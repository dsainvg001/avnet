import os
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation as R

from avnet.models.avnet import AdapterNet9Axis
from avnet.models.inekf import InEKF
from avnet.evaluate import compute_ate, compute_relative_errors

def train_adapter_offline(adapter_model, avnet_model, data_sequences,
                          epochs=5, lr=1e-3, segment_len=100,
                          checkpoint_dir='checkpoints', device='cpu'):
    """
    Offline training for AdapterNet9Axis using innovation-matching & trajectory feedback.
    Adapts process covariance Q to local motion dynamics and measurement covariance N
    to empirical speed prediction errors.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    adapter_model.to(device)
    avnet_model.to(device)
    avnet_model.eval()

    optimizer = optim.AdamW(adapter_model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"Starting Offline AdapterNet Training across {len(data_sequences)} driving sequences...")
    best_loss = float('inf')

    for epoch in range(epochs):
        adapter_model.train()
        epoch_losses = []

        for s_idx, data in enumerate(data_sequences):
            acc = data['acc']
            gyro = data['gyro']
            mag = data['mag']
            gt_speed = data['gt_speed']
            N_steps = len(acc)
            if N_steps < segment_len + 25:
                continue

            imu9 = np.concatenate([acc, gyro, mag], axis=-1).astype(np.float32)

            # Sample segments
            for _ in range(min(4, N_steps // segment_len)):
                start_i = np.random.randint(20, N_steps - segment_len)
                end_i = start_i + segment_len

                win_samples = []
                for k in range(start_i, end_i):
                    win_samples.append(imu9[k - 20:k].T) # (9, 20)
                batch_win = torch.tensor(np.array(win_samples), dtype=torch.float32).to(device)

                # 1. Forward pass AdapterNet
                params = adapter_model(batch_win) # (segment_len, 6)
                q_log_scale = params[:, 0:3] # process scale logits
                n_log_scale = params[:, 3:6] # measurement scale logits

                # 2. Compute empirical innovation targets from AVNet speed predictions
                with torch.no_grad():
                    acc_batch = batch_win[:, 0:3, :]
                    gyro_batch = batch_win[:, 3:6, :]
                    mag_batch = batch_win[:, 6:9, :]
                    v_preds, _ = avnet_model(acc_batch, gyro_batch, mag_batch)
                    v_true = torch.tensor(gt_speed[start_i:end_i], dtype=torch.float32).unsqueeze(1).to(device)
                    # Speed innovation magnitude
                    speed_innov = torch.abs(v_preds - v_true)
                    # Local sensor vibration variance
                    acc_var = torch.var(acc_batch, dim=-1) # (segment_len, 3)

                # Target log-scale: when error is high, n_log_scale should be positive (scale up N)
                # target for N scale: log10 of speed error ratio
                target_n = torch.tanh((speed_innov - 1.0) / 2.0).repeat(1, 3)
                target_q = torch.tanh((acc_var - 1.0) / 5.0)

                loss_n = F.mse_loss(n_log_scale, target_n)
                loss_q = F.mse_loss(q_log_scale, target_q)
                reg_loss = torch.mean(params ** 2) * 0.01
                total_loss = loss_n + loss_q + reg_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter_model.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(total_loss.item())

        mean_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        print(f"Adapter Epoch [{epoch+1:02d}/{epochs:02d}] Mean Innovation-Matching Loss: {mean_loss:.4f}")

        if mean_loss < best_loss and mean_loss > 0:
            best_loss = mean_loss
            ckpt_path = os.path.join(checkpoint_dir, 'best_adapter.pth')
            pkl_path = os.path.join(checkpoint_dir, 'best_adapter.pkl')
            torch.save(adapter_model.state_dict(), ckpt_path)
            with open(pkl_path, 'wb') as f:
                pickle.dump(adapter_model.state_dict(), f)
            print(f"  --> Saved new best AdapterNet weights to {ckpt_path} and {pkl_path}")

    return adapter_model
