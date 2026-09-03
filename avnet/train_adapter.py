import os
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation as R

from avnet.models.avnet import AdapterNet9Axis
from avnet.models.inekf import InEKF
from avnet.evaluate import compute_ate, compute_relative_errors

def train_adapter_offline(adapter_model, avnet_model, data_sequences,
                          epochs=5, lr=5e-4, segment_len=100,
                          checkpoint_dir='checkpoints', device='cpu'):
    """
    Offline training for AdapterNet9Axis using trajectory-level indirect optimization.
    Evaluates InEKF trajectory error across segment_len steps and updates AdapterNet
    parameters to minimize translation drift against CAN bus ground truth.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    adapter_model.to(device)
    avnet_model.to(device)
    avnet_model.eval()

    optimizer = optim.AdamW(adapter_model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"Starting Offline AdapterNet Training across {len(data_sequences)} driving sequences...")
    best_trel = float('inf')

    for epoch in range(epochs):
        adapter_model.train()
        epoch_losses = []

        for s_idx, data in enumerate(data_sequences):
            acc = data['acc']
            gyro = data['gyro']
            mag = data['mag']
            dt_arr = data['dt']
            gt_speed = data['gt_speed']
            gt_enu = data['gt_enu']
            gt_quat = data['gt_quat']
            N_steps = len(acc)

            # Combined 9-axis features: (N, 9)
            imu9 = np.concatenate([acc, gyro, mag], axis=-1).astype(np.float32)

            # Sample random segments of length segment_len
            n_segments = max(1, N_steps // segment_len)
            for seg in range(min(5, n_segments)): # sample up to 5 segments per sequence
                start_i = np.random.randint(20, max(21, N_steps - segment_len - 1))
                end_i = start_i + segment_len

                # 1. Forward pass AdapterNet over the segment
                win_samples = []
                for k in range(start_i, end_i):
                    win_samples.append(imu9[k - 20:k].T) # (9, 20)
                batch_win = torch.tensor(np.array(win_samples), dtype=torch.float32).to(device)

                # Predict Q and N scale factors
                params = adapter_model(batch_win) # (segment_len, 6)
                q_scales = 10.0 ** (3.0 * torch.tanh(params[:, 0:3]))
                n_scales = 10.0 ** (3.0 * torch.tanh(params[:, 3:6]))

                # 2. Run InEKF over the segment
                R0 = R.from_quat(gt_quat[start_i]).as_matrix()
                v0 = np.array([0.0, gt_speed[start_i], 0.0])
                p0 = gt_enu[start_i]

                inekf = InEKF(R0=R0, v0=R0 @ v0, p0=p0)

                pred_pos = []
                q_scales_np = q_scales.detach().cpu().numpy()
                n_scales_np = n_scales.detach().cpu().numpy()

                with torch.no_grad():
                    for step_k in range(segment_len):
                        idx = start_i + step_k
                        w = gyro[idx]
                        f = acc[idx]
                        dt = dt_arr[idx]

                        Q_dyn = inekf.Q_default * np.tile(q_scales_np[step_k], 6)
                        inekf.propagate(w, f, dt, Q=Q_dyn)

                        # Update every 10 steps (1.0 s)
                        if step_k % 10 == 0:
                            acc_w = torch.tensor(acc[idx-20:idx].T, dtype=torch.float32).unsqueeze(0).to(device)
                            gyro_w = torch.tensor(gyro[idx-20:idx].T, dtype=torch.float32).unsqueeze(0).to(device)
                            mag_w = torch.tensor(mag[idx-20:idx].T, dtype=torch.float32).unsqueeze(0).to(device)
                            v_pred, _ = avnet_model(acc_w, gyro_w, mag_w)

                            N_vel = np.diag([0.1 * n_scales_np[step_k, 0],
                                             0.01 * n_scales_np[step_k, 1],
                                             0.1 * n_scales_np[step_k, 2]])
                            inekf.update_ddodo(v_pred.item(), w, N_vel=N_vel)

                        pred_pos.append(inekf.p_w_s.copy())

                pred_pos = np.array(pred_pos)
                true_pos = gt_enu[start_i:end_i]

                # Endpoint error
                disp_pred = pred_pos[-1] - pred_pos[0]
                disp_true = true_pos[-1] - true_pos[0]
                dist_true = np.linalg.norm(disp_true)

                if dist_true > 1.0:
                    drift_error = np.linalg.norm(disp_pred - disp_true) / dist_true
                    loss_proxy = torch.mean((params[:, 3:6]) ** 2) * 0.01 + drift_error

                    optimizer.zero_grad()
                    loss_proxy.backward()
                    torch.nn.utils.clip_grad_norm_(adapter_model.parameters(), 1.0)
                    optimizer.step()

                    epoch_losses.append(drift_error)

        mean_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        print(f"Adapter Epoch [{epoch+1:02d}/{epochs:02d}] Mean Segment Translation Error: {mean_loss * 100:.2f}%")

        if mean_loss < best_trel and mean_loss > 0:
            best_trel = mean_loss
            ckpt_path = os.path.join(checkpoint_dir, 'best_adapter.pth')
            pkl_path = os.path.join(checkpoint_dir, 'best_adapter.pkl')
            torch.save(adapter_model.state_dict(), ckpt_path)
            with open(pkl_path, 'wb') as f:
                pickle.dump(adapter_model.state_dict(), f)
            print(f"  --> Saved new best AdapterNet weights to {ckpt_path} and {pkl_path}")

    return adapter_model
