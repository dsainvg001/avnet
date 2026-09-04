import os
import json
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

def compute_attitude_loss(pred_dq_xyz, target_dq_xyz):
    """
    Geodesic + Chordal loss on the SO(3) quaternion manifold.
    Enforces canonical hemisphere w >= 0 and penalizes rotation angle error.
    """
    # Reconstruct w = sqrt(max(0, 1 - (qx^2 + qy^2 + qz^2)))
    sum_sq_pred = torch.sum(pred_dq_xyz ** 2, dim=-1, keepdim=True)
    w_pred = torch.sqrt(torch.clamp(1.0 - sum_sq_pred, min=1e-7))
    q_pred = torch.cat([pred_dq_xyz, w_pred], dim=-1) # (B, 4)

    sum_sq_target = torch.sum(target_dq_xyz ** 2, dim=-1, keepdim=True)
    w_target = torch.sqrt(torch.clamp(1.0 - sum_sq_target, min=1e-7))
    q_target = torch.cat([target_dq_xyz, w_target], dim=-1) # (B, 4)

    # Unit normalization
    q_pred = F.normalize(q_pred, p=2, dim=-1)
    q_target = F.normalize(q_target, p=2, dim=-1)

    # Chordal distance: 1 - |q1 . q2|
    dot_product = torch.abs(torch.sum(q_pred * q_target, dim=-1))
    chordal_loss = torch.mean(1.0 - torch.clamp(dot_product, -1.0, 1.0))

    # Euclidean penalty on xyz
    mse_loss = F.mse_loss(pred_dq_xyz, target_dq_xyz)

    return chordal_loss + 2.0 * mse_loss


def evaluate_model(model, dataloader, device='cpu'):
    """
    Compute validation metrics: Speed RMSE (m/s) and Attitude Geodesic Error (degrees).
    """
    model.eval()
    total_samples = 0
    speed_sq_errors = []
    att_deg_errors = []
    val_loss_sum = 0.0

    with torch.no_grad():
        for batch in dataloader:
            acc = batch['acc'].to(device)
            gyro = batch['gyro'].to(device)
            mag = batch['mag'].to(device)
            target_speed = batch['target_speed'].to(device)
            target_dq = batch['target_delta_q'].to(device)

            pred_speed, pred_dq = model(acc, gyro, mag)
            pred_speed = torch.clamp(pred_speed, min=0.0, max=80.0)

            l_speed = F.smooth_l1_loss(pred_speed, target_speed)
            l_att = compute_attitude_loss(pred_dq, target_dq)

            if torch.isnan(l_speed) or torch.isnan(l_att) or torch.isinf(l_speed) or torch.isinf(l_att):
                continue

            loss = l_speed + 50.0 * l_att
            val_loss_sum += loss.item() * acc.size(0)
            total_samples += acc.size(0)

            # Metrics
            speed_err = (pred_speed - target_speed).cpu().numpy().flatten()
            valid_speed_err = speed_err[np.isfinite(speed_err)]
            speed_sq_errors.extend(valid_speed_err ** 2)

            # Rotation angle error in degrees: theta = 2 * arccos(|q1 . q2|)
            q_p = torch.cat([pred_dq, torch.sqrt(torch.clamp(1.0 - torch.sum(pred_dq**2, dim=-1, keepdim=True), min=1e-7))], dim=-1)
            q_t = torch.cat([target_dq, torch.sqrt(torch.clamp(1.0 - torch.sum(target_dq**2, dim=-1, keepdim=True), min=1e-7))], dim=-1)
            q_p = F.normalize(q_p, p=2, dim=-1)
            q_t = F.normalize(q_t, p=2, dim=-1)
            dots = torch.clamp(torch.abs(torch.sum(q_p * q_t, dim=-1)), 0.0, 1.0)
            angle_rad = 2.0 * torch.acos(dots)
            angle_deg = torch.rad2deg(angle_rad).cpu().numpy().flatten()
            valid_angle = angle_deg[np.isfinite(angle_deg)]
            att_deg_errors.extend(valid_angle)

    avg_val_loss = (val_loss_sum / max(1, total_samples)) if total_samples > 0 else 0.0
    speed_rmse = np.sqrt(np.mean(speed_sq_errors)) if len(speed_sq_errors) > 0 else 0.0
    mean_att_deg = np.mean(att_deg_errors) if len(att_deg_errors) > 0 else 0.0

    return {
        'val_loss': float(avg_val_loss),
        'speed_rmse_mps': float(speed_rmse),
        'att_error_deg': float(mean_att_deg)
    }


def train_tristream_avnet(model, train_loader, val_loader=None,
                          epochs=15, lr=5e-4, lambda_att=50.0,
                          weight_decay=1e-4, checkpoint_dir='checkpoints',
                          device='cpu'):
    """
    Train TriStreamAVNet with multi-task Huber + Geodesic loss and Cosine Annealing.
    Saves best and latest model checkpoints as both .pth and .pkl.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'speed_rmse': [],
        'att_error_deg': [],
        'learning_rate': []
    }

    print(f"Starting TriStreamAVNet training on device: {device}")
    print(f"Total Epochs: {epochs}, Initial LR: {lr}, Lambda Att: {lambda_att}")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_speed_loss = 0.0
        running_att_loss = 0.0
        total_samples = 0

        for step, batch in enumerate(train_loader):
            acc = batch['acc'].to(device)
            gyro = batch['gyro'].to(device)
            mag = batch['mag'].to(device)
            target_speed = batch['target_speed'].to(device)
            target_dq = batch['target_delta_q'].to(device)

            optimizer.zero_grad()

            pred_speed, pred_dq = model(acc, gyro, mag)

            loss_speed = F.smooth_l1_loss(pred_speed, target_speed)
            loss_att = compute_attitude_loss(pred_dq, target_dq)
            loss_total = loss_speed + lambda_att * loss_att

            if torch.isnan(loss_total) or torch.isinf(loss_total):
                optimizer.zero_grad()
                continue

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = acc.size(0)
            running_loss += loss_total.item() * batch_size
            running_speed_loss += loss_speed.item() * batch_size
            running_att_loss += loss_att.item() * batch_size
            total_samples += batch_size

        scheduler.step()
        epoch_train_loss = (running_loss / max(1, total_samples)) if total_samples > 0 else 0.0
        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(epoch_train_loss)
        history['learning_rate'].append(current_lr)

        # Validation
        if val_loader is not None and len(val_loader) > 0:
            val_metrics = evaluate_model(model, val_loader, device=device)
            val_loss = val_metrics['val_loss']
            speed_rmse = val_metrics['speed_rmse_mps']
            att_deg = val_metrics['att_error_deg']

            history['val_loss'].append(val_loss)
            history['speed_rmse'].append(speed_rmse)
            history['att_error_deg'].append(att_deg)

            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] "
                  f"Train Loss: {epoch_train_loss:.5f} | "
                  f"Val Loss: {val_loss:.5f} | "
                  f"Speed RMSE: {speed_rmse:.3f} m/s | "
                  f"Att Error: {att_deg:.2f}° | "
                  f"LR: {current_lr:.6f}")

            # Save best checkpoint
            if np.isfinite(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_pth = os.path.join(checkpoint_dir, 'best_avnet_tristream.pth')
                best_pkl = os.path.join(checkpoint_dir, 'best_avnet_tristream.pkl')

                checkpoint_data = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'speed_rmse': speed_rmse,
                    'att_deg': att_deg,
                    'model_config': {'window_size': model.window_size, 'hidden_dim': model.hidden_dim}
                }
                torch.save(checkpoint_data, best_pth)
                with open(best_pkl, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                print(f"  --> Saved new best checkpoint (Val Loss: {val_loss:.5f}) to {best_pth} and {best_pkl}")
        else:
            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] Train Loss: {epoch_train_loss:.5f} | LR: {current_lr:.6f}")

    # Save final model
    last_pth = os.path.join(checkpoint_dir, 'last_avnet_tristream.pth')
    last_pkl = os.path.join(checkpoint_dir, 'last_avnet_tristream.pkl')
    last_data = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'history': history,
        'model_config': {'window_size': model.window_size, 'hidden_dim': model.hidden_dim}
    }
    torch.save(last_data, last_pth)
    with open(last_pkl, 'wb') as f:
        pickle.dump(last_data, f)

    with open(os.path.join(checkpoint_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed/60:.2f} minutes. Checkpoints saved to {checkpoint_dir}/")
    return model, history


# Legacy wrapper for backward compatibility
def train_avnet(model, dataloader, epochs=5, lr=1e-4, is_ddatt=False, save_path=None, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in dataloader:
            x = batch['x'].to(device)
            if is_ddatt:
                target = batch['target_delta_q'].to(device)
            else:
                target = batch['target_speed'].to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        epoch_loss = running_loss / max(1, len(dataloader.dataset))
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.6f}")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
    return model
