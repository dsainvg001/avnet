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
from avnet.models.avnet import SPEED_SCALE

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


def evaluate_model(model, dataloader, lambda_att=3.0, lambda_zupt=2.0, device='cpu'):
    """
    Compute validation metrics: Speed RMSE (m/s), Attitude Geodesic Error (degrees), and ZUPT Accuracy (%).
    target_speed in the dataloader is normalized by SPEED_SCALE; we denormalize for RMSE reporting.
    """
    model.eval()
    total_samples = 0
    speed_sq_errors = []
    att_deg_errors = []
    zupt_correct = 0
    zupt_total = 0
    val_loss_sum = 0.0

    with torch.no_grad():
        for batch in dataloader:
            acc = batch['acc'].to(device)
            gyro = batch['gyro'].to(device)
            target_speed = batch['target_speed'].to(device)  # normalized [0, 1]
            target_dq = batch['target_delta_q'].to(device)
            target_stop = batch.get('target_is_stopped', (target_speed < 0.3 / SPEED_SCALE).float()).to(device)

            pred_speed, pred_dq, pred_stop = model(acc, gyro, return_zupt=True)
            # Clamp in normalized [0, 1] space (corresponds to 0–30 m/s)
            pred_speed = torch.clamp(pred_speed, min=0.0, max=1.0)
            pred_stop = torch.clamp(pred_stop, min=1e-7, max=1.0 - 1e-7)

            l_speed = F.smooth_l1_loss(pred_speed, target_speed, beta=0.1)
            l_att = compute_attitude_loss(pred_dq, target_dq)
            l_zupt = F.binary_cross_entropy(pred_stop, target_stop)

            if torch.isnan(l_speed) or torch.isnan(l_att) or torch.isnan(l_zupt) or torch.isinf(l_speed) or torch.isinf(l_att) or torch.isinf(l_zupt):
                continue

            loss = l_speed + lambda_att * l_att + lambda_zupt * l_zupt
            val_loss_sum += loss.item() * acc.size(0)
            total_samples += acc.size(0)

            # ZUPT accuracy
            pred_binary = (pred_stop > 0.5).float()
            zupt_correct += (pred_binary == (target_stop > 0.5).float()).sum().item()
            zupt_total += target_stop.numel()

            # Denormalize speed for human-readable RMSE in m/s
            pred_mps = (pred_speed * SPEED_SCALE).cpu().numpy().flatten()
            tgt_mps  = (target_speed * SPEED_SCALE).cpu().numpy().flatten()
            speed_err = pred_mps - tgt_mps
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
    zupt_acc = (zupt_correct / max(1, zupt_total)) * 100.0 if zupt_total > 0 else 0.0

    return {
        'val_loss': float(avg_val_loss),
        'speed_rmse_mps': float(speed_rmse),
        'att_error_deg': float(mean_att_deg),
        'zupt_acc': float(zupt_acc)
    }


def train_tristream_avnet(model, train_loader, val_loader=None,
                          epochs=25, lr=6e-4, lambda_att=2.0, lambda_zupt=2.0,
                          weight_decay=1e-4, patience=15, scheduler_type='plateau',
                          checkpoint_dir='checkpoints', device='cpu'):
    """
    Train DualStreamAVNet with multi-task Huber + Geodesic + ZUPT BCE loss and dynamic LR scheduling.
    Supports ReduceLROnPlateau (drops LR when val loss plateaus) and Cosine Annealing.
    Includes early stopping (patience) and saves best/latest checkpoints as .pth and .pkl.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-5)
    else:
        effective_tmax = min(epochs, 30)
        scheduler = CosineAnnealingLR(optimizer, T_max=effective_tmax, eta_min=1e-5)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'speed_rmse': [],
        'att_error_deg': [],
        'zupt_acc': [],
        'learning_rate': []
    }

    print(f"Starting DualStreamAVNet training on device: {device}")
    print(f"Total Epochs: {epochs}, Initial LR: {lr}, Scheduler: {scheduler_type}, Lambda Att: {lambda_att}, Lambda ZUPT: {lambda_zupt}")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_speed_loss = 0.0
        running_att_loss = 0.0
        running_zupt_loss = 0.0
        total_samples = 0

        for step, batch in enumerate(train_loader):
            acc = batch['acc'].to(device)
            gyro = batch['gyro'].to(device)
            target_speed = batch['target_speed'].to(device)
            target_dq = batch['target_delta_q'].to(device)
            target_stop = batch.get('target_is_stopped', (target_speed < 0.3 / SPEED_SCALE).float()).to(device)

            optimizer.zero_grad()

            pred_speed, pred_dq, pred_stop = model(acc, gyro, return_zupt=True)

            # target_speed is normalized to [0, 1] (SPEED_SCALE = 30 m/s) by the dataset.
            loss_speed = F.smooth_l1_loss(pred_speed, target_speed, beta=0.1)
            loss_att = compute_attitude_loss(pred_dq, target_dq)
            loss_zupt = F.binary_cross_entropy(pred_stop, target_stop)
            loss_total = loss_speed + lambda_att * loss_att + lambda_zupt * loss_zupt

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
            running_zupt_loss += loss_zupt.item() * batch_size
            total_samples += batch_size

        epoch_train_loss = (running_loss / max(1, total_samples)) if total_samples > 0 else 0.0

        # Validation
        if val_loader is not None and len(val_loader) > 0:
            val_metrics = evaluate_model(model, val_loader, lambda_att=lambda_att, lambda_zupt=lambda_zupt, device=device)
            val_loss = val_metrics['val_loss']
            speed_rmse = val_metrics['speed_rmse_mps']
            att_deg = val_metrics['att_error_deg']
            zupt_acc = val_metrics['zupt_acc']

            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
            else:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]

            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(val_loss)
            history['speed_rmse'].append(speed_rmse)
            history['att_error_deg'].append(att_deg)
            history['zupt_acc'].append(zupt_acc)
            history['learning_rate'].append(current_lr)

            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] "
                  f"Train Loss: {epoch_train_loss:.5f} | "
                  f"Val Loss: {val_loss:.5f} | "
                  f"ZUPT Acc: {zupt_acc:.1f}% | "
                  f"Att Error: {att_deg:.2f}° | "
                  f"Speed RMSE: {speed_rmse:.3f} m/s | "
                  f"LR: {current_lr:.6f}")

            # Save best checkpoint
            if np.isfinite(val_loss) and val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_pth = os.path.join(checkpoint_dir, 'best_avnet_tristream.pth')
                best_pkl = os.path.join(checkpoint_dir, 'best_avnet_tristream.pkl')

                checkpoint_data = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'speed_rmse': speed_rmse,
                    'att_deg': att_deg,
                    'zupt_acc': zupt_acc,
                    'model_config': {'window_size': getattr(model, 'window_size', 20), 'hidden_dim': getattr(model, 'hidden_dim', 64)}
                }
                torch.save(checkpoint_data, best_pth)
                with open(best_pkl, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                print(f"  --> Saved new best checkpoint (Val Loss: {val_loss:.5f}) to {best_pth} and {best_pkl}")
            else:
                epochs_without_improvement += 1
                if patience is not None and epochs_without_improvement >= patience:
                    print(f"\n[Early Stopping] Val loss did not improve for {patience} consecutive epochs. Best Val Loss: {best_val_loss:.5f}. Stopping at epoch {epoch+1}.")
                    break
        else:
            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] Train Loss: {epoch_train_loss:.5f} | LR: {current_lr:.6f}")

    # Save final model
    last_pth = os.path.join(checkpoint_dir, 'last_avnet_tristream.pth')
    last_pkl = os.path.join(checkpoint_dir, 'last_avnet_tristream.pkl')
    last_data = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'history': history,
        'model_config': {'window_size': getattr(model, 'window_size', 20), 'hidden_dim': getattr(model, 'hidden_dim', 64)}
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
