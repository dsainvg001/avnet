import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R

from avnet.dataset import load_iovnbd_csv, AVNetDataset
from avnet.models.avnet import AVNet, AdapterNet
from avnet.models.inekf import InEKF
from avnet.train import train_avnet
from avnet.evaluate import compute_ate, compute_relative_errors

def run_pipeline(csv_path, epochs=3, window_size=200, adapter_win=20, batch_size=32, device='cpu'):
    print(f"--- Loading dataset from {csv_path} ---")
    data = load_iovnbd_csv(csv_path)
    N = len(data['time_s'])
    print(f"Loaded {N} samples from dataset.")

    dataset = AVNetDataset([data], window_size=window_size, step=5)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("--- Training AVNet DDODO (Speed Model) ---")
    model_ddodo = AVNet(in_channels=6, out_dim=1)
    train_avnet(model_ddodo, dataloader, epochs=epochs, device=device)

    print("--- Training AVNet DDATT (Attitude Model) ---")
    model_ddatt = AVNet(in_channels=6, out_dim=3)
    train_avnet(model_ddatt, dataloader, epochs=epochs, is_ddatt=True, device=device)

    adapter_net = AdapterNet(in_channels=6, out_dim=6).to(device)

    model_ddodo.eval()
    model_ddatt.eval()
    adapter_net.eval()

    print("--- Running AVNet Inference & InEKF Filter ---")
    imu_features = np.concatenate([data['gyro'], data['acc']], axis=-1).astype(np.float32)

    # Initial rotation and velocity in world ENU frame
    R0 = R.from_quat(data['gt_quat'][0]).as_matrix()
    v0_body = np.array([0.0, data['gt_speed'][0], 0.0])
    v0_world = R0 @ v0_body

    inekf = InEKF(
        R0=R0,
        v0=v0_world,
        p0=data['gt_enu'][0]
    )

    pred_positions = np.zeros((N, 3))
    pred_quats = np.zeros((N, 4))

    # AVNet 1Hz update interval (every window_size steps, e.g. 200 samples)
    update_interval = window_size

    with torch.no_grad():
        for i in range(N):
            gyro = data['gyro'][i]
            accel = data['acc'][i]
            dt = data['dt'][i]

            # AdapterNet dynamic covariance scaling
            if i >= adapter_win:
                x_adapt = torch.tensor(imu_features[i - adapter_win:i], dtype=torch.float32).unsqueeze(0).to(device)
                adapter_out = adapter_net(x_adapt).cpu().numpy()[0] # 6 params

                beta = 3.0
                q_scale = 10.0 ** (beta * np.tanh(adapter_out[0:3]))
                n_scale = 10.0 ** (beta * np.tanh(adapter_out[3:6]))

                Q_dyn = inekf.Q_default * np.tile(q_scale, 6)
                N_vel_dyn = np.diag([1e-1 * n_scale[0], 1e-2 * n_scale[1], 1e-1 * n_scale[2]])
            else:
                Q_dyn = inekf.Q_default
                N_vel_dyn = None

            inekf.propagate(gyro, accel, dt, Q=Q_dyn)

            # Apply AVNet DDODO and DDATT measurements at window interval (1 Hz)
            if i >= window_size and i % update_interval == 0:
                x_win = torch.tensor(imu_features[i - window_size:i], dtype=torch.float32).unsqueeze(0).to(device)
                v_lon = model_ddodo(x_win).item()
                q_delta_vec = model_ddatt(x_win).cpu().numpy()[0]

                # Update velocity
                inekf.update_ddodo(v_lon, gyro, N_vel=N_vel_dyn)

                # Compute world attitude measurement from relative delta quaternion
                w_comp = np.sqrt(max(0.0, 1.0 - np.sum(q_delta_vec**2)))
                dq_rel = R.from_quat([q_delta_vec[0], q_delta_vec[1], q_delta_vec[2], w_comp])

                # Baseline rotation at window start
                R_win_start = R.from_quat(data['gt_quat'][i - window_size])
                R_att_meas = (R_win_start * dq_rel).as_matrix()

                inekf.update_ddatt(R_att_meas)

            pred_positions[i] = inekf.p_w_s
            pred_quats[i] = R.from_matrix(inekf.R_w_s).as_quat()

    print("--- Evaluating Trajectory Performance ---")
    ate = compute_ate(pred_positions, data['gt_enu'])
    rel_metrics = compute_relative_errors(pred_positions, data['gt_enu'], pred_quats, data['gt_quat'])

    print(f"Absolute Trajectory Error (ATE RMSE): {ate:.4f} meters")
    print(f"Relative Translation Error (E_trel): {rel_metrics['E_trel_percent']:.2f}%")
    print(f"Relative Rotational Error (E_rrel): {rel_metrics['E_rrel_deg_km']:.2f} deg/km")
    print(f"Total Trajectory Distance: {rel_metrics['total_distance_m']:.2f} meters")

    return {
        'ate': ate,
        'rel_metrics': rel_metrics,
        'pred_positions': pred_positions
    }

def main():
    parser = argparse.ArgumentParser(description="AVNet implementation on IO-VNBD Dataset")
    parser.add_argument("--csv_path", type=str, default="dataset_iovnbd/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-M.csv", help="Path to IO-VNBD CSV file")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--window_size", type=int, default=200, help="Window size for AVNet input")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run PyTorch models on (cpu/cuda)")
    args = parser.parse_args()

    run_pipeline(args.csv_path, epochs=args.epochs, window_size=args.window_size, batch_size=args.batch_size, device=args.device)

if __name__ == "__main__":
    main()
