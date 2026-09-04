import argparse
import os
import json
import pickle
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from avnet.dataset import load_iovnbd_csv, discover_paired_iovnbd_files, create_dataloaders
from avnet.models.avnet import TriStreamAVNet, AdapterNet9Axis, SPEED_SCALE
from avnet.models.inekf import InEKF
from avnet.train import train_tristream_avnet, evaluate_model
from avnet.train_adapter import train_adapter_offline
from avnet.evaluate import compute_ate, compute_relative_errors

def run_evaluation(avnet_model, adapter_model, eval_data, device='cpu', output_dir='results', max_eval_steps=None):
    """
    Run closed-loop InEKF dead-reckoning trajectory evaluation without GNSS.
    Computes ATE, KITTI E_trel, E_rrel, and exports calibration profile.
    """
    os.makedirs(output_dir, exist_ok=True)
    avnet_model.to(device)
    avnet_model.eval()

    if adapter_model is not None:
        adapter_model.to(device)
        adapter_model.eval()

    acc = eval_data['acc']
    gyro = eval_data['gyro']
    mag = eval_data['mag']
    dt_arr = eval_data['dt']
    gt_speed = eval_data['gt_speed']
    gt_enu = eval_data['gt_enu']
    gt_quat = eval_data['gt_quat']

    N = len(acc)
    if max_eval_steps is not None and max_eval_steps > 0:
        N = min(N, max_eval_steps)
        acc = acc[:N]
        gyro = gyro[:N]
        mag = mag[:N]
        dt_arr = dt_arr[:N]
        gt_speed = gt_speed[:N]
        gt_enu = gt_enu[:N]
        gt_quat = gt_quat[:N]

    print(f"Evaluating trajectory over {N} samples ({N * 0.1 / 60:.2f} minutes)...")

    # Initial state: forward direction is along axis X
    R0 = R.from_quat(gt_quat[0]).as_matrix()
    v0_body = np.array([gt_speed[0], 0.0, 0.0], dtype=np.float64)
    v0_world = R0 @ v0_body

    inekf = InEKF(R0=R0, v0=v0_world, p0=gt_enu[0])

    pred_positions = np.zeros((N, 3))
    pred_quats = np.zeros((N, 4))
    pred_quats[0] = gt_quat[0]
    window_size = 20

    imu9 = np.concatenate([acc, gyro, mag], axis=-1).astype(np.float32)

    with torch.no_grad():
        for i in range(N):
            w = gyro[i]
            f = acc[i]
            dt = dt_arr[i]

            # 1. AdapterNet Covariance Scaling
            if adapter_model is not None and i >= window_size:
                x_adapt = torch.tensor(imu9[i - window_size:i].T, dtype=torch.float32).unsqueeze(0).to(device)
                q_s, n_s = adapter_model.get_covariances(x_adapt)
                q_s = q_s.cpu().numpy()[0]
                n_s = n_s.cpu().numpy()[0]

                Q_dyn = inekf.Q_default * np.tile(q_s, 6)
                # Forward axis is X (index 0)
                N_vel_dyn = np.diag([0.01 * n_s[0], 0.1 * n_s[1], 0.1 * n_s[2]])
            else:
                Q_dyn = inekf.Q_default
                N_vel_dyn = None

            # 2. InEKF Kinematic Propagation
            inekf.propagate(w, f, dt, Q=Q_dyn)

            # 3. AVNet Measurement Update (every 100 ms sliding window)
            if i >= window_size:
                acc_w = torch.tensor(acc[i - window_size:i].T, dtype=torch.float32).unsqueeze(0).to(device)
                gyro_w = torch.tensor(gyro[i - window_size:i].T, dtype=torch.float32).unsqueeze(0).to(device)
                mag_w = torch.tensor(mag[i - window_size:i].T, dtype=torch.float32).unsqueeze(0).to(device)

                v_lon_norm, dq_xyz_t, p_stop_t = avnet_model(acc_w, gyro_w, mag_w, return_zupt=True)
                v_lon = max(0.0, v_lon_norm.item() * SPEED_SCALE)  # Denormalize speed to m/s
                dq_vec = dq_xyz_t.cpu().numpy()[0]
                p_stop = p_stop_t.item()

                # Option 1: Hybrid DMDVDR InEKF Fusion
                if p_stop > 0.5:
                    # Vehicle at standstill: lock velocity to 0 and eliminate bias drift
                    inekf.update_zupt()
                else:
                    # Vehicle moving: apply forward speed and Non-Holonomic Constraints (v_lat=0, v_up=0)
                    inekf.update_ddodo(v_lon, w, N_vel=N_vel_dyn)
                    inekf.update_nhc(w)

                # Update attitude using filter's own estimated history (pure dead reckoning)
                w_comp = np.sqrt(max(0.0, 1.0 - np.sum(dq_vec**2)))
                dq_rel = R.from_quat([dq_vec[0], dq_vec[1], dq_vec[2], w_comp])
                R_win_start = R.from_quat(pred_quats[i - window_size]) if i > window_size else R.from_matrix(R0)
                R_meas = (R_win_start * dq_rel).as_matrix()
                inekf.update_ddatt(R_meas)

            pred_positions[i] = inekf.p_w_s
            pred_quats[i] = R.from_matrix(inekf.R_w_s).as_quat()

    # Metrics
    ate = compute_ate(pred_positions, gt_enu)
    rel_metrics = compute_relative_errors(pred_positions, gt_enu, pred_quats, gt_quat)

    print("\n===========================================================")
    print("           TRAJECTORY EVALUATION BENCHMARK RESULTS           ")
    print("===========================================================")
    print(f"Total Trajectory Distance:          {rel_metrics['total_distance_m']:.2f} m")
    print(f"Absolute Trajectory Error (ATE):    {ate:.3f} m")
    print(f"Relative Translation Error (Etrel): {rel_metrics['E_trel_percent']:.2f}%")
    print(f"Relative Rotational Error (Errel):  {rel_metrics['E_rrel_deg_km']:.2f} deg/km")
    print("===========================================================\n")

    # Export Calibration Profile Vector
    mount_euler = R.from_matrix(inekf.R_v_s).as_euler('xyz', degrees=True).tolist()
    profile_data = {
        "version": 1,
        "device": "smartphone_auto_calibrated",
        "calibration_profile_14d": {
            "b_gyro_rad_s": inekf.bg.tolist(),
            "b_accel_m_s2": inekf.ba.tolist(),
            "e_mount_deg": mount_euler,
            "b_mag_uT": [0.0, 0.0, 0.0],
            "s_vib": 1.0,
            "s_gyro": 1.0
        },
        "benchmark_results": {
            "ate_rmse_m": ate,
            "etrel_percent": rel_metrics['E_trel_percent'],
            "errel_deg_km": rel_metrics['E_rrel_deg_km']
        }
    }
    profile_path = os.path.join(output_dir, "calibration_profile.json")
    with open(profile_path, "w") as f:
        json.dump(profile_data, f, indent=2)
    print(f"Exported Adaptive Calibration Profile to {profile_path}")

    return {
        'ate': ate,
        'rel_metrics': rel_metrics,
        'pred_positions': pred_positions
    }

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Tri-Stream AVNet & InEKF Engine")
    parser.add_argument("--mode", type=str, default="demo",
                        choices=["train-avnet", "train-adapter", "evaluate", "demo"],
                        help="Operating mode: train-avnet, train-adapter, evaluate, demo")
    parser.add_argument("--data_dir", type=str, default="data", help="Root directory of IO-VNBD dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--limit_files", type=int, default=None, help="Limit number of files for quick training")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model weights")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")

    args = parser.parse_args()
    print(f"AVNet Engine initialized with mode: {args.mode}, device: {args.device}")

    if args.mode == "train-avnet":
        train_l, val_l, test_l = create_dataloaders(
            root_dir=args.data_dir,
            window_size=20,
            step=2,
            batch_size=args.batch_size,
            limit_files=args.limit_files
        )
        model = TriStreamAVNet(window_size=20)
        train_tristream_avnet(
            model=model,
            train_loader=train_l,
            val_loader=val_l,
            epochs=args.epochs,
            lr=args.lr,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device
        )

    elif args.mode == "train-adapter":
        pairs = discover_paired_iovnbd_files(args.data_dir)
        if args.limit_files:
            pairs = pairs[:args.limit_files]

        print(f"Loading {len(pairs)} sequences for AdapterNet training...")
        sequences = [load_iovnbd_csv(s, v) for s, v in pairs if v is not None]

        avnet_model = TriStreamAVNet(window_size=20)
        avnet_ckpt = os.path.join(args.checkpoint_dir, "best_avnet_tristream.pth")
        if os.path.exists(avnet_ckpt):
            ckpt = torch.load(avnet_ckpt, map_location=args.device)
            avnet_model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
            print(f"Loaded trained AVNet weights from {avnet_ckpt}")

        adapter_model = AdapterNet9Axis(in_channels=9)
        train_adapter_offline(
            adapter_model=adapter_model,
            avnet_model=avnet_model,
            data_sequences=sequences,
            epochs=args.epochs,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device
        )

    elif args.mode == "evaluate":
        pairs = discover_paired_iovnbd_files(args.data_dir)
        test_s, test_v = pairs[0]
        eval_data = load_iovnbd_csv(test_s, test_v)

        avnet_model = TriStreamAVNet(window_size=20)
        avnet_ckpt = os.path.join(args.checkpoint_dir, "best_avnet_tristream.pth")
        if os.path.exists(avnet_ckpt):
            ckpt = torch.load(avnet_ckpt, map_location=args.device)
            avnet_model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
            print(f"Loaded AVNet from {avnet_ckpt}")

        adapter_model = AdapterNet9Axis(in_channels=9)
        adapter_ckpt = os.path.join(args.checkpoint_dir, "best_adapter.pth")
        if os.path.exists(adapter_ckpt):
            adapter_model.load_state_dict(torch.load(adapter_ckpt, map_location=args.device))
            print(f"Loaded AdapterNet from {adapter_ckpt}")

        run_evaluation(avnet_model, adapter_model, eval_data, device=args.device)

    elif args.mode == "demo":
        print("Running end-to-end smoke test on 2 files...")
        train_l, val_l, _ = create_dataloaders(
            root_dir=args.data_dir,
            window_size=20,
            step=10,
            batch_size=32,
            limit_files=2
        )
        model = TriStreamAVNet(window_size=20)
        train_tristream_avnet(model, train_l, val_l, epochs=1, checkpoint_dir=args.checkpoint_dir, device=args.device)

        pairs = discover_paired_iovnbd_files(args.data_dir)
        eval_data = load_iovnbd_csv(pairs[0][0], pairs[0][1])
        adapter_model = AdapterNet9Axis(in_channels=9)
        run_evaluation(model, adapter_model, eval_data, device=args.device, max_eval_steps=2000)

if __name__ == "__main__":
    main()
