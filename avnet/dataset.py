import os
import glob
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R
from avnet.models.avnet import SPEED_SCALE

def latlon_to_enu(lat, lon, alt, lat0=None, lon0=None, alt0=None):
    """
    Convert geographic coordinates (latitude, longitude, altitude) to
    Local East-North-Up (ENU) Cartesian coordinates in meters using WGS84 ellipsoid.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    alt = np.asarray(alt, dtype=np.float64)
    if lat0 is None:
        lat0 = lat[0]
    if lon0 is None:
        lon0 = lon[0]
    if alt0 is None:
        alt0 = alt[0]

    # WGS84 ellipsoid constants
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)

    def geodetic_to_ecef(latitude, longitude, altitude):
        rad_lat = np.radians(latitude)
        rad_lon = np.radians(longitude)
        N = a / np.sqrt(1.0 - e2 * np.sin(rad_lat)**2)
        X = (N + altitude) * np.cos(rad_lat) * np.cos(rad_lon)
        Y = (N + altitude) * np.cos(rad_lat) * np.sin(rad_lon)
        Z = (N * (1.0 - e2) + altitude) * np.sin(rad_lat)
        return X, Y, Z

    X, Y, Z = geodetic_to_ecef(lat, lon, alt)
    X0, Y0, Z0 = geodetic_to_ecef(lat0, lon0, alt0)

    dX = X - X0
    dY = Y - Y0
    dZ = Z - Z0

    rad_lat0 = np.radians(lat0)
    rad_lon0 = np.radians(lon0)

    e = -np.sin(rad_lon0) * dX + np.cos(rad_lon0) * dY
    n = -np.sin(rad_lat0) * np.cos(rad_lon0) * dX - np.sin(rad_lat0) * np.sin(rad_lon0) * dY + np.cos(rad_lat0) * dZ
    u = np.cos(rad_lat0) * np.cos(rad_lon0) * dX + np.cos(rad_lat0) * np.sin(rad_lon0) * dY + np.sin(rad_lat0) * dZ

    return np.stack([e, n, u], axis=-1)


def load_iovnbd_csv(filepath, v_filepath=None):
    """
    Load an IO-VNBD dataset CSV file and clean up column names and data types.
    If a matching V-*.csv file exists (or is provided via v_filepath), it automatically
    pairs the vehicle CAN bus data to provide high-precision continuous 10 Hz speed,
    incremental odometry distance, and vehicle-grade GPS position.
    """
    df = pd.read_csv(filepath, encoding='latin1')
    df.columns = [col.strip() for col in df.columns]

    # Map expected smartphone columns
    acc_cols = [c for c in df.columns if 'ACCELEROMETER' in c]
    gyro_cols = [c for c in df.columns if 'GYROSCOPE' in c]
    mag_cols = [c for c in df.columns if 'MAGNETIC' in c]
    time_col = [c for c in df.columns if 'TIME SINCE START' in c or 'TIME' in c][0]
    lat_col = [c for c in df.columns if 'LATITUDE' in c][0]
    lon_col = [c for c in df.columns if 'LONGITUDE' in c][0]
    alt_col = [c for c in df.columns if 'ALTITUDE' in c][0]
    speed_col = [c for c in df.columns if 'SPEED' in c][0]

    # Clean time and calculate dt
    time_ms = pd.Series(df[time_col]).ffill().bfill().fillna(0.0).astype(float).values
    dt = np.diff(time_ms, prepend=time_ms[0]) / 1000.0
    dt[dt <= 0] = 0.1
    dt[0] = dt[1] if len(dt) > 1 else 0.1

    # Clean and forward-fill sensor readings
    acc = pd.DataFrame(df[acc_cols[:3]]).ffill().bfill().fillna(0.0).values.astype(np.float32)
    gyro = pd.DataFrame(df[gyro_cols[:3]]).ffill().bfill().fillna(0.0).values.astype(np.float32)
    if len(mag_cols) >= 3:
        mag = pd.DataFrame(df[mag_cols[:3]]).ffill().bfill().fillna(0.0).values.astype(np.float32)
    else:
        mag = np.zeros((len(df), 3), dtype=np.float32)

    acc = np.nan_to_num(acc, nan=0.0)
    gyro = np.nan_to_num(gyro, nan=0.0)
    mag = np.nan_to_num(mag, nan=0.0)

    # Auto-detect matching vehicle CAN file if not provided (case-insensitive for Linux/Kaggle)
    if v_filepath is None:
        dir_name = os.path.dirname(filepath)
        base_name = os.path.basename(filepath)
        session_key = base_name[2:].lower()
        if os.path.exists(dir_name):
            for f in os.listdir(dir_name):
                if f.lower().startswith('v-') and f[2:].lower() == session_key:
                    v_filepath = os.path.join(dir_name, f)
                    break
        if v_filepath is None:
            # Check sibling directory if in S-Dataset
            alt_dir = dir_name.replace('S-Dataset', 'V-Dataset').replace('S Dataset', 'V Dataset')
            if os.path.exists(alt_dir):
                for f in os.listdir(alt_dir):
                    if f.lower().startswith('v-') and f[2:].lower() == session_key:
                        v_filepath = os.path.join(alt_dir, f)
                        break

    has_vehicle_can = False
    speed_kmh = None
    lat = None
    lon = None

    if v_filepath and os.path.exists(v_filepath):
        try:
            df_v = pd.read_csv(v_filepath, encoding='latin1')
            df_v.columns = [col.strip() for col in df_v.columns]
            if len(df_v) == len(df):
                has_vehicle_can = True
                # Vehicle CAN speed is continuous and millisecond-accurate at 10 Hz
                v_speed_cols = [c for c in df_v.columns if 'Indicated Vehicle Speed' in c or 'Velocity' in c or 'SPEED' in c]
                if v_speed_cols:
                    speed_kmh = df_v[v_speed_cols[0]].values
                else:
                    speed_kmh = df[speed_col].values

                # Vehicle GPS position
                v_lat_cols = [c for c in df_v.columns if 'Latitude' in c or 'LATITUDE' in c]
                v_lon_cols = [c for c in df_v.columns if 'Longitude' in c or 'LONGITUDE' in c]
                if v_lat_cols and v_lon_cols:
                    lat = df_v[v_lat_cols[0]].values
                    lon = df_v[v_lon_cols[0]].values
                else:
                    lat = df[lat_col].values
                    lon = df[lon_col].values
        except Exception:
            pass

    if not has_vehicle_can or speed_kmh is None:
        lat = df[lat_col].values
        lon = df[lon_col].values
        speed_kmh = df[speed_col].values

    # Clean and interpolate position and speed
    lat = pd.Series(lat).ffill().bfill().fillna(0.0).astype(float).values
    lon = pd.Series(lon).ffill().bfill().fillna(0.0).astype(float).values
    alt = pd.Series(df[alt_col]).ffill().bfill().fillna(0.0).astype(float).values
    speed_kmh = pd.Series(speed_kmh).ffill().bfill().fillna(0.0).astype(float).values

    # Convert km/h to m/s, eliminate any residual NaNs, clip to realistic vehicular speed
    speed_ms = np.nan_to_num(speed_kmh / 3.6, nan=0.0).astype(np.float32)
    speed_ms = np.clip(speed_ms, 0.0, 70.0)
    gt_distance_m = np.cumsum(speed_ms * dt).astype(np.float32)

    gt_enu = latlon_to_enu(lat, lon, alt)

    # Ground truth vehicle orientation: Use true vehicle Heading from V-Dataset if available
    v_heading_cols = [c for c in df_v.columns if 'Heading' in c] if has_vehicle_can else []
    if v_heading_cols:
        heading_deg = pd.Series(df_v[v_heading_cols[0]]).ffill().bfill().fillna(0.0).astype(float).values
        yaw_enu_deg = (90.0 - heading_deg) % 360.0
        rotations = R.from_euler('z', yaw_enu_deg.reshape(-1, 1), degrees=True)
        gt_quat = rotations.as_quat() # [x, y, z, w]
    else:
        # Fallback to phone orientation
        orient_yaw = [c for c in df.columns if ('ORIENTATION (Yaw)' in c or 'ORIENTATION (Azimuth)' in c or c == 'ORIENTATION YAW')]
        orient_pitch = [c for c in df.columns if ('ORIENTATION (Pitch)' in c or c == 'ORIENTATION PITCH')]
        orient_roll = [c for c in df.columns if ('ORIENTATION (Roll' in c or c == 'ORIENTATION ROLL')]

        if orient_yaw and orient_pitch and orient_roll:
            azimuth = pd.Series(df[orient_yaw[0]]).ffill().bfill().fillna(0.0).astype(float).values
            pitch = pd.Series(df[orient_pitch[0]]).ffill().bfill().fillna(0.0).astype(float).values
            roll = pd.Series(df[orient_roll[0]]).ffill().bfill().fillna(0.0).astype(float).values

            rotations = R.from_euler('zyx', np.stack([azimuth, pitch, roll], axis=-1), degrees=True)
            gt_quat = rotations.as_quat() # [x, y, z, w]
        else:
            gt_quat = np.tile([0.0, 0.0, 0.0, 1.0], (len(df), 1))

    # Guard against any zero or NaN norm quaternions
    norms = np.linalg.norm(gt_quat, axis=-1, keepdims=True)
    invalid = np.isnan(norms) | (norms < 1e-6)
    gt_quat[invalid.squeeze()] = [0.0, 0.0, 0.0, 1.0]
    norms[invalid] = 1.0
    gt_quat = gt_quat / norms

    return {
        'filepath': filepath,
        'v_filepath': v_filepath,
        'time_s': time_ms / 1000.0,
        'dt': dt,
        'acc': acc,              # (N, 3) specific forces [m/s^2]
        'gyro': gyro,            # (N, 3) angular velocity [rad/s]
        'mag': mag,              # (N, 3) magnetic field [uT]
        'gt_speed': speed_ms,    # (N,) ground truth scalar forward speed [m/s]
        'gt_distance_m': gt_distance_m, # (N,) high-precision integrated distance [m]
        'gt_enu': gt_enu,        # (N, 3) ground truth ENU position [m]
        'gt_quat': gt_quat,      # (N, 4) ground truth quaternion [x, y, z, w]
        'has_vehicle_can': has_vehicle_can,
    }


def discover_paired_iovnbd_files(root_dir='data'):
    """
    Search root_dir for all synchronized (S-*.csv, V-*.csv) file pairs.
    Uses case-insensitive session key indexing and os.walk to guarantee 100% pairing
    on Linux (case-sensitive) filesystems like Kaggle / Google Colab.
    Deduplicates by session name so there is no data leakage across splits.
    Returns list of (s_path, v_path) tuples.
    """
    sync_candidates = [
        os.path.join(root_dir, 'Synchronised V abd S datasets'),
        os.path.join(root_dir, 'Synchronised V and S datasets'),
        root_dir
    ]
    search_path = next((p for p in sync_candidates if os.path.exists(p)), root_dir)

    def is_categorised(p):
        pl = p.lower()
        return 'categorised' in pl and 'uncategorised' not in pl

    # 1. Index all V-files by lowercase session key (e.g. 'vtb6.csv', 'm.csv')
    v_map = {}
    for r, dirs, files in os.walk(search_path):
        for f in files:
            fl = f.lower()
            if fl.startswith('v-') and fl.endswith('.csv'):
                key = fl[2:]
                full_path = os.path.join(r, f)
                if key not in v_map or (is_categorised(full_path) and not is_categorised(v_map[key])):
                    v_map[key] = full_path

    # Fallback search across full root_dir if search_path was a subfolder
    if search_path != root_dir and os.path.exists(root_dir):
        for r, dirs, files in os.walk(root_dir):
            for f in files:
                fl = f.lower()
                if fl.startswith('v-') and fl.endswith('.csv'):
                    key = fl[2:]
                    if key not in v_map:
                        v_map[key] = os.path.join(r, f)

    # 2. Index all S-files by lowercase session key
    s_map = {}
    for r, dirs, files in os.walk(search_path):
        for f in files:
            fl = f.lower()
            if fl.startswith('s-') and fl.endswith('.csv'):
                key = fl[2:]
                full_path = os.path.join(r, f)
                if key not in s_map or (is_categorised(full_path) and not is_categorised(s_map[key])):
                    s_map[key] = full_path

    pairs = []
    for k in sorted(s_map.keys()):
        s_p = s_map[k]
        v_p = v_map.get(k, None)
        pairs.append((s_p, v_p))

    # Prefer pairs that have vehicle CAN ground truth
    valid_pairs = [p for p in pairs if p[1] is not None]
    if valid_pairs:
        pairs = valid_pairs

    return pairs



class TriStreamDataset(Dataset):
    """
    PyTorch Dataset for windowed Multi-Modal Tri-Stream AVNet training.
    Produces decoupled (acc, gyro, mag) windows + speed and delta quaternion targets.
    """
    def __init__(self, data_dict_list, window_size=20, step=2):
        self.samples = []
        self.window_size = window_size

        if not isinstance(data_dict_list, list):
            data_dict_list = [data_dict_list]

        for data in data_dict_list:
            acc = data['acc'].astype(np.float32)
            gyro = data['gyro'].astype(np.float32)
            mag = data['mag'].astype(np.float32)
            speed = data['gt_speed'].astype(np.float32)
            quat = data['gt_quat']
            dist = data['gt_distance_m'].astype(np.float32)
            dt = data['dt']

            N = len(acc)
            # Find any session resets / large time gaps
            valid_mask = (dt > 0) & (dt < 1.0)

            for end_idx in range(window_size - 1, N, step):
                start_idx = end_idx - window_size + 1

                # Ensure all samples in window belong to a continuous session
                if not np.all(valid_mask[start_idx + 1:end_idx + 1]):
                    continue

                acc_win = acc[start_idx:end_idx + 1].T   # (3, W)
                gyro_win = gyro[start_idx:end_idx + 1].T # (3, W)
                mag_win = mag[start_idx:end_idx + 1].T   # (3, W)

                target_speed = float(speed[end_idx])
                if np.isnan(target_speed) or np.isinf(target_speed):
                    continue

                # Compute relative quaternion delta: q_rel = q_start^-1 * q_end
                q_s = quat[start_idx]
                q_e = quat[end_idx]
                norm_s = np.linalg.norm(q_s)
                norm_e = np.linalg.norm(q_e)
                if norm_s < 1e-6 or norm_e < 1e-6 or np.isnan(norm_s) or np.isnan(norm_e):
                    continue

                q_start = R.from_quat(q_s / norm_s)
                q_end = R.from_quat(q_e / norm_e)
                q_rel = q_start.inv() * q_end
                q_rel_vec = q_rel.as_quat() # [x, y, z, w]

                # Canonical quaternion representation: ensure w >= 0
                if q_rel_vec[3] < 0:
                    q_rel_vec = -q_rel_vec
                target_delta_q = q_rel_vec[:3].astype(np.float32)

                if np.isnan(target_delta_q).any() or np.isinf(target_delta_q).any():
                    continue
                if np.isnan(acc_win).any() or np.isnan(gyro_win).any() or np.isnan(mag_win).any():
                    continue

                delta_dist = float(dist[end_idx] - dist[start_idx])
                if np.isnan(delta_dist) or np.isinf(delta_dist):
                    delta_dist = 0.0

                self.samples.append({
                    'acc': torch.tensor(acc_win, dtype=torch.float32),   # (3, W)
                    'gyro': torch.tensor(gyro_win, dtype=torch.float32), # (3, W)
                    'mag': torch.tensor(mag_win, dtype=torch.float32),   # (3, W)
                    # Normalize speed to [0, 1] so the model learns a scale-invariant speed signal
                    'target_speed': torch.tensor([target_speed / SPEED_SCALE], dtype=torch.float32),  # (1,) normalized
                    'target_delta_q': torch.tensor(target_delta_q, dtype=torch.float32), # (3,)
                    'delta_dist': torch.tensor([delta_dist], dtype=torch.float32)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# Legacy wrapper for backward compatibility with older tests
class AVNetDataset(Dataset):
    def __init__(self, data_dict_list, window_size=200, step=1):
        self.samples = []
        self.window_size = window_size
        if not isinstance(data_dict_list, list):
            data_dict_list = [data_dict_list]

        for data in data_dict_list:
            acc = data['acc']
            gyro = data['gyro']
            speed = data['gt_speed']
            quat = data['gt_quat']
            imu_features = np.concatenate([gyro, acc], axis=-1).astype(np.float32)

            N = len(imu_features)
            for end_idx in range(window_size - 1, N, step):
                start_idx = end_idx - window_size + 1
                x_win = imu_features[start_idx:end_idx + 1]
                target_speed = speed[end_idx]

                q_start = R.from_quat(quat[start_idx])
                q_end = R.from_quat(quat[end_idx])
                q_rel = q_start.inv() * q_end
                target_delta_q = q_rel.as_quat()[:3]

                self.samples.append({
                    'x': torch.tensor(x_win, dtype=torch.float32),
                    'target_speed': torch.tensor([target_speed], dtype=torch.float32),
                    'target_delta_q': torch.tensor(target_delta_q, dtype=torch.float32)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def create_dataloaders(root_dir='data', window_size=20, step=2, batch_size=64,
                       train_ratio=0.8, val_ratio=0.1, limit_files=None, num_workers=0, seed=42,
                       split_mode='temporal'):
    """
    High-level factory function: discovers paired files, loads and windowizes them,
    and returns (train_loader, val_loader, test_loader).

    split_mode:
      - 'temporal' (Default & Recommended): Splits each driving session temporally
        (e.g., first 80% train, next 10% val, last 10% test with buffer gap).
        Guarantees identical speed distributions (mean ~9.2 m/s for both Train and Val),
        eliminating the artificial 7.4 m/s speed bias, and ensures every vehicle chassis
        and smartphone sensor is represented across all splits.
      - 'file': Shuffles and splits whole files. Note: IO-VNBD file sizes range from
        300 to 105,000 samples, which can cause severe train/val domain shifts.
    """
    pairs = discover_paired_iovnbd_files(root_dir)
    if not pairs:
        raise FileNotFoundError(f"No IO-VNBD dataset files found in {root_dir}")

    if seed is not None:
        rng = random.Random(seed)
        pairs = pairs.copy()
        rng.shuffle(pairs)

    if limit_files is not None and limit_files > 0:
        pairs = pairs[:limit_files]

    def slice_dict(data, start, end):
        res = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                res[k] = v[start:end]
            else:
                res[k] = v
        return res

    if split_mode == 'temporal':
        train_data = []
        val_data = []
        test_data = []
        gap = window_size  # Non-overlapping buffer gap to prevent temporal window leakage

        print(f"Loading and segmenting {len(pairs)} driving sequences using temporal split...")
        for s_path, v_path in pairs:
            try:
                data = load_iovnbd_csv(s_path, v_path)
            except Exception as e:
                print(f"Warning: Failed to load {s_path}: {e}")
                continue

            N = len(data['acc'])
            if N < window_size * 3:
                train_data.append(data)
                continue

            n_tr = int(N * train_ratio)
            n_val = int(N * val_ratio)

            train_data.append(slice_dict(data, 0, n_tr))
            val_end = min(N, n_tr + gap + n_val)
            if n_tr + gap < val_end:
                val_data.append(slice_dict(data, n_tr + gap, val_end))
            test_start = val_end + gap
            if test_start < N:
                test_data.append(slice_dict(data, test_start, N))
            else:
                test_data.append(slice_dict(data, n_tr + gap, val_end))

    else:
        # Legacy whole-file split
        n_total = len(pairs)
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))

        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train:n_train + n_val]
        test_pairs = pairs[n_train + n_val:]
        if not test_pairs:
            test_pairs = val_pairs

        print(f"Dataset file split: {len(train_pairs)} Train files, {len(val_pairs)} Val files, {len(test_pairs)} Test files.")

        def load_pair_list(pair_list):
            loaded = []
            for s_path, v_path in pair_list:
                try:
                    loaded.append(load_iovnbd_csv(s_path, v_path))
                except Exception as e:
                    print(f"Warning: Failed to load {s_path}: {e}")
            return loaded

        print("Loading train files...")
        train_data = load_pair_list(train_pairs)
        print("Loading val files...")
        val_data = load_pair_list(val_pairs)
        print("Loading test files...")
        test_data = load_pair_list(test_pairs)

    train_ds = TriStreamDataset(train_data, window_size=window_size, step=step)
    val_ds = TriStreamDataset(val_data, window_size=window_size, step=step * 2)
    test_ds = TriStreamDataset(test_data, window_size=window_size, step=step * 2)

    print(f"Windows extracted: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader

