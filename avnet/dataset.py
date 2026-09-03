import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R

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

    time_ms = df[time_col].astype(float).values
    dt = np.diff(time_ms, prepend=time_ms[0]) / 1000.0
    # Clean negative time jumps (e.g. logger restarts/pauses)
    dt[dt <= 0] = 0.1
    dt[0] = dt[1] if len(dt) > 1 else 0.1

    acc = df[acc_cols[:3]].astype(float).values
    gyro = df[gyro_cols[:3]].astype(float).values
    mag = df[mag_cols[:3]].astype(float).values if len(mag_cols) >= 3 else np.zeros((len(df), 3))

    # Auto-detect matching vehicle CAN file if not provided
    if v_filepath is None:
        dir_name = os.path.dirname(filepath)
        base_name = os.path.basename(filepath)
        if base_name.startswith('S-'):
            candidate_v = os.path.join(dir_name, 'V-' + base_name[2:])
            if os.path.exists(candidate_v):
                v_filepath = candidate_v

    has_vehicle_can = False

    if v_filepath and os.path.exists(v_filepath):
        try:
            df_v = pd.read_csv(v_filepath, encoding='latin1')
            df_v.columns = [col.strip() for col in df_v.columns]
            if len(df_v) == len(df):
                has_vehicle_can = True
                # Vehicle CAN speed is continuous and millisecond-accurate at 10 Hz
                v_speed_cols = [c for c in df_v.columns if 'Indicated Vehicle Speed' in c or 'Velocity' in c]
                if v_speed_cols:
                    speed_kmh = df_v[v_speed_cols[0]].astype(float).values
                else:
                    speed_kmh = df[speed_col].astype(float).values

                # Vehicle GPS position
                v_lat_cols = [c for c in df_v.columns if 'Latitude' in c]
                v_lon_cols = [c for c in df_v.columns if 'Longitude' in c]
                if v_lat_cols and v_lon_cols:
                    lat = df_v[v_lat_cols[0]].astype(float).values
                    lon = df_v[v_lon_cols[0]].astype(float).values
                else:
                    lat = df[lat_col].astype(float).values
                    lon = df[lon_col].astype(float).values
        except Exception:
            pass

    if not has_vehicle_can:
        lat = df[lat_col].astype(float).values
        lon = df[lon_col].astype(float).values
        speed_kmh = df[speed_col].astype(float).values

    alt = df[alt_col].astype(float).values
    speed_ms = speed_kmh / 3.6  # convert km/h to m/s
    gt_distance_m = np.cumsum(speed_ms * dt)

    gt_enu = latlon_to_enu(lat, lon, alt)

    orient_cols = [c for c in df.columns if 'ORIENTATION' in c]
    if len(orient_cols) >= 3:
        azimuth = df[orient_cols[0]].astype(float).values
        pitch = df[orient_cols[1]].astype(float).values
        roll = df[orient_cols[2]].astype(float).values
        rotations = R.from_euler('zyx', np.stack([azimuth, pitch, roll], axis=-1), degrees=True)
        gt_quat = rotations.as_quat() # [x, y, z, w]
    else:
        gt_quat = np.tile([0.0, 0.0, 0.0, 1.0], (len(df), 1))

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
    Returns list of (s_path, v_path) tuples.
    """
    pairs = []
    # Search in Synchronised directory first
    sync_dir = os.path.join(root_dir, 'Synchronised V abd S datasets')
    search_path = sync_dir if os.path.exists(sync_dir) else root_dir

    s_files = glob.glob(os.path.join(search_path, '**', 'S-*.csv'), recursive=True)
    for s_path in s_files:
        dir_name = os.path.dirname(s_path)
        base_name = os.path.basename(s_path)
        v_name = 'V-' + base_name[2:]
        v_path = os.path.join(dir_name, v_name)
        if os.path.exists(v_path):
            pairs.append((s_path, v_path))
        else:
            pairs.append((s_path, None))

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

                target_speed = speed[end_idx]

                # Compute relative quaternion delta: q_rel = q_start^-1 * q_end
                q_start = R.from_quat(quat[start_idx])
                q_end = R.from_quat(quat[end_idx])
                q_rel = q_start.inv() * q_end
                q_rel_vec = q_rel.as_quat() # [x, y, z, w]

                # Canonical quaternion representation: ensure w >= 0
                if q_rel_vec[3] < 0:
                    q_rel_vec = -q_rel_vec
                target_delta_q = q_rel_vec[:3].astype(np.float32)

                delta_dist = dist[end_idx] - dist[start_idx]

                self.samples.append({
                    'acc': torch.tensor(acc_win, dtype=torch.float32),   # (3, W)
                    'gyro': torch.tensor(gyro_win, dtype=torch.float32), # (3, W)
                    'mag': torch.tensor(mag_win, dtype=torch.float32),   # (3, W)
                    'target_speed': torch.tensor([target_speed], dtype=torch.float32), # (1,)
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
                       train_ratio=0.8, val_ratio=0.1, limit_files=None, num_workers=0):
    """
    High-level factory function: discovers paired files, loads and windowizes them,
    and returns (train_loader, val_loader, test_loader).
    """
    pairs = discover_paired_iovnbd_files(root_dir)
    if not pairs:
        raise FileNotFoundError(f"No IO-VNBD dataset files found in {root_dir}")

    if limit_files is not None and limit_files > 0:
        pairs = pairs[:limit_files]

    # Split file pairs
    n_total = len(pairs)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]
    if not test_pairs:
        test_pairs = val_pairs

    print(f"Dataset split: {len(train_pairs)} Train files, {len(val_pairs)} Val files, {len(test_pairs)} Test files.")

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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
