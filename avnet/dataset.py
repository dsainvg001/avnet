import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

def latlon_to_enu(lat, lon, alt, lat0=None, lon0=None, alt0=None):
    """
    Convert geographic coordinates (latitude, longitude, altitude) to
    Local East-North-Up (ENU) Cartesian coordinates in meters using WGS84 ellipsoid.
    """
    lat, lon, alt = np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64), np.asarray(alt, dtype=np.float64)
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


def load_iovnbd_csv(filepath):
    """
    Load an IO-VNBD dataset CSV file and clean up column names and data types.
    """
    df = pd.read_csv(filepath, encoding='latin1')
    # Strip whitespace from column names
    df.columns = [col.strip() for col in df.columns]

    # Map expected columns
    acc_cols = [c for c in df.columns if 'ACCELEROMETER' in c]
    gyro_cols = [c for c in df.columns if 'GYROSCOPE' in c]
    time_col = [c for c in df.columns if 'TIME SINCE START' in c or 'TIME' in c][0]
    lat_col = [c for c in df.columns if 'LATITUDE' in c][0]
    lon_col = [c for c in df.columns if 'LONGITUDE' in c][0]
    alt_col = [c for c in df.columns if 'ALTITUDE' in c][0]
    speed_col = [c for c in df.columns if 'SPEED' in c][0]

    time_ms = df[time_col].astype(float).values
    dt = np.diff(time_ms, prepend=time_ms[0]) / 1000.0
    dt[0] = dt[1] if len(dt) > 1 else 0.1

    acc = df[acc_cols[:3]].astype(float).values
    gyro = df[gyro_cols[:3]].astype(float).values

    lat = df[lat_col].astype(float).values
    lon = df[lon_col].astype(float).values
    alt = df[alt_col].astype(float).values
    speed_kmh = df[speed_col].astype(float).values
    speed_ms = speed_kmh / 3.6  # convert km/h to m/s

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
        'time_s': time_ms / 1000.0,
        'dt': dt,
        'acc': acc,          # (N, 3) specific forces [m/s^2]
        'gyro': gyro,        # (N, 3) angular velocity [rad/s]
        'gt_speed': speed_ms,# (N,) ground truth scalar forward speed [m/s]
        'gt_enu': gt_enu,    # (N, 3) ground truth ENU position [m]
        'gt_quat': gt_quat,  # (N, 4) ground truth quaternion [x, y, z, w]
    }


class AVNetDataset(Dataset):
    """
    PyTorch Dataset for windowed IMU sequence training for AVNet (DDATT and DDODO).
    """
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
