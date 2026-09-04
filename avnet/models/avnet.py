import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Speed scale constant: model predicts normalized speed in [0, 1] range.
# Multiply by SPEED_SCALE to convert predictions back to m/s at inference/eval time.
SPEED_SCALE = 30.0  # m/s — covers the full IO-VNBD speed range (0–120 km/h = 33 m/s)

class SelfAttentionPooling(nn.Module):
    """
    Temporal Self-Attention Pooling layer.
    Computes an attention score for each time step in the sequence and
    pools features via a weighted sum.
    """
    def __init__(self, in_features):
        super(SelfAttentionPooling, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Tanh(),
            nn.Linear(in_features // 2, 1)
        )

    def forward(self, x):
        # x shape: (B, T, C)
        scores = self.attn(x) # (B, T, 1)
        weights = F.softmax(scores, dim=1) # (B, T, 1)
        context = torch.sum(x * weights, dim=1) # (B, C)
        return context, weights


class TriStreamAVNet(nn.Module):
    """
    Decoupled Multi-Modal Tri-Stream AVNet (AVNet-Mag).
    Engineered specifically for vehicular dead reckoning:
      1) Invariant Physical Energy: Automatically augments each stream with its
         3D Euclidean norm [X, Y, Z, ||V||], guaranteeing rotation-invariant kinetic energy.
      2) Decoupled Multi-Task Pathways: Separates the Speed branch (specializing on
         linear acceleration dynamics & vibration PSD) from the Attitude branch
         (specializing on angular rate integration & magnetometer heading fusion).
         Eliminates negative task interference / gradient conflict.
      3) Temporal Endpoint Aggregation: Combines the final recurrent state (h_last)
         with temporal mean pooling, preserving the cumulative kinematic integration
         at the window boundary.
      4) Physically Bounded Speed: Uses a Sigmoid output head strictly bounding
         normalized speed in [0, 1] (0 to 30 m/s), preventing negative speed oscillations.
    """
    def __init__(self, window_size=20, hidden_dim=64):
        super(TriStreamAVNet, self).__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        # Stream 1: Accelerometer Branch (3 axes + Euclidean Norm = 4 channels)
        self.acc_branch = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Stream 2: Gyroscope Branch (3 axes + Euclidean Norm = 4 channels)
        self.gyro_branch = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Stream 3: Magnetometer Branch (3 axes + Euclidean Norm = 4 channels)
        self.mag_branch = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Decoupled Task Recurrent Processors:
        # 1. Speed Recurrent Stream (focuses purely on acceleration dynamics)
        self.speed_gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        # 2. Attitude Recurrent Stream (fuses Gyro 64 + Mag 64 = 128 channels)
        self.att_fuse = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.att_gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        # Decoupled Regression Heads (each fed by h_last + mean_pool = hidden_dim * 4 = 256 features)
        self.ddodo_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Strictly bounded [0, 1] normalized forward speed
        )

        # Head 2: Zero Velocity Detector (DDZUPT: probability that vehicle is stopped v=0)
        self.zupt_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()  # [0, 1] probability: 1.0 = stopped at standstill, 0.0 = moving
        )

        self.ddatt_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, 64),
            nn.GELU(),
            nn.Linear(64, 3)  # relative quaternion [qx, qy, qz]
        )

    def forward(self, acc, gyro, mag=None, return_zupt=False):
        """
        Forward pass.
        Args:
            acc: (B, 3, W) or (B, W, 3) - Accelerometer stream
            gyro: (B, 3, W) or (B, W, 3) - Gyroscope stream
            mag: (B, 3, W) or (B, W, 3) - Magnetometer stream (optional, zeros if None)
            return_zupt: bool - If True, returns (v_lon, dq_xyz, p_stop); if False, returns (v_lon, dq_xyz)
        Returns:
            v_lon: (B, 1) - Predicted forward speed in normalized [0, 1] range
            dq_xyz: (B, 3) - Predicted relative quaternion imaginary part
            p_stop: (B, 1) - Predicted stationary probability [0, 1] (only if return_zupt=True)
        """
        if acc.dim() == 3 and acc.size(-1) == 3:
            acc = acc.transpose(1, 2)
        if gyro.dim() == 3 and gyro.size(-1) == 3:
            gyro = gyro.transpose(1, 2)

        if mag is None:
            mag = torch.zeros_like(acc)
        elif mag.dim() == 3 and mag.size(-1) == 3:
            mag = mag.transpose(1, 2)

        # Standardize physical sensor scales
        acc_scaled = acc / 9.81
        gyro_scaled = gyro
        mag_scaled = mag / 50.0

        # Compute rotation-invariant Euclidean norm as 4th channel
        acc_norm = torch.norm(acc_scaled, p=2, dim=1, keepdim=True)
        gyro_norm = torch.norm(gyro_scaled, p=2, dim=1, keepdim=True)
        mag_norm = torch.norm(mag_scaled, p=2, dim=1, keepdim=True)

        acc_in = torch.cat([acc_scaled, acc_norm], dim=1)   # (B, 4, W)
        gyro_in = torch.cat([gyro_scaled, gyro_norm], dim=1) # (B, 4, W)
        mag_in = torch.cat([mag_scaled, mag_norm], dim=1)   # (B, 4, W)

        # 1. Feature extraction
        f_acc = self.acc_branch(acc_in)   # (B, 64, W)
        f_gyro = self.gyro_branch(gyro_in) # (B, 64, W)
        f_mag = self.mag_branch(mag_in)   # (B, 64, W)

        # 2. Decoupled Path A: Forward Speed (DDODO) & Stationary Detection (DDZUPT)
        out_spd, _ = self.speed_gru(f_acc.transpose(1, 2))  # (B, W, 128)
        # Combine final recurrent state (h_last) with temporal mean pool
        feat_spd = torch.cat([out_spd[:, -1, :], torch.mean(out_spd, dim=1)], dim=-1) # (B, 256)
        v_lon = self.ddodo_head(feat_spd)  # (B, 1) strictly in [0, 1]
        p_stop = self.zupt_head(feat_spd)  # (B, 1) stationary probability in [0, 1]

        # 3. Decoupled Path B: Attitude Change (DDATT)
        f_rot = self.att_fuse(torch.cat([f_gyro, f_mag], dim=1)) # (B, 64, W)
        out_att, _ = self.att_gru(f_rot.transpose(1, 2))         # (B, W, 128)
        feat_att = torch.cat([out_att[:, -1, :], torch.mean(out_att, dim=1)], dim=-1) # (B, 256)
        dq_xyz = self.ddatt_head(feat_att) # (B, 3)

        if return_zupt:
            return v_lon, dq_xyz, p_stop
        return v_lon, dq_xyz


class AdapterNet9Axis(nn.Module):
    """
    Multi-Modal 9-Axis Filter Parameter Adapter Network (2-layer 1D dilated CNN).
    Takes 9-channel IMU (accel + gyro + mag) window and outputs 6 scaling parameters
    (3 for process noise Q, 3 for measurement noise N) bounded by tanh.
    """
    def __init__(self, in_channels=9, out_dim=6):
        super(AdapterNet9Axis, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, dilation=1, padding=2)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, dilation=3, padding=6)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(32, out_dim)

    def forward(self, x):
        # x shape: (B, in_channels, W) or (B, W, in_channels)
        if x.dim() == 3 and x.size(-1) == self.in_channels:
            x = x.transpose(1, 2)

        out = F.relu(self.conv1(x))
        out = self.dropout1(out)
        out = F.relu(self.conv2(out))
        out = self.dropout2(out)

        out_last = out[:, :, -1] # slice last temporal sample
        params = self.fc(out_last) # (B, out_dim)
        return params

    def get_covariances(self, x, q_default=None, n_default=None, beta=1.0):
        """
        Convenience function: computes physical Q and N covariance scaling matrices.
        Beta defaults to 1.0 (bounding adjustments to [0.1, 10.0] factor to prevent filter explosion).
        """
        params = self.forward(x) # (B, 6)
        q_raw = params[:, 0:3]
        n_raw = params[:, 3:6]

        q_scale = 10.0 ** (beta * torch.tanh(q_raw))
        n_scale = 10.0 ** (beta * torch.tanh(n_raw))

        return q_scale, n_scale


# Legacy aliases for backward compatibility
class AVNet(nn.Module):
    def __init__(self, in_channels=6, out_dim=1, hidden_dim=64):
        super(AVNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(50)
        self.fc_proj1 = nn.Linear(64 * 50, 128)
        self.fc_proj2 = nn.Linear(128, 64)
        self.gru = nn.GRU(input_size=64, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.regressor = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        if x.dim() == 3 and x.size(-1) == 6:
            x = x.transpose(1, 2)
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = self.adaptive_pool(out)
        out = out.flatten(1)
        out = F.relu(self.fc_proj1(out))
        out = F.relu(self.fc_proj2(out))
        out_gru, _ = self.gru(out.unsqueeze(1))
        out_last = out_gru[:, -1, :]
        return self.regressor(out_last)

class AdapterNet(AdapterNet9Axis):
    def __init__(self, in_channels=6, out_dim=6):
        super(AdapterNet, self).__init__(in_channels=in_channels, out_dim=out_dim)
