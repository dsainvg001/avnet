import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    Multi-Modal Tri-Stream AVNet (AVNet-Mag).
    Processes 3 decoupled 3-axis streams (Accelerometer, Gyroscope, Magnetometer)
    through dedicated 1D dilated convolutional branches, fuses them with a gated
    convolution, processes temporal dynamics via a 2-layer BiGRU, pools with
    self-attention, and regresses:
      1) DDODO: Forward longitudinal speed (1 scalar in m/s)
      2) DDATT: Relative quaternion imaginary component (3 scalars [qx, qy, qz])
    """
    def __init__(self, window_size=20, hidden_dim=64):
        super(TriStreamAVNet, self).__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        # Stream 1: Accelerometer Branch (Linear Acceleration)
        self.acc_branch = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1)
        )

        # Stream 2: Gyroscope Branch (Angular Velocity)
        self.gyro_branch = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1)
        )

        # Stream 3: Magnetometer Branch (Geomagnetic Flux)
        self.mag_branch = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1)
        )

        # Gated Fusion Block (Concatenates 64 + 64 + 64 = 192 channels)
        self.fusion_conv = nn.Conv1d(192, 128, kernel_size=1)
        self.fusion_gate = nn.Conv1d(192, 128, kernel_size=1)
        self.fusion_bn = nn.BatchNorm1d(128)

        # Bidirectional GRU (128 -> 64 each direction = 128 output)
        self.bigru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # Temporal Self-Attention Pooling
        self.attention_pool = SelfAttentionPooling(in_features=hidden_dim * 2)

        # Dual Regression Heads
        self.ddodo_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1) # scalar forward speed (v_lon)
        )

        self.ddatt_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3) # relative quaternion [qx, qy, qz]
        )

    def forward(self, acc, gyro, mag=None):
        """
        Forward pass.
        Args:
            acc: (B, 3, W) or (B, W, 3) - Accelerometer stream
            gyro: (B, 3, W) or (B, W, 3) - Gyroscope stream
            mag: (B, 3, W) or (B, W, 3) - Magnetometer stream (optional, zeros if None)
        Returns:
            v_lon: (B, 1) - Predicted forward speed in m/s
            dq_xyz: (B, 3) - Predicted relative quaternion imaginary part
        """
        if acc.dim() == 3 and acc.size(-1) == 3:
            acc = acc.transpose(1, 2)
        if gyro.dim() == 3 and gyro.size(-1) == 3:
            gyro = gyro.transpose(1, 2)

        if mag is None:
            mag = torch.zeros_like(acc)
        elif mag.dim() == 3 and mag.size(-1) == 3:
            mag = mag.transpose(1, 2)

        # 1. Decoupled feature extraction
        f_acc = self.acc_branch(acc)   # (B, 64, W)
        f_gyro = self.gyro_branch(gyro)# (B, 64, W)
        f_mag = self.mag_branch(mag)   # (B, 64, W)

        # 2. Gated Cross-Sensor Fusion
        f_cat = torch.cat([f_acc, f_gyro, f_mag], dim=1) # (B, 192, W)
        fusion = self.fusion_conv(f_cat)
        gate = torch.sigmoid(self.fusion_gate(f_cat))
        f_fused = self.fusion_bn(fusion * gate) # (B, 128, W)

        # 3. Recurrent Temporal Dynamics
        f_fused_t = f_fused.transpose(1, 2) # (B, W, 128)
        gru_out, _ = self.bigru(f_fused_t)  # (B, W, 128)

        # 4. Self-Attention Pooling
        context, _ = self.attention_pool(gru_out) # (B, 128)

        # 5. Regression Heads
        v_lon = self.ddodo_head(context)   # (B, 1)
        dq_xyz = self.ddatt_head(context) # (B, 3)

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

    def get_covariances(self, x, q_default=None, n_default=None, beta=3.0):
        """
        Convenience function: computes physical Q and N covariance matrices.
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
