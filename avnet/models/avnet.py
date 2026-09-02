import torch
import torch.nn as nn
import torch.nn.functional as F

class AVNet(nn.Module):
    """
    AVNet: Combined CNN-GRU architecture for learning data-driven measurements from IMU sequences.
    Supports dynamic window length sequences.
    """
    def __init__(self, in_channels=6, out_dim=1, hidden_dim=64):
        super(AVNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Adaptive pooling to fix spatial feature map to length 50 regardless of input sequence length
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

        out = self.adaptive_pool(out) # (Batch, 64, 50)
        out = out.flatten(1)
        out = F.relu(self.fc_proj1(out))
        out = F.relu(self.fc_proj2(out))

        out_gru, _ = self.gru(out.unsqueeze(1))
        out_last = out_gru[:, -1, :]

        pred = self.regressor(out_last)
        return pred


class AdapterNet(nn.Module):
    """
    Data-driven Filter Parameter Adapter Network (2-layer 1D dilated CNN).
    Outputs 6 scaling parameters for process Q and measurement N covariances.
    """
    def __init__(self, in_channels=6, out_dim=6):
        super(AdapterNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, dilation=1, padding=2)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, dilation=3, padding=6)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(32, out_dim)

    def forward(self, x):
        if x.dim() == 3 and x.size(-1) == 6:
            x = x.transpose(1, 2)

        out = F.relu(self.conv1(x))
        out = self.dropout1(out)
        out = F.relu(self.conv2(out))
        out = self.dropout2(out)

        out_last = out[:, :, -1]
        params = self.fc(out_last)
        return params
