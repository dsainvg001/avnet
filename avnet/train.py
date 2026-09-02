import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from avnet.dataset import load_iovnbd_csv, AVNetDataset
from avnet.models.avnet import AVNet, AdapterNet

def train_avnet(model, dataloader, epochs=5, lr=1e-4, is_ddatt=False, save_path=None, device='cpu'):
    """
    Train AVNet model (for velocity DDODO or attitude DDATT).
    """
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

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.6f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")

    return model
