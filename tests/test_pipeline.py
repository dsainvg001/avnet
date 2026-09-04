import unittest
import os
import shutil
import torch
import numpy as np
from torch.utils.data import DataLoader

from avnet.models.avnet import TriStreamAVNet
from avnet.dataset import TriStreamDataset
from avnet.train import train_tristream_avnet, evaluate_model, compute_attitude_loss

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_checkpoints_tmp'
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_attitude_loss(self):
        # Perfect match -> loss near 0
        q1 = torch.tensor([[0.0, 0.0, 0.0]])
        q2 = torch.tensor([[0.0, 0.0, 0.0]])
        loss = compute_attitude_loss(q1, q2)
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

        # Perturbed -> loss > 0
        q3 = torch.tensor([[0.2, 0.1, 0.0]])
        loss_perturbed = compute_attitude_loss(q1, q3)
        self.assertGreater(loss_perturbed.item(), 0.0)

    def test_full_training_cycle(self):
        dummy_data = {
            'time_s': np.arange(100) * 0.1,
            'dt': np.ones(100) * 0.1,
            'acc': np.random.randn(100, 3).astype(np.float32),
            'gyro': np.random.randn(100, 3).astype(np.float32),
            'gt_speed': (np.random.rand(100) * 10).astype(np.float32),
            'gt_distance_m': (np.cumsum(np.random.rand(100)) * 0.1).astype(np.float32),
            'gt_enu': np.random.randn(100, 3),
            'gt_quat': np.tile([0, 0, 0, 1], (100, 1))
        }
        dataset = TriStreamDataset([dummy_data], window_size=20, step=2)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        model = TriStreamAVNet(window_size=20, hidden_dim=32)
        trained_model, history = train_tristream_avnet(
            model=model,
            train_loader=loader,
            val_loader=loader,
            epochs=2,
            lr=1e-3,
            checkpoint_dir=self.test_dir,
            device='cpu'
        )

        self.assertEqual(len(history['train_loss']), 2)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'best_avnet_tristream.pth')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'best_avnet_tristream.pkl')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'training_history.json')))

if __name__ == '__main__':
    unittest.main()
