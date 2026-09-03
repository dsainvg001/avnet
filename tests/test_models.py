import unittest
import torch
from avnet.models.avnet import TriStreamAVNet, AdapterNet9Axis, AVNet, AdapterNet

class TestModels(unittest.TestCase):
    def test_tristream_avnet_forward(self):
        model = TriStreamAVNet(window_size=20, hidden_dim=64)
        acc = torch.randn(4, 3, 20)
        gyro = torch.randn(4, 3, 20)
        mag = torch.randn(4, 3, 20)

        v_lon, dq = model(acc, gyro, mag)
        self.assertEqual(v_lon.shape, (4, 1))
        self.assertEqual(dq.shape, (4, 3))

    def test_adapter_9axis_forward(self):
        model = AdapterNet9Axis(in_channels=9, out_dim=6)
        x = torch.randn(4, 9, 20)
        out = model(x)
        self.assertEqual(out.shape, (4, 6))

        q_s, n_s = model.get_covariances(x)
        self.assertEqual(q_s.shape, (4, 3))
        self.assertEqual(n_s.shape, (4, 3))
        # Ensure bounded by 10^(+/-3)
        self.assertTrue(torch.all(q_s >= 0.001) and torch.all(q_s <= 1000.0))
        self.assertTrue(torch.all(n_s >= 0.001) and torch.all(n_s <= 1000.0))

    def test_legacy_avnet_forward(self):
        model = AVNet(in_channels=6, out_dim=1)
        x = torch.randn(4, 200, 6)
        out = model(x)
        self.assertEqual(out.shape, (4, 1))

    def test_legacy_adapter_forward(self):
        model = AdapterNet(in_channels=6, out_dim=6)
        x = torch.randn(4, 20, 6)
        out = model(x)
        self.assertEqual(out.shape, (4, 6))

if __name__ == '__main__':
    unittest.main()
