import unittest
import torch
from avnet.models.avnet import AVNet, AdapterNet

class TestModels(unittest.TestCase):
    def test_avnet_ddodo_forward(self):
        model = AVNet(in_channels=6, out_dim=1)
        x = torch.randn(8, 200, 6)
        out = model(x)
        self.assertEqual(out.shape, (8, 1))

    def test_avnet_ddatt_forward(self):
        model = AVNet(in_channels=6, out_dim=3)
        x = torch.randn(8, 200, 6)
        out = model(x)
        self.assertEqual(out.shape, (8, 3))

    def test_adapter_forward(self):
        model = AdapterNet(in_channels=6, out_dim=6)
        x = torch.randn(8, 20, 6)
        out = model(x)
        self.assertEqual(out.shape, (8, 6))

if __name__ == '__main__':
    unittest.main()
