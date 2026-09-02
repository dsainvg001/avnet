import unittest
import torch
import numpy as np
from avnet.dataset import load_iovnbd_csv, AVNetDataset, latlon_to_enu

class TestDataset(unittest.TestCase):
    def test_latlon_to_enu(self):
        lat = np.array([52.402565, 52.402570])
        lon = np.array([-1.503471, -1.503480])
        alt = np.array([144.59, 144.60])
        enu = latlon_to_enu(lat, lon, alt)
        self.assertEqual(enu.shape, (2, 3))
        self.assertAlmostEqual(enu[0, 0], 0.0)
        self.assertAlmostEqual(enu[0, 1], 0.0)
        self.assertAlmostEqual(enu[0, 2], 0.0)

    def test_dataset_windowing(self):
        dummy_data = {
            'time_s': np.arange(300) * 0.1,
            'dt': np.ones(300) * 0.1,
            'acc': np.random.randn(300, 3),
            'gyro': np.random.randn(300, 3),
            'gt_speed': np.random.rand(300) * 10,
            'gt_enu': np.random.randn(300, 3),
            'gt_quat': np.tile([0, 0, 0, 1], (300, 1))
        }
        dataset = AVNetDataset([dummy_data], window_size=200, step=10)
        self.assertGreater(len(dataset), 0)
        sample = dataset[0]
        self.assertEqual(sample['x'].shape, (200, 6))
        self.assertEqual(sample['target_speed'].shape, (1,))
        self.assertEqual(sample['target_delta_q'].shape, (3,))

if __name__ == '__main__':
    unittest.main()
