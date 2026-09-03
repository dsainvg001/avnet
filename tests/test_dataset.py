import unittest
import torch
import numpy as np
from avnet.dataset import (
    load_iovnbd_csv,
    AVNetDataset,
    TriStreamDataset,
    discover_paired_iovnbd_files,
    latlon_to_enu
)

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

    def test_tristream_dataset_windowing(self):
        dummy_data = {
            'time_s': np.arange(100) * 0.1,
            'dt': np.ones(100) * 0.1,
            'acc': np.random.randn(100, 3).astype(np.float32),
            'gyro': np.random.randn(100, 3).astype(np.float32),
            'mag': np.random.randn(100, 3).astype(np.float32),
            'gt_speed': (np.random.rand(100) * 10).astype(np.float32),
            'gt_distance_m': (np.cumsum(np.random.rand(100)) * 0.1).astype(np.float32),
            'gt_enu': np.random.randn(100, 3),
            'gt_quat': np.tile([0, 0, 0, 1], (100, 1))
        }
        dataset = TriStreamDataset([dummy_data], window_size=20, step=2)
        self.assertGreater(len(dataset), 0)
        sample = dataset[0]
        self.assertEqual(sample['acc'].shape, (3, 20))
        self.assertEqual(sample['gyro'].shape, (3, 20))
        self.assertEqual(sample['mag'].shape, (3, 20))
        self.assertEqual(sample['target_speed'].shape, (1,))
        self.assertEqual(sample['target_delta_q'].shape, (3,))

    def test_discover_paired_iovnbd_files(self):
        pairs = discover_paired_iovnbd_files('data')
        self.assertGreater(len(pairs), 0)
        self.assertTrue(pairs[0][0].endswith('.csv'))

if __name__ == '__main__':
    unittest.main()
