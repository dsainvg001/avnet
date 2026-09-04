import unittest
import torch
import numpy as np
from avnet.dataset import (
    load_iovnbd_csv,
    AVNetDataset,
    TriStreamDataset,
    discover_paired_iovnbd_files,
    latlon_to_enu,
    resample_to_frequency,
    synthesize_vehicle_vibrations
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
        self.assertEqual(sample['target_speed'].shape, (1,))
        self.assertEqual(sample['target_delta_q'].shape, (3,))

    def test_high_frequency_resampling(self):
        t = np.arange(50) * 0.1
        dummy_data = {
            'time_s': t,
            'dt': np.ones(50) * 0.1,
            'acc': np.random.randn(50, 3).astype(np.float32),
            'gyro': np.random.randn(50, 3).astype(np.float32),
            'gt_speed': (np.ones(50) * 15.0).astype(np.float32),
            'gt_distance_m': (np.arange(50) * 1.5).astype(np.float32),
            'gt_enu': np.zeros((50, 3), dtype=np.float32),
            'gt_quat': np.tile([0.0, 0.0, 0.0, 1.0], (50, 1)).astype(np.float32)
        }

        # Resample to 100 Hz
        resampled_list = resample_to_frequency(dummy_data, target_freq=100.0, synthesize_harmonics=True)
        self.assertIsInstance(resampled_list, list)
        self.assertGreater(len(resampled_list), 0)
        res = resampled_list[0]
        self.assertAlmostEqual(res['dt'][0], 0.01, places=3)
        self.assertGreater(len(res['time_s']), 400)

        # Vibration check: speed=15 m/s -> vibrations injected
        va, vg = synthesize_vehicle_vibrations(t, dummy_data['gt_speed'], sample_rate=100.0)
        self.assertEqual(va.shape, (50, 3))
        self.assertGreater(np.std(va), 0.0)

    def test_discover_paired_iovnbd_files(self):
        pairs = discover_paired_iovnbd_files('data')
        self.assertGreater(len(pairs), 0)
        self.assertTrue(pairs[0][0].endswith('.csv'))

if __name__ == '__main__':
    unittest.main()
