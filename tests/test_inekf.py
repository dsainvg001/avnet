import unittest
import numpy as np
from avnet.models.inekf import InEKF
from avnet.evaluate import compute_ate, compute_relative_errors

class TestInEKFAndEval(unittest.TestCase):
    def test_inekf_propagation_and_update(self):
        inekf = InEKF()
        gyro = np.array([0.01, 0.02, -0.01])
        accel = np.array([0.05, 0.0, 9.81])
        inekf.propagate(gyro, accel, dt=0.1)
        self.assertEqual(inekf.P.shape, (21, 21))

        inekf.update_ddatt(np.eye(3))
        inekf.update_ddodo(v_lon_meas=5.0, gyro_meas=gyro)
        self.assertIsNotNone(inekf.p_w_s)

    def test_inekf_zupt_and_nhc(self):
        inekf = InEKF(v0=np.array([5.0, 0.0, 0.0]))
        gyro = np.array([0.0, 0.0, 0.0])
        accel = np.array([0.0, 0.0, 9.81])
        inekf.propagate(gyro, accel, dt=0.1)

        # Test NHC: preserves filter stability and updates covariance
        inekf.update_nhc(gyro_meas=gyro)
        self.assertEqual(inekf.P.shape, (21, 21))

        # Test ZUPT: reduces velocity magnitude towards zero
        inekf.update_zupt()
        self.assertLess(np.linalg.norm(inekf.v_w_s), 5.0)

    def test_evaluation_metrics(self):
        gt_pos = np.zeros((50, 3))
        gt_pos[:, 0] = np.linspace(0, 100, 50)
        pred_pos = gt_pos + 0.1

        ate = compute_ate(pred_pos, gt_pos)
        self.assertGreater(ate, 0.0)

        rel = compute_relative_errors(pred_pos, gt_pos, eval_lengths=[20, 50])
        self.assertIn('E_trel_percent', rel)

if __name__ == '__main__':
    unittest.main()
