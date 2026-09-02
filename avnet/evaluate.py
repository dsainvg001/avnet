import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_ate(pred_pos, gt_pos):
    """
    Compute Absolute Trajectory Error (ATE) RMSE in meters.
    """
    errors = np.linalg.norm(pred_pos - gt_pos, axis=-1)
    ate_rmse = np.sqrt(np.mean(errors ** 2))
    return float(ate_rmse)

def compute_relative_errors(pred_pos, gt_pos, pred_quat=None, gt_quat=None, eval_lengths=[100, 200, 300, 400, 500]):
    """
    Compute KITTI-style relative translation error E_trel (%) and relative rotational error E_rrel (deg/km).
    """
    dists = np.cumsum(np.linalg.norm(np.diff(gt_pos, axis=0, prepend=gt_pos[0:1]), axis=-1))
    total_dist = dists[-1]

    t_errs = []
    r_errs = []

    N = len(gt_pos)
    for i in range(0, N - 10, 10):
        for dist_len in eval_lengths:
            d_i = dists[i]
            target_d = d_i + dist_len
            j_indices = np.where(dists >= target_d)[0]
            if len(j_indices) == 0:
                continue
            j = j_indices[0]

            gt_trans = gt_pos[j] - gt_pos[i]
            pred_trans = pred_pos[j] - pred_pos[i]

            t_err = np.linalg.norm(pred_trans - gt_trans) / dist_len * 100.0
            t_errs.append(t_err)

            if pred_quat is not None and gt_quat is not None:
                q_gt_rel = R.from_quat(gt_quat[i]).inv() * R.from_quat(gt_quat[j])
                q_pred_rel = R.from_quat(pred_quat[i]).inv() * R.from_quat(pred_quat[j])
                q_diff = q_pred_rel.inv() * q_gt_rel
                rot_err_deg = np.degrees(q_diff.magnitude())
                r_err_per_km = rot_err_deg / (dist_len / 1000.0)
                r_errs.append(r_err_per_km)

    avg_t_err = float(np.mean(t_errs)) if len(t_errs) > 0 else 0.0
    avg_r_err = float(np.mean(r_errs)) if len(r_errs) > 0 else 0.0

    return {
        'E_trel_percent': avg_t_err,
        'E_rrel_deg_km': avg_r_err,
        'total_distance_m': float(total_dist)
    }
