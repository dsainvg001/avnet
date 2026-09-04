import numpy as np
from scipy.spatial.transform import Rotation as R

def skew(v):
    """Compute skew-symmetric matrix (v)_x for 3D vector v."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)

class InEKF:
    """
    Right-Invariant Extended Kalman Filter on GSE2(3) x GSO(3) for Vehicle Dead Reckoning.
    Equations (34) - (52) from the paper.
    """
    def __init__(self, R0=None, v0=None, p0=None, R_v_s=None, p_s_v=None, gravity=9.81):
        self.g = np.array([0.0, 0.0, -gravity], dtype=np.float64)

        self.R_w_s = R0 if R0 is not None else np.eye(3, dtype=np.float64)
        self.v_w_s = v0 if v0 is not None else np.zeros(3, dtype=np.float64)
        self.p_w_s = p0 if p0 is not None else np.zeros(3, dtype=np.float64)

        self.bg = np.zeros(3, dtype=np.float64)
        self.ba = np.zeros(3, dtype=np.float64)

        self.R_v_s = R_v_s if R_v_s is not None else np.eye(3, dtype=np.float64)
        self.p_s_v = p_s_v if p_s_v is not None else np.zeros(3, dtype=np.float64)

        # 21x21 covariance matrix
        self.P = np.eye(21, dtype=np.float64) * 1e-3
        # Default process noise Q (18x18)
        self.Q_default = np.diag([
            1e-4, 1e-4, 1e-4,  # gyro noise
            1e-3, 1e-3, 1e-3,  # accel noise
            1e-6, 1e-6, 1e-6,  # gyro bias drift
            1e-6, 1e-6, 1e-6,  # accel bias drift
            1e-6, 1e-6, 1e-6,  # R_v_s perturbation
            1e-6, 1e-6, 1e-6   # p_s_v perturbation
        ]) ** 2

    def propagate(self, gyro_meas, accel_meas, dt, Q=None):
        if Q is None:
            Q = self.Q_default

        omega_unbiased = gyro_meas - self.bg
        acc_unbiased = accel_meas - self.ba

        # Continuous / Discrete State propagation
        dR = R.from_rotvec(omega_unbiased * dt).as_matrix()
        R_next = self.R_w_s @ dR
        v_next = self.v_w_s + (self.R_w_s @ acc_unbiased + self.g) * dt
        p_next = self.p_w_s + self.v_w_s * dt + 0.5 * (self.R_w_s @ acc_unbiased + self.g) * (dt ** 2)

        # Transition Matrix A for right-invariant error dynamics
        A = np.zeros((21, 21), dtype=np.float64)
        A[0:3, 9:12] = -self.R_w_s
        A[3:6, 0:3] = skew(self.g)
        A[3:6, 12:15] = -self.R_w_s
        A[6:9, 3:6] = np.eye(3)

        F = np.eye(21, dtype=np.float64) + A * dt

        # Noise gain Matrix G (21x18) as defined in Eq. (52)
        G = np.zeros((21, 18), dtype=np.float64)
        G[0:3, 0:3] = self.R_w_s
        G[3:6, 0:3] = skew(self.v_w_s) @ self.R_w_s
        G[3:6, 3:6] = self.R_w_s
        G[6:9, 0:3] = skew(self.p_w_s) @ self.R_w_s
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)
        G[15:18, 12:15] = self.R_v_s
        G[18:21, 15:18] = np.eye(3)
        G *= dt

        # Covariance update
        self.P = F @ self.P @ F.T + G @ Q @ G.T

        # Symmetrize covariance to maintain numerical stability
        self.P = 0.5 * (self.P + self.P.T)

        self.R_w_s = R_next
        self.v_w_s = v_next
        self.p_w_s = p_next

    def update_ddatt(self, R_meas, N_att=None):
        if N_att is None:
            N_att = np.eye(3, dtype=np.float64) * 1e-2

        # Right-invariant attitude residual: log(R_meas @ R_est^T)
        R_err = R_meas @ self.R_w_s.T
        r_err = R.from_matrix(R_err).as_rotvec()

        H = np.zeros((3, 21), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)

        S = H @ self.P @ H.T + N_att
        K = self.P @ H.T @ np.linalg.inv(S)

        dx = K @ r_err
        self._apply_correction(dx)
        self.P = (np.eye(21) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

    def update_ddodo(self, v_lon_meas, gyro_meas, N_vel=None):
        if N_vel is None:
            N_vel = np.diag([1e-2, 1e-1, 1e-1])

        omega = gyro_meas - self.bg
        v_body_pred = self.R_v_s @ (self.R_w_s.T @ self.v_w_s + np.cross(omega, self.p_s_v))

        # Vehicle frame: X is forward (longitudinal), Y is lateral, Z is vertical
        v_meas = np.array([v_lon_meas, 0.0, 0.0], dtype=np.float64)
        innov = v_meas - v_body_pred

        H = np.zeros((3, 21), dtype=np.float64)
        R_s_w = self.R_w_s.T
        H[0:3, 0:3] = self.R_v_s @ skew(R_s_w @ self.v_w_s)
        H[0:3, 3:6] = self.R_v_s @ R_s_w
        H[0:3, 9:12] = self.R_v_s @ skew(self.p_s_v)
        H[0:3, 15:18] = -skew(v_body_pred)
        H[0:3, 18:21] = -self.R_v_s @ skew(omega)

        S = H @ self.P @ H.T + N_vel
        K = self.P @ H.T @ np.linalg.inv(S)

        dx = K @ innov
        self._apply_correction(dx)
        self.P = (np.eye(21) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

    def _apply_correction(self, dx):
        # Limit correction step size to prevent numerical divergence
        dx_rot = np.clip(dx[0:3], -0.1, 0.1)
        dx_vel = np.clip(dx[3:6], -5.0, 5.0)
        dx_pos = np.clip(dx[6:9], -10.0, 10.0)

        dR = R.from_rotvec(dx_rot).as_matrix()
        self.R_w_s = dR @ self.R_w_s
        self.v_w_s += dx_vel
        self.p_w_s += dx_pos
        self.bg += np.clip(dx[9:12], -0.01, 0.01)
        self.ba += np.clip(dx[12:15], -0.1, 0.1)

        dR_vs = R.from_rotvec(np.clip(dx[15:18], -0.05, 0.05)).as_matrix()
        self.R_v_s = dR_vs @ self.R_v_s
        self.p_s_v += np.clip(dx[18:21], -0.1, 0.1)
