# AVNet Architecture, Theoretical Framework & Input/Output Specification

This document provides a comprehensive technical breakdown of the **Combined Data- and Model-Driven Vehicle Dead Reckoning (DMDVDR)** architecture introduced in the research paper:

> **"Avnet: learning attitude and velocity for vehicular dead reckoning using smartphone by adapting an invariant EKF"**  
> *Long Qian, Xinchuang Lin, Xiaoguang Niu, Qihai Huang, Leilei Li, Guangyi Guo, Zexin Wang, and Ruizhi Chen*  
> *Satellite Navigation (2025) 6:15* — [DOI: 10.1186/s43020-025-00168-7](https://doi.org/10.1186/s43020-025-00168-7)

---

## 1. System Overview & Workflow Pipeline

The DMDVDR architecture solves the problem of severe sensor drift in low-cost smartphone IMUs during GNSS outages (e.g., tunnels, underground parking garages, multi-level flyovers).

It integrates four interacting blocks:
1. **AVNet (Data-Driven Measurement Estimator)**: A hybrid CNN-GRU neural network that predicts virtual odometry velocity (DDODO) and relative attitude changes (DDATT).
2. **AdapterNet (Filter Parameter Adapter)**: A dilated 1D CNN that dynamically estimates process noise covariance $Q(t_n)$ and measurement noise covariance $N(t_{n+1})$.
3. **Model-Driven INS Mechanization**: Continuous strapdown kinematics integrating raw high-frequency IMU observations.
4. **Invariant Extended Kalman Filter (InEKF)**: A Lie-group observer on $SE_2(3) \times SO(3) \times \mathbb{R}^9$ that fuses model propagation with learned 6D geometric constraints.

```
                                  +---------------------------------------+
                                  |         Raw Smartphone IMU            |
                                  |    \tilde{\omega}(t), \tilde{f}(t)    |
                                  +-------------------+-------------------+
                                                      |
                         +----------------------------+----------------------------+
                         |                                                         |
                         v                                                         v
        +----------------------------------+                      +----------------------------------+
        |       Model-Driven Module        |                      |        Data-Driven Module        |
        |  - Continuous INS Mechanization  |                      |  - AVNet: DDODO (Speed) + DDATT  |
        |  - Kinematics on SE2(3)          |                      |  - AdapterNet: Dynamic Q and N   |
        +----------------+-----------------+                      +----------------+-----------------+
                         |                                                         |
                         |  High-Rate Propagation (200 Hz)                         |  Periodic Update (1 Hz - 10 Hz)
                         |  (State x^-, Covariance P^-)                            |  (y_att, y_vel, Q_dyn, N_dyn)
                         |                                                         |
                         +----------------------------> [ InEKF Fusion Engine ] <--+
                                                                |
                                                                v
                                                +-------------------------------+
                                                |     Full Extended State       |
                                                |  - R^w_s: 3D Attitude (w)     |
                                                |  - v^w_s: 3D Velocity (w)     |
                                                |  - p^w_s: 3D Position (w)     |
                                                |  - \delta\omega, \delta f: IMU Biases
                                                |  - R^v_s, p^s_v: Extrinsics   |
                                                |  - P: State Covariance (21x21)|
                                                +-------------------------------+
```

---

## 2. Module 1: AVNet (Data-Driven Measurement Estimator)

AVNet replaces physical vehicle wheel odometers and steering angle sensors with neural virtual sensors that learn vehicle motion dynamics from raw smartphone inertial measurements.

### 2.1 Internal Architecture
AVNet uses a unified CNN-GRU backbone with dual regression heads:
1. **Input Reshape**: Formats sliding window into `(Batch, 6, 200)`.
2. **Conv1D Block 1**: `Conv1D(in=6, out=128, kernel=11, stride=1, padding=valid)` $\to$ `ReLU()` $\to$ `MaxPool1D(kernel=2, stride=2)` $\to$ Output shape: `(Batch, 128, 95)`.
3. **Conv1D Block 2**: `Conv1D(in=128, out=256, kernel=9, stride=1, padding=valid)` $\to$ `ReLU()` $\to$ `MaxPool1D(kernel=2, stride=2)` $\to$ Output shape: `(Batch, 256, 43)`.
4. **Spatial Feature Projection**: `Flatten()` to $256 \times 43 = 11,008$ $\to$ `Linear(11008, 1024)` + `ReLU()` $\to$ `Linear(1024, 512)` + `ReLU()`.
5. **Recurrent Temporal Block**: 2-layer `GRU(input_size=512, hidden_size=64, batch_first=True)` extracting hidden sequence dynamics $h_{\text{last}} \in \mathbb{R}^{64}$.
6. **Task Regression Heads**:
   - **DDODO Head**: `Linear(64, 1)` $\to$ scalar forward speed $\hat{v}^v_{v,\text{lon}}$.
   - **DDATT Head**: `Linear(64, 3)` $\to$ relative quaternion attitude vector $\Delta \hat{q}_{x,y,z}$.

```
Input IMU Window (200x6) 
  ──► Conv1D (11x1, 128) ──► MaxPool (2) 
  ──► Conv1D (9x1, 256)  ──► MaxPool (2) 
  ──► FC (11008 -> 1024 -> 512) 
  ──► 2-Layer GRU (hidden=64) 
         ├──► Regressor Head 1 (64 -> 1) ──► DDODO: Longitudinal Speed (v_lon)
         └──► Regressor Head 2 (64 -> 3) ──► DDATT: Attitude Change (dq_x, dq_y, dq_z)
```

### 2.2 Input Specification
* **Tensor Shape**: `(Batch, 200, 6)`
* **Sampling Rate & Window**: $W_{\text{est}} = 200$ samples (1.0 second history window sampled at 200 Hz).
* **Channels (6-DOF IMU)**:
  1. $\tilde{\omega}_x$: Smartphone angular velocity along body X-axis ($\text{rad/s}$)
  2. $\tilde{\omega}_y$: Smartphone angular velocity along body Y-axis ($\text{rad/s}$)
  3. $\tilde{\omega}_z$: Smartphone angular velocity along body Z-axis ($\text{rad/s}$)
  4. $\tilde{f}_x$: Smartphone specific force along body X-axis ($\text{m/s}^2$)
  5. $\tilde{f}_y$: Smartphone specific force along body Y-axis ($\text{m/s}^2$)
  6. $\tilde{f}_z$: Smartphone specific force along body Z-axis ($\text{m/s}^2$)

### 2.3 Output Specification
* **Execution Frequency**: 1 Hz (runs once every 200 IMU samples).
* **Head 1: DDODO (Data-Driven Odometry)**:
  * **Tensor Shape**: `(Batch, 1)`
  * **Value**: Forward longitudinal vehicle velocity $\hat{v}^v_{v,\text{lon}}(t_n)$ in meters per second ($\text{m/s}$).
* **Head 2: DDATT (Data-Driven Attitude)**:
  * **Tensor Shape**: `(Batch, 3)`
  * **Value**: Imaginary components of relative rotation quaternion $\Delta \hat{q} = (\Delta q_x, \Delta q_y, \Delta q_z)$.
  * **Reconstruction**: The scalar real part is computed via unit quaternion constraint:
    $$\Delta q_w = \sqrt{\max\left(0,\, 1 - (\Delta q_x^2 + \Delta q_y^2 + \Delta q_z^2)\right)}$$
  * **Attitude Measurement Formulation**: The full orientation measurement relative to world frame at window end $t_n$ is:
    $$\tilde{R}^w_s(t_n) = R^w_s(t_{n-W_{\text{est}}}) \cdot \Delta R(\Delta q)$$

### 2.4 Optimization Loss Functions
* **Attitude Loss**: $L_{R^w_s} = \frac{1}{N} \sum_{i=1}^N \|\Delta q_i^{\text{pred}} - \Delta q_i^{\text{true}}\|^2$
* **Velocity Loss**: $L_{v^v} = \frac{1}{N} \sum_{i=1}^N (v_i^{\text{pred}} - v_i^{\text{true}})^2$

---

## 3. Module 2: AdapterNet (Data-Driven Filter Parameter Adapter)

AdapterNet eliminates manual empirical noise tuning by predicting instantaneous process covariance $Q(t_n)$ and measurement covariance $N(t_{n+1})$ based on current road vibrations and vehicle motion states.

### 3.1 Internal Architecture
A 2-layer dilated 1D CNN with increasing receptive field:
1. **Conv1D Layer 1**: `Conv1D(in=6, out=32, kernel=5, dilation=1, padding=2)` + `ReplicationPad1D` + `ReLU()` + `Dropout(0.2)`.
2. **Conv1D Layer 2**: `Conv1D(in=32, out=32, kernel=5, dilation=3, padding=6)` + `ReplicationPad1D` + `ReLU()` + `Dropout(0.2)`.
3. **Linear Layer**: `Linear(32, 6)` $\to$ parameter vector $[\mathbf{q}, \mathbf{n}] \in \mathbb{R}^6$.

### 3.2 Input Specification
* **Tensor Shape**: `(Batch, 20, 6)`
* **Sampling Rate & Window**: $W_{\text{adapter}} = 20$ samples (0.1 second sliding window at 200 Hz).
* **Channels**: 6-axis raw IMU $[\tilde{\omega}_x, \tilde{\omega}_y, \tilde{\omega}_z, \tilde{f}_x, \tilde{f}_y, \tilde{f}_z]$.

### 3.3 Output Specification
* **Tensor Shape**: `(Batch, 6)`
* **Parameters**:
  1. $\mathbf{q} = [q_x, q_y, q_z]^\top \in \mathbb{R}^3$: Process noise scaling exponents for gyroscope and accelerometer.
  2. $\mathbf{n} = [n_x, n_y, n_z]^\top \in \mathbb{R}^3$: Measurement noise scaling exponents for attitude and velocity updates.
* **Covariance Transformation Law**:
  Covariances are dynamically scaled over a $[10^{-3}, 10^{+3}]$ range centered on baseline standard deviations with $\beta = 3$:
  $$Q_{s,a}(t_n) = (\sigma_{s,a}^{\text{default}})^2 \cdot 10^{\beta \tanh(q_{s,a})}, \quad a \in \{x, y, z\}$$
  $$N_{s,a}(t_{n+1}) = (\sigma_{s,a}^{\text{default}})^2 \cdot 10^{\beta \tanh(n_{s,a})}, \quad a \in \{x, y, z\}$$

---

## 4. Module 3: Invariant Extended Kalman Filter (InEKF)

The InEKF is formulated on matrix Lie groups $GSE_2(3) \times GSO(3) \times \mathbb{R}^9$. Right-invariant errors ensure state-independent error dynamics, rapid convergence, and geometric consistency.

### 4.1 State Manifold Tuple (21 Dimensions)
$$\mathbf{x} = \left(\chi^w_s,\, \delta\omega^s_s,\, \delta f^s_s,\, R^v_s,\, p^s_v\right)$$
where:
$$\chi^w_s = \begin{pmatrix} R^w_s & v^w_s & p^w_s \\ \mathbf{0}_{1\times 3} & 1 & 0 \\ \mathbf{0}_{1\times 3} & 0 & 1 \end{pmatrix} \in GSE_2(3)$$

* $R^w_s \in SO(3)$: 3D Attitude matrix mapping sensor body frame to local East-North-Up (ENU) world frame.
* $v^w_s \in \mathbb{R}^3$: 3D Sensor velocity in world frame $(\text{m/s})$.
* $p^w_s \in \mathbb{R}^3$: 3D Sensor position in world frame $(\text{m})$.
* $\delta\omega^s_s \in \mathbb{R}^3$: Tri-axial gyroscope quasi-constant bias $(\text{rad/s})$.
* $\delta f^s_s \in \mathbb{R}^3$: Tri-axial accelerometer quasi-constant bias $(\text{m/s}^2)$.
* $R^v_s \in SO(3)$: Extrinsic mounting rotation matrix mapping sensor frame to vehicle chassis frame.
* $p^s_v \in \mathbb{R}^3$: Lever-arm vector from vehicle origin to sensor origin in sensor frame $(\text{m})$.

---

### 4.2 Step 1: Propagation (Runs at Full IMU Rate: 200 Hz)

#### **Inputs to Propagation**:
1. Instantaneous gyroscope observation: $\tilde{\omega}(t_n) \in \mathbb{R}^3$
2. Instantaneous accelerometer observation: $\tilde{f}(t_n) \in \mathbb{R}^3$
3. Time step interval: $dt$ ($0.005\ \text{s}$ at 200 Hz)
4. Dynamic process noise covariance matrix: $Q(t_n) \in \mathbb{R}^{18 \times 18}$ (from AdapterNet)

#### **Propagation Dynamics**:
$$\hat{R}^w_s(t_{n+1}^-) = \hat{R}^w_s(t_n) \exp\left((\tilde{\omega} - \hat{\delta\omega}) dt\right)_\times$$
$$\hat{v}^w_s(t_{n+1}^-) = \hat{v}^w_s(t_n) + \left(\hat{R}^w_s(t_n)(\tilde{f} - \hat{\delta f}) + g^w\right) dt$$
$$\hat{p}^w_s(t_{n+1}^-) = \hat{p}^w_s(t_n) + \hat{v}^w_s(t_n) dt$$

#### **Covariance Propagation**:
$$P(t_{n+1}^-) = F P(t_n) F^\top + G Q(t_n) G^\top$$
where $F \in \mathbb{R}^{21 \times 21}$ is the state transition Jacobian and $G \in \mathbb{R}^{21 \times 18}$ is the noise input Jacobian.

---

### 4.3 Step 2: Measurement Update (Runs at 1 Hz – 10 Hz)

The filter executes a **6-Dimensional (6D) Geometric Measurement Update** by stacking 3D attitude, 1D forward speed, and 2D Non-Holonomic Constraints (DDNHC):

#### **Inputs to Measurement Update**:
1. **Learned Attitude Measurement**: $\tilde{R}^w_s(t_{n+1}) \in SO(3)$ (from AVNet DDATT)
2. **Learned Forward Velocity**: $\tilde{v}^v_{\text{lon}}(t_{n+1}) \in \mathbb{R}^1$ (from AVNet DDODO)
3. **Non-Holonomic Constraints (DDNHC)**: Zero lateral and vertical velocity $\tilde{v}^v_{\text{lat}} = 0, \tilde{v}^v_{\text{up}} = 0$
4. **Measurement Noise Covariance**: $N(t_{n+1}) \in \mathbb{R}^{6 \times 6}$ (from AdapterNet)

#### **Measurement Vectors**:
$$\tilde{y}_{R^w_s} = \tilde{R}^w_s, \quad \tilde{y}_{v^v_v} = \begin{pmatrix} 0 \\ \tilde{v}^v_{\text{lon}} \\ 0 \end{pmatrix}$$

#### **Measurement Residuals**:
* **Attitude Residual ($3 \times 1$)**:
  $$r_{R^w_s} = \log_{SO(3)}\left(\tilde{R}^w_s (\hat{R}^w_s)^\top\right)$$
* **Velocity Residual ($3 \times 1$)**:
  $$r_{v^v_v} = \tilde{y}_{v^v_v} - \hat{R}^v_s \left( (\hat{R}^w_s)^\top \hat{v}^w_s + (\omega^s_s)_\times \hat{p}^s_v \right)$$

#### **Kalman Gain & Lie Group Update**:
$$S = H P^- H^\top + N, \quad K = P^- H^\top S^{-1}$$
$$\mathbf{e}^+ = K \begin{pmatrix} r_{R^w_s} \\ r_{v^v_v} \end{pmatrix} \in \mathbb{R}^{21}$$
$$\hat{\chi}^w_s(t_{n+1}^+) = \exp_{GSE_2(3)}(\xi^w_s) \cdot \hat{\chi}^w_s(t_{n+1}^-)$$
$$\hat{R}^v_s(t_{n+1}^+) = \exp_{SO(3)}(\xi_{R^v_s}) \cdot \hat{R}^v_s(t_{n+1}^-)$$
$$\hat{\delta\omega}^+ = \hat{\delta\omega}^- + \xi_{\delta\omega}, \quad \hat{\delta f}^+ = \hat{\delta f}^- + \xi_{\delta f}, \quad \hat{p}^s_v{}^+ = \hat{p}^s_v{}^- + \xi_{p^s_v}$$
$$P^+ = (I_{21} - K H) P^-$$

---

## 5. Summary Input/Output Matrix

| Sub-System / Model | Execution Rate | Exact Inputs | Exact Outputs | Primary Function |
| :--- | :---: | :--- | :--- | :--- |
| **AVNet DDODO** | 1 Hz | `(B, 200, 6)` IMU window: $[\tilde{\omega}_{x,y,z}, \tilde{f}_{x,y,z}]$ | `(B, 1)`: Scalar forward speed $\hat{v}^v_{\text{lon}}$ (m/s) | Virtual forward wheel odometry |
| **AVNet DDATT** | 1 Hz | `(B, 200, 6)` IMU window: $[\tilde{\omega}_{x,y,z}, \tilde{f}_{x,y,z}]$ | `(B, 3)`: Quaternion delta vector $\Delta q_{x,y,z}$ | Gyro drift correction & attitude lock |
| **AdapterNet** | 10 Hz – 200 Hz | `(B, 20, 6)` IMU window: $[\tilde{\omega}_{x,y,z}, \tilde{f}_{x,y,z}]$ | `(B, 6)`: Exponents $\mathbf{q} \in \mathbb{R}^3, \mathbf{n} \in \mathbb{R}^3$ | Dynamic covariance scaling ($Q, N$) |
| **InEKF Propagation** | 200 Hz | Instantaneous $[\tilde{\omega}, \tilde{f}, dt]$ + $Q(t_n)$ | Prior state $\hat{\mathbf{x}}^-$ & covariance $P^-$ | High-rate dead-reckoning state integration |
| **InEKF Update** | 1 Hz – 10 Hz | DDATT $\tilde{R}^w_s$, DDODO $\tilde{v}^v_{\text{lon}}$, NHC, $N(t_{n+1})$ | Posterior state $\hat{\mathbf{x}}^+$ & covariance $P^+$ | 6D geometric constraint correction |
| **Final Navigation Engine** | 10 Hz – 200 Hz | Filter state tuple $\hat{\mathbf{x}}^+$ | $[p^w_s, v^w_s, R^w_s, \delta\omega, \delta f]$ | Real-time vehicle trajectory output |

---

## 6. Multi-Rate Operational Schedule

```
Timeline (ms)   0ms     5ms     10ms    15ms    ...   100ms   ...   1000ms (1.0s)
--------------------------------------------------------------------------------------
Raw IMU (200Hz)  [#1]    [#2]    [#3]    [#4]    ...   [#20]   ...   [#200]
InEKF Propagate  ──►     ──►     ──►     ──►     ...   ──►     ...   ──►
AdapterNet (Q,N)                                       [EXEC]  ...   [EXEC]
AVNet (ODO+ATT)                                                      [EXEC (1 Hz)]
InEKF 6D Update                                                      [UPDATE (1 Hz)]
Position Output  ──► (10 Hz - 200 Hz continuous vehicle pose stream) ───────────────►
```
