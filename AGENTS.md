# AVNet: Architecture, Theory & Agent Implementation Guide

This document provides a comprehensive reference of the architectural, mathematical, and algorithmic aspects of **AVNet (Data- and Model-Driven Vehicle Dead Reckoning - DMDVDR)** based on the research publication:

> **"Avnet: learning attitude and velocity for vehicular dead reckoning using smartphone by adapting an invariant EKF"**  
> *Long Qian, Xinchuang Lin, Xiaoguang Niu, Qihai Huang, Leilei Li, Guangyi Guo, Zexin Wang, and Ruizhi Chen*  
> *Satellite Navigation (2025) 6:15* — [DOI: 10.1186/s43020-025-00168-7](https://doi.org/10.1186/s43020-025-00168-7)

---

## 1. System Overview & Problem Statement

### 1.1 Challenge in Smartphone VDR
Vehicular Dead Reckoning (VDR) based exclusively on smartphone-grade MEMS IMUs (e.g., STMicroelectronics LSM6DSM) faces severe error drift due to:
- High noise density and non-stationary quasi-constant sensor biases ($\delta\omega, \delta f$).
- Unobservability of yaw drift and scale errors during long stationary periods and low-speed maneuvers.
- Limitations of traditional Non-Holonomic Constraints (NHC) and Zero Velocity Updates (ZUPT), which break down or require fragile empirical heuristic threshold detectors.

### 1.2 DMDVDR Solution Framework
The **Combined Data- and Model-Driven Vehicle Dead Reckoning (DMDVDR)** framework integrates:
1. **Model-Driven Module**: Continuous Strapdown Inertial Navigation System (INS) mechanization and vehicle kinematic constraints.
2. **Data-Driven Measurement Estimator (AVNet)**: A hybrid CNN-GRU neural network extracting:
   - **DDODO**: Data-Driven longitudinal forward velocity pseudo-measurements ($v^v_{v,\text{lon}}$).
   - **DDATT**: Data-Driven relative attitude/orientation quaternion updates ($\Delta q$).
3. **Data-Driven Filter Parameter Adapter (AdapterNet)**: A 1D dilated CNN dynamically estimating process noise covariance $Q(t_n)$ and measurement noise covariance $N(t_{n+1})$.
4. **Lie-Group Fusion Engine (InEKF)**: A right-invariant Extended Kalman Filter defined on $SE_2(3) \times SO(3) \times \mathbb{R}^9$ achieving geometric consistency, rapid convergence, and coordinate-frame invariance.

```
       +-----------------------------------------------------------+
       |                  Raw Smartphone IMU                       |
       |             \tilde{\omega}(t), \tilde{f}(t)               |
       +-----------------------------+-----------------------------+
                                     |
               +---------------------+---------------------+
               |                                           |
               v                                           v
    +----------------------+                    +----------------------+
    | Model-Driven Module  |                    |  Data-Driven Module  |
    |  - INS Mechanization |                    |  - AVNet (DDODO+ATT) |
    |  - Kinematics        |                    |  - AdapterNet (Q, N) |
    +----------+-----------+                    +----------+-----------+
               |                                           |
               |       Propagation         Update          |
               +-------------> [ InEKF Engine ] <----------+
                                     |
                                     v
                        Vehicle Extended State:
                     R^w_s, v^w_s, p^w_s, Biases,
                       Extrinsics (R^v_s, p^s_v)
```

---

## 2. Data-Driven Module Architecture

### 2.1 AVNet: Hybrid CNN-GRU Measurement Estimator

AVNet combines convolutional representation learning with recurrent temporal dynamics over a sliding window of IMU observations:

- **Input**: Window $W_{\text{est}} = 200$ samples (1.0 s at 200 Hz) of 6-axis IMU data:
  $$\mathbf{X} = \{(\tilde{\omega}_k, \tilde{f}_k)\}_{k=n-W_{\text{est}}+1}^n \in \mathbb{R}^{200 \times 6}$$
- **Output**:
  - **DDODO (Velocity)**: $\hat{v}^v_{v,\text{lon}}(t_n) \in \mathbb{R}^1$ (scalar forward speed).
  - **DDATT (Attitude Change)**: $\Delta \hat{q} = (\Delta q_x, \Delta q_y, \Delta q_z) \in \mathbb{R}^3$.  
    The scalar component is reconstructed via unit quaternion constraint:
    $$\Delta q_w = \sqrt{\max\left(0,\, 1 - (\Delta q_x^2 + \Delta q_y^2 + \Delta q_z^2)\right)}$$
    Full rotation update from window start $t_{n-W_{\text{est}}}$ to current time $t_n$:
    $$\tilde{R}^w_s(t_n) = R^w_s(t_{n-W_{\text{est}}}) \cdot \Delta R(\Delta q)$$

#### Detailed Layer Architecture:
1. **Input Reshape / Transpose**: $(B, 200, 6) \to (B, 6, 200)$
2. **Conv1D Block 1**:
   - `Conv1D(in_channels=6, out_channels=128, kernel_size=11, stride=1, padding=valid)`
   - `ReLU()`
   - `MaxPool1d(kernel_size=2, stride=2)` $\to$ output length 95
3. **Conv1D Block 2**:
   - `Conv1D(in_channels=128, out_channels=256, kernel_size=9, stride=1, padding=valid)`
   - `ReLU()`
   - `MaxPool1d(kernel_size=2, stride=2)` $\to$ output length 43
4. **Spatial Feature Projection**:
   - `Flatten()` $\to 256 \times 43 = 11,008$
   - `Linear(11008, 1024)` + `ReLU()`
   - `Linear(1024, 512)` + `ReLU()`
5. **Recurrent Sequence Module**:
   - `GRU(input_size=512, hidden_size=64, num_layers=2, batch_first=True)`
   - Output from last recurrent step: $h_{\text{last}} \in \mathbb{R}^{64}$
6. **Regression Heads**:
   - `Linear(64, 1)` for DDODO ($v^v_{\text{lon}}$)
   - `Linear(64, 3)` for DDATT ($\Delta q_{x,y,z}$)

#### Loss Functions:
- **DDATT Loss**:
  $$L_{R^w_s} = \frac{1}{N} \sum_{i=1}^N \|\Delta q_i^{\text{pred}} - \Delta q_i^{\text{true}}\|^2$$
- **DDODO Loss**:
  $$L_{v^v} = \frac{1}{N} \sum_{i=1}^N (v_i^{\text{pred}} - v_i^{\text{true}})^2$$

---

### 2.2 AdapterNet: Filter Parameter Adapter

Instead of manually tuning heuristic noise parameters, **AdapterNet** dynamically calculates the filter process covariance $Q(t_n)$ and measurement covariance $N(t_{n+1})$ directly from raw IMU signals.

- **Input**: Window $W_{\text{adapter}} = 20$ samples (0.1 s at 200 Hz). Output rate is 200 Hz.
- **Architecture**: 2-layer dilated 1D CNN:
  1. `Conv1D(in_channels=6, out_channels=32, kernel_size=5, dilation=1, padding=2)` + `ReplicationPad1D` + `ReLU()` + `Dropout(0.2)`
  2. `Conv1D(in_channels=32, out_channels=32, kernel_size=5, dilation=3, padding=6)` + `ReplicationPad1D` + `ReLU()` + `Dropout(0.2)`
  3. `Linear(32, 6)` outputting parameter vectors:
     $$\mathbf{q} \in \mathbb{R}^6, \quad \mathbf{n} \in \mathbb{R}^6$$

#### Covariance Adaptation Law:
Covariance elements are scaled exponentially around nominal baselines using hyperbolic tangent:
$$Q_{s,a} = (\sigma_{s,a}^{\text{default}})^2 \cdot 10^{\beta \tanh(q_{s,a})}, \quad a \in \{x, y, z\}$$
$$N_{s,a} = (\sigma_{s,a}^{\text{default}})^2 \cdot 10^{\beta \tanh(n_{s,a})}, \quad a \in \{x, y, z\}$$
where $\beta = 3$, providing a dynamic adjustment range of $[10^{-3}, 10^{+3}]$ relative to nominal standard deviations:
- Process noise defaults:
  - $\sigma_{\omega} = 1 \times 10^{-2}\ \text{rad/s}$
  - $\sigma_f = 1 \times 10^{-2}\ \text{m/s}^2$
  - $\sigma_{\delta\omega} = 1 \times 10^{-4}\ \text{rad/s}$
  - $\sigma_{\delta f} = 1 \times 10^{-3}\ \text{m/s}^2$
  - $\sigma_{R^v_s} = 1 \times 10^{-4}\ \text{rad}$
  - $\sigma_{p^s_v} = 1 \times 10^{-4}\ \text{m}$
- Measurement noise defaults:
  - $\sigma_{R^w_s} = 1 \times 10^{-2}\ \text{rad}$
  - $\sigma_{v^v_s} = 1.0\ \text{m/s}$

---

## 3. Model-Driven Module: Invariant Extended Kalman Filter (InEKF)

### 3.1 Coordinate Systems
- **World Frame ($w$)**: Local East-North-Up (ENU) tangent plane, flat-Earth gravity vector $g^w = [0, 0, -9.80]^\top\ \text{m/s}^2$.
- **Sensor Frame ($s$)**: Tri-axial smartphone IMU body frame.
- **Vehicle Frame ($v$)**: Forward-lateral-up vehicle chassis frame.
  - $R^v_s \in SO(3)$: mounting orientation mapping sensor to vehicle.
  - $p^s_v \in \mathbb{R}^3$: lever arm vector from vehicle origin to sensor origin in sensor frame.

### 3.2 State Vector & Lie Group Manifold
The system state is defined as a 21-dimensional manifold tuple:
$$\mathbf{x} = \left(\chi^w_s,\, \delta\omega^s_s,\, \delta f^s_s,\, R^v_s,\, p^s_v\right)$$

Where $\chi^w_s$ is embedded in the Lie group $GSE_2(3)$:
$$\chi^w_s = \begin{pmatrix} R^w_s & v^w_s & p^w_s \\ \mathbf{0}_{1\times 3} & 1 & 0 \\ \mathbf{0}_{1\times 3} & 0 & 1 \end{pmatrix} \in GSE_2(3)$$

### 3.3 Right-Invariant Error Formulation
The state error $\mathbf{e} \in \mathbb{R}^{21}$ is defined using right-invariant errors on Lie groups:
$$\eta_{\chi^w_s} = \hat{\chi}^w_s (\chi^w_s)^{-1} = \exp_{GSE_2(3)}(\xi^w_s)$$
$$\eta_{R^v_s} = \hat{R}^v_s (R^v_s)^{-1} = \exp_{GSO(3)}(\xi_{R^v_s})$$
$$\xi_{\delta\omega} = \delta\omega - \hat{\delta\omega}, \quad \xi_{\delta f} = \delta f - \hat{\delta f}, \quad \xi_{p^s_v} = p^s_v - \hat{p}^s_v$$

Explicit right-invariant error components:
$$\eta_{R^w_s} = \hat{R}^w_s (R^w_s)^\top = \exp_{SO(3)}(\xi_{R^w_s})$$
$$\xi_{v^w_s} = \hat{v}^w_s - \hat{R}^w_s (R^w_s)^\top v^w_s$$
$$\xi_{p^w_s} = \hat{p}^w_s - \hat{R}^w_s (R^w_s)^\top p^w_s$$

### 3.4 Propagation Dynamics
Between IMU samples with zero-order hold interval $dt$:
$$\hat{R}^w_s(t_{n+1}^-) = \hat{R}^w_s(t_n) \exp\left((\tilde{\omega} - \hat{\delta\omega}) dt\right)_\times$$
$$\hat{v}^w_s(t_{n+1}^-) = \hat{v}^w_s(t_n) + \left(\hat{R}^w_s(t_n)(\tilde{f} - \hat{\delta f}) + g^w\right) dt$$
$$\hat{p}^w_s(t_{n+1}^-) = \hat{p}^w_s(t_n) + \hat{v}^w_s(t_n) dt$$

#### Error State Transition Matrix $F \in \mathbb{R}^{21 \times 21}$:
$$F = I_{21} + \begin{pmatrix}
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \hat{R}^w_s & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
(g^w)_\times & \mathbf{0}_3 & \mathbf{0}_3 & (\hat{v}^w_s)_\times \hat{R}^w_s & \hat{R}^w_s & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & I_3 & \mathbf{0}_3 & (\hat{p}^w_s)_\times \hat{R}^w_s & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_{12 \times 21}
\end{pmatrix} dt$$

#### Noise Input Matrix $G \in \mathbb{R}^{21 \times 18}$:
$$G = \begin{pmatrix}
\hat{R}^w_s & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
(\hat{v}^w_s)_\times \hat{R}^w_s & \hat{R}^w_s & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
(\hat{p}^w_s)_\times \hat{R}^w_s & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & I_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & I_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \hat{R}^v_s & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & I_3
\end{pmatrix} dt$$

Covariance propagation:
$$P(t_{n+1}^-) = F P(t_n) F^\top + G Q(t_n) G^\top$$

---

### 3.5 Six-Dimensional (6D) Geometric Measurement Update

The update combines DDATT (3D attitude), DDODO (1D forward speed), and DDNHC (2D zero lateral/vertical velocity constraints):

$$y(t_{n+1}) = \begin{pmatrix} y_{R^w_s} \\ y_{v^v_v} \end{pmatrix}, \quad y_{v^v_v} = \begin{pmatrix} v^v_{\text{lat}} \approx 0 \\ v^v_{\text{lon}} (\text{DDODO}) \\ v^v_{\text{up}} \approx 0 \end{pmatrix}$$

#### Measurement Residuals:
- **Attitude Residual**:
  $$r_{R^w_s} = \log_{SO(3)}\left(\tilde{R}^w_s (\hat{R}^w_s)^\top\right)$$
- **Velocity Residual**:
  $$r_{v^v_v} = \tilde{y}_{v^v_v} - \hat{R}^v_s \left( (\hat{R}^w_s)^\top \hat{v}^w_s + (\omega^s_s)_\times \hat{p}^s_v \right)$$

#### Measurement Jacobian $H \in \mathbb{R}^{6 \times 21}$:
- **Attitude Rows ($3 \times 21$)**:
  $$H_{R^w_s} = \begin{pmatrix} I_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \end{pmatrix}$$
- **Velocity Rows ($3 \times 21$)**:
  $$H_{v^v_v} = \begin{pmatrix} \mathbf{0}_3 & \hat{R}^v_s (\hat{R}^w_s)^\top & \mathbf{0}_3 & \hat{R}^v_s (\hat{p}^s_v)_\times & \mathbf{0}_3 & H_{R^v_s} & \hat{R}^v_s (\omega^s_s)_\times \end{pmatrix}$$
  where:
  $$H_{R^v_s} = -\left( \hat{R}^v_s \left( (\hat{R}^w_s)^\top \hat{v}^w_s + (\omega^s_s)_\times \hat{p}^s_v \right) \right)_\times$$

#### Kalman Update:
$$S = H P^- H^\top + N$$
$$K = P^- H^\top S^{-1}$$
$$\mathbf{e}^+ = K \begin{pmatrix} r_{R^w_s} \\ r_{v^v_v} \end{pmatrix}$$
$$\hat{\chi}^w_s(t_{n+1}^+) = \exp_{GSE_2(3)}(\xi^w_s) \cdot \hat{\chi}^w_s(t_{n+1}^-)$$
$$\hat{R}^v_s(t_{n+1}^+) = \exp_{SO(3)}(\xi_{R^v_s}) \cdot \hat{R}^v_s(t_{n+1}^-)$$
$$\hat{\delta\omega}^+ = \hat{\delta\omega}^- + \xi_{\delta\omega}, \quad \hat{\delta f}^+ = \hat{\delta f}^- + \xi_{\delta f}, \quad \hat{p}^s_v{}^+ = \hat{p}^s_v{}^- + \xi_{p^s_v}$$
$$P^+ = (I_{21} - K H) P^-$$

---

## 4. Key Performance Characteristics & Benchmarks

From the paper's experiments across 11 test sequences on the IO-VNBD dataset (Huawei Mate 30 vs. NovAtel SPAN ISA-100C Ground Truth):

| Method | Mean $E_{rrel}$ ($^\circ/\text{km}$) | Mean $E_{trel}$ (%) | Mean $E_{thor}$ (%) | Key Failure Mode |
| :--- | :---: | :---: | :---: | :--- |
| **AI-IMU** | $180^\circ$ | $23,400\%$ | $5,600\%$ | Diverges during stationary periods & turns |
| **DeepOdo** | $145^\circ$ | $16.1\%$ | $15.5\%$ | Heading diverges on consecutive 90° turns |
| **DeepOri** | $20.3^\circ$ | $4.42\%$ | $3.57\%$ | Position drift accumulates without speed lock |
| **Proposed AVNet (DMDVDR)** | **$1.85^\circ$** | **$2.15\%$** | **$0.40\%$** | **Maintains sub-1% drift across all modes** |

### Critical Findings:
1. **Attitude Dominance**: Smartphone gyroscope errors are the primary driver of trajectory drift. Constraining attitude with DDATT reduces horizontal drift by over an order of magnitude.
2. **Stationary State Handling**: Unlike conventional INS requiring hard-coded ZUPT/ZIHR detectors, continuous DDATT and DDODO updates naturally freeze integration drift during halts.
3. **Tunnel Navigation**: Evaluated on Google Smartphone Decimeter Challenge (GSDC) Los Angeles LAX tunnel trajectory (578 m tunnel, 55 s GNSS outage): achieved **0.64% translation drift** where classical WLS GNSS solutions completely failed.

---

## 5. Codebase Directory Mapping

- [`avnet/models/avnet.py`](file:///c:/MY%20FILES/sih2026/avnet/avnet/models/avnet.py):
  - `AVNet`: PyTorch CNN-GRU model for DDODO ($v^v_{\text{lon}}$) and DDATT ($\Delta q$).
  - `AdapterNet`: 2-layer dilated CNN for dynamic noise covariance scaling ($Q, N$).
- [`avnet/models/inekf.py`](file:///c:/MY%20FILES/sih2026/avnet/avnet/models/inekf.py):
  - `InEKF`: Full $SE_2(3)$ Lie Group Invariant Extended Kalman Filter implementation with analytical Jacobians ($F, G, H$).
- [`avnet/dataset.py`](file:///c:/MY%20FILES/sih2026/avnet/avnet/dataset.py):
  - `load_iovnbd_csv`: CSV parser for IO-VNBD datasets.
  - `latlon_to_enu`: WGS-84 Geodetic to local Cartesian ENU converter.
  - `AVNetDataset`: Sliding-window dataset builder with target speed and relative quaternion delta.
- [`avnet/train.py`](file:///c:/MY%20FILES/sih2026/avnet/avnet/train.py):
  - `train_avnet`: Training loop with Adam optimizer and MSE loss for speed and attitude.
- [`avnet/evaluate.py`](file:///c:/MY%20FILES/sih2026/avnet/avnet/evaluate.py):
  - `compute_ate`: Absolute Trajectory Error (ATE RMSE).
  - `compute_relative_errors`: KITTI benchmark relative translation ($E_{trel}$) and rotation ($E_{rrel}$) metrics.
- [`main.py`](file:///c:/MY%20FILES/sih2026/avnet/main.py):
  - Command-line runner executing training, inference, InEKF filtering, and trajectory evaluation.
- [`requirements.txt`](file:///c:/MY%20FILES/sih2026/avnet/requirements.txt):
  - Project dependencies (`numpy`, `scipy`, `pandas`, `torch`, `matplotlib`, `pytest`).
- [`.venv/`](file:///c:/MY%20FILES/sih2026/avnet/.venv):
  - Dedicated virtual environment for running and testing the codebase.
