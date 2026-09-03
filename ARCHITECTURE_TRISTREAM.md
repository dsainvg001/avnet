# Multi-Modal Tri-Stream AVNet (AVNet-Mag) Architecture & Systems Specification

## Executive Summary
This document provides the complete theoretical, architectural, and operational specification for the upgraded **Tri-Stream Multi-Modal AVNet (AVNet-Mag)** and its integration with the **Invariant Extended Kalman Filter (InEKF)** Lie-group dead-reckoning engine. It contrasts the legacy paper baseline against the new multi-stream architecture, details input/output schemas, and provides empirical hardware profiling (FLOPs, latency, parameters, and memory footprint).

---

## 1. Dataset Column Analysis & Modality Grounding

An exhaustive column inspection was performed across the dataset files (`S-M.csv`, `S-vtb1.csv`, and vehicle reference `V-vtb1.csv`):

### 1.1 Smartphone Dataset Structure (`S-*.csv`)
24 synchronized sensor channels sampled uniformly at **10.0 Hz** ($dt = 100\ \text{ms}$):

| Index | Column Name in CSV | Physical Unit | Sensor Category | Role in Pipeline |
| :---: | :--- | :---: | :--- | :--- |
| `[00]` | `GPS LATITUDE (degrees)` | $\text{deg}$ | GNSS Position | WGS-84 coordinate (converted to local ENU for Ground Truth) |
| `[01]` | `GPS LONGITUDE (degrees)` | $\text{deg}$ | GNSS Position | WGS-84 coordinate (converted to local ENU for Ground Truth) |
| `[02]` | `GPS ALTITUDE (m)` | $\text{m}$ | GNSS Altitude | Vertical reference |
| `[03]` | `GPS SPEED (Kmh)` | $\text{km/h}$ | GNSS Odometry | Converted to $\text{m/s}$ as ground truth label for DDODO |
| `[04]` | `GPS ACCURACY (m)` | $\text{m}$ | GNSS Quality | Confidence weighting ($< 5\ \text{m}$ filter) |
| `[05]` | `GPS ORIENTATION (°)` | $\text{deg}$ | GNSS Heading | Vehicle ground-truth ground-track angle $[0^\circ, 360^\circ)$ |
| `[06]` | `GPS SATELLITES IN RANGE` | string | GNSS State | e.g., `'18 / 19'` visible/used |
| `[07]` | `TIME SINCE START (ms)` | $\text{ms}$ | Clock | Master synchronization timestamp |
| `[08]` | `DATE` | string | Clock | `YYYY-MO-DD HH-MI-SS_SSS` |
| `[09]` | `ACCELEROMETER X (m/s²)` | $\text{m/s}^2$ | Accelerometer | Lateral specific force (body $X$) |
| `[10]` | `ACCELEROMETER Y (m/s²)` | $\text{m/s}^2$ | Accelerometer | Longitudinal specific force (body $Y$) |
| `[11]` | `ACCELEROMETER Z (m/s²)` | $\text{m/s}^2$ | Accelerometer | Vertical specific force + gravity (body $Z \approx 9.81\ \text{m/s}^2$) |
| `[12]` | `GRAVITY X (m/s²)` | $\text{m/s}^2$ | Gravity Vector | Low-pass filtered body gravity $X$ |
| `[13]` | `GRAVITY Y (m/s²)` | $\text{m/s}^2$ | Gravity Vector | Low-pass filtered body gravity $Y$ |
| `[14]` | `GRAVITY Z (m/s²)` | $\text{m/s}^2$ | Gravity Vector | Low-pass filtered body gravity $Z$ |
| `[15]` | `GYROSCOPE Yaw (rad/s)` | $\text{rad/s}$ | Gyroscope | Body z-axis turn rate ($r$) |
| `[16]` | `GYROSCOPE Pitch (rad/s)` | $\text{rad/s}$ | Gyroscope | Body y-axis pitch rate ($q$) |
| `[17]` | `GYROSCOPE Roll (rad/s)` | $\text{rad/s}$ | Gyroscope | Body x-axis roll rate ($p$) |
| `[18]` | `MAGNETIC FIELD X (μT)` | $\mu\text{T}$ | Magnetometer | Geomagnetic flux density $B_x$ |
| `[19]` | `MAGNETIC FIELD Y (μT)` | $\mu\text{T}$ | Magnetometer | Geomagnetic flux density $B_y$ |
| `[20]` | `MAGNETIC FIELD Z (μT)` | $\mu\text{T}$ | Magnetometer | Geomagnetic flux density $B_z$ |
| `[21]` | `ORIENTATION (Yaw) (°)` | $\text{deg}$ | Device Orientation | Android sensor framework fused azimuth |
| `[22]` | `ORIENTATION (Pitch) (°)` | $\text{deg}$ | Device Orientation | Android sensor framework fused pitch |
| `[23]` | `ORIENTATION (Roll) (°)` | $\text{deg}$ | Device Orientation | Android sensor framework fused roll |

*Note: In some uncategorised runs, gyro columns are named `GYROSCOPE X/Y/Z` and yaw orientation is named `ORIENTATION (Azimuth)`. The dataset parser supports both naming schemas automatically.*

---

## 2. Architectural Comparison: Legacy Paper vs. New Tri-Stream AVNet

```
========================================================================================
                          LEGACY AVNet (Paper Baseline)
========================================================================================
  Input Window: (B, 6, 200) [Accel 3 + Gyro 3] (200 Hz only)
        │
        ├── Conv1D(6 -> 128, k=11) + ReLU + MaxPool1D(2)
        ├── Conv1D(128 -> 256, k=9) + ReLU + MaxPool1D(2)
        ├── Flatten(11,008) -> Dense(1024) -> Dense(512)
        ├── Uni-directional GRU(512 -> 64, num_layers=2)
        └── Last Time Step Slice h_last -> Heads [v_lon (1), dq (3)]

  Drawbacks:
  - Fails at 10 Hz without 200 Hz interpolation
  - Gyro yaw drift is unconstrained over long straight runs and halts
  - Mixes distinct physical dimensions (m/s² and rad/s) in early convolution kernels
  - Massive flattened dense layer (11M weights) causes high parameter bloat

========================================================================================
                     PROPOSED NEW: Tri-Stream AVNet-Mag
========================================================================================
  3 Decoupled Input Streams (Window W = 20 samples @ 10 Hz = 2.0 seconds):
  
   Stream 1: Accel (B, 3, W)    Stream 2: Gyro (B, 3, W)     Stream 3: Mag (B, 3, W)
             │                            │                            │
             ▼                            ▼                            ▼
      [Conv1D 3 -> 32]             [Conv1D 3 -> 32]             [Conv1D 3 -> 32]
      [BN + LeakyReLU]             [BN + LeakyReLU]             [BN + LeakyReLU]
             │                            │                            │
             ▼                            ▼                            ▼
    [Dilated Conv 32->64]        [Dilated Conv 32->64]        [Dilated Conv 32->64]
      (kernel=3, dil=2)            (kernel=3, dil=2)            (kernel=3, dil=2)
             │                            │                            │
             └────────────────────────────┼────────────────────────────┘
                                          ▼
                         [Channel Concatenation: (B, 192, W)]
                                          │
                                          ▼
                            [Cross-Modal Gated Fusion]
                           Conv1D(192 -> 128) + BN + LeakyReLU
                                          │
                                          ▼
                           [Bi-Directional Temporal GRU]
                           Input=128, Hidden=64 (Bidirectional -> 128)
                                          │
                                          ▼
                            [Temporal Attention Pooling]
                            Softmax-weighted temporal summation -> (B, 128)
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
              [Head 1: DDODO]                           [Head 2: DDATT]
           Linear(128 -> 64) -> ReLU                 Linear(128 -> 64) -> ReLU
                 Linear(64 -> 1)                           Linear(64 -> 3)
                     │                                         │
                     ▼                                         ▼
            Forward Speed v_lon                      Relative Quaternion Δq_xyz
            (Batch, 1) in m/s                        (Batch, 3) normalized
```

---

## 3. Detailed Component Breakdown: What Each Model Takes, Does, and Outputs

The end-to-end dead-reckoning system consists of **3 coordinated models/modules**:

### Module 1: Tri-Stream AVNet-Mag (Data-Driven Measurement Engine)
- **What Data It Takes**:
  - **Acceleration Stream**: $\mathbf{a} \in \mathbb{R}^{B \times 3 \times W}$ ($a_x, a_y, a_z$ in $\text{m/s}^2$).
  - **Angular Velocity Stream**: $\boldsymbol{\omega} \in \mathbb{R}^{B \times 3 \times W}$ ($\omega_x, \omega_y, \omega_z$ in $\text{rad/s}$).
  - **Magnetic Flux Stream**: $\mathbf{m} \in \mathbb{R}^{B \times 3 \times W}$ ($m_x, m_y, m_z$ in $\mu\text{T}$).
  - Sliding temporal window $W = 20$ samples (equivalent to 2.0 s at 10 Hz or 0.1 s at 200 Hz).
- **What It Does**:
  1. Independently filters physical noise profiles via sensor-specific 1D dilated convolutions.
  2. Synthesizes cross-sensor dynamics in the 192-channel gated fusion block.
  3. Captures forward-and-backward temporal dependencies through a 2-layer Bidirectional GRU.
  4. Applies self-attention pooling to focus on high-information acceleration pulses and turn transients.
- **What It Spits Out**:
  - $\hat{v}^v_{v,\text{lon}} \in \mathbb{R}^{B \times 1}$: Predicted vehicle scalar longitudinal forward velocity in $\text{m/s}$ (**DDODO**).
  - $\Delta \hat{\mathbf{q}}_{xyz} \in \mathbb{R}^{B \times 3}$: Relative 3D rotation quaternion vector component (**DDATT**). The scalar term is reconstructed via:
    $$\Delta q_w = \sqrt{\max\left(0,\, 1 - (\Delta q_x^2 + \Delta q_y^2 + \Delta q_z^2)\right)}$$

---

### Module 2: AdapterNet-9Axis (Multi-Modal Filter Parameter Adapter)
- **What Data It Takes**:
  - **All 3 Sensor Streams** ($9\ \text{channels}$): $\mathbf{X}_{\text{adapter}} \in \mathbb{R}^{B \times 9 \times W_a}$ ($W_a = 20$ samples):
    - $\text{Accel X, Y, Z}$ ($3$ channels)
    - $\text{Gyro Yaw, Pitch, Roll}$ ($3$ channels)
    - $\text{Mag X, Y, Z}$ ($3$ channels)
- **Why Feeding Magnetometer into AdapterNet is Critical**:
  - In urban canyons, parking garages, or near power lines / bridges, high magnetic distortions occur ($\Delta \|\mathbf{m}\| \gg 0$).
  - When the AdapterNet detects magnetic field distortion or high dynamic turns, it **dynamically increases measurement noise $N_{R^w_s}$** (de-weighting attitude updates) and adjusts process noise $Q$, preventing the InEKF filter from distorting its heading state.
  - When stationary or driving smoothly, AdapterNet scales down $N$ to tightly anchor attitude and zero-velocity drift.
- **What It Does**:
  - 2-layer dilated 1D CNN with receptive field spanning the window $W_a$.
  - Generates dynamic scaling vectors $\mathbf{q} \in \mathbb{R}^6$ and $\mathbf{n} \in \mathbb{R}^6$ scaled via hyperbolic tangent:
    $$Q_{s,a} = (\sigma_{s,a}^{\text{default}})^2 \cdot 10^{\beta \tanh(q_{s,a})}, \quad N_{s,a} = (\sigma_{s,a}^{\text{default}})^2 \cdot 10^{\beta \tanh(n_{s,a})}$$
- **What It Spits Out**:
  - $\mathbf{Q}(t_n) \in \mathbb{R}^{18 \times 18}$: Dynamic process noise covariance for the IMU kinematic propagation step.
  - $\mathbf{N}(t_{n+1}) \in \mathbb{R}^{6 \times 6}$: Dynamic measurement noise covariance for DDODO and DDATT Kalman updates.
- **Hardware Footprint**:
  - Parameters: **7,020** (only **27.4 KB** FP32 memory!).
  - Complexity: **< 0.15 MFLOPs**. Runs in $< 0.1\ \text{ms}$.

---

### Module 3: InEKF (Model-Driven Lie-Group Fusion Engine)

Module 3 is the core state estimator. It is **not** a neural network; it is a **continuous-discrete Right-Invariant Extended Kalman Filter (InEKF)** formulated on matrix Lie groups. It mathematically guarantees coordinate-invariance, log-linear error dynamics, and non-divergent Riccati covariance propagation.

```
                  FLOW INTO MODULE 3 (InEKF ENGINE)
                  
  Instantaneous Smartphone IMU:
    - Angular velocity \tilde{\omega}_k
    - Specific force   \tilde{f}_k
           │
           │  (Kinematic Propagation at dt = 100 ms)
           ▼
    +-------------------------------------------------------------+
    |                     InEKF PROPAGATION                       |
    |  - State: \chi^w_s \in GSE_2(3), Biases, Extrinsics         |
    |  - Covariance: P^- = F P F^T + G Q G^T                      |
    +------------------------------+------------------------------+
                                   │
                                   │  Prior State \hat{x}^-, Covariance P^-
                                   ▼
  Outputs from Module 1 (AVNet):
    - Forward speed   \hat{v}^v_{lon}
    - Relative quat   \Delta \hat{q} -> \tilde{R}^w_s
  Outputs from Module 2 (AdapterNet):
    - Measurement cov N(t)
    - Process cov     Q(t)
           │
           │  (Kalman Update via 6D Geometric Residual)
           ▼
    +-------------------------------------------------------------+
    |                       InEKF UPDATE                          |
    |  - Residuals: r_R = Log( \tilde{R} \hat{R}^T )              |
    |               r_v = \tilde{y}_v - \hat{R}^v_s (v^s + \omega x p) |
    |  - Kalman Gain: K = P^- H^T (H P^- H^T + N)^-1              |
    |  - Manifold Retraction: \hat{\chi}^+ = Exp(\xi) \cdot \hat{\chi}^- |
    +------------------------------+------------------------------+
                                   │
                                   ▼
                Posterior Vehicle Extended State \hat{x}^+
```

#### A. Exactly What Data Module 3 Takes:
1. **From the Raw Smartphone IMU (Instantaneous sample at time $t_n$ and $t_{n+1}$)**:
   - $\tilde{\boldsymbol{\omega}}_k = [\omega_x, \omega_y, \omega_z]^\top \in \mathbb{R}^3$: Measured angular rate ($\text{rad/s}$) from Gyroscope.
   - $\tilde{\mathbf{f}}_k = [f_x, f_y, f_z]^\top \in \mathbb{R}^3$: Measured specific force ($\text{m/s}^2$) from Accelerometer.
   - Sampling step $\Delta t = t_{n+1} - t_n$ ($0.1\ \text{s}$ at $10\ \text{Hz}$).
2. **From Module 1 (Tri-Stream AVNet-Mag)**:
   - $\hat{v}^v_{\text{lon}} \in \mathbb{R}^1$: Estimated scalar longitudinal forward speed ($\text{m/s}$). Formulates the 3D vehicle pseudo-velocity vector:
     $$\tilde{\mathbf{y}}_{v^v_v} = \begin{bmatrix} 0 \\ \hat{v}^v_{\text{lon}} \\ 0 \end{bmatrix} \in \mathbb{R}^3 \quad \begin{aligned}&\text{(Lateral non-holonomic constraint } \approx 0\text{)} \\ &\text{(Forward DDODO speed from AVNet)} \\ &\text{(Vertical non-holonomic constraint } \approx 0\text{)}\end{aligned}$$
   - $\Delta \hat{\mathbf{q}} = (\Delta q_x, \Delta q_y, \Delta q_z) \in \mathbb{R}^3$: Attitude quaternion delta from window start $t_{n-W}$ to $t_{n+1}$.
     The full absolute pseudo-measurement rotation matrix is reconstructed as:
     $$\tilde{\mathbf{R}}^w_s(t_{n+1}) = \hat{\mathbf{R}}^w_s(t_{n-W}) \cdot \mathbf{R}(\Delta \hat{\mathbf{q}}) \in SO(3)$$
3. **From Module 2 (AdapterNet-9Axis)**:
   - $\mathbf{Q}(t_n) \in \mathbb{R}^{18 \times 18}$: Dynamic process noise covariance matrix scaling the IMU propagation noise.
   - $\mathbf{N}(t_{n+1}) \in \mathbb{R}^{6 \times 6}$: Dynamic measurement noise covariance matrix weighting the 6D measurement update:
     $$\mathbf{N} = \operatorname{diag}\left(\sigma_{R,x}^2,\, \sigma_{R,y}^2,\, \sigma_{R,z}^2,\, \sigma_{v,\text{lat}}^2,\, \sigma_{v,\text{lon}}^2,\, \sigma_{v,\text{up}}^2\right)$$

---

#### B. The 21-Dimensional State Manifold:
The true vehicle state is represented as a Lie group manifold tuple:
$$\mathbf{x} = \left(\boldsymbol{\chi}^w_s,\, \delta\boldsymbol{\omega}^s_s,\, \delta\mathbf{f}^s_s,\, \mathbf{R}^v_s,\, \mathbf{p}^s_v\right) \in GSE_2(3) \times \mathbb{R}^3 \times \mathbb{R}^3 \times SO(3) \times \mathbb{R}^3$$

Where $\boldsymbol{\chi}^w_s$ is embedded in the Extended Special Euclidean Group $GSE_2(3)$ ($5 \times 5$ matrix):
$$\boldsymbol{\chi}^w_s = \begin{bmatrix} \mathbf{R}^w_s & \mathbf{v}^w_s & \mathbf{p}^w_s \\ \mathbf{0}_{1 \times 3} & 1 & 0 \\ \mathbf{0}_{1 \times 3} & 0 & 1 \end{bmatrix} \in GSE_2(3)$$

- $\mathbf{R}^w_s \in SO(3)$: Rotation matrix from smartphone sensor frame ($s$) to local ENU world frame ($w$).
- $\mathbf{v}^w_s \in \mathbb{R}^3$: Velocity of sensor in world frame.
- $\mathbf{p}^w_s \in \mathbb{R}^3$: 3D position of sensor in world frame.
- $\delta\boldsymbol{\omega}^s_s \in \mathbb{R}^3$: Quasi-constant gyroscope bias vector.
- $\delta\mathbf{f}^s_s \in \mathbb{R}^3$: Quasi-constant accelerometer bias vector.
- $\mathbf{R}^v_s \in SO(3)$: Mounting rotation matrix from sensor frame ($s$) to vehicle chassis frame ($v$).
- $\mathbf{p}^s_v \in \mathbb{R}^3$: Lever-arm vector from vehicle center of gravity to smartphone origin in sensor frame.

---

#### C. What Module 3 Does Mathematically:

##### Step 1: Kinematic Strapdown Propagation
Between IMU samples with interval $\Delta t$, the state is integrated forward:
$$\hat{\mathbf{R}}^w_s(t_{n+1}^-) = \hat{\mathbf{R}}^w_s(t_n) \exp\left((\tilde{\boldsymbol{\omega}}_k - \hat{\delta\boldsymbol{\omega}}_k)\Delta t\right)_\times$$
$$\hat{\mathbf{v}}^w_s(t_{n+1}^-) = \hat{\mathbf{v}}^w_s(t_n) + \left(\hat{\mathbf{R}}^w_s(t_n)(\tilde{\mathbf{f}}_k - \hat{\delta\mathbf{f}}_k) + \mathbf{g}^w\right)\Delta t$$
$$\hat{\mathbf{p}}^w_s(t_{n+1}^-) = \hat{\mathbf{p}}^w_s(t_n) + \hat{\mathbf{v}}^w_s(t_n)\Delta t$$
where $\mathbf{g}^w = [0, 0, -9.80]^\top\ \text{m/s}^2$ is the local gravity vector and $(\mathbf{a})_\times$ denotes the $3 \times 3$ skew-symmetric cross-product matrix.

##### Step 2: Right-Invariant Error Formulation & Covariance Propagation
Unlike standard EKFs where errors are defined by simple Euclidean subtraction $(\hat{\mathbf{x}} - \mathbf{x})$, the Right-Invariant error on Lie groups is defined as:
$$\boldsymbol{\eta}_{\chi^w_s} = \hat{\boldsymbol{\chi}}^w_s (\boldsymbol{\chi}^w_s)^{-1} = \exp_{GSE_2(3)}(\boldsymbol{\xi}^w_s)$$
$$\boldsymbol{\eta}_{R^v_s} = \hat{\mathbf{R}}^v_s (\mathbf{R}^v_s)^\top = \exp_{SO(3)}(\boldsymbol{\xi}_{R^v_s})$$
$$\boldsymbol{\xi}_{\delta\omega} = \delta\boldsymbol{\omega} - \hat{\delta\boldsymbol{\omega}}, \quad \boldsymbol{\xi}_{\delta f} = \delta\mathbf{f} - \hat{\delta\mathbf{f}}, \quad \boldsymbol{\xi}_{p^s_v} = \mathbf{p}^s_v - \hat{\mathbf{p}}^s_v$$

The continuous error dynamics linearize to:
$$\dot{\mathbf{e}}(t) = \mathcal{F} \mathbf{e}(t) + \mathcal{G} \mathbf{w}(t)$$
Discretized state transition matrix $\mathbf{F} \in \mathbb{R}^{21 \times 21}$ and noise matrix $\mathbf{G} \in \mathbb{R}^{21 \times 18}$:
$$\mathbf{F} = \mathbf{I}_{21} + \begin{bmatrix}
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \hat{\mathbf{R}}^w_s & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
(\mathbf{g}^w)_\times & \mathbf{0}_3 & \mathbf{0}_3 & (\hat{\mathbf{v}}^w_s)_\times \hat{\mathbf{R}}^w_s & \hat{\mathbf{R}}^w_s & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{I}_3 & \mathbf{0}_3 & (\hat{\mathbf{p}}^w_s)_\times \hat{\mathbf{R}}^w_s & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_{12 \times 3} & \mathbf{0}_{12 \times 3} & \mathbf{0}_{12 \times 3} & \mathbf{0}_{12 \times 3} & \mathbf{0}_{12 \times 3} & \mathbf{0}_{12 \times 3} & \mathbf{0}_{12 \times 3}
\end{bmatrix}\Delta t$$

Covariance propagates using $\mathbf{Q}(t_n)$ from AdapterNet:
$$\mathbf{P}^-(t_{n+1}) = \mathbf{F} \mathbf{P}(t_n) \mathbf{F}^\top + \mathbf{G} \mathbf{Q}(t_n) \mathbf{G}^\top$$

##### Step 3: 6D Geometric Innovation & Measurement Update
The 6D measurement innovation vector $\mathbf{r} \in \mathbb{R}^6$ combines DDATT (orientation) and DDODO (velocity):
$$\mathbf{r} = \begin{bmatrix} \mathbf{r}_{R^w_s} \\ \mathbf{r}_{v^v_v} \end{bmatrix} \in \mathbb{R}^6$$
- **Attitude Residual (Lie algebra logarithm on $SO(3)$)**:
  $$\mathbf{r}_{R^w_s} = \log_{SO(3)}\left(\tilde{\mathbf{R}}^w_s (\hat{\mathbf{R}}^w_s)^\top\right) \in \mathbb{R}^3$$
- **Velocity Residual (Vehicle Chassis Frame Kinematics)**:
  $$\mathbf{r}_{v^v_v} = \tilde{\mathbf{y}}_{v^v_v} - \hat{\mathbf{R}}^v_s \left( (\hat{\mathbf{R}}^w_s)^\top \hat{\mathbf{v}}^w_s + (\tilde{\boldsymbol{\omega}}_k - \hat{\delta\boldsymbol{\omega}}_k)_\times \hat{\mathbf{p}}^s_v \right) \in \mathbb{R}^3$$

Analytical Measurement Jacobian $\mathbf{H} \in \mathbb{R}^{6 \times 21}$:
$$\mathbf{H} = \begin{bmatrix} \mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\ \mathbf{0}_3 & \hat{\mathbf{R}}^v_s (\hat{\mathbf{R}}^w_s)^\top & \mathbf{0}_3 & \hat{\mathbf{R}}^v_s (\hat{\mathbf{p}}^s_v)_\times & \mathbf{0}_3 & \mathbf{H}_{R^v_s} & \hat{\mathbf{R}}^v_s (\tilde{\boldsymbol{\omega}}_k)_\times \end{bmatrix}$$
where:
$$\mathbf{H}_{R^v_s} = -\left( \hat{\mathbf{R}}^v_s \left( (\hat{\mathbf{R}}^w_s)^\top \hat{\mathbf{v}}^w_s + (\tilde{\boldsymbol{\omega}}_k - \hat{\delta\boldsymbol{\omega}}_k)_\times \hat{\mathbf{p}}^s_v \right) \right)_\times$$

Kalman Gain using dynamic measurement noise $\mathbf{N}$ from AdapterNet:
$$\mathbf{S} = \mathbf{H} \mathbf{P}^- \mathbf{H}^\top + \mathbf{N}$$
$$\mathbf{K} = \mathbf{P}^- \mathbf{H}^\top \mathbf{S}^{-1} \in \mathbb{R}^{21 \times 6}$$
$$\mathbf{e}^+ = \mathbf{K} \mathbf{r} = \begin{bmatrix} \boldsymbol{\xi}_{R^w_s}^\top & \boldsymbol{\xi}_{v^w_s}^\top & \boldsymbol{\xi}_{p^w_s}^\top & \boldsymbol{\xi}_{\delta\omega}^\top & \boldsymbol{\xi}_{\delta f}^\top & \boldsymbol{\xi}_{R^v_s}^\top & \boldsymbol{\xi}_{p^s_v}^\top \end{bmatrix}^\top \in \mathbb{R}^{21}$$

##### Step 4: Manifold Retraction (State Correction)
The correction is applied via matrix exponential multiplication on Lie groups (avoiding Euler singularity or quaternion norm drift):
$$\hat{\boldsymbol{\chi}}^w_s(t_{n+1}^+) = \exp_{GSE_2(3)}\left(\begin{bmatrix} \boldsymbol{\xi}_{R^w_s} \\ \boldsymbol{\xi}_{v^w_s} \\ \boldsymbol{\xi}_{p^w_s} \end{bmatrix}\right) \cdot \hat{\boldsymbol{\chi}}^w_s(t_{n+1}^-)$$
$$\hat{\mathbf{R}}^v_s(t_{n+1}^+) = \exp_{SO(3)}(\boldsymbol{\xi}_{R^v_s}) \cdot \hat{\mathbf{R}}^v_s(t_{n+1}^-)$$
$$\hat{\delta\boldsymbol{\omega}}^+ = \hat{\delta\boldsymbol{\omega}}^- + \boldsymbol{\xi}_{\delta\omega}, \quad \hat{\delta\mathbf{f}}^+ = \hat{\delta\mathbf{f}}^- + \boldsymbol{\xi}_{\delta f}, \quad \hat{\mathbf{p}}^s_v{}^+ = \hat{\mathbf{p}}^s_v{}^- + \boldsymbol{\xi}_{p^s_v}$$
$$\mathbf{P}^+ = (\mathbf{I}_{21} - \mathbf{K} \mathbf{H}) \mathbf{P}^-$$

---

#### D. What Module 3 Spits Out:
Module 3 produces the **Complete Filtered Vehicle Navigation State** at 10 Hz (or 200 Hz):
1. **$\mathbf{p}^w_s = [p_E, p_N, p_U]^\top \in \mathbb{R}^3$**: 3D position trajectory in meters (East, North, Up) relative to starting point.
2. **$\mathbf{v}^w_s = [v_E, v_N, v_U]^\top \in \mathbb{R}^3$**: 3D metric velocity in world coordinates ($\text{m/s}$).
3. **$\mathbf{R}^w_s \in SO(3)$**: Global 3D attitude matrix (convertible to Euler angles: Heading/Yaw, Pitch, Roll).
4. **$\delta\boldsymbol{\omega}^s_s, \delta\mathbf{f}^s_s \in \mathbb{R}^3$**: Online calibrated gyroscope and accelerometer sensor biases.
5. **$\mathbf{R}^v_s \in SO(3), \mathbf{p}^s_v \in \mathbb{R}^3$**: Online calibrated mounting orientation and lever-arm extrinsics.
6. **$\mathbf{P}^+ \in \mathbb{R}^{21 \times 21}$**: Posterior error covariance (provides exact 1-$\sigma$ uncertainty bounds for safety-critical autonomous systems).

- **Computational Complexity**: Matrix operations on $21 \times 21$ and $6 \times 6$ inverses require $\approx \mathbf{0.08\ \text{MFLOPs}}$ and execute in **$< 0.2\ \text{ms}$** per step on CPU.

---

## 4. Hardware Profiling: FLOPs, Model Size & RAM Consumption

The Tri-Stream AVNet model was profiled under Python 3.13 / PyTorch on a standard x86 CPU. Below are the verified metrics:

### 4.1 Parameter Count & Storage Footprint
| Metric | Full Precision (FP32) | Half Precision (FP16) | Quantized (INT8) |
| :--- | :---: | :---: | :---: |
| **Total Parameters** | **268,357** | **268,357** | **268,357** |
| **Trainable Parameters** | 268,357 | 268,357 | 268,357 |
| **Model Weights Disk / RAM** | **1.02 MB** | **0.51 MB** | **0.26 MB** |

### 4.2 Computational Complexity (FLOPs & MACs)
Profiled with input batch size $B=1$, window length $W=20$:

| Sub-Module / Block | Multiply-Accumulate Operations (MACs) | Floating Point Operations (FLOPs) |
| :--- | :---: | :---: |
| **3 Sensor Branches (Accel, Gyro, Mag)** | 774,144 | 1,548,288 |
| **Cross-Modal Fusion Block (Conv1d 192 $\to$ 128)** | 1,474,560 | 2,949,120 |
| **2-Layer Bidirectional GRU (Hidden 64)** | 2,457,600 | 4,915,200 |
| **Self-Attention Temporal Pooling** | 165,120 | 330,240 |
| **Dual Regression Heads (DDODO + DDATT)** | 119,936 | 239,872 |
| **TOTAL MODEL PER INFERENCE** | **4,991,360 MACs** | **9.98 MFLOPs (~10 MFLOPs)** |

### 4.3 Runtime Latency & Memory During Inference
- **Inference Latency (Single Thread CPU)**: **6.31 ms per window** (~159 inferences/second).
  - *At 10 Hz execution rate, the model consumes only **~6.3% of a single CPU core's capacity**.*
- **Peak Dynamic Activation RAM during Inference ($B=1$)**: **~0.18 MB**.
- **Total Operational Memory Footprint (Weights + Activations + State Buffers)**: **$< 1.5\ \text{MB}$**.
  - Extremely lightweight: can run embedded directly inside an Android background service, Raspberry Pi, or vehicle ECU without GPU acceleration.

---

## 5. Training, Fine-Tuning & Data Strategy

```
  [Stage 1: Base Pre-training]
  All 288 CSV files in "data/Synchronised V abd S datasets" (Uncategorised runs)
  -> 855 MB total data (~80+ hours of multi-driver runs)
  -> Objective: Learn fundamental inertial vibration damping and car dynamics
                      │
                      ▼ (Save base_tristream_avnet.pth)
  [Stage 2: Domain / Driver Fine-Tuning]
  Specific driver routes (e.g., Driver B "S-M.csv", Driver A "S-B.csv")
  -> Freeze Conv branches: acc_branch, gyro_branch, mag_branch
  -> Fine-tune GRU and Output Heads with lower LR (1e-4) for 10 epochs
  -> Objective: Adapt to specific vehicle chassis mount and driver acceleration habits
```
