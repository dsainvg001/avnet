# Complete Analysis: AVNet Paper, AdapterNet Deep Dive & IO-VNBD Dataset Audit

## Executive Summary

This document addresses:
1. **Paper Issues & Limitations**: Hidden assumptions, implementation gaps, and mathematical trade-offs in *Qian et al. (Satellite Navigation 2025)*.
2. **AdapterNet Explained**: Exactly what AdapterNet is, why it exists, how it works, and how it is trained.
3. **IO-VNBD Dataset Evaluation**: Complete audit of the 241 smartphone files and 323 vehicle CAN files in our repository.
4. **The Critical Discovery**: Why training on `S-*.csv` alone fails (1 Hz / 9s GPS update lag), and how pairing with `V-*.csv` from `Synchronised V abd S datasets/` solves this completely.
5. **Missing Columns & Workarounds**: What our dataset has, what it lacks compared to NovAtel SPAN, and whether we need another dataset.

---

## 1. Deep Dive: What AdapterNet Actually Is (Demystified)

### 1.1 The Fundamental Problem of the Kalman Filter
In standard inertial navigation, an Invariant Extended Kalman Filter (InEKF) relies on two critical covariance matrices:
- **$\mathbf{Q}$ (Process Noise Covariance)**: Describes how much you distrust the IMU integration physics (accelerometer and gyroscope noise, bias walk).
- **$\mathbf{N}$ (Measurement Noise Covariance)**: Describes how much you distrust the incoming corrections (AVNet's predicted speed $v_{\text{lon}}$ and attitude $\Delta q$).

In traditional aerospace / robotics, engineers **manually hand-tune** fixed numbers for $\mathbf{Q}$ and $\mathbf{N}$.
**Why fixed numbers fail on a smartphone in a moving car:**
- **At a traffic light / parked**: The car is stationary. Speed is strictly 0 m/s. Sensor noise is minimal. Here, $\mathbf{N}$ should be **tiny** (high trust) to lock the filter and prevent integration drift.
- **Over potholes, railroad tracks, or aggressive swerves**: The phone shakes violently. AVNet's neural network might output noisy or erroneous speed estimates. Here, $\mathbf{N}$ should be **large** (low trust) so the filter relies on kinematic momentum rather than erroneous neural predictions.
- **In urban canyons or parking garages**: Magnetic anomalies from rebar and high-voltage lines distort compass readings. The attitude measurement noise $\mathbf{N}_{\text{att}}$ must spike to ignore corrupted heading.

### 1.2 What AdapterNet Does
**AdapterNet is an intelligent, real-time "confidence tuner" for the Kalman filter.**
It is a lightweight 2-layer 1D dilated convolutional network (~7,020 parameters) that continuously inspects a short temporal window of raw IMU vibrations ($W = 20$ samples) and dynamically predicts:
$$\mathbf{q} \in \mathbb{R}^6 \quad \text{(process noise scales)}, \qquad \mathbf{n} \in \mathbb{R}^6 \quad \text{(measurement noise scales)}$$

### 1.3 The Bounded Exponential Formula
To ensure numerical stability, AdapterNet's outputs do not directly set the covariance; they scale nominal baseline values ($\sigma_{\text{default}}$) using hyperbolic tangent:
$$Q_{s,a} = \left(\sigma_{s,a}^{\text{default}}\right)^2 \cdot 10^{\beta \tanh(q_{s,a})}, \qquad N_{s,a} = \left(\sigma_{s,a}^{\text{default}}\right)^2 \cdot 10^{\beta \tanh(n_{s,a})}$$
where $\beta = 3$.

**Why this formula is mathematically bulletproof:**
- Since $\tanh(x) \in (-1, 1)$, $10^{3 \tanh(x)}$ is strictly bounded in:
  $$\left[10^{-3},\; 10^{+3}\right] = \left[0.001,\; 1000\right]$$
- **It can never be negative** (variance cannot be negative).
- **It can never explode to infinity** (max scaling is $1000\times$ default).
- **It can never collapse to zero** (min scaling is $0.001\times$ default, preventing singular matrix inversion in the Kalman gain $K = P H^\top (H P H^\top + N)^{-1}$).

### 1.4 How AdapterNet Was Trained in the Paper (The "Indirect" Method)
AdapterNet **does not have direct ground-truth labels**. There is no sensor on earth that measures "the true measurement covariance matrix $N$".

Instead, the paper trains AdapterNet using **indirect / end-to-end differentiable filter optimization** (adapted from AI-IMU by Brossard et al.):
1. A training driving sequence is run through the entire pipeline:
   $$\text{IMU Window} \;\xrightarrow{\text{AdapterNet}}\; Q, N \;\xrightarrow{\text{InEKF}}\; \hat{\mathbf{p}}_{1:T} \;(\text{estimated trajectory})$$
2. The final estimated trajectory $\hat{\mathbf{p}}$ is compared against the true ground-truth trajectory $\mathbf{p}_{\text{true}}$ (from the NovAtel SPAN system) using the KITTI Relative Translation Error ($E_{\text{trel}}$):
   $$\mathcal{L}_{\text{adapter}} = E_{\text{trel}}(\hat{\mathbf{p}}, \mathbf{p}_{\text{true}})$$
3. Gradients $\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{p}}}$ are backpropagated **through the Kalman Filter equations** back to $Q$ and $N$, and finally into AdapterNet's weights.

> **Key takeaway**: AdapterNet learns to output whatever $Q$ and $N$ values make the Kalman filter's final position output as accurate as possible!

---

## 2. Issues & Limitations Found in the AVNet Paper

A critical reading of *Qian et al. (2025)* reveals several important limitations:

### Issue 1: Differentiable Filter Requirement
In the paper's open-source reference (`QAIIMUDeadReckoning`), the Kalman filter propagation and Riccati equations are implemented in PyTorch tensors so that `autograd` can compute gradients through time.
- If an implementation writes InEKF in standard NumPy/SciPy (as in many offline pipelines), standard `.backward()` cannot flow through the filter.
- **Solution for our project**:
  - *Option A*: Implement the InEKF steps in PyTorch tensor operations so that `loss.backward()` works cleanly end-to-end.
  - *Option B (In-app post-trip retraining)*: Use the innovation log-likelihood / residual loss or numerical finite-difference approximation.

### Issue 2: Output Rate & Measurement Stride Mismatch
- In the paper: IMU runs at 200 Hz. The AVNet window is $W_{\text{est}} = 200$ samples. The paper states: *"the output frequency of AVNet is 1 Hz in the experiments"*.
- But InEKF propagates at 200 Hz. This means the InEKF only gets a measurement update once every 200 propagation steps (1 second). Between updates, it relies strictly on IMU integration.
- In our 10 Hz dataset, if we slide AVNet with stride = 1 (100 ms), AVNet can update the filter at **10 Hz**, giving much more frequent corrections.

### Issue 3: Rigid Mounting Assumption
- In the paper's experimental setup (Section 4.1, Page 11): *"The NovAtel SPAN antenna was rigidly mounted on the vehicle roof, while the ISA-100C IMU and the smartphone were also rigidly mounted to a custom bracket at a fixed mounting angle."*
- The paper's mathematical model assumes the sensor-to-vehicle rotation $R^v_s$ is quasi-constant.
- In real smartphone applications, phones sit in cup holders, vents, or pockets, where small vibrations or orientation changes occur. The InEKF includes $R^v_s$ in its 21-state vector to self-estimate this, but it requires vehicle maneuvers (accelerations and turns) within the first 30 seconds to observe and lock the mounting angle.

### Issue 4: Magnetometer Was Completely Excluded
- The paper explicitly notes that only 6-axis IMU (accel + gyro) was used. Magnetometer was avoided because of magnetic distortions in parking garages.
- Our Tri-Stream architecture includes the magnetometer. This is an enhancement, but AdapterNet must be given the magnetometer stream so it can learn to set high $N_{\text{att}}$ when magnetic anomalies are detected.

---

## 3. Comprehensive Audit of Our Dataset (IO-VNBD)

Our workspace contains the **IO-VNBD** (Inertial Odometry Vehicle Navigation Benchmark Dataset):
- **144 Synchronized pairs** (`S-*.csv` + `V-*.csv`) in `Synchronised V abd S datasets/`
- **97 Smartphone + 179 Vehicle files** in `Unsynchronised V and S Dataset/`
- Over **50 hours** and **4,400+ km** of driving across UK, France, and Nigeria.

### 3.1 🚨 The Critical Discovery: Smartphone GPS Stepping
When inspecting the raw `S-*.csv` files, we discovered:
- **IMU Channels** (`ACCELEROMETER`, `GYROSCOPE`, `MAGNETIC FIELD`) update every **100 ms (10 Hz)** continuously.
- **Smartphone GPS Channels** (`GPS SPEED`, `GPS LATITUDE`, `GPS LONGITUDE`) **stay frozen for 10 to 90 consecutive rows** (updating only every 1 to 9 seconds)!
  - This happens because the Android `LocationManager` only delivers GPS callbacks at ~1 Hz or slower, while `SensorManager` runs at 10 Hz. The logger simply repeated the last known GPS fix on every 10 Hz IMU tick.
- **Why this matters**: If you train AVNet using `GPS SPEED (Kmh)` from `S-*.csv` directly, the network is trying to predict a stepped, lagged staircase signal from fluid continuous IMU data!

### 3.2 The Solution: 1-to-1 Row Synchronization with `V-*.csv`
In the directory `Synchronised V abd S datasets/`:
- Every `S-*.csv` has an exact matching `V-*.csv` (e.g., `S-M.csv` has `V-M.csv`).
- **The row counts match identically** (e.g., `S-M.csv` = 105,974 rows; `V-M.csv` = 105,974 rows; `S-S1.csv` = 51,746 rows; `V-S1.csv` = 51,746 rows).
- The `V-*.csv` file contains data directly from the vehicle CAN bus and vehicle-grade GPS:
  - `Indicated Vehicle Speed (km/hr)`: Real wheel odometer speed at **continuous 10 Hz** (no stepping, no freeze).
  - `Velocity (km/hr)`: Continuous 10 Hz vehicle velocity.
  - `Latitude (degrees)` & `Longitude (degrees)`: High-frequency vehicle GPS.
  - `Yaw Rate (deg/sec)`: High-grade vehicle turn rate.

```
+-------------------------------------------------------------------------------+
|                        OPTIMAL TRAINING DATA PIPELINE                         |
|                                                                               |
|   INPUTS (from S-*.csv):                                                      |
|   - ACCELEROMETER X, Y, Z (m/s²)       [Raw Phone IMU @ 10 Hz]               |
|   - GYROSCOPE Yaw, Pitch, Roll (rad/s)  [Raw Phone IMU @ 10 Hz]               |
|   - MAGNETIC FIELD X, Y, Z (μT)         [Raw Phone Mag @ 10 Hz]               |
|                                                                               |
|   GROUND TRUTH LABELS (from paired V-*.csv):                                  |
|   - Indicated Vehicle Speed (km/hr) / 3.6  --> True Speed v_lon (m/s) @ 10 Hz |
|   - Latitude & Longitude (degrees)         --> True ENU Trajectory @ 10 Hz    |
|   - Yaw Rate (deg/sec) / Heading (degrees) --> True Attitude Change Δq @ 10 Hz |
+-------------------------------------------------------------------------------+
```

### 3.3 ⚠️ Second Discovery: Session Restarts / Negative Time Jumps
In long recordings like `S-M.csv`, `TIME SINCE START (ms)` occasionally resets:
- Example: Row 44225 ($t = 4,426,727\text{ ms}$) $\to$ Row 44226 ($t = 10\text{ ms}$).
- This indicates the logger restarted or paused.
- **Fix in Dataset Loader**: The dataset parser must split contiguous trajectories whenever $dt < 0$ or $dt > 1.0\text{ s}$, ensuring no sliding window spans across disconnected sessions.

---

## 4. Are We Missing Any Key Columns?

Let's do an itemized column-by-column comparison:

| Required Quantity | Paper Baseline | Our S-Dataset (`S-*.csv`) | Our Synced V-Dataset (`V-*.csv`) | Verdict |
| :--- | :--- | :--- | :--- | :---: |
| **Linear Acceleration** | 3-axis (m/s²) | `ACCELEROMETER X/Y/Z` | `Indicated Longitudinal/Lateral Acc` | ✅ **Present** |
| **Angular Velocity** | 3-axis (rad/s) | `GYROSCOPE Yaw/Pitch/Roll` | `Yaw Rate (deg/sec)` | ✅ **Present** |
| **Magnetic Flux** | Not used | `MAGNETIC FIELD X/Y/Z` | N/A | ✅ **Present** (Our extension) |
| **Phone Orientation** | Not used | `ORIENTATION (Yaw/Pitch/Roll)` | N/A | ✅ **Present** |
| **Ground Truth Speed** | NovAtel speed | `GPS SPEED (Kmh)` (lagged) | `Indicated Vehicle Speed` (10 Hz CAN) | ✅ **Superb (from V-file)** |
| **Ground Truth Position** | NovAtel RTK (±1cm) | `GPS LAT/LON` (phone) | `Latitude / Longitude` (vehicle GPS) | ⚠️ **Consumer GPS (±3-5m)** |
| **Ground Truth Attitude** | NovAtel SPAN (±0.001°) | Android orientation | `Heading (degrees)` + integrated yaw rate | ⚠️ **Good (from V-file / phone)** |
| **Steering / Wheel speeds** | Not used | N/A | `Wheel Speed FL/FR/RL/RR`, `Steering Angle` | ✅ **Bonus in V-file** |

### What is the ONLY thing missing?
The only difference between our dataset and the paper's dataset is:
1. **Sampling Frequency**: 10 Hz (our data) vs. 200 Hz (paper's data).
2. **Centimeter-level RTK Carrier Phase**: The paper had an expensive NovAtel SPAN ISA-100C receiver (~$30,000 unit) providing 1 cm positioning. IO-VNBD used standard vehicle GPS (~3 m positioning).

### Does this prevent training?
**NO.**
- For **AVNet (speed & attitude)**: CAN bus wheel speed is actually **more accurate than GPS Doppler velocity** for forward velocity ground truth, because it has zero multipath error and instantaneous response.
- For **AdapterNet**: Trajectory drift over 500 m to 1 km is on the order of 10 to 50 meters. A 3-meter GPS error is small enough to provide a clear gradient for covariance scaling.
- The 144 synchronized files give us over **40 hours of real-world road data** covering diverse road types, drivers, roundabouts, and traffic conditions—vastly more diverse than the paper's 11 parking lot sequences!

---

## 5. Summary & Action Plan

1. **AdapterNet**:
   - Understand it as a 2-layer CNN that outputs dynamic confidence multipliers ($10^{3 \tanh}$) to prevent filter divergence under shocks, turns, and magnetic anomalies.
2. **Dataset Pairing**:
   - Update `avnet/dataset.py` to pair `S-*.csv` with `V-*.csv` in `Synchronised V abd S datasets/`.
   - Take IMU inputs from phone `S-*.csv` and high-rate target speed / position from vehicle `V-*.csv`.
3. **Session Split**:
   - Detect and segment on negative time jumps ($dt < 0$) to guarantee window integrity.
4. **Conclusion**:
   - We do **not** need to discard or replace IO-VNBD. By correctly leveraging the synchronized vehicle CAN bus files, it is one of the richest benchmark datasets available for this task.
