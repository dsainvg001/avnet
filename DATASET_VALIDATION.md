# Dataset Validation & Architecture Gap Analysis

This document audits what the paper assumed vs. what our dataset and model actually provide. Every discrepancy is flagged, its impact assessed, and a mitigation is documented.

---

## 1. Core Architecture Differences: Paper vs. Our Implementation

### 1.1 AVNet (Module 1)

| Dimension | Paper's AVNet | Our Tri-Stream AVNet-Mag | Impact |
| :--- | :--- | :--- | :--- |
| **Input sensors** | 6-ch: Accel (3) + Gyro (3) | **9-ch: Accel (3) + Gyro (3) + Mag (3)** | ✅ Richer input — heading from mag adds yaw constraint |
| **Input sampling rate** | **200 Hz** | **10 Hz** | ⚠️ 20× slower — window temporal resolution is coarser |
| **Window size (samples)** | 200 samples | 20 samples | Same 1-second window? **No** — see §2.1 below |
| **Window duration (seconds)** | **1.0 second** at 200 Hz | **2.0 seconds** at 10 Hz | ⚠️ 2× longer window — captures more dynamics but with lower resolution |
| **Conv architecture** | Single-stream, large kernel (k=11, k=9) | **Three-stream, per-sensor dilated CNN** | ✅ Decoupled sensor-specific feature extraction |
| **Dense layer** | Flatten 11,008 → Dense(1024, 512) | **No flatten** — AdaptiveAvgPool per stream | ✅ Our design is far more parameter-efficient |
| **Recurrent layer** | Uni-directional GRU(512→64, 2 layers) | **BiGRU(192→64, 2 layers) + Self-Attention** | ✅ Bidirectional + attention = better sequence modeling |
| **Output** | `v_lon (1)`, `Δq_xyz (3)` | Same: `v_lon (1)`, `Δq_xyz (3)` | ✅ Compatible |
| **Output frequency** | 1 Hz (1 step per 200 samples) | **10 Hz** (1 step per 20 samples) | ✅ Better for InEKF update rate |

### 1.2 AdapterNet (Module 2)

| Dimension | Paper's AdapterNet | Our AdapterNet-9Axis | Impact |
| :--- | :--- | :--- | :--- |
| **Input sensors** | **6-ch: Accel + Gyro ONLY** | **9-ch: Accel + Gyro + Mag** | ✅ Mag allows detecting magnetic distortion spikes → smarter $N_{R}$ scaling |
| **Input window** | W=20 samples at **200 Hz = 0.1 sec** | W=20 samples at **10 Hz = 2.0 sec** | ⚠️ Very different temporal granularity — see §2.2 |
| **Output frequency** | **200 Hz** (one Q,N per IMU sample) | **10 Hz** (one Q,N per step) | ⚠️ Coarser — less adaptive, but still matches our loop |
| **Architecture** | 2-layer 1D dilated CNN | Same 2-layer 1D dilated CNN | ✅ Identical |
| **Training loss** | Relative translation error $E_{\text{trel}}$ (indirect) | Same + GPS endpoint loss | ✅ Compatible |

### 1.3 InEKF (Module 3)

| Dimension | Paper | Our Implementation | Impact |
| :--- | :--- | :--- | :--- |
| **Propagation rate** | 200 Hz (every IMU sample) | **10 Hz** | ⚠️ 20× fewer propagation steps — integration less accurate |
| **State space** | 21-dimensional $GSE_2(3) \times SO(3) \times \mathbb{R}^9$ | Same 21-dim | ✅ Compatible |
| **Measurement update** | 6D (3D attitude + 3D velocity) | Same 6D | ✅ Compatible |
| **DDNHC** | ✅ Zero lateral + vertical velocity | ✅ Same | ✅ Compatible |

---

## 2. Critical Dataset Invalidations

These are cases where data the paper relied on **does not exist** in our dataset, or exists in a degraded form.

---

### 2.1 ❌ INVALIDATION: Sampling Rate — 200 Hz vs. 10 Hz

**Paper assumes**: IMU sampled at 200 Hz. AVNet window = 200 samples = 1 second. AdapterNet window = 20 samples = 0.1 second.

**Our dataset provides**: All `S-*.csv` files at **10 Hz** (100 ms steps, confirmed in `data.md`).

| | Paper | Our Data | Gap |
| :--- | :---: | :---: | :---: |
| IMU rate | 200 Hz | **10 Hz** | 20× slower |
| AVNet window (samples) | 200 | 20 | Same sample count |
| AVNet window (time) | 1.0 s | **2.0 s** | 2× longer — ok if adjusted |
| AdapterNet window (samples) | 20 | 20 | Same sample count |
| AdapterNet window (time) | 0.1 s | **2.0 s** | 20× longer — significant |
| Integration accuracy | High (200 steps/sec) | Lower (10 steps/sec) | Euler errors accumulate faster |

**Mitigation**:
- AVNet at 10 Hz is viable — 2-second window captures enough dynamics for speed/attitude estimation. **Already accounted for in our architecture** (W=20 → 2.0s window).
- AdapterNet at 10 Hz means each Q/N covariance covers 0.1s of dynamics over a 2-second window. This is coarser but functional.
- InEKF at 10 Hz means position integration uses 100ms steps instead of 5ms. To compensate: use **RK4 integration** in propagation instead of simple Euler, or upsample IMU via linear interpolation before feeding InEKF.

> **Action**: Optionally add a preprocessing step to upsample 10 Hz IMU to 50 Hz via cubic spline interpolation before InEKF propagation. Not needed for training, but helps inference accuracy.

---

### 2.2 ❌ INVALIDATION: No Centimeter-Level Ground Truth Position

**Paper assumes**: NovAtel SPAN ISA-100C provides **±0.01m position** and **±0.001° attitude** at 200 Hz as training labels.

**Our dataset provides**:
- `GPS LATITUDE / LONGITUDE`: Consumer GPS, **±1–10m horizontal accuracy** depending on signal.
- `GPS ACCURACY (m)`: Reports estimated accuracy. Typically 3–8m in open roads, 10–30m in urban canyons.
- `GPS SPEED (Kmh)`: Doppler-derived — relatively reliable (±0.1 km/h), even with poor position accuracy.
- `GPS ORIENTATION (°)`: GPS track angle — only valid when moving (>5 km/h), unreliable at low speeds.

**Impact**:

| Label | Paper uses | We have | Quality |
| :--- | :--- | :--- | :--- |
| Speed label ($v_{\text{lon}}$) | NovAtel velocity ±0.01 m/s | GPS SPEED (Kmh) ÷ 3.6 | ✅ Good — Doppler GPS speed is reliable |
| Attitude label ($\Delta q$) | NovAtel SPAN attitude ±0.001° | Android ORIENTATION (fused) | ⚠️ Moderate — Android fusion is a black-box |
| Position label ($p$) for AdapterNet | NovAtel ±0.01m | GPS LAT/LON ±3–10m | ⚠️ Noisy — use strict accuracy filter |

**Mitigation**:
- For speed labels: use `GPS SPEED (Kmh)` only where `GPS ACCURACY < 5m`. This is a reliable Doppler measurement.
- For attitude labels: use `ORIENTATION (Yaw/Pitch/Roll)` from Android sensor fusion. **Important caveat**: Android orientation is itself a black-box filter — it internally uses mag + accel + gyro. This means our DDATT target is partially derived from the same magnetometer we're feeding as an input, creating a mild label-leakage path. Mitigate by using **GPS ORIENTATION** (track angle) as the primary heading label when moving, and Android orientation only for pitch/roll.
- For AdapterNet position: apply strict GPS filter (`GPS_ACCURACY < 5m`, `SATELLITES > 10`, `GPS_SPEED > 2 m/s` to avoid stationary GPS drift).

---

### 2.3 ⚠️ PARTIAL: GPS SATELLITES IN RANGE Column Format

**Paper**: Satellite count is a clean integer.

**Our dataset**: `GPS SATELLITES IN RANGE` is a **ratio string** like `"29 / 29"` (visible / used), not a simple integer.

**Impact**: Must parse before using as a quality filter.

**Fix** (in preprocessing):
```python
def parse_satellites(sat_str):
    # Handles "29 / 29" or "12/14" or simple "18"
    if '/' in str(sat_str):
        used = int(str(sat_str).split('/')[0].strip())
        return used
    return int(sat_str)
```

---

### 2.4 ⚠️ ASSUMPTION MISMATCH: Static Phone Mount vs. Dynamic Placement

**Paper assumes**: Smartphone is **rigidly mounted** to a custom bracket at a **fixed mounting angle**. The extrinsic rotation $R^v_s$ and lever arm $p^s_v$ are approximately constant throughout a recording.

**Our deployment target (app)**: Phone is placed in a **cup holder, dashboard mount, or held by hand**. Mounting orientation changes between trips. $R^v_s$ can be any rotation.

**Impact**: The InEKF estimates $R^v_s$ as part of its 21-dim state (it self-calibrates online). This is designed to handle unknown mounting. However:
- Convergence of $R^v_s$ requires the vehicle to make **turning maneuvers** within the first 30–60 seconds.
- A phone lying flat on the seat (no physical constraint) will take longer to converge because gravity and magnetic field provide weak heading discrimination.

**Mitigation**: 
- At app startup, instruct user to drive normally for 30 seconds before trusting dead-reckoning output.
- Initialize $R^v_s = I_3$ (no rotation assumed) with high initial uncertainty $\sigma_{R^v_s} = 0.5$ rad.

---

### 2.5 ⚠️ PARTIAL: No Wheel Speed / Odometer in S-Dataset

**Paper says**: 
> *"While it is usually limited for smartphones to access in-vehicle sensors' data."*
The paper's method is specifically designed to work **without** odometer/wheel speed. AVNet replaces the odometer.

**Our V-Dataset** (V-*.csv) contains: `Wheel Speed Front Left/Right/Rear Left/Right (rad/s)`, `Indicated Vehicle Speed`, `Steering Angle`, etc.

**Impact**: 
- These V-dataset columns are **bonus supervision signals** we can optionally use for offline training.
- For `GPS SPEED` substitute → use `Indicated Vehicle Speed (km/h)` from synchronized V-S pairs, which has **higher update reliability than GPS-derived speed** (no dropouts, no multipath).
- **At inference/app time**: V-dataset is unavailable. The model must run purely on S-dataset (smartphone sensors). This is by design.

**Usage**:
- **Offline training**: Optionally use `Indicated Vehicle Speed` from synced V-*.csv as a cleaner speed label.
- **In-app inference**: Only S-dataset columns available. GPS SPEED is the only speed reference.

---

### 2.6 ✅ VALID: Magnetic Field / Orientation Columns

**Paper**: Does not use magnetometer in AVNet or AdapterNet. Uses only `accel + gyro`.

**Our extension**: We add magnetometer as a **third stream** to AVNet and a **third input group** to AdapterNet.

**Dataset validation**:
- `MAGNETIC FIELD X/Y/Z (μT)` ✅ present in all S-*.csv files
- `ORIENTATION (Yaw/Pitch/Roll) (°)` ✅ present in all S-*.csv files (internally mag-fused by Android)
- Magnetic field values in dataset: typical range ±40–60 μT, occasional spikes near infrastructure

**No invalidation** — magnetometer data is fully available and properly sampled.

---

### 2.7 ⚠️ GRAVITY VECTOR — Available but Redundant

**Our dataset** has `GRAVITY X/Y/Z (m/s²)` — the Android-filtered low-pass gravity vector.

**Paper** does not use this. The InEKF estimates gravity direction itself from accelerometer over time.

**Impact**: Not needed by our model. However, `GRAVITY X/Y/Z` could be used to:
- Initialize pitch/roll at trip start more accurately.
- Subtract from raw accelerometer to get linear acceleration: `linear_accel = ACCELEROMETER - GRAVITY`.

**Recommendation**: Use `ACCELEROMETER` (raw, includes gravity) as model input — same as paper. Optionally use `linear_accel` as an alternative experiment.

---

## 3. In-App Data Collection: GPS-Triggered Dense Logging

When GPS is available during active navigation, we can collect **more data points per unit time** to build a richer training set. The key insight: we don't need to limit logging to one point per 10 Hz step — we can log sub-segment data.

### 3.1 Logging Strategy at 10 Hz with GPS Available

```
NAVIGATION LOOP (10 Hz = every 100ms):

  Every step t_k:
  ──────────────────────────────────────────────
  [ALWAYS LOG] — used for future retraining:
    - imu_window[k]     : float32 (9, 20) — current 2-sec sliding window
    - inekf_pos[k]      : float32 (3,)   — filter ENU output
    - inekf_vel[k]      : float32 (3,)   — filter velocity
    - timestamp[k]      : int64 ms

  [LOG ONLY IF GPS_RELIABLE]:
    Condition: GPS_ACCURACY < 5m AND parse_satellites(GPS_SATELLITES) > 10
               AND GPS_SPEED > 2 km/h
    
    - gps_lat[k]        : float64 degrees  ← HIGH PRECISION — always log float64
    - gps_lon[k]        : float64 degrees
    - gps_speed_mps[k]  : float32 m/s     ← GPS_SPEED (Kmh) / 3.6
    - gps_accuracy[k]   : float32 m
    - gps_heading[k]    : float32 degrees  ← GPS ORIENTATION (°)
    - n_satellites[k]   : int16            ← parsed from ratio string
    - gps_reliable[k]   : bool             ← True for this step

  [LOG ONLY IF GPS DROPOUT START]:
    - dropout_start_pos : float32 (3,)    ← ENU position at GPS loss
    - dropout_start_t   : int64 ms
```

### 3.2 Minimum Example Threshold Before Triggering Retraining

Do NOT retrain after every trip. Only retrain when enough quality examples have accumulated:

| Condition | Threshold | Reason |
| :--- | :---: | :--- |
| Total GPS-reliable steps logged | **≥ 500 steps** (50 sec GPS) | Minimum for meaningful E_trel computation |
| Number of GPS-reliable **segments** (≥50 steps each) | **≥ 5 segments** | Need diverse driving contexts |
| Total travel distance with GPS | **≥ 500 m** | Sub-100m segments excluded from E_trel |
| Time since last retrain | **≥ 24 hours** | Avoid over-fitting to single short trip |

These thresholds ensure the training signal is statistically meaningful before committing a gradient update.

### 3.3 Minimum Example Check Logic

```python
def should_retrain(trip_log):
    reliable_steps = sum(trip_log['gps_reliable'])
    segments = build_segments(trip_log, min_len=50)
    total_dist = sum(segment_distance(s) for s in segments)
    hours_since_last = (now() - last_retrain_timestamp) / 3600

    return (
        reliable_steps >= 500 and
        len(segments) >= 5 and
        total_dist >= 500.0 and          # meters
        hours_since_last >= 24.0
    )
```

---

## 4. Summary: What the Paper Had vs. What We Have

| Data / Assumption | Paper Had | We Have | Verdict |
| :--- | :--- | :--- | :---: |
| IMU sampling rate | 200 Hz raw | 10 Hz | ⚠️ Mitigated: window time extended |
| Speed ground truth | NovAtel velocity ±0.01 m/s | GPS SPEED (Doppler) ±0.1 m/s | ✅ Good enough |
| Position ground truth | NovAtel ±0.01m | GPS ±3–10m (filtered) | ⚠️ Noisier — use strict accuracy filter |
| Attitude ground truth | NovAtel SPAN ±0.001° | Android ORIENTATION (fused, ±1–3°) | ⚠️ Moderate — acceptable for heading |
| Magnetometer | Not used | ✅ Available | ✅ Our extension |
| Static phone mount | Rigid bracket | Dynamic (cup holder, hand) | ⚠️ InEKF self-calibrates $R^v_s$ |
| Wheel speed / odometer | Not used (by design) | V-dataset only (not in app) | ✅ Not needed |
| Satellite count | Clean integer | String ratio "N / M" | ⚠️ Parse needed |
| Gravity vector | Not used | Available | ✅ Optional initialization aid |
