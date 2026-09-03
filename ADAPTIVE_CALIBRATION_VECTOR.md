# Adaptive Calibration Profile Vector (ACPV) Specification

## Overview

Rather than running complex and hazardous on-device neural network backpropagation (which risks model poisoning, requires a >100 MB PyTorch training runtime, and drains battery), AVNet adapts to individual smartphone sensor variations and vehicle dynamics using an **Adaptive Calibration Profile Vector (ACPV)**:

$$\boldsymbol{\theta}_{\text{profile}} \in \mathbb{R}^{14}$$

This compact 14-dimensional vector is stored locally in a lightweight JSON configuration (`calibration_profile.json`, ~250 bytes). It **updates smoothly after each drive** using recursive Exponential Moving Average (EMA). It captures physical sensor offsets, vehicle cabin magnetic distortion, habitual phone mounting angles, and road vibration profiles without altering the frozen neural weights of AVNet or AdapterNet.

---

## 1. Vector Structure & Physical Meaning

$$\boldsymbol{\theta}_{\text{profile}} = \begin{bmatrix}
\bar{\mathbf{b}}_\omega & \text{(3 floats: Persistent Gyroscope Bias Prior in rad/s)} \\
\bar{\mathbf{b}}_a      & \text{(3 floats: Persistent Accelerometer Bias Prior in m/s²)} \\
\bar{\mathbf{e}}_{\text{mount}} & \text{(3 floats: Habitual Mounting Euler Angles in degrees)} \\
\bar{\mathbf{b}}_m      & \text{(3 floats: Vehicle Cabin Hard-Iron Magnetic Offset in } \mu\text{T)} \\
s_{\text{vib}}          & \text{(1 float: Vehicle Chassis Vibration / Road Roughness Scale)} \\
s_{\text{gyro}}         & \text{(1 float: Hardware Gyroscope Noise Density Factor)}
\end{bmatrix}$$

### Detailed Parameter Breakdown:

| Parameter | Dim | Default Value | Physical Meaning & Role |
| :--- | :---: | :---: | :--- |
| $\bar{\mathbf{b}}_\omega$ | 3 | `[0.0, 0.0, 0.0]` | **Quasi-constant Gyro Hardware Bias**: Every smartphone MEMS IMU has a factory null-rate offset. Pre-loading this removes yaw integration drift from second zero. |
| $\bar{\mathbf{b}}_a$ | 3 | `[0.0, 0.0, 0.0]` | **Accelerometer Hardware Bias**: Zero-g acceleration offset. Prevents early velocity integration drift. |
| $\bar{\mathbf{e}}_{\text{mount}}$ | 3 | `[0.0, 0.0, 0.0]` | **Driver's Habitual Phone Mounting Orientation**: Most drivers place their phone in the same vent mount, cup holder, or charging pad every day. Caching this angle eliminates the 30-second convergence lag for extrinsic rotation $R^v_s$. |
| $\bar{\mathbf{b}}_m$ | 3 | `[0.0, 0.0, 0.0]` | **Cabin Hard-Iron Magnetic Dipole Offset**: The vehicle body is made of ferromagnetic steel and generates a persistent internal magnetic field. Subtracting $\bar{\mathbf{b}}_m$ before feeding the magnetometer to Tri-Stream AVNet restores true compass heading. |
| $s_{\text{vib}}$ | 1 | `1.0` | **Vehicle Road Vibration Multiplier**: A stiff sports car on gravel produces much higher IMU vibration than a luxury sedan on smooth asphalt. Scales the nominal process noise: $Q_0 \leftarrow Q_0 \cdot s_{\text{vib}}$. |
| $s_{\text{gyro}}$ | 1 | `1.0` | **Phone Gyro Noise Quality Scale**: Budget phones have noisier MEMS gyros than flagship phones. Scales the baseline gyro uncertainty in AdapterNet. |

---

## 2. How ACPV Interacts with the 3 Modules

```
                    ┌──────────────────────────────────────────────┐
                    │  Persistent Profile: calibration_profile.json │
                    │           \theta_profile (14 floats)         │
                    └──────────────────────┬───────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         │                                 │                                 │
         ▼ (Subtract Hard-Iron)            ▼ (Scale Noise Floor)             ▼ (Seed Prior State)
+──────────────────+             +──────────────────+             +──────────────────+
|     MODULE 1     |             |     MODULE 2     |             |     MODULE 3     |
| Tri-Stream AVNet |             |    AdapterNet    |             |      InEKF       |
|                  |             |                  |             |                  |
| Inputs:          |             | Inputs:          |             | Initial State:   |
|   a_raw          |             |   Raw IMU        |             |   \delta\omega = \bar{b}_\omega |
|   \omega_raw     |             | Modulated by:    |             |   \delta f     = \bar{b}_a      |
|   m - \bar{b}_m  |             |   s_vib, s_gyro  |             |   R^v_s        = R(mount)      |
+──────────────────+             +──────────────────+             +──────────────────+
```

### Module-by-Module Integration:

1. **Module 1 (Tri-Stream AVNet-Mag)**:
   - Before feeding the magnetometer stream into Conv Branch 3:
     $$\mathbf{m}_{\text{calibrated}}(t) = \mathbf{m}_{\text{raw}}(t) - \bar{\mathbf{b}}_m$$
   - This strips the car cabin's static magnetic field, leaving clean geomagnetic flux for heading prediction.
2. **Module 2 (AdapterNet)**:
   - Modulates the baseline nominal noise covariance:
     $$\sigma_{\omega}^{\text{eff}} = \sigma_\omega^{\text{default}} \cdot s_{\text{gyro}}, \qquad Q^{\text{eff}} = Q^{\text{nominal}} \cdot s_{\text{vib}}$$
   - AdapterNet then applies its dynamic neural multiplier $10^{3 \tanh(\cdot)}$ on top of this device-tailored baseline.
3. **Module 3 (InEKF)**:
   - At trip initialization ($t = 0$), the 21-state InEKF is **seeded with the converged profile vector**:
     $$\hat{\delta\omega}(0) = \bar{\mathbf{b}}_\omega, \quad \hat{\delta f}(0) = \bar{\mathbf{b}}_a, \quad \hat{R}^v_s(0) = R(\bar{\mathbf{e}}_{\text{mount}})$$
   - **Result**: The filter begins at 100% calibration immediately upon vehicle roll-out.

---

## 3. The Usage-Based Update Law (Post-Trip Adaptation)

At the end of each drive (trip duration $> 5$ minutes), the application reads the steady-state values converged by the InEKF during the drive and performs an **Exponential Moving Average (EMA)** update:

```python
def update_calibration_profile(current_profile, trip_results):
    """
    Update persistent calibration profile vector after a completed trip.
    Zero backpropagation, zero risk of model degradation.
    """
    # Only update if trip had sufficient duration and quality
    if trip_results.duration_s < 300: # < 5 minutes
        return current_profile

    # Adaptive learning rate: faster adaptation during first 5 trips
    trip_count = current_profile.get("trip_count", 0)
    alpha = 0.30 if trip_count < 5 else 0.10

    # 1. Update Gyroscope and Accelerometer Biases (from InEKF converged state)
    current_profile["b_gyro"] = (
        (1 - alpha) * np.array(current_profile["b_gyro"]) +
        alpha * trip_results.final_inekf_gyro_bias
    ).tolist()

    current_profile["b_accel"] = (
        (1 - alpha) * np.array(current_profile["b_accel"]) +
        alpha * trip_results.final_inekf_accel_bias
    ).tolist()

    # 2. Update Habitual Mounting Orientation (if driver used a consistent mount)
    if trip_results.mount_stability_score > 0.8: # mount didn't wobble
        current_profile["e_mount"] = (
            (1 - alpha) * np.array(current_profile["e_mount"]) +
            alpha * trip_results.final_inekf_mount_euler
        ).tolist()

    # 3. Update Vehicle Vibration Scale
    mean_vibration = trip_results.measured_high_freq_accel_std / 0.15 # nominal 0.15 m/s²
    current_profile["s_vib"] = float(
        np.clip((1 - alpha) * current_profile["s_vib"] + alpha * mean_vibration, 0.5, 3.0)
    )

    # 4. Update Cabin Magnetic Hard-Iron Offset (mean mag during 360-degree turns)
    if trip_results.completed_full_turn:
        current_profile["b_mag"] = (
            (1 - alpha) * np.array(current_profile["b_mag"]) +
            alpha * trip_results.estimated_hard_iron_offset
        ).tolist()

    current_profile["trip_count"] = trip_count + 1
    current_profile["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

    save_json("calibration_profile.json", current_profile)
    return current_profile
```

---

## 4. Lifecycle Comparison: Retraining vs. ACPV

| Dimension | In-App Neural Retraining | Adaptive Calibration Profile Vector (ACPV) |
| :--- | :---: | :---: |
| **Compute Overhead** | High (backward graph, GPU/CPU epochs) | **Zero** (single vector addition at trip end) |
| **App Size Impact** | +100 MB (LibTorch autograd engine) | **0 KB** (pure JSON file) |
| **Battery Drain** | Significant (background CPU burn) | **0%** (< 1 millisecond execution) |
| **Risk of Degradation** | High (bad GPS data corrupts weights) | **Zero** (EMA parameters clipped to safe ranges) |
| **Convergence Speed** | Unpredictable (requires many gradient steps) | **Fast & Smooth** (3–5 trips to full calibration) |
| **Inspectability** | Black-box weights (`.pth`) | **Human-readable JSON** (`calibration_profile.json`) |
| **Persistent Memory** | Unstable | **Robust across app restarts and updates** |

---

## 5. Sample Storage Format (`calibration_profile.json`)

```json
{
  "version": 1,
  "device_name": "SM-G991B (Samsung S21)",
  "trip_count": 12,
  "last_updated": "2026-09-03 18:45:12",
  "vector_14d": {
    "b_gyro_rad_s": [-0.00412, 0.01083, -0.00071],
    "b_accel_m_s2": [0.0381, -0.0142, 0.0895],
    "e_mount_deg": [-1.2, -22.4, 4.8],
    "b_mag_uT": [12.4, -6.8, 18.2],
    "s_vib": 1.15,
    "s_gyro": 1.08
  }
}
```

This vector provides **personalized, usage-based adaptation** to the user's specific phone and vehicle without touching the frozen neural weights.
