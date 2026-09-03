# Training Guide: Module 3 — InEKF (Lie-Group Kalman Filter)

## Overview

Module 3 (`InEKF`) is **not a neural network**. It is a closed-form mathematical estimator operating on the matrix Lie group $GSE_2(3) \times SO(3) \times \mathbb{R}^9$. It has **zero trainable parameters** — no gradient descent, no `.pth` file, no training data required.

However, it has **tunable initialization parameters** and **physically-grounded default noise values** that must be set correctly for good performance. This guide explains those.

---

## What the InEKF "Learns" — Initialization vs. Runtime

| Component | Trainable? | How Set | Source |
| :--- | :---: | :--- | :--- |
| State transition matrix $\mathbf{F}$ | NO | Derived analytically from kinematics | Paper equations (closed-form) |
| Noise input matrix $\mathbf{G}$ | NO | Derived analytically from kinematics | Paper equations (closed-form) |
| Measurement Jacobian $\mathbf{H}$ | NO | Derived analytically from measurement model | Paper equations (closed-form) |
| Default process noise $\mathbf{Q}_0$ | NO (fixed defaults) | From IMU datasheet / empirical tuning | See defaults below |
| Default measurement noise $\mathbf{N}_0$ | NO (fixed defaults) | Empirical tuning on training sequences | See defaults below |
| Initial state $\hat{\mathbf{x}}_0$ | **Per-trip init** | First GPS fix + heading from orientation sensor | Runtime initialization |
| Initial covariance $\mathbf{P}_0$ | **Per-trip init** | Conservative diagonal prior | See defaults below |
| Scaling factor $\beta$ in AdapterNet | NO (used by Module 2) | Fixed at $\beta = 3$ | Paper specification |

---

## Fixed Default Noise Parameters

These are the nominal values around which AdapterNet dynamically scales:

### Process Noise Defaults (IMU Noise Model):

| Noise Source | Symbol | Default $\sigma$ | Physical Meaning |
| :--- | :---: | :---: | :--- |
| Gyroscope white noise | $\sigma_\omega$ | $1 \times 10^{-2}$ rad/s | Angular rate measurement noise |
| Accelerometer white noise | $\sigma_f$ | $1 \times 10^{-2}$ m/s² | Specific force measurement noise |
| Gyro bias random walk | $\sigma_{\delta\omega}$ | $1 \times 10^{-4}$ rad/s | Rate of bias drift |
| Accel bias random walk | $\sigma_{\delta f}$ | $1 \times 10^{-3}$ m/s² | Rate of bias drift |
| Mounting rotation noise | $\sigma_{R^v_s}$ | $1 \times 10^{-4}$ rad | Stability of phone mount |
| Lever arm noise | $\sigma_{p^s_v}$ | $1 \times 10^{-4}$ m | Stability of phone position |

### Measurement Noise Defaults (AVNet Output Uncertainty):

| Measurement | Symbol | Default $\sigma$ | Physical Meaning |
| :--- | :---: | :---: | :--- |
| Attitude (DDATT) | $\sigma_{R^w_s}$ | $1 \times 10^{-2}$ rad | Uncertainty in AVNet's predicted rotation |
| Forward speed (DDODO) | $\sigma_{v^v_s}$ | $1.0$ m/s | Uncertainty in AVNet's predicted speed |

> **Note**: AdapterNet multiplies these defaults by $10^{3\tanh(\cdot)}$, giving a dynamic range of $[10^{-3}, 10^{+3}]$ around each nominal value. So AdapterNet does not replace these defaults — it scales them. Getting the defaults right is important.

---

## Initial State at Trip Start

At the beginning of every trip, the InEKF state is initialized from the first $\sim$5 seconds of available sensor data:

```
Initialization Procedure (at t = 0):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Wait for GPS accuracy < 5m and satellites > 8

2. Set initial position:
   p^w_s(0) = [0, 0, 0]   ← local ENU origin at start point

3. Set initial heading:
   ψ_0 = GPS ORIENTATION (degrees) if vehicle moving (v_GPS > 2 m/s)
      or = ORIENTATION (Yaw) from phone fusion sensor if stationary

4. Set initial pitch and roll from gravity vector:
   θ_0 = arctan2(a_y, a_z)  (pitch from static accel)
   φ_0 = arctan2(-a_x, a_z) (roll from static accel)

5. Build R^w_s(0) = Rz(ψ_0) * Ry(θ_0) * Rx(φ_0)

6. Set initial velocity:
   v^w_s(0) = [0, 0, 0]  ← assume starting from rest, or
   v^w_s(0) = v_GPS_mps * [cos(ψ_0), sin(ψ_0), 0]  if moving

7. Set initial gyro/accel biases:
   δω(0) = [0, 0, 0]  (unknown; will converge within ~10 seconds)
   δf(0)  = [0, 0, 0]

8. Set initial extrinsics:
   R^v_s(0) = I₃  (assume aligned; let filter estimate true mounting)
   p^s_v(0) = [0, 0, 0]

9. Set initial covariance P(0):
   [σ_R = 0.1 rad, σ_v = 1.0 m/s, σ_p = 1.0 m,
    σ_δω = 0.01 rad/s, σ_δf = 0.1 m/s²,
    σ_Rvs = 0.1 rad, σ_psv = 0.1 m]  (all diagonal, conservative)
```

---

## How to Verify InEKF Tuning (No Training, Just Diagnosis)

Since InEKF is analytical, "training" it means verifying its noise parameters are correct. The key diagnostic tool is the **Normalized Innovation Squared (NIS)**:

$$\text{NIS}_k = \mathbf{r}_k^\top \mathbf{S}_k^{-1} \mathbf{r}_k, \quad \mathbf{S}_k = \mathbf{H}\mathbf{P}^-\mathbf{H}^\top + \mathbf{N}$$

For a well-tuned filter, $\text{NIS}_k$ should be chi-squared distributed with 6 degrees of freedom:
$$\mathbb{E}[\text{NIS}_k] = 6$$

| $\mathbb{E}[\text{NIS}]$ | Diagnosis | Fix |
| :---: | :--- | :--- |
| $\approx 6$ | ✅ Filter correctly calibrated | No action needed |
| $\gg 6$ | Filter **over-confident**: $\mathbf{N}$ too small or $\mathbf{Q}$ too small | Increase default $\sigma$ values |
| $\ll 6$ | Filter **under-confident**: $\mathbf{N}$ too large | Decrease default $\sigma$ values or let AdapterNet tighten it |

Run NIS analysis on your validation sequences to check if the defaults above need adjustment for your specific dataset.

---

## Summary

| Aspect | Detail |
| :--- | :--- |
| **Train when** | Never — no gradient-based training |
| **Tune when** | Once, on validation sequences, using NIS diagnostic |
| **What to tune** | Default $\sigma$ values for $\mathbf{Q}_0$ and $\mathbf{N}_0$ |
| **Runtime adaptation** | AdapterNet continuously scales $\mathbf{Q}$ and $\mathbf{N}$ around these defaults |
| **In-app behavior** | Initialized per-trip from GPS fix + orientation sensor |
| **Parameters** | 0 (fully analytical) |
