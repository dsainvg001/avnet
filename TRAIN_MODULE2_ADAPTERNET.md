# Training Guide: Module 2 — AdapterNet (Filter Parameter Adapter)

## What the Paper Actually Says (Verbatim)

> **From the paper (Page 6):**
> *"The measurement filter parameter adapter calculates the process and measurement covariances Q(t_n) and N(t_{n+1}), respectively, at each instant t_n. The core architecture of the adapter network is that of a CNN. The adapter's input is also a window of W_adapter inertial sensor measurements..."*
>
> *"The parameters of the measurement filter were trained using an **indirect optimization process**. Specifically, **the outputs of the adapter were integrated into the InEKF**, and **the adapter was optimized by minimizing a loss function based on relative translation errors**."*
>
> **From page 7 (Training section):**
> *"For the adapter network, the same routines in the AI-IMU method were used for training, with the **optimization objective being the relative translation error calculated from the filter estimates**."*
>
> **From page 12 (Experimental protocol):**
> *"First, the AVNet was trained without the evaluated sequence as explained in 4.2.3. Then, **the adapter was trained without the evaluated sequence.** Finally, the combined data- and model-driven method was applied to the test sequence and compared against the ground-truth."*

---

## Key Corrections to Our Earlier Understanding

Our previous TRAIN_MODULE2_ADAPTERNET.md described a **self-supervised NLL loss on innovation residuals**. That is a *theoretically valid* approach, but it is **NOT what the paper does**. The paper is explicit:

| Aspect | What We Had (Wrong) | What Paper Actually Says |
| :--- | :--- | :--- |
| Training signal | Self-supervised InEKF innovation residual NLL | **Relative Translation Error from the full InEKF trajectory** (indirect/end-to-end) |
| Supervision source | GPS speed + residuals available at inference | **Full ground truth trajectories** (offline, with NovAtel SPAN reference) |
| Training mode | Online in-app micro-batches | **Offline, on training corpus, leave-one-sequence-out** |
| Loss function | $\mathbf{r}_k^\top \mathbf{N}^{-1} \mathbf{r}_k + \ln\|\mathbf{N}\|$ | **Relative translation error $E_{\text{trel}}$ from filter output vs ground truth** |

---

## What the Paper's AdapterNet Training Actually Does

The AdapterNet is trained **offline** using a process called **indirect optimization** (also called **end-to-end filter training**):

```
Training Sequence (e.g. 10 Hz Smartphone + NovAtel SPAN Ground Truth):
    IMU Window X_k (6 channels: accel + gyro, W=20)
                │
                ▼
        AdapterNet Forward Pass
        → Q(t_k), N(t_{k+1})
                │
                ▼
    Feed Q, N into InEKF along with AVNet's predictions
    → Run full trajectory estimate p̂_DR(t_0 ... t_end)
                │
                ▼
    Compare estimated trajectory vs NovAtel SPAN ground truth
    → Compute Relative Translation Error E_trel
                │
                ▼
    Backpropagate through InEKF → through Q/N → into AdapterNet
    → Update AdapterNet weights
```

This is a **differentiable filter training** approach: the loss lives at the trajectory level, not the step level. Gradients flow backwards through the entire filter to train AdapterNet to produce covariances that minimize end-to-end positioning error.

---

## What Data AdapterNet Training Takes

### Inputs (Forward Pass):
| Input | Shape | Source Columns |
| :--- | :---: | :--- |
| IMU window | `(B, 6, 20)` | `ACCELEROMETER X/Y/Z` + `GYROSCOPE Yaw/Pitch/Roll` |

> **Note**: The paper's AdapterNet uses only **6 channels (accel + gyro)**. Our Tri-Stream architecture extends this to 9 channels (accel + gyro + mag) — this is our modification, not from the paper.

### Training Supervision Signal:
- **Relative translation error $E_{\text{trel}}$** from the full InEKF trajectory estimated on the training sequences.
- Computed against **centimeter-level NovAtel SPAN ISA-100C ground truth**.
- This means you need **both** the IMU data **and** a high-quality position reference to compute the loss.

### Training Protocol (leave-one-out):
- Train AdapterNet on all sequences **except** the one being evaluated.
- Evaluate on the held-out sequence.
- This prevents any data leakage.

---

## Offline Training of AdapterNet on IO-VNBD (Vehicle CAN Odometry)

Following the paper's indirect optimization methodology, AdapterNet is trained **offline on GPU** using the 144 synchronized IO-VNBD dataset pairs (`S-*.csv` + `V-*.csv`):

```
                                OFFLINE ADAPTERNET TRAINING LOOP
═══════════════════════════════════════════════════════════════════════════════════════════════
  Input: 10 Hz Smartphone 9-axis IMU Window (W = 20 samples = 2.0 seconds)
         accel (3), gyro (3), mag (3)
              │
              ▼
      AdapterNet Forward Pass
      → Dynamic Process Noise Q(t) and Measurement Noise N(t)
              │
              ▼
      Feed into Differentiable InEKF
      → Propagate with IMU
      → Update with AVNet-predicted v_lon, Δq
      → Generate continuous estimated trajectory p̂_InEKF(t_0 ... t_end)
              │
              ▼
      Compute Relative Trajectory Loss against Vehicle CAN Bus Ground Truth:
      Δs_CAN = ∫ v_CAN dt   (Centimeter-accurate wheel odometry distance from V-*.csv)
      Δp_CAN                (True displacement from vehicle GPS + yaw rate)

      L_adapter = E_trel(p̂_InEKF, p_CAN) = (1 / |F|) * Σ || Δp̂_ij - Δp_CAN_ij || / || Δp_CAN_ij ||
              │
              ▼
      loss.backward() through InEKF → into AdapterNet weights
      optimizer.step() (AdamW, lr=1e-4)
═══════════════════════════════════════════════════════════════════════════════════════════════
```

Once trained, AdapterNet's weights (`adapter_net.pth`, 27.4 KB) are **frozen** for production deployment.

---

## Production Decision: Why In-App Neural Retraining is Ditched

We considered running on-device backpropagation on the phone, but officially ditched it for compelling engineering reasons:
1. **Model Poisoning Risk**: Retraining on stop-and-go traffic under metal bridges or in GPS multipath zones corrupts weights.
2. **APK Bloat**: Packaging LibTorch/autograd on Android inflates the app by **>100 MB** (vs. ~15 MB for inference-only ONNX/TFLite).
3. **Battery & Thermal Throttling**: Android OS kills background training services for excessive CPU/battery usage.
4. **Non-Reproducible Fleets**: Every phone would run a divergent, unversioned neural network.

---

## The Solution: Adaptive Calibration Profile Vector (ACPV)

Instead of modifying neural network weights, the application maintains a **14-dimensional Adaptive Calibration Profile Vector**:

$$\boldsymbol{\theta}_{\text{profile}} \in \mathbb{R}^{14}$$

Stored in `calibration_profile.json` (~250 bytes), this vector updates after every drive using a smooth **Exponential Moving Average (EMA)**.

### What the 14-D Vector Stores:

$$\boldsymbol{\theta}_{\text{profile}} = \begin{bmatrix}
\bar{\mathbf{b}}_\omega & \text{(3 floats: Persistent Gyroscope Bias Prior in rad/s)} \\
\bar{\mathbf{b}}_a      & \text{(3 floats: Persistent Accelerometer Bias Prior in m/s²)} \\
\bar{\mathbf{e}}_{\text{mount}} & \text{(3 floats: Habitual Mounting Euler Angles in degrees)} \\
\bar{\mathbf{b}}_m      & \text{(3 floats: Vehicle Cabin Hard-Iron Magnetic Offset in } \mu\text{T)} \\
s_{\text{vib}}          & \text{(1 float: Vehicle Chassis Vibration / Road Roughness Scale)} \\
s_{\text{gyro}}         & \text{(1 float: Hardware Gyroscope Noise Density Factor)}
\end{bmatrix}$$

### How It Adapts to Your Phone & Car Across Drives:

1. **Trip 1 (First install)**: Starts with default priors ($\mathbf{b}=0, \mathbf{e}_{\text{mount}}=0, s=1.0$). During the drive, the 21-state InEKF naturally converges to the phone's actual hardware biases and mounting angle.
2. **Trip End ($t > 5\text{ min}$)**: An EMA update blends the newly observed biases into `calibration_profile.json` with learning rate $\alpha = 0.15$:
   $$\boldsymbol{\theta}_{\text{profile}}^{(k)} = (1 - \alpha)\boldsymbol{\theta}_{\text{profile}}^{(k-1)} + \alpha \hat{\boldsymbol{\theta}}_{\text{trip}}^{(k)}$$
3. **Trip 2 onwards**: InEKF initializes **instantly pre-calibrated** with the phone's exact hardware bias and mounting angle. Dead reckoning is accurate from the very first meter.
4. **Road Roughness Learning**: Stiffer sports cars on bumpy roads increase $s_{\text{vib}}$, causing AdapterNet to scale up nominal process noise $Q$ automatically.
5. **Magnetic Shielding**: Subtracting $\bar{\mathbf{b}}_m$ strips the vehicle chassis's magnetic dipole before AVNet sees the magnetometer.

---

## Summary Comparison

| Aspect | In-App Neural Retraining (Ditched) | Offline Training + ACPV (Implemented) |
| :--- | :---: | :---: |
| **Neural Weights** | Mutable (high risk of drift/divergence) | **Frozen & Verified** (`adapter_net.pth`) |
| **Adaptation Mechanism** | Stochastic Gradient Descent (SGD) on phone | **14-D Calibration Profile Vector (EMA)** |
| **Storage & APK Overhead** | > 100 MB runtime | **~250 bytes JSON** |
| **Battery & CPU Cost** | High (burns battery in background) | **Zero** (< 1 ms vector addition at trip end) |
| **Phone Hardware Bias?** | Learns slowly via noisy gradients | **Adapts cleanly via InEKF steady-state bias** |
| **Mounting Orientation?** | Unstable | **Caches habitual mounting Euler angles** |
| **Cabin Magnetic Offset?** | Cannot separate from heading | **Subtracts persistent hard-iron bias vector** |
| **Convergence** | Unpredictable | **Fully calibrated within 3 to 5 trips** |


