# Training Guide: Module 1 — Tri-Stream AVNet-Mag

## Overview

Module 1 (`TriStreamAVNet`) is the **universal physics backbone**. It maps raw 9-axis IMU windows to forward speed and relative rotation. This model is trained **offline** on a large diverse dataset, then kept **frozen** during normal use. It is the most expensive to train but also the most stable — you train it once and deploy everywhere.

---

## What Data It Takes as Training Input

### Input Tensors (per window):

| Tensor | Shape | Unit | Source Column(s) in S-*.csv |
| :--- | :---: | :---: | :--- |
| Accel Stream | `(B, 3, 20)` | m/s² | `ACCELEROMETER X`, `Y`, `Z` |
| Gyro Stream | `(B, 3, 20)` | rad/s | `GYROSCOPE Yaw`, `Pitch`, `Roll` |
| Mag Stream | `(B, 3, 20)` | μT | `MAGNETIC FIELD X`, `Y`, `Z` |

- Window size `W = 20` samples = **2.0 seconds at 10 Hz**
- Sliding stride = 2 samples (90% overlap for maximum training signal density)

### Supervised Labels (per window):

| Label | Shape | Source | Derivation |
| :--- | :---: | :--- | :--- |
| `y_speed` | `(B, 1)` | `GPS SPEED (Kmh)` at window-end | Divide by 3.6 → m/s. Filter: `GPS ACCURACY < 10m` |
| `y_att` | `(B, 3)` | `ORIENTATION (Yaw/Pitch/Roll)` | Compute relative quaternion `q(t_end)^{-1} ⊗ q(t_start)`, take imaginary part `[dq_x, dq_y, dq_z]`. Enforce canonical hemisphere: negate if `dq_w < 0`. |

---

## What the Training Produces

Training minimizes a **multi-task loss** jointly across both heads:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{speed}} + \lambda_{\text{att}} \cdot \mathcal{L}_{\text{att}}$$

where $\lambda_{\text{att}} = 5.0$ (attitude loss weighted higher since heading drift is the dominant source of trajectory error).

### Speed Head Loss (Smooth L1 / Huber):
$$\mathcal{L}_{\text{speed}} = \frac{1}{B} \sum_i \text{Huber}\!\left(\hat{v}_{\text{lon}}^{(i)} - y_{\text{speed}}^{(i)}\right)$$

Huber used over MSE because occasional GPS multi-path spikes produce velocity outliers that would corrupt gradients with squared loss.

### Attitude Head Loss (Geodesic Chordal):
$$\mathcal{L}_{\text{att}} = \frac{1}{B}\sum_i \left(1 - \left|\hat{\mathbf{q}}^{(i)} \cdot \mathbf{q}^{(i)}_{\text{true}}\right|\right) + \left\|\Delta\hat{\mathbf{q}}^{(i)}_{xyz} - \mathbf{y}^{(i)}_{\text{att}}\right\|^2$$

This penalizes rotation error on the $SO(3)$ manifold, not in flat Euclidean space, giving geometrically correct gradients through large rotation angles.

### What It Spits Out After Training:
- **`checkpoints/avnet_mag_base.pth`**: Frozen base weights (~268k params, 1.02 MB).
- A model that generalizes across any driver, any vehicle, any road type present in the training corpus.

---

## Training Configuration

```python
# Key hyperparameters
WINDOW_SIZE  = 20       # 2.0 s @ 10 Hz
STRIDE       = 2        # 90% overlap
BATCH_SIZE   = 64
EPOCHS       = 25       # Phase 1 (pretrain)
LR           = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_ATT   = 5.0      # Attitude loss weight
SCHEDULER    = CosineAnnealingLR(T_max=EPOCHS)
EARLY_STOP   = 7        # Patience epochs on val loss
```

### Training Data Split (from your 288 CSV files):

| Split | Files | Purpose |
| :--- | :--- | :--- |
| **Train** (80%) | ~100+ uncategorised `S-*.csv` runs | Learn diverse road geometries and maneuvers |
| **Validation** (10%) | ~15 held-out sequences | Monitor overfitting; trigger early stop |
| **Test** (10%) | Specific driver sequences (e.g. `S-M.csv` Driver B) | Final closed-loop InEKF trajectory error evaluation |

### Data Augmentation Applied During Training:
1. **Gyro bias drift injection**: $\tilde{\boldsymbol{\omega}} \mathrel{+}= \mathcal{N}(0,\, 0.01^2)$ rad/s
2. **Accel noise jitter**: $\tilde{\mathbf{a}} \mathrel{+}= \mathcal{N}(0,\, 0.05^2)$ m/s²
3. **Mag noise jitter**: $\tilde{\mathbf{m}} \mathrel{+}= \mathcal{N}(0,\, 0.5^2)$ μT
4. **Gravity axis perturbation**: Random pitch/roll frame tilt ±2° (simulates phone mount wobble)

---

## When and Why to Fine-Tune (Phase 2)

After pretraining, you can run a lightweight **domain adaptation** fine-tune if deploying on a specific device/driver:

- **Freeze**: All 3 Conv branches (`acc_branch`, `gyro_branch`, `mag_branch`)
- **Unfreeze**: BiGRU, Attention Pooling, DDODO head, DDATT head
- **LR**: `1e-4` (10× lower than pretraining)
- **Epochs**: 5–10
- **Data**: 1–2 specific sequences for that driver or phone model

This adapts temporal dynamics and head calibration without disturbing the low-level feature extractors.

---

## Summary

| Aspect | Detail |
| :--- | :--- |
| **Train when** | Once offline on full 288-file corpus |
| **Train with** | Accel + Gyro + Mag streams + GPS speed/heading labels |
| **Train for** | ~25 epochs (~2–4 hours on a laptop GPU) |
| **Output** | `avnet_mag_base.pth` (1.02 MB, frozen for deployment) |
| **In-app retraining** | **NO** — too many params, catastrophic forgetting risk |
