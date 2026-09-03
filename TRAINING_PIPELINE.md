# Training Pipeline & Dataset Engineering Guide for Tri-Stream AVNet

This document outlines the complete end-to-end training pipeline for the upgraded **Tri-Stream Multi-Modal AVNet (AVNet-Mag)** and **AdapterNet-9Axis**, designed specifically for your local dataset (`IO-VNBD` 10 Hz smartphone & vehicle benchmark). It also details data augmentation, loss formulations, training scripts, curriculum learning, and external complementary public datasets.

---

## 1. Local Dataset Structure & Inventory

Your workspace directory contains the official **IO-VNBD (Inertial and Odometry Vehicle Navigation Benchmark Dataset)**:

- **Total CSV Files**: **288 files** (**~855 MB**)
  - **Smartphone Files (`S-*.csv`)**: **144 files** (uniform $10.0\ \text{Hz}$, 24 columns).
  - **Vehicle Reference Files (`V-*.csv`)**: **144 files** (chassis CAN bus wheel odometry, indicated speed, steering angle, high-precision GNSS/INS).
- **Driver Categories Available**:
  - `M (Driver B)`: Long urban/suburban driving (~3 hours in `S-M.csv`).
  - `S (Driver A)`: Highway and city commutes.
  - `Vf`, `Vta`, `Vtb`, `Vw` (`Driver E`): Complex roundabouts, stop-and-go maneuvers, and speed variations.
  - `Y (Driver D)`: Long continuous endurance tracks.

---

## 2. Model Breakdown: Which Models Are Trainable?

In this architecture, there are **2 trainable neural network models** and **1 non-learning analytical filter**:

| Model Name | Type | Trainable? | Number of Parameters | What It Learns / Optimizes |
| :--- | :---: | :---: | :---: | :--- |
| **1. TriStream AVNet-Mag** | Deep Neural Net (CNN-BiGRU-Attention) | **YES** | **268,357** | **Physical ego-motion**: Maps raw IMU + Mag streams to longitudinal vehicle speed ($v_{\text{lon}}$) and relative attitude rotation ($\Delta q$). |
| **2. AdapterNet-9Axis** | Dilated 1D CNN | **YES** | **7,020** | **Uncertainty & Noise Adaptation**: Learns dynamic scaling factors $\mathbf{q}(t)$ and $\mathbf{n}(t)$ that scale process noise $\mathbf{Q}$ and measurement noise $\mathbf{N}$ based on magnetic disturbances and driving harshness. |
| **3. InEKF Engine** | Lie-Group Kalman Filter | **NO** (Analytical) | **0** | **Kinematic State Fusion**: Closed-form mathematical integrator on Lie group $GSE_2(3) \times SO(3)$. No backprop required; runs analytical Riccati equations. |

---

## 3. End-to-End Training Pipeline Overview

Both trainable neural networks are trained via a structured two-phase **Pretraining $\to$ Fine-Tuning** pipeline:

```
                            TRAINING PIPELINE FLOW
                            
   [Phase 1: Multi-Driver Pre-training]
   100+ S-*.csv files from Uncategorised & Diverse Drivers
   Window: W=20 (2.0s @ 10Hz) with Stride S=2
               │
               ▼
   [Data Preprocessing & Normalization]
   - Parse Accel (m/s²), Gyro (rad/s), Mag (μT)
   - Handle 'latin1' / UTF-8 sensor encoding
   - Z-score normalization per stream (fit on train split)
   - Robust outlier clipping (3-sigma)
               │
               ├───────────────────────────────────────────┐
               ▼                                           ▼
   [Trainable Model 1: AVNet-Mag]             [Trainable Model 2: AdapterNet]
   Supervised Motion Regression:              Self-Supervised / Residual Loss:
   - Huber Speed Loss (v_lon vs GPS)          - Loss = ||r_k||_N^-1 + log|N|
   - Chordal Geodesic Loss (dq vs True Rot)   - Scales Q during violent turns
   - Optimizer: AdamW (lr = 1e-3)             - Scales N up during Mag spikes
               │                                           │
               ▼ Saves: "tristream_avnet_base.pth"         ▼ Saves: "adapter_base.pth"
   [Phase 2: Target Driver / Mount Fine-Tuning]
   Target Sequence (e.g., Driver B "S-M.csv" or specific vehicle)
   - Freeze Conv branches: acc_branch, gyro_branch, mag_branch
   - Train GRU + Attention + Prediction Heads (lr = 1e-4)
               │
               ▼ Saves: "tristream_avnet_finetuned.pth"
   [Phase 3: Filter Evaluation with InEKF (Analytical - 0 Parameters)]
   Feed predicted (v_lon, dq) from Model 1 + dynamic (Q, N) from Model 2 into InEKF
   -> Compute ATE (RMSE) and Relative Translation Error (E_trel)
```

---

## 3. Dataset Preprocessing & Windowing Mathematics

### 3.1 Input Formulation ($B \times 3 \times W$)
For each window of length $W = 20$ (corresponding to $2.0\ \text{seconds}$ at $10\ \text{Hz}$):
- **Acceleration Stream**:
  $$\mathbf{X}_{\text{acc}} = \begin{bmatrix} a_x(t) & \dots & a_x(t + W - 1) \\ a_y(t) & \dots & a_y(t + W - 1) \\ a_z(t) & \dots & a_z(t + W - 1) \end{bmatrix} \in \mathbb{R}^{3 \times 20}$$
- **Angular Velocity Stream**:
  $$\mathbf{X}_{\text{gyro}} = \begin{bmatrix} \omega_x(t) & \dots & \omega_x(t + W - 1) \\ \omega_y(t) & \dots & \omega_y(t + W - 1) \\ \omega_z(t) & \dots & \omega_z(t + W - 1) \end{bmatrix} \in \mathbb{R}^{3 \times 20}$$
- **Magnetic Flux Stream**:
  $$\mathbf{X}_{\text{mag}} = \begin{bmatrix} m_x(t) & \dots & m_x(t + W - 1) \\ m_y(t) & \dots & m_y(t + W - 1) \\ m_z(t) & \dots & m_z(t + W - 1) \end{bmatrix} \in \mathbb{R}^{3 \times 20}$$

### 3.2 Target Formulation (Supervised Labels)
1. **Speed Target ($v^v_{\text{lon}}$)**:
   Extract `GPS SPEED (Kmh)` at the end of the window ($t + W - 1$) and convert to $\text{m/s}$:
   $$y_{\text{speed}} = \frac{v_{\text{GPS}}\ (\text{km/h})}{3.6} \in \mathbb{R}^1$$
2. **Relative Attitude Delta ($\Delta \mathbf{q}$)**:
   Extract orientation angles (Yaw $\psi$, Pitch $\theta$, Roll $\phi$) at window start $t$ and window end $t + W - 1$.
   Convert each to unit quaternion $\mathbf{q}(t)$ and $\mathbf{q}(t + W - 1)$.
   Compute relative rotation:
   $$\Delta \mathbf{q} = \mathbf{q}(t)^{-1} \otimes \mathbf{q}(t + W - 1)$$
   The target vector is the 3-vector imaginary component:
   $$\mathbf{y}_{\text{att}} = [\Delta q_x,\, \Delta q_y,\, \Delta q_z]^\top \in \mathbb{R}^3$$
   *(Enforcing canonical hemisphere: if $\Delta q_w < 0$, negate $\Delta \mathbf{q} \to -\Delta \mathbf{q}$)*.

---

## 4. Multi-Task Loss Function & Optimization

The network trains concurrently on velocity regression and geometric attitude change:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{speed}} + \lambda_{\text{att}} \mathcal{L}_{\text{att}} + \lambda_{\text{reg}} \|\Theta\|_2^2$$

### 4.1 Smooth L1 (Huber) Speed Loss:
$$\mathcal{L}_{\text{speed}} = \frac{1}{B} \sum_{i=1}^B \text{Huber}\left(\hat{v}_{\text{lon}}^{(i)} - y_{\text{speed}}^{(i)}\right)$$
*Huber loss prevents large GPS outliers (e.g. multi-path velocity jumps) from corrupting gradients.*

### 4.2 Quaternion Geodesic Attitude Loss:
Instead of naive Euclidean distance, we penalize the chordal distance on $SO(3)$:
$$\mathcal{L}_{\text{att}} = \frac{1}{B} \sum_{i=1}^B \left( 1 - |\hat{\mathbf{q}}^{(i)} \cdot \mathbf{q}_{\text{true}}^{(i)}| \right) + \|\Delta \hat{\mathbf{q}}_{xyz}^{(i)} - \mathbf{y}_{\text{att}}^{(i)}\|^2$$
where $\lambda_{\text{att}} = 5.0$ balances gradient magnitudes between velocity ($\text{m/s}$) and unit quaternion units.

---

## 5. Data Augmentation for Inertial Dead Reckoning

To make the model resilient against different vehicle chassis, road vibrations, and sensor imperfections:

1. **Random Bias Drift Injection**:
   Simulate low-frequency thermal gyro bias:
   $$\tilde{\boldsymbol{\omega}}_{\text{aug}} = \tilde{\boldsymbol{\omega}} + \mathbf{b}_{\text{rand}}, \quad \mathbf{b}_{\text{rand}} \sim \mathcal{N}\left(0,\, (0.01\ \text{rad/s})^2\right)$$
2. **Sensor Noise Jitter**:
   Add zero-mean Gaussian jitter to accel and mag channels:
   $$\tilde{\mathbf{a}}_{\text{aug}} = \tilde{\mathbf{a}} + \boldsymbol{\epsilon}_a, \quad \boldsymbol{\epsilon}_a \sim \mathcal{N}(0,\, 0.05^2\ \text{m/s}^2)$$
   $$\tilde{\mathbf{m}}_{\text{aug}} = \tilde{\mathbf{m}} + \boldsymbol{\epsilon}_m, \quad \boldsymbol{\epsilon}_m \sim \mathcal{N}(0,\, 0.5^2\ \mu\text{T})$$
3. **Gravity Alignment Jitter (Pitch/Roll Perturbation)**:
   Randomly rotate the 3D frame around horizontal axes by small perturbations ($\pm 2^\circ$) to simulate slight phone mount wiggles in car holders.

---

## 6. Training & Fine-Tuning Execution Script

Below is the concrete configuration for running the multi-file training and fine-tuning pipeline:

```python
# avnet/train_tristream.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from avnet.models.avnet import TriStreamAVNet
from avnet.dataset import MultiFileTriStreamDataset

# 1. Hyperparameters
WINDOW_SIZE = 20     # 2.0 s @ 10 Hz
STRIDE = 2           # 80% overlap between windows
BATCH_SIZE = 64
PRETRAIN_EPOCHS = 25
FINETUNE_EPOCHS = 10
LR_BASE = 1e-3
LR_FINETUNE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Datasets
train_files = [
    # Multiple uncategorised S-*.csv files for diverse road geometry
    "data/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-vtb1.csv",
    "data/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-vta2.csv",
    "data/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-Vw1.csv",
    "data/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-Y1.csv",
]
val_files = [
    "data/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-vtb2.csv"
]
target_finetune_file = "data/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/S-M.csv"

# 3. Model & Optimizer
model = TriStreamAVNet(window_size=WINDOW_SIZE).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR_BASE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS)

# 4. Phase 1: Pretraining on Diverse Fleet
print("Starting Phase 1: Fleet Pre-training...")
# Loop through train_loader, compute Huber speed loss + quaternion loss, backprop, validate

# 5. Phase 2: Domain Adaptation / Driver B Fine-Tuning
print("Starting Phase 2: Driver-Specific Fine-tuning...")
# Freeze Conv feature extractors:
for p in model.acc_branch.parameters(): p.requires_grad = False
for p in model.gyro_branch.parameters(): p.requires_grad = False
for p in model.mag_branch.parameters(): p.requires_grad = False

finetune_opt = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_FINETUNE
)
# Train for 10 epochs on S-M.csv
```

---

## 7. Recommended External Complementary Datasets

If you wish to augment your training with public datasets beyond the 288 files already in your repository:

| Dataset Name | Source / Institution | Sensors & Sampling Rate | Best Use Case for Our Project | Download / Access |
| :--- | :--- | :---: | :--- | :--- |
| **IO-VNBD Benchmark (Full)** | University of Nottingham / MDPI 2021 | 10 Hz Smartphone (Android) + Vehicle CAN + RTK GNSS | Matches our local dataset format 1-to-1. | [MDPI Publication & Repo](https://doi.org/10.3390/s21123992) |
| **Wuhan IO-VNBD (AVNet Paper)** | Wuhan University (Long Qian et al., 2025) | 200 Hz Smartphone (Huawei Mate 30) vs NovAtel SPAN ISA-100C | High-rate IMU baseline from the paper authors. | [Satellite Navigation Paper](https://doi.org/10.1186/s43020-025-00168-7) |
| **Google Smartphone Decimeter Challenge (GSDC)** | Google & ION GNSS+ (2021–2023) | Raw Android GNSS + Accelerometer/Gyroscope across 100+ urban & highway trips | Evaluating GNSS-denied tunnel sequences (e.g. LAX tunnel). | [Kaggle GSDC Contest](https://www.kaggle.com/c/google-smartphone-decimeter-challenge) |
| **OxIOD (Oxford Inertial Odometry Dataset)** | Oxford University (Chen et al.) | 100 Hz Phone IMU across 158 km (handheld, pocket, bag, car mount) | Training sensor invariance across different phone placement postures. | [Oxford OxIOD Webpage](http://deepio.cs.ox.ac.uk/) |
| **KITTI Odometry Benchmark** | Karlsruhe Institute of Technology | OXTS RT3003 (100 Hz IMU + RTK GPS) on vehicle roof | Golden standard benchmark for relative translation ($E_{\text{trel}}$) & rotation ($E_{\text{rrel}}$). | [KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) |

---

## 8. Summary Checklist Before Running Training

1. **Verify Sampling Uniformity**: Ensure time deltas do not have missing gaps $> 500\ \text{ms}$ (if detected, split window into separate segments).
2. **Channel Normalization**: Normalize Accel ($0 \pm 1\ \text{g}$), Gyro ($0 \pm 0.1\ \text{rad/s}$), and Mag ($0 \pm 30\ \mu\text{T}$) to comparable $[-1, 1]$ numerical bounds.
3. **Filter Velocity Label**: Discard samples where `GPS ACCURACY (m) > 10.0` or speed drops under zero due to multipath spikes.
4. **Evaluate with InEKF**: After obtaining model weights, run the InEKF state estimator on unseen test sequences to measure real-world trajectory drift.
