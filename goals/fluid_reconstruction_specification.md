# Physics-Integrated Fluid Reconstruction from Multi-View Video
## Complete Technical Specification Document

**Project:** Video-to-3D Fluid Reconstruction with Property Estimation  
**Author:** Makar Ulesov  
**Institution:** MBZUAI - Computer Vision and Metaverse Lab  
**Supervisor:** Dr. J. Alejandro Amador Herrera  
**Version:** 1.0  
**Date:** January 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Input Handling and Video Preprocessing](#3-input-handling-and-video-preprocessing)
4. [Temporal Synchronization](#4-temporal-synchronization)
5. [Camera Calibration and Multi-View Geometry](#5-camera-calibration-and-multi-view-geometry)
6. [3D Gaussian Splatting for Dynamic Fluids](#6-3d-gaussian-splatting-for-dynamic-fluids)
7. [Fluid Property Estimation](#7-fluid-property-estimation)
8. [Physics Constraints and Navier-Stokes Integration](#8-physics-constraints-and-navier-stokes-integration)
9. [Loss Functions and Optimization](#9-loss-functions-and-optimization)
10. [Complete Pipeline Architecture](#10-complete-pipeline-architecture)
11. [Data Structures](#11-data-structures)
12. [User Interface Specification](#12-user-interface-specification)
13. [Implementation Guidelines](#13-implementation-guidelines)
14. [Appendix: Mathematical Derivations](#appendix-mathematical-derivations)

---

## 1. Executive Summary

This document specifies a complete system for reconstructing 3D fluid dynamics from multi-view monocular video input. The system handles videos with different frame rates, resolutions, zoom levels, and start times (0.3-3 seconds offset). The output is a dynamic 3D Gaussian representation of the fluid with estimated physical properties: velocity field **u**, pressure field **p**, density field **ρ**, and viscosity **ν**.

### Key Challenges Addressed
- **Temporal Misalignment:** Videos may start at different times (0.3-3s offset)
- **Spatial Misalignment:** Different zoom levels and camera positions
- **Resolution Differences:** Videos may have different dimensions
- **Frame Rate Differences:** Videos may have different FPS
- **Physics Integration:** Extracted properties must satisfy Navier-Stokes equations

---

## 2. System Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT LAYER                                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                                      │
│  │ Video 1 │  │ Video 2 │  │ Video 3 │  (Different FPS, resolution, zoom)   │
│  └────┬────┘  └────┬────┘  └────┬────┘                                      │
└───────┼────────────┼────────────┼───────────────────────────────────────────┘
        │            │            │
        ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING LAYER                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ • Frame extraction at native FPS                                      │   │
│  │ • Metadata extraction (FPS, resolution, duration)                     │   │
│  │ • Initial feature detection (SIFT/ORB/SuperPoint)                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TEMPORAL SYNCHRONIZATION LAYER                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ • Optical flow computation per video                                  │   │
│  │ • Cross-correlation of motion signatures                              │   │
│  │ • Sub-frame offset estimation (0.3-3s range)                          │   │
│  │ • Common timeline establishment                                       │   │
│  │ • Frame interpolation for alignment                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CAMERA CALIBRATION LAYER                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ • Intrinsic estimation (focal length, principal point, distortion)    │   │
│  │ • Zoom-aware focal length correction                                  │   │
│  │ • Extrinsic estimation (rotation R, translation t)                    │   │
│  │ • Multi-view triangulation verification                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    3D GAUSSIAN SPLATTING LAYER                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ • Initialize Gaussians from point cloud                               │   │
│  │ • Time-varying Gaussian parameters                                    │   │
│  │ • Differentiable rendering from all views                             │   │
│  │ • Photometric loss optimization                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   FLUID PROPERTY ESTIMATION LAYER                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ • Velocity field from Gaussian motion trajectories                    │   │
│  │ • Density field from Gaussian opacity distribution                    │   │
│  │ • Pressure field from Navier-Stokes constraints                       │   │
│  │ • Viscosity estimation from velocity gradients                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ • 3D Gaussian representation G(t) = {μᵢ(t), Σᵢ(t), αᵢ(t), cᵢ(t)}     │   │
│  │ • Velocity field u(x, t) ∈ ℝ³                                         │   │
│  │ • Pressure field p(x, t) ∈ ℝ                                          │   │
│  │ • Density field ρ(x, t) ∈ ℝ⁺                                          │   │
│  │ • Viscosity ν ∈ ℝ⁺ (scalar or spatially varying)                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Components

| Component | Purpose | Key Outputs |
|-----------|---------|-------------|
| Video Preprocessor | Normalize inputs | Frames, metadata |
| Temporal Synchronizer | Align video timelines | Time offsets, common timeline |
| Camera Calibrator | Estimate camera parameters | K, R, t matrices |
| Gaussian Reconstructor | Build 3D representation | G(t) for each timestep |
| Physics Estimator | Extract fluid properties | u, p, ρ, ν fields |

---

## 3. Input Handling and Video Preprocessing

### 3.1 Input Specification

The system accepts exactly 3 video files of the same fluid scene from different viewpoints.

**Expected Variations:**
- Frame rates: 24-120 FPS (common: 30, 60 FPS)
- Resolutions: 720p to 4K
- Zoom levels: 1x to 5x optical/digital zoom
- Start time offsets: 0.3 to 3.0 seconds
- Codecs: H.264, H.265, VP9, ProRes

### 3.2 Metadata Extraction

For each video v ∈ {1, 2, 3}, extract:

```
VideoMetadata(v) = {
    fps_v:        float,           # Frames per second
    width_v:      int,             # Frame width in pixels
    height_v:     int,             # Frame height in pixels
    duration_v:   float,           # Total duration in seconds
    codec_v:      string,          # Video codec
    n_frames_v:   int,             # Total frame count
    bitrate_v:    int              # Bitrate for quality estimation
}
```

### 3.3 Frame Extraction

Extract all frames at native resolution and frame rate:

```
For video v with fps_v and n_frames_v:
    frames_v = [F_v^(i) for i in range(n_frames_v)]
    timestamps_v = [i / fps_v for i in range(n_frames_v)]
```

Where F_v^(i) ∈ ℝ^(H_v × W_v × 3) is the i-th frame of video v.

### 3.4 Initial Feature Detection

Detect features for synchronization and calibration:

**Option A: Classical Features (SIFT/ORB)**
```
For each frame F_v^(i):
    keypoints_v^(i), descriptors_v^(i) = SIFT(F_v^(i))
```

**Option B: Learned Features (SuperPoint)**
```
For each frame F_v^(i):
    keypoints_v^(i), descriptors_v^(i) = SuperPoint(F_v^(i))
```

SuperPoint is preferred for fluid scenes due to better handling of textureless regions.

### 3.5 Optical Flow Computation

Compute dense optical flow between consecutive frames:

```
For video v, for frames i and i+1:
    flow_v^(i→i+1) = OpticalFlow(F_v^(i), F_v^(i+1))
```

Where flow_v^(i→i+1) ∈ ℝ^(H_v × W_v × 2) contains (dx, dy) displacements.

**Recommended Methods:**
- RAFT (Recurrent All-Pairs Field Transforms) - highest accuracy
- PWC-Net - good balance of speed/accuracy
- Farneback - classical, fast, lower accuracy

---

## 4. Temporal Synchronization

### 4.1 Problem Statement

Given 3 videos with unknown start time offsets δ₁, δ₂, δ₃ (where one is reference with δ=0), find the offsets to align all videos to a common timeline.

**Constraint:** Offsets are in the range [0.3s, 3.0s] relative to the earliest video.

### 4.2 Motion Signature Extraction

Create a 1D motion signature from optical flow for each video:

```
Motion_v(t) = Σ_{x,y} ||flow_v^(t)||₂

Where:
- t is the frame index
- ||·||₂ is the L2 norm (magnitude) of flow vector
- Summation is over all pixels
```

This captures the total motion energy at each timestep.

**Enhanced Motion Signature (recommended):**
```
Motion_v(t) = [
    mean_magnitude:    mean(||flow_v^(t)||₂),
    std_magnitude:     std(||flow_v^(t)||₂),
    dominant_direction: arctan2(mean(flow_y), mean(flow_x)),
    spatial_variance:  var(flow_v^(t))
]
```

### 4.3 Cross-Correlation for Offset Estimation

To find the offset between video v₁ and v₂:

**Step 1: Normalize motion signatures**
```
M̃_v(t) = (Motion_v(t) - μ_v) / σ_v

Where μ_v = mean(Motion_v), σ_v = std(Motion_v)
```

**Step 2: Compute cross-correlation**
```
R₁₂(τ) = Σ_t M̃₁(t) · M̃₂(t + τ)
```

**Step 3: Find optimal offset**
```
δ₁₂ = argmax_τ R₁₂(τ)

Subject to: τ ∈ [-3.0s × fps_max, +3.0s × fps_max]
```

**Step 4: Sub-frame refinement**

Use parabolic interpolation around the peak:
```
Given R(τ-1), R(τ), R(τ+1) where τ is the discrete peak:

δ_refined = τ + 0.5 × (R(τ-1) - R(τ+1)) / (R(τ-1) - 2R(τ) + R(τ+1))
```

### 4.4 Multi-View Offset Resolution

With 3 videos, compute pairwise offsets:
```
δ₁₂ = offset(video1, video2)
δ₁₃ = offset(video1, video3)
δ₂₃ = offset(video2, video3)
```

**Consistency Check:**
```
|δ₁₂ + δ₂₃ - δ₁₃| < ε_sync

Where ε_sync ≈ 1/fps_min (one frame tolerance)
```

If inconsistent, use weighted least squares to find optimal offsets.

### 4.5 Common Timeline Establishment

Set the earliest starting video as reference (δ = 0):
```
δ_ref = min(0, δ₁₂, δ₁₃)  # Assuming video 1 is initial reference

Adjusted offsets:
δ'₁ = 0 - δ_ref
δ'₂ = δ₁₂ - δ_ref  
δ'₃ = δ₁₃ - δ_ref
```

**Common timeline parameters:**
```
t_start = 0
t_end = min(duration_v - δ'_v for v in {1,2,3})
fps_common = max(fps₁, fps₂, fps₃)  # Use highest FPS
n_frames_common = floor((t_end - t_start) × fps_common)
```

### 4.6 Frame Interpolation for Alignment

For each video v and common timeline timestamp t:

**Step 1: Find corresponding native frame position**
```
t_native = t + δ'_v
frame_idx_float = t_native × fps_v
frame_idx_low = floor(frame_idx_float)
frame_idx_high = ceil(frame_idx_float)
α = frame_idx_float - frame_idx_low
```

**Step 2: Interpolate frame**
```
F_v^aligned(t) = (1-α) × F_v^(frame_idx_low) + α × F_v^(frame_idx_high)
```

**For higher quality (optical flow-based interpolation):**
```
flow_forward = OpticalFlow(F_low, F_high)
flow_backward = OpticalFlow(F_high, F_low)

F_interpolated = FlowWarp(F_low, α × flow_forward) × (1-α) +
                 FlowWarp(F_high, (1-α) × (-flow_backward)) × α
```

---

## 5. Camera Calibration and Multi-View Geometry

### 5.1 Intrinsic Camera Parameters

The intrinsic matrix K describes the camera's internal geometry:

```
K = | f_x   0   c_x |
    |  0   f_y  c_y |
    |  0    0    1  |

Where:
- f_x, f_y: Focal lengths in pixels (f_x = f_y for square pixels)
- c_x, c_y: Principal point (usually image center)
```

### 5.2 Zoom-Aware Focal Length Estimation

Different zoom levels change the effective focal length:

**Relationship:**
```
f_zoomed = f_base × zoom_factor
```

**Estimation from video (if zoom metadata unavailable):**

Method 1: From field of view comparison
```
If known reference object size s_world is visible:
    s_pixels = measured size in pixels
    f = (s_pixels × distance) / s_world
```

Method 2: Self-calibration from motion
```
From feature tracks across frames, solve for f that minimizes
epipolar geometry error (requires sufficient camera motion)
```

Method 3: Relative zoom estimation between videos
```
Match features between videos on overlapping regions:
    zoom_ratio_v1_v2 = median(feature_scale_v1 / feature_scale_v2)
    
If f_v1 is known: f_v2 = f_v1 / zoom_ratio_v1_v2
```

### 5.3 Distortion Model

Radial and tangential distortion:
```
x_distorted = x(1 + k₁r² + k₂r⁴ + k₃r⁶) + 2p₁xy + p₂(r² + 2x²)
y_distorted = y(1 + k₁r² + k₂r⁴ + k₃r⁶) + p₁(r² + 2y²) + 2p₂xy

Where:
- r² = x² + y²
- k₁, k₂, k₃: Radial distortion coefficients
- p₁, p₂: Tangential distortion coefficients
```

For unknown cameras, estimate using feature matching across views.

### 5.4 Extrinsic Parameters Estimation

Extrinsics describe each camera's pose in world coordinates:
```
[R | t]_v = 3×4 matrix

Where:
- R ∈ SO(3): 3×3 rotation matrix
- t ∈ ℝ³: Translation vector
```

**Step 1: Feature Matching Between Views**
```
For views v_i and v_j:
    matches_{ij} = Match(descriptors_i, descriptors_j)
    
    Using: BFMatcher with ratio test (Lowe's ratio = 0.75)
    Or: SuperGlue for learned matching
```

**Step 2: Essential Matrix Estimation**
```
E = K_j^T × F × K_i

Where F is the fundamental matrix estimated from matches:
    F = RANSAC_8point(matches_{ij})
```

**Step 3: Decompose Essential Matrix**
```
[R, t] = DecomposeE(E)

This gives 4 possible solutions; select the one where
all triangulated points have positive depth in both cameras.
```

**Step 4: Bundle Adjustment**
```
Minimize reprojection error over all views and points:

min_{R_v, t_v, X_k} Σ_v Σ_k ||π(K_v, R_v, t_v, X_k) - x_v^k||²

Where:
- X_k: 3D point k
- x_v^k: 2D observation of point k in view v
- π(): Projection function
```

### 5.5 Scale Ambiguity Resolution

Multi-view reconstruction from monocular videos has scale ambiguity. Options:

**Option A: Known reference object**
```
If object of known size s_world is visible:
    scale = s_world / s_reconstructed
```

**Option B: Known baseline**
```
If distance between any two camera positions is known:
    scale = baseline_known / baseline_estimated
```

**Option C: Physics-based (for fluids)**
```
Use known fluid properties (e.g., gravity direction, expected flow speed)
to estimate scale during physics estimation phase.
```

### 5.6 Resolution Normalization

To handle different resolutions:

**Step 1: Define reference resolution**
```
ref_width = max(width_v for v in {1,2,3})
ref_height = max(height_v for v in {1,2,3})
```

**Step 2: Scale intrinsics accordingly**
```
For video v:
    scale_x = ref_width / width_v
    scale_y = ref_height / height_v
    
    K_v_normalized = | f_x × scale_x    0        c_x × scale_x |
                     |      0       f_y × scale_y  c_y × scale_y |
                     |      0           0              1         |
```

**Step 3: Upsample lower-resolution frames**
```
F_v_normalized = Resize(F_v, (ref_height, ref_width), method='bicubic')
```

---

## 6. 3D Gaussian Splatting for Dynamic Fluids

### 6.1 Gaussian Representation

Each 3D Gaussian is defined by:

```
G_i(t) = {
    μ_i(t) ∈ ℝ³:           Position (mean)
    Σ_i(t) ∈ ℝ^(3×3):      Covariance matrix (shape/orientation)
    α_i(t) ∈ [0,1]:        Opacity
    c_i(t) ∈ ℝ^k:          Color (k spherical harmonic coefficients, or RGB)
}
```

The covariance matrix is parameterized as:
```
Σ = R × S × S^T × R^T

Where:
- R ∈ SO(3): Rotation matrix (from quaternion q)
- S = diag(s_x, s_y, s_z): Scale matrix
```

### 6.2 Gaussian Splatting Rendering

**Step 1: Transform to camera space**
```
μ'_i = R_cam × μ_i + t_cam
Σ'_i = R_cam × Σ_i × R_cam^T
```

**Step 2: Project to 2D**

Using the Jacobian of the projection:
```
J = | f_x/z    0     -f_x×x/z² |
    |   0    f_y/z   -f_y×y/z² |

Σ_2D = J × Σ'_i × J^T
μ_2D = π(μ'_i) = [f_x × x/z + c_x, f_y × y/z + c_y]
```

**Step 3: Evaluate 2D Gaussian**
```
G_2D(p) = exp(-0.5 × (p - μ_2D)^T × Σ_2D^(-1) × (p - μ_2D))
```

**Step 4: Alpha compositing (front-to-back)**
```
C(p) = Σ_i c_i × α_i × G_2D_i(p) × Π_{j<i}(1 - α_j × G_2D_j(p))

Depth-sorted by μ'_i[z]
```

### 6.3 Dynamic Gaussian Representation

For time-varying fluids, extend Gaussians with temporal dynamics:

**Option A: Per-frame Gaussians**
```
G(t) = {G_i(t) for i in 1..N_t}

Each timestep has independent Gaussians (memory intensive)
```

**Option B: Trajectory-based Gaussians**
```
G_i(t) = G_i^base + ΔG_i(t)

Where ΔG_i(t) is predicted by a neural network or interpolated
```

**Option C: Velocity-integrated Gaussians (Recommended for fluids)**
```
μ_i(t+dt) = μ_i(t) + v_i(t) × dt
Σ_i(t+dt) = Σ_i(t) + dΣ_i(t)  # Optional deformation

Where v_i(t) is the velocity of Gaussian i at time t
```

### 6.4 Initialization

**Step 1: Sparse point cloud from SfM**
```
Run COLMAP or custom SfM on synchronized frames:
    points_3D = SfM(frames_aligned)
```

**Step 2: Initialize Gaussians at point locations**
```
For each 3D point P_k:
    μ_k = P_k
    Σ_k = ε × I  (small isotropic, e.g., ε = 0.001)
    α_k = 0.5
    c_k = color from observations
```

**Step 3: Densification (adaptive)**

During optimization, split/clone Gaussians based on gradients:
```
If ||∂L/∂μ_i|| > τ_grad and area(Σ_i) > τ_area:
    Split G_i into two smaller Gaussians

If ||∂L/∂μ_i|| > τ_grad and area(Σ_i) < τ_area:
    Clone G_i with small offset
```

### 6.5 Multi-View Photometric Loss

```
L_photo = Σ_v Σ_t Σ_p ||Render(G(t), K_v, [R|t]_v)(p) - F_v^aligned(t)(p)||₁

Where:
- v: View index
- t: Timestep
- p: Pixel location
- F_v^aligned(t): Ground truth frame from video v at time t
```

**Combined loss with SSIM:**
```
L_render = (1-λ_ssim) × L₁ + λ_ssim × (1 - SSIM)

Typical: λ_ssim = 0.2
```

### 6.6 Temporal Consistency Loss

Encourage smooth motion:
```
L_temporal = Σ_t Σ_i ||μ_i(t+1) - μ_i(t) - v_i(t) × dt||²

Where v_i(t) is the estimated velocity of Gaussian i
```

**Velocity smoothness:**
```
L_vel_smooth = Σ_t Σ_i ||v_i(t+1) - v_i(t)||²
```

---

## 7. Fluid Property Estimation

### 7.1 Overview

From the dynamic 3D Gaussian representation, estimate:
1. **Velocity field u(x, t)** - from Gaussian trajectories
2. **Density field ρ(x, t)** - from Gaussian distribution
3. **Pressure field p(x, t)** - from Navier-Stokes constraints
4. **Viscosity ν** - from velocity field analysis

### 7.2 Velocity Field Estimation

**Method 1: Direct from Gaussian Motion**

For each Gaussian i:
```
v_i(t) = (μ_i(t+dt) - μ_i(t)) / dt
```

To get velocity at arbitrary point x:
```
u(x, t) = Σ_i w_i(x) × v_i(t) / Σ_i w_i(x)

Where w_i(x) = exp(-0.5 × (x - μ_i)^T × Σ_i^(-1) × (x - μ_i))
```

**Method 2: Neural Velocity Field**

Train an MLP to predict velocity:
```
u(x, t) = MLP_vel(γ(x), γ(t))

Where γ() is positional encoding:
γ(p) = [sin(2^0 πp), cos(2^0 πp), ..., sin(2^L πp), cos(2^L πp)]
```

**Method 3: Velocity from Optical Flow (Hybrid)**

Project 3D velocity to 2D and compare with observed optical flow:
```
u_2D_predicted(p) = J × u(X(p), t)

Where X(p) is the 3D point corresponding to pixel p

Loss: L_flow = ||u_2D_predicted - optical_flow_observed||²
```

### 7.3 Density Field Estimation

**Method 1: From Gaussian Opacity**

```
ρ(x, t) = Σ_i α_i(t) × G_i(x, t)

Where G_i(x, t) = exp(-0.5 × (x - μ_i(t))^T × Σ_i(t)^(-1) × (x - μ_i(t)))
```

**Method 2: Normalized Density**

```
ρ(x, t) = ρ_base × (Σ_i α_i(t) × G_i(x, t)) / (Σ_i G_i(x, t))

Where ρ_base is a reference density (e.g., water = 1000 kg/m³)
```

**Method 3: Neural Density Field**

```
ρ(x, t) = softplus(MLP_density(γ(x), γ(t)))

softplus ensures ρ > 0
```

### 7.4 Pressure Field Estimation

Pressure is derived from the momentum equation. For incompressible flow:

**Pressure Poisson Equation:**
```
∇²p = -ρ × ∇·(u·∇u) = -ρ × (∂u_i/∂x_j × ∂u_j/∂x_i)
```

**Method 1: Neural Pressure Field with Physics Loss**

```
p(x, t) = MLP_pressure(γ(x), γ(t))

Train with physics residual loss (see Section 8)
```

**Method 2: Iterative Pressure Projection**

Solve the Poisson equation numerically on a grid:
```
1. Compute divergence: div = ∇·u
2. Solve: ∇²p = div using Jacobi/Gauss-Seidel/FFT
3. Project: u_corrected = u - ∇p/ρ
```

### 7.5 Viscosity Estimation

**Dynamic Viscosity μ and Kinematic Viscosity ν:**
```
ν = μ / ρ
```

**Method 1: From Velocity Gradient Decay**

In viscous flow, velocity gradients decay over time:
```
∂u/∂t = ν × ∇²u  (diffusion equation)

Estimate ν by fitting observed velocity evolution to this equation
```

**Method 2: From Strain Rate Tensor**

The strain rate tensor:
```
S_ij = 0.5 × (∂u_i/∂x_j + ∂u_j/∂x_i)
```

For Newtonian fluids:
```
τ_ij = 2μ × S_ij

Where τ is the viscous stress tensor
```

**Method 3: Neural Viscosity Estimation**

```
ν(x, t) = softplus(MLP_viscosity(γ(x), γ(t)))

Or for constant viscosity:
ν = softplus(learnable_parameter)
```

**Method 4: From Energy Dissipation**

Viscosity causes kinetic energy dissipation:
```
dE/dt = -2μ × ∫ S_ij × S_ij dV

Estimate μ from observed energy decay rate
```

### 7.6 Field Representations

Store fields on either:

**Option A: Voxel Grid**
```
Field values at regular 3D grid points:
u[i,j,k], p[i,j,k], ρ[i,j,k]

Resolution: typically 64³ to 256³
```

**Option B: Neural Fields (Implicit)**
```
Field values from neural network queries:
u(x,t) = MLP(x, t)

Continuous, memory-efficient, but slower to query
```

**Option C: Hybrid (Gaussian + Grid)**
```
Coarse grid for pressure solve, Gaussians for fine details
```

---

## 8. Physics Constraints and Navier-Stokes Integration

### 8.1 Navier-Stokes Equations

The incompressible Navier-Stokes equations govern fluid motion:

**Momentum Equation:**
```
∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f

Where:
- u: Velocity field
- p: Pressure
- ρ: Density
- ν: Kinematic viscosity
- f: External forces (e.g., gravity)
```

**Continuity Equation (Incompressibility):**
```
∇·u = 0

Or: ∂u_x/∂x + ∂u_y/∂y + ∂u_z/∂z = 0
```

### 8.2 Expanded Component Form

**x-component:**
```
∂u/∂t + u×∂u/∂x + v×∂u/∂y + w×∂u/∂z = 
    -(1/ρ)×∂p/∂x + ν×(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²) + f_x
```

**y-component:**
```
∂v/∂t + u×∂v/∂x + v×∂v/∂y + w×∂v/∂z = 
    -(1/ρ)×∂p/∂y + ν×(∂²v/∂x² + ∂²v/∂y² + ∂²v/∂z²) + f_y
```

**z-component:**
```
∂w/∂t + u×∂w/∂x + v×∂w/∂y + w×∂w/∂z = 
    -(1/ρ)×∂p/∂z + ν×(∂²w/∂x² + ∂²w/∂y² + ∂²w/∂z²) + f_z
```

### 8.3 PINN-Style Physics Loss

Following Physics-Informed Neural Networks (Raissi et al.):

**Momentum Residual:**
```
R_momentum = ∂u/∂t + (u·∇)u + ∇p/ρ - ν∇²u - f

L_momentum = ||R_momentum||²
```

**Continuity Residual:**
```
R_continuity = ∇·u

L_continuity = ||R_continuity||²
```

**Total Physics Loss:**
```
L_physics = λ_mom × L_momentum + λ_cont × L_continuity

Typical: λ_mom = 1.0, λ_cont = 10.0 (weight continuity higher)
```

### 8.4 Automatic Differentiation for Gradients

Using neural network autodiff (PyTorch/JAX):

```python
# Pseudo-code for computing physics residual
def compute_physics_residual(x, t, model):
    # Enable gradient tracking
    x.requires_grad = True
    t.requires_grad = True
    
    # Get field values
    u, v, w = model.velocity(x, t)  # velocity components
    p = model.pressure(x, t)
    rho = model.density(x, t)
    
    # Compute spatial gradients
    du_dx = grad(u, x, create_graph=True)[0][:, 0]
    du_dy = grad(u, x, create_graph=True)[0][:, 1]
    du_dz = grad(u, x, create_graph=True)[0][:, 2]
    # ... similarly for v, w, p
    
    # Compute temporal gradient
    du_dt = grad(u, t, create_graph=True)[0]
    
    # Compute Laplacian
    d2u_dx2 = grad(du_dx, x, create_graph=True)[0][:, 0]
    # ... etc
    laplacian_u = d2u_dx2 + d2u_dy2 + d2u_dz2
    
    # Momentum residual (x-component)
    R_x = du_dt + u*du_dx + v*du_dy + w*du_dz + dp_dx/rho - nu*laplacian_u - f_x
    
    # Continuity residual
    R_cont = du_dx + dv_dy + dw_dz
    
    return R_x, R_y, R_z, R_cont
```

### 8.5 Boundary Conditions

**No-slip (solid walls):**
```
u = 0 at wall surface
```

**Free surface:**
```
p = p_ambient at free surface
∂u/∂n = 0 (tangential components)
```

**Inflow/Outflow:**
```
u = u_specified at inlet
∂u/∂n = 0 at outlet (zero gradient)
```

**Implementation as loss:**
```
L_boundary = Σ_boundary_points ||u - u_BC||²
```

### 8.6 Time Integration Scheme

For forward simulation:

**Explicit Euler (simple, stability-limited):**
```
u^(n+1) = u^n + dt × [-(u^n·∇)u^n - ∇p^n/ρ + ν∇²u^n + f]
```

**Semi-implicit (Chorin's projection):**
```
Step 1 (Advection-Diffusion):
    u* = u^n + dt × [-(u^n·∇)u^n + ν∇²u^n + f]

Step 2 (Pressure solve):
    ∇²p^(n+1) = ρ/dt × ∇·u*

Step 3 (Velocity correction):
    u^(n+1) = u* - dt/ρ × ∇p^(n+1)
```

### 8.7 Differentiable Pressure Projection

For end-to-end training with physics constraints:

```
Implement pressure solve as differentiable layer:

1. Assemble Laplacian matrix L (sparse)
2. Compute RHS: b = ∇·u*
3. Solve: p = L^(-1) × b  (use conjugate gradient)
4. Backprop through the solve using implicit differentiation:
   ∂L_output/∂inputs via adjoint method
```

---

## 9. Loss Functions and Optimization

### 9.1 Complete Loss Function

```
L_total = λ_photo × L_photometric     # Multi-view rendering loss
        + λ_temp × L_temporal          # Temporal consistency
        + λ_flow × L_optical_flow      # Optical flow matching
        + λ_mom × L_momentum           # Navier-Stokes momentum
        + λ_cont × L_continuity        # Incompressibility
        + λ_bc × L_boundary            # Boundary conditions
        + λ_reg × L_regularization     # Gaussian regularization
```

### 9.2 Individual Loss Terms

**Photometric Loss:**
```
L_photometric = (1-λ) × L₁ + λ × (1 - SSIM)

L₁ = mean(|I_rendered - I_target|)
SSIM = structural_similarity(I_rendered, I_target)
```

**Temporal Consistency:**
```
L_temporal = Σ_i,t ||μ_i(t+1) - μ_i(t) - v_i(t)×dt||²
           + β × Σ_i,t ||v_i(t+1) - v_i(t)||²
```

**Optical Flow Loss:**
```
L_flow = Σ_v,t,p ||J×u(X(p),t) - flow_observed_v(p,t)||²
```

**Momentum Loss (PINN):**
```
L_momentum = Σ_samples ||∂u/∂t + (u·∇)u + ∇p/ρ - ν∇²u - f||²
```

**Continuity Loss:**
```
L_continuity = Σ_samples ||∇·u||²
```

**Boundary Condition Loss:**
```
L_boundary = Σ_bc_points ||u - u_bc||² + ||p - p_bc||²
```

**Regularization:**
```
L_reg = λ_scale × Σ_i ||log(s_i)||²      # Prevent degenerate scales
      + λ_opacity × Σ_i |α_i - 0.5|       # Opacity regularization
```

### 9.3 Loss Weights (Suggested Starting Values)

| Loss Term | Weight | Notes |
|-----------|--------|-------|
| λ_photo | 1.0 | Primary supervision |
| λ_temp | 0.1 | Temporal smoothness |
| λ_flow | 0.5 | Optical flow consistency |
| λ_mom | 0.01 | Physics momentum |
| λ_cont | 0.1 | Incompressibility (weight higher) |
| λ_bc | 1.0 | Boundary conditions |
| λ_reg | 0.001 | Regularization |

### 9.4 Optimization Strategy

**Stage 1: Geometric Reconstruction (Epochs 1-1000)**
```
Active losses: L_photo, L_reg
Learning rate: 1e-3
Goal: Establish 3D structure
```

**Stage 2: Temporal Alignment (Epochs 1000-2000)**
```
Active losses: L_photo, L_temp, L_flow, L_reg
Learning rate: 5e-4
Goal: Consistent motion across time
```

**Stage 3: Physics Integration (Epochs 2000-5000)**
```
Active losses: All
Learning rate: 1e-4
Goal: Physics-consistent fields
```

**Stage 4: Fine-tuning (Epochs 5000-10000)**
```
Active losses: All with adjusted weights
Learning rate: 1e-5
Goal: Refine details
```

### 9.5 Optimizer Configuration

```
Optimizer: Adam
Learning rates:
    - Gaussian positions: 1e-4
    - Gaussian scales: 5e-3
    - Gaussian rotations: 1e-3
    - Gaussian opacities: 5e-2
    - Gaussian colors: 2.5e-3
    - Neural network parameters: 1e-4
    - Viscosity parameter: 1e-5

Scheduler: ExponentialLR(gamma=0.995) every 100 iterations
```

---

## 10. Complete Pipeline Architecture

### 10.1 Pipeline Stages

```
┌────────────────────────────────────────────────────────────────┐
│ STAGE 0: INITIALIZATION                                         │
├────────────────────────────────────────────────────────────────┤
│ Input: 3 video file paths                                       │
│ Output: Loaded video objects, metadata                          │
│ Operations:                                                     │
│   1. Open video files                                           │
│   2. Extract metadata (fps, resolution, duration)               │
│   3. Validate inputs (min duration, codec support)              │
│   4. Report any issues to user                                  │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ STAGE 1: FRAME EXTRACTION                                       │
├────────────────────────────────────────────────────────────────┤
│ Input: Video objects                                            │
│ Output: Frame sequences per video                               │
│ Operations:                                                     │
│   1. Extract all frames at native resolution                    │
│   2. Convert to RGB float32 [0, 1]                              │
│   3. Store with native timestamps                               │
│   4. Compute optical flow for all consecutive pairs             │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ STAGE 2: TEMPORAL SYNCHRONIZATION                               │
├────────────────────────────────────────────────────────────────┤
│ Input: Frame sequences with optical flow                        │
│ Output: Time offsets, common timeline parameters                │
│ Operations:                                                     │
│   1. Compute motion signatures per video                        │
│   2. Cross-correlate to find pairwise offsets                   │
│   3. Verify consistency, resolve conflicts                      │
│   4. Establish common timeline (start, end, fps)                │
│   5. Generate aligned frame indices/interpolation params        │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ STAGE 3: CAMERA CALIBRATION                                     │
├────────────────────────────────────────────────────────────────┤
│ Input: Aligned frames                                           │
│ Output: Intrinsic K and Extrinsic [R|t] per view                │
│ Operations:                                                     │
│   1. Detect features (SuperPoint/SIFT)                          │
│   2. Match features across views (SuperGlue/BFMatcher)          │
│   3. Estimate initial intrinsics (from EXIF or motion)          │
│   4. Estimate relative poses via essential matrix               │
│   5. Run bundle adjustment                                      │
│   6. Handle zoom differences in focal lengths                   │
│   7. Output camera matrices                                     │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ STAGE 4: INITIAL POINT CLOUD                                    │
├────────────────────────────────────────────────────────────────┤
│ Input: Calibrated cameras, matched features                     │
│ Output: Sparse 3D point cloud                                   │
│ Operations:                                                     │
│   1. Triangulate matched points across views                    │
│   2. Filter outliers (reprojection error > threshold)           │
│   3. Optionally densify with MVS                                │
│   4. Estimate scene bounding box                                │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ STAGE 5: GAUSSIAN INITIALIZATION                                │
├────────────────────────────────────────────────────────────────┤
│ Input: Point cloud for first frame                              │
│ Output: Initial Gaussian set G(t=0)                             │
│ Operations:                                                     │
│   1. Place Gaussian at each 3D point                            │
│   2. Initialize scales from local point density                 │
│   3. Initialize colors from multi-view observations             │
│   4. Set opacities to 0.5                                       │
│   5. Initialize velocity to zero                                │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ STAGE 6: DYNAMIC RECONSTRUCTION (MAIN OPTIMIZATION)             │
├────────────────────────────────────────────────────────────────┤
│ Input: Initial Gaussians, all aligned frames, cameras           │
│ Output: Optimized G(t) for all timesteps                        │
│ Operations (per iteration):                                     │
│   1. Sample random timestep t and view v                        │
│   2. Render G(t) from view v                                    │
│   3. Compute photometric loss                                   │
│   4. Compute temporal consistency loss                          │
│   5. Backpropagate and update Gaussian parameters               │
│   6. Adaptive densification/pruning                             │
│   7. Log metrics and save checkpoints                           │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ STAGE 7: VELOCITY FIELD ESTIMATION                              │
├────────────────────────────────────────────────────────────────┤
│ Input: Optimized dynamic Gaussians G(t)                         │
│ Output: Velocity field u(x, t)                                  │
│ Operations:                                                     │
│   1. Compute Gaussian velocities: v_i = (μ_i(t+1) - μ_i(t))/dt  │
│   2. Train neural velocity field (optional)                     │
│   3. Validate against optical flow                              │
│   4. Smooth/regularize velocity field                           │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ STAGE 8: DENSITY FIELD ESTIMATION                               │
├────────────────────────────────────────────────────────────────┤
│ Input: Optimized Gaussians G(t)                                 │
│ Output: Density field ρ(x, t)                                   │
│ Operations:                                                     │
│   1. Compute density from Gaussian opacities                    │
│   2. Normalize to physical units (if scale known)               │
│   3. Apply smoothing for continuity                             │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ STAGE 9: PRESSURE AND VISCOSITY ESTIMATION                      │
├────────────────────────────────────────────────────────────────┤
│ Input: u(x,t), ρ(x,t)                                           │
│ Output: p(x,t), ν                                               │
│ Operations:                                                     │
│   1. Initialize pressure field                                  │
│   2. Initialize viscosity parameter                             │
│   3. Optimize with Navier-Stokes residual loss                  │
│   4. Iterate until convergence                                  │
│   5. Validate physical plausibility                             │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ STAGE 10: OUTPUT AND VALIDATION                                 │
├────────────────────────────────────────────────────────────────┤
│ Input: All estimated fields                                     │
│ Output: Final results, visualizations                           │
│ Operations:                                                     │
│   1. Save Gaussian representation                               │
│   2. Export velocity/pressure/density as VTK or numpy           │
│   3. Generate validation renders                                │
│   4. Compute error metrics                                      │
│   5. Create visualization videos                                │
└────────────────────────────────────────────────────────────────┘
```

### 10.2 Detailed Algorithm Pseudocode

```
Algorithm: FluidReconstruction

Input: video_paths[3]  # Paths to 3 video files
Output: G(t), u(x,t), p(x,t), ρ(x,t), ν

# Stage 0-1: Load and extract
videos = [load_video(path) for path in video_paths]
metadata = [get_metadata(v) for v in videos]
frames = [extract_frames(v) for v in videos]
flows = [compute_optical_flow(f) for f in frames]

# Stage 2: Temporal synchronization
motion_sigs = [compute_motion_signature(flow) for flow in flows]
offsets = compute_pairwise_offsets(motion_sigs)
offsets = verify_and_resolve(offsets)
common_timeline = establish_common_timeline(metadata, offsets)
aligned_frames = interpolate_to_common_timeline(frames, offsets, common_timeline)

# Stage 3: Camera calibration
features = [detect_features(frame_seq) for frame_seq in aligned_frames]
matches = match_features_across_views(features)
intrinsics = estimate_intrinsics(metadata, matches)
extrinsics = estimate_extrinsics(intrinsics, matches)
intrinsics, extrinsics = bundle_adjustment(intrinsics, extrinsics, matches)

# Stage 4-5: Point cloud and Gaussian initialization
point_cloud = triangulate_points(matches, intrinsics, extrinsics)
point_cloud = filter_outliers(point_cloud)
gaussians = initialize_gaussians(point_cloud, aligned_frames[0])

# Stage 6: Dynamic reconstruction
for epoch in range(n_epochs):
    t = sample_timestep()
    v = sample_view()
    
    rendered = render_gaussians(gaussians[t], intrinsics[v], extrinsics[v])
    
    loss = compute_photometric_loss(rendered, aligned_frames[v][t])
    loss += compute_temporal_loss(gaussians, t)
    loss += compute_flow_loss(gaussians, flows, t, v)
    
    backpropagate(loss)
    update_gaussians(gaussians, optimizer)
    
    if epoch % densify_interval == 0:
        densify_and_prune(gaussians)

# Stage 7-9: Physics estimation
velocity_field = estimate_velocity(gaussians)
density_field = estimate_density(gaussians)
pressure_field, viscosity = estimate_physics_fields(
    velocity_field, density_field,
    physics_loss_fn=navier_stokes_residual
)

# Stage 10: Output
save_results(gaussians, velocity_field, pressure_field, density_field, viscosity)
generate_visualizations(...)

return gaussians, velocity_field, pressure_field, density_field, viscosity
```

---

## 11. Data Structures

### 11.1 Video Data

```python
@dataclass
class VideoMetadata:
    path: str
    fps: float                    # Frames per second
    width: int                    # Frame width in pixels
    height: int                   # Frame height in pixels
    duration: float               # Duration in seconds
    n_frames: int                 # Total frame count
    codec: str                    # Video codec
    
@dataclass
class VideoData:
    metadata: VideoMetadata
    frames: np.ndarray           # Shape: (n_frames, H, W, 3), float32 [0,1]
    timestamps: np.ndarray       # Shape: (n_frames,), float64, in seconds
    optical_flow: np.ndarray     # Shape: (n_frames-1, H, W, 2), float32
```

### 11.2 Synchronization Data

```python
@dataclass
class SyncParameters:
    offsets: np.ndarray          # Shape: (3,), time offset per video in seconds
    common_fps: float            # Unified frame rate
    common_start: float          # Start time of common timeline
    common_end: float            # End time of common timeline
    n_common_frames: int         # Number of frames in common timeline
    
@dataclass
class AlignedFrames:
    frames: np.ndarray           # Shape: (n_views, n_frames, H, W, 3)
    timestamps: np.ndarray       # Shape: (n_frames,), common timeline
```

### 11.3 Camera Data

```python
@dataclass
class CameraIntrinsics:
    fx: float                    # Focal length x
    fy: float                    # Focal length y
    cx: float                    # Principal point x
    cy: float                    # Principal point y
    k1: float = 0.0              # Radial distortion
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0              # Tangential distortion
    p2: float = 0.0
    
    def to_matrix(self) -> np.ndarray:
        """Returns 3x3 intrinsic matrix K"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

@dataclass
class CameraExtrinsics:
    R: np.ndarray                # Rotation matrix, shape (3, 3)
    t: np.ndarray                # Translation vector, shape (3,)
    
    def to_matrix(self) -> np.ndarray:
        """Returns 3x4 extrinsic matrix [R|t]"""
        return np.hstack([self.R, self.t.reshape(3, 1)])
    
    def to_projection(self, K: np.ndarray) -> np.ndarray:
        """Returns 3x4 projection matrix P = K[R|t]"""
        return K @ self.to_matrix()

@dataclass
class Camera:
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    width: int
    height: int
```

### 11.4 Gaussian Data

```python
@dataclass
class Gaussian3D:
    position: np.ndarray         # μ, shape (3,)
    scale: np.ndarray            # s, shape (3,), log-scale
    rotation: np.ndarray         # q, shape (4,), quaternion (w, x, y, z)
    opacity: float               # α, logit-space
    color_sh: np.ndarray         # Spherical harmonics, shape (k, 3)
    
    # For dynamic fluids
    velocity: np.ndarray         # v, shape (3,), m/s
    
@dataclass
class GaussianCloud:
    """Collection of Gaussians at a single timestep"""
    n_gaussians: int
    positions: np.ndarray        # Shape: (N, 3)
    scales: np.ndarray           # Shape: (N, 3)
    rotations: np.ndarray        # Shape: (N, 4)
    opacities: np.ndarray        # Shape: (N,)
    colors_sh: np.ndarray        # Shape: (N, n_sh, 3)
    velocities: np.ndarray       # Shape: (N, 3)
    
@dataclass
class DynamicGaussianCloud:
    """Time-varying Gaussian representation"""
    n_timesteps: int
    timesteps: List[GaussianCloud]  # One per timestep
    # Or: shared Gaussians with time-varying parameters
```

### 11.5 Field Data

```python
@dataclass
class VectorField:
    """3D vector field on regular grid"""
    data: np.ndarray             # Shape: (T, X, Y, Z, 3) or (X, Y, Z, 3)
    origin: np.ndarray           # Shape: (3,), world coordinates of grid origin
    spacing: np.ndarray          # Shape: (3,), grid cell size
    timestamps: np.ndarray       # Shape: (T,), if time-varying
    
    def sample(self, x: np.ndarray) -> np.ndarray:
        """Trilinear interpolation at point(s) x"""
        # Implementation: convert x to grid coordinates, interpolate
        pass
        
@dataclass 
class ScalarField:
    """3D scalar field on regular grid"""
    data: np.ndarray             # Shape: (T, X, Y, Z) or (X, Y, Z)
    origin: np.ndarray
    spacing: np.ndarray
    timestamps: np.ndarray
    
@dataclass
class FluidFields:
    """Complete fluid state"""
    velocity: VectorField        # u(x, t)
    pressure: ScalarField        # p(x, t)
    density: ScalarField         # ρ(x, t)
    viscosity: float             # ν (scalar) or ScalarField (spatially varying)
```

### 11.6 Neural Network Data

```python
@dataclass
class MLPConfig:
    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    activation: str              # 'relu', 'silu', 'softplus'
    positional_encoding_L: int   # Number of frequency bands
    
@dataclass
class NeuralFields:
    velocity_mlp: nn.Module
    pressure_mlp: nn.Module
    density_mlp: nn.Module
    viscosity_param: nn.Parameter  # Learnable scalar or field
```

---

## 12. User Interface Specification

### 12.1 Terminal-Based File Selection

The system should provide an easy-to-use terminal interface for selecting video files.

**Requirements:**
- Display a file browser dialog (GUI) from terminal
- Allow selection of exactly 3 video files
- Support common video formats: .mp4, .avi, .mov, .mkv
- Validate files exist and are readable
- Show selected files for confirmation

### 12.2 UI Implementation Options

**Option A: tkinter File Dialog (Cross-platform)**
```
When user runs the program:
1. Open file dialog with title "Select Video 1 of 3"
2. Filter for video files (*.mp4, *.avi, *.mov, *.mkv)
3. Repeat for videos 2 and 3
4. Display selected files and ask for confirmation
5. If confirmed, proceed; otherwise, allow re-selection
```

**Option B: Terminal Menu with Paths**
```
================================================
      Fluid Reconstruction - Video Input
================================================

[1] Select videos using file browser
[2] Enter video paths manually
[3] Load from config file
[q] Quit

Enter choice: _
```

**Option C: Drag-and-Drop (if GUI available)**
```
Create a window with drop zone:
"Drop 3 video files here or click to browse"
```

### 12.3 Input Validation

```
For each selected video:
1. Check file exists
2. Check file is readable
3. Attempt to open with video decoder
4. Extract and display metadata
5. Warn if:
   - Duration < 1 second
   - Frame rate < 10 fps
   - Resolution too low (< 480p)
   - Codec not supported
```

### 12.4 Configuration Options

After file selection, prompt for optional configuration:

```
================================================
        Configuration (press Enter for defaults)
================================================

Output directory [./output]: _
Target frame rate (0=auto) [0]: _
Physics estimation (y/n) [y]: _
Visualization output (y/n) [y]: _
CUDA device ID [0]: _
Number of training iterations [5000]: _
```

### 12.5 Progress Display

During processing:

```
================================================
            Processing Videos
================================================

Stage 1/10: Frame Extraction
  Video 1: [████████████████████] 100% (1200/1200 frames)
  Video 2: [████████████████████] 100% (900/900 frames)
  Video 3: [████████████████████] 100% (1080/1080 frames)

Stage 2/10: Temporal Synchronization
  Computing motion signatures... done
  Finding offsets:
    Video 1 → Video 2: +1.23s
    Video 1 → Video 3: +2.15s
  Establishing common timeline... done (180 frames @ 30fps)

Stage 3/10: Camera Calibration
  [██████████░░░░░░░░░░] 50% - Bundle adjustment iteration 15/30
  
Current metrics:
  - Reprojection error: 0.82 px
  - Photometric loss: 0.0234
  
ETA: 2h 34m
```

---

## 13. Implementation Guidelines

### 13.1 Recommended Libraries

| Purpose | Library | Notes |
|---------|---------|-------|
| Video I/O | OpenCV, decord, PyAV | decord is fastest for GPU |
| Optical Flow | RAFT, torchvision | RAFT for best accuracy |
| Feature Detection | SuperPoint, OpenCV | SuperPoint preferred |
| Feature Matching | SuperGlue, OpenCV | SuperGlue for fluid scenes |
| Camera Calibration | COLMAP, OpenCV | COLMAP for full SfM |
| Gaussian Splatting | gsplat, diff-gaussian-rasterization | gsplat is more maintained |
| Neural Networks | PyTorch | With CUDA support |
| Autodiff | PyTorch autograd | For physics losses |
| Visualization | Open3D, PyVista, Matplotlib | Open3D for 3D |
| File Dialog | tkinter, PyQt | tkinter is built-in |

### 13.2 Hardware Requirements

**Minimum:**
- GPU: NVIDIA RTX 3060 (12GB VRAM)
- RAM: 32GB
- Storage: 500GB SSD

**Recommended:**
- GPU: NVIDIA RTX 4090 or A100 (24GB+ VRAM)
- RAM: 64GB
- Storage: 2TB NVMe SSD

### 13.3 Memory Management

```
For 1080p videos, 180 frames, 3 views:
- Raw frames: 180 × 3 × 1920 × 1080 × 3 × 4 bytes ≈ 13.4 GB
- Optical flow: 179 × 3 × 1920 × 1080 × 2 × 4 bytes ≈ 8.9 GB
- Gaussians (100K): 100K × (3+3+4+1+48) × 4 bytes ≈ 23.6 MB per frame

Strategy:
1. Process frames in batches
2. Use memory-mapped files for large arrays
3. Keep only current batch in GPU memory
4. Use mixed precision (FP16) where possible
```

### 13.4 Checkpointing

```
Save checkpoints every N iterations:
checkpoint = {
    'epoch': current_epoch,
    'gaussians': gaussians.state_dict(),
    'optimizer': optimizer.state_dict(),
    'loss_history': losses,
    'config': config,
}
torch.save(checkpoint, f'checkpoint_{epoch}.pt')
```

### 13.5 Error Handling

```
Key failure points and recovery:
1. Video loading: Try alternative decoders
2. Synchronization: Fall back to manual offset input
3. Calibration: Warn if reprojection error > 2px
4. Training divergence: Reduce learning rate, restart from checkpoint
5. GPU OOM: Reduce batch size, gaussian count, or resolution
```

### 13.6 Testing Strategy

```
Unit tests:
- Video loading with various codecs/resolutions
- Synchronization with known offsets
- Gaussian rendering matches reference
- Physics residual computation

Integration tests:
- Full pipeline on synthetic data
- Known fluid simulation rendered from multiple views
- Verify recovered properties match ground truth
```

---

## Appendix: Mathematical Derivations

### A.1 Cross-Correlation Derivation

The normalized cross-correlation between signals M₁ and M₂ at lag τ:

```
R₁₂(τ) = E[(M₁(t) - μ₁)(M₂(t+τ) - μ₂)] / (σ₁ × σ₂)

Expanding:
R₁₂(τ) = (1/N) × Σ_t [(M₁(t) - μ₁) × (M₂(t+τ) - μ₂)] / (σ₁ × σ₂)

Where:
- μᵢ = (1/N) × Σ_t Mᵢ(t)
- σᵢ = sqrt((1/N) × Σ_t (Mᵢ(t) - μᵢ)²)
```

### A.2 Essential Matrix Decomposition

Given essential matrix E = [t]_× R:

```
SVD: E = U × Σ × V^T

Where Σ = diag(σ, σ, 0) for valid E

Possible rotations:
R₁ = U × W × V^T
R₂ = U × W^T × V^T

Where W = | 0  -1  0 |
          | 1   0  0 |
          | 0   0  1 |

Possible translations:
t = ±U[:, 2]  (third column of U)

Select [R, t] where triangulated points have positive depth.
```

### A.3 3D Gaussian Covariance Projection

Given 3D Gaussian with mean μ and covariance Σ, project to 2D:

```
World to camera: μ' = R×μ + t, Σ' = R×Σ×R^T

Camera to image plane (perspective):
Let μ' = [x, y, z]^T

Jacobian of projection π(x,y,z) = [f_x×x/z, f_y×y/z]:
J = | f_x/z    0     -f_x×x/z² |
    |   0    f_y/z   -f_y×y/z² |

2D covariance: Σ_2D = J × Σ' × J^T

This is a 2×2 matrix defining the projected ellipse.
```

### A.4 Navier-Stokes Discretization

For finite difference computation of physics residuals:

**Spatial derivatives (central difference):**
```
∂u/∂x ≈ (u_{i+1,j,k} - u_{i-1,j,k}) / (2Δx)

∂²u/∂x² ≈ (u_{i+1,j,k} - 2u_{i,j,k} + u_{i-1,j,k}) / (Δx)²
```

**Temporal derivative:**
```
∂u/∂t ≈ (u^{n+1} - u^n) / Δt
```

**Convective term (upwind):**
```
(u·∇)u ≈ u × (u_{i,j,k} - u_{i-1,j,k})/Δx  if u > 0
       ≈ u × (u_{i+1,j,k} - u_{i,j,k})/Δx  if u < 0
```

### A.5 Pressure Poisson Equation

For incompressible flow, derive pressure equation:

```
Starting from momentum: ∂u/∂t = -∇p/ρ + ... (other terms)

Take divergence: ∂(∇·u)/∂t = -∇²p/ρ + ...

For incompressibility: ∇·u = 0

Therefore: ∇²p = ρ × (∂(∇·u*)/∂t)

Where u* is intermediate velocity before pressure correction.

In practice: ∇²p = (ρ/Δt) × ∇·u*
```

### A.6 Quaternion to Rotation Matrix

```
Given quaternion q = [w, x, y, z] (unit quaternion, ||q|| = 1):

R = | 1-2(y²+z²)   2(xy-wz)    2(xz+wy)  |
    | 2(xy+wz)     1-2(x²+z²)  2(yz-wx)  |
    | 2(xz-wy)     2(yz+wx)    1-2(x²+y²)|

Inverse (rotation matrix to quaternion):
w = 0.5 × sqrt(1 + R₀₀ + R₁₁ + R₂₂)
x = (R₂₁ - R₁₂) / (4w)
y = (R₀₂ - R₂₀) / (4w)
z = (R₁₀ - R₀₁) / (4w)
```

### A.7 Positional Encoding

For neural field inputs:

```
γ(p) = [sin(2⁰πp), cos(2⁰πp), sin(2¹πp), cos(2¹πp), ..., 
        sin(2^{L-1}πp), cos(2^{L-1}πp)]

For 3D position x = [x, y, z]:
γ(x) = [γ(x), γ(y), γ(z)]

Output dimension: 3 × 2L (for position) + 2L (for time, if included)
Typical L = 10, giving 60-dimensional encoding for position
```

---

## Document Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 2026 | Initial specification |

---

*End of Specification Document*
