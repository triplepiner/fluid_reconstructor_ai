# Physics-Integrated Neural Fluid Reconstruction

Reconstruct 3D fluid dynamics from multi-view video using **3D Gaussian Splatting** combined with **Navier-Stokes physics constraints**. This system takes 1-3 videos of fluid motion and outputs a complete 3D reconstruction with estimated velocity, pressure, density, and viscosity fields.

## What This Project Does

Given video footage of fluids (water, smoke, etc.), this pipeline:

1. **Synchronizes** videos recorded at different times (handles 0.3-3s offsets)
2. **Calibrates cameras** automatically from feature matching
3. **Reconstructs** the scene using dynamic 3D Gaussian Splatting
4. **Estimates physics** - velocity field **u(x,t)**, pressure **p(x,t)**, density **ρ(x,t)**, and viscosity **ν**

The output enables novel view synthesis, physics-based simulation, and fluid analysis.

## Key Features

- **Flexible Input**: 1-3 videos with different FPS, resolutions, and zoom levels
- **Automatic Synchronization**: Aligns videos using motion signature cross-correlation
- **Camera Shake Compensation**: Frequency-based separation of shake from intentional pans
- **Physics-Informed**: PINN-style Navier-Stokes constraints ensure physical plausibility
- **Fast Rendering**: Gaussian splatting enables real-time novel-view synthesis
- **Cross-Platform**: NVIDIA GPU (CUDA), Apple Silicon (MPS), and CPU support

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fluid_ai.git
cd fluid_ai

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- PyTorch 2.0+
- NVIDIA GPU with CUDA (recommended) / Apple Silicon (MPS) / CPU

## Quick Start

```bash
# Single video (uses depth estimation)
python scripts/run_pipeline.py my_fluid_video.mp4

# Multi-view (best quality)
python scripts/run_pipeline.py front.mp4 left.mp4 right.mp4

# Interactive mode (GUI file selector)
python scripts/run_pipeline.py

# With options
python scripts/run_pipeline.py video.mp4 --output ./results --epochs 10000 --device cuda
```

### Command Line Options

```
Options:
  videos              Input video files (1-3 videos)
  -o, --output        Output directory (default: outputs)
  --device            Device: auto, cuda, mps, cpu (default: auto)
  --max-resolution    Maximum frame resolution (default: 1080)
  --epochs            Number of training epochs (default: 5000)
  --no-physics        Skip physics estimation stage
  --resume            Resume from checkpoint directory
  --test              Run quick test with reduced settings
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐                  │
│  │  Video   │──▶│  Stabi-  │──▶│ Optical  │──▶│  Fluid   │                  │
│  │  Loader  │   │  lizer   │   │   Flow   │   │Segmenter │                  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SYNCHRONIZATION                                       │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐         │
│  │ Motion Signature │──▶│ Cross-Correlation│──▶│ Frame Interp.    │         │
│  │    Extraction    │   │  (find offset)   │   │ (align timeline) │         │
│  └──────────────────┘   └──────────────────┘   └──────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CAMERA CALIBRATION                                    │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                     │
│  │   Feature    │──▶│   Feature    │──▶│   Bundle     │                     │
│  │  Extraction  │   │   Matching   │   │  Adjustment  │                     │
│  └──────────────┘   └──────────────┘   └──────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     3D GAUSSIAN SPLATTING                                    │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                     │
│  │Triangulation │──▶│  Gaussian    │──▶│  Dynamic     │                     │
│  │ (3D points)  │   │    Init      │   │ Optimization │                     │
│  └──────────────┘   └──────────────┘   └──────────────┘                     │
│                                              │                               │
│                     Loss = L_photo + L_temporal + L_flow + L_physics         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHYSICS ESTIMATION (PINN)                                │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │              Neural Fields: MLP with Positional Encoding          │       │
│  │                                                                   │       │
│  │  Input: (x, y, z, t) ──▶ [sin/cos encoding] ──▶ MLP ──▶ (u, p, ρ)│       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│  Navier-Stokes Loss:                                                        │
│  • Momentum: ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u                                  │
│  • Continuity: ∇·u = 0 (incompressibility)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### Preprocessing (`src/preprocessing/`)

| Component | Purpose |
|-----------|---------|
| `VideoLoader` | Loads videos (mp4, avi, mov, mkv), handles different resolutions/FPS |
| `OpticalFlowEstimator` | Dense optical flow using RAFT between consecutive frames |
| `FeatureExtractor` | Keypoint detection (SuperPoint/SIFT) for camera calibration |
| `FluidSegmenter` | Isolates fluid regions from background |
| `DepthEstimator` | Monocular depth for single-view mode |

### Synchronization (`src/synchronization/`)

| Component | Purpose |
|-----------|---------|
| `TemporalAligner` | Aligns videos using cross-correlation of motion signatures |
| `MotionSignatureAnalyzer` | Computes motion energy, direction, spatial variance per frame |
| `FrameInterpolator` | Sub-frame interpolation to common timeline |

### Camera Calibration (`src/calibration/`)

| Component | Purpose |
|-----------|---------|
| `CameraEstimator` | Estimates intrinsics (focal length, principal point, distortion) |
| `FeatureMatcher` | Matches features across views (SIFT or SuperGlue) |
| `BundleAdjuster` | Joint optimization minimizing reprojection error |
| `Triangulation` | Reconstructs 3D points from 2D matches |

### 3D Gaussian Splatting (`src/gaussian_splatting/`)

| Component | Purpose |
|-----------|---------|
| `GaussianCloud` | Scene as 3D Gaussians: position μ, covariance Σ, opacity α, color (SH) |
| `DynamicGaussians` | Time-varying Gaussians with velocity-based motion |
| `Rasterizer` | Differentiable splatting renderer |
| `GaussianLoss` | Multi-view photometric loss (L1 + SSIM) |

### Physics (`src/physics/`)

| Component | Purpose |
|-----------|---------|
| `NavierStokesLoss` | PINN losses enforcing momentum and continuity equations |
| `NeuralFields` | MLPs with positional encoding for continuous field representation |
| `ViscosityEstimator` | Estimates kinematic viscosity from velocity diffusion |
| `FieldsData` | Storage for velocity, pressure, density fields |

## Key Algorithms

### 1. Motion Signature Cross-Correlation (Temporal Sync)

```python
# For each video, compute motion signature per frame:
signature = [flow_magnitude, flow_direction, spatial_variance]

# Cross-correlate signatures between video pairs
correlation = scipy.signal.correlate(sig_A, sig_B)
offset = argmax(correlation)  # with parabolic refinement
```

### 2. Slow Pan Stabilization

Separates intentional camera motion from unwanted shake using frequency analysis:
- High-frequency (>2 Hz) = camera shake → remove
- Low-frequency (<2 Hz) = intentional pan → preserve

### 3. Dynamic 3D Gaussian Splatting

Each Gaussian has parameters: `{μ, Σ, α, color_SH, velocity}`

Three temporal modes:
- **Per-frame**: Independent Gaussians per timestep
- **Trajectory**: Base Gaussians + learned time offsets
- **Velocity**: Base Gaussians with predicted velocities (best for fluids)

### 4. Physics-Informed Neural Networks (PINNs)

Neural fields predict `(u, p, ρ)` from `(x, y, z, t)` with positional encoding:
```
γ(p) = [sin(2⁰πp), cos(2⁰πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]
```

Loss includes Navier-Stokes residuals computed via automatic differentiation.

## Output Files

```
outputs/
├── checkpoints/           # Training checkpoints (resume capability)
│   └── epoch_XXXX.pt
├── gaussians/             # Reconstructed 3D Gaussians
│   ├── gaussians.pt       # PyTorch model
│   └── gaussians.ply      # Point cloud export
├── fields/                # Physics fields
│   ├── velocity.pt        # Velocity field u(x,t)
│   ├── pressure.pt        # Pressure field p(x,t)
│   ├── density.pt         # Density field ρ(x,t)
│   └── viscosity.json     # Estimated viscosity ν
├── videos/                # Rendered outputs
│   ├── reconstruction.mp4
│   └── novel_views.mp4
└── calibration/
    └── cameras.json       # Camera parameters
```

## Physics Outputs

| Field | Description | Units |
|-------|-------------|-------|
| **Velocity** u(x,t) | 3D vector field of fluid motion | m/s |
| **Pressure** p(x,t) | Scalar pressure distribution | Pa |
| **Density** ρ(x,t) | Scalar density from Gaussian opacity | relative |
| **Viscosity** ν | Kinematic viscosity (identifies fluid type) | m²/s |

## Python API

```python
from pathlib import Path
from src.config import PipelineConfig
from src.pipeline import FluidReconstructionPipeline

# Create pipeline
config = PipelineConfig(
    device="cuda",
    output_dir=Path("my_outputs"),
    max_resolution=1080,
    optimization={'n_epochs': 10000}
)
pipeline = FluidReconstructionPipeline(config)

# Run on videos
videos = [Path("view1.mp4"), Path("view2.mp4"), Path("view3.mp4")]
result = pipeline.run(videos)

# Access outputs
if result.success:
    gaussians = result.outputs['gaussians']
    velocity_field = result.outputs['velocity_field']
    viscosity = result.outputs['viscosity_estimate']
```

## Recording Tips

For best results when capturing fluid:

1. **Use 3 cameras** at 60-120° angles apart
2. **Stable mounting** (tripods recommended)
3. **Good lighting** consistent across views
4. **Start within 3 seconds** of each other
5. **60fps recommended** (different rates OK)

```
        Camera 1 (front)
             |
             v
    [===================]
    [      FLUID        ]
    [===================]
   /                     \
  v                       v
Camera 2              Camera 3
(left 60°)           (right 60°)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | `--max-resolution 720` or reduce `initial_n_gaussians` |
| Poor sync | Ensure videos show same event, check offset is <3s |
| Calibration fails | Need texture/features, ensure overlapping views |
| Blurry output | Increase `--epochs 10000`, check input quality |

## Project Structure

```
fluid_ai/
├── src/
│   ├── config.py                 # Configuration dataclasses
│   ├── preprocessing/            # Video loading, optical flow, features
│   ├── synchronization/          # Temporal alignment
│   ├── calibration/              # Camera estimation, triangulation
│   ├── gaussian_splatting/       # 3DGS implementation
│   ├── physics/                  # Navier-Stokes, neural fields
│   ├── pipeline/                 # Stage orchestration
│   └── ui/                       # File selection, progress
├── scripts/
│   ├── run_pipeline.py           # Main entry point
│   ├── visualize_output.py       # Render results
│   └── visualize_dynamic.py      # Animate Gaussians
├── goals/                        # Project specification docs
├── requirements.txt
└── README.md
```

## Technical Highlights

- **100K-500K Gaussians** with efficient splatting
- **Automatic differentiation** for physics constraints
- **Mixed precision training** for memory efficiency
- **Checkpoint/resume** capability for long training runs
- **Modular design** - each stage can run independently

## Citation

```bibtex
@misc{fluid_reconstruction_2024,
  title={Physics-Integrated Neural Fluid Reconstruction},
  author={MBZUAI Research Team},
  year={2024},
  publisher={GitHub},
}
```

## License

MIT License - see LICENSE file for details.
