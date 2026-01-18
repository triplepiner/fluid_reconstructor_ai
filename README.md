# Physics-Integrated Neural Fluid Reconstruction

Reconstruct 3D fluid dynamics from multi-view video using 3D Gaussian Splatting with Navier-Stokes physics constraints.

## Features

- **Flexible input** - Works with 1-3 videos (single video uses monocular depth estimation)
- **Multi-view support** - Handle videos with different FPS, resolutions, start times
- **Automatic synchronization** - Align videos using motion signatures (0.3-3s offset detection)
- **Camera shake compensation** - Video stabilization + per-frame motion tracking
- **Camera calibration** - Automatic intrinsics/extrinsics estimation from feature matching
- **3D Gaussian Splatting** - Dynamic scene reconstruction with time-varying Gaussians
- **Physics estimation** - Velocity, pressure, density, and viscosity fields
- **Navier-Stokes constraints** - PINN-style physics-informed optimization

## Installation

```bash
# Clone the repository
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
- NVIDIA GPU with CUDA (recommended for best performance)
- Also works on: Apple Silicon Mac (MPS), or CPU (slower)

## Quick Start

### Single Video (Easiest)

```bash
# Just one video - uses AI depth estimation
python scripts/run_pipeline.py my_fluid_video.mp4
```

### Multi-View (Best Quality)

```bash
# 2-3 videos from different angles
python scripts/run_pipeline.py front.mp4 left.mp4 right.mp4
```

### Interactive Mode

```bash
python scripts/run_pipeline.py
```

Opens a file dialog to select video files.

### All Options

```bash
python scripts/run_pipeline.py --help

Options:
  videos              Input video files (1-3 videos)
  -o, --output        Output directory (default: outputs)
  --device            Device: auto, cuda, mps, cpu (default: auto)
  --max-resolution    Maximum frame resolution (default: 1080)
  --epochs            Number of training epochs (default: 5000)
  --no-physics        Skip physics estimation stage
  --resume            Resume from checkpoint directory
  --test              Run quick test with reduced settings
  --interactive       Force interactive mode
```

### Device Support

| Device | Description | Performance |
|--------|-------------|-------------|
| `auto` | Auto-detect best available (default) | - |
| `cuda` | NVIDIA GPU with CUDA | Fastest |
| `mps` | Apple Silicon Mac (M1/M2/M3) | Fast |
| `cpu` | Any CPU | Slower |

For Mac users without NVIDIA GPU, the pipeline auto-detects and uses CPU.
Use `--max-resolution 480` for faster CPU processing.

### Quality Comparison

| Mode | Input | 3D Quality | Physics Accuracy |
|------|-------|------------|------------------|
| Single video | 1 video | Good | Good |
| Stereo | 2 videos | Better | Better |
| Multi-view | 3 videos | Best | Best |

## Recording Videos

For best results when capturing fluid:

1. **Use 3 cameras** - Place at different angles (ideally 60-120° apart)
2. **Same scene** - All cameras must see the same fluid motion
3. **Stable mounting** - Use tripods if possible (shake compensation helps but isn't perfect)
4. **Good lighting** - Consistent lighting across all views
5. **Start recording** - Start all cameras within 3 seconds of each other
6. **Frame rate** - Higher is better (60fps recommended), but different rates are OK

### Example Setup

```
        Camera 1 (front)
             |
             v
    [===================]
    [                   ]
    [      FLUID        ]
    [                   ]
    [===================]
   /                     \
  v                       v
Camera 2              Camera 3
(left 60°)           (right 60°)
```

## Pipeline Stages

1. **Video Loading** - Load and validate input videos
2. **Stabilization** - Remove camera shake, track residual motion
3. **Optical Flow** - Compute dense motion between frames
4. **Temporal Sync** - Align video timelines using motion correlation
5. **Feature Extraction** - Detect keypoints (SuperPoint/SIFT)
6. **Camera Calibration** - Estimate camera poses from matches
7. **Triangulation** - Build initial 3D point cloud
8. **Gaussian Init** - Initialize 3D Gaussians from point cloud
9. **Reconstruction** - Optimize Gaussians with photometric loss
10. **Physics Estimation** - Estimate velocity, pressure, density, viscosity
11. **Output** - Save results (Gaussians, fields, visualizations)

## Output Files

After running, find results in `outputs/`:

```
outputs/
├── checkpoints/           # Training checkpoints
│   └── epoch_XXXX.pt
├── gaussians/             # Reconstructed Gaussians
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
└── calibration/           # Camera parameters
    └── cameras.json
```

## Configuration

Edit parameters in code or pass via command line:

```python
from src.config import PipelineConfig

config = PipelineConfig(
    # Hardware
    device="cuda",
    mixed_precision=True,

    # Video processing
    max_resolution=1080,
    enable_stabilization=True,      # Remove camera shake
    track_camera_motion=True,       # Track per-frame poses

    # Synchronization
    max_offset_seconds=3.0,         # Max time offset between videos

    # Gaussian Splatting
    initial_n_gaussians=100_000,
    max_n_gaussians=500_000,
    sh_degree=3,                    # Spherical harmonics degree

    # Training
    optimization=OptimizationConfig(
        n_epochs=10000,
        lr_position=1e-4,
        lr_color=2.5e-3,
    ),

    # Physics
    physics=PhysicsConfig(
        gravity=(0.0, -9.81, 0.0),
        reference_density=1000.0,   # kg/m³ (water)
    ),
)
```

## Python API

```python
from pathlib import Path
from src.config import PipelineConfig
from src.pipeline import FluidReconstructionPipeline

# Create pipeline
config = PipelineConfig(device="cuda", output_dir=Path("my_outputs"))
pipeline = FluidReconstructionPipeline(config)

# Run on videos
videos = [Path("view1.mp4"), Path("view2.mp4"), Path("view3.mp4")]
result = pipeline.run(videos)

# Check results
if result.success:
    print(f"Reconstruction complete in {result.total_duration:.1f}s")

    # Access outputs
    gaussians = result.outputs.get('gaussians')
    velocity_field = result.outputs.get('velocity_field')
    viscosity = result.outputs.get('viscosity_estimate')
```

## Physics Outputs

The pipeline estimates physical properties of the fluid:

### Velocity Field `u(x, t)`
- 3D vector field representing fluid motion
- Units: m/s (in reconstructed coordinate system)

### Pressure Field `p(x, t)`
- Scalar field representing pressure distribution
- Estimated via Navier-Stokes constraints

### Density Field `ρ(x, t)`
- Scalar field from Gaussian opacity aggregation
- Relative density (normalized)

### Viscosity `ν`
- Kinematic viscosity estimated from velocity diffusion
- Can identify fluid type (water ~0.001, honey ~2.0 Pa·s)

## Troubleshooting

### Out of Memory
```bash
# Reduce resolution
python scripts/run_pipeline.py videos/*.mp4 --max-resolution 720

# Or reduce Gaussians in config
config.initial_n_gaussians = 50000
config.max_n_gaussians = 200000
```

### Poor Synchronization
- Ensure videos show the same motion event
- Check that offset is within 0.3-3 seconds
- Try increasing `max_offset_seconds` if needed

### Camera Calibration Fails
- Need sufficient texture/features in scene
- Ensure cameras have overlapping views
- Try different camera angles (not too parallel)

### Blurry Reconstruction
- Increase training epochs: `--epochs 10000`
- Check input video quality
- Ensure cameras are stable during capture

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
│   └── run_pipeline.py           # Main entry point
├── outputs/                      # Default output directory
├── requirements.txt
└── README.md
```

## Citation


## License

MIT License - see LICENSE file for details.
