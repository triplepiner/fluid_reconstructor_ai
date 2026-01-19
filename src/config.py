"""
Configuration dataclasses for the fluid reconstruction pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import torch


@dataclass
class VideoMetadata:
    """Metadata extracted from a video file."""
    path: Path
    fps: float
    width: int
    height: int
    duration: float
    n_frames: int
    codec: str
    bitrate: Optional[int] = None


@dataclass
class VideoData:
    """Complete video data including frames and optical flow."""
    metadata: VideoMetadata
    frames: torch.Tensor  # Shape: (n_frames, H, W, 3), float32 [0,1]
    timestamps: torch.Tensor  # Shape: (n_frames,), float32, in seconds (float32 for MPS compatibility)
    optical_flow: Optional[torch.Tensor] = None  # Shape: (n_frames-1, H, W, 2)


@dataclass
class SyncParameters:
    """Parameters for temporal synchronization."""
    offsets: List[float]  # Time offset per video in seconds
    common_fps: float
    common_start: float
    common_end: float
    n_common_frames: int


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float  # Focal length x (pixels)
    fy: float  # Focal length y (pixels)
    cx: float  # Principal point x
    cy: float  # Principal point y
    k1: float = 0.0  # Radial distortion
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0  # Tangential distortion
    p2: float = 0.0

    def to_matrix(self, device: Optional[str] = None) -> torch.Tensor:
        """Returns 3x3 intrinsic matrix K.

        Args:
            device: Optional device to place tensor on (default: cpu)
        """
        K = torch.tensor([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=torch.float32)
        if device is not None:
            K = K.to(device)
        return K

    def distortion_coeffs(self, device: Optional[str] = None) -> torch.Tensor:
        """Returns distortion coefficients [k1, k2, p1, p2, k3].

        Args:
            device: Optional device to place tensor on (default: cpu)
        """
        coeffs = torch.tensor([self.k1, self.k2, self.p1, self.p2, self.k3])
        if device is not None:
            coeffs = coeffs.to(device)
        return coeffs


@dataclass
class CameraExtrinsics:
    """Camera extrinsic parameters."""
    R: torch.Tensor  # Rotation matrix, shape (3, 3)
    t: torch.Tensor  # Translation vector, shape (3,)

    def to_matrix(self) -> torch.Tensor:
        """Returns 3x4 extrinsic matrix [R|t]."""
        return torch.cat([self.R, self.t.reshape(3, 1)], dim=1)

    def to_projection(self, K: torch.Tensor) -> torch.Tensor:
        """Returns 3x4 projection matrix P = K[R|t]."""
        # Ensure K is on same device as extrinsics
        K = K.to(self.R.device)
        return K @ self.to_matrix()


@dataclass
class Camera:
    """Complete camera model with intrinsics and extrinsics."""
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    width: int
    height: int

    def project(self, points_3d: torch.Tensor) -> torch.Tensor:
        """Project 3D points to 2D image coordinates.

        Args:
            points_3d: (N, 3) tensor of 3D points in world coordinates

        Returns:
            (N, 2) tensor of 2D pixel coordinates
        """
        # Get device from extrinsics (the R matrix)
        device = self.extrinsics.R.device
        K = self.intrinsics.to_matrix(device)
        R = self.extrinsics.R
        t = self.extrinsics.t

        # Ensure points are on same device
        points_3d = points_3d.to(device)

        # Transform to camera coordinates
        points_cam = (R @ points_3d.T).T + t

        # Project to image plane
        points_2d = (K @ points_cam.T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:3]

        return points_2d


@dataclass
class AlignedFrames:
    """Synchronized frames from all views."""
    frames: torch.Tensor  # Shape: (n_views, n_frames, H, W, 3)
    timestamps: torch.Tensor  # Shape: (n_frames,), common timeline


@dataclass
class GaussianParams:
    """Parameters for a single 3D Gaussian."""
    position: torch.Tensor  # (3,) - mean position
    scale: torch.Tensor  # (3,) - log-scale in each axis
    rotation: torch.Tensor  # (4,) - quaternion (w, x, y, z)
    opacity: torch.Tensor  # (1,) - logit-space opacity
    color_sh: torch.Tensor  # (n_sh, 3) - spherical harmonics for color
    velocity: Optional[torch.Tensor] = None  # (3,) - velocity for dynamics


@dataclass
class OptimizationConfig:
    """Configuration for training optimization."""
    # Learning rates
    lr_position: float = 1e-4
    lr_scale: float = 5e-3
    lr_rotation: float = 1e-3
    lr_opacity: float = 5e-2
    lr_color: float = 2.5e-3
    lr_velocity: float = 1e-4
    lr_neural_fields: float = 1e-4
    lr_viscosity: float = 1e-5

    # Training settings
    n_epochs: int = 10000
    batch_size: int = 1
    densify_interval: int = 100
    densify_until_epoch: int = 5000
    prune_interval: int = 100

    # Densification thresholds
    grad_threshold: float = 0.0002
    scale_threshold: float = 0.01
    opacity_threshold: float = 0.005

    # Scheduler
    lr_decay_gamma: float = 0.995
    lr_decay_step: int = 100


@dataclass
class LossWeights:
    """Weights for different loss terms."""
    # Rendering losses
    photometric: float = 1.0
    ssim: float = 0.2

    # Temporal losses
    temporal_consistency: float = 0.1
    velocity_smoothness: float = 0.05

    # Optical flow loss
    optical_flow: float = 0.5

    # Physics losses
    momentum: float = 0.01
    continuity: float = 0.1
    boundary: float = 1.0

    # Regularization
    scale_reg: float = 0.001
    opacity_reg: float = 0.001


@dataclass
class PhysicsConfig:
    """Configuration for physics estimation."""
    # Navier-Stokes parameters
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)
    reference_density: float = 1000.0  # kg/m³ (water)

    # Neural field architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256, 256])
    positional_encoding_L: int = 10  # Number of frequency bands

    # Sampling
    n_physics_samples: int = 4096
    sample_near_surface: bool = True


@dataclass
class PipelineConfig:
    """Main configuration for the entire pipeline."""
    # Paths
    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("outputs/checkpoints")
    model_dir: Path = Path("models")

    # Hardware
    device: str = "cuda"
    mixed_precision: bool = True
    num_workers: int = 4

    # Video processing
    max_resolution: int = 1080
    target_fps: Optional[float] = None  # None = use max FPS from inputs

    # Camera shake handling
    enable_stabilization: bool = True  # Apply video stabilization
    track_camera_motion: bool = True   # Track per-frame camera poses
    stabilization_smoothing: int = 30  # Temporal smoothing radius for stabilization

    # Slow pan stabilization (preserves intentional camera motion)
    preserve_slow_pan: bool = True  # Separate shake from slow pans
    shake_frequency_cutoff: float = 2.0  # Hz - motion above this is considered shake
    min_pan_velocity: float = 0.5  # pixels/frame - minimum motion to consider as pan
    pan_smoothness_factor: float = 0.8  # How much to preserve pan (0=remove all, 1=keep all)

    # Frame timing for stabilization
    stabilization_start_frame: Optional[int] = None  # Start frame for stabilization (None=auto)
    stabilization_end_frame: Optional[int] = None  # End frame for stabilization (None=auto)
    stabilization_edge_taper: int = 10  # Frames to taper stabilization at edges

    # Fluid segmentation
    fluid_motion_threshold: float = 2.0  # Optical flow magnitude threshold for fluid
    use_appearance_segmentation: bool = True  # Use color/appearance to refine masks

    # Temporal synchronization
    max_offset_seconds: float = 3.0
    min_offset_seconds: float = 0.3

    # Gaussian Splatting
    initial_n_gaussians: int = 100000
    max_n_gaussians: int = 500000
    sh_degree: int = 3  # Spherical harmonics degree

    # Optimization
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)

    # Checkpointing
    checkpoint_interval: int = 1000
    log_interval: int = 100

    def __post_init__(self):
        """Create directories if they don't exist."""
        self.output_dir = Path(self.output_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.model_dir = Path(self.model_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()
