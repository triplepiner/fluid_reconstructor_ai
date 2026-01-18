"""
Monocular depth estimation for single-view reconstruction.

Uses pretrained depth models to estimate depth from a single image/video.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Union
from pathlib import Path
from tqdm import tqdm


class DepthEstimator:
    """
    Estimate depth from monocular images using pretrained models.

    Supports multiple backends:
    - MiDaS (default, works well for general scenes)
    - Depth Anything (state-of-the-art)
    - ZoeDepth (metric depth)
    """

    def __init__(
        self,
        model_type: str = "midas",
        device: str = "cuda"
    ):
        """
        Initialize depth estimator.

        Args:
            model_type: Model to use ('midas', 'midas_small', 'zoedepth')
            device: Device for inference (cuda, mps, or cpu)
        """
        # MPS may have compatibility issues with some models, use CPU as fallback
        if device == "mps":
            print("    Note: MPS may have issues with depth models, using CPU for depth estimation")
            self.device = "cpu"
        else:
            self.device = device
        self.model_type = model_type
        self.model = None
        self.transform = None

        self._load_model()

    def _load_model(self):
        """Load the depth estimation model."""
        if self.model_type in ["midas", "midas_small"]:
            self._load_midas()
        elif self.model_type == "zoedepth":
            self._load_zoedepth()
        else:
            # Fallback to simple disparity estimation
            print(f"Unknown model type {self.model_type}, using simple estimation")
            self.model = None

    def _load_midas(self):
        """Load MiDaS model from torch hub."""
        try:
            if self.model_type == "midas_small":
                self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                self.transform = midas_transforms.small_transform
            else:
                # Try DPT-Large first, fall back to smaller models
                try:
                    self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
                    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                    self.transform = midas_transforms.dpt_transform
                except:
                    self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
                    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                    self.transform = midas_transforms.small_transform

            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"Loaded MiDaS depth model")

        except Exception as e:
            print(f"Failed to load MiDaS: {e}")
            print("Using fallback depth estimation")
            self.model = None

    def _load_zoedepth(self):
        """Load ZoeDepth model."""
        try:
            self.model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.transform = None  # ZoeDepth handles its own preprocessing
            print("Loaded ZoeDepth model")
        except Exception as e:
            print(f"Failed to load ZoeDepth: {e}, falling back to MiDaS")
            self.model_type = "midas"
            self._load_midas()

    def estimate(self, image: torch.Tensor) -> torch.Tensor:
        """
        Estimate depth for a single image.

        Args:
            image: Input image (H, W, 3) in [0, 1] range

        Returns:
            Depth map (H, W) - relative depth (larger = farther)
        """
        H, W = image.shape[:2]

        if self.model is None:
            # Fallback: simple gradient-based depth estimation
            return self._estimate_simple(image)

        with torch.no_grad():
            if self.model_type == "zoedepth":
                # ZoeDepth expects (B, C, H, W) in [0, 1]
                img = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
                depth = self.model.infer(img)
                depth = depth.squeeze()
            else:
                # MiDaS
                img_np = (image.cpu().numpy() * 255).astype(np.uint8)
                input_batch = self.transform(img_np).to(self.device)

                prediction = self.model(input_batch)
                prediction = F.interpolate(
                    prediction.unsqueeze(1),
                    size=(H, W),
                    mode="bicubic",
                    align_corners=False
                ).squeeze()

                # MiDaS outputs inverse depth (disparity)
                depth = prediction

        # Normalize to [0, 1] range
        depth = depth.cpu()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        return depth

    def estimate_sequence(
        self,
        frames: torch.Tensor,
        temporal_smooth: bool = True
    ) -> torch.Tensor:
        """
        Estimate depth for a sequence of frames.

        Args:
            frames: Video frames (T, H, W, 3)
            temporal_smooth: Apply temporal smoothing

        Returns:
            Depth maps (T, H, W)
        """
        T = frames.shape[0]
        depths = []

        for i in tqdm(range(T), desc="Estimating depth"):
            depth = self.estimate(frames[i])
            depths.append(depth)

        depths = torch.stack(depths)

        if temporal_smooth and T > 1:
            depths = self._temporal_smooth(depths)

        return depths

    def _estimate_simple(self, image: torch.Tensor) -> torch.Tensor:
        """
        Simple depth estimation fallback using image gradients and blur.

        Assumes: farther objects are blurrier and have less texture.
        """
        import cv2

        H, W = image.shape[:2]

        # Convert to grayscale
        if image.shape[-1] == 3:
            gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        else:
            gray = image

        gray_np = gray.cpu().numpy()

        # Compute Laplacian (texture/sharpness measure)
        laplacian = cv2.Laplacian(gray_np.astype(np.float32), cv2.CV_32F)
        texture = np.abs(laplacian)

        # Blur to get local average
        texture_smooth = cv2.GaussianBlur(texture, (31, 31), 0)

        # More texture = closer (inverse relationship)
        depth = 1.0 / (texture_smooth + 0.1)

        # Normalize
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        return torch.from_numpy(depth).float()

    def _temporal_smooth(
        self,
        depths: torch.Tensor,
        window: int = 3
    ) -> torch.Tensor:
        """Apply temporal smoothing to depth sequence."""
        T = depths.shape[0]
        smoothed = depths.clone()

        for t in range(T):
            start = max(0, t - window // 2)
            end = min(T, t + window // 2 + 1)
            smoothed[t] = depths[start:end].mean(dim=0)

        return smoothed


def depth_to_pointcloud(
    depth: torch.Tensor,
    image: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert depth map to 3D point cloud.

    Args:
        depth: Depth map (H, W)
        image: RGB image (H, W, 3)
        fx, fy: Focal lengths
        cx, cy: Principal point
        depth_scale: Scale factor for depth values

    Returns:
        Tuple of (points (N, 3), colors (N, 3))
    """
    H, W = depth.shape

    # Create pixel coordinate grid
    u = torch.arange(W, dtype=torch.float32)
    v = torch.arange(H, dtype=torch.float32)
    u, v = torch.meshgrid(u, v, indexing='xy')

    # Scale depth (MiDaS gives relative depth, need to scale)
    z = depth * depth_scale

    # Backproject to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack into points
    points = torch.stack([x, y, z], dim=-1)  # (H, W, 3)
    points = points.reshape(-1, 3)  # (H*W, 3)

    # Get colors
    colors = image.reshape(-1, 3)

    # Filter out invalid points (zero or very large depth)
    valid = (z.reshape(-1) > 0.01) & (z.reshape(-1) < 100)
    points = points[valid]
    colors = colors[valid]

    return points, colors


def estimate_camera_intrinsics(
    width: int,
    height: int,
    fov_degrees: float = 60.0
) -> Tuple[float, float, float, float]:
    """
    Estimate camera intrinsics from image size and assumed FOV.

    Args:
        width: Image width
        height: Image height
        fov_degrees: Assumed horizontal field of view

    Returns:
        Tuple of (fx, fy, cx, cy)
    """
    # Assume square pixels
    fov_rad = np.radians(fov_degrees)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = fx  # Square pixels
    cx = width / 2
    cy = height / 2

    return fx, fy, cx, cy
