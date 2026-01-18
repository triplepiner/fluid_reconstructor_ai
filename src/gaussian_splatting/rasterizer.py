"""
Differentiable Gaussian rasterizer.

Renders 3D Gaussians to 2D images using either gsplat library
or a pure PyTorch fallback implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np

from ..config import Camera, CameraIntrinsics, CameraExtrinsics
from .gaussian import GaussianCloud, quaternion_to_rotation_matrix


class GaussianRasterizer(nn.Module):
    """
    Differentiable rasterizer for 3D Gaussian Splatting.

    Uses gsplat library when available for GPU-accelerated rendering,
    with a pure PyTorch fallback for compatibility.
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        use_gsplat: bool = True,
        background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        device: str = "cuda"
    ):
        """
        Initialize the rasterizer.

        Args:
            image_height: Output image height
            image_width: Output image width
            use_gsplat: Whether to use gsplat library (if available)
            background_color: Background color (R, G, B)
            device: Device for computations
        """
        super().__init__()

        self.height = image_height
        self.width = image_width
        self.device = device
        self.background = torch.tensor(background_color, device=device)

        self.gsplat_available = False
        if use_gsplat:
            try:
                import gsplat
                self.gsplat_available = True
                self._gsplat = gsplat
            except ImportError:
                print("gsplat not available, using PyTorch fallback")

    def forward(
        self,
        gaussians: GaussianCloud,
        camera: Camera,
        return_depth: bool = False,
        return_alpha: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Render Gaussians from a camera viewpoint.

        Args:
            gaussians: Gaussian cloud to render
            camera: Camera to render from
            return_depth: Whether to return depth map
            return_alpha: Whether to return alpha map

        Returns:
            Dict with 'image' (H, W, 3) and optionally 'depth', 'alpha'
        """
        if self.gsplat_available:
            return self._render_gsplat(gaussians, camera, return_depth, return_alpha)
        else:
            return self._render_pytorch(gaussians, camera, return_depth, return_alpha)

    def _render_gsplat(
        self,
        gaussians: GaussianCloud,
        camera: Camera,
        return_depth: bool,
        return_alpha: bool
    ) -> Dict[str, torch.Tensor]:
        """Render using gsplat library."""
        # Prepare camera parameters
        K = camera.intrinsics.to_matrix().to(self.device)
        R = camera.extrinsics.R.to(self.device)
        t = camera.extrinsics.t.to(self.device)

        # View matrix (world to camera)
        viewmat = torch.eye(4, device=self.device)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = t

        # Projection matrix
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Get Gaussian parameters (ensure on device)
        means = gaussians.positions.to(self.device)
        scales = gaussians.scales.to(self.device)
        quats = gaussians.rotations.to(self.device)
        opacities = gaussians.opacities.to(self.device).squeeze(-1)

        # Get colors (view-dependent)
        view_dir = -R.T @ t  # Camera position in world coords
        view_dirs = F.normalize(view_dir.unsqueeze(0) - means, dim=-1)
        colors = gaussians.get_colors(view_dirs).to(self.device)

        # Render using gsplat
        rendered, alpha, info = self._gsplat.rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=self.width,
            height=self.height,
            backgrounds=self.background.unsqueeze(0),
        )

        result = {'image': rendered.squeeze(0)}  # (H, W, 3)

        if return_alpha:
            result['alpha'] = alpha.squeeze(0)

        if return_depth and 'depths' in info:
            result['depth'] = info['depths'].squeeze(0)

        return result

    def _render_pytorch(
        self,
        gaussians: GaussianCloud,
        camera: Camera,
        return_depth: bool,
        return_alpha: bool
    ) -> Dict[str, torch.Tensor]:
        """
        Pure PyTorch fallback renderer.

        This is slower but works without gsplat.
        """
        # Get camera matrices
        K = camera.intrinsics.to_matrix().to(self.device)
        R = camera.extrinsics.R.to(self.device)
        t = camera.extrinsics.t.to(self.device)

        # Transform Gaussians to camera space (ensure tensors are on device)
        means_world = gaussians.positions.to(self.device)  # (N, 3)
        means_cam = (R @ means_world.T).T + t  # (N, 3)

        # Filter Gaussians behind camera
        valid_mask = means_cam[:, 2] > 0.01
        if not valid_mask.any():
            return {
                'image': self.background.unsqueeze(0).unsqueeze(0).expand(self.height, self.width, 3)
            }

        means_cam = means_cam[valid_mask]
        n_valid = means_cam.shape[0]

        # Project to 2D
        means_2d = (K @ means_cam.T).T
        means_2d = means_2d[:, :2] / means_2d[:, 2:3]  # (N, 2)

        # Get 2D covariances (ensure on device)
        cov_3d = gaussians.get_covariance_matrices().to(self.device)[valid_mask]
        cov_2d = self._project_covariance(cov_3d, means_cam, K, R)

        # Get colors and opacities (ensure on device)
        opacities = gaussians.opacities.to(self.device)[valid_mask].squeeze(-1)
        colors = gaussians.get_colors().to(self.device)[valid_mask]  # (N, 3)

        # Sort by depth (front to back for alpha compositing)
        depths = means_cam[:, 2]
        sort_idx = torch.argsort(depths)

        means_2d = means_2d[sort_idx]
        cov_2d = cov_2d[sort_idx]
        opacities = opacities[sort_idx]
        colors = colors[sort_idx]
        depths = depths[sort_idx]

        # Rasterize
        image, alpha, depth = self._rasterize_gaussians(
            means_2d, cov_2d, opacities, colors, depths,
            return_depth
        )

        result = {'image': image}
        if return_alpha:
            result['alpha'] = alpha
        if return_depth:
            result['depth'] = depth

        return result

    def _project_covariance(
        self,
        cov_3d: torch.Tensor,
        means_cam: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor
    ) -> torch.Tensor:
        """
        Project 3D covariances to 2D.

        Args:
            cov_3d: 3D covariance matrices (N, 3, 3)
            means_cam: Points in camera space (N, 3)
            K: Intrinsic matrix (3, 3)
            R: Rotation matrix (3, 3)

        Returns:
            2D covariance matrices (N, 2, 2)
        """
        N = means_cam.shape[0]
        fx, fy = K[0, 0], K[1, 1]

        # Jacobian of projection at each point
        z = means_cam[:, 2:3]  # (N, 1)
        x = means_cam[:, 0:1]
        y = means_cam[:, 1:2]

        J = torch.zeros(N, 2, 3, device=self.device)
        J[:, 0, 0] = fx / z.squeeze()
        J[:, 0, 2] = -fx * x.squeeze() / (z.squeeze() ** 2)
        J[:, 1, 1] = fy / z.squeeze()
        J[:, 1, 2] = -fy * y.squeeze() / (z.squeeze() ** 2)

        # Transform covariance to camera space then project
        cov_cam = R.unsqueeze(0) @ cov_3d @ R.T.unsqueeze(0)  # (N, 3, 3)
        cov_2d = J @ cov_cam @ J.transpose(-1, -2)  # (N, 2, 2)

        # Add small regularization for numerical stability
        cov_2d = cov_2d + 0.3 * torch.eye(2, device=self.device).unsqueeze(0)

        return cov_2d

    def _rasterize_gaussians(
        self,
        means_2d: torch.Tensor,
        cov_2d: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        depths: torch.Tensor,
        return_depth: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Rasterize sorted Gaussians using alpha compositing.

        Uses vectorized operations for better MPS performance.
        """
        H, W = self.height, self.width
        N = means_2d.shape[0]

        # Limit number of Gaussians for memory
        max_gaussians = min(N, 5000)
        means_2d = means_2d[:max_gaussians]
        cov_2d = cov_2d[:max_gaussians]
        opacities = opacities[:max_gaussians]
        colors = colors[:max_gaussians]
        depths = depths[:max_gaussians]
        N = max_gaussians

        # Initialize output
        image = self.background.unsqueeze(0).unsqueeze(0).expand(H, W, 3).clone()
        alpha_acc = torch.zeros(H, W, device=self.device)
        depth_map = torch.zeros(H, W, device=self.device) if return_depth else None

        # Pre-compute all covariance inverse matrices (vectorized)
        # For 2x2 symmetric [[a,b],[b,c]], inv = (1/det) * [[c,-b],[-b,a]]
        a = cov_2d[:, 0, 0]  # (N,)
        b = cov_2d[:, 0, 1]  # (N,)
        c = cov_2d[:, 1, 1]  # (N,)

        det = a * c - b * b  # (N,)
        det = torch.clamp(det, min=1e-8)

        # Covariance inverses: (N, 2, 2)
        cov_inv = torch.zeros(N, 2, 2, device=self.device)
        cov_inv[:, 0, 0] = c / det
        cov_inv[:, 0, 1] = -b / det
        cov_inv[:, 1, 0] = -b / det
        cov_inv[:, 1, 1] = a / det

        # Pre-compute radii using analytical eigenvalues
        # max eigenvalue = (a+c)/2 + sqrt(((a-c)/2)² + b²)
        trace_half = (a + c) * 0.5
        det_sqrt = torch.sqrt(torch.clamp(((a - c) * 0.5) ** 2 + b ** 2, min=1e-8))
        max_eigenvalue = trace_half + det_sqrt
        radii = 3.0 * torch.sqrt(max_eigenvalue)  # (N,)

        # Create pixel coordinates grid
        y_coords = torch.arange(H, device=self.device, dtype=torch.float32)
        x_coords = torch.arange(W, device=self.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        pixel_coords = torch.stack([xx, yy], dim=-1)  # (H, W, 2)

        # Process Gaussians in batches for better GPU utilization
        batch_size = 100
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)

            for i in range(batch_start, batch_end):
                mean = means_2d[i]
                radius = radii[i]

                # Compute bounding box
                x_min = max(0, int(mean[0] - radius))
                x_max = min(W, int(mean[0] + radius) + 1)
                y_min = max(0, int(mean[1] - radius))
                y_max = min(H, int(mean[1] + radius) + 1)

                if x_min >= x_max or y_min >= y_max:
                    continue

                # Get pixel region and compute Gaussian values
                region_coords = pixel_coords[y_min:y_max, x_min:x_max]
                diff = region_coords - mean
                diff_flat = diff.reshape(-1, 2)

                # Use pre-computed inverse
                inv = cov_inv[i]
                mahal = torch.sum(diff_flat @ inv * diff_flat, dim=-1)
                gauss = torch.exp(-0.5 * mahal).reshape(diff.shape[0], diff.shape[1])

                # Alpha compositing
                alpha = opacities[i] * gauss
                region_alpha_acc = alpha_acc[y_min:y_max, x_min:x_max]
                transmittance = 1.0 - region_alpha_acc
                contrib = alpha * transmittance

                # Update outputs
                image[y_min:y_max, x_min:x_max] += contrib.unsqueeze(-1) * colors[i]
                alpha_acc[y_min:y_max, x_min:x_max] += contrib

                if return_depth:
                    depth_map[y_min:y_max, x_min:x_max] += contrib * depths[i]

        if return_depth:
            depth_map = depth_map / (alpha_acc + 1e-8)

        return image, alpha_acc, depth_map


def create_camera_from_matrices(
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    width: int,
    height: int
) -> Camera:
    """
    Create a Camera object from matrices.

    Args:
        K: Intrinsic matrix (3, 3)
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)
        width: Image width
        height: Image height

    Returns:
        Camera object
    """
    intrinsics = CameraIntrinsics(
        fx=K[0, 0].item(),
        fy=K[1, 1].item(),
        cx=K[0, 2].item(),
        cy=K[1, 2].item()
    )

    extrinsics = CameraExtrinsics(R=R, t=t)

    return Camera(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        width=width,
        height=height
    )
