"""
3D Gaussian representation for neural radiance fields.

Core data structures for representing scenes as collections of 3D Gaussians
with position, scale, rotation, opacity, and color attributes.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class Gaussian3D:
    """Single 3D Gaussian primitive."""
    position: torch.Tensor  # (3,) center position
    scale: torch.Tensor  # (3,) log-scale in each axis
    rotation: torch.Tensor  # (4,) quaternion (w, x, y, z)
    opacity: torch.Tensor  # (1,) logit-space opacity
    features: torch.Tensor  # (C,) color/SH features


class GaussianCloud(nn.Module):
    """
    Collection of 3D Gaussians representing a scene.

    Attributes are stored as nn.Parameters for gradient-based optimization.
    """

    def __init__(
        self,
        n_gaussians: int = 0,
        sh_degree: int = 3,
        device: str = "cuda"
    ):
        """
        Initialize Gaussian cloud.

        Args:
            n_gaussians: Initial number of Gaussians
            sh_degree: Spherical harmonics degree for view-dependent color
            device: Device to store tensors
        """
        super().__init__()

        self.sh_degree = sh_degree
        self.n_sh_coeffs = (sh_degree + 1) ** 2
        self.device = device

        # Initialize empty parameters
        self._positions = nn.Parameter(torch.zeros(n_gaussians, 3, device=device))
        self._scales = nn.Parameter(torch.zeros(n_gaussians, 3, device=device))
        self._rotations = nn.Parameter(torch.zeros(n_gaussians, 4, device=device))
        self._opacities = nn.Parameter(torch.zeros(n_gaussians, 1, device=device))
        self._features_dc = nn.Parameter(torch.zeros(n_gaussians, 3, device=device))
        self._features_rest = nn.Parameter(
            torch.zeros(n_gaussians, (self.n_sh_coeffs - 1) * 3, device=device)
        )

        # Initialize velocities for dynamic Gaussians (set later if needed)
        self._velocities = nn.Parameter(
            torch.zeros(n_gaussians, 3, device=device),
            requires_grad=False
        )

    @property
    def n_gaussians(self) -> int:
        """Number of Gaussians in the cloud."""
        return self._positions.shape[0]

    @property
    def positions(self) -> torch.Tensor:
        """Get Gaussian positions (N, 3)."""
        return self._positions

    @property
    def scales(self) -> torch.Tensor:
        """Get Gaussian scales in world space (N, 3)."""
        return torch.exp(self._scales)

    @property
    def rotations(self) -> torch.Tensor:
        """Get Gaussian rotations as normalized quaternions (N, 4)."""
        return torch.nn.functional.normalize(self._rotations, dim=-1)

    @property
    def opacities(self) -> torch.Tensor:
        """Get Gaussian opacities in [0, 1] (N, 1)."""
        return torch.sigmoid(self._opacities)

    @property
    def features(self) -> torch.Tensor:
        """Get full SH features (N, n_sh_coeffs, 3)."""
        dc = self._features_dc.unsqueeze(1)  # (N, 1, 3)
        rest = self._features_rest.reshape(-1, self.n_sh_coeffs - 1, 3)  # (N, n_sh-1, 3)
        return torch.cat([dc, rest], dim=1)

    @property
    def velocities(self) -> torch.Tensor:
        """Get Gaussian velocities (N, 3)."""
        return self._velocities

    def get_covariance_matrices(self) -> torch.Tensor:
        """
        Compute 3D covariance matrices from scale and rotation.

        Returns:
            Covariance matrices (N, 3, 3)
        """
        scales = self.scales
        rotations = self.rotations

        # Build rotation matrices from quaternions
        R = quaternion_to_rotation_matrix(rotations)  # (N, 3, 3)

        # Scale matrix
        S = torch.diag_embed(scales)  # (N, 3, 3)

        # Covariance = R @ S @ S^T @ R^T = R @ (S^2) @ R^T
        # More numerically stable: RS @ RS^T
        RS = R @ S  # (N, 3, 3)
        covariance = RS @ RS.transpose(-1, -2)

        return covariance

    def initialize_from_point_cloud(
        self,
        points: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
        initial_scale: Optional[float] = None,
        initial_opacity: float = 0.5,
        adaptive_scale: bool = True
    ):
        """
        Initialize Gaussians from a point cloud.

        Args:
            points: 3D points (N, 3)
            colors: Optional RGB colors (N, 3) in [0, 1]
            initial_scale: Initial scale for Gaussians (if None, computed adaptively)
            initial_opacity: Initial opacity in [0, 1] (default 0.5)
            adaptive_scale: Whether to compute scale from point cloud density
        """
        n_points = points.shape[0]

        with torch.no_grad():
            # Positions
            self._positions = nn.Parameter(points.clone().to(self.device))

            # Compute adaptive scale from nearest neighbor distances
            if initial_scale is None and adaptive_scale and n_points > 1:
                # Use k-NN to estimate local density
                k = min(4, n_points - 1)
                pts_device = points.to(self.device)

                # Compute pairwise distances (batch for memory efficiency)
                batch_size = min(1000, n_points)
                all_scales = []

                for i in range(0, n_points, batch_size):
                    end_i = min(i + batch_size, n_points)
                    batch_pts = pts_device[i:end_i]

                    # Distance to all points
                    dists = torch.cdist(batch_pts, pts_device)  # (batch, N)

                    # Set self-distance to large value
                    dists[torch.arange(end_i - i), torch.arange(i, end_i)] = float('inf')

                    # Get k nearest neighbors
                    knn_dists, _ = dists.topk(k, largest=False)  # (batch, k)

                    # Use mean of k-NN distances as initial scale
                    batch_scales = knn_dists.mean(dim=1)  # (batch,)
                    all_scales.append(batch_scales)

                scales_1d = torch.cat(all_scales)

                # Clamp to reasonable range
                min_scale = 0.0001
                max_scale = 0.5
                scales_1d = torch.clamp(scales_1d, min_scale, max_scale)

                # Use same scale for all 3 axes (isotropic)
                scales = scales_1d.unsqueeze(-1).expand(-1, 3)
                self._scales = nn.Parameter(torch.log(scales))

                print(f"    Adaptive scale: mean={scales_1d.mean():.4f}, range=[{scales_1d.min():.4f}, {scales_1d.max():.4f}]")
            else:
                # Use fixed scale
                if initial_scale is None:
                    initial_scale = 0.01  # Better default than 0.001
                self._scales = nn.Parameter(
                    torch.full((n_points, 3), np.log(initial_scale), device=self.device)
                )

            # Rotations (identity quaternion: w=1, x=y=z=0)
            rotations = torch.zeros(n_points, 4, device=self.device)
            rotations[:, 0] = 1.0  # w component
            self._rotations = nn.Parameter(rotations)

            # Opacities (convert from [0,1] to logit space)
            # sigmoid(x) = opacity => x = log(opacity / (1 - opacity))
            opacity_clamped = np.clip(initial_opacity, 0.01, 0.99)
            opacity_logit = np.log(opacity_clamped / (1 - opacity_clamped))
            self._opacities = nn.Parameter(
                torch.full((n_points, 1), opacity_logit, device=self.device)
            )

            # Colors (SH DC component)
            if colors is not None:
                # Convert RGB to SH DC (Y_0^0 = 1/(2*sqrt(pi)))
                C0 = 0.28209479177387814
                colors_device = colors.to(self.device)
                self._features_dc = nn.Parameter(
                    (colors_device - 0.5) / C0
                )
            else:
                self._features_dc = nn.Parameter(
                    torch.zeros(n_points, 3, device=self.device)
                )

            # Rest of SH coefficients
            self._features_rest = nn.Parameter(
                torch.zeros(n_points, (self.n_sh_coeffs - 1) * 3, device=self.device)
            )

            # Velocities
            self._velocities = nn.Parameter(
                torch.zeros(n_points, 3, device=self.device),
                requires_grad=False
            )

    def get_colors(self, view_dirs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get Gaussian colors, optionally with view-dependent effects.

        Args:
            view_dirs: Optional view directions (N, 3) or (1, 3)

        Returns:
            RGB colors (N, 3) in [0, 1]
        """
        if view_dirs is None or self.sh_degree == 0:
            # Just use DC component
            C0 = 0.28209479177387814
            return torch.sigmoid(self._features_dc * C0 + 0.5)

        # Evaluate spherical harmonics
        colors = eval_sh(self.sh_degree, self.features, view_dirs)
        return torch.clamp(colors + 0.5, 0.0, 1.0)

    def clone(self, indices: Optional[torch.Tensor] = None) -> "GaussianCloud":
        """
        Clone the Gaussian cloud or a subset.

        Args:
            indices: Optional indices to select subset

        Returns:
            New GaussianCloud
        """
        if indices is None:
            indices = torch.arange(self.n_gaussians, device=self.device)

        new_cloud = GaussianCloud(
            n_gaussians=len(indices),
            sh_degree=self.sh_degree,
            device=self.device
        )

        with torch.no_grad():
            new_cloud._positions = nn.Parameter(self._positions[indices].clone())
            new_cloud._scales = nn.Parameter(self._scales[indices].clone())
            new_cloud._rotations = nn.Parameter(self._rotations[indices].clone())
            new_cloud._opacities = nn.Parameter(self._opacities[indices].clone())
            new_cloud._features_dc = nn.Parameter(self._features_dc[indices].clone())
            new_cloud._features_rest = nn.Parameter(self._features_rest[indices].clone())
            new_cloud._velocities = nn.Parameter(self._velocities[indices].clone())

        return new_cloud

    def densify_and_prune(
        self,
        grad_threshold: float = 0.0002,
        scale_threshold: float = 0.01,
        opacity_threshold: float = 0.005,
        max_screen_size: float = 0.1
    ):
        """
        Densify Gaussians with high gradients and prune low opacity ones.

        This is called periodically during training to improve coverage.
        """
        # Get position gradients
        if self._positions.grad is None:
            return

        grads = self._positions.grad.norm(dim=-1)

        # Densify: split or clone based on scale
        large_scale = self.scales.max(dim=-1)[0] > scale_threshold
        high_grad = grads > grad_threshold

        # Clone small Gaussians with high gradients
        clone_mask = high_grad & ~large_scale

        # Split large Gaussians with high gradients
        split_mask = high_grad & large_scale

        # Prune low opacity
        prune_mask = self.opacities.squeeze() < opacity_threshold

        # Apply operations
        self._densify_clone(clone_mask)
        self._densify_split(split_mask)
        self._prune(prune_mask)

    def _densify_clone(self, mask: torch.Tensor):
        """Clone Gaussians at mask positions."""
        if not mask.any():
            return

        # Clone all parameters
        new_positions = self._positions[mask]
        new_scales = self._scales[mask]
        new_rotations = self._rotations[mask]
        new_opacities = self._opacities[mask]
        new_features_dc = self._features_dc[mask]
        new_features_rest = self._features_rest[mask]
        new_velocities = self._velocities[mask]

        # Add small noise to positions
        new_positions = new_positions + torch.randn_like(new_positions) * 0.001

        # Concatenate
        self._concat_tensors(
            new_positions, new_scales, new_rotations, new_opacities,
            new_features_dc, new_features_rest, new_velocities
        )

    def _densify_split(self, mask: torch.Tensor):
        """Split Gaussians at mask positions into two smaller ones."""
        if not mask.any():
            return

        # Create two Gaussians per split
        positions = self._positions[mask]
        scales = self._scales[mask]
        rotations = self._rotations[mask]

        # Compute split offset along major axis
        R = quaternion_to_rotation_matrix(self.rotations[mask])
        major_axis = R[:, :, 0]  # First column is major axis
        offset = major_axis * self.scales[mask, 0:1] * 0.5

        # New Gaussians
        new_positions = torch.cat([positions - offset, positions + offset], dim=0)
        new_scales = torch.cat([scales - 0.5, scales - 0.5], dim=0)  # Reduce scale (log space)
        new_rotations = torch.cat([rotations, rotations], dim=0)
        new_opacities = torch.cat([self._opacities[mask], self._opacities[mask]], dim=0)
        new_features_dc = torch.cat([self._features_dc[mask], self._features_dc[mask]], dim=0)
        new_features_rest = torch.cat([self._features_rest[mask], self._features_rest[mask]], dim=0)
        new_velocities = torch.cat([self._velocities[mask], self._velocities[mask]], dim=0)

        # Remove original and add new
        keep_mask = ~mask
        self._filter_tensors(keep_mask)
        self._concat_tensors(
            new_positions, new_scales, new_rotations, new_opacities,
            new_features_dc, new_features_rest, new_velocities
        )

    def _prune(self, mask: torch.Tensor):
        """Remove Gaussians at mask positions."""
        if not mask.any():
            return
        self._filter_tensors(~mask)

    def _filter_tensors(self, keep_mask: torch.Tensor):
        """Keep only Gaussians at keep_mask positions."""
        self._positions = nn.Parameter(self._positions[keep_mask])
        self._scales = nn.Parameter(self._scales[keep_mask])
        self._rotations = nn.Parameter(self._rotations[keep_mask])
        self._opacities = nn.Parameter(self._opacities[keep_mask])
        self._features_dc = nn.Parameter(self._features_dc[keep_mask])
        self._features_rest = nn.Parameter(self._features_rest[keep_mask])
        self._velocities = nn.Parameter(self._velocities[keep_mask])

    def _concat_tensors(self, positions, scales, rotations, opacities,
                        features_dc, features_rest, velocities):
        """Concatenate new Gaussians to existing ones."""
        self._positions = nn.Parameter(torch.cat([self._positions, positions], dim=0))
        self._scales = nn.Parameter(torch.cat([self._scales, scales], dim=0))
        self._rotations = nn.Parameter(torch.cat([self._rotations, rotations], dim=0))
        self._opacities = nn.Parameter(torch.cat([self._opacities, opacities], dim=0))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, features_dc], dim=0))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest, features_rest], dim=0))
        self._velocities = nn.Parameter(torch.cat([self._velocities, velocities], dim=0))


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices.

    Args:
        q: Quaternions (N, 4) as (w, x, y, z)

    Returns:
        Rotation matrices (N, 3, 3)
    """
    # Normalize
    q = torch.nn.functional.normalize(q, dim=-1)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Rotation matrix from quaternion
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
        2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)
    ], dim=-1).reshape(-1, 3, 3)

    return R


def eval_sh(degree: int, sh: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """
    Evaluate spherical harmonics.

    Args:
        degree: SH degree (0-3)
        sh: SH coefficients (N, n_coeffs, 3)
        directions: View directions (N, 3) or (1, 3)

    Returns:
        Colors (N, 3)
    """
    # SH basis functions
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    ]
    C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    ]

    result = C0 * sh[:, 0]

    if degree < 1:
        return result

    x, y, z = directions[:, 0:1], directions[:, 1:2], directions[:, 2:3]

    result = result + C1 * (-y * sh[:, 1] + z * sh[:, 2] - x * sh[:, 3])

    if degree < 2:
        return result

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z

    result = result + C2[0] * xy * sh[:, 4]
    result = result + C2[1] * yz * sh[:, 5]
    result = result + C2[2] * (2 * zz - xx - yy) * sh[:, 6]
    result = result + C2[3] * xz * sh[:, 7]
    result = result + C2[4] * (xx - yy) * sh[:, 8]

    if degree < 3:
        return result

    result = result + C3[0] * y * (3 * xx - yy) * sh[:, 9]
    result = result + C3[1] * xy * z * sh[:, 10]
    result = result + C3[2] * y * (4 * zz - xx - yy) * sh[:, 11]
    result = result + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
    result = result + C3[4] * x * (4 * zz - xx - yy) * sh[:, 13]
    result = result + C3[5] * z * (xx - yy) * sh[:, 14]
    result = result + C3[6] * x * (xx - 3 * yy) * sh[:, 15]

    return result
