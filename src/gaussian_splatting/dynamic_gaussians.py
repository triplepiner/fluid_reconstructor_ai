"""
Dynamic Gaussian representation for time-varying fluids.

Extends the static Gaussian cloud with temporal dynamics including
velocities, deformation, and time-dependent parameters.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from .gaussian import GaussianCloud, quaternion_to_rotation_matrix


@dataclass
class TemporalGaussianState:
    """State of Gaussians at a specific timestep."""
    positions: torch.Tensor  # (N, 3)
    velocities: torch.Tensor  # (N, 3)
    scales: torch.Tensor  # (N, 3)
    rotations: torch.Tensor  # (N, 4)
    opacities: torch.Tensor  # (N, 1)
    features_dc: torch.Tensor  # (N, 3)
    timestamp: float


class DynamicGaussianCloud(nn.Module):
    """
    Time-varying 3D Gaussian representation for dynamic fluids.

    Supports multiple temporal representations:
    1. Per-frame: Independent Gaussians at each timestep
    2. Trajectory: Base Gaussians with learned motion trajectories
    3. Velocity: Gaussians with velocity fields for integration
    """

    def __init__(
        self,
        base_gaussians: GaussianCloud,
        n_timesteps: int,
        temporal_mode: str = "velocity",
        device: str = "cuda"
    ):
        """
        Initialize dynamic Gaussian cloud.

        Args:
            base_gaussians: Base Gaussian cloud for initial state
            n_timesteps: Number of timesteps
            temporal_mode: 'per_frame', 'trajectory', or 'velocity'
            device: Device for tensors
        """
        super().__init__()

        self.n_timesteps = n_timesteps
        self.temporal_mode = temporal_mode
        self.device = device
        self.sh_degree = base_gaussians.sh_degree

        if temporal_mode == "per_frame":
            self._init_per_frame(base_gaussians)
        elif temporal_mode == "trajectory":
            self._init_trajectory(base_gaussians)
        elif temporal_mode == "velocity":
            self._init_velocity(base_gaussians)
        else:
            raise ValueError(f"Unknown temporal mode: {temporal_mode}")

    def _init_per_frame(self, base: GaussianCloud):
        """Initialize with independent Gaussians per frame."""
        self.gaussians_per_frame = nn.ModuleList([
            self._clone_gaussians(base) for _ in range(self.n_timesteps)
        ])

    def _init_trajectory(self, base: GaussianCloud):
        """Initialize with base Gaussians and learned trajectories."""
        self.base_gaussians = self._clone_gaussians(base)

        # Trajectory offsets: (n_timesteps, n_gaussians, 3)
        n_gaussians = base.n_gaussians
        self.position_offsets = nn.Parameter(
            torch.zeros(self.n_timesteps, n_gaussians, 3, device=self.device)
        )

        # Optional scale and rotation changes
        self.scale_offsets = nn.Parameter(
            torch.zeros(self.n_timesteps, n_gaussians, 3, device=self.device)
        )

    def _init_velocity(self, base: GaussianCloud):
        """Initialize with velocities for integration."""
        self.base_gaussians = self._clone_gaussians(base)

        # Enable velocity gradients
        self.base_gaussians._velocities.requires_grad_(True)

        # Velocity MLP for spatially-varying velocity field
        self.velocity_mlp = nn.Sequential(
            nn.Linear(4, 64),  # x, y, z, t
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # velocity components
        ).to(self.device)

    def _clone_gaussians(self, source: GaussianCloud) -> GaussianCloud:
        """Create a copy of a Gaussian cloud."""
        cloud = GaussianCloud(
            n_gaussians=source.n_gaussians,
            sh_degree=source.sh_degree,
            device=self.device
        )

        with torch.no_grad():
            cloud._positions = nn.Parameter(source._positions.clone())
            cloud._scales = nn.Parameter(source._scales.clone())
            cloud._rotations = nn.Parameter(source._rotations.clone())
            cloud._opacities = nn.Parameter(source._opacities.clone())
            cloud._features_dc = nn.Parameter(source._features_dc.clone())
            cloud._features_rest = nn.Parameter(source._features_rest.clone())
            cloud._velocities = nn.Parameter(source._velocities.clone())

        return cloud

    @property
    def n_gaussians(self) -> int:
        """Number of Gaussians."""
        if self.temporal_mode == "per_frame":
            return self.gaussians_per_frame[0].n_gaussians
        else:
            return self.base_gaussians.n_gaussians

    def get_gaussians_at_time(
        self,
        t: float,
        dt: float = 1.0 / 30.0
    ) -> GaussianCloud:
        """
        Get Gaussians at a specific time.

        Args:
            t: Time (0 to n_timesteps-1 or normalized 0 to 1)
            dt: Time step for velocity integration

        Returns:
            GaussianCloud at time t
        """
        if self.temporal_mode == "per_frame":
            return self._get_per_frame(t)
        elif self.temporal_mode == "trajectory":
            return self._get_trajectory(t)
        elif self.temporal_mode == "velocity":
            return self._get_velocity(t, dt)

    def _get_per_frame(self, t: float) -> GaussianCloud:
        """Get Gaussians for per-frame mode."""
        frame_idx = int(t)
        frame_idx = max(0, min(frame_idx, self.n_timesteps - 1))
        return self.gaussians_per_frame[frame_idx]

    def _get_trajectory(self, t: float) -> GaussianCloud:
        """Get Gaussians with trajectory interpolation."""
        frame_idx = int(t)
        alpha = t - frame_idx

        frame_idx = max(0, min(frame_idx, self.n_timesteps - 2))

        # Interpolate between frames
        offset_curr = self.position_offsets[frame_idx]
        offset_next = self.position_offsets[frame_idx + 1]
        offset = (1 - alpha) * offset_curr + alpha * offset_next

        scale_offset_curr = self.scale_offsets[frame_idx]
        scale_offset_next = self.scale_offsets[frame_idx + 1]
        scale_offset = (1 - alpha) * scale_offset_curr + alpha * scale_offset_next

        # Create result cloud
        result = self._clone_gaussians(self.base_gaussians)
        result._positions = nn.Parameter(self.base_gaussians._positions + offset)
        result._scales = nn.Parameter(self.base_gaussians._scales + scale_offset)

        return result

    def _get_velocity(self, t: float, dt: float) -> GaussianCloud:
        """Get Gaussians by integrating velocity."""
        # Start from base and integrate
        result = self._clone_gaussians(self.base_gaussians)

        # Get velocity at current positions and time
        n_steps = int(t)
        positions = self.base_gaussians._positions.clone()

        for step in range(n_steps):
            step_t = step * dt
            velocities = self._compute_velocity(positions, step_t)
            positions = positions + velocities * dt

        # Handle fractional time
        alpha = t - n_steps
        if alpha > 0:
            velocities = self._compute_velocity(positions, n_steps * dt)
            positions = positions + velocities * (alpha * dt)

        result._positions = nn.Parameter(positions)

        return result

    def _compute_velocity(
        self,
        positions: torch.Tensor,
        t: float
    ) -> torch.Tensor:
        """
        Compute velocity at given positions and time.

        Args:
            positions: (N, 3) positions
            t: Time

        Returns:
            (N, 3) velocities
        """
        # Use MLP for spatially-varying velocity
        time_tensor = torch.full(
            (positions.shape[0], 1), t, device=self.device
        )
        inputs = torch.cat([positions, time_tensor], dim=-1)

        mlp_velocity = self.velocity_mlp(inputs)

        # Combine with per-Gaussian learned velocity
        base_velocity = self.base_gaussians._velocities

        return mlp_velocity + base_velocity

    def get_velocity_field(
        self,
        points: torch.Tensor,
        t: float
    ) -> torch.Tensor:
        """
        Query velocity field at arbitrary points.

        Args:
            points: Query points (M, 3)
            t: Time

        Returns:
            Velocities at query points (M, 3)
        """
        if self.temporal_mode == "velocity":
            return self._compute_velocity(points, t)

        elif self.temporal_mode == "trajectory":
            # Interpolate from Gaussian velocities
            frame_idx = int(t)
            alpha = t - frame_idx
            frame_idx = max(0, min(frame_idx, self.n_timesteps - 2))

            # Compute velocities from position differences
            pos_curr = self.position_offsets[frame_idx]
            pos_next = self.position_offsets[frame_idx + 1]
            gaussian_velocities = pos_next - pos_curr  # Assumes dt=1

            # Interpolate to query points using Gaussian weights
            return self._interpolate_to_points(
                self.base_gaussians._positions + (1-alpha) * pos_curr + alpha * pos_next,
                gaussian_velocities,
                points
            )

        else:
            # Per-frame: compute finite differences
            if t < self.n_timesteps - 1:
                pos_curr = self.gaussians_per_frame[int(t)]._positions
                pos_next = self.gaussians_per_frame[int(t) + 1]._positions
                gaussian_velocities = pos_next - pos_curr

                return self._interpolate_to_points(pos_curr, gaussian_velocities, points)

            return torch.zeros_like(points)

    def _interpolate_to_points(
        self,
        gaussian_positions: torch.Tensor,
        gaussian_values: torch.Tensor,
        query_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate Gaussian values to query points using Gaussian weights.

        Args:
            gaussian_positions: (N, 3) Gaussian centers
            gaussian_values: (N, D) values at Gaussians
            query_points: (M, 3) query positions

        Returns:
            (M, D) interpolated values
        """
        # Compute distances
        dists = torch.cdist(query_points, gaussian_positions)  # (M, N)

        # Gaussian weights
        sigma = 0.1  # Interpolation bandwidth
        weights = torch.exp(-0.5 * (dists / sigma) ** 2)

        # Normalize weights
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Weighted sum
        return weights @ gaussian_values

    def compute_temporal_consistency_loss(self) -> torch.Tensor:
        """
        Compute loss for temporal consistency.

        Encourages smooth motion across time.
        """
        loss = torch.tensor(0.0, device=self.device)

        if self.temporal_mode == "per_frame":
            for t in range(self.n_timesteps - 1):
                pos_curr = self.gaussians_per_frame[t]._positions
                pos_next = self.gaussians_per_frame[t + 1]._positions
                vel = pos_next - pos_curr

                if t < self.n_timesteps - 2:
                    pos_future = self.gaussians_per_frame[t + 2]._positions
                    vel_next = pos_future - pos_next

                    # Velocity smoothness
                    loss = loss + ((vel_next - vel) ** 2).mean()

        elif self.temporal_mode == "trajectory":
            for t in range(self.n_timesteps - 2):
                vel = self.position_offsets[t + 1] - self.position_offsets[t]
                vel_next = self.position_offsets[t + 2] - self.position_offsets[t + 1]
                loss = loss + ((vel_next - vel) ** 2).mean()

        return loss

    def get_all_parameters(self) -> List[Dict]:
        """
        Get parameter groups for optimizer.

        Returns:
            List of parameter group dicts
        """
        groups = []

        if self.temporal_mode == "per_frame":
            for i, g in enumerate(self.gaussians_per_frame):
                groups.append({'params': [g._positions], 'name': f'positions_{i}'})
                groups.append({'params': [g._scales], 'name': f'scales_{i}'})
                groups.append({'params': [g._rotations], 'name': f'rotations_{i}'})
                groups.append({'params': [g._opacities], 'name': f'opacities_{i}'})
                groups.append({'params': [g._features_dc], 'name': f'features_dc_{i}'})
                groups.append({'params': [g._features_rest], 'name': f'features_rest_{i}'})

        else:
            g = self.base_gaussians
            groups.append({'params': [g._positions], 'name': 'positions'})
            groups.append({'params': [g._scales], 'name': 'scales'})
            groups.append({'params': [g._rotations], 'name': 'rotations'})
            groups.append({'params': [g._opacities], 'name': 'opacities'})
            groups.append({'params': [g._features_dc], 'name': 'features_dc'})
            groups.append({'params': [g._features_rest], 'name': 'features_rest'})

            if self.temporal_mode == "trajectory":
                groups.append({'params': [self.position_offsets], 'name': 'position_offsets'})
                groups.append({'params': [self.scale_offsets], 'name': 'scale_offsets'})

            elif self.temporal_mode == "velocity":
                groups.append({'params': [g._velocities], 'name': 'velocities'})
                groups.append({'params': self.velocity_mlp.parameters(), 'name': 'velocity_mlp'})

        return groups
