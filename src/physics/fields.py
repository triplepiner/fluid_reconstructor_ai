"""
Field representations for fluid properties.

Supports both grid-based and neural implicit representations
for velocity, pressure, and density fields.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class FieldBounds:
    """Axis-aligned bounding box for field domain."""
    min_corner: torch.Tensor  # (3,) minimum coordinates
    max_corner: torch.Tensor  # (3,) maximum coordinates

    @property
    def center(self) -> torch.Tensor:
        return (self.min_corner + self.max_corner) / 2

    @property
    def size(self) -> torch.Tensor:
        return self.max_corner - self.min_corner

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        """Check if points are inside bounds."""
        inside = (points >= self.min_corner.to(points.device)) & (points <= self.max_corner.to(points.device))
        return inside.all(dim=-1)

    def normalize(self, points: torch.Tensor) -> torch.Tensor:
        """Normalize points to [-1, 1] within bounds."""
        return 2.0 * (points - self.min_corner.to(points.device)) / (self.size.to(points.device) + 1e-8) - 1.0

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized coordinates back to world coordinates."""
        return (normalized + 1.0) / 2.0 * self.size.to(normalized.device) + self.min_corner.to(normalized.device)


class VectorField(nn.Module):
    """
    3D vector field on a regular grid.
    """

    def __init__(
        self,
        resolution: Tuple[int, int, int],
        bounds: FieldBounds,
        n_timesteps: int = 1,
        device: str = "cuda"
    ):
        """
        Initialize vector field.

        Args:
            resolution: (X, Y, Z) grid resolution
            bounds: Field domain bounds
            n_timesteps: Number of timesteps (for time-varying fields)
            device: Device for tensors
        """
        super().__init__()

        self.resolution = resolution
        self.bounds = bounds
        self.n_timesteps = n_timesteps
        self.device = device

        # Grid data: (T, X, Y, Z, 3)
        self.data = nn.Parameter(
            torch.zeros(n_timesteps, *resolution, 3, device=device)
        )

    @property
    def spacing(self) -> torch.Tensor:
        """Grid cell size."""
        return self.bounds.size / torch.tensor(self.resolution, device=self.device)

    def sample(
        self,
        points: torch.Tensor,
        t: int = 0,
        interpolation: str = "trilinear"
    ) -> torch.Tensor:
        """
        Sample field at arbitrary points.

        Args:
            points: Query points (N, 3)
            t: Timestep index
            interpolation: Interpolation method ('trilinear' or 'nearest')

        Returns:
            Field values at query points (N, 3)
        """
        # Normalize to grid coordinates [-1, 1]
        normalized = self.bounds.normalize(points)

        # Reshape for grid_sample: (1, 1, 1, N, 3)
        grid = normalized.view(1, 1, 1, -1, 3)

        # Field data: (1, 3, X, Y, Z)
        field_data = self.data[t].permute(3, 0, 1, 2).unsqueeze(0)

        # Sample
        mode = 'bilinear' if interpolation == 'trilinear' else 'nearest'
        sampled = F.grid_sample(
            field_data, grid,
            mode=mode,
            padding_mode='border',
            align_corners=True
        )

        # Reshape to (N, 3)
        return sampled.view(3, -1).T

    def set_from_points(
        self,
        points: torch.Tensor,
        values: torch.Tensor,
        t: int = 0
    ):
        """
        Set field values from scattered point data.

        Args:
            points: Point positions (N, 3)
            values: Values at points (N, 3)
            t: Timestep index
        """
        # Grid coordinates
        normalized = self.bounds.normalize(points)
        grid_coords = (normalized + 1) / 2 * torch.tensor(
            self.resolution, device=self.device
        ).float()

        # Scatter to grid using weighted averaging
        grid = torch.zeros_like(self.data[t])
        weights = torch.zeros(*self.resolution, 1, device=self.device)

        for i, (coord, val) in enumerate(zip(grid_coords, values)):
            # Get integer coordinates
            x, y, z = coord.long().clamp(
                torch.zeros(3, dtype=torch.long, device=self.device),
                torch.tensor(self.resolution, device=self.device) - 1
            )

            grid[x, y, z] += val
            weights[x, y, z, 0] += 1

        # Normalize
        self.data.data[t] = grid / (weights + 1e-8)

    def compute_gradient(self, t: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute spatial gradient using central differences.

        Returns:
            Tuple of (du/dx, du/dy, du/dz) each of shape (X, Y, Z, 3)
        """
        h = self.spacing
        field = self.data[t]

        # Central differences
        dudx = (torch.roll(field, -1, dims=0) - torch.roll(field, 1, dims=0)) / (2 * h[0])
        dudy = (torch.roll(field, -1, dims=1) - torch.roll(field, 1, dims=1)) / (2 * h[1])
        dudz = (torch.roll(field, -1, dims=2) - torch.roll(field, 1, dims=2)) / (2 * h[2])

        return dudx, dudy, dudz

    def compute_divergence(self, t: int = 0) -> torch.Tensor:
        """
        Compute divergence of vector field.

        Returns:
            Divergence field (X, Y, Z)
        """
        dudx, dudy, dudz = self.compute_gradient(t)

        # Divergence = du_x/dx + du_y/dy + du_z/dz
        div = dudx[..., 0] + dudy[..., 1] + dudz[..., 2]

        return div

    def compute_curl(self, t: int = 0) -> torch.Tensor:
        """
        Compute curl of vector field.

        Returns:
            Curl field (X, Y, Z, 3)
        """
        dudx, dudy, dudz = self.compute_gradient(t)

        # Curl components
        curl_x = dudy[..., 2] - dudz[..., 1]
        curl_y = dudz[..., 0] - dudx[..., 2]
        curl_z = dudx[..., 1] - dudy[..., 0]

        return torch.stack([curl_x, curl_y, curl_z], dim=-1)


class ScalarField(nn.Module):
    """
    3D scalar field on a regular grid.
    """

    def __init__(
        self,
        resolution: Tuple[int, int, int],
        bounds: FieldBounds,
        n_timesteps: int = 1,
        device: str = "cuda"
    ):
        super().__init__()

        self.resolution = resolution
        self.bounds = bounds
        self.n_timesteps = n_timesteps
        self.device = device

        # Grid data: (T, X, Y, Z)
        self.data = nn.Parameter(
            torch.zeros(n_timesteps, *resolution, device=device)
        )

    @property
    def spacing(self) -> torch.Tensor:
        return self.bounds.size / torch.tensor(self.resolution, device=self.device)

    def sample(
        self,
        points: torch.Tensor,
        t: int = 0
    ) -> torch.Tensor:
        """Sample field at arbitrary points."""
        normalized = self.bounds.normalize(points)
        grid = normalized.view(1, 1, 1, -1, 3)
        field_data = self.data[t].unsqueeze(0).unsqueeze(0)

        sampled = F.grid_sample(
            field_data, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return sampled.view(-1)

    def compute_gradient(self, t: int = 0) -> torch.Tensor:
        """
        Compute gradient of scalar field.

        Returns:
            Gradient field (X, Y, Z, 3)
        """
        h = self.spacing
        field = self.data[t]

        dfdx = (torch.roll(field, -1, dims=0) - torch.roll(field, 1, dims=0)) / (2 * h[0])
        dfdy = (torch.roll(field, -1, dims=1) - torch.roll(field, 1, dims=1)) / (2 * h[1])
        dfdz = (torch.roll(field, -1, dims=2) - torch.roll(field, 1, dims=2)) / (2 * h[2])

        return torch.stack([dfdx, dfdy, dfdz], dim=-1)

    def compute_laplacian(self, t: int = 0) -> torch.Tensor:
        """
        Compute Laplacian of scalar field.

        Returns:
            Laplacian field (X, Y, Z)
        """
        h = self.spacing
        field = self.data[t]

        d2fdx2 = (torch.roll(field, -1, dims=0) - 2*field + torch.roll(field, 1, dims=0)) / (h[0]**2)
        d2fdy2 = (torch.roll(field, -1, dims=1) - 2*field + torch.roll(field, 1, dims=1)) / (h[1]**2)
        d2fdz2 = (torch.roll(field, -1, dims=2) - 2*field + torch.roll(field, 1, dims=2)) / (h[2]**2)

        return d2fdx2 + d2fdy2 + d2fdz2


class FluidFields(nn.Module):
    """
    Complete fluid state with velocity, pressure, and density fields.
    """

    def __init__(
        self,
        resolution: Tuple[int, int, int],
        bounds: FieldBounds,
        n_timesteps: int = 1,
        device: str = "cuda"
    ):
        """
        Initialize fluid fields.

        Args:
            resolution: Grid resolution
            bounds: Domain bounds
            n_timesteps: Number of timesteps
            device: Device
        """
        super().__init__()

        self.velocity = VectorField(resolution, bounds, n_timesteps, device)
        self.pressure = ScalarField(resolution, bounds, n_timesteps, device)
        self.density = ScalarField(resolution, bounds, n_timesteps, device)

        # Learnable viscosity (can be scalar or spatially varying)
        self.viscosity = nn.Parameter(torch.tensor(0.001, device=device))

        self.bounds = bounds
        self.resolution = resolution
        self.device = device

    @property
    def kinematic_viscosity(self) -> torch.Tensor:
        """Get positive kinematic viscosity."""
        return F.softplus(self.viscosity)

    def sample_all(
        self,
        points: torch.Tensor,
        t: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample all fields at given points.

        Returns:
            Tuple of (velocity, pressure, density)
        """
        vel = self.velocity.sample(points, t)
        pres = self.pressure.sample(points, t)
        dens = self.density.sample(points, t)

        return vel, pres, dens

    def initialize_from_gaussians(
        self,
        gaussians,  # DynamicGaussianCloud
        t: int = 0
    ):
        """
        Initialize fields from Gaussian representation.

        Args:
            gaussians: Dynamic Gaussian cloud
            t: Timestep
        """
        # Get Gaussians at this timestep
        g = gaussians.get_gaussians_at_time(t)

        positions = g.positions
        velocities = g.velocities
        opacities = g.opacities.squeeze(-1)

        # Set velocity field
        self.velocity.set_from_points(positions, velocities, t)

        # Set density from opacities
        # Density proportional to Gaussian opacity
        density_grid = torch.zeros(*self.resolution, device=self.device)
        normalized = self.bounds.normalize(positions)
        grid_coords = ((normalized + 1) / 2 * torch.tensor(
            self.resolution, device=self.device
        ).float()).long()

        for coord, alpha in zip(grid_coords, opacities):
            x, y, z = coord.clamp(
                torch.zeros(3, dtype=torch.long, device=self.device),
                torch.tensor(self.resolution, device=self.device) - 1
            )
            density_grid[x, y, z] += alpha

        self.density.data.data[t] = density_grid

    def compute_kinetic_energy(self, t: int = 0) -> torch.Tensor:
        """Compute total kinetic energy."""
        vel = self.velocity.data[t]
        rho = self.density.data[t]

        # KE = 0.5 * rho * |u|^2
        speed_sq = (vel ** 2).sum(dim=-1)
        ke = 0.5 * rho * speed_sq

        # Integrate over volume
        cell_volume = self.velocity.spacing.prod()
        return ke.sum() * cell_volume

    def compute_enstrophy(self, t: int = 0) -> torch.Tensor:
        """Compute enstrophy (integral of vorticity squared)."""
        vorticity = self.velocity.compute_curl(t)
        vort_sq = (vorticity ** 2).sum(dim=-1)

        cell_volume = self.velocity.spacing.prod()
        return 0.5 * vort_sq.sum() * cell_volume
