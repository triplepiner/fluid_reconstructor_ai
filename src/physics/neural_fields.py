"""
Neural implicit field representations for fluid properties.

MLPs with positional encoding for continuous field representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """
    Fourier feature positional encoding for neural fields.

    Maps low-dimensional inputs to higher dimensions using sinusoidal functions:
    γ(p) = [sin(2^0 π p), cos(2^0 π p), ..., sin(2^{L-1} π p), cos(2^{L-1} π p)]
    """

    def __init__(
        self,
        input_dim: int,
        num_frequencies: int = 10,
        include_input: bool = True
    ):
        """
        Initialize positional encoding.

        Args:
            input_dim: Input dimensionality
            num_frequencies: Number of frequency bands (L)
            include_input: Whether to include original input
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # Frequency bands: 2^0, 2^1, ..., 2^{L-1}
        # Note: buffer will be moved to correct device when module.to(device) is called
        freqs = 2.0 ** torch.arange(num_frequencies).float() * math.pi
        self.register_buffer('frequencies', freqs)

    @property
    def output_dim(self) -> int:
        """Output dimensionality after encoding."""
        dim = self.input_dim * self.num_frequencies * 2
        if self.include_input:
            dim += self.input_dim
        return dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding.

        Args:
            x: Input tensor (..., input_dim)

        Returns:
            Encoded tensor (..., output_dim)
        """
        # x: (..., D)
        # Expand to (..., D, L) - ensure frequencies are on same device as input
        x_freq = x.unsqueeze(-1) * self.frequencies.to(x.device)

        # Apply sin and cos
        sin_x = torch.sin(x_freq)
        cos_x = torch.cos(x_freq)

        # Concatenate: (..., D*L*2)
        encoded = torch.cat([sin_x, cos_x], dim=-1).flatten(-2)

        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)

        return encoded


class NeuralField(nn.Module):
    """
    Base class for neural implicit fields.
    """

    def __init__(
        self,
        input_dim: int = 4,  # x, y, z, t
        output_dim: int = 1,
        hidden_dims: List[int] = [256, 256, 256, 256],
        num_frequencies: int = 10,
        activation: str = "relu",
        output_activation: Optional[str] = None
    ):
        """
        Initialize neural field.

        Args:
            input_dim: Input dimensionality (spatial + temporal)
            output_dim: Output dimensionality
            hidden_dims: Hidden layer dimensions
            num_frequencies: Positional encoding frequencies
            activation: Activation function
            output_activation: Optional output activation
        """
        super().__init__()

        self.encoding = PositionalEncoding(input_dim, num_frequencies)

        layers = []
        in_dim = self.encoding.output_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "silu":
                layers.append(nn.SiLU(inplace=True))
            elif activation == "tanh":
                layers.append(nn.Tanh())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        if output_activation == "softplus":
            layers.append(nn.Softplus())
        elif output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation == "tanh":
            layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Query field at positions and time.

        Args:
            x: Positions (N, 3)
            t: Time (N, 1) or scalar

        Returns:
            Field values (N, output_dim)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)

        inputs = torch.cat([x, t], dim=-1)
        encoded = self.encoding(inputs)
        return self.network(encoded)


class NeuralVelocityField(NeuralField):
    """
    Neural field for velocity.
    """

    def __init__(
        self,
        hidden_dims: List[int] = [256, 256, 256, 256],
        num_frequencies: int = 10
    ):
        super().__init__(
            input_dim=4,  # x, y, z, t
            output_dim=3,  # u, v, w
            hidden_dims=hidden_dims,
            num_frequencies=num_frequencies,
            activation="silu",
            output_activation=None  # Velocities can be positive or negative
        )


class NeuralPressureField(NeuralField):
    """
    Neural field for pressure.
    """

    def __init__(
        self,
        hidden_dims: List[int] = [256, 256, 256, 256],
        num_frequencies: int = 10
    ):
        super().__init__(
            input_dim=4,
            output_dim=1,
            hidden_dims=hidden_dims,
            num_frequencies=num_frequencies,
            activation="silu",
            output_activation=None  # Pressure can be relative (positive or negative)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Query pressure field."""
        result = super().forward(x, t)
        return result.squeeze(-1)  # Return (N,) instead of (N, 1)


class NeuralDensityField(NeuralField):
    """
    Neural field for density (always positive).
    """

    def __init__(
        self,
        hidden_dims: List[int] = [256, 256, 256, 256],
        num_frequencies: int = 10,
        base_density: float = 1000.0
    ):
        super().__init__(
            input_dim=4,
            output_dim=1,
            hidden_dims=hidden_dims,
            num_frequencies=num_frequencies,
            activation="silu",
            output_activation="softplus"
        )
        self.base_density = base_density

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Query density field."""
        result = super().forward(x, t)
        # Scale to physically meaningful range
        return result.squeeze(-1) * self.base_density


class NeuralFluidFields(nn.Module):
    """
    Combined neural fields for velocity, pressure, and density.
    """

    def __init__(
        self,
        hidden_dims: List[int] = [256, 256, 256, 256],
        num_frequencies: int = 10,
        shared_encoder: bool = False
    ):
        """
        Initialize neural fluid fields.

        Args:
            hidden_dims: Hidden layer dimensions
            num_frequencies: Positional encoding frequencies
            shared_encoder: Whether to share encoder between fields
        """
        super().__init__()

        self.velocity = NeuralVelocityField(hidden_dims, num_frequencies)
        self.pressure = NeuralPressureField(hidden_dims, num_frequencies)
        self.density = NeuralDensityField(hidden_dims, num_frequencies)

        # Learnable viscosity (device will be set when module.to(device) is called)
        self._viscosity = nn.Parameter(torch.tensor(-3.0))

    @property
    def viscosity(self) -> torch.Tensor:
        """Get positive viscosity."""
        return F.softplus(self._viscosity)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Query all fields.

        Args:
            x: Positions (N, 3)
            t: Time (N, 1) or scalar

        Returns:
            Tuple of (velocity, pressure, density)
        """
        vel = self.velocity(x, t)
        pres = self.pressure(x, t)
        dens = self.density(x, t)
        return vel, pres, dens

    def compute_physics_loss(
        self,
        sample_points: torch.Tensor,
        sample_times: torch.Tensor,
        gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)
    ) -> torch.Tensor:
        """
        Compute Navier-Stokes physics loss.

        Uses autodiff to compute derivatives.
        """
        from .navier_stokes import NavierStokesLoss

        ns_loss = NavierStokesLoss(gravity=gravity)

        result = ns_loss(
            velocity_fn=self.velocity,
            pressure_fn=self.pressure,
            density_fn=self.density,
            viscosity=self.viscosity,
            sample_points=sample_points,
            sample_times=sample_times
        )

        return result['total']


class HybridField(nn.Module):
    """
    Hybrid field combining grid and neural representations.

    Uses coarse grid for structure and neural network for details.
    """

    def __init__(
        self,
        grid_resolution: Tuple[int, int, int],
        bounds,  # FieldBounds
        hidden_dims: List[int] = [128, 128],
        num_frequencies: int = 6
    ):
        """
        Initialize hybrid field.

        Args:
            grid_resolution: Coarse grid resolution
            bounds: Field domain bounds
            hidden_dims: Neural network hidden dims
            num_frequencies: Positional encoding frequencies
        """
        super().__init__()

        from .fields import VectorField

        self.coarse_grid = VectorField(grid_resolution, bounds)
        self.detail_net = NeuralVelocityField(hidden_dims, num_frequencies)
        self.bounds = bounds

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Query hybrid field.

        Args:
            x: Positions (N, 3)
            t: Time

        Returns:
            Velocities (N, 3)
        """
        # Coarse grid interpolation
        t_idx = int(t) if isinstance(t, (int, float)) else int(t.item())
        coarse = self.coarse_grid.sample(x, t_idx)

        # Neural network detail
        detail = self.detail_net(x, t)

        return coarse + detail


def sample_random_points(
    bounds,  # FieldBounds
    n_points: int,
    n_timesteps: int,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample random points in space-time domain.

    Args:
        bounds: Spatial domain bounds
        n_points: Number of points to sample
        n_timesteps: Number of timesteps
        device: Device

    Returns:
        Tuple of (positions (N, 3), times (N, 1))
    """
    # Random positions within bounds (ensure bounds tensors are on correct device)
    positions = torch.rand(n_points, 3, device=device)
    size = bounds.size.to(device) if hasattr(bounds.size, 'to') else torch.tensor(bounds.size, device=device)
    min_corner = bounds.min_corner.to(device) if hasattr(bounds.min_corner, 'to') else torch.tensor(bounds.min_corner, device=device)
    positions = positions * size + min_corner

    # Random times
    times = torch.rand(n_points, 1, device=device) * n_timesteps

    return positions, times


def sample_near_surface(
    gaussian_positions: torch.Tensor,
    n_points: int,
    std: float = 0.1
) -> torch.Tensor:
    """
    Sample points near Gaussian positions (fluid surface).

    Args:
        gaussian_positions: Gaussian centers (N, 3)
        n_points: Number of points to sample
        std: Standard deviation for sampling

    Returns:
        Sampled points (n_points, 3)
    """
    # Randomly select Gaussians
    n_gaussians = gaussian_positions.shape[0]
    indices = torch.randint(0, n_gaussians, (n_points,), device=gaussian_positions.device)

    # Sample around selected Gaussians
    centers = gaussian_positions[indices]
    offsets = torch.randn(n_points, 3, device=gaussian_positions.device) * std

    return centers + offsets
