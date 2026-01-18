"""
Navier-Stokes equation constraints for physics-informed learning.

Implements PINN-style residual losses for enforcing incompressible
Navier-Stokes equations during optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class PhysicsResidual:
    """Residuals from Navier-Stokes equations."""
    momentum_x: torch.Tensor  # x-component residual
    momentum_y: torch.Tensor  # y-component residual
    momentum_z: torch.Tensor  # z-component residual
    continuity: torch.Tensor  # Divergence-free constraint

    @property
    def momentum(self) -> torch.Tensor:
        """Combined momentum residual magnitude."""
        return torch.sqrt(
            self.momentum_x ** 2 +
            self.momentum_y ** 2 +
            self.momentum_z ** 2
        )

    @property
    def total_mse(self) -> torch.Tensor:
        """Total mean squared error."""
        return (
            (self.momentum_x ** 2).mean() +
            (self.momentum_y ** 2).mean() +
            (self.momentum_z ** 2).mean() +
            (self.continuity ** 2).mean()
        )


class NavierStokesLoss(nn.Module):
    """
    Physics-Informed Neural Network (PINN) loss for Navier-Stokes equations.

    Enforces the incompressible Navier-Stokes equations:
        ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f
        ∇·u = 0

    Uses automatic differentiation to compute spatial and temporal derivatives.
    """

    def __init__(
        self,
        momentum_weight: float = 1.0,
        continuity_weight: float = 10.0,
        gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0),
        reference_density: float = 1000.0
    ):
        """
        Initialize Navier-Stokes loss.

        Args:
            momentum_weight: Weight for momentum equation residuals
            continuity_weight: Weight for continuity (divergence-free) constraint
            gravity: Gravitational acceleration vector
            reference_density: Reference density for normalization
        """
        super().__init__()

        self.momentum_weight = momentum_weight
        self.continuity_weight = continuity_weight
        self.register_buffer('gravity', torch.tensor(gravity))
        self.reference_density = reference_density

    def forward(
        self,
        velocity_fn,
        pressure_fn,
        density_fn,
        viscosity: torch.Tensor,
        sample_points: torch.Tensor,
        sample_times: torch.Tensor,
        external_forces: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Navier-Stokes residual loss.

        Args:
            velocity_fn: Function mapping (x, t) -> velocity (N, 3)
            pressure_fn: Function mapping (x, t) -> pressure (N,)
            density_fn: Function mapping (x, t) -> density (N,)
            viscosity: Kinematic viscosity (scalar or (N,))
            sample_points: Sample positions (N, 3), requires_grad=True
            sample_times: Sample times (N, 1), requires_grad=True
            external_forces: Optional external forces (N, 3)

        Returns:
            Dict with loss values
        """
        # Enable gradients for autodiff
        x = sample_points.requires_grad_(True)
        t = sample_times.requires_grad_(True)

        # Query fields
        u = velocity_fn(x, t)  # (N, 3)
        p = pressure_fn(x, t)  # (N,)
        rho = density_fn(x, t)  # (N,)

        # Compute derivatives using autodiff
        residual = self._compute_residual(
            u, p, rho, viscosity, x, t, external_forces
        )

        # Compute losses
        momentum_loss = (
            (residual.momentum_x ** 2).mean() +
            (residual.momentum_y ** 2).mean() +
            (residual.momentum_z ** 2).mean()
        )

        continuity_loss = (residual.continuity ** 2).mean()

        total_loss = (
            self.momentum_weight * momentum_loss +
            self.continuity_weight * continuity_loss
        )

        return {
            'total': total_loss,
            'momentum': momentum_loss,
            'continuity': continuity_loss,
            'residual': residual
        }

    def _compute_residual(
        self,
        u: torch.Tensor,
        p: torch.Tensor,
        rho: torch.Tensor,
        nu: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        f: Optional[torch.Tensor]
    ) -> PhysicsResidual:
        """
        Compute Navier-Stokes residuals using automatic differentiation.

        The momentum equation:
        ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f

        The continuity equation:
        ∇·u = 0
        """
        N = x.shape[0]

        # Velocity components
        ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]

        # Compute spatial gradients
        # du/dx, du/dy, du/dz for each component
        dux_dx, dux_dy, dux_dz = self._spatial_gradient(ux, x)
        duy_dx, duy_dy, duy_dz = self._spatial_gradient(uy, x)
        duz_dx, duz_dy, duz_dz = self._spatial_gradient(uz, x)

        # Pressure gradient
        dp_dx, dp_dy, dp_dz = self._spatial_gradient(p, x)

        # Temporal derivatives
        dux_dt = self._time_derivative(ux, t)
        duy_dt = self._time_derivative(uy, t)
        duz_dt = self._time_derivative(uz, t)

        # Laplacian (second spatial derivatives)
        laplacian_ux = self._laplacian(ux, x)
        laplacian_uy = self._laplacian(uy, x)
        laplacian_uz = self._laplacian(uz, x)

        # Convective term: (u·∇)u
        conv_x = ux * dux_dx + uy * dux_dy + uz * dux_dz
        conv_y = ux * duy_dx + uy * duy_dy + uz * duy_dz
        conv_z = ux * duz_dx + uy * duz_dy + uz * duz_dz

        # External forces (ensure gravity is on same device as input)
        if f is None:
            f = self.gravity.to(x.device).unsqueeze(0).expand(N, 3)
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]

        # Momentum residuals
        # R = ∂u/∂t + (u·∇)u + ∇p/ρ - ν∇²u - f
        R_x = dux_dt + conv_x + dp_dx / rho - nu * laplacian_ux - fx
        R_y = duy_dt + conv_y + dp_dy / rho - nu * laplacian_uy - fy
        R_z = duz_dt + conv_z + dp_dz / rho - nu * laplacian_uz - fz

        # Continuity residual
        # ∇·u = du_x/dx + du_y/dy + du_z/dz
        R_cont = dux_dx + duy_dy + duz_dz

        return PhysicsResidual(
            momentum_x=R_x,
            momentum_y=R_y,
            momentum_z=R_z,
            continuity=R_cont
        )

    def _spatial_gradient(
        self,
        scalar: torch.Tensor,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gradient of scalar w.r.t. spatial coordinates."""
        grad = torch.autograd.grad(
            scalar, x,
            grad_outputs=torch.ones_like(scalar),
            create_graph=True,
            retain_graph=True
        )[0]

        return grad[:, 0], grad[:, 1], grad[:, 2]

    def _time_derivative(
        self,
        scalar: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Compute time derivative of scalar."""
        grad = torch.autograd.grad(
            scalar, t,
            grad_outputs=torch.ones_like(scalar),
            create_graph=True,
            retain_graph=True
        )[0]

        return grad.squeeze(-1)

    def _laplacian(
        self,
        scalar: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Compute Laplacian (sum of second derivatives)."""
        # First derivatives
        grad = torch.autograd.grad(
            scalar, x,
            grad_outputs=torch.ones_like(scalar),
            create_graph=True,
            retain_graph=True
        )[0]

        # Second derivatives
        laplacian = torch.zeros_like(scalar)
        for i in range(3):
            d2f_dxi2 = torch.autograd.grad(
                grad[:, i], x,
                grad_outputs=torch.ones_like(grad[:, i]),
                create_graph=True,
                retain_graph=True
            )[0][:, i]
            laplacian = laplacian + d2f_dxi2

        return laplacian


class GridBasedNavierStokesLoss(nn.Module):
    """
    Navier-Stokes loss using finite differences on a grid.

    More efficient than autodiff for grid-based representations.
    """

    def __init__(
        self,
        momentum_weight: float = 1.0,
        continuity_weight: float = 10.0,
        gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)
    ):
        super().__init__()

        self.momentum_weight = momentum_weight
        self.continuity_weight = continuity_weight
        self.register_buffer('gravity', torch.tensor(gravity))

    def forward(
        self,
        fluid_fields,  # FluidFields
        t: int,
        dt: float
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Navier-Stokes residual on grid.

        Args:
            fluid_fields: FluidFields object
            t: Current timestep
            dt: Time step size

        Returns:
            Dict with loss values
        """
        u = fluid_fields.velocity.data[t]  # (X, Y, Z, 3)
        p = fluid_fields.pressure.data[t]  # (X, Y, Z)
        rho = fluid_fields.density.data[t]  # (X, Y, Z)
        nu = fluid_fields.kinematic_viscosity
        h = fluid_fields.velocity.spacing

        # Velocity components
        ux = u[..., 0]
        uy = u[..., 1]
        uz = u[..., 2]

        # Compute derivatives using finite differences
        # Spatial gradients
        dux_dx = self._central_diff(ux, 0, h[0])
        dux_dy = self._central_diff(ux, 1, h[1])
        dux_dz = self._central_diff(ux, 2, h[2])

        duy_dx = self._central_diff(uy, 0, h[0])
        duy_dy = self._central_diff(uy, 1, h[1])
        duy_dz = self._central_diff(uy, 2, h[2])

        duz_dx = self._central_diff(uz, 0, h[0])
        duz_dy = self._central_diff(uz, 1, h[1])
        duz_dz = self._central_diff(uz, 2, h[2])

        dp_dx = self._central_diff(p, 0, h[0])
        dp_dy = self._central_diff(p, 1, h[1])
        dp_dz = self._central_diff(p, 2, h[2])

        # Laplacians
        laplacian_ux = self._laplacian(ux, h)
        laplacian_uy = self._laplacian(uy, h)
        laplacian_uz = self._laplacian(uz, h)

        # Convective terms
        conv_x = ux * dux_dx + uy * dux_dy + uz * dux_dz
        conv_y = ux * duy_dx + uy * duy_dy + uz * duy_dz
        conv_z = ux * duz_dx + uy * duz_dy + uz * duz_dz

        # Time derivative (if multiple timesteps available)
        if t > 0:
            u_prev = fluid_fields.velocity.data[t - 1]
            dux_dt = (ux - u_prev[..., 0]) / dt
            duy_dt = (uy - u_prev[..., 1]) / dt
            duz_dt = (uz - u_prev[..., 2]) / dt
        else:
            dux_dt = torch.zeros_like(ux)
            duy_dt = torch.zeros_like(uy)
            duz_dt = torch.zeros_like(uz)

        # External forces (gravity)
        fx, fy, fz = self.gravity

        # Momentum residuals
        R_x = dux_dt + conv_x + dp_dx / (rho + 1e-8) - nu * laplacian_ux - fx
        R_y = duy_dt + conv_y + dp_dy / (rho + 1e-8) - nu * laplacian_uy - fy
        R_z = duz_dt + conv_z + dp_dz / (rho + 1e-8) - nu * laplacian_uz - fz

        # Continuity residual
        R_cont = dux_dx + duy_dy + duz_dz

        # Losses
        momentum_loss = (R_x ** 2 + R_y ** 2 + R_z ** 2).mean()
        continuity_loss = (R_cont ** 2).mean()

        total = (
            self.momentum_weight * momentum_loss +
            self.continuity_weight * continuity_loss
        )

        return {
            'total': total,
            'momentum': momentum_loss,
            'continuity': continuity_loss
        }

    def _central_diff(
        self,
        field: torch.Tensor,
        dim: int,
        h: torch.Tensor
    ) -> torch.Tensor:
        """Central difference derivative."""
        return (torch.roll(field, -1, dims=dim) - torch.roll(field, 1, dims=dim)) / (2 * h)

    def _laplacian(
        self,
        field: torch.Tensor,
        h: torch.Tensor
    ) -> torch.Tensor:
        """Laplacian using central differences."""
        lap = torch.zeros_like(field)
        for dim in range(3):
            d2f = (
                torch.roll(field, -1, dims=dim) -
                2 * field +
                torch.roll(field, 1, dims=dim)
            ) / (h[dim] ** 2)
            lap = lap + d2f
        return lap


def pressure_projection(
    velocity: torch.Tensor,
    density: torch.Tensor,
    spacing: torch.Tensor,
    n_iterations: int = 50
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project velocity field to be divergence-free using pressure solve.

    Uses Jacobi iteration to solve the pressure Poisson equation:
    ∇²p = ρ/dt * ∇·u

    Args:
        velocity: Velocity field (X, Y, Z, 3)
        density: Density field (X, Y, Z)
        spacing: Grid spacing (3,)
        n_iterations: Number of Jacobi iterations

    Returns:
        Tuple of (corrected_velocity, pressure)
    """
    device = velocity.device
    h = spacing

    # Compute divergence
    ux, uy, uz = velocity[..., 0], velocity[..., 1], velocity[..., 2]
    div = (
        (torch.roll(ux, -1, dims=0) - torch.roll(ux, 1, dims=0)) / (2 * h[0]) +
        (torch.roll(uy, -1, dims=1) - torch.roll(uy, 1, dims=1)) / (2 * h[1]) +
        (torch.roll(uz, -1, dims=2) - torch.roll(uz, 1, dims=2)) / (2 * h[2])
    )

    # RHS: ρ * ∇·u (assuming dt=1 for simplicity)
    rhs = density * div

    # Jacobi iteration for ∇²p = rhs
    p = torch.zeros_like(div)
    scale = 2 * (1 / h[0]**2 + 1 / h[1]**2 + 1 / h[2]**2)

    for _ in range(n_iterations):
        p_new = (
            (torch.roll(p, 1, dims=0) + torch.roll(p, -1, dims=0)) / h[0]**2 +
            (torch.roll(p, 1, dims=1) + torch.roll(p, -1, dims=1)) / h[1]**2 +
            (torch.roll(p, 1, dims=2) + torch.roll(p, -1, dims=2)) / h[2]**2 -
            rhs
        ) / scale
        p = p_new

    # Correct velocity: u_new = u - ∇p/ρ
    dp_dx = (torch.roll(p, -1, dims=0) - torch.roll(p, 1, dims=0)) / (2 * h[0])
    dp_dy = (torch.roll(p, -1, dims=1) - torch.roll(p, 1, dims=1)) / (2 * h[1])
    dp_dz = (torch.roll(p, -1, dims=2) - torch.roll(p, 1, dims=2)) / (2 * h[2])

    rho_safe = density + 1e-8
    u_corrected = velocity.clone()
    u_corrected[..., 0] = ux - dp_dx / rho_safe
    u_corrected[..., 1] = uy - dp_dy / rho_safe
    u_corrected[..., 2] = uz - dp_dz / rho_safe

    return u_corrected, p
