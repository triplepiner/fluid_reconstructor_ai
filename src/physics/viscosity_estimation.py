"""
Viscosity estimation from observed fluid motion.

Multiple methods for estimating dynamic and kinematic viscosity
from velocity fields and their temporal evolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ViscosityEstimate:
    """Estimated viscosity with confidence."""
    kinematic_viscosity: float  # ν (m²/s)
    dynamic_viscosity: float  # μ (Pa·s)
    confidence: float  # Estimation confidence (0-1)
    method: str  # Method used for estimation


class ViscosityEstimator:
    """
    Estimate fluid viscosity from observed motion.
    """

    # Reference viscosities for common fluids (Pa·s)
    REFERENCE_VISCOSITIES = {
        'water': 0.001,  # at 20°C
        'honey': 2.0,
        'oil': 0.1,
        'air': 1.8e-5,
        'glycerin': 1.5,
        'milk': 0.003
    }

    def __init__(
        self,
        reference_density: float = 1000.0,
        min_viscosity: float = 1e-6,
        max_viscosity: float = 10.0
    ):
        """
        Initialize viscosity estimator.

        Args:
            reference_density: Reference density (kg/m³)
            min_viscosity: Minimum valid viscosity
            max_viscosity: Maximum valid viscosity
        """
        self.reference_density = reference_density
        self.min_viscosity = min_viscosity
        self.max_viscosity = max_viscosity

    def estimate_from_velocity_diffusion(
        self,
        velocity_t0: torch.Tensor,
        velocity_t1: torch.Tensor,
        dt: float,
        spacing: torch.Tensor
    ) -> ViscosityEstimate:
        """
        Estimate viscosity from velocity diffusion rate.

        In viscous flow, velocity gradients decay according to:
        ∂u/∂t = ν ∇²u

        Args:
            velocity_t0: Velocity at time t (X, Y, Z, 3)
            velocity_t1: Velocity at time t+dt (X, Y, Z, 3)
            dt: Time step
            spacing: Grid spacing (3,)

        Returns:
            ViscosityEstimate
        """
        # Compute Laplacian of velocity at t0
        laplacian = self._compute_laplacian(velocity_t0, spacing)

        # Compute time derivative
        dudt = (velocity_t1 - velocity_t0) / dt

        # Estimate ν from: dudt ≈ ν * laplacian
        # Using least squares fit

        # Flatten and solve
        laplacian_flat = laplacian.flatten()
        dudt_flat = dudt.flatten()

        # Avoid division by zero
        valid_mask = laplacian_flat.abs() > 1e-8
        if valid_mask.sum() < 100:
            return ViscosityEstimate(
                kinematic_viscosity=0.001,
                dynamic_viscosity=self.reference_density * 0.001,
                confidence=0.0,
                method='diffusion'
            )

        laplacian_valid = laplacian_flat[valid_mask]
        dudt_valid = dudt_flat[valid_mask]

        # Least squares estimate
        nu_estimate = (dudt_valid * laplacian_valid).sum() / (laplacian_valid ** 2).sum()
        nu_estimate = nu_estimate.item()

        # Clamp to valid range
        nu_clamped = max(self.min_viscosity, min(self.max_viscosity, abs(nu_estimate)))

        # Compute confidence based on fit quality
        residual = dudt_valid - nu_clamped * laplacian_valid
        fit_error = (residual ** 2).mean().item()
        signal_energy = (dudt_valid ** 2).mean().item()
        r_squared = 1 - fit_error / (signal_energy + 1e-8)
        confidence = max(0, r_squared)

        return ViscosityEstimate(
            kinematic_viscosity=nu_clamped,
            dynamic_viscosity=self.reference_density * nu_clamped,
            confidence=confidence,
            method='diffusion'
        )

    def estimate_from_strain_rate(
        self,
        velocity: torch.Tensor,
        spacing: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute strain rate tensor and estimate effective viscosity.

        For Newtonian fluids: τ = 2μS
        where S is the strain rate tensor.

        Args:
            velocity: Velocity field (X, Y, Z, 3)
            spacing: Grid spacing (3,)

        Returns:
            Tuple of (strain_rate_tensor, estimated_viscosity)
        """
        # Compute velocity gradients
        du_dx = self._central_diff(velocity[..., 0], 0, spacing[0])
        du_dy = self._central_diff(velocity[..., 0], 1, spacing[1])
        du_dz = self._central_diff(velocity[..., 0], 2, spacing[2])

        dv_dx = self._central_diff(velocity[..., 1], 0, spacing[0])
        dv_dy = self._central_diff(velocity[..., 1], 1, spacing[1])
        dv_dz = self._central_diff(velocity[..., 1], 2, spacing[2])

        dw_dx = self._central_diff(velocity[..., 2], 0, spacing[0])
        dw_dy = self._central_diff(velocity[..., 2], 1, spacing[1])
        dw_dz = self._central_diff(velocity[..., 2], 2, spacing[2])

        # Strain rate tensor: S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
        S_xx = du_dx
        S_yy = dv_dy
        S_zz = dw_dz
        S_xy = 0.5 * (du_dy + dv_dx)
        S_xz = 0.5 * (du_dz + dw_dx)
        S_yz = 0.5 * (dv_dz + dw_dy)

        # Strain rate magnitude: sqrt(2 * S_ij * S_ij)
        strain_rate_sq = (
            2 * (S_xx**2 + S_yy**2 + S_zz**2 +
                 2*S_xy**2 + 2*S_xz**2 + 2*S_yz**2)
        )
        strain_rate = torch.sqrt(strain_rate_sq + 1e-8)

        # Mean strain rate
        mean_strain_rate = strain_rate.mean().item()

        # Stack into tensor
        S = torch.stack([
            torch.stack([S_xx, S_xy, S_xz], dim=-1),
            torch.stack([S_xy, S_yy, S_yz], dim=-1),
            torch.stack([S_xz, S_yz, S_zz], dim=-1)
        ], dim=-2)

        return S, mean_strain_rate

    def estimate_from_energy_dissipation(
        self,
        velocity_t0: torch.Tensor,
        velocity_t1: torch.Tensor,
        density: torch.Tensor,
        dt: float,
        spacing: torch.Tensor
    ) -> ViscosityEstimate:
        """
        Estimate viscosity from kinetic energy dissipation rate.

        Viscosity causes kinetic energy dissipation:
        dE/dt = -2μ ∫ S_ij S_ij dV

        Args:
            velocity_t0: Velocity at time t
            velocity_t1: Velocity at time t+dt
            density: Density field
            dt: Time step
            spacing: Grid spacing

        Returns:
            ViscosityEstimate
        """
        cell_volume = spacing.prod()

        # Kinetic energy: E = 0.5 * ∫ ρ |u|² dV
        ke_t0 = 0.5 * (density * (velocity_t0 ** 2).sum(dim=-1)).sum() * cell_volume
        ke_t1 = 0.5 * (density * (velocity_t1 ** 2).sum(dim=-1)).sum() * cell_volume

        # Energy dissipation rate
        dE_dt = (ke_t1 - ke_t0) / dt

        # Strain rate tensor
        S_t0, _ = self.estimate_from_strain_rate(velocity_t0, spacing)
        S_t1, _ = self.estimate_from_strain_rate(velocity_t1, spacing)
        S_avg = 0.5 * (S_t0 + S_t1)

        # S:S = S_ij * S_ij
        S_squared = (S_avg ** 2).sum(dim=(-1, -2))
        dissipation_integral = S_squared.sum() * cell_volume

        # Estimate: dE/dt = -2μ * ∫ S:S dV
        if dissipation_integral.abs() > 1e-8:
            mu_estimate = -dE_dt / (2 * dissipation_integral)
            mu_estimate = mu_estimate.item()
        else:
            mu_estimate = 0.001

        # Clamp and convert
        mu_clamped = max(self.min_viscosity * self.reference_density,
                        min(self.max_viscosity * self.reference_density, abs(mu_estimate)))
        nu_estimate = mu_clamped / self.reference_density

        # Confidence based on energy change significance
        relative_change = abs((ke_t1 - ke_t0) / (ke_t0 + 1e-8)).item()
        confidence = min(1.0, relative_change * 10)  # Higher change = more signal

        return ViscosityEstimate(
            kinematic_viscosity=nu_estimate,
            dynamic_viscosity=mu_clamped,
            confidence=confidence,
            method='energy_dissipation'
        )

    def estimate_combined(
        self,
        velocity_t0: torch.Tensor,
        velocity_t1: torch.Tensor,
        density: torch.Tensor,
        dt: float,
        spacing: torch.Tensor
    ) -> ViscosityEstimate:
        """
        Combine multiple estimation methods with weighted average.

        Args:
            velocity_t0: Velocity at time t
            velocity_t1: Velocity at time t+dt
            density: Density field
            dt: Time step
            spacing: Grid spacing

        Returns:
            Combined ViscosityEstimate
        """
        # Get estimates from different methods
        est_diffusion = self.estimate_from_velocity_diffusion(
            velocity_t0, velocity_t1, dt, spacing
        )
        est_energy = self.estimate_from_energy_dissipation(
            velocity_t0, velocity_t1, density, dt, spacing
        )

        # Weighted average by confidence
        total_confidence = est_diffusion.confidence + est_energy.confidence + 1e-8

        nu_combined = (
            est_diffusion.kinematic_viscosity * est_diffusion.confidence +
            est_energy.kinematic_viscosity * est_energy.confidence
        ) / total_confidence

        mu_combined = nu_combined * self.reference_density

        return ViscosityEstimate(
            kinematic_viscosity=nu_combined,
            dynamic_viscosity=mu_combined,
            confidence=(est_diffusion.confidence + est_energy.confidence) / 2,
            method='combined'
        )

    def _compute_laplacian(
        self,
        field: torch.Tensor,
        spacing: torch.Tensor
    ) -> torch.Tensor:
        """Compute Laplacian using central differences."""
        laplacian = torch.zeros_like(field)

        for dim in range(3):
            h = spacing[dim]
            d2f = (
                torch.roll(field, -1, dims=dim) -
                2 * field +
                torch.roll(field, 1, dims=dim)
            ) / (h ** 2)
            laplacian = laplacian + d2f

        return laplacian

    def _central_diff(
        self,
        field: torch.Tensor,
        dim: int,
        h: torch.Tensor
    ) -> torch.Tensor:
        """Central difference derivative."""
        return (torch.roll(field, -1, dims=dim) - torch.roll(field, 1, dims=dim)) / (2 * h)


class LearnableViscosity(nn.Module):
    """
    Learnable viscosity parameter for physics-informed optimization.
    """

    def __init__(
        self,
        initial_viscosity: float = 0.001,
        spatially_varying: bool = False,
        resolution: Optional[Tuple[int, int, int]] = None
    ):
        """
        Initialize learnable viscosity.

        Args:
            initial_viscosity: Initial kinematic viscosity
            spatially_varying: Whether viscosity varies in space
            resolution: Grid resolution if spatially varying
        """
        super().__init__()

        self.spatially_varying = spatially_varying

        if spatially_varying and resolution is not None:
            # Spatially varying viscosity field
            self._viscosity = nn.Parameter(
                torch.full(resolution, initial_viscosity).log()
            )
        else:
            # Scalar viscosity
            self._viscosity = nn.Parameter(
                torch.tensor(initial_viscosity).log()
            )

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get viscosity value(s).

        Args:
            x: Optional positions for spatially varying viscosity

        Returns:
            Viscosity value(s) (always positive via exp)
        """
        return torch.exp(self._viscosity)

    @property
    def value(self) -> torch.Tensor:
        """Get viscosity value."""
        return torch.exp(self._viscosity)


def identify_fluid_type(
    estimated_viscosity: float,
    estimated_density: float = 1000.0,
    tolerance: float = 0.5
) -> Tuple[str, float]:
    """
    Identify likely fluid type from estimated properties.

    Args:
        estimated_viscosity: Estimated dynamic viscosity (Pa·s)
        estimated_density: Estimated density (kg/m³)
        tolerance: Log-scale tolerance for matching

    Returns:
        Tuple of (fluid_name, similarity_score)
    """
    FLUID_PROPERTIES = {
        'water': {'viscosity': 0.001, 'density': 1000},
        'milk': {'viscosity': 0.003, 'density': 1030},
        'oil (light)': {'viscosity': 0.03, 'density': 900},
        'oil (heavy)': {'viscosity': 0.1, 'density': 950},
        'honey': {'viscosity': 2.0, 'density': 1400},
        'glycerin': {'viscosity': 1.5, 'density': 1260},
        'mercury': {'viscosity': 0.0015, 'density': 13500},
        'syrup': {'viscosity': 1.0, 'density': 1300},
    }

    best_match = 'unknown'
    best_score = 0.0

    for name, props in FLUID_PROPERTIES.items():
        # Log-scale distance for viscosity (spans orders of magnitude)
        log_visc_diff = abs(
            torch.log(torch.tensor(estimated_viscosity)) -
            torch.log(torch.tensor(props['viscosity']))
        ).item()

        # Linear scale for density
        density_diff = abs(estimated_density - props['density']) / props['density']

        # Combined score
        score = torch.exp(torch.tensor(-log_visc_diff / tolerance - density_diff)).item()

        if score > best_score:
            best_score = score
            best_match = name

    return best_match, best_score
