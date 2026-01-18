"""Physics modules for fluid property estimation with Navier-Stokes constraints."""

from .fields import VectorField, ScalarField, FluidFields, FieldBounds
from .navier_stokes import NavierStokesLoss, PhysicsResidual
from .neural_fields import NeuralVelocityField, NeuralPressureField, NeuralDensityField
from .viscosity_estimation import ViscosityEstimator

__all__ = [
    "VectorField",
    "ScalarField",
    "FluidFields",
    "FieldBounds",
    "NavierStokesLoss",
    "PhysicsResidual",
    "NeuralVelocityField",
    "NeuralPressureField",
    "NeuralDensityField",
    "ViscosityEstimator",
]
