"""Representation of gravitational perturbation sources.

We distinguish between two kinds of source representations:
1. Positive-real convenience wrappers: PointMass and PointMassCloud representing
   actual physical objects (vehicles, optical components, DM-like masses).
2. Generic signed/complex representations: WeightedPointSource representing
   density perturbations (δρ elements) or phase-weighted source elements.
"""

from dataclasses import dataclass

import numpy as np

from gragra._arrays import as_weights, as_xyz_table, as_xyz_vector
from gragra.array_types import ComplexArray, FloatArray


@dataclass(frozen=True)
class WeightedPointSource:
    """Generic signed/complex perturbation source representation.

    This is the core representation for gravitational perturbation solvers.
    Weights are restricted to mass elements (δm = δρ * dV) in kg, which may
    be negative (for underdensities) or complex-valued. A coherent transfer
    weight is a different abstraction with a different unit and does not
    belong in ``weights_kg``.

    Parameters
    ----------
    positions_m : array-like
        Coordinates table of shape (N, 3) in meters.
    weights_kg : array-like
        Weights vector of shape (N,) in kilograms (float or complex).
    """

    positions_m: FloatArray
    weights_kg: FloatArray | ComplexArray

    def __post_init__(self):
        positions = as_xyz_table("positions_m", self.positions_m)
        weights = as_weights("weights_kg", self.weights_kg)

        if len(positions) != len(weights):
            raise ValueError(
                "Length of positions_m and weights_kg must match: "
                f"{len(positions)} != {len(weights)}"
            )

        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "weights_kg", weights)


@dataclass(frozen=True)
class PointMass:
    """Convenience wrapper for a single positive-real point mass.

    Parameters
    ----------
    mass_kg : float
        Mass in kilograms. Must be strictly positive and finite.
    position_m : array-like
        Coordinates of shape (3,) in meters.
    """

    mass_kg: float
    position_m: FloatArray

    def __post_init__(self):
        try:
            mass = float(self.mass_kg)
        except (ValueError, TypeError) as e:
            raise TypeError(f"mass_kg must be numeric: {e}") from e

        if mass <= 0 or not np.isfinite(mass):
            raise ValueError("mass_kg must be positive and finite")

        pos = as_xyz_vector("position_m", self.position_m)
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "position_m", pos)

    def as_weighted_source(self) -> WeightedPointSource:
        """Convert to the canonical WeightedPointSource format."""
        # single (3,) vector to (1, 3) table
        positions = self.position_m[np.newaxis, :]
        weights = np.array([self.mass_kg], dtype=np.float64)
        return WeightedPointSource(positions_m=positions, weights_kg=weights)


@dataclass(frozen=True)
class PointMassCloud:
    """Convenience wrapper for multiple positive-real point masses.

    Parameters
    ----------
    masses_kg : array-like
        Masses of shape (N,) in kilograms. Must be strictly positive.
    positions_m : array-like
        Coordinates of shape (N, 3) in meters.
    """

    masses_kg: FloatArray
    positions_m: FloatArray

    def __post_init__(self):
        positions = as_xyz_table("positions_m", self.positions_m)
        masses = as_weights("masses_kg", self.masses_kg)

        if np.iscomplexobj(masses):
            raise TypeError("masses_kg must be real-valued")

        if (masses <= 0).any():
            raise ValueError("masses_kg must contain only positive values")

        if len(positions) != len(masses):
            raise ValueError(
                "Length of positions_m and masses_kg must match: "
                f"{len(positions)} != {len(masses)}"
            )

        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "masses_kg", masses)

    def as_weighted_source(self) -> WeightedPointSource:
        """Convert to the canonical WeightedPointSource format."""
        return WeightedPointSource(
            positions_m=self.positions_m, weights_kg=self.masses_kg
        )
