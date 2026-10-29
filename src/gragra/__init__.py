"""gragra: A Python package for gravitational perturbation calculation."""

from typing import TYPE_CHECKING

from gragra.constants import G_SI
from gragra.gravity_sources import PointMass, PointMassCloud, WeightedPointSource
from gragra.observables import (
    coherent_acceleration_csd,
    coherent_gravitoelastic_tensor,
    coherent_point_acceleration,
    coherent_point_gravity_gradient,
    displacement_field_acceleration,
    displacement_transfer_acceleration,
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)

__all__ = [
    "G_SI",
    "WeightedPointSource",
    "PointMass",
    "PointMassCloud",
    "TargetPoints",
    "point_potential",
    "point_acceleration",
    "point_gravity_gradient",
    "coherent_point_acceleration",
    "coherent_point_gravity_gradient",
    "coherent_gravitoelastic_tensor",
    "coherent_acceleration_csd",
    "displacement_transfer_acceleration",
    "displacement_field_acceleration",
]

if TYPE_CHECKING:
    from gragra.targets._high_points import TargetPoints


def __getattr__(name: str):
    """Load high-level target types only when callers request them."""
    if name == "TargetPoints":
        from gragra.targets._high_points import TargetPoints

        return TargetPoints
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
