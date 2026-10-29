"""Analytic models for static source geometries."""

from typing import TYPE_CHECKING

from gragra.analytic.objects._cuboid_model import uniform_cuboid
from gragra.analytic.objects._cylinder_model import uniform_cylinder
from gragra.analytic.objects._uniform_sphere import uniform_sphere

__all__ = [
    "uniform_cuboid",
    "uniform_cylinder",
    "uniform_sphere",
    "constant_velocity",
    "translational_oscillation",
    "fixed_axis_rotation",
]

if TYPE_CHECKING:
    from gragra.analytic.objects._gen1_motion import (
        constant_velocity,
        fixed_axis_rotation,
        translational_oscillation,
    )


def __getattr__(name: str):
    """Load the approved motion factories only on explicit access."""
    if name in {
        "constant_velocity",
        "translational_oscillation",
        "fixed_axis_rotation",
    }:
        from gragra.analytic.objects._gen1_motion import (
            constant_velocity,
            fixed_axis_rotation,
            translational_oscillation,
        )

        return {
            "constant_velocity": constant_velocity,
            "translational_oscillation": translational_oscillation,
            "fixed_axis_rotation": fixed_axis_rotation,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
