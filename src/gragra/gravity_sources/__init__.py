"""Gravitational source definitions."""

from typing import TYPE_CHECKING

from gragra.gravity_sources.surface import SurfaceMassSheetSource
from gragra.gravity_sources.trajectory import (
    PrescribedPointTrajectory,
    PrescribedRigidTransform,
    read_prescribed_point_trajectory_csv,
    read_prescribed_rigid_transform_csv,
)
from gragra.gravity_sources.volume import VolumeElementSource
from gragra.gravity_sources.weighted_points import (
    PointMass,
    PointMassCloud,
    WeightedPointSource,
)

__all__ = [
    "WeightedPointSource",
    "PointMass",
    "PointMassCloud",
    "VolumeElementSource",
    "SurfaceMassSheetSource",
    "PrescribedPointTrajectory",
    "PrescribedRigidTransform",
    "read_prescribed_point_trajectory_csv",
    "read_prescribed_rigid_transform_csv",
    "discretize",
    "point_sum",
    "rigid_motion",
    "ForceHistory",
    "smooth_pulse",
    "smooth_ramp_to_hold",
]

if TYPE_CHECKING:
    from gragra.gravity_sources._discretize import discretize
    from gragra.gravity_sources._force_history import (
        ForceHistory,
        smooth_pulse,
        smooth_ramp_to_hold,
    )
    from gragra.gravity_sources._models import point_sum
    from gragra.gravity_sources._rigid_motion_model import rigid_motion


def __getattr__(name: str):
    """Load C5 factories only when callers request them."""
    if name == "discretize":
        from gragra.gravity_sources._discretize import discretize

        return discretize
    if name == "point_sum":
        from gragra.gravity_sources._models import point_sum

        return point_sum
    if name == "rigid_motion":
        from gragra.gravity_sources._rigid_motion_model import rigid_motion

        return rigid_motion
    if name in {"ForceHistory", "smooth_pulse", "smooth_ramp_to_hold"}:
        from gragra.gravity_sources._force_history import (
            ForceHistory,
            smooth_pulse,
            smooth_ramp_to_hold,
        )

        return {
            "ForceHistory": ForceHistory,
            "smooth_pulse": smooth_pulse,
            "smooth_ramp_to_hold": smooth_ramp_to_hold,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
