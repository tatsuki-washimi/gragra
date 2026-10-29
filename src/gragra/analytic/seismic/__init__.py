"""High-level analytic seismic models."""

from typing import TYPE_CHECKING

from ._halfspace_psv_model import halfspace_psv
from ._rayleigh import rayleigh_halfspace
from ._rayleigh_response import rayleigh_response
from ._spherical_cavity_model import spherical_cavity
from ._spherical_shell_model import spherical_shell

__all__ = [
    "halfspace_psv",
    "rayleigh_halfspace",
    "rayleigh_response",
    "spherical_cavity",
    "spherical_shell",
    "small_cavity_wall",
    "point_force",
]

if TYPE_CHECKING:
    from ._gen1_point_force import point_force
    from ._gen1_sca_wall import small_cavity_wall


def __getattr__(name: str):
    """Load Gen1 high-level seismic factories only when requested."""
    if name == "small_cavity_wall":
        from ._gen1_sca_wall import small_cavity_wall

        return small_cavity_wall
    if name == "point_force":
        from ._gen1_point_force import point_force

        return point_force
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
