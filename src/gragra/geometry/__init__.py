"""Static high-level geometry and numerical shape primitives."""

from typing import TYPE_CHECKING

from gragra.geometry.primitives import HalfCylinder, HalfSphere
from gragra.geometry.surface_mesh import MeshResolution, PrimitiveSurfaceMesh

__all__ = [
    "Sphere",
    "Cuboid",
    "Cylinder",
    "SphericalShell",
    "Appearance",
    "Frame",
    "HalfSpace",
    "HalfSphere",
    "HalfCylinder",
    "MeshResolution",
    "PrimitiveSurfaceMesh",
]

if TYPE_CHECKING:
    from gragra.geometry._appearance import Appearance
    from gragra.geometry._cuboid import Cuboid
    from gragra.geometry._cylinder import Cylinder
    from gragra.geometry._frame import Frame
    from gragra.geometry._halfspace import HalfSpace
    from gragra.geometry._sphere import Sphere
    from gragra.geometry._spherical_shell import SphericalShell


def __getattr__(name: str):
    """Load high-level geometry types on explicit access."""
    if name == "Appearance":
        from gragra.geometry._appearance import Appearance

        return Appearance
    if name == "Cuboid":
        from gragra.geometry._cuboid import Cuboid

        return Cuboid
    if name == "Cylinder":
        from gragra.geometry._cylinder import Cylinder

        return Cylinder
    if name == "Frame":
        from gragra.geometry._frame import Frame

        return Frame
    if name == "HalfSpace":
        from gragra.geometry._halfspace import HalfSpace

        return HalfSpace
    if name == "Sphere":
        from gragra.geometry._sphere import Sphere

        return Sphere
    if name == "SphericalShell":
        from gragra.geometry._spherical_shell import SphericalShell

        return SphericalShell
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
