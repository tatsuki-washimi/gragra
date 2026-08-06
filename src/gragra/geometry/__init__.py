"""Geometry module for source discretization and primitive shapes.

This module contains shape primitives (Sphere, Cuboid, Cylinder)
and volume discretization helpers to generate WeightedPointSources.
"""

from gragra.geometry.primitives import (
    Cuboid,
    Cylinder,
    HalfCylinder,
    HalfSphere,
    Sphere,
)
from gragra.geometry.surface_mesh import MeshResolution, PrimitiveSurfaceMesh

__all__ = [
    "Sphere",
    "Cuboid",
    "Cylinder",
    "HalfSphere",
    "HalfCylinder",
    "MeshResolution",
    "PrimitiveSurfaceMesh",
]
