"""Quantity-aware construction of immutable C5 point sources."""

from __future__ import annotations

import numpy as np
from astropy import units as u

from gragra._evaluation_setup import DiscreteSource, density_kg_m3
from gragra.geometry._sphere import Sphere
from gragra.geometry.primitives import Sphere as CartesianSphere


def _spacing_m(spacing: u.Quantity) -> float:
    if not isinstance(spacing, u.Quantity):
        raise TypeError("spacing must be an astropy Quantity with length units")
    raw = np.asarray(spacing.to_value(u.m))
    if raw.shape != ():
        raise ValueError("spacing must be scalar")
    if np.iscomplexobj(raw):
        raise ValueError("spacing must be real")
    value = float(raw)
    if not np.isfinite(value):
        raise ValueError("spacing must be finite")
    if value <= 0:
        raise ValueError("spacing must be positive")
    return value


def discretize(
    geometry: Sphere,
    density: u.Quantity,
    spacing: u.Quantity,
    max_points: int | None = 1_000_000,
) -> DiscreteSource:
    """Eagerly create one immutable Cartesian midpoint source for a sphere.

    The existing ``geometry.primitives.Sphere`` implementation supplies the
    midpoint locations, cell volume, inclusion rule, and closest-point
    fallback. ``max_points`` limits its bounding-box grid before inclusion.
    No boundary-cell correction or mass renormalization is applied.
    """
    if not isinstance(geometry, Sphere):
        raise TypeError("geometry must be a high-level Sphere")
    density_value = density_kg_m3(density)
    spacing_value = _spacing_m(spacing)

    midpoint_sphere = CartesianSphere(
        center_m=geometry.center_m,
        radius_m=geometry.radius_m,
        density_kg_m3=density_value,
        spacing_m=spacing_value,
        max_points=max_points,
    )
    weighted = midpoint_sphere.to_weighted_points()

    axis_count = int(np.ceil((2.0 * geometry.radius_m) / spacing_value))
    grid_points = axis_count**3
    cell_width = (2.0 * geometry.radius_m) / axis_count
    metadata = {
        "spacing_m": spacing_value,
        "max_points": midpoint_sphere.max_points,
        "grid_points": grid_points,
        "cell_volume_m3": cell_width**3,
        "source_count": len(weighted.positions_m),
        "midpoint_rule": "cartesian",
    }
    return DiscreteSource(
        geometry=geometry,
        density=density,
        frame=geometry.frame,
        positions_m=weighted.positions_m,
        weights_kg=weighted.weights_kg,
        discretization=metadata,
    )


__all__ = ["discretize"]
