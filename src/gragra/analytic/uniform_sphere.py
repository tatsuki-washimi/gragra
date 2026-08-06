"""Analytical benchmark solutions for a uniform density sphere."""

import numpy as np

from gragra._arrays import as_xyz_table, as_xyz_vector
from gragra.array_types import ComplexArray, FloatArray


def _validate_sphere_inputs(
    center_m: list | tuple | np.ndarray,
    radius_m: float,
    density_kg_m3: float | complex,
    targets_m: list | tuple | np.ndarray,
) -> tuple[FloatArray, float, float | complex, FloatArray]:
    """Validate inputs for uniform sphere analytical benchmark."""
    if isinstance(radius_m, bool) or not isinstance(radius_m, (int, float, np.number)):
        raise TypeError("radius_m must be a numeric scalar")
    r_val = float(radius_m)
    if not np.isfinite(r_val):
        raise ValueError("radius_m must be finite")
    if r_val <= 0.0:
        raise ValueError("radius_m must be positive")

    if isinstance(density_kg_m3, bool) or not isinstance(
        density_kg_m3, (int, float, complex, np.number)
    ):
        raise TypeError("density_kg_m3 must be a numeric scalar")
    if np.iscomplexobj(density_kg_m3):
        dens = complex(density_kg_m3)
    else:
        dens = float(density_kg_m3)
    if not np.isfinite(dens):
        raise ValueError("density_kg_m3 must be finite")

    center = as_xyz_vector("center_m", center_m)
    targets = as_xyz_table("targets_m", targets_m)

    return center, r_val, dens, targets


def uniform_sphere_potential(
    center_m: list | tuple | np.ndarray,
    radius_m: float,
    density_kg_m3: float | complex,
    targets_m: list | tuple | np.ndarray,
) -> FloatArray | ComplexArray:
    """Calculate G-free gravitational potential contraction of a uniform sphere.

    Parameters
    ----------
    center_m : array-like
        Shape (3,) coordinates of the sphere center in meters.
    radius_m : float
        Radius of the sphere in meters (must be positive).
    density_kg_m3 : float or complex
        Mass density in kg/m^3 (real or complex).
    targets_m : array-like
        Shape (M, 3) coordinates of the target points in meters.

    Returns
    -------
    FloatArray or ComplexArray
        Calculated G-free potential of shape (M,).
    """
    center, r_val, dens, targets = _validate_sphere_inputs(
        center_m, radius_m, density_kg_m3, targets_m
    )

    # Compute distances relative to center
    rel_pos = targets - center  # (M, 3)
    dists = np.linalg.norm(rel_pos, axis=-1)  # (M,)

    # Total mass of the uniform sphere (G-free factor of mass: M)
    # Mass M = 4/3 * pi * R^3 * rho
    mass = (4.0 / 3.0) * np.pi * (r_val**3) * dens

    out_dtype = np.complex128 if np.iscomplexobj(dens) else np.float64
    out = np.empty(len(targets), dtype=out_dtype)

    mask_inside = dists < r_val
    mask_outside = ~mask_inside

    # Outside potential: -M / r
    if np.any(mask_outside):
        out[mask_outside] = -mass / dists[mask_outside]

    # Inside potential: -M / (2 * R^3) * (3 * R^2 - r^2)
    if np.any(mask_inside):
        r_in = dists[mask_inside]
        out[mask_inside] = -mass / (2.0 * r_val**3) * (3.0 * r_val**2 - r_in**2)

    out.setflags(write=False)
    return out


def uniform_sphere_acceleration(
    center_m: list | tuple | np.ndarray,
    radius_m: float,
    density_kg_m3: float | complex,
    targets_m: list | tuple | np.ndarray,
) -> FloatArray | ComplexArray:
    """Calculate G-free gravitational acceleration contraction of a uniform sphere.

    Parameters
    ----------
    center_m : array-like
        Shape (3,) coordinates of the sphere center in meters.
    radius_m : float
        Radius of the sphere in meters (must be positive).
    density_kg_m3 : float or complex
        Mass density in kg/m^3 (real or complex).
    targets_m : array-like
        Shape (M, 3) coordinates of the target points in meters.

    Returns
    -------
    FloatArray or ComplexArray
        Calculated G-free acceleration vector of shape (M, 3).
    """
    center, r_val, dens, targets = _validate_sphere_inputs(
        center_m, radius_m, density_kg_m3, targets_m
    )

    # Compute distances relative to center
    rel_pos = targets - center  # (M, 3)
    dists = np.linalg.norm(rel_pos, axis=-1)  # (M,)

    # Mass M = 4/3 * pi * R^3 * rho
    mass = (4.0 / 3.0) * np.pi * (r_val**3) * dens

    out_dtype = np.complex128 if np.iscomplexobj(dens) else np.float64
    out = np.empty((len(targets), 3), dtype=out_dtype)

    mask_inside = dists < r_val
    mask_outside = ~mask_inside

    # Outside: -M * r_rel / r^3
    if np.any(mask_outside):
        r_out = dists[mask_outside][:, np.newaxis]
        out[mask_outside] = -mass * rel_pos[mask_outside] / (r_out**3)

    # Inside: -M * r_rel / R^3
    if np.any(mask_inside):
        out[mask_inside] = -mass * rel_pos[mask_inside] / (r_val**3)

    out.setflags(write=False)
    return out


def uniform_sphere_gradient(
    center_m: list | tuple | np.ndarray,
    radius_m: float,
    density_kg_m3: float | complex,
    targets_m: list | tuple | np.ndarray,
) -> FloatArray | ComplexArray:
    """Calculate G-free gravity gradient tensor contraction of a uniform sphere.

    Parameters
    ----------
    center_m : array-like
        Shape (3,) coordinates of the sphere center in meters.
    radius_m : float
        Radius of the sphere in meters (must be positive).
    density_kg_m3 : float or complex
        Mass density in kg/m^3 (real or complex).
    targets_m : array-like
        Shape (M, 3) coordinates of the target points in meters.

    Returns
    -------
    FloatArray or ComplexArray
        Calculated G-free gravity gradient tensor of shape (M, 3, 3).
    """
    center, r_val, dens, targets = _validate_sphere_inputs(
        center_m, radius_m, density_kg_m3, targets_m
    )

    # Compute distances relative to center
    rel_pos = targets - center  # (M, 3)
    dists = np.linalg.norm(rel_pos, axis=-1)  # (M,)

    # Mass M = 4/3 * pi * R^3 * rho
    mass = (4.0 / 3.0) * np.pi * (r_val**3) * dens

    out_dtype = np.complex128 if np.iscomplexobj(dens) else np.float64
    out = np.empty((len(targets), 3, 3), dtype=out_dtype)

    mask_inside = dists < r_val
    mask_outside = ~mask_inside

    eye3 = np.eye(3)

    # Outside: M * (3 * r_i * r_j / r^5 - delta_ij / r^3)
    if np.any(mask_outside):
        r_out = dists[mask_outside]
        r_out_pos = rel_pos[mask_outside]
        # (M_out, 3, 3)
        outer = r_out_pos[:, :, np.newaxis] * r_out_pos[:, np.newaxis, :]
        term1 = 3.0 * outer / (r_out[:, np.newaxis, np.newaxis] ** 5)
        term2 = eye3[np.newaxis, :, :] / (r_out[:, np.newaxis, np.newaxis] ** 3)
        out[mask_outside] = mass * (term1 - term2)

    # Inside: -M * delta_ij / R^3
    if np.any(mask_inside):
        out[mask_inside] = -mass * eye3[np.newaxis, :, :] / (r_val**3)

    out.setflags(write=False)
    return out
