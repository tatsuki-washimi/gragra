"""Rigid local-to-world wrappers for closed-form analytic fields.

The wrappers in this module add only coordinate transport.  They do not
reimplement a shape's local analytic field and they do not silently repair an
invalid rotation matrix.  A rotation is interpreted as a local-to-world
proper orthogonal matrix ``Q``:

``x_local = Q.T @ (x_world - center)``
``a_world = Q @ a_local``
``Gamma_world = Q @ Gamma_local @ Q.T``

The first supported shape is the uniform cuboid.  Cylinder orientation is
deliberately left to the finite-cylinder work because its general-point local
field is still an opt-in numerical path rather than a completed closed form.
"""

from __future__ import annotations

import numpy as np

from gragra._arrays import _check_no_bool
from gragra.analytic._validation import (
    validate_real_xyz_table,
    validate_real_xyz_vector,
)
from gragra.analytic.cavity_cuboid import cuboid_observables_g_free
from gragra.analytic.values import GFreeGravityValues


def _validate_rotation_matrix(rotation_matrix: list | tuple | np.ndarray) -> np.ndarray:
    """Validate a finite proper rotation without projecting it onto SO(3)."""
    _check_no_bool(rotation_matrix, "rotation_matrix")
    raw = np.asarray(rotation_matrix)
    if (
        raw.dtype.kind == "b"
        or raw.dtype.kind == "c"
        or np.iscomplexobj(rotation_matrix)
    ):
        raise TypeError("rotation_matrix must be a real numeric array")
    if raw.dtype.kind == "O" or not np.issubdtype(raw.dtype, np.number):
        raise TypeError("rotation_matrix must be a real numeric array")
    rotation = np.asarray(rotation_matrix, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(
            f"rotation_matrix must have shape (3, 3), got {rotation.shape}"
        )
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation_matrix must be finite")
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-10):
        raise ValueError("rotation_matrix must satisfy Q.T @ Q = I")
    if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("rotation_matrix must have determinant +1")
    rotation = rotation.copy()
    rotation.setflags(write=False)
    return rotation


def oriented_cuboid_observables_g_free(
    center_m: list | tuple | np.ndarray,
    lengths_m: list | tuple | np.ndarray,
    density_kg_m3: float,
    targets_m: list | tuple | np.ndarray,
    *,
    rotation_matrix: list | tuple | np.ndarray,
) -> GFreeGravityValues:
    """Evaluate a uniformly dense cuboid in an arbitrary proper orientation.

    Parameters
    ----------
    center_m : array-like, shape (3,)
        World-coordinate cuboid center in meters.
    lengths_m : array-like, shape (3,)
        Full local-coordinate side lengths in meters.
    density_kg_m3 : float
        Non-negative uniform density in kg/m^3.
    targets_m : array-like, shape (M, 3)
        World-coordinate targets. Material-boundary targets are rejected by
        the local cuboid closed form.
    rotation_matrix : array-like, shape (3, 3), keyword-only
        Proper orthogonal local-to-world matrix ``Q``. Invalid matrices are
        rejected; they are not projected or normalized.

    Returns
    -------
    GFreeGravityValues
        G-free potential, acceleration, and 3x3 gradient in world
        coordinates. Potential is invariant under the rigid transform;
        vector and tensor components are rotated covariantly.
    """
    center = validate_real_xyz_vector(center_m, "center_m")
    targets = validate_real_xyz_table(targets_m, "targets_m")
    rotation = _validate_rotation_matrix(rotation_matrix)
    local_targets = (targets - center) @ rotation
    local_values = cuboid_observables_g_free(
        np.zeros(3), lengths_m, density_kg_m3, local_targets
    )

    acceleration = local_values.acceleration @ rotation.T
    gradient = np.einsum("ij,mjk,lk->mil", rotation, local_values.gradient, rotation)
    return GFreeGravityValues(
        potential=local_values.potential,
        acceleration=acceleration,
        gradient=gradient,
    )
