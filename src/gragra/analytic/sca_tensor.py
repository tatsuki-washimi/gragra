"""Small-cavern wall-coupling tensors from closed-surface integrals.

For an outward normal ``n`` on a cavity wall and a target ``x`` away from the
wall, this module evaluates the dimensionless tensor

``C_ij(x) = integral_boundary((s_i - x_i) * n_j / |s-x|**3 dS)``.

The corresponding G-free wall acceleration for a displacement ``u`` is
``-rho * C @ u``.  The explicit minus sign follows the SCA wall convention
used by :func:`gragra.analytic.cavity_coupling.sca_wall_acceleration`.
The public sphere and cuboid functions use the exact static filled-body
gradient inversion by default.  An explicit ``quadrature_order`` requests the
surface-integral reference and is accepted only after target-wise convergence.
Neither path is a replacement for the scalar Harms Table 1 constants.
"""

from __future__ import annotations

import numpy as np

from gragra._arrays import _check_no_bool
from gragra.analytic._validation import (
    validate_nonnegative_scalar_input,
    validate_real_xyz_table,
    validate_real_xyz_vector,
    validate_scalar_input,
)
from gragra.analytic.cavity_cuboid import cuboid_gradient_g_free
from gragra.analytic.oriented import _validate_rotation_matrix
from gragra.analytic.uniform_sphere import uniform_sphere_gradient

_MAX_SURFACE_QUADRATURE_ORDER = 128
_SURFACE_CONVERGENCE_RTOL = 1e-9
_SURFACE_CONVERGENCE_ATOL = 1e-12


def _validate_quadrature_order(order: int | None) -> int | None:
    if order is None:
        return None
    if isinstance(order, (bool, np.bool_)):
        raise TypeError("quadrature_order must be an integer, not boolean")
    if not isinstance(order, (int, np.integer)):
        raise TypeError("quadrature_order must be an integer")
    value = int(order)
    if value < 4:
        raise ValueError("quadrature_order must be at least 4")
    if value >= _MAX_SURFACE_QUADRATURE_ORDER:
        raise ValueError(
            f"quadrature_order must be below {_MAX_SURFACE_QUADRATURE_ORDER}"
        )
    return value


def _reject_sphere_boundary(targets_rel: np.ndarray, radius: float) -> None:
    atol = 1e-12 * max(radius, 1.0)
    radial_distance = np.linalg.norm(targets_rel, axis=1)
    boundary = np.isclose(radial_distance, radius, rtol=0.0, atol=atol)
    if np.any(boundary):
        index = int(np.flatnonzero(boundary)[0])
        raise ValueError(f"target point at index {index} lies on the sphere boundary")


def _reject_cuboid_boundary(
    targets_local: np.ndarray, half_lengths: np.ndarray
) -> None:
    scale = max(float(np.max(half_lengths)), 1.0)
    atol = 1e-12 * scale
    inside = np.all(np.abs(targets_local) <= half_lengths[None, :] + atol, axis=1)
    on_face = np.any(
        np.isclose(np.abs(targets_local), half_lengths[None, :], rtol=0.0, atol=atol),
        axis=1,
    )
    boundary = inside & on_face
    if np.any(boundary):
        index = int(np.flatnonzero(boundary)[0])
        raise ValueError(f"target point at index {index} lies on the cuboid boundary")


def _sphere_surface_quadrature(
    radius: float, order: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    azimuthal_order = max(32, 2 * order)
    azimuth = 2.0 * np.pi * np.arange(azimuthal_order) / azimuthal_order
    u, phi = np.meshgrid(nodes, azimuth, indexing="ij")
    radial = np.sqrt(1.0 - u * u)
    normals = np.stack(
        (radial * np.cos(phi), radial * np.sin(phi), u), axis=-1
    ).reshape(-1, 3)
    points = radius * normals
    area_weights = np.broadcast_to(
        (radius * radius * weights)[:, None] * (2.0 * np.pi / azimuthal_order),
        u.shape,
    ).reshape(-1)
    return points, normals, area_weights


def _cuboid_surface_quadrature(
    half_lengths: np.ndarray, order: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    area_weights: list[np.ndarray] = []
    for axis in range(3):
        tangential = [index for index in range(3) if index != axis]
        a, b = np.meshgrid(nodes, nodes, indexing="ij")
        for sign in (-1.0, 1.0):
            face_points = np.zeros((*a.shape, 3), dtype=np.float64)
            face_points[..., axis] = sign * half_lengths[axis]
            face_points[..., tangential[0]] = half_lengths[tangential[0]] * a
            face_points[..., tangential[1]] = half_lengths[tangential[1]] * b
            face_normals = np.zeros_like(face_points)
            face_normals[..., axis] = sign
            face_weights = (
                half_lengths[tangential[0]]
                * half_lengths[tangential[1]]
                * weights[:, None]
                * weights[None, :]
            )
            points.append(face_points.reshape(-1, 3))
            normals.append(face_normals.reshape(-1, 3))
            area_weights.append(face_weights.reshape(-1))
    return (
        np.concatenate(points),
        np.concatenate(normals),
        np.concatenate(area_weights),
    )


def _surface_coupling_tensor(
    points: np.ndarray,
    normals: np.ndarray,
    area_weights: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    delta = points[None, :, :] - targets[:, None, :]
    inverse_cubed = 1.0 / np.linalg.norm(delta, axis=2) ** 3
    return np.einsum("n,mni,nj,mn->mij", area_weights, delta, normals, inverse_cubed)


def _converged_surface_tensor(
    quadrature_builder,
    initial_order: int,
) -> np.ndarray:
    """Evaluate a surface tensor and reject target-wise non-convergence."""
    if initial_order >= _MAX_SURFACE_QUADRATURE_ORDER:
        raise ValueError(
            "quadrature_order must be below 128 so target-wise convergence "
            "can be checked"
        )

    order = initial_order
    previous = quadrature_builder(order)
    while order < _MAX_SURFACE_QUADRATURE_ORDER:
        next_order = min(2 * order, _MAX_SURFACE_QUADRATURE_ORDER)
        current = quadrature_builder(next_order)
        difference = np.linalg.norm(current - previous, axis=(1, 2))
        scale = np.maximum(np.linalg.norm(current, axis=(1, 2)), 1.0)
        converged = difference <= (
            _SURFACE_CONVERGENCE_ATOL + _SURFACE_CONVERGENCE_RTOL * scale
        )
        if np.all(converged):
            return current
        previous = current
        order = next_order

    failed = np.flatnonzero(~converged).tolist()
    raise ValueError(
        "target quadrature convergence failed for target indices "
        f"{failed}; use the closed-form default or move targets farther from "
        "the boundary"
    )


def sphere_sca_wall_coupling_tensor_g_free(
    center_m: list | tuple | np.ndarray,
    radius_m: float,
    targets_m: list | tuple | np.ndarray,
    *,
    quadrature_order: int | None = None,
) -> np.ndarray:
    """Return the dimensionless SCA wall tensor for a sphere.

    ``targets_m`` may contain general interior or exterior points, but points
    on the spherical wall are rejected because the integral is singular.  By
    default the tensor is evaluated from the exact filled-sphere gradient
    inversion ``C = -Gamma_filled/rho``.  Passing ``quadrature_order`` opts in
    to the surface-integral reference; its result is returned only when every
    target passes the target-wise order-convergence check.
    The result has shape ``(M, 3, 3)`` and is read-only.
    """
    center = validate_real_xyz_vector(center_m, "center_m")
    radius = validate_scalar_input(radius_m, "radius_m")
    targets = validate_real_xyz_table(targets_m, "targets_m")
    order = _validate_quadrature_order(quadrature_order)
    relative = targets - center
    _reject_sphere_boundary(relative, radius)
    if order is None:
        result = -uniform_sphere_gradient(center, radius, 1.0, targets)
    else:

        def evaluate_surface(current_order: int) -> np.ndarray:
            points, normals, weights = _sphere_surface_quadrature(radius, current_order)
            return _surface_coupling_tensor(points, normals, weights, relative)

        result = _converged_surface_tensor(evaluate_surface, order)
    result.setflags(write=False)
    return result


def cuboid_sca_wall_coupling_tensor_g_free(
    center_m: list | tuple | np.ndarray,
    lengths_m: list | tuple | np.ndarray,
    targets_m: list | tuple | np.ndarray,
    *,
    rotation_matrix: list | tuple | np.ndarray | None = None,
    quadrature_order: int | None = None,
) -> np.ndarray:
    """Return the dimensionless SCA wall tensor for a cuboid.

    ``rotation_matrix`` is a local-to-world proper rotation.  The cuboid
    boundary is integrated in local coordinates and the resulting rank-two
    tensor is rotated as ``Q @ C_local @ Q.T``.  By default the tensor is
    evaluated from the exact filled-cuboid gradient inversion
    ``C = -Gamma_filled/rho``.  Passing ``quadrature_order`` opts in to the
    surface-integral reference; its result is returned only when every target
    passes the target-wise order-convergence check.  Interior and exterior
    targets are supported; physical boundary targets are rejected.
    """
    center = validate_real_xyz_vector(center_m, "center_m")
    lengths = validate_real_xyz_vector(lengths_m, "lengths_m")
    if np.any(lengths <= 0.0):
        raise ValueError("lengths_m must contain only positive values")
    targets = validate_real_xyz_table(targets_m, "targets_m")
    order = _validate_quadrature_order(quadrature_order)
    rotation = (
        np.eye(3, dtype=np.float64)
        if rotation_matrix is None
        else _validate_rotation_matrix(rotation_matrix)
    )
    relative_local = (targets - center) @ rotation
    half_lengths = lengths / 2.0
    _reject_cuboid_boundary(relative_local, half_lengths)
    if order is None:
        local_result = -cuboid_gradient_g_free(
            np.zeros(3), lengths, 1.0, relative_local
        )
    else:

        def evaluate_surface(current_order: int) -> np.ndarray:
            points, normals, weights = _cuboid_surface_quadrature(
                half_lengths, current_order
            )
            return _surface_coupling_tensor(points, normals, weights, relative_local)

        local_result = _converged_surface_tensor(evaluate_surface, order)
    result = np.einsum("ij,mjk,lk->mil", rotation, local_result, rotation)
    result.setflags(write=False)
    return result


def _validate_displacement(value, name: str, count: int) -> np.ndarray:
    _check_no_bool(value, name)
    raw = np.asarray(value)
    if raw.dtype.kind in "bO":
        raise TypeError(f"{name} must be numeric")
    if raw.ndim == 1:
        if raw.shape != (3,):
            raise ValueError(f"{name} must have shape (3,) or ({count}, 3)")
        result = np.broadcast_to(
            np.asarray(
                value, dtype=np.complex128 if np.iscomplexobj(raw) else np.float64
            ),
            (count, 3),
        ).copy()
    elif raw.shape == (count, 3):
        result = np.asarray(
            value, dtype=np.complex128 if np.iscomplexobj(raw) else np.float64
        ).copy()
    else:
        raise ValueError(f"{name} must have shape (3,) or ({count}, 3)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def sca_wall_acceleration_from_tensor_g_free(
    coupling_tensor: list | tuple | np.ndarray,
    density_kg_m3: float,
    displacement_m: list | tuple | np.ndarray,
) -> np.ndarray:
    """Apply ``delta_a/G = -rho * C @ displacement`` in G-free units."""
    density = validate_nonnegative_scalar_input(density_kg_m3, "density_kg_m3")
    tensor = np.asarray(coupling_tensor)
    if tensor.ndim != 3 or tensor.shape[1:] != (3, 3):
        raise ValueError("coupling_tensor must have shape (M, 3, 3)")
    if tensor.dtype.kind in "bO" or not np.issubdtype(tensor.dtype, np.number):
        raise TypeError("coupling_tensor must be numeric")
    tensor = np.asarray(
        tensor, dtype=np.complex128 if np.iscomplexobj(tensor) else np.float64
    )
    if not np.isfinite(tensor).all():
        raise ValueError("coupling_tensor must contain only finite values")
    displacement = _validate_displacement(displacement_m, "displacement_m", len(tensor))
    result = -density * np.einsum("mij,mj->mi", tensor, displacement)
    result.setflags(write=False)
    return result
