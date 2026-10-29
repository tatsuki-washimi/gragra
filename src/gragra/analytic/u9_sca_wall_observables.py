"""Uniform wall-displacement gravity; G-free values and sample axes."""

import numpy as np

from gragra._arrays import _check_no_bool
from gragra.analytic._u9_spatial_derivatives import (
    cuboid_static_acceleration,
    cuboid_third,
    cylinder_axis_static,
    cylinder_third,
)
from gragra.analytic._validation import (
    validate_nonnegative_scalar_input,
    validate_real_xyz_table,
    validate_real_xyz_vector,
    validate_scalar_input,
)
from gragra.analytic.cavity_cuboid import cuboid_gradient_g_free
from gragra.analytic.cavity_cylinder import (
    _cylinder_far_field,
    _reject_cylinder_boundary_targets,
    cylinder_acceleration_gradient_g_free,
)
from gragra.analytic.oriented import _validate_rotation_matrix
from gragra.analytic.sca_tensor import _reject_cuboid_boundary, _reject_sphere_boundary
from gragra.analytic.values import GFreeGravityValues


class U9EvaluationError(ArithmeticError):
    """Analytic evaluation failure with call-local target diagnostics."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _inputs(targets, center, density, displacement, rotation=None):
    points = validate_real_xyz_table(targets, "targets_m")
    if len(points) == 0:
        raise ValueError("targets_m must not be empty")
    center = validate_real_xyz_vector(center, "center_m")
    rho = validate_nonnegative_scalar_input(density, "density_kg_m3")
    _check_no_bool(displacement, "displacement_m")
    raw = np.asarray(displacement)
    if raw.dtype.kind not in "iufc":
        raise TypeError("displacement_m must be real or complex numeric")
    if raw.shape != (3,) and not (raw.ndim == 2 and raw.shape[1] == 3 and len(raw)):
        raise ValueError("displacement_m must have shape (3,) or nonempty (S,3)")
    u = np.array(raw, dtype=np.complex128 if np.iscomplexobj(raw) else np.float64)
    if not np.isfinite(u).all():
        raise ValueError("displacement_m must be finite")
    q = np.eye(3) if rotation is None else _validate_rotation_matrix(rotation)
    return (points - center) @ q, np.atleast_2d(u) @ q, u.ndim == 1, rho, q


def _finish(acc, gamma, third, u, scalar, rho, rotation):
    phi = -rho * np.einsum("mi,si->sm", acc, u)
    wall_acc = rho * np.einsum("mij,sj->smi", gamma, u)
    wall_grad = rho * np.einsum("mijk,sk->smij", third, u)
    wall_acc = wall_acc @ rotation.T
    wall_grad = np.einsum("ij,smjk,lk->smil", rotation, wall_grad, rotation)
    values = (phi, wall_acc, wall_grad)
    return GFreeGravityValues(*(v[0] if scalar else v for v in values))


def _zero_values(points, u, scalar):
    dtype = np.complex128 if np.iscomplexobj(u) else np.float64
    prefix = (len(points),) if scalar else (len(u), len(points))
    return GFreeGravityValues(
        np.zeros(prefix, dtype=dtype),
        np.zeros((*prefix, 3), dtype=dtype),
        np.zeros((*prefix, 3, 3), dtype=dtype),
    )


def sphere_sca_wall_observables_g_free(
    targets_m, *, center_m, radius_m, density_kg_m3, displacement_m
):
    """Uniform spherical wall sheet: potential, acceleration, gradient.

    Geometry in meters; density kg/m³; displacement (3,) or sample table
    (S,3), finite real/complex meters. Output units kg/m, kg/m², kg/m³.
    Targets (M,3) must be nonempty and off the wall. Potential is zero at
    infinity. Complex contraction is bilinear, with no conjugation.
    """
    points, u, scalar, rho, rotation = _inputs(
        targets_m, center_m, density_kg_m3, displacement_m
    )
    radius = validate_scalar_input(radius_m, "radius_m")
    _reject_sphere_boundary(points, radius)
    if rho == 0 or not np.any(u):
        return _zero_values(points, u, scalar)
    count = len(points)
    acc, gamma, third = (
        np.zeros((count, 3)),
        np.zeros((count, 3, 3)),
        np.zeros((count, 3, 3, 3)),
    )
    k, identity = 4 * np.pi / 3, np.eye(3)
    for m, point in enumerate(points):
        distance = np.linalg.norm(point)
        if distance < radius:
            acc[m], gamma[m] = -k * point, -k * identity
        else:
            mass = k * radius**3
            acc[m] = -mass * point / distance**3
            gamma[m] = mass * (
                3 * np.outer(point, point) / distance**5 - identity / distance**3
            )
            for i in range(3):
                for j in range(3):
                    for axis in range(3):
                        third[m, i, j, axis] = mass * (
                            3
                            * (
                                point[axis] * (i == j)
                                + point[j] * (i == axis)
                                + point[i] * (j == axis)
                            )
                            / distance**5
                            - 15 * point[i] * point[j] * point[axis] / distance**7
                        )
    return _finish(acc, gamma, third, u, scalar, rho, rotation)


def cuboid_sca_wall_observables_g_free(
    targets_m,
    *,
    center_m,
    half_lengths_m,
    density_kg_m3,
    displacement_m,
    rotation=None,
):
    """Uniform cuboid wall sheet, with half lengths and local-to-world rotation.

    Displacement samples (S,3) apply independently to every target (M,3).
    Units, complex convention and output shapes match the sphere function.
    Physical faces/edges/vertices are rejected; exterior extensions are valid.
    """
    points, u, scalar, rho, q = _inputs(
        targets_m, center_m, density_kg_m3, displacement_m, rotation
    )
    half = validate_real_xyz_vector(half_lengths_m, "half_lengths_m")
    if np.any(half <= 0):
        raise ValueError("half_lengths_m must be positive")
    _reject_cuboid_boundary(points, half)
    if rho == 0 or not np.any(u):
        return _zero_values(points, u, scalar)
    acc = np.array([cuboid_static_acceleration(half, point) for point in points])
    gamma = cuboid_gradient_g_free([0, 0, 0], 2 * half, 1.0, points)
    third = np.array([cuboid_third(half, point) for point in points])
    return _finish(acc, gamma, third, u, scalar, rho, q)


def cylinder_sca_wall_observables_g_free(
    targets_m,
    *,
    center_m,
    radius_m,
    height_m,
    density_kg_m3,
    displacement_m,
    rotation=None,
):
    """Uniform finite-cylinder wall sheet with full height in meters.

    Analytic elliptic/axis/far derivatives, no spatial quadrature. Local axis
    is z; rotation maps local to world. Targets avoid physical side/caps/rim.
    An exterior result is the wall-sheet field, not the total surrounding rock.
    """
    points, u, scalar, rho, q = _inputs(
        targets_m, center_m, density_kg_m3, displacement_m, rotation
    )
    radius = validate_scalar_input(radius_m, "radius_m")
    height = validate_scalar_input(height_m, "height_m")
    _reject_cylinder_boundary_targets(points, radius, height)
    if rho == 0 or not np.any(u):
        return _zero_values(points, u, scalar)
    acc = np.empty((len(points), 3))
    gamma = np.empty((len(points), 3, 3))
    third = np.empty((len(points), 3, 3, 3))
    for index, point in enumerate(points):
        try:
            radial = np.hypot(point[0], point[1])
            distance = np.linalg.norm(point)
            if distance >= 16 * np.hypot(radius, height / 2):
                a, g = _cylinder_far_field(radius, height, 1.0, point)
            elif radial <= 0.1 * radius:
                a, g = cylinder_axis_static(radius, height, point)
            else:
                static_a, static_g = cylinder_acceleration_gradient_g_free(
                    [0, 0, 0], radius, height, 1.0, point[None, :]
                )
                a, g = static_a[0], static_g[0]
            acc[index], gamma[index] = a, g
            third[index] = cylinder_third(radius, height, point, g)
            if not all(np.isfinite(v).all() for v in (a, g, third[index])):
                raise ArithmeticError("nonfinite analytic coefficient")
        except ArithmeticError as error:
            diagnostics = {
                "status": "ANALYTIC_EVALUATION_FAILED",
                "failed_index": index,
                "evaluated_indices": list(range(index)),
                "unevaluated_indices": list(range(index + 1, len(points))),
                "local_target_m": point.tolist(),
                "samples": "all share geometry",
                "cause": str(error),
            }
            raise U9EvaluationError(
                f"cylinder analytic evaluation failed at target {index}", diagnostics
            ) from error
    return _finish(acc, gamma, third, u, scalar, rho, q)
