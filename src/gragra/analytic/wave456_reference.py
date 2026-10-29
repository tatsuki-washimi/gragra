"""Independent G-free references for the Wave 4--6 research benchmarks.

This module deliberately imports neither production Newton kernels/integrators
nor :mod:`gragra.waves`.  It uses literal Newton derivatives, tensor-product
quadrature, and independently derived horizontal-Fourier half-space formulas.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import numpy as np

ArrayLike = list | tuple | np.ndarray
Density = float | complex | Callable[[np.ndarray], np.ndarray]
Displacement = Callable[[np.ndarray], np.ndarray]


def _readonly(value: ArrayLike, *, complex_output: bool) -> np.ndarray:
    dtype = np.complex128 if complex_output else np.float64
    result = np.asarray(value, dtype=dtype).copy()
    if not np.isfinite(result).all():
        raise ValueError("observable arrays must contain only finite values")
    result.setflags(write=False)
    return result


def _contains_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        return value.dtype.kind == "b" or (
            value.dtype.kind == "O" and any(_contains_bool(item) for item in value.flat)
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_bool(item) for item in value)
    return False


def _real_array(name: str, value: ArrayLike) -> np.ndarray:
    if _contains_bool(value):
        raise TypeError(f"{name} must not contain boolean values")
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise TypeError(f"{name} must contain real values")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric and real") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True, init=False)
class GFreeObservables:
    """Potential, acceleration, and gradient with static or series axes.

    Static shapes are ``(M,)``, ``(M,3)``, ``(M,3,3)``.  Series shapes are
    ``(S,M)``, ``(S,M,3)``, ``(S,M,3,3)``.  No axis is flattened or broadcast.
    """

    potential: np.ndarray
    acceleration: np.ndarray
    gradient: np.ndarray

    def __init__(
        self, potential: ArrayLike, acceleration: ArrayLike, gradient: ArrayLike
    ):
        raw_p = np.asarray(potential)
        raw_a = np.asarray(acceleration)
        raw_g = np.asarray(gradient)
        if raw_p.ndim not in (1, 2):
            raise ValueError("potential must use static (M,) or series (S,M) shape")
        prefix = raw_p.shape
        if raw_a.shape != (*prefix, 3) or raw_g.shape != (*prefix, 3, 3):
            raise ValueError("observables must have one shared sample/target prefix")
        is_complex = any(np.iscomplexobj(item) for item in (raw_p, raw_a, raw_g))
        object.__setattr__(
            self, "potential", _readonly(raw_p, complex_output=is_complex)
        )
        object.__setattr__(
            self, "acceleration", _readonly(raw_a, complex_output=is_complex)
        )
        object.__setattr__(
            self, "gradient", _readonly(raw_g, complex_output=is_complex)
        )

    @property
    def is_series(self) -> bool:
        return self.potential.ndim == 2

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter((self.potential, self.acceleration, self.gradient))


@dataclass(frozen=True)
class QuadratureConvergence:
    """Immutable evidence for an adaptive independent quadrature."""

    status: str
    normalized_difference: float
    low_order: int
    high_order: int
    subbox_count: int
    max_depth_reached: int
    evaluations: int


def _xyz_table(name: str, value: ArrayLike) -> np.ndarray:
    result = _real_array(name, value)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must be a finite array of shape (N,3)")
    return result.copy()


def _xyz_vector(name: str, value: ArrayLike) -> np.ndarray:
    result = _real_array(name, value)
    if result.shape != (3,):
        raise ValueError(f"{name} must be a finite array of shape (3,)")
    return result.copy()


def _bounds(value: ArrayLike) -> np.ndarray:
    result = _real_array("bounds_m", value)
    if result.shape != (3, 2):
        raise ValueError("bounds_m must be finite with shape (3,2)")
    if np.any(result[:, 1] <= result[:, 0]):
        raise ValueError("each bounds_m upper endpoint must exceed its lower endpoint")
    return result.copy()


def _positive(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive finite scalar")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return result


def _order(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _evaluation_budget(value: int) -> int:
    return _order("max_evaluations", value, 1)


def _stack(results: list[GFreeObservables]) -> GFreeObservables:
    return GFreeObservables(
        np.stack([item.potential for item in results]),
        np.stack([item.acceleration for item in results]),
        np.stack([item.gradient for item in results]),
    )


def _add(left: GFreeObservables, right: GFreeObservables) -> GFreeObservables:
    return GFreeObservables(
        left.potential + right.potential,
        left.acceleration + right.acceleration,
        left.gradient + right.gradient,
    )


def _zero(target_count: int, complex_output: bool = False) -> GFreeObservables:
    dtype = np.complex128 if complex_output else np.float64
    return GFreeObservables(
        np.zeros(target_count, dtype=dtype),
        np.zeros((target_count, 3), dtype=dtype),
        np.zeros((target_count, 3, 3), dtype=dtype),
    )


def point_mass_observables_g_free(
    source_positions_m: ArrayLike, mass_kg: ArrayLike, targets_m: ArrayLike
) -> GFreeObservables:
    """Literal unsoftened Newton sums, independent of production kernels."""
    sources = _xyz_table("source_positions_m", source_positions_m)
    targets = _xyz_table("targets_m", targets_m)
    if _contains_bool(mass_kg):
        raise TypeError("mass_kg must not contain boolean values")
    raw_mass = np.asarray(mass_kg)
    dtype = np.complex128 if np.iscomplexobj(raw_mass) else np.float64
    mass = np.asarray(mass_kg, dtype=dtype)
    if mass.shape != (len(sources),) or not np.isfinite(mass).all():
        raise ValueError("mass_kg must be finite with shape (N,)")
    delta = sources[None, :, :] - targets[:, None, :]
    radius2 = np.einsum("mni,mni->mn", delta, delta)
    collision = np.argwhere(radius2 == 0.0)
    if collision.size:
        m, n = collision[0]
        raise ValueError(f"target {m} collides with source {n}")
    radius = np.sqrt(radius2)
    potential = -np.sum(mass[None, :] / radius, axis=1)
    acceleration = np.einsum("n,mni,mn->mi", mass, delta, radius**-3)
    outer = delta[..., :, None] * delta[..., None, :]
    kernel = 3.0 * outer / radius[..., None, None] ** 5
    kernel -= np.eye(3) / radius[..., None, None] ** 3
    gradient = np.einsum("n,mnij->mij", mass, kernel)
    return GFreeObservables(potential, acceleration, gradient)


def multipole_observables_g_free(
    center_m: ArrayLike,
    mass_kg: float,
    second_moment_kg_m2: ArrayLike,
    targets_m: ArrayLike,
) -> GFreeObservables:
    """Monopole plus second-central-moment expansion through ``O(r^-3)``."""
    center = _xyz_vector("center_m", center_m)
    targets = _xyz_table("targets_m", targets_m)
    mass = float(mass_kg)
    if not np.isfinite(mass) or mass < 0.0:
        raise ValueError("mass_kg must be a finite non-negative scalar")
    raw_moment = np.asarray(second_moment_kg_m2)
    dtype = np.complex128 if np.iscomplexobj(raw_moment) else np.float64
    moment = np.asarray(raw_moment, dtype=dtype)
    if moment.shape != (3, 3) or not np.isfinite(moment).all():
        raise ValueError("second_moment_kg_m2 must be finite with shape (3,3)")
    if not np.allclose(moment, moment.T, rtol=0.0, atol=1e-14):
        raise ValueError("second_moment_kg_m2 must be symmetric")
    r = center[None, :] - targets
    r2 = np.einsum("mi,mi->m", r, r)
    if np.any(r2 == 0.0):
        raise ValueError("targets_m must not contain center_m")
    radius = np.sqrt(r2)
    sr = np.einsum("ij,mj->mi", moment, r)
    trace = np.trace(moment)
    contraction = 3.0 * np.einsum("mi,mi->m", r, sr) - trace * r2
    potential = -mass / radius - 0.5 * contraction / radius**5
    acceleration = np.asarray(mass * r / radius[:, None] ** 3, dtype=dtype)
    acceleration += (trace * r - 3.0 * sr) / radius[:, None] ** 5
    acceleration += 2.5 * contraction[:, None] * r / radius[:, None] ** 7

    # Differentiate a(r) analytically, then negate because r=center-target.
    identity = np.eye(3, dtype=dtype)
    cmat = trace * identity - 3.0 * moment
    v = trace * r - 3.0 * sr
    grad_a_r = mass * (
        identity / radius[:, None, None] ** 3
        - 3.0 * r[:, :, None] * r[:, None, :] / radius[:, None, None] ** 5
    )
    grad_a_r += cmat / radius[:, None, None] ** 5
    grad_a_r -= 5.0 * v[:, :, None] * r[:, None, :] / radius[:, None, None] ** 7
    grad_contraction = 6.0 * sr - 2.0 * trace * r
    grad_a_r += (
        2.5
        * (
            r[:, :, None] * grad_contraction[:, None, :]
            + contraction[:, None, None] * identity
        )
        / radius[:, None, None] ** 7
    )
    grad_a_r -= (
        17.5
        * contraction[:, None, None]
        * (r[:, :, None] * r[:, None, :])
        / radius[:, None, None] ** 9
    )
    gradient = -grad_a_r
    return GFreeObservables(potential, acceleration, gradient)


def _box_nodes(bounds_m: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    axes = []
    axis_weights = []
    for axis in range(3):
        lo, hi = bounds_m[axis]
        axes.append(0.5 * (hi - lo) * nodes + 0.5 * (hi + lo))
        axis_weights.append(0.5 * (hi - lo) * weights)
    grid = np.meshgrid(*axes, indexing="ij")
    weight_grid = np.meshgrid(*axis_weights, indexing="ij")
    points = np.stack(grid, axis=-1).reshape(-1, 3)
    volume_weights = np.prod(np.stack(weight_grid, axis=-1), axis=-1).reshape(-1)
    return points, volume_weights


def _density_values(density: Density, points: np.ndarray) -> np.ndarray:
    raw = density(points) if callable(density) else density
    if _contains_bool(raw):
        raise TypeError("density must not contain boolean values")
    array = np.asarray(raw)
    dtype = np.complex128 if np.iscomplexobj(array) else np.float64
    if array.ndim == 0:
        array = np.full(len(points), array, dtype=dtype)
    else:
        array = np.asarray(array, dtype=dtype)
    if array.shape != (len(points),) or not np.isfinite(array).all():
        raise ValueError("density callback must return a finite array of shape (N,)")
    return array


def _explicit_zero_density(density: Density) -> bool:
    if callable(density):
        return False
    if _contains_bool(density):
        raise TypeError("density must not contain boolean values")
    raw = np.asarray(density)
    return raw.ndim == 0 and bool(raw == 0)


def _require_exterior_targets(box: np.ndarray, targets: np.ndarray) -> None:
    in_or_on = np.all(
        (targets >= box[:, 0][None, :]) & (targets <= box[:, 1][None, :]),
        axis=1,
    )
    indices = np.nonzero(in_or_on)[0]
    if indices.size:
        raise ValueError(
            "targets_m must be exterior to every nonzero material box; "
            f"target {int(indices[0])} lies inside or on the boundary"
        )


def integrate_box_observables_g_free(
    bounds_m: ArrayLike, density: Density, targets_m: ArrayLike, *, order: int = 12
) -> GFreeObservables:
    """Tensor Gauss--Legendre Newton integral over one rectangular box."""
    box = _bounds(bounds_m)
    targets = _xyz_table("targets_m", targets_m)
    n = _order("order", order, 2)
    if _explicit_zero_density(density):
        return _zero(len(targets), complex_output=np.iscomplexobj(np.asarray(density)))
    _require_exterior_targets(box, targets)
    points, volume_weights = _box_nodes(box, n)
    weights = _density_values(density, points) * volume_weights
    return point_mass_observables_g_free(points, weights, targets)


def integrate_displacement_box_observables_g_free(
    bounds_m: ArrayLike,
    density: Density,
    displacement: Displacement,
    targets_m: ArrayLike,
    *,
    order: int = 12,
) -> GFreeObservables:
    """First source-position derivative for a displaced continuous box."""
    box = _bounds(bounds_m)
    targets = _xyz_table("targets_m", targets_m)
    if _explicit_zero_density(density):
        return _zero(len(targets))
    _require_exterior_targets(box, targets)
    points, volume_weights = _box_nodes(box, _order("order", order, 2))
    rho = _density_values(density, points)
    if np.iscomplexobj(rho) or np.any(rho < 0.0):
        raise ValueError(
            "displacement background density must be real and non-negative"
        )
    raw_u = np.asarray(displacement(points))
    dtype = np.complex128 if np.iscomplexobj(raw_u) else np.float64
    u = np.asarray(raw_u, dtype=dtype)
    if u.shape != points.shape or not np.isfinite(u).all():
        raise ValueError("displacement callback must return finite shape (N,3)")
    mass = rho * volume_weights
    delta = points[None, :, :] - targets[:, None, :]
    r2 = np.einsum("mni,mni->mn", delta, delta)
    if np.any(r2 == 0.0):
        raise ValueError("a target collides with a quadrature point")
    radius = np.sqrt(r2)
    dot = np.einsum("mni,ni->mn", delta, u)
    potential = np.einsum("n,mn,mn->m", mass, dot, radius**-3)
    acceleration = np.einsum("n,ni,mn->mi", mass, u, radius**-3)
    acceleration -= 3.0 * np.einsum("n,mni,mn,mn->mi", mass, delta, dot, radius**-5)
    eye = np.eye(3)
    term = (
        3.0
        * (
            u[None, :, :, None] * delta[..., None, :]
            + delta[..., :, None] * u[None, :, None, :]
            + dot[..., None, None] * eye
        )
        / radius[..., None, None] ** 5
    )
    term -= (
        15.0
        * (delta[..., :, None] * delta[..., None, :] * dot[..., None, None])
        / radius[..., None, None] ** 7
    )
    gradient = np.einsum("n,mnij->mij", mass, term)
    return GFreeObservables(potential, acceleration, gradient)


def _norms(value: GFreeObservables) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.abs(value.potential),
        np.linalg.norm(value.acceleration, axis=-1),
        np.linalg.norm(value.gradient, axis=(-2, -1)),
    )


def _normalized_difference(
    low: GFreeObservables,
    high: GFreeObservables,
    physical_scale: GFreeObservables | None,
) -> float:
    scales = _norms(high if physical_scale is None else physical_scale)
    differences = _norms(
        GFreeObservables(
            high.potential - low.potential,
            high.acceleration - low.acceleration,
            high.gradient - low.gradient,
        )
    )
    ratios = []
    for difference, scale in zip(differences, scales, strict=True):
        denominator = np.maximum(scale, np.finfo(np.float64).tiny)
        ratios.append(float(np.max(difference / denominator, initial=0.0)))
    return max(ratios, default=0.0)


def _sum_results(
    results: list[GFreeObservables], target_count: int
) -> GFreeObservables:
    if not results:
        return _zero(target_count)
    total = results[0]
    for result in results[1:]:
        total = _add(total, result)
    return total


def adaptive_box_observables_g_free(
    regions: Iterable[tuple[ArrayLike, Density]],
    targets_m: ArrayLike,
    *,
    low_order: int = 10,
    high_order: int = 12,
    escalation_orders: tuple[int, int] = (16, 24),
    rtol: float = 1e-11,
    max_depth: int = 8,
    max_evaluations: int = 4096,
    physical_scale: GFreeObservables | None = None,
) -> tuple[GFreeObservables, QuadratureConvergence]:
    """Compare Gauss orders, then bisect the largest-error subbox."""
    targets = _xyz_table("targets_m", targets_m)
    low_n, high_n = (
        _order("low_order", low_order, 2),
        _order("high_order", high_order, 2),
    )
    if high_n <= low_n:
        raise ValueError("high_order must exceed low_order")
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError("rtol must be positive and finite")
    budget = _evaluation_budget(max_evaluations)
    leaves = [(_bounds(bounds), density, 0) for bounds, density in regions]
    if not leaves:
        raise ValueError("regions must not be empty")
    evaluations = 0

    def evaluate(order_value: int) -> list[GFreeObservables]:
        nonlocal evaluations
        values = [
            integrate_box_observables_g_free(b, d, targets, order=order_value)
            for b, d, _ in leaves
        ]
        evaluations += len(values)
        return values

    if budget < len(leaves):
        raise ValueError(
            "max_evaluations is too small to evaluate every material region once"
        )
    if budget < 2 * len(leaves):
        high_values = evaluate(high_n)
        high_total = _sum_results(high_values, len(targets))
        return high_total, QuadratureConvergence(
            "ORACLE_UNRESOLVED",
            float("inf"),
            low_n,
            high_n,
            len(leaves),
            0,
            evaluations,
        )

    low_values, high_values = evaluate(low_n), evaluate(high_n)
    low_total = _sum_results(low_values, len(targets))
    high_total = _sum_results(high_values, len(targets))
    difference = _normalized_difference(low_total, high_total, physical_scale)
    if difference <= rtol:
        return high_total, QuadratureConvergence(
            "CONVERGED", difference, low_n, high_n, len(leaves), 0, evaluations
        )

    next_low, next_high = (_order("escalation order", n, 2) for n in escalation_orders)
    if next_high <= next_low:
        raise ValueError("escalation high order must exceed low order")
    if evaluations + 2 * len(leaves) > budget:
        return high_total, QuadratureConvergence(
            "ORACLE_UNRESOLVED",
            difference,
            low_n,
            high_n,
            len(leaves),
            0,
            evaluations,
        )
    low_n, high_n = next_low, next_high
    low_values, high_values = evaluate(low_n), evaluate(high_n)

    while True:
        low_total = _sum_results(low_values, len(targets))
        high_total = _sum_results(high_values, len(targets))
        difference = _normalized_difference(low_total, high_total, physical_scale)
        depth_reached = max(depth for _, _, depth in leaves)
        if difference <= rtol:
            status = "CONVERGED"
            break
        candidates = [i for i, (_, _, depth) in enumerate(leaves) if depth < max_depth]
        if not candidates or evaluations + 4 > budget:
            status = "ORACLE_UNRESOLVED"
            break
        local_errors = [
            _normalized_difference(low_values[i], high_values[i], physical_scale)
            for i in candidates
        ]
        index = candidates[int(np.argmax(local_errors))]
        bounds, density, depth = leaves[index]
        axis = int(np.argmax(bounds[:, 1] - bounds[:, 0]))
        midpoint = 0.5 * (bounds[axis, 0] + bounds[axis, 1])
        left, right = bounds.copy(), bounds.copy()
        left[axis, 1], right[axis, 0] = midpoint, midpoint
        replacement = [(left, density, depth + 1), (right, density, depth + 1)]
        leaves[index : index + 1] = replacement
        new_low = [
            integrate_box_observables_g_free(b, d, targets, order=low_n)
            for b, d, _ in replacement
        ]
        new_high = [
            integrate_box_observables_g_free(b, d, targets, order=high_n)
            for b, d, _ in replacement
        ]
        evaluations += 4
        low_values[index : index + 1] = new_low
        high_values[index : index + 1] = new_high

    evidence = QuadratureConvergence(
        status, difference, low_n, high_n, len(leaves), depth_reached, evaluations
    )
    return high_total, evidence


def _adaptive_displacement_box_core(
    regions: Iterable[tuple[ArrayLike, Density, Displacement]],
    targets_m: ArrayLike,
    *,
    low_order: int = 10,
    high_order: int = 12,
    escalation_orders: tuple[int, int] = (16, 24),
    rtol: float = 1e-11,
    max_depth: int = 8,
    max_evaluations: int = 4096,
    physical_scale: GFreeObservables | None = None,
) -> tuple[GFreeObservables, QuadratureConvergence]:
    """Adaptive displacement core for already panelled regions."""
    targets = _xyz_table("targets_m", targets_m)
    low_n = _order("low_order", low_order, 2)
    high_n = _order("high_order", high_order, 2)
    if high_n <= low_n:
        raise ValueError("high_order must exceed low_order")
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError("rtol must be positive and finite")
    budget = _evaluation_budget(max_evaluations)
    leaves = [
        (_bounds(bounds), density, displacement, 0)
        for bounds, density, displacement in regions
    ]
    if not leaves:
        raise ValueError("regions must not be empty")
    evaluations = 0

    def evaluate(order_value: int) -> list[GFreeObservables]:
        nonlocal evaluations
        values = [
            integrate_displacement_box_observables_g_free(
                b, d, u, targets, order=order_value
            )
            for b, d, u, _ in leaves
        ]
        evaluations += len(values)
        return values

    if budget < len(leaves):
        raise ValueError(
            "max_evaluations is too small to evaluate every material region once"
        )
    if budget < 2 * len(leaves):
        high_values = evaluate(high_n)
        high_total = _sum_results(high_values, len(targets))
        return high_total, QuadratureConvergence(
            "ORACLE_UNRESOLVED",
            float("inf"),
            low_n,
            high_n,
            len(leaves),
            0,
            evaluations,
        )

    low_values, high_values = evaluate(low_n), evaluate(high_n)
    low_total = _sum_results(low_values, len(targets))
    high_total = _sum_results(high_values, len(targets))
    difference = _normalized_difference(low_total, high_total, physical_scale)
    if difference <= rtol:
        return high_total, QuadratureConvergence(
            "CONVERGED", difference, low_n, high_n, len(leaves), 0, evaluations
        )

    next_low, next_high = (_order("escalation order", n, 2) for n in escalation_orders)
    if next_high <= next_low:
        raise ValueError("escalation high order must exceed low order")
    if evaluations + 2 * len(leaves) > budget:
        return high_total, QuadratureConvergence(
            "ORACLE_UNRESOLVED",
            difference,
            low_n,
            high_n,
            len(leaves),
            0,
            evaluations,
        )
    low_n, high_n = next_low, next_high
    low_values, high_values = evaluate(low_n), evaluate(high_n)
    while True:
        low_total = _sum_results(low_values, len(targets))
        high_total = _sum_results(high_values, len(targets))
        difference = _normalized_difference(low_total, high_total, physical_scale)
        depth_reached = max(depth for _, _, _, depth in leaves)
        if difference <= rtol:
            status = "CONVERGED"
            break
        candidates = [i for i, (*_, depth) in enumerate(leaves) if depth < max_depth]
        if not candidates or evaluations + 4 > budget:
            status = "ORACLE_UNRESOLVED"
            break
        local_errors = [
            _normalized_difference(low_values[i], high_values[i], physical_scale)
            for i in candidates
        ]
        index = candidates[int(np.argmax(local_errors))]
        bounds, density, displacement, depth = leaves[index]
        axis = int(np.argmax(bounds[:, 1] - bounds[:, 0]))
        midpoint = 0.5 * (bounds[axis, 0] + bounds[axis, 1])
        left, right = bounds.copy(), bounds.copy()
        left[axis, 1], right[axis, 0] = midpoint, midpoint
        replacement = [
            (left, density, displacement, depth + 1),
            (right, density, displacement, depth + 1),
        ]
        leaves[index : index + 1] = replacement
        new_low = [
            integrate_displacement_box_observables_g_free(b, d, u, targets, order=low_n)
            for b, d, u, _ in replacement
        ]
        new_high = [
            integrate_displacement_box_observables_g_free(
                b, d, u, targets, order=high_n
            )
            for b, d, u, _ in replacement
        ]
        evaluations += 4
        low_values[index : index + 1] = new_low
        high_values[index : index + 1] = new_high
    return high_total, QuadratureConvergence(
        status, difference, low_n, high_n, len(leaves), depth_reached, evaluations
    )


def _partition_host(host: np.ndarray, inner: np.ndarray) -> list[np.ndarray]:
    if np.any(inner[:, 0] <= host[:, 0]) or np.any(inner[:, 1] >= host[:, 1]):
        raise ValueError("inclusion bounds must lie strictly inside host bounds")
    boxes = []
    for lo, hi in ((host[0, 0], inner[0, 0]), (inner[0, 1], host[0, 1])):
        box = host.copy()
        box[0] = (lo, hi)
        boxes.append(box)
    for lo, hi in ((host[1, 0], inner[1, 0]), (inner[1, 1], host[1, 1])):
        box = host.copy()
        box[0] = inner[0]
        box[1] = (lo, hi)
        boxes.append(box)
    for lo, hi in ((host[2, 0], inner[2, 0]), (inner[2, 1], host[2, 1])):
        box = host.copy()
        box[0] = inner[0]
        box[1] = inner[1]
        box[2] = (lo, hi)
        boxes.append(box)
    return boxes


def _geometric_target_partition(
    box: np.ndarray, target: np.ndarray
) -> list[np.ndarray]:
    """Panel a box around one external target without overlapping volume."""
    outside_distance = np.maximum(box[:, 0] - target, 0.0) + np.maximum(
        target - box[:, 1], 0.0
    )
    distance = float(np.linalg.norm(outside_distance))
    if distance == 0.0:
        # Host/inclusion callers only use vacuum-side targets.
        raise ValueError("target must not lie in a material region")
    edges = []
    for axis in range(3):
        lo, hi = box[axis]
        coordinate = target[axis]
        candidates = [lo, hi]
        offsets = distance * np.array((1.0, 4.0, 16.0))
        if coordinate < lo:
            candidates.extend(lo + offsets)
        elif coordinate > hi:
            candidates.extend(hi - offsets)
        else:
            candidates.extend((coordinate - offsets).tolist())
            candidates.append(coordinate)
            candidates.extend((coordinate + offsets).tolist())
        axis_edges = np.unique(np.clip(candidates, lo, hi))
        edges.append(axis_edges)
    panels = []
    for ix in range(len(edges[0]) - 1):
        for iy in range(len(edges[1]) - 1):
            for iz in range(len(edges[2]) - 1):
                panels.append(
                    np.array(
                        (
                            (edges[0][ix], edges[0][ix + 1]),
                            (edges[1][iy], edges[1][iy + 1]),
                            (edges[2][iz], edges[2][iz + 1]),
                        )
                    )
                )
    return panels


def adaptive_displacement_box_observables_g_free(
    regions: Iterable[tuple[ArrayLike, Density, Displacement]],
    targets_m: ArrayLike,
    *,
    low_order: int = 10,
    high_order: int = 12,
    escalation_orders: tuple[int, int] = (16, 24),
    rtol: float = 1e-11,
    max_depth: int = 8,
    max_evaluations: int = 4096,
    physical_scale: GFreeObservables | None = None,
) -> tuple[GFreeObservables, QuadratureConvergence]:
    """Target-panelled adaptive displacement reference.

    Material regions stay non-overlapping.  Each target is evaluated
    separately after geometric panel refinement, which resolves near-boundary
    kernels without flattening or silently merging the target axis.
    """
    targets = _xyz_table("targets_m", targets_m)
    total_budget = _evaluation_budget(max_evaluations)
    base_regions = [
        (_bounds(bounds), density, displacement)
        for bounds, density, displacement in regions
    ]
    if not base_regions:
        raise ValueError("regions must not be empty")
    results = []
    evidences = []
    target_region_sets = [
        [
            (panel, density, displacement)
            for box, density, displacement in base_regions
            for panel in _geometric_target_partition(box, target)
        ]
        for target in targets
    ]
    remaining_budget = total_budget
    for index, (target, target_regions) in enumerate(
        zip(targets, target_region_sets, strict=True)
    ):
        future_minimum = sum(
            2 * len(items) for items in target_region_sets[index + 1 :]
        )
        target_budget = remaining_budget - future_minimum
        if target_budget < len(target_regions):
            raise ValueError(
                "max_evaluations is too small to cover every target/material panel"
            )
        scale = None
        if physical_scale is not None:
            scale = GFreeObservables(
                physical_scale.potential[index : index + 1],
                physical_scale.acceleration[index : index + 1],
                physical_scale.gradient[index : index + 1],
            )
        value, evidence = _adaptive_displacement_box_core(
            target_regions,
            target[None, :],
            low_order=low_order,
            high_order=high_order,
            escalation_orders=escalation_orders,
            rtol=rtol,
            max_depth=max_depth,
            max_evaluations=target_budget,
            physical_scale=scale,
        )
        results.append(value)
        evidences.append(evidence)
        remaining_budget -= evidence.evaluations
    combined = GFreeObservables(
        np.concatenate([value.potential for value in results]),
        np.concatenate([value.acceleration for value in results]),
        np.concatenate([value.gradient for value in results]),
    )
    worst = max(evidences, key=lambda item: item.normalized_difference)
    status = (
        "CONVERGED"
        if all(item.status == "CONVERGED" for item in evidences)
        else "ORACLE_UNRESOLVED"
    )
    return combined, QuadratureConvergence(
        status=status,
        normalized_difference=worst.normalized_difference,
        low_order=worst.low_order,
        high_order=worst.high_order,
        subbox_count=sum(item.subbox_count for item in evidences),
        max_depth_reached=max(item.max_depth_reached for item in evidences),
        evaluations=sum(item.evaluations for item in evidences),
    )


def host_inclusion_observables_g_free(
    host_bounds_m: ArrayLike,
    inclusion_bounds_m: ArrayLike,
    host_density_kg_m3: float,
    inclusion_density_kg_m3: float,
    targets_m: ArrayLike,
    **adaptive_options,
) -> tuple[GFreeObservables, QuadratureConvergence]:
    """Non-overlapping host plus cavity/inclusion reference."""
    host, inner = _bounds(host_bounds_m), _bounds(inclusion_bounds_m)
    host_density = _positive("host_density_kg_m3", host_density_kg_m3)
    inclusion_density = float(inclusion_density_kg_m3)
    if not np.isfinite(inclusion_density) or inclusion_density < 0.0:
        raise ValueError("inclusion_density_kg_m3 must be finite and non-negative")
    base_regions = [(box, host_density) for box in _partition_host(host, inner)]
    if inclusion_density != 0.0:
        base_regions.append((inner, inclusion_density))
    targets = _xyz_table("targets_m", targets_m)
    total_budget = _evaluation_budget(adaptive_options.pop("max_evaluations", 4096))
    results = []
    evidences = []
    supplied_scale = adaptive_options.pop("physical_scale", None)
    target_region_sets = [
        [
            (panel, density)
            for box, density in base_regions
            for panel in _geometric_target_partition(box, target)
        ]
        for target in targets
    ]
    remaining_budget = total_budget
    for index, (target, regions) in enumerate(
        zip(targets, target_region_sets, strict=True)
    ):
        future_minimum = sum(
            2 * len(items) for items in target_region_sets[index + 1 :]
        )
        target_budget = remaining_budget - future_minimum
        if target_budget < len(regions):
            raise ValueError(
                "max_evaluations is too small to cover every target/material panel"
            )
        scale = None
        if supplied_scale is not None:
            scale = GFreeObservables(
                supplied_scale.potential[index : index + 1],
                supplied_scale.acceleration[index : index + 1],
                supplied_scale.gradient[index : index + 1],
            )
        value, evidence = adaptive_box_observables_g_free(
            regions,
            target[None, :],
            physical_scale=scale,
            max_evaluations=target_budget,
            **adaptive_options,
        )
        results.append(value)
        evidences.append(evidence)
        remaining_budget -= evidence.evaluations
    combined = GFreeObservables(
        np.concatenate([value.potential for value in results]),
        np.concatenate([value.acceleration for value in results]),
        np.concatenate([value.gradient for value in results]),
    )
    worst = max(evidences, key=lambda item: item.normalized_difference)
    status = (
        "CONVERGED"
        if all(item.status == "CONVERGED" for item in evidences)
        else "ORACLE_UNRESOLVED"
    )
    evidence = QuadratureConvergence(
        status=status,
        normalized_difference=worst.normalized_difference,
        low_order=worst.low_order,
        high_order=worst.high_order,
        subbox_count=sum(item.subbox_count for item in evidences),
        max_depth_reached=max(item.max_depth_reached for item in evidences),
        evaluations=sum(item.evaluations for item in evidences),
    )
    return combined, evidence


def solid_cylinder_observables_g_free(
    centers_m: ArrayLike,
    radius_m: float,
    length_m: float,
    density_kg_m3: float,
    targets_m: ArrayLike,
    *,
    angles_rad: ArrayLike | None = None,
    radial_order: int = 16,
    azimuthal_order: int = 64,
    axial_order: int = 16,
) -> GFreeObservables:
    """Continuous z-aligned solid-cylinder integral in cylindrical coordinates."""
    centers = _xyz_table("centers_m", centers_m)
    targets = _xyz_table("targets_m", targets_m)
    radius, length, density = (
        _positive("radius_m", radius_m),
        _positive("length_m", length_m),
        _positive("density_kg_m3", density_kg_m3),
    )
    nr, nt, nz = (
        _order("radial_order", radial_order, 2),
        _order("azimuthal_order", azimuthal_order, 4),
        _order("axial_order", axial_order, 2),
    )
    if angles_rad is None:
        angles = np.zeros(len(centers))
    else:
        angles = _real_array("angles_rad", angles_rad)
        if angles.shape != (len(centers),):
            raise ValueError("angles_rad must have shape (number_of_centers,)")
    nodes_r, weights_r = np.polynomial.legendre.leggauss(nr)
    radial_squared = 0.5 * radius**2 * (nodes_r + 1.0)
    radii = np.sqrt(radial_squared)
    # u=r**2 maps [-1,1] to [0,R**2], and r*dr=du/2.
    radial_weights = 0.25 * radius**2 * weights_r
    theta = 2.0 * np.pi * np.arange(nt) / nt
    nodes_z, weights_z = np.polynomial.legendre.leggauss(nz)
    z = 0.5 * length * nodes_z
    z_weights = 0.5 * length * weights_z
    rr, tt, zz = np.meshgrid(radii, theta, z, indexing="ij")
    wr, wt, wz = np.meshgrid(
        radial_weights, np.ones(nt) * (2 * np.pi / nt), z_weights, indexing="ij"
    )
    local = np.stack((rr * np.cos(tt), rr * np.sin(tt), zz), axis=-1).reshape(-1, 3)
    local_weights = density * (wr * wt * wz).reshape(-1)
    points, weights = [], []
    for center, angle in zip(centers, angles, strict=True):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
        points.append(local @ rotation.T + center)
        weights.append(local_weights)
    return point_mass_observables_g_free(
        np.concatenate(points), np.concatenate(weights), targets
    )


def opposed_rotor_observables_g_free(
    angles_rad: ArrayLike,
    targets_m: ArrayLike,
    *,
    radial_order: int = 16,
    azimuthal_order: int = 64,
    axial_order: int = 16,
) -> GFreeObservables:
    """Fixed opposed-cylinder benchmark, with angle as explicit leading axis."""
    angles = _real_array("angles_rad", angles_rad)
    if angles.ndim != 1:
        raise ValueError("angles_rad must be a finite one-dimensional array")
    results = []
    for angle in angles:
        centers = np.array(((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)))
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
        centers = centers @ rotation.T
        results.append(
            solid_cylinder_observables_g_free(
                centers,
                0.1,
                0.2,
                7800.0,
                targets_m,
                radial_order=radial_order,
                azimuthal_order=azimuthal_order,
                axial_order=axial_order,
            )
        )
    return _stack(results)


def opposed_rotor_leading_phasor_g_free(targets_m: ArrayLike) -> GFreeObservables:
    """Analytic 2-omega multipole phasor for the fixed opposed rotor."""
    total_mass = 2.0 * 7800.0 * np.pi * 0.1**2 * 0.2
    anisotropy = total_mass * 0.5**2
    phasor = (
        0.5
        * anisotropy
        * np.array(
            ((1.0, 1.0j, 0.0), (1.0j, -1.0, 0.0), (0.0, 0.0, 0.0)),
            dtype=np.complex128,
        )
    )
    return multipole_observables_g_free([0.0, 0.0, 0.0], 0.0, phasor, targets_m)


def passing_cuboid_observables_g_free(
    times_s: ArrayLike, targets_m: ArrayLike, *, order: int = 24
) -> GFreeObservables:
    """Fixed passing 2 x 1 x 1 m, 2700 kg/m3 cuboid time series."""
    times = _real_array("times_s", times_s)
    if times.ndim != 1:
        raise ValueError("times_s must be a finite one-dimensional array")
    local = np.array(((-1.0, 1.0), (-0.5, 0.5), (-0.5, 0.5)))
    results = []
    for time in times:
        center = np.array((10.0 * time, 5.0, 0.0))
        results.append(
            integrate_box_observables_g_free(
                local + center[:, None], 2700.0, targets_m, order=order
            )
        )
    return _stack(results)


def acoustic_finite_box_observables_g_free(
    bounds_m: ArrayLike,
    targets_m: ArrayLike,
    *,
    pressure_pa: float = 1.0,
    sound_speed_m_s: float = 343.0,
    frequency_hz: float = 10.0,
    phase_origin_m: ArrayLike = (0.0, 0.0, 0.0),
    order: int = 24,
) -> GFreeObservables:
    """Finite-air-box density phasor ``delta rho=p/c_s^2 exp(i k x)``."""
    box = _bounds(bounds_m)
    targets = _xyz_table("targets_m", targets_m)
    if box[2, 0] < 0.0:
        raise ValueError("acoustic air box must lie at z >= 0")
    if np.any(targets[:, 2] >= 0.0):
        raise ValueError("acoustic targets must be below the fixed ground (z < 0)")
    pressure = float(pressure_pa)
    if not np.isfinite(pressure):
        raise ValueError("pressure_pa must be finite")
    sound_speed, frequency = (
        _positive("sound_speed_m_s", sound_speed_m_s),
        _positive("frequency_hz", frequency_hz),
    )
    origin = _xyz_vector("phase_origin_m", phase_origin_m)
    k = 2.0 * np.pi * frequency / sound_speed
    density_amplitude = pressure / sound_speed**2
    return integrate_box_observables_g_free(
        box,
        lambda points: density_amplitude * np.exp(1j * k * (points[:, 0] - origin[0])),
        targets,
        order=order,
    )


def acoustic_infinite_halfspace_observables_g_free(
    targets_m: ArrayLike,
    *,
    pressure_pa: float = 1.0,
    sound_speed_m_s: float = 343.0,
    frequency_hz: float = 10.0,
    phase_origin_m: ArrayLike = (0.0, 0.0, 0.0),
) -> GFreeObservables:
    """Closed form for horizontally infinite, vertically semi-infinite air."""
    targets = _xyz_table("targets_m", targets_m)
    if np.any(targets[:, 2] >= 0.0):
        raise ValueError("acoustic targets must be below the fixed ground (z < 0)")
    pressure = float(pressure_pa)
    if not np.isfinite(pressure):
        raise ValueError("pressure_pa must be finite")
    sound_speed, frequency = (
        _positive("sound_speed_m_s", sound_speed_m_s),
        _positive("frequency_hz", frequency_hz),
    )
    origin = _xyz_vector("phase_origin_m", phase_origin_m)
    k = 2.0 * np.pi * frequency / sound_speed
    rho_hat = pressure / sound_speed**2
    phase = np.exp(1j * k * (targets[:, 0] - origin[0]) + k * targets[:, 2])
    potential = -(2.0 * np.pi * rho_hat / k**2) * phase
    derivative = np.array((1j * k, 0.0, k), dtype=np.complex128)
    acceleration = -potential[:, None] * derivative
    gradient = -potential[:, None, None] * np.outer(derivative, derivative)
    return GFreeObservables(potential, acceleration, gradient)


def _rayleigh_root(vp: float, vs: float) -> tuple[float, float, float]:
    if not (np.isfinite(vp) and np.isfinite(vs) and vp > np.sqrt(4.0 / 3.0) * vs > 0.0):
        raise ValueError(
            "Rayleigh reference requires a stable isotropic material with "
            "v_p/v_s > sqrt(4/3)"
        )
    lo, hi = 0.5 * vs, (1.0 - 1e-12) * vs

    def secular(c: float) -> float:
        return (2.0 - (c / vs) ** 2) ** 2 - 4.0 * np.sqrt(
            1.0 - (c / vp) ** 2
        ) * np.sqrt(1.0 - (c / vs) ** 2)

    flo = secular(lo)
    fhi = secular(hi)
    if flo * fhi > 0.0:
        raise ValueError("Rayleigh secular equation has no nonzero root bracket")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fm = secular(mid)
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    speed = 0.5 * (lo + hi)
    residual = secular(speed)
    if abs(residual) > 1e-13:
        raise RuntimeError(f"Rayleigh root residual is too large: {residual!r}")
    return (
        speed,
        float(np.sqrt(1.0 - (speed / vp) ** 2)),
        float(np.sqrt(1.0 - (speed / vs) ** 2)),
    )


def rayleigh_halfspace_observables_g_free(
    targets_m: ArrayLike,
    *,
    density_kg_m3: float = 2300.0,
    v_p_m_s: float = 2800.0,
    v_s_m_s: float = 1500.0,
    frequency_hz: float = 2.5,
    vertical_surface_amplitude_m: float = 1e-9,
    direction: ArrayLike = (1.0, 0.0, 0.0),
    phase_origin_m: ArrayLike = (0.0, 0.0, 0.0),
) -> GFreeObservables:
    """Bulk-plus-sheet Rayleigh gravity above a homogeneous half-space."""
    targets = _xyz_table("targets_m", targets_m)
    if np.any(targets[:, 2] <= 0.0):
        raise ValueError(
            "Rayleigh targets must be above the elastic half-space (z > 0)"
        )
    rho, vp, vs = (
        _positive("density_kg_m3", density_kg_m3),
        _positive("v_p_m_s", v_p_m_s),
        _positive("v_s_m_s", v_s_m_s),
    )
    if vs >= vp:
        raise ValueError("v_s_m_s must be less than v_p_m_s")
    if vp <= np.sqrt(4.0 / 3.0) * vs:
        raise ValueError(
            "Rayleigh reference requires a stable isotropic material with "
            "v_p/v_s > sqrt(4/3)"
        )
    frequency = _positive("frequency_hz", frequency_hz)
    amplitude = _positive("vertical_surface_amplitude_m", vertical_surface_amplitude_m)
    direction_array = _xyz_vector("direction", direction)
    if abs(direction_array[2]) > 1e-14 or np.linalg.norm(direction_array[:2]) == 0.0:
        raise ValueError("direction must be nonzero and horizontal")
    direction_array /= np.linalg.norm(direction_array)
    origin = _xyz_vector("phase_origin_m", phase_origin_m)
    speed, q, s = _rayleigh_root(vp, vs)
    k = 2.0 * np.pi * frequency / speed
    fz0 = 1.0 - 2.0 / (1.0 + s**2)
    potential_amplitude = amplitude / (q * k * fz0)
    bulk_density_amplitude = rho * k**2 * potential_amplitude * (1.0 - q**2)
    scalar_coefficient = rho * amplitude / k + bulk_density_amplitude / (
        k**2 * (1.0 + q)
    )
    horizontal_phase = (targets - origin) @ direction_array
    phase = np.exp(1j * k * horizontal_phase - k * targets[:, 2])
    potential = -2.0 * np.pi * scalar_coefficient * phase
    derivative = k * np.array(
        (1j * direction_array[0], 1j * direction_array[1], -1.0), dtype=np.complex128
    )
    acceleration = -potential[:, None] * derivative
    gradient = -potential[:, None, None] * np.outer(derivative, derivative)
    return GFreeObservables(potential, acceleration, gradient)
