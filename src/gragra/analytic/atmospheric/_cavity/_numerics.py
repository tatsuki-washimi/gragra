"""Shared adaptive evaluation and evidence container for cavity transforms.

The kernels reduce the spatial problem analytically and pass only a one
dimensional transform here.  Component errors are estimates from successive
Gaussian rules plus a floating-point floor; truncation tails remain separate
and must be supplied by a shape-specific outward bound.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Literal

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import roots_legendre

TransformTermination = Literal["converged", "resource_limit", "uncertainty_unresolved"]
Acceptance = Callable[[np.ndarray, np.ndarray, np.ndarray], bool]
Integrand = Callable[[float], np.ndarray]
TailBound = Callable[[float], np.ndarray | None]
_ROUNDOFF_FACTOR = 16.0
_INITIAL_RULE_ORDERS = (4, 8, 16)
_PROMOTED_INITIAL_ORDERS = (8, 16, 32)
_DEFAULT_REFINED_ORDERS = _PROMOTED_INITIAL_ORDERS
_MAX_REFINED_ORDER = 64
_Q2_FIXED_BAND = (138.4615384615381, 415.38461538461434)
_Q2_BAND_TOLERANCE = 2e-10

# GK21 abscissae and Kronrod weights transcribed from SciPy 1.17.1
# scipy/integrate/_quad_vec.py::_quadrature_gk21; SciPy is BSD-3-Clause licensed.
_GK21_NODES = (
    0.995657163025808080735527280689003,
    0.973906528517171720077964012084452,
    0.930157491355708226001207180059508,
    0.865063366688984510732096688423493,
    0.780817726586416897063717578345042,
    0.679409568299024406234327365114874,
    0.562757134668604683339000099272694,
    0.433395394129247190799265943165784,
    0.294392862701460198131126603103866,
    0.148874338981631210884826001129720,
    0.0,
    -0.148874338981631210884826001129720,
    -0.294392862701460198131126603103866,
    -0.433395394129247190799265943165784,
    -0.562757134668604683339000099272694,
    -0.679409568299024406234327365114874,
    -0.780817726586416897063717578345042,
    -0.865063366688984510732096688423493,
    -0.930157491355708226001207180059508,
    -0.973906528517171720077964012084452,
    -0.995657163025808080735527280689003,
)
_GK21_WEIGHTS = (
    0.011694638867371874278064396062192,
    0.032558162307964727478818972459390,
    0.054755896574351996031381300244580,
    0.075039674810919952767043140916190,
    0.093125454583697605535065465083366,
    0.109387158802297641899210590325805,
    0.123491976262065851077958109831074,
    0.134709217311473325928054001771707,
    0.142775938577060080797094273138717,
    0.147739104901338491374841515972068,
    0.149445554002916905664936468389821,
    0.147739104901338491374841515972068,
    0.142775938577060080797094273138717,
    0.134709217311473325928054001771707,
    0.123491976262065851077958109831074,
    0.109387158802297641899210590325805,
    0.093125454583697605535065465083366,
    0.075039674810919952767043140916190,
    0.054755896574351996031381300244580,
    0.032558162307964727478818972459390,
    0.011694638867371874278064396062192,
)


def _readonly_vector(
    value: object, name: str, *, optional: bool, complex_ok: bool = False
) -> np.ndarray | None:
    if value is None and optional:
        return None
    raw = np.asarray(value)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"{name} must be a one-dimensional numeric array")
    if np.iscomplexobj(raw) and not complex_ok:
        raise TypeError(f"{name} must be real")
    owned = np.asarray(raw, dtype=np.complex128 if complex_ok else np.float64).copy()
    if not np.isfinite(owned).all() or (
        not np.iscomplexobj(owned) and np.any(owned < 0.0)
    ):
        raise ValueError(f"{name} must contain finite nonnegative values")
    return np.frombuffer(owned.tobytes(), dtype=owned.dtype).reshape(owned.shape)


@dataclass(frozen=True, slots=True, init=False)
class TransformEstimate:
    """Immutable integral estimate; unavailable evidence is represented by None."""

    values: np.ndarray | None
    component_errors: np.ndarray | None
    tail_bounds: np.ndarray | None
    evaluations: int
    refinements: int
    cutoff: float | None
    termination: TransformTermination

    def __init__(
        self,
        *,
        values: object,
        component_errors: object,
        tail_bounds: object,
        evaluations: int,
        refinements: int,
        cutoff: float | None,
        termination: TransformTermination,
    ) -> None:
        arrays = {
            "values": _readonly_vector(
                values, "values", optional=True, complex_ok=True
            ),
            "component_errors": _readonly_vector(
                component_errors, "component_errors", optional=True
            ),
            "tail_bounds": _readonly_vector(tail_bounds, "tail_bounds", optional=True),
        }
        present = [array for array in arrays.values() if array is not None]
        if present and any(array.shape != present[0].shape for array in present[1:]):
            raise ValueError("available transform arrays must have matching shapes")
        if (
            isinstance(evaluations, (bool, np.bool_))
            or not isinstance(evaluations, (int, np.integer))
            or evaluations < 0
        ):
            raise ValueError("evaluations must be a nonnegative integer")
        if (
            isinstance(refinements, (bool, np.bool_))
            or not isinstance(refinements, (int, np.integer))
            or refinements < 0
        ):
            raise ValueError("refinements must be a nonnegative integer")
        if cutoff is not None and (not np.isfinite(cutoff) or cutoff <= 0.0):
            raise ValueError("cutoff must be finite and positive or None")
        if termination not in {
            "converged",
            "resource_limit",
            "uncertainty_unresolved",
        }:
            raise ValueError("unsupported transform termination")
        if termination == "converged" and any(
            arrays[name] is None
            for name in ("values", "component_errors", "tail_bounds")
        ):
            raise ValueError("converged estimates require values, errors, and tails")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "evaluations", int(evaluations))
        object.__setattr__(self, "refinements", int(refinements))
        object.__setattr__(self, "cutoff", None if cutoff is None else float(cutoff))
        object.__setattr__(self, "termination", termination)


def _failure_estimate(
    *,
    values: np.ndarray | None,
    errors: np.ndarray | None,
    tails: np.ndarray | None,
    evaluations: int,
    refinements: int,
    cutoff: float,
    termination: TransformTermination,
) -> TransformEstimate:
    return TransformEstimate(
        values=values,
        component_errors=errors,
        tail_bounds=tails,
        evaluations=evaluations,
        refinements=refinements,
        cutoff=cutoff,
        termination=termination,
    )


@dataclass(frozen=True, slots=True)
class _PanelEstimate:
    """Embedded Gauss estimates and rule stage for one transform interval."""

    left: float
    right: float
    values: np.ndarray
    absolute_sum: np.ndarray
    errors: np.ndarray
    orders: tuple[int, ...]
    is_initial: bool


def _fsum_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    """Accurately sum same-shaped real or complex arrays componentwise."""
    if not arrays:
        raise ValueError("at least one array is required")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("summed arrays must have matching shapes")
    flattened = [array.reshape(-1) for array in arrays]
    size = flattened[0].size
    if any(np.iscomplexobj(array) for array in arrays):
        real = np.fromiter(
            (
                math.fsum(float(array[index].real) for array in flattened)
                for index in range(size)
            ),
            dtype=np.float64,
            count=size,
        )
        imag = np.fromiter(
            (
                math.fsum(float(array[index].imag) for array in flattened)
                for index in range(size)
            ),
            dtype=np.float64,
            count=size,
        )
        return (real + 1j * imag).reshape(shape)
    return np.fromiter(
        (
            math.fsum(float(array[index]) for array in flattened)
            for index in range(size)
        ),
        dtype=np.float64,
        count=size,
    ).reshape(shape)


def _gk21_absolute_sum(
    samples: dict[float, np.ndarray],
    left: float,
    right: float,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """Reconstruct a GK21 leaf's weighted absolute integral from cached samples."""
    # Match scipy.integrate._quad_vec's c = 0.5 * (a + b), h = 0.5 * (b - a).
    midpoint = 0.5 * (left + right)
    half_width = 0.5 * (right - left)
    weighted_absolute_samples: list[np.ndarray] = []
    for node, weight in zip(_GK21_NODES, _GK21_WEIGHTS, strict=True):
        x = float(midpoint + half_width * node)
        sample = np.asarray(samples[x])
        if (
            sample.shape != expected_shape
            or sample.ndim != 1
            or not np.issubdtype(sample.dtype, np.number)
        ):
            raise ValueError("cached GK21 sample has an invalid shape or value")
        sample = np.asarray(
            sample, dtype=np.complex128 if np.iscomplexobj(sample) else np.float64
        )
        if not np.isfinite(sample).all():
            raise ValueError("cached GK21 sample has an invalid shape or value")
        weighted_absolute_samples.append(half_width * weight * np.abs(sample))
    return _fsum_arrays(weighted_absolute_samples)


class _Q2PanelError(RuntimeError):
    """Unresolved Q2 evidence, with the actual combined callback count."""

    def __init__(self, message: str, evaluations: int) -> None:
        super().__init__(message)
        self.evaluations = evaluations


class _Q2BudgetExhaustedError(_Q2PanelError):
    """The remaining combined-integrand budget cannot fund another call."""


def _q2_phase_seed_points(
    left: float, right: float, phase_length: float
) -> list[float]:
    """Interior points on the global phase grid, bounded by the Q2 leaf cap."""
    if not np.isfinite([left, right, phase_length]).all() or not (
        right > left and phase_length > 0.0
    ):
        raise ValueError("Q2 interval and phase length must be finite and positive")
    step = math.pi / (2.0 * phase_length)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("Q2 phase grid spacing is not representable")
    ratios = (left / step, right / step)
    if not np.isfinite(ratios).all():
        raise ValueError("Q2 phase grid exceeds its leaf limit")
    first, last = math.ceil(ratios[0]), math.floor(ratios[1])
    if last - first + 1 > 4_399:
        raise ValueError("Q2 phase grid exceeds its leaf limit")
    points = sorted({index * step for index in range(first, last + 1)})
    points = [point for point in points if left < point < right]
    if len(points) + 1 > 4_398:
        raise ValueError("Q2 phase grid exceeds its leaf limit")
    return points


def _evaluate_q2_panel(
    integrand: Integrand,
    left: float,
    right: float,
    expected_shape: tuple[int, ...],
    *,
    component_scales: object,
    phase_length: float,
    max_evaluations: int,
    exact_zero_mask: object = None,
    orders: tuple[int, ...] = _INITIAL_RULE_ORDERS,
) -> tuple[_PanelEstimate, int]:
    """Evaluate a seeded 13-component panel using only public GK21 evidence.

    Band selection belongs to the controller. Failures carry callback counts;
    the controller must distinguish resource exhaustion from unresolved evidence.
    Cache reconstruction never calls the combined integrand.
    """
    if expected_shape != (13,):
        raise ValueError("Q2 requires a 13-component transform")
    raw_scales = np.asarray(component_scales)
    if (
        raw_scales.shape != expected_shape
        or not np.issubdtype(raw_scales.dtype, np.number)
        or np.iscomplexobj(raw_scales)
    ):
        raise ValueError("Q2 component scales must be a real 13-vector")
    scales = np.asarray(raw_scales, dtype=np.float64).copy()
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("Q2 component scales must be finite and positive")
    mask = np.zeros(expected_shape, dtype=bool)
    if exact_zero_mask is not None:
        raw_mask = np.asarray(exact_zero_mask)
        if raw_mask.shape != expected_shape or raw_mask.dtype != np.dtype(bool):
            raise ValueError("Q2 exact-zero mask must be a boolean 13-vector")
        mask = raw_mask.copy()
    if (
        isinstance(max_evaluations, (bool, np.bool_))
        or not isinstance(max_evaluations, (int, np.integer))
        or max_evaluations < 0
    ):
        raise ValueError("Q2 callback budget must be a nonnegative integer")
    points = _q2_phase_seed_points(left, right, phase_length)
    samples: dict[float, np.ndarray] = {}
    evaluations = 0

    def scaled_callback(k: float) -> np.ndarray:
        nonlocal evaluations
        if evaluations >= max_evaluations:
            raise _Q2BudgetExhaustedError("Q2 callback budget exhausted", evaluations)
        evaluations += 1
        try:
            raw = np.asarray(integrand(float(k)))
            if (
                raw.shape != expected_shape
                or not np.issubdtype(raw.dtype, np.number)
                or not np.isfinite(raw).all()
            ):
                raise ValueError("expected a finite numeric 13-vector")
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                value = np.asarray(raw, dtype=np.complex128) / scales
            if not np.isfinite(value).all():
                raise ValueError("nonfinite scaled 13-vector")
        except Exception as exc:
            raise _Q2PanelError("Q2 callback failed", evaluations) from exc
        samples[float(k)] = value.copy()
        return value

    try:
        result, error, info = quad_vec(
            scaled_callback,
            float(left),
            float(right),
            points=points,
            quadrature="gk21",
            norm="max",
            epsabs=1e-12,
            epsrel=1e-12,
            cache_size=20_000_000,
            limit=4_398,
            workers=1,
            full_output=True,
        )
        if info.neval != evaluations:
            raise ValueError("Q2 quad_vec neval does not match callback count")
        if not info.success:
            raise ValueError("Q2 quad_vec did not succeed")
        result = np.asarray(result, dtype=np.complex128)
        intervals = np.asarray(info.intervals, dtype=np.float64)
        leaf_errors = np.asarray(info.errors, dtype=np.float64)
        leaf_integrals = np.asarray(info.integrals)
        if (
            result.shape != expected_shape
            or not np.isfinite(result).all()
            or not np.isfinite(error)
            or error < 0.0
            or intervals.ndim != 2
            or intervals.shape[1] != 2
            or not 0 < len(intervals) <= 4_398
            or not np.isfinite(intervals).all()
            or np.any(intervals[:, 1] <= intervals[:, 0])
            or np.any(intervals[:, 0] < left)
            or np.any(intervals[:, 1] > right)
            or leaf_errors.shape != (len(intervals),)
            or not np.isfinite(leaf_errors).all()
            or np.any(leaf_errors < 0.0)
            or leaf_integrals.shape != (len(intervals), 13)
            or not np.isfinite(leaf_integrals).all()
        ):
            raise ValueError("Q2 quad_vec returned invalid evidence")
        leaf_order = np.lexsort((intervals[:, 1], intervals[:, 0]))
        intervals = intervals[leaf_order]
        if (
            intervals[0, 0] != left
            or intervals[-1, 1] != right
            or not np.array_equal(intervals[:-1, 1], intervals[1:, 0])
        ):
            raise ValueError("Q2 leaves do not partition the complete panel")
        leaf_total = _fsum_arrays(list(leaf_integrals[leaf_order]))
        leaf_absolute_total = _fsum_arrays(list(np.abs(leaf_integrals[leaf_order])))
        # A nearly cancelled result cannot set the scale for summation roundoff.
        consistency_scale = np.maximum.reduce(
            [leaf_absolute_total, np.abs(result), np.abs(leaf_total)]
        )
        if not np.isfinite(consistency_scale).all() or np.any(
            np.abs(leaf_total - result) > 3e-13 * consistency_scale + 3e-300
        ):
            raise ValueError("Q2 leaf integrals are inconsistent with public result")
        absolute_scaled = _fsum_arrays(
            [
                _gk21_absolute_sum(samples, float(a), float(b), expected_shape)
                for a, b in intervals
            ]
        )
        with np.errstate(over="ignore", invalid="ignore"):
            values = result * scales
            absolute_sum = absolute_scaled * scales
            envelope = float(error) * scales
            values[mask] = 0.0j
            absolute_sum[mask] = 0.0
            envelope[mask] = 0.0
            # The controller removes this modeled roundoff filter amount.
            errors = envelope + (
                10 * _ROUNDOFF_FACTOR * np.finfo(np.float64).eps * absolute_sum
            )
        if not all(
            np.isfinite(array).all() for array in (values, absolute_sum, errors)
        ):
            raise ValueError("Q2 mapped panel evidence is nonfinite")
    except _Q2PanelError:
        raise
    except Exception as exc:
        raise _Q2PanelError(f"Q2 panel failed: {exc}", evaluations) from exc
    return _PanelEstimate(
        float(left), float(right), values, absolute_sum, errors, tuple(orders), False
    ), evaluations


@cache
def _legendre_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = roots_legendre(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _gauss_legendre_rule(
    integrand: Integrand,
    left: float,
    right: float,
    order: int,
    expected_shape: tuple[int, ...],
) -> tuple[tuple[np.ndarray, np.ndarray] | None, int]:
    nodes, weights = _legendre_rule(order)
    midpoint = left + (right - left) * 0.5
    half_width = (right - left) * 0.5
    weighted_samples: list[np.ndarray] = []
    absolute_samples: list[np.ndarray] = []
    evaluations = 0
    for node, quadrature_weight in zip(nodes, weights, strict=True):
        evaluations += 1
        try:
            sample = np.asarray(integrand(float(midpoint + half_width * node)))
        except (ArithmeticError, FloatingPointError, OverflowError, ValueError):
            return None, evaluations
        if (
            sample.shape != expected_shape
            or sample.ndim != 1
            or not np.issubdtype(sample.dtype, np.number)
            or not np.isfinite(sample).all()
        ):
            return None, evaluations
        weight = half_width * quadrature_weight
        weighted_samples.append(weight * sample)
        absolute_samples.append(abs(weight) * np.abs(sample))
    if not weighted_samples:
        return None, evaluations
    try:
        total = _fsum_arrays(weighted_samples)
        absolute_sum = _fsum_arrays(absolute_samples)
    except (OverflowError, ValueError):
        return None, evaluations
    if not np.isfinite(total).all() or not np.isfinite(absolute_sum).all():
        return None, evaluations
    return (total, absolute_sum), evaluations


def _evaluate_panel(
    integrand: Integrand,
    left: float,
    right: float,
    expected_shape: tuple[int, ...],
    *,
    orders: tuple[int, ...] = _INITIAL_RULE_ORDERS,
    is_initial: bool = True,
) -> tuple[_PanelEstimate | None, int]:
    """Evaluate one panel with successive Gauss-Legendre error evidence."""
    rules: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    evaluations = 0
    for order in orders:
        quadrature, calls = _gauss_legendre_rule(
            integrand, left, right, order, expected_shape
        )
        evaluations += calls
        if quadrature is None:
            return None, evaluations
        rules[order] = quadrature
    value_orders = tuple(rules)
    values, absolute_sum = rules[value_orders[-1]]
    successive_differences = [
        np.abs(rules[right][0] - rules[left][0])
        for left, right in zip(value_orders[:-1], value_orders[1:], strict=True)
    ]
    errors = 10.0 * np.maximum.reduce(successive_differences)
    if not np.isfinite(errors).all():
        return None, evaluations
    return (
        _PanelEstimate(
            left,
            right,
            values,
            absolute_sum,
            errors,
            tuple(orders),
            is_initial,
        ),
        evaluations,
    )


def _initial_panel_bounds(
    cutoff: float, *, origin_subdivisions: int = 0
) -> list[tuple[float, float]]:
    """Cover the interval with geometrically growing panels from the origin."""
    first_right = min(1.0, cutoff)
    if origin_subdivisions > 0:
        powers = range(-origin_subdivisions, 1)
        boundaries = [0.0] + [2.0**p * first_right for p in powers]
    else:
        boundaries = [0.0, first_right]
    right = first_right
    while right < cutoff:
        next_right = min(2.0 * right, cutoff)
        if not next_right > right:
            break
        boundaries.append(next_right)
        right = next_right
    return list(zip(boundaries[:-1], boundaries[1:], strict=True))


def _extension_panel_bounds(left: float, right: float) -> list[tuple[float, float]]:
    """Extend geometric panel coverage without subdividing the old interval."""
    boundaries = [left]
    current = left
    while current < right:
        next_right = min(max(2.0 * current, current + 1.0), right)
        if not next_right > current:
            break
        boundaries.append(next_right)
        current = next_right
    return list(zip(boundaries[:-1], boundaries[1:], strict=True))


def _subdivide_panel_bounds(
    bounds: list[tuple[float, float]],
    max_panel_width: float,
    *,
    panel_limit: int,
) -> tuple[list[tuple[float, float]] | None, TransformTermination | None]:
    """Split base intervals deterministically under a width and count cap."""
    output: list[tuple[float, float]] = []
    for left, right in bounds:
        span = right - left
        ratio = span / max_panel_width
        if not np.isfinite(ratio):
            return None, "resource_limit"
        count = max(1, math.ceil(ratio))
        while True:
            if count > panel_limit - len(output):
                return None, "resource_limit"
            points = [left]
            points.extend(left + span * (index / count) for index in range(1, count))
            points.append(right)
            widths = [
                end - start for start, end in zip(points, points[1:], strict=False)
            ]
            if any(width <= 0.0 for width in widths):
                return None, "uncertainty_unresolved"
            if all(width <= max_panel_width for width in widths):
                output.extend(zip(points[:-1], points[1:], strict=True))
                break
            count += 1
    return output, None


def integrate_transform(
    integrand: Integrand,
    tail_bound: TailBound,
    *,
    initial_cutoff: float,
    component_scales: object,
    acceptance: Acceptance,
    numerics: object,
    max_panel_width: float | None = None,
    refined_orders: tuple[int, int, int] | None = None,
    promote_initial_panels: bool = False,
    origin_subdivisions: int = 0,
    filter_panel_roundoff: bool = False,
    _q2_band: tuple[float, float] | None = None,
    _q2_phase_length: float | None = None,
    _q2_exact_zero_mask: object = None,
) -> TransformEstimate:
    """Integrate a vector transform with separate estimate and rigorous tail.

    The callback ``integrand(t)`` returns one finite vector. ``tail_bound(T)``
    returns a componentwise outward bound for the omitted integral on
    ``[T, infinity)`` or ``None`` when no bound can be represented. Acceptance
    is called only after values, successive-rule errors, and tails are all
    finite arrays. A larger cutoff is tried when the tail currently dominates
    the estimated quadrature error. ``max_panel_width`` optionally splits
    initial and tail-extension intervals before evaluation; adaptive children
    inherit the bound. ``refined_orders`` overrides the three Gauss-Legendre
    rules used for adaptive child panels, with maximum order 64.
    ``promote_initial_panels`` optionally schedules initial-panel promotions
    with (8, 16, 32) alongside bisections, in descending panel-score order;
    each completed mixed round uses one ``max_refinements`` update. Omitting
    this option preserves the default bisection behavior.
    ``origin_subdivisions`` optionally subdivides the interval [0, min(1, cutoff)]
    into dyadic sub-octaves to isolate weak origin singularities.
    ``filter_panel_roundoff`` provides a modeled error estimate that filters out
    the empirical 10x multiplier on floating-point noise from panel rule differences,
    preventing artificial noise inflation across thousands of panels while relying on
    the separate modeled global roundoff term (16 eps * scale) for binary64 accumulation.
    The private Q2 controls select only the approved fixed Cylinder band.
    A matching single band interval is exempt from width preprocessing.
    """
    if not callable(integrand) or not callable(tail_bound) or not callable(acceptance):
        raise TypeError("integrand, tail_bound, and acceptance must be callable")
    if type(promote_initial_panels) is not bool:
        raise TypeError("promote_initial_panels must be a bool")
    if type(filter_panel_roundoff) is not bool:
        raise TypeError("filter_panel_roundoff must be a bool")
    if (
        isinstance(origin_subdivisions, (bool, np.bool_))
        or not isinstance(origin_subdivisions, (int, np.integer))
    ):
        raise TypeError("origin_subdivisions must be an integer")
    if origin_subdivisions < 0:
        raise ValueError("origin_subdivisions must be nonnegative")
    cutoff = float(initial_cutoff)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("initial_cutoff must be finite and positive")
    if max_panel_width is None:
        panel_width = None
    else:
        if isinstance(max_panel_width, (bool, np.bool_)) or not isinstance(
            max_panel_width, (int, float, np.integer, np.floating)
        ):
            raise TypeError("max_panel_width must be a numeric scalar or None")
        panel_width = float(max_panel_width)
        if not np.isfinite(panel_width) or panel_width <= 0.0:
            raise ValueError("max_panel_width must be finite and positive")
    if refined_orders is None:
        refinement_rule_orders = _DEFAULT_REFINED_ORDERS
    else:
        try:
            raw_orders = tuple(refined_orders)
        except TypeError as exc:
            raise ValueError(
                "refined_orders must contain three increasing integer orders"
            ) from exc
        if len(raw_orders) != 3 or any(
            isinstance(order, (bool, np.bool_))
            or not isinstance(order, (int, np.integer))
            for order in raw_orders
        ):
            raise ValueError(
                "refined_orders must contain three increasing integer orders"
            )
        refinement_rule_orders = tuple(int(order) for order in raw_orders)
        if any(
            order <= 0 or order > _MAX_REFINED_ORDER for order in refinement_rule_orders
        ) or not (
            refinement_rule_orders[0]
            < refinement_rule_orders[1]
            < refinement_rule_orders[2]
        ):
            raise ValueError(
                "refined_orders must be three increasing positive orders "
                "no greater than 64"
            )
    scales_raw = np.asarray(component_scales)
    if scales_raw.ndim != 1 or not np.issubdtype(scales_raw.dtype, np.number):
        raise TypeError("component_scales must be a one-dimensional numeric array")
    scales = np.asarray(scales_raw, dtype=np.float64)
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("component_scales must be finite and positive")
    max_evaluations = getattr(numerics, "max_evaluations", None)
    max_refinements = getattr(numerics, "max_refinements", None)
    if (
        isinstance(max_evaluations, (bool, np.bool_))
        or not isinstance(max_evaluations, (int, np.integer))
        or max_evaluations < 1
        or isinstance(max_refinements, (bool, np.bool_))
        or not isinstance(max_refinements, (int, np.integer))
        or max_refinements < 1
    ):
        raise TypeError("numerics must provide positive integer resource limits")

    # Private Cylinder context: only the approved fixed band can enable Q2.
    q2_mask = None
    if _q2_band is None:
        if _q2_phase_length is not None or _q2_exact_zero_mask is not None:
            raise ValueError("Q2 phase and exact-zero context require the fixed band")
    else:
        raw_band = np.asarray(_q2_band)
        if (
            raw_band.shape != (2,)
            or not np.issubdtype(raw_band.dtype, np.number)
            or np.iscomplexobj(raw_band)
            or any(isinstance(value, (bool, np.bool_)) for value in _q2_band)
            or not np.array_equal(raw_band, _Q2_FIXED_BAND)
        ):
            raise ValueError("Q2 band must equal the approved fixed Cylinder band")
        if isinstance(_q2_phase_length, (bool, np.bool_)) or not isinstance(
            _q2_phase_length, (int, float, np.integer, np.floating)
        ):
            raise TypeError("Q2 phase length must be a numeric scalar")
        if not np.isfinite(_q2_phase_length) or _q2_phase_length <= 0.0:
            raise ValueError("Q2 phase length must be finite and positive")
        if scales.shape != (13,):
            raise ValueError("Q2 requires 13 component scales")
        if _q2_exact_zero_mask is not None:
            q2_mask = np.asarray(_q2_exact_zero_mask)
            if q2_mask.shape != (13,) or q2_mask.dtype != np.dtype(bool):
                raise ValueError("Q2 exact-zero mask must be a boolean 13-vector")
            q2_mask = q2_mask.copy()

    evaluations = 0
    refinements = 0
    q2_budget_exhausted = False

    def is_exact_band(left: float, right: float) -> bool:
        return _q2_band is not None and all(
            math.isclose(endpoint, boundary, rel_tol=0.0, abs_tol=_Q2_BAND_TOLERANCE)
            for endpoint, boundary in zip((left, right), _Q2_FIXED_BAND, strict=True)
        )

    def subdivide(bounds, width, *, panel_limit):
        if len(bounds) == 1 and is_exact_band(*bounds[0]):
            return list(bounds), None
        return _subdivide_panel_bounds(bounds, width, panel_limit=panel_limit)

    def evaluate_panel(integrand, left, right, expected_shape, **kwargs):
        nonlocal q2_budget_exhausted
        if (
            _q2_band is not None
            and left >= _Q2_FIXED_BAND[0] - _Q2_BAND_TOLERANCE
            and right <= _Q2_FIXED_BAND[1] + _Q2_BAND_TOLERANCE
            and right > left
        ):
            try:
                return _evaluate_q2_panel(
                    integrand,
                    left,
                    right,
                    expected_shape,
                    component_scales=scales,
                    phase_length=float(_q2_phase_length),
                    exact_zero_mask=q2_mask,
                    max_evaluations=max_evaluations - evaluations,
                    orders=kwargs.get("orders", _INITIAL_RULE_ORDERS),
                )
            except _Q2BudgetExhaustedError as exc:
                q2_budget_exhausted = True
                return None, exc.evaluations
            except _Q2PanelError as exc:
                return None, exc.evaluations
            except (ArithmeticError, ValueError):
                # Q2 validates the representability and leaf cap of its phase
                # grid before calling the integrand. Such a panel has no evidence.
                return None, 0
        if _q2_band is None:
            return _evaluate_panel(integrand, left, right, expected_shape, **kwargs)
        # Q2 has variable cost, so earlier panels in this same round can consume
        # the GL allocation reserved by the unchanged controller formulas.
        gl_calls = 0

        def budgeted_integrand(k):
            nonlocal gl_calls
            if gl_calls >= max_evaluations - evaluations:
                raise _Q2BudgetExhaustedError(
                    "combined callback budget exhausted", gl_calls
                )
            gl_calls += 1
            return integrand(k)

        try:
            return _evaluate_panel(
                budgeted_integrand, left, right, expected_shape, **kwargs
            )
        except _Q2BudgetExhaustedError as exc:
            q2_budget_exhausted = True
            return None, exc.evaluations

    if origin_subdivisions > 0:
        panel_bounds = _initial_panel_bounds(
            cutoff, origin_subdivisions=int(origin_subdivisions)
        )
    else:
        panel_bounds = _initial_panel_bounds(cutoff)
    panel_cost = sum(_INITIAL_RULE_ORDERS)
    refined_panel_cost = sum(refinement_rule_orders)
    setup_termination: TransformTermination | None = None
    if panel_width is not None:
        bounded, setup_termination = subdivide(
            panel_bounds,
            panel_width,
            panel_limit=max_evaluations // panel_cost,
        )
        panel_bounds = [] if bounded is None else bounded
    panels: list[_PanelEstimate] = []
    last_finite: tuple[np.ndarray, np.ndarray | None, np.ndarray, float] | None = None

    def failure(termination: TransformTermination) -> TransformEstimate:
        if q2_budget_exhausted:
            termination = "resource_limit"
        if last_finite is None:
            values = errors = tails = None
            evidence_cutoff = cutoff
        else:
            values, errors, tails, evidence_cutoff = last_finite
        return _failure_estimate(
            values=values,
            errors=errors,
            tails=tails,
            evaluations=evaluations,
            refinements=refinements,
            cutoff=evidence_cutoff,
            termination=termination,
        )

    if setup_termination is not None:
        return failure(setup_termination)
    if not panel_bounds or len(panel_bounds) * panel_cost > max_evaluations:
        return failure("resource_limit")
    for left, right in panel_bounds:
        panel, calls = evaluate_panel(integrand, left, right, scales.shape)
        evaluations += calls
        if panel is None:
            return failure("uncertainty_unresolved")
        panels.append(panel)

    while True:
        try:
            values = _fsum_arrays([panel.values for panel in panels])
            absolute_sum = _fsum_arrays([panel.absolute_sum for panel in panels])
            if filter_panel_roundoff:
                # Modeled truncation error filtering (C-1):
                # _evaluate_panel applies an empirical 10x safety factor to successive
                # Gauss-Legendre rule differences (10 * max(|Q_{k+1} - Q_k|)). When a panel's
                # truncation error reaches machine precision, rule differences reflect binary64
                # floating-point noise (~ 16 * eps * absolute_sum). The 10x multiplier turns this
                # noise into an artificial floor (~ 160 * eps * absolute_sum). Across thousands
                # of panels, summing this inflated floor yields ~ 10 * R_global, preventing
                # convergence even when mathematical truncation error is negligible.
                # Subtracting 10 * r_p removes the empirical 10x multiplier from the truncation
                # estimate. True floating-point accumulation roundoff is separately modeled
                # by roundoff = 16 * eps * max(scales, absolute_sum) below, summed via fsum.
                # This constitutes a modeled error estimate rather than a strict mathematical
                # theorem bound, and remains conservative against high-precision references.
                panel_trunc_errors = [
                    np.maximum(
                        panel.errors
                        - 10.0
                        * _ROUNDOFF_FACTOR
                        * np.finfo(np.float64).eps
                        * panel.absolute_sum,
                        0.0,
                    )
                    for panel in panels
                ]
                quadrature_errors = _fsum_arrays(panel_trunc_errors)
            else:
                quadrature_errors = _fsum_arrays([panel.errors for panel in panels])
        except (OverflowError, ValueError):
            return failure("uncertainty_unresolved")
        if values.shape != scales.shape:
            raise ValueError("component_scales shape must match integrand output")
        if not all(
            np.isfinite(array).all()
            for array in (values, absolute_sum, quadrature_errors)
        ):
            return failure("uncertainty_unresolved")

        scale_for_roundoff = np.maximum(scales, absolute_sum)
        # math.fsum replaces the former sequential node/panel reductions, so
        # their term-count growth is no longer represented by the old 64-eps
        # heuristic. Sixteen eps remains an estimate for binary64 kernel and
        # weight arithmetic plus conversion; it is not a rigorous bound. The
        # cancellation regression in test_gen1_cavity_numerics.py demonstrates
        # the accumulation change independently of the acceptance budget.
        roundoff = _ROUNDOFF_FACTOR * np.finfo(np.float64).eps * scale_for_roundoff
        roundoff[(roundoff == 0.0) & (scale_for_roundoff > 0.0)] = np.nextafter(
            0.0, 1.0
        )
        errors = quadrature_errors + roundoff
        if not np.isfinite(errors).all():
            return failure("uncertainty_unresolved")
        # Preserve the completed quadrature evidence even if the separate
        # truncation-tail calculation below cannot produce a usable bound.
        last_finite = (values.copy(), errors.copy(), None, cutoff)

        try:
            raw_tail = tail_bound(cutoff)
        except (ArithmeticError, OverflowError, ValueError):
            raw_tail = None
        if raw_tail is None:
            return failure("uncertainty_unresolved")
        tails = np.asarray(raw_tail, dtype=np.float64)
        if (
            tails.shape != values.shape
            or not np.isfinite(tails).all()
            or np.any(tails < 0.0)
            or np.any((tails == 0.0) & (np.asarray(raw_tail) > 0.0))
        ):
            return failure("uncertainty_unresolved")

        last_finite = (values.copy(), errors.copy(), tails.copy(), cutoff)

        if acceptance(values, errors, tails):
            return TransformEstimate(
                values=values,
                component_errors=errors,
                tail_bounds=tails,
                evaluations=evaluations,
                refinements=refinements,
                cutoff=cutoff,
                termination="converged",
            )
        if refinements >= max_refinements:
            return failure("resource_limit")

        safe_scales = np.maximum(scales, np.finfo(float).tiny)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            scaled_errors = errors / safe_scales
            scaled_tails = tails / safe_scales
        if np.max(scaled_tails) > np.max(scaled_errors):
            expanded = cutoff * 3.0
            if not np.isfinite(expanded) or expanded <= cutoff:
                return failure("uncertainty_unresolved")
            extension = (
                [(cutoff, expanded)]
                if is_exact_band(cutoff, expanded)
                else _extension_panel_bounds(cutoff, expanded)
            )
            if panel_width is not None:
                bounded_extension, extension_termination = subdivide(
                    extension,
                    panel_width,
                    panel_limit=(max_evaluations - evaluations) // panel_cost,
                )
                if bounded_extension is None:
                    if extension_termination is None:
                        return failure("uncertainty_unresolved")
                    return failure(extension_termination)
                extension = bounded_extension
            required = len(extension) * panel_cost
            if evaluations + required > max_evaluations:
                return failure("resource_limit")
            new_panels: list[_PanelEstimate] = []
            for left, right in extension:
                panel, calls = evaluate_panel(integrand, left, right, scales.shape)
                evaluations += calls
                if panel is None:
                    return failure("uncertainty_unresolved")
                new_panels.append(panel)
            panels.extend(new_panels)
            cutoff = expanded
            refinements += 1
            continue

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            if filter_panel_roundoff:
                # Prioritize panels with unresolved mathematical truncation error over
                # panels whose rule differences are dominated by machine roundoff noise.
                panel_scores = np.asarray(
                    [
                        float(
                            np.max(
                                np.maximum(
                                    panel.errors
                                    - 10.0
                                    * _ROUNDOFF_FACTOR
                                    * np.finfo(np.float64).eps
                                    * panel.absolute_sum,
                                    0.0,
                                )
                                / safe_scales
                            )
                        )
                        for panel in panels
                    ]
                )
            else:
                panel_scores = np.asarray(
                    [float(np.max(panel.errors / safe_scales)) for panel in panels]
                )
        largest_score = float(np.max(panel_scores, initial=0.0))
        if not np.isfinite(largest_score) or largest_score <= 0.0:
            return failure("uncertainty_unresolved")
        candidates = np.flatnonzero(panel_scores >= largest_score * 0.001)
        if not len(candidates):
            candidates = np.array([int(np.argmax(panel_scores))])

        if promote_initial_panels:
            promotion_cost = sum(_PROMOTED_INITIAL_ORDERS)
            split_cost = 2 * refined_panel_cost
            ordered_actions = sorted(
                (
                    (
                        float(panel_scores[int(index)]),
                        int(index),
                        "promote" if panels[int(index)].is_initial else "split",
                    )
                    for index in candidates
                ),
                key=lambda action: (-action[0], action[1]),
            )
            remaining_evaluations = max_evaluations - evaluations
            selected_actions: list[tuple[int, str]] = []
            for _score, index, action in ordered_actions:
                action_cost = promotion_cost if action == "promote" else split_cost
                if action_cost <= remaining_evaluations:
                    selected_actions.append((index, action))
                    remaining_evaluations -= action_cost
            if not selected_actions:
                return failure("resource_limit")

            replacements: dict[int, tuple[_PanelEstimate, ...]] = {}
            for index, action in selected_actions:
                panel = panels[index]
                if action == "promote":
                    promoted, calls = evaluate_panel(
                        integrand,
                        panel.left,
                        panel.right,
                        scales.shape,
                        orders=_PROMOTED_INITIAL_ORDERS,
                        is_initial=False,
                    )
                    evaluations += calls
                    if promoted is None:
                        return failure("uncertainty_unresolved")
                    replacements[index] = (promoted,)
                    continue

                midpoint = panel.left + (panel.right - panel.left) * 0.5
                if midpoint <= panel.left or midpoint >= panel.right:
                    continue
                left_panel, left_calls = evaluate_panel(
                    integrand,
                    panel.left,
                    midpoint,
                    scales.shape,
                    orders=refinement_rule_orders,
                    is_initial=False,
                )
                evaluations += left_calls
                if left_panel is None:
                    return failure("uncertainty_unresolved")
                right_panel, right_calls = evaluate_panel(
                    integrand,
                    midpoint,
                    panel.right,
                    scales.shape,
                    orders=refinement_rule_orders,
                    is_initial=False,
                )
                evaluations += right_calls
                if right_panel is None:
                    return failure("uncertainty_unresolved")
                replacements[index] = (left_panel, right_panel)

            if not replacements:
                return failure("uncertainty_unresolved")
            refined_panels: list[_PanelEstimate] = []
            for index, panel in enumerate(panels):
                refined_panels.extend(replacements.get(index, (panel,)))
            panels = refined_panels
            refinements += 1
            continue

        available_splits = (max_evaluations - evaluations) // (2 * refined_panel_cost)
        if available_splits < 1:
            return failure("resource_limit")
        if len(candidates) > available_splits:
            ordered = sorted(
                (int(index) for index in candidates),
                key=lambda index: panel_scores[index],
                reverse=True,
            )
            candidates = np.asarray(ordered[:available_splits], dtype=np.int64)

        replacements: dict[int, tuple[_PanelEstimate, _PanelEstimate]] = {}
        for index in candidates:
            panel = panels[int(index)]
            midpoint = panel.left + (panel.right - panel.left) * 0.5
            if midpoint <= panel.left or midpoint >= panel.right:
                continue
            left_panel, left_calls = evaluate_panel(
                integrand,
                panel.left,
                midpoint,
                scales.shape,
                orders=refinement_rule_orders,
                is_initial=False,
            )
            evaluations += left_calls
            if left_panel is None:
                return failure("uncertainty_unresolved")
            right_panel, right_calls = evaluate_panel(
                integrand,
                midpoint,
                panel.right,
                scales.shape,
                orders=refinement_rule_orders,
                is_initial=False,
            )
            evaluations += right_calls
            if right_panel is None:
                return failure("uncertainty_unresolved")
            replacements[int(index)] = (left_panel, right_panel)
        if not replacements:
            return failure("uncertainty_unresolved")
        refined_panels: list[_PanelEstimate] = []
        for index, panel in enumerate(panels):
            replacement = replacements.get(index)
            refined_panels.extend((panel,) if replacement is None else replacement)
        panels = refined_panels
        refinements += 1


__all__ = ["TransformEstimate", "integrate_transform"]
