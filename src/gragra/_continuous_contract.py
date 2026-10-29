"""Shared contract for continuous-time periodic and finite-window Fourier work.

This private module is the single definition of the interfaces, phase rules,
tolerances, exception reasons, and memory-budget rule used by the continuous
pose-response, periodic-coefficient, and finite-window transform lanes.

Conventions:

* Relative times are seconds measured from a fixed GPS ``epoch``.
* The Fourier phase is ``exp(-2πi f [(epoch - tref) + t])``. Shifting ``tref``
  by ``tau`` multiplies every coefficient by ``exp(+2πi f tau)``.
* Response arrays are time-first: potential ``(Ntime, Ntarget)``, acceleration
  ``(Ntime, Ntarget, 3)``, gradient ``(Ntime, Ntarget, 3, 3)``, in SI units.
* Phase reduction keeps ``Δ = epoch - tref`` as integer seconds and integer
  nanoseconds. A periodic phase fraction uses exact integer arithmetic when the
  period is an integer number of nanoseconds (within float representation
  error), and otherwise a float path restricted to ``n_max·|Δ|/T ≤ 1e6`` cycles.

This module performs no quadrature and no Fourier transform. The periodic and
finite-window functions are specified in the docstring of
:mod:`gragra._periodic_fourier` and :mod:`gragra._finite_window_fourier`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

import numpy as np
from astropy import units as u

from gragra.results._coords import normalize_gps

ResponseFunction = Callable[[u.Quantity, str], u.Quantity]
"""Callable ``response(times, observable) -> Quantity`` with relative seconds.

``times`` is a Quantity in seconds measured from the fixed ``epoch``. The
return value is time-first, see the module docstring.
"""

OBSERVABLE_COMPONENT_SHAPES: Mapping[str, tuple[int, ...]] = {
    "potential": (),
    "acceleration": (3,),
    "gradient": (3, 3),
}
"""Trailing component shape of each observable (time and target axes excluded)."""

PHASE_CYCLE_LIMIT: float = 1e6
"""Maximum ``n_max·|Δ|/T`` (float path) and ``f_max·|Δ|`` (window) in cycles."""

_INTEGER_PATH_ULP_FACTOR: int = 4
_NANOSECONDS_PER_SECOND: int = 1_000_000_000

DIAGNOSTIC_KEYS: frozenset[str] = frozenset(
    {
        "method",
        "status",
        "reason",
        "message",
        "error_estimate",
        "evaluations",
        "nodes",
        "independent_check",
    }
)
"""Exact key set of a quadrature diagnostic dict (C0-2)."""

FAILURE_REASONS: frozenset[str] = frozenset(
    {
        "independent_check_mismatch",
        "max_nodes_reached",
        "not_periodic",
        "quad_vec_status",
        "max_evaluations_exceeded",
        "nonfinite_response",
    }
)
"""Allowed ``reason`` values of :class:`ContinuousQuadratureError` (C0-2)."""


@runtime_checkable
class PoseFunction(Protocol):
    """Body-to-frame rigid pose sampled at relative times.

    Calling the pose with ``times`` (Quantity in seconds, relative to the
    epoch) returns ``(R, c)``. ``R`` is a ``scipy.spatial.transform.Rotation``
    with one rotation per time, and ``c`` is a Quantity in metres with shape
    ``(N, 3)``. The mapping is ``x = R x_body + c`` from body to computation
    frame.
    """

    def __call__(
        self, times: u.Quantity
    ) -> tuple[scipy.spatial.transform.Rotation, u.Quantity]:  # noqa: F821
        """Return the body-to-frame rotations and translations at ``times``."""
        ...


def _require_real_scalar(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def normalize_relative_times(times: object) -> u.Quantity:
    """Validate relative times and return them as a 1-D Quantity in seconds.

    A scalar becomes a length-1 array. Order and duplicates are preserved.
    Non-Quantity inputs, complex or boolean dtypes, and non-time units are
    rejected. Empty, multi-dimensional, and non-finite inputs are rejected too.

    Raises:
        TypeError: ``times`` is not a Quantity, or its dtype is not real numeric.
        ValueError: the unit is not a time unit, the shape is not 1-D or scalar,
            the array is empty, or a value is not finite.
    """
    if not isinstance(times, u.Quantity):
        raise TypeError("times must be an astropy Quantity in time units")
    raw = np.asarray(times.value)
    if raw.dtype.kind not in "fiu":
        raise TypeError("times must have a real numeric dtype (not complex or bool)")
    try:
        seconds = np.asarray(times.to_value(u.s), dtype=np.float64)
    except (u.UnitConversionError, u.UnitsError, TypeError) as exc:
        raise ValueError("times must have a time unit") from exc
    if seconds.ndim > 1:
        raise ValueError("times must be a scalar or a 1-D array")
    seconds = seconds.reshape(-1)
    if seconds.size == 0:
        raise ValueError("times must not be empty")
    if not np.isfinite(seconds).all():
        raise ValueError("times must be finite")
    return seconds * u.s


def validate_observable(name: object) -> str:
    """Return ``name`` if it is one of the three gragra observables.

    Raises:
        TypeError: ``name`` is not a string.
        ValueError: ``name`` is a string but not a known observable.
    """
    if not isinstance(name, str):
        raise TypeError("observable must be a string")
    if name not in OBSERVABLE_COMPONENT_SHAPES:
        raise ValueError(
            f"observable must be one of {tuple(OBSERVABLE_COMPONENT_SHAPES)}, "
            f"got {name!r}"
        )
    return name


def component_shape(observable: str) -> tuple[int, ...]:
    """Return the trailing component shape of a validated observable."""
    return OBSERVABLE_COMPONENT_SHAPES[validate_observable(observable)]


def gps_delta_parts(epoch: object, tref: object) -> tuple[int, int]:
    """Return ``epoch - tref`` as integer ``(seconds, nanoseconds)``.

    Both inputs are converted by :func:`gragra.results._coords.normalize_gps`.
    The result is normalized so that ``0 <= nanoseconds < 1e9``. The seconds
    part may be negative.
    """
    epoch_s, epoch_ns = normalize_gps(epoch)
    tref_s, tref_ns = normalize_gps(tref)
    total = (epoch_s - tref_s) * _NANOSECONDS_PER_SECOND + (epoch_ns - tref_ns)
    seconds, nanoseconds = divmod(total, _NANOSECONDS_PER_SECOND)
    return int(seconds), int(nanoseconds)


def _validate_delta_parts(delta_parts: object) -> tuple[int, int]:
    if not isinstance(delta_parts, tuple) or len(delta_parts) != 2:
        raise TypeError("delta_parts must be an (int seconds, int nanoseconds) tuple")
    seconds, nanoseconds = delta_parts
    for value in (seconds, nanoseconds):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError("delta_parts entries must be integers")
    return int(seconds), int(nanoseconds)


def _period_seconds(period: object) -> float:
    if not isinstance(period, u.Quantity):
        raise TypeError("period must be an astropy Quantity in time units")
    try:
        value = float(period.to_value(u.s))
    except (u.UnitConversionError, u.UnitsError, TypeError) as exc:
        raise ValueError("period must have a time unit") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("period must be positive and finite")
    return value


def integer_period_nanoseconds(period: object) -> int | None:
    """Return the nominal period in integer nanoseconds, or ``None``.

    The integer path applies when ``4·ulp(T_s)·1e9 < 0.5`` and
    ``|T_s·1e9 − T_ns| ≤ 4·ulp(T_s)·1e9``, with ``T_ns = round(T_s·1e9)``.
    The tolerance absorbs float representation error of the input period.
    """
    period_s = _period_seconds(period)
    period_ns = round(period_s * 1e9)
    if period_ns < 1:
        return None
    tolerance = _INTEGER_PATH_ULP_FACTOR * math.ulp(period_s) * 1e9
    if tolerance < 0.5 and abs(period_s * 1e9 - period_ns) <= tolerance:
        return int(period_ns)
    return None


def nominal_period_seconds(period: object) -> float:
    """Return the period used for node placement, in seconds.

    On the integer path this is ``T_ns·1e-9`` (the nominal period). Otherwise it
    is the input period as a float. Nodes and reference integrals should use
    this value so that they match the integer phase arithmetic.
    """
    period_ns = integer_period_nanoseconds(period)
    if period_ns is not None:
        return period_ns / _NANOSECONDS_PER_SECOND
    return _period_seconds(period)


def _validate_orders(orders: object) -> tuple[int, ...]:
    if isinstance(orders, (str, bytes)):
        raise TypeError("orders must be a sequence of integers")
    try:
        items = tuple(orders)  # type: ignore[call-overload]
    except TypeError as exc:
        raise TypeError("orders must be a sequence of integers") from exc
    if not items:
        raise ValueError("orders must not be empty")
    for item in items:
        if isinstance(item, (bool, np.bool_)) or not isinstance(
            item, (int, np.integer)
        ):
            raise TypeError("orders must contain only integers")
    return tuple(int(item) for item in items)


def _fraction(value: float) -> float:
    """Return the fractional part of ``value`` in the half-open interval [0, 1)."""
    frac = value - math.floor(value)
    return 0.0 if frac >= 1.0 else frac


def periodic_phase_fractions(
    delta_parts: tuple[int, int],
    period: object,
    orders: object,
) -> np.ndarray:
    """Return ``frac(n·Δ/T)`` in [0, 1) for each harmonic order ``n``.

    ``Δ`` is ``epoch - tref`` given as ``(seconds, nanoseconds)`` from
    :func:`gps_delta_parts`. The phase factor for harmonic ``n`` is
    ``exp(-2πi·frac(n·Δ/T))``.

    Integer path (period an integer number of nanoseconds): exact, with
    ``r = Δ_ns mod T_ns`` and ``frac = ((n·r) mod T_ns) / T_ns`` in Python
    integers. No range limit applies.

    Float path: requires ``n_max·|Δ|/T ≤ PHASE_CYCLE_LIMIT``.

    Raises:
        TypeError: malformed ``delta_parts``, ``period``, or ``orders``.
        ValueError: non-positive period, bad time unit, or the float path exceeds
            ``PHASE_CYCLE_LIMIT`` cycles.
    """
    seconds, nanoseconds = _validate_delta_parts(delta_parts)
    order_values = _validate_orders(orders)
    period_ns = integer_period_nanoseconds(period)
    if period_ns is not None:
        residue = (seconds * _NANOSECONDS_PER_SECOND + nanoseconds) % period_ns
        return np.array(
            [((n * residue) % period_ns) / period_ns for n in order_values],
            dtype=np.float64,
        )
    period_s = _period_seconds(period)
    delta = float(seconds) + float(nanoseconds) * 1e-9
    n_max = max(abs(n) for n in order_values)
    if n_max * abs(delta) / period_s > PHASE_CYCLE_LIMIT:
        raise ValueError(
            "float phase path requires n_max·|Δ|/T ≤ "
            f"{PHASE_CYCLE_LIMIT:g} cycles; use a period that is an integer "
            "number of nanoseconds or a reference time closer to the window"
        )
    return np.array(
        [_fraction(n * delta / period_s) for n in order_values],
        dtype=np.float64,
    )


def check_window_phase_range(delta_parts: tuple[int, int], f_max: object) -> None:
    """Raise ``ValueError`` if ``f_max·|Δ|`` exceeds :data:`PHASE_CYCLE_LIMIT`.

    Choose ``tref`` close to the window so that ``|Δ|`` is small. A real GPS
    epoch (about 1.3e9 s) against a distant ``tref`` is rejected here.
    """
    seconds, nanoseconds = _validate_delta_parts(delta_parts)
    frequency = _require_real_scalar("f_max", f_max)
    if frequency < 0.0:
        raise ValueError("f_max must be nonnegative")
    delta_abs = abs(float(seconds) + float(nanoseconds) * 1e-9)
    if frequency * delta_abs > PHASE_CYCLE_LIMIT:
        raise ValueError(
            "window phase requires f_max·|Δ| ≤ "
            f"{PHASE_CYCLE_LIMIT:g} cycles; choose tref near the window"
        )


def component_tolerance(
    atol: u.Quantity,
    rtol: float,
    magnitude: u.Quantity,
) -> np.ndarray:
    """Return the per-component tolerance ``tol_i = atol_i + rtol·ŝ_i``.

    ``atol`` has shape broadcastable to ``(Ntarget, *component)``. ``ŝ_i`` is
    ``magnitude`` (the caller's data scale, for example ``max|q_i|``) converted
    to the unit of ``atol``. ``rtol`` is a dimensionless nonnegative scalar.
    The result is a plain ndarray expressed in ``atol.unit``.

    Callers choose the unit: the periodic lane uses the response unit, and the
    finite-window lane uses response times seconds.
    """
    if not isinstance(atol, u.Quantity):
        raise TypeError("atol must be an astropy Quantity")
    if not isinstance(magnitude, u.Quantity):
        raise TypeError("magnitude must be an astropy Quantity")
    rtol_value = _require_real_scalar("rtol", rtol)
    if rtol_value < 0.0:
        raise ValueError("rtol must be nonnegative")
    atol_value = np.asarray(atol.value, dtype=np.float64)
    if not np.isfinite(atol_value).all() or (atol_value < 0.0).any():
        raise ValueError("atol must be finite and nonnegative")
    try:
        scale = np.asarray(magnitude.to_value(atol.unit), dtype=np.float64)
    except (u.UnitConversionError, u.UnitsError, TypeError) as exc:
        raise ValueError("magnitude must convert to the unit of atol") from exc
    if not np.isfinite(scale).all() or (scale < 0.0).any():
        raise ValueError("magnitude must be finite and nonnegative")
    np.broadcast_shapes(atol_value.shape, scale.shape)
    return atol_value + rtol_value * scale


class ContinuousQuadratureError(RuntimeError):
    """Quadrature did not produce a verified result.

    ``reason`` is one of :data:`FAILURE_REASONS`. ``diagnostics`` is a copy of a
    dict whose keys are exactly :data:`DIAGNOSTIC_KEYS`, and its ``reason`` entry
    must equal ``reason``. Tests should check ``reason`` as well as the type.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        diagnostics: Mapping[str, object],
    ) -> None:
        if reason not in FAILURE_REASONS:
            raise ValueError(f"reason must be one of {sorted(FAILURE_REASONS)}")
        if set(diagnostics) != DIAGNOSTIC_KEYS:
            raise ValueError(
                f"diagnostics keys must be exactly {sorted(DIAGNOSTIC_KEYS)}"
            )
        if diagnostics["reason"] != reason:
            raise ValueError("diagnostics['reason'] must equal the exception reason")
        super().__init__(message)
        self.reason = reason
        self.diagnostics: dict = dict(diagnostics)


def check_memory_budget(estimated_bytes: object, budget_bytes: object) -> None:
    """Raise ``MemoryError`` when the estimate exceeds the managed-array budget.

    Each lane implements ``_estimate_managed_bytes(...) -> int`` in its own
    module. It returns the estimate for managed arrays only (time blocks,
    responses, phases, coefficient accumulators, cache, and result copies), not
    process RSS. The lane calls this check before allocating anything.
    """
    if isinstance(estimated_bytes, (bool, np.bool_)) or not isinstance(
        estimated_bytes, (int, np.integer)
    ):
        raise TypeError("estimated_bytes must be an integer")
    if isinstance(budget_bytes, (bool, np.bool_)) or not isinstance(
        budget_bytes, (int, np.integer)
    ):
        raise TypeError("budget_bytes must be an integer")
    if estimated_bytes < 0:
        raise ValueError("estimated_bytes must be nonnegative")
    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive")
    if estimated_bytes > budget_bytes:
        raise MemoryError(
            f"managed arrays need {int(estimated_bytes)} bytes, budget is "
            f"{int(budget_bytes)} bytes"
        )


__all__ = [
    "DIAGNOSTIC_KEYS",
    "FAILURE_REASONS",
    "OBSERVABLE_COMPONENT_SHAPES",
    "PHASE_CYCLE_LIMIT",
    "ContinuousQuadratureError",
    "PoseFunction",
    "ResponseFunction",
    "check_memory_budget",
    "check_window_phase_range",
    "component_shape",
    "component_tolerance",
    "gps_delta_parts",
    "integer_period_nanoseconds",
    "nominal_period_seconds",
    "normalize_relative_times",
    "periodic_phase_fractions",
    "validate_observable",
]
