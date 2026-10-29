"""Finite-window continuous Fourier transform by adaptive quadrature.

For a time-first response ``q(t)`` (relative seconds from a GPS epoch), this
module computes the one-sided finite-window transform

    X(f) = ∫_{t_start}^{t_stop} q(t) exp(-2πi f (Δ + t)) dt,   Δ = epoch - tref,

for every frequency, target, and component. The result is not doubled, and its
unit is the response unit times seconds. The output has shape
``(Nf, Ntarget, *component)``.

Quadrature:

* The integrand is divided by the per-component tolerance
  ``tol_i = atol_i + rtol·ŝ_i`` (see
  :func:`gragra._continuous_contract.component_tolerance`) and integrated with
  ``scipy.integrate.quad_vec(epsabs=1, epsrel=0, norm="max")``. The max norm then
  judges each component against its own tolerance.
* ``ŝ_i`` is ``max|q_i|·W`` on a fixed coarse grid over the window, with spacing
  ``tau_min`` and at most 4096 points. A narrow feature that the coarse grid
  misses makes ``ŝ_i`` small, so the tolerance is then set by ``atol_i``. This
  is the safe direction, but it is not a guarantee of accuracy.
* The initial partition has panel width at most
  ``min(8·tau_min, 1/(4·f_max))``. The first term resolves features at the
  scale of ``tau_min``, and the second resolves the phase factor at the highest
  frequency.
* A result is accepted only when ``quad_vec`` reports ``status == 0``, the
  result and error estimate are finite, and a recomputation on a half-panel
  shifted partition agrees within the tolerance. The shifted partition guards
  against a false convergence that the first partition does not reveal.

Phase:

* The phase is ``exp(-2πi f (Δ + t)) = exp(-2πi (frac(f·Δ) + f·t))``. Here
  ``Δ = s + ns·1e-9`` is kept as integer seconds and nanoseconds, and
  ``frac(f·Δ)`` is computed once per frequency in exact rational arithmetic
  (``fractions.Fraction``). The float ``Δ`` is never formed, so the rounding
  error of ``Δ`` does not enter the integrand.
* Inside the integrand only ``f·t`` is rounded, where ``t`` is the relative
  time in the window (``|t| ≤`` the window extent). This is the only rounding
  left in the phase.
* The window phase range ``f_max·|Δ| ≤ 1e6`` cycles is still enforced by
  :func:`gragra._continuous_contract.check_window_phase_range`.

The quadrature error estimate reported in the diagnostics is the integrator's
own estimate, measured in units of the per-component tolerance. It is not a
scientific error bar.

Complex outputs represent linear complex amplitudes or transfer-function-like
responses, not directly real-valued time-domain gravitational fields.

The shared contract (phase rules, tolerance, failure reasons, memory-budget
rule) lives in :mod:`gragra._continuous_contract`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction
from typing import Final

import numpy as np
from astropy import units as u

from gragra._continuous_contract import (
    ContinuousQuadratureError,
    ResponseFunction,
    check_memory_budget,
    check_window_phase_range,
    component_shape,
    component_tolerance,
    gps_delta_parts,
    normalize_relative_times,
    validate_observable,
)


def quad_vec(*args, **kwargs):
    """Call :func:`scipy.integrate.quad_vec`, importing SciPy lazily.

    SciPy's module-level import is confined to ``analytic/`` (see
    ``docs/architecture/numerical_dependency_adr.md``); this thin wrapper keeps
    the name patchable in tests.
    """
    from scipy.integrate import quad_vec as _quad_vec

    return _quad_vec(*args, **kwargs)


PANEL_TAU_FACTOR: Final[float] = 8.0
"""Initial panel width is at most ``PANEL_TAU_FACTOR * tau_min`` (acceptance c1)."""

PANEL_FREQUENCY_FACTOR: Final[float] = 0.25
"""Initial panel width is at most ``PANEL_FREQUENCY_FACTOR / f_max`` (acceptance c2)."""

COARSE_MAX_POINTS: Final[int] = 4096
"""Upper bound on the coarse-grid point count used to estimate ``ŝ_i``."""

GK21_NODES_PER_PANEL: Final[int] = 21
"""Gauss-Kronrod rule evaluations per panel used by ``quad_vec`` (GK21)."""

EPSABS_NORMALIZED: Final[float] = 1.0
"""Absolute tolerance in units of ``tol_i``, which makes the max norm per-component."""

_METHOD: Final[str] = "quad_vec"
_FLOAT_BYTES: Final[int] = 8
_COMPLEX_BYTES: Final[int] = 16
_PYTHON_FLOAT_BYTES: Final[int] = 32
_INTERVAL_OVERHEAD_BYTES: Final[int] = 512
_NANOSECONDS_PER_SECOND: Final[int] = 1_000_000_000


class _EvaluationBudget:
    """Count callback evaluations and never request more than ``max_evaluations``.

    ``reserve`` is called before a batch is passed to the callback, so a batch
    that would exceed the budget is never evaluated (blind v2 N-06, Revision R3).
    ``commit`` then records the batch once the callback has returned.
    """

    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.count = 0

    def reserve(self, number: int) -> None:
        """Raise before evaluating ``number`` more points if that exceeds the budget."""
        if self.count + number > self.maximum:
            raise _failure(
                "max_evaluations_exceeded",
                f"callback evaluations exceed max_evaluations={self.maximum}",
                evaluations=self.count,
            )

    def commit(self, number: int) -> None:
        """Record ``number`` evaluations that the callback has performed."""
        self.count += number


def _failure(
    reason: str,
    message: str,
    *,
    evaluations: int,
    status: int | None = None,
    error_estimate: float | None = None,
    independent_check: dict | None = None,
) -> ContinuousQuadratureError:
    """Build a :class:`ContinuousQuadratureError` with a complete diagnostic dict."""
    diagnostics: dict = {
        "method": _METHOD,
        "status": status,
        "reason": reason,
        "message": message,
        "error_estimate": error_estimate,
        "evaluations": evaluations,
        "nodes": None,
        "independent_check": independent_check,
    }
    return ContinuousQuadratureError(message, reason=reason, diagnostics=diagnostics)


def _require_int(name: str, value: object, *, minimum: int) -> int:
    """Return ``value`` as a Python int that is at least ``minimum``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _window_bounds(t_start: object, t_stop: object) -> tuple[float, float]:
    """Return the window ``(t0, t1)`` in relative seconds with ``t0 < t1``."""
    start = normalize_relative_times(t_start)
    stop = normalize_relative_times(t_stop)
    if np.ndim(t_start) != 0 or np.ndim(t_stop) != 0:
        raise ValueError("t_start and t_stop must be scalars")
    t0 = float(start.to_value(u.s)[0])
    t1 = float(stop.to_value(u.s)[0])
    if not t1 > t0:
        raise ValueError("t_stop must be greater than t_start")
    return t0, t1


def _frequencies_hz(frequencies: object) -> np.ndarray:
    """Return a non-empty 1-D array of nonnegative finite frequencies in Hz."""
    if not isinstance(frequencies, u.Quantity):
        raise TypeError("frequencies must be an astropy Quantity in frequency units")
    raw = np.asarray(frequencies.value)
    if raw.dtype.kind not in "fiu":
        raise TypeError("frequencies must have a real numeric dtype")
    if raw.ndim > 1:
        raise ValueError("frequencies must be a scalar or a 1-D array")
    try:
        values = np.asarray(frequencies.to_value(u.Hz), dtype=np.float64).reshape(-1)
    except u.UnitsError as exc:
        raise ValueError("frequencies must have frequency units") from exc
    if values.size == 0:
        raise ValueError("frequencies must not be empty")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("frequencies must be finite and nonnegative")
    return values


def _tau_min_seconds(tau_min: object) -> float:
    """Validate the required ``tau_min`` and return it in seconds."""
    if tau_min is None:
        raise ValueError("tau_min is required and must be a positive time Quantity")
    if not isinstance(tau_min, u.Quantity):
        raise TypeError("tau_min must be an astropy Quantity in time units")
    if np.ndim(tau_min) != 0:
        raise ValueError("tau_min must be a scalar")
    try:
        tau = float(tau_min.to_value(u.s))
    except u.UnitsError as exc:
        raise ValueError("tau_min must have a time unit") from exc
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError("tau_min must be positive and finite")
    return tau


def _initial_panel_count(width: float, tau_min: float, f_max: float) -> int:
    """Return ``n_init`` for the initial partition of the window.

    The panel width is ``Δt_panel = min(c1·τ_min, c2/f_max)``, and
    ``n_init = ceil(W / Δt_panel)``. The second term is dropped when
    ``f_max`` is zero.
    """
    panel = PANEL_TAU_FACTOR * tau_min
    if f_max > 0.0:
        panel = min(panel, PANEL_FREQUENCY_FACTOR / f_max)
    return math.ceil(width / panel)


def _coarse_point_count(width: float, tau_min: float) -> int:
    """Return the number of coarse-grid points (``τ_min`` spacing, capped)."""
    intervals = min(math.ceil(width / tau_min), COARSE_MAX_POINTS)
    return intervals + 1


def _fractional_frequency_phase(
    freq_hz: np.ndarray,
    delta_parts: tuple[int, int],
) -> np.ndarray:
    """Return ``frac(f·Δ)`` for each frequency, computed exactly.

    The returned values lie in ``[0, 1]``. The exact rational residue is in
    ``[0, 1)``, but the final conversion to float64 can round a value just below
    1 up to ``1.0``. The phase ``exp(-2πi·frac)`` is the same for 0 and 1, so this
    rounding does not change the integrand.

    ``Δ = s + ns·10⁻⁹`` is the exact rational ``Fraction(s) + Fraction(ns, 10⁹)``,
    and each frequency is converted exactly with ``Fraction(float)``. The floor
    is taken on the rational value, so a negative ``Δ`` reduces to the
    same residue as a positive one. The loop runs once per frequency; the
    integrand never repeats it.
    """
    seconds, nanoseconds = delta_parts
    delta = Fraction(seconds) + Fraction(nanoseconds, _NANOSECONDS_PER_SECOND)
    fractions = np.empty(freq_hz.size, dtype=np.float64)
    for index, frequency in enumerate(freq_hz):
        product = Fraction(float(frequency)) * delta
        fractions[index] = float(product - math.floor(product))
    return fractions


def _sample(
    response: ResponseFunction,
    observable: str,
    seconds: np.ndarray,
    budget: _EvaluationBudget,
) -> u.Quantity:
    """Evaluate ``response`` at relative ``seconds`` and check the result.

    The budget is reserved before the callback runs, so a batch that would
    exceed ``max_evaluations`` is not evaluated. Each performed batch is
    committed to the count. A non-finite return value raises
    ``nonfinite_response``. Callback exceptions propagate unchanged.
    """
    budget.reserve(seconds.size)
    values = response(seconds * u.s, observable)
    budget.commit(seconds.size)
    if not isinstance(values, u.Quantity):
        raise TypeError("response must return an astropy Quantity")
    raw = np.asarray(values.value)
    if raw.dtype.kind not in "fiu":
        raise TypeError("response must return a real-valued Quantity")
    if raw.ndim < 1 or raw.shape[0] != seconds.size:
        raise ValueError("response must return one leading row per requested time")
    if not np.isfinite(raw).all():
        raise _failure(
            "nonfinite_response",
            "response returned a non-finite value",
            evaluations=budget.count,
        )
    return values


def _estimate_managed_bytes(
    *,
    coarse_points: int,
    block_size: int,
    flat_size: int,
    n_init: int,
    n_freq: int,
    limit: int,
    cache_size: int,
) -> int:
    """Estimate the managed arrays of one run, in bytes (an upper bound).

    The estimate is computed before the coarse grid is evaluated, from the
    block shape ``(Ntarget, *component)`` found by the probe evaluation. Each
    term is one array family of the run, with its multiplicity:

    * probe: the probe response and its unit-converted copy (2 · block).
    * frequency phase: the exact fractional phase ``frac(f·Δ)`` (1 · Nf floats).
    * coarse grid: the grid times (points), the response Quantity, the float
      copy, and ``|q|`` (points · block each, 3 · points · block in total).
    * tolerance: ``atol``, ``ŝ`` (magnitude), and ``tol`` (3 · block).
    * partitions: the main and shifted point lists, as Python floats in lists
      (2 · n_init · (32 + 8) bytes).
    * results: the main and shifted quad_vec vectors (2 · flat complex).
    * integrand: per-call temporaries of the phase, the product, and the
      division, plus the ravelled copy (4 · flat complex).
    * GK21 scratch: node values of one panel for both integrations
      (2 · 21 · flat complex).
    * quad_vec intervals: per interval, one flat complex vector and the
      bookkeeping overhead. The bound is ``limit``, which is at least
      ``4·n_init`` and therefore covers the initial intervals too.
    * quad_vec cache: ``cache_size`` bytes.
    * output: the returned complex array and the product with ``tol``
      (2 · flat complex).

    ``block_size`` is the product of the target and component axes, and
    ``flat_size`` is ``Nf·block_size``. The estimate excludes process memory
    outside this function.
    """
    probe = 2 * block_size * _FLOAT_BYTES
    frequency_phase = n_freq * _FLOAT_BYTES
    coarse = coarse_points * _FLOAT_BYTES * (1 + 3 * block_size)
    tolerance = 3 * _FLOAT_BYTES * block_size
    partitions = 2 * n_init * (_PYTHON_FLOAT_BYTES + _FLOAT_BYTES)
    results = 2 * flat_size * _COMPLEX_BYTES
    integrand = 4 * flat_size * _COMPLEX_BYTES
    scratch = 2 * GK21_NODES_PER_PANEL * flat_size * _COMPLEX_BYTES
    intervals = limit * (flat_size * _COMPLEX_BYTES + _INTERVAL_OVERHEAD_BYTES)
    output = 2 * flat_size * _COMPLEX_BYTES
    return int(
        probe
        + frequency_phase
        + coarse
        + tolerance
        + partitions
        + results
        + integrand
        + scratch
        + intervals
        + cache_size
        + output
    )


def _block_and_unit(values: u.Quantity, component: tuple[int, ...]) -> tuple:
    """Return ``((Ntarget, *component), unit)`` from one response evaluation.

    Raises:
        ValueError: the response does not have shape ``(n, Ntarget, *component)``
            with ``Ntarget >= 1``.
    """
    raw = np.asarray(values.value)
    if raw.ndim != 2 + len(component) or raw.shape[1] < 1:
        raise ValueError("response must return shape (Ntime, Ntarget, *component)")
    if raw.shape[2:] != component:
        raise ValueError(f"response component shape must be {component}")
    return raw.shape[1:], values.unit


def _integrate(
    integrand: Callable[[float], np.ndarray],
    t0: float,
    t1: float,
    points: list[float],
    *,
    limit: int,
    cache_size: int,
) -> tuple[np.ndarray, float, object]:
    """Run ``quad_vec`` with the fixed settings and return ``(res, err, info)``."""
    result, error, info = quad_vec(
        integrand,
        t0,
        t1,
        epsabs=EPSABS_NORMALIZED,
        epsrel=0.0,
        norm="max",
        cache_size=cache_size,
        limit=limit,
        workers=1,
        points=points,
        full_output=True,
    )
    return np.asarray(result), float(error), info


def _require_converged(
    outcome: tuple[np.ndarray, float, object],
    budget: _EvaluationBudget,
) -> None:
    """Raise ``quad_vec_status`` unless the run has status 0 and finite values."""
    result, error, info = outcome
    status = int(info.status)
    message = str(info.message)
    if status != 0:
        raise _failure(
            "quad_vec_status",
            f"quad_vec finished with status {status}: {message}",
            evaluations=budget.count,
            status=status,
            error_estimate=error,
        )
    if not (np.isfinite(result).all() and math.isfinite(error)):
        raise _failure(
            "quad_vec_status",
            "quad_vec returned a non-finite result or error estimate",
            evaluations=budget.count,
            status=status,
            error_estimate=error,
        )


def finite_window_fourier(
    response: ResponseFunction,
    observable: str,
    *,
    epoch: object,
    tref: object,
    t_start: u.Quantity,
    t_stop: u.Quantity,
    frequencies: u.Quantity,
    tau_min: u.Quantity | None = None,
    atol: u.Quantity,
    rtol: float,
    limit: int,
    cache_size: int,
    max_evaluations: int,
    memory_budget_bytes: int,
) -> tuple[u.Quantity, dict]:
    """Return the one-sided finite-window transform ``X(f)`` and diagnostics.

    Args:
        response: Callable ``response(times, observable)`` returning a real
            time-first Quantity with shape ``(Ntime, Ntarget, *component)``.
            ``times`` are relative seconds from ``epoch``.
        observable: One of ``"potential"``, ``"acceleration"``, ``"gradient"``.
        epoch: Fixed GPS epoch (``LIGOTimeGPS`` or a ``(seconds, nanoseconds)``
            tuple). Relative times are measured from it.
        tref: Reference GPS time for the Fourier phase. Choose it near the
            window: ``f_max·|epoch - tref|`` must stay within the phase range.
        t_start: Window start, scalar Quantity in time units (relative seconds).
        t_stop: Window end, scalar Quantity in time units. Must exceed ``t_start``.
        frequencies: Quantity in frequency units. Finite and nonnegative, with
            shape ``(Nf,)`` or a scalar. Duplicates and any order are kept.
        tau_min: Required positive time Quantity. It sets the coarse-grid
            spacing and the initial panel width. A feature narrower than
            ``tau_min`` can still be missed, so choose it no larger than the
            narrowest feature of interest. Omitting it raises ``ValueError``.
        atol: Absolute tolerance Quantity. Its shape must broadcast to
            ``(Ntarget, *component)``, and its unit must match the response unit
            times seconds.
        rtol: Dimensionless nonnegative relative tolerance.
        limit: ``quad_vec`` interval limit. Must be at least ``4·n_init``.
        cache_size: ``quad_vec`` cache size in bytes.
        max_evaluations: Budget for the total number of callback evaluations
            (coarse grid, main integration, and shifted-partition check).
        memory_budget_bytes: Budget for the managed arrays, in bytes.

    Returns:
        ``(X, diagnostics)``. ``X`` is a complex128 Quantity with shape
        ``(Nf, Ntarget, *component)`` and unit ``response unit · s``. The
        diagnostics dict has exactly the keys of
        :data:`gragra._continuous_contract.DIAGNOSTIC_KEYS`. Its ``nodes`` entry
        is ``None``, and ``evaluations`` holds the total callback count.

    Raises:
        TypeError: wrong argument types, or a response that is not a real
            Quantity.
        ValueError: invalid values, a missing ``tau_min``, ``limit < 4·n_init``,
            a pre-check that the budget cannot cover the minimal partition, or a
            window phase range that is too large.
        MemoryError: the managed-array estimate exceeds ``memory_budget_bytes``.
        ContinuousQuadratureError: the quadrature was not verified. Its
            ``reason`` is one of the :data:`FAILURE_REASONS` values.
    """
    name = validate_observable(observable)
    component = component_shape(name)
    if not callable(response):
        raise TypeError("response must be callable")
    if not isinstance(atol, u.Quantity):
        raise TypeError("atol must be an astropy Quantity")

    t0, t1 = _window_bounds(t_start, t_stop)
    width = t1 - t0
    freq_hz = _frequencies_hz(frequencies)
    tau = _tau_min_seconds(tau_min)
    delta_parts = gps_delta_parts(epoch, tref)
    f_max = float(freq_hz.max())
    check_window_phase_range(delta_parts, f_max)

    limit_count = _require_int("limit", limit, minimum=1)
    cache_bytes = _require_int("cache_size", cache_size, minimum=1)
    max_count = _require_int("max_evaluations", max_evaluations, minimum=1)

    n_init = _initial_panel_count(width, tau, f_max)
    coarse_points = _coarse_point_count(width, tau)
    if limit_count < 4 * n_init:
        raise ValueError(f"limit must be at least 4·n_init = {4 * n_init}")
    # One probe evaluation (window start) fixes the block shape and the unit
    # before any large array is allocated. It counts toward max_evaluations,
    # so the minimal count below includes it.
    minimal_evaluations = 1 + coarse_points + 2 * GK21_NODES_PER_PANEL * n_init
    if minimal_evaluations > max_count:
        raise ValueError(
            "max_evaluations is below the minimal count "
            f"{minimal_evaluations} (probe + coarse grid + 2·21·n_init)"
        )

    budget = _EvaluationBudget(max_count)
    probe = _sample(response, name, np.array([t0]), budget)
    block, unit = _block_and_unit(probe, component)
    base_unit = unit * u.s
    if np.broadcast_shapes(atol.shape, block) != block:
        raise ValueError("atol must broadcast to (Ntarget, *component)")
    atol_base = atol.to(base_unit)  # unit mismatch is detected here

    block_size = int(np.prod(block))
    flat_size = freq_hz.size * block_size
    estimate = _estimate_managed_bytes(
        coarse_points=coarse_points,
        block_size=block_size,
        flat_size=flat_size,
        n_init=n_init,
        n_freq=freq_hz.size,
        limit=limit_count,
        cache_size=cache_bytes,
    )
    check_memory_budget(estimate, memory_budget_bytes)

    coarse_grid = np.linspace(t0, t1, coarse_points)
    coarse_values = _sample(response, name, coarse_grid, budget)
    if coarse_values.shape != (coarse_points,) + block:
        raise ValueError("response block shape changed between evaluations")
    raw = np.asarray(coarse_values.to_value(unit), dtype=np.float64)

    magnitude = np.max(np.abs(raw), axis=0) * width
    tol = component_tolerance(atol_base, rtol, magnitude * base_unit)
    tol = np.array(np.broadcast_to(tol, block), dtype=np.float64)
    if not (tol > 0.0).all():
        raise ValueError("tolerance must be positive for every component")

    # Exact frac(f·Δ) once per frequency; the integrand only rounds f·t.
    frac_phase = _fractional_frequency_phase(freq_hz, delta_parts)
    phase_axes = (freq_hz.size,) + (1,) * len(block)

    def integrand(t: float) -> np.ndarray:
        """Return the tolerance-normalized integrand at one relative time."""
        values = _sample(response, name, np.array([t]), budget)
        sample = np.asarray(values.to_value(unit), dtype=np.float64)[0]
        if sample.shape != block:
            raise ValueError("response block shape changed between evaluations")
        phase = np.exp(-2j * np.pi * (frac_phase + freq_hz * t)).reshape(phase_axes)
        return (phase * sample[None, ...] / tol[None, ...]).ravel()

    h = width / n_init
    main_points = [t0 + k * h for k in range(1, n_init)]
    shift_points = [t0 + (k + 0.5) * h for k in range(0, n_init - 1)]

    main = _integrate(
        integrand, t0, t1, main_points, limit=limit_count, cache_size=cache_bytes
    )
    _require_converged(main, budget)

    shifted = _integrate(
        integrand, t0, t1, shift_points, limit=limit_count, cache_size=cache_bytes
    )
    _require_converged(shifted, budget)

    main_result, main_error, main_info = main
    shifted_result, _, shifted_info = shifted
    difference = float(np.max(np.abs(main_result - shifted_result)))
    check = {
        "partition": "half-panel shifted",
        "passed": bool(difference <= EPSABS_NORMALIZED),
        "max_normalized_difference": difference,
        "threshold": EPSABS_NORMALIZED,
        "shifted_status": int(shifted_info.status),
    }
    if not check["passed"]:
        raise _failure(
            "independent_check_mismatch",
            "main and half-panel shifted partitions disagree beyond tolerance",
            evaluations=budget.count,
            status=int(main_info.status),
            error_estimate=main_error,
            independent_check=check,
        )

    values = main_result.reshape((freq_hz.size,) + block) * tol[None, ...]
    diagnostics: dict = {
        "method": _METHOD,
        "status": int(main_info.status),
        "reason": None,
        "message": str(main_info.message),
        "error_estimate": main_error,
        "evaluations": budget.count,
        "nodes": None,
        "independent_check": check,
    }
    result = u.Quantity(values, unit=base_unit, dtype=np.complex128)
    return result, diagnostics


__all__ = [
    "COARSE_MAX_POINTS",
    "EPSABS_NORMALIZED",
    "PANEL_FREQUENCY_FACTOR",
    "PANEL_TAU_FACTOR",
    "finite_window_fourier",
]
