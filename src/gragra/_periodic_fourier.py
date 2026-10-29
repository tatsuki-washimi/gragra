"""Periodic Fourier coefficients of a continuous-time response (P4).

This private module computes the direct Fourier series coefficients of a
periodic response ``q(t)`` with a trapezoidal rule on a uniform grid over one
period. The conventions and shared rules (phase reduction, tolerance, error
reasons, memory budget) are defined in :mod:`gragra._continuous_contract`.

Mathematics
-----------

With ``Δ = epoch - tref`` and nominal period ``T``, the coefficient of
harmonic ``n`` is

    C_n = (1/T) ∫_0^T q(t) exp(-2πi n (Δ + t)/T) dt.

On a grid of ``M`` nodes ``t_j = j·T/M`` (``j = 0 … M-1``, no endpoint) this is
approximated by

    C_n ≈ (1/M) Σ_j q(t_j) exp(-2πi n j/M) · exp(-2πi frac(n·Δ/T)),

where ``frac(n·Δ/T)`` comes from :func:`periodic_phase_fractions`. The
time-part exponent ``n·j mod M`` is evaluated exactly with integers.

Quadrature control
------------------

* The grid size ``N`` starts at the smallest power of two that is at least
  ``max(32, 4(n_max+1), ceil(8·T/τ_min))``. ``N`` is doubled while the
  ``2N`` grid reuses the ``N`` nodes; only the new odd nodes are evaluated.
* A level is accepted when ``C(2N)`` agrees with ``C(N)`` within
  ``tol_i = atol_i + rtol·ŝ_i`` for every target and component. ``ŝ_i`` is
  ``max|q_i|`` on the initial grid, so a response with a narrow peak that the
  initial grid misses gets a tolerance dominated by ``atol``.
* An independent check recomputes the accepted coefficients on a grid of
  ``N+1`` nodes. Two-power doubling cannot see aliasing at harmonics that are
  multiples of the doubled grid, and the ``N+1`` grid cannot share those
  nodes. The check must agree with the same tolerance.
* The periodicity check ``|q(0) - q(T)| ≤ tol_i`` is done before doubling.
* The probe evaluation at ``t = 0`` (used to learn the response unit and
  shape before the budget check) is included in the ``evaluations`` count.

Applicability: the trapezoidal rule on ``M`` nodes aliases harmonics
``h`` with ``|h| ≥ M - n_max`` onto the coefficient ``n``. A finite sample
cannot guarantee that this does not happen for an arbitrary periodic
response. Callers must choose ``τ_min`` small enough that the response is
resolved on the initial grid. The value of ``τ_min`` is required, and the
caller must record it with the result.

Limitations: the reported error estimate is the difference between the last
two grid levels. It is a quadrature estimate and not a scientific error bound.
The tolerance depends on the coarse-grid magnitude ``ŝ_i``, as described in
the contract. Coefficients are returned in the response unit, with no
conversion to amplitudes.

Complex outputs represent linear complex amplitudes or transfer-function-like
responses, not directly real-valued time-domain gravitational fields.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from astropy import units as u

from gragra._continuous_contract import (
    DIAGNOSTIC_KEYS,
    ContinuousQuadratureError,
    _validate_orders,
    check_memory_budget,
    component_shape,
    component_tolerance,
    gps_delta_parts,
    nominal_period_seconds,
    periodic_phase_fractions,
    validate_observable,
)

_MIN_INITIAL_NODES: int = 32
"""Lower bound of the initial grid size."""

_NODE_BLOCK: int = 1024
"""Maximum number of time nodes evaluated by one response call."""

_BYTES_PER_FLOAT: int = 8
_BYTES_PER_INT: int = 8
_BYTES_PER_COMPLEX: int = 16


def _next_power_of_two(value: int) -> int:
    """Return the smallest power of two that is at least ``value`` (value >= 1)."""
    return 1 << (value - 1).bit_length()


def _estimate_managed_bytes(
    *,
    n_frequencies: int,
    n_targets: int,
    component_size: int,
    block_nodes: int,
) -> int:
    """Count the bytes of the managed arrays of one periodic evaluation.

    The count is made from shape and dtype, one term per array that P4 holds
    at the same time. ``F`` is the number of orders, ``B`` the number of time
    nodes in one block, ``S = Ntarget·component`` the field size, and
    ``C = F·S`` the coefficient size. The terms are:

    Time block (one block of at most ``B`` nodes):
      * node times ``t_j`` (float64, ``8·B``);
      * node indices ``j`` (int64, ``8·B``).

    Phase matrix of one block (``n·j mod M`` and the root of unity):
      * the outer product ``n·j`` and its reduction mod ``M`` (int64, ``2·8·F·B``);
      * the complex argument, its division by ``M``, and the exponential
        (complex128, ``3·16·F·B``).

    Response block (returned by the callable, as ``Ntime × Ntarget × component``):
      * the returned Quantity (float64, ``8·B·S``);
      * the unit-converted copy made by ``to_value`` (``8·B·S``);
      * ``|q|`` for the scale on the initial grid (``8·B·S``);
      * the contiguous matrix used by the tensor product (``8·B·S``).

    Coefficient accumulators (complex128, ``16·C`` each), counted at the peak of
    the doubling loop and the independent check:
      * ``sum_over``: running total, block contribution, and the new total
        while adding (3 arrays);
      * doubling: previous sums, doubled sums, previous coefficients, doubled
        coefficients, and their difference (5 arrays);
      * accepted coefficients (1 array);
      * independent check: check sums, check coefficients, and their
        difference (3 arrays);
      * the result copy returned to the caller (1 array).

    Field-sized float arrays (``8·S`` each): the initial scale, the tolerance,
    the periodicity start and end values, and the periodicity difference
    (5 arrays).

    The count does not depend on the grid size ``N`` because the grid indices
    are generated block by block. It counts only the arrays of
    ``periodic_fourier``. The memory budget therefore bounds these managed
    arrays, and it is not a bound on the process RSS. The response callable
    keeps its own temporaries, which are checked by the memory tests and not
    counted here.
    """
    if block_nodes < 1:
        raise ValueError("block_nodes must be positive")
    if n_frequencies < 1 or n_targets < 1 or component_size < 1:
        raise ValueError("sizes must be positive")
    field = n_targets * component_size
    coefficients = n_frequencies * field
    block = block_nodes
    node_times = block * _BYTES_PER_FLOAT
    node_indices = block * _BYTES_PER_INT
    phase_matrix = n_frequencies * block * (2 * _BYTES_PER_INT + 3 * _BYTES_PER_COMPLEX)
    response_block = 4 * block * field * _BYTES_PER_FLOAT
    coefficient_arrays = (3 + 5 + 1 + 3 + 1) * coefficients * _BYTES_PER_COMPLEX
    field_arrays = 5 * field * _BYTES_PER_FLOAT
    return int(
        node_times
        + node_indices
        + phase_matrix
        + response_block
        + coefficient_arrays
        + field_arrays
    )


class _PeriodicRun:
    """State of one ``periodic_fourier`` call: response evaluation and counters.

    The response is evaluated in blocks of time nodes. Each block contributes
    to the coefficient sums, so no full set of node responses is kept.
    """

    def __init__(
        self,
        response: Callable[[u.Quantity, str], u.Quantity],
        observable: str,
        period_s: float,
        orders: tuple[int, ...],
    ) -> None:
        self._response = response
        self._observable = observable
        self._period_s = period_s
        self._orders = np.asarray(orders, dtype=np.int64)
        self.unit: u.UnitBase | None = None
        self.field_shape: tuple[int, ...] | None = None
        self.evaluations: int = 0
        self.nodes: list[int] = []
        self.scale: np.ndarray | None = None

    def _check_block(self, value: object, n_times: int) -> np.ndarray:
        """Validate one response block and return it as float64 in ``self.unit``."""
        if not isinstance(value, u.Quantity):
            raise TypeError("response must return an astropy Quantity")
        if self.unit is None:
            self.unit = value.unit
        try:
            array = np.asarray(value.to_value(self.unit), dtype=np.float64)
        except (u.UnitConversionError, u.UnitsError, TypeError) as exc:
            raise ValueError("response units must stay consistent") from exc
        expected_tail = component_shape(self._observable)
        if array.ndim != 2 + len(expected_tail) or array.shape[0] != n_times:
            raise ValueError(
                "response must have shape (Ntime, Ntarget, *component) "
                f"with Ntime={n_times}, got {array.shape}"
            )
        if array.shape[2:] != expected_tail:
            raise ValueError(
                f"response component shape must be {expected_tail}, "
                f"got {array.shape[2:]}"
            )
        if self.field_shape is None:
            self.field_shape = array.shape[1:]
        elif array.shape[1:] != self.field_shape:
            raise ValueError("response target shape must not change between calls")
        return array

    def values_at(self, times_s: np.ndarray) -> np.ndarray:
        """Evaluate the response at ``times_s`` (seconds) and return a float block."""
        times_quantity = u.Quantity(np.asarray(times_s, dtype=np.float64), u.s)
        raw = self._response(times_quantity, self._observable)
        array = self._check_block(raw, times_s.shape[0])
        self.evaluations += int(times_s.shape[0])
        if not np.isfinite(array).all():
            self.fail(
                "nonfinite_response",
                "response returned a non-finite value",
            )
        return array

    def fail(
        self,
        reason: str,
        message: str,
        *,
        error_estimate: float | None = None,
        independent: bool | None = None,
    ) -> None:
        """Raise :class:`ContinuousQuadratureError` with the current diagnostics."""
        diagnostics: dict[str, Any] = {
            "method": "periodic_trapezoid",
            "status": "failed",
            "reason": reason,
            "message": message,
            "error_estimate": error_estimate,
            "evaluations": self.evaluations,
            "nodes": tuple(self.nodes),
            "independent_check": independent,
        }
        assert set(diagnostics) == DIAGNOSTIC_KEYS
        raise ContinuousQuadratureError(message, reason=reason, diagnostics=diagnostics)

    def sum_over(
        self,
        first: int,
        stop: int,
        step: int,
        grid: int,
        *,
        track_scale: bool,
    ) -> np.ndarray:
        """Return ``Σ_j q_j exp(-2πi n j / grid)`` over ``j = first, first+step, …``.

        The indices run over ``range(first, stop, step)`` and are generated
        one block at a time, so the index arrays do not scale with the grid.
        The result has shape ``(Nf, Ntarget, *component)``. When
        ``track_scale`` is true, the running maximum of ``|q|`` is kept in
        ``self.scale``.
        """
        total: np.ndarray | None = None
        block_stride = step * _NODE_BLOCK
        for block_start in range(first, stop, block_stride):
            block_stop = min(block_start + block_stride, stop)
            block = np.arange(block_start, block_stop, step, dtype=np.int64)
            values = self.values_at(block * self._period_s / grid)
            if track_scale:
                magnitude = np.max(np.abs(values), axis=0)
                self.scale = (
                    magnitude
                    if self.scale is None
                    else np.maximum(self.scale, magnitude)
                )
            # Exact integer exponent n·j mod grid, then the root of unity.
            exponent = np.outer(self._orders, block) % grid
            phases = np.exp(-2j * np.pi * exponent / grid)
            contribution = np.tensordot(phases, values, axes=(1, 0))
            total = contribution if total is None else total + contribution
        assert total is not None
        return total


def _finalize(
    sums: np.ndarray,
    grid: int,
    phase_factor: np.ndarray,
) -> np.ndarray:
    """Convert unnormalized sums to coefficients with the time-phase factor."""
    shape = (phase_factor.shape[0],) + (1,) * (sums.ndim - 1)
    return sums / grid * phase_factor.reshape(shape)


def _within(difference: np.ndarray, tolerance: np.ndarray) -> bool:
    """Return True when every component is within its tolerance."""
    return bool(np.all(np.abs(difference) <= tolerance))


def periodic_fourier(
    response: Callable[[u.Quantity, str], u.Quantity],
    observable: str,
    *,
    epoch: object,
    tref: object,
    period: u.Quantity,
    orders: object,
    tau_min: u.Quantity | None = None,
    atol: u.Quantity,
    rtol: float,
    max_nodes: int,
    memory_budget_bytes: int,
) -> tuple[u.Quantity, dict]:
    """Return the periodic Fourier coefficients ``C_n`` of a response.

    Parameters
    ----------
    response:
        ``response(times, observable) -> Quantity`` with ``times`` in seconds
        from ``epoch``. The return value is time-first with shape
        ``(Ntime, Ntarget, *component)``.
    observable:
        One of ``"potential"``, ``"acceleration"``, or ``"gradient"``.
    epoch, tref:
        GPS epoch and reference time (anything :func:`normalize_gps` accepts).
        The phase is ``exp(-2πi n (Δ + t)/T)`` with ``Δ = epoch - tref``.
    period:
        Period as a time Quantity. The response is evaluated on the nominal
        period from :func:`nominal_period_seconds`.
    orders:
        Integer harmonic orders ``n``. The output keeps this order.
    tau_min:
        Smallest time scale of the response, as a time Quantity. It is
        required, and omitting it raises ``ValueError``. A non-positive value
        also raises ``ValueError``. It sets the initial grid size.
    atol:
        Absolute tolerance Quantity, convertible to the response unit and
        broadcastable to ``(Ntarget, *component)``.
    rtol:
        Dimensionless nonnegative relative tolerance. The per-component
        tolerance is ``atol + rtol·max|q_i|`` on the initial grid.
    max_nodes:
        Upper bound on the number of nodes of one quadrature rule. It is not a
        cumulative count.
    memory_budget_bytes:
        Budget for the managed arrays counted by ``_estimate_managed_bytes``,
        checked before the computation. It is not a bound on process RSS.

    Returns
    -------
    tuple[Quantity, dict]
        ``C_n`` with shape ``(Nf, Ntarget, *component)``, complex128, in the
        response unit (no conversion to amplitudes), and the diagnostic dict
        with exactly the keys in ``DIAGNOSTIC_KEYS``. ``nodes`` lists the grid
        size of each level in evaluation order, followed by the independent
        check grid. ``evaluations`` includes the probe and the periodicity
        evaluations.

    Raises
    ------
    TypeError
        A value has the wrong type.
    ValueError
        ``tau_min`` is missing or not positive, the initial grid (or ``N+1``)
        exceeds ``max_nodes``, the shape of the response is wrong, or the phase
        range limit is exceeded.
    MemoryError
        The managed arrays exceed ``memory_budget_bytes``.
    ContinuousQuadratureError
        The quadrature was not verified. ``reason`` is one of
        ``max_nodes_reached``, ``independent_check_mismatch``,
        ``not_periodic``, or ``nonfinite_response``.

    Notes
    -----
    The phase is reduced by :func:`periodic_phase_fractions`, so an integer
    nanosecond period has exact phases. Node times are the nominal period
    times ``j/M``. ``τ_min`` and ``rtol`` are not stored in the output; the
    caller must record them with the result.
    """
    observable = validate_observable(observable)
    if not callable(response):
        raise TypeError("response must be callable")
    if not isinstance(atol, u.Quantity):
        raise TypeError("atol must be an astropy Quantity")
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, (int, np.integer)):
        raise TypeError("max_nodes must be an integer")
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")

    if tau_min is None:
        raise ValueError("tau_min is required for periodic_fourier")
    if not isinstance(tau_min, u.Quantity):
        raise TypeError("tau_min must be an astropy Quantity in time units")
    try:
        tau_s = float(tau_min.to_value(u.s))
    except (u.UnitConversionError, u.UnitsError, TypeError) as exc:
        raise ValueError("tau_min must have a time unit") from exc
    if not math.isfinite(tau_s) or tau_s <= 0.0:
        raise ValueError("tau_min must be positive and finite")

    order_values = _validate_orders(orders)
    period_s = nominal_period_seconds(period)
    delta = gps_delta_parts(epoch, tref)
    frac = periodic_phase_fractions(delta, period, order_values)
    phase_factor = np.exp(-2j * np.pi * frac)

    n_max = max(abs(n) for n in order_values)
    n_phys = math.ceil(8.0 * period_s / tau_s)
    initial = _next_power_of_two(
        max(_MIN_INITIAL_NODES, 4 * (n_max + 1), n_phys)
    )
    if initial > max_nodes or initial + 1 > max_nodes:
        raise ValueError(
            f"initial grid N={initial} (and N+1) must not exceed max_nodes={max_nodes}"
        )

    run = _PeriodicRun(response, observable, period_s, order_values)
    # The probe reveals Ntarget and the response unit before the budget check.
    # Its evaluation stays in the evaluations count.
    run.values_at(np.array([0.0]))
    if run.unit is None or run.field_shape is None:
        raise ValueError("response did not return a usable array")
    field_shape = run.field_shape
    component_size = int(np.prod(component_shape(observable), dtype=np.int64))
    estimated = _estimate_managed_bytes(
        n_frequencies=len(order_values),
        n_targets=field_shape[0],
        component_size=component_size,
        block_nodes=min(_NODE_BLOCK, max_nodes),
    )
    check_memory_budget(estimated, memory_budget_bytes)

    unit = run.unit
    try:
        atol_unit = atol.to(unit)
    except (u.UnitConversionError, u.UnitsError) as exc:
        raise ValueError("atol must be convertible to the response unit") from exc

    # Initial grid, periodicity check, and the tolerance from the initial scale.
    run.nodes.append(initial)
    sums = run.sum_over(0, initial, 1, initial, track_scale=True)
    assert run.scale is not None
    tolerance = component_tolerance(
        atol_unit, rtol, u.Quantity(run.scale, unit)
    )
    coefficients = _finalize(sums, initial, phase_factor)

    start_value = run.values_at(np.array([0.0]))[0]
    end_value = run.values_at(np.array([period_s]))[0]
    if not _within(start_value - end_value, tolerance):
        run.fail(
            "not_periodic",
            "response is not periodic: |q(0) - q(T)| exceeds the tolerance",
        )

    grid = initial
    previous_coefficients = coefficients
    previous_sums = sums
    error_estimate = 0.0
    while True:
        if 2 * grid + 1 > max_nodes:
            run.fail(
                "max_nodes_reached",
                f"doubling to N={2 * grid} needs {2 * grid + 1} nodes, "
                f"above max_nodes={max_nodes}",
            )
        doubled = 2 * grid
        doubled_sums = previous_sums + run.sum_over(
            1, doubled, 2, doubled, track_scale=False
        )
        doubled_coefficients = _finalize(doubled_sums, doubled, phase_factor)
        run.nodes.append(doubled)
        difference = doubled_coefficients - previous_coefficients
        error_estimate = float(np.max(np.abs(difference)))
        grid = doubled
        if _within(difference, tolerance):
            coefficients = doubled_coefficients
            break
        previous_coefficients = doubled_coefficients
        previous_sums = doubled_sums

    # Independent check: recompute on N+1 nodes, which cannot reuse the grid.
    check_grid = grid + 1
    check_sums = run.sum_over(0, check_grid, 1, check_grid, track_scale=False)
    check_coefficients = _finalize(check_sums, check_grid, phase_factor)
    run.nodes.append(check_grid)
    if not _within(check_coefficients - coefficients, tolerance):
        run.fail(
            "independent_check_mismatch",
            f"coefficients on N={grid} and N+1={check_grid} nodes disagree",
            error_estimate=error_estimate,
            independent=False,
        )

    diagnostics: dict[str, Any] = {
        "method": "periodic_trapezoid",
        "status": "converged",
        "reason": None,
        "message": f"converged on N={grid} nodes, independent check passed",
        "error_estimate": error_estimate,
        "evaluations": run.evaluations,
        "nodes": tuple(run.nodes),
        "independent_check": True,
    }
    assert set(diagnostics) == DIAGNOSTIC_KEYS
    result = u.Quantity(np.asarray(coefficients, dtype=np.complex128), unit)
    return result, diagnostics


__all__ = ["periodic_fourier"]
