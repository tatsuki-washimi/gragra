"""Tests for the periodic Fourier coefficients (P4).

Each test function is named after a required case ID in
``tests/continuous_acceptance.json``. The fixtures come from the shared
``_continuous_fixtures`` module (analytic responses only). This file does not
import the P3 modules.

Acceptance thresholds and test settings are read from
``tests/continuous_acceptance.json`` (the normalized error limit, ``atol``,
``rtol``, ``max_nodes``, the aliasing and control ``tau_min`` settings, and the
managed-array budget). The file is not modified here. The phase-fraction
checks use float-precision bounds that are local to this file and are not
part of the acceptance table.
"""

from __future__ import annotations

import json
import math
import tracemalloc
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest
from _continuous_fixtures import (
    OBSERVABLE_UNITS,
    ScalarSignal,
    constant,
    cosine,
    expand_components,
    make_response,
    periodic_coefficient_closed_form,
)
from astropy import units as u

from gragra._continuous_contract import (
    DIAGNOSTIC_KEYS,
    ContinuousQuadratureError,
    component_shape,
    normalize_relative_times,
    periodic_phase_fractions,
    validate_observable,
)
from gragra._periodic_fourier import _estimate_managed_bytes, periodic_fourier

ORDERS = (0, 1, 2, 3, 7, 8)
"""Requested orders from the P4 acceptance table."""

_ACCEPTANCE_PATH = Path(__file__).resolve().parent / "continuous_acceptance.json"
_ACCEPTANCE = json.loads(_ACCEPTANCE_PATH.read_text(encoding="utf-8"))
_P4 = _ACCEPTANCE["test_settings"]["P4"]

PERIOD = 1.0 * u.s
"""Test period. The acceptance table does not fix one for P4."""

MEMORY_BUDGET = _ACCEPTANCE["test_settings"]["memory"]["managed_budget_bytes"]
"""Managed-array budget from the acceptance table (64 MiB)."""

NORMALIZED_LIMIT = _ACCEPTANCE["thresholds"]["P4_P5_integration"][
    "max_normalized_error"
]
"""Normalized error limit from the acceptance table."""

ATOL_COEFFICIENT = _P4["atol_coefficient"]
RTOL = _P4["rtol"]
MAX_NODES = _P4["max_nodes"]
ALIASING_MAX_NODES = _P4["aliasing_max_nodes"]
ALIASING_TAU_SPEC = _P4["aliasing_tau_min"]
CONTROL_TAU_SPEC = _P4["control_tau_min"]
CONTROL_MAX_NODES = _P4["control_max_nodes"]
ALIASING_EXPECTED_N = _P4["aliasing_expected_initial_N"]


def _tau_from_spec(spec: str, period: u.Quantity) -> u.Quantity:
    """Convert a table entry such as ``"T/4"`` into a time Quantity."""
    if not spec.startswith("T/"):
        raise ValueError(f"unsupported tau_min spec {spec!r}")
    return period / float(spec[2:])


def _call(
    signal: ScalarSignal,
    observable: str,
    *,
    period: u.Quantity = PERIOD,
    orders: tuple[int, ...] = ORDERS,
    epoch: tuple[int, int] = (0, 0),
    tref: tuple[int, int] = (0, 0),
    tau_min: u.Quantity | None = None,
    max_nodes: int = MAX_NODES,
    budget: int = MEMORY_BUDGET,
    scale: float = 1.0,
):
    """Run ``periodic_fourier`` for an analytic signal with the acceptance settings."""
    if tau_min is None:
        tau_min = period / 8
    atol = ATOL_COEFFICIENT * scale * OBSERVABLE_UNITS[observable]
    return periodic_fourier(
        make_response(signal),
        observable,
        epoch=epoch,
        tref=tref,
        period=period,
        orders=orders,
        tau_min=tau_min,
        atol=atol,
        rtol=RTOL,
        max_nodes=max_nodes,
        memory_budget_bytes=budget,
    )


def _reference(
    signal: ScalarSignal,
    observable: str,
    *,
    period_s: float = 1.0,
    delta_s: float = 0.0,
    orders: tuple[int, ...] = ORDERS,
) -> u.Quantity:
    """Closed-form coefficients expanded to ``(Nf, Ntarget, *component)``."""
    scalar = periodic_coefficient_closed_form(
        signal, period_s=period_s, delta_s=delta_s, orders=orders
    )
    return expand_components(scalar, observable) * OBSERVABLE_UNITS[observable]


def _max_error(result: u.Quantity, reference: u.Quantity) -> float:
    """Largest absolute deviation between two coefficient arrays, in response units."""
    return float(np.max(np.abs(result.value - reference.value)))


def _assert_matches(result, reference, scale: float) -> None:
    """Check the normalized error against the acceptance limit."""
    assert result.shape == reference.shape
    assert _max_error(result, reference) <= NORMALIZED_LIMIT * scale


def _mp_cosine_coefficients(
    amplitude: float,
    phase: float,
    harmonic: int,
    period_ns: int,
    delta_ns: int,
    orders: tuple[int, ...],
) -> np.ndarray:
    """High-precision reference ``C_n`` for ``a cos(2π h t/T + φ)``.

    The period is ``period_ns`` nanoseconds and ``Δ = delta_ns`` nanoseconds.
    Only ``n = h`` is nonzero: ``C_h = (a/2) e^{iφ} exp(-2πi h Δ/T)``.
    """
    with mp.workdps(80):
        period = mp.mpf(period_ns) / 10**9
        delta = mp.mpf(delta_ns) / 10**9
        values = []
        for n in orders:
            if n == harmonic:
                value = (
                    (mp.mpf(amplitude) / 2)
                    * mp.exp(mp.mpc(0, 1) * mp.mpf(phase))
                    * mp.exp(-2j * mp.pi * n * delta / period)
                )
                values.append(complex(value))
            else:
                values.append(0j)
    return np.array(values, dtype=np.complex128)


# ---------------------------------------------------------------------------
# Analytic fixtures (P4 acceptance table IDs).
# ---------------------------------------------------------------------------


def test_p4_constant_coefficients():
    """A constant response has only C_0 = c, for all observables."""
    signal = constant(2.5)
    for observable in ("potential", "acceleration", "gradient"):
        result, diagnostics = _call(signal, observable, scale=2.5)
        reference = _reference(signal, observable)
        _assert_matches(result, reference, 2.5)
        assert diagnostics["status"] == "converged"


@pytest.mark.parametrize("harmonic", [1, 2, 7])
def test_p4_harmonics_1_2_7(harmonic):
    """Cosines with harmonics 1, 2, and 7 match the closed form for every observable."""
    signal = cosine(1.0, 0.3, float(harmonic))
    for observable in ("potential", "acceleration", "gradient"):
        result, _ = _call(signal, observable)
        _assert_matches(result, _reference(signal, observable), 1.0)


@pytest.mark.parametrize("amplitude", [1.0, 0.1, 1e-10])
def test_p4_amplitudes_1_0p1_1e_minus10(amplitude):
    # ID p4_amplitudes_1_0p1_1e-10 (a hyphen is not valid in a Python name).
    """The small amplitude 1e-10 is measured on its own signal scale."""
    signal = cosine(amplitude, 0.0, 1.0)
    result, _ = _call(signal, "acceleration", scale=amplitude)
    _assert_matches(result, _reference(signal, "acceleration"), amplitude)


@pytest.mark.parametrize("phase", [0.0, 0.37])
def test_p4_phases_0_0p37(phase):
    """Signal phase and reference offset Δ = epoch - tref enter the coefficients."""
    signal = cosine(1.0, phase, 1.0)
    result, _ = _call(signal, "acceleration", epoch=(0, 0), tref=(0, 370_000_000))
    # Δ = epoch - tref = -0.37 s, so the reference uses delta_s = -0.37.
    reference = _reference(signal, "acceleration", delta_s=-0.37)
    _assert_matches(result, reference, 1.0)


@pytest.mark.parametrize("period_s", [0.5, 1.0, 2.0])
def test_p4_periods_0p5_1_2(period_s):
    """The periods 0.5, 1, and 2 s give the closed-form coefficients."""
    period = period_s * u.s
    signal = cosine(1.0, 0.3, 2.0 / period_s)
    result, _ = _call(signal, "potential", period=period)
    reference = _reference(signal, "potential", period_s=period_s)
    _assert_matches(result, reference, 1.0)


def test_p4_requested_orders_and_tref_shift():
    """Requested orders are returned in order, and a tref shift multiplies C_n.

    Shifting tref by τ multiplies C_n by exp(+2πi n τ / T).
    """
    signal = cosine(1.0, 0.3, 2.0)
    base, _ = _call(signal, "acceleration", tref=(0, 0))
    shift_s = 0.123
    shifted, _ = _call(
        signal, "acceleration", tref=(0, 123_000_000)
    )
    factors = np.exp(2j * np.pi * np.array(ORDERS) * shift_s / 1.0)
    expected = base.value * factors[:, None, None]
    assert np.max(np.abs(shifted.value - expected)) <= NORMALIZED_LIMIT
    _assert_matches(base, _reference(signal, "acceleration"), 1.0)


def test_p4_output_contract_shape_dtype_unit():
    """Output is complex128, shape (Nf, Ntarget, *component), response unit."""
    signal = cosine(1.0, 0.0, 1.0)
    result, diagnostics = _call(signal, "gradient")
    assert result.shape == (len(ORDERS), 3, 3, 3)
    assert result.dtype == np.complex128
    assert result.unit == OBSERVABLE_UNITS["gradient"]
    assert set(diagnostics) == DIAGNOSTIC_KEYS
    assert diagnostics["reason"] is None
    assert diagnostics["independent_check"] is True
    assert diagnostics["nodes"] == (64, 128, 129)


# ---------------------------------------------------------------------------
# Aliasing and quadrature control.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frequency", [128.0, 256.0])
def test_p4_aliasing_cos128_independent_check_mismatch(frequency):
    """Aliasing counterexample: the N+1 check catches a missed harmonic.

    With τ_min = T/4 and max_nodes = 256 the initial grid is N = 64. The
    cos(2π·128t/T) and cos(2π·256t/T) responses alias onto the 64 and 128 grids,
    so N and 2N agree, and the N+1 independent check fails.
    """
    signal = cosine(1.0, 0.0, frequency)
    with pytest.raises(ContinuousQuadratureError) as excinfo:
        _call(
            signal,
            "acceleration",
            tau_min=_tau_from_spec(ALIASING_TAU_SPEC, PERIOD),
            max_nodes=ALIASING_MAX_NODES,
        )
    assert excinfo.value.reason == "independent_check_mismatch"
    assert excinfo.value.diagnostics["nodes"][0] == ALIASING_EXPECTED_N
    assert excinfo.value.diagnostics["nodes"] == (64, 128, 129)
    assert excinfo.value.diagnostics["reason"] == "independent_check_mismatch"


@pytest.mark.parametrize("frequency", [128.0, 256.0])
def test_p4_aliasing_control_fine_grid_matches_zero(frequency):
    """Control: a fine grid resolves the same signal, and the coefficients are zero."""
    signal = cosine(1.0, 0.0, frequency)
    result, diagnostics = _call(
        signal,
        "acceleration",
        tau_min=_tau_from_spec(CONTROL_TAU_SPEC, PERIOD),
        max_nodes=CONTROL_MAX_NODES,
    )
    _assert_matches(result, _reference(signal, "acceleration"), 1.0)
    assert diagnostics["reason"] is None


def test_p4_nonperiodic_input_not_periodic():
    """A ramp is not periodic on [0, T), so the periodicity check fails first."""
    ramp = ScalarSignal(name="ramp", evaluate=lambda t: t)
    with pytest.raises(ContinuousQuadratureError) as excinfo:
        _call(ramp, "potential")
    assert excinfo.value.reason == "not_periodic"


def test_p4_tau_min_omitted_value_error():
    """A missing or non-positive tau_min is rejected with ValueError."""
    signal = cosine(1.0, 0.0, 1.0)
    response = make_response(signal)
    common = {
        "epoch": (0, 0),
        "tref": (0, 0),
        "period": PERIOD,
        "orders": ORDERS,
        "atol": ATOL_COEFFICIENT * OBSERVABLE_UNITS["potential"],
        "rtol": RTOL,
        "max_nodes": MAX_NODES,
        "memory_budget_bytes": MEMORY_BUDGET,
    }
    with pytest.raises(ValueError):
        periodic_fourier(response, "potential", **common)
    with pytest.raises(ValueError):
        periodic_fourier(response, "potential", tau_min=0.0 * u.s, **common)
    with pytest.raises(ValueError):
        periodic_fourier(response, "potential", tau_min=-1.0 * u.s, **common)


def test_p4_max_nodes_reached_is_not_success():
    """A grid limit that stops doubling raises max_nodes_reached, not a result."""
    signal = cosine(1.0, 0.0, 128.0)
    with pytest.raises(ContinuousQuadratureError) as excinfo:
        _call(signal, "acceleration", tau_min=PERIOD / 4, max_nodes=128)
    assert excinfo.value.reason == "max_nodes_reached"


def test_p4_initial_grid_above_max_nodes_value_error():
    """An initial grid (or N+1) above max_nodes is rejected before computing."""
    signal = cosine(1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        _call(signal, "potential", tau_min=PERIOD / 4, max_nodes=64)


def test_p4_negative_control_sign_flip_and_double_fail():
    """Wrong answers (sign flip, doubled amplitude) fail the acceptance limit."""
    signal = cosine(1.0, 0.3, 2.0)
    reference = _reference(signal, "acceleration")
    flipped = ScalarSignal(name="flipped", evaluate=lambda t: -signal.evaluate(t))
    doubled = ScalarSignal(name="doubled", evaluate=lambda t: 2.0 * signal.evaluate(t))
    for wrong in (flipped, doubled):
        result, _ = _call(wrong, "acceleration")
        assert _max_error(result, reference) > NORMALIZED_LIMIT


# ---------------------------------------------------------------------------
# Phase reduction at GPS scale and in the float range.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "period_s,period_ns",
    [(0.1, 100_000_000), (0.3, 300_000_000), (0.7, 700_000_000)],
)
def test_p4_phase_integer_path_gps_scale(period_s, period_ns):
    """Integer-path phases at Δ ≈ 1e9 s match an 80-digit reference."""
    epoch = (1_000_000_123, 456_000_000)
    tref = (0, 0)
    delta_ns = 1_000_000_123 * 10**9 + 456_000_000
    fractions = periodic_phase_fractions(
        (1_000_000_123, 456_000_000), period_s * u.s, ORDERS
    )
    with mp.workdps(80):
        period = mp.mpf(period_ns) / 10**9
        delta = mp.mpf(delta_ns) / 10**9
        reference = [float(mp.frac(n * delta / period)) for n in ORDERS]
    for got, want in zip(fractions, reference, strict=True):
        difference = got - want
        difference -= round(difference)
        assert abs(difference) <= 1e-15

    # End to end: a cosine at this Δ matches the reference coefficient.
    # The harmonic-1 cosine has frequency 1/T for the period under test.
    signal = cosine(1.0, 0.3, 1.0 / period_s)
    result, _ = _call(
        signal,
        "acceleration",
        period=period_s * u.s,
        epoch=epoch,
        tref=tref,
    )
    expected = _mp_cosine_coefficients(
        1.0, 0.3, 1, period_ns, delta_ns, ORDERS
    )
    expected_full = expand_components(expected, "acceleration")
    assert np.max(np.abs(result.value - expected_full)) <= NORMALIZED_LIMIT


def test_p4_phase_float_path_inside_range():
    """T = 1/3 s (not an integer nanosecond count) matches mpmath inside the range."""
    period = (1.0 / 3.0) * u.s
    seconds, nanoseconds = 41_000, 123_000_000
    fractions = periodic_phase_fractions((seconds, nanoseconds), period, ORDERS)
    with mp.workdps(80):
        t_ref = mp.mpf(float(1.0 / 3.0))
        delta = mp.mpf(seconds) + mp.mpf(nanoseconds) / 10**9
        reference = [float(mp.frac(n * delta / t_ref)) for n in ORDERS]
    for got, want in zip(fractions, reference, strict=True):
        difference = got - want
        difference -= round(difference)
        assert abs(difference) <= 1e-9


def test_p4_phase_float_path_outside_range_value_error():
    """Outside n_max·|Δ|/T ≤ 1e6 the float path raises ValueError."""
    period = (1.0 / 3.0) * u.s
    with pytest.raises(ValueError):
        periodic_phase_fractions((41_667, 0), period, ORDERS)
    signal = cosine(1.0, 0.0, 1.0)
    response = make_response(signal)
    with pytest.raises(ValueError):
        periodic_fourier(
            response,
            "potential",
            epoch=(41_667, 0),
            tref=(0, 0),
            period=period,
            orders=ORDERS,
            tau_min=PERIOD / 8,
            atol=ATOL_COEFFICIENT * OBSERVABLE_UNITS["potential"],
            rtol=RTOL,
            max_nodes=MAX_NODES,
            memory_budget_bytes=MEMORY_BUDGET,
        )


# ---------------------------------------------------------------------------
# Memory.
# ---------------------------------------------------------------------------


ESTIMATE_OBSERVABLES = ("potential", "acceleration", "gradient")
ESTIMATE_TARGETS = (1, 8, 64)
ESTIMATE_ORDERS = {
    1: (0,),
    6: ORDERS,
    32: tuple(range(32)),
}
"""Order sets with 1, 6, and 32 members."""

# (initial N target, tau_min as a fraction of the period): tau = T / k gives
# N_phys = ceil(8 k); the realized initial N also obeys 4(n_max+1).
ESTIMATE_GRIDS = {64: 8, 1024: 128, 8192: 1024}


def _multi_target_response(n_targets: int, frequency_hz: float = 2.0):
    """Analytic response with ``n_targets`` targets, for the memory matrix.

    The signal is ``cos(2π f t + 0.3)``, which is periodic on ``T = 1 s`` for
    ``f = 2 Hz``. Each target has its own scale. Each component has weight 1.
    """

    def response(times: u.Quantity, observable: str) -> u.Quantity:
        name = validate_observable(observable)
        seconds = normalize_relative_times(times).to_value(u.s)
        tail = component_shape(name)
        base = np.cos(2.0 * np.pi * frequency_hz * seconds + 0.3)
        scales = 1.0 + 0.5 * np.arange(n_targets, dtype=np.float64)
        shape_base = (seconds.shape[0], 1) + (1,) * len(tail)
        shape_scale = (1, n_targets) + (1,) * len(tail)
        values = base.reshape(shape_base) * scales.reshape(shape_scale)
        values = np.broadcast_to(values, (seconds.shape[0], n_targets) + tail)
        return values * OBSERVABLE_UNITS[name]

    return response


@pytest.mark.parametrize("observable", ESTIMATE_OBSERVABLES)
@pytest.mark.parametrize("n_targets", ESTIMATE_TARGETS)
@pytest.mark.parametrize("n_orders", sorted(ESTIMATE_ORDERS))
@pytest.mark.parametrize("grid", sorted(ESTIMATE_GRIDS))
def test_p4_estimate_bounds_tracemalloc_peak(observable, n_targets, n_orders, grid):
    """The managed-array estimate is at least the traced peak, across 81 conditions.

    The conditions vary the observable (3), the target count (1, 8, 64), the
    order count (1, 6, 32), and the initial grid size (about 64, 1024, 8192).
    The estimate is computed from the shape and dtype of the managed arrays
    and is used as the budget, so the call must also succeed under it.
    """
    orders = ESTIMATE_ORDERS[n_orders]
    response = _multi_target_response(n_targets)
    tau_min = PERIOD / ESTIMATE_GRIDS[grid]
    component_size = int(np.prod(component_shape(observable), dtype=np.int64))
    estimate = _estimate_managed_bytes(
        n_frequencies=len(orders),
        n_targets=n_targets,
        component_size=component_size,
        block_nodes=min(1024, MAX_NODES),
    )
    atol = ATOL_COEFFICIENT * OBSERVABLE_UNITS[observable]
    tracemalloc.start()
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        _, diagnostics = periodic_fourier(
            response,
            observable,
            epoch=(0, 0),
            tref=(0, 0),
            period=PERIOD,
            orders=orders,
            tau_min=tau_min,
            atol=atol,
            rtol=RTOL,
            max_nodes=MAX_NODES,
            memory_budget_bytes=estimate,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    traced = peak - baseline
    assert traced <= estimate, (
        f"estimate {estimate} B below traced peak {traced} B "
        f"(initial N={diagnostics['nodes'][0]})"
    )


def test_p4_insufficient_budget_memory_error():
    """A budget smaller than the estimate raises MemoryError."""
    signal = cosine(1.0, 0.3, 2.0)
    with pytest.raises(MemoryError):
        _call(signal, "acceleration", budget=1000)


def test_p4_diagnostics_keys_are_exact_on_success():
    """A success returns exactly the diagnostic keys, with a None reason."""
    signal = cosine(1.0, 0.0, 1.0)
    _, diagnostics = _call(signal, "potential")
    assert set(diagnostics) == DIAGNOSTIC_KEYS
    assert diagnostics["method"] == "periodic_trapezoid"
    assert all(isinstance(n, int) for n in diagnostics["nodes"])
    assert math.isfinite(diagnostics["error_estimate"])


def test_p4_nonfinite_response_raises():
    """A non-finite response value stops the quadrature with nonfinite_response."""
    bad = ScalarSignal(name="bad", evaluate=lambda t: np.where(t > 0.5, np.nan, 0.0))
    with pytest.raises(ContinuousQuadratureError) as excinfo:
        _call(bad, "potential")
    assert excinfo.value.reason == "nonfinite_response"
