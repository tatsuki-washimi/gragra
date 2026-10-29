"""Contract tests for the continuous-time lanes (C0)."""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest
from _continuous_fixtures import (
    TARGET_SCALES,
    circular_pose,
    constant,
    constant_rotation_pose,
    constant_velocity_pose,
    cosine,
    gaussian,
    make_response,
    periodic_coefficient_closed_form,
    static_pose,
    window_transform_closed_form,
)
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeriesMatrix
from gwpy.time import LIGOTimeGPS

from gragra._continuous_contract import (
    DIAGNOSTIC_KEYS,
    FAILURE_REASONS,
    OBSERVABLE_COMPONENT_SHAPES,
    PHASE_CYCLE_LIMIT,
    ContinuousQuadratureError,
    PoseFunction,
    check_memory_budget,
    check_window_phase_range,
    component_shape,
    component_tolerance,
    gps_delta_parts,
    integer_period_nanoseconds,
    nominal_period_seconds,
    normalize_relative_times,
    periodic_phase_fractions,
    validate_observable,
)
from gragra.geometry import Frame
from gragra.gravity_sources import (
    PointMassCloud,
    PrescribedRigidTransform,
    rigid_motion,
)
from gragra.results import EvaluationContext, EvaluationResult, GravityResult
from gragra.results._spectral import (
    AmplitudeSpectrumRepresentation,
    FiniteRecordInterpretation,
    FourierTransformRepresentation,
    Sidedness,
)
from gragra.results._time import TimeSampling
from gragra.results._time_grid import build_uniform_time_s
from gragra.targets import TargetPoints

NS = 10**9
PHASE_TOL_CYCLES = 1e-15
FLOAT_OUTPUT_TOL = 1e-15


# normalize_relative_times -------------------------------------------------------


def test_normalize_scalar_becomes_length_one_in_seconds():
    out = normalize_relative_times(0.25 * u.s)
    assert out.shape == (1,)
    assert out.unit == u.s
    assert out.value[0] == 0.25


def test_normalize_converts_units_and_preserves_order_and_duplicates():
    out = normalize_relative_times(u.Quantity([3.0, 1.0, 1.0, 500.0], u.ms))
    assert out.unit == u.s
    np.testing.assert_allclose(out.value, [0.003, 0.001, 0.001, 0.5])


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ([0.0, 1.0], TypeError),  # plain list, not a Quantity
        (np.array([0.0, 1.0]), TypeError),  # plain ndarray
        (u.Quantity([1 + 1j], u.s), TypeError),  # complex
        (u.Quantity(np.array([True]), u.s, dtype=bool), TypeError),  # bool
        (u.Quantity([0.0, 1.0], u.m), ValueError),  # not a time unit
        (u.Quantity([], u.s), ValueError),  # empty
        (u.Quantity([[0.0, 1.0]], u.s), ValueError),  # multi-dimensional
        (u.Quantity([0.0, np.nan], u.s), ValueError),  # non-finite
        (u.Quantity([np.inf], u.s), ValueError),  # non-finite
    ],
)
def test_normalize_rejects_invalid_times(value, error):
    with pytest.raises(error):
        normalize_relative_times(value)


# observables and component shapes ----------------------------------------------


def test_observable_names_and_shapes():
    for name in ("potential", "acceleration", "gradient"):
        assert validate_observable(name) == name
    assert component_shape("potential") == ()
    assert component_shape("acceleration") == (3,)
    assert component_shape("gradient") == (3, 3)
    assert set(OBSERVABLE_COMPONENT_SHAPES) == {"potential", "acceleration", "gradient"}


def test_observable_rejects_unknown_and_non_string():
    with pytest.raises(ValueError):
        validate_observable("velocity")
    with pytest.raises(TypeError):
        validate_observable(1)


# GPS phase bookkeeping ----------------------------------------------------------


def test_gps_delta_parts_carries_nanoseconds_and_keeps_sign():
    assert gps_delta_parts((10, 0), (3, 500_000_000)) == (6, 500_000_000)
    assert gps_delta_parts((3, 0), (10, 0)) == (-7, 0)
    assert gps_delta_parts((5, 900_000_000), (5, 100_000_000)) == (0, 800_000_000)


@pytest.mark.parametrize(
    ("period_s", "expected_ns"),
    [(0.1, 100_000_000), (0.3, 300_000_000), (0.7, 700_000_000), (1.5, 1_500_000_000)],
)
def test_integer_path_applies_to_decimal_periods(period_s, expected_ns):
    assert integer_period_nanoseconds(period_s * u.s) == expected_ns
    assert nominal_period_seconds(period_s * u.s) == pytest.approx(
        expected_ns * 1e-9, rel=1e-15
    )


def test_one_third_second_uses_float_path():
    period = 1.0 / 3.0 * u.s
    assert integer_period_nanoseconds(period) is None
    assert nominal_period_seconds(period) == 1.0 / 3.0


@pytest.mark.parametrize("period_s", [0.1, 0.3, 0.7])
@pytest.mark.parametrize("delta", [(NS, 0), (NS, 123_456_789), (-NS, 987_654_321)])
def test_integer_phase_matches_mpmath_at_gps_scale(period_s, delta):
    orders = (0, 1, 2, 3, 7, 8, -3)
    got = periodic_phase_fractions(delta, period_s * u.s, orders)
    period_ns = integer_period_nanoseconds(period_s * u.s)
    with mp.workdps(60):
        t_mp = mp.mpf(period_ns) / NS
        d_mp = mp.mpf(delta[0]) + mp.mpf(delta[1]) / NS
        ref = []
        for n in orders:
            q = n * d_mp / t_mp
            ref.append(float(q - mp.floor(q)))
    assert got.shape == (len(orders),)
    assert np.all((got >= 0.0) & (got < 1.0))
    assert np.max(np.abs(got - np.array(ref))) < PHASE_TOL_CYCLES


def test_float_phase_inside_range_matches_mpmath():
    period = 1.0 / 3.0
    orders = (0, 1, 2, 3, 7, 8)
    delta = (10, 250_000_000)
    got = periodic_phase_fractions(delta, period * u.s, orders)
    with mp.workdps(60):
        t_mp = mp.mpf(period)
        d_mp = mp.mpf(delta[0]) + mp.mpf(delta[1]) / NS
        ref = []
        for n in orders:
            q = n * d_mp / t_mp
            ref.append(float(q - mp.floor(q)))
    assert np.max(np.abs(got - np.array(ref))) < 1e-9


def test_float_phase_outside_range_raises_value_error():
    with pytest.raises(ValueError, match="float phase path"):
        periodic_phase_fractions((NS, 0), (1.0 / 3.0) * u.s, (0, 1, 8))


def test_float_phase_range_boundary():
    # n_max = 8, T = 1/3 s: 8 * 41666 * 3 = 999984 <= 1e6 (accepted),
    # 8 * 41667 * 3 = 1000008 > 1e6 (rejected).
    periodic_phase_fractions((41_666, 0), (1.0 / 3.0) * u.s, (8,))
    with pytest.raises(ValueError):
        periodic_phase_fractions((41_667, 0), (1.0 / 3.0) * u.s, (8,))


def test_window_phase_range():
    check_window_phase_range((1_000_000, 0), 1.0)
    check_window_phase_range((NS, 0), 0.0)
    with pytest.raises(ValueError):
        check_window_phase_range((1_000_001, 0), 1.0)
    with pytest.raises(ValueError):
        check_window_phase_range((NS, 0), -1.0)


def test_tref_shift_multiplies_periodic_phase_by_exp_plus_i_harmonic():
    """tref -> tref + tau multiplies C_n by exp(+2πi n tau / T)."""
    period = 1.0
    tau = 0.125
    orders = (1, 2, 3)
    signal = cosine(amplitude=1.3, phase=0.4, frequency_hz=3.0)
    delta = 0.25
    base = periodic_coefficient_closed_form(
        signal, period_s=period, delta_s=delta, orders=orders
    )
    shifted = periodic_coefficient_closed_form(
        signal, period_s=period, delta_s=delta - tau, orders=orders
    )
    expected = np.exp(2j * np.pi * 3.0 * tau)
    np.testing.assert_allclose(shifted[2] / base[2], expected, rtol=1e-12)


def test_tref_shift_window_transform_multiplies_by_exp_plus_i_f_tau():
    tau = 0.125
    signal = cosine(amplitude=0.9, phase=-0.2, frequency_hz=7.5)
    freqs = np.array([0.0, 0.125, 1.0, 4.0])
    base = window_transform_closed_form(
        signal, delta_s=0.37, t0=-2.0, t1=2.0, frequencies_hz=freqs
    )
    shifted = window_transform_closed_form(
        signal, delta_s=0.37 - tau, t0=-2.0, t1=2.0, frequencies_hz=freqs
    )
    np.testing.assert_allclose(
        shifted / base, np.exp(2j * np.pi * freqs * tau), rtol=1e-12
    )


# quadrature error and memory ---------------------------------------------------


def _diagnostics(reason: str) -> dict:
    values = dict.fromkeys(DIAGNOSTIC_KEYS)
    values["reason"] = reason
    return values


@pytest.mark.parametrize("reason", sorted(FAILURE_REASONS))
def test_quadrature_error_accepts_each_reason(reason):
    err = ContinuousQuadratureError(
        "failed", reason=reason, diagnostics=_diagnostics(reason)
    )
    assert isinstance(err, RuntimeError)
    assert err.reason == reason
    assert err.diagnostics["reason"] == reason
    assert set(err.diagnostics) == DIAGNOSTIC_KEYS


def test_quadrature_error_rejects_unknown_reason():
    with pytest.raises(ValueError):
        ContinuousQuadratureError(
            "failed", reason="made_up", diagnostics=_diagnostics("made_up")
        )


def test_quadrature_error_rejects_wrong_diagnostic_keys():
    values = _diagnostics("not_periodic")
    values.pop("nodes")
    with pytest.raises(ValueError, match="diagnostics keys"):
        ContinuousQuadratureError("failed", reason="not_periodic", diagnostics=values)


def test_quadrature_error_rejects_mismatched_reason():
    with pytest.raises(ValueError, match="must equal"):
        ContinuousQuadratureError(
            "failed",
            reason="not_periodic",
            diagnostics=_diagnostics("max_nodes_reached"),
        )


def test_memory_budget_raises_when_exceeded():
    check_memory_budget(10, 10)
    with pytest.raises(MemoryError):
        check_memory_budget(11, 10)
    with pytest.raises(TypeError):
        check_memory_budget(1.0, 10)
    with pytest.raises(ValueError):
        check_memory_budget(1, 0)


def test_component_tolerance_combines_atol_and_scaled_rtol():
    atol = np.array([1e-10, 2e-10]) * u.m
    magnitude = np.array([1.0, 0.0]) * u.mm
    tol = component_tolerance(atol, 1e-3, magnitude)
    np.testing.assert_allclose(tol, [1e-10 + 1e-3 * 1e-3, 2e-10])
    with pytest.raises(TypeError):
        component_tolerance(np.array([1.0]), 1e-3, magnitude)
    with pytest.raises(ValueError):
        component_tolerance(atol, -1e-3, magnitude)


# independent mpmath references for the analytic fixtures ------------------------


def _mp_periodic_coefficient(signal_fn, period, delta, n):
    """C_n by mpmath.quad over one period, with the closed-form phase convention."""
    t0 = mp.mpf("-0.3") * period
    pts = [t0 + period * k / 8 for k in range(9)]

    def integrand(t):
        return signal_fn(t) * mp.exp(-2j * mp.pi * n * (delta + t) / period)

    return mp.quad(integrand, pts) / period


@pytest.mark.parametrize(
    ("signal", "signal_mp", "orders"),
    [
        (
            constant(0.8),
            lambda _t: mp.mpf("0.8"),
            (0, 1, 2),
        ),
        (
            cosine(1.7, 0.37, 2.0),
            lambda t: mp.mpf("1.7") * mp.cos(2 * mp.pi * 2 * t + mp.mpf("0.37")),
            (0, 1, 2, 3),
        ),
    ],
)
def test_periodic_closed_form_matches_mpmath_quad(signal, signal_mp, orders):
    period = 1.0
    delta = 0.25
    with mp.workdps(50):
        ref = [
            _mp_periodic_coefficient(
                signal_mp, mp.mpf(period), mp.mpf(delta), mp.mpf(n)
            )
            for n in orders
        ]
    closed = periodic_coefficient_closed_form(
        signal, period_s=period, delta_s=delta, orders=orders
    )
    for value, expected in zip(closed, ref, strict=True):
        assert abs(complex(value) - complex(expected)) <= FLOAT_OUTPUT_TOL * max(
            1.0, abs(complex(expected))
        )


def _mp_window_transform(signal_mp, freq, delta, breakpoints):
    def integrand(t):
        return signal_mp(t) * mp.exp(-2j * mp.pi * freq * (delta + t))

    return mp.quad(integrand, breakpoints)


@pytest.mark.parametrize(
    ("signal", "signal_mp"),
    [
        (constant(0.8), lambda _t: mp.mpf("0.8")),
        (
            cosine(0.9, -0.2, 7.5),
            lambda t: (
                mp.mpf("0.9") * mp.cos(2 * mp.pi * mp.mpf("7.5") * t - mp.mpf("0.2"))
            ),
        ),
    ],
)
def test_finite_window_closed_form_matches_mpmath_quad(signal, signal_mp):
    t0, t1, delta = -2.0, 2.0, 0.37
    freqs = np.array([0.0, 0.125, 1.0, 4.0, 16.0])
    closed = window_transform_closed_form(
        signal, delta_s=delta, t0=t0, t1=t1, frequencies_hz=freqs
    )
    breakpoints = [mp.mpf(t0) + (mp.mpf(t1) - t0) * k / 64 for k in range(65)]
    with mp.workdps(50):
        for value, freq in zip(closed, freqs, strict=True):
            ref = _mp_window_transform(
                signal_mp,
                mp.mpf(float(freq)),
                mp.mpf(delta),
                breakpoints,
            )
            assert abs(complex(value) - complex(ref)) <= FLOAT_OUTPUT_TOL * max(
                1.0, abs(complex(ref))
            )


def test_gaussian_window_closed_form_matches_mpmath_quad():
    amplitude, centre, sigma = 1.0, 0.37, 1e-4
    signal = gaussian(amplitude, centre, sigma)
    t0, t1, delta = -2.0, 2.0, 0.0
    freqs = np.array([0.0, 1.0, 1000.0])
    closed = window_transform_closed_form(
        signal, delta_s=delta, t0=t0, t1=t1, frequencies_hz=freqs
    )
    with mp.workdps(60):
        tc, sg = mp.mpf(centre), mp.mpf(sigma)
        gauss = lambda t: mp.exp(-((t - tc) ** 2) / (2 * sg**2))  # noqa: E731
        points = [
            mp.mpf(t0),
            tc - 10 * sg,
            tc - 3 * sg,
            tc,
            tc + 3 * sg,
            tc + 10 * sg,
            mp.mpf(t1),
        ]
        for value, freq in zip(closed, freqs, strict=True):
            f = mp.mpf(float(freq))
            ref = mp.quad(lambda t, f=f: gauss(t) * mp.exp(-2j * mp.pi * f * t), points)
            assert abs(complex(value) - complex(ref)) <= FLOAT_OUTPUT_TOL * max(
                1.0, abs(complex(ref))
            )


def test_response_fixture_is_time_first_with_si_units():
    response = make_response(cosine(1.0, 0.0, 2.0))
    times = u.Quantity([0.0, 0.25, 0.5], u.s)
    potential = response(times, "potential")
    acceleration = response(times, "acceleration")
    gradient = response(times, "gradient")
    assert potential.shape == (3, 3)
    assert acceleration.shape == (3, 3, 3)
    assert gradient.shape == (3, 3, 3, 3)
    assert potential.unit == u.m**2 / u.s**2
    assert acceleration.unit == u.m / u.s**2
    assert gradient.unit == 1 / u.s**2
    np.testing.assert_allclose(
        potential.value[1], TARGET_SCALES * math.cos(2 * math.pi * 2.0 * 0.25)
    )


def test_pose_fixtures_satisfy_pose_protocol_and_shapes():
    times = u.Quantity([0.0, 0.25, 0.5], u.s)
    poses = [
        static_pose(),
        constant_velocity_pose((1.0, 0.0, 0.0)),
        constant_rotation_pose(),
        circular_pose(),
    ]
    for pose in poses:
        assert isinstance(pose, PoseFunction)
        rotation, translation = pose(times)
        assert len(rotation) == 3
        assert translation.shape == (3, 3)
        assert translation.unit == u.m


def test_circular_pose_has_unit_radius_and_unit_period():
    rotation, translation = circular_pose()(u.Quantity([0.0, 0.25, 1.0], u.s))
    radii = np.hypot(translation.value[:, 0], translation.value[:, 1])
    np.testing.assert_allclose(radii, 1.0, atol=1e-15)
    np.testing.assert_allclose(translation.value[0], translation.value[2], atol=1e-12)


def test_constant_velocity_pose_translation_is_linear():
    _, translation = constant_velocity_pose((2.0, 0.0, -1.0))(
        u.Quantity([0.0, 0.5], u.s)
    )
    np.testing.assert_allclose(translation.value, [[0, 0, 0], [1, 0, -0.5]])


# Result-type runtime compatibility (C0-5) --------------------------------------

WORLD = Frame("world")
NONUNIFORM_HZ = np.array([0.0, 0.7, 3.0, 11.0])
SINGLETON_DC_HZ = np.array([0.0])


@pytest.fixture(scope="module")
def base_result():
    """Time-series rigid-motion result used only for its model spec and record."""
    source = PointMassCloud([1.0], [[0.0, 0.0, 0.0]])
    m, dt, epoch = 64, 1.0 / 64.0, (0, 0)
    sampling = TimeSampling(epoch=LIGOTimeGPS(*epoch), dt=dt * u.s, count=m)
    t = build_uniform_time_s(epoch, dt, m)
    translations = np.zeros((m, 3))
    translations[:, 0] = 0.5 * np.sin(2 * np.pi * 4.0 * t)
    transform = PrescribedRigidTransform(
        t, translations, np.repeat(np.eye(3)[None], m, axis=0)
    )
    targets = TargetPoints(
        np.array([[5.0, 0.0, 0.0], [0.0, 4.0, 0.0]]) * u.m,
        ("t0", "t1"),
        frame=WORLD,
    )
    return (
        rigid_motion(source, transform, frame=WORLD, sampling=sampling)
        .at(targets, observables=("potential", "acceleration"))
        .run()
    )


def _fourier_context(
    base, frequencies_hz, representation, observables=("acceleration",)
):
    return EvaluationContext(
        model_spec=base.context.model_spec,
        targets=base.context.targets,
        frame=base.context.frame,
        observables=observables,
        representation=representation,
        frequencies=np.asarray(frequencies_hz, dtype=np.float64) * u.Hz,
        reference_time=(0, 0),
    )


def _spectral_matrix(context, observable, si_unit):
    """Build a complex FrequencySeriesMatrix with the context's labels and axis."""
    binding = context.binding(observable)
    rows = list(binding.target_ids)
    cols = ["x", "y", "z"] if observable == "acceleration" else ["potential"]
    freqs = np.asarray(binding.frequencies_hz, dtype=np.float64)
    shape = (len(rows), len(cols), len(freqs))
    values = np.arange(np.prod(shape), dtype=np.complex128).reshape(shape) + 1j
    units = np.empty((len(rows), len(cols)), dtype=object)
    units[...] = si_unit
    return FrequencySeriesMatrix(
        values,
        frequencies=freqs * u.Hz,
        units=units,
        rows=rows,
        cols=cols,
        name=f"c0_{observable}",
    )


FOURIER = FourierTransformRepresentation(sidedness=Sidedness.ONE_SIDED)
AMPLITUDE = AmplitudeSpectrumRepresentation(
    record_interpretation=FiniteRecordInterpretation.PERIODIC
)


@pytest.mark.parametrize("frequencies", [NONUNIFORM_HZ, SINGLETON_DC_HZ])
def test_fourier_context_accepts_nonuniform_and_dc_frequencies(
    base_result, frequencies
):
    ctx = _fourier_context(base_result, frequencies, FOURIER)
    assert isinstance(ctx.representation, FourierTransformRepresentation)
    assert ctx.representation.sidedness is Sidedness.ONE_SIDED
    assert ctx.frequencies_hz == tuple(float(f) for f in frequencies)
    ctx.validate()


@pytest.mark.parametrize("frequencies", [NONUNIFORM_HZ, SINGLETON_DC_HZ])
def test_amplitude_context_accepts_nonuniform_and_dc_frequencies(
    base_result, frequencies
):
    ctx = _fourier_context(base_result, frequencies, AMPLITUDE)
    assert isinstance(ctx.representation, AmplitudeSpectrumRepresentation)
    assert (
        ctx.representation.record_interpretation is FiniteRecordInterpretation.PERIODIC
    )
    assert ctx.frequencies_hz == tuple(float(f) for f in frequencies)
    ctx.validate()


@pytest.mark.parametrize("representation", [FOURIER, AMPLITUDE])
@pytest.mark.parametrize("frequencies", [NONUNIFORM_HZ, SINGLETON_DC_HZ])
def test_evaluation_result_constructs_and_validates(
    base_result, representation, frequencies
):
    ctx = _fourier_context(
        base_result,
        frequencies,
        representation,
        observables=("potential", "acceleration"),
    )
    # Fourier payloads carry unit Q·s, amplitude spectra carry unit Q.
    if isinstance(representation, FourierTransformRepresentation):
        acc_unit, pot_unit = u.m / u.s, u.m**2 / u.s
    else:
        acc_unit, pot_unit = u.m / u.s**2, u.m**2 / u.s**2
    matrix = _spectral_matrix(ctx, "acceleration", acc_unit)
    potential_matrix = _spectral_matrix(ctx, "potential", pot_unit)
    result = EvaluationResult(
        GravityResult(
            name="c0",
            potential=potential_matrix,
            acceleration=matrix,
        ),
        evaluation=base_result.evaluation,
        record=base_result.record,
        context=ctx,
    )
    result.validate()
    assert result.observables == ("potential", "acceleration")


def test_evaluation_result_select_narrows_observables(base_result):
    ctx = _fourier_context(
        base_result, NONUNIFORM_HZ, FOURIER, observables=("potential", "acceleration")
    )
    result = EvaluationResult(
        GravityResult(
            name="c0",
            potential=_spectral_matrix(ctx, "potential", u.m**2 / u.s),
            acceleration=_spectral_matrix(ctx, "acceleration", u.m / u.s),
        ),
        evaluation=base_result.evaluation,
        record=base_result.record,
        context=ctx,
    )
    narrowed = result.select("acceleration")
    narrowed.validate()
    assert narrowed.observables == ("acceleration",)
    assert narrowed.potential is None


def test_frequency_axis_slice_keeps_labels_and_values(base_result):
    ctx = _fourier_context(base_result, NONUNIFORM_HZ, FOURIER)
    matrix = _spectral_matrix(ctx, "acceleration", u.m / u.s)
    freqs = np.asarray(matrix.frequencies.to_value(u.Hz))
    np.testing.assert_allclose(freqs, NONUNIFORM_HZ)
    sliced = matrix[:, :, 1:3]
    assert sliced.shape == (2, 3, 2)
    np.testing.assert_allclose(
        np.asarray(sliced.frequencies.to_value(u.Hz)), NONUNIFORM_HZ[1:3]
    )
    assert tuple(sliced.rows.keys()) == ("t0", "t1")
    assert tuple(sliced.cols.keys()) == ("x", "y", "z")


def test_spectral_units_and_labels_are_carried(base_result):
    ctx = _fourier_context(base_result, NONUNIFORM_HZ, FOURIER)
    matrix = _spectral_matrix(ctx, "acceleration", u.m / u.s)
    assert matrix.meta[0, 0].unit == u.m / u.s
    assert tuple(matrix.rows.keys()) == ("t0", "t1")
    assert tuple(matrix.cols.keys()) == ("x", "y", "z")


# Acceptance table (C0c) ---------------------------------------------------------

ACCEPTANCE_PATH = Path(__file__).resolve().parent / "continuous_acceptance.json"
LANES = ("C0", "P3", "P4", "P5", "I")
CASE_ID = re.compile(r"^[a-z0-9_.+\-]+$")


def _acceptance() -> dict:
    return json.loads(ACCEPTANCE_PATH.read_text(encoding="utf-8"))


def test_acceptance_table_fields_types_and_section_9_values():
    data = _acceptance()
    for key in (
        "thresholds",
        "scale_definitions",
        "test_settings",
        "p5_panel_rule",
        "p5_fixture_n_init",
        "p5_measured_evaluations",
        "p5_max_evaluations",
        "p4_expected_representation",
        "required_cases",
        "failure_policy",
    ):
        assert key in data, key
    thresholds = data["thresholds"]
    assert thresholds["P3"]["max_normalized_error"] == 1e-12
    assert thresholds["P3"]["reference_stability"] == 1e-14
    assert thresholds["P4_P5_integration"]["max_normalized_error"] == 1e-8
    assert thresholds["P4_P5_integration"]["reference_stability"] == 1e-10
    for section in thresholds.values():
        if isinstance(section, dict):
            for value in section.values():
                assert isinstance(value, float)
    settings = data["test_settings"]
    assert settings["phase_cycle_limit"] == PHASE_CYCLE_LIMIT
    assert settings["memory"]["managed_budget_bytes"] == 64 * 1024**2
    assert settings["P5"]["cache_size_bytes"] == 8 * 1024**2
    assert settings["P5"]["epsabs"] == 1.0
    assert settings["P5"]["epsrel"] == 0.0
    assert settings["P5"]["norm"] == "max"
    assert settings["P5"]["workers"] == 1
    assert settings["P4"]["max_nodes"] == 65536
    assert settings["P4"]["aliasing_max_nodes"] == 256
    assert settings["P4"]["control_max_nodes"] == 65536
    assert data["p4_expected_representation"]["descriptor"] == (
        "AmplitudeSpectrumRepresentation(record_interpretation=PERIODIC)"
    )


def test_acceptance_p5_panel_and_max_evaluations_are_consistent():
    data = _acceptance()
    rule = data["p5_panel_rule"]
    assert rule["c1"] in (4, 8, 16)
    assert rule["c2"] in (0.125, 0.25, 0.5)
    measured = data["p5_measured_evaluations"]
    limits = data["p5_max_evaluations"]
    assert set(measured) == set(limits) - {"rule"} == set(data["p5_fixture_n_init"])
    for name, evaluations in measured.items():
        assert isinstance(evaluations, int) and evaluations > 0
        expected = math.ceil(2 * evaluations / 1000) * 1000
        assert limits[name] == expected, name
        assert limits[name] % 1000 == 0


def test_acceptance_required_cases_are_complete_and_well_formed():
    cases = _acceptance()["required_cases"]
    assert set(cases) == set(LANES)
    seen: set[str] = set()
    for lane in LANES:
        ids = cases[lane]
        assert isinstance(ids, list) and ids, lane
        for case_id in ids:
            assert isinstance(case_id, str) and CASE_ID.match(case_id), case_id
            assert case_id not in seen, case_id
            seen.add(case_id)


def test_node_map_c0_functions_exist_in_this_file():
    """Each C0 required case maps to real test functions in this file.

    Only the C0 lane is checked here. The P3, P4, P5 and I lists are filled by
    their own lanes and are not verified by this test.
    """
    data = _acceptance()
    c0_ids = data["required_cases"]["C0"]
    c0_map = data["test_node_map"]["C0"]
    assert set(c0_map) == set(c0_ids)
    this_file = Path(__file__).resolve()
    tree = ast.parse(this_file.read_text(encoding="utf-8"))
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    for case_id, node_ids in c0_map.items():
        assert node_ids, case_id
        for node_id in node_ids:
            path, separator, function = node_id.partition("::")
            assert separator == "::" and function, node_id
            target = (this_file.parent.parent / path).resolve()
            assert target == this_file, node_id
            assert function in defined, node_id
