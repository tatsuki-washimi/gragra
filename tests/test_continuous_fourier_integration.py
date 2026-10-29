"""Integration tests for the continuous Fourier adapter (lane I).

Required case IDs are listed in ``tests/continuous_acceptance.json`` under
``required_cases.I``. The mapping to pytest node ids lives in the separate
manifest ``~/work/gragra-archive/p3p4p5-20261018/sealed/test_node_map.json``
(not in the acceptance table).
"""

from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

pytest.importorskip("mpmath")
pytest.importorskip("scipy")

from _continuous_fixtures import circular_pose, constant_velocity_pose  # noqa: E402
from astropy import units as u  # noqa: E402
from gwpy.time import LIGOTimeGPS  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from gragra._continuous_contract import (  # noqa: E402
    gps_delta_parts,
)
from gragra._continuous_fourier import (  # noqa: E402
    DEFAULT_MAX_EVALUATIONS,
    DEFAULT_MEMORY_BUDGET_BYTES,
    ContinuousFourierModel,
)
from gragra._continuous_rigid_motion import make_rigid_pose_response  # noqa: E402
from gragra._periodic_fourier import periodic_fourier  # noqa: E402
from gragra.constants import G_SI  # noqa: E402
from gragra.geometry._frame import Frame  # noqa: E402
from gragra.gravity_sources import PointMassCloud  # noqa: E402
from gragra.results._spectral import (  # noqa: E402
    AmplitudeSpectrumRepresentation,
    FiniteRecordInterpretation,
    FourierTransformRepresentation,
    Sidedness,
)
from gragra.targets._high_points import TargetPoints  # noqa: E402
from gragra.targets.points import TargetPoints as NumericalTargetPoints  # noqa: E402

_ROOT = Path(__file__).resolve().parent
_ACCEPTANCE = json.loads((_ROOT / "continuous_acceptance.json").read_text())
_LIMIT = _ACCEPTANCE["thresholds"]["P4_P5_integration"]["max_normalized_error"]
_P4 = _ACCEPTANCE["test_settings"]["P4"]
_P5 = _ACCEPTANCE["test_settings"]["P5"]
_MEMORY = _ACCEPTANCE["test_settings"]["memory"]["managed_budget_bytes"]
_OBS_UNITS = {
    "potential": u.m**2 / u.s**2,
    "acceleration": u.m / u.s**2,
    "gradient": 1 / u.s**2,
}
WORLD = Frame("world")
_EPOCH_GPS = LIGOTimeGPS(1_000_000_000, 0)


def _hl_targets(
    positions_m: np.ndarray,
    *,
    ids: tuple[str, ...] | None = None,
    frame: Frame = WORLD,
) -> TargetPoints:
    count = len(positions_m)
    if ids is None:
        ids = tuple(f"t{i}" for i in range(count))
    return TargetPoints(positions_m * u.m, ids=ids, frame=frame)


def _local_frame() -> Frame:
    return Frame(
        "local",
        origin=np.array([1.0, -0.5, 2.0]) * u.m,
        basis=Rotation.from_euler("z", 0.4).as_matrix(),
        parent=WORLD,
    )


def _normalized_error(got: np.ndarray, expected: np.ndarray, scale: float) -> float:
    diff = np.asarray(got) - np.asarray(expected)
    denom = scale if scale > 0.0 else 1.0
    return float(np.max(np.abs(diff)) / denom)


def _circular_source() -> PointMassCloud:
    """1 kg at the body origin (world position follows the pose translation)."""
    return PointMassCloud(np.array([1.0]), np.array([[0.0, 0.0, 0.0]]))


def _integration_atol(observable: str, scale: float) -> u.Quantity:
    return _P4["atol_coefficient"] * scale * _OBS_UNITS[observable]


def _window_atol(observable: str, scale: float, width: float) -> u.Quantity:
    return _P5["atol_coefficient"] * scale * width * _OBS_UNITS[observable] * u.s


def _mp_periodic_coefficient(
    response,
    *,
    observable: str,
    period_s: float,
    delta_s: float,
    order: int,
    n_samples: int = 4096,
) -> complex:
    """Reference ``C_n`` by trapezoidal rule in mpmath (50 digits)."""
    times = np.linspace(0.0, period_s, n_samples, endpoint=False)
    values = response(times * u.s, observable).to_value(u.m / u.s**2)
    # Single target at origin, x component only for this fixture.
    series = values[:, 0, 0]
    with mp.workdps(50):
        total = mp.mpc(0)
        dt = mp.mpf(period_s / n_samples)
        for t, q in zip(times, series, strict=True):
            exponent = (
                -2j
                * mp.pi
                * mp.mpf(order)
                * (mp.mpf(delta_s) + mp.mpf(t))
                / mp.mpf(period_s)
            )
            phase = mp.exp(exponent)
            total += mp.mpf(q) * phase
        return complex(total * dt / mp.mpf(period_s))


def test_i_circular_motion_acceleration_n1_closed_form():
    """Circular 1 kg, 1 m, 1 s: only harmonic n=1 is significant at the origin."""
    model = ContinuousFourierModel(_circular_source(), circular_pose(), frame=WORLD)
    period = 1.0 * u.s
    orders = np.array([0, 1, 2, 3], dtype=int)
    scale = G_SI  # minimum separation is 1 m
    evaluation = model.periodic_at(
        _hl_targets(np.zeros((1, 3))),
        epoch=LIGOTimeGPS(0, 0),
        tref=LIGOTimeGPS(0, 0),
        period=period,
        orders=orders,
        tau_min=period / 8.0,
        atol=_integration_atol("acceleration", scale),
        rtol=_P4["rtol"],
        max_nodes=_P4["max_nodes"],
        memory_budget_bytes=_MEMORY,
        observables=("acceleration",),
    )
    result, _ = evaluation.run_with_diagnostics()
    matrix = result.acceleration
    freqs = np.asarray(matrix.frequencies.to_value(u.Hz))
    values = np.asarray(matrix[0, 0])  # target 0, x component
    delta = gps_delta_parts(LIGOTimeGPS(0, 0), LIGOTimeGPS(0, 0))
    delta_s = delta[0] + delta[1] * 1e-9
    ref1 = _mp_periodic_coefficient(
        make_rigid_pose_response(
            _circular_source(),
            circular_pose(),
            NumericalTargetPoints(np.zeros((1, 3))),
            frame=WORLD,
            memory_budget_bytes=_MEMORY,
        ),
        observable="acceleration",
        period_s=1.0,
        delta_s=delta_s,
        order=1,
    )
    # Stored amplitudes: A_n = 2 C_n for n > 0.
    idx1 = int(np.where(np.isclose(freqs, 1.0))[0][0])
    err1 = _normalized_error(np.array([values[idx1]]), np.array([2.0 * ref1]), scale)
    assert err1 <= _LIMIT
    for order in (0, 2, 3):
        idx = int(np.where(np.isclose(freqs, float(order)))[0][0])
        err = _normalized_error(np.array([values[idx]]), np.array([0.0 + 0.0j]), scale)
        assert err <= _LIMIT


def test_i_passage_window_mpmath_integral():
    """Passing motion c(t)=(t,1,0) m; mpmath quad oracle on the Newton response."""
    pose = constant_velocity_pose(
        velocity_m_s=(1.0, 0.0, 0.0),
        origin_m=(0.0, 1.0, 0.0),
    )
    model = ContinuousFourierModel(_circular_source(), pose, frame=WORLD)
    freqs = np.array([0.0, 0.125, 1.0, 4.0]) * u.Hz
    t0, t1 = -2.0 * u.s, 2.0 * u.s
    width = 4.0
    scale = G_SI  # closest approach 1 m
    evaluation = model.window_at(
        _hl_targets(np.zeros((1, 3))),
        epoch=_EPOCH_GPS,
        tref=_EPOCH_GPS,
        t_start=t0,
        t_stop=t1,
        frequencies=freqs,
        tau_min=width / 8.0 * u.s,
        atol=_window_atol("acceleration", scale, width),
        rtol=_P5["rtol"],
        max_evaluations=DEFAULT_MAX_EVALUATIONS,
        memory_budget_bytes=_MEMORY,
        observables=("acceleration",),
    )
    result, _ = evaluation.run_with_diagnostics()
    response = make_rigid_pose_response(
        _circular_source(),
        pose,
        _hl_targets(np.zeros((1, 3))),
        frame=WORLD,
        memory_budget_bytes=_MEMORY,
    )
    delta = gps_delta_parts(_EPOCH_GPS, _EPOCH_GPS)
    delta_s = delta[0] + delta[1] * 1e-9
    matrix = result.acceleration
    got = np.asarray(matrix[0, 0])
    t0_s = t0.to_value(u.s)
    t1_s = t1.to_value(u.s)
    with mp.workdps(60):

        def sample_q(t_s: float) -> float:
            accel = response(np.array([t_s]) * u.s, "acceleration")
            return accel.to_value(u.m / u.s**2)[0, 0, 0]

        for k, f_hz in enumerate(freqs.to_value(u.Hz)):
            f_val = mp.mpf(float(f_hz))
            d_val = mp.mpf(delta_s)

            def integrand(t: mp.mpf, *, f=f_val, d=d_val) -> mp.mpc:
                q = mp.mpf(sample_q(float(t)))
                return q * mp.e ** (-2j * mp.pi * f * (d + t))

            ref = complex(mp.quad(integrand, [mp.mpf(t0_s), mp.mpf(t1_s)]))
            denom = max(abs(ref), abs(got[k]), scale * width, 1e-30)
            assert abs(got[k] - ref) / denom <= _LIMIT


def test_i_epoch_zero_relative_time_alignment():
    """Changing epoch while shifting relative times leaves coefficients unchanged."""
    model = ContinuousFourierModel(_circular_source(), circular_pose(), frame=WORLD)
    period = 1.0 * u.s
    orders = np.array([1], dtype=int)
    scale = G_SI
    common = {
        "period": period,
        "orders": orders,
        "tau_min": period / 8.0,
        "atol": _integration_atol("acceleration", scale),
        "rtol": _P4["rtol"],
        "max_nodes": _P4["max_nodes"],
        "memory_budget_bytes": _MEMORY,
        "observables": ("acceleration",),
    }
    a = model.periodic_at(
        _hl_targets(np.zeros((1, 3))),
        epoch=LIGOTimeGPS(0, 0),
        tref=LIGOTimeGPS(0, 0),
        **common,
    ).run()
    b = model.periodic_at(
        _hl_targets(np.zeros((1, 3))),
        epoch=LIGOTimeGPS(100, 0),
        tref=LIGOTimeGPS(100, 0),
        **common,
    ).run()
    np.testing.assert_allclose(
        np.asarray(a.acceleration[0, 0]),
        np.asarray(b.acceleration[0, 0]),
        rtol=0,
        atol=_LIMIT * scale,
    )


def test_i_end_to_end_newton_response_fourier():
    """Adapter periodic path matches direct P4 on the same Newton response."""
    targets = _hl_targets(np.zeros((1, 3)))
    model = ContinuousFourierModel(_circular_source(), circular_pose(), frame=WORLD)
    response = make_rigid_pose_response(
        _circular_source(),
        circular_pose(),
        NumericalTargetPoints(np.zeros((1, 3))),
        frame=WORLD,
        memory_budget_bytes=_MEMORY,
    )
    period = 1.0 * u.s
    orders = np.array([0, 1], dtype=int)
    scale = G_SI
    coeff, _ = periodic_fourier(
        response,
        "acceleration",
        epoch=LIGOTimeGPS(0, 0),
        tref=LIGOTimeGPS(0, 0),
        period=period,
        orders=orders,
        tau_min=period / 16.0,
        atol=_integration_atol("acceleration", scale),
        rtol=_P4["rtol"],
        max_nodes=_P4["max_nodes"],
        memory_budget_bytes=_MEMORY,
    )
    adapted, _ = model.periodic_at(
        targets,
        epoch=LIGOTimeGPS(0, 0),
        tref=LIGOTimeGPS(0, 0),
        period=period,
        orders=orders,
        tau_min=period / 16.0,
        atol=_integration_atol("acceleration", scale),
        rtol=_P4["rtol"],
        max_nodes=_P4["max_nodes"],
        memory_budget_bytes=_MEMORY,
        observables=("acceleration",),
    ).run_with_diagnostics()
    direct = np.asarray(coeff.value[:, 0, 0])
    amp = np.asarray(adapted.acceleration[0, 0])
    # Matrix stores A_n = 2 C_n for n > 0.
    expected = np.array([direct[0], 2.0 * direct[1]])
    assert _normalized_error(amp, expected, scale) <= _LIMIT


def test_i_fft_auxiliary_comparison():
    """Dense FFT of a time series is auxiliary; it must not gate acceptance alone."""
    model = ContinuousFourierModel(_circular_source(), circular_pose(), frame=WORLD)
    period = 1.0 * u.s
    orders = np.array([0, 1, 2], dtype=int)
    scale = G_SI
    result, _ = model.periodic_at(
        _hl_targets(np.zeros((1, 3))),
        epoch=LIGOTimeGPS(0, 0),
        tref=LIGOTimeGPS(0, 0),
        period=period,
        orders=orders,
        tau_min=period / 32.0,
        atol=_integration_atol("acceleration", scale),
        rtol=_P4["rtol"],
        max_nodes=_P4["max_nodes"],
        memory_budget_bytes=_MEMORY,
        observables=("acceleration",),
    ).run_with_diagnostics()
    response = make_rigid_pose_response(
        _circular_source(),
        circular_pose(),
        _hl_targets(np.zeros((1, 3))),
        frame=WORLD,
        memory_budget_bytes=_MEMORY,
    )
    n = 2048
    times = np.linspace(0.0, period.to_value(u.s), n, endpoint=False)
    series = response(times * u.s, "acceleration").to_value(u.m / u.s**2)[:, 0, 0]
    spectrum = np.fft.rfft(series) / n
    freqs = np.fft.rfftfreq(n, d=period.to_value(u.s) / n)
    adapter = np.asarray(result.acceleration[0, 0])
    for order in orders:
        if order == 0:
            expected = spectrum[0]
            got = adapter[int(order)]
        else:
            idx = int(np.argmin(np.abs(freqs - float(order))))
            expected = 2.0 * spectrum[idx]
            match = np.isclose(result.context.frequencies_hz, float(order))
            got = adapter[int(np.where(match)[0][0])]
        # Loose auxiliary tolerance (independent discretization).
        assert _normalized_error(np.array([got]), np.array([expected]), scale) <= 1e-2


def test_i_descriptor_amplitude_periodic_roundtrip():
    result = (
        ContinuousFourierModel(_circular_source(), circular_pose(), frame=WORLD)
        .periodic_at(
            _hl_targets(np.zeros((1, 3))),
            epoch=LIGOTimeGPS(0, 0),
            tref=LIGOTimeGPS(0, 0),
            period=1.0 * u.s,
            orders=np.array([0, 1]),
            tau_min=0.1 * u.s,
            atol=_integration_atol("acceleration", G_SI),
            rtol=_P4["rtol"],
            max_nodes=512,
            memory_budget_bytes=_MEMORY,
        )
        .run()
    )
    rep = result.context.representation
    assert isinstance(rep, AmplitudeSpectrumRepresentation)
    assert rep.record_interpretation is FiniteRecordInterpretation.PERIODIC
    assert rep.record_interpretation is FiniteRecordInterpretation.PERIODIC
    assert "AmplitudeSpectrumRepresentation" in repr(rep)
    assert "PERIODIC" in repr(rep) or "periodic" in repr(rep)


def test_i_descriptor_fourier_one_sided_roundtrip():
    result = (
        ContinuousFourierModel(
            _circular_source(),
            constant_velocity_pose((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            frame=WORLD,
        )
        .window_at(
            _hl_targets(np.zeros((1, 3))),
            epoch=LIGOTimeGPS(0, 0),
            tref=LIGOTimeGPS(0, 0),
            t_start=-1.0 * u.s,
            t_stop=1.0 * u.s,
            frequencies=np.array([0.0, 1.0]) * u.Hz,
            tau_min=0.25 * u.s,
            atol=_window_atol("acceleration", G_SI, 2.0),
            rtol=_P5["rtol"],
            max_evaluations=DEFAULT_MAX_EVALUATIONS,
            memory_budget_bytes=_MEMORY,
        )
        .run()
    )
    rep = result.context.representation
    assert isinstance(rep, FourierTransformRepresentation)
    assert rep.sidedness is Sidedness.ONE_SIDED
    assert rep.sidedness is Sidedness.ONE_SIDED
    assert "FourierTransformRepresentation" in repr(rep)
    assert "ONE_SIDED" in repr(rep) or "one_sided" in repr(rep)


def test_i_gps_tref_offset_phase_identity():
    model = ContinuousFourierModel(_circular_source(), circular_pose(), frame=WORLD)
    period = 1.0 * u.s
    orders = np.array([1], dtype=int)
    scale = G_SI
    base, _ = model.periodic_at(
        _hl_targets(np.zeros((1, 3))),
        epoch=_EPOCH_GPS,
        tref=_EPOCH_GPS,
        period=period,
        orders=orders,
        tau_min=period / 8.0,
        atol=_integration_atol("acceleration", scale),
        rtol=_P4["rtol"],
        max_nodes=_P4["max_nodes"],
        memory_budget_bytes=_MEMORY,
    ).run_with_diagnostics()
    shifted, _ = model.periodic_at(
        _hl_targets(np.zeros((1, 3))),
        epoch=_EPOCH_GPS,
        tref=LIGOTimeGPS(999_999_990, 0),
        period=period,
        orders=orders,
        tau_min=period / 8.0,
        atol=_integration_atol("acceleration", scale),
        rtol=_P4["rtol"],
        max_nodes=_P4["max_nodes"],
        memory_budget_bytes=_MEMORY,
    ).run_with_diagnostics()
    tau = 10.0
    factor = np.exp(2j * np.pi * 1.0 * tau)
    np.testing.assert_allclose(
        np.asarray(shifted.acceleration[0, 0]),
        np.asarray(base.acceleration[0, 0]) * factor,
        rtol=0,
        atol=_LIMIT * scale,
    )


def test_i_memory_budget_propagation():
    """Managed-array budgets propagate; an impossibly small budget fails early."""
    model = ContinuousFourierModel(_circular_source(), circular_pose(), frame=WORLD)
    with pytest.raises(MemoryError):
        model.periodic_at(
            _hl_targets(np.zeros((1, 3))),
            epoch=LIGOTimeGPS(0, 0),
            tref=LIGOTimeGPS(0, 0),
            period=1.0 * u.s,
            orders=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
            tau_min=0.05 * u.s,
            atol=_integration_atol("acceleration", G_SI),
            rtol=_P4["rtol"],
            max_nodes=_P4["max_nodes"],
            memory_budget_bytes=1024,
        ).run()


def test_i_newton_framed_targets_match_manual_transform():
    """Framed targets are transformed into the computation frame (P3 fix)."""
    local = _local_frame()
    local_positions = np.array([[0.2, 0.0, -0.1]])
    framed = _hl_targets(local_positions, ids=("a",), frame=local)
    world_positions = local.transform_points_m(local_positions, WORLD)
    pose = circular_pose()
    model = ContinuousFourierModel(_circular_source(), pose, frame=WORLD)
    via_framed, _ = model.window_at(
        framed,
        epoch=LIGOTimeGPS(0, 0),
        tref=LIGOTimeGPS(0, 0),
        t_start=-1.0 * u.s,
        t_stop=1.0 * u.s,
        frequencies=np.array([0.0, 1.0]) * u.Hz,
        tau_min=0.25 * u.s,
        atol=_window_atol("acceleration", G_SI, 2.0),
        rtol=_P5["rtol"],
        max_evaluations=DEFAULT_MAX_EVALUATIONS,
        memory_budget_bytes=_MEMORY,
    ).run_with_diagnostics()
    via_manual, _ = model.window_at(
        _hl_targets(world_positions, ids=("a",)),
        epoch=LIGOTimeGPS(0, 0),
        tref=LIGOTimeGPS(0, 0),
        t_start=-1.0 * u.s,
        t_stop=1.0 * u.s,
        frequencies=np.array([0.0, 1.0]) * u.Hz,
        tau_min=0.25 * u.s,
        atol=_window_atol("acceleration", G_SI, 2.0),
        rtol=_P5["rtol"],
        max_evaluations=DEFAULT_MAX_EVALUATIONS,
        memory_budget_bytes=_MEMORY,
    ).run_with_diagnostics()
    np.testing.assert_allclose(
        np.asarray(via_framed.acceleration[0, 0]),
        np.asarray(via_manual.acceleration[0, 0]),
        rtol=0,
        atol=_LIMIT * G_SI,
    )


def test_i_p4_newton_memory_estimate_within_budget():
    """Newton periodic path succeeds under the 64 MiB managed budget (P4 counting)."""
    targets = _hl_targets(np.random.default_rng(0).normal(size=(8, 3)))
    model = ContinuousFourierModel(_circular_source(), circular_pose(), frame=WORLD)
    result, diag = model.periodic_at(
        targets,
        epoch=LIGOTimeGPS(0, 0),
        tref=LIGOTimeGPS(0, 0),
        period=1.0 * u.s,
        orders=np.arange(9),
        tau_min=0.1 * u.s,
        atol=_integration_atol("acceleration", G_SI),
        rtol=_P4["rtol"],
        max_nodes=2048,
        memory_budget_bytes=DEFAULT_MEMORY_BUDGET_BYTES,
        observables=("acceleration",),
    ).run_with_diagnostics()
    assert result.acceleration is not None
    assert diag["acceleration"]["evaluations"] > 0


def test_i_p5_newton_window_not_rejected_by_default_budget():
    """Default max_evaluations and budget allow the passage-style Newton window."""
    pose = constant_velocity_pose((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    model = ContinuousFourierModel(_circular_source(), pose, frame=WORLD)
    result, diag = model.window_at(
        _hl_targets(np.zeros((1, 3))),
        epoch=LIGOTimeGPS(0, 0),
        tref=LIGOTimeGPS(0, 0),
        t_start=-2.0 * u.s,
        t_stop=2.0 * u.s,
        frequencies=np.array([0.0, 0.125, 1.0, 4.0]) * u.Hz,
        tau_min=0.5 * u.s,
        atol=_window_atol("acceleration", G_SI, 4.0),
        rtol=_P5["rtol"],
        max_evaluations=DEFAULT_MAX_EVALUATIONS,
        memory_budget_bytes=DEFAULT_MEMORY_BUDGET_BYTES,
    ).run_with_diagnostics()
    assert result.acceleration is not None
    assert diag["acceleration"]["evaluations"] <= DEFAULT_MAX_EVALUATIONS
