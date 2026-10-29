"""M3-B H2/H3: .fft() and .spectrum() on rigid_motion time-series results."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from gwpy.time import LIGOTimeGPS

from gragra.geometry import Frame
from gragra.gravity_sources import (
    PointMassCloud,
    PrescribedRigidTransform,
    rigid_motion,
)
from gragra.results import TimeSampling, compare
from gragra.results._spectral import (
    AmplitudeSpectrumRepresentation,
    FourierTransformRepresentation,
    Sidedness,
)
from gragra.results._time_grid import build_uniform_time_s
from gragra.targets import TargetPoints

WORLD = Frame("world")


def _sin_series_result():
    source = PointMassCloud([1.0], [[0.0, 0.0, 0.0]])
    m = 256
    dt = 1.0 / 256.0
    epoch = (0, 0)
    sampling = TimeSampling(epoch=LIGOTimeGPS(*epoch), dt=dt * u.s, count=m)
    t = build_uniform_time_s(epoch, dt, m)
    translations = np.zeros((m, 3))
    translations[:, 0] = 0.5 * np.sin(2 * np.pi * 10.0 * t)
    transform = PrescribedRigidTransform(
        t,
        translations,
        np.repeat(np.eye(3)[None], m, axis=0),
    )
    targets = TargetPoints(np.array([[5.0, 0.0, 0.0]]) * u.m, ("t0",), frame=WORLD)
    ts = (
        rigid_motion(source, transform, frame=WORLD, sampling=sampling)
        .at(targets, observables=("acceleration",))
        .run()
    )
    return ts, dt, m


def test_fft_one_sided_and_dc():
    ts, dt, m = _sin_series_result()
    tref = (0, 0)
    ft = ts.fft(tref=tref)
    assert isinstance(ft.context.representation, FourierTransformRepresentation)
    assert ft.context.representation.sidedness is Sidedness.ONE_SIDED
    freqs = np.asarray(ft.acceleration.frequencies.to_value(u.Hz))
    assert freqs[0] == 0.0
    assert len(freqs) == m // 2 + 1


def test_spectrum_phase_ratio():
    ts, dt, m = _sin_series_result()
    tau = 0.125
    tref_shift = (0, int(tau * 1e9))
    amp0 = ts.spectrum("transient", tref=(0, 0))
    amp1 = ts.spectrum("transient", tref=tref_shift)
    k = 10
    f0 = 10.0
    ratio = amp1.acceleration.value[0, 0, k] / amp0.acceleration.value[0, 0, k]
    expected = np.exp(2j * np.pi * f0 * tau)
    assert abs(ratio - expected) < 1e-10


def test_compare_rejects_spectrum():
    ts, _, _ = _sin_series_result()
    ref = ts.spectrum("periodic", tref=(0, 0))
    cand = ts.spectrum("periodic", tref=(0, 0))
    with pytest.raises(ValueError, match="does not support Fourier"):
        compare(reference=ref, candidate=cand)


def test_static_result_rejects_fft():
    from gragra.gravity_sources import point_sum
    from gragra.gravity_sources.weighted_points import PointMass

    pm = PointMass(1.0, [0.0, 0.0, 0.0])
    targets = TargetPoints(np.array([[1.0, 0.0, 0.0]]) * u.m, ("t0",), frame=WORLD)
    static = point_sum(pm, frame=WORLD).at(targets, observables=("potential",)).run()
    with pytest.raises(ValueError, match="time-series"):
        static.fft(tref=(0, 0))


def test_amplitude_representation_kind():
    ts, _, _ = _sin_series_result()
    amp = ts.spectrum("periodic", tref=(0, 0))
    assert isinstance(amp.context.representation, AmplitudeSpectrumRepresentation)


def test_fft_one_frame_dc_only():
    source = PointMassCloud([1.0], [[0.0, 0.0, 0.0]])
    epoch = (1_000_000_000, 0)
    dt = 1e-7
    sampling = TimeSampling(epoch=LIGOTimeGPS(*epoch), dt=dt * u.s, count=1)
    time_s = build_uniform_time_s(epoch, dt, 1)
    transform = PrescribedRigidTransform(
        time_s,
        np.zeros((1, 3)),
        np.eye(3)[None],
    )
    targets = TargetPoints(np.array([[3.0, 0.0, 0.0]]) * u.m, ("t0",), frame=WORLD)
    ts = (
        rigid_motion(source, transform, frame=WORLD, sampling=sampling)
        .at(targets, observables=("potential",))
        .run()
    )
    ft = ts.fft(tref=epoch)
    vals = ft.potential.value[0, 0, :]
    assert len(vals) == 1
    assert np.isfinite(vals[0])


def test_amplitude_even_nyquist_known_alternating_series():
    """M=8 alternating ±1 → Nyquist amplitude 1 (step3-num L2 reference)."""
    from gragra._spectral_from_timeseries import transform_time_payloads
    from gragra._time_matrix import make_time_matrix

    m = 8
    dt = 0.125
    epoch = (0, 0)
    sampling = TimeSampling(epoch=LIGOTimeGPS(*epoch), dt=dt * u.s, count=m)
    y = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    values = np.zeros((1, 3, m), dtype=np.float64)
    values[0, 0, :] = y
    matrix = make_time_matrix(
        values,
        sampling=sampling,
        unit=u.m / u.s**2,
        rows=("t0",),
        cols=("x", "y", "z"),
        name="nyquist_fixture",
    )
    amp = transform_time_payloads(
        {"acceleration": matrix},
        sampling=sampling,
        observables=("acceleration",),
        tref=(0, 0),
        mode="amplitude",
    )["acceleration"]
    assert abs(amp.value[0, 0, 0]) < 1e-12
    assert abs(abs(amp.value[0, 0, -1]) - 1.0) < 1e-12


def test_spectrum_mixed_cell_units_converts_to_si_before_dft():
    from gragra._spectral_from_timeseries import transform_time_payloads
    from gragra._time_matrix import make_time_matrix

    m = 32
    dt = 1.0 / 64.0
    epoch = (0, 0)
    sampling = TimeSampling(epoch=LIGOTimeGPS(*epoch), dt=dt * u.s, count=m)
    values = np.ones((1, 3, m), dtype=np.float64)
    matrix = make_time_matrix(
        values,
        sampling=sampling,
        unit=u.m / u.s**2,
        rows=("t0",),
        cols=("x", "y", "z"),
        name="mixed_units",
    )
    matrix.meta[0, 1].unit = u.km / u.s**2
    amp = transform_time_payloads(
        {"acceleration": matrix},
        sampling=sampling,
        observables=("acceleration",),
        tref=(0, 0),
        mode="amplitude",
    )["acceleration"]
    dc_x = amp.value[0, 0, 0]
    dc_y = amp.value[0, 1, 0]
    assert abs(dc_x - 1.0) < 1e-12
    assert abs(dc_y - 1000.0) < 1e-9
    assert amp.meta[0, 0].unit == u.m / u.s**2
    assert amp.meta[0, 1].unit == u.m / u.s**2


def test_select_then_fft():
    ts, _, _ = _sin_series_result()
    subset = ts.select("acceleration")
    ft = subset.fft(tref=(0, 0))
    assert ft.potential is None
    assert ft.acceleration is not None


def test_spectrum_rejects_unknown_kind():
    ts, _, _ = _sin_series_result()
    with pytest.raises(ValueError, match="spectrum kind"):
        ts.spectrum("psd", tref=(0, 0))
