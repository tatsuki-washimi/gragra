"""P2 regression/independence tests for M3-B H2/H3 (gradient units, direct DFT, Frame x DFT).

Thresholds are the scale-normalised budgets proposed in 06-tolerance-budget.md
(``atol = c * eps * max|reference|``); they are NOT the M3-A 3e-14 gravity values.
"""

from __future__ import annotations

import math

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
from gragra.results import TimeSampling
from gragra.results._time_grid import build_uniform_time_s
from gragra.targets import TargetPoints

from test_m3b_rot_dft import assert_l1b_exchange

WORLD = Frame("world")
EPS = float(np.finfo(float).eps)
OBSERVABLES = ("potential", "acceleration", "gradient")
FOURIER_UNITS = {
    "potential": u.m**2 / u.s,
    "acceleration": u.m / u.s,
    "gradient": 1 / u.s,
}
AMPLITUDE_UNITS = {
    "potential": u.m**2 / u.s**2,
    "acceleration": u.m / u.s**2,
    "gradient": 1 / u.s**2,
}


def _moving_result(m: int, dt: float, epoch=(1_000_000_000, 250_000_000), out=None):
    ang = 2 * np.pi * 0.37 * np.arange(m) * dt
    rot = np.zeros((m, 3, 3))
    rot[:, 0, 0] = np.cos(ang)
    rot[:, 0, 1] = -np.sin(ang)
    rot[:, 1, 0] = np.sin(ang)
    rot[:, 1, 1] = np.cos(ang)
    rot[:, 2, 2] = 1.0
    trans = np.stack(
        [
            0.5 * np.sin(2 * np.pi * 1.3 * np.arange(m) * dt),
            0.2 * np.arange(m) * dt,
            np.zeros(m),
        ],
        axis=1,
    )
    sampling = TimeSampling(epoch=LIGOTimeGPS(*epoch), dt=dt * u.s, count=m)
    transform = PrescribedRigidTransform(build_uniform_time_s(epoch, dt, m), trans, rot)
    source = PointMassCloud([1.0, 2.0], [[0.0, 0.0, 0.0], [1.0, -1.0, 0.5]])
    targets = TargetPoints(
        np.array([[5.0, 3.0, -4.0], [-6.0, 2.0, 4.0]]) * u.m, ("a", "b"), frame=WORLD
    )
    result = (
        rigid_motion(source, transform, frame=WORLD, sampling=sampling)
        .at(targets, observables=OBSERVABLES, output_frame=out)
        .run()
    )
    return result, epoch


def _direct_dft_reference(y, dt, tau, mode):
    """Explicit-matrix DFT (no FFT call); float64, phase built from exact tau."""
    m = len(y)
    k = np.arange(m // 2 + 1)
    j = np.arange(m)
    x = np.exp(-2j * np.pi * (np.outer(k, j) % m) / m) @ y
    phase = np.exp(-2j * np.pi * (k / (m * dt)) * tau)
    if mode == "fourier":
        return dt * phase * x
    ck = np.full(len(k), 2.0)
    ck[0] = 1.0
    if m % 2 == 0:
        ck[-1] = 1.0
    return ck / m * phase * x


@pytest.mark.parametrize("m", [7, 8, 33])
@pytest.mark.parametrize("tau", [0.0, 0.125, -0.3])
@pytest.mark.parametrize("mode", ["fourier", "amplitude"])
def test_all_observables_match_direct_dft_with_units(m, tau, mode):
    dt = 1.0 / 32.0
    result, epoch = _moving_result(m, dt)
    # tref = epoch - tau  =>  t0 - tref = +tau
    total_ns = epoch[0] * 10**9 + epoch[1] - int(round(tau * 1e9))
    tref = divmod(total_ns, 10**9)
    spec = result.fft(tref=tref) if mode == "fourier" else result.spectrum("periodic", tref=tref)
    unit_map = FOURIER_UNITS if mode == "fourier" else AMPLITUDE_UNITS
    f_nyq = (m // 2) / (m * dt)
    phase_arg = 2 * math.pi * f_nyq * abs(tau)
    for obs in OBSERVABLES:
        series = np.asarray(getattr(result, obs))
        out = getattr(spec, obs)
        assert out.meta[0, 0].unit == unit_map[obs]
        values = np.asarray(out.value)
        for t_idx in range(series.shape[0]):
            for c in range(series.shape[1]):
                ref = _direct_dft_reference(series[t_idx, c], dt, tau, mode)
                scale = np.abs(ref).max()
                err = np.abs(values[t_idx, c] - ref).max()
                assert err <= 4 * EPS * (4 + phase_arg) * max(scale, 1e-300)


def test_gradient_fft_and_spectrum_accept_one_over_second_units():
    """Regression: np.full(..., Unit) used to raise UnitConversionError for 1/s, 1/s^2."""
    result, epoch = _moving_result(8, 0.125)
    ft = result.fft(tref=epoch)
    amp = result.spectrum("transient", tref=epoch)
    assert ft.gradient.meta[0, 0].unit == 1 / u.s
    assert amp.gradient.meta[0, 0].unit == 1 / u.s**2
    assert all(c.unit == 1 / u.s for c in ft.gradient.meta.flat)
    assert all(c.unit == 1 / u.s**2 for c in amp.gradient.meta.flat)
    y = np.asarray(result.gradient)[0, 0]
    assert abs(amp.gradient.value[0, 0, 0] - y.mean()) <= 4 * EPS * abs(y).max()
    assert abs(ft.gradient.value[0, 0, 0] - y.sum() * 0.125) <= 4 * EPS * abs(y).sum()


def test_fixed_frame_rotation_commutes_with_fft_through_pipeline():
    """F[R a] = R F[a] and F[R G R^T] = R F[G] R^T via real H1 + .fft()."""
    basis = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )  # exact 90 deg
    generic = np.linalg.qr(np.random.default_rng(11).normal(size=(3, 3)))[0]
    if np.linalg.det(generic) < 0:
        generic[:, 0] *= -1
    for q in (basis, generic):
        local = Frame("loc", basis=q, parent=WORLD)
        world, epoch = _moving_result(64, 1.0 / 64.0)
        moved, _ = _moving_result(64, 1.0 / 64.0, out=local)
        for obs, shape in (("acceleration", (3,)), ("gradient", (3, 3))):
            fw = np.asarray(getattr(world.fft(tref=epoch), obs).value)
            fl = np.asarray(getattr(moved.fft(tref=epoch), obs).value)
            n = fw.shape[0]
            if obs == "acceleration":
                ref = np.einsum("ij,njk->nik", q.T, fw)
            else:
                ref = np.einsum(
                    "ia,nabk,jb->nijk", q.T, fw.reshape(n, 3, 3, -1), q.T
                ).reshape(n, 9, -1)
            assert_l1b_exchange(fl, ref)


def test_periodic_and_transient_are_numerically_identical_interpretation_labels():
    """C3-v1: PERIODIC/TRANSIENT differ in interpretation, not in A_k values."""
    result, epoch = _moving_result(33, 1.0 / 32.0)
    periodic = result.spectrum("periodic", tref=epoch)
    transient = result.spectrum("transient", tref=epoch)
    assert periodic.context.representation != transient.context.representation
    for obs in OBSERVABLES:
        np.testing.assert_array_equal(
            np.asarray(getattr(periodic, obs).value),
            np.asarray(getattr(transient, obs).value),
        )
