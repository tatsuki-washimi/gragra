"""Tests for PlaneWave, PlanePWave, and PlaneWaveSources evaluation API."""

import numpy as np

from gragra.gravity_sources import PointMassCloud
from gragra.waves import (
    PlanePWave,
    PlaneWave,
    PlaneWaveSources,
    directional_power_average,
)


def _make_sample_cloud():
    """Create a sample 2-point cloud."""
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]], dtype=np.float64)
    mass = np.array([1000.0, 2000.0], dtype=np.float64)
    return PointMassCloud(mass, pos)


def test_plane_p_wave_evaluation_three_observables():
    """Verify PlanePWave evaluates three observables with correct shapes."""
    cloud = _make_sample_cloud()

    dirs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)  # D = 2
    freqs = np.array([2.0, 5.0, 10.0], dtype=np.float64)  # F = 3
    k_vals = freqs / 3000.0  # c = 3000 m/s

    wave = PlanePWave(
        directions=dirs,
        frequencies_hz=freqs,
        wavenumbers_rad_m=k_vals,
        amplitudes_m=1.0e-8,
        phase_origin_m=[0.0, 0.0, 0.0],
    )

    bundle = wave.evaluate_on(cloud)
    assert isinstance(bundle, PlaneWaveSources)
    assert bundle.displacement_m.shape == (2, 2, 3, 3)  # (N, D, F, 3)

    targets = np.array([[0.0, 0.0, 10.0], [10.0, 0.0, 0.0]], dtype=np.float64)  # M = 2

    # 1. Potential: (M, D, F)
    pot = bundle.evaluate(targets, observable="potential")
    assert pot.shape == (2, 2, 3)
    assert pot.dtype == np.complex128
    assert np.all(np.isfinite(pot))

    # 2. Acceleration: (M, D, F, 3)
    acc = bundle.evaluate(targets, observable="acceleration")
    assert acc.shape == (2, 2, 3, 3)
    assert acc.dtype == np.complex128
    assert np.all(np.isfinite(acc))

    # 3. Gradient: (M, D, F, 3, 3)
    grad = bundle.evaluate(targets, observable="gradient")
    assert grad.shape == (2, 2, 3, 3, 3)
    assert grad.dtype == np.complex128
    assert np.all(np.isfinite(grad))


def test_plane_wave_singular_direction_sh_sv():
    """Verify Minor 2: SH/SV with k parallel to z-axis evaluate deterministically."""
    cloud = _make_sample_cloud()

    # Singular propagation direction k = [0, 0, 1] parallel to z-axis
    dirs = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    freqs = np.array([5.0], dtype=np.float64)
    k_vals = np.array([0.01], dtype=np.float64)

    # SH wave along z-axis
    wave_sh = PlaneWave(
        wave_type="SH",
        directions=dirs,
        frequencies_hz=freqs,
        wavenumbers_rad_m=k_vals,
        amplitudes_m=1.0e-9,
        phase_origin_m=[0.0, 0.0, 0.0],
    )
    bundle_sh = wave_sh.evaluate_on(cloud)
    assert bundle_sh.displacement_m.shape == (2, 1, 1, 3)
    # SH displacement is along y [0, 1, 0] by the deterministic x fallback
    assert np.allclose(np.abs(bundle_sh.displacement_m[:, 0, 0, 0]), 0.0)  # x=0
    assert np.all(np.abs(bundle_sh.displacement_m[:, 0, 0, 1]) > 0.0)  # y!=0
    assert np.allclose(np.abs(bundle_sh.displacement_m[:, 0, 0, 2]), 0.0)  # z=0

    # SV wave along z-axis
    wave_sv = PlaneWave(
        wave_type="SV",
        directions=dirs,
        frequencies_hz=freqs,
        wavenumbers_rad_m=k_vals,
        amplitudes_m=1.0e-9,
        phase_origin_m=[0.0, 0.0, 0.0],
    )
    bundle_sv = wave_sv.evaluate_on(cloud)
    # SV displacement should be orthogonal to k and SH (along x-axis)
    assert np.all(np.abs(bundle_sv.displacement_m[:, 0, 0, 0]) > 0.0)  # x!=0
    assert np.allclose(np.abs(bundle_sv.displacement_m[:, 0, 0, 1]), 0.0)  # y=0
    assert np.allclose(np.abs(bundle_sv.displacement_m[:, 0, 0, 2]), 0.0)  # z=0


def test_directional_power_average():
    """Verify directional power average computes non-negative weighted power."""
    # Dummy field of shape (M=3, D=4, F=2)
    rng = np.random.default_rng(42)
    field = rng.standard_normal((3, 4, 2)) + 1j * rng.standard_normal((3, 4, 2))

    # Uniform weights
    power_uniform = directional_power_average(field, axis=1)
    assert power_uniform.shape == (3, 2)
    assert np.all(power_uniform >= 0.0)
    expected_uniform = np.mean(np.abs(field) ** 2, axis=1)
    assert np.allclose(power_uniform, expected_uniform)

    # Custom weights
    w = np.array([1.0, 2.0, 3.0, 4.0])
    power_weighted = directional_power_average(field, weights_dir=w, axis=1)
    assert power_weighted.shape == (3, 2)
    assert np.all(power_weighted >= 0.0)
    w_norm = w / 10.0
    expected_weighted = np.sum((np.abs(field) ** 2) * w_norm[None, :, None], axis=1)
    assert np.allclose(power_weighted, expected_weighted)
