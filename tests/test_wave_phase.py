"""Tests for wave phase and complex factor utilities."""

import numpy as np
import pytest

from gragra.waves.phase import (
    evanescent_complex_factor,
    plane_wave_complex_factor,
    plane_wave_phase,
)


def test_plane_wave_phase_basics():
    # 2 positions, 2 directions, 3 frequencies
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    k_hat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    wavenumbers = np.array([0.1, 0.2, 0.5])
    phase0 = 0.5

    phase = plane_wave_phase(positions, k_hat, wavenumbers, phase0=phase0)

    # Output shape is strictly (N, Ndir, Nf) = (2, 2, 3)
    assert phase.shape == (2, 2, 3)
    assert phase.dtype == np.float64
    assert not phase.flags.writeable

    # Verify manual calculations
    # pos[1] = [1, 2, 3], k_hat[0] = [1, 0, 0] -> dot = 1.0
    # wave[2] = 0.5 -> phase = 0.5 * 1.0 + 0.5 = 1.0
    assert np.abs(phase[1, 0, 2] - 1.0) < 1e-15

    # pos[1] = [1, 2, 3], k_hat[1] = [0, 1, 0] -> dot = 2.0
    # wave[1] = 0.2 -> phase = 0.2 * 2.0 + 0.5 = 0.9
    assert np.abs(phase[1, 1, 1] - 0.9) < 1e-15


def test_plane_wave_phase_scalar_promotion():
    positions = np.array([[1.0, 0.0, 0.0]])
    k_hat = np.array([1.0, 0.0, 0.0])  # promoted to (1, 3) -> Ndir=1
    wavenumber = 0.1  # promoted to (1,) -> Nf=1

    phase = plane_wave_phase(positions, k_hat, wavenumber)
    # Output must be (1, 1, 1) and not squeezed
    assert phase.shape == (1, 1, 1)
    np.testing.assert_allclose(phase, [[[0.1]]])


def test_plane_wave_phase_invalid():
    positions = np.array([[1.0, 0.0, 0.0]])
    k_hat = np.array([1.0, 0.0, 0.0])

    # Negative wavenumber
    with pytest.raises(ValueError, match="non-negative"):
        plane_wave_phase(positions, k_hat, -0.1)

    # Non-finite wavenumber
    with pytest.raises(ValueError, match="finite numbers"):
        plane_wave_phase(positions, k_hat, np.nan)

    # Boolean wavenumber
    with pytest.raises(TypeError, match="numeric"):
        plane_wave_phase(positions, k_hat, True)

    # Invalid positions shape
    with pytest.raises(ValueError, match="shape"):
        plane_wave_phase([1.0, 2.0], k_hat, 0.1)


def test_plane_wave_complex_factor():
    positions = np.array([[1.0, 0.0, 0.0]])
    k_hat = np.array([1.0, 0.0, 0.0])
    wavenumber = np.pi  # Phase = pi * 1.0 = pi (if phase0=0)

    # sign = +1: exp(1j * pi) = -1
    factor_pos = plane_wave_complex_factor(positions, k_hat, wavenumber, sign=1)
    assert factor_pos.shape == (1, 1, 1)
    assert factor_pos.dtype == np.complex128
    assert not factor_pos.flags.writeable
    np.testing.assert_allclose(factor_pos, [[[-1.0 + 0j]]], atol=1e-15)

    # sign = -1
    factor_neg = plane_wave_complex_factor(positions, k_hat, wavenumber, sign=-1)
    np.testing.assert_allclose(factor_neg, [[[-1.0 + 0j]]], atol=1e-15)

    # Test oblique phase with sign +/-
    # Phase = pi/2
    factor_pos_half = plane_wave_complex_factor(positions, k_hat, np.pi / 2, sign=1)
    np.testing.assert_allclose(factor_pos_half, [[[0.0 + 1j]]], atol=1e-15)

    factor_neg_half = plane_wave_complex_factor(positions, k_hat, np.pi / 2, sign=-1)
    np.testing.assert_allclose(factor_neg_half, [[[0.0 - 1j]]], atol=1e-15)

    # Invalid sign
    with pytest.raises(ValueError, match="sign must be"):
        plane_wave_complex_factor(positions, k_hat, wavenumber, sign=2)


def test_evanescent_complex_factor_sign():
    positions = np.array([[1.0, 0.0, -1.0]])
    k_hat = np.array([[1.0, 0.0, 0.0]])
    wavenumber = np.pi / 2
    decay = 0.5
    # sign = +1: exp(1j * pi/2) * exp(-0.5) = 1j * exp(-0.5)
    out_pos = evanescent_complex_factor(positions, k_hat, wavenumber, decay, sign=1)
    np.testing.assert_allclose(out_pos[0, 0, 0], 1j * np.exp(-0.5), rtol=1e-12)

    # sign = -1: exp(-1j * pi/2) * exp(-0.5) = -1j * exp(-0.5)
    out_neg = evanescent_complex_factor(positions, k_hat, wavenumber, decay, sign=-1)
    np.testing.assert_allclose(out_neg[0, 0, 0], -1j * np.exp(-0.5), rtol=1e-12)


def test_evanescent_complex_factor_invalid_bool():
    positions = np.array([[1.0, 0.0, -1.0]])
    k_hat = np.array([[1.0, 0.0, 0.0]])
    wavenumber = 1.0

    # Test that Python bool raises TypeError
    with pytest.raises(TypeError, match="numeric, not bool"):
        evanescent_complex_factor(positions, k_hat, wavenumber, True)

    # Test that numpy bool_ raises TypeError
    with pytest.raises(TypeError, match="numeric, not bool"):
        evanescent_complex_factor(positions, k_hat, wavenumber, np.bool_(True))
