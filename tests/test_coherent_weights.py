"""Tests for plane-wave coherent mass weights."""

import numpy as np
import pytest

from gragra.coherent.weights import plane_wave_mass_weights


def test_plane_wave_mass_weights_basics():
    # N=3, Ndir=2, Nf=3
    base_weights = np.array([1.5, 2.5, 3.5])
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    wavenumbers = np.array([0.1, 0.2, 0.5])
    phase0 = 0.1

    weights = plane_wave_mass_weights(
        base_weights,
        positions,
        directions,
        wavenumbers,
        phase0=phase0,
        sign=1,
    )

    # Shape: (N, Ndir, Nf) = (3, 2, 3)
    assert weights.shape == (3, 2, 3)
    assert weights.dtype == np.complex128
    assert not weights.flags.writeable

    # Verify a few manual calculations
    # pos[1] = [1, 0, 0], dir[0] = [1, 0, 0] -> dot = 1.0
    # wave[2] = 0.5 -> phase = 0.5 * 1.0 + 0.1 = 0.6
    # factor = exp(1j * 0.6)
    # expected = 2.5 * factor
    expected = 2.5 * np.exp(1j * 0.6)
    np.testing.assert_allclose(weights[1, 0, 2], expected, rtol=1e-12)


def test_plane_wave_mass_weights_sign():
    base_weights = np.array([1.5])
    positions = np.array([[1.0, 0.0, 0.0]])
    directions = np.array([1.0, 0.0, 0.0])
    wavenumber = np.pi

    w_pos = plane_wave_mass_weights(
        base_weights, positions, directions, wavenumber, sign=1
    )
    np.testing.assert_allclose(w_pos, [[[-1.5 + 0j]]], atol=1e-15)

    w_neg = plane_wave_mass_weights(
        base_weights, positions, directions, wavenumber, sign=-1
    )
    np.testing.assert_allclose(w_neg, [[[-1.5 + 0j]]], atol=1e-15)


def test_plane_wave_mass_weights_invalid():
    base_weights = np.array([1.5, 2.5])
    positions = np.array([[1.0, 0.0, 0.0]])  # N mismatch (2 vs 1)
    directions = np.array([1.0, 0.0, 0.0])
    wavenumber = 0.1

    with pytest.raises(
        ValueError, match="Length of base_weights_kg and positions_m must match"
    ):
        plane_wave_mass_weights(base_weights, positions, directions, wavenumber)


def test_plane_wave_mass_weights_positional_compatibility():
    base_weights = np.array([1.5, 2.5])
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    wavenumber = 0.5
    phase0 = 0.1
    sign = -1

    w_pos = plane_wave_mass_weights(
        base_weights, positions, directions, wavenumber, phase0, sign, backend="numpy"
    )
    w_default = plane_wave_mass_weights(
        base_weights, positions, directions, wavenumber, phase0, sign
    )

    np.testing.assert_allclose(w_pos, w_default, rtol=1e-12)
    assert w_pos.shape == (2, 2, 1)
