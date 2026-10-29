"""Tests for the witness plane-wave displacement forward model
(Phase 15-a-2, spec: docs/design/phase15a_cross_statistics.md).
"""

import inspect

import numpy as np
import pytest

from gragra.waves.displacement import plane_wave_displacement
from gragra.waves.phase import plane_wave_complex_factor
from gragra.waves.polarization import displacement_direction


def test_w1_matches_component_product():
    """W1: output must equal the product of the two existing building blocks."""
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    k_hat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    wavenumber = np.array([1.0, 2.0])
    amplitude = 2.0

    result = plane_wave_displacement(
        positions, k_hat, wavenumber, "P", amplitude_m=amplitude
    )

    direction = displacement_direction("P", k_hat)  # (Ndir, 3)
    factor = plane_wave_complex_factor(positions, k_hat, wavenumber)  # (N, Ndir, Nf)
    expected = amplitude * direction[None, :, None, :] * factor[..., None]

    np.testing.assert_array_equal(result, expected)


def test_w2_literal_golden():
    """W2: P-wave, k_hat=x, k=pi, position (1,0,0), amplitude 2.0 -> factor=-1."""
    positions = np.array([[1.0, 0.0, 0.0]])
    k_hat = np.array([[1.0, 0.0, 0.0]])
    wavenumber = np.array([np.pi])
    result = plane_wave_displacement(positions, k_hat, wavenumber, "P", amplitude_m=2.0)
    np.testing.assert_allclose(result[0, 0, 0], [-2.0 + 0j, 0.0, 0.0], atol=1e-12)


def test_w3_shape_dtype_readonly():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    k_hat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    wavenumber = np.array([1.0, 2.0, 3.0])
    result = plane_wave_displacement(positions, k_hat, wavenumber, "P")
    assert result.shape == (2, 3, 3, 3)
    assert result.dtype == np.complex128
    assert result.flags.writeable is False


def test_w4_sh_transversality():
    positions = np.array([[0.3, 0.4, 0.0], [1.0, -0.5, 0.2]])
    k_hat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    wavenumber = np.array([1.5])
    result = plane_wave_displacement(positions, k_hat, wavenumber, "SH")
    # dot(u, k_hat) == 0 for every point/direction/frequency
    dots = np.einsum("nkfc,kc->nkf", result, k_hat)
    np.testing.assert_allclose(dots, 0.0, atol=1e-12)


def test_w5_valueerror_delegation():
    """W5 (Codex review #39): non-unit-but-nonzero k_hat is NOT rejected --
    normalize_directions silently normalizes it. Only zero/non-finite/
    wrong-shape directions and invalid wave_type must raise."""
    positions = np.array([[1.0, 0.0, 0.0]])
    wavenumber = np.array([1.0])

    with pytest.raises(ValueError):
        plane_wave_displacement(positions, [[0.0, 0.0, 0.0]], wavenumber, "P")

    with pytest.raises(ValueError):
        plane_wave_displacement(positions, [[np.inf, 0.0, 0.0]], wavenumber, "P")

    with pytest.raises(ValueError):
        plane_wave_displacement(positions, [[1.0, 0.0]], wavenumber, "P")

    with pytest.raises(ValueError):
        plane_wave_displacement(positions, [[1.0, 0.0, 0.0]], wavenumber, "not_a_wave")

    # Non-unit but nonzero k_hat must NOT raise (normalize_directions accepts it).
    plane_wave_displacement(positions, [[2.0, 0.0, 0.0]], wavenumber, "P")


def test_w6_no_sign_parameter():
    """W6: sign argument must not exist (locked to +1 -- conventions §7)."""
    sig = inspect.signature(plane_wave_displacement)
    assert "sign" not in sig.parameters
