"""Tests for Phase 15-a-3 gravitoelastic cross statistics (c_ij / CSD).

Golden values are independently hand-derived in
docs/design/phase15a_cross_statistics.md ("15-a-3 golden") and reproduced
verbatim here; see that document for the full derivation (setups G-c1/G-c2).
"""

import numpy as np

from gragra.coherent import (
    directional_outer_cross_average,
    plane_wave_bulk_mass_weights,
)
from gragra.constants import G_SI
from gragra.observables import coherent_acceleration_csd, coherent_gravitoelastic_tensor
from gragra.observables.coherent import coherent_point_acceleration
from gragra.waves import plane_wave_displacement

# Shared G-c1/G-c2 setup: single P-wave bulk mass element at the origin.
_POSITIONS = np.array([[0.0, 0.0, 0.0]])
_VOLUMES = np.array([1.0])
_RHO0 = 1000.0
_K_HAT = np.array([[1.0, 0.0, 0.0]])
_K = 2.0
_XI_AMP = 1e-6
_TARGET = np.array([[1.0, 0.0, 0.0]])
_WITNESS = np.array([[0.5, 0.0, 0.0]])
_W = np.array([1.0])


def _p_wave_weights(wave_type="P"):
    return plane_wave_bulk_mass_weights(
        _VOLUMES,
        _POSITIONS,
        _RHO0,
        _K_HAT,
        _K,
        wave_type=wave_type,
        displacement_m=_XI_AMP,
        phase0=0.0,
    )


def _witness(wave_type="P"):
    return plane_wave_displacement(
        _WITNESS, _K_HAT, _K, wave_type, amplitude_m=_XI_AMP, phase0=0.0
    )


def test_gc1_c_ij_absolute_golden():
    """G-c1: c_xx = 2e-9 * G_SI * (sin(1) + i*cos(1)); other components zero."""
    weights = _p_wave_weights()
    witness = _witness()
    c_ij = coherent_gravitoelastic_tensor(_POSITIONS, weights, _TARGET, witness, _W)

    assert c_ij.shape == (1, 1, 1, 3, 3)
    expected_xx = 2e-9 * G_SI * (np.sin(1.0) + 1j * np.cos(1.0))
    np.testing.assert_allclose(c_ij[0, 0, 0, 0, 0], expected_xx, rtol=1e-12)

    other = c_ij.copy()
    other[0, 0, 0, 0, 0] = 0.0
    np.testing.assert_allclose(other, 0.0, atol=1e-30)


def test_gc2_csd_absolute_golden():
    """G-c2: CSD_xx = 1e-6 * G_SI**2 (real, positive; G**2 detects mis-ordering)."""
    weights = _p_wave_weights()
    targets_a = np.array([[1.0, 0.0, 0.0]])
    targets_b = np.array([[2.0, 0.0, 0.0]])
    csd = coherent_acceleration_csd(_POSITIONS, weights, targets_a, targets_b, _W)

    assert csd.shape == (1, 1, 1, 3, 3)
    expected_xx = 1e-6 * G_SI**2
    np.testing.assert_allclose(csd[0, 0, 0, 0, 0], expected_xx, rtol=1e-12)


def test_s_wave_zero_fixed_point():
    """SH wave: k_hat . e_pol == 0 structurally, so c_ij and CSD are exact zero."""
    weights = _p_wave_weights(wave_type="SH")
    witness = _witness(wave_type="SH")

    c_ij = coherent_gravitoelastic_tensor(_POSITIONS, weights, _TARGET, witness, _W)
    assert np.all(c_ij == 0)

    targets_b = np.array([[2.0, 0.0, 0.0]])
    csd = coherent_acceleration_csd(_POSITIONS, weights, _TARGET, targets_b, _W)
    assert np.all(csd == 0)


def test_conjugate_symmetry():
    """c_ij(a, xi) == conj(c_ji(xi, a)) transposed (conventions §3), Ndir=2."""
    k_hat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    weights = plane_wave_bulk_mass_weights(
        _VOLUMES,
        _POSITIONS,
        _RHO0,
        k_hat,
        _K,
        wave_type="P",
        displacement_m=_XI_AMP,
        phase0=0.0,
    )
    witness = plane_wave_displacement(
        _WITNESS, k_hat, _K, "P", amplitude_m=_XI_AMP, phase0=0.0
    )
    w = np.array([0.7, 0.3])

    acc = coherent_point_acceleration(_POSITIONS, weights, _TARGET)
    c_ij = directional_outer_cross_average(acc, witness, w)  # (M, W, Nf, 3, 3)
    c_ji = directional_outer_cross_average(witness, acc, w)  # (W, M, Nf, 3, 3)

    lhs = c_ij
    rhs = np.conj(c_ji).transpose(1, 0, 2, 4, 3)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-14)


def test_shape_fixed_multi_point():
    """Output shape (M, W, Nf, 3, 3) = (2, 3, 2, 3, 3) for a multi-point config."""
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    volumes = np.array([1.0, 1.0])
    k_hat = np.array([[1.0, 0.0, 0.0]])
    wavenumbers = np.array([1.0, 2.0])
    weights = plane_wave_bulk_mass_weights(
        volumes,
        positions,
        _RHO0,
        k_hat,
        wavenumbers,
        wave_type="P",
        displacement_m=_XI_AMP,
        phase0=0.0,
    )
    targets = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    witness_positions = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]])
    witness = plane_wave_displacement(
        witness_positions, k_hat, wavenumbers, "P", amplitude_m=_XI_AMP, phase0=0.0
    )
    w = np.array([1.0])

    c_ij = coherent_gravitoelastic_tensor(positions, weights, targets, witness, w)
    assert c_ij.shape == (2, 3, 2, 3, 3)


def test_read_only_outputs():
    weights = _p_wave_weights()
    witness = _witness()
    c_ij = coherent_gravitoelastic_tensor(_POSITIONS, weights, _TARGET, witness, _W)
    assert not c_ij.flags.writeable

    targets_b = np.array([[2.0, 0.0, 0.0]])
    csd = coherent_acceleration_csd(_POSITIONS, weights, _TARGET, targets_b, _W)
    assert not csd.flags.writeable

    acc = coherent_point_acceleration(_POSITIONS, weights, _TARGET)
    outer = directional_outer_cross_average(acc, witness, _W)
    assert not outer.flags.writeable


def test_chunk_invariance():
    weights = _p_wave_weights()
    witness = _witness()
    c_none = coherent_gravitoelastic_tensor(
        _POSITIONS, weights, _TARGET, witness, _W, chunk_size=None
    )
    c_chunked = coherent_gravitoelastic_tensor(
        _POSITIONS, weights, _TARGET, witness, _W, chunk_size=1
    )
    np.testing.assert_allclose(c_none, c_chunked, rtol=1e-12)

    targets_b = np.array([[2.0, 0.0, 0.0]])
    csd_none = coherent_acceleration_csd(
        _POSITIONS, weights, _TARGET, targets_b, _W, chunk_size=None
    )
    csd_chunked = coherent_acceleration_csd(
        _POSITIONS, weights, _TARGET, targets_b, _W, chunk_size=1
    )
    np.testing.assert_allclose(csd_none, csd_chunked, rtol=1e-12)


def test_directional_outer_cross_average_literal():
    """Direct outer-product check independent of the physical wrapper.

    A = [[1+1j, 0, 0]] (Ma=1, Ndir=1, Nf=1, 3), B = [[2, 3j, 0]] (Mb=1, same),
    w = [1.0] -> c_ij = A_i * conj(B_j) for the single retained direction.
    """
    a = np.array([[[[1 + 1j, 0.0, 0.0]]]])  # (Ma=1, Ndir=1, Nf=1, 3)
    b = np.array([[[[2.0, 3j, 0.0]]]])  # (Mb=1, Ndir=1, Nf=1, 3)
    w = np.array([1.0])

    result = directional_outer_cross_average(a, b, w)
    assert result.shape == (1, 1, 1, 3, 3)

    expected = np.outer(a[0, 0, 0], np.conj(b[0, 0, 0]))
    np.testing.assert_allclose(result[0, 0, 0], expected, rtol=1e-15)


def test_directional_outer_cross_average_shape_errors():
    a = np.zeros((2, 1, 1, 3), dtype=np.complex128)
    b_wrong_ndir = np.zeros((3, 2, 1, 3), dtype=np.complex128)
    try:
        directional_outer_cross_average(a, b_wrong_ndir, np.array([1.0]))
        raise AssertionError("expected ValueError for Ndir mismatch")
    except ValueError:
        pass

    b_wrong_ndim = np.zeros((3, 1, 3), dtype=np.complex128)
    try:
        directional_outer_cross_average(a, b_wrong_ndim, np.array([1.0]))
        raise AssertionError("expected ValueError for ndim mismatch")
    except ValueError:
        pass
