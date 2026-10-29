"""Golden tests for the polarization-projected bulk mass-weight helper (pre-impl).

対象: gragra.coherent.plane_wave_bulk_mass_weights（design spec
docs/design/phase12a_analytic_benchmarks.md §4）。

    δm_j = −ρ₀ · (i k) · (k̂·ê_pol) · ξ · V_j · e^{i k k̂·r_j + iφ₀}   → (N, Ndir, Nf)

期待値・許容誤差の変更は禁止（追記のみ可）。
"""

import numpy as np
import pytest

from gragra.coherent import plane_wave_bulk_mass_weights

RHO = 2700.0
XI = 1e-6


def _small_case():
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.3, -0.2, 1.0]])
    volumes = np.array([1.0, 2.0, 0.5])
    directions = np.array([[0.0, 0.0, 1.0]])
    wavenumber = np.array([2.0, 5.0])
    return positions, volumes, directions, wavenumber


def test_pwave_weights_match_hand_formula():
    pos, vol, dirs, k = _small_case()
    w = plane_wave_bulk_mass_weights(
        vol, pos, RHO, dirs, k, wave_type="P", displacement_m=XI
    )
    # P 波: k̂·ê_pol = 1（ê_pol = k̂）。手計算式と rtol 1e-12 で一致。
    kz = pos[:, 2][:, None, None] * k[None, None, :]
    expected = -1j * RHO * XI * k[None, None, :] * vol[:, None, None] * np.exp(1j * kz)
    assert w.shape == (3, 1, 2)
    assert w.dtype == np.complex128
    np.testing.assert_allclose(w, expected, rtol=1e-12)


@pytest.mark.parametrize("wave_type", ["SH", "SV"])
def test_swave_weights_exactly_zero(wave_type):
    # ∇·ξ = 0 ⇔ k̂·ê_pol = 0（構造的）→ 丸め誤差の ≈0 ではなく厳密 0 を要求。
    pos, vol, dirs, k = _small_case()
    w = plane_wave_bulk_mass_weights(
        vol, pos, RHO, dirs, k, wave_type=wave_type, displacement_m=XI
    )
    assert w.dtype == np.complex128
    assert np.all(w == 0)


def test_multiple_directions_broadcast():
    pos, vol, _, k = _small_case()
    dirs = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    w = plane_wave_bulk_mass_weights(
        vol, pos, RHO, dirs, k, wave_type="P", displacement_m=XI
    )
    assert w.shape == (3, 2, 2)
    # 方向 x̂ の位相は x 座標で決まる
    kx = pos[:, 0][:, None] * k[None, :]
    expected_dir1 = -1j * RHO * XI * k[None, :] * vol[:, None] * np.exp(1j * kx)
    np.testing.assert_allclose(w[:, 1, :], expected_dir1, rtol=1e-12)


@pytest.mark.parametrize("bad_rho", [0.0, -1.0, np.nan])
def test_invalid_density_raises(bad_rho):
    pos, vol, dirs, k = _small_case()
    with pytest.raises(ValueError):
        plane_wave_bulk_mass_weights(vol, pos, bad_rho, dirs, k, wave_type="P")


def test_negative_wavenumber_raises():
    pos, vol, dirs, _ = _small_case()
    with pytest.raises(ValueError):
        plane_wave_bulk_mass_weights(vol, pos, RHO, dirs, [-1.0], wave_type="P")


def test_scalar_wavenumber():
    pos, vol, dirs, _ = _small_case()
    # scalar wavenumber_rad_m = 2.0
    w = plane_wave_bulk_mass_weights(
        vol, pos, RHO, dirs, 2.0, wave_type="P", displacement_m=XI
    )
    assert w.shape == (3, 1, 1)
    assert w.dtype == np.complex128


def test_single_direction_3d_vector():
    pos, vol, _, k = _small_case()
    # (3,) direction vector
    single_dir = np.array([0.0, 0.0, 1.0])
    w = plane_wave_bulk_mass_weights(
        vol, pos, RHO, single_dir, k, wave_type="P", displacement_m=XI
    )
    assert w.shape == (3, 1, 2)


@pytest.mark.parametrize("bad_vol", [0.0, -1.0, np.nan, np.inf])
def test_invalid_volumes_raises(bad_vol):
    pos, vol, dirs, k = _small_case()
    bad_volumes = vol.copy()
    bad_volumes[0] = bad_vol
    with pytest.raises(ValueError):
        plane_wave_bulk_mass_weights(bad_volumes, pos, RHO, dirs, k, wave_type="P")


def test_nan_positions_raises():
    pos, vol, dirs, k = _small_case()
    bad_pos = pos.copy()
    bad_pos[0, 0] = np.nan
    with pytest.raises(ValueError):
        plane_wave_bulk_mass_weights(vol, bad_pos, RHO, dirs, k, wave_type="P")


def test_output_is_readonly():
    pos, vol, dirs, k = _small_case()
    w = plane_wave_bulk_mass_weights(
        vol, pos, RHO, dirs, k, wave_type="P", displacement_m=XI
    )
    assert w.flags.writeable is False
