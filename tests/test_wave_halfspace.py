"""Golden tests for Phase 13 T1b — 自由表面係数 + evanescent API（pre-impl）.

実装（agy 委任）より先にコミットする golden / overflow guard テスト。期待値・許容誤差の
変更は禁止（追記のみ可）。規約の一次資料は docs/architecture/halfspace_conventions.md
§4（overflow ガード）・§6（鉛直入射・変位規約）。

- 反射係数は値の単独固定でなく **恒等式 1 + R_PP = 表面変位倍率 = 2.0** で規約ごと固定
  （ポテンシャル規約の文献では符号が逆になるため — conventions §6）。
- evanescent 因子は e^{i k (k̂_h·r_xy) + iφ₀} · e^{−κ h}（h = −z ≥ 0）。z > 0 と κ < 0 は
  exp 成長方向として ValueError（構造的 overflow ガード）。
"""

import numpy as np
import pytest

from gragra.waves import (
    evanescent_complex_factor,
    free_surface_reflection_vertical,
    plane_wave_complex_factor,
    surface_displacement_factor_vertical,
)

K = 2.0  # rad/m
KAPPA = 0.7  # 1/m


# ---------------------------------------------------------------------------
# 鉛直入射自由表面係数（変位規約）
# ---------------------------------------------------------------------------


def test_psv_vertical_decoupling_and_identity():
    coeff = free_surface_reflection_vertical("P")
    # 法線入射では P/SV がデカップルしモード変換はゼロ（厳密値）
    assert coeff["R_PP"] == 1.0
    assert coeff["R_PS"] == 0.0
    assert coeff["R_SP"] == 0.0
    assert coeff["R_SS"] == 1.0
    # 恒等式: 1 + R_PP = 表面変位倍率 = 2.0（変位規約ごと固定）
    assert 1.0 + coeff["R_PP"] == surface_displacement_factor_vertical() == 2.0


def test_sh_vertical_reflection():
    coeff = free_surface_reflection_vertical("SH")
    assert coeff["R_SH"] == 1.0
    assert 1.0 + coeff["R_SH"] == surface_displacement_factor_vertical()


def test_sv_vertical_same_as_psv_block():
    coeff = free_surface_reflection_vertical("SV")
    assert coeff["R_SS"] == 1.0
    assert coeff["R_SP"] == 0.0


def test_unknown_wave_type_raises():
    with pytest.raises(ValueError):
        free_surface_reflection_vertical("X")


# ---------------------------------------------------------------------------
# evanescent 位相 API（overflow ガード込み）
# ---------------------------------------------------------------------------


def test_evanescent_factor_hand_formula():
    # e^{i(k k̂·r_xy + φ₀)} · e^{κ z}（z ≤ 0 なので e^{−κh} ≤ 1）
    positions = np.array([[0.5, -0.3, 0.0], [0.5, -0.3, -2.0]])
    k_hat = np.array([[1.0, 0.0, 0.0]])
    phase0 = 0.25
    out = evanescent_complex_factor(positions, k_hat, [K], KAPPA, phase0=phase0)
    assert out.shape == (2, 1, 1)
    assert out.dtype == np.complex128
    expected_surface = np.exp(1j * (K * 0.5 + phase0))
    np.testing.assert_allclose(out[0, 0, 0], expected_surface, rtol=1e-12)
    np.testing.assert_allclose(
        out[1, 0, 0], expected_surface * np.exp(-KAPPA * 2.0), rtol=1e-12
    )


def test_evanescent_zero_decay_matches_plane_wave_horizontal():
    # κ = 0 で水平伝播の plane_wave_complex_factor と厳密一致（互換の固定点）
    positions = np.array([[0.3, 0.7, -1.5], [-0.2, 0.1, -0.4]])
    k_hat = np.array([[0.6, 0.8, 0.0]])
    out = evanescent_complex_factor(positions, k_hat, [K], 0.0)
    # z 成分を落とした水平位相と等価
    pos_h = positions.copy()
    pos_h[:, 2] = 0.0
    ref = plane_wave_complex_factor(pos_h, k_hat, [K])
    np.testing.assert_allclose(out, ref, rtol=1e-12)


def test_evanescent_decay_monotone_with_depth():
    depths = np.array([[0.0, 0.0, -z] for z in (0.0, 0.5, 1.0, 3.0, 10.0)])
    out = np.abs(evanescent_complex_factor(depths, [[1.0, 0.0, 0.0]], [K], KAPPA))[
        :, 0, 0
    ]
    assert np.all(np.diff(out) < 0.0)
    assert out[0] == 1.0


def test_evanescent_above_surface_rejected():
    # z > 0 は exp 成長方向 — 構造的拒否（overflow ガード）
    with pytest.raises(ValueError):
        evanescent_complex_factor([[0.0, 0.0, 0.1]], [[1.0, 0.0, 0.0]], [K], KAPPA)


def test_evanescent_negative_decay_rejected():
    with pytest.raises(ValueError):
        evanescent_complex_factor([[0.0, 0.0, -1.0]], [[1.0, 0.0, 0.0]], [K], -0.1)


def test_evanescent_nonhorizontal_direction_rejected():
    with pytest.raises(ValueError):
        evanescent_complex_factor([[0.0, 0.0, -1.0]], [[0.0, 0.0, 1.0]], [K], KAPPA)


def test_evanescent_broadcast_shape():
    positions = np.zeros((3, 3))
    positions[:, 2] = -1.0
    out = evanescent_complex_factor(
        positions, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [1.0, 2.0, 3.0], KAPPA
    )
    assert out.shape == (3, 2, 3)


def test_evanescent_output_readonly():
    out = evanescent_complex_factor([[0.0, 0.0, -1.0]], [[1.0, 0.0, 0.0]], [K], KAPPA)
    assert out.flags.writeable is False
