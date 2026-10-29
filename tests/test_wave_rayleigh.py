"""Tests for the Rayleigh-wave kinematic basis (Phase 13, T1a-1).

Golden values（Poisson 固体 v_P/v_S = √3、2026-07-04 に一次文献から独立導出・
数値検証済み — docs/architecture/halfspace_conventions.md §3）:

    c_R/v_S = √(2 − 2/√3) = 0.9194016868（厳密閉形式）
    表面楕円率 H/V = 0.681250038633
    水平変位ノード k·h = 1.209481350128

恒等式（正規化非依存）:
    secular 残差 |R(c_R)| < 1e-12
    トラクションフリー: 構成項の最大絶対値で正規化した残差 < 1e-10
    粒子運動位相: Im(u_x conj(u_z)) は表面で負（retrograde）、ノード以深で正
"""

import numpy as np
import pytest

from gragra.waves import (
    rayleigh_displacement,
    rayleigh_ellipticity,
    rayleigh_secular,
    rayleigh_speed,
)

VS = 1.0
VP = np.sqrt(3.0)  # Poisson solid
C_R_EXACT = np.sqrt(2.0 - 2.0 / np.sqrt(3.0))  # 0.9194016868
HV_SURFACE = 0.681250038633
NODE_KH = 1.209481350128
K = 1.0  # rad/m


def _qs(c_r):
    q = np.sqrt(1.0 - c_r**2 / VP**2)
    s = np.sqrt(1.0 - c_r**2 / VS**2)
    return q, s


# ---------------------------------------------------------------------------
# 分散関係（secular / speed）
# ---------------------------------------------------------------------------


def test_rayleigh_speed_poisson_exact_closed_form():
    c_r = rayleigh_speed(VP, VS)
    np.testing.assert_allclose(c_r, C_R_EXACT, rtol=1e-9)


def test_secular_residual_at_root():
    # 求根と独立に secular 式そのものを検証（恒等式）
    c_r = rayleigh_speed(VP, VS)
    assert abs(float(rayleigh_secular(c_r, VP, VS))) < 1e-12


def test_secular_domain_rejected():
    for bad in (0.0, -0.1, VS, 1.5 * VS):
        with pytest.raises(ValueError):
            rayleigh_secular(bad, VP, VS)


def test_speed_validation():
    with pytest.raises(ValueError):
        rayleigh_speed(1.0, 1.0)  # v_s >= v_p
    with pytest.raises(ValueError):
        rayleigh_speed(-3.0, 1.0)
    with pytest.raises(ValueError):
        rayleigh_speed(np.nan, 1.0)
    with pytest.raises(TypeError):
        rayleigh_speed(True, 1.0)


@pytest.mark.parametrize("ratio", [1.05, 1.2, np.sqrt(2.0), 5.0, 50.0])
def test_speed_robust_over_vp_vs_ratios(ratio):
    # 極端な v_P/v_S 比でも 0 < c_R < v_S と小さい secular 残差を保つ
    c_r = rayleigh_speed(ratio * VS, VS)
    assert 0.0 < c_r < VS
    assert abs(float(rayleigh_secular(c_r, ratio * VS, VS))) < 1e-10


# ---------------------------------------------------------------------------
# トラクションフリー境界（解析再構成 — 実装と独立な式で検証）
# ---------------------------------------------------------------------------


def test_traction_free_normalized_residual_analytic():
    """ポテンシャル表現から σ_zz, σ_xz を独立に再構成し、規格化残差 < 1e-10。

    0 目標への rtol 判定は ill-posed のため、残差を構成項の最大絶対値で
    規格化する（halfspace_conventions.md §3.2）。
    """
    rho = 1.0
    mu = rho * VS**2
    lam = rho * VP**2 - 2.0 * mu
    c_r = rayleigh_speed(VP, VS)
    q, s = _qs(c_r)
    k = 1.0
    amp_b = -2j * q / (1.0 + s * s)  # B/A（トラクションフリーから）

    # ポテンシャル φ=e^{-qkh}, ψ=B e^{-skh}（h=深さ）から表面 h=0 で:
    # u_x 項: [i k, s k B]、u_h 項: [-q k, i k B]
    ux_terms = np.array([1j * k, s * k * amp_b])
    uh_terms = np.array([-q * k, 1j * k * amp_b])
    dux_dx = 1j * k * ux_terms
    dux_dh = np.array([-q * k, -s * k]) * ux_terms
    duh_dh = np.array([-q * k, -s * k]) * uh_terms
    duh_dx = 1j * k * uh_terms

    s_hh_terms = lam * (dux_dx + duh_dh) + 2.0 * mu * duh_dh
    s_xh_terms = mu * (dux_dh + duh_dx)

    res_hh = abs(np.sum(s_hh_terms)) / np.max(np.abs(s_hh_terms))
    res_xh = abs(np.sum(s_xh_terms)) / np.max(np.abs(s_xh_terms))
    assert res_hh < 1e-10
    assert res_xh < 1e-10


def test_traction_free_finite_difference_auxiliary():
    """補助検証: rayleigh_displacement の FD 微分でトラクション ≈ 0（rtol≈1e-5 級）。

    表面 z=0 では媒質側（z<=0）しか評価できないため z 方向は 2 次片側差分。
    """
    rho = 1.0
    mu = rho * VS**2
    lam = rho * VP**2 - 2.0 * mu
    delta = 1e-5

    def u(x, z):
        return rayleigh_displacement(
            [[x, 0.0, z]], [1.0, 0.0, 0.0], K, VP, VS, amplitude_m=1.0
        )[0]

    u0 = u(0.0, 0.0)
    # x 方向は中心差分、z 方向は片側 2 次: f'(0) = (3f(0) - 4f(-δ) + f(-2δ))/(2δ)
    du_dx = (u(delta, 0.0) - u(-delta, 0.0)) / (2.0 * delta)
    du_dz = (3.0 * u0 - 4.0 * u(0.0, -delta) + u(0.0, -2.0 * delta)) / (2.0 * delta)

    sigma_zz = lam * (du_dx[0] + du_dz[2]) + 2.0 * mu * du_dz[2]
    sigma_xz = mu * (du_dx[2] + du_dz[0])
    scale = mu * K  # 応力スケール（|u|~1, k=1）
    assert abs(sigma_zz) / scale < 1e-4
    assert abs(sigma_xz) / scale < 1e-4


# ---------------------------------------------------------------------------
# 固有関数の golden 値
# ---------------------------------------------------------------------------


def test_surface_ellipticity_golden():
    hv = rayleigh_ellipticity(0.0, K, VP, VS)
    np.testing.assert_allclose(hv, HV_SURFACE, rtol=1e-9)


def test_horizontal_node_location():
    # ノード位置で楕円率がゼロ交差（単一指数実装では再現不能）
    assert float(rayleigh_ellipticity(NODE_KH / K, K, VP, VS)) < 1e-9
    hv = rayleigh_ellipticity(np.array([1.15, 1.25]) / K, K, VP, VS)
    assert hv[0] > 1e-2 and hv[1] > 1e-3  # 両側で有限（符号はゼロ交差で反転）
    # u_x の quadrature 成分の符号がノードを跨いで反転する
    u = rayleigh_displacement(
        [[0.0, 0.0, -1.15 / K], [0.0, 0.0, -1.25 / K]],
        [1.0, 0.0, 0.0],
        K,
        VP,
        VS,
    )
    assert np.imag(u[0, 0]) * np.imag(u[1, 0]) < 0.0


def test_displacement_normalization_and_closed_form():
    # 正規化: 表面・位相原点で u_z = +amplitude（実数）
    amp = 2.5e-7
    u0 = rayleigh_displacement(
        [[0.0, 0.0, 0.0]], [1.0, 0.0, 0.0], K, VP, VS, amplitude_m=amp
    )[0]
    np.testing.assert_allclose(np.real(u0[2]), amp, rtol=1e-12)
    assert abs(np.imag(u0[2])) < amp * 1e-12
    # 表面の H/V が golden と一致
    np.testing.assert_allclose(abs(u0[0]) / abs(u0[2]), HV_SURFACE, rtol=1e-9)

    # 深さ依存: 閉形式 f_x, f_z（テスト内独立実装）と一致
    c_r = rayleigh_speed(VP, VS)
    q, s = _qs(c_r)
    coef_x = 2.0 * q * s / (1.0 + s * s)
    coef_z = 2.0 / (1.0 + s * s)
    for kh in (0.5, 1.0, 2.0):
        u = rayleigh_displacement(
            [[0.0, 0.0, -kh / K]], [1.0, 0.0, 0.0], K, VP, VS, amplitude_m=amp
        )[0]
        f_x = np.exp(-q * kh) - coef_x * np.exp(-s * kh)
        f_z = np.exp(-q * kh) - coef_z * np.exp(-s * kh)
        f_z0 = 1.0 - coef_z
        np.testing.assert_allclose(np.real(u[2]), amp * f_z / f_z0, rtol=1e-12)
        np.testing.assert_allclose(np.imag(u[0]), amp * f_x / (q * f_z0), rtol=1e-12)


def test_displacement_matches_potential_superposition():
    """実装出力をポテンシャル重ね合わせ場（B=−2iq/(1+s²) の 2 項和 — 実装の
    coef_x/coef_z 結合形とは独立な構成）と解析精度で照合する。"""
    c_r = rayleigh_speed(VP, VS)
    q, s = _qs(c_r)
    amp_b = -2j * q / (1.0 + s * s)
    k = K
    amp = 1.0

    for kh in (0.0, 0.3, 1.0, 2.5):
        # ポテンシャル項の直和（深さ座標フレーム → u_z = −u_h）
        e_q, e_s = np.exp(-q * kh), np.exp(-s * kh)
        u_x_pot = 1j * k * e_q + s * k * amp_b * e_s
        u_h_pot = -q * k * e_q + 1j * k * amp_b * e_s
        u_z_pot = -u_h_pot
        # 実装と同じ正規化（表面 u_z = +amp）に合わせる
        u_z0_pot = -(-q * k + 1j * k * amp_b)
        scale = amp / u_z0_pot
        u_impl = rayleigh_displacement(
            [[0.0, 0.0, -kh / k]], [1.0, 0.0, 0.0], k, VP, VS, amplitude_m=amp
        )[0]
        np.testing.assert_allclose(u_impl[0], scale * u_x_pot, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(u_impl[2], scale * u_z_pot, rtol=1e-12, atol=1e-15)


def test_particle_motion_retrograde_then_prograde():
    # Im(u_x conj(u_z)): 表面で負（retrograde）、ノード以深（kh=2）で正
    u = rayleigh_displacement(
        [[0.0, 0.0, 0.0], [0.0, 0.0, -2.0 / K]], [1.0, 0.0, 0.0], K, VP, VS
    )
    phase_surface = np.imag(u[0, 0] * np.conj(u[0, 2]))
    phase_deep = np.imag(u[1, 0] * np.conj(u[1, 2]))
    assert phase_surface < 0.0
    assert phase_deep > 0.0


def test_general_horizontal_direction():
    # 伝播方向 (1,1,0)/√2: 変位は span{k̂, ẑ} 内（横方向成分ゼロ）
    k_hat = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    perp = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    u = rayleigh_displacement([[0.3, -0.2, -0.5]], [1.0, 1.0, 0.0], K, VP, VS)[0]
    assert abs(u @ perp) < 1e-15
    # 水平成分は k̂ 方向に射影した u_x に一致（位相は k̂·r_xy で決まる）
    assert abs(u[:2] @ k_hat[:2]) > 0.0


# ---------------------------------------------------------------------------
# ガード（overflow・入力検証）
# ---------------------------------------------------------------------------


def test_above_surface_rejected_with_index():
    with pytest.raises(ValueError, match="index: 1"):
        rayleigh_displacement(
            [[0.0, 0.0, -1.0], [0.0, 0.0, 0.1]], [1.0, 0.0, 0.0], K, VP, VS
        )


def test_direction_validation():
    with pytest.raises(ValueError, match="horizontal"):
        rayleigh_displacement([[0.0, 0.0, 0.0]], [0.0, 0.0, 1.0], K, VP, VS)
    with pytest.raises(ValueError, match="non-zero"):
        rayleigh_displacement([[0.0, 0.0, 0.0]], [0.0, 0.0, 0.0], K, VP, VS)


def test_wavenumber_and_amplitude_validation():
    with pytest.raises(ValueError):
        rayleigh_displacement([[0.0, 0.0, 0.0]], [1.0, 0.0, 0.0], -1.0, VP, VS)
    with pytest.raises(ValueError):
        rayleigh_displacement([[0.0, 0.0, 0.0]], [1.0, 0.0, 0.0], 0.0, VP, VS)
    with pytest.raises(ValueError):
        rayleigh_displacement(
            [[0.0, 0.0, 0.0]], [1.0, 0.0, 0.0], K, VP, VS, amplitude_m=0.0
        )


def test_ellipticity_deep_no_nan_and_deep_limit():
    # 深部では素朴な f_x/f_z が両方 underflow して 0/0 = NaN になる
    # （Codex レビュー指摘）。factored form により全深度で有限のまま、
    # 深部極限 H/V → s に収束する。
    c_r = rayleigh_speed(VP, VS)
    _, s = _qs(c_r)
    hv = rayleigh_ellipticity([1.0e3, 5.0e3], K, VP, VS)
    assert np.all(np.isfinite(hv))
    np.testing.assert_allclose(hv, s, rtol=1e-9)
    # 浅部（表面 golden）との整合は test_surface_ellipticity_golden が担保


def test_ellipticity_validation():
    with pytest.raises(ValueError):
        rayleigh_ellipticity(-0.1, K, VP, VS)
    with pytest.raises(ValueError):
        rayleigh_ellipticity(np.nan, K, VP, VS)


def test_output_dtype_shape_readonly():
    u = rayleigh_displacement(
        [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]], [1.0, 0.0, 0.0], K, VP, VS
    )
    assert u.shape == (2, 3)
    assert u.dtype == np.complex128
    assert u.flags.writeable is False
    hv = rayleigh_ellipticity([0.0, 1.0], K, VP, VS)
    assert hv.dtype == np.float64
    assert hv.flags.writeable is False
