"""Golden tests for Phase 12-a spherical-cavity closed forms (committed pre-impl).

期待値・許容誤差は docs/design/phase12a_analytic_benchmarks.md §5 と
docs/architecture/halfspace_conventions.md に基づく。実装者（agy）による
期待値・許容誤差の変更は禁止（追記のみ可）。各許容誤差の設計根拠はコメントに明記。

閉形式（G-free、x = k_P R）:
    bulk(P)  =  4πρ₀ξ · j₀(x)
    wall(P)  = −4πρ₀ξ · (j₀(x) − 2j₁(x)/x)
    wall(S)  = −4πρ₀ξ · j₁(x)/x
    total(P) =  bulk + wall = 8πρ₀ξ · j₁(x)/x
"""

import numpy as np
import pytest

from gragra import G_SI, coherent_point_acceleration
from gragra.analytic import (
    spherical_cavity_bulk_p,
    spherical_cavity_total_p,
    spherical_cavity_wall,
    spherical_shell_dipole_factor,
)
from gragra.analytic._bessel import (
    cavity_wall_geometric,
    spherical_j0,
    spherical_j1,
)

RHO = 1000.0  # kg/m^3
XI = 1e-6  # m (displacement amplitude)
R = 1.0  # m (cavity radius)

# 高精度基準値（級数を x^12 まで、誤差 << 1e-16。導出は design spec §1/§2）
WALL_GEOMETRIC_AT_0P1 = 0.3323339284171301644


# ---------------------------------------------------------------------------
# 小空洞（長波長）極限 — 項別固定（factor 2–3 の取り違えを構造的に検出）
# ---------------------------------------------------------------------------


def test_small_cavity_limit_bulk_p():
    # bulk = 4πρξ j₀(x)、j₀(x) = 1 − x²/6 + O(x⁴) → x=1e-3 の相対偏差 = x²/6 = 1.67e-7
    x = 1e-3
    val = spherical_cavity_bulk_p(R, x / R, RHO, XI)
    np.testing.assert_allclose(val, 4 * np.pi * RHO * XI, rtol=2.5e-7)


def test_small_cavity_limit_wall_p():
    # wall = −4πρξ(1/3 − x²/10 + …) → 極限 −(4π/3)ρξ、相対偏差 = 3x²/10 = 3.0e-7
    x = 1e-3
    val = spherical_cavity_wall(R, x / R, RHO, XI, wave_type="P")
    np.testing.assert_allclose(val, -(4 * np.pi / 3) * RHO * XI, rtol=4.5e-7)


def test_small_cavity_limit_wall_s():
    # wall(S) = −4πρξ(1/3 − x²/30 + …) → 極限は P/S 共通、相対偏差 = x²/10
    x = 1e-3
    val = spherical_cavity_wall(R, x / R, RHO, XI, wave_type="S")
    np.testing.assert_allclose(val, -(4 * np.pi / 3) * RHO * XI, rtol=1.5e-7)


def test_small_cavity_limit_total_p():
    # total = 8πρξ j₁(x)/x = (8π/3)ρξ(1 − x²/10 + …) → 相対偏差 = x²/10 = 1.0e-7
    x = 1e-3
    val = spherical_cavity_total_p(R, x / R, RHO, XI)
    np.testing.assert_allclose(val, (8 * np.pi / 3) * RHO * XI, rtol=1.5e-7)


# ---------------------------------------------------------------------------
# 固定点 x = π
# ---------------------------------------------------------------------------


def test_fixed_point_pi_bulk_vanishes():
    # j₀(π) = 0（厳密）→ bulk = 0。atol は振幅スケール 4πρξ × 1e-13。
    val = spherical_cavity_bulk_p(R, np.pi / R, RHO, XI)
    assert abs(val) < 4 * np.pi * RHO * XI * 1e-13


def test_fixed_point_pi_total():
    # j₁(π) = 1/π（厳密）→ total(π) = 8πρξ·(1/π)/π = 8ρξ/π
    val = spherical_cavity_total_p(R, np.pi / R, RHO, XI)
    np.testing.assert_allclose(val, 8 * RHO * XI / np.pi, rtol=1e-12)


# ---------------------------------------------------------------------------
# _bessel: 級数係数・閾値連続性・scipy 突合
# ---------------------------------------------------------------------------


def test_wall_geometric_small_x_series_coefficient():
    # 1/3 − x²/10 + x⁴/168 − x⁶/6480（x⁴ 係数 1/168 が正 — nn-sus の 1/280 はバグ）。
    # x=0.1 で 1/280 系列は基準値と 2.4e-7 ずれるため rtol 1e-12 が混入を検出する。
    np.testing.assert_allclose(
        cavity_wall_geometric(0.1), WALL_GEOMETRIC_AT_0P1, rtol=1e-12
    )


def test_bessel_threshold_continuity():
    # 閾値 X0 = 0.1 の両側で級数と直接式が一致（直接式の桁落ちは x=0.1 で ~1e-13）
    for f in (spherical_j0, spherical_j1, cavity_wall_geometric):
        below = f(np.nextafter(0.1, 0.0))
        above = f(np.nextafter(0.1, 1.0))
        np.testing.assert_allclose(below, above, rtol=1e-12)


def test_bessel_at_zero_and_j0_pi():
    np.testing.assert_allclose(spherical_j0(0.0), 1.0, rtol=1e-15)
    assert abs(spherical_j1(0.0)) == 0.0
    np.testing.assert_allclose(cavity_wall_geometric(0.0), 1.0 / 3.0, rtol=1e-15)
    assert abs(spherical_j0(np.pi)) < 1e-15


def test_bessel_against_scipy():
    scipy_special = pytest.importorskip("scipy.special")
    x = np.logspace(-8, 1, 200)
    np.testing.assert_allclose(
        spherical_j0(x), scipy_special.spherical_jn(0, x), rtol=1e-12
    )
    np.testing.assert_allclose(
        spherical_j1(x), scipy_special.spherical_jn(1, x), rtol=1e-12
    )


# ---------------------------------------------------------------------------
# 恒等式
# ---------------------------------------------------------------------------


def test_identity_total_equals_bulk_plus_wall():
    k = np.linspace(0.05, 8.0, 160) / R
    total = spherical_cavity_total_p(R, k, RHO, XI)
    bulk = spherical_cavity_bulk_p(R, k, RHO, XI)
    wall = spherical_cavity_wall(R, k, RHO, XI, wave_type="P")
    np.testing.assert_allclose(total, bulk + wall, rtol=1e-12, atol=0.0)


def test_identity_total_vs_shell_dipole_factor():
    # total(P) = −8πρξ · shell_dipole_factor(x)、shell_dipole_factor = −j₁(x)/x
    k = np.linspace(0.05, 8.0, 160) / R
    total = spherical_cavity_total_p(R, k, RHO, XI)
    shell = spherical_shell_dipole_factor(k * R)
    np.testing.assert_allclose(total, -8 * np.pi * RHO * XI * shell, rtol=1e-12)


# ---------------------------------------------------------------------------
# 入力検証・dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_invalid_radius_raises(bad):
    with pytest.raises(ValueError):
        spherical_cavity_bulk_p(bad, 1.0, RHO, XI)


def test_output_is_readonly_float64():
    k = np.array([0.5, 1.0])
    out = spherical_cavity_total_p(R, k, RHO, XI)
    assert out.dtype == np.float64
    assert out.flags.writeable is False


# ---------------------------------------------------------------------------
# 数値経路突合（同じ項同士の比較 — design spec §0 の定式化規約）
# ---------------------------------------------------------------------------


def _temporary_analytic_test_sphere_surface(radius, n_theta=40, n_phi=80):
    # TODO: T3 の geometry.surface_mesh マージ後にこの一時 helper を置き換える
    # （roadmap Backlog 登録済み）。lat-lon 中点グリッドの centroids/normals/areas。
    theta = (np.arange(n_theta) + 0.5) * np.pi / n_theta
    phi = (np.arange(n_phi) + 0.5) * 2.0 * np.pi / n_phi
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    sin_t = np.sin(tt)
    normals = np.stack(
        [sin_t * np.cos(pp), sin_t * np.sin(pp), np.cos(tt)], axis=-1
    ).reshape(-1, 3)
    centroids = radius * normals
    areas = (radius**2 * sin_t * (np.pi / n_theta) * (2.0 * np.pi / n_phi)).reshape(-1)
    return centroids, normals, areas


def test_numerical_bulk_finite_shell_matches_closed_form():
    # 有限シェル R..2R の bulk 閉形式: 4πρξ[j₀(kR) − j₀(2kR)]（設計 spec §5.6）。
    # kR = 1、spacing = R/15、rtol 0.05（収束 slope アサーションは置かない）。
    from gragra.coherent import plane_wave_bulk_mass_weights

    k = 1.0 / R
    spacing = R / 15.0
    axis = np.arange(-2.0 * R + spacing / 2, 2.0 * R, spacing)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    pos = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    r = np.linalg.norm(pos, axis=1)
    mask = (r >= R) & (r <= 2.0 * R)
    pos = pos[mask]
    vol = np.full(len(pos), spacing**3)

    weights = plane_wave_bulk_mass_weights(
        vol, pos, RHO, [[0.0, 0.0, 1.0]], [k], wave_type="P", displacement_m=XI
    )
    acc = coherent_point_acceleration(pos, weights, [[0.0, 0.0, 0.0]])
    a_z = acc[0, 0, 0, 2]

    expected = G_SI * 4 * np.pi * RHO * XI * (np.sin(1.0) / 1.0 - np.sin(2.0) / 2.0)
    np.testing.assert_allclose(np.real(a_z), expected, rtol=0.05)
    assert abs(np.imag(a_z)) < 0.01 * abs(np.real(a_z))


def test_numerical_wall_surface_matches_closed_form():
    # wall 項: w_j = −ρ₀ (ê_pol·n̂_j) ξ e^{i k k̂·r_j} dS_j（n̂ = 空洞外向き = 媒質向き）。
    # kR = 1、lat-lon 40×80（離散化誤差 ~(π/40)² ≈ 6e-3）→ rtol 0.02。
    k = 1.0 / R
    centroids, normals, areas = _temporary_analytic_test_sphere_surface(R)
    e_pol = np.array([0.0, 0.0, 1.0])
    phase = np.exp(1j * k * centroids @ e_pol)
    w = (-RHO * XI * (normals @ e_pol) * areas * phase).reshape(-1, 1, 1)

    acc = coherent_point_acceleration(centroids, w, [[0.0, 0.0, 0.0]])
    a_z = acc[0, 0, 0, 2]

    j0_1 = np.sin(1.0)
    j1_1 = np.sin(1.0) - np.cos(1.0)
    expected = G_SI * (-4 * np.pi * RHO * XI) * (j0_1 - 2.0 * j1_1)
    np.testing.assert_allclose(np.real(a_z), expected, rtol=0.02)
    assert abs(np.imag(a_z)) < 0.01 * abs(np.real(a_z))
