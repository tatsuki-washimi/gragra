"""Tests for the G-free half-space transfer conventions (Phase 13, T1a-2).

恒等式・固定点は docs/architecture/halfspace_conventions.md §5 の定義関係
δa_z(H) = −4πGρ₀ξ_inc·T(kH) に基づく。golden 深度低減値は 2 指数固有関数から
2026-07-04 に独立導出（Poisson 固体）。

NOTE: analytic/__init__.py は T2 レーン専有のため、本テストはモジュール直 import
（export 配線は T1a マージ時にコーディネータが実施 — チェックリスト項目）。
"""

import numpy as np
import pytest

from gragra.analytic.halfspace import (
    halfspace_transfer_bulk,
    halfspace_transfer_full,
    rayleigh_depth_reduction,
)

VS = 1.0
VP = np.sqrt(3.0)  # Poisson solid
# 2 指数固有関数からの独立導出値（|u(h)|/|u(0)|、単一指数では再現不能）
R_KH_05 = 0.899944442512
R_KH_1 = 0.837384637935


# ---------------------------------------------------------------------------
# transfer 恒等式・固定点（規約 §5）
# ---------------------------------------------------------------------------


def test_identity_full_minus_bulk_is_minus_one():
    # T_full − T_bulk = −1（全 kH の恒等式 — 表面質量シート項 T_sheet = −1）
    k_h = np.linspace(0.0, 12.0, 481)
    np.testing.assert_allclose(
        halfspace_transfer_full(k_h) - halfspace_transfer_bulk(k_h),
        -1.0,
        rtol=0.0,
        atol=1e-15,
    )


def test_endpoints():
    np.testing.assert_allclose(halfspace_transfer_bulk(0.0), -1.0, rtol=1e-15)
    np.testing.assert_allclose(halfspace_transfer_full(0.0), -2.0, rtol=1e-15)


def test_null_frequencies():
    # T_bulk = 0 ⇔ kH = π/3（f = v_P/6H）、T_full = 0 ⇔ kH = π/2（f = v_P/4H）
    assert abs(float(halfspace_transfer_bulk(np.pi / 3.0))) < 1e-15
    assert abs(float(halfspace_transfer_full(np.pi / 2.0))) < 1e-15
    # null の位置がずれていないこと（両側で符号反転）
    assert float(halfspace_transfer_bulk(np.pi / 3.0 - 0.1)) < 0.0
    assert float(halfspace_transfer_bulk(np.pi / 3.0 + 0.1)) > 0.0


def test_transfer_validation_and_output():
    with pytest.raises(ValueError):
        halfspace_transfer_bulk(-0.1)
    with pytest.raises(ValueError):
        halfspace_transfer_full([1.0, np.nan])
    out = halfspace_transfer_bulk([0.5, 1.0])
    assert out.dtype == np.float64
    assert out.flags.writeable is False


# ---------------------------------------------------------------------------
# Rayleigh 深度低減（厳密 2 指数形）
# ---------------------------------------------------------------------------


def test_depth_reduction_surface_is_unity():
    np.testing.assert_allclose(
        rayleigh_depth_reduction(0.0, 1.0, VP, VS), 1.0, rtol=1e-15
    )


def test_depth_reduction_golden_values():
    r = rayleigh_depth_reduction([0.5, 1.0], 1.0, VP, VS)
    np.testing.assert_allclose(r, [R_KH_05, R_KH_1], rtol=1e-9)


def test_depth_reduction_monotone_decreasing_poisson():
    # 単調減少は v_P/v_S ≲ 2.7 の低比率域（Poisson 固体を含む）でのみ成立
    depth = np.linspace(0.0, 6.0, 241)
    r = rayleigh_depth_reduction(depth, 1.0, VP, VS)
    assert np.all(np.diff(r) < 0.0)


def test_depth_reduction_subsurface_maximum_high_ratio():
    # 高 v_P/v_S 比（含水堆積層級）では表面下極大で R > 1 になる実在の物理
    # （v_P/v_S = 5 で max R ≈ 1.031 @ kh ≈ 0.81 — 独立検証済み）。
    # 「reduction」を無条件 R ≤ 1 と読ませないための固定。
    depth = np.linspace(0.0, 6.0, 601)
    r = rayleigh_depth_reduction(depth, 1.0, 5.0 * VS, VS)
    assert float(np.max(r)) > 1.02
    assert 0.5 < float(depth[np.argmax(r)]) < 1.2  # 極大位置 kh ~ 0.81
    assert r[-1] < 0.5  # 十分深部では減衰


def test_depth_reduction_is_not_single_exponential():
    # 単一指数 e^{−kh} なら R(kh=1) = 0.368。厳密 2 指数形は 0.837 で大きく異なる
    # — γ 固定単一指数近似（nn-sus 形）をこのテストが排除する。
    r1 = float(rayleigh_depth_reduction(1.0, 1.0, VP, VS))
    assert abs(r1 - np.exp(-1.0)) > 0.4
    # 単一指数なら R(2h) = R(h)² が厳密に成立する。2 指数形では破れる。
    r_half = float(rayleigh_depth_reduction(0.5, 1.0, VP, VS))
    assert abs(r1 - r_half**2) > 0.02


def test_depth_reduction_wavenumber_scaling():
    # R は無次元組合せ k·h のみに依存する（k を変えて同じ kh なら同じ値）
    r_a = rayleigh_depth_reduction(2.0, 0.5, VP, VS)
    r_b = rayleigh_depth_reduction(0.5, 2.0, VP, VS)
    np.testing.assert_allclose(r_a, r_b, rtol=1e-12)


def test_depth_reduction_validation():
    with pytest.raises(ValueError):
        rayleigh_depth_reduction(-1.0, 1.0, VP, VS)
    with pytest.raises(ValueError):
        rayleigh_depth_reduction(1.0, 0.0, VP, VS)
    with pytest.raises(ValueError):
        rayleigh_depth_reduction(1.0, 1.0, VS, VS)  # v_s >= v_p
    out = rayleigh_depth_reduction([0.0, 1.0], 1.0, VP, VS)
    assert out.dtype == np.float64
    assert out.flags.writeable is False
