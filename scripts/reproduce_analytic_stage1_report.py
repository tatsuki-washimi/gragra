#!/usr/bin/env python3
"""Reproduction script for Stage 1 analytic verification report tables.

This script executes the exact independent quadrature comparisons and convergence studies
documented in `docs/validation/analytic-stage1-report.md` (§4.1 and §4.2), and outputs the
resulting tables containing:
  - Evaluation condition
  - Observable name & physical unit
  - Analytical closed-form value (float64 tracking precision)
  - Independent quadrature reference value (float64 tracking precision)
  - Normalization scale (denominator definition)
  - Absolute difference
  - Observed relative error / normalized residual
  - Acceptance threshold and PASS/FAIL verdict

Supported CLI options:
  (default)     Print text table summary and verification verdict.
  --markdown    Print comprehensive Markdown table with all numerical columns.
  --json        Print structured JSON output for machine inspection.

Exit codes:
  0: All §4.1 verification items passed.
  1: One or more §4.1 verification items failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
import numpy as np

try:
    from scipy import integrate
except ImportError:
    integrate = None  # type: ignore[assignment]

from gragra.constants import G_SI
from gragra.analytic import (
    cuboid_observables_g_free,
    cylinder_axis_observables_g_free,
    spherical_shell_p_acceleration_components_g_free,
)
from gragra.observables import cuboid_observables


def run_table_4_1() -> list[dict[str, Any]]:
    """Run all benchmarks for §4.1 and collect atomic component-by-component rows."""
    if integrate is None:
        raise RuntimeError("scipy is required for numerical quadrature benchmarks. Install scipy to run §4 benchmarks.")
    rows: list[dict[str, Any]] = []
    center = np.zeros(3)

    # -------------------------------------------------------------------------
    # 1. Cuboid External Edge Extension [1, 1, 2]
    # -------------------------------------------------------------------------
    cube_lengths = np.array([2.0, 2.0, 2.0])
    cube_rho = 1.0
    edge_target = np.array([1.0, 1.0, 2.0])

    obs_edge = cuboid_observables_g_free(center, cube_lengths, cube_rho, np.array([edge_target]))
    calc_gxy = float(obs_edge.gradient[0, 0, 1])

    def integrand_edge_gxy(z: float, y: float, x: float) -> float:
        d = np.array([x - edge_target[0], y - edge_target[1], z - edge_target[2]])
        r = float(np.linalg.norm(d))
        return 3.0 * float(d[0] * d[1]) / (r**5)

    ref_edge_gxy, _ = integrate.tplquad(
        integrand_edge_gxy, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, epsabs=1e-12, epsrel=1e-11
    )
    diff_edge_gxy = abs(calc_gxy - ref_edge_gxy)
    scale_edge_gxy = abs(ref_edge_gxy)
    rel_edge_gxy = diff_edge_gxy / scale_edge_gxy
    thresh_edge_gxy = 1.0e-10

    rows.append({
        "shape": "直方体",
        "condition": "外部稜線延長点 [1, 1, 2] ([-1,1]^3, rho=1)",
        "observable": "Gamma_xy",
        "unit": "kg/m^3",
        "method": "3D tplquad",
        "quad_cond": "epsabs=1e-12, epsrel=1e-11",
        "scale": scale_edge_gxy,
        "scale_str": f"{scale_edge_gxy:.6f}",
        "analytic": calc_gxy,
        "analytic_str": f"{calc_gxy:+.16e}",
        "reference": ref_edge_gxy,
        "reference_str": f"{ref_edge_gxy:+.16e}",
        "abs_diff": diff_edge_gxy,
        "abs_diff_str": f"{diff_edge_gxy:.3e}",
        "rel_err": rel_edge_gxy,
        "rel_err_str": f"{rel_edge_gxy:.2e}",
        "threshold": thresh_edge_gxy,
        "threshold_str": f"{thresh_edge_gxy:.1e}",
        "status": "PASS" if rel_edge_gxy <= thresh_edge_gxy else "FAIL",
    })

    # -------------------------------------------------------------------------
    # 2. Cuboid Interior Point [0.2, 0.3, 0.4]
    # -------------------------------------------------------------------------
    c_lengths = np.array([2.0, 3.0, 4.0])
    c_half = c_lengths / 2.0
    c_rho = 2500.0
    c_target = np.array([0.2, 0.3, 0.4])
    xt, yt, zt = c_target

    obs_in = cuboid_observables_g_free(center, c_lengths, c_rho, np.array([c_target]))

    # 2a. Potential: 3D tplquad
    def integrand_c_phi(z: float, y: float, x: float) -> float:
        d = np.array([x - xt, y - yt, z - zt])
        return 1.0 / float(np.linalg.norm(d))

    val_c_phi, _ = integrate.tplquad(
        integrand_c_phi, -c_half[0], c_half[0], -c_half[1], c_half[1], -c_half[2], c_half[2],
        epsabs=1e-12, epsrel=1e-11
    )
    ref_c_phi = float(-c_rho * val_c_phi)
    calc_c_phi = float(obs_in.potential[0])
    diff_c_phi = abs(calc_c_phi - ref_c_phi)
    scale_c_phi = abs(ref_c_phi)
    rel_c_phi = diff_c_phi / scale_c_phi
    thresh_c_phi = 1.0e-8

    rows.append({
        "shape": "直方体",
        "condition": "内部点 [0.2, 0.3, 0.4] ([2,3,4], rho=2500)",
        "observable": "Phi",
        "unit": "kg/m",
        "method": "3D tplquad",
        "quad_cond": "epsabs=1e-12, epsrel=1e-11",
        "scale": scale_c_phi,
        "scale_str": f"{scale_c_phi:.4e}",
        "analytic": calc_c_phi,
        "analytic_str": f"{calc_c_phi:+.16e}",
        "reference": ref_c_phi,
        "reference_str": f"{ref_c_phi:+.16e}",
        "abs_diff": diff_c_phi,
        "abs_diff_str": f"{diff_c_phi:.3e}",
        "rel_err": rel_c_phi,
        "rel_err_str": f"{rel_c_phi:.2e}",
        "threshold": thresh_c_phi,
        "threshold_str": f"{thresh_c_phi:.1e}",
        "status": "PASS" if rel_c_phi <= thresh_c_phi else "FAIL",
    })

    # 2b. Acceleration: 6-face non-singular surface integrals
    ref_c_a = np.zeros(3)
    for i in range(3):
        axes = [0, 1, 2]
        axes.remove(i)
        ax1, ax2 = axes

        def face_pos(u2: float, u1: float, i: int = i, ax1: int = ax1, ax2: int = ax2) -> float:
            pt = [0.0, 0.0, 0.0]
            pt[i] = float(c_half[i])
            pt[ax1] = u1
            pt[ax2] = u2
            d = np.array(pt) - c_target
            return 1.0 / float(np.linalg.norm(d))

        def face_neg(u2: float, u1: float, i: int = i, ax1: int = ax1, ax2: int = ax2) -> float:
            pt = [0.0, 0.0, 0.0]
            pt[i] = float(-c_half[i])
            pt[ax1] = u1
            pt[ax2] = u2
            d = np.array(pt) - c_target
            return -1.0 / float(np.linalg.norm(d))

        vp, _ = integrate.dblquad(face_pos, -c_half[ax1], c_half[ax1], -c_half[ax2], c_half[ax2], epsabs=1e-13, epsrel=1e-12)
        vm, _ = integrate.dblquad(face_neg, -c_half[ax1], c_half[ax1], -c_half[ax2], c_half[ax2], epsabs=1e-13, epsrel=1e-12)
        ref_c_a[i] = -c_rho * (vp + vm)

    calc_c_a = obs_in.acceleration[0]
    thresh_c_a = 1.0e-10
    a_names = ["a_x", "a_y", "a_z"]
    for i in range(3):
        calc_ai = float(calc_c_a[i])
        ref_ai = float(ref_c_a[i])
        diff_ai = abs(calc_ai - ref_ai)
        scale_ai = abs(ref_ai)
        rel_ai = diff_ai / scale_ai
        rows.append({
            "shape": "直方体",
            "condition": "内部点 [0.2, 0.3, 0.4] ([2,3,4], rho=2500)",
            "observable": a_names[i],
            "unit": "kg/m^2",
            "method": "6面表面積分 2D dblquad (発散定理)",
            "quad_cond": "epsabs=1e-13, epsrel=1e-12",
            "scale": scale_ai,
            "scale_str": f"{scale_ai:.4e}",
            "analytic": calc_ai,
            "analytic_str": f"{calc_ai:+.16e}",
            "reference": ref_ai,
            "reference_str": f"{ref_ai:+.16e}",
            "abs_diff": diff_ai,
            "abs_diff_str": f"{diff_ai:.3e}",
            "rel_err": rel_ai,
            "rel_err_str": f"{rel_ai:.2e}",
            "threshold": thresh_c_a,
            "threshold_str": f"{thresh_c_a:.1e}",
            "status": "PASS" if rel_ai <= thresh_c_a else "FAIL",
        })

    # 2c. Gradient: 6-face non-singular surface integrals
    ref_c_g = np.zeros((3, 3))
    for i in range(3):
        axes = [0, 1, 2]
        axes.remove(i)
        ax1, ax2 = axes
        for j in range(3):
            def face_g_pos(u2: float, u1: float, i: int = i, j: int = j, ax1: int = ax1, ax2: int = ax2) -> float:
                pt = [0.0, 0.0, 0.0]
                pt[i] = float(c_half[i])
                pt[ax1] = u1
                pt[ax2] = u2
                d = np.array(pt) - c_target
                r = float(np.linalg.norm(d))
                return float(d[j]) / (r**3)

            def face_g_neg(u2: float, u1: float, i: int = i, j: int = j, ax1: int = ax1, ax2: int = ax2) -> float:
                pt = [0.0, 0.0, 0.0]
                pt[i] = float(-c_half[i])
                pt[ax1] = u1
                pt[ax2] = u2
                d = np.array(pt) - c_target
                r = float(np.linalg.norm(d))
                return float(-d[j]) / (r**3)

            vp, _ = integrate.dblquad(face_g_pos, -c_half[ax1], c_half[ax1], -c_half[ax2], c_half[ax2], epsabs=1e-13, epsrel=1e-12)
            vm, _ = integrate.dblquad(face_g_neg, -c_half[ax1], c_half[ax1], -c_half[ax2], c_half[ax2], epsabs=1e-13, epsrel=1e-12)
            ref_c_g[i, j] = -c_rho * (vp + vm)

    calc_c_g = obs_in.gradient[0]
    thresh_c_g = 1.0e-10
    g_components = [
        ("Gamma_xx", 0, 0), ("Gamma_yy", 1, 1), ("Gamma_zz", 2, 2),
        ("Gamma_xy", 0, 1), ("Gamma_xz", 0, 2), ("Gamma_yz", 1, 2),
    ]
    for gname, i, j in g_components:
        calc_gij = float(calc_c_g[i, j])
        ref_gij = float(ref_c_g[i, j])
        diff_gij = abs(calc_gij - ref_gij)
        scale_gij = abs(ref_gij)
        rel_gij = diff_gij / scale_gij if scale_gij > 0.0 else diff_gij
        rows.append({
            "shape": "直方体",
            "condition": "内部点 [0.2, 0.3, 0.4] ([2,3,4], rho=2500)",
            "observable": gname,
            "unit": "kg/m^3",
            "method": "6面表面積分 2D dblquad (発散定理)",
            "quad_cond": "epsabs=1e-13, epsrel=1e-12",
            "scale": scale_gij,
            "scale_str": f"{scale_gij:.4e}",
            "analytic": calc_gij,
            "analytic_str": f"{calc_gij:+.16e}",
            "reference": ref_gij,
            "reference_str": f"{ref_gij:+.16e}",
            "abs_diff": diff_gij,
            "abs_diff_str": f"{diff_gij:.3e}",
            "rel_err": rel_gij,
            "rel_err_str": f"{rel_gij:.2e}",
            "threshold": thresh_c_g,
            "threshold_str": f"{thresh_c_g:.1e}",
            "status": "PASS" if rel_gij <= thresh_c_g else "FAIL",
        })

    # 2d. Interior Poisson trace check
    trace_in = float(np.trace(calc_c_g))
    expected_trace_in = float(-4.0 * np.pi * c_rho)
    diff_trace_in = abs(trace_in - expected_trace_in)
    scale_trace_in = abs(expected_trace_in)
    rel_trace_in = diff_trace_in / scale_trace_in
    thresh_trace_in = 1.0e-12

    rows.append({
        "shape": "直方体",
        "condition": "内部点 [0.2, 0.3, 0.4] ([2,3,4], rho=2500)",
        "observable": "tr(Gamma) + 4*pi*rho",
        "unit": "kg/m^3",
        "method": "代数 Poisson 則",
        "quad_cond": "厳密恒等式",
        "scale": scale_trace_in,
        "scale_str": f"4*pi*rho = {scale_trace_in:.4e}",
        "analytic": trace_in,
        "analytic_str": f"{trace_in:+.16e}",
        "reference": expected_trace_in,
        "reference_str": f"{expected_trace_in:+.16e}",
        "abs_diff": diff_trace_in,
        "abs_diff_str": f"{diff_trace_in:.3e}",
        "rel_err": rel_trace_in,
        "rel_err_str": f"{rel_trace_in:.2e}",
        "threshold": thresh_trace_in,
        "threshold_str": f"{thresh_trace_in:.1e}",
        "status": "PASS" if rel_trace_in <= thresh_trace_in else "FAIL",
    })

    # -------------------------------------------------------------------------
    # 3. Cuboid Exterior Point [0.0, 5.0, 0.0] (Laplace trace check)
    # -------------------------------------------------------------------------
    ext_target = np.array([[0.0, 5.0, 0.0]])
    ext_rho = 2700.0
    si_ext = cuboid_observables(center, c_lengths, ext_rho, ext_target)
    trace_ext_si = float(np.trace(si_ext.gradient[0]))
    abs_res_ext_si = abs(trace_ext_si)
    poisson_scale_si = float(4.0 * np.pi * G_SI * ext_rho)
    norm_res_ext = abs_res_ext_si / poisson_scale_si
    thresh_ext_trace = 1.0e-14

    rows.append({
        "shape": "直方体",
        "condition": "外部点 [0.0, 5.0, 0.0] ([2,3,4], rho=2700)",
        "observable": "tr(Gamma) (Laplace 則)",
        "unit": "s^-2",
        "method": "代数 Laplace 則",
        "quad_cond": "厳密恒等式",
        "scale": poisson_scale_si,
        "scale_str": f"4*pi*G*rho = {poisson_scale_si:.4e}",
        "analytic": trace_ext_si,
        "analytic_str": f"{trace_ext_si:+.16e}",
        "reference": 0.0,
        "reference_str": "0.0000000000000000e+00",
        "abs_diff": abs_res_ext_si,
        "abs_diff_str": f"{abs_res_ext_si:.3e}",
        "rel_err": norm_res_ext,
        "rel_err_str": f"{norm_res_ext:.2e} (無次元規格化残差)",
        "threshold": thresh_ext_trace,
        "threshold_str": f"{thresh_ext_trace:.1e} (無次元規格化残差)",
        "status": "PASS" if norm_res_ext <= thresh_ext_trace else "FAIL",
    })

    # -------------------------------------------------------------------------
    # 4. Cylinder Interior Point z_t = 0.5
    # -------------------------------------------------------------------------
    cyl_r = 2.0
    cyl_h = 4.0
    cyl_rho = 1500.0
    cyl_in_target = np.array([[0.0, 0.0, 0.5]])

    obs_cyl_in = cylinder_axis_observables_g_free(center, cyl_r, cyl_h, cyl_rho, cyl_in_target)
    calc_cyl_phi = float(obs_cyl_in.potential[0])
    calc_cyl_az = float(obs_cyl_in.acceleration[0, 2])
    calc_cyl_gzz = float(obs_cyl_in.gradient[0, 2, 2])
    calc_cyl_gxx = float(obs_cyl_in.gradient[0, 0, 0])

    # 4a. Cylinder Phi
    def integrand_cyl_phi(z_prime: float) -> float:
        u = z_prime - 0.5
        return float(np.sqrt(cyl_r**2 + u**2) - abs(u))

    val_cyl_phi, _ = integrate.quad(integrand_cyl_phi, -2.0, 2.0, epsabs=1e-12, epsrel=1e-11)
    ref_cyl_phi = float(-2.0 * np.pi * cyl_rho * val_cyl_phi)
    diff_cyl_phi = abs(calc_cyl_phi - ref_cyl_phi)
    scale_cyl_phi = abs(ref_cyl_phi)
    rel_cyl_phi = diff_cyl_phi / scale_cyl_phi
    thresh_cyl_phi = 1.0e-11

    rows.append({
        "shape": "円柱",
        "condition": "内部点 z_t = 0.5 (R=2, H=4, rho=1500)",
        "observable": "Phi",
        "unit": "kg/m",
        "method": "1D quad",
        "quad_cond": "epsabs=1e-12, epsrel=1e-11",
        "scale": scale_cyl_phi,
        "scale_str": f"{scale_cyl_phi:.4e}",
        "analytic": calc_cyl_phi,
        "analytic_str": f"{calc_cyl_phi:+.16e}",
        "reference": ref_cyl_phi,
        "reference_str": f"{ref_cyl_phi:+.16e}",
        "abs_diff": diff_cyl_phi,
        "abs_diff_str": f"{diff_cyl_phi:.3e}",
        "rel_err": rel_cyl_phi,
        "rel_err_str": f"{rel_cyl_phi:.2e}",
        "threshold": thresh_cyl_phi,
        "threshold_str": f"{thresh_cyl_phi:.1e}",
        "status": "PASS" if rel_cyl_phi <= thresh_cyl_phi else "FAIL",
    })

    # 4b. Cylinder az
    def integrand_cyl_az(z_prime: float) -> float:
        u = z_prime - 0.5
        return float(np.sign(u) - u / np.sqrt(cyl_r**2 + u**2))

    val_cyl_az, _ = integrate.quad(integrand_cyl_az, -2.0, 2.0, points=[0.5], epsabs=1e-12, epsrel=1e-11)
    ref_cyl_az = float(2.0 * np.pi * cyl_rho * val_cyl_az)
    diff_cyl_az = abs(calc_cyl_az - ref_cyl_az)
    scale_cyl_az = abs(ref_cyl_az)
    rel_cyl_az = diff_cyl_az / scale_cyl_az
    thresh_cyl_az = 1.0e-11

    rows.append({
        "shape": "円柱",
        "condition": "内部点 z_t = 0.5 (R=2, H=4, rho=1500)",
        "observable": "az",
        "unit": "kg/m^2",
        "method": "1D quad (points=[0.5])",
        "quad_cond": "epsabs=1e-12, epsrel=1e-11",
        "scale": scale_cyl_az,
        "scale_str": f"{scale_cyl_az:.4e}",
        "analytic": calc_cyl_az,
        "analytic_str": f"{calc_cyl_az:+.16e}",
        "reference": ref_cyl_az,
        "reference_str": f"{ref_cyl_az:+.16e}",
        "abs_diff": diff_cyl_az,
        "abs_diff_str": f"{diff_cyl_az:.3e}",
        "rel_err": rel_cyl_az,
        "rel_err_str": f"{rel_cyl_az:.2e}",
        "threshold": thresh_cyl_az,
        "threshold_str": f"{thresh_cyl_az:.1e}",
        "status": "PASS" if rel_cyl_az <= thresh_cyl_az else "FAIL",
    })

    # 4c. Cylinder Gzz
    def integrand_cyl_gzz(z_prime: float) -> float:
        u = z_prime - 0.5
        return float(cyl_r**2 / ((cyl_r**2 + u**2)**1.5))

    val_cyl_gzz, _ = integrate.quad(integrand_cyl_gzz, -2.0, 2.0, epsabs=1e-12, epsrel=1e-11)
    ref_cyl_gzz = float(-4.0 * np.pi * cyl_rho + 2.0 * np.pi * cyl_rho * val_cyl_gzz)
    diff_cyl_gzz = abs(calc_cyl_gzz - ref_cyl_gzz)
    scale_cyl_gzz = abs(ref_cyl_gzz)
    rel_cyl_gzz = diff_cyl_gzz / scale_cyl_gzz
    thresh_cyl_gzz = 1.0e-11

    rows.append({
        "shape": "円柱",
        "condition": "内部点 z_t = 0.5 (R=2, H=4, rho=1500)",
        "observable": "Gamma_zz",
        "unit": "kg/m^3",
        "method": "1D quad (特異分離)",
        "quad_cond": "epsabs=1e-12, epsrel=1e-11",
        "scale": scale_cyl_gzz,
        "scale_str": f"{scale_cyl_gzz:.4e}",
        "analytic": calc_cyl_gzz,
        "analytic_str": f"{calc_cyl_gzz:+.16e}",
        "reference": ref_cyl_gzz,
        "reference_str": f"{ref_cyl_gzz:+.16e}",
        "abs_diff": diff_cyl_gzz,
        "abs_diff_str": f"{diff_cyl_gzz:.3e}",
        "rel_err": rel_cyl_gzz,
        "rel_err_str": f"{rel_cyl_gzz:.2e}",
        "threshold": thresh_cyl_gzz,
        "threshold_str": f"{thresh_cyl_gzz:.1e}",
        "status": "PASS" if rel_cyl_gzz <= thresh_cyl_gzz else "FAIL",
    })

    # 4d. Cylinder Gxx (Poisson constrained)
    ref_cyl_gxx = float((-4.0 * np.pi * cyl_rho - ref_cyl_gzz) / 2.0)
    diff_cyl_gxx = abs(calc_cyl_gxx - ref_cyl_gxx)
    scale_cyl_gxx = abs(ref_cyl_gxx)
    rel_cyl_gxx = diff_cyl_gxx / scale_cyl_gxx if scale_cyl_gxx > 0.0 else diff_cyl_gxx
    thresh_cyl_gxx = 1.0e-12

    rows.append({
        "shape": "円柱",
        "condition": "内部点 z_t = 0.5 (R=2, H=4, rho=1500)",
        "observable": "Gamma_xx",
        "unit": "kg/m^3",
        "method": "Poisson 拘束",
        "quad_cond": "厳密拘束式",
        "scale": scale_cyl_gxx,
        "scale_str": f"{scale_cyl_gxx:.4e}",
        "analytic": calc_cyl_gxx,
        "analytic_str": f"{calc_cyl_gxx:+.16e}",
        "reference": ref_cyl_gxx,
        "reference_str": f"{ref_cyl_gxx:+.16e}",
        "abs_diff": diff_cyl_gxx,
        "abs_diff_str": f"{diff_cyl_gxx:.3e}",
        "rel_err": rel_cyl_gxx,
        "rel_err_str": f"{rel_cyl_gxx:.2e}",
        "threshold": thresh_cyl_gxx,
        "threshold_str": f"{thresh_cyl_gxx:.1e}",
        "status": "PASS" if rel_cyl_gxx <= thresh_cyl_gxx else "FAIL",
    })

    # -------------------------------------------------------------------------
    # 5. Cylinder Exterior Point z_t = 3.0 (2D transverse gradient)
    # -------------------------------------------------------------------------
    cyl_ext_target = np.array([[0.0, 0.0, 3.0]])
    obs_cyl_ext = cylinder_axis_observables_g_free(center, cyl_r, cyl_h, cyl_rho, cyl_ext_target)
    calc_cyl_ext_gxx = float(obs_cyl_ext.gradient[0, 0, 0])

    def integrand_cyl_ext_gxx(z_prime: float, r_prime: float) -> float:
        u = z_prime - 3.0
        r_dist = float(np.sqrt(r_prime**2 + u**2))
        return float(r_prime * (3.0 * (r_prime**2 / 2.0) - (r_dist**2)) / (r_dist**5))

    val_cyl_ext_gxx, _ = integrate.dblquad(
        integrand_cyl_ext_gxx, 0.0, cyl_r, -2.0, 2.0, epsabs=1e-12, epsrel=1e-11
    )
    ref_cyl_ext_gxx = float(2.0 * np.pi * cyl_rho * val_cyl_ext_gxx)
    diff_cyl_ext_gxx = abs(calc_cyl_ext_gxx - ref_cyl_ext_gxx)
    scale_cyl_ext_gxx = abs(ref_cyl_ext_gxx)
    rel_cyl_ext_gxx = diff_cyl_ext_gxx / scale_cyl_ext_gxx
    thresh_cyl_ext_gxx = 1.0e-8

    rows.append({
        "shape": "円柱",
        "condition": "外部点 z_t = 3.0 (R=2, H=4, rho=1500)",
        "observable": "Gamma_xx (横勾配)",
        "unit": "kg/m^3",
        "method": "2D dblquad",
        "quad_cond": "epsabs=1e-12, epsrel=1e-11",
        "scale": scale_cyl_ext_gxx,
        "scale_str": f"{scale_cyl_ext_gxx:.4e}",
        "analytic": calc_cyl_ext_gxx,
        "analytic_str": f"{calc_cyl_ext_gxx:+.16e}",
        "reference": ref_cyl_ext_gxx,
        "reference_str": f"{ref_cyl_ext_gxx:+.16e}",
        "abs_diff": diff_cyl_ext_gxx,
        "abs_diff_str": f"{diff_cyl_ext_gxx:.3e}",
        "rel_err": rel_cyl_ext_gxx,
        "rel_err_str": f"{rel_cyl_ext_gxx:.2e}",
        "threshold": thresh_cyl_ext_gxx,
        "threshold_str": f"{thresh_cyl_ext_gxx:.1e}",
        "status": "PASS" if rel_cyl_ext_gxx <= thresh_cyl_ext_gxx else "FAIL",
    })

    # Cylinder trace check at z=0, 1
    for z_chk in [0.0, 1.0]:
        chk_obs = cylinder_axis_observables_g_free(center, cyl_r, cyl_h, cyl_rho, np.array([[0.0, 0.0, z_chk]]))
        tr = float(np.trace(chk_obs.gradient[0]))
        exp_tr = float(-4.0 * np.pi * cyl_rho)
        diff_tr = abs(tr - exp_tr)
        scale_tr = abs(exp_tr)
        rel_tr = diff_tr / scale_tr
        rows.append({
            "shape": "円柱",
            "condition": f"内部点 z_t = {z_chk:.1f} (R=2, H=4, rho=1500)",
            "observable": "tr(Gamma) + 4*pi*rho",
            "unit": "kg/m^3",
            "method": "代数 Poisson 則",
            "quad_cond": "厳密恒等式",
            "scale": scale_tr,
            "scale_str": f"4*pi*rho = {scale_tr:.4e}",
            "analytic": tr,
            "analytic_str": f"{tr:+.16e}",
            "reference": exp_tr,
            "reference_str": f"{exp_tr:+.16e}",
            "abs_diff": diff_tr,
            "abs_diff_str": f"{diff_tr:.3e}",
            "rel_err": rel_tr,
            "rel_err_str": f"{rel_tr:.2e}",
            "threshold": 1.0e-11,
            "threshold_str": "1.0e-11",
            "status": "PASS" if rel_tr <= 1.0e-11 else "FAIL",
        })

    # -------------------------------------------------------------------------
    # 6. Spherical Shell P-Wave Cavity Decomposition
    # -------------------------------------------------------------------------
    p_a = 10.0
    p_r = 40.0
    p_k = 0.02
    p_rho = 2000.0
    p_xi = 1.0e-5
    p_dir = [0.0, 0.0, 1.0]

    comps = spherical_shell_p_acceleration_components_g_free(
        inner_radius_m=p_a, outer_radius_m=p_r, wavenumber_rad_m=p_k,
        density_kg_m3=p_rho, displacement_m=p_xi, direction=p_dir
    )

    # 6a. Bulk
    def integrand_p_bulk(theta: float, r: float) -> float:
        return float(np.sin(theta) * np.cos(theta) * np.sin(p_k * r * np.cos(theta)))

    val_p_bulk, _ = integrate.dblquad(integrand_p_bulk, p_a, p_r, 0.0, np.pi, epsabs=1e-12, epsrel=1e-10)
    ref_p_bulk = float(2.0 * np.pi * p_rho * p_k * p_xi * val_p_bulk)
    calc_p_bulk = float(comps.bulk[0, 2])
    diff_p_bulk = abs(calc_p_bulk - ref_p_bulk)
    scale_p_bulk = abs(ref_p_bulk)
    rel_p_bulk = diff_p_bulk / scale_p_bulk
    thresh_p_comps = 1.0e-7

    rows.append({
        "shape": "球殻 P波",
        "condition": "a=10, R=40, k=0.02 (rho=2000, xi=1e-5)",
        "observable": "Bulk 寄与",
        "unit": "kg/m^2",
        "method": "2D dblquad",
        "quad_cond": "epsabs=1e-12, epsrel=1e-10",
        "scale": scale_p_bulk,
        "scale_str": f"{scale_p_bulk:.5e}",
        "analytic": calc_p_bulk,
        "analytic_str": f"{calc_p_bulk:+.16e}",
        "reference": ref_p_bulk,
        "reference_str": f"{ref_p_bulk:+.16e}",
        "abs_diff": diff_p_bulk,
        "abs_diff_str": f"{diff_p_bulk:.3e}",
        "rel_err": rel_p_bulk,
        "rel_err_str": f"{rel_p_bulk:.2e}",
        "threshold": thresh_p_comps,
        "threshold_str": f"{thresh_p_comps:.1e}",
        "status": "PASS" if rel_p_bulk <= thresh_p_comps else "FAIL",
    })

    # 6b. Inner wall
    def integrand_p_inner(theta: float) -> float:
        return float(np.sin(theta) * (np.cos(theta)**2) * np.cos(p_k * p_a * np.cos(theta)))

    val_p_inner, _ = integrate.quad(integrand_p_inner, 0.0, np.pi, epsabs=1e-12, epsrel=1e-10)
    ref_p_inner = float(-2.0 * np.pi * p_rho * p_xi * val_p_inner)
    calc_p_inner = float(comps.inner_wall[0, 2])
    diff_p_inner = abs(calc_p_inner - ref_p_inner)
    scale_p_inner = abs(ref_p_inner)
    rel_p_inner = diff_p_inner / scale_p_inner

    rows.append({
        "shape": "球殻 P波",
        "condition": "a=10, R=40, k=0.02 (rho=2000, xi=1e-5)",
        "observable": "Inner wall 寄与",
        "unit": "kg/m^2",
        "method": "1D quad (内表面)",
        "quad_cond": "epsabs=1e-12, epsrel=1e-10",
        "scale": scale_p_inner,
        "scale_str": f"{scale_p_inner:.5e}",
        "analytic": calc_p_inner,
        "analytic_str": f"{calc_p_inner:+.16e}",
        "reference": ref_p_inner,
        "reference_str": f"{ref_p_inner:+.16e}",
        "abs_diff": diff_p_inner,
        "abs_diff_str": f"{diff_p_inner:.3e}",
        "rel_err": rel_p_inner,
        "rel_err_str": f"{rel_p_inner:.2e}",
        "threshold": thresh_p_comps,
        "threshold_str": f"{thresh_p_comps:.1e}",
        "status": "PASS" if rel_p_inner <= thresh_p_comps else "FAIL",
    })

    # 6c. Outer boundary
    def integrand_p_outer(theta: float) -> float:
        return float(np.sin(theta) * (np.cos(theta)**2) * np.cos(p_k * p_r * np.cos(theta)))

    val_p_outer, _ = integrate.quad(integrand_p_outer, 0.0, np.pi, epsabs=1e-12, epsrel=1e-10)
    ref_p_outer = float(2.0 * np.pi * p_rho * p_xi * val_p_outer)
    calc_p_outer = float(comps.outer_boundary[0, 2])
    diff_p_outer = abs(calc_p_outer - ref_p_outer)
    scale_p_outer = abs(ref_p_outer)
    rel_p_outer = diff_p_outer / scale_p_outer

    rows.append({
        "shape": "球殻 P波",
        "condition": "a=10, R=40, k=0.02 (rho=2000, xi=1e-5)",
        "observable": "Outer boundary 寄与",
        "unit": "kg/m^2",
        "method": "1D quad (外表面)",
        "quad_cond": "epsabs=1e-12, epsrel=1e-10",
        "scale": scale_p_outer,
        "scale_str": f"{scale_p_outer:.5e}",
        "analytic": calc_p_outer,
        "analytic_str": f"{calc_p_outer:+.16e}",
        "reference": ref_p_outer,
        "reference_str": f"{ref_p_outer:+.16e}",
        "abs_diff": diff_p_outer,
        "abs_diff_str": f"{diff_p_outer:.3e}",
        "rel_err": rel_p_outer,
        "rel_err_str": f"{rel_p_outer:.2e}",
        "threshold": thresh_p_comps,
        "threshold_str": f"{thresh_p_comps:.1e}",
        "status": "PASS" if rel_p_outer <= thresh_p_comps else "FAIL",
    })

    # 6d. Total vs 3-component quadrature sum
    ref_p_total_3sum = ref_p_bulk + ref_p_inner + ref_p_outer
    calc_p_total = float(comps.total[0, 2])
    diff_p_total_3sum = abs(calc_p_total - ref_p_total_3sum)
    scale_p_total_3sum = abs(ref_p_total_3sum)
    rel_p_total_3sum = diff_p_total_3sum / scale_p_total_3sum
    thresh_p_total = 1.0e-10

    rows.append({
        "shape": "球殻 P波",
        "condition": "a=10, R=40, k=0.02 (rho=2000, xi=1e-5)",
        "observable": "Total (3成分求積和比較)",
        "unit": "kg/m^2",
        "method": "3成分独立求積の代数和",
        "quad_cond": "各成分 epsabs=1e-12, epsrel=1e-10",
        "scale": scale_p_total_3sum,
        "scale_str": f"{scale_p_total_3sum:.5e}",
        "analytic": calc_p_total,
        "analytic_str": f"{calc_p_total:+.16e}",
        "reference": ref_p_total_3sum,
        "reference_str": f"{ref_p_total_3sum:+.16e}",
        "abs_diff": diff_p_total_3sum,
        "abs_diff_str": f"{diff_p_total_3sum:.3e}",
        "rel_err": rel_p_total_3sum,
        "rel_err_str": f"{rel_p_total_3sum:.2e}",
        "threshold": thresh_p_total,
        "threshold_str": f"{thresh_p_total:.1e}",
        "status": "PASS" if rel_p_total_3sum <= thresh_p_total else "FAIL",
    })

    # 6e. Total vs direct dipole quadrature (ANNA v4 Eq. 17)
    def integrand_p_dipole(theta: float, r: float) -> float:
        return float((1.0 - 3.0 * np.cos(theta)**2) * np.cos(p_k * r * np.cos(theta)) * np.sin(theta) / r)

    val_p_dipole, _ = integrate.dblquad(integrand_p_dipole, p_a, p_r, 0.0, np.pi, epsabs=1e-13, epsrel=1e-12)
    ref_p_total_dipole = float(2.0 * np.pi * p_rho * p_xi * val_p_dipole)
    diff_p_total_dipole = abs(calc_p_total - ref_p_total_dipole)
    scale_p_total_dipole = abs(ref_p_total_dipole)
    rel_p_total_dipole = diff_p_total_dipole / scale_p_total_dipole

    rows.append({
        "shape": "球殻 P波",
        "condition": "a=10, R=40, k=0.02 (rho=2000, xi=1e-5)",
        "observable": "Total (direct dipole 積分比較)",
        "unit": "kg/m^2",
        "method": "ANNA v4 式(17) 2D dblquad",
        "quad_cond": "epsabs=1e-13, epsrel=1e-12",
        "scale": scale_p_total_dipole,
        "scale_str": f"{scale_p_total_dipole:.5e}",
        "analytic": calc_p_total,
        "analytic_str": f"{calc_p_total:+.16e}",
        "reference": ref_p_total_dipole,
        "reference_str": f"{ref_p_total_dipole:+.16e}",
        "abs_diff": diff_p_total_dipole,
        "abs_diff_str": f"{diff_p_total_dipole:.3e}",
        "rel_err": rel_p_total_dipole,
        "rel_err_str": f"{rel_p_total_dipole:.2e}",
        "threshold": thresh_p_total,
        "threshold_str": f"{thresh_p_total:.1e}",
        "status": "PASS" if rel_p_total_dipole <= thresh_p_total else "FAIL",
    })

    # 6f. Zero density control
    comps_zero = spherical_shell_p_acceleration_components_g_free(
        inner_radius_m=p_a, outer_radius_m=p_r, wavenumber_rad_m=p_k,
        density_kg_m3=0.0, displacement_m=p_xi, direction=p_dir
    )
    is_bit_zero = bool(
        np.all(comps_zero.bulk == 0.0) and
        np.all(comps_zero.inner_wall == 0.0) and
        np.all(comps_zero.outer_boundary == 0.0) and
        np.all(comps_zero.total == 0.0)
    )
    rows.append({
        "shape": "球殻 P波",
        "condition": "rho = 0.0 (ゼロ密度対照)",
        "observable": "全4成分ゼロ一致",
        "unit": "kg/m^2",
        "method": "恒等式検査",
        "quad_cond": "厳密代数対照",
        "scale": 0.0,
        "scale_str": "0.0",
        "analytic": 0.0,
        "analytic_str": "0.0000000000000000e+00",
        "reference": 0.0,
        "reference_str": "0.0000000000000000e+00",
        "abs_diff": 0.0,
        "abs_diff_str": "0.000e+00",
        "rel_err": 0.0,
        "rel_err_str": "0.00e+00 (bit一致)",
        "threshold": 0.0,
        "threshold_str": "厳密一致",
        "status": "PASS" if is_bit_zero else "FAIL",
    })

    return rows


def run_table_4_2() -> list[dict[str, Any]]:
    """Run representative cases convergence study for §4.2."""
    if integrate is None:
        raise RuntimeError("scipy is required for numerical quadrature benchmarks. Install scipy to run §4 benchmarks.")
    cases = [
        ("標準球殻", 10.0, 40.0, 0.02, 2000.0, 1e-5),
        ("薄殻", 10.0, 10.5, 0.02, 2000.0, 1e-5),
        ("小非ゼロ波数", 10.0, 40.0, 1e-5, 2000.0, 1e-5),
    ]
    tols = [1e-6, 1e-9, 1e-12]
    rows: list[dict[str, Any]] = []

    for name, a, R, k, rho, xi in cases:
        comps = spherical_shell_p_acceleration_components_g_free(
            inner_radius_m=a, outer_radius_m=R, wavenumber_rad_m=k,
            density_kg_m3=rho, displacement_m=xi, direction=[0, 0, 1]
        )
        tot_analytic = float(comps.total[0, 2])

        def integrand_dipole(theta: float, r: float, k: float = k) -> float:
            return float((1.0 - 3.0 * np.cos(theta)**2) * np.cos(k * r * np.cos(theta)) * np.sin(theta) / r)

        k_str = f"{k:.0e}" if k < 0.01 else f"{k}"
        for tol in tols:
            val_dipole, _ = integrate.dblquad(
                integrand_dipole, a, R, 0.0, np.pi,
                epsabs=tol * 1e-2, epsrel=tol
            )
            tot_dipole = float(2.0 * np.pi * rho * xi * val_dipole)
            err = abs(tot_dipole - tot_analytic)
            rel_err = err / abs(tot_analytic) if abs(tot_analytic) > 0.0 else err

            rows.append({
                "case": name,
                "condition": f"a={a}, R={R}, k={k_str}, rho={rho}, xi={xi:.0e}",
                "unit": "kg/m^2",
                "analytic": tot_analytic,
                "analytic_str": f"{tot_analytic:+.14e}",
                "epsrel": f"{tol:.0e}",
                "quadrature": tot_dipole,
                "quadrature_str": f"{tot_dipole:+.14e}",
                "abs_err": err,
                "abs_err_str": f"{err:.2e}",
                "rel_err": rel_err,
                "rel_err_str": f"{rel_err:.2e}",
            })

    return rows


def format_text_tables(rows_4_1: list[dict[str, Any]], rows_4_2: list[dict[str, Any]]) -> str:
    """Format human-readable summary text table."""
    lines: list[str] = []
    lines.append("=" * 135)
    lines.append("gragra Stage 1 Analytic Verification Report Tables (Reproduction)")
    lines.append("=" * 135)

    lines.append(f"\n[Section 4.1] 独立求積比較の詳細成績表 ({len(rows_4_1)} 項目)")
    lines.append("-" * 135)
    hdr_fmt = "{:<8} | {:<28} | {:<12} | {:<8} | {:<24} | {:<24} | {:<10} | {:<6}"
    lines.append(hdr_fmt.format("対象形状", "評価条件", "検証量", "単位", "解析値 (Analytic)", "独立参照値 (Reference)", "実測相対誤差", "判定"))
    lines.append("-" * 135)
    for r in rows_4_1:
        lines.append(hdr_fmt.format(
            r["shape"],
            r["condition"][:28],
            r["observable"][:12],
            r["unit"][:8],
            r["analytic_str"][:24],
            r["reference_str"][:24],
            r["rel_err_str"][:10],
            r["status"],
        ))

    lines.append(f"\n[Section 4.2] 代表ケースにおける求積条件と参照値収束の記録 ({len(rows_4_2)} 観測点記録 ※判定対象外)")
    lines.append("-" * 135)
    c_fmt = "{:<12} | {:<8} | {:<24} | {:<24} | {:<10} | {:<10}"
    lines.append(c_fmt.format("ケース", "epsrel", "解析解 Total", "求積値", "絶対誤差", "相対誤差"))
    lines.append("-" * 135)
    for r in rows_4_2:
        lines.append(c_fmt.format(
            r["case"],
            r["epsrel"],
            r["analytic_str"],
            r["quadrature_str"],
            r["abs_err_str"],
            r["rel_err_str"],
        ))
    lines.append("=" * 135)

    exit_code, summary_msg = evaluate_status(rows_4_1)
    lines.append(summary_msg)
    return "\n".join(lines)


def format_markdown_tables(rows_4_1: list[dict[str, Any]], rows_4_2: list[dict[str, Any]]) -> str:
    """Format comprehensive detailed Markdown tables containing all numerical fields."""
    lines: list[str] = []
    lines.append(f"\n### 4.1 独立求積比較の詳細成績表 ({len(rows_4_1)} 項目)\n")
    lines.append("| 対象形状 | 評価点 / 条件 | 検証量 | 単位 | 解析値 (Analytic) | 独立参照値 (Reference) | 規格化分母 (Scale) | 絶対差 (Abs Diff) | 規格化相対誤差 (Rel Err) | 受入閾値 | 判定 |")
    lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in rows_4_1:
        lines.append(
            f"| **{r['shape']}** | {r['condition']} | `{r['observable']}` | `{r['unit']}` "
            f"| `{r['analytic_str']}` | `{r['reference_str']}` | `{r['scale_str']}` "
            f"| `{r['abs_diff_str']}` | **`{r['rel_err_str']}`** | `{r['threshold_str']}` | {r['status']} |"
        )

    lines.append(f"\n### 4.2 代表ケースにおける求積条件と参照値収束の記録 ({len(rows_4_2)} 観測点記録 ※判定対象外)\n")
    lines.append("| ケース | 幾何・媒質条件 | 単位 | 解析解 Total | 求積公差 (epsrel) | 求積値 | 実測絶対誤差 | 実測相対誤差 |")
    lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in rows_4_2:
        lines.append(
            f"| **{r['case']}** | {r['condition']} | `{r['unit']}` | `{r['analytic_str']}` "
            f"| `{r['epsrel']}` | `{r['quadrature_str']}` | `{r['abs_err_str']}` | **`{r['rel_err_str']}`** |"
        )

    exit_code, summary_msg = evaluate_status(rows_4_1)
    lines.append(f"\n> **検証集計**: {summary_msg}\n")
    return "\n".join(lines)


def format_json_report(rows_4_1: list[dict[str, Any]], rows_4_2: list[dict[str, Any]]) -> str:
    """Format machine-readable structured JSON report."""
    total = len(rows_4_1)
    passed = sum(1 for r in rows_4_1 if r["status"] == "PASS")
    failed = total - passed

    data = {
        "summary": {
            "total_items": total,
            "passed_items": passed,
            "failed_items": failed,
            "overall_status": "PASS" if failed == 0 else "FAIL",
            "section_4_2_observational_records": len(rows_4_2),
        },
        "table_4_1": rows_4_1,
        "table_4_2_convergence": rows_4_2,
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def evaluate_status(rows_4_1: list[dict[str, Any]]) -> tuple[int, str]:
    """Evaluate overall status across §4.1 rows.

    Returns:
      (exit_code, summary_message)
    """
    total = len(rows_4_1)
    passed = sum(1 for r in rows_4_1 if r.get("status") == "PASS")
    failed = total - passed

    if failed > 0:
        msg = f"[FAILURE] {failed}/{total} verification item(s) failed in §4.1! (§4.2 は観測記録のため判定対象外)"
        return 1, msg
    msg = f"[SUCCESS] All {total}/{total} verification items in §4.1 passed successfully! (§4.2 は観測記録のため判定対象外)"
    return 0, msg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce Stage 1 verification report tables.")
    parser.add_argument("--markdown", action="store_true", help="Print detailed report tables in Markdown format.")
    parser.add_argument("--json", action="store_true", help="Print structured report in JSON format.")
    args = parser.parse_args(argv)

    rows_4_1 = run_table_4_1()
    rows_4_2 = run_table_4_2()

    exit_code, _ = evaluate_status(rows_4_1)

    if args.json:
        print(format_json_report(rows_4_1, rows_4_2))
    elif args.markdown:
        print(format_markdown_tables(rows_4_1, rows_4_2))
    else:
        print(format_text_tables(rows_4_1, rows_4_2))

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
