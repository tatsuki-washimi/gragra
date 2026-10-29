"""Example 15: Analytic stage 1 benchmarks and SI observables.

Execution modes:
- Default mode (no `--report`): Demonstrates and verifies the 4 standard benchmark scenarios:
  1. Uniform sphere: G-free and SI observable bundles.
  2. Uniform cuboid: Closed-form potential, acceleration, and full 3x3 gradient tensor.
  3. Uniform cylinder on-axis: Axis-aligned potential, acceleration, and gradient satisfying Poisson's equation.
  4. Spherical shell P-wave cavity: Closed-form 4-component acceleration decomposition
     (bulk, inner_wall, outer_boundary, total) and infinite difference relation.
  All 4 scenarios include assert checks confirming physical and mathematical consistency.
- Switch mode (`--report`): Switches execution to invoke `scripts/reproduce_analytic_stage1_report.py`
  to generate and display the full verification report tables (§4.1 and §4.2), supporting
  `--markdown` and `--json` formatting options.
"""

import numpy as np

from gragra.constants import G_SI
from gragra.analytic import (
    cuboid_observables_g_free,
    cylinder_axis_observables_g_free,
    spherical_shell_p_acceleration_components_g_free,
    spherical_shell_p_infinite_difference_g_free,
    uniform_sphere_observables_g_free,
)
from gragra.observables import (
    cuboid_observables,
    cylinder_axis_observables,
    spherical_shell_p_acceleration_components,
    spherical_shell_p_infinite_difference,
    to_gravity_observables,
    to_spherical_shell_p_components,
    uniform_sphere_observables,
)


def benchmark_sphere() -> None:
    print("=== Scenario 1: Uniform Sphere ===")
    center = [0.0, 0.0, 0.0]
    radius_m = 1.0
    density_kg_m3 = 2500.0
    targets = np.array([
        [0.0, 0.0, 0.5],   # Inside
        [0.0, 0.0, 2.0],   # Outside
    ])

    gfree = uniform_sphere_observables_g_free(center, radius_m, density_kg_m3, targets)
    si = uniform_sphere_observables(center, radius_m, density_kg_m3, targets)

    # Verification: SI is strictly G_SI * G-free
    np.testing.assert_allclose(si.potential, G_SI * gfree.potential)
    np.testing.assert_allclose(si.acceleration, G_SI * gfree.acceleration)
    np.testing.assert_allclose(si.gradient, G_SI * gfree.gradient)

    # Inside Poisson check: trace(Gamma) = -4*pi*G*rho
    trace_inside = np.trace(si.gradient[0])
    expected_trace_inside = -4.0 * np.pi * G_SI * density_kg_m3
    assert np.isclose(trace_inside, expected_trace_inside, rtol=1e-14)

    # Outside Laplace check: trace(Gamma) = 0
    trace_outside = np.trace(si.gradient[1])
    assert np.isclose(trace_outside, 0.0, atol=1e-15)

    print(f"  Sphere inside trace(Gamma):  {trace_inside:.6e} s^-2 (expected {expected_trace_inside:.6e})")
    print(f"  Sphere outside trace(Gamma): {trace_outside:.6e} s^-2 (expected 0)")
    print("  [PASS] Sphere checks passed.\n")


def benchmark_cuboid() -> None:
    print("=== Scenario 2: Uniform Cuboid (Closed Form) ===")
    center = [0.0, 0.0, 0.0]
    lengths = [2.0, 3.0, 4.0]
    density_kg_m3 = 2700.0
    targets = np.array([
        [0.2, 0.3, 0.4],    # Inside
        [5.0, 0.0, 0.0],    # Outside along x
        [0.0, 5.0, 0.0],    # Outside along y
        [0.0, 0.0, 5.0],    # Outside along z
    ])

    gfree = cuboid_observables_g_free(center, lengths, density_kg_m3, targets)
    si = cuboid_observables(center, lengths, density_kg_m3, targets)

    # Conversion consistency
    si_from_conv = to_gravity_observables(gfree)
    np.testing.assert_allclose(si.potential, si_from_conv.potential)
    np.testing.assert_allclose(si.acceleration, si_from_conv.acceleration)
    np.testing.assert_allclose(si.gradient, si_from_conv.gradient)

    # Inside Poisson check: trace(Gamma) = -4*pi*G*rho
    trace_inside = np.trace(si.gradient[0])
    expected_trace_inside = -4.0 * np.pi * G_SI * density_kg_m3
    assert np.isclose(trace_inside, expected_trace_inside, rtol=1e-12)

    # Outside Laplace check: trace(Gamma) = 0 for outside targets
    for i in range(1, 4):
        trace_out = np.trace(si.gradient[i])
        assert np.isclose(trace_out, 0.0, atol=1e-14)

    # Symmetry of gradient tensor: Gamma_ij = Gamma_ji
    for i in range(len(targets)):
        grad = si.gradient[i]
        np.testing.assert_allclose(grad, grad.T, atol=1e-15)

    print(f"  Cuboid inside trace(Gamma):  {trace_inside:.6e} s^-2 (expected {expected_trace_inside:.6e})")
    print(f"  Cuboid outside trace max:    {max(abs(np.trace(si.gradient[i])) for i in range(1, 4)):.6e} s^-2")
    print("  [PASS] Cuboid checks passed.\n")


def benchmark_cylinder() -> None:
    print("=== Scenario 3: Uniform Cylinder On-Axis ===")
    center = [0.0, 0.0, 0.0]
    radius_m = 1.0
    height_m = 4.0
    density_kg_m3 = 2600.0
    targets = np.array([
        [0.0, 0.0, 0.0],    # Center (inside)
        [0.0, 0.0, 1.0],    # Inside off-center
        [0.0, 0.0, 3.0],    # Outside along z-axis
    ])

    gfree = cylinder_axis_observables_g_free(center, radius_m, height_m, density_kg_m3, targets)
    si = cylinder_axis_observables(center, radius_m, height_m, density_kg_m3, targets)

    np.testing.assert_allclose(si.potential, G_SI * gfree.potential)
    np.testing.assert_allclose(si.acceleration, G_SI * gfree.acceleration)
    np.testing.assert_allclose(si.gradient, G_SI * gfree.gradient)

    # In center (z=0), by symmetry, acceleration = 0
    np.testing.assert_allclose(si.acceleration[0], [0.0, 0.0, 0.0], atol=1e-15)

    # Poisson check at target 0 (inside)
    trace_inside = np.trace(si.gradient[0])
    expected_trace_inside = -4.0 * np.pi * G_SI * density_kg_m3
    assert np.isclose(trace_inside, expected_trace_inside, rtol=1e-14)

    # Laplace check at target 2 (outside)
    trace_outside = np.trace(si.gradient[2])
    assert np.isclose(trace_outside, 0.0, atol=1e-15)

    print(f"  Cylinder center acc:        {np.linalg.norm(si.acceleration[0]):.6e} m/s^2 (expected 0)")
    print(f"  Cylinder inside trace:      {trace_inside:.6e} s^-2 (expected {expected_trace_inside:.6e})")
    print(f"  Cylinder outside trace:     {trace_outside:.6e} s^-2 (expected 0)")
    print("  [PASS] Cylinder on-axis checks passed.\n")


def benchmark_spherical_shell_p() -> None:
    print("=== Scenario 4: Spherical Shell P-Wave Cavity Decomposition ===")
    a = 10.0      # Inner cavity radius
    R = 100.0     # Outer boundary radius
    k = 0.02      # Wavenumber rad/m
    rho = 2800.0  # Density kg/m^3
    xi = 1e-6     # Displacement 1 um
    direction = [1.0, 0.0, 0.0]

    gfree_comp = spherical_shell_p_acceleration_components_g_free(
        a, R, k, rho, xi, direction
    )
    si_comp = spherical_shell_p_acceleration_components(
        a, R, k, rho, xi, direction
    )

    # Conversion consistency via to_spherical_shell_p_components
    si_from_conv = to_spherical_shell_p_components(gfree_comp)
    np.testing.assert_allclose(si_comp.total, si_from_conv.total)
    np.testing.assert_allclose(si_comp.bulk, si_from_conv.bulk)

    # Component sum identity: bulk + inner_wall + outer_boundary == total
    sum_comps = si_comp.bulk + si_comp.inner_wall + si_comp.outer_boundary
    np.testing.assert_allclose(sum_comps, si_comp.total, rtol=1e-14)

    # Infinite difference check
    si_diff = spherical_shell_p_infinite_difference(
        a, R, k, rho, xi, direction
    )
    # Expected: 8*pi*G*rho*(j1(kR)/(kR))*xi*e_hat
    np.testing.assert_allclose(si_diff, G_SI * spherical_shell_p_infinite_difference_g_free(
        a, R, k, rho, xi, direction
    ))

    print(f"  P-wave bulk accel:           {np.linalg.norm(si_comp.bulk):.6e} m/s^2")
    print(f"  P-wave inner wall accel:     {np.linalg.norm(si_comp.inner_wall):.6e} m/s^2")
    print(f"  P-wave outer boundary accel: {np.linalg.norm(si_comp.outer_boundary):.6e} m/s^2")
    print(f"  P-wave total accel:          {np.linalg.norm(si_comp.total):.6e} m/s^2")
    print(f"  P-wave infinite diff:        {np.linalg.norm(si_diff):.6e} m/s^2")
    print("  [PASS] Spherical shell P-wave cavity checks passed.\n")


def main() -> None:
    import argparse
    import subprocess
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Run Stage 1 analytic benchmarks and SI observables (or switch to validation report generation via --report)."
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Switch mode: generate and display full validation report tables (§4.1 / §4.2) instead of running the 4 benchmark scenarios.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print report tables in Markdown format (valid when --report is set).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print report tables in JSON format (valid when --report is set).",
    )
    args = parser.parse_args()

    if args.report:
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "reproduce_analytic_stage1_report.py"
        cmd = [sys.executable, str(script_path)]
        if args.markdown:
            cmd.append("--markdown")
        if args.json:
            cmd.append("--json")
        res = subprocess.run(cmd)
        sys.exit(res.returncode)

    print("Running gragra Stage 1 Analytic Benchmarks...\n")
    benchmark_sphere()
    benchmark_cuboid()
    benchmark_cylinder()
    benchmark_spherical_shell_p()
    print("All Stage 1 Analytic Benchmarks passed successfully!")


if __name__ == "__main__":
    import sys
    main()
