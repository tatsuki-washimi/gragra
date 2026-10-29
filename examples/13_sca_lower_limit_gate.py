"""Example 13: Phase 12-b Slice 4 -- SCA lower-limit consistency gate.

Design: docs/design/phase12b_sca_cavity_coupling.md Sec.3.

Purpose
-------
Cross-verify the Harms et al. 2022 Table 1 small-cavern-approximation (SCA)
cavern-wall coupling constants (`gragra.analytic.sca_wall_coupling_factor`,
Slice 1) against the exact static "filled body" internal-field closed forms
(`cuboid_filled_internal_field` / `cylinder_filled_axis_field`, Slice 2/3),
via the inversion principle:

    delta_a_wall_i = (d a_filled_i / d x_j) * xi_j

i.e. the cavern-wall coupling equals the *gradient* of the filled body's own
internal gravitational field (no extra sign flip -- see the module docstring
of `cavity_cuboid.py` for the derivation and the sign check against Table 1).

Scope note (IMPORTANT -- read before extending this gate)
-----------------------------------------------------------
Only the sphere/arbitrary and cube/3D-center Table 1 rows are cross-checked
here. The sphere calculation is a representative center check, not an
independent verification over every position described by ``arbitrary``. The
other three rows (cube/floor_center_2d,
half_sphere/floor_center_2d, elongated_cuboid_2to1to1/floor_center_2d) are
**not** cross-verified by this gate and are printed as explicitly unverified.
They remain source-value gates: Table 1 alone does not define their physical
boundary condition, direction, tensor component, or one-sided limit. Slice
2/3 provide closed-body static fields, so this example does not infer any
such relationship for the floor rows.

Also included: two sanity checks that are not Table 1 comparisons but confirm
the filled-body closed forms are self-consistent (Poisson's equation,
G-free: div(a_filled) = -4*pi*rho): a generic (non-cube, off-center) prism
interior point, and an axisymmetric decomposition of the cylinder's on-axis
gradient (using Poisson + axisymmetry to determine the transverse diagonal
component algebraically). The independent off-axis numerical cross-check for
the cylinder decomposition lives in
tests/test_analytic_cavity_cylinder.py::
test_offaxis_poisson_axisymmetry_matches_independent_integration (run via
`pytest -m slow`) rather than here: that cross-check integrates through the
filled cylinder's own 1/r^2 singularity at near-machine-precision tolerance
and costs several minutes, which is inappropriate for the examples/ smoke-run
(tests/test_examples.py) that executes on every default `pytest` invocation.

No files are written to disk; output is stdout only.
"""

import numpy as np

from gragra.analytic import (
    cuboid_filled_internal_field,
    cylinder_filled_axis_field,
    sca_wall_coupling_factor,
    spherical_cavity_wall,
)

RHO = 2500.0  # kg/m^3, arbitrary rock density
EPS = 1e-6  # central-difference step (m) for the cuboid gradient


def _numeric_gradient_cuboid(
    lengths: np.ndarray, point: np.ndarray, rho: float, eps: float = EPS
) -> np.ndarray:
    """Central-difference gradient d(a_filled_i)/d(x_j) at one point."""
    grad = np.zeros((3, 3))
    for j in range(3):
        dp = np.zeros(3)
        dp[j] = eps
        gp = cuboid_filled_internal_field(lengths, np.array([point + dp]), rho)[0]
        gm = cuboid_filled_internal_field(lengths, np.array([point - dp]), rho)[0]
        grad[:, j] = (gp - gm) / (2 * eps)
    return grad


def _check(name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    if detail:
        print(f"        {detail}")
    return passed


def row_sphere_arbitrary() -> bool:
    print("\n--- Row: sphere / arbitrary (Table 1; representative center check) ---")
    table_value = sca_wall_coupling_factor("sphere", "arbitrary")

    radius, xi, tiny_x = 1.0, 1e-6, 1e-6
    wall_p = spherical_cavity_wall(radius, [tiny_x / radius], RHO, xi, wave_type="P")[0]
    implied_factor = -float(wall_p) / (RHO * xi)

    return _check(
        "sphere/arbitrary: representative center x->0 check vs Table 1",
        np.isclose(implied_factor, table_value, rtol=1e-6),
        f"table={table_value:.6f}  implied(x={tiny_x:g})={implied_factor:.6f}",
    )


def row_cube_center_3d() -> bool:
    print("\n--- Row: cube / center_3d (whole-space, inversion principle) ---")
    a = 2.0
    cube = np.array([a, a, a])
    grad = _numeric_gradient_cuboid(cube, np.zeros(3), RHO)

    diag = np.diagonal(grad)
    off_diag_max = float(np.max(np.abs(grad - np.diag(diag))))
    isotropy_ok = _check(
        "cube/center_3d: gradient is isotropic (off-diag ~ 0, diag elements equal)",
        off_diag_max < 1e-4 * RHO and float(np.ptp(diag)) < 1e-4 * RHO,
        f"diag={diag}  max|off-diag|={off_diag_max:.3e}",
    )

    implied_factor = -float(np.mean(diag)) / RHO
    table_value = sca_wall_coupling_factor("cube", "center_3d")
    value_ok = _check(
        "cube/center_3d: -mean(diag(grad_a_filled))/rho vs Table 1",
        np.isclose(implied_factor, table_value, rtol=1e-6),
        f"table={table_value:.6f}  implied={implied_factor:.6f}",
    )

    trace = float(np.trace(grad))
    poisson_ok = _check(
        "cube/center_3d: Poisson trace(grad_a_filled) == -4*pi*rho",
        np.isclose(trace, -4.0 * np.pi * RHO, rtol=1e-6),
        f"trace={trace:.4f}  want={-4.0 * np.pi * RHO:.4f}",
    )
    return isotropy_ok and value_ok and poisson_ok


def row_poisson_generic_prism() -> bool:
    print("\n--- Sanity: generic prism interior Poisson (not a Table 1 row) ---")
    lengths = np.array([3.0, 2.0, 1.5])
    point = np.array([0.7, -0.4, 0.3])
    grad = _numeric_gradient_cuboid(lengths, point, RHO)
    trace = float(np.trace(grad))
    return _check(
        "generic prism interior: trace(grad_a_filled) == -4*pi*rho",
        np.isclose(trace, -4.0 * np.pi * RHO, rtol=1e-6),
        f"trace={trace:.4f}  want={-4.0 * np.pi * RHO:.4f}",
    )


def row_cylinder_poisson_axisymmetry() -> None:
    """Print the on-axis Poisson+axisymmetry-derived transverse gradient.

    Informational only (no PASS/FAIL gate): this is a fast, scipy-free
    algebraic decomposition. The independent off-axis numerical cross-check
    against this value lives in tests/test_analytic_cavity_cylinder.py::
    test_offaxis_poisson_axisymmetry_matches_independent_integration (see
    module docstring for why it is not run here).
    """
    print(
        "\n--- Sanity: cylinder axisymmetric Poisson decomposition "
        "(no Table 1 counterpart) ---"
    )
    radius, height = 1.2, 2.5
    z0 = 0.4  # interior on-axis point
    dz = 1e-5

    az_p = cylinder_filled_axis_field(radius, height, z0 + dz, RHO)
    az_m = cylinder_filled_axis_field(radius, height, z0 - dz, RHO)
    g_zz = float((az_p - az_m) / (2 * dz))

    # Poisson + axisymmetry (g_xx == g_yy on-axis): 2*g_xx + g_zz = -4*pi*rho.
    # This determines the transverse diagonal component algebraically,
    # without needing an off-axis closed form (Slice 3 is axis-only by scope).
    g_xx_from_poisson = (-4.0 * np.pi * RHO - g_zz) / 2.0
    print(f"        axial g_zz (finite difference)     = {g_zz:.4f}")
    print(f"        transverse g_xx=g_yy (Poisson+axisym) = {g_xx_from_poisson:.4f}")
    print(
        "        (independent off-axis numerical cross-check: "
        "pytest -m slow tests/test_analytic_cavity_cylinder.py)"
    )


def print_unverified_floor_rows() -> None:
    print("\n--- NOT cross-verified (Table 1 floor-label source values) ---")
    for shape, loc in (
        ("cube", "floor_center_2d"),
        ("half_sphere", "floor_center_2d"),
        ("elongated_cuboid_2to1to1", "floor_center_2d"),
    ):
        value = sca_wall_coupling_factor(shape, loc)
        print(
            f"    {shape}/{loc}: Table 1 value = {value} "
            "-- source-value gate only; no boundary condition, tensor, "
            "or image-method interpretation is inferred"
        )


def main() -> None:
    print("=== Phase 12-b Slice 4: SCA lower-limit consistency gate ===")
    print(
        f"rho = {RHO} kg/m^3 (G-free comparison; both sides are G-free "
        "by convention -- G_SI never enters)"
    )

    all_ok = True
    all_ok &= row_sphere_arbitrary()
    all_ok &= row_cube_center_3d()
    all_ok &= row_poisson_generic_prism()
    row_cylinder_poisson_axisymmetry()
    print_unverified_floor_rows()

    print("\n=== OVERALL:", "PASS" if all_ok else "FAIL", "===")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
