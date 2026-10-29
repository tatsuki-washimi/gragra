"""Phase 12-b Slice 3: filled-cylinder on-axis field tests.

Design: docs/design/phase12b_sca_cavity_coupling.md; derivation/CAS
verification record: docs/references/02_NN_Modeling_and_Simulation/
02-1_Analytical/Blakely_1995_cylinder_axis_gravity.md.
"""

import warnings

import numpy as np
import pytest
from oracles.cylinder_axis_potential_check import run_isolated_reference_case

from gragra.analytic import cylinder_filled_axis_field

RHO = 1234.5  # kg/m^3, arbitrary positive density (matches CAS script)


def test_golden_matches_direct_integration():
    pytest.importorskip("scipy")
    from scipy import integrate

    radius, height = 1.2, 2.5
    z_points = np.array([-3.0, -0.6, 0.33, 2.0])

    got = cylinder_filled_axis_field(radius, height, z_points, RHO)

    def integrand(z, r, z_t):
        dz = z - z_t
        return 2.0 * np.pi * r * dz / (r * r + dz * dz) ** 1.5

    want = []
    for z_t in z_points:
        val, _ = integrate.dblquad(
            integrand,
            0.0,
            radius,
            -height / 2.0,
            height / 2.0,
            args=(z_t,),
            epsabs=1e-12,
            epsrel=1e-11,
        )
        want.append(RHO * val)
    np.testing.assert_allclose(got, want, rtol=1e-7)


def test_center_and_symmetry():
    radius, height = 1.0, 2.0
    got = cylinder_filled_axis_field(radius, height, 0.0, RHO)
    assert got == pytest.approx(0.0, abs=1e-12)

    z_pos = cylinder_filled_axis_field(radius, height, 0.7, RHO)
    z_neg = cylinder_filled_axis_field(radius, height, -0.7, RHO)
    assert z_neg == pytest.approx(-z_pos, rel=1e-12)


@pytest.mark.parametrize("z", [0.0, 0.25, 2.0, 1e6])
def test_thin_axis_potential_retains_float64_precision(z):
    """A numerical regression, not a new scientific acceptance budget."""
    if not run_isolated_reference_case("thin-axis", z):
        pytest.skip("mpmath is required for the independent cylinder oracle")


@pytest.mark.parametrize("height", [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 2e-2, 1, 200])
def test_axis_potential_height_sweep_against_independent_volume(height):
    if not run_isolated_reference_case("height-sweep", height):
        pytest.skip("mpmath is required for the independent cylinder oracle")


def test_thin_center_independent_references_refine_in_precision_and_quadrature():
    if not run_isolated_reference_case("center-refinement"):
        pytest.skip("mpmath is required for the independent cylinder oracle")


def test_disk_limit():
    radius = 1.2
    h_thin = 1e-7
    z_t = 0.8
    got = cylinder_filled_axis_field(radius, h_thin, z_t, RHO)
    want = -2.0 * np.pi * RHO * h_thin * (1.0 - z_t / np.sqrt(radius**2 + z_t**2))
    assert got == pytest.approx(want, rel=1e-6)


def test_far_field_point_mass_asymptote():
    radius, height = 1.2, 2.5
    z_far = 300.0
    volume = np.pi * radius**2 * height
    got = cylinder_filled_axis_field(radius, height, z_far, RHO)
    want = -RHO * volume / z_far**2
    assert got == pytest.approx(want, rel=1e-3)


def test_boundary_and_interior_finite():
    radius, height = 1.2, 2.5
    z_points = np.array([-height / 2.0, 0.0, height / 2.0, 0.6])
    got = cylinder_filled_axis_field(radius, height, z_points, RHO)
    assert np.all(np.isfinite(got))


def test_array_shape_matches_input():
    radius, height = 1.0, 1.0
    z_points = np.array([-0.3, 0.0, 0.4, 0.9])
    got = cylinder_filled_axis_field(radius, height, z_points, RHO)
    assert got.shape == z_points.shape


@pytest.mark.parametrize("radius_m", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_radius(radius_m):
    with pytest.raises(ValueError):
        cylinder_filled_axis_field(radius_m, 1.0, 0.0, RHO)


@pytest.mark.parametrize("height_m", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_height(height_m):
    with pytest.raises(ValueError):
        cylinder_filled_axis_field(1.0, height_m, 0.0, RHO)


@pytest.mark.parametrize("density_kg_m3", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_density(density_kg_m3):
    with pytest.raises(ValueError):
        cylinder_filled_axis_field(1.0, 1.0, 0.0, density_kg_m3)


def test_rejects_non_finite_z():
    with pytest.raises(ValueError):
        cylinder_filled_axis_field(1.0, 1.0, float("nan"), RHO)
    with pytest.raises(ValueError):
        cylinder_filled_axis_field(1.0, 1.0, np.array([0.0, float("inf")]), RHO)


def test_output_read_only():
    out = cylinder_filled_axis_field(1.0, 1.0, np.array([0.3, -0.2]), RHO)
    assert out.flags.writeable is False
    assert out.dtype == np.float64


@pytest.mark.slow
def test_offaxis_poisson_axisymmetry_matches_independent_integration():
    """Cross-check the on-axis Poisson+axisymmetry-derived transverse gradient
    against an independent off-axis numerical integration.

    Moved here from examples/13_sca_lower_limit_gate.py (code review,
    2026-07-13): the off-axis probe points are interior to the filled
    cylinder, so scipy.integrate.tplquad's domain contains the field's own
    1/r^2 singularity, and certifying the rtol=1e-4 comparison below at
    epsabs=epsrel=1e-8 costs several minutes -- keeping that in the
    examples/ smoke-run (tests/test_examples.py) added ~5 minutes to every
    default `pytest` invocation, contradicting this project's slow-test
    convention (pyproject.toml addopts = "-m 'not slow'"). Only the
    x-component is integrated (the example's original version integrated
    all 3 components per probe point via a `for comp in range(3)` loop but
    only ever used component 0, wasting 2 of every 3 tplquad calls).
    """
    pytest.importorskip("scipy")
    from scipy import integrate

    radius, height = 1.2, 2.5
    z0 = 0.4  # interior on-axis point
    dz = 1e-5

    az_p = cylinder_filled_axis_field(radius, height, z0 + dz, RHO)
    az_m = cylinder_filled_axis_field(radius, height, z0 - dz, RHO)
    g_zz = float((az_p - az_m) / (2 * dz))

    # Poisson + axisymmetry (g_xx == g_yy on-axis): 2*g_xx + g_zz = -4*pi*rho.
    g_xx_from_poisson = (-4.0 * np.pi * RHO - g_zz) / 2.0

    def integrand(phi, rr, z, tx, tz):
        sx = rr * np.cos(phi)
        sy = rr * np.sin(phi)
        dx = sx - tx
        dy = sy
        dzc = z - tz
        return rr * dx / (dx * dx + dy * dy + dzc * dzc) ** 1.5

    def ax_component(tx: float) -> float:
        # target is interior to the integration domain, so the 1/r^2
        # singularity sits inside it -- quadpack cannot certify a
        # near-machine-precision tolerance here. epsabs/epsrel are set to
        # what the rtol=1e-4 comparison below actually needs, and the
        # resulting IntegrationWarning (a "this is my best effort" notice,
        # not a failure) is caught and silenced deliberately.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)
            val, _ = integrate.tplquad(
                integrand,
                -height / 2,
                height / 2,
                0.0,
                radius,
                0.0,
                2 * np.pi,
                args=(tx, z0),
                epsabs=1e-8,
                epsrel=1e-8,
            )
        return RHO * val

    dr = 1e-4
    g_xx_numeric = (ax_component(dr) - ax_component(-dr)) / (2 * dr)

    np.testing.assert_allclose(g_xx_from_poisson, g_xx_numeric, rtol=1e-4)
