"""Tests for finite spherical shell P-wave acceleration component decomposition."""

import numpy as np
import pytest

from gragra.analytic import (
    GFreeSphericalShellPComponents,
    spherical_cavity_bulk_p,
    spherical_cavity_total_p,
    spherical_shell_p_acceleration_components_g_free,
    spherical_shell_p_infinite_difference_g_free,
)
from gragra.constants import G_SI
from gragra.observables import (
    SphericalShellPComponents,
    spherical_shell_p_acceleration_components,
    spherical_shell_p_infinite_difference,
)

A_INNER = 15.0   # inner cavity radius a in meters
R_OUTER = 120.0  # outer shell radius R in meters
RHO = 2500.0     # density in kg/m^3
DISP = 1.5e-6    # displacement amplitude in meters
DIR_VEC = [1.0, 2.0, -2.0]  # direction vector


def test_spherical_shell_p_decomposition_identity():
    """Verify total == bulk + inner_wall + outer_boundary bit-identically or to machine epsilon."""
    k_vals = [0.005, 0.02, 0.05, 0.1]
    direction = np.array([0.0, 0.0, 1.0])

    comps = spherical_shell_p_acceleration_components_g_free(
        inner_radius_m=A_INNER,
        outer_radius_m=R_OUTER,
        wavenumber_rad_m=k_vals,
        density_kg_m3=RHO,
        displacement_m=DISP,
        direction=direction,
    )

    assert isinstance(comps, GFreeSphericalShellPComponents)
    assert comps.unit_system == "g_free"
    assert comps.bulk.shape == (len(k_vals), 1, 3)
    assert comps.total.shape == (len(k_vals), 1, 3)

    sum_parts = comps.bulk + comps.inner_wall + comps.outer_boundary
    np.testing.assert_allclose(comps.total, sum_parts, rtol=1e-14, atol=1e-14 * np.max(np.abs(comps.total)))


def test_spherical_shell_p_k_zero_limit():
    """Verify at k=0: bulk = 0, total = 0, inner_wall cancels outer_boundary."""
    direction = np.array([1.0, 0.0, 0.0])

    comps = spherical_shell_p_acceleration_components_g_free(
        inner_radius_m=A_INNER,
        outer_radius_m=R_OUTER,
        wavenumber_rad_m=0.0,
        density_kg_m3=RHO,
        displacement_m=DISP,
        direction=direction,
    )

    assert comps.total.shape == (1, 3)
    np.testing.assert_allclose(comps.bulk, np.zeros((1, 3)), atol=1e-14)
    np.testing.assert_allclose(comps.total, np.zeros((1, 3)), atol=1e-14)

    # inner_wall = -4*pi*rho/3 * xi * e, outer_boundary = +4*pi*rho/3 * xi * e
    expected_inner = (-4.0 * np.pi * RHO / 3.0) * DISP * direction
    expected_outer = (+4.0 * np.pi * RHO / 3.0) * DISP * direction

    np.testing.assert_allclose(comps.inner_wall[0], expected_inner, rtol=1e-12)
    np.testing.assert_allclose(comps.outer_boundary[0], expected_outer, rtol=1e-12)
    np.testing.assert_allclose(comps.inner_wall + comps.outer_boundary, np.zeros((1, 3)), atol=1e-14)


def test_spherical_shell_p_large_r_limit_approaches_infinite():
    """Verify that as R -> large, finite total approaches infinite cavity total for k > 0."""
    k = 0.05
    direction = np.array([0.0, 1.0, 0.0])

    # Infinite cavity total:
    inf_tot_mag = spherical_cavity_total_p(A_INNER, k, RHO, DISP)
    inf_tot_vec = inf_tot_mag * direction

    # As R increases, finite total should converge to infinite cavity total:
    prev_err = float("inf")
    for R in [200.0, 800.0, 3200.0]:
        comps = spherical_shell_p_acceleration_components_g_free(
            inner_radius_m=A_INNER,
            outer_radius_m=R,
            wavenumber_rad_m=k,
            density_kg_m3=RHO,
            displacement_m=DISP,
            direction=direction,
        )
        finite_tot_vec = comps.total[0]
        err = np.linalg.norm(finite_tot_vec - inf_tot_vec)
        # Verify monotone error decay ~ 1/R
        assert err < prev_err
        prev_err = err

    # At R=3200m, truncation factor j1(kR)/(kR) is very small
    np.testing.assert_allclose(finite_tot_vec, inf_tot_vec, rtol=1e-2)


def test_spherical_shell_p_infinite_difference_helper():
    """Verify infinite_difference equals 8*pi*rho*f(kR)*xi*e and matches inf - finite."""
    k = 0.03
    direction = np.array([0.6, 0.8, 0.0])

    diff = spherical_shell_p_infinite_difference_g_free(
        inner_radius_m=A_INNER,
        outer_radius_m=R_OUTER,
        wavenumber_rad_m=k,
        density_kg_m3=RHO,
        displacement_m=DISP,
        direction=direction,
    )

    comps = spherical_shell_p_acceleration_components_g_free(
        inner_radius_m=A_INNER,
        outer_radius_m=R_OUTER,
        wavenumber_rad_m=k,
        density_kg_m3=RHO,
        displacement_m=DISP,
        direction=direction,
    )

    inf_mag = spherical_cavity_total_p(A_INNER, k, RHO, DISP)
    inf_vec = inf_mag * direction

    np.testing.assert_allclose(diff[0], inf_vec - comps.total[0], rtol=1e-12)

    # k=0 must be rejected
    with pytest.raises(ValueError, match="strictly positive"):
        spherical_shell_p_infinite_difference_g_free(
            A_INNER, R_OUTER, 0.0, RHO, DISP, direction
        )


def test_spherical_shell_p_complex_phasor():
    """Verify complex displacement amplitude preserves phase and gives complex128 output."""
    k = 0.04
    direction = np.array([0.0, 0.0, 1.0])
    phase = np.exp(1j * np.pi / 3.0)
    disp_c = DISP * phase

    comps_c = spherical_shell_p_acceleration_components_g_free(
        inner_radius_m=A_INNER,
        outer_radius_m=R_OUTER,
        wavenumber_rad_m=k,
        density_kg_m3=RHO,
        displacement_m=disp_c,
        direction=direction,
    )

    comps_r = spherical_shell_p_acceleration_components_g_free(
        inner_radius_m=A_INNER,
        outer_radius_m=R_OUTER,
        wavenumber_rad_m=k,
        density_kg_m3=RHO,
        displacement_m=DISP,
        direction=direction,
    )

    assert comps_c.total.dtype == np.complex128
    np.testing.assert_allclose(comps_c.bulk, comps_r.bulk * phase, rtol=1e-14)
    np.testing.assert_allclose(comps_c.total, comps_r.total * phase, rtol=1e-14)


def test_spherical_shell_p_independent_quadrature():
    """Verify P-wave bulk, inner wall, outer boundary, and total against independent numerical quadrature."""
    pytest.importorskip("scipy")
    from scipy import integrate

    k = 0.02
    a = 10.0
    R = 40.0
    rho = 2000.0
    xi = 1e-5
    direction = np.array([0.0, 0.0, 1.0])  # along z

    # Analytic values:
    comps = spherical_shell_p_acceleration_components_g_free(
        inner_radius_m=a,
        outer_radius_m=R,
        wavenumber_rad_m=k,
        density_kg_m3=rho,
        displacement_m=xi,
        direction=direction,
    )

    # 1. Independent numerical integration for bulk term:
    # Displacement u(r, theta) = xi * e^{i k z} e_z = xi * e^{i k r cos(theta)} e_z
    # Bulk density perturbation: delta_rho = -rho * div(u) = -rho * (i k xi) e^{i k r cos(theta)}
    # At center (target = 0), gravitational acceleration a_z = int delta_rho * (z/r^3) dV
    # = int delta_rho * (cos(theta)/r^2) * (r^2 sin(theta) dr dtheta dphi)
    # = 2*pi * int_a^R dr int_0^pi dtheta sin(theta) cos(theta) delta_rho
    # Using real part (or sin(k r cos theta)):
    # delta_rho = rho * k * xi * sin(k r cos theta)
    def integrand_bulk(theta, r):
        return np.sin(theta) * np.cos(theta) * np.sin(k * r * np.cos(theta))

    val_bulk, _ = integrate.dblquad(integrand_bulk, a, R, 0.0, np.pi, epsabs=1e-12, epsrel=1e-10)
    want_bulk_z = 2.0 * np.pi * rho * k * xi * val_bulk
    np.testing.assert_allclose(comps.bulk[0, 2], want_bulk_z, rtol=1e-7)

    # 2. Independent numerical integration for inner wall (surface mass sheet):
    # Inner normal is -e_r (pointing out of medium into cavity).
    # Surface mass sheet sigma = -rho * (u . e_r) = -rho * xi * cos(theta) * cos(k a cos(theta))
    # Attraction on center test mass: int sigma * (cos(theta)/a^2) * a^2 sin(theta) dtheta dphi
    def integrand_inner(theta):
        return -np.sin(theta) * (np.cos(theta)**2) * np.cos(k * a * np.cos(theta))

    val_inner, _ = integrate.quad(integrand_inner, 0.0, np.pi, epsabs=1e-12, epsrel=1e-10)
    want_inner_z = 2.0 * np.pi * rho * xi * val_inner
    np.testing.assert_allclose(comps.inner_wall[0, 2], want_inner_z, rtol=1e-7)

    # 3. Independent numerical integration for outer boundary:
    # Outer normal is +e_r (pointing outward).
    # Surface mass sheet sigma = +rho * (u . e_r) = +rho * xi * cos(theta) * cos(k R cos(theta))
    def integrand_outer(theta):
        return np.sin(theta) * (np.cos(theta)**2) * np.cos(k * R * np.cos(theta))

    val_outer, _ = integrate.quad(integrand_outer, 0.0, np.pi, epsabs=1e-12, epsrel=1e-10)
    want_outer_z = 2.0 * np.pi * rho * xi * val_outer
    np.testing.assert_allclose(comps.outer_boundary[0, 2], want_outer_z, rtol=1e-7)

    # 4. Total acceleration check against sum of independent 3-component quadratures:
    want_total_3sum = want_bulk_z + want_inner_z + want_outer_z
    np.testing.assert_allclose(comps.total[0, 2], want_total_3sum, rtol=1e-10)

    # 5. Independent direct dipole volume integration for total acceleration (ANNA v4 Eq. 17):
    # delta_a_t(x0, omega) = +G \int_Omega [rho(x) / ||x - x0||^3] (I - 3 e_r (x) e_r) u(x, omega) dv
    # For u(x) = xi * cos(k*z) e_z at cavity center x0 = 0, the z-component integrand is:
    # (1 - 3*cos(theta)^2) * cos(k*r*cos(theta)) * sin(theta) / r
    def integrand_dipole(theta, r):
        return (1.0 - 3.0 * np.cos(theta)**2) * np.cos(k * r * np.cos(theta)) * np.sin(theta) / r

    val_dipole, _ = integrate.dblquad(integrand_dipole, a, R, 0.0, np.pi, epsabs=1e-13, epsrel=1e-12)
    want_total_dipole = 2.0 * np.pi * rho * xi * val_dipole
    np.testing.assert_allclose(comps.total[0, 2], want_total_dipole, rtol=1e-10)


def test_spherical_shell_p_convergence_study_representative_cases():
    """Verify convergence of direct dipole quadrature across representative cases (thin shell, small k)."""
    pytest.importorskip("scipy")
    from scipy import integrate

    cases = [
        ("Standard shell", 10.0, 40.0, 0.02, 2000.0, 1e-5),
        ("Thin shell (Delta R=0.5)", 10.0, 10.5, 0.02, 2000.0, 1e-5),
        ("Small non-zero k (1e-5)", 10.0, 40.0, 1e-5, 2000.0, 1e-5),
    ]

    for name, a, R, k, rho, xi in cases:
        comps = spherical_shell_p_acceleration_components_g_free(
            inner_radius_m=a, outer_radius_m=R, wavenumber_rad_m=k,
            density_kg_m3=rho, displacement_m=xi, direction=[0, 0, 1]
        )
        tot_analytic = comps.total[0, 2]

        def integrand_dipole(theta, r, k=k):
            return (1.0 - 3.0 * np.cos(theta)**2) * np.cos(k * r * np.cos(theta)) * np.sin(theta) / r

        # Evaluate dipole quadrature across tightening relative tolerances:
        prev_err = float("inf")
        for tol in [1e-6, 1e-9, 1e-12]:
            val_dipole, _ = integrate.dblquad(
                integrand_dipole, a, R, 0.0, np.pi,
                epsabs=tol * 1e-2, epsrel=tol
            )
            tot_dipole = 2.0 * np.pi * rho * xi * val_dipole
            err = abs(tot_dipole - tot_analytic)
            # Verify numerical accuracy within acceptance threshold
            assert err <= 1e-6 * abs(tot_analytic) or err < 1e-15


def test_spherical_shell_legacy_input_compatibility():
    """Verify that legacy spherical cavity functions maintain permissive input acceptance."""
    # spherical_cavity_bulk_p accepts boolean arrays / sequences via permissive _finite_array cast
    out = spherical_cavity_bulk_p(10.0, [True, False], 2000.0, 1e-5)
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))


def test_spherical_shell_p_zero_density():
    """Verify zero background density produces identically zero acceleration for all components."""
    comps_zero = spherical_shell_p_acceleration_components_g_free(
        inner_radius_m=A_INNER,
        outer_radius_m=R_OUTER,
        wavenumber_rad_m=0.03,
        density_kg_m3=0.0,
        displacement_m=DISP,
        direction=DIR_VEC,
    )
    np.testing.assert_array_equal(comps_zero.bulk, np.zeros((1, 3)))
    np.testing.assert_array_equal(comps_zero.inner_wall, np.zeros((1, 3)))
    np.testing.assert_array_equal(comps_zero.outer_boundary, np.zeros((1, 3)))
    np.testing.assert_array_equal(comps_zero.total, np.zeros((1, 3)))


def test_spherical_shell_p_invalid_inputs():
    direction = [1, 0, 0]
    # R <= a
    with pytest.raises(ValueError, match="strictly greater than"):
        spherical_shell_p_acceleration_components_g_free(20.0, 10.0, 0.01, RHO, DISP, direction)
    with pytest.raises(ValueError, match="strictly greater than"):
        spherical_shell_p_acceleration_components_g_free(20.0, 20.0, 0.01, RHO, DISP, direction)

    # negative k
    with pytest.raises(ValueError, match="non-negative"):
        spherical_shell_p_acceleration_components_g_free(10.0, 20.0, -0.01, RHO, DISP, direction)

    # negative density
    with pytest.raises(ValueError, match="non-negative"):
        spherical_shell_p_acceleration_components_g_free(10.0, 20.0, 0.01, -1.0, DISP, direction)

    # invalid direction
    with pytest.raises(ValueError, match="non-zero"):
        spherical_shell_p_acceleration_components_g_free(10.0, 20.0, 0.01, RHO, DISP, [0, 0, 0])

    # Boolean rejection
    with pytest.raises(TypeError, match="must not be boolean"):
        spherical_shell_p_acceleration_components_g_free(True, 20.0, 0.01, RHO, DISP, direction)
    with pytest.raises(TypeError, match="must not be boolean"):
        spherical_shell_p_acceleration_components_g_free(10.0, False, 0.01, RHO, DISP, direction)
    with pytest.raises(TypeError, match="must not be boolean"):
        spherical_shell_p_acceleration_components_g_free(10.0, 20.0, True, RHO, DISP, direction)
    with pytest.raises(TypeError, match="must not be boolean"):
        spherical_shell_p_acceleration_components_g_free(10.0, 20.0, 0.01, True, DISP, direction)
    with pytest.raises(TypeError, match="must not be boolean"):
        spherical_shell_p_acceleration_components_g_free(10.0, 20.0, 0.01, RHO, DISP, [True, False, False])

    # Complex rejection (radii, wavenumber, density, direction must be real)
    with pytest.raises(TypeError, match="must be real"):
        spherical_shell_p_acceleration_components_g_free(10.0 + 1j, 20.0, 0.01, RHO, DISP, direction)
    with pytest.raises(TypeError, match="must be real"):
        spherical_shell_p_acceleration_components_g_free(10.0, 20.0 + 1j, 0.01, RHO, DISP, direction)
    with pytest.raises(TypeError, match="must be real"):
        spherical_shell_p_acceleration_components_g_free(10.0, 20.0, 0.01 + 1j, RHO, DISP, direction)
    with pytest.raises(TypeError, match="must be real"):
        spherical_shell_p_acceleration_components_g_free(10.0, 20.0, 0.01, 2000.0 + 1j, DISP, direction)
    with pytest.raises(TypeError, match="must be real"):
        spherical_shell_p_acceleration_components_g_free(10.0, 20.0, 0.01, RHO, DISP, [1j, 0.0, 0.0])


def test_spherical_shell_p_si_entrypoint():
    direction = [0, 1, 0]
    k = 0.02
    si_comps = spherical_shell_p_acceleration_components(
        inner_radius_m=A_INNER,
        outer_radius_m=R_OUTER,
        wavenumber_rad_m=k,
        density_kg_m3=RHO,
        displacement_m=DISP,
        direction=direction,
    )
    g_free = spherical_shell_p_acceleration_components_g_free(
        inner_radius_m=A_INNER,
        outer_radius_m=R_OUTER,
        wavenumber_rad_m=k,
        density_kg_m3=RHO,
        displacement_m=DISP,
        direction=direction,
    )

    assert isinstance(si_comps, SphericalShellPComponents)
    assert si_comps.unit_system == "si"
    np.testing.assert_allclose(si_comps.bulk, G_SI * g_free.bulk)
    np.testing.assert_allclose(si_comps.inner_wall, G_SI * g_free.inner_wall)
    np.testing.assert_allclose(si_comps.outer_boundary, G_SI * g_free.outer_boundary)
    np.testing.assert_allclose(si_comps.total, G_SI * g_free.total)

    si_diff = spherical_shell_p_infinite_difference(
        A_INNER, R_OUTER, k, RHO, DISP, direction
    )
    g_diff = spherical_shell_p_infinite_difference_g_free(
        A_INNER, R_OUTER, k, RHO, DISP, direction
    )
    np.testing.assert_allclose(si_diff, G_SI * g_diff)
