"""Tests for uniform cylinder on-axis closed-form potential, acceleration, and gradient."""

import numpy as np
import pytest

from gragra.analytic import (
    cylinder_axis_observables_g_free,
    cylinder_filled_axis_field,
    cylinder_gradient_axis_g_free,
    cylinder_potential_axis_g_free,
)
from gragra.constants import G_SI
from gragra.observables import GravityObservables, cylinder_axis_observables

RHO = 1500.0  # kg/m^3
RADIUS = 2.0  # m
HEIGHT = 4.0  # m


def test_cylinder_acceleration_matches_legacy_filled_axis_field():
    center = np.zeros(3)
    z_targets = np.array([-6.0, -3.0, -1.0, 0.0, 1.2, 3.5, 7.0])
    targets_3d = np.column_stack([np.zeros_like(z_targets), np.zeros_like(z_targets), z_targets])

    legacy_az = cylinder_filled_axis_field(RADIUS, HEIGHT, z_targets, RHO)
    obs = cylinder_axis_observables_g_free(center, RADIUS, HEIGHT, RHO, targets_3d)

    np.testing.assert_array_equal(obs.acceleration[:, 2], legacy_az)
    np.testing.assert_array_equal(obs.acceleration[:, 0], np.zeros_like(z_targets))
    np.testing.assert_array_equal(obs.acceleration[:, 1], np.zeros_like(z_targets))
    assert not obs.acceleration.flags.writeable


def test_cylinder_potential_derivative_matches_acceleration():
    """Verify a_z = -d(Phi)/dz using central finite difference."""
    center = np.array([0.5, -1.0, 2.0])
    z_offsets = np.array([-5.0, -1.5, 0.0, 1.2, 4.5])
    eps = 1e-6

    for z_off in z_offsets:
        target_z = center[2] + z_off
        target_3d = np.array([[center[0], center[1], target_z]])

        obs = cylinder_axis_observables_g_free(center, RADIUS, HEIGHT, RHO, target_3d)
        az_expected = obs.acceleration[0, 2]

        t_p = np.array([[center[0], center[1], target_z + eps]])
        t_m = np.array([[center[0], center[1], target_z - eps]])
        phi_p = cylinder_potential_axis_g_free(center, RADIUS, HEIGHT, RHO, t_p)[0]
        phi_m = cylinder_potential_axis_g_free(center, RADIUS, HEIGHT, RHO, t_m)[0]
        az_fd = -(phi_p - phi_m) / (2.0 * eps)

        np.testing.assert_allclose(az_fd, az_expected, rtol=1e-5, atol=1e-5 * abs(az_expected))


def test_cylinder_gradient_derivative_matches_acceleration():
    """Verify Gamma_zz = d(a_z)/dz using central finite difference."""
    center = np.array([1.0, 2.0, -3.0])
    z_offsets = np.array([-6.0, -1.0, 0.0, 0.8, 5.0])
    eps = 1e-6

    for z_off in z_offsets:
        target_z = center[2] + z_off
        target_3d = np.array([[center[0], center[1], target_z]])

        obs = cylinder_axis_observables_g_free(center, RADIUS, HEIGHT, RHO, target_3d)
        gzz_expected = obs.gradient[0, 2, 2]

        t_p = np.array([[center[0], center[1], target_z + eps]])
        t_m = np.array([[center[0], center[1], target_z - eps]])
        az_p = cylinder_axis_observables_g_free(center, RADIUS, HEIGHT, RHO, t_p).acceleration[0, 2]
        az_m = cylinder_axis_observables_g_free(center, RADIUS, HEIGHT, RHO, t_m).acceleration[0, 2]
        gzz_fd = (az_p - az_m) / (2.0 * eps)

        np.testing.assert_allclose(gzz_expected, gzz_fd, rtol=1e-5, atol=1e-4 * RHO)


def test_cylinder_gradient_axis_symmetry_and_trace():
    """Verify Poisson relation tr(Gamma) = -4*pi*rho_local and Gamma_xx == Gamma_yy."""
    center = np.zeros(3)
    half_h = HEIGHT / 2.0

    z_interior = np.array([-1.5, 0.0, 1.2])
    targets_in = np.column_stack([np.zeros_like(z_interior), np.zeros_like(z_interior), z_interior])
    obs_in = cylinder_axis_observables_g_free(center, RADIUS, HEIGHT, RHO, targets_in)

    for g in obs_in.gradient:
        # Diagonal equality
        np.testing.assert_allclose(g[0, 0], g[1, 1], atol=1e-12)
        # Off-diagonals are zero
        assert g[0, 1] == 0.0 and g[0, 2] == 0.0 and g[1, 2] == 0.0
        # Interior trace == -4*pi*rho
        np.testing.assert_allclose(np.trace(g), -4.0 * np.pi * RHO, rtol=1e-11)

    z_exterior = np.array([-6.0, -2.5, 2.5, 8.0])
    targets_out = np.column_stack([np.zeros_like(z_exterior), np.zeros_like(z_exterior), z_exterior])
    obs_out = cylinder_axis_observables_g_free(center, RADIUS, HEIGHT, RHO, targets_out)

    for g in obs_out.gradient:
        np.testing.assert_allclose(g[0, 0], g[1, 1], atol=1e-12)
        assert g[0, 1] == 0.0 and g[0, 2] == 0.0 and g[1, 2] == 0.0
        # Exterior trace == 0
        np.testing.assert_allclose(np.trace(g), 0.0, atol=1e-11 * RHO)


def test_cylinder_off_axis_and_end_face_rejection():
    center = np.zeros(3)
    half_h = HEIGHT / 2.0

    # Off-axis point must be rejected
    off_axis = np.array([[0.1, 0.0, 1.0]])
    with pytest.raises(ValueError, match="is off-axis"):
        cylinder_potential_axis_g_free(center, RADIUS, HEIGHT, RHO, off_axis)
    with pytest.raises(ValueError, match="is off-axis"):
        cylinder_gradient_axis_g_free(center, RADIUS, HEIGHT, RHO, off_axis)
    with pytest.raises(ValueError, match="is off-axis"):
        cylinder_axis_observables_g_free(center, RADIUS, HEIGHT, RHO, off_axis)

    # End-face points z = ±H/2 must be rejected for gradient and observables
    end_face = np.array([[0.0, 0.0, half_h]])
    with pytest.raises(ValueError, match="lies on the cylinder end face"):
        cylinder_gradient_axis_g_free(center, RADIUS, HEIGHT, RHO, end_face)
    with pytest.raises(ValueError, match="lies on the cylinder end face"):
        cylinder_axis_observables_g_free(center, RADIUS, HEIGHT, RHO, end_face)

    # But potential allows end face:
    phi_face = cylinder_potential_axis_g_free(center, RADIUS, HEIGHT, RHO, end_face)
    assert np.isfinite(phi_face[0])


def test_cylinder_symmetry_and_center_reflection():
    center = np.zeros(3)

    # At center (z = 0): az = 0
    obs_center = cylinder_axis_observables_g_free(center, RADIUS, HEIGHT, RHO, np.zeros((1, 3)))
    np.testing.assert_allclose(obs_center.acceleration[0, 2], 0.0, atol=1e-12)

    # Symmetrical z points
    z = 3.0
    obs_p = cylinder_axis_observables_g_free(center, RADIUS, HEIGHT, RHO, np.array([[0.0, 0.0, z]]))
    obs_m = cylinder_axis_observables_g_free(center, RADIUS, HEIGHT, RHO, np.array([[0.0, 0.0, -z]]))

    np.testing.assert_allclose(obs_p.potential, obs_m.potential, atol=1e-12)
    np.testing.assert_allclose(obs_p.acceleration[0, 2], -obs_m.acceleration[0, 2], atol=1e-12)
    np.testing.assert_allclose(obs_p.gradient, obs_m.gradient, atol=1e-12)


def test_cylinder_translation_invariance():
    c0 = np.zeros(3)
    shift = np.array([10.0, -20.0, 30.0])
    c1 = c0 + shift

    t0 = np.array([[0.0, 0.0, 1.5]])
    t1 = t0 + shift

    obs0 = cylinder_axis_observables_g_free(c0, RADIUS, HEIGHT, RHO, t0)
    obs1 = cylinder_axis_observables_g_free(c1, RADIUS, HEIGHT, RHO, t1)

    np.testing.assert_allclose(obs0.potential, obs1.potential, atol=1e-12)
    np.testing.assert_allclose(obs0.acceleration, obs1.acceleration, atol=1e-12)
    np.testing.assert_allclose(obs0.gradient, obs1.gradient, atol=1e-12)


def test_cylinder_far_field_point_mass_asymptote():
    center = np.zeros(3)
    volume = np.pi * (RADIUS**2) * HEIGHT
    mass = RHO * volume

    far_z = 300.0
    far_pt = np.array([[0.0, 0.0, far_z]])

    obs = cylinder_axis_observables_g_free(center, RADIUS, HEIGHT, RHO, far_pt)

    # Potential: Phi -> -M / z
    want_phi = -mass / far_z
    np.testing.assert_allclose(obs.potential[0], want_phi, rtol=1e-3)

    # Acceleration: az -> -M / z^2
    want_az = -mass / (far_z**2)
    np.testing.assert_allclose(obs.acceleration[0, 2], want_az, rtol=1e-3)

    # Gradient: Gzz -> 2*M / z^3, Gxx == Gyy -> -M / z^3
    want_gzz = 2.0 * mass / (far_z**3)
    want_gxx = -mass / (far_z**3)
    np.testing.assert_allclose(obs.gradient[0, 2, 2], want_gzz, rtol=1e-3)
    np.testing.assert_allclose(obs.gradient[0, 0, 0], want_gxx, rtol=1e-3)


def test_cylinder_independent_quadrature():
    pytest.importorskip("scipy")
    from scipy import integrate

    center = np.zeros(3)
    target_z = 5.0
    target = np.array([[0.0, 0.0, target_z]])

    obs = cylinder_axis_observables_g_free(center, RADIUS, HEIGHT, RHO, target)

    # 2D double integral in (r', z') for potential on axis:
    # dV' = 2*pi * r' dr' dz'
    # distance = sqrt(r'^2 + (z' - z_t)^2)
    def integrand_phi(r, z):
        return 2.0 * np.pi * r / np.sqrt(r * r + (z - target_z)**2)

    val_phi, _ = integrate.dblquad(
        integrand_phi, -HEIGHT / 2.0, HEIGHT / 2.0, 0.0, RADIUS,
        epsabs=1e-12, epsrel=1e-11,
    )
    want_phi = -RHO * val_phi
    np.testing.assert_allclose(obs.potential[0], want_phi, rtol=1e-8)

    # 2D double integral for acceleration z:
    # integrand: 2*pi * r' * (z' - z_t) / (r'^2 + (z' - z_t)^2)^(3/2)
    def integrand_az(r, z):
        return 2.0 * np.pi * r * (z - target_z) / ((r * r + (z - target_z)**2)**(1.5))

    val_az, _ = integrate.dblquad(
        integrand_az, -HEIGHT / 2.0, HEIGHT / 2.0, 0.0, RADIUS,
        epsabs=1e-12, epsrel=1e-11,
    )
    want_az = RHO * val_az
    np.testing.assert_allclose(obs.acceleration[0, 2], want_az, rtol=1e-8)


def test_cylinder_independent_quadrature_interior():
    """Verify on-axis interior potential, acceleration, and gradient via independent 1D quadrature."""
    pytest.importorskip("scipy")
    from scipy import integrate

    center = np.zeros(3)
    target_z_in = 0.5
    target_in = np.array([[0.0, 0.0, target_z_in]])
    obs_in = cylinder_axis_observables_g_free(center, RADIUS, HEIGHT, RHO, target_in)

    # 1D integral for interior potential
    def integrand_phi_1d(z):
        u = z - target_z_in
        return 2.0 * np.pi * (np.sqrt(RADIUS**2 + u**2) - np.abs(u))

    val_phi, _ = integrate.quad(integrand_phi_1d, -HEIGHT / 2.0, HEIGHT / 2.0, epsabs=1e-12, epsrel=1e-11)
    want_phi = -RHO * val_phi
    np.testing.assert_allclose(obs_in.potential[0], want_phi, rtol=1e-11)

    # 1D integral for interior acceleration
    def integrand_az_1d(z):
        u = z - target_z_in
        return 2.0 * np.pi * (np.sign(u) - u / np.sqrt(RADIUS**2 + u**2))

    val_az, _ = integrate.quad(
        integrand_az_1d, -HEIGHT / 2.0, HEIGHT / 2.0, points=[target_z_in], epsabs=1e-12, epsrel=1e-11
    )
    want_az = RHO * val_az
    np.testing.assert_allclose(obs_in.acceleration[0, 2], want_az, rtol=1e-11)

    # 1D integral for interior Gzz (smooth part - 4*pi)
    def integrand_gzz_smooth_1d(z):
        u = z - target_z_in
        return 2.0 * np.pi * (RADIUS**2) / (RADIUS**2 + u**2)**1.5

    val_gzz_smooth, _ = integrate.quad(integrand_gzz_smooth_1d, -HEIGHT / 2.0, HEIGHT / 2.0, epsabs=1e-12, epsrel=1e-11)
    want_gzz = RHO * (-4.0 * np.pi + val_gzz_smooth)
    np.testing.assert_allclose(obs_in.gradient[0, 2, 2], want_gzz, rtol=1e-11)

    # Poisson relation for transverse components
    want_gxx = (-4.0 * np.pi * RHO - obs_in.gradient[0, 2, 2]) / 2.0
    np.testing.assert_allclose(obs_in.gradient[0, 0, 0], want_gxx, rtol=1e-12)
    np.testing.assert_allclose(obs_in.gradient[0, 1, 1], want_gxx, rtol=1e-12)


def test_cylinder_independent_quadrature_transverse_gradient():
    """Verify on-axis exterior transverse gradient Gamma_xx == Gamma_yy via 2D quadrature."""
    pytest.importorskip("scipy")
    from scipy import integrate

    center = np.zeros(3)
    target_z = 3.0
    target = np.array([[0.0, 0.0, target_z]])
    obs = cylinder_axis_observables_g_free(center, RADIUS, HEIGHT, RHO, target)

    # 2D integral for Gamma_xx:
    # dV' = r' dr' dphi' dz', integral over phi' gives pi*(r'^2 - 2*u^2)
    def integrand_gxx(r, z):
        u = z - target_z
        return np.pi * r * (r**2 - 2.0 * u**2) / (r**2 + u**2)**2.5

    val_gxx, _ = integrate.dblquad(
        integrand_gxx, -HEIGHT / 2.0, HEIGHT / 2.0, 0.0, RADIUS,
        epsabs=1e-12, epsrel=1e-11,
    )
    want_gxx = RHO * val_gxx
    np.testing.assert_allclose(obs.gradient[0, 0, 0], want_gxx, rtol=1e-8)
    np.testing.assert_allclose(obs.gradient[0, 1, 1], want_gxx, rtol=1e-8)


def test_cylinder_input_validation_rejection():
    """Verify cylinder entry points strictly reject boolean and complex inputs."""
    center = [0.0, 0.0, 0.0]
    target = [[0.0, 0.0, 3.0]]

    # Boolean rejection
    with pytest.raises(TypeError, match="must not be boolean"):
        cylinder_axis_observables_g_free([True, 0.0, 0.0], RADIUS, HEIGHT, RHO, target)
    with pytest.raises(TypeError, match="must not be boolean"):
        cylinder_axis_observables_g_free(center, True, HEIGHT, RHO, target)
    with pytest.raises(TypeError, match="must not be boolean"):
        cylinder_axis_observables_g_free(center, RADIUS, False, RHO, target)
    with pytest.raises(TypeError, match="must not be boolean"):
        cylinder_axis_observables_g_free(center, RADIUS, HEIGHT, True, target)
    with pytest.raises(TypeError, match="must not be boolean"):
        cylinder_axis_observables_g_free(center, RADIUS, HEIGHT, RHO, [[0.0, 0.0, True]])

    # Complex rejection
    with pytest.raises(TypeError, match="must be real"):
        cylinder_axis_observables_g_free([0.0, 0.0, 1.0j], RADIUS, HEIGHT, RHO, target)
    with pytest.raises(TypeError, match="must be real"):
        cylinder_axis_observables_g_free(center, 1.0 + 1.0j, HEIGHT, RHO, target)
    with pytest.raises(TypeError, match="must be real"):
        cylinder_axis_observables_g_free(center, RADIUS, HEIGHT, RHO, [[0.0, 0.0, 1.0 + 2.0j]])


def test_cylinder_filled_axis_field_legacy_permissive_inputs():
    """Verify that legacy cylinder_filled_axis_field maintains permissive input acceptance."""
    # Boolean coordinates accepted and cast by legacy validate_coordinate_input
    out = cylinder_filled_axis_field(RADIUS, HEIGHT, [True, False], RHO)
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))
    # Scalar input returns scalar
    scalar_out = cylinder_filled_axis_field(RADIUS, HEIGHT, 1.5, RHO)
    assert scalar_out.shape == ()
    assert np.isfinite(scalar_out)


def test_cylinder_si_entrypoint():
    center = np.array([0.0, 0.0, 1.0])
    targets = np.array([[0.0, 0.0, 3.5]])

    si_obs = cylinder_axis_observables(center, RADIUS, HEIGHT, RHO, targets)
    g_free = cylinder_axis_observables_g_free(center, RADIUS, HEIGHT, RHO, targets)

    assert isinstance(si_obs, GravityObservables)
    np.testing.assert_allclose(si_obs.potential, G_SI * g_free.potential)
    np.testing.assert_allclose(si_obs.acceleration, G_SI * g_free.acceleration)
    np.testing.assert_allclose(si_obs.gradient, G_SI * g_free.gradient)
