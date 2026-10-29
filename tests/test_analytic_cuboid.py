"""Tests for uniform cuboid closed-form potential, acceleration, and gradient."""

import numpy as np
import pytest

from gragra.analytic import (
    cuboid_acceleration_g_free,
    cuboid_filled_internal_field,
    cuboid_gradient_g_free,
    cuboid_observables_g_free,
    cuboid_potential_g_free,
)
from gragra.constants import G_SI
from gragra.observables import GravityObservables, cuboid_observables

RHO = 1234.5  # kg/m^3


def test_cuboid_acceleration_matches_existing_filled_field():
    lengths = np.array([3.0, 2.0, 1.5])
    targets = np.array([
        [0.5, -0.4, 0.3],
        [2.0, 3.0, -4.0],
        [-1.2, 0.8, -0.5],
    ])
    center = np.zeros(3)

    got_new = cuboid_acceleration_g_free(center, lengths, RHO, targets)
    got_legacy = cuboid_filled_internal_field(lengths, targets, RHO)

    np.testing.assert_array_equal(got_new, got_legacy)
    assert not got_new.flags.writeable


def test_cuboid_potential_derivative_matches_acceleration():
    """Verify a_i = -d(Phi)/d(x_{t,i}) using central finite difference on potential."""
    lengths = np.array([2.5, 3.5, 1.8])
    center = np.array([1.0, -0.5, 2.0])
    points = np.array([
        [1.3, -0.2, 2.4],   # interior
        [5.0, -4.0, 6.0],   # exterior
    ])
    eps = 1e-6
    for pt in points:
        a_expected = cuboid_acceleration_g_free(center, lengths, RHO, np.array([pt]))[0]
        a_fd = np.zeros(3)
        for comp in range(3):
            dp = np.zeros(3)
            dp[comp] = eps
            phi_p = cuboid_potential_g_free(center, lengths, RHO, np.array([pt + dp]))[0]
            phi_m = cuboid_potential_g_free(center, lengths, RHO, np.array([pt - dp]))[0]
            a_fd[comp] = -(phi_p - phi_m) / (2.0 * eps)

        np.testing.assert_allclose(a_fd, a_expected, rtol=1e-5, atol=1e-5 * np.max(np.abs(a_expected)))


def test_cuboid_gradient_derivative_matches_acceleration():
    """Verify Gamma_{ij} = d(a_i)/d(x_{t,j}) using central finite difference on acceleration."""
    lengths = np.array([2.0, 3.0, 4.0])
    center = np.array([-1.0, 0.5, 1.5])
    points = np.array([
        [-0.7, 0.2, 2.0],   # interior
        [3.0, 4.0, -2.0],   # exterior
    ])
    eps = 1e-6
    for pt in points:
        grad_expected = cuboid_gradient_g_free(center, lengths, RHO, np.array([pt]))[0]
        grad_fd = np.zeros((3, 3))
        for comp in range(3):
            dp = np.zeros(3)
            dp[comp] = eps
            ap = cuboid_acceleration_g_free(center, lengths, RHO, np.array([pt + dp]))[0]
            am = cuboid_acceleration_g_free(center, lengths, RHO, np.array([pt - dp]))[0]
            grad_fd[:, comp] = (ap - am) / (2.0 * eps)

        np.testing.assert_allclose(grad_expected, grad_fd, rtol=1e-5, atol=1e-4 * RHO)


def test_cuboid_gradient_symmetry():
    lengths = np.array([3.0, 2.0, 1.5])
    center = np.array([0.5, -0.5, 1.0])
    targets = np.array([
        [0.7, -0.4, 1.2],    # interior
        [4.0, 5.0, -3.0],    # exterior
    ])
    grads = cuboid_gradient_g_free(center, lengths, RHO, targets)
    for g in grads:
        np.testing.assert_allclose(g, g.T, atol=1e-12)


def test_cuboid_poisson_trace():
    lengths = np.array([3.0, 2.0, 1.5])
    center = np.zeros(3)

    interior_targets = np.array([
        [0.0, 0.0, 0.0],
        [0.8, -0.5, 0.4],
        [-1.0, 0.7, -0.3],
    ])
    exterior_targets = np.array([
        [3.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [5.0, -6.0, 7.0],
    ])

    int_grads = cuboid_gradient_g_free(center, lengths, RHO, interior_targets)
    for g in int_grads:
        np.testing.assert_allclose(np.trace(g), -4.0 * np.pi * RHO, rtol=1e-11)

    ext_grads = cuboid_gradient_g_free(center, lengths, RHO, exterior_targets)
    for g in ext_grads:
        np.testing.assert_allclose(np.trace(g), 0.0, atol=1e-11 * RHO)


def test_cuboid_boundary_rejection():
    lengths = np.array([2.0, 3.0, 4.0])
    center = np.array([0.0, 0.0, 0.0])
    half = lengths / 2.0

    # Points on face, edge, vertex must be rejected by gradient and observables_g_free
    face_pt = np.array([[half[0], 0.5, 0.5]])
    edge_pt = np.array([[half[0], half[1], 0.5]])
    vert_pt = np.array([[half[0], half[1], half[2]]])

    for pt in [face_pt, edge_pt, vert_pt]:
        with pytest.raises(ValueError, match="lies on the cuboid boundary"):
            cuboid_gradient_g_free(center, lengths, RHO, pt)
        with pytest.raises(ValueError, match="lies on the cuboid boundary"):
            cuboid_observables_g_free(center, lengths, RHO, pt)

    # Face extension plane point OUTSIDE the box must NOT be rejected
    face_ext_pt = np.array([[half[0], half[1] + 1.0, 0.5]])
    obs = cuboid_observables_g_free(center, lengths, RHO, face_ext_pt)
    assert obs.potential.shape == (1,)
    assert obs.gradient.shape == (1, 3, 3)
    np.testing.assert_allclose(np.trace(obs.gradient[0]), 0.0, atol=1e-10 * RHO)


def test_cuboid_translation_invariance():
    lengths = np.array([2.0, 3.0, 1.5])
    c0 = np.array([0.0, 0.0, 0.0])
    shift = np.array([12.3, -45.6, 78.9])
    c1 = c0 + shift

    target0 = np.array([[0.5, -0.3, 0.2]])
    target1 = target0 + shift

    obs0 = cuboid_observables_g_free(c0, lengths, RHO, target0)
    obs1 = cuboid_observables_g_free(c1, lengths, RHO, target1)

    np.testing.assert_allclose(obs0.potential, obs1.potential, atol=1e-12)
    np.testing.assert_allclose(obs0.acceleration, obs1.acceleration, atol=1e-12)
    np.testing.assert_allclose(obs0.gradient, obs1.gradient, atol=1e-12)


def test_cuboid_symmetry_center_and_reflection():
    lengths = np.array([2.0, 3.0, 4.0])
    center = np.zeros(3)

    # At center: acceleration is zero
    obs_center = cuboid_observables_g_free(center, lengths, RHO, np.zeros((1, 3)))
    np.testing.assert_allclose(obs_center.acceleration, np.zeros((1, 3)), atol=1e-12)

    # Opposite points: potential is symmetric, acceleration is anti-symmetric, gradient is symmetric
    pt = np.array([[0.7, -0.8, 0.9]])
    obs_pos = cuboid_observables_g_free(center, lengths, RHO, pt)
    obs_neg = cuboid_observables_g_free(center, lengths, RHO, -pt)

    np.testing.assert_allclose(obs_pos.potential, obs_neg.potential, atol=1e-12)
    np.testing.assert_allclose(obs_pos.acceleration, -obs_neg.acceleration, atol=1e-12)
    np.testing.assert_allclose(obs_pos.gradient, obs_neg.gradient, atol=1e-12)


def test_cuboid_density_linearity_and_zero():
    lengths = np.array([2.0, 2.5, 3.0])
    center = np.zeros(3)
    target = np.array([[1.0, 2.0, 3.0]])

    obs_rho = cuboid_observables_g_free(center, lengths, RHO, target)
    obs_2rho = cuboid_observables_g_free(center, lengths, 2.0 * RHO, target)

    np.testing.assert_allclose(obs_2rho.potential, 2.0 * obs_rho.potential, atol=1e-12)
    np.testing.assert_allclose(obs_2rho.acceleration, 2.0 * obs_rho.acceleration, atol=1e-12)
    np.testing.assert_allclose(obs_2rho.gradient, 2.0 * obs_rho.gradient, atol=1e-12)

    obs_zero = cuboid_observables_g_free(center, lengths, 0.0, target)
    np.testing.assert_allclose(obs_zero.potential, np.zeros(1), atol=1e-14)
    np.testing.assert_allclose(obs_zero.acceleration, np.zeros((1, 3)), atol=1e-14)
    np.testing.assert_allclose(obs_zero.gradient, np.zeros((1, 3, 3)), atol=1e-14)


def test_cuboid_far_field_point_mass_asymptote():
    lengths = np.array([2.0, 3.0, 1.5])
    volume = np.prod(lengths)
    mass = RHO * volume
    center = np.zeros(3)

    far_pt = np.array([[200.0, -300.0, 400.0]])
    r = np.linalg.norm(far_pt[0])

    obs = cuboid_observables_g_free(center, lengths, RHO, far_pt)

    # Point mass limit: Phi -> -M / r
    want_phi = -mass / r
    np.testing.assert_allclose(obs.potential[0], want_phi, rtol=1e-3)

    # Point mass limit: a -> -M * r / r^3
    want_a = -mass * far_pt[0] / (r**3)
    np.testing.assert_allclose(obs.acceleration[0], want_a, rtol=1e-3)

    # Point mass limit: Gamma -> M * (3 * r_i * r_j / r^5 - delta_ij / r^3)
    outer = np.outer(far_pt[0], far_pt[0])
    want_grad = mass * (3.0 * outer / (r**5) - np.eye(3) / (r**3))
    np.testing.assert_allclose(obs.gradient[0], want_grad, rtol=1e-3)


def test_cuboid_gradient_external_edge_extension():
    """Verify gradient at external edge-extension points (e.g. cube=[-1,1]^3, target=[1,1,2]).

    Regression test for the issue where single-corner log arguments vanished
    on edge lines, leading to Gamma_xy = -0.850053... instead of +0.248558956790044...
    Checks:
    - Target [1, 1, 2] against exact independent numerical volume integral.
    - All 3x3 gradient tensor components against tplquad.
    - Coordinate permutations ([1, 2, 1], [2, 1, 1]).
    - Coordinate reflections ([±1, ±1, ±2], [±1, ±2, ±1], [±2, ±1, ±1]).
    - Nearby points ([1 ± 1e-4, 1 ± 1e-4, 2], [1, 1, 2 ± 1e-4]).
    """
    pytest.importorskip("scipy")
    from scipy import integrate

    lengths = np.array([2.0, 2.0, 2.0])
    center = np.zeros(3)
    rho = 1.0

    # 1. Exact check at [1, 1, 2]
    target_edge = np.array([[1.0, 1.0, 2.0]])
    grad_edge = cuboid_gradient_g_free(center, lengths, rho, target_edge)[0]

    # Reference independent volume integral for Gamma_xy
    def integrand_gxy(z, y, x):
        d = np.array([x - 1.0, y - 1.0, z - 2.0])
        r = np.linalg.norm(d)
        return 3.0 * d[0] * d[1] / (r**5)

    ref_gxy, _ = integrate.tplquad(integrand_gxy, -1, 1, -1, 1, -1, 1, epsabs=1e-12, epsrel=1e-11)
    np.testing.assert_allclose(grad_edge[0, 1], ref_gxy, rtol=1e-10, atol=1e-12)
    assert np.isclose(grad_edge[0, 1], 0.24855895679004425, atol=1e-12)

    # Full tensor trace check (exterior point -> Laplace tr(Gamma) == 0)
    assert np.isclose(np.trace(grad_edge), 0.0, atol=1e-14)

    # 2. Permutations and reflections
    test_cases = [
        ([1.0, 2.0, 1.0], (0, 2)),  # permutation: edge along y
        ([2.0, 1.0, 1.0], (1, 2)),  # permutation: edge along x
        ([-1.0, 1.0, 2.0], (0, 1)), # reflection in x
        ([1.0, -1.0, 2.0], (0, 1)), # reflection in y
        ([1.0, 1.0, -2.0], (0, 1)), # reflection in z
    ]
    for pt, offdiag in test_cases:
        target_arr = np.array([pt])
        g = cuboid_gradient_g_free(center, lengths, rho, target_arr)[0]
        # Trace must be zero
        assert np.isclose(np.trace(g), 0.0, atol=1e-14)
        # Symmetry must hold
        np.testing.assert_allclose(g, g.T, atol=1e-14)
        # Compare off-diagonal with tplquad
        i, j = offdiag
        def integrand_ij(z, y, x, i=i, j=j, p=pt):
            d = np.array([x - p[0], y - p[1], z - p[2]])
            r = np.linalg.norm(d)
            return 3.0 * d[i] * d[j] / (r**5)
        ref_val, _ = integrate.tplquad(integrand_ij, -1, 1, -1, 1, -1, 1, epsabs=1e-9, epsrel=1e-8)
        np.testing.assert_allclose(g[i, j], ref_val, rtol=1e-6, atol=1e-8)

    # 3. Continuity at nearby points
    eps_offsets = [1e-4, -1e-4]
    for eps in eps_offsets:
        near_target = np.array([[1.0 + eps, 1.0 + eps, 2.0]])
        near_g = cuboid_gradient_g_free(center, lengths, rho, near_target)[0]
        np.testing.assert_allclose(near_g[0, 1], grad_edge[0, 1], rtol=1e-3)


def test_cuboid_independent_quadrature_interior():
    r"""Verify potential, full acceleration vector, and full gradient tensor at an interior point.

    The gradient and acceleration are verified against non-singular 2D surface integrals
    over the 6 bounding faces of the cuboid derived via the divergence theorem:
      a_i = -rho * \int_{partial V} (n_i / |r' - r_t|) dS'
      Gamma_ij = -rho * \int_{partial V} ((x'_j - x_{t,j}) / |r' - r_t|^3) n_i dS'
    which completely avoids the 1/R^3 and 3*x_i*x_j/R^5 singular volume kernels.
    """
    pytest.importorskip("scipy")
    from scipy import integrate

    lengths = np.array([2.0, 3.0, 4.0])
    half = lengths / 2.0
    center = np.zeros(3)
    rho = 2500.0
    target = np.array([0.2, 0.3, 0.4])
    xt, yt, zt = target

    obs = cuboid_observables_g_free(center, lengths, rho, np.array([target]))

    # 1. Potential: 3D volume integral -rho * int 1/R dV
    def integrand_phi(z, y, x):
        d = np.array([x - xt, y - yt, z - zt])
        return 1.0 / np.linalg.norm(d)

    val_phi, _ = integrate.tplquad(
        integrand_phi,
        -half[0], half[0],
        -half[1], half[1],
        -half[2], half[2],
        epsabs=1e-12, epsrel=1e-11,
    )
    want_phi = -rho * val_phi
    np.testing.assert_allclose(obs.potential[0], want_phi, rtol=1e-8)

    # 2. Acceleration: all 3 components via non-singular 2D surface integration over bounding faces
    ref_a = np.zeros(3)
    for i in range(3):
        axes = [0, 1, 2]
        axes.remove(i)
        ax1, ax2 = axes

        def face_pos(u2, u1, i=i, ax1=ax1, ax2=ax2):
            pt = [0.0, 0.0, 0.0]
            pt[i] = half[i]
            pt[ax1] = u1
            pt[ax2] = u2
            d = np.array(pt) - target
            return 1.0 / np.linalg.norm(d)

        def face_neg(u2, u1, i=i, ax1=ax1, ax2=ax2):
            pt = [0.0, 0.0, 0.0]
            pt[i] = -half[i]
            pt[ax1] = u1
            pt[ax2] = u2
            d = np.array(pt) - target
            return -1.0 / np.linalg.norm(d)

        vp, _ = integrate.dblquad(face_pos, -half[ax1], half[ax1], -half[ax2], half[ax2], epsabs=1e-13, epsrel=1e-12)
        vm, _ = integrate.dblquad(face_neg, -half[ax1], half[ax1], -half[ax2], half[ax2], epsabs=1e-13, epsrel=1e-12)
        ref_a[i] = -rho * (vp + vm)

    np.testing.assert_allclose(obs.acceleration[0], ref_a, rtol=1e-10)

    # 3. Gradient: all 6 unique tensor components via non-singular 2D surface integration
    ref_g = np.zeros((3, 3))
    for i in range(3):
        axes = [0, 1, 2]
        axes.remove(i)
        ax1, ax2 = axes
        for j in range(3):
            def face_g_pos(u2, u1, i=i, j=j, ax1=ax1, ax2=ax2):
                pt = [0.0, 0.0, 0.0]
                pt[i] = half[i]
                pt[ax1] = u1
                pt[ax2] = u2
                d = np.array(pt) - target
                r = np.linalg.norm(d)
                return d[j] / (r**3)

            def face_g_neg(u2, u1, i=i, j=j, ax1=ax1, ax2=ax2):
                pt = [0.0, 0.0, 0.0]
                pt[i] = -half[i]
                pt[ax1] = u1
                pt[ax2] = u2
                d = np.array(pt) - target
                r = np.linalg.norm(d)
                return -d[j] / (r**3)

            vp, _ = integrate.dblquad(face_g_pos, -half[ax1], half[ax1], -half[ax2], half[ax2], epsabs=1e-13, epsrel=1e-12)
            vm, _ = integrate.dblquad(face_g_neg, -half[ax1], half[ax1], -half[ax2], half[ax2], epsabs=1e-13, epsrel=1e-12)
            ref_g[i, j] = -rho * (vp + vm)

    np.testing.assert_allclose(obs.gradient[0], ref_g, rtol=1e-10)

    # 4. Trace check (Poisson: tr(Gamma) == -4*pi*rho)
    np.testing.assert_allclose(np.trace(obs.gradient[0]), -4.0 * np.pi * rho, rtol=1e-12)


def test_cuboid_independent_quadrature_exterior():
    pytest.importorskip("scipy")
    from scipy import integrate

    lengths = np.array([3.0, 2.0, 1.5])
    center = np.zeros(3)
    target = np.array([4.0, -3.0, 2.5])
    hx, hy, hz = lengths / 2.0

    obs = cuboid_observables_g_free(center, lengths, RHO, np.array([target]))

    # Integrate potential: -rho * int 1/R dV
    def integrand_phi(z, y, x):
        d = np.array([x - target[0], y - target[1], z - target[2]])
        return 1.0 / np.linalg.norm(d)

    val_phi, _ = integrate.tplquad(
        integrand_phi, -hx, hx, -hy, hy, -hz, hz,
        epsabs=1e-12, epsrel=1e-11,
    )
    want_phi = -RHO * val_phi
    np.testing.assert_allclose(obs.potential[0], want_phi, rtol=1e-7)

    # Integrate acceleration: rho * int (x' - target) / R^3 dV
    for comp in range(3):
        def integrand_a(z, y, x, c=comp):
            d = np.array([x - target[0], y - target[1], z - target[2]])
            return d[c] / (np.linalg.norm(d)**3)

        val_a, _ = integrate.tplquad(
            integrand_a, -hx, hx, -hy, hy, -hz, hz,
            epsabs=1e-12, epsrel=1e-11,
        )
        want_a = RHO * val_a
        np.testing.assert_allclose(obs.acceleration[0, comp], want_a, rtol=1e-7)

    # Integrate off-diagonal gradient Gamma_xy: rho * int 3 (x' - target_x)(y' - target_y) / R^5 dV
    def integrand_gxy(z, y, x):
        d = np.array([x - target[0], y - target[1], z - target[2]])
        return 3.0 * d[0] * d[1] / (np.linalg.norm(d)**5)

    val_gxy, _ = integrate.tplquad(
        integrand_gxy, -hx, hx, -hy, hy, -hz, hz,
        epsabs=1e-12, epsrel=1e-11,
    )
    want_gxy = RHO * val_gxy
    np.testing.assert_allclose(obs.gradient[0, 0, 1], want_gxy, rtol=1e-7)


def test_cuboid_input_validation_rejection():
    """Reject boolean, complex coordinates, and invalid shapes."""
    center = [0.0, 0.0, 0.0]
    lengths = [2.0, 2.0, 2.0]
    targets = [[1.0, 2.0, 3.0]]

    # Rejection of bool before numeric cast
    with pytest.raises(TypeError, match="must not be boolean"):
        cuboid_observables_g_free([True, 0.0, 0.0], lengths, RHO, targets)
    with pytest.raises(TypeError, match="must not be boolean"):
        cuboid_observables_g_free(center, [2.0, False, 2.0], RHO, targets)
    with pytest.raises(TypeError, match="must not be boolean"):
        cuboid_observables_g_free(center, lengths, True, targets)
    with pytest.raises(TypeError, match="must not be boolean"):
        cuboid_observables_g_free(center, lengths, RHO, [[1.0, True, 3.0]])

    # Rejection of complex coordinates
    with pytest.raises(TypeError, match="must be real"):
        cuboid_observables_g_free([1.0 + 1j, 0.0, 0.0], lengths, RHO, targets)
    with pytest.raises(TypeError, match="must be real"):
        cuboid_observables_g_free(center, [2.0, 2.0 + 1j, 2.0], RHO, targets)
    with pytest.raises(TypeError, match="must be real"):
        cuboid_observables_g_free(center, lengths, RHO, [[1.0, 2.0 + 1j, 3.0]])


def test_cuboid_si_entrypoint():
    lengths = np.array([2.0, 3.0, 4.0])
    center = np.array([0.5, -0.5, 1.0])
    targets = np.array([[1.0, 2.0, 3.0]])

    si_obs = cuboid_observables(center, lengths, RHO, targets)
    g_free = cuboid_observables_g_free(center, lengths, RHO, targets)

    assert isinstance(si_obs, GravityObservables)
    np.testing.assert_allclose(si_obs.potential, G_SI * g_free.potential)
    np.testing.assert_allclose(si_obs.acceleration, G_SI * g_free.acceleration)
    np.testing.assert_allclose(si_obs.gradient, G_SI * g_free.gradient)

