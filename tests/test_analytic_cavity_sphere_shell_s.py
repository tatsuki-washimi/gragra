"""Tests for finite spherical shell S-wave acceleration component decomposition."""

import numpy as np
import pytest

from gragra.analytic import (
    GFreeSphericalShellSComponents,
    spherical_cavity_wall,
    spherical_shell_s_acceleration_components_g_free,
    spherical_shell_s_infinite_difference_g_free,
)
from gragra.constants import G_SI
from gragra.observables import (
    SphericalShellSComponents,
    spherical_shell_s_acceleration_components,
    spherical_shell_s_infinite_difference,
)

A_INNER = 15.0   # inner cavity radius a in meters
R_OUTER = 120.0  # outer shell radius R in meters
RHO = 2500.0     # density in kg/m^3
DISP = 1.5e-6    # displacement amplitude in meters


def _orthogonal_pair():
    """A convenient exactly-orthogonal (propagation, polarization) pair for reuse."""
    n = np.array([1.0, 0.0, 0.0])
    e = np.array([0.0, 1.0, 0.0])
    return n, e


def test_spherical_shell_s_decomposition_identity():
    """Verify total == bulk + inner_wall + outer_boundary, and bulk is identically zero."""
    n_hat, e_hat = _orthogonal_pair()
    k_vals = [0.005, 0.02, 0.05, 0.1]

    comps = spherical_shell_s_acceleration_components_g_free(
        inner_radius_m=A_INNER,
        outer_radius_m=R_OUTER,
        wavenumber_rad_m=k_vals,
        density_kg_m3=RHO,
        displacement_m=DISP,
        propagation_direction=n_hat,
        polarization_direction=e_hat,
    )

    assert isinstance(comps, GFreeSphericalShellSComponents)
    assert comps.unit_system == "g_free"
    assert comps.bulk.shape == (len(k_vals), 1, 3)
    assert comps.total.shape == (len(k_vals), 1, 3)

    np.testing.assert_array_equal(comps.bulk, np.zeros_like(comps.bulk))

    sum_parts = comps.bulk + comps.inner_wall + comps.outer_boundary
    np.testing.assert_allclose(
        comps.total, sum_parts, rtol=1e-14, atol=1e-14 * np.max(np.abs(comps.total))
    )


def test_shell_s_direction_normalization_is_scale_safe():
    """Verify direction vectors normalize consistently regardless of overall scale.

    A naive ``vec / np.linalg.norm(vec)`` overflows to the zero vector for a
    finite component around 1e200 (its square overflows float64) and
    underflows the norm to exactly 0.0 for a finite component around 1e-200
    (its square underflows to 0.0), so those two regimes previously (a)
    silently produced a wrong (zero) polarization instead of the intended
    direction, and (b) raised a spurious "must be non-zero" ValueError,
    respectively. Both must now match the same unscaled direction.
    """
    n_hat, e_hat = _orthogonal_pair()
    k = 0.02

    baseline = spherical_shell_s_acceleration_components_g_free(
        A_INNER, R_OUTER, k, RHO, DISP, n_hat, e_hat
    )

    huge = spherical_shell_s_acceleration_components_g_free(
        A_INNER, R_OUTER, k, RHO, DISP, n_hat, e_hat * 1e200
    )
    np.testing.assert_allclose(huge.total, baseline.total, rtol=1e-13)
    np.testing.assert_allclose(huge.inner_wall, baseline.inner_wall, rtol=1e-13)

    tiny = spherical_shell_s_acceleration_components_g_free(
        A_INNER, R_OUTER, k, RHO, DISP, n_hat, e_hat * 1e-200
    )
    np.testing.assert_allclose(tiny.total, baseline.total, rtol=1e-13)
    np.testing.assert_allclose(tiny.inner_wall, baseline.inner_wall, rtol=1e-13)

    huge_prop = spherical_shell_s_acceleration_components_g_free(
        A_INNER, R_OUTER, k, RHO, DISP, n_hat * 1e200, e_hat
    )
    np.testing.assert_allclose(huge_prop.total, baseline.total, rtol=1e-13)

    # A large-magnitude propagation direction that is only *nearly*
    # orthogonal (not exactly) must still be caught by the transversality
    # check -- overflowing its norm to zero must not silently zero out the
    # dot product and let a non-orthogonal pair through.
    with pytest.raises(ValueError, match="orthogonal"):
        spherical_shell_s_acceleration_components_g_free(
            A_INNER, R_OUTER, k, RHO, DISP, (e_hat + 0.1 * n_hat) * 1e200, e_hat
        )


def test_spherical_shell_s_orthogonality_validation():
    """Verify transversality is enforced: non-orthogonal inputs rejected, near-orthogonal accepted."""
    n_hat, e_hat = _orthogonal_pair()

    # Exactly parallel (non-orthogonal): must raise.
    with pytest.raises(ValueError, match="orthogonal"):
        spherical_shell_s_acceleration_components_g_free(
            A_INNER, R_OUTER, 0.02, RHO, DISP, n_hat, n_hat
        )

    # Exactly orthogonal: must succeed.
    comps = spherical_shell_s_acceleration_components_g_free(
        A_INNER, R_OUTER, 0.02, RHO, DISP, n_hat, e_hat
    )
    assert comps.total.shape == (1, 3)

    # Nearly orthogonal within default tolerance (1e-5): must succeed.
    tiny_tilt = e_hat + 1e-6 * n_hat
    comps_tilt = spherical_shell_s_acceleration_components_g_free(
        A_INNER, R_OUTER, 0.02, RHO, DISP, n_hat, tiny_tilt
    )
    assert comps_tilt.total.shape == (1, 3)

    # Beyond tolerance: must raise.
    with pytest.raises(ValueError, match="orthogonal"):
        spherical_shell_s_acceleration_components_g_free(
            A_INNER, R_OUTER, 0.02, RHO, DISP, n_hat, e_hat + 0.1 * n_hat
        )


def test_shell_s_propagation_direction_invariance():
    """Verify the closed form has no explicit dependence on propagation_direction.

    Fixing the polarization direction e_hat_S and varying propagation_direction
    (n_hat) across several vectors orthogonal to e_hat_S must leave
    inner_wall/outer_boundary/total unchanged, per the rotational-symmetry
    argument documented in spherical_shell_s_acceleration_components_g_free.
    """
    e_hat = np.array([0.0, 1.0, 0.0])
    n_options = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0),
        np.array([3.0, 0.0, -4.0]) / 5.0,
    ]
    k = 0.03

    results = [
        spherical_shell_s_acceleration_components_g_free(
            A_INNER, R_OUTER, k, RHO, DISP, n, e_hat
        )
        for n in n_options
    ]

    reference = results[0]
    for comps in results[1:]:
        np.testing.assert_allclose(comps.inner_wall, reference.inner_wall, rtol=1e-14)
        np.testing.assert_allclose(comps.outer_boundary, reference.outer_boundary, rtol=1e-14)
        np.testing.assert_allclose(comps.total, reference.total, rtol=1e-14)


def test_spherical_shell_s_k_zero_limit():
    """Verify at k=0: bulk=0, total=0, inner_wall = -(4pi/3) rho xi e, outer_boundary = +(4pi/3) rho xi e."""
    n_hat, e_hat = _orthogonal_pair()

    comps = spherical_shell_s_acceleration_components_g_free(
        A_INNER, R_OUTER, 0.0, RHO, DISP, n_hat, e_hat
    )

    assert comps.total.shape == (1, 3)
    np.testing.assert_allclose(comps.bulk, np.zeros((1, 3)), atol=1e-14)
    np.testing.assert_allclose(comps.total, np.zeros((1, 3)), atol=1e-14)

    expected_inner = (-4.0 * np.pi * RHO / 3.0) * DISP * e_hat
    expected_outer = (+4.0 * np.pi * RHO / 3.0) * DISP * e_hat

    np.testing.assert_allclose(comps.inner_wall[0], expected_inner, rtol=1e-12)
    np.testing.assert_allclose(comps.outer_boundary[0], expected_outer, rtol=1e-12)


def test_spherical_shell_s_finite_kr_exact_identity():
    """Verify finite.total + infinite_difference == infinite-cavity wall(S) at several finite kR.

    This is the primary R -> infinity consistency check (per plan review), not a
    single-huge-R spot check: it is an *exact* algebraic identity that must hold
    for every k > 0, derived from ANNA v4 Eq.(71)/(72) (see
    spherical_shell_s_infinite_difference_g_free docstring).
    """
    n_hat, e_hat = _orthogonal_pair()

    for k in [1e-3, 0.01, 0.1, 0.5, 1.0, 3.0, 10.0]:
        finite = spherical_shell_s_acceleration_components_g_free(
            A_INNER, R_OUTER, k, RHO, DISP, n_hat, e_hat
        ).total[0]
        diff = spherical_shell_s_infinite_difference_g_free(
            A_INNER, R_OUTER, k, RHO, DISP, n_hat, e_hat
        )[0]
        inf_wall = spherical_cavity_wall(A_INNER, k, RHO, DISP, wave_type="S") * e_hat

        np.testing.assert_allclose(finite + diff, inf_wall, rtol=1e-11, atol=1e-20)


def test_spherical_shell_s_outer_term_analytic_upper_bound():
    """Verify ||outer_boundary|| <= 4*pi*rho*|xi| * (1/(kR)^2 + 1/(kR)^3) as R grows.

    Uses the analytic bound |j1(x)/x| <= 1/x^2 + 1/x^3 (from
    j1(x) = sin(x)/x^2 - cos(x)/x), so Bessel oscillation is fully accounted
    for and no monotone-decay assumption is required.
    """
    n_hat, e_hat = _orthogonal_pair()
    k = 0.05

    for R in [50.0, 200.0, 800.0, 3200.0, 12800.0]:
        comps = spherical_shell_s_acceleration_components_g_free(
            A_INNER, R, k, RHO, DISP, n_hat, e_hat
        )
        x = k * R
        bound = 4.0 * np.pi * RHO * abs(DISP) * (1.0 / x**2 + 1.0 / x**3)
        norm_outer = np.linalg.norm(comps.outer_boundary[0])
        assert norm_outer <= bound * (1.0 + 1e-12)


def test_spherical_shell_s_infinite_difference_helper():
    """Verify infinite_difference == -4*pi*rho*f(kR)*xi*e and matches inf - finite; k<=0 rejected."""
    n_hat, e_hat = _orthogonal_pair()
    k = 0.03

    diff = spherical_shell_s_infinite_difference_g_free(
        A_INNER, R_OUTER, k, RHO, DISP, n_hat, e_hat
    )
    comps = spherical_shell_s_acceleration_components_g_free(
        A_INNER, R_OUTER, k, RHO, DISP, n_hat, e_hat
    )
    inf_vec = spherical_cavity_wall(A_INNER, k, RHO, DISP, wave_type="S") * e_hat

    np.testing.assert_allclose(diff[0], inf_vec - comps.total[0], rtol=1e-12)

    with pytest.raises(ValueError, match="strictly positive"):
        spherical_shell_s_infinite_difference_g_free(
            A_INNER, R_OUTER, 0.0, RHO, DISP, n_hat, e_hat
        )


def test_spherical_shell_s_complex_phasor():
    """Verify complex displacement amplitude preserves phase and gives complex128 output."""
    n_hat, e_hat = _orthogonal_pair()
    k = 0.04
    phase = np.exp(1j * np.pi / 3.0)
    disp_c = DISP * phase

    comps_c = spherical_shell_s_acceleration_components_g_free(
        A_INNER, R_OUTER, k, RHO, disp_c, n_hat, e_hat
    )
    comps_r = spherical_shell_s_acceleration_components_g_free(
        A_INNER, R_OUTER, k, RHO, DISP, n_hat, e_hat
    )

    assert comps_c.total.dtype == np.complex128
    np.testing.assert_allclose(comps_c.inner_wall, comps_r.inner_wall * phase, rtol=1e-14)
    np.testing.assert_allclose(comps_c.total, comps_r.total * phase, rtol=1e-14)


def test_spherical_shell_s_independent_quadrature():
    """Verify inner_wall and outer_boundary against a direct Eq.(22)-style surface quadrature.

    Does not go through any production closed-form helper (e.g. _j1_over_x) or
    a "surface density sigma -> Newton kernel" abstraction to generate the
    expected values. Instead, the ANNA v4 Eq.(22) vector integrand

        I(r) = rho * integral_{S_r} (s / |s|^3) * [u(s) . n] dA

    is coded directly, with u(s) = xi * e_hat_S * exp(+i k (n_hat . s)) (gragra
    phase convention e^{+i k.(x - x_ref)}, x_ref = shell center = origin), and
    the material-region outward normals n_inner = -r_hat (at r=a) /
    n_outer = +r_hat (at r=R). Coordinates are chosen with n_hat along z and
    e_hat_S along x (both orthogonal by construction) so that
    r_hat(theta, phi) = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
    gives n_hat . s = r*cos(theta) and e_hat_S . r_hat = sin(theta)cos(phi).
    Only the x-component of I(r) is nonzero by symmetry; it is verified against
    the analytic inner_wall_x / outer_boundary_x.
    """
    pytest.importorskip("scipy")
    from scipy import integrate

    k = 0.02
    a = 10.0
    R = 40.0
    rho = 2000.0
    xi = 1e-5
    n_hat = np.array([0.0, 0.0, 1.0])
    e_hat = np.array([1.0, 0.0, 0.0])

    comps = spherical_shell_s_acceleration_components_g_free(
        a, R, k, rho, xi, n_hat, e_hat
    )

    # scipy.integrate.dblquad(func, a, b, gfun, hfun) integrates the outer
    # variable x over [a, b] and calls func(y, x) with the inner variable y
    # ranging over [gfun(x), hfun(x)]. Below, theta is the outer variable
    # ([0, pi]) and phi is the inner variable ([0, 2*pi]), so the integrand
    # signature must be (phi, theta, ...), not (theta, phi, ...).
    def real_integrand(phi, theta, r, sign):
        # sign = -1 for inner wall (n = -r_hat), +1 for outer boundary (n = +r_hat)
        return (
            sign
            * np.sin(theta) ** 3
            * np.cos(phi) ** 2
            * np.cos(k * r * np.cos(theta))
        )

    def imag_integrand(phi, theta, r, sign):
        return (
            sign
            * np.sin(theta) ** 3
            * np.cos(phi) ** 2
            * np.sin(k * r * np.cos(theta))
        )

    # Inner wall (r=a, n_inner=-r_hat -> sign=-1):
    val_inner_re, _ = integrate.dblquad(
        real_integrand, 0.0, np.pi, 0.0, 2.0 * np.pi, args=(a, -1.0),
        epsabs=1e-13, epsrel=1e-12,
    )
    val_inner_im, _ = integrate.dblquad(
        imag_integrand, 0.0, np.pi, 0.0, 2.0 * np.pi, args=(a, -1.0),
        epsabs=1e-13, epsrel=1e-12,
    )
    want_inner_x = rho * xi * val_inner_re
    np.testing.assert_allclose(val_inner_im, 0.0, atol=1e-12)
    np.testing.assert_allclose(comps.inner_wall[0, 0], want_inner_x, rtol=1e-9)
    np.testing.assert_allclose(comps.inner_wall[0, 1:], 0.0, atol=1e-18)

    # Outer boundary (r=R, n_outer=+r_hat -> sign=+1):
    val_outer_re, _ = integrate.dblquad(
        real_integrand, 0.0, np.pi, 0.0, 2.0 * np.pi, args=(R, 1.0),
        epsabs=1e-13, epsrel=1e-12,
    )
    val_outer_im, _ = integrate.dblquad(
        imag_integrand, 0.0, np.pi, 0.0, 2.0 * np.pi, args=(R, 1.0),
        epsabs=1e-13, epsrel=1e-12,
    )
    want_outer_x = rho * xi * val_outer_re
    np.testing.assert_allclose(val_outer_im, 0.0, atol=1e-12)
    np.testing.assert_allclose(comps.outer_boundary[0, 0], want_outer_x, rtol=1e-9)
    np.testing.assert_allclose(comps.outer_boundary[0, 1:], 0.0, atol=1e-18)


def test_spherical_shell_s_zero_density():
    """Verify zero background density produces identically zero acceleration for all components."""
    n_hat, e_hat = _orthogonal_pair()

    comps_zero = spherical_shell_s_acceleration_components_g_free(
        A_INNER, R_OUTER, 0.03, 0.0, DISP, n_hat, e_hat
    )
    np.testing.assert_array_equal(comps_zero.bulk, np.zeros((1, 3)))
    np.testing.assert_array_equal(comps_zero.inner_wall, np.zeros((1, 3)))
    np.testing.assert_array_equal(comps_zero.outer_boundary, np.zeros((1, 3)))
    np.testing.assert_array_equal(comps_zero.total, np.zeros((1, 3)))


def test_spherical_shell_s_invalid_inputs():
    n_hat, e_hat = _orthogonal_pair()

    # R <= a
    with pytest.raises(ValueError, match="strictly greater than"):
        spherical_shell_s_acceleration_components_g_free(20.0, 10.0, 0.01, RHO, DISP, n_hat, e_hat)
    with pytest.raises(ValueError, match="strictly greater than"):
        spherical_shell_s_acceleration_components_g_free(20.0, 20.0, 0.01, RHO, DISP, n_hat, e_hat)

    # negative k
    with pytest.raises(ValueError, match="non-negative"):
        spherical_shell_s_acceleration_components_g_free(10.0, 20.0, -0.01, RHO, DISP, n_hat, e_hat)

    # negative density
    with pytest.raises(ValueError, match="non-negative"):
        spherical_shell_s_acceleration_components_g_free(10.0, 20.0, 0.01, -1.0, DISP, n_hat, e_hat)

    # zero-vector directions
    with pytest.raises(ValueError, match="non-zero"):
        spherical_shell_s_acceleration_components_g_free(10.0, 20.0, 0.01, RHO, DISP, [0, 0, 0], e_hat)
    with pytest.raises(ValueError, match="non-zero"):
        spherical_shell_s_acceleration_components_g_free(10.0, 20.0, 0.01, RHO, DISP, n_hat, [0, 0, 0])

    # Boolean rejection
    with pytest.raises(TypeError, match="must not be boolean"):
        spherical_shell_s_acceleration_components_g_free(True, 20.0, 0.01, RHO, DISP, n_hat, e_hat)
    with pytest.raises(TypeError, match="must not be boolean"):
        spherical_shell_s_acceleration_components_g_free(10.0, False, 0.01, RHO, DISP, n_hat, e_hat)
    with pytest.raises(TypeError, match="must not be boolean"):
        spherical_shell_s_acceleration_components_g_free(10.0, 20.0, True, RHO, DISP, n_hat, e_hat)
    with pytest.raises(TypeError, match="must not be boolean"):
        spherical_shell_s_acceleration_components_g_free(10.0, 20.0, 0.01, True, DISP, n_hat, e_hat)
    with pytest.raises(TypeError, match="must not be boolean"):
        spherical_shell_s_acceleration_components_g_free(
            10.0, 20.0, 0.01, RHO, DISP, [True, False, False], e_hat
        )
    with pytest.raises(TypeError, match="must not be boolean"):
        spherical_shell_s_acceleration_components_g_free(
            10.0, 20.0, 0.01, RHO, DISP, n_hat, [True, False, False]
        )

    # Complex rejection (radii, wavenumber, density, both direction vectors must be real)
    with pytest.raises(TypeError, match="must be real"):
        spherical_shell_s_acceleration_components_g_free(10.0 + 1j, 20.0, 0.01, RHO, DISP, n_hat, e_hat)
    with pytest.raises(TypeError, match="must be real"):
        spherical_shell_s_acceleration_components_g_free(10.0, 20.0 + 1j, 0.01, RHO, DISP, n_hat, e_hat)
    with pytest.raises(TypeError, match="must be real"):
        spherical_shell_s_acceleration_components_g_free(10.0, 20.0, 0.01 + 1j, RHO, DISP, n_hat, e_hat)
    with pytest.raises(TypeError, match="must be real"):
        spherical_shell_s_acceleration_components_g_free(10.0, 20.0, 0.01, 2000.0 + 1j, DISP, n_hat, e_hat)
    with pytest.raises(TypeError, match="must be real"):
        spherical_shell_s_acceleration_components_g_free(10.0, 20.0, 0.01, RHO, DISP, [1j, 0.0, 0.0], e_hat)
    with pytest.raises(TypeError, match="must be real"):
        spherical_shell_s_acceleration_components_g_free(10.0, 20.0, 0.01, RHO, DISP, n_hat, [1j, 0.0, 0.0])

    # orthogonality_tol input contract: bool / non-finite / out-of-range rejected
    with pytest.raises(TypeError, match="must not be boolean"):
        spherical_shell_s_acceleration_components_g_free(
            10.0, 20.0, 0.01, RHO, DISP, n_hat, e_hat, orthogonality_tol=True
        )
    with pytest.raises(ValueError, match="finite"):
        spherical_shell_s_acceleration_components_g_free(
            10.0, 20.0, 0.01, RHO, DISP, n_hat, e_hat, orthogonality_tol=float("inf")
        )
    with pytest.raises(ValueError, match=r"\(0, 1\.0\)"):
        spherical_shell_s_acceleration_components_g_free(
            10.0, 20.0, 0.01, RHO, DISP, n_hat, e_hat, orthogonality_tol=0.0
        )
    with pytest.raises(ValueError, match=r"\(0, 1\.0\)"):
        spherical_shell_s_acceleration_components_g_free(
            10.0, 20.0, 0.01, RHO, DISP, n_hat, e_hat, orthogonality_tol=-0.5
        )
    with pytest.raises(ValueError, match=r"\(0, 1\.0\)"):
        spherical_shell_s_acceleration_components_g_free(
            10.0, 20.0, 0.01, RHO, DISP, n_hat, e_hat, orthogonality_tol=1.5
        )
    # The upper bound is now strict (0, 1.0), not (0, 1.0]: orthogonality_tol
    # == 1.0 itself must be rejected, so that no valid tolerance can ever let
    # an exactly parallel/antiparallel pair (|dot| == 1.0 exactly) through.
    with pytest.raises(ValueError, match=r"\(0, 1\.0\)"):
        spherical_shell_s_acceleration_components_g_free(
            10.0, 20.0, 0.01, RHO, DISP, n_hat, e_hat, orthogonality_tol=1.0
        )


def test_spherical_shell_s_exact_parallel_always_rejected():
    """Verify no valid orthogonality_tol lets an exactly parallel/antiparallel pair through.

    orthogonality_tol's upper bound is a strict (0, 1.0), and there is also
    a hard, tolerance-independent rejection for abs(dot(n_hat, e_hat)) >
    1.0 - 1e-9. Both matter: the strict upper bound alone is not sufficient
    once floating-point normalization noise is accounted for, because
    dot_np for a genuinely parallel pair is not guaranteed to be exactly
    1.0 -- normalizing [1, 1, 0] against itself gives 0.9999999999999998
    (two ULPs below 1.0). A permissive orthogonality_tol requested at
    np.nextafter(1.0, 0.0) (one ULP below 1.0) would sit *above* that
    dot_np, so `dot_np > orthogonality_tol` alone would silently accept it;
    the hard margin catches it regardless.
    """
    n_hat, e_hat = _orthogonal_pair()

    with pytest.raises(ValueError, match="orthogonal"):
        spherical_shell_s_acceleration_components_g_free(
            A_INNER, R_OUTER, 0.02, RHO, DISP, n_hat, n_hat, orthogonality_tol=0.999999
        )
    # Antiparallel: |dot(n_hat, -n_hat)| == 1.0 too.
    with pytest.raises(ValueError, match="orthogonal"):
        spherical_shell_s_acceleration_components_g_free(
            A_INNER, R_OUTER, 0.02, RHO, DISP, n_hat, -n_hat, orthogonality_tol=0.999999
        )

    # The floating-point edge case: [1, 1, 0] normalized against itself has
    # dot_np == 0.9999999999999998, strictly below np.nextafter(1.0, 0.0)
    # == 0.9999999999999999 -- so the tol-only check `dot_np >
    # orthogonality_tol` would NOT trigger here. The hard
    # _PARALLEL_REJECT_MARGIN-based guard must still reject it.
    v = np.array([1.0, 1.0, 0.0])
    tight_tol = float(np.nextafter(1.0, 0.0))
    with pytest.raises(ValueError, match="parallel"):
        spherical_shell_s_acceleration_components_g_free(
            A_INNER, R_OUTER, 0.02, RHO, DISP, v, v, orthogonality_tol=tight_tol
        )
    with pytest.raises(ValueError, match="parallel"):
        spherical_shell_s_acceleration_components_g_free(
            A_INNER, R_OUTER, 0.02, RHO, DISP, v, -v, orthogonality_tol=tight_tol
        )


def test_spherical_shell_s_si_entrypoint():
    n_hat, e_hat = _orthogonal_pair()
    k = 0.02

    si_comps = spherical_shell_s_acceleration_components(
        A_INNER, R_OUTER, k, RHO, DISP, n_hat, e_hat
    )
    g_free = spherical_shell_s_acceleration_components_g_free(
        A_INNER, R_OUTER, k, RHO, DISP, n_hat, e_hat
    )

    assert isinstance(si_comps, SphericalShellSComponents)
    assert si_comps.unit_system == "si"
    np.testing.assert_allclose(si_comps.bulk, G_SI * g_free.bulk)
    np.testing.assert_allclose(si_comps.inner_wall, G_SI * g_free.inner_wall)
    np.testing.assert_allclose(si_comps.outer_boundary, G_SI * g_free.outer_boundary)
    np.testing.assert_allclose(si_comps.total, G_SI * g_free.total)

    si_diff = spherical_shell_s_infinite_difference(A_INNER, R_OUTER, k, RHO, DISP, n_hat, e_hat)
    g_diff = spherical_shell_s_infinite_difference_g_free(A_INNER, R_OUTER, k, RHO, DISP, n_hat, e_hat)
    np.testing.assert_allclose(si_diff, G_SI * g_diff)
