"""Independent verification oracle for LIT-B (Eq. 86-90).

The oracle separates the literal H19 Eq. (90) transcription from the
physical reflected-P coefficient derived from the same traction-free
displacement field. The private H19 helper is imported only to assert the
literature transcription; all boundary-value and Green-function expectations
are constructed independently of production's physical helper.

1. ``test_delta_matches_boundary_value_solve`` (fast): solves the free-surface
   traction-free boundary conditions (``sigma_zz = sigma_xz = 0`` at ``z=0``)
   directly via Helmholtz displacement potentials (2x2 linear system,
   ``numpy.linalg.solve``), then compares ``(1+PP) - i*(C/A)`` against the
   private literal-H19 helper and ``(1+PP) + i*(C/A)`` against the physical
   helper. This keeps the published coefficient and the production physical
   coefficient as two explicit contracts (see ``docs/design/literature-b.md``).

2. ``test_green_function_reconstructs_single_p_potential`` (slow,
   ``scipy.integrate.quad``): re-derives the single-P ground/underground
   potential from the horizontal-Fourier-mode reduction of Poisson's
   equation, ``(d^2/dz^2 - k_rho^2) Phi_hat(z) = 4*pi*delta_rho_hat(z)``,
   Green's function ``G(z,z') = -exp(-k_rho|z-z'|)/(2*k_rho)``, with the bulk
   source ``delta_rho = -rho*div(xi_P)`` (``xi_P`` the P-wave's own
   displacement field, polarization ``sin(theta)*h + cos(theta)*e_z``) and a
   free-surface mass sheet ``sigma = rho*xi0_p*cos(theta)``. This is
   independent of Eq. (86)/(87)'s closed form; it reconstructs the field
   from Poisson's equation and the wave's own displacement, not from
   production's amplitude prefactors.

3. ``test_green_function_reconstructs_reflected_p_composite`` (slow,
   ``scipy.integrate.quad``): reconstructs the full reflected-P composite
   gravity potential (incident P + reflected P + converted SV) by integrating
   the bulk density ``delta_rho = -rho*div(xi_P)`` and the free-surface mass
   sheet ``sigma = rho*xi_z(0)`` of the independently-solved boundary-value
   displacement field over the half-space. The boundary-value coefficients
   (B, C) come from the same Helmholtz-potential solve as method 1; the
   integration is independent of production's Eq. (90) and its amplitude
   prefactors. This verifies the complete composite field including the
   converted SV contribution.

4. ``test_green_function_reconstructs_full_observables`` (slow,
   ``scipy.integrate.quad``): reconstructs potential, acceleration, and
   gravity gradient for single-P, single-SV, and reflected-P composite by
   integrating the same bulk density and surface sheet. The horizontal
   components of acceleration and gradient are evaluated analytically from the
   reconstructed potential (``a_x = -i*qx*Phi``, ``Gamma_xx = qx^2*Phi``,
   etc.); vertical components use numerical quadrature of ``dG/dz`` and the
   *regular part* of ``d^2G/dz^2`` (``k_rho^2*G(z,z')``; the distributional
   ``delta(z-z')`` content is added back explicitly for underground
   observation points inside the bulk source domain -- see the
   ``_d2g_dz2``/``_reconstruct_single_p_observables`` docstrings). The
   horizontal components agree to machine precision; vertical components are
   limited by ``scipy.integrate.quad``'s own achievable precision for these
   infinite-domain, partly-oscillatory integrals (measured, not assumed --
   see ``docs/validation/literature-b.md``).

5. ``test_*_exact_closed_form`` (fast, no ``scipy`` dependency): the bulk
   density here is a pure decaying/oscillating complex exponential in z', so
   its Green's-function convolution (and first/second z-derivatives) has an
   elementary closed form -- no numerical quadrature needed at all. For
   single-P/SV this is a full, unambiguous machine-precision independent
   check (ground AND underground, all angles). For the reflected-P composite
   it reconstructs the actual total surface displacement and therefore the
   physical coefficient used by production; a separate test keeps the
   literal Eq. (90) value as a non-production literature reference.

6. ``test_*_poisson_boundary_value_solution`` (fast): solves the horizontal-
   Fourier Poisson equation with a particular solution for each bulk source,
   one decaying homogeneous term in the lower half-space, and one decaying
   homogeneous term in the upper half-space. The two homogeneous coefficients
   are obtained directly from potential continuity and the surface-sheet
   normal-derivative jump, independently of the Green convolution.

**Reflected-P surface-source contract:** the boundary-value solve and the
Green-function reconstruction use the actual total vertical displacement at
the free surface. The old minus-convention comparison remains a regression
record of why the literal H19 coefficient and the physical coefficient must
not be conflated; it is not used as production's expected physical result.

   (a) The oracle's boundary-value solve returns (B, C) for
   ``psi_y = C * exp(i*kx*x - i*kzs*z)`` under the potential decomposition
   ``u = grad(phi) + curl(0, psi_y, 0)`` -- the SAME decomposition whose
   traction-free boundary conditions produce the coded 2x2 linear system.
   For THIS decomposition, the ACTUAL elastic-wavefield vertical
   displacement is ``xi_z(0) = xi0_p*[cos(theta)*(1-B) + sin(theta)*C]``
   (PLUS) -- confirmed three independent ways: matching the coded
   traction-free matrix (a11..a22) to sigma_zz=sigma_xz=0 term-by-term via
   exact algebra, a finite-difference evaluation of u_z(0) from phi, psi_y
   directly (exact-zero agreement, not merely small), and an independent
   external reviewer's from-scratch Cartesian-displacement/stress
   re-derivation (matching the coded matrix to ~2.9e-16).
   (b) Reconstructing the gravity potential as
   ``4*pi*[bulk(incident+reflected P divergence, using this SAME B) +
   G(z,0)*sigma]`` with the actual surface sheet
   ``sigma = rho*xi0_p*[cos(theta)*(1-B) + sin(theta)*C]`` produces the
   physical coefficient used by production. The literal Eq. (90) helper is
   retained separately and is tested against the minus-sign literature
   combination; this is a distinction between two contracts, not a claim
   that H19 contains a confirmed typographical error.

   The prior implementation used the minus combination in production and
   differed from the actual-source reconstruction by about 1.39 relatively
   at the representative hard-rock condition. That historical mismatch is
   now resolved in production by using the plus/physical coefficient, while
   the literal H19 value remains covered by
   ``test_delta_matches_boundary_value_solve``.

Scope note: this file does not re-derive the SV-mode potential via quadrature
(the SV surface mass-sheet formula is instead confirmed exactly, to O(eps)
Taylor precision, by the jump-condition test in
``tests/test_analytic_bodywave_literature.py::test_sv_surface_mass_sheet_jump``,
and to machine precision by Method 5's exact closed-form single-SV test --
the SV field has no bulk term to integrate, so these checks together cover
the same ground as a quadrature would for that mode).
"""

from __future__ import annotations

import numpy as np
import pytest

from gragra.analytic.bodywave_literature import (
    _harms2019_reflected_p_delta,
    _physical_reflected_p_delta,
    halfspace_reflected_p_above_observables_g_free,
    halfspace_single_bodywave_observables_g_free,
)

POISSON = {"name": "poisson", "v_p_m_s": 1500.0 * np.sqrt(3.0), "v_s_m_s": 1500.0}
HARD_ROCK = {"name": "hard_rock", "v_p_m_s": 5800.0, "v_s_m_s": 3200.0}
MATERIALS = [POISSON, HARD_ROCK]
ANGLES_DEG = [10.0, 45.0, 80.0]  # same three angles H19 Fig.10 plots
RHO = 2700.0
FREQ = 5.0


def _solve_reflection_coefficients(v_p, v_s, theta, rho=1.0, omega=1.0):
    """Solve free-surface traction-free BCs for incident P via Helmholtz
    potentials (see docs/design/literature-b.md). Returns (kP,kS,kx,kzP,kzS,B,C)
    with A (incident amplitude) = 1.
    """
    k_p = omega / v_p
    k_s = omega / v_s
    kx = k_p * np.sin(theta)
    kzp = np.sqrt(k_p**2 - kx**2 + 0j)
    kzs = np.sqrt(k_s**2 - kx**2 + 0j)
    lam = rho * (v_p**2 - 2 * v_s**2)
    mu = rho * v_s**2

    a11 = -lam * (kx**2 + kzp**2) - 2 * mu * kzp**2
    a12 = 2 * mu * kx * kzs
    rhs1 = lam * (kx**2 + kzp**2) + 2 * mu * kzp**2
    a21 = 2 * mu * kx * kzp
    a22 = mu * (kzs**2 - kx**2)
    rhs2 = 2 * mu * kx * kzp

    m = np.array([[a11, a12], [a21, a22]], dtype=complex)
    rhs = np.array([rhs1, rhs2], dtype=complex)
    b, c = np.linalg.solve(m, rhs)
    return k_p, k_s, kx, kzp, kzs, b, c


def _poisson_boundary_value_observables(
    z_target, *, qx, qy, source_terms, surface_density
):
    """Solve the horizontal-Fourier Poisson interface problem directly.

    ``source_terms`` contains ``(D, lam)`` pairs for bulk sources
    ``delta_rho_hat(z) = D*exp(lam*z)`` in ``z < 0``.  For each source the
    particular solution is ``P*exp(lam*z)``, where
    ``(lam**2-q**2)*P = 4*pi*D``.  The lower half-space then has one remaining
    homogeneous term ``H*exp(q*z)`` and the upper half-space has ``O*exp(-q*z)``.
    The two unknowns are solved from potential continuity and the surface-sheet
    derivative jump.  No production potential or Green-function helper enters
    this calculation.

    Returns ``((potential, acceleration, gradient), (O, H))``.  The horizontal
    phase is one at the target's reference point; ``qx`` and ``qy`` are kept so
    that the differentiated observables still test the Fourier algebra.
    """
    q = float(np.hypot(qx, qy))
    if q <= 0.0:
        raise ValueError("the Fourier boundary-value oracle requires q > 0")

    particular_terms = tuple(
        (
            4.0 * np.pi * amplitude / (exponent**2 - q**2),
            exponent,
        )
        for amplitude, exponent in source_terms
    )
    particular_value = sum(coefficient for coefficient, _ in particular_terms)
    particular_derivative = sum(
        exponent * coefficient for coefficient, exponent in particular_terms
    )

    # O - H = Phi_particular(0), from potential continuity,
    # -q*O - q*H - Phi_particular'(0) = 4*pi*sigma.
    interface_matrix = np.array([[1.0, -1.0], [-q, -q]], dtype=np.complex128)
    interface_rhs = np.array(
        [
            particular_value,
            4.0 * np.pi * surface_density + particular_derivative,
        ],
        dtype=np.complex128,
    )
    outside_coefficient, inside_homogeneous_coefficient = np.linalg.solve(
        interface_matrix, interface_rhs
    )

    if z_target > 0.0:
        evaluation_terms = ((outside_coefficient, -q),)
    elif z_target < 0.0:
        evaluation_terms = particular_terms + ((inside_homogeneous_coefficient, q),)
    else:
        raise ValueError("the Poisson boundary-value oracle excludes z == 0")

    potential = sum(
        coefficient * np.exp(exponent * z_target)
        for coefficient, exponent in evaluation_terms
    )
    first_z_derivative = sum(
        exponent * coefficient * np.exp(exponent * z_target)
        for coefficient, exponent in evaluation_terms
    )
    second_z_derivative = sum(
        exponent**2 * coefficient * np.exp(exponent * z_target)
        for coefficient, exponent in evaluation_terms
    )

    acceleration = np.array(
        [-1j * qx * potential, -1j * qy * potential, -first_z_derivative],
        dtype=np.complex128,
    )
    gradient = np.empty((3, 3), dtype=np.complex128)
    gradient[0, 0] = qx * qx * potential
    gradient[0, 1] = qx * qy * potential
    gradient[1, 0] = gradient[0, 1]
    gradient[1, 1] = qy * qy * potential
    gradient[0, 2] = -1j * qx * first_z_derivative
    gradient[2, 0] = gradient[0, 2]
    gradient[1, 2] = -1j * qy * first_z_derivative
    gradient[2, 1] = gradient[1, 2]
    gradient[2, 2] = -second_z_derivative
    return (potential, acceleration, gradient), (
        outside_coefficient,
        inside_homogeneous_coefficient,
    )


@pytest.mark.parametrize("material", MATERIALS, ids=lambda m: m["name"])
@pytest.mark.parametrize("theta_deg", ANGLES_DEG)
def test_delta_matches_boundary_value_solve(material, theta_deg):
    theta = np.deg2rad(theta_deg)
    omega = 2.0 * np.pi * FREQ
    k_p, k_s, kx, kzp, kzs, b, c = _solve_reflection_coefficients(
        material["v_p_m_s"], material["v_s_m_s"], theta, omega=omega
    )
    delta_boundary_value = (1.0 + b) - 1j * c

    denom = (k_s**2 - 2.0 * kx**2) ** 2 + 4.0 * kx**2 * kzp * kzs
    numer = 8.0 * kx**2 * kzp * kzs - 1j * 4.0 * kx * kzp * (k_s**2 - 2.0 * kx**2)
    delta_eq90 = numer / denom  # literal H19 Eq. (90) reference value

    assert abs(delta_boundary_value - delta_eq90) <= 1e-10 * abs(delta_eq90)

    delta_helper = _harms2019_reflected_p_delta(kx, kzp, kzs, k_s)
    assert abs(delta_helper - delta_eq90) <= 1e-14 * abs(delta_eq90)

    delta_physical = _physical_reflected_p_delta(kx, kzp, kzs, k_s)
    assert abs(delta_physical - ((1.0 + b) + 1j * c)) <= 1e-14 * abs(delta_physical)


@pytest.mark.parametrize("material", MATERIALS, ids=lambda m: m["name"])
@pytest.mark.parametrize("theta_deg", ANGLES_DEG)
def test_reflected_p_potential_matches_boundary_value_reconstruction(
    material, theta_deg
):
    """Reconstruct the physical Phi from the independently-solved (B, C).

    This uses the actual total surface displacement, which is distinct from
    the literal H19 Eq. (90) coefficient retained by the private literature
    helper.
    """
    theta = np.deg2rad(theta_deg)
    azimuth = np.deg2rad(15.0)
    xi0_p = 1e-9 + 0.4e-9j
    z_target = 7.0

    omega = 2.0 * np.pi * FREQ
    k_p, k_s, kx, kzp, kzs, b, c = _solve_reflection_coefficients(
        material["v_p_m_s"], material["v_s_m_s"], theta, omega=omega
    )
    delta = (1.0 + b) + 1j * c
    c3 = -RHO * xi0_p * delta / (1j * k_p)
    phi_expected = (
        2.0 * np.pi * c3 * np.exp(-kx * z_target)
    )  # at x=y=phase_origin -> phase=1

    r = halfspace_reflected_p_above_observables_g_free(
        [[0.0, 0.0, z_target]],
        density_kg_m3=RHO,
        v_p_m_s=material["v_p_m_s"],
        v_s_m_s=material["v_s_m_s"],
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        incident_displacement_m=xi0_p,
    )
    assert abs(r.potential[0] - phi_expected) <= 1e-10 * abs(phi_expected)


@pytest.mark.parametrize("material", MATERIALS, ids=lambda m: m["name"])
@pytest.mark.parametrize("theta_deg", ANGLES_DEG)
@pytest.mark.parametrize("azimuth", [0.0, np.pi / 4.0])
def test_reflected_p_observables_match_actual_displacement_reconstruction(
    material, theta_deg, azimuth
):
    """The reflected-P API must use the source's actual surface displacement.

    ``(1+B)+i*C`` is reconstructed from the independently solved elastic
    boundary-value coefficients.  The expected potential is then obtained
    from the P bulk density plus the actual total surface sheet, and the
    acceleration/gradient are differentiated independently from that field.
    """
    theta = np.deg2rad(theta_deg)
    xi0_p = 1e-9 + 0.4e-9j
    target = np.array([[0.37, -0.41, 7.0]])
    omega = 2.0 * np.pi * FREQ
    k_p, k_s, kx, kzp, kzs, b, c = _solve_reflection_coefficients(
        material["v_p_m_s"], material["v_s_m_s"], theta, omega=omega
    )

    # Independent boundary-value reconstruction of the physical coefficient:
    # the actual total vertical displacement is the PLUS combination, so the
    # Green-function bulk+surface sum gives (1+B)+i*C.
    delta_phys = (1.0 + b) + 1j * c
    phase = np.exp(
        1j * kx * (target[0, 0] * np.cos(azimuth) + target[0, 1] * np.sin(azimuth))
    )
    phi = (
        2.0
        * np.pi
        * (-RHO * xi0_p * delta_phys / (1j * k_p))
        * np.exp(-kx * target[0, 2])
        * phase
    )
    qx = kx * np.cos(azimuth)
    qy = kx * np.sin(azimuth)
    expected_acceleration = np.array([-1j * qx * phi, -1j * qy * phi, kx * phi])
    expected_gradient = np.array(
        [
            [qx * qx * phi, qx * qy * phi, 1j * qx * kx * phi],
            [qx * qy * phi, qy * qy * phi, 1j * qy * kx * phi],
            [1j * qx * kx * phi, 1j * qy * kx * phi, -kx * kx * phi],
        ]
    )

    actual = halfspace_reflected_p_above_observables_g_free(
        target,
        density_kg_m3=RHO,
        v_p_m_s=material["v_p_m_s"],
        v_s_m_s=material["v_s_m_s"],
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        incident_displacement_m=xi0_p,
    )

    np.testing.assert_allclose(actual.potential[0], phi, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(
        actual.acceleration[0], expected_acceleration, rtol=1e-10, atol=0.0
    )
    np.testing.assert_allclose(
        actual.gradient[0], expected_gradient, rtol=1e-9, atol=0.0
    )


@pytest.mark.parametrize("material", MATERIALS, ids=lambda m: m["name"])
@pytest.mark.parametrize("theta_deg", ANGLES_DEG)
def test_composite_traction_is_zero_at_solved_coefficients(material, theta_deg):
    """Consistency check on the oracle's own boundary-value solve: with the
    solved (B, C), the residual traction at z=0 is zero (not a check on
    production -- a check that _solve_reflection_coefficients itself is
    self-consistent)."""
    theta = np.deg2rad(theta_deg)
    v_p, v_s = material["v_p_m_s"], material["v_s_m_s"]
    omega = 2.0 * np.pi * FREQ
    k_p, k_s, kx, kzp, kzs, b, c = _solve_reflection_coefficients(
        v_p, v_s, theta, omega=omega
    )
    lam = v_p**2 - 2 * v_s**2
    mu = v_s**2
    sigma_zz0 = -lam * (1 + b) * (kx**2 + kzp**2) + 2 * mu * (
        c * kx * kzs - kzp**2 * (1 + b)
    )
    sigma_xz0 = mu * (-c * kx**2 + c * kzs**2 - 2 * kx * kzp * (1 - b))
    scale = mu * kzp**2
    assert abs(sigma_zz0) <= 1e-9 * scale
    assert abs(sigma_xz0) <= 1e-9 * scale


@pytest.mark.parametrize("z_target", [3.0, -3.0, 0.5, -0.5])
def test_single_p_poisson_boundary_value_solution_matches_production(z_target):
    """Solve the Poisson interface problem directly for a single P wave.

    The independent solve uses a decaying homogeneous solution on each side of
    the free surface, imposes potential continuity, and imposes the normal
    derivative jump from the surface sheet.  It is deliberately separate from
    the Green-function convolution and from the production amplitude formula.
    """
    theta = np.deg2rad(37.0)
    azimuth = np.deg2rad(15.0)
    xi0_p = 1.0e-9 + 0.4e-9j
    v_p, v_s = HARD_ROCK["v_p_m_s"], HARD_ROCK["v_s_m_s"]
    k_p = 2.0 * np.pi * FREQ / v_p
    k_rho = k_p * np.sin(theta)
    k_zp = k_p * np.cos(theta)
    delta_rho_amplitude = RHO * k_p**2 * xi0_p / (1j * k_p)
    sigma = RHO * xi0_p * np.cos(theta)

    actual, coefficients = _poisson_boundary_value_observables(
        z_target,
        qx=k_rho * np.cos(azimuth),
        qy=k_rho * np.sin(azimuth),
        source_terms=((delta_rho_amplitude, 1j * k_zp),),
        surface_density=sigma,
    )
    production = halfspace_single_bodywave_observables_g_free(
        [[0.0, 0.0, z_target]],
        wave_type="P",
        density_kg_m3=RHO,
        v_p_m_s=v_p,
        v_s_m_s=v_s,
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        displacement_m=xi0_p,
    )

    assert np.isfinite(coefficients[0]) and np.isfinite(coefficients[1])
    np.testing.assert_allclose(actual[0], production.potential[0], rtol=1e-10)
    np.testing.assert_allclose(actual[1], production.acceleration[0], rtol=1e-10)
    np.testing.assert_allclose(actual[2], production.gradient[0], rtol=1e-9)


@pytest.mark.parametrize("material", MATERIALS, ids=lambda m: m["name"])
@pytest.mark.parametrize("theta_deg", ANGLES_DEG)
def test_reflected_p_poisson_boundary_value_solution_matches_production(
    material, theta_deg
):
    """Solve the reflected-P Poisson interface problem from its source terms."""
    theta = np.deg2rad(theta_deg)
    azimuth = np.deg2rad(15.0)
    xi0_p = 1.0e-9 + 0.4e-9j
    z_target = 7.0
    omega = 2.0 * np.pi * FREQ
    k_p, _k_s, kx, kzp, _kzs, b, c = _solve_reflection_coefficients(
        material["v_p_m_s"], material["v_s_m_s"], theta, omega=omega
    )
    delta_rho_amplitude = RHO * k_p**2 * xi0_p / (1j * k_p)
    reflected_delta_rho_amplitude = b * delta_rho_amplitude
    sigma = RHO * xi0_p * (np.cos(theta) * (1.0 - b) + np.sin(theta) * c)

    actual, coefficients = _poisson_boundary_value_observables(
        z_target,
        qx=kx * np.cos(azimuth),
        qy=kx * np.sin(azimuth),
        source_terms=(
            (delta_rho_amplitude, 1j * kzp),
            (reflected_delta_rho_amplitude, -1j * kzp),
        ),
        surface_density=sigma,
    )
    production = halfspace_reflected_p_above_observables_g_free(
        [[0.0, 0.0, z_target]],
        density_kg_m3=RHO,
        v_p_m_s=material["v_p_m_s"],
        v_s_m_s=material["v_s_m_s"],
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        incident_displacement_m=xi0_p,
    )

    assert np.isfinite(coefficients[0]) and np.isfinite(coefficients[1])
    np.testing.assert_allclose(actual[0], production.potential[0], rtol=1e-10)
    np.testing.assert_allclose(actual[1], production.acceleration[0], rtol=1e-10)
    np.testing.assert_allclose(actual[2], production.gradient[0], rtol=1e-9)


@pytest.mark.parametrize("material", MATERIALS, ids=lambda m: m["name"])
@pytest.mark.parametrize("theta_deg", ANGLES_DEG)
def test_harms_eq54_surface_potential_matches_production(material, theta_deg):
    """Evaluate H19 Eq. (54) from surface seismic potentials independently.

    For an above-surface target and a horizontal wavevector along x, the
    surface integral in Eq. (54) reduces to
    ``-2*pi*rho*exp(-q*z)*(phi_s(0) + 1j*psi_y(0))``.  The surface potentials
    below come from the independently solved elastic displacement field; no
    production delta helper or surface density is used.
    """
    theta = np.deg2rad(theta_deg)
    xi0_p = 1.0e-9 + 0.4e-9j
    target = np.array([0.37, 0.0, 7.0])
    omega = 2.0 * np.pi * FREQ
    k_p, _k_s, kx, _kzp, _kzs, b, c = _solve_reflection_coefficients(
        material["v_p_m_s"], material["v_s_m_s"], theta, omega=omega
    )
    seismic_potential = xi0_p / (1j * k_p)
    phi_surface = seismic_potential * (1.0 + b)
    psi_y_surface = seismic_potential * c
    phi_eq54 = (
        -2.0
        * np.pi
        * RHO
        * np.exp(-kx * target[2])
        * np.exp(1j * kx * target[0])
        * (phi_surface + 1j * psi_y_surface)
    )

    production = halfspace_reflected_p_above_observables_g_free(
        [target],
        density_kg_m3=RHO,
        v_p_m_s=material["v_p_m_s"],
        v_s_m_s=material["v_s_m_s"],
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=0.0,
        incident_displacement_m=xi0_p,
    )
    np.testing.assert_allclose(production.potential[0], phi_eq54, rtol=1.0e-10)


@pytest.mark.slow
def test_reflected_p_displacement_dipole_integral_matches_potential():
    """Build gravity from the same displacement field without delta-rho/sigma.

    Horizontal Fourier reduction of the first-order displaced-volume
    potential gives, for an above-surface target,

    ``Phi/G = 2*pi*rho*exp(i*q*x-q*z) * int[-inf,0]
    ``exp(q*zp) * (1j*xi_x(zp) - xi_z(zp)) dzp``.

    The integral receives only the elastic displacement field and therefore
    does not pass a precomputed density perturbation or surface sheet into the
    reconstruction.
    """
    scipy_integrate = pytest.importorskip("scipy.integrate")
    material = HARD_ROCK
    theta = np.deg2rad(45.0)
    xi0_p = 1.0e-9 + 0.4e-9j
    target = np.array([0.37, 0.0, 7.0])
    omega = 2.0 * np.pi * FREQ
    k_p, k_s, kx, kzp, kzs, b, c = _solve_reflection_coefficients(
        material["v_p_m_s"], material["v_s_m_s"], theta, omega=omega
    )
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)

    def displacement(z_prime):
        incident = (
            xi0_p * np.array((sin_theta, 0.0, cos_theta)) * np.exp(1j * kzp * z_prime)
        )
        reflected = (
            xi0_p
            * b
            * np.array((sin_theta, 0.0, -cos_theta))
            * np.exp(-1j * kzp * z_prime)
        )
        converted = (
            xi0_p * c / k_p * np.array((kzs, 0.0, kx)) * np.exp(-1j * kzs * z_prime)
        )
        return incident + reflected + converted

    def integrand(z_prime):
        xi = displacement(z_prime)
        return np.exp(kx * z_prime) * (1j * xi[0] - xi[2])

    real, real_error = scipy_integrate.quad(
        lambda value: integrand(value).real,
        -np.inf,
        0.0,
        epsabs=1.0e-14,
        epsrel=1.0e-12,
        limit=1000,
    )
    imag, imag_error = scipy_integrate.quad(
        lambda value: integrand(value).imag,
        -np.inf,
        0.0,
        epsabs=1.0e-14,
        epsrel=1.0e-12,
        limit=1000,
    )
    assert real_error < 1.0e-12
    assert imag_error < 1.0e-12
    phi_dipole = (
        2.0
        * np.pi
        * RHO
        * np.exp(1j * kx * target[0] - kx * target[2])
        * (real + 1j * imag)
    )

    production = halfspace_reflected_p_above_observables_g_free(
        [target],
        density_kg_m3=RHO,
        v_p_m_s=material["v_p_m_s"],
        v_s_m_s=material["v_s_m_s"],
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=0.0,
        incident_displacement_m=xi0_p,
    )
    np.testing.assert_allclose(production.potential[0], phi_dipole, rtol=1.0e-10)


def test_physical_surface_sheet_is_distinct_from_harms2019_and_is_production_source():
    """Track the literature/physical coefficient split without masking it.

    The traction-free solve gives the PLUS surface displacement.  The literal
    H19 Eq. (90) coefficient remains a distinct regression reference, while
    production must agree with the actual bulk-plus-surface reconstruction.
    """
    v_p, v_s = HARD_ROCK["v_p_m_s"], HARD_ROCK["v_s_m_s"]
    theta = np.deg2rad(45.0)
    xi0_p = 1e-9 + 0.4e-9j
    z_target = 7.0
    omega = 2.0 * np.pi * FREQ
    k_p, k_s, kx, kzp, kzs, b, c = _solve_reflection_coefficients(
        v_p, v_s, theta, omega=omega
    )
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    xi_z_actual = cos_t * (1 - b) + sin_t * c
    xi_z_harms = cos_t * (1 - b) - sin_t * c

    assert abs(xi_z_actual - xi_z_harms) / abs(xi_z_actual) > 0.5

    # Full potential reconstruction using the ACTUAL displacement (PLUS) as
    # the surface sheet, compared directly to production.
    d0 = -RHO * 1j * k_p * xi0_p
    sigma_plus = RHO * xi0_p * xi_z_actual
    coef_bulk = -d0 / (2.0 * kx) * (1.0 / (kx + 1j * kzp) + b / (kx - 1j * kzp))
    coef_surf_plus = -sigma_plus / (2.0 * kx)
    phi_plus = 4.0 * np.pi * (coef_bulk + coef_surf_plus) * np.exp(-kx * z_target)

    r = halfspace_reflected_p_above_observables_g_free(
        [[0.0, 0.0, z_target]],
        density_kg_m3=RHO,
        v_p_m_s=v_p,
        v_s_m_s=v_s,
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=0.0,
        incident_displacement_m=xi0_p,
    )
    np.testing.assert_allclose(phi_plus, r.potential[0], rtol=1e-10, atol=0.0)


def _green_function(z, z_prime, k_rho):
    """Half-space Green's function for Poisson's equation in horizontal
    Fourier domain: G(z, z') = -exp(-k_rho * |z - z'|) / (2 * k_rho).
    """
    return -np.exp(-k_rho * abs(z - z_prime)) / (2.0 * k_rho)


def _dg_dz(z, z_prime, k_rho):
    """Derivative dG/dz."""
    if z > z_prime:
        return k_rho * np.exp(-k_rho * (z - z_prime)) / (2.0 * k_rho)
    return -k_rho * np.exp(-k_rho * (z_prime - z)) / (2.0 * k_rho)


def _d2g_dz2(z, z_prime, k_rho):
    """Regular part of d^2G/dz^2 away from the z==z' coincidence.

    G'' (distributionally) = delta(z-z') + k_rho^2 * G(z,z') -- the
    Green's-function identity (d^2/dz^2 - k_rho^2) G = delta(z-z'), since
    dG/dz has a unit jump at z=z' (jump = +1/2 - (-1/2) = 1) and G itself
    has none. This function returns only the REGULAR part, k_rho^2*G(z,z')
    = -k_rho*exp(-k_rho|z-z'|)/2; a caller evaluating the bulk second
    z-derivative at an observation point INSIDE the source's integration
    domain must add the local density term (delta_rho_hat(z_target))
    separately -- see the underground bulk correction in
    ``_reconstruct_single_p_observables``.
    """
    return -k_rho * np.exp(-k_rho * abs(z - z_prime)) / 2.0


@pytest.mark.slow
@pytest.mark.parametrize("z_target", [3.0, -3.0, 0.5, -0.5, -20.0])
def test_green_function_reconstructs_single_p_potential(z_target):
    scipy_integrate = pytest.importorskip("scipy.integrate")

    v_p, v_s = HARD_ROCK["v_p_m_s"], HARD_ROCK["v_s_m_s"]
    theta = np.deg2rad(37.0)
    xi0_p = 1e-9 + 0.4e-9j
    k_p = 2.0 * np.pi * FREQ / v_p
    k_zp = k_p * np.cos(theta)
    k_rho = k_p * np.sin(theta)

    def delta_rho_hat(z_prime):
        return -RHO * 1j * k_p * xi0_p * np.exp(1j * k_zp * z_prime)

    bulk, _ = _split_quad(
        scipy_integrate,
        lambda zp: delta_rho_hat(zp).real * _green_function(z_target, zp, k_rho),
        lambda zp: delta_rho_hat(zp).imag * _green_function(z_target, zp, k_rho),
        z_target,
        limit=1000,
        epsabs=1e-14,
        epsrel=1e-12,
    )
    sigma = RHO * xi0_p * np.cos(theta)
    surf = _green_function(z_target, 0.0, k_rho) * sigma
    phi_green = 4.0 * np.pi * (bulk + surf)

    r = halfspace_single_bodywave_observables_g_free(
        [[0.0, 0.0, z_target]],
        wave_type="P",
        density_kg_m3=RHO,
        v_p_m_s=v_p,
        v_s_m_s=v_s,
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=0.0,
        displacement_m=xi0_p,
    )
    # With tightened epsabs/epsrel and domain splitting at z_target (see
    # _split_quad), this reaches ~1.1e-14 to 1.2e-14 relative across all
    # tested z_target (measured; see docs/validation/literature-b.md
    # Method 3) -- restoring the original well-conditioned 1e-10 (Phi)
    # criterion instead of the previous 3e-8..1e-5 default-quad-tolerance
    # floor.
    tol_rel = 1e-10
    allowed = tol_rel * abs(r.potential[0])
    assert abs(phi_green - r.potential[0]) <= allowed


@pytest.mark.slow
@pytest.mark.parametrize("material", MATERIALS, ids=lambda m: m["name"])
@pytest.mark.parametrize("theta_deg", ANGLES_DEG)
def test_green_function_reconstructs_reflected_p_composite(material, theta_deg):
    """Reconstruct the full reflected-P composite potential (incident P +
    reflected P + converted SV) via Green's function integration of the
    independently-solved boundary-value displacement field.

    The displacement field uses the Helmholtz potentials with coefficients
    (B, C) from the traction-free boundary-value solve. The bulk density
    comes from the P-wave divergence only (SV is divergence-free), and the
    surface mass sheet is the total vertical displacement at z=0.
    """
    scipy_integrate = pytest.importorskip("scipy.integrate")

    theta = np.deg2rad(theta_deg)
    azimuth = np.deg2rad(15.0)
    xi0_p = 1e-9 + 0.4e-9j
    z_target = 7.0

    omega = 2.0 * np.pi * FREQ
    k_p, k_s, kx, kzp, kzs, b, c = _solve_reflection_coefficients(
        material["v_p_m_s"], material["v_s_m_s"], theta, omega=omega
    )

    # Bulk density: delta_rho = -rho * div(xi_P)
    # phi_inc = xi0_p / (i k_p) * exp(i kzp z)
    # phi_ref = b * xi0_p / (i k_p) * exp(-i kzp z)
    # div(xi_P) = -k_p^2 * (phi_inc + phi_ref)
    # delta_rho = rho * k_p^2 * (phi_inc + phi_ref)
    def delta_rho_hat(z_prime):
        phi_inc = xi0_p / (1j * k_p) * np.exp(1j * kzp * z_prime)
        phi_ref = b * xi0_p / (1j * k_p) * np.exp(-1j * kzp * z_prime)
        return RHO * k_p**2 * (phi_inc + phi_ref)

    # Surface mass sheet from the actual total vertical displacement.
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    xi_z_0 = xi0_p * (cos_theta * (1 - b) + sin_theta * c)
    sigma = RHO * xi_z_0

    bulk, _ = _split_quad(
        scipy_integrate,
        lambda zp: delta_rho_hat(zp).real * _green_function(z_target, zp, kx),
        lambda zp: delta_rho_hat(zp).imag * _green_function(z_target, zp, kx),
        z_target,
        limit=1000,
        epsabs=1e-14,
        epsrel=1e-12,
    )
    surf = _green_function(z_target, 0.0, kx) * sigma
    phi_green = 4.0 * np.pi * (bulk + surf)

    r = halfspace_reflected_p_above_observables_g_free(
        [[0.0, 0.0, z_target]],
        density_kg_m3=RHO,
        v_p_m_s=material["v_p_m_s"],
        v_s_m_s=material["v_s_m_s"],
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        incident_displacement_m=xi0_p,
    )
    # With tightened epsabs/epsrel, this reaches ~1e-14 relative across all
    # tested (material, theta) combinations (measured; see
    # docs/validation/literature-b.md Method 3).
    tol_rel = 1e-10
    allowed = tol_rel * abs(r.potential[0])
    assert abs(phi_green - r.potential[0]) <= allowed


# ============================================================
# Full three-observable reconstruction (Phi, a, Gamma)
# ============================================================


def _split_quad(scipy_integrate, real_fn, imag_fn, z_target, **kwargs):
    """Integrate a complex integrand over z' in (-inf, 0), splitting the
    domain at z'=z_target when z_target lies inside it (z_target < 0).

    The kink in dG/dz (and the regular part of d^2G/dz^2) at z'=z_target
    slows quad's adaptive convergence when the discontinuity is buried
    inside a single (-inf, 0) call; telling quad about the split point
    directly (two calls meeting exactly at z_target) restores its normal
    convergence rate. Returns (value, summed_abserr).
    """
    if z_target < 0.0:
        re1, err1 = scipy_integrate.quad(real_fn, -np.inf, z_target, **kwargs)
        re2, err2 = scipy_integrate.quad(real_fn, z_target, 0.0, **kwargs)
        im1, err3 = scipy_integrate.quad(imag_fn, -np.inf, z_target, **kwargs)
        im2, err4 = scipy_integrate.quad(imag_fn, z_target, 0.0, **kwargs)
        return (re1 + re2) + 1j * (im1 + im2), err1 + err2 + err3 + err4
    re, err_re = scipy_integrate.quad(real_fn, -np.inf, 0.0, **kwargs)
    im, err_im = scipy_integrate.quad(imag_fn, -np.inf, 0.0, **kwargs)
    return re + 1j * im, err_re + err_im


def _reconstruct_single_p_observables(z_target, theta, azimuth, xi0_p, v_p, _v_s):
    """Reconstruct Phi, a, Gamma for single P via Green's function.

    For underground z_target (< 0, inside the bulk source domain z'<0),
    Gamma_zz needs an extra "local density" term beyond the regular-part
    quadrature: the defining ODE (d^2/dz^2 - k_rho^2) Phi_hat = 4*pi*
    delta_rho_hat(z) means Phi_bulk''(z) = k_rho^2*Phi_bulk(z) +
    delta_rho_hat(z) classically whenever z is an interior point of a
    continuous source (the Green's function's own delta(z-z') distributional
    content, picked up only when the observation point coincides with a
    point inside the integration domain -- see the docstring on
    ``_d2g_dz2``). This was verified as an exact symbolic identity against
    production's closed-form Eq.(87) (see the exact closed-form fast tests
    below); here it is applied as a correction to the quadrature-computed
    "regular part only" result.
    """
    scipy_integrate = pytest.importorskip("scipy.integrate")

    k_p = 2.0 * np.pi * FREQ / v_p
    qx = k_p * np.sin(theta) * np.cos(azimuth)
    qy = k_p * np.sin(theta) * np.sin(azimuth)
    kzp = k_p * np.cos(theta)
    k_rho = np.sqrt(qx**2 + qy**2)

    def delta_rho(zp):
        phi = xi0_p / (1j * k_p) * np.exp(1j * kzp * zp)
        return RHO * k_p**2 * phi

    sigma = RHO * xi0_p * np.cos(theta)

    quad_kwargs = {"limit": 1000, "epsabs": 1e-14, "epsrel": 1e-12}

    # Phi
    bulk, phi_err = _split_quad(
        scipy_integrate,
        lambda zp: delta_rho(zp).real * _green_function(z_target, zp, k_rho),
        lambda zp: delta_rho(zp).imag * _green_function(z_target, zp, k_rho),
        z_target,
        **quad_kwargs,
    )
    surf = _green_function(z_target, 0.0, k_rho) * sigma
    phi_green = 4.0 * np.pi * (bulk + surf)

    # Acceleration
    a_x = -1j * qx * phi_green
    a_y = -1j * qy * phi_green
    a_z_bulk_raw, _ = _split_quad(
        scipy_integrate,
        lambda zp: delta_rho(zp).real * _dg_dz(z_target, zp, k_rho),
        lambda zp: delta_rho(zp).imag * _dg_dz(z_target, zp, k_rho),
        z_target,
        **quad_kwargs,
    )
    a_z_bulk = -4.0 * np.pi * a_z_bulk_raw
    a_z_surf = -4.0 * np.pi * _dg_dz(z_target, 0.0, k_rho) * sigma
    a_z = a_z_bulk + a_z_surf

    # Gradient
    gamma = np.zeros((3, 3), dtype=complex)
    gamma[0, 0] = qx**2 * phi_green
    gamma[0, 1] = qx * qy * phi_green
    gamma[1, 0] = gamma[0, 1]
    gamma[1, 1] = qy**2 * phi_green
    # Gamma_xz = d(a_x)/dz = d(-i*qx*Phi)/dz = -i*qx*(dPhi/dz) = -i*qx*(-a_z)
    # = +i*qx*a_z (a_z = -dPhi/dz throughout this module's a=-grad(Phi)
    # convention); likewise Gamma_yz = +i*qy*a_z. Verified independently by
    # matching the coded traction-free matrix (a11..a22) to
    # u=grad(phi)+curl(0,psi,0) and differentiating a_x=-i*qx*Phi directly --
    # do not flip this sign to "match" a quadrature result.
    gamma[0, 2] = 1j * qx * a_z
    gamma[2, 0] = gamma[0, 2]
    gamma[1, 2] = 1j * qy * a_z
    gamma[2, 1] = gamma[1, 2]
    gzz_bulk_regular_raw, gzz_err = _split_quad(
        scipy_integrate,
        lambda zp: delta_rho(zp).real * _d2g_dz2(z_target, zp, k_rho),
        lambda zp: delta_rho(zp).imag * _d2g_dz2(z_target, zp, k_rho),
        z_target,
        limit=1000,
        epsabs=1e-14,
        epsrel=1e-12,
    )
    gzz_bulk = -4.0 * np.pi * gzz_bulk_regular_raw
    if z_target < 0.0:
        # Local density term (see _reconstruct_single_p_observables and the
        # _d2g_dz2 docstring); a no-op here for the reflected-P composite,
        # whose bulk source domain is z'<0 and whose only defined
        # observation points are z_target>0 (ground only), but kept for
        # symmetry/reuse if that scope ever changes.
        gzz_bulk = gzz_bulk - 4.0 * np.pi * delta_rho(z_target)
    gzz_surf = -4.0 * np.pi * _d2g_dz2(z_target, 0.0, k_rho) * sigma
    gamma[2, 2] = gzz_bulk + gzz_surf

    quad_err = phi_err + gzz_err
    return phi_green, np.array([a_x, a_y, a_z]), gamma, quad_err


def _reconstruct_single_sv_observables(z_target, theta, azimuth, xi0_sv, v_s):
    """Reconstruct Phi, a, Gamma for single SV (surface only)."""
    k_s = 2.0 * np.pi * FREQ / v_s
    qx = k_s * np.sin(theta) * np.cos(azimuth)
    qy = k_s * np.sin(theta) * np.sin(azimuth)
    k_rho = np.sqrt(qx**2 + qy**2)

    sigma = -RHO * xi0_sv * np.sin(theta)

    phi_green = 4.0 * np.pi * _green_function(z_target, 0.0, k_rho) * sigma
    a_x = -1j * qx * phi_green
    a_y = -1j * qy * phi_green
    a_z = -4.0 * np.pi * _dg_dz(z_target, 0.0, k_rho) * sigma

    gamma = np.zeros((3, 3), dtype=complex)
    gamma[0, 0] = qx**2 * phi_green
    gamma[0, 1] = qx * qy * phi_green
    gamma[1, 0] = gamma[0, 1]
    gamma[1, 1] = qy**2 * phi_green
    # Gamma_xz = d(a_x)/dz = d(-i*qx*Phi)/dz = -i*qx*(dPhi/dz) = -i*qx*(-a_z)
    # = +i*qx*a_z (a_z = -dPhi/dz throughout this module's a=-grad(Phi)
    # convention); likewise Gamma_yz = +i*qy*a_z. Verified independently by
    # matching the coded traction-free matrix (a11..a22) to
    # u=grad(phi)+curl(0,psi,0) and differentiating a_x=-i*qx*Phi directly --
    # do not flip this sign to "match" a quadrature result.
    gamma[0, 2] = 1j * qx * a_z
    gamma[2, 0] = gamma[0, 2]
    gamma[1, 2] = 1j * qy * a_z
    gamma[2, 1] = gamma[1, 2]
    gamma[2, 2] = -4.0 * np.pi * _d2g_dz2(z_target, 0.0, k_rho) * sigma

    return phi_green, np.array([a_x, a_y, a_z]), gamma, 0.0


def _reconstruct_reflected_p_observables(z_target, theta, azimuth, xi0_p, v_p, v_s):
    """Reconstruct Phi, a, Gamma for reflected P composite via Green's function."""
    scipy_integrate = pytest.importorskip("scipy.integrate")

    omega = 2.0 * np.pi * FREQ
    k_p = omega / v_p
    k_s = omega / v_s
    kx_mode = k_p * np.sin(theta)
    kzp = np.sqrt(k_p**2 - kx_mode**2 + 0j)
    kzs = np.sqrt(k_s**2 - kx_mode**2 + 0j)

    lam = v_p**2 - 2 * v_s**2
    mu = v_s**2

    a11 = -lam * (kx_mode**2 + kzp**2) - 2 * mu * kzp**2
    a12 = 2 * mu * kx_mode * kzs
    rhs1 = lam * (kx_mode**2 + kzp**2) + 2 * mu * kzp**2
    a21 = 2 * mu * kx_mode * kzp
    a22 = mu * (kzs**2 - kx_mode**2)
    rhs2 = 2 * mu * kx_mode * kzp

    m = np.array([[a11, a12], [a21, a22]], dtype=complex)
    rhs = np.array([rhs1, rhs2], dtype=complex)
    b, c = np.linalg.solve(m, rhs)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    # Actual total vertical displacement from the traction-free field.
    xi_z_0 = xi0_p * (cos_theta * (1 - b) + sin_theta * c)
    sigma = RHO * xi_z_0

    def delta_rho(zp):
        phi_inc = xi0_p / (1j * k_p) * np.exp(1j * kzp * zp)
        phi_ref = b * xi0_p / (1j * k_p) * np.exp(-1j * kzp * zp)
        return RHO * k_p**2 * (phi_inc + phi_ref)

    qx = k_p * np.sin(theta) * np.cos(azimuth)
    qy = k_p * np.sin(theta) * np.sin(azimuth)
    k_rho = np.sqrt(qx**2 + qy**2)

    quad_kwargs = {"limit": 1000, "epsabs": 1e-14, "epsrel": 1e-12}

    # Phi
    bulk, phi_err = _split_quad(
        scipy_integrate,
        lambda zp: delta_rho(zp).real * _green_function(z_target, zp, k_rho),
        lambda zp: delta_rho(zp).imag * _green_function(z_target, zp, k_rho),
        z_target,
        **quad_kwargs,
    )
    surf = _green_function(z_target, 0.0, k_rho) * sigma
    phi_green = 4.0 * np.pi * (bulk + surf)

    # Acceleration
    a_x = -1j * qx * phi_green
    a_y = -1j * qy * phi_green
    a_z_bulk_raw, _ = _split_quad(
        scipy_integrate,
        lambda zp: delta_rho(zp).real * _dg_dz(z_target, zp, k_rho),
        lambda zp: delta_rho(zp).imag * _dg_dz(z_target, zp, k_rho),
        z_target,
        **quad_kwargs,
    )
    a_z_bulk = -4.0 * np.pi * a_z_bulk_raw
    a_z_surf = -4.0 * np.pi * _dg_dz(z_target, 0.0, k_rho) * sigma
    a_z = a_z_bulk + a_z_surf

    # Gradient
    gamma = np.zeros((3, 3), dtype=complex)
    gamma[0, 0] = qx**2 * phi_green
    gamma[0, 1] = qx * qy * phi_green
    gamma[1, 0] = gamma[0, 1]
    gamma[1, 1] = qy**2 * phi_green
    # Gamma_xz = d(a_x)/dz = d(-i*qx*Phi)/dz = -i*qx*(dPhi/dz) = -i*qx*(-a_z)
    # = +i*qx*a_z (a_z = -dPhi/dz throughout this module's a=-grad(Phi)
    # convention); likewise Gamma_yz = +i*qy*a_z. Verified independently by
    # matching the coded traction-free matrix (a11..a22) to
    # u=grad(phi)+curl(0,psi,0) and differentiating a_x=-i*qx*Phi directly --
    # do not flip this sign to "match" a quadrature result.
    gamma[0, 2] = 1j * qx * a_z
    gamma[2, 0] = gamma[0, 2]
    gamma[1, 2] = 1j * qy * a_z
    gamma[2, 1] = gamma[1, 2]
    gzz_bulk_regular_raw, gzz_err = _split_quad(
        scipy_integrate,
        lambda zp: delta_rho(zp).real * _d2g_dz2(z_target, zp, k_rho),
        lambda zp: delta_rho(zp).imag * _d2g_dz2(z_target, zp, k_rho),
        z_target,
        limit=1000,
        epsabs=1e-14,
        epsrel=1e-12,
    )
    gzz_bulk = -4.0 * np.pi * gzz_bulk_regular_raw
    if z_target < 0.0:
        # Local density term (see _reconstruct_single_p_observables and the
        # _d2g_dz2 docstring); a no-op here for the reflected-P composite,
        # whose bulk source domain is z'<0 and whose only defined
        # observation points are z_target>0 (ground only), but kept for
        # symmetry/reuse if that scope ever changes.
        gzz_bulk = gzz_bulk - 4.0 * np.pi * delta_rho(z_target)
    gzz_surf = -4.0 * np.pi * _d2g_dz2(z_target, 0.0, k_rho) * sigma
    gamma[2, 2] = gzz_bulk + gzz_surf

    quad_err = phi_err + gzz_err
    return phi_green, np.array([a_x, a_y, a_z]), gamma, quad_err


@pytest.mark.slow
@pytest.mark.parametrize("z_target", [3.0, 10.0, -3.0, -10.0])
def test_green_function_reconstructs_single_p_full_observables(z_target):
    """Reconstruct Phi, a, Gamma for single P wave (ground AND underground).

    With the ``_d2g_dz2`` regular-part sign fix, the Gamma_xz/Gamma_yz sign
    fix, tightened ``quad`` tolerances (epsabs=1e-14, epsrel=1e-12), and the
    underground local-density correction + domain split (see
    ``_reconstruct_single_p_observables``), this reaches ~1.3e-14 relative
    precision at theta=37 deg across ground and underground targets --
    restoring the original well-conditioned ``1e-10`` (Phi/acceleration) /
    ``1e-9`` (gradient) acceptance criteria (no per-regime tolerance
    loosening needed here, see docs/validation/literature-b.md).
    """
    theta = np.deg2rad(37.0)
    azimuth = 0.0
    xi0_p = 1e-9 + 0.4e-9j
    v_p, v_s = HARD_ROCK["v_p_m_s"], HARD_ROCK["v_s_m_s"]

    phi_g, a_g, gamma_g, quad_err = _reconstruct_single_p_observables(
        z_target, theta, azimuth, xi0_p, v_p, v_s
    )

    r = halfspace_single_bodywave_observables_g_free(
        [[0.0, 0.0, z_target]],
        wave_type="P",
        density_kg_m3=RHO,
        v_p_m_s=v_p,
        v_s_m_s=v_s,
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        displacement_m=xi0_p,
    )

    # tol_phi_a: original well-conditioned criterion for Phi/acceleration;
    # tol_gamma: one derivative order noisier, for the gradient.
    tol_phi_a = 1e-10
    tol_gamma = 1e-9
    assert abs(phi_g - r.potential[0]) <= tol_phi_a * abs(r.potential[0])
    assert abs(a_g[0] - r.acceleration[0, 0]) <= tol_phi_a * abs(r.acceleration[0, 0])
    assert abs(a_g[1] - r.acceleration[0, 1]) <= tol_phi_a * abs(r.acceleration[0, 1])
    assert abs(a_g[2] - r.acceleration[0, 2]) <= tol_phi_a * abs(r.acceleration[0, 2])
    assert abs(gamma_g[0, 0] - r.gradient[0, 0, 0]) <= tol_gamma * abs(
        r.gradient[0, 0, 0]
    )
    assert abs(gamma_g[0, 1] - r.gradient[0, 0, 1]) <= tol_gamma * abs(
        r.gradient[0, 0, 1]
    )
    assert abs(gamma_g[1, 1] - r.gradient[0, 1, 1]) <= tol_gamma * abs(
        r.gradient[0, 1, 1]
    )
    assert abs(gamma_g[0, 2] - r.gradient[0, 0, 2]) <= tol_gamma * abs(
        r.gradient[0, 0, 2]
    )
    assert abs(gamma_g[1, 2] - r.gradient[0, 1, 2]) <= tol_gamma * abs(
        r.gradient[0, 1, 2]
    )
    assert abs(gamma_g[2, 2] - r.gradient[0, 2, 2]) <= tol_gamma * abs(
        r.gradient[0, 2, 2]
    )


@pytest.mark.slow
@pytest.mark.parametrize("z_target", [3.0, 10.0, -3.0, -10.0])
def test_green_function_reconstructs_single_sv_full_observables(z_target):
    """Reconstruct Phi, a, Gamma for single SV wave (surface only, ground
    AND underground -- SV has no bulk term, so no quadrature is involved and
    this reaches machine precision, ~1e-16, on both sides of the surface)."""
    theta = np.deg2rad(37.0)
    azimuth = np.deg2rad(15.0)
    xi0_sv = 1e-9 - 0.2e-9j
    v_p, v_s = HARD_ROCK["v_p_m_s"], HARD_ROCK["v_s_m_s"]

    phi_g, a_g, gamma_g, _ = _reconstruct_single_sv_observables(
        z_target, theta, azimuth, xi0_sv, v_s
    )

    r = halfspace_single_bodywave_observables_g_free(
        [[0.0, 0.0, z_target]],
        wave_type="SV",
        density_kg_m3=RHO,
        v_p_m_s=v_p,
        v_s_m_s=v_s,
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        displacement_m=xi0_sv,
    )

    # tol_phi_a: original well-conditioned criterion for Phi/acceleration;
    # tol_gamma: one derivative order noisier, for the gradient.
    tol_phi_a = 1e-10
    tol_gamma = 1e-9
    assert abs(phi_g - r.potential[0]) <= tol_phi_a * abs(r.potential[0])
    assert abs(a_g[0] - r.acceleration[0, 0]) <= tol_phi_a * abs(r.acceleration[0, 0])
    assert abs(a_g[1] - r.acceleration[0, 1]) <= tol_phi_a * abs(r.acceleration[0, 1])
    assert abs(a_g[2] - r.acceleration[0, 2]) <= tol_phi_a * abs(r.acceleration[0, 2])
    assert abs(gamma_g[0, 0] - r.gradient[0, 0, 0]) <= tol_gamma * abs(
        r.gradient[0, 0, 0]
    )
    assert abs(gamma_g[0, 1] - r.gradient[0, 0, 1]) <= tol_gamma * abs(
        r.gradient[0, 0, 1]
    )
    assert abs(gamma_g[1, 1] - r.gradient[0, 1, 1]) <= tol_gamma * abs(
        r.gradient[0, 1, 1]
    )
    assert abs(gamma_g[0, 2] - r.gradient[0, 0, 2]) <= tol_gamma * abs(
        r.gradient[0, 0, 2]
    )
    assert abs(gamma_g[1, 2] - r.gradient[0, 1, 2]) <= tol_gamma * abs(
        r.gradient[0, 1, 2]
    )
    assert abs(gamma_g[2, 2] - r.gradient[0, 2, 2]) <= tol_gamma * abs(
        r.gradient[0, 2, 2]
    )


@pytest.mark.slow
@pytest.mark.parametrize("material", MATERIALS, ids=lambda m: m["name"])
@pytest.mark.parametrize("theta_deg", [45.0])  # representative moderate angle
def test_green_function_reconstructs_reflected_p_full_observables(material, theta_deg):
    """Reconstruct Phi, a, Gamma for reflected P composite (ground only).

    Restricted to theta=45 deg (moderate k_rho) for THIS quadrature-based
    method: at theta=10 deg (small k_rho), Gamma_zz's quadrature converges
    only to ~1e-6..4e-6 relative (measured; the long oscillatory tail of the
    Green's-function second-derivative integral at small k_rho), well short
    of the 1e-9 criterion used here. The exact closed-form test below sweeps
    all three angles including 10 deg and reaches machine precision.
    """
    theta = np.deg2rad(theta_deg)
    azimuth = np.deg2rad(15.0)
    xi0_p = 1e-9 + 0.4e-9j
    z_target = 7.0

    phi_g, a_g, gamma_g, quad_err = _reconstruct_reflected_p_observables(
        z_target, theta, azimuth, xi0_p, material["v_p_m_s"], material["v_s_m_s"]
    )

    r = halfspace_reflected_p_above_observables_g_free(
        [[0.0, 0.0, z_target]],
        density_kg_m3=RHO,
        v_p_m_s=material["v_p_m_s"],
        v_s_m_s=material["v_s_m_s"],
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        incident_displacement_m=xi0_p,
    )

    # Measured worst case across both materials at theta=45 deg is ~2e-11
    # (well under 1e-10); see docs/validation/literature-b.md.
    tol_phi_a = 1e-10
    tol_gamma = 1e-9

    assert abs(phi_g - r.potential[0]) <= tol_phi_a * abs(r.potential[0])
    assert abs(a_g[0] - r.acceleration[0, 0]) <= tol_phi_a * abs(r.acceleration[0, 0])
    assert abs(a_g[1] - r.acceleration[0, 1]) <= tol_phi_a * abs(r.acceleration[0, 1])
    assert abs(a_g[2] - r.acceleration[0, 2]) <= tol_phi_a * abs(r.acceleration[0, 2])
    assert abs(gamma_g[0, 0] - r.gradient[0, 0, 0]) <= tol_gamma * abs(
        r.gradient[0, 0, 0]
    )
    assert abs(gamma_g[0, 1] - r.gradient[0, 0, 1]) <= tol_gamma * abs(
        r.gradient[0, 0, 1]
    )
    assert abs(gamma_g[1, 1] - r.gradient[0, 1, 1]) <= tol_gamma * abs(
        r.gradient[0, 1, 1]
    )
    assert abs(gamma_g[0, 2] - r.gradient[0, 0, 2]) <= tol_gamma * abs(
        r.gradient[0, 0, 2]
    )
    assert abs(gamma_g[1, 2] - r.gradient[0, 1, 2]) <= tol_gamma * abs(
        r.gradient[0, 1, 2]
    )
    assert abs(gamma_g[2, 2] - r.gradient[0, 2, 2]) <= tol_gamma * abs(
        r.gradient[0, 2, 2]
    )


# ============================================================
# Method 5: exact closed-form reconstruction (no scipy, non-slow)
# ============================================================
#
# The bulk density in every case here is a sum of decaying/oscillating
# complex exponentials in z', so its Green's-function convolution (and
# first/second z-derivatives) is an elementary closed form -- no numerical
# quadrature is needed, and no quadrature-precision floor applies. These
# tests are fast (pure ``numpy`` exp() arithmetic) and run in the normal,
# non-slow CI matrix (registering a fast independent-reference case per the
# review, without needing a new ci.yml/slow.yml entry: no
# ``pytest.importorskip`` is used, so ``tests/test_ci_coverage.py``'s
# reachability check does not apply). The closed forms themselves were
# derived and cross-checked with ``sympy`` (exact symbolic zero-diff against
# production's Eq.(86)/(87)/(90), for Phi, dPhi/dz, and d^2Phi/dz^2 all
# three); this file re-implements them directly in plain arithmetic so the
# test itself does not depend on sympy being installed.


def _eval_exp_sum(terms, z, order):
    """Evaluate sum(coef*exp(rate*z)) or its z-derivative of the given order
    (0, 1, or 2), for a list of (coef, rate) pairs. Elementary/exact -- each
    derivative of exp(rate*z) is just another factor of ``rate``."""
    total = 0j
    for coef, rate in terms:
        total += coef * (rate**order) * np.exp(rate * z)
    return total


def _closed_form_single_p(z, theta, xi0_p, v_p):
    """Exact closed-form (Phi, dPhi/dz, d^2Phi/dz^2) for single P, ground
    (z>0) or underground (z<0), via elementary integration of the Green's
    function against the P-wave's own bulk divergence density plus the
    free-surface mass sheet -- see the module docstring's Method 5 and
    ``docs/design/literature-b.md``."""
    k_p = 2.0 * np.pi * FREQ / v_p
    k_rho = k_p * np.sin(theta)
    kzp = k_p * np.cos(theta)
    d0 = -RHO * 1j * k_p * xi0_p  # delta_rho_hat(z') = d0*exp(i*kzp*z')
    sigma = RHO * xi0_p * np.cos(theta)

    if z > 0.0:
        coef_bulk = -d0 / (2.0 * k_rho * (k_rho + 1j * kzp))
        coef_surf = -sigma / (2.0 * k_rho)
        terms = [(coef_bulk + coef_surf, -k_rho)]
    else:
        c1 = -d0 / (2.0 * k_rho * (k_rho + 1j * kzp))
        c2 = -d0 / (2.0 * k_rho * (-k_rho + 1j * kzp))
        c3 = -sigma / (2.0 * k_rho)
        terms = [(c1 - c2, 1j * kzp), (c2 + c3, k_rho)]

    phi = 4.0 * np.pi * _eval_exp_sum(terms, z, 0)
    s1 = 4.0 * np.pi * _eval_exp_sum(terms, z, 1)
    s2 = 4.0 * np.pi * _eval_exp_sum(terms, z, 2)
    return phi, s1, s2


def _closed_form_reflected_p(z, theta, xi0_p, v_p, v_s):
    """Exact closed-form (Phi, dPhi/dz, d^2Phi/dz^2) for the reflected-P
    composite (ground only, z>0, matching production's contract), using the
    boundary-value (B, C) and the actual total vertical displacement as the
    surface sheet."""
    omega = 2.0 * np.pi * FREQ
    k_p = omega / v_p
    k_s = omega / v_s
    kx = k_p * np.sin(theta)
    kzp = np.sqrt(k_p**2 - kx**2 + 0j)
    kzs = np.sqrt(k_s**2 - kx**2 + 0j)
    lam = RHO * (v_p**2 - 2 * v_s**2)
    mu = RHO * v_s**2
    a11 = -lam * (kx**2 + kzp**2) - 2 * mu * kzp**2
    a12 = 2 * mu * kx * kzs
    rhs1 = lam * (kx**2 + kzp**2) + 2 * mu * kzp**2
    a21 = 2 * mu * kx * kzp
    a22 = mu * (kzs**2 - kx**2)
    rhs2 = 2 * mu * kx * kzp
    m = np.array([[a11, a12], [a21, a22]], dtype=complex)
    rhs = np.array([rhs1, rhs2], dtype=complex)
    b, c = np.linalg.solve(m, rhs)

    k_rho = kx
    d0 = -RHO * 1j * k_p * xi0_p
    sigma = RHO * xi0_p * (np.cos(theta) * (1 - b) + np.sin(theta) * c)
    coef_bulk = (
        -d0 / (2.0 * k_rho) * (1.0 / (k_rho + 1j * kzp) + b / (k_rho - 1j * kzp))
    )
    coef_surf = -sigma / (2.0 * k_rho)
    terms = [(coef_bulk + coef_surf, -k_rho)]

    phi = 4.0 * np.pi * _eval_exp_sum(terms, z, 0)
    s1 = 4.0 * np.pi * _eval_exp_sum(terms, z, 1)
    s2 = 4.0 * np.pi * _eval_exp_sum(terms, z, 2)
    return phi, s1, s2


def _assert_full_observables_match(
    phi, s1, s2, qx, qy, r, tol_phi_a=1e-10, tol_gamma=1e-9
):
    """tol_phi_a applies to Phi and acceleration (the original well-
    conditioned criterion); tol_gamma applies to the gradient (one
    derivative order noisier). Measured worst case across all exact
    closed-form cases is ~1.9e-14, far under either."""
    a_z = -s1
    gamma_zz = -s2
    assert abs(phi - r.potential[0]) <= tol_phi_a * abs(r.potential[0])
    assert abs(-1j * qx * phi - r.acceleration[0, 0]) <= tol_phi_a * abs(
        r.acceleration[0, 0]
    )
    assert abs(-1j * qy * phi - r.acceleration[0, 1]) <= tol_phi_a * abs(
        r.acceleration[0, 1]
    )
    assert abs(a_z - r.acceleration[0, 2]) <= tol_phi_a * abs(r.acceleration[0, 2])
    assert abs(qx * qx * phi - r.gradient[0, 0, 0]) <= tol_gamma * abs(
        r.gradient[0, 0, 0]
    )
    assert abs(qx * qy * phi - r.gradient[0, 0, 1]) <= tol_gamma * abs(
        r.gradient[0, 0, 1]
    )
    assert abs(qy * qy * phi - r.gradient[0, 1, 1]) <= tol_gamma * abs(
        r.gradient[0, 1, 1]
    )
    assert abs(1j * qx * a_z - r.gradient[0, 0, 2]) <= tol_gamma * abs(
        r.gradient[0, 0, 2]
    )
    assert abs(1j * qy * a_z - r.gradient[0, 1, 2]) <= tol_gamma * abs(
        r.gradient[0, 1, 2]
    )
    assert abs(gamma_zz - r.gradient[0, 2, 2]) <= tol_gamma * abs(r.gradient[0, 2, 2])


@pytest.mark.parametrize("z_target", [3.0, 10.0, -3.0, -10.0, -0.5])
@pytest.mark.parametrize("theta_deg", [10.0, 37.0, 80.0])
def test_single_p_exact_closed_form(z_target, theta_deg):
    """Fast, non-slow, machine-precision independent check of single-P
    (ground and underground, all angles including small k_rho=theta=10 deg,
    which the quadrature method above does not cover at 1e-9)."""
    theta = np.deg2rad(theta_deg)
    azimuth = np.deg2rad(20.0)
    xi0_p = 1e-9 + 0.4e-9j
    v_p, v_s = HARD_ROCK["v_p_m_s"], HARD_ROCK["v_s_m_s"]

    phi, s1, s2 = _closed_form_single_p(z_target, theta, xi0_p, v_p)

    r = halfspace_single_bodywave_observables_g_free(
        [[0.0, 0.0, z_target]],
        wave_type="P",
        density_kg_m3=RHO,
        v_p_m_s=v_p,
        v_s_m_s=v_s,
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        displacement_m=xi0_p,
    )
    k_p = 2.0 * np.pi * FREQ / v_p
    qx = k_p * np.sin(theta) * np.cos(azimuth)
    qy = k_p * np.sin(theta) * np.sin(azimuth)
    _assert_full_observables_match(phi, s1, s2, qx, qy, r)


@pytest.mark.parametrize("material", MATERIALS, ids=lambda m: m["name"])
@pytest.mark.parametrize("theta_deg", ANGLES_DEG)
def test_reflected_p_exact_closed_form(material, theta_deg):
    """Fast, non-slow, machine-precision independent check of the
    reflected-P composite across all three H19 Fig.10 angles (including
    theta=10 deg, small k_rho, which the quadrature method above does not
    cover at 1e-9 for Gamma_zz)."""
    theta = np.deg2rad(theta_deg)
    azimuth = np.deg2rad(20.0)
    xi0_p = 1e-9 + 0.4e-9j
    z_target = 7.0
    v_p, v_s = material["v_p_m_s"], material["v_s_m_s"]

    phi, s1, s2 = _closed_form_reflected_p(z_target, theta, xi0_p, v_p, v_s)

    r = halfspace_reflected_p_above_observables_g_free(
        [[0.0, 0.0, z_target]],
        density_kg_m3=RHO,
        v_p_m_s=v_p,
        v_s_m_s=v_s,
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        incident_displacement_m=xi0_p,
    )
    k_p = 2.0 * np.pi * FREQ / v_p
    qx = k_p * np.sin(theta) * np.cos(azimuth)
    qy = k_p * np.sin(theta) * np.sin(azimuth)
    _assert_full_observables_match(phi, s1, s2, qx, qy, r)
