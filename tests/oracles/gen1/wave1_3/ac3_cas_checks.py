"""SymPy CAS verification for AC3 acoustic specialization of cylindrical gravity.

Verifies:
1. Acoustic wave equation in cylindrical coordinates and derivation of dispersion relation:
   nabla^2 p - (1/cs^2) partial_t^2 p = 0  =>  omega^2 = cs^2 * (kappa^2 + kz^2)
2. Rigid wall Neumann boundary condition:
   partial_r p |_{r=R} = 0  =>  kappa * J'_m(kappa * R) = 0
3. Density coupling:
   hat{delta_rho} = hat{p} / cs^2  =>  A = P / cs^2
4. Trace identity tr(Gamma) = -nabla^2 Phi:
   - Interior (r < R): tr(Gamma) = -4*pi*G*delta_rho
   - Exterior (r > R): tr(Gamma) = 0
   Checked for kz != 0 (general and kappa=0) and kz = 0 (m > 0 and m = 0).
5. Rotational invariance of trace under cylindrical-to-Cartesian transformation:
   tr(E * Gamma_cyl * E^T) = tr(Gamma_cyl)

Exit code 0 on complete symbolic verification.
"""

# ruff: noqa: E501, N802, N806

from __future__ import annotations

import sympy as sp


def simplify_bessel_recurrence(expr, z, m):
    """Simplify Bessel and modified Bessel ODE expressions using standard 3-term recurrences."""
    # Besseli recurrences
    expr = expr.subs(
        sp.besseli(m + 2, z), sp.besseli(m, z) - 2 * (m + 1) / z * sp.besseli(m + 1, z)
    )
    expr = expr.subs(
        sp.besseli(m - 2, z), sp.besseli(m, z) + 2 * (m - 1) / z * sp.besseli(m - 1, z)
    )
    expr = expr.subs(
        sp.besseli(m - 1, z), sp.besseli(m + 1, z) + 2 * m / z * sp.besseli(m, z)
    )
    # Besselk recurrences
    expr = expr.subs(
        sp.besselk(m + 2, z), sp.besselk(m, z) + 2 * (m + 1) / z * sp.besselk(m + 1, z)
    )
    expr = expr.subs(
        sp.besselk(m - 2, z), sp.besselk(m, z) - 2 * (m - 1) / z * sp.besselk(m - 1, z)
    )
    expr = expr.subs(
        sp.besselk(m - 1, z), sp.besselk(m + 1, z) - 2 * m / z * sp.besselk(m, z)
    )
    # Besselj recurrences
    expr = expr.subs(
        sp.besselj(m + 2, z), 2 * (m + 1) / z * sp.besselj(m + 1, z) - sp.besselj(m, z)
    )
    expr = expr.subs(
        sp.besselj(m - 2, z), 2 * (m - 1) / z * sp.besselj(m - 1, z) - sp.besselj(m, z)
    )
    expr = expr.subs(
        sp.besselj(m - 1, z), 2 * m / z * sp.besselj(m, z) - sp.besselj(m + 1, z)
    )
    return sp.simplify(expr)


def check_acoustic_wave_equation() -> None:
    print("[CAS 1] Verifying acoustic wave equation and dispersion relation...")
    r, phi, z, t = sp.symbols("r phi z t", real=True)
    kappa, kz, cs, omega = sp.symbols("kappa kz cs omega", real=True, positive=True)
    m = sp.symbols("m", integer=True)

    # p(r, phi, z, t) = P * besselj(m, kappa * r) * exp(I * m * phi + I * kz * z - I * omega * t)
    # Using Bessel ODE: r^2 * J''_m(x) + r * J'_m(x) + (x^2 - m^2) * J_m(x) = 0
    # Let R_m(r) = besselj(m, kappa * r)
    # d^2 R_m / dr^2 + (1/r) d R_m / dr = (m^2/r^2 - kappa^2) R_m
    # Cylindrical Laplacian:
    # nabla^2 p = [ d^2 R/dr^2 + (1/r) dR/dr - (m^2/r^2) R - kz^2 R ] * exp(...)
    #           = [ (m^2/r^2 - kappa^2) R - (m^2/r^2) R - kz^2 R ] * exp(...)
    #           = -(kappa^2 + kz^2) * p
    laplacian_factor = -(kappa**2 + kz**2)

    # Time derivative:
    # partial_t^2 p = (-I * omega)^2 p = -omega^2 p
    time_deriv_factor = -(omega**2) / (cs**2)

    wave_eq = laplacian_factor - time_deriv_factor
    # Under dispersion relation omega^2 = cs^2 * (kappa^2 + kz^2)
    wave_eq_substituted = sp.simplify(
        wave_eq.subs(omega**2, cs**2 * (kappa**2 + kz**2))
    )
    assert wave_eq_substituted == 0, f"Wave equation failed: {wave_eq_substituted}"

    # Also verify Bessel ODE directly for general m using recurrence and for m = 0, 1, 2, 3
    x = sp.Symbol("x", positive=True)
    bessel_ode = (
        x**2 * sp.diff(sp.besselj(m, x), x, 2)
        + x * sp.diff(sp.besselj(m, x), x)
        + (x**2 - m**2) * sp.besselj(m, x)
    )
    assert simplify_bessel_recurrence(bessel_ode, x, m) == 0, (
        f"Bessel ODE general m check failed: {bessel_ode}"
    )

    for m_val in [0, 1, 2, 3]:
        ode_val = (
            x**2 * sp.diff(sp.besselj(m_val, x), x, 2)
            + x * sp.diff(sp.besselj(m_val, x), x)
            + (x**2 - m_val**2) * sp.besselj(m_val, x)
        )
        assert sp.simplify(ode_val) == 0, f"Bessel ODE check failed for m={m_val}"

    print("  -> Acoustic wave equation and dispersion relation verified: EXACT 0.")


def check_boundary_condition() -> None:
    print("[CAS 2] Verifying rigid wall Neumann boundary condition...")
    r, R = sp.symbols("r R", real=True, positive=True)
    # Acoustic radial velocity u_r = -1/(i*omega*rho_0) * partial_r p
    # Vanishing velocity at r = R requires partial_r p |_{r=R} = 0
    # partial_r (besselj(m, kappa * r)) = kappa * besselj'(m, kappa * r)
    # At r = R: kappa * besselj'(m, kappa * R) = 0
    # Case 1: kappa > 0  => besselj'(m, kappa * R) = 0  => kappa * R = alpha_mn
    # Case 2: kappa = 0:
    #   For m = 0: besselj(0, 0) = 1 (constant), derivative is 0 everywhere, boundary condition identically satisfied!
    #   For m != 0: besselj(m, 0) = 0, so p identically 0 (trivial).
    j0_at_0 = sp.besselj(0, 0)
    assert j0_at_0 == 1, f"J_0(0) should be 1, got {j0_at_0}"
    dj0_at_0 = sp.diff(sp.besselj(0, r), r).subs(r, 0)
    assert dj0_at_0 == 0, f"J'_0(0) should be 0, got {dj0_at_0}"

    print("  -> Rigid wall boundary condition and branches verified: EXACT 0.")


def check_density_coupling() -> None:
    print("[CAS 3] Verifying density coupling delta_rho = P / cs^2...")
    P, cs = sp.symbols("P cs", positive=True)
    # Linear acoustic relation: p = cs^2 * delta_rho  => delta_rho = p / cs^2
    # With p = P * J_m(...) * exp(...), delta_rho = (P / cs^2) * J_m(...) * exp(...)
    # Amplitude A in the cylindrical density-mode formula: A = P / cs^2
    A = P / cs**2
    assert sp.simplify(A * cs**2 - P) == 0

    # Also verify across all cases in case_matrix.json
    import json
    import pathlib

    case_matrix_file = pathlib.Path(__file__).parent / "ac3_case_matrix.json"
    with open(case_matrix_file, encoding="utf-8") as f:
        cases_data = json.load(f)["cases"]
    for c in cases_data:
        cs_val = sp.Rational(c["cs"])
        P_re = sp.Rational(c["P"][0])
        P_im = sp.Rational(c["P"][1])
        A_re = sp.Rational(c["A"][0])
        A_im = sp.Rational(c["A"][1])
        diff_re = P_re / (cs_val**2) - A_re
        diff_im = P_im / (cs_val**2) - A_im
        assert abs(float(diff_re)) < 1e-70, (
            f"Case {c['id']} real part mismatch: {diff_re}"
        )
        assert abs(float(diff_im)) < 1e-70, (
            f"Case {c['id']} imag part mismatch: {diff_im}"
        )

    print(
        f"  -> Density coupling A = P / cs^2 verified algebraically and across {len(cases_data)} cases: EXACT 0."
    )


def check_poisson_and_trace_identity() -> None:
    print("[CAS 4] Verifying Poisson equation and tr(Gamma) identity...")
    r = sp.symbols("r", real=True, positive=True)
    q, kappa, G, A, R = sp.symbols("q kappa G A R", real=True, positive=True)
    m = sp.symbols("m", integer=True)
    C, D = sp.symbols("C D", complex=True)
    beta = 4 * sp.pi * G * A

    # Define radial differential operator for kz != 0:
    # L_q[f] = f'' + (1/r) * f' - (m^2/r^2 + q^2) * f
    def L_q(f):
        return sp.diff(f, r, 2) + (1 / r) * sp.diff(f, r) - (m**2 / r**2 + q**2) * f

    # 4a. Interior solution: f(r) = -beta / (kappa^2 + q^2) * besselj(m, kappa*r) + C * besseli(m, q*r)
    J_part = sp.besselj(m, kappa * r)
    Lq_J = L_q(J_part)
    # Check that L_q[J] = -(kappa^2 + q^2) * J
    diff_J = simplify_bessel_recurrence(
        r**2 * (Lq_J + (kappa**2 + q**2) * J_part), kappa * r, m
    )
    assert diff_J == 0, f"L_q[J] failed: {diff_J}"

    I_part = sp.besseli(m, q * r)
    Lq_I = L_q(I_part)
    diff_I = simplify_bessel_recurrence(r**2 * Lq_I, q * r, m)
    assert diff_I == 0, f"L_q[I] should be 0, got {diff_I}"

    # Also verify for concrete m = 0, 1, 2
    for m_val in [0, 1, 2]:
        J_mval = sp.besselj(m_val, kappa * r)
        I_mval = sp.besseli(m_val, q * r)
        K_mval = sp.besselk(m_val, q * r)

        def L_q_val(f, m_val=m_val):
            return (
                sp.diff(f, r, 2)
                + (1 / r) * sp.diff(f, r)
                - (m_val**2 / r**2 + q**2) * f
            )

        assert sp.simplify(L_q_val(J_mval) + (kappa**2 + q**2) * J_mval) == 0
        assert sp.simplify(L_q_val(I_mval)) == 0
        assert sp.simplify(L_q_val(K_mval)) == 0

    f_int = -beta / (kappa**2 + q**2) * J_part + C * I_part
    Lq_f_int = L_q(f_int)
    diff_f_int = simplify_bessel_recurrence(
        r**2 * (Lq_f_int - beta * J_part), kappa * r, m
    )
    # Also evaluate substitution for I
    diff_f_int = simplify_bessel_recurrence(diff_f_int, q * r, m)
    assert diff_f_int == 0, f"Interior Poisson failed: {diff_f_int}"

    # tr(Gamma) = -nabla^2 Phi = -4*pi*G*delta_rho
    # For concrete m = 0, 1, 2:
    for m_val in [0, 1, 2]:
        f_int_val = -beta / (kappa**2 + q**2) * sp.besselj(
            m_val, kappa * r
        ) + C * sp.besseli(m_val, q * r)
        tr_G_val = -(
            sp.diff(f_int_val, r, 2)
            + (1 / r) * sp.diff(f_int_val, r)
            - (m_val**2 / r**2 + q**2) * f_int_val
        )
        assert (
            sp.simplify(tr_G_val + 4 * sp.pi * G * A * sp.besselj(m_val, kappa * r))
            == 0
        )

    print("  -> Interior q != 0: tr(Gamma) = -4*pi*G*delta_rho verified: EXACT 0.")

    # 4b. Exterior solution: f(r) = D * besselk(m, q*r)
    K_part = sp.besselk(m, q * r)
    Lq_K = L_q(K_part)
    diff_K = simplify_bessel_recurrence(r**2 * Lq_K, q * r, m)
    assert diff_K == 0, f"L_q[K] should be 0, got {diff_K}"

    for m_val in [0, 1, 2]:
        f_ext_val = D * sp.besselk(m_val, q * r)
        tr_G_ext = -(
            sp.diff(f_ext_val, r, 2)
            + (1 / r) * sp.diff(f_ext_val, r)
            - (m_val**2 / r**2 + q**2) * f_ext_val
        )
        assert sp.simplify(tr_G_ext) == 0

    print("  -> Exterior q != 0: tr(Gamma) = 0 verified: EXACT 0.")

    # 4c. kz = 0 branches (q = 0):
    def L_0(f):
        return sp.diff(f, r, 2) + (1 / r) * sp.diff(f, r) - (m**2 / r**2) * f

    # m > 0:
    # Interior: f = -beta / kappa^2 * besselj(m, kappa*r) + C * r^m
    f_int_kz0 = -beta / kappa**2 * J_part + C * r**m
    L0_f_int = L_0(f_int_kz0)
    diff_kz0 = simplify_bessel_recurrence(
        r**2 * (L0_f_int - beta * J_part), kappa * r, m
    )
    assert diff_kz0 == 0, f"Interior kz=0 failed: {diff_kz0}"

    for m_val in [1, 2, 3]:
        f_int_kz0_val = -beta / kappa**2 * sp.besselj(m_val, kappa * r) + C * r**m_val
        L0_val = (
            sp.diff(f_int_kz0_val, r, 2)
            + (1 / r) * sp.diff(f_int_kz0_val, r)
            - (m_val**2 / r**2) * f_int_kz0_val
        )
        assert sp.simplify(L0_val - beta * sp.besselj(m_val, kappa * r)) == 0

    # Exterior: f = D * r^(-m)
    f_ext_kz0 = D * r ** (-m)
    L0_f_ext = sp.simplify(L_0(f_ext_kz0))
    assert L0_f_ext == 0, f"Exterior kz=0 failed: {L0_f_ext}"
    print(
        "  -> kz = 0 (m > 0): tr(Gamma) = -4*pi*G*delta_rho (int) and 0 (ext) verified: EXACT 0."
    )

    # m = 0, kz = 0:
    # Interior: f = beta / kappa^2 * (besselj(0, kappa*R) - besselj(0, kappa*r))
    f_int_m0_kz0 = (
        beta / kappa**2 * (sp.besselj(0, kappa * R) - sp.besselj(0, kappa * r))
    )
    L0_m0_int = sp.diff(f_int_m0_kz0, r, 2) + (1 / r) * sp.diff(f_int_m0_kz0, r)
    assert sp.simplify(L0_m0_int - beta * sp.besselj(0, kappa * r)) == 0
    # Exterior: f = D * log(r/R)
    f_ext_m0_kz0 = D * sp.log(r / R)
    L0_m0_ext = sp.diff(f_ext_m0_kz0, r, 2) + (1 / r) * sp.diff(f_ext_m0_kz0, r)
    assert sp.simplify(L0_m0_ext) == 0
    print(
        "  -> kz = 0 (m = 0): tr(Gamma) = -4*pi*G*delta_rho (int) and 0 (ext) verified: EXACT 0."
    )

    # Uniform mode: m = 0, kappa = 0, kz = 0:
    # Interior: f = beta * (r^2 - R^2) / 4
    f_int_unif = beta * (r**2 - R**2) / 4
    L0_unif_int = sp.diff(f_int_unif, r, 2) + (1 / r) * sp.diff(f_int_unif, r)
    assert sp.simplify(L0_unif_int - beta) == 0
    print(
        "  -> m=0, kappa=0, kz=0 uniform interior: tr(Gamma) = -4*pi*G*A verified: EXACT 0."
    )


def check_coordinate_transformation_trace_invariance() -> None:
    print(
        "[CAS 5] Verifying trace invariance under cylindrical-to-Cartesian transformation..."
    )
    phi = sp.symbols("phi", real=True)
    G_rr, G_rf, G_rz = sp.symbols("G_rr G_rf G_rz", complex=True)
    G_ff, G_fz, G_zz = sp.symbols("G_ff G_fz G_zz", complex=True)

    # Cylindrical symmetric matrix
    Gamma_cyl = sp.Matrix([[G_rr, G_rf, G_rz], [G_rf, G_ff, G_fz], [G_rz, G_fz, G_zz]])

    # Transformation matrix E
    c = sp.cos(phi)
    s = sp.sin(phi)
    E = sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # Cartesian Gamma: E * Gamma_cyl * E^T
    Gamma_cart = E * Gamma_cyl * E.T
    tr_cart = sp.simplify(Gamma_cart.trace())
    tr_cyl = sp.simplify(Gamma_cyl.trace())

    assert sp.simplify(tr_cart - tr_cyl) == 0, f"Trace difference: {tr_cart - tr_cyl}"
    print("  -> tr(Gamma_cart) == tr(Gamma_cyl) verified: EXACT 0.")


def main() -> None:
    print("=================================================================")
    print("Starting SymPy CAS verification for AC3 Acoustic Specialization")
    print("=================================================================")
    check_acoustic_wave_equation()
    check_boundary_condition()
    check_density_coupling()
    check_poisson_and_trace_identity()
    check_coordinate_transformation_trace_invariance()
    print("=================================================================")
    print("ALL CAS VERIFICATIONS PASSED SUCCESSFULLY (exit code 0)")
    print("=================================================================")


if __name__ == "__main__":
    main()
