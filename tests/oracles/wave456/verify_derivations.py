"""CAS checks for Wave 4--6 independent reference formulas."""

from __future__ import annotations

import sympy as sp


def main() -> None:
    checks: list[str] = []

    def check(condition: bool, name: str) -> None:
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    x, y, z, mass = sp.symbols("x y z mass", real=True)
    k, q, s, rho, amplitude = sp.symbols(
        "k q s rho amplitude", positive=True, real=True
    )
    r = sp.sqrt(x**2 + y**2 + z**2)
    delta = sp.Matrix([x, y, z])

    potential = -mass / r
    acceleration = sp.Matrix([sp.diff(potential, v) for v in (x, y, z)])
    expected_acceleration = mass * delta / r**3
    check(
        all(
            sp.simplify(a - b) == 0
            for a, b in zip(acceleration, expected_acceleration, strict=True)
        ),
        "point acceleration",
    )

    target_gradient = -acceleration.jacobian((x, y, z))
    expected_gradient = mass * (3 * delta * delta.T / r**5 - sp.eye(3) / r**3)
    # x,y,z here are source-minus-target; target differentiation adds -1.
    check(
        all(
            sp.simplify(a - b) == 0
            for a, b in zip(target_gradient, expected_gradient, strict=True)
        ),
        "point target gradient",
    )

    # Central-second-moment multipole used by the opposed-rotor phasor.
    sxx, syy, szz, sxy, sxz, syz = sp.symbols("sxx syy szz sxy sxz syz", real=True)
    moment = sp.Matrix(((sxx, sxy, sxz), (sxy, syy, syz), (sxz, syz, szz)))
    sr = moment * delta
    trace = sp.trace(moment)
    contraction = 3 * (delta.dot(sr)) - trace * r**2
    multipole_potential = -mass / r - contraction / (2 * r**5)
    derived_acceleration = sp.Matrix(
        [sp.diff(multipole_potential, value) for value in (x, y, z)]
    )
    formula_acceleration = (
        mass * delta / r**3
        + (trace * delta - 3 * sr) / r**5
        + sp.Rational(5, 2) * contraction * delta / r**7
    )
    check(
        all(
            sp.simplify(a - b) == 0
            for a, b in zip(derived_acceleration, formula_acceleration, strict=True)
        ),
        "multipole acceleration",
    )
    identity = sp.eye(3)
    c_matrix = trace * identity - 3 * moment
    vector = trace * delta - 3 * sr
    gradient_contraction = 6 * sr - 2 * trace * delta
    literal_delta_gradient = (
        mass * (identity / r**3 - 3 * delta * delta.T / r**5)
        + c_matrix / r**5
        - 5 * vector * delta.T / r**7
        + sp.Rational(5, 2)
        * (delta * gradient_contraction.T + contraction * identity)
        / r**7
        - sp.Rational(35, 2) * contraction * delta * delta.T / r**9
    )
    literal_target_gradient = -literal_delta_gradient
    check(
        all(
            sp.simplify(a - b) == 0
            for a, b in zip(
                -derived_acceleration.jacobian((x, y, z)),
                literal_target_gradient,
                strict=True,
            )
        ),
        "literal expanded multipole target gradient",
    )

    # Fourier harmonic in the lower vacuum: Phi=-C exp(i*k*x+k*z).
    phi_air = -sp.symbols("C", real=True) * sp.exp(sp.I * k * x + k * z)
    d_air = sp.Matrix([sp.I * k, 0, k])
    air_acceleration = -sp.Matrix([sp.diff(phi_air, v) for v in (x, y, z)])
    check(
        all(
            sp.simplify(a - b) == 0
            for a, b in zip(air_acceleration, -phi_air * d_air, strict=True)
        ),
        "air acceleration Fourier derivative",
    )
    air_gradient = air_acceleration.jacobian((x, y, z))
    check(
        all(
            sp.simplify(a + b) == 0
            for a, b in zip(air_gradient, phi_air * (d_air * d_air.T), strict=True)
        ),
        "air gradient Fourier derivative",
    )
    check(
        sp.simplify(sum(sp.diff(phi_air, v, 2) for v in (x, y, z))) == 0,
        "air vacuum Laplacian",
    )

    # The s-exponential Rayleigh shear-potential contribution is divergence free.
    coef_x = 2 * q * s / (1 + s**2)
    coef_z = 2 / (1 + s**2)
    check(sp.simplify(q * coef_z * s - coef_x) == 0, "Rayleigh shear divergence")

    # Full bulk+sheet potential coefficient equals the usual Rayleigh gamma
    # only on the positive Rayleigh secular branch
    # (1+s^2)^2=4*q*s.  Substitute that constraint before simplifying.
    fz0 = 1 - 2 / (1 + s**2)
    coefficient_ratio = 1 + (1 - q) / (q * fz0)
    root_q_over_s = (1 + s**2) / (2 * s)
    gamma = (1 - root_q_over_s) / (q - root_q_over_s)
    secular_q = (1 + s**2) ** 2 / (4 * s)
    check(
        sp.factor((coefficient_ratio - gamma).subs(q, secular_q)) == 0,
        "Rayleigh bulk-sheet gamma",
    )

    # Dimension scaling is checked algebraically: mass scales by mu, every
    # length by ell, density by mu/ell**3, and k by 1/ell.
    ell, mu = sp.symbols("ell mu", positive=True, real=True)
    scaled_point_potential = potential.subs(
        {mass: mu * mass, x: ell * x, y: ell * y, z: ell * z}, simultaneous=True
    )
    check(
        sp.simplify(scaled_point_potential / potential - mu / ell) == 0,
        "point potential dimensions",
    )
    rayleigh_prefactor = rho * amplitude / k
    scaled_rayleigh_prefactor = rayleigh_prefactor.subs(
        {rho: mu * rho / ell**3, amplitude: ell * amplitude, k: k / ell},
        simultaneous=True,
    )
    check(
        sp.simplify(scaled_rayleigh_prefactor / rayleigh_prefactor - mu / ell) == 0,
        "Rayleigh potential dimensions",
    )

    # Zero-density/amplitude limits and reflection symmetry.
    rayleigh_phi = -2 * sp.pi * rho * amplitude * sp.exp(-k * z)
    check(sp.limit(rayleigh_phi, rho, 0) == 0, "zero density limit")
    check(sp.limit(rayleigh_phi, amplitude, 0) == 0, "zero amplitude limit")
    check(sp.simplify(potential.subs(x, -x) - potential) == 0, "point reflection")
    print(f"wave456 symbolic checks: {len(checks)} independent assertions passed")


if __name__ == "__main__":
    main()
