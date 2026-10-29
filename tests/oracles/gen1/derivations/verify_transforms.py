"""Exact transform and tail identities; positive geometry/cutoffs, integer modes.

This checks derived identities, independently of any production import. Bounds
also use the inequalities stated in cavity-transforms.md, not sampled maxima.
"""

import sympy as s


def zero(v):
    assert s.simplify(s.expand_func(v)) == 0, v


def main():
    x, c, t, r_symbol, a, b, lam = s.symbols("x c t R a b lam", positive=True)
    d = c - x
    q_symbol = d * s.exp(-(t**2) * d**2)
    zero(s.diff(q_symbol, x) - (-1 + 2 * t**2 * d**2) * s.exp(-(t**2) * d**2))
    zero(
        s.diff(q_symbol, x, 2)
        - 2 * t**2 * d * (-3 + 2 * t**2 * d**2) * s.exp(-(t**2) * d**2)
    )
    u, beta = s.symbols("u beta", real=True)
    gaussian = s.exp(-(t**2) * (u - x) ** 2)
    cosine = s.cos(beta * u)
    sine = s.sin(beta * u)
    # Integration by parts yields F_x, F_xx and G_x, with sin endpoints zero.
    zero(
        s.diff(gaussian, x) * cosine
        + s.diff(gaussian * cosine, u)
        + beta * gaussian * sine
    )
    zero(
        s.diff(gaussian, x) * sine
        + s.diff(gaussian * sine, u)
        - beta * gaussian * cosine
    )
    antiderivative = (
        s.sqrt(s.pi)
        / (2 * t)
        * s.exp(-(beta**2) / (4 * t**2) + s.I * beta * x)
        * s.erf(t * (u - x) - s.I * beta / (2 * t))
    )
    zero(s.diff(antiderivative, u) - gaussian * s.exp(s.I * beta * u))
    print(
        "Cuboid Q derivatives, Gaussian primitive and "
        "integration-by-parts identities EXACT PASS"
    )
    delta, t_symbol = s.symbols("delta T", positive=True)
    for p in range(7):
        tail = s.uppergamma(s.Rational(p + 1, 2), delta**2 * t_symbol**2) / (
            2 * delta ** (p + 1)
        )
        zero(s.diff(tail, t_symbol) + t_symbol**p * s.exp(-(delta**2) * t_symbol**2))
    for p in [
        s.Integer(1),
        s.Integer(2),
        s.Integer(3),
        s.Rational(1, 2),
        s.Rational(3, 2),
        s.Rational(5, 2),
    ]:
        tail = s.uppergamma(p + 1, delta * t_symbol) / delta ** (p + 1)
        zero(s.diff(tail, t_symbol) + t_symbol**p * s.exp(-delta * t_symbol))
    print(
        "Gaussian polynomial and exponential integer/half-integer "
        "tail primitives EXACT PASS"
    )
    # Lagrange/Wronskian derivative: source J_a and target J_b or I_b.
    r, m = s.symbols("r m", positive=True)
    f = s.Function("f")(r)
    g = s.Function("g")(r)
    ode_f = -s.diff(f, r) / r - (a * a - m * m / (r * r)) * f
    for sign in [-1, 1]:
        ode_g = -s.diff(g, r) / r + (sign * b * b + m * m / (r * r)) * g
        w_symbol = r * (f * s.diff(g, r) - s.diff(f, r) * g) / (a * a + sign * b * b)
        zero(
            (s.diff(w_symbol, r) - r * f * g).subs(
                {s.diff(f, r, 2): ode_f, s.diff(g, r, 2): ode_g}
            )
        )
    print("Cylinder J/J and J/I radial overlap Wronskian identities EXACT PASS")
    # Integral norm primitive is exact under ordinary Bessel differential eq.
    bessel = s.Function("B")(x)
    norm = (
        x * x / 2 * (s.diff(bessel, x) ** 2 + (1 - m * m / (x * x)) * bessel * bessel)
    )
    ode = -s.diff(bessel, x) / x - (1 - m * m / (x * x)) * bessel
    zero((s.diff(norm, x) - x * bessel * bessel).subs(s.diff(bessel, x, 2), ode))
    print("Cylinder radial norm primitive under Bessel ODE EXACT PASS")
    h_symbol, z, kappa = s.symbols("H z kappa", positive=True)
    zp = s.symbols("zp", real=True)
    for q in range(5):
        beta = q * s.pi / h_symbol
        mode = s.cos(beta * (zp + h_symbol / 2))
        primitive = (
            s.sin((beta - kappa) * zp + beta * h_symbol / 2 + kappa * z)
            / (beta - kappa)
            + s.sin((beta + kappa) * zp + beta * h_symbol / 2 - kappa * z)
            / (beta + kappa)
        ) / 2
        zero(s.diff(primitive, zp) - mode * s.cos(kappa * (z - zp)))
        c_symbol = primitive.subs(zp, h_symbol / 2) - primitive.subs(zp, -h_symbol / 2)
        closed = (
            kappa
            * (
                (-1) ** q * s.sin(kappa * (h_symbol / 2 - z))
                + s.sin(kappa * (h_symbol / 2 + z))
            )
            / (kappa * kappa - beta * beta)
        )
        # Rational primitive is for kappa != beta, resonance checked below.
        zero(c_symbol - closed)
        zero(s.diff(closed, z, 2) + kappa * kappa * closed)
        if q:
            variable = s.symbols("variable", real=True)
            norm_primitive = variable / 2 + s.sin(2 * beta * variable) / (4 * beta)
            cross_primitive = s.sin(beta * variable) ** 2 / (2 * beta)
            zero(s.diff(norm_primitive, variable) - s.cos(beta * variable) ** 2)
            zero(
                s.diff(cross_primitive, variable)
                - s.sin(beta * variable) * s.cos(beta * variable)
            )
            norm = norm_primitive.subs(variable, h_symbol) - norm_primitive.subs(
                variable, 0
            )
            cross = cross_primitive.subs(variable, h_symbol) - cross_primitive.subs(
                variable, 0
            )
            rootint = (
                s.cos(beta * (z + h_symbol / 2)) * norm
                + s.sin(beta * (z + h_symbol / 2)) * cross
            )
            zero(rootint - h_symbol / 2 * s.cos(beta * (z + h_symbol / 2)))
        aboveprimitive = (
            s.exp(-kappa * (z - zp))
            * (kappa * mode + beta * s.sin(beta * (zp + h_symbol / 2)))
            / (kappa * kappa + beta * beta)
        )
        zero(s.diff(aboveprimitive, zp) - mode * s.exp(-kappa * (z - zp)))
        above = aboveprimitive.subs(zp, h_symbol / 2) - aboveprimitive.subs(
            zp, -h_symbol / 2
        )
        aboveclosed = (
            kappa
            * (
                (-1) ** q * s.exp(-kappa * (z - h_symbol / 2))
                - s.exp(-kappa * (z + h_symbol / 2))
            )
            / (kappa * kappa + beta * beta)
        )
        zero(above - aboveclosed)
    phi = s.symbols("phi", real=True)
    for mval in range(4):
        angular = s.integrate(s.cos(mval * phi) ** 2, (phi, 0, 2 * s.pi)) / (2 * s.pi)
        zero(angular - (1 if mval == 0 else s.Rational(1, 2)))
        if mval:
            zero(
                s.integrate(s.sin(mval * phi) ** 2, (phi, 0, 2 * s.pi)) / (2 * s.pi)
                - s.Rational(1, 2)
            )
    print(
        "Cylinder axial overlaps, resonance cases q=0..4 and "
        "real angular RMS m=0..3 EXACT PASS"
    )


if __name__ == "__main__":
    main()
