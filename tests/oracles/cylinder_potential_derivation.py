"""CAS certificate for V's own potential reduction and series recurrences.

Assumptions: R,r>0; source displacement is nonzero; d,x>0 for logarithms;
the material boundary is excluded. Limits d=0,x=0 are separate numerical
tests. Divergence identities hold off the target; the omitted sphere's
potential/first-derivative flux tends to zero for an interior target.
"""

import sympy as s


def residuals():
    x, y, z = coordinates = s.symbols("x y z", real=True)
    distance = s.sqrt(x * x + y * y + z * z)
    # Newton potential -> physical boundary potential by divergence theorem.
    yield s.simplify(sum(s.diff(v / distance, v) for v in coordinates) - 2 / distance)
    for j, value in enumerate(coordinates):
        # Its target derivative is the C surface acceleration, after a
        # second divergence identity. The source derivative has opposite sign.
        divergence = sum(s.diff(value * v / distance**3, v) for v in coordinates)
        yield s.simplify(divergence + s.diff(1 / distance, value))
        for k, other in enumerate(coordinates):
            expected = (
                3 * value * other - (distance**2 if j == k else 0)
            ) / distance**5
            yield s.simplify(-s.diff(value / distance**3, other) - expected)
    radius, radial, u, q = s.symbols("R r u q", positive=True)
    angular = s.symbols("c", real=True)
    side = radius * (radius - radial * angular) * s.asinh(u / s.sqrt(q))
    log_part = radius**2 * s.asinh(u / s.sqrt(q))
    radial_part = radial * (-radius * angular * s.asinh(u / s.sqrt(q)))
    yield s.simplify(side - log_part - radial_part)
    yield s.simplify(s.diff(s.asinh(u / s.sqrt(q)), u) - 1 / s.sqrt(q + u * u))
    # Integrated K-series: I_n(y)=integral_0^y(1-v²)^(n-1)dv.
    v = s.symbols("v", real=True)
    for n in range(2, 10):
        previous = s.integrate((1 - v**2) ** (n - 2), (v, 0, y))
        recurrence = ((2 * n - 2) * previous + y * (1 - y**2) ** (n - 1)) / (2 * n - 1)
        yield s.simplify(s.diff(recurrence, y) - (1 - y**2) ** (n - 1))
    # Complete-series coefficients equal log(2/(1+sqrt(1-t))).
    t = s.symbols("t")
    complete = s.log(2 / (1 + s.sqrt(1 - t)))
    yield s.simplify(s.diff(complete, t) - (1 / s.sqrt(1 - t) - 1) / (2 * t))
    n = s.symbols("n", integer=True, positive=True)
    alpha = (s.rf(s.Rational(1, 2), n) / s.factorial(n)) ** 2
    beta_integral = s.sqrt(s.pi) * s.gamma(n) / (2 * s.gamma(n + s.Rational(1, 2)))
    yield s.simplify(
        s.combsimp(
            alpha * beta_integral - s.rf(s.Rational(1, 2), n) / (2 * n * s.factorial(n))
        )
    )
    for n in range(1, 9):
        coefficient = (s.rf(s.Rational(1, 2), n) / s.factorial(n)) ** 2
        integral = s.integrate((1 - v**2) ** (n - 1), (v, 0, 1))
        yield s.simplify(
            s.diff(complete, t, n).subs(t, 0) / s.factorial(n) - coefficient * integral
        )
    # Local log moments: integration by parts and rational recurrence.
    d, x = s.symbols("d x", positive=True)
    rational = s.atan(x / d) / d
    for p in range(8):
        rational = x ** (2 * p + 1) / (2 * p + 1) - d**2 * rational
        primitive = (x ** (2 * p + 1) * s.log(d * d + x * x) - 2 * rational) / (
            2 * p + 1
        )
        yield s.simplify(s.diff(primitive, x) - x ** (2 * p) * s.log(d * d + x * x))
    # All far-field potentials differentiate to C's acceleration formula.
    r, z = s.symbols("r z", positive=True)
    distance = s.sqrt(r * r + z * z)
    for degree in range(0, 10, 2):
        poly = s.legendre(degree, t)
        potential = -poly.subs(t, z / distance) / distance ** (degree + 1)
        dp = s.diff(poly, t).subs(t, z / distance)
        angular = (degree + 1) * poly.subs(t, z / distance) + z / distance * dp
        yield s.simplify(-s.diff(potential, r) + angular * r / distance ** (degree + 3))
        yield s.simplify(
            -s.diff(potential, z)
            - (dp - angular * z / distance) / distance ** (degree + 2)
        )


def derived_expression():
    checks = list(residuals())
    assert all(value == 0 for value in checks), checks
    return sum(value**2 for value in checks)


def reference_expression():
    return s.Integer(0)


if __name__ == "__main__":
    checks = list(residuals())
    assert all(value == 0 for value in checks), checks
    print(f"{len(checks)} exact cylinder-potential identities verified")
