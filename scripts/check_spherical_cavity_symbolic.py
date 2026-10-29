"""CAS checks: real Cartesian coordinates; k,a,r>0; integer l>=0.

Run with ai-harness/bin/verify-symbolic --file <this file>.
General radial coefficient identities use DLMF recurrences as assumptions.
Polynomial/target-derivative checks cover degrees 0..5 explicitly.
"""

import sympy as sp


def derived_expression():
    x, y, z = sp.symbols("x y z", real=True)
    r, a, k = sp.symbols("r a k", positive=True)
    ell = sp.symbols("ell", integer=True, nonnegative=True)
    jl, next_j, previous_j = sp.symbols("jl next_j previous_j", real=True)
    residuals = []
    # d/dr [r^(l+2) j_(l+1)(kr)/k] = r^(l+2) j_l(kr).
    radial_derivative = (ell + 2) * r ** (ell + 1) * next_j / k + r ** (ell + 2) * (
        jl - (ell + 2) / (k * r) * next_j
    )
    residuals.append(sp.simplify(radial_derivative - r ** (ell + 2) * jl))
    # Full-space boundary value minus removed sphere; interior harmonic coefficient.
    boundary = (2 * ell + 1) / (k * a) * jl - next_j
    residuals.append(
        sp.simplify(
            boundary.subs(next_j, (2 * ell + 1) / (k * a) * jl - previous_j)
            - previous_j
        )
    )
    # Both radial branches solve the homogeneous Laplace radial equation.
    for power in (ell, -ell - 1):
        residuals.append(
            sp.simplify(
                sp.diff(r**power, r, 2)
                + 2 / r * sp.diff(r**power, r)
                - ell * (ell + 1) / r**2 * r**power
            )
        )
    radius2 = x * x + y * y + z * z
    harmonic, previous = sp.Integer(1), sp.Integer(0)
    angular, angular_previous = sp.Integer(0), sp.Integer(0)
    t = sp.symbols("t", real=True)
    for degree in range(6):
        if degree:
            factor = sp.Rational(2 * degree - 1, degree)
            harmonic, previous = (
                sp.expand(
                    factor * z * harmonic
                    - sp.Rational(degree - 1, degree) * radius2 * previous
                ),
                harmonic,
            )
            angular, angular_previous = (
                sp.expand(
                    factor * (x * previous + z * angular)
                    - sp.Rational(degree - 1, degree) * radius2 * angular_previous
                ),
                angular,
            )
        legendre = sp.expand(sp.legendre(degree, t))
        expected = sp.expand(
            sum(
                coefficient
                * z ** monomial[0]
                * radius2 ** ((degree - monomial[0]) // 2)
                for monomial, coefficient in sp.Poly(legendre, t).terms()
            )
        )
        residuals.append(sp.expand(harmonic - expected))
        residuals.append(sp.expand(sum(sp.diff(harmonic, v, 2) for v in (x, y, z))))
        expected_angular = sp.expand(
            sum(
                coefficient
                * monomial[0]
                * x
                * z ** (monomial[0] - 1)
                * radius2 ** ((degree - monomial[0]) // 2)
                for monomial, coefficient in sp.Poly(legendre, t).terms()
                if monomial[0]
            )
        )
        residuals.append(sp.expand(angular - expected_angular))
    # Cylindrical radius primitive used by the independent volume oracle.
    b, h = sp.symbols("b h", positive=True)
    residuals.append(
        sp.simplify(sp.diff(sp.sqrt(b * b + h * h) - h, b) - b / sp.sqrt(b * b + h * h))
    )
    # Hessian product-rule radial factor, for any real power.
    power = sp.symbols("power", real=True)
    radial = radius2 ** (power / 2)
    residuals.append(
        sp.simplify(
            sp.diff(radial, x, z)
            - power * (power - 2) * x * z * radius2 ** (power / 2 - 2)
        )
    )
    return sp.Add(*(sp.simplify(residual) ** 2 for residual in residuals))


def reference_expression():
    return sp.Integer(0)
