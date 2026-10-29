"""CAS verification of the U4 potential derivatives.

Assumptions: r > 0, c > 0, and the source history is C1 at
the retarded query.  The radial identities used are
J'(r) = r F(t-r/c)/c**2 and
J''(r) = F(t-r/c)/c**2 - r F'(t-r/c)/c**3.
"""

from __future__ import annotations

import sympy as sp


def main() -> None:
    x, y, z, c, t = sp.symbols("x y z c t", positive=True, finite=True)
    e0, e1, e2 = sp.symbols("e0 e1 e2", real=True)
    coords = (x, y, z)
    e = sp.Matrix((e0, e1, e2))
    r = sp.sqrt(x**2 + y**2 + z**2)
    n = sp.Matrix((x, y, z)) / r
    mu = e.dot(n)
    j = sp.Function("J")(r)
    f = sp.Function("f")(r)
    f1 = sp.Function("f1")(r)
    phi = -mu * j / r**2
    acceleration_from_phi = sp.Matrix(
        [-sp.diff(phi, coordinate) for coordinate in coords]
    )
    acceleration_contract = j / r**3 * e + (f / (c**2 * r) - 3 * j / r**3) * mu * n

    def replace_radial_derivatives(expression):
        def replacement(item):
            if not isinstance(item, sp.Subs):
                return item
            derivative = item.expr
            if not isinstance(derivative, sp.Derivative):
                return item
            if derivative.expr.func != sp.Function("J"):
                return item
            if derivative.derivative_count == 1:
                return r * f / c**2
            if derivative.derivative_count == 2:
                return f / c**2 - r * f1 / c**3
            return item

        return expression.replace(lambda item: isinstance(item, sp.Subs), replacement)

    acceleration_difference = acceleration_from_phi - acceleration_contract
    acceleration_difference = acceleration_difference.applyfunc(
        replace_radial_derivatives
    )
    acceleration_difference = acceleration_difference.applyfunc(sp.factor)
    if any(item != 0 for item in acceleration_difference):
        raise AssertionError(f"a != -grad(Phi): {acceleration_difference}")

    # Differentiate the unsimplified -grad(Phi), then apply both radial J rules.
    gamma_from_phi = sp.Matrix(
        3,
        3,
        lambda i, k: sp.diff(acceleration_from_phi[i], coords[k]),
    )
    c_term = f / (c**2 * r**2) - 3 * j / r**4
    d_term = 15 * j / r**4 - 6 * f / (c**2 * r**2) - f1 / (c**3 * r)
    gamma_contract = (
        c_term * (e * n.T + n * e.T + mu * sp.eye(3)) + mu * d_term * n * n.T
    )
    gamma_difference = gamma_from_phi - gamma_contract
    gamma_difference = gamma_difference.applyfunc(replace_radial_derivatives)
    gamma_difference = gamma_difference.applyfunc(sp.factor)
    if any(item != 0 for item in gamma_difference):
        raise AssertionError(f"Gamma != grad(a): {gamma_difference}")
    trace_expected = -mu * (f / (c**2 * r**2) + f1 / (c**3 * r))
    if sp.factor(sp.trace(gamma_contract) - trace_expected) != 0:
        raise AssertionError("point-force density omits retarded-force contribution")
    trace_actual = replace_radial_derivatives(sp.trace(gamma_from_phi))
    if sp.factor(trace_actual - trace_expected) != 0:
        raise AssertionError("density != laplacian(Phi)/(4*pi*G)")
    print("U4 symbolic checks passed: derivatives and full force/dforce density trace")


if __name__ == "__main__":
    main()
