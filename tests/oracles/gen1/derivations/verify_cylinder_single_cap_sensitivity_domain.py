"Independent symbolic review of the one-cap sensitivity formulas, not a total-error bound."  # noqa: E501

import json

import sympy as s

a, gap, radius, alpha, eta = s.symbols("a gap R alpha eta", positive=True)
beta, m = s.symbols("beta m", nonnegative=True)
direction = s.symbols("direction", real=True)
checks = []
polynomials = [-gap, 1 - gap * a, 2 * a - gap * a * a]
poly_a = [0, -gap, 2 - 2 * gap * a]
poly_gap = [-1, -a, -a * a]
for n in range(3):
    value = (-direction) ** n * a**n * s.exp(-a * gap)
    proposed = (-direction) ** n * s.exp(-a * gap) * polynomials[n] / radius
    for name, residual in (
        ("alpha", s.diff(value, a) / radius - proposed),
        (
            "alpha_then_a",
            s.diff(proposed, a)
            - (-direction) ** n
            * s.exp(-a * gap)
            * (poly_a[n] - gap * polynomials[n])
            / radius,
        ),
        (
            "alpha_then_gap",
            s.diff(proposed, gap)
            - (-direction) ** n
            * s.exp(-a * gap)
            * (poly_gap[n] - a * polynomials[n])
            / radius,
        ),
    ):
        assert s.simplify(residual) == 0, (n, name, residual)
        checks.append(f"axial{n}:{name}")
# Cover the two nonzero real Bessel branches explicitly; J=0 is outside the contract.
J = s.Function("positive_J", positive=True)
for polarity in (-1, 1):
    signed = polarity * J(alpha)
    normalization = 1 / s.sqrt(eta * signed**2 * (1 - m * m / (alpha * alpha)))
    coefficient = normalization / ((alpha / radius) ** 2 + beta * beta)
    expected = (
        -s.diff(signed, alpha) / signed
        - m * m / (alpha * (alpha * alpha - m * m))
        - 2 * alpha / (alpha * alpha + radius * radius * beta * beta)
    )
    assert s.simplify(s.diff(coefficient, alpha) / coefficient - expected) == 0
    checks.append(f"normalization_and_inverse_wavenumber_squared_chain_Jsign{polarity}")
x, y = s.symbols("x y", real=True)
for order in range(4):
    harmonic = sum(
        (-1) ** j
        * (a / 2) ** (order + 2 * j)
        * (x + s.I * y) ** order
        * (x * x + y * y) ** j
        / (s.factorial(j) * s.factorial(order + j))
        for j in range(2)
    )
    for azimuth, field in (
        ("cos", s.re(s.expand(harmonic))),
        ("sin", s.im(s.expand(harmonic))),
    ):
        for label, derivatives in (
            ("value", ()),
            ("x", (x,)),
            ("y", (y,)),
            ("xx", (x, x)),
            ("xy", (x, y)),
            ("yy", (y, y)),
        ):
            obtained = (
                s.diff(field, *derivatives).subs({x: 0, y: 0})
                if derivatives
                else field.subs({x: 0, y: 0})
            )
            expected_value = 0
            if order == 0 and azimuth == "cos":
                expected_value = (
                    1
                    if label == "value"
                    else -a * a / 2
                    if label in ("xx", "yy")
                    else 0
                )
            if order == 1 and label == ("x" if azimuth == "cos" else "y"):
                expected_value = a / 2
            if order == 2:
                if azimuth == "cos" and label in ("xx", "yy"):
                    expected_value = a * a / 4 * (1 if label == "xx" else -1)
                elif azimuth == "sin" and label == "xy":
                    expected_value = a * a / 4
            assert s.simplify(obtained - expected_value) == 0, (
                order,
                azimuth,
                label,
                obtained,
            )
            checks.append(f"axis_m{order}_{azimuth}:{label}")
print(
    json.dumps(
        {
            "status": "passed",
            "symbolic_checks": len(checks),
            "checks": checks,
            "assumptions": (
                "a,gap,R>0; direction=+/-1; alpha>m>=0, eta>0, J(alpha)!=0; "
                "beta>=0. Axis Bessel series terms j>=2 have Cartesian degree>=4 "
                "and cannot affect derivatives of order<=2."
            ),
            "claim": (
                "Exact axial/alpha chain and m0..3 axis identities only; "
                "no finite-precision or total uncertainty bound claim."
            ),
            "sympy_version": s.__version__,
        },
        indent=2,
    )
)
