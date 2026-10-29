"""Exact cylindrical Cartesian derivative, axis and tail-inequality checks.

Assumptions r>0 for cylindrical chart, positive K arguments; m integer >=0.
Axis statements instead use regular Cartesian Bessel power series.
"""

import mpmath as mp
import sympy as s


def zero(v):
    assert s.simplify(s.expand_func(v)) == 0, v


def main():
    r, phi, z = s.symbols("r phi z", real=True)
    m = s.symbols("m", integer=True, nonnegative=True)
    q_symbol = s.Function("Q")(r, z)
    phase = s.exp(s.I * m * phi)
    c, si = s.cos(phi), s.sin(phi)
    rotation = s.Matrix([[c, -si, 0], [si, c, 0], [0, 0, 1]])
    gradient = rotation * (
        phase
        * s.Matrix([s.diff(q_symbol, r), s.I * m * q_symbol / r, s.diff(q_symbol, z)])
    )
    ops = [
        lambda f: c * s.diff(f, r) - si / r * s.diff(f, phi),
        lambda f: si * s.diff(f, r) + c / r * s.diff(f, phi),
        lambda f: s.diff(f, z),
    ]
    for i in range(3):
        zero(gradient[i] - ops[i](phase * q_symbol))
    cylindrical = phase * s.Matrix(
        [
            [
                s.diff(q_symbol, r, 2),
                s.I * m * (s.diff(q_symbol, r) / r - q_symbol / r**2),
                s.diff(q_symbol, r, z),
            ],
            [
                s.I * m * (s.diff(q_symbol, r) / r - q_symbol / r**2),
                s.diff(q_symbol, r) / r - m * m * q_symbol / r**2,
                s.I * m * s.diff(q_symbol, z) / r,
            ],
            [
                s.diff(q_symbol, r, z),
                s.I * m * s.diff(q_symbol, z) / r,
                s.diff(q_symbol, z, 2),
            ],
        ]
    )
    cartesian = rotation * cylindrical * rotation.T
    for i in range(3):
        for j in range(3):
            zero(cartesian[i, j] - ops[j](gradient[i]))
    print(
        "Cylinder arbitrary m Cartesian gradient and all Hessian components EXACT PASS"
    )
    x, y, lam = s.symbols("x y lam", real=True)
    for mm in range(5):
        for sign in [-1, 1]:
            polynomial = sum(
                sign**j
                * (lam / 2) ** (mm + 2 * j)
                * (x + s.I * y) ** mm
                * (x * x + y * y) ** j
                / (s.factorial(j) * s.factorial(mm + j))
                for j in range(3)
            )
            center = {x: 0, y: 0}
            if mm >= 1:
                zero(polynomial.subs(center))
            if mm >= 2:
                zero(s.diff(polynomial, x).subs(center))
                zero(s.diff(polynomial, y).subs(center))
            if mm >= 3:
                for a in [x, y]:
                    for b in [x, y]:
                        zero(s.diff(polynomial, a, b).subs(center))
            rr = s.symbols("rr", positive=True)
            reference = (
                s.besselj(mm, lam * rr) if sign == -1 else s.besseli(mm, lam * rr)
            )
            zero(
                s.series(reference, rr, 0, mm + 6).removeO()
                - polynomial.subs({x: rr, y: 0})
            )
    print("Cylinder regular J/I Cartesian axis series and m selection EXACT PASS")
    a_symbol, b_symbol = s.symbols("A B", nonnegative=True)
    zero(
        s.cosh(a_symbol + b_symbol)
        - s.cosh(a_symbol) * s.cosh(b_symbol)
        - s.sinh(a_symbol) * s.sinh(b_symbol)
    )
    xx, mu, u = s.symbols("xx mu u", positive=True)
    gaussian = s.integrate(s.exp(-xx * u * u / 2) * s.cosh(mu * u), (u, 0, s.oo))
    zero(gaussian - s.sqrt(s.pi / (2 * xx)) * s.exp(mu * mu / (2 * xx)))
    print(
        "K derivative envelope Gaussian integral and nonnegative "
        "cosh product identity EXACT PASS"
    )
    mp.mp.dps = 40
    for mm in range(4):
        for dd in range(4):
            for arg in [mp.mpf(".1"), mp.mpf("1"), mp.mpf("20")]:
                actual = abs(mp.diff(lambda v, mm=mm: mp.besselk(mm, v), arg, dd))
                bound = (
                    mp.exp(-arg)
                    * mp.sqrt(mp.pi / (2 * arg))
                    * mp.exp((mm + dd) ** 2 / (2 * arg))
                )
                assert actual <= bound
    print(
        "K derivative inequality: finite independent mpmath counterexample search PASS"
    )


if __name__ == "__main__":
    main()
