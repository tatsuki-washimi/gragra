"""Independent identities and finite checks; no production or oracle imports.

Domains: R,H>0, m,q nonnegative integers; r<R, |z|<H/2;
positive Neumann root alpha>m or the separate m=alpha=0,q>0 branch.
Numeric checks search for counterexamples, not special-function error bounds.
"""

import json
import platform

import mpmath as mp
import sympy as s


def exact_checks():
    a, h, z, beta = s.symbols("a h z beta", real=True)
    c, d = s.symbols("c d", real=True)
    bc = s.exp(-a * h) * s.cosh(a * z)
    bs = s.exp(-a * h) * s.sinh(a * z)
    even = [
        c * s.cos(beta * z) - bc,
        -c * beta * s.sin(beta * z) - a * bs,
        -c * beta**2 * s.cos(beta * z) - a**2 * bc,
    ]
    odd = [
        d * s.sin(beta * z) + bs,
        d * beta * s.cos(beta * z) + a * bc,
        -d * beta**2 * s.sin(beta * z) + a**2 * bs,
    ]
    ae = [
        h * bc - z * bs,
        (a * h - 1) * bs - a * z * bc,
        a * (a * h - 2) * bc - a**2 * z * bs,
    ]
    ao = [
        z * bc - h * bs,
        (1 - a * h) * bc + a * z * bs,
        a * (2 - a * h) * bs + a**2 * z * bc,
    ]
    count = 0
    for f, fa in zip(even + odd, ae + ao, strict=True):
        assert s.simplify(s.diff(f, a) - fa) == 0
        count += 1
    for fs in (even, odd):
        for j in (1, 2):
            assert s.simplify(s.diff(fs[0], z, j) - fs[j]) == 0
            count += 1
    assert even[1].subs(z, 0) == ae[1].subs(z, 0) == 0
    assert odd[0].subs(z, 0) == ao[0].subs(z, 0) == 0
    assert odd[2].subs(z, 0) == ao[2].subs(z, 0) == 0
    count += 3
    # Keep the verified mathematical symbol names without changing the proof AST.
    alpha, R, m = s.symbols("alpha R m", positive=True)  # noqa: N806
    J = s.Function("J")(alpha)  # noqa: N806
    j1 = s.diff(J, alpha)
    n2 = J**2 * (1 - m**2 / alpha**2)
    lam_n = -j1 / J - m**2 / (alpha * (alpha**2 - m**2))
    assert s.simplify(-s.diff(n2, alpha) / (2 * n2) - lam_n) == 0
    full_n2 = n2 + j1**2
    assert (
        s.simplify((-s.diff(full_n2, alpha) / (2 * full_n2) - lam_n).subs(j1, 0)) == 0
    )
    k2 = alpha**2 / R**2 + beta**2
    assert s.simplify(s.diff(s.log(1 / k2), alpha) + 2 * alpha / (R**2 * k2)) == 0
    count += 3
    # Polar D+ and D- acting on J_m(ar)e^(im phi); the two Bessel
    # recurrences J_{m-1}=mJ/u+J', J_{m+1}=mJ/u-J' are explicit premises.
    r, phi, order, j, jp = s.symbols("r phi order j jp", real=True)
    u = a * r
    dp = s.exp(s.I * (order + 1) * phi) * (a * jp - order * j / r)
    dm = s.exp(s.I * (order - 1) * phi) * (a * jp + order * j / r)
    assert (
        s.simplify(dp + a * (order * j / u - jp) * s.exp(s.I * (order + 1) * phi)) == 0
    )
    assert (
        s.simplify(dm - a * (order * j / u + jp) * s.exp(s.I * (order - 1) * phi)) == 0
    )
    count += 2

    # Ladder algebra then gives all Cartesian derivatives, without 1/r.
    def shift(poly, sign):
        out = {}
        for n, v in poly.items():
            out[n + sign] = out.get(n + sign, 0) + (-a if sign == 1 else a) * v
        return out

    def combine(p, q, cp, cq):
        return {
            n: s.simplify(cp * p.get(n, 0) + cq * q.get(n, 0)) for n in set(p) | set(q)
        }

    def dx(p):
        return combine(shift(p, -1), shift(p, 1), s.Rational(1, 2), s.Rational(1, 2))

    def dy(p):
        return combine(shift(p, -1), shift(p, 1), s.I / 2, -s.I / 2)

    expected = [
        {-1: a / 2, 1: -a / 2},
        {-1: s.I * a / 2, 1: s.I * a / 2},
        {-2: a * a / 4, 0: -a * a / 2, 2: a * a / 4},
        {-2: s.I * a * a / 4, 2: -s.I * a * a / 4},
        {-2: -a * a / 4, 0: -a * a / 2, 2: -a * a / 4},
    ]
    actual = [dx({0: 1}), dy({0: 1}), dx(dx({0: 1})), dx(dy({0: 1})), dy(dy({0: 1}))]
    for p, q in zip(actual, expected, strict=True):
        assert all(s.simplify(p.get(n, 0) - q.get(n, 0)) == 0 for n in set(p) | set(q))
        count += 1
    assert s.simplify(actual[2][0] + actual[4][0] + a * a) == 0
    count += 1
    return count


TABLE = {
    (0, 0): [(0, 1)],
    (1, 0): [(-1, mp.mpf(".5")), (1, -mp.mpf(".5"))],
    (0, 1): [(-1, mp.j / 2), (1, mp.j / 2)],
    (2, 0): [(-2, mp.mpf(".25")), (0, -mp.mpf(".5")), (2, mp.mpf(".25"))],
    (1, 1): [(-2, mp.j / 4), (2, -mp.j / 4)],
    (0, 2): [(-2, -mp.mpf(".25")), (0, -mp.mpf(".5")), (2, -mp.mpf(".25"))],
}


def harmonic(m, a, x, y, phase):
    rr = x * x + y * y
    # Entire Cartesian series, independent of the neighboring-order formulas.
    radial = mp.hyper([], [m + 1], -a * a * rr / 4) * (a / 2) ** m / mp.factorial(m)
    value = (x + mp.j * y) ** m * radial
    return mp.re(value) if phase == "cos" else mp.im(value)


def series_derivative(m, a, x, y, phase, gamma):
    # Differentiate the independent entire Cartesian 0F1 series analytically.
    # Ordinary finite differences cannot resolve a 1e-200 displacement at 80 dps.
    w = x + mp.j * y
    t = -a * a * (x * x + y * y) / 4
    hs = [mp.hyper([], [m + 1 + k], t) for k in range(3)]
    hx = -a * a * x * hs[1] / (2 * (m + 1))
    hy = -a * a * y * hs[1] / (2 * (m + 1))
    hxx = -a * a * hs[1] / (2 * (m + 1)) + a**4 * x * x * hs[2] / (
        4 * (m + 1) * (m + 2)
    )
    hxy = a**4 * x * y * hs[2] / (4 * (m + 1) * (m + 2))
    hyy = -a * a * hs[1] / (2 * (m + 1)) + a**4 * y * y * hs[2] / (
        4 * (m + 1) * (m + 2)
    )
    wp = m * w ** (m - 1) if m else 0
    wpp = m * (m - 1) * w ** (m - 2) if m >= 2 else 0
    values = {
        (0, 0): w**m * hs[0],
        (1, 0): wp * hs[0] + w**m * hx,
        (0, 1): mp.j * wp * hs[0] + w**m * hy,
        (2, 0): wpp * hs[0] + 2 * wp * hx + w**m * hxx,
        (1, 1): mp.j * wpp * hs[0] + wp * hy + mp.j * wp * hx + w**m * hxy,
        (0, 2): -wpp * hs[0] + 2 * mp.j * wp * hy + w**m * hyy,
    }
    value = (a / 2) ** m / mp.factorial(m) * values[gamma]
    return mp.re(value) if phase == "cos" else mp.im(value)


def ladder(m, a, x, y, phase, gamma):
    r = mp.hypot(x, y)
    phi = mp.atan2(y, x) if r else mp.mpf(0)
    degree = sum(gamma)

    def project(v):
        return mp.re(v) if phase == "cos" else mp.im(v)

    b = ba = mp.mpf(0)
    for delta, coefficient in TABLE[gamma]:
        n = m + delta
        p = project(coefficient * mp.exp(mp.j * n * phi))
        jn = mp.besselj(n, a * r)
        jprime = (mp.besselj(n - 1, a * r) - mp.besselj(n + 1, a * r)) / 2
        b += a**degree * p * jn
        ba += p * (
            (degree * a ** (degree - 1) * jn if degree else 0) + a**degree * r * jprime
        )
    return b, ba


def numeric_checks():
    mp.mp.dps = 80
    rows = []
    points = [
        ("generic", mp.mpf(".31"), mp.mpf("-.19")),
        ("axis", mp.mpf(0), mp.mpf(0)),
        ("sin_plane", mp.mpf(".31"), mp.mpf(0)),
        ("tiny_y", mp.mpf(".31"), mp.mpf("1e-200")),
    ]
    max_residual = mp.mpf(0)
    tiny_nonzero_max_relative = mp.mpf(0)
    for m in (0, 1, 2, 4):
        a = mp.mpf("3.2")
        for phase in ("cos", "sin"):
            if m == 0 and phase == "sin":
                continue
            for name, x, y in points:
                for gamma in TABLE:
                    b, ba = ladder(m, a, x, y, phase, gamma)
                    ref = series_derivative(m, a, x, y, phase, gamma)
                    # mp.diff evaluates this callback synchronously in this iteration.
                    refa = mp.diff(
                        lambda aa: series_derivative(m, aa, x, y, phase, gamma),  # noqa: B023
                        a,
                    )
                    error = max(abs(b - ref), abs(ba - refa))
                    max_residual = max(max_residual, error)
                    assert error < mp.mpf("1e-70")
                    if name == "tiny_y":
                        for got, expected in ((b, ref), (ba, refa)):
                            if expected:
                                relative = abs(got - expected) / abs(expected)
                                tiny_nonzero_max_relative = max(
                                    tiny_nonzero_max_relative, relative
                                )
                                assert relative < mp.mpf("1e-65")
                    rows.append(
                        {
                            "m": m,
                            "phase": phase,
                            "point": name,
                            "gamma": list(gamma),
                            "residual": str(error),
                        }
                    )
    # Independently index the first J0 derivative root by a bracket.
    alpha = mp.findroot(lambda t: mp.besselj(1, t), (mp.mpf("3.5"), mp.mpf("4")))
    assert abs(alpha - mp.besseljzero(1, 1)) < mp.mpf("1e-75")
    r = mp.mpf(".627612237576135")
    h = mp.mpf(".05")

    def y(aa):
        N = 1 / abs(mp.besselj(0, aa))  # noqa: N806
        return 4 * mp.pi * N / aa**2 * mp.besselj(0, aa * r) * (-mp.expm1(-aa * h))

    N = 1 / abs(mp.besselj(0, alpha))  # noqa: N806
    j = mp.besselj(0, alpha * r)
    ja = -r * mp.besselj(1, alpha * r)
    f = -mp.expm1(-alpha * h)
    fa = h * mp.exp(-alpha * h)
    lam = mp.besselj(1, alpha) / mp.besselj(0, alpha) - 2 / alpha
    explicit = 4 * mp.pi * N / alpha**2 * (ja * f + j * fa + lam * j * f)
    direct = mp.diff(y, alpha)
    assert abs(explicit - direct) < mp.mpf("1e-70")
    return {
        "finite_counterexample_checks": len(rows),
        "max_abs_residual": str(max_residual),
        "tiny_y_nonzero_max_relative_residual": str(tiny_nonzero_max_relative),
        "rows": rows,
        "radial_zero_probe": {
            "independent_root": str(alpha),
            "radius": str(r),
            "paired_value_without_G_amplitude": str(y(alpha)),
            "absolute_root_derivative": str(explicit),
            "derivative_check_residual": str(abs(explicit - direct)),
            "delta_alpha_example": "2e-15",
            "linear_absolute_sensitivity_example": str(abs(explicit) * mp.mpf("2e-15")),
        },
    }


if __name__ == "__main__":
    result = {
        "assumptions": __doc__,
        "python": platform.python_version(),
        "sympy": s.__version__,
        "mpmath": mp.__version__,
        "exact_checks": exact_checks(),
        "numeric_checks": numeric_checks(),
        "claim": (
            "Exact identities plus finite counterexample searches; "
            "no rigorous floating-point envelope."
        ),
    }
    print(json.dumps(result, indent=2))
