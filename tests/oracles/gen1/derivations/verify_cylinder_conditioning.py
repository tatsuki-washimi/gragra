"""Read-only Cylinder conditioning advisory; no production/oracle imports.

Domains: H=2h>0, a>=0, kappa>=0, q nonnegative integer,
beta=q*pi/H; paired cap/local expression requires |z|<h and r<R.
a=0 is only the radialconstant m=0,q>0 branch; a=beta=0 is rejected.
SymPy identities are exact; finite MP checks are counterexample searches.
"""

import json
import platform
import sys

import mpmath as mp
import sympy as sp


def symbolic_checks():
    h, a = sp.symbols("h a", positive=True)
    z, s = sp.symbols("z s", real=True)
    beta, kappa = sp.symbols("beta kappa", positive=True)
    v = sp.symbols("v", integer=True, nonnegative=True)
    for epsilon, paired in (
        (1, -sp.exp(-a * h) * sp.cosh(a * z)),
        (-1, sp.exp(-a * h) * sp.sinh(a * z)),
    ):
        direct = -(sp.exp(-a * (h + z)) + epsilon * sp.exp(-a * (h - z))) / 2
        assert sp.simplify(direct - paired.rewrite(sp.exp)) == 0
    assert (
        sp.trigsimp(
            sp.expand_trig(sp.cos(v * sp.pi + beta * s)) - (-1) ** v * sp.cos(beta * s)
        )
        == 0
    )
    assert (
        sp.trigsimp(
            sp.expand_trig(sp.cos((v + sp.Rational(1, 2)) * sp.pi + beta * s))
            - (-1) ** (v + 1) * sp.sin(beta * s)
        )
        == 0
    )
    e = sp.exp(-a * h)
    d = 1 - e * sp.cosh(a * z)
    d_stable = -(sp.exp(-a * (h - z)) - 1 + sp.exp(-a * (h + z)) - 1) / 2
    assert sp.simplify(d - d_stable.rewrite(sp.exp)) == 0
    for sign in (1, -1):
        even = sign * sp.cos(beta * z) - e * sp.cosh(a * z)
        trig_minus_one = (
            -2 * sp.sin(beta * z / 2) ** 2
            if sign == 1
            else -2 * sp.cos(beta * z / 2) ** 2
        )
        assert sp.trigsimp(even - (trig_minus_one + d)) == 0
        even_d = -sign * beta * sp.sin(beta * z) - a * e * sp.sinh(a * z)
        even_dd = -sign * beta**2 * sp.cos(beta * z) - a**2 * e * sp.cosh(a * z)
        odd = sign * sp.sin(beta * z) + e * sp.sinh(a * z)
        odd_d = sign * beta * sp.cos(beta * z) + a * e * sp.cosh(a * z)
        odd_dd = -sign * beta**2 * sp.sin(beta * z) + a**2 * e * sp.sinh(a * z)
        for f, first, second, local in (
            (even, even_d, even_dd, sign * sp.cos(beta * z)),
            (odd, odd_d, odd_dd, sign * sp.sin(beta * z)),
        ):
            assert sp.simplify(sp.diff(f, z) - first) == 0
            assert sp.simplify(sp.diff(f, z, 2) - second) == 0
            assert sp.simplify(second - a**2 * f + (a**2 + beta**2) * local) == 0
        assert sp.simplify(even_d.subs(z, 0)) == 0
        assert odd.subs(z, 0) == odd_dd.subs(z, 0) == 0
        sinc_u = sp.sin(beta * z) / (beta * z)
        sinhc = sp.sinh(a * z) / (a * z)
        assert sp.simplify(odd - z * (sign * beta * sinc_u + a * e * sinhc)) == 0
        assert (
            sp.simplify(even_d + z * (sign * beta**2 * sinc_u + a**2 * e * sinhc)) == 0
        )
        assert (
            sp.simplify(odd_dd - z * (-sign * beta**3 * sinc_u + a**3 * e * sinhc)) == 0
        )
        assert sp.limit(even, a, 0, dir="+") == sign * sp.cos(beta * z) - 1
        assert sp.limit(odd, a, 0, dir="+") == sign * sp.sin(beta * z)
    argument = sp.symbols("argument", real=True)
    assert sp.limit(sp.sin(argument) / argument, argument, 0) == 1
    assert sp.limit(sp.sinh(argument) / argument, argument, 0) == 1
    magnitude = sp.symbols("magnitude", positive=True)
    assert (
        sp.simplify(
            sp.exp(-a * h) * sp.sinh(magnitude).rewrite(sp.exp)
            - sp.exp(-a * h + magnitude) * (1 - sp.exp(-2 * magnitude)) / 2
        )
        == 0
    )
    # Product-to-sum radial-free axial integral, before Neumann parity selection.
    cc_primitive = sp.sin((kappa - beta) * s) / (2 * (kappa - beta)) + sp.sin(
        (kappa + beta) * s
    ) / (2 * (kappa + beta))
    ss_primitive = sp.sin((kappa - beta) * s) / (2 * (kappa - beta)) - sp.sin(
        (kappa + beta) * s
    ) / (2 * (kappa + beta))
    assert (
        sp.trigsimp(sp.diff(cc_primitive, s) - sp.cos(beta * s) * sp.cos(kappa * s))
        == 0
    )
    assert (
        sp.trigsimp(sp.diff(ss_primitive, s) - sp.sin(beta * s) * sp.sin(kappa * s))
        == 0
    )
    assert (
        sp.simplify(
            sp.limit(sp.sin((kappa - beta) * h) / (kappa - beta), kappa, beta) - h
        )
        == 0
    )
    for q in range(11):
        b = q * sp.pi / (2 * h)
        if q == 0:
            zero = 2 * h
            resonant = 2 * h
        elif q % 2 == 0:
            zero = 2 * (-1) ** (q // 2) * sp.sin(b * h) / b
            resonant = (-1) ** (q // 2) * h * sp.cos(b * z)
        else:
            zero = sp.Integer(0)
            resonant = (-1) ** ((q + 1) // 2) * h * sp.sin(b * z)
        assert sp.simplify(zero - (2 * h if q == 0 else 0)) == 0
        assert (
            sp.trigsimp(resonant - (2 * h if q == 0 else h * sp.cos(b * (z + h)))) == 0
        )
        if q:
            primitive = cc_primitive if q % 2 == 0 else ss_primitive
            overlap = (primitive.subs(s, h) - primitive.subs(s, -h)).subs(beta, b)
            overlap *= (-1) ** (q // 2 if q % 2 == 0 else (q + 1) // 2)
            rational = (
                2
                * kappa
                * (sp.sin(kappa * h) if q % 2 == 0 else sp.cos(kappa * h))
                / (kappa**2 - b**2)
            )
            assert sp.trigsimp(sp.expand_trig(overlap - rational)) == 0
    even_rational = (
        2 * kappa * sp.sin(kappa * h) * sp.cos(kappa * z) / (kappa**2 - beta**2)
    )
    odd_rational = (
        2 * kappa * sp.cos(kappa * h) * sp.sin(kappa * z) / (kappa**2 - beta**2)
    )
    assert (
        sp.simplify(sp.limit(even_rational / kappa**2, kappa, 0) + 2 * h / beta**2) == 0
    )
    assert (
        sp.simplify(sp.limit(odd_rational / kappa**2, kappa, 0) + 2 * z / beta**2) == 0
    )
    print(
        "Exact CAS cap pair/local derivatives/Poisson/parity/"
        "Fourier/removable limits PASS"
    )


def stable_hyperbolic(a, h, z):
    magnitude = abs(a * z)
    positive = mp.exp(-a * h + magnitude)
    negative = mp.exp(-a * h - magnitude)
    cosh = (positive + negative) / 2
    sinh = mp.sign(z) * positive * (-mp.expm1(-2 * magnitude)) / 2 if a else mp.mpf(0)
    return cosh, sinh


def stable_combination(q, a, h, z):
    beta = q * mp.pi / (2 * h)
    e_cosh, e_sinh = stable_hyperbolic(a, h, z)
    if q % 2 == 0:
        sign = (-1) ** (q // 2)
        d = -(mp.expm1(-a * (h - z)) + mp.expm1(-a * (h + z))) / 2
        trig = -2 * (
            mp.sin(beta * z / 2) ** 2 if sign == 1 else mp.cos(beta * z / 2) ** 2
        )
        return (
            trig + d,
            -sign * beta * mp.sin(beta * z) - a * e_sinh,
            -sign * beta**2 * mp.cos(beta * z) - a * a * e_cosh,
        )
    sign = (-1) ** ((q + 1) // 2)
    return (
        sign * mp.sin(beta * z) + e_sinh,
        sign * beta * mp.cos(beta * z) + a * e_cosh,
        -sign * beta**2 * mp.sin(beta * z) + a * a * e_sinh,
    )


def finite_checks():
    mp.mp.dps = 90
    h = mp.mpf("0.7")
    counts = {"cap_pair_derivatives": 0, "Fourier_integral_derivatives": 0}
    for q in range(7):
        beta = q * mp.pi / (2 * h)
        for ah in ("0", "1e-40", "1e-8", "0.2", "20", "1000"):
            a = mp.mpf(ah) / h
            if not a and q == 0:
                continue
            for fraction in ("0", "1e-30", "-0.03", "0.8"):
                z = mp.mpf(fraction) * h

                def direct(value, beta=beta, a=a, q=q):
                    local = mp.cos(beta * (value + h))
                    caps = (
                        -(
                            mp.exp(-a * (h + value))
                            + (-1) ** q * mp.exp(-a * (h - value))
                        )
                        / 2
                    )
                    return local + caps

                values = stable_combination(q, a, h, z)
                for order, actual in enumerate(values):
                    expected = mp.diff(direct, z, order)
                    assert abs(actual - expected) < mp.mpf("1e-75") * max(
                        1, abs(expected)
                    )
                    counts["cap_pair_derivatives"] += 1
    for q in range(7):
        beta = q * mp.pi / (2 * h)
        for kappa in (
            mp.mpf(0),
            mp.mpf("1e-35"),
            beta,
            beta + mp.mpf("1e-35"),
            mp.mpf("0.4"),
            mp.mpf("3.7"),
        ):
            for fraction in ("0", "1e-20", "-0.13", "0.72"):
                z = mp.mpf(fraction) * h
                sinc_minus, sinc_plus = (
                    mp.sinc((kappa - beta) * h),
                    mp.sinc((kappa + beta) * h),
                )
                factor = (
                    (-1) ** (q // 2) * h * (sinc_minus + sinc_plus)
                    if q % 2 == 0
                    else (-1) ** ((q + 1) // 2) * h * (sinc_minus - sinc_plus)
                )

                def spectral(value, factor=factor, kappa=kappa, q=q):
                    return factor * (
                        mp.cos(kappa * value) if q % 2 == 0 else mp.sin(kappa * value)
                    )

                for order in range(3):

                    def integrand(s, order=order, beta=beta, kappa=kappa, z=z):
                        return (
                            mp.cos(beta * (s + h))
                            * kappa**order
                            * mp.cos(kappa * (z - s) + order * mp.pi / 2)
                        )

                    direct = mp.quad(integrand, [-h, 0, h])
                    expected = mp.diff(spectral, z, order)
                    assert abs(direct - expected) < mp.mpf("1e-75") * max(
                        1, abs(direct)
                    )
                    counts["Fourier_integral_derivatives"] += 1
    print(json.dumps({"finite_mp90_counterexample_checks": counts, "status": "PASS"}))


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "sympy": sp.__version__,
                "mpmath": mp.__version__,
            }
        )
    )
    symbolic_checks()
    finite_checks()
