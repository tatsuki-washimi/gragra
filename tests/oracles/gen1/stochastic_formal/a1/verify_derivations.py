"""CAS identities supporting acoustic oracle methods, not numerical acceptance."""

import sympy as s


def verify():
    q, b, k, t, z = s.symbols("q b k t z", positive=True, real=True)
    # A1 air-side split Green integral; q,b>0, z>=0 upper-side limit.
    green = (2 * q * s.cos(b * z) - q * s.exp(-q * z)) / (q * q + b * b)
    gradient = s.diff(green, z)
    assert s.simplify(s.limit(green, z, 0, dir="+") - q / (q * q + b * b)) == 0
    assert s.simplify(s.limit(gradient, z, 0, dir="+") / q - q / (q * q + b * b)) == 0
    # A1 theta->mu transformation, normalized hemisphere measure.
    theta = s.symbols("theta", real=True)
    assert s.integrate(s.sin(theta) ** 3, (theta, 0, s.pi / 2)) == s.Rational(2, 3)
    u = s.symbols("u", real=True)
    assert s.integrate(1 - u * u, (u, 0, 1)) == s.Rational(2, 3)
    # A3 integration-by-parts identity and regular origin, t>0.
    antiderivative = 1 - s.sin(t) / t
    radial = (s.sin(t) - t * s.cos(t)) / t**2
    assert s.simplify(s.diff(antiderivative, t) - radial) == 0
    assert s.limit(antiderivative, t, 0, dir="+") == 0
    assert s.limit(radial, t, 0, dir="+") == 0
    x = s.symbols("x", positive=True, real=True)
    assert (
        s.simplify(s.integrate(1 - s.cos(x * u), (u, 0, 1)) - (1 - s.sin(x) / x)) == 0
    )
    return {
        "assumptions": "q,b,k,t>0; air height z->0+; x=kR>=0",
        "exact_identities": 7,
        "status": "CAS_IDENTITIES_PASS_NOT_SCIENCE_ACCEPTANCE",
    }


if __name__ == "__main__":
    print(verify())
