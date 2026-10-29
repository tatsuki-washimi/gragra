"""Independent high-precision definitions of the cylinder logarithmic function."""

import numpy as np
import pytest


@pytest.mark.parametrize(
    "radius,r,u",
    [
        (2.0, 0.3, 0.4),
        (2.0, 3.1, -0.8),
        (2.0, 2.0, 0.0),
        (2.0, 2.0, 1e-10),
        (2.0, 2.0001, 0.003),
        (2.0, 1.999, 0.00001),
        (2.0, 2.0, 1.0),
        (2.0, 1.2, 0.799999999),
        (2.0, 1.2, 0.8),
        (2.0, 2.0, 100.0),
        (2.0, 1.200000001, 0.79999999),
        (2.0, 1.200000001, 0.80000001),
        (2.0, 1.199999999, 0.8),
        (2.0, 2.0, 0.999999999),
        (2.0, 2.0, 1.000000001),
    ],
)
def test_logarithmic_function_matches_definition(radius, r, u):
    mp = pytest.importorskip("mpmath")
    from gragra.analytic.cylinder_potential import _cylinder_logarithmic_function

    with mp.workdps(60):
        a, b, z = map(mp.mpf, (radius, r, u))
        if u == 0:
            expected = 0
        else:
            expected = 2 * mp.quad(
                lambda phi: mp.asinh(
                    z / mp.sqrt((a - b) ** 2 + 4 * a * b * mp.sin(phi / 2) ** 2)
                ),
                [0, mp.mpf("1e-12"), mp.mpf("1e-6"), mp.mpf("0.01"), mp.pi],
            )
    np.testing.assert_allclose(
        _cylinder_logarithmic_function(radius, r, u),
        float(expected),
        rtol=2e-14,
        atol=2e-18,
    )


def test_documented_uniform_truncation_majorants():
    """Check the numeric bounds of the documented exact-arithmetic inequalities."""
    from math import factorial, log, prod

    assert (15 / 16) ** 1025 / (1 - 15 / 16) < 3e-28
    assert 1024**2 * (16 / 17) ** 1023 < 1.3e-21
    assert (16 / 17) ** 1024 / (1 - 16 / 17) < 2e-26
    w = 2 / 17
    assert w**32 * (log(4) - log(w) / 2 + 2 * log(2)) / (1 - w) < 9e-30
    # Inner binomial/log-weighted tail after j=63, integrated on x<=1/4.
    # Polynomial absolute sum <=(d²+x²)^n<=1/8^n; alpha_n<=1.
    ratio = 3 / 32
    majorant = 0
    for n in range(32):
        coefficient = prod(n + 0.5 + j for j in range(64)) / factorial(64)
        harmonic = sum(1 / (n + 0.5 + j) for j in range(64))
        logarithmic_bound = log(4) + 2 * log(2) + 1 - log(0.25)
        logarithmic_bound += (harmonic + 1 / ((n + 64.5) * (1 - ratio))) / 2
        majorant += (
            (1 / 8) ** n
            * coefficient
            * (1 / 16) ** 64
            * logarithmic_bound
            / (1 - ratio)
        )
    assert majorant < 1e-70
    assert 2**-81 / (81 * (1 - 0.5)) < 1.1e-26
