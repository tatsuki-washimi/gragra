"""Independent special-function checks for the analytic cylinder path."""

import numpy as np
import pytest

import gragra.analytic.cavity_cylinder as cylinder


@pytest.mark.parametrize(
    "kc,p,a,b",
    [
        (1.0, 1.0, 1.0, 1.0),
        (0.2, 0.1, 1.0, 1.0),
        (1e-8, 1e-10, 1.0, -0.3),
        (0.99, 1.0, 1.0, -0.5),
    ],
)
def test_cel_matches_high_precision_definition(kc, p, a, b):
    mp = pytest.importorskip("mpmath")
    with mp.workdps(60):
        expected = (
            a * mp.elliprf(0, mp.mpf(kc) ** 2, 1)
            + (b - p * a) * mp.elliprj(0, mp.mpf(kc) ** 2, 1, p) / 3
        )
    assert hasattr(cylinder, "_cel"), "analytic elliptic evaluator is missing"
    np.testing.assert_allclose(cylinder._cel(kc, p, a, b), float(expected), rtol=3e-14)
