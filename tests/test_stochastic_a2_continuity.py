"""Continuity, transition-boundary, and dense-domain tests for the A2 depth factor."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mpmath
import numpy as np
import pytest

from gragra.analytic.stochastic_acoustic import (
    _A_COEFFS,
    _S_COEFFS,
    _isotropic_depth_factor_large,
    _isotropic_depth_factor_small,
    isotropic_acoustic_depth_factor,
)


def _mpmath_reference(x: float) -> float:
    """Independent 60-digit precision reference integral via mpmath.quad."""
    if x == 0.0:
        return 1.0 / 3.0
    with mpmath.workdps(60):
        val = mpmath.quad(
            lambda th: (
                0.5
                * (mpmath.sin(th) ** 3)
                * mpmath.exp(-mpmath.mpf(x) * mpmath.sin(th))
            ),
            [0.0, mpmath.pi / 2.0],
        )
    return float(val)


def test_import_does_not_change_global_mpmath_precision() -> None:
    code = (
        "import mpmath, runpy, sys\n"
        "before = mpmath.mp.dps\n"
        "runpy.run_path(sys.argv[1])\n"
        "assert mpmath.mp.dps == before, (before, mpmath.mp.dps)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve())],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_mpmath_reference_uses_scoped_sixty_digit_precision(monkeypatch) -> None:
    observed = {}

    def inspect_precision(*_args, **_kwargs):
        observed["dps"] = mpmath.mp.dps
        return mpmath.mpf("1.25")

    monkeypatch.setattr(mpmath, "quad", inspect_precision)
    with mpmath.workdps(27):
        result = _mpmath_reference(0.5)
        assert mpmath.mp.dps == 27

    assert observed["dps"] == 60
    assert result == 1.25


def test_a2_transition_at_0_75_branch_agreement() -> None:
    """Verify sub-ULP continuity across Taylor-to-exponential transition at x = 0.75."""
    ref_075 = _mpmath_reference(0.75)
    val_taylor = _isotropic_depth_factor_small(0.75)
    val_exp = float(np.sum(_A_COEFFS * np.exp(-_S_COEFFS * 0.75)))
    val_pub = float(isotropic_acoustic_depth_factor(0.75))

    # Public dispatch selects the exponential branch at 0.75
    assert val_pub == val_exp

    # Relative difference between the Taylor and exponential branches is
    # below 5e-16 (within 3 ULPs)
    rel_diff_branches = abs(val_taylor - val_exp) / ref_075
    assert rel_diff_branches < 5.0e-16

    # Both branches agree with 60-digit mpmath reference to < 5e-16
    assert abs(val_taylor - ref_075) / ref_075 < 5.0e-16
    assert abs(val_exp - ref_075) / ref_075 < 5.0e-16


def test_a2_transition_at_100_branch_agreement() -> None:
    """Verify continuity across exponential-to-Watson transition at x = 100."""
    ref_100 = _mpmath_reference(100.0)
    val_exp = float(np.sum(_A_COEFFS * np.exp(-_S_COEFFS * 100.0)))
    val_watson = _isotropic_depth_factor_large(100.0)
    val_pub = float(isotropic_acoustic_depth_factor(100.0))

    # Public dispatch selects the exponential branch at 100.0 (x <= 100)
    assert val_pub == val_exp

    # Relative difference between Exponential and Watson branches is < 2.3e-14
    rel_diff_branches = abs(val_exp - val_watson) / ref_100
    assert rel_diff_branches < 2.3e-14

    # Both branches agree with 60-digit mpmath reference
    assert abs(val_exp - ref_100) / ref_100 < 2.4e-14
    assert abs(val_watson - ref_100) / ref_100 < 2.0e-15


@pytest.mark.parametrize(
    "x",
    np.geomspace(1.0e-8, 1000.0, num=40),
)
def test_a2_dense_domain_against_independent_quadrature(x: float) -> None:
    """Verify relative error < 2.5e-14 on 40 log-spaced points against mpmath."""
    actual = float(isotropic_acoustic_depth_factor(x))
    expected = _mpmath_reference(x)
    np.testing.assert_allclose(actual, expected, rtol=2.5e-14, atol=1.0e-25)


def test_a2_monotonicity_and_positivity_on_dense_grid() -> None:
    """Verify I_iso(x) is strictly positive and strictly decreasing for x >= 0."""
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 50),
                np.linspace(1.0, 20.0, 50),
                np.linspace(20.0, 500.0, 50),
            ]
        )
    )
    values = np.asarray([float(isotropic_acoustic_depth_factor(x)) for x in grid])
    assert np.all(values > 0.0)
    assert np.all(np.diff(values) < 0.0)


def test_a2_exact_origin_and_asymptotic_limit() -> None:
    """Verify exact origin value and Watson leading behavior at large x."""
    assert isotropic_acoustic_depth_factor(0.0) == 1.0 / 3.0

    large_x = 1.0e4
    val_large = float(isotropic_acoustic_depth_factor(large_x))
    expected_lead = 3.0 / (large_x**4) + 30.0 / (large_x**6)
    np.testing.assert_allclose(val_large, expected_lead, rtol=1.0e-12, atol=0.0)


def _watson_coefficient(n: int) -> int:
    """Exact c_n = (1/2)_n (2n+3)! / (2 n!) as an integer (Issue #180)."""
    from fractions import Fraction
    from math import factorial

    poch = Fraction(1)
    for i in range(n):
        poch *= Fraction(1, 2) + i
    value = poch * factorial(2 * n + 3) / (2 * factorial(n))
    assert value.denominator == 1
    return int(value)


def test_a2_watson_coefficients_match_closed_form() -> None:
    """All 8 Watson coefficients equal the exact closed form (no rounding)."""
    from gragra.analytic.stochastic_acoustic import _WATSON_COEFFS

    assert len(_WATSON_COEFFS) == 8
    exact = [_watson_coefficient(n) for n in range(8)]
    assert [int(c) for c in _WATSON_COEFFS] == exact
    assert all(float(int(c)) == c for c in _WATSON_COEFFS)
    assert _WATSON_COEFFS[5:] == (766215450.0, 147496474125.0, 37253395179000.0)


@pytest.mark.parametrize(
    "x",
    [0.75, 0.7500000000000001, 1.0, 5.0, 30.0, 99.99999999999999, 100.0],
)
def test_a2_taylor_and_exponential_branches_tight_against_quadrature(x: float) -> None:
    """Gauss-Legendre exponential sum is accurate to ~1e-15 on [0.75, 100].

    Measured max relative error on 60 log-spaced points in [0.75, 100] is
    2.2e-16 (previously 2.3e-14); the tolerance here is looser than measured
    and applies only to the tested inputs.
    """
    expected = _mpmath_reference(x)
    actual = float(np.sum(_A_COEFFS * np.exp(-_S_COEFFS * x)))
    assert abs(actual - expected) / expected < 2.0e-15


@pytest.mark.parametrize(
    "x",
    [np.nextafter(100.0, 200.0), 100.0001, 101.0, 150.0, 300.0, 1000.0],
)
def test_a2_watson_branch_against_quadrature(x: float) -> None:
    """Public value for x > 100 agrees with 60-digit quadrature.

    Measured max relative error on 40 log-spaced points in (100, 1000] plus
    {100.0000001, 101, 150} is 4.1e-16 (previously 1.8e-15); tolerance is looser.
    """
    expected = _mpmath_reference(float(x))
    actual = float(isotropic_acoustic_depth_factor(float(x)))
    assert abs(actual - expected) / expected < 2.0e-15


@pytest.mark.parametrize("x", [0.75, 100.0])
def test_a2_no_discontinuity_across_dispatch_boundaries(x: float) -> None:
    """Values just below, at, and just above each dispatch boundary are continuous."""
    lo = float(isotropic_acoustic_depth_factor(np.nextafter(x, 0.0)))
    mid = float(isotropic_acoustic_depth_factor(x))
    hi = float(isotropic_acoustic_depth_factor(np.nextafter(x, 1.0e9)))
    # One-ULP steps in x are below the branch rounding noise (~1e-16), so only
    # closeness is asserted, not strict monotonicity.
    assert abs(lo - mid) / mid < 1.0e-15
    assert abs(hi - mid) / mid < 1.0e-15
