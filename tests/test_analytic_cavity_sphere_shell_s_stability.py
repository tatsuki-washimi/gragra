"""High-precision regression tests for the finite spherical shell S-wave total.

Guards the fix for a catastrophic-cancellation bug: computing the finite-shell
S-wave total as the naive difference of two independently-rounded
``j1(x)/x`` evaluations loses essentially all significant digits once both
``ka`` and ``kR`` are small (small wavenumber and/or a thin shell, R ~ a),
because both terms round to ~1/3 in float64 before the subtraction ever
happens. ``gragra.analytic._bessel._j1_over_x_diff`` fixes this by evaluating
the difference directly as a series in ``ka^2 - kR^2`` (see its docstring).

This file is deliberately separate from
``tests/test_analytic_cavity_sphere_shell_s.py`` (which is registered under
the ``lebedev`` CI row): mixing an ``mpmath``-gated test into a file that
also runs, under ``GRAGRA_REQUIRE_EXTRAS=1``, in a job that installs only the
``lebedev`` extra (scipy, not mpmath) would turn the missing-dependency skip
into a hard CI failure in that job. This file is registered only under the
``validation`` extra row instead, which installs both scipy and mpmath.
"""

import numpy as np
import pytest

pytest.importorskip("mpmath")

from gragra.analytic import spherical_shell_s_acceleration_components_g_free

A_INNER = 15.0
R_OUTER = 120.0
RHO = 2500.0
DISP = 1.5e-6


def _orthogonal_pair():
    return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])


def _reference_total_over_g(
    a: float, R: float, k: float, rho: float, xi: float
) -> float:
    """Independent high-precision reference for total/G along e_hat_S.

    Uses mpmath at 120 decimal digits and the spherical Bessel function's
    trigonometric closed form j1(x) = sin(x)/x**2 - cos(x)/x directly (not
    gragra's own ``_j1_over_x``/``_j1_over_x_diff`` helpers, and not
    mpmath's built-in ``besselj``, which is the *cylindrical* Bessel
    function J1 and has a different small-x limit -- j1(x)/x -> 1/3 for the
    spherical function used here, vs J1(x)/x -> 1/2 for the cylindrical
    one). At 120 digits of precision, evaluating the trig formula directly
    carries far more headroom than float64's ~16 digits, so the
    cancellation that corrupts the float64 path here is not a concern.
    """
    import mpmath as mp

    mp.mp.dps = 120

    def f(x):
        x = mp.mpf(x)
        if x == 0:
            return mp.mpf(1) / 3
        return (mp.sin(x) / x**2 - mp.cos(x) / x) / x

    ka = mp.mpf(k) * mp.mpf(a)
    kR = mp.mpf(k) * mp.mpf(R)
    total_over_g = -4 * mp.pi * mp.mpf(rho) * (f(ka) - f(kR)) * mp.mpf(xi)
    return float(total_over_g)


@pytest.mark.parametrize(
    "a,R,k",
    [
        (10.0, 40.0, 1e-5),
        (10.0, 40.0, 1e-8),
        (10.0, 40.0, 1e-10),
        (10.0, 10.01, 1e-7),
        (1.0, 1.0001, 1e-3),
        (1.0, 1.0001, 0.05),
    ],
    ids=[
        "small-k-a10-R40",
        "smaller-k-a10-R40",
        "tiny-k-a10-R40",
        "thin-shell-tiny-k",
        "thin-shell-small-k",
        "thin-shell-near-threshold",
    ],
)
def test_shell_s_total_matches_high_precision_reference_at_small_k_and_thin_shell(
    a, R, k
):
    """The production total must track the mpmath reference even where the
    naive f(ka) - f(kR) difference would lose all significant digits.

    Before the ``_j1_over_x_diff`` fix, some of these cases (e.g. k=1e-10)
    produced an exact-zero total against a nonzero reference (100% relative
    error), and the thin-shell case had ~66% relative error.
    """
    n_hat, e_hat = _orthogonal_pair()
    rho = 2000.0
    xi = 1e-5

    comps = spherical_shell_s_acceleration_components_g_free(
        a, R, k, rho, xi, n_hat, e_hat
    )
    got = comps.total[0]

    ref_scalar = _reference_total_over_g(a, R, k, rho, xi)
    ref_vec = ref_scalar * e_hat

    np.testing.assert_allclose(got, ref_vec, rtol=1e-9, atol=0.0)


@pytest.mark.parametrize(
    "a,R,k",
    [
        (1.0, 1.0001, 0.1),
        (1.0, 1.000001, 0.1),
        (10.0, 10.00000001, 0.02),
        (1.0, 1.000000001, 0.09999999995),
        (100.0, 100.001, 5.0),
        (1000.0, 1000.001, 50.0),
        (1.0, 1.00000000001, 0.05),
    ],
    ids=[
        "thin-shell-at-threshold",
        "thin-shell-closer-at-threshold",
        "thin-shell-moderate-k",
        "thin-shell-straddling-threshold",
        "thin-shell-large-x",
        "thin-shell-huge-x",
        "thin-shell-extreme-R-close-to-a",
    ],
)
def test_shell_s_total_matches_high_precision_reference_for_thin_shells_outside_small_x(
    a, R, k
):
    """A thin shell (R close to a) is not the same condition as "ka and kR
    both small": ka=0.1, kR=0.10001 are both >= the small-argument
    threshold, yet naively differencing `_j1_over_x(ka) - _j1_over_x(kR)`
    still loses most significant digits, because `_j1_over_x`'s own
    direct-evaluation branch internally cancels O(1/x^2) terms to produce
    an O(1) result (~1e-13 relative error), which is then amplified once
    the desired difference is itself much smaller than either term. Before
    the sum-to-product "close" branch was added to `_j1_over_x_diff`, these
    cases had relative errors from ~6e-8 up to ~4.7e-3.

    The "thin-shell-extreme-R-close-to-a" case additionally exercises
    `_j1_over_x_diff`'s `x_minus_y` parameter (passed by
    `spherical_shell_s_acceleration_components_g_free` as
    `-(k_arr*(R-a))`, a single subtraction done before multiplying by k):
    without it, deriving `ka - kR` from the separately-rounded `ka = k*a`
    and `kR = k*R` loses relative precision once R is extremely close to
    a (this case had ~5.6e-6 relative error before `x_minus_y` was
    threaded through).

    This 1e-9 tolerance matches `_j1_over_x_diff`'s documented worst-case
    bound (a few times 1e-10 with `x_minus_y`) with a comfortable margin,
    not a loosened bound chosen to make these specific cases pass.
    """
    n_hat, e_hat = _orthogonal_pair()
    rho = 2000.0
    xi = 1e-5

    comps = spherical_shell_s_acceleration_components_g_free(
        a, R, k, rho, xi, n_hat, e_hat
    )
    got = comps.total[0]

    ref_vec = _reference_total_over_g(a, R, k, rho, xi) * e_hat

    np.testing.assert_allclose(got, ref_vec, rtol=1e-9, atol=0.0)


def test_shell_s_total_matches_high_precision_reference_at_ordinary_k():
    """Sanity check: the fix must not regress the already-accurate ordinary regime."""
    n_hat, e_hat = _orthogonal_pair()
    a, R, k = 10.0, 40.0, 0.02
    rho, xi = 2000.0, 1e-5

    comps = spherical_shell_s_acceleration_components_g_free(
        a, R, k, rho, xi, n_hat, e_hat
    )
    got = comps.total[0]
    ref_vec = _reference_total_over_g(a, R, k, rho, xi) * e_hat

    np.testing.assert_allclose(got, ref_vec, rtol=1e-9, atol=0.0)
