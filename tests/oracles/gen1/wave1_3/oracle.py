"""Independent Newton/Fourier-Green radial-quadrature oracle.

This module integrates the radial Green kernel directly. It never calls the
closed-form candidate or copies its boundary coefficients. For kz != 0, the
starting Newton potential is -G int rho/|x-x'| d^3x'; analytic z integration
and the angular Fourier coefficient reduce it to a split radial quadrature.
"""

# ruff: noqa: N803, N806

from __future__ import annotations

import mpmath as mp
from gravity import G_SI, density_sign


def _beta(m: int, A: mp.mpc) -> mp.mpc:
    return 4 * mp.pi * mp.mpf(G_SI) * density_sign(m) * A


def _zero_density(m: int, kappa: mp.mpf) -> bool:
    return kappa == 0 and abs(m) > 0


def q_nonzero_oracle_radial(
    m: int,
    kappa: mp.mpf,
    kz: mp.mpf,
    R: mp.mpf,
    A: mp.mpc,
    r: mp.mpf,
    side: str | None = None,
) -> tuple[mp.mpc, mp.mpc, mp.mpc]:
    """Potential and its first two radial derivatives from quadrature."""
    if kz == 0:
        raise ValueError("q_nonzero_oracle_radial requires kz != 0")
    n = abs(m)
    if _zero_density(m, kappa):
        return mp.mpc(0), mp.mpc(0), mp.mpc(0)
    q = abs(kz)
    beta = _beta(m, A)
    if r == 0:
        if n == 0:
            H0 = mp.quad(
                lambda s: s * mp.besselj(0, kappa * s) * mp.besselk(0, q * s), [0, R]
            )
            f0 = -beta * H0
            return f0, mp.mpc(0), (beta + q * q * f0) / 2
        if n == 1:
            H0 = mp.quad(
                lambda s: s * mp.besselj(1, kappa * s) * mp.besselk(1, q * s), [0, R]
            )
            return mp.mpc(0), -beta * q * H0 / 2, mp.mpc(0)
        if n == 2:
            H0 = mp.quad(
                lambda s: s * mp.besselj(2, kappa * s) * mp.besselk(2, q * s), [0, R]
            )
            c2 = -beta * q * q * H0 / 8
            return mp.mpc(0), mp.mpc(0), 2 * c2
        return mp.mpc(0), mp.mpc(0), mp.mpc(0)

    if side not in (None, "inside", "outside"):
        raise ValueError("side must be None, 'inside', or 'outside'")

    def source(s):
        return s * mp.besselj(n, kappa * s)

    def inside_integral():
        return mp.quad(lambda s: source(s) * mp.besseli(n, q * s), [0, r])

    def outside_integral():
        return mp.quad(lambda s: source(s) * mp.besselk(n, q * s), [r, R])

    if r < R:
        F = inside_integral()
        H = outside_integral()
    else:
        F = mp.quad(lambda s: source(s) * mp.besseli(n, q * s), [0, R])
        H = mp.mpf(0)
    Iv = mp.besseli(n, q * r)
    K = mp.besselk(n, q * r)
    Ip = mp.besseli(n + 1, q * r) + n * Iv / (q * r)
    Kp = n * K / (q * r) - mp.besselk(n + 1, q * r)
    Ipp = -Ip / (q * r) + (1 + mp.mpf(n * n) / ((q * r) ** 2)) * Iv
    Kpp = -Kp / (q * r) + (1 + mp.mpf(n * n) / ((q * r) ** 2)) * K
    J = mp.besselj(n, kappa * r)
    f = -beta * (Iv * H + K * F)
    fp = -beta * q * (Ip * H + Kp * F)
    inside = (r < R) if side is None else side == "inside"
    fpp = -beta * q * q * (Ipp * H + Kpp * F) + (beta * J if inside else 0)
    return f, fp, fpp


def q_zero_oracle_radial(
    m: int, kappa: mp.mpf, R: mp.mpf, A: mp.mpc, r: mp.mpf, side: str | None = None
) -> tuple[mp.mpc, mp.mpc, mp.mpc]:
    """Independent two-dimensional Poisson radial integral with frozen gauge."""
    n = abs(m)
    beta = _beta(m, A)
    if _zero_density(m, kappa):
        return mp.mpc(0), mp.mpc(0), mp.mpc(0)
    if side not in (None, "inside", "outside"):
        raise ValueError("side must be None, 'inside', or 'outside'")

    if n == 0:

        def cumulative(s):
            upper = min(s, R)
            if kappa == 0:
                return upper * upper / 2
            return upper * mp.besselj(1, kappa * upper) / kappa

        MR = cumulative(R)
        if r == 0:

            def reference_integrand(s):
                return mp.mpf(0) if s == 0 else cumulative(s) / s

            f0 = -beta * mp.quad(reference_integrand, [0, R])
            return f0, mp.mpc(0), beta / 2
        M = cumulative(r)
        inside = (r < R) if side is None else side == "inside"
        if inside:
            f = -beta * mp.quad(lambda s: cumulative(s) / s, [r, R])
            fpp = beta * (mp.besselj(0, kappa * r) - M / (r * r))
        else:
            f = beta * MR * mp.log(r / R)
            fpp = -beta * MR / (r * r)
        return f, beta * M / r, fpp

    if kappa == 0:
        return mp.mpc(0), mp.mpc(0), mp.mpc(0)
    Fmax = mp.quad(lambda s: s ** (n + 1) * mp.besselj(n, kappa * s), [0, R])
    if r == 0:
        H0 = mp.quad(lambda s: s ** (1 - n) * mp.besselj(n, kappa * s), [0, R])
        return (
            mp.mpc(0),
            -beta * H0 / (2 * n) if n == 1 else mp.mpc(0),
            (-beta * H0 / n if n == 2 else mp.mpc(0)),
        )
    upper = min(r, R)
    F = mp.quad(lambda s: s ** (n + 1) * mp.besselj(n, kappa * s), [0, upper])
    inside = (r < R) if side is None else side == "inside"
    if inside:
        H = mp.quad(lambda s: s ** (1 - n) * mp.besselj(n, kappa * s), [r, R])
        f = -beta * (r ** (-n) * F + r**n * H) / (2 * n)
        fp = beta * (r ** (-n - 1) * F - r ** (n - 1) * H) / 2
        fpp = (
            beta * mp.besselj(n, kappa * r)
            - beta * ((n + 1) * r ** (-n - 2) * F + (n - 1) * r ** (n - 2) * H) / 2
        )
        return f, fp, fpp
    f = -beta * Fmax * r ** (-n) / (2 * n)
    fp = beta * Fmax * r ** (-n - 1) / 2
    fpp = -beta * (n + 1) * Fmax * r ** (-n - 2) / 2
    return f, fp, fpp
