"""Integral-free cylindrical density-mode candidate used only for evaluation.

The module is an isolated mathematical prototype, not a gragra production API.
All inputs are interpreted with m in Z, kappa >= 0, R > 0, real kz, and complex
linear amplitude A in kg m^-3. mpmath is used for the declared high precision.
"""

# ruff: noqa: N803, N806

from __future__ import annotations

import mpmath as mp

G_SI = "6.67430e-11"


def density_sign(m: int) -> int:
    n = abs(m)
    return (-1) ** n if m < 0 else 1


def _beta(m: int, A: mp.mpc) -> mp.mpc:
    return 4 * mp.pi * mp.mpf(G_SI) * density_sign(m) * A


def _j(n: int, x: mp.mpf) -> mp.mpf:
    return mp.besselj(n, x)


def _jp(n: int, x: mp.mpf) -> mp.mpf:
    return (mp.besselj(n - 1, x) - mp.besselj(n + 1, x)) / 2


def _jpp(n: int, x: mp.mpf) -> mp.mpf:
    y = _j(n, x)
    yp = _jp(n, x)
    return -yp / x - (1 - mp.mpf(n * n) / (x * x)) * y


def _ip(n: int, x: mp.mpf) -> mp.mpf:
    return mp.besseli(n + 1, x) + mp.mpf(n) * mp.besseli(n, x) / x


def _ipp(n: int, x: mp.mpf) -> mp.mpf:
    y = mp.besseli(n, x)
    yp = _ip(n, x)
    return -yp / x + (1 + mp.mpf(n * n) / (x * x)) * y


def _kp(n: int, x: mp.mpf) -> mp.mpf:
    return mp.mpf(n) * mp.besselk(n, x) / x - mp.besselk(n + 1, x)


def _kpp(n: int, x: mp.mpf) -> mp.mpf:
    y = mp.besselk(n, x)
    yp = _kp(n, x)
    return -yp / x + (1 + mp.mpf(n * n) / (x * x)) * y


def q_nonzero_coefficients(
    m: int, kappa: mp.mpf, kz: mp.mpf, R: mp.mpf, A: mp.mpc
) -> tuple[mp.mpf, mp.mpc, mp.mpc, mp.mpc]:
    """Return q, beta, interior-I coefficient C and exterior-K coefficient D."""
    n = abs(m)
    q = abs(kz)
    if q == 0:
        raise ValueError("q_nonzero_coefficients requires kz != 0")
    beta = _beta(m, A)
    den = kappa * kappa + q * q
    jR = _j(n, kappa * R)
    iR = mp.besseli(n, q * R)
    kR = mp.besselk(n, q * R)
    primitive_R = (
        R
        / den
        * (q * mp.besseli(n + 1, q * R) * jR + kappa * iR * _j(n + 1, kappa * R))
    )
    C = beta * (jR / den - primitive_R * kR) / iR
    D = -beta * primitive_R
    return q, beta, C, D


def q_nonzero_radial(
    m: int,
    kappa: mp.mpf,
    kz: mp.mpf,
    R: mp.mpf,
    A: mp.mpc,
    r: mp.mpf,
    side: str | None = None,
) -> tuple[mp.mpc, mp.mpc, mp.mpc]:
    """Return f, df/dr, d2f/dr2 for Phi=f(r) exp(i*m*phi+i*kz*z)."""
    n = abs(m)
    if kz == 0:
        raise ValueError("q_nonzero_radial requires kz != 0")
    if kappa == 0 and n > 0:
        return mp.mpc(0), mp.mpc(0), mp.mpc(0)
    q, beta, C, D = q_nonzero_coefficients(m, kappa, kz, R, A)
    den = kappa * kappa + q * q
    if r == 0:
        if n == 0:
            f0 = -beta / den + C
            f2 = beta * kappa * kappa / (2 * den) + C * q * q / 2
            return f0, mp.mpc(0), f2
        if n == 1:
            c1 = (-beta * kappa / den + C * q) / 2
            return mp.mpc(0), c1, mp.mpc(0)
        if n == 2:
            c2 = -beta * kappa * kappa / (8 * den) + C * q * q / 8
            return mp.mpc(0), mp.mpc(0), 2 * c2
        return mp.mpc(0), mp.mpc(0), mp.mpc(0)

    if side not in (None, "inside", "outside"):
        raise ValueError("side must be None, 'inside', or 'outside'")
    inside = (r < R) if side is None else side == "inside"
    if inside:
        xj = kappa * r
        xi = q * r
        jv = _j(n, xj)
        jvp = _jp(n, xj) if kappa != 0 else mp.mpf(0)
        jvpp = _jpp(n, xj) if kappa != 0 else mp.mpf(0)
        iv = mp.besseli(n, xi)
        ivp = _ip(n, xi)
        ivpp = _ipp(n, xi)
        f = -beta * jv / den + C * iv
        fp = -beta * kappa * jvp / den + C * q * ivp
        fpp = -beta * kappa * kappa * jvpp / den + C * q * q * ivpp
        return f, fp, fpp

    xk = q * r
    kv = mp.besselk(n, xk)
    kvp = _kp(n, xk)
    kvpp = _kpp(n, xk)
    return D * kv, D * q * kvp, D * q * q * kvpp


def q_zero_radial(
    m: int, kappa: mp.mpf, R: mp.mpf, A: mp.mpc, r: mp.mpf, side: str | None = None
) -> tuple[mp.mpc, mp.mpc, mp.mpc]:
    """Return the kz=0 radial solution with the frozen branch gauge."""
    n = abs(m)
    beta = _beta(m, A)
    if kappa == 0 and n > 0:
        return mp.mpc(0), mp.mpc(0), mp.mpc(0)
    if side not in (None, "inside", "outside"):
        raise ValueError("side must be None, 'inside', or 'outside'")
    inside = (r < R) if side is None else side == "inside"

    if n == 0:
        if r == 0:
            if kappa == 0:
                f0 = -beta * R * R / 4
            else:
                f0 = beta * (mp.besselj(0, kappa * R) - 1) / (kappa * kappa)
            return f0, mp.mpc(0), beta / 2
        if inside:
            if kappa == 0:
                return (beta * (r * r - R * R) / 4, beta * r / 2, beta / 2)
            f = (
                beta
                * (mp.besselj(0, kappa * R) - mp.besselj(0, kappa * r))
                / (kappa * kappa)
            )
            fp = beta * mp.besselj(1, kappa * r) / kappa
            fpp = beta * (mp.besselj(0, kappa * r) - mp.besselj(2, kappa * r)) / 2
            return f, fp, fpp
        if kappa == 0:
            D = beta * R * R / 2
        else:
            D = beta * R * mp.besselj(1, kappa * R) / kappa
        return D * mp.log(r / R), D / r, -D / (r * r)

    if kappa == 0:
        return mp.mpc(0), mp.mpc(0), mp.mpc(0)
    if r == 0:
        if n == 1:
            # The boundary solution is f=-beta*J_n/kappa^2+C*r^n.
            D = -beta * R ** (n + 1) * mp.besselj(n + 1, kappa * R) / (2 * n * kappa)
            C = beta * mp.besselj(n, kappa * R) / (kappa * kappa * R**n) + D / R ** (
                2 * n
            )
            c1 = -beta / (2 * kappa) + C
            return mp.mpc(0), c1, mp.mpc(0)
        if n == 2:
            D = -beta * R ** (n + 1) * mp.besselj(n + 1, kappa * R) / (2 * n * kappa)
            C = beta * mp.besselj(n, kappa * R) / (kappa * kappa * R**n) + D / R ** (
                2 * n
            )
            c2 = -beta / 8 + C
            return mp.mpc(0), mp.mpc(0), 2 * c2
        return mp.mpc(0), mp.mpc(0), mp.mpc(0)

    D = -beta * R ** (n + 1) * mp.besselj(n + 1, kappa * R) / (2 * n * kappa)
    C = beta * mp.besselj(n, kappa * R) / (kappa * kappa * R**n) + D / R ** (2 * n)
    if inside:
        x = kappa * r
        jv = mp.besselj(n, x)
        jvp = _jp(n, x)
        jvpp = _jpp(n, x)
        f = -beta * jv / (kappa * kappa) + C * r**n
        fp = -beta * jvp / kappa + n * C * r ** (n - 1)
        fpp = -beta * jvpp + n * (n - 1) * C * r ** (n - 2)
        return f, fp, fpp
    return D * r ** (-n), -n * D * r ** (-n - 1), n * (n + 1) * D * r ** (-n - 2)


def field_from_radial(
    m: int,
    kz: mp.mpf,
    r: mp.mpf,
    phi: mp.mpf,
    z: mp.mpf,
    radial: tuple[mp.mpc, mp.mpc, mp.mpc],
) -> dict:
    """Convert radial amplitude and derivatives to cylindrical and Cartesian fields."""
    f, fp, fpp = radial
    phase = mp.e ** (1j * m * phi + 1j * kz * z)
    f, fp, fpp = f * phase, fp * phase, fpp * phase
    a = mp.matrix([-fp, -1j * m * f / r, -1j * kz * f])
    gamma = mp.matrix(
        [
            [-fpp, -1j * m * (fp / r - f / (r * r)), -1j * kz * fp],
            [
                -1j * m * (fp / r - f / (r * r)),
                -fp / r + m * m * f / (r * r),
                m * kz * f / r,
            ],
            [-1j * kz * fp, m * kz * f / r, kz * kz * f],
        ]
    )
    c = mp.cos(phi)
    s = mp.sin(phi)
    E = mp.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    ac = E * a
    gc = E * gamma * E.T
    return {"Phi": f, "a_cyl": a, "Gamma_cyl": gamma, "a_cart": ac, "Gamma_cart": gc}


def axis_fields(
    m: int, kz: mp.mpf, z: mp.mpf, radial: tuple[mp.mpc, mp.mpc, mp.mpc]
) -> dict:
    """Cartesian axis limits from the regular Taylor coefficient of f(r)."""
    n = abs(m)
    f, fp, fpp = radial
    phase = mp.e ** (1j * kz * z)
    if n == 0:
        a = mp.matrix([0, 0, -1j * kz * f]) * phase
        gamma = mp.matrix([[-fpp, 0, 0], [0, -fpp, 0], [0, 0, kz * kz * f]]) * phase
        return {"Phi": f * phase, "a_cart": a, "Gamma_cart": gamma}
    if n == 1:
        c1 = fp
        a = mp.matrix([-c1, -1j * m * c1, 0]) * phase
        gamma = (
            mp.matrix(
                [
                    [0, 0, -1j * kz * c1],
                    [0, 0, m * kz * c1],
                    [-1j * kz * c1, m * kz * c1, 0],
                ]
            )
            * phase
        )
        return {"Phi": mp.mpc(0), "a_cart": a, "Gamma_cart": gamma}
    if n == 2:
        c2 = fpp / 2
        gamma = (
            mp.matrix(
                [
                    [-2 * c2, -1j * m * c2, 0],
                    [-1j * m * c2, 2 * c2, 0],
                    [0, 0, 0],
                ]
            )
            * phase
        )
        return {"Phi": mp.mpc(0), "a_cart": mp.matrix([0, 0, 0]), "Gamma_cart": gamma}
    return {"Phi": mp.mpc(0), "a_cart": mp.matrix([0, 0, 0]), "Gamma_cart": mp.zeros(3)}


def field(
    m: int,
    kappa: mp.mpf,
    kz: mp.mpf,
    R: mp.mpf,
    A: mp.mpc,
    r: mp.mpf,
    phi: mp.mpf,
    z: mp.mpf,
    branch: str,
    side: str | None = None,
) -> dict:
    if branch == "q_nonzero":
        radial = q_nonzero_radial(m, kappa, kz, R, A, r, side)
    elif branch == "q_zero":
        radial = q_zero_radial(m, kappa, R, A, r, side)
    else:
        raise ValueError("branch must be 'q_nonzero' or 'q_zero'")
    if r == 0:
        return axis_fields(m, kz, z, radial)
    return field_from_radial(m, kz, r, phi, z, radial)
