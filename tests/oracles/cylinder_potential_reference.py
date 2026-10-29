"""Numerical references that do not import any gragra production functions."""

import numpy as np


def boundary_potential(radius, height, density, point, tolerance=1e-11):
    """Physical surface integration; z/r antiderivatives and adaptive azimuth."""
    from scipy.integrate import quad

    x, y, z = point
    r = np.hypot(x, y)
    lo, hi = -height / 2 - z, height / 2 - z

    def integrand(phi):
        c, s = np.cos(phi), np.sin(phi)
        q = (radius - r) ** 2 + 4 * radius * r * np.sin(phi / 2) ** 2
        side = (
            radius
            * (radius - r * c)
            * (np.arcsinh(hi / np.sqrt(q)) - np.arcsinh(lo / np.sqrt(q)))
        )

        def cap(u):
            distance = np.sqrt(q + u * u)
            b = np.hypot(r * s, u)
            return (
                distance
                - np.hypot(r, u)
                + r * c * (np.arcsinh((radius - r * c) / b) + np.arcsinh(r * c / b))
            )

        return side + hi * cap(hi) - lo * cap(lo)

    result, error = quad(
        integrand,
        0,
        np.pi,
        epsabs=tolerance,
        epsrel=tolerance,
        points=np.unique(
            np.r_[
                np.linspace(0, np.pi, 5 if tolerance >= 1e-8 else 8),
                np.geomspace(1e-9, 0.1, 9 if tolerance >= 1e-8 else 17),
            ]
        ),
        limit=1000,
    )
    return -density * result, density * error


def volume_potential(radius, height, density, point, tolerance=1e-10):
    """Newton volume integral, analytic source z and numerical source r/phi."""
    from scipy.integrate import quad

    x, y, z = point
    r = np.hypot(x, y)
    lo, hi = -height / 2 - z, height / 2 - z

    def ring(s):
        def azimuth(phi):
            separation = np.sqrt((s - r) ** 2 + 4 * s * r * np.sin(phi / 2) ** 2)
            return np.arcsinh(hi / separation) - np.arcsinh(lo / separation)

        return s * quad(azimuth, 0, np.pi, epsabs=tolerance, epsrel=tolerance)[0]

    result, error = quad(
        ring,
        0,
        radius,
        points=[r] if 0 < r < radius else None,
        epsabs=tolerance,
        epsrel=tolerance,
        limit=500,
    )
    return -2 * density * result, 2 * density * error
