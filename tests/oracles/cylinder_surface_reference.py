"""Independent adaptive surface integration; never imports production helpers.

Side z and cap radius integrations are elementary; scipy integrates the
remaining physical azimuth. This is a numerical reference, not an analytic
production route. Components are evaluated independently, including trace.
"""

import numpy as np


def surface_reference(radius, height, density, point, tolerance=1e-11):
    from scipy.integrate import quad_vec

    x, y, z = point
    r = np.hypot(x, y)
    lo, hi = -height / 2 - z, height / 2 - z

    def integrand(phi):
        c, s = np.cos(phi), np.sin(phi)
        q = (radius - r) ** 2 + 4 * radius * r * np.sin(phi / 2) ** 2
        dl, dh = np.sqrt(q + lo * lo), np.sqrt(q + hi * hi)
        axial = hi / (q * dh) - lo / (q * dl)
        ar = -radius * c * (np.arcsinh(hi / np.sqrt(q)) - np.arcsinh(lo / np.sqrt(q)))
        rr = -radius * (radius * c - r) * c * axial
        pp = -(radius**2) * s * s * axial
        rz = radius * c * (1 / dh - 1 / dl)

        def cap(u, distance):
            b2 = r * r * s * s + u * u
            d0 = np.hypot(r, u)
            potential = (
                distance
                - d0
                + r
                * c
                * (
                    np.arcsinh((radius - r * c) / np.sqrt(b2))
                    + np.arcsinh(r * c / np.sqrt(b2))
                )
            )
            omega = u * ((r * c * radius - r * r - u * u) / distance + d0) / b2
            return potential, omega

        sl, ol = cap(lo, dl)
        sh, oh = cap(hi, dh)
        return np.array([ar, sl - sh, rr, pp, rz, ol - oh])

    values, error = quad_vec(
        integrand,
        0.0,
        np.pi,
        epsabs=tolerance,
        epsrel=tolerance,
        limit=1000,
        points=np.linspace(0.0, np.pi, 5 if tolerance >= 1e-8 else 8),
    )
    ar, az, rr, pp, rz, zz = 2 * density * values
    radial = np.array([x / r, y / r, 0.0])
    tangent = np.array([-y / r, x / r, 0.0])
    axial = np.array([0.0, 0.0, 1.0])
    acc = ar * radial + az * axial
    grad = (
        rr * np.outer(radial, radial)
        + pp * np.outer(tangent, tangent)
        + zz * np.outer(axial, axial)
    )
    grad += rz * (np.outer(radial, axial) + np.outer(axial, radial))
    return acc, grad, 2 * density * error
