"""Independent centered spherical Newton volume kernel for Amann Eq13.

The published half-spherical-cavern interpretation is an explicit assumption.
This integral verifies the spherical volume kernel/prefactor, never arbitrary
cavern boundary acoustics. delta-rho=p/c**2, volume gradient/G is
 i*n*(4*pi/c**2/k)*integral_0^1[1-cos(kR*u)]du.
"""

from ..a1.generator import context, read_inputs


def volume_kernel(c, x, resolution=4, route="angular"):
    if x < 0 or resolution not in (1, 2, 4):
        raise ValueError("x>=0 and resolution1/2/4 required")
    if not x:
        return c.mpf(0)
    n = max(1, int(c.ceil(x / c.pi))) * resolution
    if n > 10000:
        raise ValueError("volume oscillation resource limit")
    if route == "angular":
        return c.quad(
            lambda u: 2 * c.sin(x * u / 2) ** 2,
            [c.mpf(i) / n for i in range(n + 1)],
            maxdegree=8,
        )
    if route != "radial":
        raise ValueError("unknown kernel route")

    def radial(t):
        if t == 0:
            return c.mpf(0)
        if abs(t) < c.mpf("0.5"):
            # Entire Taylor series independently avoids small-t subtraction.
            return c.nsum(
                lambda m: (
                    (-1) ** (m + 1) * 2 * m * t ** (2 * m - 1) / c.factorial(2 * m + 1)
                ),
                [1, c.inf],
            )
        return (c.sin(t) - t * c.cos(t)) / t**2

    return c.quad(radial, [x * i / n for i in range(n + 1)], maxdegree=8)


def cavern_matrix(inputs, *, dps=80, resolution=4, route="angular"):
    c = context(dps)
    p = read_inputs(c, inputs)
    k = 2 * c.pi * p["frequency_hz"] / p["sound_speed_m_s"]
    kernel = volume_kernel(c, k * p["radius_m"], resolution, route)
    amplitude = 4 * c.pi / (p["sound_speed_m_s"] ** 2 * k) * kernel
    theta = [c.pi * i / resolution for i in range(resolution + 1)]
    phi = [c.pi * i / resolution for i in range(2 * resolution + 1)]
    polar = [lambda t: c.sin(t), lambda t: c.sin(t), lambda t: c.cos(t)]
    azi = [lambda a: c.cos(a), lambda a: c.sin(a), lambda _a: c.mpf(1)]
    return [
        p["pressure_psd_pa2_hz"]
        * amplitude**2
        * c.quad(
            lambda t, i=i, j=j: polar[i](t) * polar[j](t) * c.sin(t), theta, maxdegree=8
        )
        * c.quad(lambda a, i=i, j=j: azi[i](a) * azi[j](a), phi, maxdegree=8)
        / (4 * c.pi)
        for i in range(3)
        for j in range(3)
    ]
