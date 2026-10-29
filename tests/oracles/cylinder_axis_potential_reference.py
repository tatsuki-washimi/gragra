"""Independent high-precision Newton-volume references for axis potential.

No production helper is imported.  The volume integral integrates source
azimuth and source height analytically, then integrates source radius with
mpmath tanh-sinh quadrature.  It does not integrate the production primitive.
Float inputs are converted exactly, so comparisons describe the actual binary64
geometry supplied to the public API.  Results are G-free, in kg/m.
"""

import mpmath as mp


def center_potential(radius, height, density, *, dps=100):
    """Independent center identity, with the infinity-zero potential gauge."""
    with mp.workdps(dps):
        radius, height, density = map(mp.mpf, (radius, height, density))
        half = height / 2
        return (
            -2
            * mp.pi
            * density
            * (
                half * mp.sqrt(radius**2 + half**2)
                + radius**2 * mp.asinh(half / radius)
                - half**2
            )
        )


def center_axial_potential(radius, height, density, *, dps=100, refinement=0):
    """Numerically integrate -4*pi*rho*int_0^(H/2)(sqrt(R²+z²)-z) dz."""
    with mp.workdps(dps):
        radius, height, density = map(mp.mpf, (radius, height, density))
        half = height / 2
        splits = [half * i / 2**refinement for i in range(2**refinement + 1)]
        return (
            -4
            * mp.pi
            * density
            * mp.quad(
                lambda z: mp.sqrt(radius**2 + z**2) - z,
                splits,
                method="tanh-sinh",
            )
        )


def volume_potential(radius, height, density, z, *, dps=100, refinement=0):
    """Return potential and radial splits for an independent volume integral.

    Integrate ``-2*pi*rho*s*integral dz'/sqrt(s²+(z'-z)²) ds``.
    Refinement inserts midpoints in every radial interval independently of dps.
    Splits near the endpoint separations resolve very thin source cylinders.
    The returned split strings permit exact reconstruction at the stated dps.
    """
    with mp.workdps(dps):
        radius, height, density, z = map(mp.mpf, (radius, height, density, z))
        lower, upper = -height / 2 - z, height / 2 - z
        splits = sorted(
            {mp.mpf(0), radius / 16, radius / 4, radius}
            | {abs(u) for u in (lower, upper) if 0 < abs(u) < radius}
        )
        for _ in range(refinement):
            splits = sorted(
                set(splits)
                | {(a + b) / 2 for a, b in zip(splits, splits[1:], strict=False)}
            )

        def ring(s):
            if s == 0:
                return mp.mpf(0)
            return s * (mp.asinh(upper / s) - mp.asinh(lower / s))

        value = -2 * mp.pi * density * mp.quad(ring, splits, method="tanh-sinh")
        return value, tuple(mp.nstr(s, dps) for s in splits)
