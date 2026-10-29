"""Independent high-precision Newton integrals on the propagation axis.

Validation only: mpmath spatial quadrature after analytic azimuth/cylindrical
radius integration. No production coefficients, Bessel functions or derivatives
are imported. Fixed radius=density=displacement=1; dimensionless target r>0.
"""

import argparse

import numpy as np


def axis_reference(r, k, wave, dps=40):
    import mpmath as mp

    with mp.workdps(dps):
        r, k = mp.mpf(float(r)), mp.mpf(float(k))
        if r <= 0 or r == 1 or wave not in ("P", "S"):
            raise ValueError("require r>0 off wall, and P/S")
        low, high = abs(1 - r), 1 + r
        # Geometric distance intervals resolve the near-wall peak directly.
        cuts = [low]
        while cuts[-1] * 4 < high:
            cuts.append(cuts[-1] * 4)
        cuts.append(high)

        def wall_integral(kind):
            def integrand(distance):
                mu = (1 + r * r - distance * distance) / (2 * r)
                h = mu - r
                phase = mp.exp(1j * k * mu)
                if wave == "P":
                    factors = [
                        1 / distance,
                        -h / distance**3,
                        -(
                            mp.mpf("1.5") * (1 - mu * mu) / distance**5
                            - 1 / distance**3
                        ),
                        -(3 * h * h / distance**5 - 1 / distance**3),
                    ]
                    return 2 * mp.pi * mu * phase * factors[kind] * distance / r
                factors = [-1 / distance**3, -3 * h / distance**5]
                return mp.pi * (1 - mu * mu) * phase * factors[kind] * distance / r

            return mp.quad(integrand, cuts)

        wall = np.zeros(13, complex)
        bulk = np.zeros(13, complex)
        if wave == "S":
            wall[1] = complex(wall_integral(0))
            wall[6] = wall[10] = complex(wall_integral(1))
        else:
            wall[0] = complex(wall_integral(0))
            wall[3] = complex(wall_integral(1))
            wall[4] = wall[8] = complex(wall_integral(2))
            wall[12] = complex(wall_integral(3))
            # Removed sphere: integrate cylindrical radius and azimuth exactly,
            # then integrate z numerically, retaining the interior local term.
            zcuts = sorted(
                set(
                    [-mp.mpf(1), mp.mpf(0), mp.mpf(1)]
                    + ([r] if r < 1 else [])
                    + [1 - min(mp.mpf(2), low * 4**n) for n in range(len(cuts))]
                )
            )

            def volume_integral(kind):
                def integrand(z):
                    distance = mp.sqrt(1 + r * r - 2 * r * z)
                    h = r - z
                    factors = [
                        distance - abs(h),
                        h / distance - mp.sign(h),
                        (1 - z * z) / distance**3,
                    ]
                    return mp.exp(1j * k * z) * factors[kind]

                return mp.quad(integrand, zcuts)

            removed = np.zeros(13, complex)
            removed[0] = complex(2j * mp.pi * k * volume_integral(0))
            removed[3] = complex(-2j * mp.pi * k * volume_integral(1))
            transverse = 1j * mp.pi * k * volume_integral(2)
            removed[4] = removed[8] = complex(transverse)
            removed[12] = complex(
                -2 * transverse + (4j * mp.pi * k * mp.exp(1j * k * r) if r < 1 else 0)
            )
            bulk[0] = complex(4j * mp.pi / k * mp.exp(1j * k * r))
            bulk[3] = complex(4 * mp.pi * mp.exp(1j * k * r))
            bulk[12] = complex(4j * mp.pi * k * mp.exp(1j * k * r))
            bulk -= removed
        return {"bulk": bulk, "wall": wall, "total": bulk + wall}


def main():
    from gragra.analytic import spherical_cavity_observables_g_free

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distances", nargs="+", type=float, default=[0.1, 0.01, 1e-3, 1e-6, 1e-9]
    )
    args = parser.parse_args()
    print("wave,ka,distance,side,observable,reference_delta,series_error")
    for wave in ("P", "S"):
        for k in (0.01, 0.8, 5.0):
            for delta in args.distances:
                for sign in (-1, 1):
                    r = 1 + sign * delta
                    low, high = (axis_reference(r, k, wave, dps) for dps in (35, 50))
                    actual = spherical_cavity_observables_g_free(
                        [[0, 0, r]],
                        radius_m=1,
                        density_kg_m3=1,
                        displacement_m=1,
                        wavenumber_rad_m=k,
                        propagation_direction=[0, 0, 1],
                        polarization=[0, 0, 1] if wave == "P" else [1, 0, 0],
                        wave_type=wave,
                    )
                    for name, sl in zip(
                        ("potential", "acceleration", "gradient"),
                        (slice(0, 1), slice(1, 4), slice(4, 13)),
                        strict=True,
                    ):
                        ref_error = max(
                            np.linalg.norm(high[p][sl] - low[p][sl]) for p in high
                        ) / (4 * np.pi)
                        error = max(
                            np.linalg.norm(
                                getattr(actual[p], name).ravel() - high[p][sl]
                            )
                            for p in high
                        ) / (4 * np.pi)
                        budget = 1e-7
                        if ref_error > budget / 10 or error > budget:
                            raise ValueError(f"unresolved {wave} ka={k} r={r}: {name}")
                        print(
                            f"{wave},{k},{delta},{sign},{name},{ref_error:.3e},{error:.3e}",
                            flush=True,
                        )


if __name__ == "__main__":
    main()
