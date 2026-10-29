"""Example 10: Rayleigh wave displacement field and evanescent factor in half-space.

This is a smoke example that demonstrates how to calculate the complex displacement
field of a Rayleigh wave and the corresponding evanescent decay factor at various depths
in a homogeneous elastic half-space.
Plotting and file output are disabled.
"""

import numpy as np

from gragra.waves import (
    evanescent_complex_factor,
    rayleigh_displacement,
    rayleigh_ellipticity,
    rayleigh_speed,
)


def main() -> None:
    # Medium properties (Poisson solid)
    v_p = 5500.0  # m/s
    v_s = 3175.0  # m/s

    print("--- Homogeneous Elastic Half-Space Parameters ---")
    print(f"P-wave speed (v_P): {v_p} m/s")
    print(f"S-wave speed (v_S): {v_s} m/s")

    # Calculate Rayleigh speed
    c_r = rayleigh_speed(v_p, v_s)
    print(f"Rayleigh speed (c_R): {c_r:.4f} m/s (c_R/v_S = {c_r / v_s:.6f})")

    # Wave parameters
    wavenumber = 2.0  # rad/m
    decay = 0.7  # 1/m (attenuation factor kappa)
    direction = np.array([1.0, 0.0, 0.0])  # Propagation along x-axis
    amplitude = 1.0e-6  # 1 micron amplitude at surface

    print("\n--- Wave Parameters ---")
    print(f"Wavenumber (k): {wavenumber} rad/m")
    print(f"Decay parameter (kappa): {decay} 1/m")
    print(f"Propagation direction: {direction}")
    print(f"Surface amplitude: {amplitude:.1e} m")

    # Evaluation depths (z <= 0)
    depths = np.array([0.0, 0.5, 0.6047405, 1.0, 2.0, 5.0])
    positions = np.zeros((len(depths), 3))
    positions[:, 0] = 0.5  # evaluate at x = 0.5 m
    positions[:, 2] = -depths  # z = -depth

    # 1. Calculate Rayleigh displacements
    disp = rayleigh_displacement(
        positions, direction, wavenumber, v_p, v_s, amplitude_m=amplitude
    )

    # 2. Calculate H/V ellipticity
    ellipticity = rayleigh_ellipticity(depths, wavenumber, v_p, v_s)

    # 3. Calculate evanescent complex factor (with kappa)
    factor = evanescent_complex_factor(
        positions, direction[np.newaxis, :], wavenumber, decay
    )

    print("\n--- Numerical Results ---")
    header = (
        "  Depth (m) | u_x (m)                       | u_z (m)            "
        "           | H/V Ratio | Evanescent Factor"
    )
    print(header)
    print("  " + "-" * len(header))
    for i, h in enumerate(depths):
        ux = disp[i, 0]
        uz = disp[i, 2]
        ev_fac = factor[i, 0, 0]
        print(
            f"  {h:9.4f} | "
            f"{ux.real:+11.3e} + {ux.imag:10.3e}j | "
            f"{uz.real:+11.3e} + {uz.imag:10.3e}j | "
            f"{ellipticity[i]:9.6f} | "
            f"{np.abs(ev_fac):.6f}"
        )

    print(
        "\nNote: The evanescent factor shows exponential decay"
        " e^(-kappa * depth) with depth."
    )
    print(
        "      H/V Ratio is 0.681201 at the surface (h=0) for the actual"
        " v_P/v_S ratio (5500/3175 = 1.732283),"
    )
    print(
        "      which differs from the golden value 0.681250 (which assumes"
        " Poisson solid v_P/v_S = sqrt(3) = 1.732051)."
    )
    print(
        "      The horizontal displacement node h = 0.6047405 m is included"
        " in the depth samples (H/V Ratio is zero)."
    )


if __name__ == "__main__":
    main()
