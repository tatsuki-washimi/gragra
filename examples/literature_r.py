"""LIT-R example: Rayleigh-wave gravity above and below a homogeneous half-space.

Self-contained smoke example for H19 (Harms, Living Reviews in Relativity
22, 6 (2019)) Eq. (96)-(98), as implemented in
``gragra.analytic.rayleigh_literature`` and
``gragra.observables.rayleigh_literature``. No external data, plotting, or
file output. See ``docs/design/literature-r.md`` for the full literature
mapping and evidence classification.
"""

from __future__ import annotations

import numpy as np

from gragra.observables.rayleigh_literature import (
    rayleigh_halfspace_allspace_observables,
)


def main() -> None:
    # Poisson-solid half-space (v_P/v_S = sqrt(3)).
    v_s = 1500.0
    v_p = np.sqrt(3.0) * v_s
    density = 2300.0
    frequency = 2.5  # Hz
    vertical_surface_amplitude = 1e-9  # m, complex amplitude at phase origin

    print("--- Half-space parameters ---")
    print(f"v_P = {v_p:.1f} m/s, v_S = {v_s:.1f} m/s, density = {density} kg/m^3")
    print(
        f"frequency = {frequency} Hz, vertical surface amplitude = "
        f"{vertical_surface_amplitude:.1e} m"
    )

    # A vertical profile straddling the free surface (z=0 itself is rejected).
    depths_above = np.array([50.0, 10.0, 1.0, 0.1])
    depths_below = np.array([0.1, 1.0, 10.0, 50.0])
    targets = np.zeros((8, 3))
    targets[:4, 2] = depths_above
    targets[4:, 2] = -depths_below

    out = rayleigh_halfspace_allspace_observables(
        targets,
        density_kg_m3=density,
        v_p_m_s=v_p,
        v_s_m_s=v_s,
        frequency_hz=frequency,
        vertical_surface_amplitude_m=vertical_surface_amplitude,
        direction=(1.0, 0.0, 0.0),
    )

    print(
        "\n--- SI potential and vertical acceleration vs. height (m/s^2, m^2/s^2) ---"
    )
    print("   z (m)   |            Phi (m^2/s^2)            |   a_z (m/s^2)")
    for i in range(targets.shape[0]):
        z = targets[i, 2]
        phi = out.potential[i]
        az = out.acceleration[i, 2]
        print(
            f"  {z:8.2f} | {phi.real:+.4e} {phi.imag:+.4e}j | "
            f"{az.real:+.4e} {az.imag:+.4e}j"
        )

    print(
        "\nNote: trace(gradient) is 0 above ground (vacuum Poisson equation) "
        "and tracks -4*pi*G*delta_rho(target) below ground; both are exact "
        "identities checked in tests/test_analytic_rayleigh_literature.py, "
        "not approximations."
    )


if __name__ == "__main__":
    main()
