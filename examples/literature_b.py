"""LIT-B example: single-mode P/SV and reflected-P half-space gravity.

Self-contained smoke example for H19 (Harms, Living Reviews in Relativity
22, 6 (2019)) Eq. (86)-(90), as implemented in
``gragra.analytic.bodywave_literature`` and
``gragra.observables.bodywave_literature``. No external data, plotting, or
file output. The reflected-P production observable follows the independently
reconstructed physical surface source; the literal Eq. (90) coefficient is
kept as a private reference helper. See ``docs/design/literature-b.md`` for
the full literature mapping and evidence classification.

Demonstrates two distinct physical models -- do not confuse them:

1. **Single mode** (``halfspace_single_bodywave_observables``): a plane P or
   SV wave *without* compositing a free-surface reflection. Not by itself a
   solution of the traction-free boundary condition -- it is a building
   block.
2. **Reflected P** (``halfspace_reflected_p_above_observables``): the full
   incident-P + reflected-P + converted-SV composite above ground, which
   *does* satisfy the traction-free boundary condition (independently
   verified in ``tests/test_bodywave_literature_oracle.py``).
"""

from __future__ import annotations

import numpy as np

from gragra.observables.bodywave_literature import (
    halfspace_reflected_p_above_observables,
    halfspace_single_bodywave_observables,
)


def main() -> None:
    v_s = 3200.0
    v_p = 5800.0
    density = 2700.0
    frequency = 5.0  # Hz
    theta = np.deg2rad(30.0)
    azimuth = 0.0
    xi0 = 1e-9  # m, complex displacement amplitude

    print("--- Half-space parameters ---")
    print(f"v_P = {v_p:.1f} m/s, v_S = {v_s:.1f} m/s, density = {density} kg/m^3")
    print(f"frequency = {frequency} Hz, incidence = {np.rad2deg(theta):.0f} deg")

    print("\n--- 1. Single P wave, no reflection (H19 Eq. 86/87) ---")
    targets = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, -10.0]])
    single_p = halfspace_single_bodywave_observables(
        targets,
        wave_type="P",
        density_kg_m3=density,
        v_p_m_s=v_p,
        v_s_m_s=v_s,
        frequency_hz=frequency,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        displacement_m=xi0,
    )
    for i in range(targets.shape[0]):
        z = targets[i, 2]
        print(f"  z={z:+6.1f} m: Phi={single_p.potential[i]:.4e} m^2/s^2")

    print("\n--- 2. Single SV wave, no reflection (H19 Eq. 88) ---")
    single_sv = halfspace_single_bodywave_observables(
        targets,
        wave_type="SV",
        density_kg_m3=density,
        v_p_m_s=v_p,
        v_s_m_s=v_s,
        frequency_hz=frequency,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        displacement_m=xi0,
    )
    for i in range(targets.shape[0]):
        z = targets[i, 2]
        print(f"  z={z:+6.1f} m: Phi={single_sv.potential[i]:.4e} m^2/s^2")

    print("\n--- 3. Reflected P + converted SV composite, ground only ---")
    ground_targets = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 50.0]])
    reflected = halfspace_reflected_p_above_observables(
        ground_targets,
        density_kg_m3=density,
        v_p_m_s=v_p,
        v_s_m_s=v_s,
        frequency_hz=frequency,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        incident_displacement_m=xi0,
    )
    for i in range(ground_targets.shape[0]):
        z = ground_targets[i, 2]
        print(f"  z={z:+6.1f} m: Phi={reflected.potential[i]:.4e} m^2/s^2")

    print(
        "\n--- 4. H19's stated zero at vertical incidence (theta=0) ---\n"
        "The paper states the reflected-P composite vanishes for a vertically "
        "propagating incident P-wave (interference of incident+reflected P, no "
        "P->S conversion at theta=0). The production API uses the physical "
        "reflected-P coefficient; the literal Eq. (90) coefficient is retained "
        "as a private literature reference. Neither path special-cases theta=0 "
        "-- the zero falls out on its own:"
    )
    reflected_normal = halfspace_reflected_p_above_observables(
        [[0.0, 0.0, 10.0]],
        density_kg_m3=density,
        v_p_m_s=v_p,
        v_s_m_s=v_s,
        frequency_hz=frequency,
        incidence_angle_rad=0.0,
        azimuth_rad=azimuth,
        incident_displacement_m=xi0,
    )
    print(f"  theta=0: Phi = {reflected_normal.potential[0]!r} (exactly zero)")


if __name__ == "__main__":
    main()
