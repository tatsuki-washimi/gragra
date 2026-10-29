"""Example 12: Gravitoelastic cross statistics + witness forward model.

Demonstrates the Phase 15-a cross-statistics workflow:

1. Build a bulk mass-element source and apply a P-wave plane-wave modulation
   via ``plane_wave_bulk_mass_weights`` to obtain complex bulk mass weights
   of shape ``(N, Ndir, Nf)``.
2. Build a G-free witness displacement field at a set of witness positions
   with ``plane_wave_displacement``, using the *same* wave parameters
   (k_hat, wavenumber, phase0) as the source -- this is required for the
   cross statistics to encode a physically meaningful phase relationship.
3. Compute the gravitoelastic cross tensor c_ij = <delta_a_i * conj(xi_j)>
   with ``coherent_gravitoelastic_tensor`` (G^1: G_SI is multiplied exactly
   once, inside the acceleration computation).
4. Compute the mirror-pair acceleration cross-spectral density
   CSD_ij = <delta_a_i(r_a) * conj(delta_a_j(r_b))> with
   ``coherent_acceleration_csd`` (G^2: two independently G_SI-included
   acceleration responses).

Note on complex outputs
-----------------------
Complex outputs are second-order cross statistics (cross-spectral
quantities), not time-domain fields; the phase encodes relative propagation
delay/coherence between the two inputs, not an absolute field phase.
"""

import numpy as np

from gragra import (
    G_SI,
    coherent_acceleration_csd,
    coherent_gravitoelastic_tensor,
)
from gragra.coherent import plane_wave_bulk_mass_weights
from gragra.waves import normalize_directions, plane_wave_displacement


def main() -> None:
    print(f"G_SI = {G_SI:.5e}  m^3 kg^-1 s^-2")

    # ------------------------------------------------------------------
    # 1. Bulk mass-element source: N=4 elements in the z=0 plane
    # ------------------------------------------------------------------
    n_sources = 4
    angles = np.linspace(0, 2 * np.pi, n_sources, endpoint=False)
    radius_m = 0.3  # m
    positions_m = np.column_stack(
        [
            radius_m * np.cos(angles),
            radius_m * np.sin(angles),
            np.zeros(n_sources),
        ]
    )
    volumes_m3 = np.full(n_sources, 0.1)
    rho0_kg_m3 = 2700.0  # crustal rock density

    # ------------------------------------------------------------------
    # 2. Shared wave parameters: Ndir=2 directions, Nf=1 wavenumber
    # ------------------------------------------------------------------
    raw_dirs = np.array(
        [
            [1.0, 0.0, 0.0],  # propagating in +x
            [0.0, 1.0, 0.0],  # propagating in +y
        ]
    )
    k_hat = normalize_directions(raw_dirs)  # (Ndir, 3)
    wavenumber_rad_m = np.array([2 * np.pi * 10.0])  # 10 Hz equivalent, (Nf,)
    ndir = len(k_hat)
    nf = len(wavenumber_rad_m)

    weights_kg = plane_wave_bulk_mass_weights(
        volumes_m3=volumes_m3,
        positions_m=positions_m,
        density_kg_m3=rho0_kg_m3,
        directions=k_hat,
        wavenumber_rad_m=wavenumber_rad_m,
        wave_type="P",
        displacement_m=1e-8,
        phase0=0.0,
    )
    print("\nplane_wave_bulk_mass_weights output:")
    print(f"  weights shape : {weights_kg.shape}")  # (N, Ndir, Nf)
    assert weights_kg.shape == (n_sources, ndir, nf)

    # ------------------------------------------------------------------
    # 3. G-free witness displacement at W=2 witness positions.
    #    Must share (k_hat, wavenumber_rad_m, phase0) with the source above.
    # ------------------------------------------------------------------
    witness_positions_m = np.array(
        [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
        ]
    )
    witness_displacement_m = plane_wave_displacement(
        witness_positions_m,
        k_hat,
        wavenumber_rad_m,
        wave_type="P",
        amplitude_m=1e-8,
        phase0=0.0,
    )
    print("\nplane_wave_displacement output:")
    print(f"  witness shape : {witness_displacement_m.shape}")  # (W, Ndir, Nf, 3)
    assert witness_displacement_m.shape == (len(witness_positions_m), ndir, nf, 3)
    assert not witness_displacement_m.flags.writeable

    # ------------------------------------------------------------------
    # 4. Gravitoelastic cross tensor c_ij (G^1)
    # ------------------------------------------------------------------
    targets_m = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])  # (M=2, 3)
    direction_weights = np.array([0.5, 0.5])  # uniform, (Ndir,)

    c_ij = coherent_gravitoelastic_tensor(
        source_positions_m=positions_m,
        weights_kg=weights_kg,
        targets_m=targets_m,
        witness_displacement_m=witness_displacement_m,
        direction_weights=direction_weights,
    )
    # Output shape: (M, W, Nf, 3, 3)
    print("\ncoherent_gravitoelastic_tensor output:")
    print(f"  c_ij shape : {c_ij.shape}")
    print(f"  c_ij dtype : {c_ij.dtype}")
    print(f"  c_ij[target=0, witness=0, freq=0] =\n{c_ij[0, 0, 0]}")

    assert c_ij.shape == (len(targets_m), len(witness_positions_m), nf, 3, 3)
    assert c_ij.dtype == np.complex128
    assert not c_ij.flags.writeable

    # ------------------------------------------------------------------
    # 5. Mirror-pair acceleration CSD (G^2)
    # ------------------------------------------------------------------
    targets_a_m = np.array([[3.0, 0.0, 0.0]])
    targets_b_m = np.array([[-3.0, 0.0, 0.0]])

    csd = coherent_acceleration_csd(
        source_positions_m=positions_m,
        weights_kg=weights_kg,
        targets_a_m=targets_a_m,
        targets_b_m=targets_b_m,
        direction_weights=direction_weights,
    )
    # Output shape: (Ma, Mb, Nf, 3, 3)
    print("\ncoherent_acceleration_csd output:")
    print(f"  csd shape : {csd.shape}")
    print(f"  csd dtype : {csd.dtype}")
    print(f"  csd[0, 0, freq=0] =\n{csd[0, 0, 0]}")

    assert csd.shape == (len(targets_a_m), len(targets_b_m), nf, 3, 3)
    assert csd.dtype == np.complex128
    assert not csd.flags.writeable

    print("\nAll assertions passed.")
    print("Example 12 completed successfully.")


if __name__ == "__main__":
    main()
