"""Example 07: Coherent gravitational response from a plane-wave displacement.

Demonstrates the coherent-response workflow:

1. Build a set of discrete mass elements (``WeightedPointSource``).
2. Apply plane-wave modulation via ``plane_wave_mass_weights`` to obtain
   complex coherent mass weights of shape ``(N, Ndir, Nf)``.
3. Compute coherent gravitational acceleration at a target with
   ``coherent_point_acceleration``, yielding shape ``(M, Ndir, Nf, 3)``.
4. Compute the directional power average with ``directional_power_average``.

Note on complex outputs
-----------------------
Complex outputs represent linear complex amplitudes or transfer-function-like
responses, not directly real-valued time-domain gravitational fields.
"""

import numpy as np

from gragra import G_SI, WeightedPointSource, coherent_point_acceleration
from gragra.coherent import directional_power_average, plane_wave_mass_weights
from gragra.waves import normalize_directions


def main() -> None:
    print(f"G_SI = {G_SI:.5e}  m^3 kg^-1 s^-2")

    # ------------------------------------------------------------------
    # 1. Mass model: small cluster of N=6 point masses in the z=0 plane
    # ------------------------------------------------------------------
    n_sources = 6
    angles = np.linspace(0, 2 * np.pi, n_sources, endpoint=False)
    radius_m = 0.5  # m
    positions_m = np.column_stack(
        [
            radius_m * np.cos(angles),
            radius_m * np.sin(angles),
            np.zeros(n_sources),
        ]
    )
    # 10 kg each
    base_weights_kg = np.full(n_sources, 10.0)

    source = WeightedPointSource(
        positions_m=positions_m,
        weights_kg=base_weights_kg,
    )
    print(
        f"\nSource: {n_sources} point masses, "
        f"positions shape: {source.positions_m.shape}"
    )

    # ------------------------------------------------------------------
    # 2. Wave parameters
    #    Ndir = 3 propagation directions, Nf = 2 wavenumbers
    # ------------------------------------------------------------------
    raw_dirs = np.array(
        [
            [1.0, 0.0, 0.0],  # propagating in +x
            [0.0, 1.0, 0.0],  # propagating in +y
            [1.0, 1.0, 0.0],  # propagating in x-y diagonal
        ]
    )
    directions = normalize_directions(raw_dirs)  # (Ndir, 3), unit normalised

    # wavenumbers in rad/m — 10 Hz and 50 Hz equivalent
    wavenumbers = np.array([2 * np.pi * 10.0, 2 * np.pi * 50.0])

    ndir = len(directions)
    nf = len(wavenumbers)

    # ------------------------------------------------------------------
    # 3. Coherent mass weights: shape (N, Ndir, Nf)
    # ------------------------------------------------------------------
    weights = plane_wave_mass_weights(
        base_weights_kg=source.weights_kg,
        positions_m=source.positions_m,
        directions=directions,
        wavenumber_rad_m=wavenumbers,
        phase0=0.0,
        sign=1,
    )

    print("\nplane_wave_mass_weights output:")
    print(f"  weights shape : {weights.shape}")  # expected (N, Ndir, Nf)
    print(f"  weights dtype : {weights.dtype}")  # complex128
    print(f"  weights writeable: {weights.flags.writeable}")  # False

    assert weights.shape == (n_sources, ndir, nf), (
        f"Unexpected weights shape: {weights.shape}, "
        f"expected ({n_sources}, {ndir}, {nf})"
    )
    assert weights.dtype == np.complex128
    assert not weights.flags.writeable

    # ------------------------------------------------------------------
    # 4. Coherent gravitational acceleration at a single target
    #    target at 5 m above the cluster along z
    # ------------------------------------------------------------------
    targets_m = np.array([[0.0, 0.0, 5.0]])  # (M=1, 3)

    acc = coherent_point_acceleration(
        source_positions_m=source.positions_m,
        weights_kg=weights,  # (N, Ndir, Nf)
        targets_m=targets_m,
        softening_m=1e-6,
    )

    # Output shape: (M, Ndir, Nf, 3)
    m_targets = len(targets_m)
    print("\ncoherent_point_acceleration output:")
    print(f"  coherent acceleration shape : {acc.shape}")
    print(f"  coherent acceleration dtype : {acc.dtype}")  # complex128
    print(f"  acc writeable               : {acc.flags.writeable}")

    assert acc.shape == (m_targets, ndir, nf, 3), (
        f"Unexpected acc shape: {acc.shape}, expected ({m_targets}, {ndir}, {nf}, 3)"
    )
    assert acc.dtype == np.complex128
    assert not acc.flags.writeable

    # Print a sample value (direction 0, frequency 0, all 3 components)
    print(f"\n  acc[target=0, dir=0, freq=0] = {acc[0, 0, 0]}")

    # ------------------------------------------------------------------
    # 5. Directional power average
    #    Sum over direction axis (axis=1) with uniform weights
    # ------------------------------------------------------------------
    direction_weights = np.ones(ndir) / ndir  # uniform average

    # acc has shape (M, Ndir, Nf, 3); direction_axis=1
    power_avg = directional_power_average(
        response=acc,
        direction_weights=direction_weights,
        direction_axis=1,
    )

    # Output shape: (M, Nf, 3) — direction axis summed out
    print("\ndirectional_power_average output:")
    print(f"  power_avg shape : {power_avg.shape}")  # (M, Nf, 3)
    print(f"  power_avg dtype : {power_avg.dtype}")  # float64
    print(f"  power_avg writeable: {power_avg.flags.writeable}")
    print(f"  power_avg[0, 0]   = {power_avg[0, 0]}")  # (3,) power per component

    assert power_avg.shape == (m_targets, nf, 3), (
        f"Unexpected power_avg shape: {power_avg.shape}"
    )
    assert power_avg.dtype == np.float64
    assert not power_avg.flags.writeable

    # Power must be non-negative (it's sum of |...|^2)
    assert (power_avg >= 0.0).all(), "Power average must be non-negative"

    print("\nAll assertions passed.")
    print("Example 07 completed successfully.")


if __name__ == "__main__":
    main()
