"""Tests for the displacement-transfer acceleration observable (Phase 14-d)."""

import numpy as np

from gragra import G_SI
from gragra.integrators import direct_point_dipole_sum
from gragra.observables.dipole import displacement_transfer_acceleration


def test_matches_g_si_times_dipole_sum():
    """displacement_transfer_acceleration == G_SI * direct_point_dipole_sum,
    for both real and complex displacement_m (numpy backend)."""
    rng = np.random.default_rng(0)
    n_sources = 5
    n_targets = 3
    sources_xyz = rng.uniform(-1, 1, (n_sources, 3))
    targets_xyz = rng.uniform(2, 3, (n_targets, 3))
    mass_kg = rng.uniform(0, 5, n_sources)

    for displacement_m in (
        rng.uniform(-1, 1, (n_sources, 3)),
        rng.uniform(-1, 1, (n_sources, 3)) + 1j * rng.uniform(-1, 1, (n_sources, 3)),
    ):
        expected = G_SI * direct_point_dipole_sum(
            mass_kg, displacement_m, sources_xyz, targets_xyz
        )
        actual = displacement_transfer_acceleration(
            sources_xyz, mass_kg, displacement_m, targets_xyz
        )
        np.testing.assert_allclose(actual, expected)


def test_sign_fixed_golden():
    """Source@origin, target(1,0,0), softening=0, mass=1, d=(1,0,0)
    -> G_SI * (-2, 0, 0) (dipole_contract golden times G_SI)."""
    sources_xyz = [[0.0, 0.0, 0.0]]
    mass_kg = [1.0]
    displacement_m = [[1.0, 0.0, 0.0]]
    targets_xyz = [[1.0, 0.0, 0.0]]

    out = displacement_transfer_acceleration(
        sources_xyz, mass_kg, displacement_m, targets_xyz, softening_m=0.0
    )

    np.testing.assert_allclose(out[0], G_SI * np.array([-2.0, 0.0, 0.0]))


def test_read_only_output():
    sources_xyz = [[0.0, 0.0, 0.0]]
    mass_kg = [1.0]
    displacement_m = [[1.0, 0.0, 0.0]]
    targets_xyz = [[1.0, 0.0, 0.0]]

    out = displacement_transfer_acceleration(
        sources_xyz, mass_kg, displacement_m, targets_xyz
    )

    assert out.flags.writeable is False
