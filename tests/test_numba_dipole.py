"""Correctness tests for the Numba-accelerated dipole/total contraction
(Phase 14-c).

These tests ensure the 'numba' backend for the vector-weight (dipole/total)
contraction is mathematically identical to the default 'numpy' reference
backend (Phase 14-a), mirroring test_numba_backend_available.py's structure
for the scalar-weight contractions.
"""

import numpy as np
import pytest

# Skip all tests in this module if numba is not installed
pytest.importorskip("numba")

from gragra.integrators import direct_point_dipole_sum
from gragra.kernels.inverse_square import acceleration_contract, dipole_contract


def test_numba_dipole_sign_convention():
    """Source@origin, target(1,0,0), softening=0, mass=1, d=(1,0,0) -> (-2,0,0).

    Same golden as test_kernel_dipole_contract.py::test_dipole_sign_convention.
    """
    sources_xyz = [[0.0, 0.0, 0.0]]
    mass_kg = [1.0]
    displacement_m = [[1.0, 0.0, 0.0]]
    targets_xyz = [[1.0, 0.0, 0.0]]

    out = direct_point_dipole_sum(
        mass_kg, displacement_m, sources_xyz, targets_xyz, backend="numba"
    )

    assert out.shape == (1, 3)
    assert np.allclose(out[0], [-2.0, 0.0, 0.0])


def test_numba_numpy_dipole_equivalence_real():
    """Verify numpy and numba dipole_contract agree on random real inputs."""
    np.random.seed(42)
    n_sources = 25
    sources_xyz = np.random.uniform(-10, 10, (n_sources, 3))
    mass_kg = np.random.uniform(0, 5, n_sources)
    displacement_m = np.random.uniform(-1, 1, (n_sources, 3))
    targets_xyz = [[1.0, 0.0, -1.0], [0.0, 2.0, 3.0]]

    ref = dipole_contract(sources_xyz, mass_kg, displacement_m, targets_xyz)
    numba_out = direct_point_dipole_sum(
        mass_kg, displacement_m, sources_xyz, targets_xyz, backend="numba"
    )
    assert np.allclose(ref, numba_out, rtol=1e-12, atol=1e-15)


def test_numba_numpy_dipole_equivalence_complex():
    """Verify numpy and numba dipole_contract agree on complex displacement."""
    np.random.seed(7)
    n_sources = 15
    sources_xyz = np.random.uniform(-10, 10, (n_sources, 3))
    mass_kg = np.random.uniform(0, 5, n_sources)
    displacement_m = np.random.uniform(-1, 1, (n_sources, 3)) + 1j * np.random.uniform(
        -1, 1, (n_sources, 3)
    )
    targets_xyz = [[1.0, 0.0, -1.0], [0.0, 2.0, 3.0]]

    ref = dipole_contract(sources_xyz, mass_kg, displacement_m, targets_xyz)
    numba_out = direct_point_dipole_sum(
        mass_kg, displacement_m, sources_xyz, targets_xyz, backend="numba"
    )
    assert numba_out.dtype == np.complex128
    assert np.allclose(ref, numba_out, rtol=1e-12, atol=1e-15)


def test_numba_dipole_finite_difference_oracle():
    """Source-shifted central difference of acceleration_contract converges
    directly to the numba dipole_contract (no extra sign flip) as eps -> 0.
    """
    source_xyz = np.array([0.0, 0.0, 0.0])
    mass = 3.0
    d_hat = np.array([0.0, 1.0, 0.0])
    targets_xyz = [[1.0, 0.0, 0.0]]

    dipole_out = direct_point_dipole_sum(
        [mass], [d_hat], [source_xyz], targets_xyz, backend="numba"
    )[0]

    eps = 1e-6
    a_plus = acceleration_contract([source_xyz + eps * d_hat], [mass], targets_xyz)[0]
    a_minus = acceleration_contract([source_xyz - eps * d_hat], [mass], targets_xyz)[0]
    central_diff = (a_plus - a_minus) / (2 * eps)

    assert np.allclose(central_diff, dipole_out, rtol=1e-5)


def test_numba_dipole_chunk_size_ignored():
    """Numba backend ignores chunk_size (compiled CPU backends do not chunk
    the source axis; see backend_strategy.md), but must still accept it.
    """
    np.random.seed(1)
    n_sources = 10
    sources_xyz = np.random.uniform(-5, 5, (n_sources, 3))
    mass_kg = np.random.uniform(0, 5, n_sources)
    displacement_m = np.random.uniform(-1, 1, (n_sources, 3))
    targets_xyz = np.random.uniform(-5, 5, (4, 3))

    out_none = direct_point_dipole_sum(
        mass_kg,
        displacement_m,
        sources_xyz,
        targets_xyz,
        chunk_size=None,
        backend="numba",
    )
    out_small = direct_point_dipole_sum(
        mass_kg,
        displacement_m,
        sources_xyz,
        targets_xyz,
        chunk_size=1,
        backend="numba",
    )
    assert np.allclose(out_none, out_small, rtol=1e-15)


def test_numba_dipole_collision_detection():
    """Verify ValueError is raised on target-source collision, softening=0."""
    sources_xyz = [[0.0, 0.0, 0.0]]
    mass_kg = [1.0]
    displacement_m = [[1.0, 0.0, 0.0]]
    targets_xyz = [[0.0, 0.0, 0.0]]

    with pytest.raises(
        ValueError,
        match=r"(?i)collision.*target.*0.*source.*0.*zero softening",
    ):
        direct_point_dipole_sum(
            mass_kg,
            displacement_m,
            sources_xyz,
            targets_xyz,
            softening_m=0.0,
            backend="numba",
        )


def test_numba_dipole_invalid_inputs():
    """Verify mass_kg/displacement_m validation errors propagate through the
    numba backend (validation happens in _validate_dipole_inputs, shared
    with the numpy backend).
    """
    sources_xyz = [[0.0, 0.0, 0.0]]
    displacement_m = [[1.0, 0.0, 0.0]]
    targets_xyz = [[1.0, 0.0, 0.0]]

    with pytest.raises(TypeError, match="real-valued"):
        direct_point_dipole_sum(
            [1.0 + 1.0j], displacement_m, sources_xyz, targets_xyz, backend="numba"
        )

    with pytest.raises(ValueError, match="non-negative"):
        direct_point_dipole_sum(
            [-1.0], displacement_m, sources_xyz, targets_xyz, backend="numba"
        )

    with pytest.raises(TypeError, match="bool"):
        direct_point_dipole_sum(
            [True], displacement_m, sources_xyz, targets_xyz, backend="numba"
        )


def test_numba_dipole_read_only_output():
    out = direct_point_dipole_sum(
        [1.0],
        [[1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0]],
        backend="numba",
    )
    assert out.flags.writeable is False
