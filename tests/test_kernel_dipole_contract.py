"""Tests for the G-free vector-weight (dipole/total) contraction (Phase 14-a)."""

import numpy as np
import pytest

from gragra.gravity_sources import WeightedPointSource
from gragra.integrators import direct_point_dipole_sum
from gragra.integrators.point_direct import (
    direct_point_dipole_sum as _direct_point_dipole_sum,
)
from gragra.kernels import dipole_contract
from gragra.kernels.inverse_square import acceleration_contract


def test_dipole_sign_convention():
    """Source@origin, target(1,0,0), softening=0, mass=1, d=(1,0,0) -> (-2,0,0).

    This is -diag(2,-1,-1)@(1,0,0), matching the K_T golden of
    test_observables.py::test_tensor_sign_convention with the sign flip
    documented in kernel_contract.md §2.1/§2.2.
    """
    sources_xyz = [[0.0, 0.0, 0.0]]
    mass_kg = [1.0]
    displacement_m = [[1.0, 0.0, 0.0]]
    targets_xyz = [[1.0, 0.0, 0.0]]

    out = dipole_contract(
        sources_xyz, mass_kg, displacement_m, targets_xyz, softening_m=0.0
    )

    assert out.shape == (1, 3)
    assert np.allclose(out[0], [-2.0, 0.0, 0.0])


def test_dipole_finite_difference_oracle():
    """Source-shifted central difference of acceleration_contract converges
    directly to dipole_contract (no extra sign flip) as eps -> 0."""
    source_xyz = np.array([0.0, 0.0, 0.0])
    mass = 3.0
    d_hat = np.array([0.0, 1.0, 0.0])
    targets_xyz = [[1.0, 0.0, 0.0]]

    dipole_out = dipole_contract(
        [source_xyz], [mass], [d_hat], targets_xyz, softening_m=0.0
    )[0]

    eps = 1e-6
    a_plus = acceleration_contract([source_xyz + eps * d_hat], [mass], targets_xyz)[0]
    a_minus = acceleration_contract([source_xyz - eps * d_hat], [mass], targets_xyz)[0]
    central_diff = (a_plus - a_minus) / (2 * eps)

    assert np.allclose(central_diff, dipole_out, rtol=1e-5)


def test_dipole_softening_finite_difference():
    """With softening > 0, the dipole output still matches the source-shifted
    finite-difference oracle (K_T is no longer trace-free, but the vector
    contraction remains well-defined and finite)."""
    source_xyz = np.array([0.2, -0.1, 0.05])
    mass = 2.0
    d_hat = np.array([1.0, 0.0, 0.0])
    targets_xyz = [[1.5, 0.3, -0.2]]
    softening_m = 0.5

    dipole_out = dipole_contract(
        [source_xyz], [mass], [d_hat], targets_xyz, softening_m=softening_m
    )[0]
    assert np.all(np.isfinite(dipole_out))

    eps = 1e-6
    a_plus = acceleration_contract(
        [source_xyz + eps * d_hat], [mass], targets_xyz, softening_m=softening_m
    )[0]
    a_minus = acceleration_contract(
        [source_xyz - eps * d_hat], [mass], targets_xyz, softening_m=softening_m
    )[0]
    central_diff = (a_plus - a_minus) / (2 * eps)

    assert np.allclose(central_diff, dipole_out, rtol=1e-5)


def test_dipole_complex_displacement_dtype():
    """Complex displacement_m promotes the output to complex128; real stays float64."""
    sources_xyz = [[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]]
    mass_kg = [1.5, 0.5]
    targets_xyz = [[1.0, 1.0, 1.0], [0.0, -1.0, 2.0]]

    real_disp = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    out_real = dipole_contract(sources_xyz, mass_kg, real_disp, targets_xyz)
    assert out_real.dtype == np.float64

    complex_disp = [[1.0 + 2.0j, 0.0, 0.0], [0.0, 1.0 - 1.0j, 0.0]]
    out_complex = dipole_contract(sources_xyz, mass_kg, complex_disp, targets_xyz)
    assert out_complex.dtype == np.complex128


def test_dipole_chunk_size_invariance():
    """Varying chunk_size does not affect the result."""
    np.random.seed(42)
    n_sources = 25
    sources_xyz = np.random.uniform(-10, 10, (n_sources, 3))
    mass_kg = np.random.uniform(0, 5, n_sources)
    displacement_m = np.random.uniform(-1, 1, (n_sources, 3))
    targets_xyz = [[1.0, 0.0, -1.0], [0.0, 2.0, 3.0]]

    ref = dipole_contract(sources_xyz, mass_kg, displacement_m, targets_xyz)

    for chunk_size in [1, 5, 10, 100]:
        chunked = dipole_contract(
            sources_xyz, mass_kg, displacement_m, targets_xyz, chunk_size=chunk_size
        )
        assert np.allclose(ref, chunked)


def test_dipole_shape_mismatch_raises():
    sources_xyz = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    targets_xyz = [[1.0, 0.0, 0.0]]

    with pytest.raises(ValueError, match="must match"):
        dipole_contract(
            sources_xyz, [1.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], targets_xyz
        )

    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        dipole_contract(sources_xyz, [1.0, 1.0], [[1.0, 0.0], [0.0, 1.0]], targets_xyz)


def test_dipole_mass_kg_validation():
    sources_xyz = [[0.0, 0.0, 0.0]]
    displacement_m = [[1.0, 0.0, 0.0]]
    targets_xyz = [[1.0, 0.0, 0.0]]

    with pytest.raises(TypeError, match="real-valued"):
        dipole_contract(sources_xyz, [1.0 + 1.0j], displacement_m, targets_xyz)

    with pytest.raises(ValueError, match="non-negative"):
        dipole_contract(sources_xyz, [-1.0], displacement_m, targets_xyz)

    with pytest.raises(TypeError, match="bool"):
        dipole_contract(sources_xyz, [True], displacement_m, targets_xyz)

    with pytest.raises(ValueError, match="finite"):
        dipole_contract(sources_xyz, [np.inf], displacement_m, targets_xyz)


def test_dipole_softening_and_collision_errors():
    sources_xyz = [[0.0, 0.0, 0.0]]
    mass_kg = [1.0]
    displacement_m = [[1.0, 0.0, 0.0]]

    with pytest.raises(ValueError, match="softening_m"):
        dipole_contract(
            sources_xyz, mass_kg, displacement_m, [[1.0, 0.0, 0.0]], softening_m=-0.1
        )

    with pytest.raises(ValueError, match="Collision"):
        dipole_contract(
            sources_xyz, mass_kg, displacement_m, [[0.0, 0.0, 0.0]], softening_m=0.0
        )


def test_dipole_read_only_output():
    out = dipole_contract(
        [[0.0, 0.0, 0.0]], [1.0], [[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]
    )
    assert out.flags.writeable is False


def test_weighted_point_source_rejects_vector_weights():
    """Regression: displacement_m (N,3) must not be accepted as weights_kg."""
    with pytest.raises(ValueError, match="1D"):
        WeightedPointSource(positions_m=[[0.0, 0.0, 0.0]], weights_kg=[[1.0, 0.0, 0.0]])


def test_direct_point_dipole_sum_numpy_delegation():
    sources_xyz = [[0.0, 0.0, 0.0], [1.0, 2.0, -1.0]]
    mass_kg = [1.0, 2.0]
    displacement_m = [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]]
    targets_xyz = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.5]]

    direct = direct_point_dipole_sum(mass_kg, displacement_m, sources_xyz, targets_xyz)
    kernel = dipole_contract(sources_xyz, mass_kg, displacement_m, targets_xyz)
    assert np.allclose(direct, kernel)
    assert _direct_point_dipole_sum is direct_point_dipole_sum


def test_direct_point_dipole_sum_backend_guards():
    """Unknown backends are rejected.

    The old expectation that backend='cupy' raises ValueError was removed in
    Phase C-1, which added dipole_contract to the cupy backend; the cupy
    missing/device-unavailable guards are pinned in test_cupy_backend_missing.py
    and the parity path in test_cupy_dipole.py.
    """
    sources_xyz = [[0.0, 0.0, 0.0]]
    mass_kg = [1.0]
    displacement_m = [[1.0, 0.0, 0.0]]
    targets_xyz = [[1.0, 0.0, 0.0]]

    with pytest.raises(ValueError, match="Unsupported backend"):
        direct_point_dipole_sum(
            mass_kg, displacement_m, sources_xyz, targets_xyz, backend="bogus"
        )


def test_dipole_subpackage_exports():
    from gragra.integrators import direct_point_dipole_sum as integrators_export
    from gragra.kernels import dipole_contract as kernels_export

    assert callable(kernels_export)
    assert callable(integrators_export)


def test_dipole_not_promoted_to_top_level():
    import gragra

    assert not hasattr(gragra, "dipole_contract")
    assert not hasattr(gragra, "direct_point_dipole_sum")
