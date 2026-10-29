"""Correctness tests for the C++ / OpenMP accelerated dipole/total
contraction (Phase 14-c).

These tests ensure the 'cpp' backend for the vector-weight (dipole/total)
contraction is mathematically identical to the default 'numpy' reference
backend (Phase 14-a) and the 'numba' backend when available, mirroring
test_cpp_backend_available.py's structure for the scalar-weight
contractions.
"""

import numpy as np
import pytest

# Skip all tests in this module if C++ backend is not compiled
pytest.importorskip("gragra._cpp_backend")

from gragra.integrators import direct_point_dipole_sum
from gragra.kernels.inverse_square import acceleration_contract, dipole_contract


def test_numpy_cpp_dipole_equivalence_real():
    """Verify numpy and cpp dipole_contract agree on random real inputs."""
    np.random.seed(42)
    n_sources = 25
    sources_xyz = np.random.uniform(-10, 10, (n_sources, 3))
    mass_kg = np.random.uniform(0, 5, n_sources)
    displacement_m = np.random.uniform(-1, 1, (n_sources, 3))
    targets_xyz = [[1.0, 0.0, -1.0], [0.0, 2.0, 3.0]]

    ref = dipole_contract(sources_xyz, mass_kg, displacement_m, targets_xyz)
    cpp_out = direct_point_dipole_sum(
        mass_kg, displacement_m, sources_xyz, targets_xyz, backend="cpp"
    )
    assert np.allclose(ref, cpp_out, rtol=1e-12, atol=1e-15)


def test_numpy_cpp_dipole_equivalence_complex():
    """Verify numpy and cpp dipole_contract agree on complex displacement."""
    np.random.seed(7)
    n_sources = 15
    sources_xyz = np.random.uniform(-10, 10, (n_sources, 3))
    mass_kg = np.random.uniform(0, 5, n_sources)
    displacement_m = np.random.uniform(-1, 1, (n_sources, 3)) + 1j * np.random.uniform(
        -1, 1, (n_sources, 3)
    )
    targets_xyz = [[1.0, 0.0, -1.0], [0.0, 2.0, 3.0]]

    ref = dipole_contract(sources_xyz, mass_kg, displacement_m, targets_xyz)
    cpp_out = direct_point_dipole_sum(
        mass_kg, displacement_m, sources_xyz, targets_xyz, backend="cpp"
    )
    assert cpp_out.dtype == np.complex128
    assert np.allclose(ref, cpp_out, rtol=1e-12, atol=1e-15)


def test_numpy_numba_cpp_dipole_three_way_equivalence():
    """Verify three-way equivalence numpy == numba == cpp when numba is
    available.
    """
    try:
        import numba  # noqa: F401

        has_numba = True
    except ImportError:
        has_numba = False

    if not has_numba:
        pytest.skip("Numba not available for 3-way match")

    np.random.seed(3)
    n_sources = 12
    sources_xyz = np.random.uniform(-10, 10, (n_sources, 3))
    mass_kg = np.random.uniform(0, 5, n_sources)
    displacement_m = np.random.uniform(-1, 1, (n_sources, 3))
    targets_xyz = [[1.0, 0.0, -1.0], [0.0, 2.0, 3.0]]

    ref = dipole_contract(sources_xyz, mass_kg, displacement_m, targets_xyz)
    numba_out = direct_point_dipole_sum(
        mass_kg, displacement_m, sources_xyz, targets_xyz, backend="numba"
    )
    cpp_out = direct_point_dipole_sum(
        mass_kg, displacement_m, sources_xyz, targets_xyz, backend="cpp"
    )

    assert np.allclose(ref, numba_out, rtol=1e-12, atol=1e-15)
    assert np.allclose(ref, cpp_out, rtol=1e-12, atol=1e-15)


def test_cpp_dipole_sign_convention():
    """Source@origin, target(1,0,0), softening=0, mass=1, d=(1,0,0) -> (-2,0,0)."""
    out = direct_point_dipole_sum(
        [1.0], [[1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], backend="cpp"
    )
    assert out.shape == (1, 3)
    assert np.allclose(out[0], [-2.0, 0.0, 0.0])


def test_cpp_dipole_finite_difference_oracle():
    """Source-shifted central difference of acceleration_contract converges
    directly to the cpp dipole_contract (no extra sign flip) as eps -> 0.
    """
    source_xyz = np.array([0.0, 0.0, 0.0])
    mass = 3.0
    d_hat = np.array([0.0, 1.0, 0.0])
    targets_xyz = [[1.0, 0.0, 0.0]]

    dipole_out = direct_point_dipole_sum(
        [mass], [d_hat], [source_xyz], targets_xyz, backend="cpp"
    )[0]

    eps = 1e-6
    a_plus = acceleration_contract([source_xyz + eps * d_hat], [mass], targets_xyz)[0]
    a_minus = acceleration_contract([source_xyz - eps * d_hat], [mass], targets_xyz)[0]
    central_diff = (a_plus - a_minus) / (2 * eps)

    assert np.allclose(central_diff, dipole_out, rtol=1e-5)


def test_cpp_dipole_chunk_size_ignored():
    """C++ backend ignores chunk_size but must still accept/validate it."""
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
        backend="cpp",
    )
    out_small = direct_point_dipole_sum(
        mass_kg,
        displacement_m,
        sources_xyz,
        targets_xyz,
        chunk_size=1,
        backend="cpp",
    )
    assert np.allclose(out_none, out_small, rtol=1e-15)


def test_cpp_dipole_collision_detection():
    """Verify ValueError is raised on target-source collision, softening=0."""
    with pytest.raises(
        ValueError,
        match=r"(?i)collision.*target.*0.*source.*0.*zero softening",
    ):
        direct_point_dipole_sum(
            [1.0],
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            softening_m=0.0,
            backend="cpp",
        )


def test_cpp_dipole_invalid_inputs():
    """Verify mass_kg/displacement_m validation errors propagate through the
    cpp backend (validation happens in _validate_dipole_inputs, shared with
    the numpy backend).
    """
    sources_xyz = [[0.0, 0.0, 0.0]]
    displacement_m = [[1.0, 0.0, 0.0]]
    targets_xyz = [[1.0, 0.0, 0.0]]

    with pytest.raises(TypeError, match="real-valued"):
        direct_point_dipole_sum(
            [1.0 + 1.0j], displacement_m, sources_xyz, targets_xyz, backend="cpp"
        )

    with pytest.raises(ValueError, match="non-negative"):
        direct_point_dipole_sum(
            [-1.0], displacement_m, sources_xyz, targets_xyz, backend="cpp"
        )


def test_cpp_dipole_read_only_output():
    out = direct_point_dipole_sum(
        [1.0],
        [[1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0]],
        backend="cpp",
    )
    assert out.flags.writeable is False


def test_cpp_dipole_empty_inputs():
    """Verify that C++ backend handles N=0 or M=0 gracefully (zero-filled
    output via std::fill in the wrapper's early-return path).
    """
    # N=0 case
    out_empty_n = direct_point_dipole_sum(
        np.zeros((0,)),
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        [[1.0, 2.0, 3.0]],
        backend="cpp",
    )
    assert out_empty_n.shape == (1, 3)
    assert np.all(out_empty_n == 0.0)

    # M=0 case
    out_empty_m = direct_point_dipole_sum(
        [1.0],
        [[1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]],
        np.zeros((0, 3)),
        backend="cpp",
    )
    assert out_empty_m.shape == (0, 3)


def test_cpp_dipole_direct_validation():
    """Verify that gragra._cpp_backend checks dipole inputs and raises
    ValueError, mirroring test_cpp_direct_validation for the scalar
    contracts.
    """
    import gragra._cpp_backend as b

    sources = np.zeros((2, 3))
    mass = np.zeros(2)
    disp = np.zeros((2, 3))
    disp_c = np.zeros((2, 3), dtype=np.complex128)
    targets = np.zeros((1, 3))
    collision = np.full(1, -1, dtype=np.int64)

    functions = [
        (b.dipole_contract_real, disp),
        (b.dipole_contract_complex, disp_c),
    ]

    for func, d in functions:
        # 1. Invalid sources shape
        with pytest.raises(ValueError, match="sources_xyz must have shape"):
            func(np.zeros((2, 2)), mass, d, targets, 0.0, collision)

        # 2. Invalid targets shape
        with pytest.raises(ValueError, match="targets_xyz must have shape"):
            func(sources, mass, d, np.zeros((1, 4)), 0.0, collision)

        # 3. Invalid mass length (too short)
        with pytest.raises(ValueError, match="mass length must match"):
            func(sources, np.zeros(1), d, targets, 0.0, collision)

        # 4. Invalid disp shape (wrong N)
        with pytest.raises(ValueError, match="disp must have shape"):
            func(
                sources, mass, np.zeros((1, 3), dtype=d.dtype), targets, 0.0, collision
            )

        # 5. Invalid disp shape (wrong trailing axis)
        with pytest.raises(ValueError, match="disp must have shape"):
            func(
                sources, mass, np.zeros((2, 4), dtype=d.dtype), targets, 0.0, collision
            )

        # 6. Invalid collision_source_index length
        with pytest.raises(
            ValueError, match="collision_source_index length must match"
        ):
            func(sources, mass, d, targets, 0.0, np.full(0, -1, dtype=np.int64))

        # 7. Invalid softening (eps2 < 0)
        with pytest.raises(ValueError, match="eps2 must be non-negative"):
            func(sources, mass, d, targets, -1.0, collision)
