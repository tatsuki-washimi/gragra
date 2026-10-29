"""Correctness tests for the C++ / OpenMP accelerated backend.

These tests ensure the 'cpp' backend is mathematically identical to
the default 'numpy' reference backend and the 'numba' backend when available.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

# Skip all tests in this module if C++ backend is not compiled
pytest.importorskip("gragra._cpp_backend")

from gragra.constants import G_SI
from gragra.gravity_sources import PointMass, WeightedPointSource
from gragra.observables import (
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)
from gragra.observables.projection import point_projected_acceleration


def test_numpy_cpp_equivalence_real():
    """Verify that numpy and cpp backends return identical real results."""
    source = WeightedPointSource(
        positions_m=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        weights_kg=[10.0, 20.0],
    )
    targets = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]

    # 1. Potential
    v_numpy = point_potential(source, targets, backend="numpy")
    v_cpp = point_potential(source, targets, backend="cpp")
    assert np.allclose(v_numpy, v_cpp, rtol=1e-12, atol=1e-15)

    # 2. Acceleration
    a_numpy = point_acceleration(source, targets, backend="numpy")
    a_cpp = point_acceleration(source, targets, backend="cpp")
    assert np.allclose(a_numpy, a_cpp, rtol=1e-12, atol=1e-15)

    # 3. Projected Acceleration
    direction = [1.0, 1.0, 0.0]
    ap_numpy = point_projected_acceleration(source, targets, direction, backend="numpy")
    ap_cpp = point_projected_acceleration(source, targets, direction, backend="cpp")
    assert np.allclose(ap_numpy, ap_cpp, rtol=1e-12, atol=1e-15)

    # 4. Gravity Gradient
    g_numpy = point_gravity_gradient(source, targets, backend="numpy")
    g_cpp = point_gravity_gradient(source, targets, backend="cpp")
    assert np.allclose(g_numpy, g_cpp, rtol=1e-12, atol=1e-15)


def test_cpp_tensor_sign_convention():
    """Verify the sign convention of the gravity gradient tensor on cpp backend.

    For a 1kg point mass at the origin, and target at (1, 0, 0):
    T = G_SI * diag(2, -1, -1).
    """
    source = PointMass(1.0, [0.0, 0.0, 0.0])
    target = [[1.0, 0.0, 0.0]]

    t_tensor = point_gravity_gradient(source, target, softening_m=0.0, backend="cpp")

    assert t_tensor.shape == (1, 3, 3)
    expected = G_SI * np.diag([2.0, -1.0, -1.0])
    assert np.allclose(t_tensor[0], expected, rtol=1e-12, atol=1e-15)


def test_numpy_cpp_equivalence_complex():
    """Verify that numpy and cpp backends return identical complex results."""
    source = WeightedPointSource(
        positions_m=[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        weights_kg=[10.0 + 5.0j, -20.0 + 3.0j],
    )
    targets = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]

    v_numpy = point_potential(source, targets, backend="numpy")
    v_cpp = point_potential(source, targets, backend="cpp")
    assert v_cpp.dtype == np.complex128
    assert np.allclose(v_numpy, v_cpp, rtol=1e-12, atol=1e-15)

    a_numpy = point_acceleration(source, targets, backend="numpy")
    a_cpp = point_acceleration(source, targets, backend="cpp")
    assert a_cpp.dtype == np.complex128
    assert np.allclose(a_numpy, a_cpp, rtol=1e-12, atol=1e-15)

    g_numpy = point_gravity_gradient(source, targets, backend="numpy")
    g_cpp = point_gravity_gradient(source, targets, backend="cpp")
    assert g_cpp.dtype == np.complex128
    assert np.allclose(g_numpy, g_cpp, rtol=1e-12, atol=1e-15)


def test_numpy_numba_cpp_three_way_equivalence():
    """Verify three-way equivalence numpy == numba == cpp when numba is available."""
    try:
        import numba  # noqa: F401

        has_numba = True
    except ImportError:
        has_numba = False

    if not has_numba:
        pytest.skip("Numba not available for 3-way match")

    source = WeightedPointSource(
        positions_m=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        weights_kg=[10.0 + 3.0j, 20.0 - 5.0j],
    )
    targets = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]

    v_numpy = point_potential(source, targets, backend="numpy")
    v_numba = point_potential(source, targets, backend="numba")
    v_cpp = point_potential(source, targets, backend="cpp")

    assert np.allclose(v_numpy, v_numba, rtol=1e-12, atol=1e-15)
    assert np.allclose(v_numpy, v_cpp, rtol=1e-12, atol=1e-15)


def test_cpp_chunk_independence():
    """Verify that C++ backend ignores chunk_size but yields same results."""
    source = WeightedPointSource(
        positions_m=np.random.rand(10, 3),
        weights_kg=np.random.rand(10),
    )
    targets = np.random.rand(5, 3)

    v_chunk_none = point_potential(source, targets, chunk_size=None, backend="cpp")
    v_chunk_small = point_potential(source, targets, chunk_size=2, backend="cpp")
    assert np.allclose(v_chunk_none, v_chunk_small, rtol=1e-15)


def test_cpp_collision_detection():
    """Verify ValueError is raised on target-source collision on C++ backend."""
    source = WeightedPointSource(
        positions_m=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        weights_kg=[10.0, 20.0],
    )
    targets = [[1.0, 1.0, 1.0]]  # Collides with source index 1

    with pytest.raises(
        ValueError,
        match=r"(?i)collision.*target.*0.*source.*1.*zero softening",
    ):
        point_potential(source, targets, softening_m=0.0, backend="cpp")


def test_cpp_invalid_inputs():
    """Verify standard input validation errors are raised."""
    source = PointMass(1.0, [0.0, 0.0, 0.0])
    target = [[1.0, 0.0, 0.0]]

    # softening_m < 0
    with pytest.raises(ValueError, match="softening_m must be non-negative"):
        point_potential(source, target, softening_m=-0.1, backend="cpp")

    # chunk_size <= 0
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        point_potential(source, target, chunk_size=0, backend="cpp")


def test_cpp_read_only_returns():
    """Verify C++ return arrays are marked as read-only."""
    source = PointMass(1.0, [0.0, 0.0, 0.0])
    target = [[1.0, 0.0, 0.0]]

    v = point_potential(source, target, backend="cpp")
    assert not v.flags.writeable

    a = point_acceleration(source, target, backend="cpp")
    assert not a.flags.writeable

    g = point_gravity_gradient(source, target, backend="cpp")
    assert not g.flags.writeable


def test_cpp_openmp_determinism():
    """Verify OpenMP determinism under different number of threads via subprocess."""
    script = """
import numpy as np
from gragra.gravity_sources import WeightedPointSource
from gragra.observables import point_potential

np.random.seed(42)
sources_xyz = np.random.rand(200, 3)
weights = np.random.rand(200) + 1j * np.random.rand(200)
targets_xyz = np.random.rand(64, 3)

source = WeightedPointSource(sources_xyz, weights)
v_cpp = point_potential(source, targets_xyz, backend="cpp")
print(repr(v_cpp.tolist()))
"""
    # Run with OMP_NUM_THREADS=2
    res2_a = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "OMP_NUM_THREADS": "2"},
        capture_output=True,
        text=True,
        check=True,
    )
    res2_b = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "OMP_NUM_THREADS": "2"},
        capture_output=True,
        text=True,
        check=True,
    )
    # Run with OMP_NUM_THREADS=4
    res4 = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "OMP_NUM_THREADS": "4"},
        capture_output=True,
        text=True,
        check=True,
    )

    # Verify identical output across identical settings
    assert res2_a.stdout == res2_b.stdout
    # Verify identical output across thread configurations (determinism)
    assert res2_a.stdout == res4.stdout


def test_cpp_empty_inputs():
    """Verify that C++ backend handles N=0 or M=0 gracefully."""
    # N=0 case
    source_empty = WeightedPointSource(positions_m=np.zeros((0, 3)), weights_kg=[])
    targets = [[1.0, 2.0, 3.0]]
    v_empty_n = point_potential(source_empty, targets, backend="cpp")
    assert v_empty_n.shape == (1,)
    assert np.all(v_empty_n == 0.0)

    a_empty_n = point_acceleration(source_empty, targets, backend="cpp")
    assert a_empty_n.shape == (1, 3)
    assert np.all(a_empty_n == 0.0)

    g_empty_n = point_gravity_gradient(source_empty, targets, backend="cpp")
    assert g_empty_n.shape == (1, 3, 3)
    assert np.all(g_empty_n == 0.0)

    # M=0 case
    source = WeightedPointSource(positions_m=[[0.0, 0.0, 0.0]], weights_kg=[10.0])
    targets_empty = np.zeros((0, 3))
    v_empty_m = point_potential(source, targets_empty, backend="cpp")
    assert v_empty_m.shape == (0,)

    a_empty_m = point_acceleration(source, targets_empty, backend="cpp")
    assert a_empty_m.shape == (0, 3)

    g_empty_m = point_gravity_gradient(source, targets_empty, backend="cpp")
    assert g_empty_m.shape == (0, 3, 3)


def test_cpp_direct_validation():
    """Verify that gragra._cpp_backend checks inputs and raises ValueError."""
    import gragra._cpp_backend as b

    sources = np.zeros((2, 3))
    weights = np.zeros(2)
    weights_c = np.zeros(2, dtype=np.complex128)
    targets = np.zeros((1, 3))
    collision = np.full(1, -1, dtype=np.int64)

    # Validate all 6 contract functions
    # (real and complex for potential, acceleration, gradient)
    functions = [
        (b.potential_contract_real, weights),
        (b.potential_contract_complex, weights_c),
        (b.acceleration_contract_real, weights),
        (b.acceleration_contract_complex, weights_c),
        (b.gradient_contract_real, weights),
        (b.gradient_contract_complex, weights_c),
    ]

    for func, w in functions:
        # 1. Invalid sources shape (N, 2 instead of N, 3)
        with pytest.raises(ValueError, match="sources_xyz must have shape"):
            func(np.zeros((2, 2)), w, targets, 0.0, collision)

        # 2. Invalid targets shape (M, 4 instead of M, 3)
        with pytest.raises(ValueError, match="targets_xyz must have shape"):
            func(sources, w, np.zeros((1, 4)), 0.0, collision)

        # 3. Invalid weights length (too short)
        with pytest.raises(ValueError, match="weights length must match"):
            func(sources, np.zeros(1, dtype=w.dtype), targets, 0.0, collision)

        # 4. Invalid weights length (too long)
        with pytest.raises(ValueError, match="weights length must match"):
            func(sources, np.zeros(3, dtype=w.dtype), targets, 0.0, collision)

        # 5. Invalid collision_source_index length (too short)
        with pytest.raises(
            ValueError, match="collision_source_index length must match"
        ):
            func(sources, w, targets, 0.0, np.full(0, -1, dtype=np.int64))

        # 6. Invalid collision_source_index length (too long)
        with pytest.raises(
            ValueError, match="collision_source_index length must match"
        ):
            func(sources, w, targets, 0.0, np.full(2, -1, dtype=np.int64))

        # 7. Invalid softening (eps2 < 0)
        with pytest.raises(ValueError, match="eps2 must be non-negative"):
            func(sources, w, targets, -1.0, collision)
