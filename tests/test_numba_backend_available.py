"""Correctness tests for the Numba-accelerated backend.

These tests ensure the 'numba' backend is mathematically identical to
the default 'numpy' reference backend.
"""

import numpy as np
import pytest

# Skip all tests in this module if numba is not installed
pytest.importorskip("numba")

from gragra.constants import G_SI
from gragra.gravity_sources import PointMass, WeightedPointSource
from gragra.observables import (
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)
from gragra.observables.projection import point_projected_acceleration


def test_numpy_numba_equivalence_real():
    """Verify that numpy and numba backends return identical results."""
    source = WeightedPointSource(
        positions_m=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        weights_kg=[10.0, 20.0],
    )
    targets = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]

    # 1. Potential
    v_numpy = point_potential(source, targets, backend="numpy")
    v_numba = point_potential(source, targets, backend="numba")
    assert np.allclose(v_numpy, v_numba, rtol=1e-12, atol=1e-15)

    # 2. Acceleration
    a_numpy = point_acceleration(source, targets, backend="numpy")
    a_numba = point_acceleration(source, targets, backend="numba")
    assert np.allclose(a_numpy, a_numba, rtol=1e-12, atol=1e-15)

    # 3. Projected Acceleration
    direction = [1.0, 1.0, 0.0]
    ap_numpy = point_projected_acceleration(source, targets, direction, backend="numpy")
    ap_numba = point_projected_acceleration(source, targets, direction, backend="numba")
    assert np.allclose(ap_numpy, ap_numba, rtol=1e-12, atol=1e-15)

    # 4. Gravity Gradient
    g_numpy = point_gravity_gradient(source, targets, backend="numpy")
    g_numba = point_gravity_gradient(source, targets, backend="numba")
    assert np.allclose(g_numpy, g_numba, rtol=1e-12, atol=1e-15)


def test_numba_tensor_sign_convention():
    """Verify the sign convention of the gravity gradient tensor on numba backend.

    For a 1kg point mass at the origin, and target at (1, 0, 0):
    T = G_SI * diag(2, -1, -1).
    """
    source = PointMass(1.0, [0.0, 0.0, 0.0])
    target = [[1.0, 0.0, 0.0]]

    t_tensor = point_gravity_gradient(source, target, softening_m=0.0, backend="numba")

    assert t_tensor.shape == (1, 3, 3)
    expected = G_SI * np.diag([2.0, -1.0, -1.0])
    assert np.allclose(t_tensor[0], expected, rtol=1e-12, atol=1e-15)


def test_numpy_numba_equivalence_complex():
    """Verify that numpy and numba backends return identical complex results."""
    source = WeightedPointSource(
        positions_m=[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        weights_kg=[10.0 + 5.0j, -20.0 + 3.0j],
    )
    targets = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]

    v_numpy = point_potential(source, targets, backend="numpy")
    v_numba = point_potential(source, targets, backend="numba")
    assert v_numba.dtype == np.complex128
    assert np.allclose(v_numpy, v_numba, rtol=1e-12, atol=1e-15)

    a_numpy = point_acceleration(source, targets, backend="numpy")
    a_numba = point_acceleration(source, targets, backend="numba")
    assert a_numba.dtype == np.complex128
    assert np.allclose(a_numpy, a_numba, rtol=1e-12, atol=1e-15)

    g_numpy = point_gravity_gradient(source, targets, backend="numpy")
    g_numba = point_gravity_gradient(source, targets, backend="numba")
    assert g_numba.dtype == np.complex128
    assert np.allclose(g_numpy, g_numba, rtol=1e-12, atol=1e-15)


def test_numba_chunk_independence():
    """Verify that Numba backend ignores chunk_size but yields same results."""
    source = WeightedPointSource(
        positions_m=np.random.rand(10, 3),
        weights_kg=np.random.rand(10),
    )
    targets = np.random.rand(5, 3)

    v_chunk_none = point_potential(source, targets, chunk_size=None, backend="numba")
    v_chunk_small = point_potential(source, targets, chunk_size=2, backend="numba")
    assert np.allclose(v_chunk_none, v_chunk_small, rtol=1e-15)


def test_numba_collision_detection():
    """Verify ValueError is raised on target-source collision.

    Checks that the raised error message contains target index, source index,
    and mention of zero softening (case-insensitive, loosely matched).
    """
    source = WeightedPointSource(
        positions_m=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        weights_kg=[10.0, 20.0],
    )
    targets = [[1.0, 1.0, 1.0]]  # Collides with source index 1

    # Using match= to verify target index, source index, and zero softening collision
    with pytest.raises(
        ValueError,
        match=r"(?i)collision.*target.*0.*source.*1.*zero softening",
    ):
        point_potential(source, targets, softening_m=0.0, backend="numba")


def test_numba_invalid_inputs():
    """Verify standard input validation errors are raised."""
    source = PointMass(1.0, [0.0, 0.0, 0.0])
    target = [[1.0, 0.0, 0.0]]

    # softening_m < 0
    with pytest.raises(ValueError, match="softening_m must be non-negative"):
        point_potential(source, target, softening_m=-0.1, backend="numba")

    # chunk_size <= 0
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        point_potential(source, target, chunk_size=0, backend="numba")


def test_numba_read_only_returns():
    """Verify Numba return arrays are marked as read-only."""
    source = PointMass(1.0, [0.0, 0.0, 0.0])
    target = [[1.0, 0.0, 0.0]]

    v = point_potential(source, target, backend="numba")
    assert not v.flags.writeable

    a = point_acceleration(source, target, backend="numba")
    assert not a.flags.writeable

    g = point_gravity_gradient(source, target, backend="numba")
    assert not g.flags.writeable
