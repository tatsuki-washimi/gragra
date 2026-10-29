"""Tests for G-free kernel contractions."""

import numpy as np
import pytest

from gragra.constants import G_SI
from gragra.gravity_sources import WeightedPointSource
from gragra.kernels.inverse_square import (
    acceleration_contract,
    gradient_contract,
    potential_contract,
)
from gragra.observables import (
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)


def test_kernel_vs_observable_real():
    """Verify that (contract * G_SI) matches the observable output (real weights)."""
    sources_xyz = [[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]]
    weights = [1.5, -0.5]
    targets_xyz = [[1.0, 1.0, 1.0], [0.0, -1.0, 2.0]]

    # 1. Potential
    p_contract = potential_contract(sources_xyz, weights, targets_xyz)
    p_obs = point_potential(WeightedPointSource(sources_xyz, weights), targets_xyz)
    assert np.allclose(p_contract * G_SI, p_obs)

    # 2. Acceleration
    a_contract = acceleration_contract(sources_xyz, weights, targets_xyz)
    a_obs = point_acceleration(WeightedPointSource(sources_xyz, weights), targets_xyz)
    assert np.allclose(a_contract * G_SI, a_obs)

    # 3. Gradient
    g_contract = gradient_contract(sources_xyz, weights, targets_xyz)
    g_obs = point_gravity_gradient(
        WeightedPointSource(sources_xyz, weights), targets_xyz
    )
    assert np.allclose(g_contract * G_SI, g_obs)


def test_kernel_vs_observable_complex():
    """Verify that complex weight contractions also match observables."""
    sources_xyz = [[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]]
    weights = [1.5 + 2.0j, -0.5 - 1.0j]
    targets_xyz = [[1.0, 1.0, 1.0], [0.0, -1.0, 2.0]]

    p_contract = potential_contract(sources_xyz, weights, targets_xyz)
    p_obs = point_potential(WeightedPointSource(sources_xyz, weights), targets_xyz)
    assert p_contract.dtype == np.complex128
    assert p_obs.dtype == np.complex128
    assert np.allclose(p_contract * G_SI, p_obs)


def test_chunk_size_invariance():
    """Verify that varying the chunk_size does not affect the calculation results."""
    # Create a reasonably large number of sources to chunk
    np.random.seed(42)
    n_sources = 25
    sources_xyz = np.random.uniform(-10, 10, (n_sources, 3))
    weights = np.random.uniform(-5, 5, n_sources)
    targets_xyz = [[1.0, 0.0, -1.0], [0.0, 2.0, 3.0]]

    # Compute with no chunking (defaults to N)
    p_ref = potential_contract(sources_xyz, weights, targets_xyz)
    a_ref = acceleration_contract(sources_xyz, weights, targets_xyz)
    g_ref = gradient_contract(sources_xyz, weights, targets_xyz)

    # Compute with different chunk sizes
    for chunk_size in [1, 5, 10, 100]:
        p_chunk = potential_contract(
            sources_xyz, weights, targets_xyz, chunk_size=chunk_size
        )
        a_chunk = acceleration_contract(
            sources_xyz, weights, targets_xyz, chunk_size=chunk_size
        )
        g_chunk = gradient_contract(
            sources_xyz, weights, targets_xyz, chunk_size=chunk_size
        )

        assert np.allclose(p_ref, p_chunk)
        assert np.allclose(a_ref, a_chunk)
        assert np.allclose(g_ref, g_chunk)


def test_invalid_chunk_size_raises():
    """Verify that chunk_size <= 0 raises ValueError."""
    sources_xyz = [[0.0, 0.0, 0.0]]
    weights = [1.0]
    targets_xyz = [[1.0, 0.0, 0.0]]

    with pytest.raises(ValueError, match="chunk_size"):
        potential_contract(sources_xyz, weights, targets_xyz, chunk_size=0)

    with pytest.raises(ValueError, match="chunk_size"):
        potential_contract(sources_xyz, weights, targets_xyz, chunk_size=-5)
