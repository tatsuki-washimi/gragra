"""Tests for parametric gravity kernel contractions."""

import numpy as np
import pytest

from gragra.kernels.inverse_square import (
    acceleration_contract,
    gradient_contract,
    potential_contract,
)
from gragra.kernels.parametric import (
    acceleration_contract_parametric,
    gradient_contract_parametric,
    potential_contract_parametric,
)


def test_parametric_scalar_weights_match():
    """Test that 1D weights produce identical results to non-parametric."""
    sources = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    weights = np.array([1.5, 2.5, 3.5])
    targets = np.array([[0.5, 0.5, 0.0], [1.0, 1.0, 1.0]])

    for eps in [0.0, 0.1]:
        # Potential
        ref_phi = potential_contract(sources, weights, targets, softening_m=eps)
        test_phi = potential_contract_parametric(
            sources, weights, targets, softening_m=eps
        )
        assert test_phi.shape == (2,)
        np.testing.assert_allclose(test_phi, ref_phi, rtol=1e-12)
        assert test_phi.dtype == np.float64
        assert not test_phi.flags.writeable

        # Acceleration
        ref_acc = acceleration_contract(sources, weights, targets, softening_m=eps)
        test_acc = acceleration_contract_parametric(
            sources, weights, targets, softening_m=eps
        )
        assert test_acc.shape == (2, 3)
        np.testing.assert_allclose(test_acc, ref_acc, rtol=1e-12)
        assert test_acc.dtype == np.float64
        assert not test_acc.flags.writeable

        # Gradient
        ref_grad = gradient_contract(sources, weights, targets, softening_m=eps)
        test_grad = gradient_contract_parametric(
            sources, weights, targets, softening_m=eps
        )
        assert test_grad.shape == (2, 3, 3)
        np.testing.assert_allclose(test_grad, ref_grad, rtol=1e-12)
        assert test_grad.dtype == np.float64
        assert not test_grad.flags.writeable


def test_parametric_frequency_axis():
    """Test weights with a frequency axis: shape (N, Nf)."""
    sources = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # Nf = 4
    weights = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    targets = np.array([[0.5, 0.5, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    phi = potential_contract_parametric(sources, weights, targets, softening_m=0.1)
    assert phi.shape == (3, 4)
    assert phi.dtype == np.float64

    acc = acceleration_contract_parametric(sources, weights, targets, softening_m=0.1)
    assert acc.shape == (3, 4, 3)
    assert acc.dtype == np.float64

    grad = gradient_contract_parametric(sources, weights, targets, softening_m=0.1)
    assert grad.shape == (3, 4, 3, 3)
    assert grad.dtype == np.float64

    # Verify matching for a single slice
    for f in range(4):
        ref_phi = potential_contract(sources, weights[:, f], targets, softening_m=0.1)
        np.testing.assert_allclose(phi[:, f], ref_phi, rtol=1e-12)

        ref_acc = acceleration_contract(
            sources, weights[:, f], targets, softening_m=0.1
        )
        np.testing.assert_allclose(acc[:, f, :], ref_acc, rtol=1e-12)

        ref_grad = gradient_contract(sources, weights[:, f], targets, softening_m=0.1)
        np.testing.assert_allclose(grad[:, f, :, :], ref_grad, rtol=1e-12)


def test_parametric_direction_frequency_axis():
    """Test weights with direction and frequency axes: shape (N, Ndir, Nf)."""
    sources = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # Ndir = 2, Nf = 3
    weights = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    targets = np.array([[0.5, 0.5, 0.0], [1.0, 1.0, 1.0]])

    phi = potential_contract_parametric(sources, weights, targets, softening_m=0.1)
    assert phi.shape == (2, 2, 3)

    acc = acceleration_contract_parametric(sources, weights, targets, softening_m=0.1)
    assert acc.shape == (2, 2, 3, 3)

    grad = gradient_contract_parametric(sources, weights, targets, softening_m=0.1)
    assert grad.shape == (2, 2, 3, 3, 3)

    # Verify a single slice
    ref_phi = potential_contract(sources, weights[:, 1, 2], targets, softening_m=0.1)
    np.testing.assert_allclose(phi[:, 1, 2], ref_phi, rtol=1e-12)


def test_parametric_complex_weights():
    """Test complex weights yield complex128 outputs."""
    sources = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    weights = np.array([[1.0 + 1j, 2.0 - 2j], [3.0 + 3j, 4.0 - 4j]])
    targets = np.array([[0.5, 0.5, 0.0], [1.0, 1.0, 1.0]])

    phi = potential_contract_parametric(sources, weights, targets, softening_m=0.1)
    assert phi.shape == (2, 2)
    assert phi.dtype == np.complex128
    assert not phi.flags.writeable

    acc = acceleration_contract_parametric(sources, weights, targets, softening_m=0.1)
    assert acc.shape == (2, 2, 3)
    assert acc.dtype == np.complex128
    assert not acc.flags.writeable

    grad = gradient_contract_parametric(sources, weights, targets, softening_m=0.1)
    assert grad.shape == (2, 2, 3, 3)
    assert grad.dtype == np.complex128
    assert not grad.flags.writeable

    # Verify accuracy
    ref_phi = potential_contract(sources, weights[:, 1], targets, softening_m=0.1)
    np.testing.assert_allclose(phi[:, 1], ref_phi, rtol=1e-12)


def test_parametric_chunk_invariance():
    """Test that results are invariant to the chunk_size parameter."""
    sources = np.random.default_rng(42).random((20, 3))
    weights = np.random.default_rng(42).random((20, 2, 3))
    targets = np.random.default_rng(42).random((5, 3))

    for fn in [
        potential_contract_parametric,
        acceleration_contract_parametric,
        gradient_contract_parametric,
    ]:
        res_none = fn(sources, weights, targets, softening_m=0.05, chunk_size=None)
        for cs in [1, 3, 7, 20, 50]:
            res_chunk = fn(sources, weights, targets, softening_m=0.05, chunk_size=cs)
            np.testing.assert_allclose(res_chunk, res_none, rtol=1e-12)


def test_parametric_gradient_axes_last():
    """Test that tensor components for gradient are at the very end of axes."""
    sources = np.array([[0.0, 0.0, 0.0]])
    weights = np.ones((1, 2, 3, 4))
    targets = np.array([[1.0, 0.0, 0.0]])

    grad = gradient_contract_parametric(sources, weights, targets, softening_m=0.0)
    # Expected shape: (M, P1, P2, P3, 3, 3) where M=1, P=(2,3,4)
    # => (1, 2, 3, 4, 3, 3)
    assert grad.shape == (1, 2, 3, 4, 3, 3)


def test_parametric_collision_raises():
    """Test that collision with softening=0 raises ValueError with indices."""
    sources = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    weights = np.ones((2, 2))
    # target 1 collides with source 1
    targets = np.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])

    for fn in [
        potential_contract_parametric,
        acceleration_contract_parametric,
        gradient_contract_parametric,
    ]:
        with pytest.raises(ValueError) as exc_info:
            fn(sources, weights, targets, softening_m=0.0)
        assert "Collision detected" in str(exc_info.value)
        assert "target point 1" in str(exc_info.value)
        assert "source point 1" in str(exc_info.value)


def test_parametric_invalid_inputs():
    """Test various invalid inputs for parametric contractions."""
    sources = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    weights = np.ones((2, 3))
    targets = np.array([[0.5, 0.5, 0.0]])

    for fn in [
        potential_contract_parametric,
        acceleration_contract_parametric,
        gradient_contract_parametric,
    ]:
        # negative softening
        with pytest.raises(ValueError, match="softening_m must be non-negative"):
            fn(sources, weights, targets, softening_m=-0.1)

        # chunk_size <= 0
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            fn(sources, weights, targets, chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            fn(sources, weights, targets, chunk_size=-5)

        # weights first axis mismatch
        bad_weights = np.ones((3, 3))
        msg_mismatch = "Length of sources_xyz and weights first axis must match"
        with pytest.raises(ValueError, match=msg_mismatch):
            fn(sources, bad_weights, targets)

        # non-finite weights
        inf_weights = np.array([[1.0, 2.0], [3.0, np.inf]])
        msg_finite = "weights must contain only finite numbers"
        with pytest.raises(ValueError, match=msg_finite):
            fn(sources, inf_weights, targets)

        # 0D weights (at least 1D required)
        scalar_w = 1.0
        with pytest.raises(ValueError, match="weights must have at least 1 dimension"):
            fn(sources, scalar_w, targets)
