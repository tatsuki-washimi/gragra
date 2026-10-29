"""Tests for wave direction utilities."""

import numpy as np
import pytest

from gragra.waves.directions import (
    normalize_directions,
    orthonormal_basis_from_direction,
)


def test_normalize_directions_basics():
    # 1D input (3,) promoted to (1, 3)
    d1 = np.array([3.0, 0.0, 4.0])
    res1 = normalize_directions(d1)
    assert res1.shape == (1, 3)
    np.testing.assert_allclose(res1, [[0.6, 0.0, 0.8]])
    assert res1.dtype == np.float64
    assert not res1.flags.writeable

    # 2D input (Ndir, 3)
    d2 = np.array([[3.0, 0.0, 4.0], [0.0, -2.0, 0.0]])
    res2 = normalize_directions(d2)
    assert res2.shape == (2, 3)
    np.testing.assert_allclose(res2, [[0.6, 0.0, 0.8], [0.0, -1.0, 0.0]])


def test_normalize_directions_invalid():
    # Zero vector
    with pytest.raises(ValueError, match="zero vectors"):
        normalize_directions([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="zero vectors"):
        normalize_directions([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    # Non-finite
    with pytest.raises(ValueError, match="finite numbers"):
        normalize_directions([1.0, np.nan, 0.0])
    with pytest.raises(ValueError, match="finite numbers"):
        normalize_directions([1.0, np.inf, 0.0])

    # Boolean
    with pytest.raises(TypeError, match="numeric"):
        normalize_directions([True, False, True])
    with pytest.raises(TypeError, match="numeric"):
        normalize_directions(True)

    # Invalid shape
    with pytest.raises(ValueError, match="1D direction must have shape"):
        normalize_directions([1.0, 2.0])
    with pytest.raises(ValueError, match="2D directions must have shape"):
        normalize_directions([[1.0, 2.0, 3.0, 4.0]])
    with pytest.raises(ValueError, match="must be 1D or 2D"):
        normalize_directions(np.ones((2, 2, 3)))


def test_orthonormal_basis_basics():
    # Basic construction
    k_hat = np.array([0.0, 0.0, 1.0])
    e1, e2 = orthonormal_basis_from_direction(k_hat)

    # Check shapes and dtype
    assert e1.shape == (3,)
    assert e2.shape == (3,)
    assert e1.dtype == np.float64
    assert e2.dtype == np.float64
    assert not e1.flags.writeable
    assert not e2.flags.writeable

    # Check orthogonality
    assert np.abs(np.dot(e1, k_hat)) < 1e-15
    assert np.abs(np.dot(e2, k_hat)) < 1e-15
    assert np.abs(np.dot(e1, e2)) < 1e-15

    # Check normalization
    assert np.abs(np.linalg.norm(e1) - 1.0) < 1e-15
    assert np.abs(np.linalg.norm(e2) - 1.0) < 1e-15

    # Check right-handedness: k x e1 = e2
    np.testing.assert_allclose(np.cross(k_hat, e1), e2, atol=1e-15)


def test_orthonormal_basis_with_reference():
    k_hat = np.array([0.0, 0.0, 1.0])
    # reference_axis is orthogonal
    ref = np.array([1.0, 0.0, 0.0])
    e1, e2 = orthonormal_basis_from_direction(k_hat, reference_axis=ref)
    np.testing.assert_allclose(e1, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(e2, [0.0, 1.0, 0.0])

    # reference_axis is oblique
    ref_oblique = np.array([1.0, 0.0, 2.0])  # has k_hat component
    e1, e2 = orthonormal_basis_from_direction(k_hat, reference_axis=ref_oblique)
    # e1 should be normalized [1, 0, 0]
    np.testing.assert_allclose(e1, [1.0, 0.0, 0.0])


def test_orthonormal_basis_parallel_or_none_ref():
    # If reference_axis is parallel to k_hat
    k_hat = np.array([0.0, 0.0, 1.0])
    ref_parallel = np.array([0.0, 0.0, 1.5])
    e1, e2 = orthonormal_basis_from_direction(k_hat, reference_axis=ref_parallel)
    # Should fall back to a stable basis (e.g. using standard axis)
    assert np.abs(np.dot(e1, k_hat)) < 1e-15
    assert np.abs(np.dot(e2, k_hat)) < 1e-15
    assert np.abs(np.dot(e1, e2)) < 1e-15

    # Test deterministic fallback for all coordinate axes
    for axis in [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]:
        e1, e2 = orthonormal_basis_from_direction(axis)
        assert np.abs(np.dot(e1, axis)) < 1e-15
        assert np.abs(np.dot(e2, axis)) < 1e-15
        assert np.abs(np.dot(e1, e2)) < 1e-15
        np.testing.assert_allclose(np.cross(axis, e1), e2, atol=1e-15)


def test_orthonormal_basis_invalid():
    # Zero vector k_hat
    with pytest.raises(ValueError, match="zero vector"):
        orthonormal_basis_from_direction([0.0, 0.0, 0.0])

    # Non-finite k_hat
    with pytest.raises(ValueError, match="finite numbers"):
        orthonormal_basis_from_direction([1.0, np.nan, 0.0])

    # Bad shape k_hat
    with pytest.raises(ValueError, match="single 3D vector"):
        orthonormal_basis_from_direction([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    # Invalid reference axis shape
    with pytest.raises(ValueError, match="reference_axis must have shape"):
        orthonormal_basis_from_direction([0.0, 0.0, 1.0], reference_axis=[1.0, 0.0])
