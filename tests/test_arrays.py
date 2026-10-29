"""Test the internal array validation and normalization helper functions."""

import numpy as np
import pytest

from gragra._arrays import as_weights, as_xyz_table, as_xyz_vector


def test_as_xyz_vector_valid():
    """Test as_xyz_vector with valid inputs."""
    val = [1.0, 2.0, 3.0]
    res = as_xyz_vector("pos", val)
    assert isinstance(res, np.ndarray)
    assert res.shape == (3,)
    assert res.dtype == np.float64
    assert np.allclose(res, [1.0, 2.0, 3.0])
    # Immutability
    assert not res.flags.writeable
    with pytest.raises(ValueError):
        res[0] = 9.9


def test_as_xyz_vector_invalid_shape():
    """Test as_xyz_vector with incorrect shapes."""
    with pytest.raises(ValueError, match="shape"):
        as_xyz_vector("pos", [1.0, 2.0])
    with pytest.raises(ValueError, match="shape"):
        as_xyz_vector("pos", [[1.0, 2.0, 3.0]])


def test_as_xyz_vector_non_finite():
    """Test as_xyz_vector with non-finite values."""
    with pytest.raises(ValueError, match="finite"):
        as_xyz_vector("pos", [1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="finite"):
        as_xyz_vector("pos", [1.0, np.inf, 3.0])


def test_as_xyz_table_valid():
    """Test as_xyz_table with valid inputs."""
    val = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    res = as_xyz_table("pos_table", val)
    assert isinstance(res, np.ndarray)
    assert res.shape == (2, 3)
    assert res.dtype == np.float64
    assert np.allclose(res, val)
    assert not res.flags.writeable


def test_as_xyz_table_rejects_vector():
    """Test as_xyz_table rejects single vectors of shape (3,)."""
    with pytest.raises(ValueError, match="1D vector"):
        as_xyz_table("pos_table", [1.0, 2.0, 3.0])


def test_as_xyz_table_invalid_shape():
    """Test as_xyz_table with invalid shape."""
    with pytest.raises(ValueError, match="shape"):
        as_xyz_table("pos_table", [[1.0, 2.0], [3.0, 4.0]])


def test_as_xyz_table_non_finite():
    """Test as_xyz_table with non-finite values."""
    with pytest.raises(ValueError, match="finite"):
        as_xyz_table("pos_table", [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]])


def test_as_weights_real():
    """Test as_weights with real values."""
    val = [0.1, 0.2, 0.3]
    res = as_weights("weights", val)
    assert res.shape == (3,)
    assert res.dtype == np.float64
    assert np.allclose(res, val)
    assert not res.flags.writeable


def test_as_weights_complex():
    """Test as_weights with complex values."""
    val = [0.1 + 0.2j, 0.3, 0.4 - 0.5j]
    res = as_weights("weights", val)
    assert res.shape == (3,)
    assert res.dtype == np.complex128
    assert np.allclose(res, val)
    assert not res.flags.writeable


def test_as_weights_invalid_shape():
    """Test as_weights with non-1D shape."""
    with pytest.raises(ValueError, match="1D array"):
        as_weights("weights", [[1.0, 2.0]])


def test_as_weights_non_finite():
    """Test as_weights with non-finite values."""
    with pytest.raises(ValueError, match="finite"):
        as_weights("weights", [1.0, np.nan])
