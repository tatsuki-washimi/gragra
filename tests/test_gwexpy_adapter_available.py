"""Tests for gwexpy adapter when gwexpy is available."""

import gwexpy
import numpy as np
import pytest

from gragra.adapters import (
    to_gwexpy_scalar_field,
    to_gwexpy_tensor_field,
    to_gwexpy_vector_field,
)


def test_scalar_field_flat_mode():
    """Test converting 1D data to ScalarField in flat point mode."""
    n_pts = 5
    data = np.arange(n_pts, dtype=np.float64)
    positions = np.zeros((n_pts, 3), dtype=np.float64)

    sf = to_gwexpy_scalar_field(data, positions_m=positions, unit="m", t_s=1.23)

    assert isinstance(sf, gwexpy.fields.ScalarField)
    # The internal shape must be (1, n_pts, 1, 1)
    assert sf.value.shape == (1, n_pts, 1, 1)
    np.testing.assert_allclose(sf.value[0, :, 0, 0], data)

    # Validate metadata
    assert sf.unit == "m"
    np.testing.assert_allclose(sf.axes[0].index.value, [1.23])
    assert sf.axes[0].name == "time"
    assert sf.axes[1].name == "point"
    assert sf.axes[2].name == "dummy_y"
    assert sf.axes[3].name == "dummy_z"
    assert sf.axis0_domain == "time"
    assert sf.space_domains["point"] == "real"
    assert sf.space_domains["dummy_y"] == "real"
    assert sf.space_domains["dummy_z"] == "real"


def test_vector_field_flat_mode():
    """Test converting 2D data to VectorField in flat point mode."""
    n_pts = 5
    data = np.arange(n_pts * 3, dtype=np.float64).reshape((n_pts, 3))

    vf = to_gwexpy_vector_field(data, unit="m/s", t_s=0.0)

    assert isinstance(vf, gwexpy.fields.VectorField)
    arr = vf.to_array()
    assert arr.shape == (1, n_pts, 1, 1, 3)
    np.testing.assert_allclose(arr[0, :, 0, 0, :], data)

    # Component check
    for i, comp in enumerate(["x", "y", "z"]):
        sf_comp = vf[comp]
        assert sf_comp.unit == "m/s"
        assert sf_comp.value.shape == (1, n_pts, 1, 1)
        np.testing.assert_allclose(sf_comp.value[0, :, 0, 0], data[:, i])
        assert sf_comp.axes[0].name == "time"
        assert sf_comp.axes[1].name == "point"
        assert sf_comp.axis0_domain == "time"
        assert sf_comp.space_domains["point"] == "real"


def test_tensor_field_flat_mode():
    """Test converting 3D data to TensorField in flat point mode."""
    n_pts = 5
    data = np.arange(n_pts * 9, dtype=np.float64).reshape((n_pts, 3, 3))

    tf = to_gwexpy_tensor_field(data, unit="1/s^2", t_s=0.5)

    assert isinstance(tf, gwexpy.fields.TensorField)
    arr = tf.to_array(order="last")
    assert arr.shape == (1, n_pts, 1, 1, 3, 3)
    np.testing.assert_allclose(arr[0, :, 0, 0, :, :], data)

    # Component check
    for i in range(3):
        for j in range(3):
            sf_comp = tf[(i, j)]
            assert sf_comp.unit == "1/s^2"
            assert sf_comp.value.shape == (1, n_pts, 1, 1)
            np.testing.assert_allclose(sf_comp.value[0, :, 0, 0], data[:, i, j])


def test_grid_mode():
    """Test grid mode reshaping with grid_shape."""
    grid_shape = (2, 3, 4)
    n_pts = 2 * 3 * 4  # 24
    data_scalar = np.arange(n_pts, dtype=np.float64)
    data_vector = np.arange(n_pts * 3, dtype=np.float64).reshape((n_pts, 3))
    data_tensor = np.arange(n_pts * 9, dtype=np.float64).reshape((n_pts, 3, 3))

    # Scalar
    sf = to_gwexpy_scalar_field(data_scalar, grid_shape=grid_shape, t_s=1.0)
    assert sf.value.shape == (1, 2, 3, 4)
    assert sf.axes[0].name == "time"
    assert sf.axes[1].name == "x"
    assert sf.axes[2].name == "y"
    assert sf.axes[3].name == "z"
    assert sf.axis0_domain == "time"
    assert sf.space_domains["x"] == "real"
    assert sf.space_domains["y"] == "real"
    assert sf.space_domains["z"] == "real"
    np.testing.assert_allclose(sf.value[0], data_scalar.reshape(grid_shape))

    # Vector
    vf = to_gwexpy_vector_field(data_vector, grid_shape=grid_shape, t_s=1.0)
    assert vf.to_array().shape == (1, 2, 3, 4, 3)
    np.testing.assert_allclose(vf.to_array()[0], data_vector.reshape(grid_shape + (3,)))

    # Tensor
    tf = to_gwexpy_tensor_field(data_tensor, grid_shape=grid_shape, t_s=1.0)
    assert tf.to_array(order="last").shape == (1, 2, 3, 4, 3, 3)
    np.testing.assert_allclose(
        tf.to_array(order="last")[0], data_tensor.reshape(grid_shape + (3, 3))
    )


def test_grid_shape_product_mismatch():
    """Test ValueError is raised when grid_shape product doesn't match M."""
    data = np.zeros(8)
    with pytest.raises(
        ValueError, match="grid_shape product 27 must match data points 8"
    ):
        to_gwexpy_scalar_field(data, grid_shape=(3, 3, 3))


def test_complex_dtype_preservation():
    """Test that complex input dtypes are preserved in the fields."""
    n_pts = 4
    data_complex = np.ones(n_pts, dtype=np.complex128) * (1.0 + 2.0j)
    sf = to_gwexpy_scalar_field(data_complex)
    assert sf.value.dtype == np.complex128
    np.testing.assert_allclose(sf.value[0, :, 0, 0], data_complex)


def test_invalid_grid_shape():
    """Test validation of grid_shape elements."""
    data = np.zeros(8)

    # Not a tuple
    with pytest.raises(TypeError, match="grid_shape must be a tuple"):
        to_gwexpy_scalar_field(data, grid_shape=[2, 2, 2])  # type: ignore

    # Wrong length
    with pytest.raises(ValueError, match="grid_shape must be a 3-tuple"):
        to_gwexpy_scalar_field(data, grid_shape=(2, 4))  # type: ignore

    # Non-integer / float
    with pytest.raises(
        TypeError, match="grid_shape elements must be positive integers"
    ):
        to_gwexpy_scalar_field(data, grid_shape=(2.0, 2, 2))  # type: ignore

    # Boolean (which is subclass of Integral)
    with pytest.raises(
        TypeError, match="grid_shape elements must be positive integers"
    ):
        to_gwexpy_scalar_field(data, grid_shape=(2, True, 2))  # type: ignore

    # Non-positive integer
    with pytest.raises(
        ValueError, match="grid_shape elements must be positive integers"
    ):
        to_gwexpy_scalar_field(data, grid_shape=(2, 0, 2))


def test_invalid_ts():
    """Test validation of time coordinate t_s."""
    data = np.zeros(5)

    # Non-numeric
    with pytest.raises(TypeError, match="t_s must be a finite real number"):
        to_gwexpy_scalar_field(data, t_s="1.0")  # type: ignore

    # Boolean
    with pytest.raises(TypeError, match="t_s must be a finite real number"):
        to_gwexpy_scalar_field(data, t_s=True)  # type: ignore

    # Complex (np.complex128 or python complex)
    with pytest.raises(TypeError, match="t_s must be a finite real number"):
        to_gwexpy_scalar_field(data, t_s=np.complex128(1 + 2j))  # type: ignore
    with pytest.raises(TypeError, match="t_s must be a finite real number"):
        to_gwexpy_scalar_field(data, t_s=1 + 2j)  # type: ignore

    # Non-finite (inf)
    with pytest.raises(ValueError, match="t_s must be a finite real number"):
        to_gwexpy_scalar_field(data, t_s=np.inf)


def test_invalid_positions():
    """Test validation of positions_m."""
    data = np.zeros(5)

    # Wrong shape
    with pytest.raises(ValueError, match="positions_m must have shape"):
        to_gwexpy_scalar_field(data, positions_m=np.zeros((5, 2)))

    # Complex positions
    with pytest.raises(TypeError, match="positions_m must be real-valued"):
        to_gwexpy_scalar_field(data, positions_m=np.zeros((5, 3), dtype=np.complex128))

    # Inf in positions
    inf_pos = np.zeros((5, 3))
    inf_pos[2, 1] = np.inf
    with pytest.raises(ValueError, match="positions_m must contain only finite"):
        to_gwexpy_scalar_field(data, positions_m=inf_pos)


def test_invalid_data_shape():
    """Test validation of data shapes for scalar, vector, and tensor fields."""
    # Scalar expects 1D
    with pytest.raises(ValueError, match="Scalar data must have shape"):
        to_gwexpy_scalar_field(np.zeros((5, 2)))

    # Vector expects (M, 3)
    with pytest.raises(ValueError, match="Vector data must have shape"):
        to_gwexpy_vector_field(np.zeros((5, 4)))
    with pytest.raises(ValueError, match="Vector data must have shape"):
        to_gwexpy_vector_field(np.zeros(5))

    # Tensor expects (M, 3, 3)
    with pytest.raises(ValueError, match="Tensor data must have shape"):
        to_gwexpy_tensor_field(np.zeros((5, 3, 2)))
    with pytest.raises(ValueError, match="Tensor data must have shape"):
        to_gwexpy_tensor_field(np.zeros(5))
