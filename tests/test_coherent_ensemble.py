"""Tests for realization-axis ensemble cross statistics."""

import numpy as np
import pytest

from gragra.coherent.ensemble import ensemble_outer_cross_average


def test_ensemble_outer_cross_average_reduces_only_realization_axis():
    """The R axis is averaged while the corresponding N samples are retained."""
    a = np.array(
        [
            [[1.0 + 1.0j, 2.0, 0.0], [3.0, 0.0, 1.0j]],
            [[4.0, 0.0, 2.0], [0.0, 5.0 - 1.0j, 6.0]],
        ]
    )
    b = np.array(
        [
            [[2.0, 1.0j, 1.0], [1.0 - 1.0j, 2.0, 3.0]],
            [[0.0, 2.0, 1.0j], [3.0, 4.0 + 1.0j, 0.0]],
        ]
    )

    actual = ensemble_outer_cross_average(a, b)
    expected = np.stack(
        [
            np.mean(a[n, :, :, None] * np.conj(b[n, :, None, :]), axis=0)
            for n in range(2)
        ]
    )

    assert actual.shape == (2, 3, 3)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    ("a_shape", "b_shape"),
    [((1, 4, 3), (5, 4, 3)), ((2, 4, 3), (2, 5, 3))],
)
def test_ensemble_outer_cross_average_rejects_sample_or_realization_axis_mismatch(
    a_shape, b_shape
):
    with pytest.raises(ValueError):
        ensemble_outer_cross_average(np.ones(a_shape), np.ones(b_shape))


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 3, 1), (2, 3, 2)])
def test_ensemble_outer_cross_average_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError):
        ensemble_outer_cross_average(np.ones(shape), np.ones((2, 3, 3)))


def test_ensemble_outer_cross_average_rejects_empty_or_nonfinite_realizations():
    with pytest.raises(ValueError):
        ensemble_outer_cross_average(np.empty((2, 0, 3)), np.empty((2, 0, 3)))

    nonfinite = np.ones((1, 2, 3), dtype=np.complex128)
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        ensemble_outer_cross_average(nonfinite, np.ones((1, 2, 3)))


def test_ensemble_outer_cross_average_rejects_bool_inside_object_array():
    mixed = np.ones((1, 2, 3), dtype=object)
    mixed[0, 0, 0] = True

    with pytest.raises(TypeError):
        ensemble_outer_cross_average(mixed, np.ones((1, 2, 3)))


def test_ensemble_outer_cross_average_returns_readonly_complex128_for_empty_samples():
    actual = ensemble_outer_cross_average(
        np.empty((0, 2, 3), dtype=np.float32),
        np.empty((0, 2, 3), dtype=np.float64),
    )

    assert actual.shape == (0, 3, 3)
    assert actual.dtype == np.complex128
    assert not actual.flags.writeable
