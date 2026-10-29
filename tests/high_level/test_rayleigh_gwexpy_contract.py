"""GWexpy matrix boundary used by Rayleigh evaluation results."""

import numpy as np
from astropy import units as u

from gragra._rayleigh_evaluation_setup import _native_matrices


def _arrays(target_count: int, frequency_count: int):
    values = np.arange(frequency_count * target_count * 9, dtype=np.float64).reshape(
        frequency_count, target_count, 3, 3
    )
    gradient = values + 1j * (values + 0.125)
    return {
        "potential": np.arange(target_count * frequency_count).reshape(
            frequency_count, target_count
        )
        + 1j,
        "acceleration": np.arange(frequency_count * target_count * 3).reshape(
            frequency_count, target_count, 3
        )
        + 1j,
        "gradient": gradient,
    }


def test_native_matrices_support_irregular_and_single_bin_axes() -> None:
    target_ids = ("p1", "p2", "p3", "p4")
    for frequencies in ((0.75, 1.375, 5.25), (3.125,)):
        matrices = _native_matrices(
            _arrays(len(target_ids), len(frequencies)), target_ids, frequencies
        )
        expected_shapes = {
            "potential": (4, 1, len(frequencies)),
            "acceleration": (4, 3, len(frequencies)),
            "gradient": (4, 9, len(frequencies)),
        }
        expected_units = {
            "potential": u.m**2 / u.s**2,
            "acceleration": u.m / u.s**2,
            "gradient": u.s**-2,
        }
        expected_columns = {
            "potential": ("potential",),
            "acceleration": ("x", "y", "z"),
            "gradient": ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"),
        }
        for observable, matrix in matrices.items():
            assert matrix.shape == expected_shapes[observable]
            assert np.asarray(matrix).dtype == np.complex128
            np.testing.assert_array_equal(
                matrix.frequencies.to_value(u.Hz), frequencies
            )
            assert tuple(matrix.rows.keys()) == target_ids
            assert tuple(matrix.cols.keys()) == expected_columns[observable]
            assert all(
                matrix.meta[row, column].unit == expected_units[observable]
                for row in range(matrix.shape[0])
                for column in range(matrix.shape[1])
            )


def test_native_gradient_matrix_copy_and_partial_label_selection() -> None:
    target_ids = ("p1", "p2", "p3", "p4")
    frequencies = (0.75, 1.375, 5.25)
    gradient = _native_matrices(
        _arrays(len(target_ids), len(frequencies)), target_ids, frequencies
    )["gradient"]

    copied = gradient.copy()
    before = np.asarray(gradient).copy()
    np.asarray(copied)[0, 0, 0] += 1
    np.testing.assert_array_equal(np.asarray(gradient), before)

    partial = gradient.loc[["p3"], ["xy", "yz"]]
    assert partial.shape == (1, 2, 3)
    assert tuple(partial.rows.keys()) == ("p3",)
    assert tuple(partial.cols.keys()) == ("xy", "yz")
    np.testing.assert_array_equal(partial.frequencies.to_value(u.Hz), frequencies)
