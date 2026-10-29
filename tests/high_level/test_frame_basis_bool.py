"""Regression coverage: ``Frame(basis=...)`` must reject bool input.

``np.asarray(basis)`` on a raw Python list silently upcasts a mixed
``bool``/``float`` list to ``float64`` (numpy type promotion), and a
plain all-``bool`` ndarray/Quantity has a ``dtype.kind == "b"`` that is
only detectable *before* the ``dtype=np.float64`` conversion discards it.
Both cases must be rejected with ``TypeError`` before any float64
conversion happens, matching the existing ``_check_no_bool`` guard used
elsewhere in the package (e.g. ``gragra._arrays.as_index_table``,
``gragra.analytic.rayleigh_literature._reject_non_real_dtype``).
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gragra.geometry._frame import Frame

_ROTATION_BASIS = np.array(
    [
        [np.sqrt(0.5), -np.sqrt(0.5), 0.0],
        [np.sqrt(0.5), np.sqrt(0.5), 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def test_frame_rejects_bool_mixed_nested_list_basis() -> None:
    basis = [
        [True, False, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    with pytest.raises(TypeError, match="basis"):
        Frame("child", basis=basis, parent=Frame("root"))


def test_frame_rejects_all_bool_ndarray_basis() -> None:
    basis = np.array(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
        ]
    )

    with pytest.raises(TypeError, match="basis"):
        Frame("child", basis=basis, parent=Frame("root"))


def test_frame_rejects_numpy_bool_element_in_list_basis() -> None:
    basis = [
        [np.bool_(True), 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    with pytest.raises(TypeError, match="basis"):
        Frame("child", basis=basis, parent=Frame("root"))


def test_frame_rejects_bool_dimensionless_quantity_basis() -> None:
    basis = u.Quantity(
        np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
            ]
        ),
        u.dimensionless_unscaled,
    )

    with pytest.raises(TypeError, match="basis"):
        Frame("child", basis=basis, parent=Frame("root"))


def test_frame_cyclic_basis_keeps_array_conversion_value_error() -> None:
    basis: list[object] = []
    basis.append(basis)

    with pytest.raises(ValueError):
        Frame("child", basis=basis, parent=Frame("root"))


def test_frame_still_accepts_valid_float_basis() -> None:
    root = Frame("root")
    child = Frame("child", basis=_ROTATION_BASIS, parent=root)

    np.testing.assert_allclose(child.basis, _ROTATION_BASIS)


def test_frame_still_accepts_default_identity_basis() -> None:
    root = Frame("root")

    np.testing.assert_allclose(root.basis, np.eye(3))
