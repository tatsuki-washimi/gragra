"""Regression coverage for bool input at the Frame origin API boundary."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gragra.geometry._frame import Frame


def test_frame_rejects_bool_length_quantity_origin_before_float_conversion() -> None:
    origin = np.array([False, False, False]) * u.m
    assert origin.dtype.kind == "b"

    with pytest.raises(TypeError, match="origin.*bool"):
        Frame("child", origin=origin, parent=Frame("root"))


def test_frame_accepts_numeric_zero_and_converts_real_origin_units() -> None:
    root = Frame("root")
    zero = Frame("zero", origin=np.zeros(3) * u.m, parent=root)
    converted = Frame("converted", origin=[100, 200, 300] * u.cm, parent=root)

    np.testing.assert_array_equal(zero.origin_m, np.zeros(3))
    np.testing.assert_allclose(converted.origin_m, [1.0, 2.0, 3.0], rtol=0, atol=0)


@pytest.mark.parametrize(
    "basis",
    (
        np.eye(3, dtype=np.int64),
        np.eye(3, dtype=np.float64) * u.dimensionless_unscaled,
    ),
)
def test_frame_accepts_numeric_zero_one_identity_basis(basis: object) -> None:
    child = Frame("child", basis=basis, parent=Frame("root"))

    np.testing.assert_array_equal(child.basis, np.eye(3))
