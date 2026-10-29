"""Regression coverage for frame transforms used by precision checks."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gragra._evaluation_setup import CoordinatePrecisionError, DiscreteSource
from gragra.geometry._frame import Frame
from gragra.geometry._sphere import Sphere
from gragra.gravity_sources._models import point_sum
from gragra.targets._high_points import TargetPoints


def test_same_physical_point_with_accepted_rounded_basis_stops_validation() -> None:
    root = Frame("root")
    cosine = np.sqrt(0.5)
    basis = np.round(
        [
            [cosine, -cosine, 0.0],
            [cosine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        13,
    )
    source_frame = Frame("source", basis=basis, parent=root)
    source_position_m = np.array([1.0, 0.0, 0.0])
    target_position_m = basis @ source_position_m
    geometry = Sphere(np.zeros(3) * u.m, 1.0 * u.m, source_frame)
    source = DiscreteSource(
        geometry,
        1000.0 * u.kg / u.m**3,
        source_frame,
        source_position_m.reshape(1, 3),
        np.ones(1),
        {"spacing_m": 1.0, "max_points": 1, "grid_points": 1},
    )
    targets = TargetPoints(
        target_position_m.reshape(1, 3) * u.m,
        ("same-physical-point",),
        root,
    )

    with pytest.raises(CoordinatePrecisionError) as caught:
        point_sum(source).at(targets, observables=("potential",)).validate()

    assert caught.value.target_id == "same-physical-point"
    assert caught.value.source_index == 0
    assert caught.value.reason in {"collision", "coordinate_precision"}
    assert caught.value.observables == ("potential",)
