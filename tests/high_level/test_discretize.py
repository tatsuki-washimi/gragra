"""Tests for the immutable C5 uniform-sphere midpoint source."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gragra.geometry._frame import Frame
from gragra.geometry._sphere import Sphere
from gragra.geometry.primitives import Sphere as CartesianSphere
from gragra.gravity_sources._discretize import discretize
from gragra.gravity_sources._models import point_sum
from gragra.targets._high_points import TargetPoints


def _geometry() -> Sphere:
    return Sphere(
        center=np.array([1.25, -0.5, 2.0]) * u.m,
        radius=1.2 * u.m,
        frame=Frame("body"),
    )


def test_discretize_reuses_existing_midpoints_and_records_actual_grid() -> None:
    geometry = _geometry()
    density = 2.7 * u.g / u.cm**3
    spacing = 0.7 * u.m

    source = discretize(geometry, density, spacing)
    reference = CartesianSphere(
        geometry.center_m,
        geometry.radius_m,
        density.to_value(u.kg / u.m**3),
        0.7,
        1_000_000,
    ).to_weighted_points()

    np.testing.assert_array_equal(source.positions_m, reference.positions_m)
    np.testing.assert_array_equal(source.weights_kg, reference.weights_kg)
    grid_axis = int(np.ceil(2 * geometry.radius_m / spacing.to_value(u.m)))
    assert source.discretization["grid_points"] == grid_axis**3
    assert source.discretization["source_count"] == len(reference.positions_m)
    assert source.discretization["spacing_m"] == pytest.approx(0.7)
    assert source.discretization["cell_volume_m3"] == pytest.approx(
        (2 * geometry.radius_m / grid_axis) ** 3
    )
    assert source.discretization["midpoint_rule"] == "cartesian"
    assert source.density_kg_m3 == pytest.approx(density.to_value(u.kg / u.m**3))
    assert source.frame is geometry.frame


def test_discretize_equivalent_si_and_centimeter_inputs() -> None:
    frame = Frame("body")
    geometry = Sphere([25, -50, 100] * u.cm, 100 * u.cm, frame)
    si = discretize(geometry, 1000 * u.kg / u.m**3, 0.5 * u.m)
    centimeters = discretize(geometry, 1 * u.g / u.cm**3, 50 * u.cm)

    np.testing.assert_array_equal(si.positions_m, centimeters.positions_m)
    scale = float(np.max(np.abs(si.weights_kg)))
    np.testing.assert_allclose(
        si.weights_kg,
        centimeters.weights_kg,
        rtol=1e-12,
        atol=1e-12 * scale,
    )


def test_discretize_is_eager_once_and_source_arrays_are_immutable(monkeypatch) -> None:
    geometry = _geometry()
    original = CartesianSphere.to_weighted_points
    calls = 0

    def counted(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(CartesianSphere, "to_weighted_points", counted)
    source = discretize(geometry, -3 * u.kg / u.m**3, 0.6 * u.m)
    assert calls == 1
    positions = source.positions_m.copy()
    weights = source.weights_kg.copy()

    model = point_sum(source)
    assert model.spec is model.spec
    request = model.at(
        # A separated frame avoids a target/source collision.
        TargetPoints([[10, 0, 0]] * u.m, ("far",), geometry.frame)
    )
    assert request.resolved.model_spec is model.spec
    request.validate()
    assert len(request.parameters) == 1
    assert len(request.formulas) == 3
    assert len(request.references) >= 1
    assert calls == 1
    np.testing.assert_array_equal(source.positions_m, positions)
    np.testing.assert_array_equal(source.weights_kg, weights)
    with pytest.raises(ValueError):
        source.positions_m.setflags(write=True)
    with pytest.raises(ValueError):
        source.weights_kg.setflags(write=True)


@pytest.mark.parametrize("max_points", [7, 26])
def test_bbox_grid_cap_is_checked_before_sphere_inclusion(max_points: int) -> None:
    geometry = Sphere(np.zeros(3) * u.m, 1 * u.m, Frame("body"))
    with pytest.raises(ValueError, match="bbox would require 27 cells"):
        discretize(geometry, 1 * u.kg / u.m**3, 0.7 * u.m, max_points=max_points)


def test_discretize_none_cap_and_zero_density_keep_midpoint_volume() -> None:
    geometry = Sphere(np.zeros(3) * u.m, 1 * u.m, Frame("body"))
    source = discretize(geometry, 0 * u.kg / u.m**3, 0.25 * u.m, max_points=None)
    assert source.discretization["max_points"] is None
    assert source.discretization["grid_points"] == 512
    assert source.discretization["source_count"] > 0
    assert source.discretization["cell_volume_m3"] == pytest.approx(0.25**3)
    np.testing.assert_array_equal(source.weights_kg, 0.0)


@pytest.mark.parametrize(
    ("density", "spacing", "max_points", "exception"),
    [
        (1.0, 1 * u.m, 10, TypeError),
        ((1 + 2j) * u.kg / u.m**3, 1 * u.m, 10, ValueError),
        (np.nan * u.kg / u.m**3, 1 * u.m, 10, ValueError),
        (1 * u.kg / u.m**3, 1 * u.s, 10, u.UnitConversionError),
        (1 * u.kg / u.m**3, 0 * u.m, 10, ValueError),
        (1 * u.kg / u.m**3, 1 * u.m, True, TypeError),
    ],
)
def test_discretize_rejects_invalid_quantity_and_cap_inputs(
    density, spacing, max_points, exception
) -> None:
    geometry = Sphere(np.zeros(3) * u.m, 1 * u.m, Frame("body"))
    with pytest.raises(exception):
        discretize(geometry, density, spacing, max_points=max_points)
