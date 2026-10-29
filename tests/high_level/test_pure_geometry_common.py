"""Contracts for pure, unit-aware high-level geometry."""

import numpy as np
import pytest
from astropy import units as u

import gragra.geometry as geometry
from gragra.geometry import Frame
from gragra.geometry.primitives import Cuboid as NumericCuboid
from gragra.geometry.primitives import Cylinder as NumericCylinder


def test_pure_shapes_are_separate_from_numeric_primitives() -> None:
    assert geometry.Cuboid is not NumericCuboid
    assert geometry.Cylinder is not NumericCylinder

    frame = Frame("body")
    center = np.array([100.0, -200.0, 300.0]) * u.cm
    lengths = np.array([2.0, 3.0, 4.0]) * u.m
    cuboid = geometry.Cuboid(center, lengths, frame)
    cylinder = geometry.Cylinder(center, 50 * u.cm, 2 * u.m, frame)
    shell = geometry.SphericalShell(center, 1 * u.m, 3 * u.m, frame)

    assert cuboid.frame is frame
    np.testing.assert_allclose(cuboid.center_m, [1.0, -2.0, 3.0])
    np.testing.assert_allclose(cuboid.lengths_m, [2.0, 3.0, 4.0])
    assert cylinder.frame is frame
    np.testing.assert_allclose(cylinder.center_m, [1.0, -2.0, 3.0])
    assert cylinder.radius_m == 0.5
    assert cylinder.height_m == 2.0
    assert shell.frame is frame
    np.testing.assert_allclose(shell.center_m, [1.0, -2.0, 3.0])
    assert shell.inner_radius_m == 1.0
    assert shell.outer_radius_m == 3.0


def test_pure_geometry_owns_center_and_dimensions() -> None:
    frame = Frame("body")
    center = np.array([1.0, 2.0, 3.0]) * u.m
    lengths = np.array([4.0, 5.0, 6.0]) * u.m
    cuboid = geometry.Cuboid(center, lengths, frame)

    center.value[0] = 20.0
    lengths.value[1] = 50.0
    exposed_center = cuboid.center_m
    exposed_lengths = cuboid.lengths_m
    exposed_center[2] = 30.0
    exposed_lengths[2] = 60.0

    np.testing.assert_allclose(cuboid.center_m, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(cuboid.lengths_m, [4.0, 5.0, 6.0])


@pytest.mark.parametrize(
    ("constructor", "dimensions"),
    [
        ("cuboid", [0.0, 1.0, 1.0]),
        ("cuboid", [1.0, np.inf, 1.0]),
        ("cylinder", [0.0, 1.0]),
        ("cylinder", [1.0, np.nan]),
        ("shell", [0.0, 1.0]),
        ("shell", [1.0, 1.0]),
        ("shell", [2.0, 1.0]),
    ],
)
def test_pure_shapes_reject_invalid_dimensions(
    constructor: str, dimensions: list[float]
) -> None:
    frame = Frame("body")
    center = np.zeros(3) * u.m
    with pytest.raises(ValueError):
        if constructor == "cuboid":
            geometry.Cuboid(center, np.asarray(dimensions) * u.m, frame)
        elif constructor == "cylinder":
            geometry.Cylinder(center, dimensions[0] * u.m, dimensions[1] * u.m, frame)
        else:
            geometry.SphericalShell(
                center, dimensions[0] * u.m, dimensions[1] * u.m, frame
            )


def test_appearance_is_optional_metadata_on_pure_geometry() -> None:
    frame = Frame("body")
    appearance = geometry.Appearance(color="#123456", alpha=0.4, label="test body")
    plain = geometry.Sphere(np.zeros(3) * u.m, 2 * u.m, frame)
    styled = geometry.Sphere(np.zeros(3) * u.m, 2 * u.m, frame, appearance=appearance)
    half_space = geometry.HalfSpace(frame, appearance=appearance)
    cuboid = geometry.Cuboid(
        np.zeros(3) * u.m, np.ones(3) * u.m, frame, appearance=appearance
    )

    assert appearance.color == "#123456"
    assert appearance.alpha == 0.4
    assert appearance.label == "test body"
    assert styled.appearance is appearance
    assert plain.appearance is None
    assert half_space.appearance is appearance
    assert cuboid.appearance is appearance
    assert styled.radius_m == plain.radius_m
    np.testing.assert_array_equal(styled.center_m, plain.center_m)


def test_invalid_geometry_is_rejected_independently_of_appearance() -> None:
    frame = Frame("body")
    appearance = geometry.Appearance(color="red", alpha=1.0, label="visual only")
    with pytest.raises(ValueError, match="positive"):
        geometry.Cylinder(
            np.zeros(3) * u.m, 1 * u.m, 0 * u.m, frame, appearance=appearance
        )


def test_appearance_accepts_only_immutable_display_values() -> None:
    appearance = geometry.Appearance(
        color="#123456", alpha=np.float64(0.4), label="body"
    )

    assert appearance.alpha == 0.4
    assert isinstance(appearance.alpha, float)
    with pytest.raises(TypeError, match="color"):
        geometry.Appearance(color=[0.1, 0.2, 0.3])
    with pytest.raises(TypeError, match="label"):
        geometry.Appearance(label=["body"])


@pytest.mark.parametrize("alpha", [True, "0.5"])
def test_appearance_rejects_non_real_alpha(alpha: object) -> None:
    with pytest.raises(TypeError, match="alpha"):
        geometry.Appearance(alpha=alpha)


@pytest.mark.parametrize("alpha", [np.nan, np.inf, -0.1, 1.1])
def test_appearance_rejects_alpha_outside_finite_unit_interval(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        geometry.Appearance(alpha=alpha)
