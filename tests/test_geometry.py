"""Tests for geometry grid helper functions and validation routines."""

import numpy as np
import pytest

from gragra.geometry._grid import (
    _generate_grid,
    _validate_density,
    _validate_max_points,
    _validate_positive_scalar,
)
from gragra.geometry.primitives import Cuboid, Cylinder, Sphere
from gragra.gravity_sources import WeightedPointSource


def test_validate_positive_scalar():
    """Verify validation of positive numeric scalars."""
    # Valid cases
    assert _validate_positive_scalar("test", 5) == 5.0
    assert _validate_positive_scalar("test", 3.14) == 3.14
    assert _validate_positive_scalar("test", np.array(2.5)) == 2.5
    assert _validate_positive_scalar("test", np.float64(1.2)) == 1.2

    # Invalid types/values
    with pytest.raises(TypeError, match="numeric, not bool"):
        _validate_positive_scalar("test", True)

    with pytest.raises(TypeError, match="numeric, not bool"):
        _validate_positive_scalar("test", np.array(True, dtype=bool))

    with pytest.raises(TypeError, match="scalar"):
        _validate_positive_scalar("test", [1.0, 2.0])

    with pytest.raises(TypeError, match="scalar"):
        _validate_positive_scalar("test", np.array([1.0]))

    with pytest.raises(ValueError, match="finite"):
        _validate_positive_scalar("test", np.nan)

    with pytest.raises(ValueError, match="finite"):
        _validate_positive_scalar("test", np.inf)

    with pytest.raises(ValueError, match="positive"):
        _validate_positive_scalar("test", 0)

    with pytest.raises(ValueError, match="positive"):
        _validate_positive_scalar("test", -1.0)


def test_validate_density():
    """Verify validation of mass densities (real and complex)."""
    # Valid cases
    assert _validate_density(10) == 10.0
    assert _validate_density(1.5) == 1.5
    assert _validate_density(2.0 + 3.0j) == 2.0 + 3.0j
    assert _validate_density(np.array(5.0)) == 5.0
    assert _validate_density(np.complex128(1.0j)) == 1.0j

    # Invalid cases
    with pytest.raises(TypeError, match="must not be bool"):
        _validate_density(False)

    with pytest.raises(ValueError, match="scalar"):
        _validate_density([1.0])

    with pytest.raises(TypeError, match="numeric"):
        _validate_density("string")

    with pytest.raises(ValueError, match="finite"):
        _validate_density(np.nan)

    with pytest.raises(ValueError, match="finite"):
        _validate_density(np.complex128(complex(1.0, np.nan)))


def test_validate_max_points():
    """Verify validation of max_points parameter."""
    # Valid cases
    assert _validate_max_points(None) is None
    assert _validate_max_points(100) == 100
    assert _validate_max_points(np.array(5)) == 5

    # Invalid cases
    with pytest.raises(TypeError, match="integer, not bool"):
        _validate_max_points(True)

    with pytest.raises(TypeError, match="integer"):
        _validate_max_points(5.5)

    with pytest.raises(ValueError, match="positive"):
        _validate_max_points(0)

    with pytest.raises(ValueError, match="positive"):
        _validate_max_points(-10)


def test_generate_grid_basic():
    """Verify midpoint grid generation coordinates and cell volume."""
    center = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    bbox = np.array([2.0, 4.0, 6.0], dtype=np.float64)
    spacing = 2.0

    # Lx = 2.0 -> n_x = 1 (width 2.0) -> center 1.0
    # Ly = 4.0 -> n_y = 2 (width 2.0) -> centers 1.0, 3.0
    # Lz = 6.0 -> n_z = 3 (width 2.0) -> centers 1.0, 3.0, 5.0
    # Total points = 1 * 2 * 3 = 6
    # dv = 2.0 * 2.0 * 2.0 = 8.0
    points, dv = _generate_grid(center, bbox, spacing, max_points=10)

    assert points.shape == (6, 3)
    assert np.isclose(dv, 8.0)
    assert points.flags.writeable is False

    # Check midpoint coordinates
    # For x: center is 1.0 (range [0, 2]) -> midpoint is 1.0
    # For y: range [0, 4] -> boundaries [0, 2, 4] -> midpoints [1.0, 3.0]
    # For z: range [0, 6] -> boundaries [0, 2, 4, 6] -> midpoints [1.0, 3.0, 5.0]
    # Since indexing is "ij", x varies slowest, z varies fastest.
    expected_points = []
    for x in [1.0]:
        for y in [1.0, 3.0]:
            for z in [1.0, 3.0, 5.0]:
                expected_points.append([x, y, z])

    assert np.allclose(points, expected_points)


def test_generate_grid_max_points_guard():
    """Verify that _generate_grid raises ValueError if bbox exceeds max_points."""
    center = np.array([0.0, 0.0, 0.0])
    bbox = np.array([1.0, 1.0, 1.0])
    spacing = 0.01  # requires 100 * 100 * 100 = 1,000,000 cells

    # Limit to 1000, should raise ValueError
    with pytest.raises(ValueError, match="exceeds max_points"):
        _generate_grid(center, bbox, spacing, max_points=1000)


def test_cuboid_properties_and_validation():
    """Verify Cuboid validation of parameters and read-only status."""
    # Valid construction
    c = Cuboid(
        center_m=[1.0, 2.0, 3.0],
        lengths_m=[1.5, 2.5, 3.5],
        density_kg_m3=5.0,
        spacing_m=0.5,
    )
    assert np.allclose(c.center_m, [1.0, 2.0, 3.0])
    assert np.allclose(c.lengths_m, [1.5, 2.5, 3.5])
    assert c.density_kg_m3 == 5.0
    assert c.spacing_m == 0.5
    assert c.max_points == 1_000_000

    # Verify center_m and lengths_m arrays are read-only
    assert c.center_m.flags.writeable is False
    assert c.lengths_m.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        c.center_m[0] = 9.0
    with pytest.raises(ValueError, match="read-only"):
        c.lengths_m[0] = 9.0

    # Invalid spacing
    with pytest.raises(ValueError, match="positive"):
        Cuboid(
            center_m=[0, 0, 0], lengths_m=[1, 1, 1], density_kg_m3=1.0, spacing_m=-0.1
        )
    with pytest.raises(TypeError, match="numeric, not bool"):
        Cuboid(
            center_m=[0, 0, 0], lengths_m=[1, 1, 1], density_kg_m3=1.0, spacing_m=True
        )

    # Invalid lengths
    with pytest.raises(ValueError, match="positive"):
        Cuboid(
            center_m=[0, 0, 0], lengths_m=[1, -1, 1], density_kg_m3=1.0, spacing_m=0.5
        )
    with pytest.raises(TypeError, match="must not be bool"):
        Cuboid(
            center_m=[0, 0, 0], lengths_m=[1, True, 1], density_kg_m3=1.0, spacing_m=0.5
        )
    with pytest.raises(TypeError, match="numeric, not bool"):
        Cuboid(center_m=[0, 0, 0], lengths_m=False, density_kg_m3=1.0, spacing_m=0.5)

    # Invalid density
    with pytest.raises(ValueError, match="finite"):
        Cuboid(
            center_m=[0, 0, 0], lengths_m=[1, 1, 1], density_kg_m3=np.nan, spacing_m=0.5
        )
    with pytest.raises(TypeError, match="density_kg_m3 must not be bool"):
        Cuboid(
            center_m=[0, 0, 0], lengths_m=[1, 1, 1], density_kg_m3=False, spacing_m=0.5
        )


def test_cuboid_discretization_and_mass_conservation():
    """Verify Cuboid strict mass conservation and dtype propagation."""
    # 1. Real density -> float64 weights
    c = Cuboid(
        center_m=[0.0, 0.0, 0.0],
        lengths_m=[2.0, 3.0, 4.0],
        density_kg_m3=1.5,
        spacing_m=0.5,
    )
    src = c.to_weighted_points()
    assert isinstance(src, WeightedPointSource)
    assert src.positions_m.dtype == np.float64
    assert src.weights_kg.dtype == np.float64
    assert src.positions_m.flags.writeable is False
    assert src.weights_kg.flags.writeable is False

    # Exact volume = 2 * 3 * 4 = 24. Density = 1.5. Total mass = 36.0
    expected_mass = 24.0 * 1.5
    assert np.isclose(np.sum(src.weights_kg), expected_mass)

    # 2. Complex density -> complex128 weights
    c_comp = Cuboid(
        center_m=[0.0, 0.0, 0.0],
        lengths_m=[2.0, 3.0, 4.0],
        density_kg_m3=1.5 + 2.0j,
        spacing_m=0.5,
    )
    src_comp = c_comp.to_weighted_points()
    assert src_comp.weights_kg.dtype == np.complex128
    assert np.isclose(np.sum(src_comp.weights_kg), 24.0 * (1.5 + 2.0j))

    # 3. spacing > side_length still generates at least 1 cell per axis
    # and matches strict mass
    c_large = Cuboid(
        center_m=[1.0, 2.0, 3.0],
        lengths_m=[1.0, 2.0, 3.0],
        density_kg_m3=2.0,
        spacing_m=10.0,
    )
    src_large = c_large.to_weighted_points()
    # n_x = ceil(1/10) = 1, n_y = ceil(2/10) = 1, n_z = ceil(3/10) = 1
    assert src_large.positions_m.shape == (1, 3)
    assert np.allclose(src_large.positions_m[0], [1.0, 2.0, 3.0])
    assert np.isclose(np.sum(src_large.weights_kg), 1.0 * 2.0 * 3.0 * 2.0)


def test_sphere_properties_and_validation():
    """Verify Sphere validation and discretization."""
    # Valid Sphere
    s = Sphere(center_m=[0.0, 0.0, 0.0], radius_m=2.0, density_kg_m3=3.0, spacing_m=0.5)
    assert np.allclose(s.center_m, [0.0, 0.0, 0.0])
    assert s.radius_m == 2.0
    assert s.center_m.flags.writeable is False

    # Invalid radius
    with pytest.raises(ValueError, match="positive"):
        Sphere(center_m=[0, 0, 0], radius_m=0.0, density_kg_m3=1.0, spacing_m=0.5)
    with pytest.raises(TypeError, match="numeric, not bool"):
        Sphere(center_m=[0, 0, 0], radius_m=True, density_kg_m3=1.0, spacing_m=0.5)


def test_cylinder_properties_and_validation():
    """Verify Cylinder validation and discretization."""
    # Valid Cylinder
    cyl = Cylinder(
        center_m=[0.0, 0.0, 0.0],
        radius_m=1.0,
        height_m=5.0,
        axis="z",
        density_kg_m3=2.0,
        spacing_m=0.5,
    )
    assert np.allclose(cyl.center_m, [0.0, 0.0, 0.0])
    assert cyl.radius_m == 1.0
    assert cyl.height_m == 5.0
    assert cyl.axis == "z"
    assert cyl.center_m.flags.writeable is False

    # Invalid axis
    with pytest.raises(ValueError, match="axis must be 'x', 'y', or 'z'"):
        Cylinder(
            center_m=[0, 0, 0],
            radius_m=1.0,
            height_m=5.0,
            axis="w",
            density_kg_m3=2.0,
            spacing_m=0.5,
        )


def test_regression_validators_harden():
    """Verify hardening of validators against wrapped bools and strings."""
    # 1. np.array(True, dtype=object) should raise TypeError
    with pytest.raises(TypeError, match="must be numeric|not bool|not be string"):
        Sphere(
            center_m=[0, 0, 0],
            radius_m=np.array(True, dtype=object),
            density_kg_m3=1.0,
            spacing_m=0.5,
        )

    with pytest.raises(
        TypeError, match="must be numeric|not bool|not be string|integer"
    ):
        Sphere(
            center_m=[0, 0, 0],
            radius_m=1.0,
            density_kg_m3=1.0,
            spacing_m=0.5,
            max_points=np.array(True, dtype=object),
        )

    # 2. String "1.2" should raise TypeError
    with pytest.raises(TypeError, match="must be numeric|not be string|not string"):
        Sphere(
            center_m=[0, 0, 0],
            radius_m="1.2",
            density_kg_m3=1.0,
            spacing_m=0.5,
        )

    with pytest.raises(
        TypeError, match="must be numeric|not be string|not string|integer"
    ):
        Sphere(
            center_m=[0, 0, 0],
            radius_m=1.0,
            density_kg_m3=1.0,
            spacing_m=0.5,
            max_points="10",
        )

    # 3. np.array([1, np.bool_(True), 1], dtype=object) and ["1", "2", "3"]
    # passed to lengths_m should raise TypeError
    with pytest.raises(
        TypeError, match="must not be bool|not bool|must not be string|string"
    ):
        Cuboid(
            center_m=[0, 0, 0],
            lengths_m=np.array([1, np.bool_(True), 1], dtype=object),
            density_kg_m3=1.0,
            spacing_m=0.5,
        )

    with pytest.raises(
        TypeError, match="must not be bool|not bool|must not be string|string"
    ):
        Cuboid(
            center_m=[0, 0, 0],
            lengths_m=["1", "2", "3"],
            density_kg_m3=1.0,
            spacing_m=0.5,
        )
