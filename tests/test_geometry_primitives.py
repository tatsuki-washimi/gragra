import numpy as np
import pytest

from gragra.geometry import HalfCylinder, HalfSphere
from gragra.geometry.primitives import Cuboid, Cylinder, Sphere


def test_sphere_contains():
    sphere = Sphere(
        center_m=[1.0, 2.0, 3.0], radius_m=2.0, density_kg_m3=2700.0, spacing_m=0.2
    )

    # Points inside, boundary, outside
    pts = np.array(
        [
            [1.0, 2.0, 3.0],  # center
            [3.0, 2.0, 3.0],  # boundary
            [1.0, 4.0, 3.0],  # boundary
            [1.0, 2.0, 5.0],  # boundary
            [3.1, 2.0, 3.0],  # outside
            [1.0, 2.0, 5.1],  # outside
        ],
        dtype=np.float64,
    )

    res = sphere.contains(pts)
    assert np.all(res[:4])
    assert not np.any(res[4:])

    # With tolerance
    res_tol = sphere.contains(pts, tol_m=0.15)
    assert np.all(res_tol)

    # Negative/NaN/Inf tolerance raises ValueError
    with pytest.raises(ValueError):
        sphere.contains(pts, tol_m=-0.1)
    with pytest.raises(ValueError):
        sphere.contains(pts, tol_m=np.nan)
    with pytest.raises(ValueError):
        sphere.contains(pts, tol_m=np.inf)

    # Invalid points shape
    with pytest.raises(ValueError):
        sphere.contains(np.array([1.0, 2.0, 3.0]))  # 1D vector


def test_cuboid_contains():
    cuboid = Cuboid(
        center_m=[0.0, 0.0, 0.0],
        lengths_m=[2.0, 4.0, 6.0],
        density_kg_m3=2700.0,
        spacing_m=0.2,
    )

    pts = np.array(
        [
            [0.0, 0.0, 0.0],  # center
            [1.0, 2.0, 3.0],  # corner (boundary)
            [-1.0, -2.0, -3.0],  # corner (boundary)
            [1.1, 2.0, 3.0],  # outside
            [1.0, 2.1, 3.0],  # outside
            [1.0, 2.0, 3.1],  # outside
        ],
        dtype=np.float64,
    )

    res = cuboid.contains(pts)
    assert np.all(res[:3])
    assert not np.any(res[3:])

    # With tolerance
    res_tol = cuboid.contains(pts, tol_m=0.15)
    assert np.all(res_tol)

    # Negative/NaN/Inf tolerance raises ValueError
    with pytest.raises(ValueError):
        cuboid.contains(pts, tol_m=-0.05)
    with pytest.raises(ValueError):
        cuboid.contains(pts, tol_m=np.nan)
    with pytest.raises(ValueError):
        cuboid.contains(pts, tol_m=np.inf)

    # Invalid points shape
    with pytest.raises(ValueError):
        cuboid.contains(np.array([1.0, 2.0, 3.0]))  # 1D vector


def test_cylinder_contains():
    # Test Cylinder in z, x, y axes
    cyl_z = Cylinder(
        center_m=[0.0, 0.0, 0.0],
        radius_m=1.0,
        height_m=4.0,
        axis="z",
        density_kg_m3=2700.0,
        spacing_m=0.2,
    )

    pts_z = np.array(
        [
            [0.0, 0.0, 0.0],  # center
            [1.0, 0.0, 2.0],  # boundary
            [0.0, 1.0, -2.0],  # boundary
            [1.1, 0.0, 2.0],  # outside radius
            [1.0, 0.0, 2.1],  # outside height
        ]
    )
    res_z = cyl_z.contains(pts_z)
    assert np.all(res_z[:3])
    assert not np.any(res_z[3:])

    # With tolerance
    res_z_tol = cyl_z.contains(pts_z, tol_m=0.15)
    assert np.all(res_z_tol)

    # Negative/NaN/Inf tolerance raises ValueError
    with pytest.raises(ValueError):
        cyl_z.contains(pts_z, tol_m=-0.1)
    with pytest.raises(ValueError):
        cyl_z.contains(pts_z, tol_m=np.nan)
    with pytest.raises(ValueError):
        cyl_z.contains(pts_z, tol_m=np.inf)

    # Invalid points shape
    with pytest.raises(ValueError):
        cyl_z.contains(np.array([1.0, 2.0, 3.0]))  # 1D vector

    # Test x-axis cylinder
    cyl_x = Cylinder(
        center_m=[0.0, 0.0, 0.0],
        radius_m=1.0,
        height_m=4.0,
        axis="x",
        density_kg_m3=2700.0,
        spacing_m=0.2,
    )
    pts_x = np.array(
        [
            [0.0, 0.0, 0.0],  # center
            [2.0, 1.0, 0.0],  # boundary
            [-2.0, 0.0, 1.0],  # boundary
            [2.0, 1.1, 0.0],  # outside radius
            [2.1, 1.0, 0.0],  # outside height
        ]
    )
    res_x = cyl_x.contains(pts_x)
    assert np.all(res_x[:3])
    assert not np.any(res_x[3:])

    # Test y-axis cylinder
    cyl_y = Cylinder(
        center_m=[0.0, 0.0, 0.0],
        radius_m=1.0,
        height_m=4.0,
        axis="y",
        density_kg_m3=2700.0,
        spacing_m=0.2,
    )
    pts_y = np.array(
        [
            [0.0, 0.0, 0.0],  # center
            [1.0, 2.0, 0.0],  # boundary
            [0.0, -2.0, 1.0],  # boundary
            [1.1, 2.0, 0.0],  # outside radius
            [1.0, 2.1, 0.0],  # outside height
        ]
    )
    res_y = cyl_y.contains(pts_y)
    assert np.all(res_y[:3])
    assert not np.any(res_y[3:])


def test_halfsphere_contains():
    hs = HalfSphere(
        center_m=[1.0, 1.0, 1.0], radius_m=2.0, density_kg_m3=2700.0, spacing_m=0.2
    )

    pts = np.array(
        [
            [1.0, 1.0, 1.0],  # center
            [1.0, 1.0, 3.0],  # top boundary
            [3.0, 1.0, 1.0],  # side boundary
            [1.0, 1.0, 0.9],  # inside sphere but below floor (z=0.9 relative to cz=1.0)
            [1.0, 1.0, 3.1],  # outside sphere
        ]
    )
    # z relative to cz must be >= -tol_m. Here cz = 1.0. So relative z is z - cz.
    # pts[3]: z = 0.9. relative z = 0.9 - 1.0 = -0.1 < 0.0.
    res = hs.contains(pts)
    assert np.all(res[:3])
    assert not np.any(res[3:])

    # With tolerance
    res_tol = hs.contains(pts, tol_m=0.15)
    assert np.all(res_tol)

    # Negative/NaN/Inf tolerance raises ValueError
    with pytest.raises(ValueError):
        hs.contains(pts, tol_m=-0.1)
    with pytest.raises(ValueError):
        hs.contains(pts, tol_m=np.nan)
    with pytest.raises(ValueError):
        hs.contains(pts, tol_m=np.inf)

    # Invalid points shape
    with pytest.raises(ValueError):
        hs.contains(np.array([1.0, 2.0, 3.0]))  # 1D vector


def test_halfcylinder_contains():
    hc = HalfCylinder(
        center_m=[0.0, 0.0, 0.0],
        radius_m=2.0,
        length_m=4.0,
        density_kg_m3=2700.0,
        spacing_m=0.2,
    )

    pts = np.array(
        [
            [0.0, 0.0, 0.0],  # center
            [2.0, 2.0, 0.0],  # end-cap boundary
            [0.0, 0.0, 2.0],  # arch boundary
            [0.0, 0.0, -0.1],  # below floor
            [2.1, 0.0, 1.0],  # outside length
            [0.0, 2.1, 0.0],  # outside radius
        ]
    )
    res = hc.contains(pts)
    assert np.all(res[:3])
    assert not np.any(res[3:])

    # With tolerance
    res_tol = hc.contains(pts, tol_m=0.15)
    assert np.all(res_tol)

    # Negative/NaN/Inf tolerance raises ValueError
    with pytest.raises(ValueError):
        hc.contains(pts, tol_m=-0.1)
    with pytest.raises(ValueError):
        hc.contains(pts, tol_m=np.nan)
    with pytest.raises(ValueError):
        hc.contains(pts, tol_m=np.inf)

    # Invalid points shape
    with pytest.raises(ValueError):
        hc.contains(np.array([1.0, 2.0, 3.0]))  # 1D vector


def test_mass_consistency_halfsphere():
    """Verify total mass of to_weighted_points is within 5% of ρ·V for HalfSphere."""
    # spacing = R / 20
    radius = 2.0
    spacing = radius / 20.0
    density = 2700.0

    hs = HalfSphere(
        center_m=[0.0, 0.0, 0.0],
        radius_m=radius,
        density_kg_m3=density,
        spacing_m=spacing,
    )
    source = hs.to_weighted_points()

    total_mass = np.sum(source.weights_kg)
    analytic_mass = hs.analytic_volume_m3 * density

    rel_err = abs(total_mass - analytic_mass) / analytic_mass
    assert rel_err < 0.05, f"HalfSphere mass relative error {rel_err:.5f} exceeds 5%"


def test_mass_consistency_halfcylinder():
    """Verify total mass of to_weighted_points is within 5% of ρ·V for HalfCylinder."""

    radius = 1.5
    length = 4.0
    spacing = radius / 20.0
    density = 2700.0

    hc = HalfCylinder(
        center_m=[0.0, 0.0, 0.0],
        radius_m=radius,
        length_m=length,
        density_kg_m3=density,
        spacing_m=spacing,
    )
    source = hc.to_weighted_points()

    total_mass = np.sum(source.weights_kg)
    analytic_mass = hc.analytic_volume_m3 * density

    rel_err = abs(total_mass - analytic_mass) / analytic_mass
    assert rel_err < 0.05, f"HalfCylinder mass relative error {rel_err:.5f} exceeds 5%"
