"""Integration tests for geometry primitives and observables."""

import numpy as np

from gragra import G_SI, point_acceleration
from gragra.geometry.primitives import Cuboid, Cylinder, Sphere
from gragra.gravity_sources import PointMass


def test_sphere_mass_convergence():
    """Verify Sphere mass convergence as spacing becomes finer.

    Expected:
      - spacing = R/10 (0.1m) -> relative mass error < 0.25
      - spacing = R/20 (0.05m) -> relative mass error < 0.15
    """
    center = [0.0, 0.0, 0.0]
    radius = 1.0
    density = 1000.0  # kg/m^3
    exact_mass = density * (4.0 / 3.0) * np.pi * (radius**3)

    # 1. Spacing = R/10 = 0.1
    s1 = Sphere(center_m=center, radius_m=radius, density_kg_m3=density, spacing_m=0.1)
    src1 = s1.to_weighted_points()
    mass1 = np.sum(src1.weights_kg)
    rel_err1 = abs(mass1 - exact_mass) / exact_mass
    assert rel_err1 < 0.25

    # 2. Spacing = R/20 = 0.05
    s2 = Sphere(center_m=center, radius_m=radius, density_kg_m3=density, spacing_m=0.05)
    src2 = s2.to_weighted_points()
    mass2 = np.sum(src2.weights_kg)
    rel_err2 = abs(mass2 - exact_mass) / exact_mass
    assert rel_err2 < 0.15


def test_cylinder_mass_convergence():
    """Verify Cylinder mass convergence as spacing becomes finer.

    V = pi * r^2 * h.
    """
    center = [0.0, 0.0, 0.0]
    radius = 1.0
    height = 2.0
    density = 500.0
    exact_mass = density * np.pi * (radius**2) * height

    # Spacing = 0.1
    cyl = Cylinder(
        center_m=center,
        radius_m=radius,
        height_m=height,
        axis="z",
        density_kg_m3=density,
        spacing_m=0.1,
    )
    src = cyl.to_weighted_points()
    mass = np.sum(src.weights_kg)
    rel_err = abs(mass - exact_mass) / exact_mass
    # A cylinder discretized on a 0.1 spacing grid should have small relative error
    assert rel_err < 0.1


def test_cylinder_axes_behavior():
    """Verify that Cylinder generates correct bounding box and masks for each axis."""
    center = [1.0, 2.0, 3.0]
    radius = 1.0
    height = 4.0
    density = 100.0
    spacing = 0.5

    # Test "x" axis
    cyl_x = Cylinder(
        center_m=center,
        radius_m=radius,
        height_m=height,
        axis="x",
        density_kg_m3=density,
        spacing_m=spacing,
    )
    src_x = cyl_x.to_weighted_points()
    # Coordinates along x should span from center[0]-height/2 to center[0]+height/2
    assert (src_x.positions_m[:, 0] >= center[0] - height / 2.0).all()
    assert (src_x.positions_m[:, 0] <= center[0] + height / 2.0).all()
    # Radial coords (y, z) should be within radius
    dy = src_x.positions_m[:, 1] - center[1]
    dz = src_x.positions_m[:, 2] - center[2]
    assert (np.sqrt(dy**2 + dz**2) <= radius).all()

    # Test "y" axis
    cyl_y = Cylinder(
        center_m=center,
        radius_m=radius,
        height_m=height,
        axis="y",
        density_kg_m3=density,
        spacing_m=spacing,
    )
    src_y = cyl_y.to_weighted_points()
    assert (src_y.positions_m[:, 1] >= center[1] - height / 2.0).all()
    assert (src_y.positions_m[:, 1] <= center[1] + height / 2.0).all()
    dx = src_y.positions_m[:, 0] - center[0]
    dz = src_y.positions_m[:, 2] - center[2]
    assert (np.sqrt(dx**2 + dz**2) <= radius).all()

    # Test "z" axis
    cyl_z = Cylinder(
        center_m=center,
        radius_m=radius,
        height_m=height,
        axis="z",
        density_kg_m3=density,
        spacing_m=spacing,
    )
    src_z = cyl_z.to_weighted_points()
    assert (src_z.positions_m[:, 2] >= center[2] - height / 2.0).all()
    assert (src_z.positions_m[:, 2] <= center[2] + height / 2.0).all()
    dx = src_z.positions_m[:, 0] - center[0]
    dy = src_z.positions_m[:, 1] - center[1]
    assert (np.sqrt(dx**2 + dy**2) <= radius).all()


def test_far_field_multipole_and_e2e():
    """Verify far-field fields approach point-mass limits.

    1. Multipole test: compare fields with point mass of the SAME mass.
       Verify rtol is tight (e.g., 1e-3).
    2. End-to-end test: compare fields with exact analytical continuous mass.
       Verify rtol is looser (e.g., 0.05).
    """
    center = [0.0, 0.0, 0.0]
    radius = 1.0
    density = 1000.0
    spacing = 0.05

    s = Sphere(
        center_m=center, radius_m=radius, density_kg_m3=density, spacing_m=spacing
    )
    src = s.to_weighted_points()
    discrete_mass = np.sum(src.weights_kg)
    continuous_mass = density * (4.0 / 3.0) * np.pi * (radius**3)

    # Target far away at [15.0, 0.0, 0.0]
    target = [[15.0, 0.0, 0.0]]

    # Compute acceleration from discretized sphere
    acc_disc = point_acceleration(src, target)[0]

    # 1. Multipole test: PointMass with discrete_mass at center
    pm_disc = PointMass(discrete_mass, center)
    acc_pm_disc = point_acceleration(pm_disc, target)[0]
    # Verify tight rtol (rtol=1e-3)
    assert np.allclose(acc_disc, acc_pm_disc, rtol=1e-3)

    # 2. E2E test: compare with analytical field using continuous_mass
    # a_exact = - G_SI * M / r^2
    expected_acc_x = -G_SI * continuous_mass / (15.0**2)
    expected_acc = np.array([expected_acc_x, 0.0, 0.0])
    # Verify looser rtol (rtol=0.05)
    assert np.allclose(acc_disc, expected_acc, rtol=0.05)


def test_symmetry():
    """Verify that centered Sphere cancels transverse acceleration.

    For target on x-axis [10, 0, 0], acceleration y and z components must
    be close to zero.
    """
    s = Sphere(
        center_m=[0.0, 0.0, 0.0], radius_m=1.0, density_kg_m3=1000.0, spacing_m=0.1
    )
    src = s.to_weighted_points()
    target = [[10.0, 0.0, 0.0]]

    acc = point_acceleration(src, target)[0]
    # Transverse components must cancel to zero within numerical precision
    # (e.g. < 1e-12)
    assert abs(acc[1]) < 1e-12
    assert abs(acc[2]) < 1e-12


def test_complex_density_and_dtype_propagation():
    """Verify complex density produces complex weights and propagates to observables.

    For negative or complex densities, PointMass wrapper is not allowed. We must
    use WeightedPointSource directly.
    """
    c = Cuboid(
        center_m=[0.0, 0.0, 0.0],
        lengths_m=[1.0, 1.0, 1.0],
        density_kg_m3=1.0 + 2.0j,
        spacing_m=0.5,
    )
    src = c.to_weighted_points()
    assert src.weights_kg.dtype == np.complex128

    target = [[5.0, 0.0, 0.0]]
    acc = point_acceleration(src, target)
    assert acc.dtype == np.complex128
    # Assert non-zero imaginary parts
    assert abs(acc[0, 0].imag) > 1e-13


def test_fallback_generates_at_least_one_point():
    """Verify Sphere and Cylinder generate at least 1 point for very large spacing."""
    # Sphere with very large spacing relative to radius
    s = Sphere(
        center_m=[1.0, 2.0, 3.0], radius_m=0.1, density_kg_m3=1.0, spacing_m=10.0
    )
    src_s = s.to_weighted_points()
    assert len(src_s.positions_m) >= 1
    assert np.allclose(src_s.positions_m[0], [1.0, 2.0, 3.0])

    # Cylinder with very large spacing relative to radius/height
    cyl = Cylinder(
        center_m=[1.0, 2.0, 3.0],
        radius_m=0.1,
        height_m=0.1,
        axis="y",
        density_kg_m3=1.0,
        spacing_m=10.0,
    )
    src_cyl = cyl.to_weighted_points()
    assert len(src_cyl.positions_m) >= 1
    assert np.allclose(src_cyl.positions_m[0], [1.0, 2.0, 3.0])
