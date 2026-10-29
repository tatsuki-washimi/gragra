import numpy as np

from gragra import G_SI, point_acceleration, point_potential
from gragra.analytic import (
    uniform_sphere_acceleration,
    uniform_sphere_potential,
)
from gragra.geometry.primitives import Sphere


def test_sphere_outside_acceleration_vs_analytic():
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    density = 1000.0
    # spacing_m = R/20 = 0.05
    spacing = 0.05

    s = Sphere(
        center_m=center, radius_m=radius, density_kg_m3=density, spacing_m=spacing
    )
    src = s.to_weighted_points()

    # Target far outside at 3R
    target = np.array([[3.0, 0.0, 0.0]])

    acc_disc = point_acceleration(src, target)
    acc_exact = G_SI * uniform_sphere_acceleration(center, radius, density, target)

    # Discretization error is expected, verify rtol <= 0.05
    # The actual error should be much smaller (around 1%)
    np.testing.assert_allclose(acc_disc, acc_exact, rtol=0.05, atol=1e-15)


def test_sphere_inside_acceleration_vs_analytic():
    center = np.array([0.0, 0.0, 0.0])
    radius = 2.0
    density = 1000.0
    # spacing_m = R/20 = 0.1
    spacing = 0.1

    s = Sphere(
        center_m=center, radius_m=radius, density_kg_m3=density, spacing_m=spacing
    )
    src = s.to_weighted_points()

    # Target inside at 0.5R, choosing a point off the grid axes
    # to avoid exactly hitting cell boundaries/centers
    target = np.array([[0.5, 0.3, -0.4]])

    acc_disc = point_acceleration(src, target)
    acc_exact = G_SI * uniform_sphere_acceleration(center, radius, density, target)

    # Inside error is larger due to cell inclusion step function representation
    # verify rtol <= 0.15
    np.testing.assert_allclose(acc_disc, acc_exact, rtol=0.15)


def test_sphere_outside_potential_vs_analytic():
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    density = 1000.0
    spacing = 0.05

    s = Sphere(
        center_m=center, radius_m=radius, density_kg_m3=density, spacing_m=spacing
    )
    src = s.to_weighted_points()

    target = np.array([[3.0, 0.0, 0.0]])

    pot_disc = point_potential(src, target)
    pot_exact = G_SI * uniform_sphere_potential(center, radius, density, target)

    np.testing.assert_allclose(pot_disc, pot_exact, rtol=0.05)


def test_sphere_complex_density_dtype():
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    density = 1000.0 + 500j
    spacing = 0.1

    s = Sphere(
        center_m=center, radius_m=radius, density_kg_m3=density, spacing_m=spacing
    )
    src = s.to_weighted_points()

    assert src.weights_kg.dtype == np.complex128

    target = np.array([[3.0, 0.0, 0.0]])
    pot_disc = point_potential(src, target)
    assert pot_disc.dtype == np.complex128
