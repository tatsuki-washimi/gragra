import numpy as np

from gragra import G_SI
from gragra.gravity_sources import PointMass
from gragra.observables import (
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)


def test_point_potential_matches_analytic():
    # Source at origin, mass = 1.0 kg
    pm = PointMass(mass_kg=1.0, position_m=[0.0, 0.0, 0.0])
    # Target at distance 2.0 m along x-axis
    targets = np.array([[2.0, 0.0, 0.0]])

    pot = point_potential(pm, targets)
    # Expected: -G_SI * mass / r
    expected = -G_SI * 1.0 / 2.0
    np.testing.assert_allclose(pot, [expected], rtol=1e-12)


def test_point_acceleration_matches_analytic():
    # Source at origin, mass = 1.0 kg
    pm = PointMass(mass_kg=1.0, position_m=[0.0, 0.0, 0.0])
    # Target at distance 1.0 m along x-axis
    targets = np.array([[1.0, 0.0, 0.0]])

    acc = point_acceleration(pm, targets)
    # Expected: a = -G_SI * mass * r_rel / r^3 = [-G_SI, 0, 0] (accel toward source)
    expected = np.array([[-G_SI, 0.0, 0.0]])
    np.testing.assert_allclose(acc, expected, rtol=1e-12)


def test_point_gradient_matches_analytic():
    # Source at origin, mass = 1.0 kg
    pm = PointMass(mass_kg=1.0, position_m=[0.0, 0.0, 0.0])
    # Target at distance 1.0 m along x-axis
    targets = np.array([[1.0, 0.0, 0.0]])

    grad = point_gravity_gradient(pm, targets)
    # Expected: G_SI * diag(2, -1, -1)
    expected = G_SI * np.diag([2.0, -1.0, -1.0])
    np.testing.assert_allclose(grad[0], expected, rtol=1e-12)
