import numpy as np
import pytest

from gragra import G_SI
from gragra.analytic.uniform_sphere import (
    uniform_sphere_acceleration,
    uniform_sphere_gradient,
    uniform_sphere_potential,
)
from gragra.gravity_sources import PointMass
from gragra.observables import (
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)


def test_outside_matches_point_mass():
    center = np.array([1.0, 2.0, 3.0])
    radius = 2.0
    density = 1500.0
    mass = (4.0 / 3.0) * np.pi * (radius**3) * density

    # Target points far outside
    targets = np.array(
        [
            [10.0, 2.0, 3.0],
            [1.0, 8.0, 3.0],
            [1.0, 2.0, -5.0],
        ]
    )

    # Observables from point mass
    pm = PointMass(mass_kg=mass, position_m=center)
    expected_pot = point_potential(pm, targets)
    expected_acc = point_acceleration(pm, targets)
    expected_grad = point_gravity_gradient(pm, targets)

    # G-free analytic * G_SI
    calc_pot = G_SI * uniform_sphere_potential(center, radius, density, targets)
    calc_acc = G_SI * uniform_sphere_acceleration(center, radius, density, targets)
    calc_grad = G_SI * uniform_sphere_gradient(center, radius, density, targets)

    np.testing.assert_allclose(calc_pot, expected_pot, rtol=1e-12)
    np.testing.assert_allclose(calc_acc, expected_acc, rtol=1e-12)
    np.testing.assert_allclose(calc_grad, expected_grad, rtol=1e-12)


def test_inside_acceleration_linear():
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    density = 1000.0

    targets = np.array(
        [
            [0.5, 0.0, 0.0],
            [0.0, 0.25, 0.0],
            [0.0, 0.0, -0.75],
        ]
    )

    acc_gfree = uniform_sphere_acceleration(center, radius, density, targets)
    # Inside acceleration: -4/3 * pi * rho * r_rel
    expected_acc = -(4.0 / 3.0) * np.pi * density * targets
    np.testing.assert_allclose(acc_gfree, expected_acc, rtol=1e-12)


def test_center_acceleration_zero():
    center = np.array([1.0, 1.0, 1.0])
    radius = 5.0
    density = 2000.0
    targets = np.array([center])

    acc_gfree = uniform_sphere_acceleration(center, radius, density, targets)
    np.testing.assert_allclose(acc_gfree, np.zeros((1, 3)), atol=1e-15)


def test_inside_gradient_isotropic():
    center = np.array([0.0, 0.0, 0.0])
    radius = 2.0
    density = 3000.0
    targets = np.array(
        [
            [0.5, 0.2, -0.1],
            [0.0, 0.0, 0.0],
        ]
    )

    grad_gfree = uniform_sphere_gradient(center, radius, density, targets)
    # Inside gradient: -4/3 * pi * rho * delta_ij
    expected_value = -(4.0 / 3.0) * np.pi * density
    expected_grad = expected_value * np.eye(3)

    for i in range(len(targets)):
        np.testing.assert_allclose(grad_gfree[i], expected_grad, rtol=1e-12)


def test_potential_continuous_at_surface():
    center = np.array([0.0, 0.0, 0.0])
    radius = 3.0
    density = 1000.0

    # Compute at r = radius (exact limit)
    surface_targets = np.array(
        [
            [3.0, 0.0, 0.0],
            [0.0, -3.0, 0.0],
        ]
    )
    pot = uniform_sphere_potential(center, radius, density, surface_targets)

    # Compute limit from inside
    inside_targets = surface_targets * 0.999999
    pot_inside = uniform_sphere_potential(center, radius, density, inside_targets)

    # Compute limit from outside
    outside_targets = surface_targets * 1.000001
    pot_outside = uniform_sphere_potential(center, radius, density, outside_targets)

    np.testing.assert_allclose(pot, pot_inside, rtol=1e-5)
    np.testing.assert_allclose(pot, pot_outside, rtol=1e-5)


def test_acceleration_continuous_at_surface():
    center = np.array([0.0, 0.0, 0.0])
    radius = 3.0
    density = 1000.0

    surface_targets = np.array(
        [
            [3.0, 0.0, 0.0],
            [0.0, -3.0, 0.0],
        ]
    )
    acc = uniform_sphere_acceleration(center, radius, density, surface_targets)

    inside_targets = surface_targets * 0.999999
    acc_inside = uniform_sphere_acceleration(center, radius, density, inside_targets)

    outside_targets = surface_targets * 1.000001
    acc_outside = uniform_sphere_acceleration(center, radius, density, outside_targets)

    np.testing.assert_allclose(acc, acc_inside, rtol=1e-5)
    np.testing.assert_allclose(acc, acc_outside, rtol=1e-5)


def test_complex_density_dtype():
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    density = 1000.0 + 500j
    targets = np.array([[2.0, 0.0, 0.0]])

    pot = uniform_sphere_potential(center, radius, density, targets)
    acc = uniform_sphere_acceleration(center, radius, density, targets)
    grad = uniform_sphere_gradient(center, radius, density, targets)

    assert pot.dtype == np.complex128
    assert acc.dtype == np.complex128
    assert grad.dtype == np.complex128

    # Test linearity: real part matches real density, imaginary matches imaginary
    pot_real = uniform_sphere_potential(center, radius, 1000.0, targets)
    pot_imag = uniform_sphere_potential(center, radius, 500.0, targets)
    np.testing.assert_allclose(pot, pot_real + 1j * pot_imag, rtol=1e-12)


def test_signed_density_behavior():
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    targets = np.array([[2.0, 0.0, 0.0], [0.5, 0.0, 0.0]])

    pos_density = 1000.0
    neg_density = -1000.0

    pot_pos = uniform_sphere_potential(center, radius, pos_density, targets)
    acc_pos = uniform_sphere_acceleration(center, radius, pos_density, targets)
    grad_pos = uniform_sphere_gradient(center, radius, pos_density, targets)

    pot_neg = uniform_sphere_potential(center, radius, neg_density, targets)
    acc_neg = uniform_sphere_acceleration(center, radius, neg_density, targets)
    grad_neg = uniform_sphere_gradient(center, radius, neg_density, targets)

    # Dtype check
    assert pot_neg.dtype == np.float64
    assert acc_neg.dtype == np.float64
    assert grad_neg.dtype == np.float64

    # Linearity / Sign flip check
    np.testing.assert_allclose(pot_neg, -pot_pos, rtol=1e-12)
    np.testing.assert_allclose(acc_neg, -acc_pos, rtol=1e-12)
    np.testing.assert_allclose(grad_neg, -grad_pos, rtol=1e-12)


def test_rejects_invalid_radius():
    center = np.array([0.0, 0.0, 0.0])
    targets = np.array([[2.0, 0.0, 0.0]])
    density = 1000.0

    with pytest.raises(ValueError, match="radius_m must be positive"):
        uniform_sphere_potential(center, 0.0, density, targets)

    with pytest.raises(ValueError, match="radius_m must be positive"):
        uniform_sphere_potential(center, -1.0, density, targets)

    with pytest.raises(TypeError, match="radius_m must be a numeric scalar"):
        uniform_sphere_potential(center, "1.0", density, targets)

    with pytest.raises(TypeError, match="radius_m must be a numeric scalar"):
        uniform_sphere_potential(center, True, density, targets)


def test_outputs_read_only():
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    density = 1000.0
    targets = np.array([[2.0, 0.0, 0.0]])

    pot = uniform_sphere_potential(center, radius, density, targets)
    acc = uniform_sphere_acceleration(center, radius, density, targets)
    grad = uniform_sphere_gradient(center, radius, density, targets)

    assert not pot.flags.writeable
    assert not acc.flags.writeable
    assert not grad.flags.writeable
