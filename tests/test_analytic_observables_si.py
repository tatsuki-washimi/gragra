"""Tests for high-level SI observable entry points in gragra.observables.analytic."""

import numpy as np
import pytest

from gragra.analytic.values import GFreeGravityValues, GFreeSphericalShellPComponents
from gragra.constants import G_SI
from gragra.observables.analytic import (
    SphericalShellPComponents,
    cuboid_observables,
    cylinder_axis_observables,
    spherical_shell_p_acceleration_components,
    spherical_shell_p_infinite_difference,
    to_gravity_observables,
    to_spherical_shell_p_components,
    uniform_sphere_observables,
)
from gragra.analytic.cavity_cuboid import cuboid_observables_g_free
from gragra.analytic.cavity_cylinder import cylinder_axis_observables_g_free
from gragra.analytic.cavity_sphere import (
    spherical_shell_p_acceleration_components_g_free,
    spherical_shell_p_infinite_difference_g_free,
)
from gragra.analytic.uniform_sphere import uniform_sphere_observables_g_free


def test_uniform_sphere_observables_si():
    center = [0.0, 0.0, 0.0]
    target = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 0.5]])
    radius = 1.0
    rho = 2000.0

    gfree = uniform_sphere_observables_g_free(center, radius, rho, target)
    si = uniform_sphere_observables(center, radius, rho, target)

    np.testing.assert_allclose(si.potential, G_SI * gfree.potential)
    np.testing.assert_allclose(si.acceleration, G_SI * gfree.acceleration)
    np.testing.assert_allclose(si.gradient, G_SI * gfree.gradient)

    # Read-only check
    assert not si.potential.flags.writeable
    assert not si.acceleration.flags.writeable
    assert not si.gradient.flags.writeable


def test_cuboid_observables_si():
    center = [0.0, 0.0, 0.0]
    target = np.array([[0.0, 0.0, 5.0], [0.1, 0.2, 0.3]])
    lengths = [2.0, 3.0, 4.0]
    rho = 2500.0

    gfree = cuboid_observables_g_free(center, lengths, rho, target)
    si = cuboid_observables(center, lengths, rho, target)

    np.testing.assert_allclose(si.potential, G_SI * gfree.potential)
    np.testing.assert_allclose(si.acceleration, G_SI * gfree.acceleration)
    np.testing.assert_allclose(si.gradient, G_SI * gfree.gradient)

    assert not si.potential.flags.writeable
    assert not si.acceleration.flags.writeable
    assert not si.gradient.flags.writeable


def test_cylinder_axis_observables_si():
    center = [0.0, 0.0, 0.0]
    targets = np.array([[0.0, 0.0, -5.0], [0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    radius = 1.0
    height = 4.0
    rho = 2700.0

    gfree = cylinder_axis_observables_g_free(center, radius, height, rho, targets)
    si = cylinder_axis_observables(center, radius, height, rho, targets)

    np.testing.assert_allclose(si.potential, G_SI * gfree.potential)
    np.testing.assert_allclose(si.acceleration, G_SI * gfree.acceleration)
    np.testing.assert_allclose(si.gradient, G_SI * gfree.gradient)

    assert not si.potential.flags.writeable
    assert not si.acceleration.flags.writeable
    assert not si.gradient.flags.writeable


def test_spherical_shell_p_acceleration_components_si():
    a = 10.0
    R = 100.0
    k = 0.05
    rho = 2500.0
    xi = 1e-6
    d = [1.0, 0.0, 0.0]

    gfree = spherical_shell_p_acceleration_components_g_free(a, R, k, rho, xi, d)
    si = spherical_shell_p_acceleration_components(a, R, k, rho, xi, d)

    np.testing.assert_allclose(si.bulk, G_SI * gfree.bulk)
    np.testing.assert_allclose(si.inner_wall, G_SI * gfree.inner_wall)
    np.testing.assert_allclose(si.outer_boundary, G_SI * gfree.outer_boundary)
    np.testing.assert_allclose(si.total, G_SI * gfree.total)

    assert not si.bulk.flags.writeable
    assert not si.inner_wall.flags.writeable
    assert not si.outer_boundary.flags.writeable
    assert not si.total.flags.writeable


def test_spherical_shell_p_infinite_difference_si():
    a = 10.0
    R = 100.0
    k = 0.05
    rho = 2500.0
    xi = 1e-6
    d = [0.0, 1.0, 0.0]

    gfree = spherical_shell_p_infinite_difference_g_free(a, R, k, rho, xi, d)
    si = spherical_shell_p_infinite_difference(a, R, k, rho, xi, d)

    np.testing.assert_allclose(si, G_SI * gfree)
    assert not si.flags.writeable


def test_to_spherical_shell_p_components_conversion():
    a = 10.0
    R = 100.0
    k = 0.05
    rho = 2500.0
    xi = 1e-6
    d = [1.0, 0.0, 0.0]

    gfree = spherical_shell_p_acceleration_components_g_free(a, R, k, rho, xi, d)
    assert isinstance(gfree, GFreeSphericalShellPComponents)
    assert gfree.unit_system == "g_free"

    si = to_spherical_shell_p_components(gfree)
    assert isinstance(si, SphericalShellPComponents)
    assert si.unit_system == "si"
    np.testing.assert_allclose(si.bulk, G_SI * gfree.bulk)
    np.testing.assert_allclose(si.inner_wall, G_SI * gfree.inner_wall)
    np.testing.assert_allclose(si.outer_boundary, G_SI * gfree.outer_boundary)
    np.testing.assert_allclose(si.total, G_SI * gfree.total)

    # Calling to_spherical_shell_p_components with an SI bundle must fail
    with pytest.raises(TypeError, match="already SphericalShellPComponents"):
        to_spherical_shell_p_components(si)

    # Calling to_gravity_observables with GFreeSphericalShellPComponents must fail with helpful guide
    with pytest.raises(TypeError, match="to_spherical_shell_p_components"):
        to_gravity_observables(gfree)
