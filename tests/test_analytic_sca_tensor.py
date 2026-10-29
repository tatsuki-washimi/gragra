"""Independent surface-integral checks for SCA wall coupling tensors."""

import numpy as np
import pytest

from gragra.analytic import (
    cuboid_gradient_g_free,
    cuboid_sca_wall_coupling_tensor_g_free,
    sca_wall_acceleration_from_tensor_g_free,
    sphere_sca_wall_coupling_tensor_g_free,
    uniform_sphere_gradient,
)
from gragra.constants import G_SI
from gragra.observables import sca_wall_acceleration_from_tensor


def _sphere_reference(center, radius, targets, order=96):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    azimuthal_order = 2 * order
    phi = 2.0 * np.pi * np.arange(azimuthal_order) / azimuthal_order
    u, phi_grid = np.meshgrid(nodes, phi, indexing="ij")
    normal = np.stack(
        (
            np.sqrt(1.0 - u * u) * np.cos(phi_grid),
            np.sqrt(1.0 - u * u) * np.sin(phi_grid),
            u,
        ),
        axis=-1,
    ).reshape(-1, 3)
    points = center + radius * normal
    area_weights = np.broadcast_to(
        (radius * radius * weights)[:, None] * (2.0 * np.pi / azimuthal_order),
        u.shape,
    ).reshape(-1)
    delta = points[None, :, :] - targets[:, None, :]
    inverse_cubed = 1.0 / np.linalg.norm(delta, axis=2) ** 3
    return np.einsum("n,mni,nj,mn->mij", area_weights, delta, normal, inverse_cubed)


def _cuboid_reference(center, lengths, targets, order=48):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    half = lengths / 2.0
    points = []
    normals = []
    area_weights = []
    for axis in range(3):
        tangential = [index for index in range(3) if index != axis]
        for sign in (-1.0, 1.0):
            a, b = np.meshgrid(nodes, nodes, indexing="ij")
            local = np.zeros((*a.shape, 3))
            local[..., axis] = sign * half[axis]
            local[..., tangential[0]] = half[tangential[0]] * a
            local[..., tangential[1]] = half[tangential[1]] * b
            normal = np.zeros_like(local)
            normal[..., axis] = sign
            weight = (
                half[tangential[0]]
                * half[tangential[1]]
                * weights[:, None]
                * weights[None, :]
            )
            points.append(local.reshape(-1, 3))
            normals.append(normal.reshape(-1, 3))
            area_weights.append(weight.reshape(-1))
    points = np.concatenate(points)
    normals = np.concatenate(normals)
    area_weights = np.concatenate(area_weights)
    delta = center[None, None, :] + points[None, :, :] - targets[:, None, :]
    inverse_cubed = 1.0 / np.linalg.norm(delta, axis=2) ** 3
    return np.einsum("n,mni,nj,mn->mij", area_weights, delta, normals, inverse_cubed)


def test_sphere_center_tensor_and_wall_acceleration_match_closed_form():
    center = np.array([0.5, -0.25, 0.75])
    tensor = sphere_sca_wall_coupling_tensor_g_free(
        center, 2.0, center[None, :], quadrature_order=32
    )
    expected = 4.0 * np.pi / 3.0 * np.eye(3)
    np.testing.assert_allclose(tensor[0], expected, rtol=1e-12, atol=1e-12)

    displacement = np.array([1.0e-9, -2.0e-9, 0.5e-9])
    acceleration = sca_wall_acceleration_from_tensor_g_free(
        tensor, 2300.0, displacement
    )
    np.testing.assert_allclose(
        acceleration[0], -(4.0 * np.pi / 3.0) * 2300.0 * displacement
    )


def test_sphere_general_tensor_matches_independent_surface_integral():
    center = np.zeros(3)
    targets = np.array([[0.3, -0.4, 0.2], [1.5, 0.2, -0.1], [4.0, 2.0, 3.0]])
    actual = sphere_sca_wall_coupling_tensor_g_free(
        center, 2.0, targets, quadrature_order=48
    )
    expected = _sphere_reference(center, 2.0, targets)
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(np.trace(actual[0]), 4.0 * np.pi, rtol=2e-7)
    np.testing.assert_allclose(np.trace(actual[2]), 0.0, atol=2e-7)
    np.testing.assert_allclose(actual, np.swapaxes(actual, 1, 2), atol=2e-7)


def test_sphere_default_matches_filled_gradient_at_general_internal_points():
    center = np.zeros(3)
    radius = 2.0
    targets = np.array(
        [
            [0.3, -0.4, 0.2],
            [1.8, 0.0, 0.0],
            [1.98, 0.0, 0.0],
            [0.0, 0.0, 1.98],
        ]
    )

    actual = sphere_sca_wall_coupling_tensor_g_free(center, radius, targets)
    filled_gradient = uniform_sphere_gradient(center, radius, 1.0, targets)

    # C = -Gamma_filled / rho for the wall displacement inversion.
    np.testing.assert_allclose(actual, -filled_gradient, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        actual,
        np.broadcast_to((4.0 * np.pi / 3.0) * np.eye(3), actual.shape),
        rtol=1e-12,
        atol=1e-12,
    )


def test_surface_quadrature_rejects_unconverged_sphere_target():
    with pytest.raises(ValueError, match=r"target.*convergence"):
        sphere_sca_wall_coupling_tensor_g_free(
            np.zeros(3), 2.0, [[1.98, 0.0, 0.0]], quadrature_order=16
        )


def test_cuboid_center_symmetry_and_general_surface_integral():
    center = np.array([1.0, 2.0, -1.0])
    lengths = np.array([2.0, 2.0, 2.0])
    center_tensor = cuboid_sca_wall_coupling_tensor_g_free(
        center, lengths, center[None, :], quadrature_order=24
    )
    np.testing.assert_allclose(
        center_tensor[0], (4.0 * np.pi / 3.0) * np.eye(3), rtol=2e-12, atol=2e-12
    )

    targets = center + np.array([[0.2, 0.3, -0.1], [3.0, -2.0, 4.0]])
    actual = cuboid_sca_wall_coupling_tensor_g_free(
        center, lengths, targets, quadrature_order=32
    )
    expected = _cuboid_reference(center, lengths, targets)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(np.trace(actual[0]), 4.0 * np.pi, rtol=2e-10)
    np.testing.assert_allclose(np.trace(actual[1]), 0.0, atol=2e-10)


def test_cuboid_default_matches_filled_gradient_near_an_internal_face():
    center = np.zeros(3)
    lengths = np.array([2.0, 2.0, 2.0])
    targets = np.array([[0.999, 0.2, 0.3], [0.2, 0.3, -0.1]])

    actual = cuboid_sca_wall_coupling_tensor_g_free(center, lengths, targets)
    filled_gradient = cuboid_gradient_g_free(center, lengths, 1.0, targets)

    np.testing.assert_allclose(actual, -filled_gradient, rtol=1e-12, atol=1e-12)


def test_oriented_cuboid_tensor_rotates_covariantly():
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    center = np.array([1.0, -2.0, 0.5])
    lengths = np.array([2.0, 3.0, 4.0])
    local_target = np.array([[0.2, 0.3, 0.4], [4.0, -2.0, 3.0]])
    world_target = center + local_target @ rotation.T
    local = cuboid_sca_wall_coupling_tensor_g_free(
        np.zeros(3), lengths, local_target, quadrature_order=24
    )
    actual = cuboid_sca_wall_coupling_tensor_g_free(
        center,
        lengths,
        world_target,
        rotation_matrix=rotation,
        quadrature_order=24,
    )
    expected = np.einsum("ij,mjk,lk->mil", rotation, local, rotation)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_tensor_acceleration_preserves_complex_displacement_and_si_scaling():
    tensor = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    displacement = np.array(
        [[1.0 + 2.0j, 0.0, -1.0j], [0.5j, 1.0 - 0.2j, 0.0]], dtype=np.complex128
    )
    g_free = sca_wall_acceleration_from_tensor_g_free(tensor, 3.0, displacement)
    si = sca_wall_acceleration_from_tensor(tensor, 3.0, displacement)
    np.testing.assert_allclose(g_free, -3.0 * displacement)
    np.testing.assert_allclose(si, G_SI * g_free)
    assert not g_free.flags.writeable


def test_tensor_acceleration_zero_density_is_exact_zero():
    tensor = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    displacement = np.array([1.0, -2.0, 0.5])
    result = sca_wall_acceleration_from_tensor_g_free(tensor, 0.0, displacement)
    np.testing.assert_array_equal(result, np.zeros((2, 3)))
    assert not result.flags.writeable


def test_sca_tensor_rejects_boundary_and_invalid_rotation():
    with pytest.raises(ValueError, match="boundary"):
        sphere_sca_wall_coupling_tensor_g_free(np.zeros(3), 2.0, [[2.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="boundary"):
        cuboid_sca_wall_coupling_tensor_g_free(
            np.zeros(3), [2.0, 4.0, 6.0], [[1.0, 0.0, 0.0]]
        )
    with pytest.raises(ValueError, match="rotation_matrix"):
        cuboid_sca_wall_coupling_tensor_g_free(
            np.zeros(3),
            [2.0, 4.0, 6.0],
            [[3.0, 0.0, 0.0]],
            rotation_matrix=np.diag([-1.0, 1.0, 1.0]),
        )


def test_sca_tensor_rejects_mixed_boolean_displacement_and_order_128():
    tensor = np.broadcast_to(np.eye(3), (1, 3, 3))
    with pytest.raises(TypeError, match="displacement_m"):
        sca_wall_acceleration_from_tensor_g_free(tensor, 2700.0, [True, 0.0, 1.0])
    with pytest.raises(ValueError, match="below 128"):
        sphere_sca_wall_coupling_tensor_g_free(
            np.zeros(3), 2.0, [[0.0, 0.0, 0.0]], quadrature_order=128
        )
