"""Tests for the proper-rotation wrapper around the cuboid closed form."""

import numpy as np
import pytest

from gragra.analytic import (
    cuboid_observables_g_free,
    oriented_cuboid_observables_g_free,
)
from gragra.constants import G_SI
from gragra.observables import oriented_cuboid_observables

CENTER = np.array([1.0, -2.0, 0.5])
LENGTHS = np.array([2.0, 3.0, 4.0])
DENSITY = 1800.0


def _rotation() -> np.ndarray:
    cz, sz = np.cos(0.37), np.sin(0.37)
    cx, sx = np.cos(-0.23), np.sin(-0.23)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ rx


def _world_targets(rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 1.0, 2.0],
            [-2.5, 2.5, 3.5],
        ]
    )
    return local, CENTER + local @ rotation.T


def _independent_world_quadrature_reference(
    rotation: np.ndarray, target: np.ndarray, order: int
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate a world-coordinate volume reference without production helpers."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    half_lengths = LENGTHS / 2.0
    local_sources = (
        np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(
            -1, 3
        )
        * half_lengths
    )
    volume_weights = np.einsum("i,j,k->ijk", weights, weights, weights).reshape(
        -1
    ) * np.prod(half_lengths)
    world_sources = CENTER + local_sources @ rotation.T
    delta = world_sources - target
    inverse_distance = 1.0 / np.linalg.norm(delta, axis=1)
    inverse_distance3 = inverse_distance**3
    inverse_distance5 = inverse_distance**5
    potential = -DENSITY * np.dot(volume_weights, inverse_distance)
    acceleration = DENSITY * np.einsum(
        "n,ni,n->i", volume_weights, delta, inverse_distance3
    )
    eye = np.eye(3)
    gradient = DENSITY * np.einsum(
        "n,nij->ij",
        volume_weights,
        3.0 * delta[:, :, None] * delta[:, None, :] * inverse_distance5[:, None, None]
        - eye[None, :, :] * inverse_distance3[:, None, None],
    )
    return potential, acceleration, gradient


def test_oriented_cuboid_identity_matches_existing_entrypoint():
    targets = CENTER + np.array([[0.0, 0.0, 0.0], [2.5, -2.0, 3.0]])
    expected = cuboid_observables_g_free(CENTER, LENGTHS, DENSITY, targets)
    actual = oriented_cuboid_observables_g_free(
        CENTER,
        LENGTHS,
        DENSITY,
        targets,
        rotation_matrix=np.eye(3),
    )

    np.testing.assert_array_equal(actual.potential, expected.potential)
    np.testing.assert_array_equal(actual.acceleration, expected.acceleration)
    np.testing.assert_array_equal(actual.gradient, expected.gradient)


def test_oriented_cuboid_rotates_acceleration_and_gradient_covariantly():
    rotation = _rotation()
    local_targets, targets = _world_targets(rotation)
    local = cuboid_observables_g_free(np.zeros(3), LENGTHS, DENSITY, local_targets)
    actual = oriented_cuboid_observables_g_free(
        CENTER,
        LENGTHS,
        DENSITY,
        targets,
        rotation_matrix=rotation,
    )

    np.testing.assert_allclose(
        actual.potential, local.potential, rtol=1e-15, atol=1e-10
    )
    np.testing.assert_allclose(
        actual.acceleration, local.acceleration @ rotation.T, rtol=1e-14, atol=1e-10
    )
    expected_gradient = np.einsum("ij,mjk,lk->mil", rotation, local.gradient, rotation)
    np.testing.assert_allclose(
        actual.gradient, expected_gradient, rtol=1e-14, atol=1e-10
    )
    assert not actual.acceleration.flags.writeable
    assert not actual.gradient.flags.writeable


def test_oriented_cuboid_matches_independent_world_quadrature_at_external_target():
    rotation = _rotation()
    target = CENTER + np.array([[7.0, -5.0, 6.0]])
    actual = oriented_cuboid_observables_g_free(
        CENTER,
        LENGTHS,
        DENSITY,
        target,
        rotation_matrix=rotation,
    )

    references = {
        order: _independent_world_quadrature_reference(rotation, target[0], order)
        for order in (12, 20, 32)
    }
    np.testing.assert_allclose(
        references[12][0], references[20][0], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        references[12][1], references[20][1], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        references[12][2], references[20][2], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        actual.potential[0], references[32][0], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        actual.acceleration[0], references[32][1], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        actual.gradient[0], references[32][2], rtol=1e-9, atol=1e-10
    )


def test_oriented_cuboid_quarter_turn_and_translation_are_covariant():
    quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    center = np.array([-3.0, 1.5, 2.0])
    local_target = np.array([[3.5, -2.0, 2.5]])
    target = center + local_target @ quarter_turn.T
    local = cuboid_observables_g_free(np.zeros(3), LENGTHS, DENSITY, local_target)
    actual = oriented_cuboid_observables_g_free(
        center, LENGTHS, DENSITY, target, rotation_matrix=quarter_turn
    )
    np.testing.assert_allclose(
        actual.potential, local.potential, rtol=1e-14, atol=1e-10
    )
    np.testing.assert_allclose(
        actual.acceleration,
        local.acceleration @ quarter_turn.T,
        rtol=1e-14,
        atol=1e-10,
    )
    expected_gradient = np.einsum(
        "ij,mjk,lk->mil", quarter_turn, local.gradient, quarter_turn
    )
    np.testing.assert_allclose(
        actual.gradient, expected_gradient, rtol=1e-14, atol=1e-10
    )


def test_oriented_cuboid_si_wrapper_scales_all_observables():
    rotation = _rotation()
    _, targets = _world_targets(rotation)
    g_free = oriented_cuboid_observables_g_free(
        CENTER, LENGTHS, DENSITY, targets, rotation_matrix=rotation
    )
    si = oriented_cuboid_observables(
        CENTER, LENGTHS, DENSITY, targets, rotation_matrix=rotation
    )

    np.testing.assert_allclose(si.potential, G_SI * g_free.potential)
    np.testing.assert_allclose(si.acceleration, G_SI * g_free.acceleration)
    np.testing.assert_allclose(si.gradient, G_SI * g_free.gradient)


def test_oriented_cuboid_rejects_improper_or_non_orthogonal_rotation():
    targets = np.array([[4.0, 5.0, 6.0]])
    with pytest.raises(ValueError, match="rotation_matrix"):
        oriented_cuboid_observables_g_free(
            CENTER,
            LENGTHS,
            DENSITY,
            targets,
            rotation_matrix=np.diag([-1.0, 1.0, 1.0]),
        )
    with pytest.raises(ValueError, match="rotation_matrix"):
        oriented_cuboid_observables_g_free(
            CENTER,
            LENGTHS,
            DENSITY,
            targets,
            rotation_matrix=np.array(
                [[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
        )


def test_oriented_cuboid_rejects_mixed_nested_boolean_rotation():
    with pytest.raises(TypeError, match="rotation_matrix"):
        oriented_cuboid_observables_g_free(
            CENTER,
            LENGTHS,
            DENSITY,
            [[4.0, 5.0, 6.0]],
            rotation_matrix=[[True, 0, 0], [0, 1, 0], [0, 0, 1]],
        )


def test_oriented_cuboid_zero_density_is_zero_without_relaxing_rotation_checks():
    rotation = _rotation()
    _, targets = _world_targets(rotation)
    values = oriented_cuboid_observables_g_free(
        CENTER, LENGTHS, 0.0, targets, rotation_matrix=rotation
    )

    np.testing.assert_array_equal(values.potential, np.zeros(len(targets)))
    np.testing.assert_array_equal(values.acceleration, np.zeros((len(targets), 3)))
    np.testing.assert_array_equal(values.gradient, np.zeros((len(targets), 3, 3)))
