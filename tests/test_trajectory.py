"""Contract tests for prescribed point trajectories and rigid transforms."""

import dataclasses

import numpy as np
import pytest

from gragra.constants import G_SI
from gragra.gravity_sources import PointMass, PointMassCloud
from gragra.gravity_sources.trajectory import (
    PrescribedPointTrajectory,
    PrescribedRigidTransform,
)
from gragra.observables.acceleration import point_acceleration
from gragra.observables.gradient import point_gravity_gradient
from gragra.observables.potential import point_potential
from gragra.observables.trajectory import (
    rigid_transform_acceleration,
    rigid_transform_gravity_gradient,
    rigid_transform_potential,
    trajectory_acceleration,
    trajectory_gravity_gradient,
    trajectory_potential,
)


def _rotation_z(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_point_trajectory_validates_and_defensively_copies():
    times = np.array([0.0, 0.25, 1.5])
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 2.0, 3.0]])
    trajectory = PrescribedPointTrajectory(2.0, times, positions)

    times[0] = 9.0
    positions[0, 0] = 9.0
    assert trajectory.mass_kg == 2.0
    assert trajectory.time_s.dtype == np.float64
    assert trajectory.positions_m.dtype == np.float64
    assert np.array_equal(trajectory.time_s, [0.0, 0.25, 1.5])
    assert np.array_equal(trajectory.positions_m[0], [0.0, 0.0, 0.0])
    assert not trajectory.time_s.flags.writeable
    assert not trajectory.positions_m.flags.writeable
    with pytest.raises(dataclasses.FrozenInstanceError):
        trajectory.mass_kg = 3.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mass_kg": 0.0, "time_s": [0.0], "positions_m": [[0.0, 0.0, 0.0]]},
        {"mass_kg": -1.0, "time_s": [0.0], "positions_m": [[0.0, 0.0, 0.0]]},
        {"mass_kg": np.nan, "time_s": [0.0], "positions_m": [[0.0, 0.0, 0.0]]},
        {"mass_kg": True, "time_s": [0.0], "positions_m": [[0.0, 0.0, 0.0]]},
        {
            "mass_kg": 1.0 + 0.0j,
            "time_s": [0.0],
            "positions_m": [[0.0, 0.0, 0.0]],
        },
        {
            "mass_kg": 1.0,
            "time_s": [0.0, 0.0],
            "positions_m": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        },
        {
            "mass_kg": 1.0,
            "time_s": [0.0, np.inf],
            "positions_m": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        },
        {
            "mass_kg": 1.0,
            "time_s": [0.0, 1.0],
            "positions_m": [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        },
        {
            "mass_kg": 1.0,
            "time_s": [0.0, 1.0],
            "positions_m": [[0.0, 0.0, 0.0], [1.0, np.nan, 0.0]],
        },
    ],
)
def test_point_trajectory_rejects_invalid_inputs(kwargs):
    with pytest.raises((TypeError, ValueError)):
        PrescribedPointTrajectory(**kwargs)


def test_point_trajectory_rejects_mixed_bool_and_complex_array_values():
    with pytest.raises(TypeError, match="bool"):
        PrescribedPointTrajectory(1.0, [0.0, 1.0], [[0.0, True, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(TypeError, match="complex"):
        PrescribedPointTrajectory(1.0, [0.0, 1.0], [[0.0, 1.0j, 0.0], [1.0, 0.0, 0.0]])


def test_rigid_transform_validates_so3_and_storage():
    times = np.array([0.0, 0.7])
    translations = np.array([[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]])
    rotations = np.stack([np.eye(3), _rotation_z(0.2)])
    transform = PrescribedRigidTransform(times, translations, rotations)

    times[0] = 8.0
    translations[0, 0] = 8.0
    rotations[0, 0, 0] = 8.0
    assert np.array_equal(transform.time_s, [0.0, 0.7])
    assert np.array_equal(transform.translations_m[0], [0.0, 0.0, 0.0])
    assert np.array_equal(transform.rotation_matrices[0], np.eye(3))
    assert all(
        not array.flags.writeable
        for array in (
            transform.time_s,
            transform.translations_m,
            transform.rotation_matrices,
        )
    )


@pytest.mark.parametrize(
    "rotations",
    [
        np.zeros((1, 3, 3)),
        np.diag([1.0, 1.0, -1.0])[None, :, :],
        np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.1]]]),
    ],
)
def test_rigid_transform_rejects_non_so3(rotations):
    with pytest.raises(ValueError, match="rotation"):
        PrescribedRigidTransform([0.0], [[0.0, 0.0, 0.0]], rotations)


def test_rigid_transform_rejects_complex_and_mixed_bool():
    with pytest.raises(TypeError, match="complex"):
        PrescribedRigidTransform([0.0], [[0.0, 0.0, 0.0]], np.eye(3)[None] + 0j)
    with pytest.raises(TypeError, match="bool"):
        PrescribedRigidTransform([0.0], [[0.0, True, 0.0]], np.eye(3)[None])


def test_trajectory_point_flyby_closed_form_and_readonly():
    trajectory = PrescribedPointTrajectory(
        mass_kg=2.0,
        time_s=[-1.0, 0.5],
        positions_m=[[-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )
    targets = [[0.0, 0.0, 0.0]]

    potential = trajectory_potential(trajectory, targets)
    acceleration = trajectory_acceleration(trajectory, targets)
    gradient = trajectory_gravity_gradient(trajectory, targets)

    assert np.allclose(potential[:, 0], [-2.0 * G_SI, -G_SI])
    assert np.allclose(acceleration[:, 0, 0], [-2.0 * G_SI, 0.5 * G_SI])
    assert np.allclose(acceleration[:, :, 1:], 0.0)
    assert np.allclose(gradient[0, 0], G_SI * np.diag([4.0, -2.0, -2.0]))
    assert np.allclose(gradient[1, 0], G_SI * np.diag([0.25, -0.125, -0.125]))
    assert all(
        not result.flags.writeable for result in (potential, acceleration, gradient)
    )


@pytest.mark.parametrize(
    "observable, expected_shape",
    [
        (trajectory_potential, (2, 0)),
        (trajectory_acceleration, (2, 0, 3)),
        (trajectory_gravity_gradient, (2, 0, 3, 3)),
    ],
)
def test_trajectory_empty_targets(observable, expected_shape):
    trajectory = PrescribedPointTrajectory(1.0, [0.0, 1.0], [[0.0, 0.0, 0.0]] * 2)
    result = observable(trajectory, np.empty((0, 3)))
    assert result.shape == expected_shape
    assert not result.flags.writeable


def test_rigid_empty_body_returns_zero_for_all_observables():
    source = PointMassCloud(np.empty(0), np.empty((0, 3)))
    transform = PrescribedRigidTransform(
        [0.0, 2.0], np.zeros((2, 3)), np.repeat(np.eye(3)[None], 2, axis=0)
    )
    targets = [[1.0, 2.0, 3.0]]

    for function, shape in (
        (rigid_transform_potential, (2, 1)),
        (rigid_transform_acceleration, (2, 1, 3)),
        (rigid_transform_gravity_gradient, (2, 1, 3, 3)),
    ):
        result = function(source, transform, targets)
        assert result.shape == shape
        assert np.all(result == 0.0)
        assert not result.flags.writeable


def test_rigid_static_transform_equals_point_cloud():
    source = PointMassCloud([2.0, 3.0], [[-1.0, 0.5, 2.0], [1.0, -0.5, -2.0]])
    transform = PrescribedRigidTransform(
        [0.0, 1.0, 4.0], np.zeros((3, 3)), np.repeat(np.eye(3)[None], 3, axis=0)
    )
    targets = [[0.0, 0.0, 5.0], [3.0, -1.0, 1.0]]

    for trajectory_fn, point_fn in (
        (rigid_transform_potential, point_potential),
        (rigid_transform_acceleration, point_acceleration),
        (rigid_transform_gravity_gradient, point_gravity_gradient),
    ):
        values = trajectory_fn(source, transform, targets)
        expected = point_fn(source, targets)
        assert np.allclose(values, np.broadcast_to(expected, values.shape))


def test_rigid_rotation_covariance_for_all_observables():
    source = PointMassCloud([2.0, 1.5], [[-0.7, 0.4, 1.2], [0.8, -1.1, -0.2]])
    rotation = _rotation_z(0.37)
    transform = PrescribedRigidTransform([0.0], [[0.0, 0.0, 0.0]], rotation[None])
    body_targets = np.array([[2.0, -1.0, 3.0], [-1.5, 0.5, 2.0]])
    world_targets = body_targets @ rotation.T

    assert np.allclose(
        rigid_transform_potential(source, transform, world_targets)[0],
        point_potential(source, body_targets),
    )
    expected_acc = point_acceleration(source, body_targets) @ rotation.T
    assert np.allclose(
        rigid_transform_acceleration(source, transform, world_targets)[0], expected_acc
    )
    expected_gradient = np.einsum(
        "ij,mjk,lk->mil",
        rotation,
        point_gravity_gradient(source, body_targets),
        rotation,
    )
    assert np.allclose(
        rigid_transform_gravity_gradient(source, transform, world_targets)[0],
        expected_gradient,
    )


def test_opposed_rigid_rotor_has_pi_periodicity():
    source = PointMassCloud([1.0, 1.0], [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    transform = PrescribedRigidTransform(
        [0.0, np.pi],
        np.zeros((2, 3)),
        np.stack([_rotation_z(0.3), _rotation_z(0.3 + np.pi)]),
    )
    targets = [[3.5, 1.2, 0.8]]
    for function in (
        rigid_transform_potential,
        rigid_transform_acceleration,
        rigid_transform_gravity_gradient,
    ):
        result = function(source, transform, targets)
        assert np.allclose(result[0], result[1], rtol=1e-13, atol=1e-25)


def test_trajectory_superposition_and_chunk_size_invariance():
    trajectory = PrescribedPointTrajectory(
        3.0,
        [0.0, 0.5, 2.0],
        [[-1.0, 0.0, 0.0], [0.5, 0.2, 0.0], [2.0, 0.0, 0.0]],
    )
    targets = [[0.0, 1.0, 0.0], [1.0, -1.0, 2.0]]
    direct = trajectory_acceleration(trajectory, targets, chunk_size=1)
    ref = np.stack(
        [
            point_acceleration(PointMass(trajectory.mass_kg, position), targets)
            for position in trajectory.positions_m
        ]
    )
    assert np.allclose(direct, ref)
    assert np.allclose(
        direct, trajectory_acceleration(trajectory, targets, chunk_size=99)
    )


def test_collision_reports_lexicographically_earliest_frame_target_source():
    trajectory = PrescribedPointTrajectory(1.0, [0.0], [[9.0, 0.0, 0.0]])
    transform = PrescribedRigidTransform([0.0], np.zeros((1, 3)), np.eye(3)[None])
    source = PointMassCloud([1.0, 1.0], [[9.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match=r"frame 0.*target 0.*source 1"):
        rigid_transform_potential(
            source,
            transform,
            [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]],
            chunk_size=1,
        )

    with pytest.raises(ValueError, match=r"frame 0.*target 0.*source 0"):
        trajectory_potential(trajectory, [[9.0, 0.0, 0.0]], chunk_size=1)


@pytest.mark.parametrize(
    "function",
    [trajectory_potential, trajectory_acceleration, trajectory_gravity_gradient],
)
def test_invalid_options_are_rejected_before_calculation(function):
    trajectory = PrescribedPointTrajectory(1.0, [0.0], [[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="chunk_size"):
        function(trajectory, [[1.0, 0.0, 0.0]], chunk_size=0)
    with pytest.raises(ValueError, match="softening"):
        function(trajectory, [[1.0, 0.0, 0.0]], softening_m=-1.0)
    with pytest.raises(ValueError, match="backend"):
        function(trajectory, [[1.0, 0.0, 0.0]], backend="not-a-backend")
