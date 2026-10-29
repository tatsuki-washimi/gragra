"""M3-B H1: high-level rigid_motion vs M3-A and time-grid contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest
from astropy import units as u
from gwpy.time import LIGOTimeGPS

from gragra.geometry import Frame
from gragra.gravity_sources import (
    PointMassCloud,
    PrescribedRigidTransform,
    rigid_motion,
)
from gragra.observables import rigid_motion_observable
from gragra.results import TimeSampling
from gragra.results._time_grid import (
    build_uniform_time_s,
    validate_representable_uniform_grid,
)
from gragra.targets import TargetPoints

WORLD = Frame("world")


def _fixture(
    *, frames: int = 3, epoch: tuple[int, int] = (1_000_000_000, 0), dt: float = 1.0
):
    frame = WORLD
    source = PointMassCloud(
        [2.0, 3.0, 4.0],
        [[-1.0, 0.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, -1.0]],
    )
    dt_q = dt * u.s
    sampling = TimeSampling(epoch=LIGOTimeGPS(*epoch), dt=dt_q, count=frames)
    time_s = build_uniform_time_s(epoch, dt, frames)
    transform = PrescribedRigidTransform(
        time_s,
        np.column_stack((np.arange(frames) * 0.25, np.zeros((frames, 2)))),
        np.repeat(np.eye(3)[None], frames, axis=0),
    )
    targets = TargetPoints(
        np.array([[4.0, -3.0, 5.0], [5.0, 2.0, -4.0]], dtype=np.float64) * u.m,
        ("t0", "t1"),
        frame=frame,
    )
    return source, transform, sampling, targets, frame


def _nct_from_m3a(values: np.ndarray, observable: str) -> np.ndarray:
    if observable == "potential":
        return np.transpose(values, (1, 0))[:, np.newaxis, :]
    if observable == "acceleration":
        return np.transpose(values, (1, 2, 0))
    n_target, n_motion = values.shape[1], values.shape[0]
    return np.transpose(values, (1, 2, 3, 0)).reshape(n_target, 9, n_motion)


@pytest.mark.parametrize("observable", ("potential", "acceleration", "gradient"))
def test_h1_matches_m3a_dense(observable):
    source, transform, sampling, targets, frame = _fixture()
    result = (
        rigid_motion(source, transform, frame=frame, sampling=sampling)
        .at(targets, observables=(observable,))
        .run()
    )
    matrix = getattr(result, observable)
    assert matrix is not None
    raw = np.asarray(matrix)
    expected = _nct_from_m3a(
        rigid_motion_observable(
            source,
            transform,
            targets.positions_m,
            observable=observable,
        ),
        observable,
    )
    np.testing.assert_allclose(raw, expected, rtol=0, atol=0)
    assert result.record.g_applications == 1
    assert result.record.backend == "m3a_rigid_motion_h1"


def test_h1_dt_one_over_1024():
    source, _, _, targets, frame = _fixture(frames=8)
    epoch = (1_000_000_000, 0)
    dt = 1.0 / 1024.0
    sampling = TimeSampling(epoch=LIGOTimeGPS(*epoch), dt=dt * u.s, count=8)
    time_s = build_uniform_time_s(epoch, dt, 8)
    transform = PrescribedRigidTransform(
        time_s,
        np.zeros((8, 3)),
        np.repeat(np.eye(3)[None], 8, axis=0),
    )
    result = (
        rigid_motion(source, transform, frame=frame, sampling=sampling)
        .at(targets, observables=("acceleration",))
        .run()
    )
    assert np.asarray(result.acceleration).shape == (2, 3, 8)


def test_h1_count_one_frame():
    source, _, _, targets, frame = _fixture(frames=1)
    epoch = (1_000_000_000, 0)
    sampling = TimeSampling(epoch=LIGOTimeGPS(*epoch), dt=1e-7 * u.s, count=1)
    time_s = build_uniform_time_s(epoch, 1e-7, 1)
    transform = PrescribedRigidTransform(
        time_s,
        np.zeros((1, 3)),
        np.eye(3)[None],
    )
    rigid_motion(source, transform, frame=frame, sampling=sampling).at(
        targets, observables=("potential",)
    ).run()


def test_h1_rejects_microsecond_jitter():
    source, transform, sampling, targets, frame = _fixture(frames=11, dt=0.1)
    bad = np.array(transform.time_s, copy=True)
    bad[4] += 0.5e-6
    transform = PrescribedRigidTransform(
        bad,
        transform.translations_m,
        transform.rotation_matrices,
    )
    with pytest.raises(ValueError, match="deviates from uniform"):
        rigid_motion(source, transform, frame=frame, sampling=sampling).at(
            targets, observables=("acceleration",)
        )


def test_h1_rejects_count_mismatch():
    source, transform, sampling, targets, frame = _fixture(frames=3)
    bad_sampling = TimeSampling(
        epoch=LIGOTimeGPS(*sampling.epoch_gps), dt=sampling.dt, count=4
    )
    with pytest.raises(ValueError, match="frame count"):
        rigid_motion(source, transform, frame=frame, sampling=bad_sampling).at(
            targets, observables=("acceleration",)
        )


def test_h1_rejects_non_representable_grid():
    with pytest.raises(ValueError, match="strictly increasing float64"):
        validate_representable_uniform_grid((1_000_000_000, 0), 1e-20, 100)


def test_h1_negative_epoch_canonical_grid():
    neg_epoch = (-1_000_000_000, 0)
    source, _, _, targets, frame = _fixture(frames=10, epoch=neg_epoch, dt=0.1)
    epoch = (-1_000_000_000, 0)
    dt = 0.1
    sampling = TimeSampling(epoch=LIGOTimeGPS(*epoch), dt=dt * u.s, count=10)
    time_s = build_uniform_time_s(epoch, dt, 10)
    transform = PrescribedRigidTransform(
        time_s,
        np.zeros((10, 3)),
        np.repeat(np.eye(3)[None], 10, axis=0),
    )
    rigid_motion(source, transform, frame=frame, sampling=sampling).at(
        targets, observables=("potential",)
    ).run()


def test_h1_output_frame_rotates_acceleration():
    source, transform, sampling, targets, root = _fixture()
    angle = math.pi / 2
    rot = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    local = Frame("local", basis=rot, parent=root)
    root_values = (
        rigid_motion(source, transform, frame=root, sampling=sampling)
        .at(targets, observables=("acceleration",), output_frame=root)
        .run()
    )
    local_values = (
        rigid_motion(source, transform, frame=root, sampling=sampling)
        .at(targets, observables=("acceleration",), output_frame=local)
        .run()
    )
    rotation = root.rotation_to(local)
    expected = np.einsum("ij,njt->nit", rotation, np.asarray(root_values.acceleration))
    np.testing.assert_allclose(
        np.asarray(local_values.acceleration),
        expected,
        rtol=3e-14,
        atol=0,
    )
