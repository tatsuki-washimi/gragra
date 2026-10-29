"""Tests for the continuous rigid-motion pose response (M3 P3).

Reference values come from the independent 80-digit Newton oracle in
``tests/oracles/rigid_motion.py`` (loaded by path, with no gragra imports).
Scales follow the acceptance table: ``S = GM / L**p`` globally, and
``GM / r(t)**p`` per time. Pass thresholds are read from the acceptance file.
"""

from __future__ import annotations

import importlib.util
import json
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mpmath")
pytest.importorskip("scipy")

from astropy import units as u  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from gragra._continuous_rigid_motion import (  # noqa: E402
    _estimate_managed_bytes,
    make_rigid_pose_response,
)
from gragra.constants import G_SI  # noqa: E402
from gragra.geometry._frame import Frame  # noqa: E402
from gragra.gravity_sources import PointMassCloud  # noqa: E402
from gragra.observables import (  # noqa: E402
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)
from gragra.targets._high_points import TargetPoints as FramedTargetPoints  # noqa: E402
from gragra.targets.points import TargetPoints  # noqa: E402

_ROOT = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "m3a_independent_oracle", _ROOT / "oracles" / "rigid_motion.py"
)
oracle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(oracle)

_ACCEPTANCE = json.loads((_ROOT / "continuous_acceptance.json").read_text())
_P3_TOLERANCE = _ACCEPTANCE["thresholds"]["P3"]["max_normalized_error"]
_P3_STABILITY = _ACCEPTANCE["thresholds"]["P3"]["reference_stability"]

OBSERVABLES = ("potential", "acceleration", "gradient")
UNITS = {
    "potential": u.m**2 / u.s**2,
    "acceleration": u.m / u.s**2,
    "gradient": u.s**-2,
}
POWERS = {"potential": 1, "acceleration": 2, "gradient": 3}
WORLD = Frame("world")

BODY = np.array([[-1.0, 0.0, 0.0], [0.0, 2.0, 1.0]])
MASSES = np.array([2.0, 3.0])
TARGETS = np.array([[4.0, -3.0, 5.0], [0.5, 1.5, -2.0]])


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _pose_stub(_times):
    """Pose placeholder for argument-validation tests."""


def _sampled_pose(times_s, rotations, translations):
    """Return a pose callable that interpolates exactly at the given samples.

    The samples are looked up by value, so this helper returns exactly the
    sampled matrices and translations for the requested unique times.
    """
    lookup = {float(t): i for i, t in enumerate(times_s)}

    def pose(times):
        index = [lookup[float(t)] for t in times.to_value(u.s)]
        mats = np.asarray(rotations)[index]
        return Rotation.from_matrix(mats), translations[index] * u.m

    return pose


def _response(pose, *, targets=TARGETS, source=None, softening=0.0, budget=2**26):
    source = source if source is not None else PointMassCloud(MASSES, BODY)
    return make_rigid_pose_response(
        source,
        pose,
        TargetPoints(targets),
        frame=WORLD,
        memory_budget_bytes=budget,
        softening_m=softening,
    )


def _oracle_gfree(times_s, rotations, translations, targets, observable, softening=0.0):
    assert len(times_s) == len(rotations) == len(translations)
    return oracle.rigid_motion_gfree(
        BODY,
        MASSES,
        np.asarray(rotations),
        np.asarray(translations),
        targets,
        observable=observable,
        softening_m=softening,
    )


def _min_length(times_s, rotations, translations, targets):
    """Return the minimum body-to-target distance at each time (metres)."""
    assert len(times_s) == len(rotations)
    lengths = []
    for rot, trans in zip(rotations, translations, strict=True):
        world = BODY @ np.asarray(rot).T + np.asarray(trans)
        diff = targets[None, :, :] - world[:, None, :]
        lengths.append(float(np.sqrt((diff**2).sum(-1)).min()))
    return np.asarray(lengths)


def _normalized_errors(
    result, gfree, times_s, rotations, translations, targets, observable
):
    """Return (global, local) maximum normalized errors against the oracle.

    ``result`` is the SI Quantity from the response. ``gfree`` is the
    oracle's G-free mp array. The result is divided by ``G_SI`` before
    comparing, so the G factor is removed from the response only for this
    comparison.
    """
    assert len(times_s) == len(rotations)
    got = np.asarray(result.to_value(UNITS[observable])) / G_SI
    lengths = _min_length(times_s, rotations, translations, targets)
    global_length = float(lengths.min())
    global_scale = oracle.reference_scale(MASSES, global_length, observable)
    ref_global = oracle.normalized_float64(gfree, global_scale)
    global_error = float(np.max(np.abs(got / float(global_scale) - ref_global)))

    local_error = 0.0
    for t, length in enumerate(lengths):
        scale = oracle.reference_scale(MASSES, float(length), observable)
        ref_local = oracle.normalized_float64(gfree[t], scale)
        local_error = max(
            local_error,
            float(np.max(np.abs(got[t] / float(scale) - ref_local))),
        )
    return global_error, local_error


def _assert_oracle(result, gfree, times_s, rotations, translations, observable):
    assert len(times_s) == len(rotations)
    global_error, local_error = _normalized_errors(
        result, gfree, times_s, rotations, translations, TARGETS, observable
    )
    assert global_error <= _P3_TOLERANCE
    assert local_error <= _P3_TOLERANCE


def _pose_samples(times_s):
    """Return a nontrivial rotation and translation sampled at ``times_s``."""
    rotations = [_rot_z(0.7 * t + 0.2) for t in times_s]
    translations = [[0.3 * t, -0.2 * t + 0.1, 0.5] for t in times_s]
    return np.asarray(rotations), np.asarray(translations)


# ---- Single time, reference against the Newton oracle --------------------


@pytest.mark.parametrize("observable", OBSERVABLES)
def test_p3_single_time_sampled_pose_newton_oracle(observable):
    """One time, one sampled pose: result matches the 80-digit oracle."""
    times_s = np.array([0.4])
    rotations, translations = _pose_samples(times_s)
    response = _response(_sampled_pose(times_s, rotations, translations))
    result = response(times_s * u.s, observable)
    assert result.unit == UNITS[observable]
    assert result.shape == (1, len(TARGETS), *oracle.COMPONENTS[observable])
    gfree = _oracle_gfree(times_s, rotations, translations, TARGETS, observable)
    _assert_oracle(result, gfree, times_s, rotations, translations, observable)


def test_p3_order_and_duplicates_preserved():
    """Unsorted input with duplicates returns each time's own value."""
    grid = np.array([0.1, 0.3, 0.5])
    rotations, translations = _pose_samples(grid)
    response = _response(_sampled_pose(grid, rotations, translations))
    request = np.array([0.5, 0.1, 0.5, 0.3])
    result = response(request * u.s, "acceleration").to_value(UNITS["acceleration"])
    assert result.shape == (4, len(TARGETS), 3)
    # Duplicated requests return identical rows.
    np.testing.assert_array_equal(result[0], result[2])
    # Each row equals the single-time response at that time.
    for position, t in enumerate(request):
        single = response(np.array([t]) * u.s, "acceleration").to_value(
            UNITS["acceleration"]
        )
        np.testing.assert_allclose(
            result[position], single[0], rtol=_P3_TOLERANCE, atol=0.0
        )


@pytest.mark.parametrize("observable", OBSERVABLES)
def test_p3_translation_newton_oracle(observable):
    """Pure translation over several times matches the oracle."""
    times_s = np.array([0.0, 0.25, 0.9])
    rotations = np.tile(np.eye(3), (3, 1, 1))
    translations = np.array([[0.2 * t, -0.4 * t, 0.1] for t in times_s])
    response = _response(_sampled_pose(times_s, rotations, translations))
    result = response(times_s * u.s, observable)
    gfree = _oracle_gfree(times_s, rotations, translations, TARGETS, observable)
    _assert_oracle(result, gfree, times_s, rotations, translations, observable)


@pytest.mark.parametrize("observable", OBSERVABLES)
def test_p3_nontrivial_rotation_newton_oracle(observable):
    """Time-dependent rotation and translation match the oracle."""
    times_s = np.array([0.0, 0.37, 1.1, 2.0])
    rotations, translations = _pose_samples(times_s)
    response = _response(_sampled_pose(times_s, rotations, translations))
    result = response(times_s * u.s, observable)
    gfree = _oracle_gfree(times_s, rotations, translations, TARGETS, observable)
    _assert_oracle(result, gfree, times_s, rotations, translations, observable)


def test_p3_newton_reference_stability_80_to_100_digits(monkeypatch):
    """Normalized oracle values agree between 80 and 100 digits."""
    times_s = np.array([0.0, 0.37, 1.1])
    rotations, translations = _pose_samples(times_s)
    diffs = []
    for observable in OBSERVABLES:
        scale_global = oracle.reference_scale(MASSES, 1.0, observable)
        monkeypatch.setattr(oracle, "DPS", 80)
        at80 = oracle.normalized_float64(
            _oracle_gfree(times_s, rotations, translations, TARGETS, observable),
            scale_global,
        )
        monkeypatch.setattr(oracle, "DPS", 100)
        at100 = oracle.normalized_float64(
            _oracle_gfree(times_s, rotations, translations, TARGETS, observable),
            scale_global,
        )
        diffs.append(float(np.max(np.abs(at80 - at100))))
    assert max(diffs) <= _P3_STABILITY


# ---- Observables, G, and frame -------------------------------------------


def test_p3_three_observables_match_point_observables():
    """At a fixed pose, all three observables match the point observables."""
    shift = np.array([0.4, -0.3, 0.2])
    times_s = np.array([0.7])
    rotations = np.tile(np.eye(3), (1, 1, 1))
    translations = np.array([shift])
    response = _response(_sampled_pose(times_s, rotations, translations))
    moved = PointMassCloud(MASSES, BODY + shift)
    expected = {
        "potential": point_potential(moved, TARGETS),
        "acceleration": point_acceleration(moved, TARGETS),
        "gradient": point_gravity_gradient(moved, TARGETS),
    }
    for observable in OBSERVABLES:
        got = response(times_s * u.s, observable).to_value(UNITS[observable])[0]
        np.testing.assert_allclose(
            got, expected[observable], rtol=_P3_TOLERANCE, atol=0.0
        )


def test_p3_g_applied_once_matches_point_observables():
    """The response equals the point observable (ratio 1), so G is applied once."""
    shift = np.array([0.1, 0.2, -0.6])
    times_s = np.array([1.5])
    rotations = np.tile(np.eye(3), (1, 1, 1))
    response = _response(_sampled_pose(times_s, rotations, np.array([shift])))
    moved = PointMassCloud(MASSES, BODY + shift)
    got = response(times_s * u.s, "potential").to_value(UNITS["potential"])[0]
    expected = point_potential(moved, TARGETS)
    ratio = got / expected
    np.testing.assert_allclose(ratio, np.ones_like(ratio), rtol=_P3_TOLERANCE, atol=0.0)
    assert not np.allclose(ratio, G_SI) and not np.allclose(ratio, 1 / G_SI)


def test_p3_frame_transform():
    """A global rotation of pose and targets rotates the response covariantly."""
    times_s = np.array([0.0, 0.6])
    rotations, translations = _pose_samples(times_s)
    q = _rot_z(1.1) @ _rot_z(0.3)
    base = _response(_sampled_pose(times_s, rotations, translations))
    rotated_rot = np.einsum("ij,tjk->tik", q, rotations)
    rotated_trans = translations @ q.T
    rotated_targets = TARGETS @ q.T
    rotated = _response(
        _sampled_pose(times_s, rotated_rot, rotated_trans), targets=rotated_targets
    )
    pot = base(times_s * u.s, "potential").to_value(UNITS["potential"])
    pot_r = rotated(times_s * u.s, "potential").to_value(UNITS["potential"])
    acc = base(times_s * u.s, "acceleration").to_value(UNITS["acceleration"])
    acc_r = rotated(times_s * u.s, "acceleration").to_value(UNITS["acceleration"])
    grad = base(times_s * u.s, "gradient").to_value(UNITS["gradient"])
    grad_r = rotated(times_s * u.s, "gradient").to_value(UNITS["gradient"])
    scale = np.max(np.abs(acc))
    tol = _P3_TOLERANCE
    np.testing.assert_allclose(pot_r, pot, rtol=tol, atol=tol * np.max(np.abs(pot)))
    np.testing.assert_allclose(acc_r, acc @ q.T, rtol=0, atol=tol * scale)
    np.testing.assert_allclose(
        grad_r,
        np.einsum("ij,tnjk,lk->tnil", q, grad, q),
        rtol=0,
        atol=tol * np.max(np.abs(grad)),
    )


# ---- Target frames ---------------------------------------------------------


def _local_frame():
    """A child frame rotated and offset from the computation frame WORLD."""
    return Frame(
        "local",
        origin=np.array([1.0, -0.5, 2.0]) * u.m,
        basis=_rot_z(0.4),
        parent=WORLD,
    )


def test_p3_framed_targets_match_manual_transform():
    """A framed TargetPoints equals the manually transformed numerical targets."""
    local = _local_frame()
    local_positions = np.array([[0.5, 1.0, -0.7], [-1.2, 0.3, 0.9]])
    framed = FramedTargetPoints(local_positions * u.m, ids=("a", "b"), frame=local)
    world_positions = local.transform_points_m(local_positions, WORLD)
    # The offset frame must actually move the targets, or the test is vacuous.
    assert np.max(np.abs(world_positions - local_positions)) > 0.5

    times_s = np.array([0.0, 0.8])
    rotations, translations = _pose_samples(times_s)
    pose = _sampled_pose(times_s, rotations, translations)
    via_framed = make_rigid_pose_response(
        PointMassCloud(MASSES, BODY),
        pose,
        framed,
        frame=WORLD,
        memory_budget_bytes=2**26,
    )
    via_manual = _response(pose, targets=world_positions)
    for observable in OBSERVABLES:
        got = via_framed(times_s * u.s, observable).to_value(UNITS[observable])
        expected = via_manual(times_s * u.s, observable).to_value(UNITS[observable])
        scale = np.max(np.abs(expected))
        np.testing.assert_allclose(got, expected, rtol=0, atol=_P3_TOLERANCE * scale)


def test_p3_framed_targets_without_frame_path_raise():
    """A target frame with no path to the computation frame raises ValueError."""
    island = Frame("island", origin=np.array([0.0, 0.0, 0.0]) * u.m)
    framed = FramedTargetPoints(
        np.array([[0.5, 1.0, -0.7]]) * u.m, ids=("x",), frame=island
    )
    with pytest.raises(ValueError):
        make_rigid_pose_response(
            PointMassCloud(MASSES, BODY),
            _pose_stub,
            framed,
            frame=WORLD,
            memory_budget_bytes=2**26,
        )


# ---- Failure paths and memory ---------------------------------------------


def test_p3_collision_value_error():
    """A target on a source with zero softening raises the kernel ValueError."""
    times_s = np.array([0.0])
    rotations = np.tile(np.eye(3), (1, 1, 1))
    translations = np.zeros((1, 3))
    targets = np.array([BODY[0]])
    response = _response(
        _sampled_pose(times_s, rotations, translations), targets=targets
    )
    with pytest.raises(ValueError):
        response(times_s * u.s, "potential")


def test_p3_insufficient_budget_memory_error():
    """A tiny managed budget raises MemoryError before any work."""
    times_s = np.array([0.0, 0.5])
    rotations, translations = _pose_samples(times_s)
    response = _response(_sampled_pose(times_s, rotations, translations), budget=1024)
    with pytest.raises(MemoryError):
        response(times_s * u.s, "gradient")


def test_p3_single_point_cost_measurement():
    """A single time and target stays within the managed-array budget."""
    budget = 2**26
    times_s = np.array([0.2])
    rotations, translations = _pose_samples(times_s)
    response = _response(
        _sampled_pose(times_s, rotations, translations),
        targets=TARGETS[:1],
        budget=budget,
    )
    tracemalloc.start()
    try:
        response(times_s * u.s, "acceleration")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak <= budget
    assert _estimate_managed_bytes(n_input=1, n_unique=1, n_target=1, n_component=3) > 0


# ---- Contract checks -------------------------------------------------------


def test_p3_rejects_bad_input_types():
    """Wrong argument types raise TypeError at construction or call time."""
    with pytest.raises(TypeError):
        make_rigid_pose_response(
            object(),
            _pose_stub,
            TargetPoints(TARGETS),
            frame=WORLD,
            memory_budget_bytes=2**20,
        )
    with pytest.raises(TypeError):
        make_rigid_pose_response(
            PointMassCloud(MASSES, BODY),
            _pose_stub,
            TargetPoints(TARGETS),
            frame="world",
            memory_budget_bytes=2**20,
        )
    response = _response(_pose_stub)
    with pytest.raises(TypeError):
        response([0.0], "potential")
    with pytest.raises(ValueError):
        response(np.array([0.0]) * u.m, "potential")
    with pytest.raises(ValueError):
        response(np.array([0.0]) * u.s, "laplacian")
