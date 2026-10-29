"""M3-A science: independent MP transforms/Newton sums and scoped Wave 6-C."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mpmath")
import mpmath as mp

from gragra.constants import G_SI
from gragra.gravity_sources import PointMassCloud
from gragra.gravity_sources.trajectory import PrescribedRigidTransform
from gragra.integrators.rigid_motion import iter_rigid_motion_gfree_blocks
from gragra.observables import rigid_motion_observable

_SPEC = importlib.util.spec_from_file_location(
    "m3a_independent_oracle", Path(__file__).parent / "oracles" / "rigid_motion.py"
)
oracle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(oracle)
OBSERVABLES = ("potential", "acceleration", "gradient")


def _fixtures():
    identity = np.eye(3)[None]
    zero = np.zeros((1, 3))
    for name, target in (
        ("single", [1.0, 2.0, 4.0]),
        ("tiny_component", [1.0, 2.0, 1e-12]),
        ("100m", [100.0, 0.0, 0.0]),
        ("1e6m", [1e6, 0.0, 0.0]),
    ):
        yield name, np.zeros((1, 3)), np.ones(1), identity, zero, np.array([target])
    body = np.array([[-1.0, 0.0, 0.0], [0.0, 2.0, 1.0]])
    mass = np.array([2.0, 3.0])
    target = np.array([[4.0, -3.0, 5.0]])
    yield (
        "multi_motion",
        body,
        mass,
        np.array([np.eye(3), [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]),
        np.array([[0.0, 0.0, 0.0], [0.5, -0.25, 0.75]]),
        target,
    )
    yield (
        "approximate_rotation",
        body,
        mass,
        np.array([[[1.0, 5e-11, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]),
        np.array([[0.5, -0.25, 0.75]]),
        target,
    )


FIXTURES = list(_fixtures())


@pytest.mark.parametrize("observable", OBSERVABLES)
@pytest.mark.parametrize("fixture", FIXTURES, ids=[item[0] for item in FIXTURES])
def test_frozen_fixtures_match_mp_transform_oracle_in_gfree_and_si(fixture, observable):
    name, body, mass, rotation, translation, targets = fixture
    transform = PrescribedRigidTransform(
        np.arange(len(rotation)), translation, rotation
    )
    source = PointMassCloud(mass, body)
    expected_mp = oracle.rigid_motion_gfree(
        body,
        mass,
        transform.rotation_matrices,
        transform.translations_m,
        targets,
        observable=observable,
    )
    # L_ref is exact distance from the first body point to the first target
    # in frame zero; fixed from input values, not the computed response.
    with mp.workdps(80):
        length = mp.sqrt(
            sum(
                (oracle.exact_float64(targets[0, i]) - oracle.exact_float64(body[0, i]))
                ** 2
                for i in range(3)
            )
        )
        scale = oracle.reference_scale(mass, length, observable)
        expected = oracle.normalized_float64(expected_mp, scale)
        scale_float = float(scale)
    blocks = list(
        iter_rigid_motion_gfree_blocks(
            source,
            transform,
            targets,
            observable=observable,
            motion_block_size=1,
            target_block_size=1,
            source_block_size=1,
            kernel_chunk_size=1,
        )
    )
    actual = np.concatenate([block.values for block in blocks], axis=0)
    si = rigid_motion_observable(source, transform, targets, observable=observable)
    np.testing.assert_allclose(actual / scale_float, expected, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(
        si / (G_SI * scale_float), expected, rtol=3e-14, atol=3e-14
    )
    zero = np.array([value == 0 for value in expected_mp.flat]).reshape(
        expected_mp.shape
    )
    np.testing.assert_array_equal(actual[zero], 0.0)
    np.testing.assert_array_equal(si[zero], 0.0)
    nonzero = ~zero
    np.testing.assert_allclose(
        actual[nonzero] / scale_float, expected[nonzero], rtol=3e-14, atol=0
    )
    np.testing.assert_allclose(
        si[nonzero] / (G_SI * scale_float), expected[nonzero], rtol=3e-14, atol=0
    )
    if name == "tiny_component" and observable == "acceleration":
        assert 0 < abs(expected[0, 0, 2]) < 1e-11
    for mutant in (np.zeros_like(si), -si, si / G_SI, si * G_SI):
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                mutant[nonzero] / (G_SI * scale_float),
                expected[nonzero],
                rtol=3e-14,
                atol=0,
            )


def test_oracle_exact_ingestion_and_transform_are_not_float64_shortcuts():
    with mp.workdps(80):
        numerator, denominator = (0.1).as_integer_ratio()
        assert oracle.exact_float64(0.1) == mp.mpf(numerator) / denominator
        assert oracle.exact_float64(0.1) != mp.mpf("0.1")
    body = np.array([[1e16, 1.0, 0.0]])
    rotation = np.array([[[1.0, 5e-11, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    translation = np.array([[-1e16, 0.0, 0.0]])
    targets = np.array([[2.0, 3.0, 4.0]])
    primary = oracle.rigid_motion_gfree(
        body, [1.0], rotation, translation, targets, observable="acceleration"
    )
    world = np.einsum("fij,sj->fsi", rotation, body) + translation[:, None]
    diagnostic = oracle.transformed_world_gfree(
        world, [1.0], targets, observable="acceleration"
    )
    assert primary[0, 0, 0] != diagnostic[0, 0, 0]


def _wave_errors(times, counts):
    from gragra.observables.wave456_reference import passing_cuboid_observables
    from scripts.research_wave6_finite_motion_mesh import (
        PASSING_TARGETS_M,
        passing_transform,
        uniform_box_cloud,
    )

    reference = passing_cuboid_observables(times, PASSING_TARGETS_M, order=24)
    errors = []
    for cells in counts:
        source = uniform_box_cloud(cells)
        row = {"source_count": len(source.masses_kg), "relative_l2_per_target": {}}
        for observable in OBSERVABLES:
            actual = rigid_motion_observable(
                source,
                passing_transform(times),
                PASSING_TARGETS_M,
                observable=observable,
            )
            expected = getattr(reference, observable)
            axes = (0,) + tuple(range(2, actual.ndim))
            error = np.sqrt(
                np.sum((actual - expected) ** 2, axis=axes)
                / np.sum(expected**2, axis=axes)
            )
            assert np.isfinite(error).all()
            row["relative_l2_per_target"][observable] = error.tolist()
        errors.append(row)
    return errors


def test_wave6c_five_frame_criterion_is_scoped_to_the_existing_fixture():
    rows = _wave_errors(np.linspace(-2.0, 2.0, 5), [(2, 1, 1), (8, 4, 4)])
    for observable in OBSERVABLES:
        coarse, fine = [max(row["relative_l2_per_target"][observable]) for row in rows]
        assert fine < coarse
        assert fine < 2e-3


@pytest.mark.slow
def test_wave6c_41_frame_research_series_records_mesh_errors_and_trend(record_property):
    import json

    times = np.arange(-2.0, 2.0 + 0.05, 0.1, dtype=np.float64)
    assert len(times) == 41
    rows = _wave_errors(times, [(2, 1, 1), (4, 2, 2), (8, 4, 4), (16, 8, 8)])
    assert [row["source_count"] for row in rows] == [12, 96, 768, 6144]
    ratios = {}
    for observable in OBSERVABLES:
        errors = np.array([row["relative_l2_per_target"][observable] for row in rows])
        assert np.all(errors[1:] < errors[:-1])
        ratios[observable] = (errors[:-1] / errors[1:]).tolist()
        # Empirical trend for this fixed fixture only; no universal order claim.
        assert np.all(
            (errors[:-1] / errors[1:] > 2.0) & (errors[:-1] / errors[1:] < 6.0)
        )
    record_property("wave6c_41_frame_mesh_errors", json.dumps(rows, sort_keys=True))
    record_property("wave6c_41_frame_error_ratios", json.dumps(ratios, sort_keys=True))
