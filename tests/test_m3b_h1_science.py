"""M3-B H1-Sci (L3): high-level rigid_motion vs 80-digit independent oracle.

Separate from L1 M3-A packaging parity in ``test_m3b_h1_rigid_motion.py``.
Oracle thresholds here are **provisional** until G-最終 / ``[user]`` sign-off.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from gwpy.time import LIGOTimeGPS

pytest.importorskip("mpmath")
import mpmath as mp

from gragra.constants import G_SI
from gragra.geometry import Frame
from gragra.gravity_sources import (
    PointMassCloud,
    PrescribedRigidTransform,
    rigid_motion,
)
from gragra.results import TimeSampling
from gragra.results._time_grid import build_uniform_time_s
from gragra.targets import TargetPoints

_SPEC = importlib.util.spec_from_file_location(
    "m3b_independent_oracle", Path(__file__).parent / "oracles" / "rigid_motion.py"
)
oracle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(oracle)

# Provisional (mirrors existing M3-A science test; not G-最終).
_PROVISIONAL_RTOL = 3e-14
_PROVISIONAL_ATOL = 3e-14

OBSERVABLES = ("potential", "acceleration", "gradient")
WORLD = Frame("world")


def _fixtures():
    identity = np.eye(3)[None]
    zero = np.zeros((1, 3))
    for name, target in (
        ("single", [1.0, 2.0, 4.0]),
        ("tiny_component", [1.0, 2.0, 1e-12]),
        ("100m", [100.0, 0.0, 0.0]),
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


FIXTURES = list(_fixtures())


def _motion_first_si(result, observable: str) -> np.ndarray:
    nct = np.asarray(getattr(result, observable))
    if observable == "potential":
        return np.transpose(nct[:, 0, :], (1, 0))
    if observable == "acceleration":
        return np.transpose(nct, (2, 0, 1))
    n_target, _, n_motion = nct.shape
    flat = nct.reshape(n_target, 3, 3, n_motion)
    return np.transpose(flat, (3, 0, 1, 2))


def _run_h1(
    body,
    mass,
    rotation,
    translation,
    targets,
    *,
    observable: str,
):
    frames = len(rotation)
    epoch = (0, 0)
    dt = 1.0
    sampling = TimeSampling(epoch=LIGOTimeGPS(*epoch), dt=dt * u.s, count=frames)
    time_s = build_uniform_time_s(epoch, dt, frames)
    transform = PrescribedRigidTransform(time_s, translation, rotation)
    source = PointMassCloud(mass, body)
    target_pts = TargetPoints(
        np.asarray(targets, dtype=np.float64) * u.m,
        tuple(f"t{i}" for i in range(len(targets))),
        frame=WORLD,
    )
    return (
        rigid_motion(source, transform, frame=WORLD, sampling=sampling)
        .at(targets=target_pts, observables=(observable,))
        .run()
    )


@pytest.mark.parametrize("observable", OBSERVABLES)
@pytest.mark.parametrize("fixture", FIXTURES, ids=[item[0] for item in FIXTURES])
def test_h1_wrapper_matches_independent_oracle_provisional(fixture, observable):
    name, body, mass, rotation, translation, targets = fixture
    expected_mp = oracle.rigid_motion_gfree(
        body,
        mass,
        rotation,
        translation,
        targets,
        observable=observable,
    )
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

    result = _run_h1(body, mass, rotation, translation, targets, observable=observable)
    si = _motion_first_si(result, observable)
    assert result.record.g_applications == 1
    np.testing.assert_allclose(
        si / (G_SI * scale_float),
        expected,
        rtol=_PROVISIONAL_RTOL,
        atol=_PROVISIONAL_ATOL,
    )
    if name == "tiny_component" and observable == "acceleration":
        assert 0 < abs(expected[0, 0, 2]) < 1e-11


def test_oracle_does_not_import_gragra_integrator():
    text = (Path(__file__).parent / "oracles" / "rigid_motion.py").read_text()
    assert "gragra.integrators" not in text
    assert "gragra.kernels" not in text
