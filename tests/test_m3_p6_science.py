"""P6 scientific validation of the continuous Fourier path (V1-V8).

Acceptance criteria are fixed before any run and are the module constants
below. Each ``measure_*`` helper returns the raw numbers that the tests assert
on; ``scripts/m3_p6_report.py`` imports the same helpers so the report and the
tests cannot drift apart.

Normalisation: a coefficient error is divided by the scale ``S = G M / L^p``
(``p = 1, 2, 3`` for potential, acceleration, gradient; ``M = 1 kg``, ``L = 1 m``).
Window coefficients use ``S * W`` with ``W`` the window width in seconds.

Oracles are in ``tests/_m3_p6_oracles.py`` and never import gragra.
"""

from __future__ import annotations

import functools
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mpmath")
pytest.importorskip("scipy")

import mpmath as mp  # noqa: E402

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from _continuous_fixtures import (  # noqa: E402
    circular_pose,
    constant_velocity_pose,
    expand_components,
    gaussian,
    make_response,
    static_pose,
    window_transform_closed_form,
)
from _m3_p6_oracles import (  # noqa: E402
    circular_centre_closed_form,
    field_components,
    gravitational_constant,
    periodic_coefficients,
    periodic_coefficients_all,
    rigid_field_numpy,
    rodrigues_mp,
    rodrigues_numpy,
    scale_si,
    window_transform,
)
from astropy import units as u  # noqa: E402
from gwpy.time import LIGOTimeGPS  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from gragra._continuous_contract import ContinuousQuadratureError  # noqa: E402
from gragra._continuous_fourier import (  # noqa: E402
    DEFAULT_MAX_EVALUATIONS,
    DEFAULT_MEMORY_BUDGET_BYTES,
    ContinuousFourierModel,
)
from gragra._continuous_rigid_motion import make_rigid_pose_response  # noqa: E402
from gragra._finite_window_fourier import finite_window_fourier  # noqa: E402
from gragra.geometry import Frame  # noqa: E402
from gragra.gravity_sources import PointMassCloud, rigid_motion  # noqa: E402
from gragra.gravity_sources.trajectory import PrescribedRigidTransform  # noqa: E402
from gragra.observables.acceleration import point_acceleration  # noqa: E402
from gragra.observables.rigid_motion import rigid_motion_observable  # noqa: E402
from gragra.results import TimeSampling  # noqa: E402
from gragra.results._time_grid import build_uniform_time_s  # noqa: E402
from gragra.targets import TargetPoints  # noqa: E402
from gragra.targets._high_points import TargetPoints as FramedTargets  # noqa: E402
from gragra.targets.points import TargetPoints as NumericalTargets  # noqa: E402

# ---------------------------------------------------------------------------
# Acceptance thresholds (fixed before the first run)
# ---------------------------------------------------------------------------
THR_V1 = 1e-12  # M3-A vs continuous response, normalised by S
THR_V2 = 1e-8  # periodic Fourier vs mpmath oracle, normalised by S
THR_V2_CONV_SLACK = 1e-12  # allowed non-monotonic slack in convergence tables
THR_V3 = 1e-8  # finite window vs mpmath oracle, normalised by S*W
THR_V4 = 1e-6  # direct periodic vs discrete FFT, low orders (n <= 8), by S
THR_V5 = 1e-10  # units / frame / phase / epoch identities, normalised by S
THR_V6 = 1e-12  # one application of G (ratio to point_acceleration)
THR_V1_INDEP = 1e-12  # M1 a: independent rigid-body Newton sum, normalised by S
THR_V2_ROT = 1e-8  # M1 b: rotating asymmetric body, periodic vs mpmath, by S
THR_FRAME = 1e-12  # M1 j / V5: frame chain identities, normalised by S
THR_IDENT = 1e-12  # M1 c: gradient symmetry and trace, normalised by S
THR_NEG = 1e-3  # M3 g: a wrong expectation must exceed this to be rejected
ORDERS_V2 = np.arange(0, 7, dtype=int)
MEM = DEFAULT_MEMORY_BUDGET_BYTES  # 64 MiB, managed-array budget
RTOL_LADDER_V2 = (1e-6, 1e-8, 1e-10, 1e-12)
RTOL_LADDER_V3 = (1e-4, 1e-6, 1e-8, 1e-10)
MAX_NODES_LADDER_V2 = (4096, 8192, 16384)
PERIOD_S = 1.0
WORLD = Frame("world")
EPOCH0 = LIGOTimeGPS(0, 0)

_UNIT = {
    "potential": u.m**2 / u.s**2,
    "acceleration": u.m / u.s**2,
    "gradient": u.s**-2,
}
_POWER = {"potential": 1, "acceleration": 2, "gradient": 3}
_COMP = {"potential": 1, "acceleration": 3, "gradient": 9}


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _circle_source() -> PointMassCloud:
    return PointMassCloud(np.array([1.0]), np.array([[0.0, 0.0, 0.0]]))


def _framed(positions_m) -> FramedTargets:
    pos = np.atleast_2d(np.asarray(positions_m, dtype=np.float64))
    ids = tuple(f"t{i}" for i in range(len(pos)))
    return FramedTargets(pos * u.m, ids=ids, frame=WORLD)


def _circle_xyz(t_s):
    """Circular body path c(t) (numpy), radius 1 m, period 1 s."""
    a = 2.0 * np.pi * np.asarray(t_s, dtype=np.float64)
    return np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=1)


def _periodic(
    model,
    targets,
    *,
    orders,
    observables,
    epoch=EPOCH0,
    tref=EPOCH0,
    rtol=1e-10,
    max_nodes=65536,
    tau_min=PERIOD_S / 8.0,
    atol_scale=1e-10,
    scale=None,
    budget=MEM,
):
    """Build a periodic evaluation with the P6 settings.

    ``tau_min=None`` is passed through unchanged so the negative test can see it.
    """
    if scale is None:
        scale = scale_si(1.0, 1.0, 2)
    atol = atol_scale * scale * _UNIT[observables[0]]
    return model.periodic_at(
        targets,
        epoch=epoch,
        tref=tref,
        period=PERIOD_S * u.s,
        orders=np.asarray(orders, dtype=int),
        tau_min=None if tau_min is None else tau_min * u.s,
        atol=atol,
        rtol=rtol,
        max_nodes=max_nodes,
        memory_budget_bytes=budget,
        observables=observables,
    )


def _values(result, name: str) -> np.ndarray:
    """Return the payload as ``(target, component, frequency)`` complex."""
    return np.asarray(getattr(result, name).value, dtype=np.complex128)


def normalised_error(got, ref, scale: float) -> float:
    """The judge used by every check: ``max |got - ref| / scale``.

    Negative controls call this same function with wrong expectations, so a
    passing test cannot rely on a judge that accepts everything.
    """
    got_arr = np.asarray(got, dtype=np.complex128)
    ref_arr = np.broadcast_to(np.asarray(ref, dtype=np.complex128), got_arr.shape)
    return float(np.max(np.abs(got_arr - ref_arr)) / scale)


def wrong_answer_candidates(ref) -> dict:
    """Plausible wrong expectations: conjugate, factor 2, sign flip."""
    ref_arr = np.asarray(ref, dtype=np.complex128)
    return {
        "conjugate": np.conj(ref_arr),
        "double": 2.0 * ref_arr,
        "sign_flip": -ref_arr,
    }


# ---------------------------------------------------------------------------
# V1: continuous response vs M3-A rigid_motion_observable
# ---------------------------------------------------------------------------


def _v1_case():
    """Rotation plus translation, three body points, three targets."""
    masses = np.array([1.0, 2.0, 0.5])
    body = np.array([[0.3, 0.0, 0.1], [-0.2, 0.25, 0.0], [0.0, -0.4, 0.3]])
    omega = np.array([0.3, 0.5, 1.1])
    origin = np.array([3.0, 0.0, 0.0])

    def pose(times):
        t = np.asarray(times.to_value(u.s), dtype=np.float64)
        rot = Rotation.from_rotvec(t[:, None] * omega[None, :])
        trans = (
            origin[None, :]
            + 0.5 * t[:, None] * np.array([0.2, 0.1, 0.0])[None, :]
            + 0.4 * np.stack([np.sin(t), np.cos(2 * t), np.zeros_like(t)], axis=1)
        )
        return rot, trans * u.m

    targets = np.array([[0.0, 0.0, 0.0], [0.5, -1.0, 0.2], [-1.0, 0.3, 0.8]])
    times = np.linspace(0.0, 4.0, 41)
    return masses, body, pose, targets, times


def measure_v1() -> list[dict]:
    """Max normalised error of continuous response vs M3-A, per observable."""
    masses, body, pose, targets, times = _v1_case()
    source = PointMassCloud(masses, body)
    response = make_rigid_pose_response(
        source, pose, NumericalTargets(targets), frame=WORLD, memory_budget_bytes=MEM
    )
    rot, trans = pose(times * u.s)
    transform = PrescribedRigidTransform(
        time_s=times,
        translations_m=np.asarray(trans.to_value(u.m)),
        rotation_matrices=rot.as_matrix(),
    )
    rows = []
    for name in ("potential", "acceleration", "gradient"):
        cont = response(times * u.s, name).to_value(_UNIT[name])
        ref = rigid_motion_observable(
            source,
            transform,
            NumericalTargets(targets),
            observable=name,
            memory_budget_bytes=MEM,
        )
        s = scale_si(float(masses.sum()), 1.0, _POWER[name])
        err = float(np.max(np.abs(cont - ref)) / s)
        rows.append(
            {
                "observable": name,
                "shape": list(np.shape(cont)),
                "max_normalised_error": err,
                "scale_S": s,
            }
        )
    return rows


def test_v1_continuous_matches_m3a_rigid_motion():
    """V1 acceptance: normalised max error <= THR_V1 for potential/acc/gradient."""
    rows = measure_v1()
    for row in rows:
        assert row["max_normalised_error"] <= THR_V1, row


# M1 a: independent Newton sum (numpy, own Rodrigues) at unsorted, duplicated times.
V1_TIMES_S = np.array([3.1, 0.2, 1.7, 0.2, 2.5, 0.9, 3.1, 0.0, 4.0, 1.7, 0.45])


def _v1_independent_pose(t: float) -> tuple[np.ndarray, np.ndarray]:
    """Restatement of the ``_v1_case`` pose, written without scipy."""
    omega = np.array([0.3, 0.5, 1.1])
    origin = np.array([3.0, 0.0, 0.0])
    rot = rodrigues_numpy(t * omega)
    trans = (
        origin
        + 0.5 * t * np.array([0.2, 0.1, 0.0])
        + 0.4 * np.array([np.sin(t), np.cos(2 * t), 0.0])
    )
    return rot, trans


def measure_v1_independent() -> dict:
    """M1 a: continuous response vs independent Newton sum (3 observables)."""
    masses, body, pose, targets, _ = _v1_case()
    response = make_rigid_pose_response(
        PointMassCloud(masses, body),
        pose,
        NumericalTargets(targets),
        frame=WORLD,
        memory_budget_bytes=MEM,
    )
    frames = [_v1_independent_pose(float(t)) for t in V1_TIMES_S]
    rows = []
    for name in ("potential", "acceleration", "gradient"):
        cont = np.asarray(
            response(V1_TIMES_S * u.s, name).to_value(_UNIT[name]), dtype=np.float64
        ).reshape(len(V1_TIMES_S), len(targets), -1)
        ref = np.stack(
            [
                rigid_field_numpy(masses, body, rot, trans, targets, name)
                for rot, trans in frames
            ]
        )
        s = scale_si(float(masses.sum()), 1.0, _POWER[name])
        rows.append(
            {
                "observable": name,
                "n_times": len(V1_TIMES_S),
                "n_targets": len(targets),
                "max_normalised_error": normalised_error(cont, ref, s),
                "scale_S": s,
            }
        )
    return {"rows": rows, "threshold": THR_V1_INDEP}


def test_v1_independent_newton_sum_unsorted_duplicate_times():
    """M1 a: independent oracle, error <= THR_V1_INDEP for all 3 observables."""
    for row in measure_v1_independent()["rows"]:
        assert row["max_normalised_error"] <= THR_V1_INDEP, row


# ---------------------------------------------------------------------------
# V2: periodic Fourier vs closed form, mpmath oracle, convergence
# ---------------------------------------------------------------------------

V2_TARGETS = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, -0.2, 0.1]])


@functools.cache
def _v2_oracle(delta_s: float, observable: str, panels: int = 64, dps: int = 50):
    """Return ``(n_target, n_comp, n_order)`` complex amplitudes ``A_n`` (mp)."""
    with mp.workdps(dps):
        out = np.zeros(
            (len(V2_TARGETS), _COMP[observable], len(ORDERS_V2)), dtype=np.complex128
        )
        for it, tgt in enumerate(V2_TARGETS):
            tgt_mp = [mp.mpf(x) for x in tgt]
            for k in range(_COMP[observable]):

                def series(t, k=k, tgt_mp=tgt_mp):
                    src = [
                        (
                            mp.mpf(1),
                            [mp.cos(2 * mp.pi * t), mp.sin(2 * mp.pi * t), mp.mpf(0)],
                        )
                    ]
                    return field_components(src, tgt_mp, observable)[k]

                coeffs = periodic_coefficients(
                    series, 1, delta_s, ORDERS_V2, panels=panels
                )
                for io, n in enumerate(ORDERS_V2):
                    amp = coeffs[io] * (2 if n > 0 else 1)
                    out[it, k, io] = complex(amp)
        return out


def _v2_got_ref(observable: str = "acceleration", *, rtol=1e-10, max_nodes=65536):
    """Production periodic coefficients and their mpmath reference (same shape)."""
    model = ContinuousFourierModel(_circle_source(), circular_pose(), frame=WORLD)
    ev = _periodic(
        model,
        _framed(V2_TARGETS),
        orders=ORDERS_V2,
        observables=(observable,),
        rtol=rtol,
        max_nodes=max_nodes,
    )
    return _values(ev.run(), observable), _v2_oracle(0.0, observable)


def measure_v2(observable: str = "acceleration", *, rtol=1e-10, max_nodes=65536):
    """Max normalised error of the periodic path vs the mpmath oracle."""
    got, ref = _v2_got_ref(observable, rtol=rtol, max_nodes=max_nodes)
    s = scale_si(1.0, 1.0, _POWER[observable])
    return {
        "observable": observable,
        "rtol": rtol,
        "max_nodes": max_nodes,
        "max_normalised_error": normalised_error(got, ref, s),
        "scale_S": s,
    }


def measure_v2_negative_controls(observable: str = "acceleration") -> dict:
    """M3 g (V2): the judge must reject conjugate, 2x and sign-flipped answers."""
    got, ref = _v2_got_ref(observable)
    s = scale_si(1.0, 1.0, _POWER[observable])
    return {
        "observable": observable,
        "correct": normalised_error(got, ref, s),
        "wrong": {
            name: normalised_error(got, cand, s)
            for name, cand in wrong_answer_candidates(ref).items()
        },
        "threshold_wrong": THR_NEG,
        "scale_S": s,
    }


def test_v2_negative_controls_reject_wrong_answers():
    """M3 g (V2): the judge accepts the correct answer and rejects wrong ones."""
    row = measure_v2_negative_controls()
    assert row["correct"] <= THR_V2, row
    for name, err in row["wrong"].items():
        assert err > THR_NEG, (name, row)


def measure_v2_closed_form() -> dict:
    """Centred target: only n=1 is non-zero (closed form, period 1 s)."""
    model = ContinuousFourierModel(_circle_source(), circular_pose(), frame=WORLD)
    res = _periodic(
        model,
        _framed([[0.0, 0.0, 0.0]]),
        orders=ORDERS_V2,
        observables=("acceleration",),
    ).run()
    got = _values(res, "acceleration")[0]  # (3, nf)
    ref = np.array(circular_centre_closed_form(0.0)).T  # (3, 4)
    s = scale_si(1.0, 1.0, 2)
    err = float(np.max(np.abs(got[:, :4] - ref)) / s)
    zeros_err = float(np.max(np.abs(got[:, 4:])) / s)
    return {
        "max_normalised_error_n0_3": err,
        "max_normalised_error_n4_6": zeros_err,
        "scale_S": s,
    }


def measure_v2_quadrature_independence() -> dict:
    """mpmath quadrature sensitivity: 32 vs 128 panels, and 80 vs 100 digits."""
    s = scale_si(1.0, 1.0, 2)
    a = _v2_oracle(0.0, "acceleration", panels=32, dps=50)
    b = _v2_oracle(0.0, "acceleration", panels=128, dps=50)
    c = _v2_oracle(0.0, "acceleration", panels=64, dps=80)
    d = _v2_oracle(0.0, "acceleration", panels=64, dps=100)
    return {
        "panels_32_vs_128_normalised": float(np.max(np.abs(a - b)) / s),
        "dps_80_vs_100_normalised": float(np.max(np.abs(c - d)) / s),
    }


def measure_v2_convergence() -> dict:
    """Error ladder in rtol and in max_nodes (4 levels each)."""
    rtol_rows = [measure_v2(rtol=r) for r in RTOL_LADDER_V2]
    node_rows = []
    base = None
    for mn in MAX_NODES_LADDER_V2:
        model = ContinuousFourierModel(_circle_source(), circular_pose(), frame=WORLD)
        res = _periodic(
            model,
            _framed(V2_TARGETS),
            orders=ORDERS_V2,
            observables=("acceleration",),
            max_nodes=mn,
        ).run()
        got = _values(res, "acceleration")
        if base is None:
            base = got
        s = scale_si(1.0, 1.0, 2)
        node_rows.append(
            {
                "max_nodes": mn,
                "max_change_vs_first_normalised": float(np.max(np.abs(got - base)) / s),
            }
        )
    return {"rtol_ladder": rtol_rows, "max_nodes_ladder": node_rows}


def test_v2_closed_form_centred_target():
    row = measure_v2_closed_form()
    assert row["max_normalised_error_n0_3"] <= THR_V2, row
    assert row["max_normalised_error_n4_6"] <= THR_V2, row


@pytest.mark.parametrize(
    "observable",
    [
        "potential",
        "acceleration",
        pytest.param("gradient", marks=pytest.mark.slow),
    ],
)
def test_v2_mpmath_oracle_eccentric_targets(observable):
    row = measure_v2(observable)
    assert row["max_normalised_error"] <= THR_V2, row


@pytest.mark.slow
def test_v2_quadrature_independence_of_oracle():
    row = measure_v2_quadrature_independence()
    assert row["panels_32_vs_128_normalised"] <= 1e-12, row
    assert row["dps_80_vs_100_normalised"] <= 1e-10, row


@pytest.mark.slow
def test_v2_convergence_rtol_and_max_nodes():
    conv = measure_v2_convergence()
    errs = [r["max_normalised_error"] for r in conv["rtol_ladder"]]
    for prev, cur in zip(errs, errs[1:], strict=False):
        assert cur <= prev + THR_V2_CONV_SLACK, conv
    assert errs[-1] <= THR_V2, conv
    for r in conv["max_nodes_ladder"]:
        assert r["max_change_vs_first_normalised"] <= THR_V2, conv


# M1 b, c: rotating asymmetric 3-mass body, period 1 s (|omega| = 2 pi, so the pose
# repeats after one turn and the translation is periodic as well).
ROT_MASSES = np.array([1.0, 2.5, 0.7])
ROT_BODY = np.array([[0.3, 0.0, 0.1], [-0.2, 0.25, 0.0], [0.0, -0.4, 0.3]])
_ROT_AXIS_RAW = np.array([0.3, -0.2, 0.9])
ROT_AXIS = _ROT_AXIS_RAW / np.linalg.norm(_ROT_AXIS_RAW)
ROT_OMEGA = 2.0 * np.pi * ROT_AXIS
ROT_TARGETS = np.array([[2.0, 0.0, 0.0], [1.5, -1.0, 0.7], [-1.2, 1.1, 0.4]])


def _rot_centre_numpy(t: np.ndarray) -> np.ndarray:
    """Body-frame origin path c(t) (numpy), period 1 s."""
    return np.stack(
        [
            0.5 * np.cos(2 * np.pi * t),
            0.4 * np.sin(2 * np.pi * t + 0.3),
            0.1 * np.sin(4 * np.pi * t),
        ],
        axis=1,
    )


def _rot_pose(times):
    t = np.asarray(times.to_value(u.s), dtype=np.float64)
    rot = Rotation.from_rotvec(t[:, None] * ROT_OMEGA[None, :])
    return rot, _rot_centre_numpy(t) * u.m


@functools.cache
def _rot_oracle(observable: str, dps: int = 30, panels: int = 64) -> np.ndarray:
    """Independent periodic coefficients ``(target, comp, order)`` (mpmath).

    Positions follow ``x = R(t) x_b + c(t)`` with ``R`` from an mpmath Rodrigues
    formula; the all-order quadrature shares its samples between orders.
    """
    with mp.workdps(dps):
        omega = [2 * mp.pi * mp.mpf(float(a)) for a in ROT_AXIS]
        masses = [mp.mpf(float(m)) for m in ROT_MASSES]
        body = [[mp.mpf(float(x)) for x in row] for row in ROT_BODY]
        targets = [[mp.mpf(float(x)) for x in row] for row in ROT_TARGETS]

        def field(t):
            rot = rodrigues_mp([w * t for w in omega])
            c = [
                mp.mpf(1) / 2 * mp.cos(2 * mp.pi * t),
                mp.mpf(2) / 5 * mp.sin(2 * mp.pi * t + mp.mpf(3) / 10),
                mp.mpf(1) / 10 * mp.sin(4 * mp.pi * t),
            ]
            pos = [
                [sum(rot[i][j] * p[j] for j in range(3)) + c[i] for i in range(3)]
                for p in body
            ]
            sources = list(zip(masses, pos, strict=True))
            out = []
            for tg in targets:
                out.extend(field_components(sources, tg, observable))
            return out

        coeffs = periodic_coefficients_all(field, 1, ORDERS_V2, panels=panels)
        ncomp = _COMP[observable]
        arr = np.zeros((len(targets), ncomp, len(ORDERS_V2)), dtype=np.complex128)
        for io, n in enumerate(ORDERS_V2):
            factor = 2 if n > 0 else 1
            for it in range(len(targets)):
                for k in range(ncomp):
                    arr[it, k, io] = complex(coeffs[io][it * ncomp + k]) * factor
        return arr


@functools.cache
def _rot_production(observable: str) -> np.ndarray:
    """Production periodic coefficients for the rotating body (cached, read-only)."""
    model = ContinuousFourierModel(
        PointMassCloud(ROT_MASSES, ROT_BODY), _rot_pose, frame=WORLD
    )
    s = scale_si(float(ROT_MASSES.sum()), 1.0, _POWER[observable])
    res = _periodic(
        model,
        _framed(ROT_TARGETS),
        orders=ORDERS_V2,
        observables=(observable,),
        scale=s,
    ).run()
    got = _values(res, observable)
    got.setflags(write=False)
    return got


def measure_v2_rotating(observable: str) -> dict:
    """M1 b: periodic path for the rotating body vs the independent mpmath oracle."""
    s = scale_si(float(ROT_MASSES.sum()), 1.0, _POWER[observable])
    return {
        "observable": observable,
        "max_normalised_error": normalised_error(
            _rot_production(observable), _rot_oracle(observable), s
        ),
        "scale_S": s,
    }


@pytest.mark.parametrize("observable", ["potential", "acceleration", "gradient"])
def test_v2_rotating_asymmetric_body_matches_mpmath(observable):
    """M1 b: each observable run separately, error <= THR_V2_ROT."""
    row = measure_v2_rotating(observable)
    assert row["max_normalised_error"] <= THR_V2_ROT, row


def measure_v2_gradient_structure() -> dict:
    """M1 c: symmetry ``T_ij = T_ji`` and tracelessness of the periodic gradient."""
    got = _rot_production("gradient")  # (target, 9, order)
    s = scale_si(float(ROT_MASSES.sum()), 1.0, 3)
    tens = got.reshape(len(ROT_TARGETS), 3, 3, len(ORDERS_V2))
    return {
        "symmetry_normalised": float(
            np.max(np.abs(tens - np.swapaxes(tens, 1, 2))) / s
        ),
        "trace_normalised": float(
            np.max(np.abs(tens[:, 0, 0] + tens[:, 1, 1] + tens[:, 2, 2])) / s
        ),
        "threshold": THR_IDENT,
        "scale_S": s,
    }


def test_v2_gradient_symmetric_and_traceless():
    """M1 c: gradient of the rotating body is symmetric and traceless (<= 1e-12 S)."""
    row = measure_v2_gradient_structure()
    assert row["symmetry_normalised"] <= THR_IDENT, row
    assert row["trace_normalised"] <= THR_IDENT, row


# M2/d: close approach at distance d from the orbit (target at x = 1 + d on the x axis).
NEAR_TARGET_X = {0.1: 1.1, 0.03: 1.03}
NEAR_DISTANCES_M = (0.1, 0.03)
RTOL_LADDER_NEAR = (1e-4, 1e-6, 1e-8, 1e-10)
NEAR_PANELS = 2000


@functools.cache
def _near_oracle(distance_m: float, panels: int = NEAR_PANELS) -> np.ndarray:
    """Independent acceleration coefficients ``(comp, order)`` for the near target."""
    with mp.workdps(40):
        x = NEAR_TARGET_X[distance_m]
        tgt = [mp.mpf(x), mp.mpf(0), mp.mpf(0)]

        def field(t):
            src = [
                (
                    mp.mpf(1),
                    [mp.cos(2 * mp.pi * t), mp.sin(2 * mp.pi * t), mp.mpf(0)],
                )
            ]
            return field_components(src, tgt, "acceleration")

        coeffs = periodic_coefficients_all(field, 1, ORDERS_V2, panels=panels)
        out = np.zeros((3, len(ORDERS_V2)), dtype=np.complex128)
        for io, n in enumerate(ORDERS_V2):
            factor = 2 if n > 0 else 1
            for k in range(3):
                out[k, io] = complex(coeffs[io][k]) * factor
        return out


def measure_d_near_rtol(distance_m: float) -> dict:
    """M3 d: error and node count vs rtol at a close (0.1 m / 0.03 m) target."""
    ref = _near_oracle(distance_m)
    s = scale_si(1.0, 1.0, 2)
    rows = []
    for rtol in RTOL_LADDER_NEAR:
        model = ContinuousFourierModel(_circle_source(), circular_pose(), frame=WORLD)
        ev = _periodic(
            model,
            _framed([[NEAR_TARGET_X[distance_m], 0.0, 0.0]]),
            orders=ORDERS_V2,
            observables=("acceleration",),
            rtol=rtol,
        )
        res, diag = ev.run_with_diagnostics()
        got = _values(res, "acceleration")[0]
        info = diag["acceleration"]
        rows.append(
            {
                "rtol": rtol,
                "max_normalised_error": normalised_error(got, ref, s),
                "nodes": len(info["nodes"]),
                "status": info["status"],
                "reason": info["reason"],
            }
        )
    oracle_check = normalised_error(_near_oracle(distance_m, NEAR_PANELS // 2), ref, s)
    return {
        "distance_m": distance_m,
        "rows": rows,
        "oracle_panels": NEAR_PANELS,
        "oracle_halved_panels_normalised": oracle_check,
        "scale_S": s,
    }


@pytest.mark.slow
@pytest.mark.parametrize("distance_m", NEAR_DISTANCES_M)
def test_d_near_target_rtol_monotone_and_nodes(distance_m):
    """M3 d: error is non-increasing as rtol tightens; nodes do not decrease."""
    row = measure_d_near_rtol(distance_m)
    errs = [r["max_normalised_error"] for r in row["rows"]]
    nodes = [r["nodes"] for r in row["rows"]]
    for prev, cur in zip(errs, errs[1:], strict=False):
        assert cur <= prev + THR_V2_CONV_SLACK, row
    assert errs[-1] <= THR_V2, row
    assert nodes[0] <= nodes[-1], row
    assert row["oracle_halved_panels_normalised"] <= 1e-10, row


# ---------------------------------------------------------------------------
# V3: finite-window Fourier vs mpmath oracle
# ---------------------------------------------------------------------------

V3_BS = (0.5, 1.0, 2.0)
V3_FREQS = (0.0, 0.125, 1.0, 4.0)
V3_WINDOW = (-3.0, 3.0)


def _passage_model(b: float) -> ContinuousFourierModel:
    pose = constant_velocity_pose(velocity_m_s=(1.0, 0.0, 0.0), origin_m=(0.0, b, 0.0))
    return ContinuousFourierModel(_circle_source(), pose, frame=WORLD)


def _window_eval(
    model,
    *,
    t0,
    t1,
    rtol=1e-10,
    limit=None,
    freqs=V3_FREQS,
    observables=("acceleration",),
    tau_min=0.25,
    budget=MEM,
    targets=None,
):
    width = t1 - t0
    obs = observables[0]
    scale = scale_si(1.0, 1.0, _POWER[obs]) * width
    if targets is None:
        targets = _framed([[0.0, 0.0, 0.0]])
    # The window transform carries the response unit times seconds.
    return model.window_at(
        targets,
        epoch=EPOCH0,
        tref=EPOCH0,
        t_start=t0 * u.s,
        t_stop=t1 * u.s,
        frequencies=np.asarray(freqs) * u.Hz,
        tau_min=tau_min * u.s,
        atol=1e-10 * scale * _UNIT[obs] * u.s,
        rtol=rtol,
        limit=limit,
        max_evaluations=DEFAULT_MAX_EVALUATIONS,
        memory_budget_bytes=budget,
        observables=observables,
    )


@functools.cache
def _v3_oracle(
    b: float, t0: float, t1: float, panels: int = 400, t_peak: float = 0.0
) -> np.ndarray:
    """(3, nf) complex window transforms of the passage acceleration (mp).

    The source sits at ``(t - t_peak, b, 0)``, so the closest approach is at
    ``t = t_peak``.
    """
    with mp.workdps(40):
        out = np.zeros((3, len(V3_FREQS)), dtype=np.complex128)
        bm = mp.mpf(b)
        tp = mp.mpf(t_peak)
        for k in range(3):

            def series(t, k=k):
                src = [(mp.mpf(1), [t - tp, bm, mp.mpf(0)])]
                return field_components(src, [mp.mpf(0)] * 3, "acceleration")[k]

            vals = window_transform(series, t0, t1, V3_FREQS, 0.0, panels=panels)
            out[k] = [complex(v) for v in vals]
        return out


def measure_v3(
    b: float, *, t0=V3_WINDOW[0], t1=V3_WINDOW[1], rtol=1e-10, limit=None
) -> dict:
    res = _window_eval(_passage_model(b), t0=t0, t1=t1, rtol=rtol, limit=limit).run()
    got = _values(res, "acceleration")[0]  # (3, nf)
    ref = _v3_oracle(b, t0, t1)
    sw = scale_si(1.0, 1.0, 2) * (t1 - t0)
    return {
        "b_m": b,
        "window_s": [t0, t1],
        "rtol": rtol,
        "limit": limit,
        "max_normalised_error": float(np.max(np.abs(got - ref)) / sw),
        "scale_SW": sw,
    }


def measure_v3_convergence(b: float = 1.0) -> dict:
    """rtol ladder, and invariance under a change of the initial panel split.

    The initial split is set by ``tau_min``; halving or doubling it must not move
    the result beyond THR_V3 (normalised by S*W).
    """
    rtol_rows = [measure_v3(b, rtol=r) for r in RTOL_LADDER_V3]
    sw = scale_si(1.0, 1.0, 2) * 6.0
    base = _values(
        _window_eval(_passage_model(b), t0=-3.0, t1=3.0, tau_min=0.25).run(),
        "acceleration",
    )
    tau_rows = {}
    for tau in (0.125, 0.5):
        other = _values(
            _window_eval(_passage_model(b), t0=-3.0, t1=3.0, tau_min=tau).run(),
            "acceleration",
        )
        tau_rows[f"tau_min_{tau}_vs_0.25"] = normalised_error(other, base, sw)
    return {"rtol_ladder": rtol_rows, "tau_min_partition_normalised": tau_rows}


def measure_v3_negative_controls(b: float = 1.0) -> dict:
    """M3 g (V3): the judge must reject conjugate, 2x and sign-flipped answers."""
    got = _values(
        _window_eval(_passage_model(b), t0=-3.0, t1=3.0).run(), "acceleration"
    )
    ref = _v3_oracle(b, -3.0, 3.0)
    sw = scale_si(1.0, 1.0, 2) * 6.0
    return {
        "b_m": b,
        "correct": normalised_error(got[0], ref, sw),
        "wrong": {
            name: normalised_error(got[0], cand, sw)
            for name, cand in wrong_answer_candidates(ref).items()
        },
        "threshold_wrong": THR_NEG,
    }


def test_v3_negative_controls_reject_wrong_answers():
    """M3 g (V3): the window judge accepts the correct transform, rejects wrong ones."""
    row = measure_v3_negative_controls()
    assert row["correct"] <= THR_V3, row
    for name, err in row["wrong"].items():
        assert err > THR_NEG, (name, row)


def measure_v3_edge() -> dict:
    """Window ending at the closest approach of b = 0.5 m (large edge value)."""
    return measure_v3(0.5, t0=-1.0, t1=0.0)


@pytest.mark.parametrize("b", V3_BS)
def test_v3_passage_mpmath_oracle(b):
    row = measure_v3(b)
    assert row["max_normalised_error"] <= THR_V3, row


def test_v3_edge_case_large_endpoint_value():
    row = measure_v3_edge()
    assert row["max_normalised_error"] <= THR_V3, row


@pytest.mark.slow
def test_v3_convergence_rtol_ladder_and_tau_min_partition():
    """rtol ladder is non-increasing; the result is invariant to the panel split.

    This replaces an earlier check that doubled the ``quad_vec`` interval limit.
    The limit only caps the number of intervals, and it does not depend on the
    initial partition floor. Changing the initial split through ``tau_min`` is a
    more sensitive test of partition invariance, so it is used instead.
    """
    conv = measure_v3_convergence()
    errs = [r["max_normalised_error"] for r in conv["rtol_ladder"]]
    for prev, cur in zip(errs, errs[1:], strict=False):
        assert cur <= prev + THR_V2_CONV_SLACK, conv
    for key, err in conv["tau_min_partition_normalised"].items():
        assert err <= THR_V3, (key, conv)


# ---------------------------------------------------------------------------
# V4: direct periodic Fourier vs discrete FFT (M3-B), helpful not oracle
# ---------------------------------------------------------------------------


def measure_v4(n_samples: int, target=(0.0, 0.0, 0.0)) -> dict:
    dt = PERIOD_S / n_samples
    t = build_uniform_time_s((0, 0), dt, n_samples)
    c = _circle_xyz(t)
    transform = PrescribedRigidTransform(
        t, c, np.repeat(np.eye(3)[None], n_samples, axis=0)
    )
    targets = TargetPoints(np.array([list(target)]) * u.m, ("t0",), frame=WORLD)
    ts = (
        rigid_motion(
            _circle_source(),
            transform,
            frame=WORLD,
            sampling=TimeSampling(epoch=EPOCH0, dt=dt * u.s, count=n_samples),
        )
        .at(targets, observables=("acceleration",))
        .run()
    )
    spec = ts.spectrum("periodic", tref=(0, 0))
    disc = np.asarray(spec.acceleration.value)[0, :, :9]  # (3, 9)
    model = ContinuousFourierModel(_circle_source(), circular_pose(), frame=WORLD)
    res = _periodic(
        model,
        _framed([list(target)]),
        orders=np.arange(0, 9),
        observables=("acceleration",),
    ).run()
    cont = _values(res, "acceleration")[0][:, :9]
    s = scale_si(1.0, 1.0, 2)
    per_order = [float(np.max(np.abs(disc[:, n] - cont[:, n])) / s) for n in range(9)]
    return {
        "n_samples": n_samples,
        "max_normalised_error_n_le_8": float(max(per_order)),
        "per_order_normalised": per_order,
        "discrete_bin_count": int(disc.shape[1]),
    }


def test_v4_direct_vs_fft_low_orders():
    row = measure_v4(1024)
    assert row["max_normalised_error_n_le_8"] <= THR_V4, row


def test_v4_error_vs_sample_count_recorded():
    rows = [measure_v4(n) for n in (512, 1024, 2048)]
    for r in rows:
        assert r["max_normalised_error_n_le_8"] <= THR_V4, r


def test_v4_near_passage_fft_error_falls_with_samples():
    """M2 e: at 0.1 m from the orbit, each doubling of N cuts the error by >= 10x."""
    rows = [measure_v4(n, target=(1.1, 0.0, 0.0)) for n in (128, 256, 512)]
    errs = [r["max_normalised_error_n_le_8"] for r in rows]
    assert errs[1] <= errs[0] / 10, rows
    assert errs[2] <= errs[1] / 10, rows
    assert errs[2] <= 1e-9, rows


# ---------------------------------------------------------------------------
# V5: units, frame basis, phase convention, epoch shift
# ---------------------------------------------------------------------------


def _first_unit(matrix) -> str:
    return str(np.asarray(matrix.units).reshape(-1)[0])


def measure_v5_units() -> dict:
    """Payload unit per observable. One observable per run: a single atol is
    checked against the response unit, so potential and acceleration cannot share
    one evaluation.
    """
    model = ContinuousFourierModel(_circle_source(), circular_pose(), frame=WORLD)
    per: dict[str, dict] = {}
    for name in ("potential", "acceleration", "gradient"):
        try:
            res = _periodic(
                model,
                _framed([[0.0, 0.0, 0.0]]),
                orders=np.arange(0, 3),
                observables=(name,),
            ).run()
            per[f"periodic_{name}"] = {
                "unit": _first_unit(getattr(res, name)),
                "expected": str(_UNIT[name]),
            }
        except Exception as exc:  # recorded as a failure, not hidden
            per[f"periodic_{name}"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "expected": str(_UNIT[name]),
            }
    passage = _passage_model(1.0)
    win = _window_eval(passage, t0=-3.0, t1=3.0, observables=("acceleration",)).run()
    per["window_acceleration"] = {
        "unit": _first_unit(win.acceleration),
        "expected": str(u.m / u.s),
    }
    return per


def _frame_setup():
    basis = Rotation.from_euler("z", 0.4).as_matrix()
    origin = np.array([1.0, -0.5, 2.0])
    return Frame("local", origin=origin * u.m, basis=basis, parent=WORLD), basis, origin


def measure_v5_frame() -> dict:
    """Output components are in the frame basis: a_local = B^T a_world."""
    frame, basis, origin = _frame_setup()
    target_w = np.array([[0.3, 0.1, -0.2]])
    base = _periodic(
        ContinuousFourierModel(_circle_source(), circular_pose(), frame=WORLD),
        _framed(target_w),
        orders=np.arange(0, 3),
        observables=("acceleration",),
    ).run()
    a_w = _values(base, "acceleration")[0]  # (3, nf)

    def pose_local(times):
        t = np.asarray(times.to_value(u.s), dtype=np.float64)
        c_world = _circle_xyz(t)
        c_local = (c_world - origin[None, :]) @ basis  # rows: B^T (x - o)
        return Rotation.identity(len(t)), c_local * u.m

    local = _periodic(
        ContinuousFourierModel(_circle_source(), pose_local, frame=frame),
        _framed(target_w),
        orders=np.arange(0, 3),
        observables=("acceleration",),
    ).run()
    a_l = _values(local, "acceleration")[0]
    expected = basis.T @ a_w
    s = scale_si(1.0, 1.0, 2)
    return {
        "max_normalised_error": float(np.max(np.abs(a_l - expected)) / s),
        "scale_S": s,
    }


def measure_v5_phase(tau_s: float = 0.1) -> dict:
    """tref shifted by tau multiplies C_n by exp(+2 pi i n tau / T)."""
    model = ContinuousFourierModel(_circle_source(), circular_pose(), frame=WORLD)
    base = _periodic(
        model,
        _framed([[0.0, 0.0, 0.0]]),
        orders=np.arange(0, 4),
        observables=("acceleration",),
    ).run()
    shifted = _periodic(
        model,
        _framed([[0.0, 0.0, 0.0]]),
        orders=np.arange(0, 4),
        observables=("acceleration",),
        tref=LIGOTimeGPS(0, int(round(tau_s * 1e9))),
    ).run()
    a0 = _values(base, "acceleration")[0]
    a1 = _values(shifted, "acceleration")[0]
    n = np.arange(0, 4)
    factor = np.exp(2j * np.pi * n * tau_s / PERIOD_S)
    expected = a0 * factor[None, :]
    s = scale_si(1.0, 1.0, 2)
    return {
        "tau_s": tau_s,
        "max_normalised_error": float(np.max(np.abs(a1 - expected)) / s),
        "scale_S": s,
    }


def measure_v5_epoch(tau_s: float = 0.1) -> dict:
    """Moving epoch by tau with pose(t + tau) leaves the coefficients unchanged."""
    base_model = ContinuousFourierModel(_circle_source(), circular_pose(), frame=WORLD)
    base = _periodic(
        base_model,
        _framed([[0.0, 0.0, 0.0]]),
        orders=np.arange(0, 4),
        observables=("acceleration",),
    ).run()
    inner = circular_pose()

    def shifted_pose(times):
        t = np.asarray(times.to_value(u.s), dtype=np.float64) + tau_s
        return inner(t * u.s)

    model = ContinuousFourierModel(_circle_source(), shifted_pose, frame=WORLD)
    moved = _periodic(
        model,
        _framed([[0.0, 0.0, 0.0]]),
        orders=np.arange(0, 4),
        observables=("acceleration",),
        epoch=LIGOTimeGPS(0, int(round(tau_s * 1e9))),
    ).run()
    s = scale_si(1.0, 1.0, 2)
    err = float(
        np.max(
            np.abs(_values(moved, "acceleration") - _values(base, "acceleration")) / s
        )
    )
    return {"tau_s": tau_s, "max_normalised_error": err, "scale_S": s}


def test_v5_units():
    per = measure_v5_units()
    for name, row in per.items():
        assert "error" not in row, (name, row)
        assert _same_unit(row["unit"], row["expected"]), (name, row)


def _same_unit(a: str, b: str) -> bool:
    return u.Unit(a) == u.Unit(b)


def test_v5_frame_basis():
    """Frame basis check: the output components are expressed in the local frame.

    The expected values also come from the Frame API (``_frame_setup`` builds the
    frame with it). This test therefore checks that the implementation agrees with
    the Frame API. It is not an independent verification of the Frame API itself.
    """
    row = measure_v5_frame()
    assert row["max_normalised_error"] <= THR_V5, row


def test_v5_phase_convention():
    row = measure_v5_phase()
    assert row["max_normalised_error"] <= THR_V5, row


def test_v5_epoch_shift_invariance():
    row = measure_v5_epoch()
    assert row["max_normalised_error"] <= THR_V5, row


def measure_v5_closed_form_nonzero_delta(delta_s: float = 0.1) -> dict:
    """M2 k: centred target with Delta = epoch - tref = delta_s != 0.

    Compares complex coefficients and the modulus of A_1 (which must equal G).
    """
    model = ContinuousFourierModel(_circle_source(), circular_pose(), frame=WORLD)
    res = _periodic(
        model,
        _framed([[0.0, 0.0, 0.0]]),
        orders=ORDERS_V2,
        observables=("acceleration",),
        epoch=LIGOTimeGPS(0, int(round(delta_s * 1e9))),
        tref=EPOCH0,
    ).run()
    got = _values(res, "acceleration")[0]  # (3, nf)
    ref = np.array(circular_centre_closed_form(delta_s)).T  # (3, 4)
    s = scale_si(1.0, 1.0, 2)
    # Modulus of each component of A_1 must match the closed form (phase-free check).
    modulus_err = float(np.max(np.abs(np.abs(got[:, 1]) - np.abs(ref[:, 1]))) / s)
    return {
        "delta_s": delta_s,
        "max_normalised_error_complex_n0_3": normalised_error(got[:, :4], ref, s),
        "max_normalised_error_n4_6": normalised_error(
            got[:, 4:], np.zeros_like(got[:, 4:]), s
        ),
        "modulus_A1_minus_closed_form_normalised": modulus_err,
        "scale_S": s,
    }


def test_v5_closed_form_nonzero_delta():
    row = measure_v5_closed_form_nonzero_delta()
    assert row["max_normalised_error_complex_n0_3"] <= THR_V5, row
    assert row["max_normalised_error_n4_6"] <= THR_V5, row
    assert row["modulus_A1_minus_closed_form_normalised"] <= THR_V5, row


def _chain_local_frame() -> Frame:
    """Two-level parent chain world -> mid -> local with 3-axis rotations."""
    mid = Frame(
        "mid",
        origin=np.array([0.5, -1.0, 0.25]) * u.m,
        basis=Rotation.from_euler("xyz", [0.3, -0.5, 0.7]).as_matrix(),
        parent=WORLD,
    )
    return Frame(
        "local",
        origin=np.array([-0.3, 0.6, 1.1]) * u.m,
        basis=Rotation.from_euler("zyx", [-0.4, 0.2, 0.9]).as_matrix(),
        parent=mid,
    )


def _in_frame(values: np.ndarray, observable: str, dest: Frame) -> np.ndarray:
    """Re-express the world-basis payload in ``dest`` with the Frame API.

    ``values`` is ``(comp, nf)``; vectors use ``transform_vectors_m`` and gradients
    use ``transform_tensors``, applied to the real and imaginary parts separately.
    """
    nf = values.shape[-1]
    if observable == "acceleration":
        vec = values.T  # (nf, 3)
        out = WORLD.transform_vectors_m(
            vec.real, dest
        ) + 1j * WORLD.transform_vectors_m(vec.imag, dest)
        return out.T
    ten = values.reshape(3, 3, nf).transpose(2, 0, 1)  # (nf, 3, 3)
    out = WORLD.transform_tensors(ten.real, dest) + 1j * WORLD.transform_tensors(
        ten.imag, dest
    )
    return out.transpose(1, 2, 0).reshape(9, nf)


def measure_v5_frame_chain() -> dict:
    """M1 j: 3-axis rotation and two parent levels, periodic and window paths.

    The body positions are given in the local frame (via the Frame point API), and
    the expectation is the world-frame result re-expressed with the Frame API.
    """
    local = _chain_local_frame()
    tgt_w = np.array([[0.3, 0.1, -0.2]])

    def pose_local(times):
        t = np.asarray(times.to_value(u.s), dtype=np.float64)
        c_local = WORLD.transform_points_m(_circle_xyz(t), local)
        return Rotation.identity(len(t)), c_local * u.m

    out: dict[str, float] = {}
    for name in ("acceleration", "gradient"):
        s = scale_si(1.0, 1.0, _POWER[name])
        world_model = ContinuousFourierModel(
            _circle_source(), circular_pose(), frame=WORLD
        )
        local_model = ContinuousFourierModel(_circle_source(), pose_local, frame=local)
        base = _values(
            _periodic(
                world_model,
                _framed(tgt_w),
                orders=np.arange(0, 3),
                observables=(name,),
                scale=s,
            ).run(),
            name,
        )[0]
        got = _values(
            _periodic(
                local_model,
                _framed(tgt_w),
                orders=np.arange(0, 3),
                observables=(name,),
                scale=s,
            ).run(),
            name,
        )[0]
        out[f"periodic_{name}"] = normalised_error(got, _in_frame(base, name, local), s)

        base_w = _values(
            _window_eval(
                world_model,
                t0=-2.0,
                t1=2.0,
                observables=(name,),
                targets=_framed(tgt_w),
            ).run(),
            name,
        )[0]
        got_w = _values(
            _window_eval(
                local_model,
                t0=-2.0,
                t1=2.0,
                observables=(name,),
                targets=_framed(tgt_w),
            ).run(),
            name,
        )[0]
        out[f"window_{name}"] = normalised_error(
            got_w, _in_frame(base_w, name, local), s
        )
    out["threshold"] = THR_FRAME
    return out


def test_v5_frame_chain_general_rotation_two_parent_levels():
    """M1 j: periodic and window, acceleration and gradient, <= 1e-12 S."""
    row = measure_v5_frame_chain()
    for key, err in row.items():
        if key != "threshold":
            assert err <= THR_FRAME, (key, row)


# ---------------------------------------------------------------------------
# V6: G applied exactly once
# ---------------------------------------------------------------------------


def measure_v6(factor: float = 1.0, monkeypatch=None) -> dict:
    """Static source, one G application; compare with point_acceleration."""
    source = PointMassCloud(np.array([2.0]), np.array([[0.0, 0.0, 0.0]]))
    tgt = np.array([[1.0, 0.0, 0.0]])
    # point_acceleration returns a plain SI ndarray (G applied once inside).
    ref = np.asarray(point_acceleration(source, NumericalTargets(tgt))).reshape(3)
    if monkeypatch is not None:
        import gragra.observables.rigid_motion as rm

        monkeypatch.setattr(rm, "G_SI", rm.G_SI * factor)
    res = _periodic(
        ContinuousFourierModel(source, static_pose(), frame=WORLD),
        _framed(tgt),
        orders=np.array([0]),
        observables=("acceleration",),
    ).run()
    got = _values(res, "acceleration")[0][:, 0]
    return {"factor": factor, "got": got.real.tolist(), "ref": ref.tolist()}


def test_v6_g_applied_once_matches_point_acceleration():
    row = measure_v6()
    s = scale_si(2.0, 1.0, 2)
    err = float(np.max(np.abs(np.asarray(row["got"]) - np.asarray(row["ref"]))) / s)
    assert err <= THR_V6, (row, err)


def test_v6_g_scaling_is_linear(monkeypatch):
    """Doubling G_SI in the observable layer doubles the result (not 4x)."""
    base = np.asarray(measure_v6()["got"])
    doubled = np.asarray(measure_v6(2.0, monkeypatch=monkeypatch)["got"])
    s = scale_si(2.0, 1.0, 2)
    # Zero components are allowed, so compare against 2 * base in absolute terms.
    assert float(np.max(np.abs(doubled - 2.0 * base)) / s) <= THR_V6, (base, doubled)
    assert float(np.max(np.abs(base))) > 0.0
    assert gravitational_constant() > 0


def test_v6_negative_control_rejects_double_g():
    """M3 g (V6): an answer with G applied twice (ref * G) is rejected by the judge."""
    row = measure_v6()
    s = scale_si(2.0, 1.0, 2)
    got = np.asarray(row["got"], dtype=np.float64)
    ref = np.asarray(row["ref"], dtype=np.float64)
    assert normalised_error(got, ref, s) <= THR_V6, row
    double_g = ref * float(gravitational_constant())
    assert normalised_error(got, double_g, s) > THR_NEG, row


# ---------------------------------------------------------------------------
# V7: memory budget and performance (record only, except MemoryError)
# ---------------------------------------------------------------------------


def _periodic_run(
    n_targets: int, n_orders: int, *, budget=MEM, observables=("acceleration",)
):
    rng = np.random.default_rng(0)
    tgt = rng.uniform(-0.5, 0.5, size=(n_targets, 3)) + np.array([0.0, 0.0, 2.0])
    model = ContinuousFourierModel(_circle_source(), circular_pose(), frame=WORLD)
    return _periodic(
        model,
        _framed(tgt),
        orders=np.arange(0, n_orders),
        observables=observables,
        budget=budget,
    ).run()


def test_v7_tiny_budget_raises_memory_error():
    with pytest.raises(MemoryError):
        _periodic_run(1, 3, budget=1024)


def test_v7_default_budget_completes_typical_case():
    res = _periodic_run(1, 6)
    assert _values(res, "acceleration").shape[0] == 1


def measure_v7_memory_grid() -> list[dict]:
    rows = []
    for nt in (1, 8, 64):
        for no in (1, 6, 32):
            tracemalloc.start()
            try:
                _periodic_run(nt, no)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            rows.append(
                {
                    "n_targets": nt,
                    "n_orders": no,
                    "tracemalloc_peak_bytes": peak,
                    "budget_bytes": MEM,
                    "peak_over_budget": peak / MEM,
                }
            )
    return rows


def _median_seconds(fn, repeats: int = 3) -> dict:
    """Median wall time of ``repeats`` calls; errors are recorded, not raised."""
    times: list[float] = []
    try:
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
    except Exception as exc:
        return {"median_s": None, "error": f"{type(exc).__name__}: {exc}"}
    return {"median_s": statistics.median(times), "repeats": repeats, "error": None}


def measure_v7_timing() -> dict:
    """Performance record only (no pass/fail). Each observable is timed separately."""

    def window():
        model = ContinuousFourierModel(_circle_source(), circular_pose(), frame=WORLD)
        # P5 timing case from the plan: 5 frequencies and 5 targets.
        targets = _framed(np.linspace(-0.4, 0.4, 5)[:, None] * [1.0, 0.0, 0.0])
        _window_eval(
            model,
            t0=-2.0,
            t1=2.0,
            freqs=(0.0, 0.5, 1.0, 2.0, 4.0),
            tau_min=0.25,
            targets=targets,
        ).run()

    periodic = {
        name: _median_seconds(
            lambda name=name: _periodic_run(8, 7, observables=(name,))
        )
        for name in ("potential", "acceleration", "gradient")
    }
    return {
        "periodic_P4_targets8_order6": periodic,
        "window_P5_targets5_freq5": _median_seconds(window),
    }


@pytest.mark.slow
def test_v7_memory_grid_recorded():
    rows = measure_v7_memory_grid()
    assert len(rows) == 9


# ---------------------------------------------------------------------------
# V8: negative tests and documented limits
# ---------------------------------------------------------------------------


def test_v8_nonperiodic_motion_rejected_by_p4():
    model = ContinuousFourierModel(
        _circle_source(),
        constant_velocity_pose(velocity_m_s=(1.0, 0.0, 0.0), origin_m=(0.0, 1.0, 0.0)),
        frame=WORLD,
    )
    with pytest.raises(ContinuousQuadratureError) as excinfo:
        _periodic(
            model,
            _framed([[0.0, 0.0, 0.0]]),
            orders=np.arange(0, 3),
            observables=("acceleration",),
        ).run()
    assert excinfo.value.reason == "not_periodic"


def test_v8_missing_tau_min_raises_value_error():
    model = ContinuousFourierModel(_circle_source(), circular_pose(), frame=WORLD)
    with pytest.raises(ValueError, match="tau_min is required"):
        _periodic(
            model,
            _framed([[0.0, 0.0, 0.0]]),
            orders=np.arange(0, 3),
            observables=("acceleration",),
            tau_min=None,
        ).run()


def test_v8_window_phase_range_exceeded_raises_value_error():
    model = ContinuousFourierModel(_circle_source(), circular_pose(), frame=WORLD)
    with pytest.raises(ValueError, match="window phase requires"):
        model.window_at(
            _framed([[0.0, 0.0, 0.0]]),
            epoch=LIGOTimeGPS(1_000_000_000, 0),
            tref=EPOCH0,
            t_start=-1.0 * u.s,
            t_stop=1.0 * u.s,
            frequencies=np.array([2e-3]) * u.Hz,
            tau_min=0.25 * u.s,
            atol=1e-10 * u.m / u.s,
            rtol=1e-10,
            max_evaluations=DEFAULT_MAX_EVALUATIONS,
            memory_budget_bytes=MEM,
        ).run()


def test_v8_insufficient_max_evaluations_raises():
    model = _passage_model(0.5)
    with pytest.raises(ValueError, match="max_evaluations is below the minimal count"):
        model.window_at(
            _framed([[0.0, 0.0, 0.0]]),
            epoch=EPOCH0,
            tref=EPOCH0,
            t_start=-3.0 * u.s,
            t_stop=3.0 * u.s,
            frequencies=np.asarray(V3_FREQS) * u.Hz,
            tau_min=0.25 * u.s,
            atol=1e-10 * u.m / u.s,
            rtol=1e-10,
            max_evaluations=10,
            memory_budget_bytes=MEM,
        ).run()


def test_v8_nonfinite_pose_raises():
    def bad_pose(times):
        n = len(times)
        return Rotation.identity(n), np.full((n, 3), np.nan) * u.m

    model = ContinuousFourierModel(_circle_source(), bad_pose, frame=WORLD)
    with pytest.raises(ValueError, match="must contain only finite"):
        _periodic(
            model,
            _framed([[0.0, 0.0, 0.0]]),
            orders=np.arange(0, 3),
            observables=("acceleration",),
        ).run()


# M2 h: close passage (b = 0.01 m) whose peak is moved off the coarse grid (t = 0.37 s).
V8_PEAK_S = 0.37
V8_TAU_MIN_S = (0.01, 0.1, 0.5, 1.0, 3.0, 6.0)


def _offcentre_passage_model(b: float, t_peak: float) -> ContinuousFourierModel:
    """Straight passage at speed 1 m/s whose closest approach is at ``t_peak``."""
    pose = constant_velocity_pose(
        velocity_m_s=(1.0, 0.0, 0.0), origin_m=(-t_peak, b, 0.0)
    )
    return ContinuousFourierModel(_circle_source(), pose, frame=WORLD)


def measure_v8_passage_tau_profile(
    b: float = 0.01, t_peak: float = V8_PEAK_S, tau_list=V8_TAU_MIN_S
) -> list[dict]:
    """M2 h: error vs tau_min for the off-grid close passage (recorded per point)."""
    model = _offcentre_passage_model(b, t_peak)
    ref = _v3_oracle(b, -3.0, 3.0, 3000, t_peak)
    sw = scale_si(1.0, 1.0, 2) * 6.0
    rows = []
    for tau in tau_list:
        try:
            res = _window_eval(model, t0=-3.0, t1=3.0, tau_min=tau).run()
        except Exception as exc:  # an exception is recorded and fails the test
            rows.append(
                {
                    "tau_min_s": tau,
                    "outcome": "exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        got = _values(res, "acceleration")[0]
        rows.append(
            {
                "tau_min_s": tau,
                "outcome": "returned",
                "max_normalised_error": normalised_error(got, ref, sw),
            }
        )
    return rows


@pytest.mark.slow
def test_v8_offgrid_close_passage_tau_min_profile():
    """M2 h: every tau_min in the list returns, and each point matches the oracle."""
    rows = measure_v8_passage_tau_profile()
    for row in rows:
        assert row["outcome"] == "returned", row
        assert row["max_normalised_error"] <= THR_V3, row


def measure_characterization_gaussian(tau_list=(1e-3, 1.0)) -> list[dict]:
    """M2 h(2): Gaussian fixture sigma = 1e-4 s, window [-2, 2] s, centre 0.37 s.

    Settings follow ``tests/test_finite_window_fourier.py``: epoch - tref = 0.25 s.
    Relative error is ``max|got - exp| / max|exp|``.

    The frequency set is fixed to ``[0, 1]`` Hz. The false convergence depends on
    the maximum frequency (it sets the initial panel split): with ``[0, 1, 4]`` Hz
    the same ``tau_min`` values happen to resolve the pulse (exploratory run,
    relative error about 1e-15), so that set is not a stable characterization.
    """
    signal = gaussian(1.0, 0.37, 1e-4)
    freqs = np.array([0.0, 1.0])
    closed = window_transform_closed_form(
        signal, delta_s=0.25, t0=-2.0, t1=2.0, frequencies_hz=freqs
    )
    expected = expand_components(closed, "acceleration")  # (nf, 3, 3)
    rows = []
    for tau in tau_list:
        val, diag = finite_window_fourier(
            make_response(signal),
            "acceleration",
            epoch=(1_000_000_000, 0),
            tref=(999_999_999, 750_000_000),
            t_start=-2.0 * u.s,
            t_stop=2.0 * u.s,
            frequencies=freqs * u.Hz,
            tau_min=tau * u.s,
            atol=1e-10 * 4.0 * u.m / u.s,
            rtol=1e-10,
            limit=40000,
            cache_size=8 * 1024**2,
            max_evaluations=2_000_000,
            memory_budget_bytes=MEM,
        )
        got = np.asarray(val.value, dtype=np.complex128)
        rows.append(
            {
                "tau_min_s": tau,
                "status": int(diag["status"]),
                "reason": diag["reason"],
                "relative_error": float(
                    np.max(np.abs(got - expected)) / np.max(np.abs(expected))
                ),
            }
        )
    return rows


def test_characterization_gaussian_tau_min_applicability():
    """Characterization, not a pass/fail criterion (contract sections 3 and 4.4).

    This fixes the applicability limit of ``tau_min``: a fine ``tau_min`` resolves
    the narrow pulse, while an oversized one still returns success with a wrong
    answer. The second point is recorded as a documented limit.
    """
    fine, coarse = measure_characterization_gaussian()
    assert fine["reason"] is None, fine
    assert fine["relative_error"] <= THR_V3, fine
    assert coarse["reason"] is None, coarse  # status success is kept
    assert coarse["relative_error"] > 0.5, coarse
