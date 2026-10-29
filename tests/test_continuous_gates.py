"""Regression gates for the M3 continuous Fourier integration (Revision R1).

These tests pin three review findings:

* The P5 pre-check floor counts the probe evaluation, so the floor is
  ``1 + coarse + 2·21·n_init``. The expected value is recomputed here from the
  written formula, not read from the implementation.
* The lane-specific memory budget reaches P3, P4, and P5 unchanged. Each lane
  checks it independently; the budgets are not summed.
* ``parameters`` records the absolute tolerance ``atol`` as a Quantity column
  at the end of the table, and the round trip keeps value and unit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _continuous_fixtures import circular_pose, constant, make_response
from astropy import units as u

import gragra._continuous_fourier as cf
from gragra._continuous_contract import ContinuousQuadratureError
from gragra._continuous_fourier import ContinuousFourierModel
from gragra._finite_window_fourier import finite_window_fourier
from gragra.geometry._frame import Frame
from gragra.gravity_sources import PointMassCloud
from gragra.targets._high_points import TargetPoints

WORLD = Frame("world")
EPOCH = (1_000_000_000, 0)
TREF = (999_999_999, 750_000_000)
GK21_NODES = 21
COARSE_MAX_POINTS = 4096
CACHE_BYTES = 8 * 1024**2
MEMORY_BYTES = 64 * 1024**2
RTOL = 1e-10
ATOL_COEFFICIENT = 1e-10


def _independent_floor(width_s: float, tau_s: float, f_max_hz: float) -> int:
    """Return the P5 pre-check floor, computed from the written formula.

    The floor is ``1 + coarse + 2·21·n_init``. The leading 1 is the probe at the
    window start. ``n_init = ceil(W/Δt_panel)`` with
    ``Δt_panel = min(8·τ_min, 1/(4·f_max))``, the second term dropped at
    ``f_max = 0``. The coarse grid has ``min(ceil(W/τ_min), 4096) + 1`` points.
    """
    panel = 8.0 * tau_s
    if f_max_hz > 0.0:
        panel = min(panel, 0.25 / f_max_hz)
    n_init = math.ceil(width_s / panel)
    coarse_points = min(math.ceil(width_s / tau_s), COARSE_MAX_POINTS) + 1
    return 1 + coarse_points + 2 * GK21_NODES * n_init


def _run_floor(
    freqs_hz: tuple[float, ...], width_s: float, tau_s: float, max_evaluations: int
):
    """Run P5 on a constant response with the given pre-check budget."""
    return finite_window_fourier(
        make_response(constant(1.0)),
        "acceleration",
        epoch=EPOCH,
        tref=TREF,
        t_start=0.0 * u.s,
        t_stop=width_s * u.s,
        frequencies=np.array(freqs_hz, dtype=np.float64) * u.Hz,
        tau_min=tau_s * u.s,
        atol=ATOL_COEFFICIENT * width_s * (u.m / u.s**2) * u.s,
        rtol=RTOL,
        limit=10000,
        cache_size=CACHE_BYTES,
        max_evaluations=max_evaluations,
        memory_budget_bytes=MEMORY_BYTES,
    )


# (frequencies, width, tau). The first case has f_max = 0 and n_init = 1.
# The second case has f_max = 4 Hz, so the frequency term sets the panel width.
FLOOR_CASES = [
    pytest.param((0.0,), 1.0, 0.5, id="zero-frequency"),
    pytest.param((0.0, 4.0), 1.0, 0.5, id="frequency-term"),
]


@pytest.mark.parametrize(("freqs_hz", "width_s", "tau_s"), FLOOR_CASES)
def test_p5_precheck_rejects_one_below_independent_floor(freqs_hz, width_s, tau_s):
    floor = _independent_floor(width_s, tau_s, max(freqs_hz))
    with pytest.raises(ValueError, match=f"minimal count {floor}"):
        _run_floor(freqs_hz, width_s, tau_s, max_evaluations=floor - 1)


@pytest.mark.parametrize(("freqs_hz", "width_s", "tau_s"), FLOOR_CASES)
def test_p5_precheck_passes_at_independent_floor(freqs_hz, width_s, tau_s):
    # At the floor the pre-check must not fire. The floor is a lower bound that
    # excludes subdivision, and quad_vec subdivides here even for a constant
    # integrand, so the run then stops at the budget. Revision R3 (blind v2
    # N-06): the budget is checked before each callback batch, so the run stops
    # with exactly ``floor`` evaluations performed. The batch that would exceed
    # the budget is never passed to the callback. The pre-check still let the run
    # start, and the counted evaluations are consistent with the floor.
    floor = _independent_floor(width_s, tau_s, max(freqs_hz))
    with pytest.raises(ContinuousQuadratureError) as info:
        _run_floor(freqs_hz, width_s, tau_s, max_evaluations=floor)
    assert info.value.reason == "max_evaluations_exceeded"
    assert info.value.diagnostics["evaluations"] == floor


def _model() -> ContinuousFourierModel:
    source = PointMassCloud(np.array([1.0]), np.array([[0.0, 0.0, 0.0]]))
    return ContinuousFourierModel(source, circular_pose(), frame=WORLD)


def _targets(count: int) -> TargetPoints:
    positions = np.array([[2.0 + i, 0.0, 0.0] for i in range(count)])
    ids = tuple(f"t{i}" for i in range(count))
    return TargetPoints(positions * u.m, ids=ids, frame=WORLD)


def _evaluation(
    kind: str, *, atol: u.Quantity, budget: int, tau_min: u.Quantity | None = None
):
    """Build (without running) a P4 or P5 evaluation with the given settings."""
    model = _model()
    targets = _targets(2)
    if kind == "periodic":
        return model.periodic_at(
            targets,
            epoch=EPOCH,
            tref=TREF,
            period=1.0 * u.s,
            orders=[0, 1],
            tau_min=0.1 * u.s if tau_min is None else tau_min,
            atol=atol,
            rtol=1e-6,
            max_nodes=1024,
            memory_budget_bytes=budget,
        )
    return model.window_at(
        targets,
        epoch=EPOCH,
        tref=TREF,
        t_start=0.0 * u.s,
        t_stop=1.0 * u.s,
        frequencies=np.array([0.0, 1.0]) * u.Hz,
        tau_min=0.25 * u.s if tau_min is None else tau_min,
        atol=atol,
        rtol=1e-6,
        memory_budget_bytes=budget,
    )


class _StopAfterCaptureError(Exception):
    """Raised by the lane spy to stop the run once the budget is captured."""


@pytest.mark.parametrize("kind", ["periodic", "window"])
def test_memory_budget_reaches_p3_and_lane_unchanged(monkeypatch, kind):
    # A value distinct from the default so that a silent default fallback fails.
    budget = 12_345_678
    captured: dict[str, int] = {}
    real_pose_response = cf.make_rigid_pose_response

    def spy_pose_response(*args, **kwargs):
        captured["p3"] = kwargs["memory_budget_bytes"]
        return real_pose_response(*args, **kwargs)

    def spy_lane(*_args, **kwargs):
        captured["lane"] = kwargs["memory_budget_bytes"]
        raise _StopAfterCaptureError

    lane_name = "periodic_fourier" if kind == "periodic" else "finite_window_fourier"
    monkeypatch.setattr(cf, "make_rigid_pose_response", spy_pose_response)
    monkeypatch.setattr(cf, lane_name, spy_lane)

    evaluation = _evaluation(kind, atol=1e-6 * u.m / u.s**2, budget=budget)
    with pytest.raises(_StopAfterCaptureError):
        evaluation.run()
    assert captured == {"p3": budget, "lane": budget}


# A budget far below any managed-array estimate. The lane must reject it before
# doing work, and the traceback must pass through the lane itself, so the
# rejection is not an accidental error from the P3 pose response.
TINY_LANE_BUDGET = 1024


@pytest.mark.parametrize(
    ("kind", "lane_name", "atol"),
    [
        ("periodic", "periodic_fourier", 1e-6 * u.m / u.s**2),
        ("window", "finite_window_fourier", 1e-6 * u.m / u.s**2 * u.s),
    ],
    ids=["periodic-P4", "finite-window-P5"],
)
def test_tiny_lane_budget_is_rejected_by_the_lane(kind, lane_name, atol):
    evaluation = _evaluation(kind, atol=atol, budget=TINY_LANE_BUDGET)
    with pytest.raises(MemoryError) as info:
        evaluation.run()
    lane_frames = {entry.name for entry in info.traceback}
    assert lane_name in lane_frames


@pytest.mark.parametrize("kind", ["periodic", "window"])
@pytest.mark.parametrize("shape", [(), (2, 3)], ids=["scalar", "target-component"])
def test_parameters_record_atol_round_trip(kind, shape):
    count = int(np.prod(shape)) if shape else 1
    values = 1e-9 * (np.arange(count, dtype=np.float64) + 1.0).reshape(shape)
    unit = u.m / u.s**2 if kind == "periodic" else u.m / u.s**2 * u.s
    atol = values * unit

    table = _evaluation(kind, atol=atol, budget=MEMORY_BYTES).parameters

    assert table.colnames[-1] == "atol"
    column = table["atol"]
    assert column.shape == (1,) + shape
    row = column[0]
    assert row.unit == unit
    np.testing.assert_array_equal(np.asarray(row.value), values)
    assert table["rtol"][0] == pytest.approx(1e-6)


def test_window_transform_rejects_unsorted_frequencies():
    """Frequencies must be strictly increasing, as the public docstring states."""
    import numpy as np
    from _continuous_fixtures import circular_pose
    from astropy import units as u
    from gwpy.time import LIGOTimeGPS

    from gragra.constants import G_SI
    from gragra.geometry import Frame
    from gragra.gravity_sources import PointMassCloud, rigid_motion
    from gragra.targets import TargetPoints

    world = Frame("world")
    cloud = PointMassCloud(np.array([1.0]), np.array([[0.0, 0.0, 0.0]]))
    epoch = LIGOTimeGPS(1_000_000_000, 0)
    targets = TargetPoints(np.zeros((1, 3)) * u.m, ids=("t0",), frame=world)
    view = rigid_motion(cloud, circular_pose(), frame=world, epoch=epoch).at(
        targets, observables=("acceleration",)
    )
    with pytest.raises(ValueError):
        view.fourier_transform(
            t_start=0.0 * u.s,
            t_stop=1.0 * u.s,
            frequencies=[0.0, 2.0, 1.0] * u.Hz,
            tref=epoch,
            tau_min=0.125 * u.s,
            atol=1e-9 * G_SI * u.m / u.s,
            rtol=1e-10,
        )


def test_amplitude_without_orders_is_rejected():
    """Doubling needs the harmonic orders, so amplitude=True requires orders."""
    coefficients = np.zeros((2, 1, 3), dtype=np.complex128) * u.m / u.s**2
    with pytest.raises(ValueError, match="orders is required"):
        cf._coefficients_to_matrix(
            coefficients,
            orders=None,
            frequencies_hz=np.array([0.0, 1.0]),
            target_ids=("t0",),
            observable="acceleration",
            amplitude=True,
        )


def test_transform_coefficients_without_orders_are_accepted():
    """amplitude=False does not double, so orders=None is allowed."""
    coefficients = np.zeros((2, 1, 3), dtype=np.complex128) * u.m / u.s**2
    matrix = cf._coefficients_to_matrix(
        coefficients,
        orders=None,
        frequencies_hz=np.array([0.0, 1.0]),
        target_ids=("t0",),
        observable="acceleration",
        amplitude=False,
    )
    assert matrix.shape[:2] == (1, 3)


@pytest.mark.parametrize("softening", [-1.0, -math.inf, math.nan, math.inf])
def test_model_rejects_nonfinite_or_negative_softening(softening):
    """softening_m must be finite and nonnegative; the message shows the value."""
    source = PointMassCloud(np.array([1.0]), np.array([[0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="softening_m") as info:
        ContinuousFourierModel(
            source, circular_pose(), frame=WORLD, softening_m=softening
        )
    assert repr(softening) in str(info.value)


@pytest.mark.parametrize("softening", [0.0, 0.5])
def test_model_accepts_zero_and_positive_softening(softening):
    source = PointMassCloud(np.array([1.0]), np.array([[0.0, 0.0, 0.0]]))
    model = ContinuousFourierModel(
        source, circular_pose(), frame=WORLD, softening_m=softening
    )
    assert model.softening_m == softening


# Blind holdout v2 N-04 (Revision R3): an invalid tau_min must be a ValueError
# raised before any callback runs, not a ZeroDivisionError from the default
# limit. The P4 and P5 paths must reject the same values.
BAD_TAU_MIN = [
    pytest.param(0.0 * u.s, id="zero"),
    pytest.param(-0.5 * u.s, id="negative"),
    pytest.param(math.nan * u.s, id="nan"),
    pytest.param(math.inf * u.s, id="inf"),
    pytest.param(1.0 * u.m, id="length-unit"),
]


@pytest.mark.parametrize("tau_min", BAD_TAU_MIN)
@pytest.mark.parametrize(
    ("kind", "atol"),
    [
        ("periodic", 1e-6 * u.m / u.s**2),
        ("window", 1e-6 * u.m / u.s**2 * u.s),
    ],
    ids=["periodic-P4", "finite-window-P5"],
)
def test_invalid_tau_min_is_value_error_before_any_callback(
    monkeypatch, kind, atol, tau_min
):
    calls = [0]
    real_pose_response = cf.make_rigid_pose_response

    def counting_pose_response(*args, **kwargs):
        response = real_pose_response(*args, **kwargs)

        def counted(times, observable):
            calls[0] += times.size
            return response(times, observable)

        return counted

    monkeypatch.setattr(cf, "make_rigid_pose_response", counting_pose_response)
    evaluation = _evaluation(kind, atol=atol, budget=MEMORY_BYTES, tau_min=tau_min)
    with pytest.raises(ValueError, match="tau_min"):
        evaluation.run()
    assert calls[0] == 0


# Blind holdout v2 N-06 (Revision R3): the callback budget is checked before a
# batch is evaluated, so the total number of callback elements never exceeds
# max_evaluations. The width-1, tau_min=0.5 constant case is the one used by the
# pre-check floor tests above.
@pytest.mark.parametrize("margin", [0, 1, 2])
def test_max_evaluations_never_passes_budget_to_callback(margin):
    width_s, tau_s, freqs_hz = 1.0, 0.5, (0.0,)
    maximum = _independent_floor(width_s, tau_s, max(freqs_hz)) + margin
    seen = [0]
    base = make_response(constant(1.0))

    def counting(times, observable):
        seen[0] += times.size
        return base(times, observable)

    with pytest.raises(ContinuousQuadratureError) as info:
        finite_window_fourier(
            counting,
            "acceleration",
            epoch=EPOCH,
            tref=TREF,
            t_start=0.0 * u.s,
            t_stop=width_s * u.s,
            frequencies=np.array(freqs_hz, dtype=np.float64) * u.Hz,
            tau_min=tau_s * u.s,
            atol=ATOL_COEFFICIENT * width_s * (u.m / u.s**2) * u.s,
            rtol=RTOL,
            limit=10000,
            cache_size=CACHE_BYTES,
            max_evaluations=maximum,
            memory_budget_bytes=MEMORY_BYTES,
        )
    assert info.value.reason == "max_evaluations_exceeded"
    assert seen[0] <= maximum
    assert info.value.diagnostics["evaluations"] <= maximum
