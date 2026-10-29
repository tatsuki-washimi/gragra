"""Public continuous rigid-motion API: ``rigid_motion(..., epoch=...)``.

The continuous form returns a private ``ContinuousFourierModel``. Its ``.at``
view offers ``fourier_series`` and ``fourier_transform``, which delegate to the
P4 and P5 integrators. These tests check the argument dispatch, the closed
forms, bitwise agreement with the private window path, the ``tref`` phase
identity, and the returned ``EvaluationResult`` metadata.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from _continuous_fixtures import circular_pose, constant_velocity_pose
from astropy import units as u
from astropy.time import Time
from gwpy.time import LIGOTimeGPS

from gragra._continuous_fourier import (
    DEFAULT_MAX_EVALUATIONS,
    DEFAULT_MEMORY_BUDGET_BYTES,
    ContinuousFourierModel,
)
from gragra.constants import G_SI
from gragra.geometry import Frame
from gragra.gravity_sources import PointMassCloud, rigid_motion
from gragra.gravity_sources.trajectory import PrescribedRigidTransform
from gragra.results import EvaluationResult, TimeSampling
from gragra.results._coords import normalize_gps
from gragra.results._spectral import (
    AmplitudeSpectrumRepresentation,
    FiniteRecordInterpretation,
    FourierTransformRepresentation,
    Sidedness,
)
from gragra.results._time_grid import build_uniform_time_s
from gragra.targets import TargetPoints

WORLD = Frame("world")
EPOCH_TUPLE = (1_000_000_000, 0)
EPOCH = LIGOTimeGPS(*EPOCH_TUPLE)
# Unit mass at the body origin. The pose translation moves it in the frame.
CLOUD = PointMassCloud(np.array([1.0]), np.array([[0.0, 0.0, 0.0]]))
ACC_UNIT = u.m / u.s**2
# Absolute tolerance scaled by G, because the acceleration scale is G * M / r^2.
ATOL_ACC = 1e-9 * G_SI * ACC_UNIT
# The finite-window coefficients carry an extra factor of seconds.
ATOL_FT = ATOL_ACC * u.s


def _targets(positions=((0.0, 0.0, 0.0),)) -> TargetPoints:
    arr = np.asarray(positions, dtype=np.float64)
    return TargetPoints(
        arr * u.m, ids=tuple(f"t{i}" for i in range(len(arr))), frame=WORLD
    )


def _sampled_model():
    transform, sampling = _sampled_transform_and_sampling()
    return rigid_motion(CLOUD, transform, frame=WORLD, sampling=sampling)


def _sampled_transform_and_sampling():
    frames = 4
    time_s = build_uniform_time_s(EPOCH_TUPLE, 1.0, frames)
    transform = PrescribedRigidTransform(
        time_s,
        np.zeros((frames, 3)),
        np.repeat(np.eye(3)[None], frames, axis=0),
    )
    sampling = TimeSampling(epoch=EPOCH, dt=1.0 * u.s, count=frames)
    return transform, sampling


# -- (a) the sampled path is unchanged --------------------------------------


def test_sampled_form_still_returns_time_series_result():
    model = _sampled_model()
    assert not isinstance(model, ContinuousFourierModel)
    # The sampled body sits at the origin, so the target is placed away from it.
    targets = _targets(((3.0, 0.0, 0.0),))
    result = model.at(targets, observables=("acceleration",)).run()
    assert isinstance(result, EvaluationResult)
    assert np.asarray(result.acceleration).shape == (1, 3, 4)


# -- (b) argument dispatch ---------------------------------------------------


def test_continuous_form_returns_private_model_with_epoch():
    model = rigid_motion(CLOUD, circular_pose(), frame=WORLD, epoch=EPOCH)
    assert isinstance(model, ContinuousFourierModel)
    assert model.epoch is EPOCH


def test_sampled_transform_with_epoch_raises_type_error():
    transform, sampling = _sampled_transform_and_sampling()
    with pytest.raises(TypeError, match="epoch"):
        rigid_motion(CLOUD, transform, frame=WORLD, sampling=sampling, epoch=EPOCH)


def test_sampled_transform_without_sampling_raises_type_error():
    transform, _ = _sampled_transform_and_sampling()
    with pytest.raises(TypeError, match="sampling"):
        rigid_motion(CLOUD, transform, frame=WORLD)


def test_callable_pose_with_sampling_raises_type_error():
    _, sampling = _sampled_transform_and_sampling()
    with pytest.raises(TypeError, match="sampling"):
        rigid_motion(CLOUD, circular_pose(), frame=WORLD, sampling=sampling)


def test_callable_pose_without_epoch_raises_type_error():
    with pytest.raises(TypeError, match="epoch"):
        rigid_motion(CLOUD, circular_pose(), frame=WORLD)


def test_callable_pose_with_non_gps_epoch_raises_type_error():
    with pytest.raises(TypeError):
        rigid_motion(CLOUD, circular_pose(), frame=WORLD, epoch=1.5)


@pytest.mark.parametrize(
    "bad_epoch", [1.5, (1.5, 0), (1, 2.0)], ids=["float", "float-seconds", "float-ns"]
)
def test_bad_epoch_error_names_the_epoch_argument(bad_epoch):
    # The message names the public argument, not the internal reference_time,
    # and the original exception is kept as the cause.
    with pytest.raises(TypeError, match="^epoch must be a LIGOTimeGPS") as info:
        rigid_motion(CLOUD, circular_pose(), frame=WORLD, epoch=bad_epoch)
    assert "reference_time" not in str(info.value)
    assert isinstance(info.value.__cause__, TypeError)


def test_astropy_time_epoch_is_accepted_and_normalized():
    epoch = Time(EPOCH_TUPLE[0], format="gps")
    model = rigid_motion(CLOUD, circular_pose(), frame=WORLD, epoch=epoch)
    assert isinstance(model, ContinuousFourierModel)
    assert normalize_gps(model.epoch) == EPOCH_TUPLE


def test_non_callable_non_sampled_transform_raises_type_error():
    with pytest.raises(TypeError):
        rigid_motion(CLOUD, 3.0, frame=WORLD, epoch=EPOCH)


def test_provenance_is_rejected_in_both_forms():
    transform, sampling = _sampled_transform_and_sampling()
    with pytest.raises(TypeError, match="provenance"):
        rigid_motion(CLOUD, transform, frame=WORLD, sampling=sampling, provenance={})
    with pytest.raises(TypeError, match="provenance"):
        rigid_motion(CLOUD, circular_pose(), frame=WORLD, epoch=EPOCH, provenance={})


# -- (c) closed form: circular motion, n = 1 only ---------------------------


def test_circular_motion_fourier_series_matches_closed_form():
    # c(t) = (cos 2 pi t, sin 2 pi t, 0) m, unit mass, target at the origin:
    # a(t) = G (cos, sin, 0) m/s^2. For this pose the one-sided amplitudes are
    # A_1x = G and A_1y = -i G. All other orders vanish.
    result = (
        rigid_motion(CLOUD, circular_pose(), frame=WORLD, epoch=EPOCH)
        .at(_targets(), observables=("acceleration",))
        .fourier_series(
            period=1.0 * u.s,
            orders=[0, 1, 2, 3],
            tref=EPOCH,
            tau_min=0.125 * u.s,
            atol=ATOL_ACC,
            rtol=1e-10,
        )
    )
    freqs = np.asarray(result.acceleration.frequencies.to_value(u.Hz))
    index = {int(round(f)): i for i, f in enumerate(freqs)}
    ax = np.asarray(result.acceleration[0, 0])
    ay = np.asarray(result.acceleration[0, 1])
    az = np.asarray(result.acceleration[0, 2])
    tol = 1e-6 * G_SI
    assert abs(ax[index[1]] - G_SI) <= tol
    assert abs(ay[index[1]] - (-1j * G_SI)) <= tol
    assert abs(az[index[1]]) <= tol
    for order in (0, 2, 3):
        assert abs(ax[index[order]]) <= tol
        assert abs(ay[index[order]]) <= tol


# -- (d) bitwise agreement with the private window path ----------------------


def test_passage_fourier_transform_is_bitwise_identical_to_private_path():
    pose = constant_velocity_pose(
        velocity_m_s=(1.0, 0.0, 0.0), origin_m=(0.0, 1.0, 0.0)
    )
    freqs = np.array([0.0, 0.125, 1.0, 4.0]) * u.Hz
    t_start, t_stop = -2.0 * u.s, 2.0 * u.s
    tau_min = 0.5 * u.s
    public = (
        rigid_motion(CLOUD, pose, frame=WORLD, epoch=EPOCH)
        .at(_targets(), observables=("acceleration",))
        .fourier_transform(
            t_start=t_start,
            t_stop=t_stop,
            frequencies=freqs,
            tref=EPOCH,
            tau_min=tau_min,
            atol=ATOL_FT,
            rtol=1e-10,
        )
    )
    private = (
        ContinuousFourierModel(CLOUD, pose, frame=WORLD)
        .window_at(
            _targets(),
            epoch=EPOCH,
            tref=EPOCH,
            t_start=t_start,
            t_stop=t_stop,
            frequencies=freqs,
            tau_min=tau_min,
            atol=ATOL_FT,
            rtol=1e-10,
            max_evaluations=DEFAULT_MAX_EVALUATIONS,
            memory_budget_bytes=DEFAULT_MEMORY_BUDGET_BYTES,
            observables=("acceleration",),
        )
        .run()
    )
    np.testing.assert_array_equal(
        np.asarray(public.acceleration), np.asarray(private.acceleration)
    )
    np.testing.assert_array_equal(
        np.asarray(public.acceleration.frequencies.value),
        np.asarray(private.acceleration.frequencies.value),
    )


# -- (e) tref moves only the phase -------------------------------------------


def test_tref_shift_multiplies_coefficients_by_the_phase_identity():
    # F_tref'(f) = F_tref(f) * exp(+2 pi i f (tref' - tref)) under
    # exp[-2 pi i f (t - tref)].
    pose = constant_velocity_pose(
        velocity_m_s=(1.0, 0.0, 0.0), origin_m=(0.0, 1.0, 0.0)
    )
    freqs = np.array([0.0, 0.5, 1.0]) * u.Hz
    shift_s = 0.25
    shifted_tref = LIGOTimeGPS(1_000_000_000, 250_000_000)
    common = {
        "t_start": -2.0 * u.s,
        "t_stop": 2.0 * u.s,
        "frequencies": freqs,
        "tau_min": 0.5 * u.s,
        "atol": ATOL_FT,
        "rtol": 1e-10,
    }
    model = rigid_motion(CLOUD, pose, frame=WORLD, epoch=EPOCH)
    base = (
        model.at(_targets(), observables=("acceleration",))
        .fourier_transform(tref=EPOCH, **common)
        .acceleration
    )
    moved = (
        model.at(_targets(), observables=("acceleration",))
        .fourier_transform(tref=shifted_tref, **common)
        .acceleration
    )
    phase = np.exp(2j * np.pi * freqs.to_value(u.Hz) * shift_s)
    expected = np.asarray(base[0, 0]) * phase
    got = np.asarray(moved[0, 0])
    scale = max(float(np.max(np.abs(expected))), G_SI)
    assert float(np.max(np.abs(got - expected))) <= 1e-6 * scale


# -- (f) missing or invalid required arguments ------------------------------


def test_fourier_methods_require_explicit_tref_and_tau_min():
    evaluation = rigid_motion(CLOUD, circular_pose(), frame=WORLD, epoch=EPOCH).at(
        _targets()
    )
    with pytest.raises(TypeError):
        evaluation.fourier_series(
            period=1.0 * u.s,
            orders=[0, 1],
            tau_min=0.125 * u.s,
            atol=ATOL_ACC,
        )
    with pytest.raises(TypeError):
        evaluation.fourier_transform(
            t_start=-1.0 * u.s,
            t_stop=1.0 * u.s,
            frequencies=np.array([0.0]) * u.Hz,
            tref=EPOCH,
            atol=ATOL_ACC,
        )


def test_output_frame_is_rejected_for_continuous_form():
    model = rigid_motion(CLOUD, circular_pose(), frame=WORLD, epoch=EPOCH)
    with pytest.raises(ValueError, match="computation frame"):
        model.at(_targets(), output_frame=WORLD)


def test_model_without_epoch_cannot_build_an_evaluation_view():
    model = ContinuousFourierModel(CLOUD, circular_pose(), frame=WORLD)
    with pytest.raises(ValueError, match="epoch"):
        model.at(_targets())


# -- (g) returned EvaluationResult metadata ----------------------------------


def test_fourier_series_result_metadata():
    result = (
        rigid_motion(CLOUD, circular_pose(), frame=WORLD, epoch=EPOCH)
        .at(_targets(), observables=("acceleration",))
        .fourier_series(
            period=1.0 * u.s,
            orders=[0, 1],
            tref=EPOCH,
            tau_min=0.125 * u.s,
            atol=ATOL_ACC,
        )
    )
    assert isinstance(result, EvaluationResult)
    representation = result.context.representation
    assert isinstance(representation, AmplitudeSpectrumRepresentation)
    assert representation.record_interpretation is FiniteRecordInterpretation.PERIODIC
    freqs = result.acceleration.frequencies
    np.testing.assert_allclose(freqs.to_value(u.Hz), [0.0, 1.0])
    assert "atol" in result.parameters.colnames
    selected = result.select("acceleration")
    assert selected.observables == ("acceleration",)


def test_fourier_transform_result_metadata():
    result = (
        rigid_motion(CLOUD, circular_pose(), frame=WORLD, epoch=EPOCH)
        .at(_targets(), observables=("acceleration",))
        .fourier_transform(
            t_start=-1.0 * u.s,
            t_stop=1.0 * u.s,
            frequencies=np.array([0.0, 1.0]) * u.Hz,
            tref=EPOCH,
            tau_min=0.25 * u.s,
            atol=ATOL_FT,
        )
    )
    assert isinstance(result, EvaluationResult)
    representation = result.context.representation
    assert isinstance(representation, FourierTransformRepresentation)
    assert representation.sidedness is Sidedness.ONE_SIDED
    np.testing.assert_allclose(
        result.acceleration.frequencies.to_value(u.Hz), [0.0, 1.0]
    )
    assert "atol" in result.parameters.colnames


# -- (h) the sampled-only import does not load the continuous module ---------


def test_sampled_import_does_not_load_continuous_module():
    code = (
        "import sys\n"
        "import gragra.gravity_sources as g\n"
        "g.rigid_motion\n"
        "print('gragra._continuous_fourier' in sys.modules)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "False"


# -- gradient and potential closed forms (circular motion, central target) ----


def test_circular_motion_gradient_series_matches_closed_form():
    # Delta = (cos th, sin th, 0), r = 1, Gamma = G (3 d d^T - I) with th = 2 pi t.
    # C_0 = G diag(1/2, 1/2, -1); the 2 f_0 harmonic gives A_2,xx = 3G/2,
    # A_2,yy = -3G/2, A_2,xy = A_2,yx = -3iG/2. Components are row-major.
    result = (
        rigid_motion(CLOUD, circular_pose(), frame=WORLD, epoch=EPOCH)
        .at(_targets(), observables=("gradient",))
        .fourier_series(
            period=1.0 * u.s,
            orders=[0, 1, 2, 3],
            tref=EPOCH,
            tau_min=0.125 * u.s,
            atol=1e-9 * G_SI / u.s**2,
            rtol=1e-10,
        )
    )
    freqs = np.asarray(result.gradient.frequencies.to_value(u.Hz))
    index = {int(round(f)): i for i, f in enumerate(freqs)}
    comp = {
        name: np.asarray(result.gradient[0, j])
        for j, name in enumerate(("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"))
    }
    tol = 1e-6 * G_SI
    assert abs(comp["xx"][index[0]] - 0.5 * G_SI) <= tol
    assert abs(comp["yy"][index[0]] - 0.5 * G_SI) <= tol
    assert abs(comp["zz"][index[0]] + G_SI) <= tol
    assert abs(comp["xx"][index[2]] - 1.5 * G_SI) <= tol
    assert abs(comp["yy"][index[2]] + 1.5 * G_SI) <= tol
    assert abs(comp["xy"][index[2]] + 1.5j * G_SI) <= tol
    assert abs(comp["yx"][index[2]] + 1.5j * G_SI) <= tol
    for order in (1, 3):
        for name in ("xx", "xy", "yy", "zz"):
            assert abs(comp[name][index[order]]) <= tol
    assert result.gradient.units.shape == (1, 9)


def test_circular_motion_gradient_transform_runs_and_has_nine_components():
    result = (
        rigid_motion(CLOUD, circular_pose(), frame=WORLD, epoch=EPOCH)
        .at(_targets(), observables=("gradient",))
        .fourier_transform(
            t_start=0.0 * u.s,
            t_stop=1.0 * u.s,
            frequencies=[0.0, 2.0] * u.Hz,
            tref=EPOCH,
            tau_min=0.125 * u.s,
            atol=1e-9 * G_SI * u.s / u.s**2,
            rtol=1e-10,
        )
    )
    # int_0^1 Gamma_xx dt = G/2 (Gamma_zz = -G).
    zz = np.asarray(result.gradient[0, 8])
    xx = np.asarray(result.gradient[0, 0])
    assert abs(zz[0] + G_SI) <= 1e-6 * G_SI
    assert abs(xx[0] - 0.5 * G_SI) <= 1e-6 * G_SI


def test_circular_motion_potential_series_matches_closed_form():
    # r = 1 for all t: Phi = -G, so A_0 = -G and every other order vanishes.
    result = (
        rigid_motion(CLOUD, circular_pose(), frame=WORLD, epoch=EPOCH)
        .at(_targets(), observables=("potential",))
        .fourier_series(
            period=1.0 * u.s,
            orders=[0, 1, 2],
            tref=EPOCH,
            tau_min=0.125 * u.s,
            atol=1e-9 * G_SI * u.m**2 / u.s**2,
            rtol=1e-10,
        )
    )
    phi = np.asarray(result.potential[0, 0])
    assert abs(phi[0] + G_SI) <= 1e-6 * G_SI
    assert np.all(np.abs(phi[1:]) <= 1e-6 * G_SI)
