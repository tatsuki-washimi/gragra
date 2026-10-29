"""Application and coefficient-matrix contracts for gravity responses."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesMatrix
from gwpy.time import LIGOTimeGPS

from gragra._evaluation_setup import EvaluationRecord
from gragra._rayleigh_contract import (
    RayleighResponseEvaluation,
    RayleighResponseSpec,
    _rayleigh_output_definition,
)
from gragra.constants import G_SI
from gragra.geometry._frame import Frame
from gragra.geometry._halfspace import HalfSpace
from gragra.results._response import GravityResponse, ResponseContract
from gragra.results._spectral import PhasorRepresentation, PSDRepresentation
from gragra.targets._high_points import TargetPoints

_FREQUENCIES_HZ = (0.5, 2.0)
_INPUT_NAME = "vertical_displacement"


def _coefficient_arrays(
    target_count: int = 2,
    frequency_count: int = len(_FREQUENCIES_HZ),
    *,
    scale: float = 1e-4,
) -> dict[str, np.ndarray]:
    component_counts = {"potential": 1, "acceleration": 3, "gradient": 9}
    result = {}
    for observable, components in component_counts.items():
        indices = np.arange(
            1, target_count * components * frequency_count + 1, dtype=np.float64
        ).reshape(target_count, components, 1, frequency_count)
        result[observable] = np.asarray(
            scale * indices * (1.0 + 0.25j), dtype=np.complex128
        )
    return result


def _fake_contract(
    *,
    input_location: np.ndarray | None = None,
    law_parameters: dict | None = None,
    input_unit: u.UnitBase = u.m,
) -> ResponseContract:
    return ResponseContract(
        input_name=_INPUT_NAME,
        input_quantity="surface_vertical_displacement",
        input_unit=input_unit,
        input_channel_ids=(_INPUT_NAME,),
        input_components=("z",),
        input_frame=Frame("fake input"),
        input_location_m=(
            np.array([3.0, -2.0, 0.0]) if input_location is None else input_location
        ),
        input_distribution="rayleigh_plane_wave",
        normalization="complex phasor amplitude at input_location",
        basis_amplitude=1 * input_unit,
        mapping_kind="local_linearization",
        action="per_frequency",
        storage="precomputed",
        varied="source",
        target_position="fixed",
        law_name="newton",
        law_parameters={} if law_parameters is None else law_parameters,
        gravitational_constant_si=G_SI,
        linearization_assessment="not_assessed",
    )


class _FakeEvaluation:
    frequencies_hz = _FREQUENCIES_HZ
    target_ids = ("T0", "T1")
    parameters = {"fixture": True}
    formulas = ()
    references = ()

    def __init__(self, context_factory=None):
        self.context_factory = context_factory
        self.context_calls = 0

    def _applied_context(self, value, *, representation, reference_time):
        self.context_calls += 1
        if self.context_factory is None:
            raise AssertionError("the fake context should not be requested")
        return self.context_factory(value, representation, reference_time)


def _fake_response(
    *, coefficients=None, contract=None, context_factory=None
) -> tuple[GravityResponse, _FakeEvaluation]:
    evaluation = _FakeEvaluation(context_factory)
    response = GravityResponse(
        contract=_fake_contract() if contract is None else contract,
        output_definition=_rayleigh_output_definition(),
        evaluation=evaluation,
        coefficients=(_coefficient_arrays() if coefficients is None else coefficients),
        record=EvaluationRecord(
            backend="fixture:coefficient_source",
            versions={"fixture": "1"},
            gravitational_constant_si=G_SI,
            g_applications=1,
        ),
    )
    return response, evaluation


def _rayleigh_response(
    *, coefficients=None, frequencies_hz=_FREQUENCIES_HZ, target_ids=("T0", "T1")
) -> GravityResponse:
    frame = Frame("surface")
    spec = RayleighResponseSpec(
        geometry=HalfSpace(frame),
        density=2500 * u.kg / u.m**3,
        vp=4000 * u.m / u.s,
        vs=2300 * u.m / u.s,
        frequencies=np.asarray(frequencies_hz) * u.Hz,
        phase_origin=np.array([3.0, -2.0, 0.0]) * u.m,
        direction=(3.0, 4.0, 0.0),
    )
    model = SimpleNamespace(spec=spec)
    positions = np.asarray(
        [
            [1.5 * index, -0.5 * index, -10.0 - 5.0 * index]
            for index in range(len(target_ids))
        ]
    )
    targets = TargetPoints(positions * u.m, tuple(target_ids), frame)
    evaluation = RayleighResponseEvaluation(model, targets, frame)
    coefficient_values = (
        _coefficient_arrays(len(target_ids), len(frequencies_hz))
        if coefficients is None
        else coefficients
    )
    from gragra._rayleigh_evaluation_setup import _rayleigh_generation_record

    return GravityResponse(
        **evaluation._constructor_kwargs(
            coefficients=coefficient_values,
            record=_rayleigh_generation_record(),
        )
    )


def _series(
    values,
    *,
    unit=u.m,
    frequencies=_FREQUENCIES_HZ,
    frequency_unit=u.Hz,
    epoch=None,
    name="sensor displacement",
    channel="TEST:VERTICAL",
):
    return FrequencySeries(
        np.asarray(values, dtype=np.complex128),
        frequencies=np.asarray(frequencies) * frequency_unit,
        unit=unit,
        epoch=epoch,
        name=name,
        channel=channel,
    )


def _apply(response, series, *, reference_time=(1_700_000_000, 123_456_789)):
    return response.apply(
        representation=PhasorRepresentation(),
        reference_time=reference_time,
        **{_INPUT_NAME: series},
    )


def test_coefficients_copy_isolated():
    response, _ = _fake_response()
    original = response._coefficient_array("acceleration").copy()

    first = response.coefficients("acceleration")
    assert isinstance(first, FrequencySeriesMatrix)
    assert tuple(first.rows.keys()) == response.target_ids
    assert tuple(first.cols.keys()) == ("x", "y", "z")
    assert first.frequencies.unit == u.Hz
    np.testing.assert_array_equal(
        first.frequencies.to_value(u.Hz), response.frequencies.to_value(u.Hz)
    )
    np.testing.assert_array_equal(np.asarray(first), original[:, :, 0, :])
    for row in range(first.shape[0]):
        for column in range(first.shape[1]):
            assert first.meta[row, column].unit == u.s**-2
            assert _INPUT_NAME in str(first.meta[row, column].name)
            assert _INPUT_NAME in str(first.meta[row, column].channel)

    first[0, 0, 0] = 123 + 456j
    first.meta[0, 0].unit = u.kg
    second = response.coefficients("acceleration")
    np.testing.assert_array_equal(np.asarray(second), original[:, :, 0, :])
    assert second.meta[0, 0].unit == u.s**-2
    assert _INPUT_NAME in str(second.meta[0, 0].channel)
    np.testing.assert_array_equal(response._coefficient_array("acceleration"), original)

    potential = response.coefficients("potential")
    gradient = response.coefficients("gradient")
    assert potential.meta[0, 0].unit == u.m / u.s**2
    assert gradient.meta[0, 0].unit == u.m**-1 / u.s**2
    with pytest.raises(ValueError, match="unknown observable"):
        response.coefficients("jerk")


def test_response_owns_constructor_inputs():
    source_coefficients = _coefficient_arrays()
    location = np.array([3.0, -2.0, 0.0])
    law_parameters = {}
    contract = _fake_contract(input_location=location, law_parameters=law_parameters)
    response, _ = _fake_response(coefficients=source_coefficients, contract=contract)
    expected = response._coefficient_array("potential").copy()

    source_coefficients["potential"][:] = 9 + 2j
    location[:] = 99.0
    law_parameters["label"] = "mutated"
    assert response._coefficient_array("potential")[0, 0, 0, 0] == expected[0, 0, 0, 0]
    np.testing.assert_array_equal(contract.input_location_m.to_value(u.m), [3, -2, 0])
    assert not contract.law_parameters

    frequencies = response.frequencies
    frequencies.value[0] = 99
    assert response.frequencies[0] == 0.5 * u.Hz
    location_quantity = contract.input_location_m
    location_quantity.value[0] = -100
    np.testing.assert_array_equal(contract.input_location_m.to_value(u.m), [3, -2, 0])


def test_coefficients_has_independent_native_values_and_metadata():
    response = _rayleigh_response()
    matrix = response.coefficients("potential")
    expected = response._coefficient_array("potential")[:, :, 0, :].copy()
    assert isinstance(matrix, FrequencySeriesMatrix)
    assert tuple(matrix.rows.keys()) == response.target_ids
    assert tuple(matrix.cols.keys()) == ("potential",)
    assert matrix.name == f"potential response per {_INPUT_NAME}"
    matrix[0, 0, 0] = -900j
    matrix.meta[0, 0].unit = u.kg
    fresh = response.coefficients("potential")
    np.testing.assert_array_equal(np.asarray(fresh), expected)
    assert fresh.meta[0, 0].unit == u.m / u.s**2


def test_coefficients_derive_units_from_input_unit():
    response, _ = _fake_response(contract=_fake_contract(input_unit=u.cm))
    coefficient = response.coefficients("potential")
    expected_unit = (u.m**2 / u.s**2) / u.cm
    assert coefficient.meta[0, 0].unit == expected_unit


def test_apply_length_units():
    response = _rayleigh_response()
    si_values = np.array([1.25 - 0.5j, -2.0 + 0.75j]) * 1e-9
    metre_input = _series(si_values, unit=u.m)
    nanometre_input = _series(si_values / 1e-9, unit=u.nm)

    metres = _apply(response, metre_input)
    nanometres = _apply(response, nanometre_input)
    for observable in response.observables:
        np.testing.assert_allclose(
            np.asarray(getattr(metres, observable)),
            np.asarray(getattr(nanometres, observable)),
            rtol=2e-15,
            atol=0,
        )

    with pytest.raises(ValueError):
        _apply(response, _series([1, 2], unit=u.s))
    with pytest.raises(ValueError, match="finite amplitudes"):
        _apply(response, _series([np.nan, 2]))


def test_apply_matches_linear_combination_and_zero_exactly():
    response = _rayleigh_response()
    x_values = np.array([0.75 + 0.5j, -1.25j]) * 1e-8
    y_values = np.array([-0.25j, 2.0 - 0.75j]) * 1e-8
    alpha = -0.75 + 1.5j
    beta = 2.25 - 0.5j
    x = _apply(response, _series(x_values))
    y = _apply(response, _series(y_values))
    combined = _apply(response, _series(alpha * x_values + beta * y_values))

    for observable in response.observables:
        expected = alpha * np.asarray(getattr(x, observable)) + beta * np.asarray(
            getattr(y, observable)
        )
        actual = np.asarray(getattr(combined, observable))
        scale = np.abs(alpha * np.asarray(getattr(x, observable))) + np.abs(
            beta * np.asarray(getattr(y, observable))
        )
        assert np.all(np.abs(actual - expected) <= 2e-15 * scale + np.finfo(float).tiny)

    zero = _apply(response, _series(np.zeros(2, dtype=np.complex128)))
    for observable in response.observables:
        assert np.count_nonzero(np.asarray(getattr(zero, observable))) == 0
        assert np.isfinite(np.asarray(getattr(zero, observable))).all()


def test_apply_builds_real_result_and_retains_explicit_gps():
    response = _rayleigh_response()
    phasors = np.array([2.0 - 3.0j, -0.5 + 0.75j]) * 1e-9
    epoch = LIGOTimeGPS(123, 0)
    explicit_reference = (1_700_000_000, 123_456_789)
    before = {
        name: response._coefficient_array(name).copy() for name in response.observables
    }
    input_series = _series(phasors / 1e-9, unit=u.nm, epoch=epoch)
    result = _apply(response, input_series, reference_time=explicit_reference)

    assert result.record.backend == "gravity_response:rayleigh_literature"
    assert result.record.versions == response.record.versions
    assert (
        result.record.gravitational_constant_si
        == response.record.gravitational_constant_si
    )
    assert result.record.g_applications == 1
    assert result.context.reference_time_gps == explicit_reference
    assert result.context.model_spec.displacement_epoch_gps == (
        epoch.gpsSeconds,
        epoch.gpsNanoSeconds,
    )
    for name in response.observables:
        input_si = input_series.to_value(u.m)
        expected = before[name][:, :, 0, :] * input_si[None, None, :]
        np.testing.assert_array_equal(np.asarray(getattr(result, name)), expected)
        assert result.bindings[name].reference_time_gps == explicit_reference


def test_apply_checks_keys_types_and_representation_before_context():
    response, evaluation = _fake_response()
    series = _series([1, 2])
    representation = PhasorRepresentation()

    with pytest.raises(TypeError, match="exactly"):
        response.apply(representation=representation, reference_time=(1, 0))
    with pytest.raises(TypeError, match="exactly"):
        response.apply(
            representation=representation,
            reference_time=(1, 0),
            wrong_input=series,
        )
    with pytest.raises(TypeError, match="FrequencySeries"):
        response.apply(
            representation=representation,
            reference_time=(1, 0),
            **{_INPUT_NAME: np.ones(2)},
        )
    with pytest.raises(TypeError, match="spectral representation"):
        response.apply(
            representation="phasor",
            reference_time=(1, 0),
            **{_INPUT_NAME: series},
        )
    with pytest.raises(ValueError, match="only PhasorRepresentation"):
        response.apply(
            representation=PSDRepresentation(),
            reference_time=(1, 0),
            **{_INPUT_NAME: series},
        )
    assert evaluation.context_calls == 0


def test_apply_accepts_exact_millihertz_and_rejects_rounded_mismatch():
    response = _rayleigh_response(frequencies_hz=(0.75, 1.375, 5.25))
    accepted = _series(
        [1 + 2j, -3j, 0.5j],
        frequencies=(750.0, 1375.0, 5250.0),
        frequency_unit=u.mHz,
    )
    result = _apply(response, accepted)
    assert result.context.frequencies_hz == response.evaluation.frequencies_hz

    rounded_response = _rayleigh_response(frequencies_hz=(0.1, 0.3, 0.7))
    rejected = _series(
        [1 + 2j, -3j, 0.5j],
        frequencies=(100.0, 300.0, 700.0),
        frequency_unit=u.mHz,
    )
    with pytest.raises(ValueError, match="frequencies must match"):
        _apply(rounded_response, rejected)

    with pytest.raises(ValueError, match="frequencies must match"):
        _apply(response, _series([1 + 2j, -3j], frequencies=(0.75, 1.375)))


def test_apply_rejects_nonphasor_and_malformed_actual_inputs():
    response = _rayleigh_response()
    series = _series([1 + 2j, -3j])
    with pytest.raises(ValueError, match="frequencies must match"):
        _apply(response, _series([1, 2], frequencies=(0.5, 2.01)))
    with pytest.raises(TypeError, match="GPS"):
        _apply(response, series, reference_time=123.0)
    with pytest.raises(TypeError, match="reference_time"):
        response.apply(representation=PhasorRepresentation(), **{_INPUT_NAME: series})


def test_apply_rejects_overflow_before_packing():
    coefficients = _coefficient_arrays(scale=1.0)
    coefficients["potential"][:] = 1e308 + 0j

    class Context:
        input_values = np.array([1e308, 1.0], dtype=np.complex128)
        backend = "fixture:overflow"
        bindings = {}
        evaluation = None

        def pack(self, _values):
            raise AssertionError("nonfinite products must not be packed")

    response, evaluation = _fake_response(
        coefficients=coefficients,
        context_factory=lambda *_: Context(),
    )
    with pytest.raises(ValueError, match="applied potential values must be finite"):
        response.apply(
            representation=PhasorRepresentation(),
            reference_time=(1, 0),
            **{_INPUT_NAME: _series([1, 1])},
        )
    assert evaluation.context_calls == 1
