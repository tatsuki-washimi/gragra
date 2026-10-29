"""Contract and construction checks for Rayleigh gravity responses."""

import subprocess
import sys
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u

from gragra.geometry._frame import Frame
from gragra.results._field_definition import (
    ChangeDefinition,
    ChangeForm,
    Contribution,
    GravityFieldDefinition,
    PotentialReference,
    ReferenceDefinition,
    ReferenceStatus,
    ReferenceTemporalKind,
)


def _definition(form=ChangeForm.LINEARIZED_RESPONSE):
    return GravityFieldDefinition.field_change(
        "halfspace_perturbation",
        Contribution("total", ("bulk", "surface")),
        PotentialReference.VERTICAL_FAR_FIELD_DECAY_POSITIVE_KH,
        ChangeDefinition(
            form,
            ReferenceDefinition(
                ReferenceStatus.DEFINED,
                ReferenceTemporalKind.STATE,
                "zero_displacement_halfspace",
            ),
            "displacement",
            order=1 if form is ChangeForm.LINEARIZED_RESPONSE else None,
        ),
    )


def _contract(**changes):
    from gragra.results._response import ResponseContract

    fields = {
        "input_name": "vertical_displacement",
        "input_quantity": "surface_vertical_displacement",
        "input_unit": u.m,
        "input_channel_ids": ("vertical_displacement",),
        "input_components": ("z",),
        "input_frame": Frame("root"),
        "input_location_m": np.array([1.0, 2.0, 0.0]),
        "input_distribution": "rayleigh_plane_wave",
        "normalization": "complex phasor amplitude at input_location",
        "basis_amplitude": 1 * u.m,
        "mapping_kind": "local_linearization",
        "action": "per_frequency",
        "storage": "precomputed",
        "varied": "source",
        "target_position": "fixed",
        "law_name": "newton",
        "law_parameters": {},
        "gravitational_constant_si": 6.67430e-11,
        "linearization_assessment": "not_assessed",
    }
    fields.update(changes)
    return ResponseContract(**fields)


def test_contract_owns_mutable_inputs_and_quantities():
    location = np.array([1.0, 2.0, 0.0])
    parameters = {"alpha": 2.0}
    contract = _contract(input_location_m=location, law_parameters=parameters)
    location[0] = 99.0
    parameters["alpha"] = 99.0
    np.testing.assert_array_equal(contract.input_location_m.to_value(u.m), [1, 2, 0])
    assert contract.law_parameters["alpha"] == 2.0
    contract.input_location_m.value[0] = -9
    assert contract.input_location_m[0] == 1 * u.m
    with pytest.raises(TypeError):
        contract.law_parameters["alpha"] = 3.0


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"input_name": "representation"}, "reserved"),
        ({"action": "other"}, "action"),
        ({"law_parameters": {"alpha": []}}, "law_parameters"),
        ({"basis_amplitude": 1 * u.s}, "basis_amplitude"),
        ({"input_location_m": np.array([1 + 2j, 0, 0])}, "input_location_m"),
    ],
)
def test_contract_rejects_invalid_descriptor(change, error):
    with pytest.raises(ValueError, match=error):
        _contract(**change)


class _FakeEvaluation:
    frequencies_hz = (1.0, 2.0)
    target_ids = ("a", "b")
    parameters = "parameters"
    formulas = "formulas"
    references = "references"

    def _applied_context(self, value, *, representation, reference_time):
        raise NotImplementedError


def _coefficients(bins=2):
    return {
        "potential": np.ones((2, 1, 1, bins), dtype=np.complex128),
        "acceleration": np.ones((2, 3, 1, bins), dtype=np.complex128),
        "gradient": np.ones((2, 9, 1, bins), dtype=np.complex128),
    }


def _response(contract=None, definition=None, evaluation=None, coefficients=None):
    from gragra._evaluation_setup import EvaluationRecord
    from gragra.results._response import GravityResponse

    return GravityResponse(
        contract=_contract() if contract is None else contract,
        output_definition=_definition() if definition is None else definition,
        evaluation=_FakeEvaluation() if evaluation is None else evaluation,
        coefficients=_coefficients() if coefficients is None else coefficients,
        record=EvaluationRecord(backend="test", versions={"gragra": "test"}),
    )


def test_response_constructor_owns_coefficients_and_frequencies():
    coefficients = _coefficients()
    response = _response(coefficients=coefficients)
    coefficients["potential"][0, 0, 0, 0] = 99
    assert response._coefficient_array("potential")[0, 0, 0, 0] == 1
    with pytest.raises(ValueError):
        response._coefficient_array("potential")[0, 0, 0, 0] = 9
    assert response.frequencies.unit == u.Hz
    assert response.target_ids == ("a", "b")
    assert response.observables == ("potential", "acceleration", "gradient")
    assert response.parameters == "parameters"


@pytest.mark.parametrize("action", ["instantaneous", "time_dependent"])
def test_response_constructor_rejects_unimplemented_action(action):
    with pytest.raises(ValueError, match="action"):
        _response(contract=_contract(action=action))


def test_response_constructor_rejects_local_linearization_mismatch():
    with pytest.raises(ValueError, match="output_definition"):
        _response(definition=_definition(ChangeForm.FINITE_STATE_DIFFERENCE))


def test_exact_linear_does_not_restrict_change_form():
    _response(contract=_contract(mapping_kind="exact_linear"))
    _response(
        contract=_contract(mapping_kind="exact_linear"),
        definition=_definition(ChangeForm.FINITE_STATE_DIFFERENCE),
    )


def test_response_spec_and_applied_spec_share_direct_interpretation():
    from gwexpy.frequencyseries import FrequencySeries

    from gragra._rayleigh_contract import RayleighModelSpec, RayleighResponseSpec
    from gragra.geometry._halfspace import HalfSpace
    from gragra.results._spectral import PhasorRepresentation

    frame = Frame("surface")
    shared = {
        "geometry": HalfSpace(frame),
        "density": 2300 * u.kg / u.m**3,
        "vp": 4000 * u.m / u.s,
        "vs": 2300 * u.m / u.s,
        "phase_origin": [1.0, 2.0, 0.0] * u.m,
        "direction": (3.0, 4.0, 0.0),
    }
    axis = np.array([0.75, 1.375, 5.25])
    input_series = FrequencySeries(
        [1 + 2j, -3j, 0.5],
        frequencies=axis * u.Hz,
        unit=u.nm,
        name="sensor",
        channel="TEST:Z",
    )
    response_spec = RayleighResponseSpec(**shared, frequencies=axis * u.Hz)
    direct = RayleighModelSpec(
        **shared,
        vertical_displacement=input_series,
        representation=PhasorRepresentation(),
        reference_time=(1600000000, 250000000),
    )
    applied = RayleighModelSpec._from_response(
        response_spec,
        input_series,
        PhasorRepresentation(),
        (1600000000, 250000000),
    )
    for field in fields(RayleighModelSpec):
        name = field.name
        assert getattr(applied, name) == getattr(direct, name)
    np.testing.assert_array_equal(
        applied.vertical_displacement_m, direct.vertical_displacement_m
    )
    np.testing.assert_array_equal(applied.phase_origin_m, direct.phase_origin_m)
    np.testing.assert_array_equal(applied.direction, direct.direction)
    assert applied.field_definition is response_spec.output_definition
    assert applied.field_definition == direct.field_definition


def test_response_spec_preserves_validation_priority():
    from gragra._rayleigh_contract import RayleighResponseSpec

    with pytest.raises(TypeError, match="geometry must be a HalfSpace"):
        RayleighResponseSpec(
            geometry=None,
            density=-1 * u.kg / u.m**3,
            vp=1 * u.m / u.s,
            vs=2 * u.m / u.s,
            frequencies=[0] * u.Hz,
            phase_origin=[0, 0, 1] * u.m,
            direction=(0, 0, 0),
        )


def test_response_spec_rejects_complex_frequency_axis():
    from gragra._rayleigh_contract import RayleighResponseSpec
    from gragra.geometry._halfspace import HalfSpace

    with pytest.raises(ValueError, match="frequencies must be real"):
        RayleighResponseSpec(
            geometry=HalfSpace(Frame("root")),
            density=2300 * u.kg / u.m**3,
            vp=4000 * u.m / u.s,
            vs=2300 * u.m / u.s,
            frequencies=np.array([1 + 1j, 2 + 0j]) * u.Hz,
            phase_origin=[0, 0, 0] * u.m,
            direction=(1, 0, 0),
        )


@pytest.mark.parametrize(
    ("changes", "error_type", "message"),
    [
        ({"geometry": None, "density": -1 * u.kg / u.m**3}, TypeError, "geometry"),
        ({"density": -1 * u.kg / u.m**3, "vp": 1 * u.m / u.s}, ValueError, "density"),
        (
            {"vp": 1 * u.m / u.s, "vs": 2 * u.m / u.s, "phase_origin": [0, 0, 1] * u.m},
            ValueError,
            "speeds",
        ),
        (
            {"phase_origin": [0, 0, 1] * u.m, "direction": (0, 0, 0)},
            ValueError,
            "phase_origin",
        ),
    ],
)
def test_direct_and_response_spec_keep_earliest_invalid_stage(
    changes, error_type, message
):
    from gwexpy.frequencyseries import FrequencySeries

    from gragra._rayleigh_contract import RayleighModelSpec, RayleighResponseSpec
    from gragra.geometry._halfspace import HalfSpace
    from gragra.results._spectral import PhasorRepresentation

    inputs = {
        "geometry": HalfSpace(Frame("root")),
        "density": 2300 * u.kg / u.m**3,
        "vp": 4000 * u.m / u.s,
        "vs": 2300 * u.m / u.s,
        "phase_origin": [0, 0, 0] * u.m,
        "direction": (1, 0, 0),
    }
    inputs.update(changes)
    with pytest.raises(error_type, match=message):
        RayleighResponseSpec(**inputs, frequencies=[1, 2] * u.Hz)
    with pytest.raises(error_type, match=message):
        RayleighModelSpec(
            **inputs,
            vertical_displacement=FrequencySeries(
                [1, 2], frequencies=[1, 2] * u.Hz, unit=u.m
            ),
            representation=PhasorRepresentation(),
            reference_time=(1, 0),
        )


def test_real_rayleigh_evaluation_constructs_response():
    from gragra._rayleigh_contract import (
        RayleighResponseEvaluation,
        RayleighResponseSpec,
    )
    from gragra._rayleigh_evaluation_setup import _rayleigh_generation_record
    from gragra.geometry._halfspace import HalfSpace
    from gragra.results._response import GravityResponse
    from gragra.targets._high_points import TargetPoints

    frame = Frame("surface")
    spec = RayleighResponseSpec(
        geometry=HalfSpace(frame),
        density=2300 * u.kg / u.m**3,
        vp=4000 * u.m / u.s,
        vs=2300 * u.m / u.s,
        frequencies=[0.75, 1.375, 5.25] * u.Hz,
        phase_origin=[1, 2, 0] * u.m,
        direction=(1, 0, 0),
    )

    class Model:
        def __init__(self, spec):
            self.spec = spec

    targets = TargetPoints([[0, 0, -1], [1, 0, 1]] * u.m, ("a", "b"), frame)
    evaluation = RayleighResponseEvaluation(Model(spec), targets, frame)
    response = GravityResponse(
        **evaluation._constructor_kwargs(
            coefficients=_coefficients(bins=3),
            record=_rayleigh_generation_record(),
        )
    )
    assert response.contract is spec.contract
    assert response.output_definition is spec.output_definition
    assert response.evaluation is evaluation
    assert response.target_ids == targets.ids
    from gwexpy.frequencyseries import FrequencySeries

    from gragra.results._spectral import PhasorRepresentation

    input_series = FrequencySeries(
        [1, 2j, -3],
        frequencies=[0.75, 1.375, 5.25] * u.Hz,
        unit=u.nm,
        name="sensor",
        channel="TEST:Z",
    )
    context = evaluation._applied_context(
        input_series,
        representation=PhasorRepresentation(),
        reference_time=(1600000000, 250000000),
    )
    assert context.evaluation.model_spec.field_definition is response.output_definition
    assert context.backend == "gravity_response:rayleigh_literature"
    np.testing.assert_array_equal(context.input_values, input_series.to_value(u.m))
    packed = context.pack(
        {
            name: np.ones((2, components, 3), dtype=np.complex128)
            for name, components in (
                ("potential", 1),
                ("acceleration", 3),
                ("gradient", 9),
            )
        }
    )
    assert tuple(packed.acceleration.rows.keys()) == targets.ids

    from gragra._plotting._rayleigh_setup import plot_rayleigh_setup

    figure = plot_rayleigh_setup(evaluation, view="xy")
    assert figure.axes[0].get_title()


def test_generic_phasor_matrix_uses_supplied_metadata():
    from gragra._phasor_matrix import _phasor_matrices

    data = {
        name: np.ones((2, channels, 3), dtype=np.complex128)
        for name, channels in (("potential", 1), ("acceleration", 3), ("gradient", 9))
    }
    packed = _phasor_matrices(
        data,
        ("a", "b"),
        (1.0, 2.0, 4.0),
        units={
            "potential": u.m / u.s**2,
            "acceleration": u.s**-2,
            "gradient": u.m**-1 * u.s**-2,
        },
        cell_name=lambda name, target, component, _row, _column: (
            f"{target}:{component}:{name}"
        ),
        cell_channel=lambda name, _target, _component, row, column: (
            f"H:{name}:{row}:{column}"
        ),
        matrix_name=lambda name: f"{name} response",
    )
    assert packed["potential"].name == "potential response"
    assert packed["acceleration"].meta[1, 2].name == "b:z:acceleration"
    assert packed["gradient"].meta[0, 0].unit == u.m**-1 * u.s**-2


def test_generation_record_contains_versions_and_one_g_application():
    from gragra._rayleigh_evaluation_setup import _rayleigh_generation_record

    record = _rayleigh_generation_record()
    assert record.backend == "rayleigh_literature"
    assert {"gragra", "python", "numpy", "astropy", "gwexpy"} <= set(record.versions)
    assert record.g_applications == 1


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"storage": "on_demand"}, "storage"),
        ({"law_name": "yukawa"}, "law_name"),
        ({"law_parameters": {"alpha": 1.0}}, "law_parameters"),
        ({"input_channel_ids": ("a", "b")}, "one input"),
        ({"linearization_assessment": "assessed"}, "linearization_assessment"),
        ({"varied": "target"}, "varied"),
        ({"target_position": "moving"}, "target_position"),
    ],
)
def test_response_constructor_rejects_unsupported_capability(changes, error):
    with pytest.raises(ValueError, match=error):
        _response(contract=_contract(**changes))


@pytest.mark.parametrize(
    ("observable", "replacement"),
    [
        ("potential", np.ones((2, 1, 2), dtype=np.complex128)),
        ("acceleration", np.ones((2, 3, 1, 2), dtype=np.float64)),
        ("gradient", np.full((2, 9, 1, 2), np.nan + 0j)),
    ],
)
def test_response_constructor_rejects_invalid_coefficient(observable, replacement):
    coefficients = _coefficients()
    coefficients[observable] = replacement
    with pytest.raises(ValueError, match=observable):
        _response(coefficients=coefficients)


def test_response_constructor_prioritizes_definition_before_capability():
    with pytest.raises(ValueError, match="output_definition"):
        _response(
            contract=_contract(action="instantaneous"),
            definition=_definition(ChangeForm.FINITE_STATE_DIFFERENCE),
        )


def test_response_constructor_rejects_evaluation_metadata_mismatch():
    contract = _contract()
    evaluation = _FakeEvaluation()
    evaluation.model_spec = SimpleNamespace(
        contract=_contract(), output_definition=_definition()
    )
    with pytest.raises(ValueError, match="evaluation.model_spec.contract"):
        _response(contract=contract, evaluation=evaluation)


def test_response_constructor_requires_configured_newton_g():
    with pytest.raises(ValueError, match="gravitational_constant_si"):
        _response(contract=_contract(gravitational_constant_si=1e-10))


@pytest.mark.parametrize(
    "order",
    [
        (
            "import gragra.results._response; import gragra._rayleigh_contract; "
            "import gragra.results._evaluation"
        ),
        (
            "import gragra.results._evaluation; import gragra._rayleigh_contract; "
            "import gragra.results._response"
        ),
        (
            "import gragra._rayleigh_contract; import gragra.results._response; "
            "import gragra.results._evaluation"
        ),
    ],
)
def test_fresh_process_import_order(order):
    completed = subprocess.run(
        [sys.executable, "-c", order], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
