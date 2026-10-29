from dataclasses import FrozenInstanceError

import pytest

import gragra.results as results


def _resolve_enum_markers(value: object) -> object:
    if isinstance(value, str) and value.startswith("$"):
        enum_name, member_name = value[1:].split(".", 1)
        return getattr(getattr(results, enum_name), member_name)
    if isinstance(value, dict):
        return {key: _resolve_enum_markers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_enum_markers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_enum_markers(item) for item in value)
    return value


def test_field_definition_symbols_are_public_exports() -> None:
    symbols = {
        "FieldKind",
        "ReferenceStatus",
        "ReferenceTemporalKind",
        "ChangeForm",
        "PotentialReference",
        "ReferenceDefinition",
        "ChangeDefinition",
        "Contribution",
        "GravityFieldDefinition",
    }

    assert len(results.__all__) == 31
    assert symbols <= set(results.__all__)
    assert all(getattr(results, symbol) is not None for symbol in symbols)


def test_evaluation_context_types_and_interactions_are_public() -> None:
    import gragra
    import gragra.interactions as interactions

    result_names = {
        "EvaluationContext",
        "StaticRepresentation",
        "TimeSeriesRepresentation",
        "TimeSampling",
        "ModelSpec",
        "FieldEvaluation",
        "EvaluationRecord",
        "NumericalDiagnostic",
    }
    assert result_names <= set(results.__all__)
    assert all(getattr(results, name) is not None for name in result_names)
    assert interactions.__all__ == ["Interaction", "Newtonian"]
    assert interactions.Interaction is not None
    assert interactions.Newtonian is not None
    assert not (result_names | set(interactions.__all__)).intersection(gragra.__all__)


def test_enums_have_the_contract_values_and_string_behavior() -> None:
    expected = {
        "FieldKind": {"SOURCE_FIELD": "source_field", "FIELD_CHANGE": "field_change"},
        "ReferenceStatus": {
            "DEFINED": "defined",
            "UNKNOWN": "unknown",
            "EXPLICIT_ZERO": "explicit_zero",
        },
        "ReferenceTemporalKind": {"STATE": "state", "HISTORY": "history"},
        "ChangeForm": {
            "FINITE_STATE_DIFFERENCE": "finite_state_difference",
            "LINEARIZED_RESPONSE": "linearized_response",
        },
        "PotentialReference": {
            "ZERO_AT_INFINITY": "zero_at_infinity",
            "VERTICAL_FAR_FIELD_DECAY_POSITIVE_KH": (
                "vertical_far_field_decay_positive_kh"
            ),
            "SPHERICAL_MEAN_ZERO_AT_INFINITY": ("spherical_mean_zero_at_infinity"),
            "HALFSPACE_FOURIER_GREEN_POSITIVE_Q": (
                "halfspace_fourier_green_positive_q"
            ),
            "ACOUSTIC_HALFSPACE_FOURIER_POSITIVE_Q": (
                "acoustic_halfspace_fourier_positive_q"
            ),
            "PERIODIC_ZERO_MEAN_NONZERO_K": "periodic_zero_mean_nonzero_k",
        },
    }

    for enum_name, members in expected.items():
        enum_type = getattr(results, enum_name)
        assert {member.name: member.value for member in enum_type} == members
        assert all(isinstance(member, str) for member in enum_type)


def test_reference_definitions_cover_each_status_and_are_immutable() -> None:
    defined = results.ReferenceDefinition(
        results.ReferenceStatus.DEFINED,
        results.ReferenceTemporalKind.STATE,
        "equilibrium",
        "model.py:42",
        "diagnostics/run-1",
    )
    unknown = results.ReferenceDefinition(results.ReferenceStatus.UNKNOWN)
    unknown_history = results.ReferenceDefinition(
        results.ReferenceStatus.UNKNOWN,
        results.ReferenceTemporalKind.HISTORY,
        "unresolved_history",
    )
    explicit_zero = results.ReferenceDefinition(
        results.ReferenceStatus.EXPLICIT_ZERO,
        results.ReferenceTemporalKind.HISTORY,
        "zero_perturbation",
    )

    assert defined.temporal_kind is results.ReferenceTemporalKind.STATE
    assert defined.state_kind == "equilibrium"
    assert defined.origin == "model.py:42"
    assert defined.diagnostics_ref == "diagnostics/run-1"
    assert unknown.temporal_kind is None and unknown.state_kind is None
    assert unknown_history.temporal_kind is results.ReferenceTemporalKind.HISTORY
    assert explicit_zero.status is results.ReferenceStatus.EXPLICIT_ZERO
    with pytest.raises(FrozenInstanceError):
        defined.origin = "changed"


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"status": "defined"}, TypeError),
        ({"status": True}, TypeError),
        ({"status": "$ReferenceStatus.DEFINED"}, ValueError),
        (
            {
                "status": "$ReferenceStatus.DEFINED",
                "temporal_kind": "state",
                "state_kind": "equilibrium",
            },
            TypeError,
        ),
        (
            {
                "status": "$ReferenceStatus.DEFINED",
                "temporal_kind": "$ReferenceTemporalKind.STATE",
            },
            ValueError,
        ),
        (
            {
                "status": "$ReferenceStatus.DEFINED",
                "state_kind": "equilibrium",
            },
            ValueError,
        ),
        (
            {
                "status": "$ReferenceStatus.UNKNOWN",
                "state_kind": "unresolved",
            },
            ValueError,
        ),
        (
            {
                "status": "$ReferenceStatus.EXPLICIT_ZERO",
                "temporal_kind": "$ReferenceTemporalKind.STATE",
            },
            ValueError,
        ),
        (
            {
                "status": "$ReferenceStatus.EXPLICIT_ZERO",
                "state_kind": "zero",
            },
            ValueError,
        ),
        (
            {"status": "$ReferenceStatus.UNKNOWN", "temporal_kind": True},
            TypeError,
        ),
        (
            {"status": "$ReferenceStatus.UNKNOWN", "state_kind": ""},
            ValueError,
        ),
        (
            {
                "status": "$ReferenceStatus.UNKNOWN",
                "temporal_kind": "$ReferenceTemporalKind.STATE",
                "state_kind": 1,
            },
            TypeError,
        ),
        ({"status": "$ReferenceStatus.UNKNOWN", "origin": ""}, ValueError),
        ({"status": "$ReferenceStatus.UNKNOWN", "origin": False}, TypeError),
        (
            {"status": "$ReferenceStatus.UNKNOWN", "diagnostics_ref": ""},
            ValueError,
        ),
        (
            {"status": "$ReferenceStatus.UNKNOWN", "diagnostics_ref": 1},
            TypeError,
        ),
    ],
)
def test_reference_definition_rejects_invalid_fields(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        results.ReferenceDefinition(**_resolve_enum_markers(kwargs))


def test_change_definitions_validate_form_and_normalize_conditions() -> None:
    reference = results.ReferenceDefinition(
        results.ReferenceStatus.DEFINED,
        results.ReferenceTemporalKind.HISTORY,
        "baseline",
    )
    conditions = ["fixed_mass", "fixed_frame"]
    linear = results.ChangeDefinition(
        results.ChangeForm.LINEARIZED_RESPONSE,
        reference,
        "density",
        order=1,
        fixed_conditions=conditions,
    )
    conditions.append("later_mutation")
    finite = results.ChangeDefinition(
        results.ChangeForm.FINITE_STATE_DIFFERENCE,
        reference,
        "density",
    )

    assert linear.fixed_conditions == ("fixed_frame", "fixed_mass")
    assert finite.order is None and finite.fixed_conditions == ()
    with pytest.raises(FrozenInstanceError):
        linear.order = 2


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        (
            {
                "form": "finite_state_difference",
                "reference": None,
                "changed_input": "x",
            },
            TypeError,
        ),
        (
            {
                "form": "$ChangeForm.FINITE_STATE_DIFFERENCE",
                "reference": None,
                "changed_input": "x",
            },
            TypeError,
        ),
        (
            {
                "form": "$ChangeForm.FINITE_STATE_DIFFERENCE",
                "reference": "reference",
                "changed_input": "x",
            },
            TypeError,
        ),
        (
            {
                "form": "$ChangeForm.FINITE_STATE_DIFFERENCE",
                "reference": "valid",
                "changed_input": "",
            },
            ValueError,
        ),
        (
            {
                "form": "$ChangeForm.FINITE_STATE_DIFFERENCE",
                "reference": "valid",
                "changed_input": True,
            },
            TypeError,
        ),
        (
            {
                "form": "$ChangeForm.FINITE_STATE_DIFFERENCE",
                "reference": "valid",
                "changed_input": "x",
                "order": 1,
            },
            ValueError,
        ),
        (
            {
                "form": "$ChangeForm.FINITE_STATE_DIFFERENCE",
                "reference": "valid",
                "changed_input": "x",
                "order": True,
            },
            TypeError,
        ),
        (
            {
                "form": "$ChangeForm.LINEARIZED_RESPONSE",
                "reference": "valid",
                "changed_input": "x",
            },
            ValueError,
        ),
        (
            {
                "form": "$ChangeForm.LINEARIZED_RESPONSE",
                "reference": "valid",
                "changed_input": "x",
                "order": 2,
            },
            ValueError,
        ),
        (
            {
                "form": "$ChangeForm.LINEARIZED_RESPONSE",
                "reference": "valid",
                "changed_input": "x",
                "order": True,
            },
            TypeError,
        ),
        (
            {
                "form": "$ChangeForm.LINEARIZED_RESPONSE",
                "reference": "valid",
                "changed_input": "x",
                "order": "1",
            },
            TypeError,
        ),
        (
            {
                "form": "$ChangeForm.LINEARIZED_RESPONSE",
                "reference": "valid",
                "changed_input": "x",
                "order": 1,
                "fixed_conditions": "frame",
            },
            TypeError,
        ),
        (
            {
                "form": "$ChangeForm.LINEARIZED_RESPONSE",
                "reference": "valid",
                "changed_input": "x",
                "order": 1,
                "fixed_conditions": [""],
            },
            ValueError,
        ),
        (
            {
                "form": "$ChangeForm.LINEARIZED_RESPONSE",
                "reference": "valid",
                "changed_input": "x",
                "order": 1,
                "fixed_conditions": [1],
            },
            TypeError,
        ),
        (
            {
                "form": "$ChangeForm.LINEARIZED_RESPONSE",
                "reference": "valid",
                "changed_input": "x",
                "order": 1,
                "fixed_conditions": ["frame", "frame"],
            },
            ValueError,
        ),
    ],
)
def test_change_definition_rejects_invalid_fields(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    kwargs = _resolve_enum_markers(kwargs)
    if kwargs["reference"] == "valid":
        kwargs["reference"] = results.ReferenceDefinition(
            results.ReferenceStatus.UNKNOWN
        )
    with pytest.raises(error):
        results.ChangeDefinition(**kwargs)


def test_contribution_normalizes_components_and_checks_coverage() -> None:
    components = ["potential", "acceleration"]
    selected = results.Contribution("selected", components)
    components.append("gradient")
    total = results.Contribution("total")

    assert selected.components == ("acceleration", "potential")
    assert total.components == ()
    with pytest.raises(FrozenInstanceError):
        selected.coverage = "total"


@pytest.mark.parametrize(
    ("coverage", "components", "error"),
    [
        (True, (), TypeError),
        ("unknown", (), ValueError),
        ("selected", (), ValueError),
        ("total", "potential", TypeError),
        ("total", [""], ValueError),
        ("selected", ["potential", "potential"], ValueError),
        ("selected", [1], TypeError),
    ],
)
def test_contribution_rejects_invalid_values(
    coverage: object, components: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        results.Contribution(coverage, components)


def test_gravity_field_definition_builds_source_and_change_fields() -> None:
    contribution = results.Contribution("total")
    reference = results.PotentialReference.ZERO_AT_INFINITY
    source = results.GravityFieldDefinition.source_field(
        "point_mass", contribution, reference
    )
    change = results.ChangeDefinition(
        results.ChangeForm.LINEARIZED_RESPONSE,
        results.ReferenceDefinition(
            results.ReferenceStatus.UNKNOWN,
            results.ReferenceTemporalKind.STATE,
        ),
        "source_strength",
        order=1,
    )
    changed = results.GravityFieldDefinition.field_change(
        "point_mass", contribution, reference, change
    )

    assert source.kind is results.FieldKind.SOURCE_FIELD
    assert source.change is None
    assert changed.kind is results.FieldKind.FIELD_CHANGE
    assert changed.change is change
    assert changed.change.reference.temporal_kind is results.ReferenceTemporalKind.STATE
    with pytest.raises(FrozenInstanceError):
        source.source_kind = "changed_source"


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        (
            {
                "kind": "source_field",
                "source_kind": "mass",
                "contribution": None,
                "potential_reference": "$PotentialReference.ZERO_AT_INFINITY",
            },
            TypeError,
        ),
        (
            {
                "kind": "$FieldKind.SOURCE_FIELD",
                "source_kind": "",
                "contribution": "valid",
                "potential_reference": "$PotentialReference.ZERO_AT_INFINITY",
            },
            ValueError,
        ),
        (
            {
                "kind": "$FieldKind.SOURCE_FIELD",
                "source_kind": 1,
                "contribution": "valid",
                "potential_reference": "$PotentialReference.ZERO_AT_INFINITY",
            },
            TypeError,
        ),
        (
            {
                "kind": "$FieldKind.SOURCE_FIELD",
                "source_kind": "mass",
                "contribution": "valid",
                "potential_reference": "zero_at_infinity",
            },
            TypeError,
        ),
        (
            {
                "kind": "$FieldKind.SOURCE_FIELD",
                "source_kind": "mass",
                "contribution": "valid",
                "potential_reference": "$PotentialReference.ZERO_AT_INFINITY",
                "change": "change",
            },
            ValueError,
        ),
        (
            {
                "kind": "$FieldKind.FIELD_CHANGE",
                "source_kind": "mass",
                "contribution": "valid",
                "potential_reference": "$PotentialReference.ZERO_AT_INFINITY",
            },
            ValueError,
        ),
        (
            {
                "kind": "$FieldKind.FIELD_CHANGE",
                "source_kind": "mass",
                "contribution": "valid",
                "potential_reference": "$PotentialReference.ZERO_AT_INFINITY",
                "change": "change",
            },
            TypeError,
        ),
    ],
)
def test_gravity_field_definition_rejects_invalid_fields(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    kwargs = _resolve_enum_markers(kwargs)
    if kwargs["contribution"] == "valid":
        kwargs["contribution"] = results.Contribution("total")
    if (
        kwargs.get("kind") is results.FieldKind.SOURCE_FIELD
        and kwargs.get("change") == "change"
    ):
        kwargs["change"] = results.ChangeDefinition(
            results.ChangeForm.FINITE_STATE_DIFFERENCE,
            results.ReferenceDefinition(results.ReferenceStatus.UNKNOWN),
            "mass",
        )
    with pytest.raises(error):
        results.GravityFieldDefinition(**kwargs)


def test_gravity_field_definition_classmethods_reject_invalid_inputs() -> None:
    with pytest.raises(TypeError):
        results.GravityFieldDefinition.source_field(
            "mass", "not a contribution", results.PotentialReference.ZERO_AT_INFINITY
        )
    with pytest.raises(TypeError):
        results.GravityFieldDefinition.field_change(
            "mass",
            results.Contribution("total"),
            results.PotentialReference.ZERO_AT_INFINITY,
            "not a change",
        )
