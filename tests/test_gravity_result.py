"""Contract tests for the experimental GravityResult envelope."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from dataclasses import FrozenInstanceError, fields, is_dataclass

import numpy as np
import pytest


def _gravity_result_type():
    spec = importlib.util.find_spec("gragra.results")
    assert spec is not None, "the gragra.results package is missing"
    return importlib.import_module("gragra.results").GravityResult


@pytest.mark.parametrize(
    ("potential", "acceleration", "gradient"),
    [
        (object(), None, None),
        (None, object(), None),
        (None, None, object()),
        (object(), object(), None),
        (object(), None, object()),
        (None, object(), object()),
        (object(), object(), object()),
    ],
)
def test_any_nonempty_observable_mask_is_accepted(potential, acceleration, gradient):
    gravity_result_class = _gravity_result_type()

    result = gravity_result_class(
        potential=potential, acceleration=acceleration, gradient=gradient
    )

    assert result.potential is potential
    assert result.acceleration is acceleration
    assert result.gradient is gradient
    assert result.name is None


@pytest.mark.parametrize("kwargs", [{}, {"name": "label"}])
def test_all_none_observables_are_rejected_even_with_name(kwargs):
    gravity_result_class = _gravity_result_type()

    with pytest.raises(ValueError, match="at least one"):
        gravity_result_class(**kwargs)


@pytest.mark.parametrize("payload", [False, 0, np.array([])])
def test_falsey_payloads_are_provided(payload):
    gravity_result_class = _gravity_result_type()

    result = gravity_result_class(potential=payload)

    assert result.potential is payload


class _NoTruthValue:
    def __bool__(self):
        raise AssertionError("payload truth value must not be inspected")


def test_payload_with_raising_bool_is_provided_without_inspection():
    gravity_result_class = _gravity_result_type()
    payload = _NoTruthValue()

    result = gravity_result_class(potential=payload)

    assert result.potential is payload


@pytest.mark.parametrize("payload", [None, object()])
def test_non_string_name_raises_type_error_before_observable_validation(payload):
    gravity_result_class = _gravity_result_type()
    kwargs = {} if payload is None else {"potential": payload}

    with pytest.raises(TypeError, match="name"):
        gravity_result_class(name=3, **kwargs)


def test_empty_string_name_is_accepted():
    gravity_result_class = _gravity_result_type()

    result = gravity_result_class(name="", potential=object())

    assert result.name == ""


def test_payloads_are_retained_by_identity_and_mutation_is_shared():
    gravity_result_class = _gravity_result_type()
    payload = [1]
    result = gravity_result_class(potential=payload)

    assert result.potential is payload
    payload.append(2)
    assert result.potential == [1, 2]
    result.potential.append(3)
    assert payload == [1, 2, 3]


def test_payload_metadata_is_left_untouched():
    gravity_result_class = _gravity_result_type()

    class Payload:
        name = "cell label"
        channel = "target_a:x"

    payload = Payload()
    result = gravity_result_class(name="container label", acceleration=payload)

    assert result.acceleration is payload
    assert payload.name == "cell label"
    assert payload.channel == "target_a:x"
    assert result.name == "container label"


def test_only_outer_field_bindings_are_frozen():
    gravity_result_class = _gravity_result_type()
    payload = []
    result = gravity_result_class(potential=payload)

    with pytest.raises(FrozenInstanceError):
        result.potential = object()
    with pytest.raises(FrozenInstanceError):
        del result.potential

    payload.append("caller mutation")
    assert result.potential == ["caller mutation"]


def test_dataclass_fields_annotations_and_keyword_only_constructor():
    gravity_result_class = _gravity_result_type()

    assert is_dataclass(gravity_result_class)
    assert [field.name for field in fields(gravity_result_class)] == [
        "potential",
        "acceleration",
        "gradient",
        "name",
    ]
    signature = inspect.signature(gravity_result_class)
    assert list(signature.parameters) == [
        "potential",
        "acceleration",
        "gradient",
        "name",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )

    with pytest.raises(TypeError):
        gravity_result_class(object())


def test_numpy_payloads_do_not_use_fieldwise_equality():
    gravity_result_class = _gravity_result_type()
    payload = np.array([1.0, 2.0])
    left = gravity_result_class(potential=payload)
    right = gravity_result_class(potential=payload)

    assert left is not right
    assert left != right


def test_identity_hash_and_container_membership_survive_payload_mutation():
    gravity_result_class = _gravity_result_type()
    payload = [1]
    result = gravity_result_class(potential=payload)
    identity_hash = hash(result)
    mapping = {result: "stored"}
    members = {result}

    payload.append(2)

    assert mapping[result] == "stored"
    assert result in members
    assert hash(result) == identity_hash


@pytest.mark.parametrize(
    ("operation", "other"),
    [
        (lambda value, other: value + other, 1),
        (lambda value, other: value - other, 1),
        (lambda value, other: value * other, 1),
        (lambda value, other: value / other, 1),
    ],
)
def test_binary_arithmetic_is_unsupported(operation, other):
    gravity_result_class = _gravity_result_type()
    result = gravity_result_class(potential=object())

    with pytest.raises(TypeError):
        operation(result, other)


@pytest.mark.parametrize(
    "operation",
    [
        lambda left, right: left + right,
        lambda left, right: left - right,
        lambda left, right: left * right,
        lambda left, right: left / right,
    ],
)
def test_binary_arithmetic_between_results_is_unsupported(operation):
    gravity_result_class = _gravity_result_type()
    left = gravity_result_class(potential=object())
    right = gravity_result_class(acceleration=object())

    with pytest.raises(TypeError):
        operation(left, right)


def test_unary_negation_is_unsupported():
    gravity_result_class = _gravity_result_type()
    result = gravity_result_class(potential=object())

    with pytest.raises(TypeError):
        _ = -result


def test_subpackage_export_and_root_non_export_contract():
    gravity_result_class = _gravity_result_type()
    import gragra
    import gragra.results as results

    assert "GravityResult" in results.__all__
    assert results.GravityResult is gravity_result_class
    assert "GravityResult" not in gragra.__all__
    assert not hasattr(gragra, "GravityResult")
