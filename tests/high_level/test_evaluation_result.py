"""Static evaluation result views preserve payload identity and context."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from astropy import units as u

from gragra._evaluation_setup import (
    DiscreteSource,
    EvaluationRecord,
    ResolvedEvaluation,
    UniformSphereModelSpec,
)
from gragra.evaluation import EvaluationSetup
from gragra.geometry._frame import Frame
from gragra.geometry._sphere import Sphere
from gragra.results import GravityResult
from gragra.results._evaluation import EvaluationResult
from gragra.targets._high_points import TargetPoints


def _make_result(
    *,
    observables: tuple[str, ...] = ("potential", "acceleration", "gradient"),
    values: dict[str, u.Quantity] | None = None,
    ids: tuple[str, ...] = ("A", "B"),
    positions: u.Quantity | None = None,
    root: Frame | None = None,
    target_frame: Frame | None = None,
    output_frame: Frame | None = None,
    source: bool = False,
    backend: str = "numpy",
    g_si: float = 6.67430e-11,
) -> EvaluationResult:
    root = root or Frame("world")
    target_frame = target_frame or root
    output_frame = output_frame or root
    if positions is None:
        positions = (
            np.arange(1, len(ids) + 1, dtype=float)[:, None]
            * np.array([[0.0, 0.0, 1.0]])
            * u.m
        )
    sphere = Sphere(np.zeros(3) * u.m, 1 * u.m, root)
    spec = UniformSphereModelSpec(sphere, 1 * u.kg / u.m**3)
    targets = TargetPoints(positions, ids, target_frame)
    discrete_source = None
    if source:
        discrete_source = DiscreteSource(
            sphere,
            spec.density,
            root,
            [[0.0, 0.0, 0.0]],
            [1.0],
            {"spacing_m": 0.25, "max_points": 1000, "grid_points": 64},
        )
    evaluation = ResolvedEvaluation(
        spec,
        targets,
        root,
        output_frame,
        observables,
        discrete_source,
    )
    if values is None:
        defaults = {
            "potential": np.arange(1, len(ids) + 1, dtype=float) * u.m**2 / u.s**2,
            "acceleration": np.ones((len(ids), 3)) * u.m / u.s**2,
            "gradient": np.ones((len(ids), 3, 3)) / u.s**2,
        }
        values = {name: defaults[name] for name in observables}
    gravity_values = GravityResult(
        potential=values.get("potential"),
        acceleration=values.get("acceleration"),
        gradient=values.get("gradient"),
    )
    record = EvaluationRecord(
        backend=backend,
        versions={"fixture": "1"},
        gravitational_constant_si=g_si,
    )
    return EvaluationResult(
        values=gravity_values,
        evaluation=EvaluationSetup(evaluation),
        record=record,
    )


def test_result_exposes_native_quantities_and_delegates_original_context() -> None:
    potential = np.array([-2.0, -3.0]) * u.m**2 / u.s**2
    acceleration = np.arange(6.0).reshape(2, 3) * u.m / u.s**2
    gradient = np.arange(18.0).reshape(2, 3, 3) / u.s**2
    result = _make_result(
        values={
            "potential": potential,
            "acceleration": acceleration,
            "gradient": gradient,
        },
        source=True,
        backend="direct",
    )

    assert result.potential is potential
    assert result.acceleration is acceleration
    assert result.gradient is gradient
    assert result.values.potential is potential
    assert result.evaluation.resolved.source is not None
    assert result.parameters["continuous_mass"].unit == u.kg
    assert any("sum_j" in expression for expression in result.formulas["expression"])
    assert "gragra.observables.potential.point_potential" in set(
        result.references["source"]
    )
    assert result.record.backend == result.backend == "direct"
    assert result.versions == {"fixture": "1"}
    assert result.g_applications == 1
    assert result.gravitational_constant_si == 6.67430e-11


def test_select_keeps_payload_identity_context_and_current_mutated_values() -> None:
    acceleration = np.arange(6.0).reshape(2, 3) * u.m / u.s**2
    result = _make_result(
        observables=("potential", "acceleration"),
        values={
            "potential": np.ones(2) * u.m**2 / u.s**2,
            "acceleration": acceleration,
        },
    )

    selected = result.select("acceleration")
    assert selected.acceleration is acceleration
    assert selected.values.acceleration is acceleration
    assert selected.values.potential is None
    assert selected.bindings["acceleration"] is result.bindings["acceleration"]
    assert selected.evaluation is result.evaluation
    assert selected.record is result.record
    acceleration[0, 0] = 42 * u.m / u.s**2
    assert selected.acceleration[0, 0].to_value(u.m / u.s**2) == 42


def test_select_rejects_empty_unknown_duplicate_and_unavailable_requests() -> None:
    result = _make_result(observables=("potential",))
    with pytest.raises(ValueError, match="at least one|requires at least one"):
        result.select()
    with pytest.raises(ValueError, match="unknown"):
        result.select("density")
    with pytest.raises(ValueError, match="unique"):
        result.select("potential", "potential")
    with pytest.raises(ValueError, match="not present"):
        result.select("acceleration")


def test_operation_revalidates_mutable_quantity_shape() -> None:
    acceleration = np.ones((2, 3)) * u.m / u.s**2
    result = _make_result(
        observables=("acceleration",), values={"acceleration": acceleration}
    )
    acceleration.shape = (3, 2)
    with pytest.raises(ValueError, match="acceleration must have shape"):
        result.select("acceleration")


def test_operation_revalidates_mutated_quantity_unit() -> None:
    potential = np.ones(2) * u.m**2 / u.s**2
    result = _make_result(observables=("potential",), values={"potential": potential})
    potential._unit = u.m
    with pytest.raises(ValueError, match="units equivalent"):
        _ = result.potential


@pytest.mark.parametrize(
    ("name", "quantity", "message"),
    [
        ("potential", np.ones(2) * u.m, "units equivalent"),
        ("potential", np.ones(3) * u.m**2 / u.s**2, "shape"),
        ("potential", np.array([1.0, np.nan]) * u.m**2 / u.s**2, "finite"),
        ("potential", np.array([1 + 1j, 2]) * u.m**2 / u.s**2, "real"),
    ],
)
def test_constructor_rejects_malformed_native_quantity(
    name: str, quantity: u.Quantity, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _make_result(observables=(name,), values={name: quantity})


def test_constructor_rejects_equal_but_distinct_field_definition_and_g_record() -> None:
    result = _make_result(observables=("potential",))
    definition = result.field_definition
    context = result.context
    object.__setattr__(context.model_spec, "field_definition", replace(definition))
    assert context.model_spec.field_definition == definition
    assert context.model_spec.field_definition is not definition
    with pytest.raises(ValueError, match="field_definition changed"):
        result.validate()

    fresh = _make_result(observables=("potential",))
    noncanonical_record = EvaluationRecord(
        backend="numpy",
        versions={"fixture": "1"},
        gravitational_constant_si=2 * fresh.gravitational_constant_si,
    )
    with pytest.raises(ValueError, match="configured G_SI"):
        EvaluationResult(
            values=fresh.values,
            evaluation=fresh.evaluation,
            record=noncanonical_record,
        )


def test_context_owns_binding_axes_and_rejects_inconsistent_g_convention() -> None:
    result = _make_result(observables=("potential",))
    binding = result.bindings["potential"]
    assert binding.component_axes == ()
    assert binding.representation is result.context.representation
    object.__setattr__(result.context, "g_convention", "G_applied_twice")
    with pytest.raises(ValueError, match="G convention"):
        result.validate()
