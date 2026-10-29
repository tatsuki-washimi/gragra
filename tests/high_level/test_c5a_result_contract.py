"""Result operations use an external evaluation's stable shared context."""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable
from pandas import DataFrame

from gragra.geometry._frame import Frame
from gragra.results import (
    Contribution,
    GravityFieldDefinition,
    GravityResult,
    PotentialReference,
)
from gragra.results._comparison import compare
from gragra.results._context import EvaluationContext
from gragra.results._evaluation import EvaluationResult
from gragra.results._record import EvaluationRecord
from gragra.results._representation import StaticRepresentation
from gragra.targets._high_points import TargetPoints


@dataclass(frozen=True)
class _Interaction:
    name: str
    parameters: object = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class _Spec:
    field_definition: GravityFieldDefinition
    interaction: _Interaction


class _Evaluation:
    def __init__(self, context: EvaluationContext):
        self._context = context
        self.accesses = 0

    @property
    def context(self) -> EvaluationContext:
        self.accesses += 1
        return self._context

    @property
    def parameters(self) -> QTable:
        return QTable()

    @property
    def formulas(self) -> DataFrame:
        return DataFrame()

    @property
    def references(self) -> DataFrame:
        return DataFrame()

    def validate(self) -> None:
        raise AssertionError("producer validation must not run")

    def run(self) -> None:
        raise AssertionError("producer run must not run")


def _result(
    interaction_name: str = "law",
    *,
    frame: Frame | None = None,
    parameters: object = MappingProxyType({}),
) -> EvaluationResult:
    frame = frame or Frame("world")
    targets = TargetPoints(np.zeros((1, 3)) * u.m, ("one",), frame)
    definition = GravityFieldDefinition.source_field(
        "external", Contribution("total"), PotentialReference.ZERO_AT_INFINITY
    )
    context = EvaluationContext(
        model_spec=_Spec(definition, _Interaction(interaction_name, parameters)),
        targets=targets,
        frame=frame,
        observables=("acceleration", "potential"),
        representation=StaticRepresentation(),
    )
    evaluation = _Evaluation(context)
    result = EvaluationResult(
        GravityResult(
            acceleration=np.zeros((1, 3)) * u.m / u.s**2,
            potential=np.zeros(1) * u.m**2 / u.s**2,
        ),
        evaluation=evaluation,
        record=EvaluationRecord(backend="external", versions={}),
    )
    assert evaluation.accesses == 1
    return result


def test_selection_keeps_context_evaluation_payload_and_binding_identity() -> None:
    result = _result()
    selected = result.select("potential")
    assert result.context is result.evaluation.context
    assert selected.context is result.context
    assert selected.evaluation is result.evaluation
    assert selected.record is result.record
    assert selected.potential is result.potential
    assert selected.bindings["potential"] is result.bindings["potential"]
    assert selected.observables == ("potential",)
    assert selected.context.observables == ("acceleration", "potential")
    assert result.potential.value.tolist() == [0.0]


def test_compare_rejects_different_interactions_without_running_producer() -> None:
    with pytest.raises(ValueError, match="interaction"):
        compare(reference=_result("law-a"), candidate=_result("law-b"))
    with pytest.raises(ValueError, match="interaction"):
        compare(
            reference=_result(parameters=MappingProxyType({"scale": 1.0})),
            candidate=_result(parameters=MappingProxyType({"scale": 2.0})),
        )


@pytest.mark.parametrize(
    ("reference_value", "candidate_value"), [(1, 1.0), (1, True), (1.0, True)]
)
def test_compare_rejects_interaction_parameter_type_difference(
    reference_value: object, candidate_value: object
) -> None:
    frame = Frame("world")
    reference = _result(
        frame=frame,
        parameters=MappingProxyType({"scale": reference_value}),
    )
    candidate = _result(
        frame=frame,
        parameters=MappingProxyType({"scale": candidate_value}),
    )
    with pytest.raises(ValueError, match="interactions differ"):
        compare(reference=reference, candidate=candidate)


def test_result_and_compare_reject_interaction_parameter_drift() -> None:
    backing = {"scale": 1.0}
    reference = _result(parameters=MappingProxyType(backing))
    candidate = _result(parameters=MappingProxyType({"scale": 1.0}))
    backing["scale"] = 2.0
    with pytest.raises(ValueError, match="interaction.*parameters"):
        reference.validate()
    with pytest.raises(ValueError, match="interaction.*parameters"):
        compare(reference=reference, candidate=candidate)


def test_constructor_properties_select_and_compare_do_not_run_kernels(
    monkeypatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("result operation invoked a kernel")

    monkeypatch.setattr("gragra.evaluation._compute_values", forbidden)
    monkeypatch.setattr(
        "gragra.observables.rayleigh_literature.rayleigh_halfspace_allspace_observables",
        forbidden,
    )
    frame = Frame("world")
    reference = _result(frame=frame)
    candidate = _result(frame=frame)
    assert reference.parameters.colnames == []
    assert reference.formulas.empty
    assert reference.references.empty
    assert reference.select("potential").potential is reference.potential
    comparison = compare(reference=reference, candidate=candidate)
    np.testing.assert_array_equal(comparison.differences.potential.value, [0.0])


def test_real_uniform_sphere_result_operations_do_not_run_kernel(monkeypatch) -> None:
    from gragra.analytic.objects import uniform_sphere
    from gragra.geometry._sphere import Sphere

    frame = Frame("world")
    sphere = Sphere(np.zeros(3) * u.m, 1 * u.m, frame)
    targets = TargetPoints(np.array([[0.0, 0.0, 2.0]]) * u.m, ("one",), frame)
    result = uniform_sphere(sphere, 1 * u.kg / u.m**3).at(targets).run()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("uniform sphere kernel was called")

    monkeypatch.setattr("gragra.evaluation._compute_values", forbidden)
    monkeypatch.setattr(
        "gragra.observables.analytic.uniform_sphere_observables", forbidden
    )
    with pytest.raises(AssertionError, match="kernel was called"):
        result.evaluation.run()

    rebuilt = EvaluationResult(
        result.values, evaluation=result.evaluation, record=result.record
    )
    assert rebuilt.parameters.colnames
    assert not rebuilt.formulas.empty
    assert not rebuilt.references.empty
    selected = rebuilt.select("potential")
    comparison = compare(reference=selected, candidate=result.select("potential"))
    np.testing.assert_array_equal(comparison.differences.potential.value, [0.0])


def test_result_and_compare_have_no_concrete_model_imports_or_branches() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "gragra" / "results"
    banned_names = {
        "ResolvedEvaluation",
        "RayleighResolvedEvaluation",
        "PhasorObservableBinding",
    }
    for name in ("_evaluation.py", "_comparison.py"):
        tree = ast.parse((root / name).read_text())
        assert not banned_names.intersection(
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        )
        assert not any(
            isinstance(node, ast.Constant) and node.value in {"static_real", "newton"}
            for node in ast.walk(tree)
        )
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        imports = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        ]
        assert not any(
            module.startswith(("gragra._evaluation_setup", "gragra._rayleigh_"))
            for module in imports
        ), path
