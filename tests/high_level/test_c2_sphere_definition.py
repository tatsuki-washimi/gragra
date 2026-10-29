"""Producer contract for the canonical uniform-sphere field definition."""

from __future__ import annotations

import inspect

import numpy as np
from astropy import units as u

from gragra._evaluation_setup import (
    ResolvedEvaluation,
    UniformSphereModelSpec,
)
from gragra.evaluation import EvaluationSetup
from gragra.geometry._frame import Frame
from gragra.geometry._sphere import Sphere
from gragra.results import (
    Contribution,
    FieldKind,
    GravityFieldDefinition,
    PotentialReference,
)
from gragra.targets._high_points import TargetPoints


def _evaluation() -> ResolvedEvaluation:
    frame = Frame("world")
    sphere = Sphere(np.zeros(3) * u.m, 1 * u.m, frame)
    model = UniformSphereModelSpec(sphere, 1 * u.kg / u.m**3)
    targets = TargetPoints(np.array([[0.0, 0.0, 2.0]]) * u.m, ("A",), frame)
    return ResolvedEvaluation(model, targets, frame, frame)


def test_uniform_sphere_spec_has_one_canonical_field_definition() -> None:
    evaluation = _evaluation()
    spec = evaluation.model_spec

    assert isinstance(spec.field_definition, GravityFieldDefinition)
    assert spec.field_definition.kind is FieldKind.SOURCE_FIELD
    assert spec.field_definition.source_kind == "uniform_sphere"
    assert spec.field_definition.contribution == Contribution("total")
    potential_reference = spec.field_definition.potential_reference
    assert potential_reference is PotentialReference.ZERO_AT_INFINITY
    assert spec.field_definition.change is None
    assert not hasattr(spec, "contribution")
    assert not hasattr(spec, "potential_reference")


def test_all_observable_bindings_share_the_spec_definition_by_identity() -> None:
    evaluation = _evaluation()

    context = EvaluationSetup(evaluation).context
    bindings = {name: context.binding(name) for name in evaluation.observables}

    assert set(bindings) == set(evaluation.observables)
    assert all(
        binding.field_definition is evaluation.model_spec.field_definition
        for binding in bindings.values()
    )


def test_bindings_drop_legacy_fields_and_contribution_override() -> None:
    evaluation = _evaluation()
    binding = EvaluationSetup(evaluation).context.binding("potential")

    assert not hasattr(binding, "physical_meaning")
    assert not hasattr(binding, "contribution")
    assert (
        "contribution"
        not in inspect.signature(EvaluationSetup(evaluation).context.binding).parameters
    )
