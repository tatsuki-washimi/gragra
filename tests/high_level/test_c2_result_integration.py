"""Result and comparison consumers integrate with typed C2 field definitions."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeries

from gragra._evaluation_setup import (
    EvaluationRecord,
    ResolvedEvaluation,
    UniformSphereModelSpec,
)
from gragra.analytic.objects import uniform_sphere
from gragra.analytic.seismic import rayleigh_halfspace
from gragra.evaluation import EvaluationSetup
from gragra.geometry import Frame, HalfSpace, Sphere
from gragra.results import (
    ChangeDefinition,
    ChangeForm,
    Contribution,
    GravityFieldDefinition,
    GravityResult,
    PhasorRepresentation,
    PotentialReference,
    ReferenceDefinition,
    ReferenceStatus,
    ReferenceTemporalKind,
    compare,
)
from gragra.results._evaluation import EvaluationResult
from gragra.targets import TargetPoints


def _source_definition(
    *,
    source_kind: str = "test_source",
    contribution: Contribution | None = None,
    potential_reference: PotentialReference = PotentialReference.ZERO_AT_INFINITY,
) -> GravityFieldDefinition:
    return GravityFieldDefinition.source_field(
        source_kind,
        contribution or Contribution("total"),
        potential_reference,
    )


def _change_definition(
    *,
    source_kind: str = "test_change",
    contribution: Contribution | None = None,
    potential_reference: PotentialReference = PotentialReference.ZERO_AT_INFINITY,
    status: ReferenceStatus = ReferenceStatus.DEFINED,
    temporal_kind: ReferenceTemporalKind | None = ReferenceTemporalKind.STATE,
    state_kind: str | None = "baseline",
    origin: str | None = None,
    diagnostics_ref: str | None = None,
    form: ChangeForm = ChangeForm.LINEARIZED_RESPONSE,
    changed_input: str = "displacement",
    order: int | None = 1,
    fixed_conditions: tuple[str, ...] = ("density", "geometry"),
) -> GravityFieldDefinition:
    reference = ReferenceDefinition(
        status,
        temporal_kind,
        state_kind,
        origin=origin,
        diagnostics_ref=diagnostics_ref,
    )
    change = ChangeDefinition(
        form,
        reference,
        changed_input=changed_input,
        order=order,
        fixed_conditions=fixed_conditions,
    )
    return GravityFieldDefinition.field_change(
        source_kind,
        contribution or Contribution("total"),
        potential_reference,
        change,
    )


def _static_result(
    field_definition: GravityFieldDefinition | None = None,
    *,
    root: Frame | None = None,
    potential_value: float = 1.0,
) -> EvaluationResult:
    root = root or Frame("comparison-root")
    sphere = Sphere(np.zeros(3) * u.m, 1 * u.m, root)
    model_spec = UniformSphereModelSpec(sphere, 1 * u.kg / u.m**3)
    if field_definition is not None:
        # Test-only injection isolates compare projections from producer contracts.
        object.__setattr__(model_spec, "field_definition", field_definition)
    targets = TargetPoints([[0.0, 0.0, 2.0]] * u.m, ("A",), root)
    evaluation = ResolvedEvaluation(
        model_spec,
        targets,
        root,
        root,
        ("potential",),
    )
    return EvaluationResult(
        values=GravityResult(potential=np.array([potential_value]) * u.m**2 / u.s**2),
        evaluation=EvaluationSetup(evaluation),
        record=EvaluationRecord(backend="test", versions={"fixture": "1"}),
    )


def test_static_result_exposes_its_shared_typed_field_definition() -> None:
    frame = Frame("world")
    sphere = Sphere(np.zeros(3) * u.m, 1 * u.m, frame)
    targets = TargetPoints([[0.0, 0.0, 2.0]] * u.m, ("A",), frame)

    result = uniform_sphere(sphere, 1 * u.kg / u.m**3).at(targets).run()

    assert isinstance(result.field_definition, GravityFieldDefinition)
    assert result.field_definition is result.context.model_spec.field_definition
    assert all(
        binding.field_definition is result.field_definition
        for binding in result.bindings.values()
    )


def test_rayleigh_result_exposes_its_shared_typed_field_definition() -> None:
    frame = Frame("surface")
    targets = TargetPoints([[0.0, 0.0, -1.0]] * u.m, ("A",), frame)
    model = rayleigh_halfspace(
        geometry=HalfSpace(frame),
        density=2500 * u.kg / u.m**3,
        vp=4000 * u.m / u.s,
        vs=2300 * u.m / u.s,
        vertical_displacement=FrequencySeries(
            np.asarray([1 + 2j]), frequencies=np.asarray([1.0]) * u.Hz, unit=u.m
        ),
        representation=PhasorRepresentation(),
        reference_time=(1_000_000_000, 0),
        phase_origin=np.zeros(3) * u.m,
        direction=(1.0, 0.0, 0.0),
    )

    result = model.at(targets).run()

    assert isinstance(result.field_definition, GravityFieldDefinition)
    assert result.field_definition is result.context.model_spec.field_definition
    assert all(
        binding.field_definition is result.field_definition
        for binding in result.bindings.values()
    )


@pytest.mark.parametrize(
    ("left", "right", "compatible", "reason"),
    [
        (
            _source_definition(source_kind="sphere"),
            _source_definition(source_kind="point_cloud"),
            True,
            None,
        ),
        (_source_definition(), _change_definition(), False, "kind"),
        (
            _change_definition(status=ReferenceStatus.DEFINED),
            _change_definition(
                status=ReferenceStatus.UNKNOWN, temporal_kind=None, state_kind=None
            ),
            False,
            "status",
        ),
        (
            _change_definition(status=ReferenceStatus.DEFINED),
            _change_definition(
                status=ReferenceStatus.EXPLICIT_ZERO,
                temporal_kind=ReferenceTemporalKind.STATE,
                state_kind="baseline",
            ),
            False,
            "status",
        ),
        (
            _change_definition(
                status=ReferenceStatus.UNKNOWN, temporal_kind=None, state_kind=None
            ),
            _change_definition(
                status=ReferenceStatus.EXPLICIT_ZERO,
                temporal_kind=ReferenceTemporalKind.STATE,
                state_kind="baseline",
            ),
            False,
            "status",
        ),
        (
            _change_definition(origin="paper-A", diagnostics_ref="diag-A"),
            _change_definition(origin="paper-B", diagnostics_ref="diag-B"),
            True,
            None,
        ),
        (
            _change_definition(),
            _change_definition(temporal_kind=ReferenceTemporalKind.HISTORY),
            False,
            "temporal_kind",
        ),
        (
            _change_definition(),
            _change_definition(state_kind="another_baseline"),
            False,
            "state_kind",
        ),
        (
            _change_definition(
                status=ReferenceStatus.UNKNOWN, temporal_kind=None, state_kind=None
            ),
            _change_definition(
                status=ReferenceStatus.UNKNOWN, temporal_kind=None, state_kind=None
            ),
            True,
            None,
        ),
        (
            _change_definition(status=ReferenceStatus.UNKNOWN),
            _change_definition(status=ReferenceStatus.UNKNOWN),
            True,
            None,
        ),
        (
            _change_definition(
                status=ReferenceStatus.UNKNOWN, temporal_kind=None, state_kind=None
            ),
            _change_definition(
                status=ReferenceStatus.UNKNOWN,
                temporal_kind=ReferenceTemporalKind.STATE,
                state_kind=None,
            ),
            False,
            "temporal_kind",
        ),
        (
            _change_definition(status=ReferenceStatus.UNKNOWN, state_kind=None),
            _change_definition(status=ReferenceStatus.UNKNOWN, state_kind="baseline"),
            False,
            "state_kind",
        ),
        (
            _change_definition(
                status=ReferenceStatus.EXPLICIT_ZERO,
                temporal_kind=ReferenceTemporalKind.HISTORY,
                state_kind="reference_history",
            ),
            _change_definition(
                status=ReferenceStatus.EXPLICIT_ZERO,
                temporal_kind=ReferenceTemporalKind.HISTORY,
                state_kind="reference_history",
            ),
            True,
            None,
        ),
        (
            _change_definition(
                status=ReferenceStatus.EXPLICIT_ZERO,
                temporal_kind=ReferenceTemporalKind.STATE,
                state_kind="baseline",
            ),
            _change_definition(
                status=ReferenceStatus.EXPLICIT_ZERO,
                temporal_kind=ReferenceTemporalKind.HISTORY,
                state_kind="baseline",
            ),
            False,
            "temporal_kind",
        ),
        (
            _change_definition(
                status=ReferenceStatus.EXPLICIT_ZERO,
                temporal_kind=ReferenceTemporalKind.STATE,
                state_kind="baseline",
            ),
            _change_definition(
                status=ReferenceStatus.EXPLICIT_ZERO,
                temporal_kind=ReferenceTemporalKind.STATE,
                state_kind="another_baseline",
            ),
            False,
            "state_kind",
        ),
        (
            _source_definition(
                contribution=Contribution("selected", ("surface", "bulk"))
            ),
            _source_definition(
                contribution=Contribution("selected", ("bulk", "surface"))
            ),
            True,
            None,
        ),
        (
            _source_definition(contribution=Contribution("total")),
            _source_definition(contribution=Contribution("selected", ("bulk",))),
            False,
            "contribution",
        ),
        (
            _source_definition(
                contribution=Contribution("selected", ("bulk", "surface"))
            ),
            _source_definition(
                contribution=Contribution("selected", ("bulk", "volume"))
            ),
            False,
            "contribution",
        ),
        (
            _source_definition(),
            _source_definition(
                potential_reference=PotentialReference.VERTICAL_FAR_FIELD_DECAY_POSITIVE_KH
            ),
            False,
            "potential_reference",
        ),
        (
            _change_definition(),
            _change_definition(
                form=ChangeForm.FINITE_STATE_DIFFERENCE,
                changed_input="displacement",
                order=None,
            ),
            False,
            "form",
        ),
        (
            _change_definition(),
            _change_definition(changed_input="density"),
            False,
            "changed_input",
        ),
        (
            _change_definition(),
            _change_definition(fixed_conditions=("geometry", "vp")),
            False,
            "fixed_conditions",
        ),
    ],
)
def test_compare_uses_symmetric_c2_compatibility_projection(
    left: GravityFieldDefinition,
    right: GravityFieldDefinition,
    compatible: bool,
    reason: str | None,
) -> None:
    root = Frame("shared-comparison-root")
    reference = _static_result(left, root=root, potential_value=1.0)
    candidate = _static_result(right, root=root, potential_value=2.0)

    for first, second in ((reference, candidate), (candidate, reference)):
        if compatible:
            comparison = compare(reference=first, candidate=second)
            expected = 1.0 if first is reference else -1.0
            assert comparison.differences.potential.value[0] == pytest.approx(expected)
        else:
            assert reason is not None
            with pytest.raises(ValueError, match=f"field definition.*{reason}"):
                compare(reference=first, candidate=second)
