"""C2 contracts for the Rayleigh model's typed field definition."""

import numpy as np
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeries

from gragra._rayleigh_contract import (
    RayleighModelSpec,
    RayleighResolvedEvaluation,
)
from gragra._rayleigh_evaluation_setup import RayleighEvaluationSetup
from gragra.geometry._frame import Frame
from gragra.geometry._halfspace import HalfSpace
from gragra.results import (
    ChangeForm,
    Contribution,
    FieldKind,
    PhasorRepresentation,
    PotentialReference,
    ReferenceDefinition,
    ReferenceStatus,
    ReferenceTemporalKind,
)
from gragra.targets._high_points import TargetPoints


def _resolved() -> RayleighResolvedEvaluation:
    frame = Frame("root")
    spec = RayleighModelSpec(
        geometry=HalfSpace(frame),
        density=2500 * u.kg / u.m**3,
        vp=4000 * u.m / u.s,
        vs=2300 * u.m / u.s,
        vertical_displacement=FrequencySeries(
            np.asarray([1 + 2j, 3 + 4j]),
            frequencies=np.asarray([1.0, 2.0]) * u.Hz,
            unit=u.m,
        ),
        representation=PhasorRepresentation(),
        reference_time=(1_000_000_000, 0),
        phase_origin=np.zeros(3) * u.m,
        direction=(1.0, 0.0, 0.0),
    )

    class Model:
        pass

    model = Model()
    model.spec = spec
    targets = TargetPoints([[0.0, 0.0, -1.0]] * u.m, ("target",), frame)
    return RayleighResolvedEvaluation(model, targets, frame)


def test_rayleigh_spec_owns_canonical_field_definition() -> None:
    spec = _resolved().model_spec

    definition = spec.field_definition
    assert definition.kind is FieldKind.FIELD_CHANGE
    assert definition.source_kind == "halfspace_perturbation"
    assert definition.contribution == Contribution("total", ("bulk", "surface"))
    assert definition.contribution.components == ("bulk", "surface")
    assert (
        definition.potential_reference
        is PotentialReference.VERTICAL_FAR_FIELD_DECAY_POSITIVE_KH
    )
    assert definition.change is not None
    assert definition.change.form is ChangeForm.LINEARIZED_RESPONSE
    assert definition.change.reference == ReferenceDefinition(
        ReferenceStatus.DEFINED,
        ReferenceTemporalKind.STATE,
        "zero_displacement_halfspace",
    )
    assert definition.change.changed_input == "displacement"
    assert definition.change.order == 1
    assert definition.change.fixed_conditions == (
        "density",
        "free_surface_boundary",
        "geometry",
        "vp",
        "vs",
    )


def test_rayleigh_bindings_share_spec_definition_and_hide_legacy_fields() -> None:
    resolved = _resolved()
    context = RayleighEvaluationSetup(resolved).context
    bindings = {name: context.binding(name) for name in resolved.observables}

    assert set(bindings) == set(resolved.observables)
    assert all(
        binding.field_definition is resolved.model_spec.field_definition
        for binding in bindings.values()
    )
    assert not hasattr(resolved.model_spec, "contribution")
    assert not hasattr(resolved.model_spec, "reference_state")
    assert not hasattr(resolved.model_spec, "potential_reference")
    for binding in bindings.values():
        assert not hasattr(binding, "physical_meaning")
        assert not hasattr(binding, "contribution")
        assert not hasattr(binding, "reference_state")
        assert not hasattr(binding, "potential_reference")
