"""Cross-factory assertions for the adopted Generation 1 result contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeries

from gragra.analytic.objects import uniform_cuboid, uniform_cylinder, uniform_sphere
from gragra.analytic.seismic import (
    halfspace_psv,
    rayleigh_halfspace,
    rayleigh_response,
    spherical_cavity,
    spherical_shell,
)
from gragra.geometry import Cuboid, Cylinder, Frame, HalfSpace, Sphere, SphericalShell
from gragra.results import (
    ChangeForm,
    Contribution,
    FieldKind,
    PhasorRepresentation,
    PotentialReference,
    StaticRepresentation,
)
from gragra.targets import TargetPoints


@dataclass(frozen=True)
class ContractCase:
    """Independent expected contract for one public factory route."""

    name: str
    physical_setup: str
    representation: str
    kind: FieldKind
    source_kind: str
    contribution: Contribution
    potential_reference: PotentialReference
    change_form: ChangeForm | None


# Keep physical inputs and expected semantics in the same explicit table. The
# builder below supplies these inputs; expected values are not copied from the
# result being checked.
CASES = (
    ContractCase(
        "uniform_sphere",
        "static 1 m radius sphere, density 2500 kg/m^3",
        "static",
        FieldKind.SOURCE_FIELD,
        "uniform_sphere",
        Contribution("total"),
        PotentialReference.ZERO_AT_INFINITY,
        None,
    ),
    ContractCase(
        "uniform_cuboid",
        "static 4 m cube, density 2500 kg/m^3",
        "static",
        FieldKind.SOURCE_FIELD,
        "uniform_cuboid",
        Contribution("total"),
        PotentialReference.ZERO_AT_INFINITY,
        None,
    ),
    ContractCase(
        "uniform_cylinder",
        "static radius 2 m, height 4 m cylinder, density 2500 kg/m^3",
        "static",
        FieldKind.SOURCE_FIELD,
        "uniform_cylinder",
        Contribution("total"),
        PotentialReference.ZERO_AT_INFINITY,
        None,
    ),
    ContractCase(
        "halfspace_psv",
        "SV phasor in elastic half-space; positive horizontal wavenumber",
        "phasor",
        FieldKind.FIELD_CHANGE,
        "halfspace_psv_perturbation",
        Contribution("total"),
        PotentialReference.HALFSPACE_FOURIER_GREEN_POSITIVE_Q,
        ChangeForm.LINEARIZED_RESPONSE,
    ),
    ContractCase(
        "rayleigh_halfspace",
        "vertical Rayleigh displacement phasor in an elastic half-space",
        "phasor",
        FieldKind.FIELD_CHANGE,
        "halfspace_perturbation",
        Contribution("total", ("bulk", "surface")),
        PotentialReference.VERTICAL_FAR_FIELD_DECAY_POSITIVE_KH,
        ChangeForm.LINEARIZED_RESPONSE,
    ),
    ContractCase(
        "rayleigh_response.apply",
        "Rayleigh field after applying coefficients to displacement",
        "phasor",
        FieldKind.FIELD_CHANGE,
        "halfspace_perturbation",
        Contribution("total", ("bulk", "surface")),
        PotentialReference.VERTICAL_FAR_FIELD_DECAY_POSITIVE_KH,
        ChangeForm.LINEARIZED_RESPONSE,
    ),
    ContractCase(
        "spherical_cavity",
        "P-wave linear response of radius 1 m cavity, total bulk and wall",
        "phasor",
        FieldKind.FIELD_CHANGE,
        "spherical_cavity_perturbation",
        Contribution("total", ("bulk", "wall")),
        PotentialReference.SPHERICAL_MEAN_ZERO_AT_INFINITY,
        ChangeForm.LINEARIZED_RESPONSE,
    ),
    ContractCase(
        "spherical_shell",
        "P-wave linear response of 0.7–5 m shell, total bulk and boundaries",
        "phasor",
        FieldKind.FIELD_CHANGE,
        "spherical_shell_perturbation",
        Contribution("total", ("bulk", "inner_wall", "outer_boundary")),
        PotentialReference.ZERO_AT_INFINITY,
        ChangeForm.LINEARIZED_RESPONSE,
    ),
)

_FREQUENCIES_HZ = np.array([1.25, 2.75])
_AMPLITUDES_M = np.array([1.0e-5 + 2.0e-6j, -3.0e-6 + 4.0e-6j])
_REFERENCE_TIME = (1_000_000_000, 125_000_000)
_PHASE_ORIGIN_M = np.array([0.25, -0.4, 0.7])
_RAYLEIGH_PHASE_ORIGIN_M = np.array([0.25, -0.4, 0.0])


def _inputs():
    frame = Frame("world")
    geometry_sphere = Sphere(np.zeros(3) * u.m, 1.0 * u.m, frame)
    displacement = FrequencySeries(
        _AMPLITUDES_M.copy(), frequencies=_FREQUENCIES_HZ * u.Hz, unit=u.m
    )
    representation = PhasorRepresentation()
    phase_origin = _PHASE_ORIGIN_M * u.m
    targets = TargetPoints(
        np.array([[0.0, 0.0, 10.0], [2.0, -1.0, 12.0]]) * u.m,
        ("far-a", "far-b"),
        frame,
    )
    return {
        "frame": frame,
        "sphere": geometry_sphere,
        "displacement": displacement,
        "representation": representation,
        "phase_origin": phase_origin,
        "targets": targets,
    }


def _model(case: ContractCase, shared: dict[str, object]):
    frame = shared["frame"]
    sphere = shared["sphere"]
    if case.name == "uniform_sphere":
        return uniform_sphere(sphere, 2500 * u.kg / u.m**3)
    if case.name == "uniform_cuboid":
        return uniform_cuboid(
            Cuboid(np.zeros(3) * u.m, np.full(3, 4.0) * u.m, frame),
            2500 * u.kg / u.m**3,
        )
    if case.name == "uniform_cylinder":
        return uniform_cylinder(
            Cylinder(np.zeros(3) * u.m, 2.0 * u.m, 4.0 * u.m, frame),
            2500 * u.kg / u.m**3,
        )
    if case.name == "halfspace_psv":
        return halfspace_psv(
            HalfSpace(frame),
            2500 * u.kg / u.m**3,
            4000 * u.m / u.s,
            2300 * u.m / u.s,
            shared["displacement"],
            shared["representation"],
            _REFERENCE_TIME,
            "SV",
            0.4,
            0.65,
            shared["phase_origin"],
        )
    if case.name in {"rayleigh_halfspace", "rayleigh_response.apply"}:
        common = {
            "geometry": HalfSpace(frame),
            "density": 2500 * u.kg / u.m**3,
            "vp": 4000 * u.m / u.s,
            "vs": 2300 * u.m / u.s,
            # Rayleigh inputs are anchored on the local free surface z=0.
            "phase_origin": _RAYLEIGH_PHASE_ORIGIN_M * u.m,
            "direction": np.array([1.0, 0.0, 0.0]),
        }
        if case.name == "rayleigh_halfspace":
            return rayleigh_halfspace(
                **common,
                vertical_displacement=shared["displacement"],
                representation=shared["representation"],
                reference_time=_REFERENCE_TIME,
            )
        response = (
            rayleigh_response(
                **common,
                frequencies=_FREQUENCIES_HZ * u.Hz,
            )
            .at(shared["targets"])
            .run()
        )
        return (
            response,
            response.apply(
                vertical_displacement=shared["displacement"],
                representation=shared["representation"],
                reference_time=_REFERENCE_TIME,
            ),
        )
    if case.name == "spherical_cavity":
        return spherical_cavity(
            sphere,
            2500 * u.kg / u.m**3,
            3000 * u.m / u.s,
            shared["displacement"],
            shared["representation"],
            _REFERENCE_TIME,
            "P",
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
            shared["phase_origin"],
            part="total",
        )
    if case.name == "spherical_shell":
        return spherical_shell(
            SphericalShell(np.zeros(3) * u.m, 0.7 * u.m, 5.0 * u.m, frame),
            2500 * u.kg / u.m**3,
            3000 * u.m / u.s,
            shared["displacement"],
            shared["representation"],
            _REFERENCE_TIME,
            shared["phase_origin"],
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
            "P",
            contribution="total",
        )
    raise AssertionError(f"unmapped contract case: {case.name}")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_public_factories_produce_the_declared_common_field_contract(
    case: ContractCase,
) -> None:
    shared = _inputs()
    model_or_result = _model(case, shared)
    if case.name == "rayleigh_response.apply":
        response, result = model_or_result
        assert response.output_definition is result.context.field_definition
    else:
        result = model_or_result.at(shared["targets"]).run()

    definition = result.context.field_definition
    assert definition.kind is case.kind
    assert definition.source_kind == case.source_kind
    assert definition.contribution == case.contribution
    assert definition.potential_reference is case.potential_reference
    change_form = None if definition.change is None else definition.change.form
    assert change_form is case.change_form
    assert result.observables == ("potential", "acceleration", "gradient")
    expected_representation = (
        StaticRepresentation
        if case.representation == "static"
        else PhasorRepresentation
    )
    assert isinstance(result.context.representation, expected_representation)
    assert result.g_applications == 1
    assert result.context.frame is shared["targets"].frame
