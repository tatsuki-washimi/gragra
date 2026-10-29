"""Canonical field-meaning descriptors for approved Gen1 model families."""

from __future__ import annotations

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

_FIELD_CHANGE = {
    "sca_wall": (
        "cavity_wall_perturbation",
        "displacement",
        ReferenceTemporalKind.STATE,
        "zero_displacement_cavity",
        PotentialReference.ZERO_AT_INFINITY,
        Contribution("selected", ("wall",)),
        ("density", "geometry"),
    ),
    "point_force": (
        "point_force_perturbation",
        "force_history",
        ReferenceTemporalKind.HISTORY,
        "zero_force_history",
        PotentialReference.ZERO_AT_INFINITY,
        Contribution("total"),
        ("direction", "position", "vp"),
    ),
    "acoustic_halfspace": (
        "acoustic_halfspace_perturbation",
        "pressure_perturbation",
        ReferenceTemporalKind.STATE,
        "zero_acoustic_pressure_perturbation",
        PotentialReference.ACOUSTIC_HALFSPACE_FOURIER_POSITIVE_Q,
        Contribution("total"),
        (
            "azimuth",
            "frequency",
            "geometry",
            "incidence_angle",
            "phase_origin",
            "sound_speed",
        ),
    ),
    "temperature": (
        "isobaric_temperature_perturbation",
        "temperature_perturbation",
        ReferenceTemporalKind.STATE,
        "zero_temperature_perturbation",
        PotentialReference.ZERO_AT_INFINITY,
        Contribution("total"),
        ("T0", "pressure", "rho0", "support"),
    ),
    "oscillation": (
        "point_mass_displacement_perturbation",
        "displacement",
        ReferenceTemporalKind.STATE,
        "stationary_point_mass_configuration",
        PotentialReference.ZERO_AT_INFINITY,
        Contribution("total"),
        ("masses", "reference_positions"),
    ),
    "acoustic_cavity": (
        "acoustic_cavity_perturbation",
        "pressure_perturbation",
        ReferenceTemporalKind.STATE,
        "zero_acoustic_pressure_perturbation",
        PotentialReference.ZERO_AT_INFINITY,
        Contribution("total"),
        ("geometry", "mode", "sound_speed"),
    ),
}

_SOURCE_FIELD = {"constant_speed", "rotation"}


def gen1_field_definition(
    model: str,
    *,
    periodic_temperature: bool = False,
) -> GravityFieldDefinition:
    """Create the approved field definition for one Gen1 model family.

    ``periodic_temperature`` selects the nonzero-wavevector periodic reference
    used by the plane-wave temperature model. Other temperature models retain
    the zero-at-infinity convention.
    """
    if not isinstance(model, str):
        raise TypeError("model must be a string")
    if not isinstance(periodic_temperature, bool):
        raise TypeError("periodic_temperature must be bool")
    if model in _SOURCE_FIELD:
        if periodic_temperature:
            raise ValueError("periodic_temperature applies only to temperature")
        return GravityFieldDefinition.source_field(
            source_kind="moving_point_masses",
            contribution=Contribution("total"),
            potential_reference=PotentialReference.ZERO_AT_INFINITY,
        )
    try:
        (
            source_kind,
            changed_input,
            temporal_kind,
            state_kind,
            potential_reference,
            contribution,
            fixed_conditions,
        ) = _FIELD_CHANGE[model]
    except KeyError as exc:
        raise ValueError(f"unknown Gen1 model {model!r}") from exc
    if periodic_temperature:
        if model != "temperature":
            raise ValueError("periodic_temperature applies only to temperature")
        potential_reference = PotentialReference.PERIODIC_ZERO_MEAN_NONZERO_K
    reference = ReferenceDefinition(
        status=ReferenceStatus.DEFINED,
        temporal_kind=temporal_kind,
        state_kind=state_kind,
    )
    change = ChangeDefinition(
        form=ChangeForm.LINEARIZED_RESPONSE,
        reference=reference,
        changed_input=changed_input,
        order=1,
        fixed_conditions=fixed_conditions,
    )
    return GravityFieldDefinition.field_change(
        source_kind=source_kind,
        contribution=contribution,
        potential_reference=potential_reference,
        change=change,
    )


__all__ = ["gen1_field_definition"]
