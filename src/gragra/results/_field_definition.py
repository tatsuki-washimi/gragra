"""Immutable descriptors for the meaning and provenance of gravity fields."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FieldKind(StrEnum):
    """Distinguish a source field from a field change."""

    SOURCE_FIELD = "source_field"
    FIELD_CHANGE = "field_change"


class ReferenceStatus(StrEnum):
    """Describe whether a field reference is defined, unknown, or zero."""

    DEFINED = "defined"
    UNKNOWN = "unknown"
    EXPLICIT_ZERO = "explicit_zero"


class ReferenceTemporalKind(StrEnum):
    """Temporal interpretation of a field reference."""

    STATE = "state"
    HISTORY = "history"


class ChangeForm(StrEnum):
    """Mathematical form used to define a field change."""

    FINITE_STATE_DIFFERENCE = "finite_state_difference"
    LINEARIZED_RESPONSE = "linearized_response"


class PotentialReference(StrEnum):
    """Reference convention for a potential field."""

    ZERO_AT_INFINITY = "zero_at_infinity"
    VERTICAL_FAR_FIELD_DECAY_POSITIVE_KH = "vertical_far_field_decay_positive_kh"
    SPHERICAL_MEAN_ZERO_AT_INFINITY = "spherical_mean_zero_at_infinity"
    HALFSPACE_FOURIER_GREEN_POSITIVE_Q = "halfspace_fourier_green_positive_q"
    ACOUSTIC_HALFSPACE_FOURIER_POSITIVE_Q = "acoustic_halfspace_fourier_positive_q"
    PERIODIC_ZERO_MEAN_NONZERO_K = "periodic_zero_mean_nonzero_k"


def _non_empty_string(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must be non-empty")


def _sorted_unique_strings(values: object, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an iterable of strings, not a string")
    try:
        normalized = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of strings") from exc

    for value in normalized:
        _non_empty_string(value, name)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class ReferenceDefinition:
    """Describe the reference state or history used for a field."""

    status: ReferenceStatus
    temporal_kind: ReferenceTemporalKind | None = None
    state_kind: str | None = None
    origin: str | None = None
    diagnostics_ref: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ReferenceStatus):
            raise TypeError("status must be a ReferenceStatus member")
        if self.temporal_kind is not None and not isinstance(
            self.temporal_kind, ReferenceTemporalKind
        ):
            raise TypeError(
                "temporal_kind must be a ReferenceTemporalKind member or None"
            )
        if self.state_kind is not None:
            _non_empty_string(self.state_kind, "state_kind")
        for name in ("origin", "diagnostics_ref"):
            value = getattr(self, name)
            if value is not None:
                _non_empty_string(value, name)

        if self.status in (ReferenceStatus.DEFINED, ReferenceStatus.EXPLICIT_ZERO):
            if self.temporal_kind is None:
                raise ValueError(f"{self.status.value} requires temporal_kind")
            if self.state_kind is None:
                raise ValueError(f"{self.status.value} requires state_kind")
        elif self.state_kind is not None and self.temporal_kind is None:
            raise ValueError("state_kind requires temporal_kind when status is unknown")


@dataclass(frozen=True, slots=True)
class ChangeDefinition:
    """Describe how a source field was varied and under which conditions."""

    form: ChangeForm
    reference: ReferenceDefinition
    changed_input: str
    order: int | None = None
    fixed_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.form, ChangeForm):
            raise TypeError("form must be a ChangeForm member")
        if not isinstance(self.reference, ReferenceDefinition):
            raise TypeError("reference must be a ReferenceDefinition")
        _non_empty_string(self.changed_input, "changed_input")
        if self.order is not None and (
            isinstance(self.order, bool) or not isinstance(self.order, int)
        ):
            raise TypeError("order must be an integer or None")
        if self.form is ChangeForm.LINEARIZED_RESPONSE:
            if self.order != 1:
                raise ValueError("linearized_response requires order=1")
        elif self.order is not None:
            raise ValueError("finite_state_difference requires order=None")

        object.__setattr__(
            self,
            "fixed_conditions",
            _sorted_unique_strings(self.fixed_conditions, "fixed_conditions"),
        )


@dataclass(frozen=True, slots=True)
class Contribution:
    """Describe whether a field contains total or selected contributions."""

    coverage: str
    components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.coverage, str):
            raise TypeError("coverage must be a string")
        if self.coverage not in ("total", "selected"):
            raise ValueError("coverage must be 'total' or 'selected'")
        components = _sorted_unique_strings(self.components, "components")
        if self.coverage == "selected" and not components:
            raise ValueError("selected coverage requires at least one component")
        object.__setattr__(self, "components", components)


@dataclass(frozen=True, slots=True)
class GravityFieldDefinition:
    """Describe the source kind, contribution, and reference for a field."""

    kind: FieldKind
    source_kind: str
    contribution: Contribution
    potential_reference: PotentialReference
    change: ChangeDefinition | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, FieldKind):
            raise TypeError("kind must be a FieldKind member")
        _non_empty_string(self.source_kind, "source_kind")
        if not isinstance(self.contribution, Contribution):
            raise TypeError("contribution must be a Contribution")
        if not isinstance(self.potential_reference, PotentialReference):
            raise TypeError("potential_reference must be a PotentialReference member")
        if self.change is not None and not isinstance(self.change, ChangeDefinition):
            raise TypeError("change must be a ChangeDefinition or None")

        if self.kind is FieldKind.SOURCE_FIELD and self.change is not None:
            raise ValueError("source_field must not have a change definition")
        if self.kind is FieldKind.FIELD_CHANGE and self.change is None:
            raise ValueError("field_change requires a change definition")

    @classmethod
    def source_field(
        cls,
        source_kind: str,
        contribution: Contribution,
        potential_reference: PotentialReference,
    ) -> GravityFieldDefinition:
        """Create a descriptor for a source field."""
        return cls(
            FieldKind.SOURCE_FIELD,
            source_kind,
            contribution,
            potential_reference,
        )

    @classmethod
    def field_change(
        cls,
        source_kind: str,
        contribution: Contribution,
        potential_reference: PotentialReference,
        change: ChangeDefinition,
    ) -> GravityFieldDefinition:
        """Create a descriptor for a field change."""
        return cls(
            FieldKind.FIELD_CHANGE,
            source_kind,
            contribution,
            potential_reference,
            change,
        )
