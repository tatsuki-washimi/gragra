"""Experimental result containers for gravitational observables."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gragra.results._field_definition import (
    ChangeDefinition,
    ChangeForm,
    Contribution,
    FieldKind,
    GravityFieldDefinition,
    PotentialReference,
    ReferenceDefinition,
    ReferenceStatus,
    ReferenceTemporalKind,
)
from gragra.results._spectral import (
    AmplitudeSpectrumRepresentation,
    ASDRepresentation,
    CSDRepresentation,
    FiniteRecordInterpretation,
    FourierTransformRepresentation,
    PhasorRepresentation,
    PSDRepresentation,
    Sidedness,
    SpectralRepresentation,
)


@dataclass(frozen=True, kw_only=True, eq=False)
class GravityResult:
    """Group optional observable payloads produced by one evaluation.

    Only the field bindings are frozen. Payload objects are retained by
    identity and may remain mutable; this class does not create a snapshot.
    At least one observable must be provided. ``name`` is a container label
    and is not copied into payload metadata.
    """

    potential: object | None = None
    acceleration: object | None = None
    gradient: object | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("name must be a string or None")
        if (
            self.potential is None
            and self.acceleration is None
            and self.gradient is None
        ):
            raise ValueError("at least one observable must be provided")


__all__ = [
    "GravityResult",
    "FieldKind",
    "ReferenceStatus",
    "ReferenceTemporalKind",
    "ChangeForm",
    "PotentialReference",
    "ReferenceDefinition",
    "ChangeDefinition",
    "Contribution",
    "GravityFieldDefinition",
    "Sidedness",
    "FiniteRecordInterpretation",
    "FourierTransformRepresentation",
    "AmplitudeSpectrumRepresentation",
    "PhasorRepresentation",
    "PSDRepresentation",
    "ASDRepresentation",
    "CSDRepresentation",
    "SpectralRepresentation",
    "EvaluationResult",
    "compare",
    "GravityResponse",
    "ResponseContract",
    "EvaluationContext",
    "StaticRepresentation",
    "TimeSeriesRepresentation",
    "TimeSampling",
    "ModelSpec",
    "FieldEvaluation",
    "EvaluationRecord",
    "NumericalDiagnostic",
]

if TYPE_CHECKING:
    from gragra.exceptions import NumericalDiagnostic
    from gragra.results._comparison import compare
    from gragra.results._context import EvaluationContext
    from gragra.results._evaluation import EvaluationResult
    from gragra.results._protocols import FieldEvaluation, ModelSpec
    from gragra.results._record import EvaluationRecord
    from gragra.results._representation import (
        StaticRepresentation,
        TimeSeriesRepresentation,
    )
    from gragra.results._response import GravityResponse, ResponseContract
    from gragra.results._time import TimeSampling


def __getattr__(name: str):
    """Load evaluation result and comparison APIs on explicit access."""
    if name == "EvaluationResult":
        from gragra.results._evaluation import EvaluationResult

        return EvaluationResult
    if name == "compare":
        from gragra.results._comparison import compare

        return compare
    if name == "EvaluationContext":
        from gragra.results._context import EvaluationContext

        return EvaluationContext
    if name == "StaticRepresentation":
        from gragra.results._representation import StaticRepresentation

        return StaticRepresentation
    if name == "TimeSeriesRepresentation":
        from gragra.results._representation import TimeSeriesRepresentation

        return TimeSeriesRepresentation
    if name == "TimeSampling":
        from gragra.results._time import TimeSampling

        return TimeSampling
    if name in {"ModelSpec", "FieldEvaluation"}:
        from gragra.results._protocols import FieldEvaluation, ModelSpec

        return {"ModelSpec": ModelSpec, "FieldEvaluation": FieldEvaluation}[name]
    if name == "EvaluationRecord":
        from gragra.results._record import EvaluationRecord

        return EvaluationRecord
    if name == "NumericalDiagnostic":
        from gragra.exceptions import NumericalDiagnostic

        return NumericalDiagnostic
    if name in {"GravityResponse", "ResponseContract"}:
        from gragra.results._response import GravityResponse, ResponseContract

        return {
            "GravityResponse": GravityResponse,
            "ResponseContract": ResponseContract,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
