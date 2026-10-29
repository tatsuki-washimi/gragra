"""Immutable result records and exceptions for the optional reference API."""

from dataclasses import dataclass
from typing import Any


class ReferenceResourceError(RuntimeError):
    """Raised when a reference calculation exceeds its fixed work limit."""


class ReferenceConvergenceError(RuntimeError):
    """Raised when a reference calculation cannot produce a converged result."""


@dataclass(frozen=True)
class ReferenceEvaluation:
    """Metadata for one completed route evaluation."""

    route: str
    dps: int
    resolution: int
    n: int


@dataclass(frozen=True)
class ReferenceErrorComponents:
    """The three independently estimated error contributions."""

    e_p: Any
    e_r: Any
    e_a: Any


@dataclass(frozen=True)
class ReferenceScalar:
    """Scalar result and immutable evaluation metadata."""

    value: Any
    estimated_abs_error: Any
    requested_dps: int
    actual_dps: int
    actual_resolution: int
    actual_route: str
    evaluations: tuple[ReferenceEvaluation, ...]
    error_components: ReferenceErrorComponents
    converged: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluations", tuple(self.evaluations))
