"""Dependency-light typed diagnostics and numerical convergence errors."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np

ObservableName = Literal["potential", "acceleration", "gradient"]
DiagnosticNorm = Literal["absolute", "euclidean", "frobenius"]
AccuracyPolicy = Literal["gen1_cavity_v1"]
DiagnosticTermination = Literal[
    "converged", "exact_zero", "resource_limit", "uncertainty_unresolved"
]

_VALID_OBSERVABLES = {"potential", "acceleration", "gradient"}
_VALID_NORMS = {"absolute", "euclidean", "frobenius"}
_EXPECTED_NORM = {
    "potential": "absolute",
    "acceleration": "euclidean",
    "gradient": "frobenius",
}
_VALID_TERMINATIONS = {
    "converged",
    "exact_zero",
    "resource_limit",
    "uncertainty_unresolved",
}


def _finite_nonnegative(value: object, name: str, *, optional: bool = False):
    if value is None and optional:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite nonnegative real number")
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer, not bool")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


@dataclass(frozen=True, slots=True)
class NumericalDiagnostic:
    """One target/observable accuracy and termination record in SI units."""

    method: str
    target_index: int
    target_id: str | None
    observable: ObservableName
    norm: DiagnosticNorm
    accuracy_policy: AccuracyPolicy
    scale_si: float
    value_norm_si: float | None
    requested_tolerance_si: float | None
    error_estimate_si: float | None
    rigorous_tail_bound_si: float | None
    evaluations: int
    refinements: int
    termination: DiagnosticTermination

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("method must be a nonempty string")
        object.__setattr__(
            self,
            "target_index",
            _nonnegative_integer(self.target_index, "target_index"),
        )
        if self.target_id is not None and (
            not isinstance(self.target_id, str) or not self.target_id
        ):
            raise ValueError("target_id must be a nonempty string or None")
        if (
            not isinstance(self.observable, str)
            or self.observable not in _VALID_OBSERVABLES
        ):
            raise ValueError("observable must be potential, acceleration, or gradient")
        if not isinstance(self.norm, str) or self.norm not in _VALID_NORMS:
            raise ValueError("norm must be absolute, euclidean, or frobenius")
        if self.norm != _EXPECTED_NORM[self.observable]:
            raise ValueError("norm is inconsistent with observable")
        if self.accuracy_policy != "gen1_cavity_v1":
            raise ValueError("accuracy_policy must be 'gen1_cavity_v1'")
        for name in (
            "scale_si",
            "value_norm_si",
            "requested_tolerance_si",
            "error_estimate_si",
            "rigorous_tail_bound_si",
        ):
            value = _finite_nonnegative(
                getattr(self, name), name, optional=name != "scale_si"
            )
            if value is not None:
                object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "evaluations",
            _nonnegative_integer(self.evaluations, "evaluations"),
        )
        object.__setattr__(
            self,
            "refinements",
            _nonnegative_integer(self.refinements, "refinements"),
        )
        if self.termination not in _VALID_TERMINATIONS:
            raise ValueError("termination is not a supported diagnostic state")
        if self.termination == "converged" and any(
            getattr(self, name) is None
            for name in ("value_norm_si", "requested_tolerance_si", "error_estimate_si")
        ):
            raise ValueError(
                "converged diagnostics require value norm, tolerance, and "
                "error estimate"
            )
        if self.termination == "exact_zero":
            required_zero = (
                self.scale_si,
                self.value_norm_si,
                self.requested_tolerance_si,
                self.error_estimate_si,
                self.evaluations,
                self.refinements,
            )
            if any(value != 0 for value in required_zero):
                raise ValueError(
                    "exact_zero diagnostics require zero values and counts"
                )
            if self.rigorous_tail_bound_si not in (None, 0, 0.0):
                raise ValueError("exact_zero tail bound must be zero or None")


class ConvergenceError(RuntimeError):
    """Numerical failure carrying immutable, nonempty diagnostic evidence."""

    __slots__ = ("_diagnostics",)

    def __init__(
        self,
        message: str,
        diagnostics: Iterable[NumericalDiagnostic],
    ) -> None:
        if not isinstance(message, str):
            raise TypeError("message must be a string")
        if isinstance(diagnostics, (str, bytes)):
            raise TypeError("diagnostics must be an iterable of NumericalDiagnostic")
        try:
            values = tuple(diagnostics)
        except TypeError as exc:
            raise TypeError(
                "diagnostics must be an iterable of NumericalDiagnostic"
            ) from exc
        if not values:
            raise ValueError("diagnostics must be nonempty")
        if any(not isinstance(item, NumericalDiagnostic) for item in values):
            raise TypeError("diagnostics must contain only NumericalDiagnostic values")
        if not any(
            item.termination in {"resource_limit", "uncertainty_unresolved"}
            for item in values
        ):
            raise ValueError(
                "ConvergenceError diagnostics must include a failed target/observable"
            )
        super().__init__(message)
        self._diagnostics = values

    @property
    def diagnostics(self) -> tuple[NumericalDiagnostic, ...]:
        return self._diagnostics


__all__ = ["NumericalDiagnostic", "ConvergenceError"]
