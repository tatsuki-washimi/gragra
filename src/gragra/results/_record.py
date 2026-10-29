"""Observable names and execution facts shared by result producers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from gragra.constants import G_SI
from gragra.exceptions import NumericalDiagnostic

OBSERVABLES = ("potential", "acceleration", "gradient")


@dataclass(frozen=True, init=False)
class EvaluationRecord:
    """Execution facts only; physical assumptions stay in the model context."""

    backend: str
    versions: Mapping[str, str]
    gravitational_constant_si: float
    g_applications: int
    numerical_diagnostics: tuple[NumericalDiagnostic, ...]

    def __init__(
        self,
        *,
        backend: str,
        versions: Mapping[str, str],
        gravitational_constant_si: float = G_SI,
        g_applications: int = 1,
        numerical_diagnostics: tuple[NumericalDiagnostic, ...] = (),
    ) -> None:
        if not isinstance(backend, str) or not backend:
            raise ValueError("backend must be a nonempty string")
        if not isinstance(versions, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in versions.items()
        ):
            raise TypeError("versions must map strings to strings")
        if not np.isfinite(gravitational_constant_si) or gravitational_constant_si <= 0:
            raise ValueError("gravitational_constant_si must be positive and finite")
        if g_applications != 1:
            raise ValueError("G must be applied exactly once")
        if isinstance(numerical_diagnostics, (str, bytes)):
            raise TypeError("numerical_diagnostics must be an iterable of diagnostics")
        try:
            diagnostics = tuple(numerical_diagnostics)
        except TypeError as exc:
            raise TypeError(
                "numerical_diagnostics must be an iterable of diagnostics"
            ) from exc
        if any(not isinstance(item, NumericalDiagnostic) for item in diagnostics):
            raise TypeError(
                "numerical_diagnostics must contain only NumericalDiagnostic values"
            )
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "versions", MappingProxyType(dict(versions)))
        object.__setattr__(
            self, "gravitational_constant_si", float(gravitational_constant_si)
        )
        object.__setattr__(self, "g_applications", 1)
        object.__setattr__(self, "numerical_diagnostics", diagnostics)
