"""Shared evaluation bridge for analytic model families."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import version
from platform import python_version
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import QTable
from gwexpy.frequencyseries import FrequencySeriesMatrix

from gragra._interactions import Newtonian
from gragra._time_matrix import make_time_matrix
from gragra.constants import G_SI
from gragra.exceptions import ConvergenceError, NumericalDiagnostic
from gragra.geometry._frame import Frame
from gragra.results import GravityResult
from gragra.results._context import EvaluationContext
from gragra.results._evaluation import EvaluationResult
from gragra.results._record import OBSERVABLES, EvaluationRecord
from gragra.results._representation import (
    StaticRepresentation,
    TimeSeriesRepresentation,
)
from gragra.results._spectral import PhasorRepresentation

NativeValues: TypeAlias = Mapping[str, object]

_UNITS = {
    "potential": u.m**2 / u.s**2,
    "acceleration": u.m / u.s**2,
    "gradient": u.s**-2,
}
_COMPONENTS = {
    "potential": ("potential",),
    "acceleration": ("x", "y", "z"),
    "gradient": ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"),
}


@dataclass(frozen=True, slots=True, init=False)
class NativeEvaluation:
    """G-free family output before common frame transforms and SI packaging."""

    values: NativeValues
    diagnostics: tuple[NumericalDiagnostic, ...]

    def __init__(
        self,
        values: Mapping[str, object],
        diagnostics: tuple[NumericalDiagnostic, ...] = (),
    ) -> None:
        if not isinstance(values, Mapping):
            raise TypeError("native values must be an observable mapping")
        snapshot = dict(values)
        if any(
            not isinstance(name, str) or name not in OBSERVABLES for name in snapshot
        ):
            raise ValueError("native values contain an unknown observable")
        if isinstance(diagnostics, (str, bytes)):
            raise TypeError("diagnostics must be an iterable of NumericalDiagnostic")
        try:
            diagnostic_snapshot = tuple(diagnostics)
        except TypeError as exc:
            raise TypeError(
                "diagnostics must be an iterable of NumericalDiagnostic"
            ) from exc
        if any(
            not isinstance(item, NumericalDiagnostic) for item in diagnostic_snapshot
        ):
            raise TypeError("diagnostics must contain NumericalDiagnostic values")
        object.__setattr__(self, "values", MappingProxyType(snapshot))
        object.__setattr__(self, "diagnostics", diagnostic_snapshot)


NativeCallback: TypeAlias = Callable[[np.ndarray, tuple[str, ...]], NativeEvaluation]


@dataclass(frozen=True, eq=False, slots=True, init=False)
class AnalyticEvaluation:
    """Shared static, phasor, and time-series evaluation adapter.

    Family callbacks return G-free arrays. This adapter owns validation of the
    shared context, component transforms, one application of ``G_SI``, and
    native result construction. Family-specific physics and target rejection
    stay in the model callback and optional validator.
    """

    context: EvaluationContext
    computation_frame: Frame
    _target_positions_bytes: bytes
    _target_count: int
    evaluate_native: NativeCallback
    validate_targets: Callable[[np.ndarray], object] | None

    def __init__(
        self,
        *,
        context: EvaluationContext,
        computation_frame: Frame,
        target_positions_m: object,
        evaluate_native: NativeCallback,
        validate_targets: Callable[[np.ndarray], object] | None = None,
    ) -> None:
        if not isinstance(context, EvaluationContext):
            raise TypeError("context must be an EvaluationContext")
        if not isinstance(getattr(context.model_spec, "interaction", None), Newtonian):
            raise TypeError("analytic evaluation requires a Newtonian interaction")
        if not isinstance(computation_frame, Frame):
            raise TypeError("computation_frame must be a Frame")
        if not callable(evaluate_native):
            raise TypeError("evaluate_native must be callable")
        if validate_targets is not None and not callable(validate_targets):
            raise TypeError("validate_targets must be callable or None")
        context.targets.frame._path_to(computation_frame)
        computation_frame._path_to(context.frame)

        raw = np.asarray(target_positions_m)
        if np.iscomplexobj(raw):
            raise ValueError("target_positions_m must contain only real coordinates")
        if np.issubdtype(raw.dtype, np.bool_):
            raise TypeError("target_positions_m must not contain bool values")
        try:
            positions = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("target_positions_m must contain real coordinates") from exc
        expected_shape = (len(context.targets), 3)
        if positions.shape != expected_shape:
            raise ValueError(
                "target_positions_m must have shape "
                f"{expected_shape}, got {positions.shape}"
            )
        if not np.isfinite(positions).all():
            raise ValueError("target_positions_m must contain only finite coordinates")

        object.__setattr__(self, "context", context)
        object.__setattr__(self, "computation_frame", computation_frame)
        object.__setattr__(self, "_target_positions_bytes", positions.copy().tobytes())
        object.__setattr__(self, "_target_count", len(positions))
        object.__setattr__(self, "evaluate_native", evaluate_native)
        object.__setattr__(self, "validate_targets", validate_targets)

    @property
    def target_positions_m(self) -> np.ndarray:
        """Read-only positions in the numerical kernel's computation frame."""
        return np.frombuffer(self._target_positions_bytes, dtype=np.float64).reshape(
            self._target_count, 3
        )

    @property
    def model_spec(self):
        return self.context.model_spec

    @property
    def parameters(self) -> QTable:
        value = getattr(self.model_spec, "parameters", None)
        return QTable() if value is None else value

    @property
    def formulas(self) -> pd.DataFrame:
        value = getattr(self.model_spec, "formulas", None)
        return pd.DataFrame() if value is None else value

    @property
    def references(self) -> pd.DataFrame:
        value = getattr(self.model_spec, "references", None)
        return pd.DataFrame() if value is None else value

    def validate(self) -> None:
        """Recheck shared interpretation and then run family target checks."""
        self.context.validate()
        if not isinstance(
            getattr(self.context.model_spec, "interaction", None), Newtonian
        ):
            raise TypeError("analytic evaluation requires a Newtonian interaction")
        positions = self.target_positions_m
        if (
            positions.shape != (len(self.context.targets), 3)
            or not np.isfinite(positions).all()
        ):
            raise ValueError("analytic evaluation target coordinates are invalid")
        self.context.targets.frame._path_to(self.computation_frame)
        self.computation_frame._path_to(self.context.frame)
        if self.validate_targets is not None:
            self.validate_targets(positions)

    def run(self) -> EvaluationResult:
        """Evaluate once and return all requested observables or raise."""
        self.validate()
        try:
            native = self.evaluate_native(
                self.target_positions_m, self.context.observables
            )
        except Exception as cause:
            raise self._evaluation_error(cause) from cause
        if not isinstance(native, NativeEvaluation):
            raise TypeError("evaluate_native must return NativeEvaluation")
        self._validate_diagnostics(native.diagnostics)
        if any(
            diagnostic.termination in {"resource_limit", "uncertainty_unresolved"}
            for diagnostic in native.diagnostics
        ):
            raise ValueError(
                "successful evaluations cannot contain failed numerical diagnostics"
            )
        arrays = self._validate_native_values(native.values)
        arrays = self._rotate_and_apply_g(arrays)
        values = self._package_values(arrays)
        return EvaluationResult(
            values=GravityResult(**values),
            evaluation=self,
            record=self._record(native.diagnostics),
        )

    def _evaluation_error(self, cause: Exception):
        from gragra.evaluation import EvaluationError

        target_index = 0 if self._target_count else None
        observable = self.context.observables[0]
        if isinstance(cause, ConvergenceError):
            try:
                self._validate_diagnostics(cause.diagnostics)
            except (TypeError, ValueError):
                pass
            else:
                failure = next(
                    diagnostic
                    for diagnostic in cause.diagnostics
                    if diagnostic.termination
                    in {"resource_limit", "uncertainty_unresolved"}
                )
                target_index = failure.target_index
                observable = failure.observable
        target_id = (
            None if target_index is None else self.context.targets.ids[target_index]
        )
        return EvaluationError(
            target_id=target_id,
            source_index=None,
            observables=self.context.observables,
            observable=observable,
            cause=cause,
        )

    def _validate_diagnostics(
        self, diagnostics: tuple[NumericalDiagnostic, ...]
    ) -> None:
        for diagnostic in diagnostics:
            index = diagnostic.target_index
            if index >= len(self.context.targets):
                raise ValueError("diagnostic target_index is outside context targets")
            expected_id = self.context.targets.ids[index]
            if diagnostic.target_id != expected_id:
                raise ValueError(
                    "diagnostic target_id must match context target ID at target_index"
                )

    def _validate_native_values(
        self, values: Mapping[str, object]
    ) -> dict[str, np.ndarray]:
        names = self.context.observables
        if set(values) != set(names):
            raise ValueError("native values must match requested observables exactly")
        representation = self.context.representation
        target_count = len(self.context.targets)
        if isinstance(representation, StaticRepresentation):
            axis_count = None
        elif isinstance(representation, PhasorRepresentation):
            axis_count = len(self.context.frequencies_hz or ())
        elif isinstance(representation, TimeSeriesRepresentation):
            sampling = self.context.time_sampling
            if sampling is None:
                raise ValueError("time_series context requires TimeSampling")
            axis_count = sampling.count
        else:
            raise TypeError("analytic representation is unsupported")

        arrays: dict[str, np.ndarray] = {}
        for name in names:
            raw = np.asarray(values[name])
            if not np.issubdtype(raw.dtype, np.number):
                raise TypeError(f"native {name} values must be numeric")
            if isinstance(representation, StaticRepresentation):
                expected_shape = {
                    "potential": (target_count,),
                    "acceleration": (target_count, 3),
                    "gradient": (target_count, 3, 3),
                }[name]
                complex_required = False
            else:
                expected_shape = (
                    target_count,
                    len(_COMPONENTS[name]),
                    axis_count,
                )
                complex_required = isinstance(representation, PhasorRepresentation)
            if raw.shape != expected_shape:
                raise ValueError(
                    f"native {name} values must have shape {expected_shape}, "
                    f"got {raw.shape}"
                )
            if complex_required and not np.iscomplexobj(raw):
                raise ValueError(f"native phasor {name} values must be complex")
            if not complex_required and np.iscomplexobj(raw):
                raise ValueError(f"native {name} values must be real")
            dtype = np.complex128 if complex_required else np.float64
            try:
                array = np.asarray(raw, dtype=dtype)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"native {name} values cannot be represented") from exc
            if not np.isfinite(array).all():
                raise ValueError(f"native {name} values must be finite")
            arrays[name] = array
        return arrays

    def _rotate_and_apply_g(
        self, arrays: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        rotation = self.computation_frame.rotation_to(self.context.frame)
        representation = self.context.representation
        result: dict[str, np.ndarray] = {}
        for name, values in arrays.items():
            with np.errstate(over="ignore", invalid="ignore"):
                if name == "potential":
                    output = G_SI * values
                elif isinstance(representation, StaticRepresentation):
                    if name == "acceleration":
                        output = G_SI * np.einsum("ij,nj->ni", rotation, values)
                    else:
                        output = G_SI * np.einsum(
                            "ia,nab,jb->nij", rotation, values, rotation
                        )
                elif name == "acceleration":
                    output = G_SI * np.einsum("ij,njt->nit", rotation, values)
                else:
                    count, _, axis_count = values.shape
                    tensors = values.reshape(count, 3, 3, axis_count)
                    output = G_SI * np.einsum(
                        "ia,nabt,jb->nijt", rotation, tensors, rotation
                    ).reshape(count, 9, axis_count)
            if not np.isfinite(output).all():
                raise ValueError(f"SI {name} values became non-finite")
            result[name] = output
        return result

    def _package_values(self, arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
        representation = self.context.representation
        if isinstance(representation, StaticRepresentation):
            return {
                name: arrays[name] * _UNITS[name] for name in self.context.observables
            }
        if isinstance(representation, PhasorRepresentation):
            matrices = {}
            frequencies = (
                np.asarray(self.context.frequencies_hz, dtype=np.float64) * u.Hz
            )
            target_ids = self.context.targets.ids
            for name in self.context.observables:
                raw = arrays[name]
                components = _COMPONENTS[name]
                matrices[name] = FrequencySeriesMatrix(
                    raw.copy(),
                    frequencies=frequencies,
                    units=np.full(
                        (len(target_ids), len(components)), _UNITS[name], dtype=object
                    ),
                    rows=list(target_ids),
                    cols=list(components),
                    name=f"Gen1 {name}",
                )
            return matrices
        if isinstance(representation, TimeSeriesRepresentation):
            sampling = self.context.time_sampling
            if sampling is None:
                raise ValueError("time_series context requires TimeSampling")
            return {
                name: make_time_matrix(
                    arrays[name],
                    sampling=sampling,
                    unit=_UNITS[name],
                    rows=self.context.targets.ids,
                    cols=_COMPONENTS[name],
                    name=f"Gen1 {name}",
                )
                for name in self.context.observables
            }
        raise TypeError("analytic representation is unsupported")

    @staticmethod
    def _record(diagnostics: tuple[NumericalDiagnostic, ...]) -> EvaluationRecord:
        return EvaluationRecord(
            backend="gen1_analytic_native",
            versions={
                "gragra": version("gragra"),
                "python": python_version(),
                "numpy": np.__version__,
                "scipy": version("scipy"),
                "astropy": version("astropy"),
                "gwexpy": version("gwexpy"),
            },
            gravitational_constant_si=G_SI,
            g_applications=1,
            numerical_diagnostics=diagnostics,
        )


__all__ = ["AnalyticEvaluation", "NativeEvaluation"]
