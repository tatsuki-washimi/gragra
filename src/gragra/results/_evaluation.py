"""Result views interpreted through a shared model-independent context."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from astropy import units as u
from astropy.table import QTable
from gwexpy.frequencyseries import FrequencySeriesMatrix
from gwexpy.timeseries import TimeSeriesMatrix

from gragra._spectral_from_timeseries import (
    parse_spectrum_kind,
    transform_time_payloads,
)
from gragra._time_matrix import validate_time_axis
from gragra.constants import G_SI
from gragra.results import GravityResult
from gragra.results._binding import ObservableBinding
from gragra.results._context import EvaluationContext
from gragra.results._coords import normalize_gps, one_sided_frequencies_hz
from gragra.results._field_definition import GravityFieldDefinition
from gragra.results._protocols import FieldEvaluation
from gragra.results._record import OBSERVABLES, EvaluationRecord
from gragra.results._representation import (
    StaticRepresentation,
    TimeSeriesRepresentation,
)
from gragra.results._spectral import (
    AmplitudeSpectrumRepresentation,
    FiniteRecordInterpretation,
    FourierTransformRepresentation,
    PhasorRepresentation,
    Sidedness,
)

_EXPECTED_UNITS = {
    "potential": u.m**2 / u.s**2,
    "acceleration": u.m / u.s**2,
    "gradient": 1 / u.s**2,
}
_FOURIER_UNITS = {
    "potential": u.m**2 / u.s,
    "acceleration": u.m / u.s,
    "gradient": 1 / u.s,
}
_EXPECTED_AXES = {
    "potential": (),
    "acceleration": (("x", "y", "z"),),
    "gradient": (("x", "y", "z"), ("x", "y", "z")),
}
_MATRIX_COMPONENTS = {
    "potential": ("potential",),
    "acceleration": ("x", "y", "z"),
    "gradient": ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"),
}


def _validate_quantity(
    observable: str, quantity: object, shape: tuple[int, ...]
) -> None:
    if not isinstance(quantity, u.Quantity):
        raise TypeError(f"{observable} must be an astropy Quantity")
    raw = np.asarray(quantity.value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{observable} must contain only real values")
    if raw.shape != shape:
        raise ValueError(f"{observable} must have shape {shape}, got {raw.shape}")
    try:
        si_values = np.asarray(quantity.to_value(_EXPECTED_UNITS[observable]))
    except (TypeError, ValueError, u.UnitConversionError) as exc:
        raise ValueError(
            f"{observable} must have units equivalent to {_EXPECTED_UNITS[observable]}"
        ) from exc
    if not np.isfinite(raw).all() or not np.isfinite(si_values).all():
        raise ValueError(f"{observable} must contain only finite values")


def _validate_spectral_matrix(
    observable: str,
    matrix: object,
    binding: ObservableBinding,
    *,
    expected_units: dict[str, u.UnitBase],
) -> None:
    if not isinstance(matrix, FrequencySeriesMatrix):
        raise TypeError(
            f"{observable} spectral payload must be a FrequencySeriesMatrix"
        )
    expected_shape = (
        len(binding.target_ids),
        len(_MATRIX_COMPONENTS[observable]),
        len(binding.frequencies_hz or ()),
    )
    raw = np.asarray(matrix)
    if raw.shape != expected_shape:
        raise ValueError(
            f"{observable} matrix must have shape {expected_shape}, got {raw.shape}"
        )
    if not np.iscomplexobj(raw):
        raise ValueError(f"{observable} spectral values must have a complex dtype")
    if not np.isfinite(raw).all():
        raise ValueError(f"{observable} spectral values must be finite")
    if tuple(matrix.rows.keys()) != binding.target_ids:
        raise ValueError(f"{observable} matrix row labels must match target IDs")
    if tuple(matrix.cols.keys()) != _MATRIX_COMPONENTS[observable]:
        raise ValueError(f"{observable} matrix component labels are inconsistent")
    try:
        frequencies_hz = np.asarray(matrix.frequencies.to_value(u.Hz), dtype=np.float64)
    except (AttributeError, TypeError, ValueError, u.UnitConversionError) as exc:
        raise ValueError(f"{observable} matrix frequencies must convert to Hz") from exc
    expected_frequencies = np.asarray(binding.frequencies_hz, dtype=np.float64)
    if (
        frequencies_hz.shape != expected_frequencies.shape
        or not np.isfinite(frequencies_hz).all()
        or not np.array_equal(frequencies_hz, expected_frequencies)
    ):
        raise ValueError(
            f"{observable} matrix frequency axis must exactly match binding "
            "frequencies in Hz"
        )
    expected_unit = expected_units[observable]
    for row in range(expected_shape[0]):
        for column in range(expected_shape[1]):
            unit = matrix.meta[row, column].unit
            try:
                scale = u.Unit(unit).to(expected_unit)
                si_values = raw[row, column] * scale
            except (TypeError, ValueError, u.UnitConversionError) as exc:
                raise ValueError(
                    f"{observable} cell ({row}, {column}) must have units "
                    f"equivalent to {expected_unit}"
                ) from exc
            if not np.isfinite(si_values).all():
                raise ValueError(
                    f"{observable} cell ({row}, {column}) is non-finite in SI units"
                )


def _validate_phasor_matrix(
    observable: str, matrix: object, binding: ObservableBinding
) -> None:
    if not isinstance(matrix, FrequencySeriesMatrix):
        raise TypeError(f"{observable} phasor payload must be a FrequencySeriesMatrix")
    expected_shape = (
        len(binding.target_ids),
        len(_MATRIX_COMPONENTS[observable]),
        len(binding.frequencies_hz),
    )
    raw = np.asarray(matrix)
    if raw.shape != expected_shape:
        raise ValueError(
            f"{observable} matrix must have shape {expected_shape}, got {raw.shape}"
        )
    if not np.iscomplexobj(raw):
        raise ValueError(f"{observable} phasor values must have a complex dtype")
    if not np.isfinite(raw).all():
        raise ValueError(f"{observable} phasor values must be finite")
    if tuple(matrix.rows.keys()) != binding.target_ids:
        raise ValueError(f"{observable} matrix row labels must match target IDs")
    if tuple(matrix.cols.keys()) != _MATRIX_COMPONENTS[observable]:
        raise ValueError(f"{observable} matrix component labels are inconsistent")

    try:
        frequencies_hz = np.asarray(matrix.frequencies.to_value(u.Hz), dtype=np.float64)
    except (AttributeError, TypeError, ValueError, u.UnitConversionError) as exc:
        raise ValueError(f"{observable} matrix frequencies must convert to Hz") from exc
    expected_frequencies = np.asarray(binding.frequencies_hz, dtype=np.float64)
    if (
        frequencies_hz.shape != expected_frequencies.shape
        or not np.isfinite(frequencies_hz).all()
        or not np.array_equal(frequencies_hz, expected_frequencies)
    ):
        raise ValueError(
            f"{observable} matrix frequency axis must exactly match binding "
            "frequencies in Hz"
        )

    expected_unit = _EXPECTED_UNITS[observable]
    for row in range(expected_shape[0]):
        for column in range(expected_shape[1]):
            unit = matrix.meta[row, column].unit
            try:
                scale = u.Unit(unit).to(expected_unit)
                si_values = raw[row, column] * scale
            except (TypeError, ValueError, u.UnitConversionError) as exc:
                raise ValueError(
                    f"{observable} cell ({row}, {column}) must have units "
                    f"equivalent to {expected_unit}"
                ) from exc
            if not np.isfinite(si_values).all():
                raise ValueError(
                    f"{observable} cell ({row}, {column}) is non-finite in SI units"
                )


def _validate_time_series_matrix(
    observable: str, matrix: object, binding: ObservableBinding
) -> None:
    if not isinstance(matrix, TimeSeriesMatrix):
        raise TypeError(f"{observable} time-series payload must be a TimeSeriesMatrix")
    sampling = binding.time_sampling
    if sampling is None:
        raise ValueError(f"{observable} time-series binding has no TimeSampling")
    component_count = len(_MATRIX_COMPONENTS[observable])
    expected_shape = (
        len(binding.target_ids),
        component_count,
        sampling.count,
    )
    raw = np.asarray(matrix)
    if raw.shape != expected_shape:
        raise ValueError(
            f"{observable} TimeSeriesMatrix must have shape {expected_shape}, "
            f"got {raw.shape}"
        )
    if np.iscomplexobj(raw):
        raise ValueError(f"{observable} time-series values must be real")
    if not np.isfinite(raw).all():
        raise ValueError(f"{observable} time-series values must be finite")
    if tuple(matrix.rows.keys()) != binding.target_ids:
        raise ValueError(f"{observable} matrix row labels must match target IDs")
    if tuple(matrix.cols.keys()) != _MATRIX_COMPONENTS[observable]:
        raise ValueError(f"{observable} matrix component labels are inconsistent")
    validate_time_axis(matrix, sampling, observable=observable)

    expected_unit = _EXPECTED_UNITS[observable]
    units = getattr(matrix, "units", None)
    if units is None or np.shape(units) != expected_shape[:2]:
        raise ValueError(
            f"{observable} matrix units must match its row/component cells"
        )
    for row in range(expected_shape[0]):
        for column in range(expected_shape[1]):
            try:
                scale = u.Unit(units[row, column]).to(expected_unit)
                si_values = raw[row, column] * scale
            except (TypeError, ValueError, u.UnitConversionError) as exc:
                raise ValueError(
                    f"{observable} cell ({row}, {column}) must have units "
                    f"equivalent to {expected_unit}"
                ) from exc
            if not np.isfinite(si_values).all():
                raise ValueError(
                    f"{observable} cell ({row}, {column}) is non-finite in SI units"
                )


@dataclass(frozen=True, eq=False, init=False)
class EvaluationResult:
    """A validated view of payloads from one stable field evaluation.

    Payloads are retained by identity and rechecked on result operations. The
    constructor reads ``evaluation.context`` once and never runs the producer.
    """

    _values: GravityResult
    _bindings: Mapping[str, ObservableBinding]
    _evaluation: FieldEvaluation
    _context: EvaluationContext
    _record: EvaluationRecord
    _observable_order: tuple[str, ...]

    def __init__(
        self,
        values: GravityResult,
        *,
        evaluation: FieldEvaluation,
        record: EvaluationRecord,
        context: EvaluationContext | None = None,
    ) -> None:
        if not isinstance(values, GravityResult):
            raise TypeError("values must be a GravityResult")
        if not isinstance(record, EvaluationRecord):
            raise TypeError("record must be an EvaluationRecord")
        if context is None:
            context = getattr(evaluation, "context", None)
        if not isinstance(context, EvaluationContext):
            raise TypeError("evaluation.context must be an EvaluationContext")
        names = tuple(
            name for name in context.observables if getattr(values, name) is not None
        )
        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "_evaluation", evaluation)
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_record", record)
        object.__setattr__(self, "_observable_order", names)
        object.__setattr__(
            self,
            "_bindings",
            MappingProxyType({name: context.binding(name) for name in names}),
        )
        self.validate()

    def validate(self) -> None:
        """Recheck the mutable payload against its fixed interpretation."""
        context = self._context
        context.validate()
        if self._record.g_applications != 1:
            raise ValueError("evaluation record must apply G exactly once")
        if self._record.gravitational_constant_si != G_SI:
            raise ValueError(
                "evaluation record gravitational_constant_si must match configured G_SI"
            )
        if context.g_convention != "SI_G_applied_once":
            raise ValueError("context G convention is inconsistent")
        if not self._observable_order:
            raise ValueError("result must contain at least one observable")
        present = {
            name for name in OBSERVABLES if getattr(self._values, name) is not None
        }
        provided = set(self._observable_order)
        if present != provided:
            raise ValueError("GravityResult payloads must match result observables")
        if not provided <= set(context.observables):
            raise ValueError(
                "result observables must be a subset of context observables"
            )
        if tuple(self._bindings) != self._observable_order:
            raise ValueError("bindings must follow result observable order")
        for name in self._observable_order:
            if self._bindings[name] is not context.binding(name):
                raise ValueError(f"{name} binding must belong to result context")
        if isinstance(context.representation, StaticRepresentation):
            self._validate_static()
        elif isinstance(context.representation, PhasorRepresentation):
            self._validate_phasor()
        elif isinstance(context.representation, TimeSeriesRepresentation):
            self._validate_time_series()
        elif isinstance(context.representation, FourierTransformRepresentation):
            self._validate_fourier()
        elif isinstance(context.representation, AmplitudeSpectrumRepresentation):
            self._validate_amplitude()
        else:
            raise TypeError("result representation is unsupported")

    def _validate_static(self) -> None:
        count = len(self._context.targets)
        for name in self._observable_order:
            shape = {
                "potential": (count,),
                "acceleration": (count, 3),
                "gradient": (count, 3, 3),
            }[name]
            _validate_quantity(name, getattr(self._values, name), shape)

    def _validate_phasor(self) -> None:
        for name in self._observable_order:
            _validate_phasor_matrix(
                name, getattr(self._values, name), self._bindings[name]
            )

    def _validate_time_series(self) -> None:
        for name in self._observable_order:
            _validate_time_series_matrix(
                name, getattr(self._values, name), self._bindings[name]
            )

    def _validate_fourier(self) -> None:
        for name in self._observable_order:
            _validate_spectral_matrix(
                name,
                getattr(self._values, name),
                self._bindings[name],
                expected_units=_FOURIER_UNITS,
            )

    def _validate_amplitude(self) -> None:
        for name in self._observable_order:
            _validate_spectral_matrix(
                name,
                getattr(self._values, name),
                self._bindings[name],
                expected_units=_EXPECTED_UNITS,
            )

    @property
    def values(self) -> GravityResult:
        self.validate()
        return self._values

    @property
    def bindings(self) -> Mapping[str, ObservableBinding]:
        self.validate()
        return self._bindings

    @property
    def evaluation(self) -> FieldEvaluation:
        self.validate()
        return self._evaluation

    @property
    def context(self) -> EvaluationContext:
        self.validate()
        return self._context

    @property
    def field_definition(self) -> GravityFieldDefinition:
        self.validate()
        return self._context.field_definition

    @property
    def record(self) -> EvaluationRecord:
        self.validate()
        return self._record

    @property
    def observables(self) -> tuple[str, ...]:
        self.validate()
        return self._observable_order

    @property
    def potential(self) -> u.Quantity | FrequencySeriesMatrix | TimeSeriesMatrix | None:
        self.validate()
        return self._values.potential

    @property
    def acceleration(
        self,
    ) -> u.Quantity | FrequencySeriesMatrix | TimeSeriesMatrix | None:
        self.validate()
        return self._values.acceleration

    @property
    def gradient(self) -> u.Quantity | FrequencySeriesMatrix | TimeSeriesMatrix | None:
        self.validate()
        return self._values.gradient

    @property
    def parameters(self) -> QTable:
        self.validate()
        return self._evaluation.parameters

    @property
    def formulas(self):
        self.validate()
        return self._evaluation.formulas

    @property
    def references(self):
        self.validate()
        return self._evaluation.references

    @property
    def backend(self) -> str:
        return self.record.backend

    @property
    def versions(self) -> Mapping[str, str]:
        return self.record.versions

    @property
    def gravitational_constant_si(self) -> float:
        return self.record.gravitational_constant_si

    @property
    def g_applications(self) -> int:
        return self.record.g_applications

    def select(self, *observables: str) -> EvaluationResult:
        """Return a narrower view sharing evaluation, record, context and payloads."""
        self.validate()
        if not observables:
            raise ValueError("select requires at least one observable")
        if any(not isinstance(name, str) for name in observables):
            raise TypeError("selected observable names must be strings")
        if len(set(observables)) != len(observables):
            raise ValueError("selected observables must be unique")
        unknown = set(observables) - set(OBSERVABLES)
        if unknown:
            raise ValueError(f"unknown observables: {sorted(unknown)!r}")
        unavailable = set(observables) - set(self._observable_order)
        if unavailable:
            raise ValueError(
                f"observables are not present in this result: {sorted(unavailable)!r}"
            )
        selected = set(observables)
        values = GravityResult(
            name=self._values.name,
            potential=self._values.potential if "potential" in selected else None,
            acceleration=(
                self._values.acceleration if "acceleration" in selected else None
            ),
            gradient=self._values.gradient if "gradient" in selected else None,
        )
        result = object.__new__(EvaluationResult)
        object.__setattr__(result, "_values", values)
        object.__setattr__(result, "_evaluation", self._evaluation)
        object.__setattr__(result, "_context", self._context)
        object.__setattr__(result, "_record", self._record)
        object.__setattr__(result, "_observable_order", tuple(observables))
        object.__setattr__(
            result,
            "_bindings",
            MappingProxyType({name: self._bindings[name] for name in observables}),
        )
        result.validate()
        return result

    def _spectral_from_time_series(
        self,
        *,
        mode: str,
        tref: object,
        representation: object,
    ) -> EvaluationResult:
        self.validate()
        context = self._context
        if not isinstance(context.representation, TimeSeriesRepresentation):
            raise ValueError(".fft() and .spectrum() require a time-series result")
        sampling = context.time_sampling
        if sampling is None:
            raise ValueError("time-series result has no TimeSampling")
        payloads = {
            name: getattr(self._values, name)
            for name in self._observable_order
            if getattr(self._values, name) is not None
        }
        transformed = transform_time_payloads(
            payloads,
            sampling=sampling,
            observables=self._observable_order,
            tref=tref,
            mode=mode,
        )
        freqs = one_sided_frequencies_hz(sampling.count, sampling.dt_seconds) * u.Hz
        new_context = EvaluationContext(
            model_spec=context.model_spec,
            targets=context.targets,
            frame=context.frame,
            observables=context.observables,
            representation=representation,
            frequencies=freqs,
            reference_time=tref,
            g_convention=context.g_convention,
        )
        values = GravityResult(
            name=self._values.name,
            **{name: transformed[name] for name in self._observable_order},
        )
        return EvaluationResult(
            values=values,
            evaluation=self._evaluation,
            record=self._record,
            context=new_context,
        )

    def fft(self, *, tref: object) -> EvaluationResult:
        """Discrete Fourier transform (C3 · ONE_SIDED · unit Q·s)."""
        normalize_gps(tref)
        return self._spectral_from_time_series(
            mode="fourier",
            tref=tref,
            representation=FourierTransformRepresentation(
                sidedness=Sidedness.ONE_SIDED
            ),
        )

    def spectrum(
        self, kind: str | FiniteRecordInterpretation, *, tref: object
    ) -> EvaluationResult:
        """Complex amplitude spectrum (C3 · periodic or transient · unit Q)."""
        interpretation = parse_spectrum_kind(kind)
        normalize_gps(tref)
        return self._spectral_from_time_series(
            mode="amplitude",
            tref=tref,
            representation=AmplitudeSpectrumRepresentation(
                record_interpretation=interpretation
            ),
        )
