"""Contracts and owned transfer coefficients for gravity responses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

import numpy as np
from astropy import units as u

from gragra.constants import G_SI
from gragra.geometry._frame import Frame
from gragra.results._context import EvaluationContext
from gragra.results._field_definition import (
    ChangeForm,
    FieldKind,
    GravityFieldDefinition,
)
from gragra.results._record import OBSERVABLES, EvaluationRecord


class _MappingKind(StrEnum):
    EXACT_LINEAR = "exact_linear"
    LOCAL_LINEARIZATION = "local_linearization"


class _ResponseAction(StrEnum):
    INSTANTANEOUS = "instantaneous"
    PER_FREQUENCY = "per_frequency"
    TIME_DEPENDENT = "time_dependent"


class _CoefficientStorage(StrEnum):
    PRECOMPUTED = "precomputed"
    ON_DEMAND = "on_demand"


class _LinearizationAssessment(StrEnum):
    NOT_ASSESSED = "not_assessed"
    ASSESSED = "assessed"


def _descriptor(name: str, value: object, choices: type[StrEnum]) -> StrEnum:
    try:
        return choices(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be one of {[member.value for member in choices]}"
        ) from exc


def _nonempty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _strings(name: str, value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty sequence of strings")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonempty sequence of strings") from exc
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must be a nonempty sequence of strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _scalar_quantity(name: str, value: object, unit: u.UnitBase) -> float:
    if not isinstance(value, u.Quantity):
        raise ValueError(f"{name} must be an astropy Quantity")
    try:
        raw = np.asarray(value.to_value(unit))
    except (TypeError, ValueError, u.UnitConversionError) as exc:
        raise ValueError(f"{name} must have units equivalent to {unit}") from exc
    if raw.shape != () or np.iscomplexobj(raw) or not np.isfinite(raw):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(raw)


@dataclass(frozen=True, slots=True, init=False)
class ResponseContract:
    """Describe the physical input, mapping, gravity law, and applicability.

    The Rayleigh contract identifies vertical surface displacement in
    ``input_frame`` at ``input_location_m`` and records its normalization and
    basis amplitude. ``mapping_kind`` describes the physical map independently
    of the output definition's C2 change form. A concrete response currently
    supports ``per_frequency`` action, ``precomputed`` storage, Newton gravity
    with no law parameters, a varied source, fixed targets, one input channel
    and component, and ``not_assessed`` applicability. Both ``exact_linear``
    and ``local_linearization`` are valid mapping kinds. Future action, storage,
    law, or assessment vocabulary does not imply execution support.

    Mutable constructor inputs are copied; the descriptor is immutable.
    """

    input_name: str
    input_quantity: str
    input_unit: u.UnitBase
    input_channel_ids: tuple[str, ...]
    input_components: tuple[str, ...]
    input_frame: Frame
    _input_location_tuple_m: tuple[float, float, float]
    input_distribution: str
    normalization: str
    _basis_amplitude_si: float
    mapping_kind: _MappingKind
    action: _ResponseAction
    storage: _CoefficientStorage
    varied: str
    target_position: str
    law_name: str
    law_parameters: Mapping[str, object]
    gravitational_constant_si: float
    linearization_assessment: _LinearizationAssessment

    def __init__(
        self,
        *,
        input_name: str,
        input_quantity: str,
        input_unit: u.UnitBase,
        input_channel_ids: object,
        input_components: object,
        input_frame: Frame,
        input_location_m: object,
        input_distribution: str,
        normalization: str,
        basis_amplitude: u.Quantity,
        mapping_kind: str,
        action: str,
        storage: str,
        varied: str,
        target_position: str,
        law_name: str,
        law_parameters: Mapping[str, object],
        gravitational_constant_si: float,
        linearization_assessment: str,
    ) -> None:
        name = _nonempty("input_name", input_name)
        if name in {"representation", "reference_time"}:
            raise ValueError("input_name is reserved")
        object.__setattr__(self, "input_name", name)
        object.__setattr__(
            self, "input_quantity", _nonempty("input_quantity", input_quantity)
        )
        try:
            unit = u.Unit(input_unit)
        except (TypeError, ValueError) as exc:
            raise ValueError("input_unit must be a valid unit") from exc
        object.__setattr__(self, "input_unit", unit)
        object.__setattr__(
            self, "input_channel_ids", _strings("input_channel_ids", input_channel_ids)
        )
        object.__setattr__(
            self, "input_components", _strings("input_components", input_components)
        )
        if not isinstance(input_frame, Frame):
            raise ValueError("input_frame must be a Frame")
        object.__setattr__(self, "input_frame", input_frame)
        if isinstance(input_location_m, u.Quantity):
            try:
                raw_location = np.asarray(input_location_m.to_value(u.m))
            except (TypeError, ValueError, u.UnitConversionError) as exc:
                raise ValueError(
                    "input_location_m must be a position in metres"
                ) from exc
        else:
            raw_location = np.asarray(input_location_m)
        if np.iscomplexobj(raw_location):
            raise ValueError("input_location_m must be real")
        location = np.asarray(raw_location, dtype=np.float64)
        if location.shape != (3,) or not np.isfinite(location).all():
            raise ValueError("input_location_m must be a finite three-vector")
        object.__setattr__(
            self, "_input_location_tuple_m", tuple(float(x) for x in location)
        )
        object.__setattr__(
            self,
            "input_distribution",
            _nonempty("input_distribution", input_distribution),
        )
        object.__setattr__(
            self, "normalization", _nonempty("normalization", normalization)
        )
        object.__setattr__(
            self,
            "_basis_amplitude_si",
            _scalar_quantity("basis_amplitude", basis_amplitude, unit),
        )
        object.__setattr__(
            self,
            "mapping_kind",
            _descriptor("mapping_kind", mapping_kind, _MappingKind),
        )
        object.__setattr__(
            self, "action", _descriptor("action", action, _ResponseAction)
        )
        object.__setattr__(
            self, "storage", _descriptor("storage", storage, _CoefficientStorage)
        )
        object.__setattr__(self, "varied", _nonempty("varied", varied))
        object.__setattr__(
            self, "target_position", _nonempty("target_position", target_position)
        )
        object.__setattr__(self, "law_name", _nonempty("law_name", law_name))
        if not isinstance(law_parameters, Mapping):
            raise ValueError("law_parameters must be a mapping")
        copied = {}
        for key, value in law_parameters.items():
            if not isinstance(key, str) or not key:
                raise ValueError("law_parameters keys must be nonempty strings")
            if isinstance(value, np.generic):
                value = value.item()
            if value is not None and not isinstance(
                value, (str, bool, int, float, complex)
            ):
                raise ValueError("law_parameters values must be immutable scalars")
            if isinstance(value, (float, complex)) and not np.isfinite(value):
                raise ValueError("law_parameters values must be finite")
            copied[key] = value
        object.__setattr__(self, "law_parameters", MappingProxyType(copied))
        try:
            g = float(gravitational_constant_si)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "gravitational_constant_si must be positive and finite"
            ) from exc
        if not np.isfinite(g) or g <= 0:
            raise ValueError("gravitational_constant_si must be positive and finite")
        object.__setattr__(self, "gravitational_constant_si", g)
        object.__setattr__(
            self,
            "linearization_assessment",
            _descriptor(
                "linearization_assessment",
                linearization_assessment,
                _LinearizationAssessment,
            ),
        )

    @property
    def input_location_m(self) -> u.Quantity:
        """Return an independent position in metres in ``input_frame``."""
        return np.asarray(self._input_location_tuple_m) * u.m

    @property
    def basis_amplitude(self) -> u.Quantity:
        """Return an independent amplitude in the declared input unit."""
        return self._basis_amplitude_si * self.input_unit


class _AppliedContext(Protocol):
    context: EvaluationContext
    input_values: np.ndarray
    backend: str
    parameters: object
    formulas: object
    references: object

    def pack(self, values: Mapping[str, np.ndarray]) -> object: ...


class _ResponseEvaluation(Protocol):
    frequencies_hz: tuple[float, ...]
    target_ids: tuple[str, ...]
    parameters: object
    formulas: object
    references: object

    def _applied_context(
        self, value: object, *, representation: object, reference_time: object
    ) -> _AppliedContext: ...


_COMPONENT_COUNT = {"potential": 1, "acceleration": 3, "gradient": 9}
_CHANGED_INPUT = {"surface_vertical_displacement": "displacement"}


@dataclass(frozen=True, slots=True, init=False, eq=False)
class GravityResponse:
    """Own the coefficients and interpretation of a single-input response.

    Obtain a Rayleigh response from ``rayleigh_response(...).at(...).run()``;
    direct construction is an internal factory operation. Its stored transfer
    coefficients are separate from fields produced by applying an input.
    Complex outputs represent linear complex amplitudes or transfer-function-like
    responses, not directly real-valued time-domain gravitational fields.
    """

    contract: ResponseContract
    output_definition: GravityFieldDefinition
    evaluation: _ResponseEvaluation
    record: EvaluationRecord
    _frequencies_hz: tuple[float, ...]
    target_ids: tuple[str, ...]
    _coefficient_bytes: Mapping[str, bytes]

    def __init__(
        self,
        *,
        contract: ResponseContract,
        output_definition: GravityFieldDefinition,
        evaluation: _ResponseEvaluation,
        coefficients: Mapping[str, np.ndarray],
        record: EvaluationRecord,
    ) -> None:
        if not isinstance(contract, ResponseContract):
            raise TypeError("contract must be a ResponseContract")
        if not isinstance(output_definition, GravityFieldDefinition):
            raise TypeError("output_definition must be a GravityFieldDefinition")
        if contract.mapping_kind is _MappingKind.LOCAL_LINEARIZATION:
            change = output_definition.change
            if (
                output_definition.kind is not FieldKind.FIELD_CHANGE
                or change is None
                or change.form is not ChangeForm.LINEARIZED_RESPONSE
                or change.order != 1
                or change.changed_input != _CHANGED_INPUT.get(contract.input_quantity)
            ):
                raise ValueError(
                    "output_definition is inconsistent with local_linearization"
                )
        if contract.action is not _ResponseAction.PER_FREQUENCY:
            raise ValueError("action is not supported for a concrete response")
        if contract.storage is not _CoefficientStorage.PRECOMPUTED:
            raise ValueError("storage is not supported for a concrete response")
        if contract.varied != "source":
            raise ValueError("varied must be 'source' for a concrete response")
        if contract.target_position != "fixed":
            raise ValueError("target_position must be 'fixed' for a concrete response")
        if contract.law_name != "newton" or contract.law_parameters:
            raise ValueError(
                "law_name and law_parameters are not supported for a concrete response"
            )
        if contract.gravitational_constant_si != G_SI:
            raise ValueError("gravitational_constant_si must match configured G_SI")
        if len(contract.input_channel_ids) != 1 or len(contract.input_components) != 1:
            raise ValueError(
                "a concrete response requires one input channel and component"
            )
        if (
            contract.linearization_assessment
            is not _LinearizationAssessment.NOT_ASSESSED
        ):
            raise ValueError("linearization_assessment is not supported")
        for name in (
            "frequencies_hz",
            "target_ids",
            "parameters",
            "formulas",
            "references",
            "_applied_context",
        ):
            if not hasattr(evaluation, name):
                raise TypeError(f"evaluation must provide {name}")
        if not callable(evaluation._applied_context):
            raise TypeError("evaluation._applied_context must be callable")
        model_spec = getattr(evaluation, "model_spec", None)
        if model_spec is not None:
            if hasattr(model_spec, "contract") and model_spec.contract is not contract:
                raise ValueError("evaluation.model_spec.contract must match contract")
            if (
                hasattr(model_spec, "output_definition")
                and model_spec.output_definition is not output_definition
            ):
                raise ValueError(
                    "evaluation.model_spec.output_definition must match "
                    "output_definition"
                )
        frequencies = tuple(float(x) for x in evaluation.frequencies_hz)
        if (
            not frequencies
            or not np.isfinite(frequencies).all()
            or any(x <= 0 for x in frequencies)
            or any(b <= a for a, b in zip(frequencies, frequencies[1:], strict=False))
        ):
            raise ValueError(
                "evaluation frequencies_hz must be positive, finite, "
                "and strictly increasing"
            )
        target_ids = _strings("evaluation target_ids", evaluation.target_ids)
        if not isinstance(record, EvaluationRecord):
            raise TypeError("record must be an EvaluationRecord")
        if (
            record.gravitational_constant_si != contract.gravitational_constant_si
            or record.g_applications != 1
        ):
            raise ValueError(
                "record gravity constant and application count must match contract"
            )
        if not isinstance(coefficients, Mapping) or set(coefficients) != set(
            OBSERVABLES
        ):
            raise ValueError("coefficients must contain all three observables")
        owned = {}
        for observable in OBSERVABLES:
            array = np.asarray(coefficients[observable])
            expected = (
                len(target_ids),
                _COMPONENT_COUNT[observable],
                1,
                len(frequencies),
            )
            if (
                array.shape != expected
                or array.dtype != np.dtype(np.complex128)
                or not np.isfinite(array).all()
            ):
                raise ValueError(
                    f"{observable} coefficients must be finite complex128 "
                    f"shape {expected}"
                )
            owned[observable] = array.copy().tobytes()
        object.__setattr__(self, "contract", contract)
        object.__setattr__(self, "output_definition", output_definition)
        object.__setattr__(self, "evaluation", evaluation)
        object.__setattr__(self, "record", record)
        object.__setattr__(self, "_frequencies_hz", frequencies)
        object.__setattr__(self, "target_ids", target_ids)
        object.__setattr__(self, "_coefficient_bytes", MappingProxyType(owned))

    @property
    def observables(self) -> tuple[str, ...]:
        return OBSERVABLES

    @property
    def frequencies(self) -> u.Quantity:
        return np.asarray(self._frequencies_hz) * u.Hz

    @property
    def parameters(self) -> object:
        return self.evaluation.parameters

    @property
    def formulas(self) -> object:
        return self.evaluation.formulas

    @property
    def references(self) -> object:
        return self.evaluation.references

    def _coefficient_array(self, observable: str) -> np.ndarray:
        """Return a protected internal coefficient view for response operations."""
        if observable not in OBSERVABLES:
            raise ValueError(f"unknown observable {observable!r}")
        return np.frombuffer(
            self._coefficient_bytes[observable], dtype=np.complex128
        ).reshape(
            len(self.target_ids),
            _COMPONENT_COUNT[observable],
            1,
            len(self._frequencies_hz),
        )

    def coefficients(self, observable: str):
        """Return a fresh native ``FrequencySeriesMatrix`` of coefficients.

        Values and metadata are independent of the response. Rows identify
        targets, columns identify components, and the frequency axis is last.
        Potential has one component, acceleration three, and gradient nine.
        Units are the field SI unit divided by ``contract.input_unit``; for a
        Rayleigh metre input these are m s⁻², s⁻², and m⁻¹ s⁻², respectively.
        These coefficients are not applied fields.
        """
        from gragra._phasor_matrix import _phasor_matrices

        if observable not in OBSERVABLES:
            raise ValueError(f"unknown observable {observable!r}")

        field_units = {
            "potential": u.m**2 / u.s**2,
            "acceleration": u.m / u.s**2,
            "gradient": u.s**-2,
        }
        input_name = self.contract.input_name
        arrays = {
            name: self._coefficient_array(name)[:, :, 0, :] for name in OBSERVABLES
        }
        units = {
            name: field_units[name] / self.contract.input_unit for name in OBSERVABLES
        }
        matrices = _phasor_matrices(
            arrays,
            self.target_ids,
            self._frequencies_hz,
            units=units,
            cell_name=lambda name, target, component, _row, _column: (
                f"{target} {component} {name} response per {input_name}"
            ),
            cell_channel=lambda name, _target, _component, row, column: (
                f"GRAGRA:response:{input_name}:{name}:{row}:{column}"
            ),
            matrix_name=lambda name: f"{name} response per {input_name}",
        )
        return matrices[observable]

    def apply(
        self, *, representation: object, reference_time: object, **inputs: object
    ):
        """Apply a phasor to the stored transfer and return an ``EvaluationResult``.

        For the current Rayleigh response, provide ``vertical_displacement``
        as a GWexpy ``FrequencySeries`` with a length unit. Values are converted
        to metres. Only ``PhasorRepresentation`` is supported, and an explicit
        GPS ``reference_time`` is required; the series epoch is not a fallback.
        Frequencies are converted to hertz and must match the stored strictly
        increasing axis exactly. No interpolation or reordering is performed.

        Application multiplies the stored coefficients by the finite SI input
        amplitudes without rerunning the kernel or applying G again. The result
        retains the actual input and its reference time. The Rayleigh output is
        a first-order field change at fixed observation points, with no reference
        field added; linearization applicability is ``not_assessed``.
        """
        from gwexpy.frequencyseries import FrequencySeries

        from gragra.results._evaluation import EvaluationResult
        from gragra.results._spectral import (
            AmplitudeSpectrumRepresentation,
            ASDRepresentation,
            CSDRepresentation,
            FourierTransformRepresentation,
            PhasorRepresentation,
            PSDRepresentation,
        )

        expected_inputs = {self.contract.input_name}
        if set(inputs) != expected_inputs:
            raise TypeError(f"inputs must contain exactly {self.contract.input_name!r}")
        value = inputs[self.contract.input_name]
        if not isinstance(value, FrequencySeries):
            raise TypeError(f"{self.contract.input_name} must be a FrequencySeries")

        known_representations = (
            FourierTransformRepresentation,
            AmplitudeSpectrumRepresentation,
            PhasorRepresentation,
            PSDRepresentation,
            ASDRepresentation,
            CSDRepresentation,
        )
        if not isinstance(representation, known_representations):
            raise TypeError("representation must be a spectral representation")
        if not isinstance(representation, PhasorRepresentation):
            raise ValueError("only PhasorRepresentation is supported")

        context = self.evaluation._applied_context(
            value,
            representation=representation,
            reference_time=reference_time,
        )
        input_values = np.asarray(context.input_values, dtype=np.complex128)
        if (
            input_values.shape != (len(self._frequencies_hz),)
            or not np.isfinite(input_values).all()
        ):
            raise ValueError(
                "applied SI input values must be finite and match frequencies"
            )

        products = {}
        with np.errstate(over="ignore", invalid="ignore"):
            for observable in OBSERVABLES:
                coefficients = self._coefficient_array(observable)[:, :, 0, :]
                product = coefficients * input_values[None, None, :]
                if not np.isfinite(product).all():
                    raise ValueError(f"applied {observable} values must be finite")
                products[observable] = product

        values = context.pack(products)
        record = EvaluationRecord(
            backend=context.backend,
            versions=self.record.versions,
            gravitational_constant_si=self.record.gravitational_constant_si,
            g_applications=1,
        )
        return EvaluationResult(
            values=values,
            evaluation=context,
            record=record,
        )
