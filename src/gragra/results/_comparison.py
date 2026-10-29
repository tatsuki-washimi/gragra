"""Comparison of compatible evaluation results by representation descriptor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.table import QTable
from gwexpy.frequencyseries import FrequencySeriesMatrix
from gwexpy.timeseries import TimeSeriesMatrix

from gragra._time_matrix import make_time_matrix
from gragra.results import GravityResult
from gragra.results._evaluation import EvaluationResult
from gragra.results._field_definition import (
    FieldKind,
    GravityFieldDefinition,
)
from gragra.results._representation import (
    StaticRepresentation,
    TimeSeriesRepresentation,
)
from gragra.results._spectral import PhasorRepresentation

_POSITION_ATOL_M = 1e-12
_POSITION_RTOL = 1e-12
_FRAME_ATOL = 1e-12


def _float_values(quantity: u.Quantity, unit: u.UnitBase) -> np.ndarray:
    """Convert values for stable float64 comparison arithmetic."""
    try:
        values = np.asarray(quantity.to_value(unit), dtype=np.float64)
    except (TypeError, ValueError, OverflowError, u.UnitConversionError) as exc:
        raise ValueError(
            "observable cannot be represented in float64 comparison units"
        ) from exc
    if not np.isfinite(values).all():
        raise ValueError("observable conversion produced non-finite comparison values")
    return values


@dataclass(frozen=True, eq=False)
class ComparisonResult:
    """Aligned candidate-minus-reference values and norm/phase summaries."""

    reference: EvaluationResult
    candidate: EvaluationResult
    differences: GravityResult
    summary: QTable
    phase_differences: QTable | None = None


def _positions_in_root(result: EvaluationResult) -> np.ndarray:
    targets = result.context.targets
    root = targets.frame.root
    return targets.frame.transform_points_m(targets.positions_m, root)


def _target_order_and_check_positions(
    reference: EvaluationResult, candidate: EvaluationResult
) -> np.ndarray:
    reference_targets = reference.context.targets
    candidate_targets = candidate.context.targets
    root = reference_targets.frame.root
    if candidate_targets.frame.root is not root:
        raise ValueError("target frames must share the same connected root Frame")

    reference_ids = reference_targets.ids
    candidate_ids = candidate_targets.ids
    if len(reference_ids) != len(candidate_ids) or set(reference_ids) != set(
        candidate_ids
    ):
        raise ValueError("reference and candidate target IDs must match exactly")
    candidate_indices = {
        target_id: index for index, target_id in enumerate(candidate_ids)
    }
    order = np.asarray(
        [candidate_indices[target_id] for target_id in reference_ids], dtype=np.intp
    )

    reference_positions = _positions_in_root(reference)
    candidate_positions = _positions_in_root(candidate)[order]
    delta_m = np.abs(candidate_positions - reference_positions)
    tolerance_m = _POSITION_ATOL_M + _POSITION_RTOL * np.abs(reference_positions)
    if np.any(delta_m > tolerance_m):
        raise ValueError(
            "target positions differ in the common root Frame beyond "
            "abs(candidate-reference) <= 1e-12 m + 1e-12*abs(reference)"
        )
    return order


def _c2_field_definition_projection(
    definition: GravityFieldDefinition,
) -> tuple[tuple[str, object], ...]:
    """Return only typed field-definition items required for C2 comparison."""
    projection: list[tuple[str, object]] = [
        ("kind", definition.kind),
        ("contribution", definition.contribution),
        ("potential_reference", definition.potential_reference),
    ]
    if definition.kind is FieldKind.SOURCE_FIELD:
        return tuple(projection)

    change = definition.change
    if change is None:
        raise ValueError("field definition change is missing")
    projection.extend(
        (
            ("change.form", change.form),
            ("change.changed_input", change.changed_input),
            ("change.order", change.order),
            ("change.fixed_conditions", change.fixed_conditions),
            ("change.reference.status", change.reference.status),
            ("change.reference.temporal_kind", change.reference.temporal_kind),
            ("change.reference.state_kind", change.reference.state_kind),
        )
    )
    return tuple(projection)


def _check_c2_field_definition_compatibility(
    reference: GravityFieldDefinition,
    candidate: GravityFieldDefinition,
) -> None:
    """Reject differences in the field definition's C2 compatibility view."""
    if reference.kind is not candidate.kind:
        raise ValueError("field definition kind differs")
    reference_projection = _c2_field_definition_projection(reference)
    candidate_projection = _c2_field_definition_projection(candidate)
    for (field, reference_value), (candidate_field, candidate_value) in zip(
        reference_projection, candidate_projection, strict=True
    ):
        if field != candidate_field:
            raise RuntimeError("internal C2 field definition projection labels differ")
        if reference_value != candidate_value:
            raise ValueError(f"field definition {field} differs")


def _check_binding_compatibility(
    reference: EvaluationResult,
    candidate: EvaluationResult,
    observables: tuple[str, ...],
) -> None:
    reference_root = reference.context.targets.frame.root
    candidate_root = candidate.context.targets.frame.root
    _check_c2_field_definition_compatibility(
        reference.field_definition, candidate.field_definition
    )
    for name in observables:
        ref_binding = reference.bindings[name]
        cand_binding = candidate.bindings[name]
        for attribute, label in (
            ("representation", "representation"),
            ("g_convention", "G convention"),
            ("component_axes", "component axes"),
        ):
            if getattr(ref_binding, attribute) != getattr(cand_binding, attribute):
                raise ValueError(f"{name} bindings have different {label}")
        if name == "potential":
            continue
        reference_frame = ref_binding.frame
        candidate_frame = cand_binding.frame
        if reference_frame is None or candidate_frame is None:
            raise ValueError(f"{name} requires an output component Frame")
        if (
            reference_frame.root is not reference_root
            or candidate_frame.root is not candidate_root
        ):
            raise ValueError(
                f"{name} output Frame must connect to the common target root"
            )
        reference_basis = reference_frame.rotation_to(reference_root)
        candidate_basis = candidate_frame.rotation_to(candidate_root)
        if not np.allclose(reference_basis, candidate_basis, atol=_FRAME_ATOL, rtol=0):
            raise ValueError(
                f"{name} output component bases differ; reevaluate both results "
                "with a common output_frame"
            )


def _aligned_difference(
    reference: u.Quantity, candidate: u.Quantity, candidate_order: np.ndarray
) -> u.Quantity:
    reference_values = _float_values(reference, reference.unit)
    candidate_values = _float_values(candidate, reference.unit)[candidate_order]
    with np.errstate(over="ignore", invalid="ignore"):
        difference_values = candidate_values - reference_values
    if not np.isfinite(difference_values).all():
        raise ValueError("candidate-reference difference is non-finite in float64")
    return difference_values * reference.unit


def _norms(observable: str, quantity: u.Quantity) -> np.ndarray:
    values = _float_values(quantity, quantity.unit)
    if observable == "potential":
        norms = np.abs(values)
    elif observable == "acceleration":
        norms = np.hypot.reduce(np.abs(values), axis=1)
    elif observable == "gradient":
        flat_values = np.abs(values).reshape(len(values), 9)
        norms = np.hypot.reduce(flat_values, axis=1)
    else:
        raise ValueError(f"unknown observable {observable!r}")
    if not np.isfinite(norms).all():
        raise ValueError(f"{observable} norm is not representable in float64")
    return norms


def _summary(reference: EvaluationResult, differences: GravityResult) -> QTable:
    target_ids = reference.context.targets.ids
    table = QTable({"target_id": np.asarray(target_ids, dtype=str)})
    table.meta["relative_difference_zero_reference"] = (
        "NaN and defined=False when the reference norm is exactly zero"
    )
    for observable in reference.observables:
        reference_quantity = getattr(reference._values, observable)
        difference = getattr(differences, observable)
        absolute = _norms(observable, difference)
        reference_norm = _norms(observable, reference_quantity)
        defined = reference_norm != 0
        relative = np.full(absolute.shape, np.nan, dtype=np.float64)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            np.divide(absolute, reference_norm, out=relative, where=defined)
        if not np.isfinite(relative[defined]).all():
            raise ValueError(
                f"{observable} relative difference is not representable in float64"
            )
        table[f"{observable}_absolute_difference"] = absolute * difference.unit
        table[f"{observable}_relative_difference"] = relative * u.dimensionless_unscaled
        table[f"{observable}_relative_defined"] = defined
    return table


def _si_matrix_values(matrix: FrequencySeriesMatrix, observable: str) -> np.ndarray:
    units = {
        "potential": u.m**2 / u.s**2,
        "acceleration": u.m / u.s**2,
        "gradient": u.s**-2,
    }
    raw = np.asarray(matrix, dtype=np.complex128)
    converted = np.empty(raw.shape, dtype=np.complex128)
    for row in range(raw.shape[0]):
        for column in range(raw.shape[1]):
            unit = u.Unit(matrix.meta[row, column].unit)
            converted[row, column] = raw[row, column] * unit.to(units[observable])
    if not np.isfinite(converted).all():
        raise ValueError(f"{observable} values are not finite in SI units")
    return converted


def _difference_matrix(
    reference: FrequencySeriesMatrix,
    candidate: FrequencySeriesMatrix,
    candidate_order: np.ndarray,
    observable: str,
) -> FrequencySeriesMatrix:
    reference_values = np.asarray(reference, dtype=np.complex128)
    candidate_raw = np.asarray(candidate, dtype=np.complex128)[candidate_order]
    difference = np.empty(reference_values.shape, dtype=np.complex128)
    units = np.empty(reference_values.shape[:2], dtype=object)
    si_unit = {
        "potential": u.m**2 / u.s**2,
        "acceleration": u.m / u.s**2,
        "gradient": u.s**-2,
    }[observable]
    for row in range(reference_values.shape[0]):
        for column in range(reference_values.shape[1]):
            reference_unit = u.Unit(reference.meta[row, column].unit)
            candidate_unit = u.Unit(candidate.meta[candidate_order[row], column].unit)
            try:
                reference_scale = reference_unit.to(si_unit)
                candidate_scale = candidate_unit.to(si_unit)
            except (TypeError, ValueError, u.UnitConversionError) as exc:
                raise ValueError(
                    f"{observable} reference and candidate cell units must "
                    "convert to SI"
                ) from exc
            converted = candidate_raw[row, column] * candidate_scale
            reference_si = reference_values[row, column] * reference_scale
            with np.errstate(over="ignore", invalid="ignore"):
                difference[row, column] = converted - reference_si
            units[row, column] = si_unit
    if not np.isfinite(difference).all():
        raise ValueError(f"{observable} candidate-reference difference is non-finite")
    return FrequencySeriesMatrix(
        difference,
        frequencies=reference.frequencies.copy(),
        units=units,
        names=reference.names,
        channels=reference.channels,
        rows=list(reference.rows.keys()),
        cols=list(reference.cols.keys()),
        name=f"candidate - reference {observable}",
    )


def _phasor_norms(matrix: FrequencySeriesMatrix, observable: str) -> np.ndarray:
    si_values = _si_matrix_values(matrix, observable)
    magnitudes = np.abs(si_values)
    if observable == "potential":
        norms = magnitudes[:, 0, :]
    else:
        norms = np.hypot.reduce(magnitudes, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError(f"{observable} norm is not representable in float64")
    return norms


def _phasor_summary(reference: EvaluationResult, differences: GravityResult) -> QTable:
    targets = reference.context.targets
    frequencies_hz = np.asarray(
        reference.bindings[reference.observables[0]].frequencies_hz,
        dtype=np.float64,
    )
    row_count = len(targets) * len(frequencies_hz)
    table = QTable(
        {
            "target_id": np.repeat(
                np.asarray(targets.ids, dtype=str), len(frequencies_hz)
            ),
            "frequency": np.tile(frequencies_hz, len(targets)) * u.Hz,
        }
    )
    table.meta["relative_difference_zero_reference"] = (
        "NaN and defined=False when the reference norm is exactly zero"
    )
    for observable in reference.observables:
        reference_matrix = getattr(reference._values, observable)
        difference_matrix = getattr(differences, observable)
        absolute = _phasor_norms(difference_matrix, observable)
        reference_norm = _phasor_norms(reference_matrix, observable)
        defined = reference_norm != 0
        relative = np.full(reference_norm.shape, np.nan, dtype=np.float64)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            np.divide(absolute, reference_norm, out=relative, where=defined)
        if not np.isfinite(relative[defined]).all():
            raise ValueError(
                f"{observable} relative difference is not representable in float64"
            )
        table[f"{observable}_absolute_difference"] = (
            absolute.reshape(row_count)
            * {
                "potential": u.m**2 / u.s**2,
                "acceleration": u.m / u.s**2,
                "gradient": u.s**-2,
            }[observable]
        )
        table[f"{observable}_relative_difference"] = (
            relative.reshape(row_count) * u.dimensionless_unscaled
        )
        table[f"{observable}_relative_defined"] = defined.reshape(row_count)
    return table


def _phase_summary(
    reference: EvaluationResult,
    candidate: EvaluationResult,
    candidate_order: np.ndarray,
) -> QTable:
    target_ids: list[str] = []
    observable_names: list[str] = []
    component_names: list[str] = []
    frequencies_hz: list[float] = []
    phases: list[float] = []
    defined_values: list[bool] = []
    for observable in reference.observables:
        reference_matrix = getattr(reference._values, observable)
        candidate_matrix = getattr(candidate._values, observable)
        reference_values = np.asarray(reference_matrix, dtype=np.complex128)
        candidate_values = np.asarray(candidate_matrix, dtype=np.complex128)[
            candidate_order
        ]
        axis = np.asarray(
            reference.bindings[observable].frequencies_hz, dtype=np.float64
        )
        for component_index, component in enumerate(reference_matrix.cols.keys()):
            for target_index, target_id in enumerate(reference.context.targets.ids):
                ref = reference_values[target_index, component_index]
                cand = candidate_values[target_index, component_index]
                defined = (ref != 0) & (cand != 0)
                phase = np.full(len(axis), np.nan, dtype=np.float64)
                delta = np.angle(cand) - np.angle(ref)
                phase[defined] = np.angle(np.exp(1j * delta[defined]))
                target_ids.extend([target_id] * len(axis))
                observable_names.extend([observable] * len(axis))
                component_names.extend([component] * len(axis))
                frequencies_hz.extend(axis.tolist())
                phases.extend(phase.tolist())
                defined_values.extend(defined.tolist())
    return QTable(
        {
            "target_id": np.asarray(target_ids, dtype=str),
            "observable": np.asarray(observable_names, dtype=str),
            "component": np.asarray(component_names, dtype=str),
            "frequency": np.asarray(frequencies_hz, dtype=np.float64) * u.Hz,
            "phase_difference": np.asarray(phases, dtype=np.float64) * u.rad,
            "phase_defined": np.asarray(defined_values, dtype=bool),
        }
    )


def _timeseries_si_values(matrix: TimeSeriesMatrix, observable: str) -> np.ndarray:
    """Return real TimeSeriesMatrix cells converted to canonical SI units."""
    raw = np.asarray(matrix)
    units = matrix.units
    si_unit = _EXPECTED_UNITS[observable]
    converted = np.empty(raw.shape, dtype=np.float64)
    for row in range(raw.shape[0]):
        for column in range(raw.shape[1]):
            try:
                scale = u.Unit(units[row, column]).to(si_unit)
            except (TypeError, ValueError, u.UnitConversionError) as exc:
                raise ValueError(
                    f"{observable} TimeSeriesMatrix cell units must convert to SI"
                ) from exc
            with np.errstate(over="ignore", invalid="ignore"):
                converted[row, column] = raw[row, column] * scale
    if not np.isfinite(converted).all():
        raise ValueError(f"{observable} TimeSeriesMatrix is non-finite in SI units")
    return converted


_EXPECTED_UNITS = {
    "potential": u.m**2 / u.s**2,
    "acceleration": u.m / u.s**2,
    "gradient": u.s**-2,
}


def _timeseries_difference_matrix(
    reference: TimeSeriesMatrix,
    candidate: TimeSeriesMatrix,
    candidate_order: np.ndarray,
    observable: str,
    sampling,
) -> TimeSeriesMatrix:
    reference_values = _timeseries_si_values(reference, observable)
    candidate_values = _timeseries_si_values(candidate, observable)[candidate_order]
    with np.errstate(over="ignore", invalid="ignore"):
        difference = candidate_values - reference_values
    if not np.isfinite(difference).all():
        raise ValueError(f"{observable} candidate-reference difference is non-finite")
    return make_time_matrix(
        difference,
        sampling=sampling,
        unit=_EXPECTED_UNITS[observable],
        rows=tuple(reference.rows.keys()),
        cols=tuple(reference.cols.keys()),
        name=f"candidate - reference {observable}",
    )


def _timeseries_norms(matrix: TimeSeriesMatrix, observable: str) -> np.ndarray:
    values = np.abs(_timeseries_si_values(matrix, observable))
    if observable == "potential":
        norms = values[:, 0, :]
    else:
        norms = np.hypot.reduce(values, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError(f"{observable} time-series norm is not finite")
    return norms


def _timeseries_summary(
    reference: EvaluationResult,
    differences: GravityResult,
) -> QTable:
    sampling = reference.context.time_sampling
    if sampling is None:
        raise ValueError("TimeSeries comparison requires a shared TimeSampling")
    target_ids = reference.context.targets.ids
    count = sampling.count
    row_count = len(target_ids) * count
    table = QTable(
        {
            "target_id": np.repeat(np.asarray(target_ids, dtype=str), count),
            "sample_index": np.tile(np.arange(count, dtype=np.int64), len(target_ids)),
            "time_offset": np.tile(
                np.arange(count, dtype=np.float64) * sampling.dt_seconds,
                len(target_ids),
            )
            * u.s,
        }
    )
    table.meta["epoch_gps"] = sampling.epoch_gps
    table.meta["relative_difference_zero_reference"] = (
        "NaN and defined=False when the reference norm is exactly zero"
    )
    for observable in reference.observables:
        reference_values = _timeseries_norms(
            getattr(reference._values, observable), observable
        )
        difference_values = _timeseries_norms(
            getattr(differences, observable), observable
        )
        defined = reference_values != 0.0
        relative = np.full(reference_values.shape, np.nan, dtype=np.float64)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            np.divide(difference_values, reference_values, out=relative, where=defined)
        if not np.isfinite(relative[defined]).all():
            raise ValueError(
                f"{observable} relative difference is not representable in float64"
            )
        table[f"{observable}_absolute_difference"] = (
            difference_values.reshape(row_count) * _EXPECTED_UNITS[observable]
        )
        table[f"{observable}_relative_difference"] = (
            relative.reshape(row_count) * u.dimensionless_unscaled
        )
        table[f"{observable}_relative_defined"] = defined.reshape(row_count)
    return table


def _compare_timeseries(
    reference: EvaluationResult,
    candidate: EvaluationResult,
    candidate_order: np.ndarray,
    observables: tuple[str, ...],
) -> ComparisonResult:
    sampling = reference.context.time_sampling
    if sampling is None or candidate.context.time_sampling != sampling:
        raise ValueError("TimeSeries comparison time axis differs exactly")
    difference_values: dict[str, object] = {"name": "candidate - reference"}
    for observable in observables:
        reference_matrix = getattr(reference._values, observable)
        candidate_matrix = getattr(candidate._values, observable)
        difference_values[observable] = _timeseries_difference_matrix(
            reference_matrix,
            candidate_matrix,
            candidate_order,
            observable,
            sampling,
        )
    differences = GravityResult(**difference_values)
    return ComparisonResult(
        reference,
        candidate,
        differences,
        _timeseries_summary(reference, differences),
        None,
    )


def _compare_phasors(
    reference: EvaluationResult,
    candidate: EvaluationResult,
    candidate_order: np.ndarray,
    observables: tuple[str, ...],
) -> ComparisonResult:
    for name in observables:
        reference_binding = reference.bindings[name]
        candidate_binding = candidate.bindings[name]
        if reference_binding.reference_time_gps != candidate_binding.reference_time_gps:
            raise ValueError(f"{name} phasor reference time differs")
        if reference_binding.frequencies_hz != candidate_binding.frequencies_hz:
            raise ValueError(f"{name} frequency coordinates differ exactly")

    difference_values: dict[str, object] = {"name": "candidate - reference"}
    for observable in observables:
        difference_values[observable] = _difference_matrix(
            getattr(reference._values, observable),
            getattr(candidate._values, observable),
            candidate_order,
            observable,
        )
    differences = GravityResult(**difference_values)
    return ComparisonResult(
        reference,
        candidate,
        differences,
        _phasor_summary(reference, differences),
        _phase_summary(reference, candidate, candidate_order),
    )


def compare(
    *, reference: EvaluationResult, candidate: EvaluationResult
) -> ComparisonResult:
    """Compare compatible results after ID and SI-position alignment.

    Static Quantity payloads are promoted to float64, converted to reference
    units, and subtracted. Phasor Matrix cells are converted to SI before
    complex subtraction; their difference matrices preserve the native target,
    component, and frequency axes. The operation never intersects observable
    sets, rotates output components, interpolates positions, or rephases data.
    """
    if not isinstance(reference, EvaluationResult):
        raise TypeError("reference must be an EvaluationResult")
    if not isinstance(candidate, EvaluationResult):
        raise TypeError("candidate must be an EvaluationResult")
    reference.validate()
    candidate.validate()

    reference_observables = set(reference.observables)
    candidate_observables = set(candidate.observables)
    if reference_observables != candidate_observables:
        raise ValueError(
            "reference and candidate observable sets must match exactly; "
            "select a common set explicitly"
        )
    observables = tuple(reference.observables)
    if (
        reference.record.gravitational_constant_si
        != candidate.record.gravitational_constant_si
    ):
        raise ValueError("reference and candidate gravitational constants differ")
    if reference.record.g_applications != candidate.record.g_applications:
        raise ValueError("reference and candidate G application counts differ")

    reference_representation = reference.context.representation
    candidate_representation = candidate.context.representation
    if reference_representation != candidate_representation:
        raise ValueError("reference and candidate result representations differ")
    if (
        reference.context._interaction_name != candidate.context._interaction_name
        or reference.context._interaction_parameters
        != candidate.context._interaction_parameters
    ):
        raise ValueError("reference and candidate interactions differ")

    candidate_order = _target_order_and_check_positions(reference, candidate)
    _check_binding_compatibility(reference, candidate, observables)

    if isinstance(reference_representation, PhasorRepresentation):
        return _compare_phasors(reference, candidate, candidate_order, observables)
    if isinstance(reference_representation, TimeSeriesRepresentation):
        return _compare_timeseries(reference, candidate, candidate_order, observables)
    from gragra.results._spectral import (
        AmplitudeSpectrumRepresentation,
        FourierTransformRepresentation,
    )

    if isinstance(
        reference_representation,
        (FourierTransformRepresentation, AmplitudeSpectrumRepresentation),
    ):
        raise ValueError(
            "compare() does not support Fourier or amplitude spectrum results in M3-B"
        )
    if not isinstance(reference_representation, StaticRepresentation):
        raise TypeError("result representation is unsupported")

    difference_values: dict[str, object] = {"name": "candidate - reference"}
    for observable in observables:
        reference_quantity = getattr(reference._values, observable)
        candidate_quantity = getattr(candidate._values, observable)
        difference_values[observable] = _aligned_difference(
            reference_quantity, candidate_quantity, candidate_order
        )
    differences = GravityResult(**difference_values)
    summary = _summary(reference, differences)
    return ComparisonResult(reference, candidate, differences, summary)
