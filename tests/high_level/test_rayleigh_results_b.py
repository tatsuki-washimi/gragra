"""B-lane contracts for phasor result validation and comparison."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeriesMatrix

from gragra._evaluation_setup import EvaluationRecord
from gragra._rayleigh_contract import (
    RayleighModelSpec,
    RayleighResolvedEvaluation,
)
from gragra._rayleigh_evaluation_setup import RayleighEvaluationSetup, _native_matrices
from gragra.geometry._frame import Frame
from gragra.geometry._halfspace import HalfSpace
from gragra.results import GravityResult, PhasorRepresentation, compare
from gragra.results._evaluation import EvaluationResult
from gragra.targets._high_points import TargetPoints


def _resolved(
    *,
    ids: tuple[str, ...] = ("A", "B"),
    frequencies: tuple[float, ...] = (1.0, 1.7),
    output_frame: Frame | None = None,
    root_frame: Frame | None = None,
    reference_time: tuple[int, int] = (1_000_000_000, 5),
    phase_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> RayleighResolvedEvaluation:
    from gwexpy.frequencyseries import FrequencySeries

    root = Frame("surface") if root_frame is None else root_frame
    spec = RayleighModelSpec(
        geometry=HalfSpace(root),
        density=2500 * u.kg / u.m**3,
        vp=4000 * u.m / u.s,
        vs=2300 * u.m / u.s,
        vertical_displacement=FrequencySeries(
            np.ones(len(frequencies), dtype=np.complex128),
            frequencies=np.asarray(frequencies) * u.Hz,
            unit=u.m,
        ),
        representation=PhasorRepresentation(),
        reference_time=reference_time,
        phase_origin=np.asarray(phase_origin) * u.m,
        direction=(1.0, 0.0, 0.0),
    )

    class Model:
        pass

    model = Model()
    model.spec = spec
    # Reversed input order is intentional: comparisons align by ID.
    positions = {"A": (0.0, 0.0, -10.0), "B": (2.0, 0.0, -20.0)}
    targets = TargetPoints(
        np.asarray([positions[target_id] for target_id in ids]) * u.m,
        ids,
        root,
    )
    return RayleighResolvedEvaluation(model, targets, output_frame or root)


def _arrays(ids: tuple[str, ...], frequencies: int, offset: float = 0.0):
    per_target = {
        target_id: index for index, target_id in enumerate(("A", "B"), start=1)
    }
    potential = np.empty((frequencies, len(ids)), dtype=np.complex128)
    acceleration = np.empty((frequencies, len(ids), 3), dtype=np.complex128)
    gradient = np.empty((frequencies, len(ids), 3, 3), dtype=np.complex128)
    for target_index, target_id in enumerate(ids):
        target_value = per_target[target_id]
        for frequency_index in range(frequencies):
            base = target_value + frequency_index * 10
            potential[frequency_index, target_index] = base + 1j * (base + 0.5)
            acceleration[frequency_index, target_index] = (
                np.arange(3) + base + 1j * (np.arange(3) + base + 0.25)
            )
            gradient[frequency_index, target_index] = (
                np.arange(9).reshape(3, 3)
                + base
                + 1j * (np.arange(9).reshape(3, 3) + base + 0.75)
            )
    return {
        "potential": potential + offset * (1 + 1j),
        "acceleration": acceleration + offset * (1 + 1j),
        "gradient": gradient + offset * (1 + 1j),
    }


def _matrix_with_units(
    original: FrequencySeriesMatrix,
    unit: u.UnitBase,
    frequency_unit: u.UnitBase = u.Hz,
) -> FrequencySeriesMatrix:
    si_unit = original.meta[0, 0].unit
    values = np.asarray(original) * si_unit.to(unit)
    return FrequencySeriesMatrix(
        values,
        frequencies=original.frequencies.to(frequency_unit),
        units=np.full(original.shape[:2], unit, dtype=object),
        names=original.names,
        channels=original.channels,
        rows=list(original.rows.keys()),
        cols=list(original.cols.keys()),
        name=original.name,
    )


def _result(
    *,
    ids: tuple[str, ...] = ("A", "B"),
    frequencies: tuple[float, ...] = (1.0, 1.7),
    observables: Iterable[str] = ("potential", "acceleration", "gradient"),
    offset: float = 0.0,
    output_frame: Frame | None = None,
    root_frame: Frame | None = None,
    reference_time: tuple[int, int] = (1_000_000_000, 5),
    phase_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    potential_unit: u.UnitBase | None = None,
    frequency_unit: u.UnitBase = u.Hz,
) -> EvaluationResult:
    resolved = _resolved(
        ids=ids,
        frequencies=frequencies,
        output_frame=output_frame,
        root_frame=root_frame,
        reference_time=reference_time,
        phase_origin=phase_origin,
    )
    matrices = _native_matrices(
        _arrays(ids, len(frequencies), offset), ids, frequencies
    )
    if potential_unit is not None:
        matrices["potential"] = _matrix_with_units(
            matrices["potential"], potential_unit, frequency_unit
        )
    selected = tuple(observables)
    values = GravityResult(
        name="phasors",
        potential=matrices["potential"] if "potential" in selected else None,
        acceleration=matrices["acceleration"] if "acceleration" in selected else None,
        gradient=matrices["gradient"] if "gradient" in selected else None,
    )
    return EvaluationResult(
        values=values,
        evaluation=RayleighEvaluationSetup(resolved),
        record=EvaluationRecord(backend="fixture", versions={"fixture": "1"}),
    )


def test_phasor_result_accepts_native_matrices_and_select_preserves_payload() -> None:
    result = _result(frequencies=(1.0, 1.7, 4.25))

    assert result.observables == ("potential", "acceleration", "gradient")
    assert result.potential.shape == (2, 1, 3)
    assert result.acceleration.shape == (2, 3, 3)
    assert result.gradient.shape == (2, 9, 3)
    np.testing.assert_array_equal(
        np.asarray(result.gradient)[0, :, 0],
        _arrays(("A", "B"), 3)["gradient"][0, 0].reshape(9),
    )

    selected = result.select("gradient", "potential")
    assert selected.values.gradient is result.values.gradient
    assert selected.values.potential is result.values.potential
    assert selected.values.acceleration is None
    assert selected.evaluation is result.evaluation


def test_phasor_result_revalidates_mutable_units_and_column_labels() -> None:
    result = _result(observables=("potential",))
    result.values.potential.meta[0, 0].unit = u.s
    with pytest.raises(ValueError, match="equivalent"):
        result.validate()

    result = _result(observables=("gradient",))
    matrix = result.values.gradient
    label = matrix.cols.pop("xy")
    matrix.cols["bad"] = label
    with pytest.raises(ValueError, match="component labels"):
        result.validate()

    result = _result(observables=("potential",))
    result.values.potential.frequencies[0] = 1.1 * u.Hz
    with pytest.raises(ValueError, match="frequency axis"):
        result.validate()


def test_phasor_result_rejects_real_or_nonfinite_payload_mutations() -> None:
    result = _result(observables=("potential",))
    result.values.potential[0, 0, 0] = np.nan + 0j
    with pytest.raises(ValueError, match="finite"):
        result.validate()

    result = _result(observables=("potential",))
    matrix = result.values.potential
    real_matrix = FrequencySeriesMatrix(
        np.asarray(matrix.real),
        frequencies=matrix.frequencies,
        units=matrix.units,
        names=matrix.names,
        channels=matrix.channels,
        rows=list(matrix.rows.keys()),
        cols=list(matrix.cols.keys()),
    )
    with pytest.raises(ValueError, match="complex"):
        EvaluationResult(
            values=GravityResult(potential=real_matrix),
            evaluation=result.evaluation,
            record=result.record,
        )


def test_phasor_compare_aligns_ids_and_converts_units_and_frequency_units() -> None:
    reference = _result(observables=("potential",))
    candidate = _result(
        ids=("B", "A"),
        observables=("potential",),
        offset=2.0,
        potential_unit=u.cm**2 / u.s**2,
        frequency_unit=u.mHz,
        root_frame=reference.evaluation.resolved.computation_frame,
    )

    comparison = compare(reference=reference, candidate=candidate)

    assert comparison.differences.potential.shape == (2, 1, 2)
    assert comparison.differences.potential.meta[0, 0].unit == u.m**2 / u.s**2
    np.testing.assert_allclose(
        np.asarray(comparison.differences.potential)[:, 0, :],
        np.full((2, 2), 2 + 2j),
    )
    assert comparison.summary["target_id"].tolist() == ["A", "A", "B", "B"]
    np.testing.assert_array_equal(
        comparison.summary["frequency"].to_value(u.Hz), [1.0, 1.7, 1.0, 1.7]
    )
    np.testing.assert_allclose(
        comparison.summary["potential_absolute_difference"].to_value(u.m**2 / u.s**2),
        np.full(4, np.sqrt(8)),
    )
    assert comparison.phase_differences is not None
    assert set(comparison.phase_differences.colnames) >= {
        "target_id",
        "observable",
        "component",
        "frequency",
        "phase_difference",
        "phase_defined",
    }
    expected_phase = np.angle((3 + 3.5j) * np.conj(1 + 1.5j))
    assert comparison.phase_differences["phase_difference"][0].to_value(
        u.rad
    ) == pytest.approx(expected_phase)


def test_phasor_compare_uses_scalar_vector_and_frobenius_norms() -> None:
    reference = _result()
    candidate = _result(
        offset=2.0, root_frame=reference.evaluation.resolved.computation_frame
    )

    comparison = compare(reference=reference, candidate=candidate)

    np.testing.assert_allclose(
        comparison.summary["potential_absolute_difference"].to_value(u.m**2 / u.s**2),
        np.full(4, np.sqrt(8)),
    )
    np.testing.assert_allclose(
        comparison.summary["acceleration_absolute_difference"].to_value(u.m / u.s**2),
        np.full(4, np.sqrt(24)),
    )
    np.testing.assert_allclose(
        comparison.summary["gradient_absolute_difference"].to_value(u.s**-2),
        np.full(4, np.sqrt(72)),
    )


def test_phasor_compare_reports_undefined_phase_for_zero_components() -> None:
    reference = _result(observables=("potential",))
    candidate = _result(
        observables=("potential",),
        offset=-1.0,
        root_frame=reference.evaluation.resolved.computation_frame,
    )
    reference.values.potential[0, 0, 0] = 0j
    candidate.values.potential[0, 0, 0] = 0j

    comparison = compare(reference=reference, candidate=candidate)

    phase_rows = comparison.phase_differences
    assert not bool(phase_rows["phase_defined"][0])
    assert np.isnan(phase_rows["phase_difference"][0].value)
    assert not bool(comparison.summary["potential_relative_defined"][0])
    assert np.isnan(comparison.summary["potential_relative_difference"][0].value)


def test_phasor_compare_rejects_position_basis_gps_and_frequency_mismatches() -> None:
    reference = _result(observables=("acceleration",))
    # Rebuild with a different position but the same stable target IDs.
    root = reference.evaluation.resolved.computation_frame
    moved = _resolved(ids=("A", "B"), frequencies=(1.0, 1.7), root_frame=root)
    moved_spec = moved.model_spec
    # A distinct resolved target set is supplied by the helper below.
    new_targets = TargetPoints([[0, 0, -10], [2, 0, -20.01]] * u.m, ("A", "B"), root)
    from gragra._rayleigh_contract import RayleighResolvedEvaluation

    class Model:
        spec = moved_spec

    moved = RayleighResolvedEvaluation(Model(), new_targets, root)
    mats = _native_matrices(_arrays(("A", "B"), 2), ("A", "B"), (1.0, 1.7))
    moved_result = EvaluationResult(
        values=GravityResult(acceleration=mats["acceleration"]),
        evaluation=RayleighEvaluationSetup(moved),
        record=EvaluationRecord(backend="fixture", versions={"fixture": "1"}),
    )
    with pytest.raises(ValueError, match="target positions differ"):
        compare(reference=reference, candidate=moved_result)

    with pytest.raises(ValueError, match="sets must match exactly"):
        compare(reference=_result(root_frame=root), candidate=reference)

    angle = np.pi / 2
    rotated = Frame(
        "rotated output",
        basis=np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        ),
        parent=reference.evaluation.resolved.computation_frame,
    )
    rotated_result = _result(
        observables=("acceleration",), output_frame=rotated, root_frame=root
    )
    with pytest.raises(ValueError, match="common output_frame"):
        compare(reference=reference, candidate=rotated_result)

    with pytest.raises(ValueError, match="reference time"):
        compare(
            reference=reference,
            candidate=_result(
                observables=("acceleration",),
                reference_time=(1_000_000_000, 6),
                root_frame=root,
            ),
        )
    with pytest.raises(ValueError, match="frequency"):
        compare(
            reference=reference,
            candidate=_result(
                observables=("acceleration",),
                frequencies=(1.0, 1.8),
                root_frame=root,
            ),
        )

    # Input phase origins are provenance; compensated or otherwise differing
    # values do not add a compare rejection condition.
    compare(
        reference=reference,
        candidate=_result(
            observables=("acceleration",),
            phase_origin=(8.0, -3.0, 0.0),
            root_frame=root,
        ),
    )
