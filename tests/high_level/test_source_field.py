"""Contracts for gridded, source-position contribution fields."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from gwexpy import fields

from gragra._evaluation_setup import CoordinatePrecisionError
from gragra._plotting._source_contracts import SourceContributionResult
from gragra._plotting._source_data import SourceContributionData
from gragra._plotting._source_field import source_field, source_field_result
from gragra.constants import G_SI
from gragra.geometry import Frame
from gragra.gravity_sources import VolumeElementSource, WeightedPointSource
from gragra.gravity_sources._models import point_sum
from gragra.results import Contribution, GravityFieldDefinition, PotentialReference
from gragra.targets import TargetPoints as EvaluationTargetPoints

ROOT = Frame("source-field-root")
_GRID = (
    np.array([0.0, 1.0, 2.0]),
    np.array([0.0, 1.0, 2.0]),
    np.array([0.0, 1.0, 2.0, 3.0]),
)


def _setup(positions, weights, *, volumes=None, density=None, target=(0, 0, 0)):
    definition = GravityFieldDefinition.source_field(
        "source-field-test", Contribution("total"), PotentialReference.ZERO_AT_INFINITY
    )
    if volumes is None:
        source = WeightedPointSource(positions, weights)
    else:
        source = VolumeElementSource(positions, volumes, density)
    model = point_sum(source, frame=ROOT, field_definition=definition)
    targets = EvaluationTargetPoints([target] * u.m, ("target-0",), ROOT)
    return model.at(targets)


def _dynamic_record(
    source_values,
    positions,
    *,
    targets=None,
    frequencies=(10.0, 20.0),
    unit="m/s^2",
    observable="acceleration",
    softening=0.0,
    provider=None,
    source_ids=None,
    global_metadata=None,
):
    source_values = np.asarray(source_values, dtype=np.complex128)
    if source_values.ndim < 3:
        raise ValueError("test source_values need (F,M,N,...) axes")
    n_frequencies, n_targets, n_sources = source_values.shape[:3]
    groups = source_values.sum(axis=2, keepdims=True)
    totals = source_values.sum(axis=2)
    membership = np.zeros((n_targets, n_sources), dtype=np.int64)
    return SourceContributionResult(
        SourceContributionData(
            source_values=None if provider is not None else source_values,
            source_block_provider=provider,
            group_values=groups,
            total_values=totals,
            positions_m=positions,
            targets_m=([[0.0, 0.0, 0.0]] * n_targets if targets is None else targets),
            target_ids=tuple(f"target-{i}" for i in range(n_targets)),
            frequencies_hz=frequencies,
            unit=unit,
            observable=observable,
            frame="source-field-world",
            membership=membership,
            group_ids=("all",),
            source_ids=source_ids,
            global_metadata=global_metadata,
            softening_m=softening,
            coordinate_origin_m=[4.0, 5.0, 6.0],
            t_ref_s=2.5,
            phase_origin_m=[1.0, 2.0, 3.0],
            fourier_convention="Re[q_hat exp(-2pi i f(t-t_ref_s))]",
        )
    )


def _field_array(value):
    if isinstance(value, fields.ScalarField):
        return np.asarray(value)
    return value.to_array()


def test_static_scalar_fields_keep_all_physical_axes_and_exact_edges():
    positions = [
        [0.2, 0.2, 0.2],
        [0.8, 0.8, 0.8],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 3.0],
    ]
    setup = _setup(
        positions,
        [2.0, 12.0, 10.0, 7.0],
        volumes=[1.0, 3.0, 2.0, 1.0],
        density=[2.0, 4.0, 5.0, 7.0],
    )
    edges = tuple(np.array(axis, copy=True) * u.m for axis in _GRID)
    result = source_field_result(setup, "density", grid=edges, aggregation="mean")
    field = result.field

    assert isinstance(field, fields.ScalarField)
    assert field.shape == (1, 2, 2, 3)
    assert field.axis0_domain == "time"
    np.testing.assert_array_equal(field.axis("time").index.to_value(u.s), [0.0])
    assert field.axis("time").unit == u.s
    for axis, centers in zip(
        ("x", "y", "z"),
        ([0.5, 1.5], [0.5, 1.5], [0.5, 1.5, 2.5]),
        strict=True,
    ):
        np.testing.assert_array_equal(field.axis(axis).index.to_value(u.m), centers)
        assert field.axis(axis).unit == u.m
    values = _field_array(field)
    assert values[0, 0, 0, 0] == pytest.approx(3.5)
    assert values[0, 1, 1, 1] == pytest.approx(5.0)
    assert values[0, 1, 1, 2] == pytest.approx(7.0)
    assert np.isnan(values[0, 0, 0, 1])
    assert result.valid.shape == field.shape
    assert result.valid.sum() == 3
    assert not result.valid[0, 0, 0, 1]
    assert result.metadata["grid_edges_m"][2] == (0.0, 1.0, 2.0, 3.0)
    assert result.metadata["aggregation"] == "mean"
    with pytest.raises(TypeError):
        result.metadata["quantity"] = "changed"


def test_extensive_static_sum_conserves_values_and_snapshots_grid():
    positions = [[0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [2.0, 2.0, 3.0]]
    masses = np.array([2.0, 12.0, 7.0])
    setup = _setup(positions, masses)
    mutable_edges = [np.array(axis, copy=True) for axis in _GRID]
    result = source_field_result(setup, "mass", grid=mutable_edges, aggregation="sum")
    mutable_edges[0][0] = -100.0

    values = _field_array(result.field)
    assert values.shape == (1, 2, 2, 3)
    assert values.sum() == pytest.approx(masses.sum())
    assert values[0, 0, 0, 0] == pytest.approx(14.0)
    assert result.metadata["grid_edges_m"][0] != (-100.0, 1.0, 2.0)
    with pytest.raises(ValueError, match="aggregation"):
        source_field(setup, "mass", grid=_GRID, aggregation="mean")


def test_empty_cell_validity_and_unknown_volume_rejection():
    setup = _setup([[0.2, 0.2, 0.2]], [2.0])
    with pytest.raises(ValueError, match="volume"):
        source_field_result(setup, "density", grid=_GRID, aggregation="mean")

    volume_setup = _setup([[0.2, 0.2, 0.2]], [2.0], volumes=[1.0], density=[2.0])
    result = source_field_result(
        volume_setup, "density", grid=_GRID, aggregation="mean"
    )
    values = _field_array(result.field)
    assert np.isnan(values[0, 1, 1, 2])
    assert not result.valid[0, 1, 1, 2]
    assert result.valid[0, 0, 0, 0]


def test_grid_validation_rejects_coverage_shape_unit_and_nonfinite_edges():
    setup = _setup([[0.2, 0.2, 0.2]], [1.0])
    with pytest.raises(ValueError, match="cover"):
        source_field(
            setup, "mass", grid=([0.5, 1.0], [0, 1], [0, 1]), aggregation="sum"
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        source_field(setup, "mass", grid=([0, 1, 1], [0, 1], [0, 1]), aggregation="sum")
    with pytest.raises(ValueError, match="finite"):
        source_field(
            setup, "mass", grid=([0, np.inf], [0, 1], [0, 1]), aggregation="sum"
        )
    with pytest.raises(u.UnitConversionError):
        source_field(
            setup,
            "mass",
            grid=([0, 1] * u.s, [0, 1] * u.m, [0, 1] * u.m),
            aggregation="sum",
        )


def test_dynamic_tensor_selects_one_exact_frequency_and_keeps_nine_components():
    positions = [[0.25, 0.25, 0.25], [1.25, 1.25, 1.25]]
    values = np.empty((2, 2, 2, 3, 3), dtype=np.complex128)
    for f in range(2):
        for target in range(2):
            for source in range(2):
                values[f, target, source] = (
                    np.arange(1, 10).reshape(3, 3)
                    + 100 * f
                    + 10 * target
                    + source
                    + 1j * (np.arange(1, 10).reshape(3, 3) + source)
                )
    record = _dynamic_record(
        values,
        positions,
        targets=[[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        observable="gravity_gradient",
        unit="1/s2",
        source_ids=("s-a", "s-b"),
    )
    result = source_field_result(
        record,
        "complex_contribution",
        target_id="target-1",
        grid=([0, 1, 2] * u.m, [0, 1, 2] * u.m, [0, 1, 2] * u.m),
        aggregation="sum",
        frequency_hz=20 * u.Hz,
    )
    field = result.field
    assert isinstance(field, fields.TensorField)
    component = field[(0, 0)]
    assert component.axis0_domain == "frequency"
    np.testing.assert_array_equal(
        component.axis("frequency").index.to_value(u.Hz), [20.0]
    )
    assert set(field.keys()) == {(i, j) for i in range(3) for j in range(3)}
    actual = field.to_array()
    assert actual.shape == (1, 2, 2, 2, 3, 3)
    np.testing.assert_array_equal(actual[0, 0, 0, 0], values[1, 1, 0])
    np.testing.assert_array_equal(actual[0, 1, 1, 1], values[1, 1, 1])
    assert np.all(actual[0, 0, 0, 0][~np.eye(3, dtype=bool)] != 0)
    assert result.metadata["frequency_hz"] == 20.0
    assert result.metadata["target_id"] == "target-1"
    assert result.metadata["target_xyz_m"] == (20.0, 0.0, 0.0)
    assert result.metadata["source_ids"] == ("s-a", "s-b")
    assert result.metadata["t_ref_s"] == 2.5
    with pytest.raises(ValueError, match="exactly match"):
        source_field(
            record,
            "complex_contribution",
            target_id="target-1",
            grid=_GRID,
            aggregation="sum",
            frequency_hz=15 * u.Hz,
        )


@pytest.mark.parametrize(
    ("component_shape", "observable", "field_type"),
    [
        ((), "potential", fields.ScalarField),
        ((3,), "acceleration", fields.VectorField),
        ((3, 3), "gravity_gradient", fields.TensorField),
    ],
)
def test_dynamic_field_defaults_to_all_stored_frequencies(
    component_shape, observable, field_type
):
    count = int(np.prod(component_shape)) if component_shape else 1
    raw = np.arange(1, 3 * 1 * 2 * count + 1, dtype=np.float64).reshape(
        (3, 1, 2, *component_shape)
    )
    values = raw + 1j * (raw + 0.5)
    record = _dynamic_record(
        values,
        [[0.2, 0.2, 0.2], [1.2, 0.2, 0.2]],
        frequencies=(10.0, 20.0, 30.0),
        observable=observable,
        unit="m2/s2" if not component_shape else "m/s2" if count == 3 else "1/s2",
    )
    grid = ([0, 1, 2] * u.m, [0, 1] * u.m, [0, 1] * u.m)
    all_frequencies = source_field(
        record,
        "complex_contribution",
        target_id="target-0",
        grid=grid,
        aggregation="sum",
    )

    assert isinstance(all_frequencies, field_type)
    actual = _field_array(all_frequencies)
    assert actual.shape == (3, 2, 1, 1, *component_shape)
    component = (
        all_frequencies
        if isinstance(all_frequencies, fields.ScalarField)
        else all_frequencies["x"]
        if isinstance(all_frequencies, fields.VectorField)
        else all_frequencies[(0, 0)]
    )
    assert component.axis0_domain == "frequency"
    np.testing.assert_array_equal(
        component.axis("frequency").index.to_value(u.Hz), [10.0, 20.0, 30.0]
    )

    for frequency_index, frequency_hz in enumerate((10.0, 20.0, 30.0)):
        selected = source_field(
            record,
            "complex_contribution",
            target_id="target-0",
            grid=grid,
            aggregation="sum",
            frequency_hz=frequency_hz,
        )
        np.testing.assert_array_equal(
            _field_array(selected)[0], actual[frequency_index]
        )


def test_dynamic_phase_and_interference_keep_all_frequencies():
    values = np.array(
        [
            [[1.0 + 1.0j, 2.0 - 1.0j]],
            [[2.0 + 0.0j, -1.0 + 1.0j]],
            [[1.0 - 1.0j, -1.0 - 1.0j]],
        ]
    )
    record = _dynamic_record(
        values,
        [[0.2, 0.2, 0.2], [1.2, 0.2, 0.2]],
        frequencies=(10.0, 20.0, 30.0),
        observable="potential",
        unit="m2/s2",
    )
    kwargs = {
        "target_id": "target-0",
        "grid": ([0, 1, 2] * u.m, [0, 1] * u.m, [0, 1] * u.m),
        "aggregation": "sum",
    }
    phase = source_field_result(record, "phase", **kwargs)
    interference = source_field_result(record, "interference_fraction", **kwargs)

    phases = _field_array(phase.field)
    fractions = _field_array(interference.field)
    assert phases.shape == (3, 2, 1, 1)
    assert fractions.shape == phases.shape
    np.testing.assert_allclose(phases[:, 0, 0, 0], np.angle(values[:, 0, 0]))
    totals = values.sum(axis=2)[:, 0]
    expected_fractions = (
        np.real(values[:, 0] * np.conjugate(totals[:, None]))
        / np.abs(totals[:, None]) ** 2
    )
    np.testing.assert_allclose(fractions[:, :, 0, 0], expected_fractions)
    assert phase.valid.all()
    assert interference.valid.all()


def test_static_spatial_projection_conserves_sum_and_volume_weights_mean():
    grid = ([0, 1, 2] * u.m, [0, 1, 2] * u.m, [0, 1, 2] * u.m)
    positions = [[0.2, 0.2, 0.2], [0.2, 0.2, 1.2]]
    extensive = source_field_result(
        _setup(positions, [2.0, 6.0]),
        "mass",
        grid=grid,
        aggregation="sum",
        spatial_projection="XY",
    )
    extensive_values = _field_array(extensive.field)
    assert extensive_values.shape == (1, 2, 2, 1)
    assert np.sum(extensive_values) == pytest.approx(8.0)
    assert extensive.metadata["spatial_projection"] == "xy"
    assert extensive.metadata["projected_axis"] == "z"
    assert extensive.metadata["projection_range_m"] == (0.0, 2.0)
    assert extensive.metadata["unprojected_grid_edges_m"][2] == (0.0, 1.0, 2.0)
    np.testing.assert_array_equal(extensive.field.axis("z").index.to_value(u.m), [1.0])

    intensive = source_field_result(
        _setup(positions, [2.0, 12.0], volumes=[1.0, 3.0], density=[2.0, 4.0]),
        "density",
        grid=grid,
        aggregation="mean",
        spatial_projection="xy",
    )
    intensive_values = _field_array(intensive.field)
    assert intensive_values.shape == (1, 2, 2, 1)
    assert intensive_values[0, 0, 0, 0] == pytest.approx(3.5)
    assert intensive.valid[0, 0, 0, 0]
    assert not intensive.valid[0, 1, 1, 0]


@pytest.mark.parametrize(
    ("plane", "projected_axis"),
    (("xy", 2), ("xz", 1), ("yz", 0)),
)
def test_each_spatial_projection_preserves_total_and_projected_range(
    plane, projected_axis
):
    positions = [[0.2, 0.2, 0.2], [1.2, 0.2, 1.2], [1.2, 1.2, 0.2]]
    result = source_field_result(
        _setup(positions, [2.0, 6.0, 4.0]),
        "mass",
        grid=([0, 1, 2] * u.m, [0, 1, 2] * u.m, [0, 1, 2] * u.m),
        aggregation="sum",
        spatial_projection=plane,
    )

    values = _field_array(result.field)
    assert values.shape[projected_axis + 1] == 1
    assert np.sum(values) == pytest.approx(12.0)
    original_edges = result.metadata["unprojected_grid_edges_m"][projected_axis]
    assert result.metadata["projection_range_m"] == (
        original_edges[0],
        original_edges[-1],
    )
    assert result.field.axis("xyz"[projected_axis]).index.to_value(u.m)[0] == 1.0


def test_spatial_projection_sums_tensor_components_before_norm():
    setup = _setup(
        [[0.2, 0.2, 0.2], [0.2, 0.2, 1.2]],
        [2.0, 6.0],
    )
    grid = ([0, 1, 2] * u.m, [0, 1, 2] * u.m, [0, 1, 2] * u.m)
    full = source_field(
        setup,
        "gravity_gradient_contribution",
        target_id="target-0",
        grid=grid,
        aggregation="sum",
    ).to_array()
    projected_tensor = source_field(
        setup,
        "gravity_gradient_contribution",
        target_id="target-0",
        grid=grid,
        aggregation="sum",
        spatial_projection="xy",
    ).to_array()
    projected_norm = source_field(
        setup,
        "gravity_gradient_contribution",
        target_id="target-0",
        grid=grid,
        aggregation="sum",
        spatial_projection="xy",
        norm="frobenius",
    )
    expected = full.sum(axis=3, keepdims=True)

    assert projected_tensor.shape == (1, 2, 2, 1, 3, 3)
    np.testing.assert_allclose(projected_tensor, expected)
    np.testing.assert_allclose(
        _field_array(projected_norm),
        np.sqrt(np.sum(np.abs(expected) ** 2, axis=(-2, -1))),
    )


def test_dynamic_spatial_projection_sums_complex_values_before_phase():
    values = np.array([[[1.0 + 1.0j, 2.0 - 1.0j]]])
    record = _dynamic_record(
        values,
        [[0.2, 0.2, 0.2], [0.2, 0.2, 1.2]],
        frequencies=(10.0,),
        observable="potential",
        unit="m2/s2",
    )
    kwargs = {
        "target_id": "target-0",
        "grid": ([0, 1] * u.m, [0, 1] * u.m, [0, 1, 2] * u.m),
        "aggregation": "sum",
        "spatial_projection": "xy",
    }
    complex_result = source_field_result(record, "complex_contribution", **kwargs)
    phase = source_field_result(record, "phase", **kwargs)
    interference = source_field_result(record, "interference_fraction", **kwargs)

    assert _field_array(complex_result.field).shape == (1, 1, 1, 1)
    assert _field_array(complex_result.field)[0, 0, 0, 0] == 3.0 + 0.0j
    assert _field_array(phase.field)[0, 0, 0, 0] == 0.0
    assert phase.valid[0, 0, 0, 0]
    assert _field_array(interference.field)[0, 0, 0, 0] == pytest.approx(1.0)
    assert interference.valid[0, 0, 0, 0]


def test_dynamic_field_preserves_global_metadata_array_dtypes_and_values():
    global_metadata = {
        "flags": np.array([True, False], dtype=np.bool_),
        "labels": np.array(["north", "south"], dtype="U5"),
        "large_ids": np.array([2**53 + 1, 2**53 + 3], dtype=np.int64),
        "nan_scores": np.array([np.nan], dtype=np.float64),
        "records": np.array(
            [(2**53 + 5, "east")], dtype=[("id", np.int64), ("tag", "U4")]
        ),
    }
    record = _dynamic_record(
        np.array([[[1.0 + 2.0j]]]),
        [[0.5, 0.5, 0.5]],
        frequencies=(10.0,),
        observable="potential",
        unit="m2/s2",
        global_metadata=global_metadata,
    )
    result = source_field_result(
        record,
        "complex_contribution",
        target_id="target-0",
        grid=([0, 1] * u.m, [0, 1] * u.m, [0, 1] * u.m),
        aggregation="sum",
        frequency_hz=10.0,
    )

    preserved = result.metadata["global_metadata"]
    input_provenance = result.metadata["input_provenance"]
    assert record.global_metadata is record._data.global_metadata
    assert result.metadata["source_metadata"] is None
    for key, original in global_metadata.items():
        actual = preserved[key]
        provenance_value = input_provenance["global_metadata"][key]
        assert actual.dtype == original.dtype
        assert provenance_value.dtype == original.dtype
        np.testing.assert_array_equal(actual, original)
        np.testing.assert_array_equal(provenance_value, original)
        assert not actual.flags.writeable
        with pytest.raises(ValueError):
            actual.setflags(write=True)
    assert preserved["large_ids"][0] == 2**53 + 1


def test_dynamic_vector_preserves_all_three_components_and_projection():
    vals = np.array([[[[1 + 2j, 3 + 4j, 5 + 6j]]]], dtype=np.complex128)
    record = _dynamic_record(
        vals,
        [[0.25, 0.25, 0.25]],
        frequencies=(10.0,),
        observable="acceleration",
        unit="m/s2",
    )
    kwargs = {
        "target_id": "target-0",
        "grid": ([0, 1, 2] * u.m, [0, 1] * u.m, [0, 1] * u.m),
        "aggregation": "sum",
        "frequency_hz": 10.0,
    }
    field = source_field(record, "complex_contribution", **kwargs)
    assert isinstance(field, fields.VectorField)
    assert set(field.keys()) == {"x", "y", "z"}
    np.testing.assert_array_equal(field.to_array()[0, 0, 0, 0], vals[0, 0, 0])
    projected = source_field(
        record,
        "complex_contribution",
        projection=[0.0, 0.0, 2.0],
        **kwargs,
    )
    assert isinstance(projected, fields.ScalarField)
    assert _field_array(projected)[0, 0, 0, 0] == pytest.approx(vals[0, 0, 0, 2])


def test_complex_phasors_cancel_in_cell_before_norm_and_phase():
    vals = np.array(
        [[[[1 + 1j, 2 + 0j, 3 - 1j], [-1 - 1j, -2 + 0j, -3 + 1j]]]],
        dtype=np.complex128,
    )
    record = _dynamic_record(
        vals,
        [[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]],
        frequencies=(10.0,),
        observable="acceleration",
        unit="m/s2",
    )
    kwargs = {
        "target_id": "target-0",
        "grid": ([0, 1, 2] * u.m, [0, 1] * u.m, [0, 1] * u.m),
        "aggregation": "sum",
        "frequency_hz": 10.0,
    }
    raw = source_field_result(record, "complex_contribution", **kwargs)
    np.testing.assert_array_equal(_field_array(raw.field)[0, 0, 0, 0], [0.0j] * 3)
    assert raw.valid[0, 0, 0, 0].all()

    phase = source_field_result(record, "phase", **kwargs)
    np.testing.assert_array_equal(_field_array(phase.field)[0, 0, 0, 0], [0.0] * 3)
    assert not phase.valid[0, 0, 0, 0].any()
    assert not phase.valid[0, 1, 0, 0].any()
    with pytest.raises(ValueError, match="cannot be combined"):
        source_field_result(record, "phase", norm="euclidean", **kwargs)

    norm = source_field_result(
        record, "complex_contribution", norm="euclidean", **kwargs
    )
    assert _field_array(norm.field)[0, 0, 0, 0] == 0.0
    assert norm.valid[0, 0, 0, 0]
    assert (
        source_field_result(record, "phase", **kwargs).metadata["quantity"] == "phase"
    )


def test_interference_fraction_uses_complex_cell_sum_and_total_mask():
    vals = np.array([[[1 + 1j, 2 - 1j]]], dtype=np.complex128)
    record = _dynamic_record(
        vals,
        [[0.2, 0.2, 0.2], [1.2, 0.2, 0.2]],
        frequencies=(10.0,),
        observable="potential",
        unit="m2/s2",
    )
    result = source_field_result(
        record,
        "interference_fraction",
        target_id="target-0",
        grid=([0, 1, 2, 3] * u.m, [0, 1] * u.m, [0, 1] * u.m),
        aggregation="sum",
        frequency_hz=10.0,
    )
    # total = 3 + 0j; first cell is Re((1+i)*3)/9 = 1/3.
    actual = _field_array(result.field)
    assert actual[0, 0, 0, 0] == pytest.approx(1.0 / 3.0)
    assert actual[0, 1, 0, 0] == pytest.approx(2.0 / 3.0)
    assert actual[0, 2, 0, 0] == 0.0
    assert result.valid.all()  # empty cells are valid zero when total is nonzero

    canceled = _dynamic_record(
        np.array([[[1 + 0j, -1 + 0j]]]),
        [[0.2, 0.2, 0.2], [1.2, 0.2, 0.2]],
        frequencies=(10.0,),
        observable="potential",
        unit="m2/s2",
    )
    invalid = source_field_result(
        canceled,
        "interference_fraction",
        target_id="target-0",
        grid=([0, 1, 2, 3] * u.m, [0, 1] * u.m, [0, 1] * u.m),
        aggregation="sum",
        frequency_hz=10.0,
    )
    assert np.count_nonzero(_field_array(invalid.field)) == 0
    assert not invalid.valid.any()


@pytest.mark.parametrize("component_shape", [(), (3,), (3, 3)])
def test_interference_fraction_is_stable_for_tiny_nonzero_totals(component_shape):
    if component_shape:
        component_count = int(np.prod(component_shape))
        component_scale = np.arange(1, component_count + 1).reshape(component_shape)
        tiny_values = -6.6743e-171 * component_scale * (1.0 + 0.25j)
    else:
        tiny_values = np.array(-6.6743e-171 + 0.0j)
    values = np.asarray(tiny_values, dtype=np.complex128).reshape(
        (1, 1, 1) + component_shape
    )
    record = _dynamic_record(
        values,
        [[0.2, 0.2, 0.2]],
        frequencies=(10.0,),
        observable=(
            "potential"
            if not component_shape
            else "acceleration"
            if component_shape == (3,)
            else "gravity_gradient"
        ),
        unit=(
            "m2/s2"
            if not component_shape
            else "m/s2"
            if component_shape == (3,)
            else "1/s2"
        ),
    )

    result = source_field_result(
        record,
        "interference_fraction",
        target_id="target-0",
        grid=([0, 1, 2] * u.m, [0, 1] * u.m, [0, 1] * u.m),
        aggregation="sum",
        frequency_hz=10.0,
    )

    actual = _field_array(result.field)
    np.testing.assert_allclose(actual[0, 0, 0, 0], np.ones(component_shape or ()))
    np.testing.assert_array_equal(actual[0, 1, 0, 0], np.zeros(component_shape or ()))
    assert result.valid[0, 0, 0, 0].all()
    assert result.valid[0, 1, 0, 0].all()


@pytest.mark.parametrize("component_shape", [(), (3,), (3, 3)])
def test_interference_fraction_is_stable_for_subnormal_complex_totals(component_shape):
    if component_shape:
        component_count = int(np.prod(component_shape))
        component_scale = np.arange(1, component_count + 1).reshape(component_shape)
        subnormal_values = (-6.6743e-310 + 1.668575e-310j) * component_scale
    else:
        subnormal_values = np.array(-6.6743e-310 + 1.668575e-310j)
    values = np.asarray(subnormal_values, dtype=np.complex128).reshape(
        (1, 1, 1) + component_shape
    )
    record = _dynamic_record(
        values,
        [[0.2, 0.2, 0.2]],
        frequencies=(10.0,),
        observable=(
            "potential"
            if not component_shape
            else "acceleration"
            if component_shape == (3,)
            else "gravity_gradient"
        ),
        unit=(
            "m2/s2"
            if not component_shape
            else "m/s2"
            if component_shape == (3,)
            else "1/s2"
        ),
    )

    result = source_field_result(
        record,
        "interference_fraction",
        target_id="target-0",
        grid=([0, 1, 2] * u.m, [0, 1] * u.m, [0, 1] * u.m),
        aggregation="sum",
        frequency_hz=10.0,
    )

    actual = _field_array(result.field)
    np.testing.assert_allclose(actual[0, 0, 0, 0], np.ones(component_shape or ()))
    np.testing.assert_array_equal(actual[0, 1, 0, 0], np.zeros(component_shape or ()))
    assert result.valid[0, 0, 0, 0].all()
    assert result.valid[0, 1, 0, 0].all()


def test_interference_fraction_preserves_tiny_aligned_component():
    record = _dynamic_record(
        np.array([[[1e200j, 1e-200 - 1e200j]]]),
        [[0.2, 0.2, 0.2], [1.2, 0.2, 0.2]],
        frequencies=(10.0,),
        observable="potential",
        unit="m2/s2",
    )

    result = source_field_result(
        record,
        "interference_fraction",
        target_id="target-0",
        grid=([0, 1, 2] * u.m, [0, 1] * u.m, [0, 1] * u.m),
        aggregation="sum",
        frequency_hz=10.0,
    )

    actual = _field_array(result.field)
    assert actual[0, 0, 0, 0] == 0.0
    assert actual[0, 1, 0, 0] == 1.0
    assert result.valid[0, :, 0, 0].all()


def test_source_blocks_are_consumed_without_full_source_materialization():
    n_sources = 300
    positions = np.tile([[0.5, 0.5, 0.5]], (n_sources, 1))
    provider_widths = []
    source_values = np.ones((1, 1, n_sources), dtype=np.complex128)

    def provider(start, stop):
        provider_widths.append(stop - start)
        if stop - start >= n_sources:
            raise AssertionError("the source field materialized every source at once")
        return source_values[:, :, start:stop]

    record = _dynamic_record(
        source_values,
        positions,
        frequencies=(10.0,),
        observable="potential",
        unit="m2/s2",
        provider=provider,
    )
    field = source_field(
        record,
        "complex_contribution",
        target_id="target-0",
        grid=([0, 1] * u.m, [0, 1] * u.m, [0, 1] * u.m),
        aggregation="sum",
        frequency_hz=10.0,
    )
    assert _field_array(field)[0, 0, 0, 0] == 300.0
    assert len(provider_widths) == 2
    assert max(provider_widths) < n_sources


def test_zero_softening_and_plummer_softening_use_existing_kernel_terms():
    setup = _setup([[2.0, 0.0, 0.0]], [5.0])
    grid = ([1.5, 2.5] * u.m, [-0.5, 0.5] * u.m, [-0.5, 0.5] * u.m)
    unsoftened = source_field(
        setup,
        "acceleration_kernel",
        target_id="target-0",
        grid=grid,
        aggregation="sum",
        softening_m=0.0,
    )
    softened = source_field(
        setup,
        "acceleration_kernel",
        target_id="target-0",
        grid=grid,
        aggregation="sum",
        softening_m=1.0,
    )
    # Source-target kernel: Δ/r^3, then Δ/(r²+eps²)^(3/2).
    assert unsoftened.to_array()[0, 0, 0, 0, 0] == pytest.approx(0.25)
    assert softened.to_array()[0, 0, 0, 0, 0] == pytest.approx(2 / 5**1.5)
    assert softened.to_array()[0, 0, 0, 0, 0] != pytest.approx(1 / (4 + 1))

    physical = source_field(
        setup,
        "acceleration_contribution",
        target_id="target-0",
        grid=grid,
        aggregation="sum",
        softening_m=1.0,
    )
    assert physical.to_array()[0, 0, 0, 0, 0] == pytest.approx(5.0 * G_SI * 2 / 5**1.5)
    gradient = source_field(
        setup,
        "gravity_gradient_contribution",
        target_id="target-0",
        grid=grid,
        aggregation="sum",
        softening_m=1.0,
    )
    alias = source_field(
        setup,
        "GG",
        target_id="target-0",
        grid=grid,
        aggregation="sum",
        softening_m=1.0,
    )
    np.testing.assert_array_equal(alias.to_array(), gradient.to_array())


def test_static_physical_field_allows_softened_source_target_coincidence():
    setup = _setup([[0.0, 0.0, 0.0]], [2.0])
    grid = ([-1, 1] * u.m, [-1, 1] * u.m, [-1, 1] * u.m)
    field = source_field(
        setup,
        "gravity_gradient_contribution",
        target_id="target-0",
        grid=grid,
        aggregation="sum",
        softening_m=0.5,
    )
    assert np.isfinite(field.to_array()).all()
    with pytest.raises(CoordinatePrecisionError, match="collision"):
        source_field(
            setup,
            "gravity_gradient_contribution",
            target_id="target-0",
            grid=grid,
            aggregation="sum",
            softening_m=0.0,
        )


def test_static_physical_field_rejects_cross_frame_near_coincidence():
    root = Frame("source-field-precision-root")
    source_frame = Frame(
        "source-field-precision-local",
        origin=[1e6, 0.0, 0.0] * u.m,
        parent=root,
    )
    definition = GravityFieldDefinition.source_field(
        "source-field-precision-test",
        Contribution("total"),
        PotentialReference.ZERO_AT_INFINITY,
    )
    source = WeightedPointSource([[0.0, 0.0, 0.0]], [1.0])
    model = point_sum(source, frame=source_frame, field_definition=definition)
    targets = EvaluationTargetPoints(
        [[1e6 + 1e-8, 0.0, 0.0]] * u.m,
        ("near-target",),
        root,
    )
    setup = model.at(targets)
    grid = ([-0.5, 0.5] * u.m, [-0.5, 0.5] * u.m, [-0.5, 0.5] * u.m)

    with pytest.raises(CoordinatePrecisionError) as caught:
        source_field(
            setup,
            "acceleration_kernel",
            target_id="near-target",
            grid=grid,
            aggregation="sum",
            softening_m=0.5,
        )
    assert caught.value.reason == "coordinate_precision"
    assert caught.value.target_id == "near-target"
    assert caught.value.source_index == 0


def test_geometry_quantities_are_unsoftened_and_norm_projection_are_explicit():
    setup = _setup([[2.0, 0.0, 0.0]], [1.0])
    grid = ([1.5, 2.5] * u.m, [-0.5, 0.5] * u.m, [-0.5, 0.5] * u.m)
    with pytest.raises(ValueError, match="softening"):
        source_field(
            setup,
            "mass_over_r",
            target_id="target-0",
            grid=grid,
            aggregation="sum",
            softening_m=0.1,
        )

    record = _dynamic_record(
        np.array([[[[1 + 1j, 2 + 0j, 2 - 1j]]]]),
        [[0.5, 0.5, 0.5]],
        frequencies=(10.0,),
        observable="acceleration",
        unit="m/s2",
    )
    kwargs = {
        "target_id": "target-0",
        "grid": ([0, 1] * u.m, [0, 1] * u.m, [0, 1] * u.m),
        "aggregation": "sum",
        "frequency_hz": 10.0,
    }
    with pytest.raises(ValueError, match="norm and projection"):
        source_field(
            record,
            "complex_contribution",
            norm="euclidean",
            projection=[1, 0, 0],
            **kwargs,
        )
    with pytest.raises(ValueError, match="nonzero"):
        source_field(record, "complex_contribution", projection=[0, 0, 0], **kwargs)
    with pytest.raises(ValueError, match="norm"):
        source_field(record, "complex_contribution", norm="frobenius", **kwargs)
    with pytest.raises(ValueError, match="aggregation"):
        source_field(
            record,
            "complex_contribution",
            aggregation="mean",
            **{k: v for k, v in kwargs.items() if k != "aggregation"},
        )


def test_tensor_norms_and_bilateral_projection_use_all_nine_components():
    matrix = np.array(
        [[1 + 1j, 2, 3], [4, 5 - 1j, 6], [7, 8, 9 + 2j]],
        dtype=np.complex128,
    )
    record = _dynamic_record(
        matrix.reshape(1, 1, 1, 3, 3),
        [[0.5, 0.5, 0.5]],
        frequencies=(10.0,),
        observable="gravity_gradient",
        unit="1/s2",
    )
    kwargs = {
        "target_id": "target-0",
        "grid": ([0, 1] * u.m, [0, 1] * u.m, [0, 1] * u.m),
        "aggregation": "sum",
        "frequency_hz": 10.0,
    }
    frobenius = source_field(record, "complex_contribution", norm="frobenius", **kwargs)
    expected_frobenius = np.sqrt(np.sum(np.abs(matrix) ** 2))
    assert _field_array(frobenius)[0, 0, 0, 0] == pytest.approx(expected_frobenius)
    spectral = source_field(record, "complex_contribution", norm="spectral", **kwargs)
    assert _field_array(spectral)[0, 0, 0, 0] == pytest.approx(
        np.linalg.svd(matrix, compute_uv=False)[0]
    )
    projected = source_field(
        record,
        "complex_contribution",
        projection=([2.0, 0.0, 0.0], [0.0, 3.0, 0.0]),
        **kwargs,
    )
    assert _field_array(projected)[0, 0, 0, 0] == pytest.approx(matrix[0, 1])
    with pytest.raises(ValueError, match=r"requires \(left, right\)"):
        source_field(record, "complex_contribution", projection=[1, 0, 0], **kwargs)
    with pytest.raises(ValueError, match="nonzero"):
        source_field(
            record,
            "complex_contribution",
            projection=([1, 0, 0], [0, 0, 0]),
            **kwargs,
        )


def test_dynamic_softening_is_inherited_and_override_must_match():
    record = _dynamic_record(
        np.ones((2, 1, 1), dtype=np.complex128),
        [[0.5, 0.5, 0.5]],
        frequencies=(10.0, 20.0),
        observable="potential",
        unit="m2/s2",
        softening=0.25,
    )
    kwargs = {
        "target_id": "target-0",
        "grid": ([0, 1] * u.m, [0, 1] * u.m, [0, 1] * u.m),
        "aggregation": "sum",
        "frequency_hz": 10.0,
    }
    inherited = source_field_result(record, "complex_contribution", **kwargs)
    assert inherited.metadata["softening_m"] == 0.25
    with pytest.raises(ValueError, match="must match"):
        source_field(record, "complex_contribution", softening_m=0.5, **kwargs)


def test_static_gg_aliases_preserve_full_tensor_and_select_named_component():
    setup = _setup([[0.2, 0.2, 0.2]], [2.0])
    kwargs = {
        "target_id": "target-0",
        "grid": _GRID,
        "aggregation": "sum",
    }
    tensor = source_field(setup, "GG", **kwargs)
    tensor_alias = source_field(setup, "GG_kernel", **kwargs)
    xx_result = source_field_result(setup, "GG_xx", **kwargs)
    xx = xx_result.field
    canonical_tensor = source_field(setup, "gravity_gradient_contribution", **kwargs)
    canonical_kernel = source_field(setup, "gravity_gradient_kernel", **kwargs)

    assert isinstance(tensor, fields.TensorField)
    assert isinstance(tensor_alias, fields.TensorField)
    assert isinstance(xx, fields.ScalarField)
    np.testing.assert_array_equal(tensor.to_array(), canonical_tensor.to_array())
    np.testing.assert_array_equal(tensor_alias.to_array(), canonical_kernel.to_array())
    np.testing.assert_allclose(_field_array(xx), tensor.to_array()[..., 0, 0])
    assert u.Unit(xx_result.metadata["unit"]) == u.s**-2
