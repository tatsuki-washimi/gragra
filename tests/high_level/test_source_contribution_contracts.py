"""Neutral source-contribution records and GWexpy matrix wrapper contracts."""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from gwexpy.frequencyseries import FrequencySeriesMatrix

from gragra._plotting._source_contracts import SourceContributionResult
from gragra._plotting._source_data import SourceContributionData


def _record(*, lazy=False):
    source_values = (
        np.arange(2 * 2 * 3 * 9, dtype=np.float64)
        .reshape(2, 2, 3, 3, 3)
        .astype(complex)
    )
    groups = np.stack(
        [source_values[:, :, :1].sum(axis=2), source_values[:, :, 1:].sum(axis=2)],
        axis=2,
    )
    total = source_values.sum(axis=2)
    provider = None
    stored = source_values
    if lazy:
        stored = None

        def provider(start, stop):
            return source_values[:, :, start:stop, :]

    membership = np.array([[0, 0, 1], [1, 0, 1]])
    return SourceContributionData(
        source_values=stored,
        group_values=groups,
        total_values=total,
        source_block_provider=provider,
        positions_m=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        source_ids=("S0", "S1", "S2"),
        source_metadata=({"label": "a"}, {"label": "b"}, {"label": "c"}),
        targets_m=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        target_ids=("T0", "T1"),
        frequencies_hz=[1.0, 2.0],
        unit="m/s^2",
        observable="gravity_gradient",
        frame="world",
        coordinate_origin_m=[0.0, 0.0, 0.0],
        softening_m=0.25,
        membership=membership,
        group_ids=("near", "far"),
        fourier_convention="Re[q_hat exp(-2pi i f(t-t_ref_s))]",
        t_ref_s=1.5,
        phase_origin_m=[0.0, 0.0, 0.0],
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_record_provider_snapshot_and_block_iteration(lazy):
    record = _record(lazy=lazy)
    blocks = list(record.iter_source_blocks(block_size=2))
    assert [(start, stop) for start, stop, _ in blocks] == [(0, 2), (2, 3)]
    np.testing.assert_array_equal(
        np.concatenate([block for _, _, block in blocks], axis=2),
        _record(lazy=False).source_values,
    )
    assert not blocks[0][2].flags.writeable
    assert record.source_ids == ("S0", "S1", "S2")
    with pytest.raises(TypeError):
        record.source_metadata[0]["label"] = "changed"
    with pytest.raises(ValueError):
        record.positions_m.setflags(write=True)
    with pytest.raises(ValueError):
        record.total_values.setflags(write=True)
    selected_frequency = list(record.iter_source_blocks(frequency_index=1))
    assert selected_frequency[0][2].shape == (2, 3, 3, 3)
    with pytest.raises(IndexError, match="frequency_index"):
        list(record.iter_source_blocks(frequency_index=2))


def test_matrix_wrapper_preserves_components_labels_units_and_frequency():
    record = _record(lazy=True)
    result = SourceContributionResult(record)
    matrix = result.source("S1")
    assert isinstance(matrix, FrequencySeriesMatrix)
    assert tuple(matrix.rows.keys()) == ("T0", "T1")
    assert tuple(matrix.cols.keys()) == (
        "xx",
        "xy",
        "xz",
        "yx",
        "yy",
        "yz",
        "zx",
        "zy",
        "zz",
    )
    np.testing.assert_array_equal(matrix.frequencies.to_value("Hz"), [1.0, 2.0])
    assert matrix.units[0, 0].to_string() == "m / s2"
    np.testing.assert_array_equal(
        np.asarray(matrix),
        np.moveaxis(
            _record(lazy=False).source_values[:, :, 1, ...].reshape(2, 2, 9),
            0,
            -1,
        ),
    )
    grouped = result.group("far")
    np.testing.assert_array_equal(
        np.asarray(grouped),
        np.moveaxis(record.group_values[:, :, 1, ...].reshape(2, 2, 9), 0, -1),
    )
    np.testing.assert_array_equal(
        np.asarray(result.total),
        np.moveaxis(record.total_values.reshape(2, 2, 9), 0, -1),
    )


@pytest.mark.parametrize(
    "layout, expected_shape",
    [("sample_first", (2, 2, 3, 3, 3)), ("target_first", (2, 3, 2, 3, 3))],
)
def test_explicit_array_layout_materializes_all_components(layout, expected_shape):
    result = SourceContributionResult(_record(lazy=True))
    values = result.to_array(kind="source", layout=layout)
    assert values.shape == expected_shape
    assert not values.flags.writeable
    with pytest.raises(ValueError, match="layout"):
        result.to_array(kind="source", layout="native")
    with pytest.raises(ValueError, match="kind"):
        result.to_array(kind="raw", layout="sample_first")


def test_total_array_has_no_item_axis_and_matrix_is_an_independent_copy():
    record = _record()
    result = SourceContributionResult(record)
    exported = result.to_array(kind="total", layout="sample_first")
    assert exported.shape == (2, 2, 3, 3)
    matrix = result.total
    matrix[0, 0, 0] = -900.0
    assert record.total_values[0, 0, 0, 0] != -900.0


def test_explicit_array_layout_preserves_group_and_total_axis_order():
    result = SourceContributionResult(_record(lazy=True))
    sample_first = result.to_array(kind="group", layout="sample_first")
    target_first = result.to_array(kind="group", layout="target_first")
    assert sample_first.shape == (2, 2, 2, 3, 3)
    assert target_first.shape == (2, 2, 2, 3, 3)
    np.testing.assert_array_equal(target_first, np.moveaxis(sample_first, 0, 2))

    total_sample_first = result.to_array(kind="total", layout="sample_first")
    total_target_first = result.to_array(kind="total", layout="target_first")
    assert total_sample_first.shape == (2, 2, 3, 3)
    assert total_target_first.shape == (2, 2, 3, 3)
    np.testing.assert_array_equal(
        total_target_first, np.moveaxis(total_sample_first, 0, 1)
    )


def test_result_keeps_missing_source_ids_missing():
    record = _record(lazy=False)
    no_ids = SourceContributionData(
        source_values=record.source_values,
        group_values=record.group_values,
        total_values=record.total_values,
        positions_m=record.positions_m,
        targets_m=record.targets_m,
        target_ids=record.target_ids,
        frequencies_hz=record.frequencies_hz,
        unit=record.unit,
        observable=record.observable,
        frame=record.frame,
        membership=record.membership,
        group_ids=record.group_ids,
    )
    with pytest.raises(TypeError, match="integer source index"):
        SourceContributionResult(no_ids).source("S0")
    assert SourceContributionResult(no_ids).source(0).shape == (2, 9, 2)
    assert SourceContributionResult(no_ids).source_ids is None
    np.testing.assert_array_equal(
        SourceContributionResult(no_ids).source_index, [0, 1, 2]
    )


def test_single_lazy_source_selection_requests_only_that_source():
    eager = _record()
    requested = []

    def provider(start, stop):
        requested.append((start, stop))
        return eager.source_values[:, :, start:stop, :]

    deferred = SourceContributionData(
        source_values=None,
        source_block_provider=provider,
        group_values=eager.group_values,
        total_values=eager.total_values,
        positions_m=eager.positions_m,
        source_ids=eager.source_ids,
        targets_m=eager.targets_m,
        target_ids=eager.target_ids,
        frequencies_hz=eager.frequencies_hz,
        unit=eager.unit,
        observable=eager.observable,
        frame=eager.frame,
        membership=eager.membership,
        group_ids=eager.group_ids,
    )
    selected = SourceContributionResult(deferred).source("S2")
    assert selected.shape == (2, 9, 2)
    assert requested == [(2, 3)]


def test_provider_blocks_are_copied_and_bounds_are_checked():
    source = _record()
    backing = np.array(source.source_values, copy=True)
    requested = []

    def provider(start, stop):
        requested.append((start, stop))
        return backing[:, :, start:stop, ...]

    deferred = SourceContributionData(
        source_values=None,
        source_block_provider=provider,
        group_values=source.group_values,
        total_values=source.total_values,
        positions_m=source.positions_m,
        targets_m=source.targets_m,
        target_ids=source.target_ids,
        frequencies_hz=source.frequencies_hz,
        unit=source.unit,
        observable=source.observable,
        frame=source.frame,
        membership=source.membership,
        group_ids=source.group_ids,
    )
    values = deferred.source_block(1, 2)
    before = values.copy()
    # Simulate a provider backed by a writable external buffer changing after
    # a block was returned; the exposed block remains its own snapshot.
    backing[:, :, 1, ...] = -100
    np.testing.assert_array_equal(values, before)
    assert requested == [(1, 2)]
    with pytest.raises(IndexError, match="bounds"):
        deferred.source_block(-1, 1)
    with pytest.raises(IndexError, match="bounds"):
        deferred.source_block(2, 2)
    with pytest.raises(TypeError, match="integer"):
        deferred.source_block(True, 2)


@pytest.mark.parametrize(
    "mode, error, message",
    [
        ("shape", ValueError, "shape"),
        ("nonfinite", ValueError, "finite"),
        ("dtype", TypeError, "numeric"),
    ],
)
def test_provider_blocks_are_shape_dtype_and_finiteness_validated(mode, error, message):
    base = _record()

    def provider(start, stop):
        if mode == "shape":
            return np.zeros((2, 2, stop - start, 8), dtype=np.complex128)
        if mode == "nonfinite":
            values = np.zeros((2, 2, stop - start, 3, 3), dtype=np.complex128)
            values[0, 0, 0, 0, 0] = np.inf
            return values
        return np.full((2, 2, stop - start, 3, 3), "not numeric", dtype=object)

    deferred = SourceContributionData(
        source_values=None,
        source_block_provider=provider,
        group_values=base.group_values,
        total_values=base.total_values,
        positions_m=base.positions_m,
        targets_m=base.targets_m,
        target_ids=base.target_ids,
        frequencies_hz=base.frequencies_hz,
        unit=base.unit,
        observable=base.observable,
        frame=base.frame,
        membership=base.membership,
        group_ids=base.group_ids,
    )
    with pytest.raises(error, match=message):
        deferred.source_block(0, 1)


def test_record_defensively_snapshots_arrays_and_nested_metadata():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    values = (
        np.arange(2 * 2 * 3 * 9, dtype=np.float64)
        .reshape(2, 2, 3, 3, 3)
        .astype(complex)
    )
    metadata = {"labels": ["source", {"weights": np.array([1.0, 2.0])}]}
    base = _record()
    snapshot = SourceContributionData(
        source_values=values,
        group_values=base.group_values,
        total_values=base.total_values,
        positions_m=positions,
        source_metadata=(metadata, None, None),
        targets_m=base.targets_m,
        target_ids=base.target_ids,
        frequencies_hz=base.frequencies_hz,
        unit="m/s^2",
        observable=base.observable,
        frame=base.frame,
        membership=base.membership,
        group_ids=base.group_ids,
    )
    values[:] = -1
    positions[:] = -2
    metadata["labels"][1]["weights"][:] = -3
    assert np.all(snapshot.source_values >= 0)
    np.testing.assert_array_equal(snapshot.positions_m, base.positions_m)
    np.testing.assert_array_equal(
        snapshot.source_metadata[0]["labels"][1]["weights"], [1.0, 2.0]
    )
    with pytest.raises((TypeError, ValueError)):
        snapshot.source_metadata[0]["labels"][1]["weights"][0] = 8.0


def test_deepcopy_reuses_immutable_record_and_preserves_wrapper_behavior():
    record = _record()
    copied_record = deepcopy(record)
    assert copied_record is record
    assert not copied_record.total_values.flags.writeable
    with pytest.raises(ValueError):
        copied_record.total_values.setflags(write=True)

    result = SourceContributionResult(record)
    copied_result = deepcopy(result)
    assert copied_result is not result
    assert copied_result.metadata == result.metadata
    np.testing.assert_array_equal(
        np.asarray(copied_result.source("S1")), np.asarray(result.source("S1"))
    )


def test_global_metadata_is_one_immutable_snapshot_with_original_array_dtypes():
    source_count = 128
    integer_values = np.arange(source_count, dtype=np.int64) + 2**60
    boolean_values = np.array([True, False] * (source_count // 2), dtype=bool)
    string_values = np.array(["left", "right"] * (source_count // 2), dtype="U5")
    structured_dtype = np.dtype([("identifier", np.int64), ("label", "U5")])
    structured_values = np.array(
        [(index, "left") for index in range(source_count)], dtype=structured_dtype
    )
    datetime_values = np.datetime64("2026-11-05") + np.arange(
        source_count, dtype="timedelta64[D]"
    )
    nonfinite_values = np.array([np.nan, np.inf, -np.inf])
    structured_snapshot = structured_values.copy()
    datetime_snapshot = datetime_values.copy()
    nonfinite_snapshot = nonfinite_values.copy()
    input_metadata = {
        "integer_values": integer_values,
        "boolean_values": boolean_values,
        "string_values": string_values,
        "structured_values": structured_values,
        "datetime_values": datetime_values,
        "nonfinite_values": nonfinite_values,
    }
    source_positions = np.column_stack(
        (np.arange(source_count, dtype=np.float64), np.zeros((source_count, 2)))
    )
    record = SourceContributionData(
        source_values=np.zeros((1, 1, source_count), dtype=np.complex128),
        group_values=np.zeros((1, 1, 1), dtype=np.complex128),
        total_values=np.zeros((1, 1), dtype=np.complex128),
        positions_m=source_positions,
        targets_m=[[0.0, 0.0, 1.0]],
        target_ids=("T0",),
        frequencies_hz=[1.0],
        unit="m2/s2",
        observable="potential",
        frame="world",
        membership=np.zeros((1, source_count), dtype=np.int64),
        group_ids=("all",),
        global_metadata=input_metadata,
    )
    integer_values[:] = 0
    boolean_values[:] = ~boolean_values
    string_values[:] = "mutated"
    structured_values["identifier"][:] = -1
    structured_values["label"][:] = "other"
    datetime_values[:] = np.datetime64("2030-01-01")
    nonfinite_values[:] = 0.0

    frozen = record.global_metadata
    assert frozen is not None
    assert frozen["integer_values"].dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(
        frozen["integer_values"], np.arange(source_count, dtype=np.int64) + 2**60
    )
    assert frozen["boolean_values"].dtype == np.dtype(bool)
    np.testing.assert_array_equal(
        frozen["boolean_values"], np.array([True, False] * (source_count // 2))
    )
    assert frozen["string_values"].dtype == np.dtype("U5")
    assert set(frozen["string_values"]) == {"left", "right"}
    assert frozen["structured_values"].dtype == structured_dtype
    np.testing.assert_array_equal(frozen["structured_values"], structured_snapshot)
    assert frozen["datetime_values"].dtype == datetime_values.dtype
    np.testing.assert_array_equal(frozen["datetime_values"], datetime_snapshot)
    np.testing.assert_equal(frozen["nonfinite_values"], nonfinite_snapshot)
    assert frozen["integer_values"].nbytes == source_count * np.dtype(np.int64).itemsize
    assert record.source_metadata is None
    assert not np.shares_memory(
        frozen["integer_values"], input_metadata["integer_values"]
    )
    with pytest.raises(ValueError):
        frozen["integer_values"].setflags(write=True)
    with pytest.raises(ValueError):
        frozen["boolean_values"][0] = False
    with pytest.raises(TypeError):
        frozen["new_key"] = "not allowed"
    with pytest.raises(TypeError, match="Python objects"):
        replace(record, global_metadata={"object_array": np.array([object()])})

    result = SourceContributionResult(record)
    assert result.global_metadata is frozen
    assert result.metadata["global_metadata"] is frozen
    copied = deepcopy(result)
    assert copied.global_metadata is frozen


@pytest.mark.parametrize(
    "unit, error, message",
    [
        (None, TypeError, "unit text"),
        (object(), TypeError, "unit text"),
        ("", ValueError, "unit text"),
        ("   ", ValueError, "unit text"),
    ],
)
def test_numeric_record_requires_nonempty_unit_text(unit, error, message):
    with pytest.raises(error, match=message):
        replace(_record(), unit=unit)


def test_public_wrapper_rejects_unparseable_unit_text_at_construction():
    malformed = replace(_record(), unit="not a valid physical unit")
    with pytest.raises(ValueError, match="unit must be valid unit text"):
        SourceContributionResult(malformed)


def test_result_metadata_preserves_coordinates_phase_and_grouping_immutably():
    result = SourceContributionResult(_record())
    metadata = result.metadata
    assert metadata["positions_m"][1] == (1.0, 0.0, 0.0)
    assert metadata["targets_m"][0] == (2.0, 0.0, 0.0)
    assert metadata["frequencies_hz"] == (1.0, 2.0)
    assert metadata["frame"] == "world"
    assert metadata["t_ref_s"] == 1.5
    assert metadata["phase_origin_m"] == (0.0, 0.0, 0.0)
    assert metadata["fourier_convention"].startswith("Re[q_hat")
    assert metadata["group_ids"] == ("near", "far")
    with pytest.raises(TypeError):
        metadata["frame"] = "changed"


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"frequencies_hz": [1.0, 1.0]}, ValueError, "strictly increasing"),
        ({"frequencies_hz": [1.0, 0.0]}, ValueError, "positive"),
        ({"membership": [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]}, TypeError, "integer"),
        ({"membership": [[0, 0, 2], [1, 0, 1]]}, ValueError, "identify a group"),
        ({"observable": "unknown"}, ValueError, "observable"),
    ],
)
def test_record_rejects_invalid_frequency_membership_and_observable(
    kwargs, error, message
):
    record = _record(lazy=False)
    arguments = {
        field: getattr(record, field)
        for field in (
            "source_values",
            "group_values",
            "total_values",
            "positions_m",
            "targets_m",
            "target_ids",
            "frequencies_hz",
            "unit",
            "observable",
            "frame",
            "membership",
            "group_ids",
        )
    }
    arguments.update(kwargs)
    with pytest.raises(error, match=message):
        SourceContributionData(**arguments)


@pytest.mark.parametrize("layout", ["sample_first", "target_first"])
def test_array_export_is_an_independent_immutable_snapshot(layout):
    record = _record(lazy=True)
    result = SourceContributionResult(record)
    values = result.to_array(kind="source", layout=layout)
    assert values.dtype == np.complex128
    assert not values.flags.writeable
    with pytest.raises(ValueError):
        values.setflags(write=True)


def test_numpy_record_module_has_no_astropy_or_gwexpy_imports():
    path = (
        Path(__file__).parents[2] / "src" / "gragra" / "_plotting" / "_source_data.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(
                alias.name.split(".", maxsplit=1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", maxsplit=1)[0])
    assert "astropy" not in imported
    assert "gwexpy" not in imported
