"""Tests for the internal parameterized source-provider adapter."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from gragra.integrators.generic import (
    DisplacementSourceBlock,
    ScalarSourceBlock,
    SeriesProviderAdapter,
    SourceProvider,
)
from gragra.integrators.streaming import (
    DisplacementSourceSeriesBlock,
    ScalarSourceSeriesBlock,
    SeriesMetadata,
    SeriesProvider,
)


def _metadata(*, quantity="density", axis_kind="time", source_count=5):
    return SeriesMetadata(
        positions_m=np.column_stack(
            (np.arange(1.0, source_count + 1.0), np.zeros((source_count, 2)))
        ),
        source_ids=tuple(f"s{i}" for i in range(source_count)),
        coordinate_frame="ENU",
        coordinate_origin_m=np.zeros(3),
        axis_kind=axis_kind,
        axis_values=np.array([0.0, 0.5, 1.0, 1.5])
        if axis_kind == "time"
        else np.array([1.0, 2.0, 3.0, 4.0]),
        sample_count=4,
        source_count=source_count,
        t_ref_s=0.25 if axis_kind == "frequency" else None,
        phase_origin_m=np.zeros(3) if axis_kind == "frequency" else None,
        quantity=quantity,
        density_kind="delta_rho" if quantity == "density" else None,
    )


def _scalar_provider(meta, values, *, alter=None):
    def factory(parameter_slice, requested_size):
        for start in range(0, meta.source_count, requested_size):
            stop = min(start + requested_size, meta.source_count)
            block_values = values[parameter_slice, start:stop]
            if alter is not None:
                block_values = alter(block_values)
            yield ScalarSourceSeriesBlock(
                meta.positions_m[start:stop],
                parameter_slice,
                start,
                block_values,
                meta.source_ids[start:stop],
            )

    return SeriesProvider(meta, factory)


def _displacement_provider(meta, masses, values):
    def factory(parameter_slice, requested_size):
        for start in range(0, meta.source_count, requested_size):
            stop = min(start + requested_size, meta.source_count)
            yield DisplacementSourceSeriesBlock(
                meta.positions_m[start:stop],
                parameter_slice,
                start,
                masses[start:stop],
                values[parameter_slice, start:stop],
                meta.source_ids[start:stop],
            )

    return SeriesProvider(meta, factory)


def test_scalar_adapter_transposes_source_axis_and_preserves_arbitrary_slice():
    meta = _metadata()
    values = np.arange(20.0).reshape(4, 5)
    provider = _scalar_provider(meta, values)
    provider._set_resident_bytes(1234)
    adapter = SeriesProviderAdapter(provider)

    assert isinstance(adapter, SourceProvider)
    assert adapter.kind == "scalar"
    assert adapter.source_count == 5
    assert adapter.parameter_count == 4
    assert adapter.replayable is True
    assert adapter.random_access is True
    assert adapter.resident_bytes == 1234

    requested = slice(1, 4)
    blocks = list(adapter.iter_blocks(requested, 2))
    assert [type(block) for block in blocks] == [
        ScalarSourceBlock,
        ScalarSourceBlock,
        ScalarSourceBlock,
    ]
    assert [block.source_start for block in blocks] == [0, 2, 4]
    assert [block.parameter_slice for block in blocks] == [requested] * 3
    assert [block.weights_kg.shape for block in blocks] == [(2, 3), (2, 3), (1, 3)]
    for block in blocks:
        start = block.source_start
        stop = start + len(block.positions_m)
        assert np.array_equal(block.weights_kg, values[requested, start:stop].T)
        assert block.weights_kg.dtype == np.float64


def test_displacement_adapter_keeps_mass_and_transposes_parameter_axis():
    meta = _metadata(quantity="displacement", axis_kind="frequency")
    masses = np.arange(1.0, 6.0)
    displacement = (
        np.arange(4 * 5 * 3).reshape(4, 5, 3).astype(np.float64) * (1.0 + 0.5j)
    ).astype(np.complex128)
    adapter = SeriesProviderAdapter(
        _displacement_provider(meta, masses, displacement)
    )

    assert adapter.kind == "displacement"
    blocks = list(adapter.iter_blocks(slice(1, 3), 2))
    assert [type(block) for block in blocks] == [
        DisplacementSourceBlock,
        DisplacementSourceBlock,
        DisplacementSourceBlock,
    ]
    assert [block.source_start for block in blocks] == [0, 2, 4]
    assert [block.displacement_m.shape for block in blocks] == [
        (2, 2, 3),
        (2, 2, 3),
        (1, 2, 3),
    ]
    replay = list(adapter.iter_blocks(slice(1, 3), 2))
    for block, replay_block in zip(blocks, replay, strict=True):
        start = block.source_start
        stop = start + len(block.mass_kg)
        assert block.source_start == replay_block.source_start
        assert block.parameter_slice == replay_block.parameter_slice
        assert np.array_equal(block.mass_kg, masses[start:stop])
        assert np.array_equal(
            block.displacement_m, displacement[1:3, start:stop].transpose(1, 0, 2)
        )
        assert np.array_equal(block.displacement_m, replay_block.displacement_m)
        assert block.displacement_m.dtype == np.complex128
        assert not block.positions_m.flags.writeable
        assert not block.mass_kg.flags.writeable
        assert not block.displacement_m.flags.writeable


def test_adapter_replays_equal_blocks_for_same_request():
    meta = _metadata()
    values = np.arange(20.0).reshape(4, 5)
    adapter = SeriesProviderAdapter(_scalar_provider(meta, values))

    first = list(adapter.iter_blocks(slice(1, 4), 2))
    second = list(adapter.iter_blocks(slice(1, 4), 2))

    assert len(first) == len(second)
    for left, right in zip(first, second, strict=True):
        assert left.source_start == right.source_start
        assert left.parameter_slice == right.parameter_slice
        assert np.array_equal(left.positions_m, right.positions_m)
        assert np.array_equal(left.weights_kg, right.weights_kg)


def test_adapter_rejects_changed_replay_before_iterator_completes():
    meta = _metadata(source_count=1)
    calls = 0

    def changing(values):
        nonlocal calls
        calls += 1
        return values + calls

    adapter = SeriesProviderAdapter(
        _scalar_provider(meta, np.ones((4, 1)), alter=changing)
    )

    blocks = adapter.iter_blocks(slice(0, 1), 1)
    assert next(blocks).weights_kg[0, 0] == 3.0
    assert calls == 2  # complete preflight scan ran before this replay block
    with pytest.raises(ValueError, match="replay"):
        next(blocks)


def test_many_requests_do_not_retain_preflight_digests():
    meta = _metadata(source_count=1)
    provider = _scalar_provider(meta, np.ones((4, 1)))
    adapter = SeriesProviderAdapter(provider)

    tracemalloc.start()
    baseline_bytes, _ = tracemalloc.get_traced_memory()
    for source_block_size in range(1, 1025):
        list(adapter.iter_blocks(slice(0, 1), source_block_size))
    retained_bytes, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert retained_bytes - baseline_bytes < 32 * 1024
    assert adapter.resident_bytes == provider.resident_bytes


@pytest.mark.parametrize(
    "parts,match",
    [
        ([(0, 1), (2, 3)], "gap"),
        ([(0, 2), (1, 4)], "overlap or reverse"),
        ([(0, 1)], "missing tail"),
    ],
)
def test_adapter_rejects_noncontiguous_or_incomplete_source_coverage(parts, match):
    meta = _metadata()

    def broken(parameter_slice, _requested_size):
        for start, stop in parts:
            yield ScalarSourceSeriesBlock(
                meta.positions_m[start:stop],
                parameter_slice,
                start,
                np.ones((parameter_slice.stop - parameter_slice.start, stop - start)),
                meta.source_ids[start:stop],
            )

    with pytest.raises(ValueError, match=match):
        list(
            SeriesProviderAdapter(SeriesProvider(meta, broken)).iter_blocks(
                slice(0, 2), 3
            )
        )


def test_adapter_rejects_source_ids_that_do_not_follow_metadata_order():
    meta = _metadata()

    def reordered(parameter_slice, _requested_size):
        yield ScalarSourceSeriesBlock(
            meta.positions_m[:2],
            parameter_slice,
            0,
            np.ones((parameter_slice.stop - parameter_slice.start, 2)),
            ("s1", "s0"),
        )

    with pytest.raises(ValueError, match="source ID mismatch"):
        list(
            SeriesProviderAdapter(SeriesProvider(meta, reordered)).iter_blocks(
                slice(0, 1), 2
            )
        )


def test_adapter_rejects_wrong_parameter_coverage_and_block_dtype():
    meta = _metadata(axis_kind="frequency")

    def wrong_slice(parameter_slice, _requested_size):
        yield ScalarSourceSeriesBlock(
            meta.positions_m,
            slice(parameter_slice.start + 1, parameter_slice.stop),
            0,
            np.ones(
                (
                    parameter_slice.stop - parameter_slice.start - 1,
                    meta.source_count,
                ),
                dtype=np.complex128,
            ),
            meta.source_ids,
        )

    with pytest.raises(ValueError, match="sample coverage mismatch"):
        list(
            SeriesProviderAdapter(SeriesProvider(meta, wrong_slice)).iter_blocks(
                slice(0, 2), 8
            )
        )

    real_on_frequency_axis = SeriesProviderAdapter(
        _scalar_provider(meta, np.ones((4, 5), dtype=np.float64))
    )
    with pytest.raises(TypeError, match="frequency.*complex128"):
        list(real_on_frequency_axis.iter_blocks(slice(0, 2), 5))


def test_displacement_adapter_rejects_negative_background_mass_and_nonfinite_values():
    meta = _metadata(quantity="displacement")
    displacement = np.zeros((4, 5, 3))
    with pytest.raises(ValueError, match="mass_kg"):
        list(
            SeriesProviderAdapter(
                _displacement_provider(
                    meta, np.array([1.0, -1.0, 1.0, 1.0, 1.0]), displacement
                )
            ).iter_blocks(slice(0, 1), 2)
        )

    displacement[2, 3, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        list(
            SeriesProviderAdapter(
                _displacement_provider(meta, np.ones(5), displacement)
            ).iter_blocks(slice(0, 4), 3)
        )


def test_adapter_outputs_immutable_blocks_without_mutating_inputs():
    meta = _metadata()
    positions = meta.positions_m.copy()
    values = np.arange(20.0).reshape(4, 5)
    values_before = values.copy()
    block = next(
        SeriesProviderAdapter(_scalar_provider(meta, values)).iter_blocks(
            slice(1, 3), 3
        )
    )

    assert np.array_equal(meta.positions_m, positions)
    assert np.array_equal(values, values_before)
    for array in (block.positions_m, block.weights_kg):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 99.0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"parameter_slice": slice(1, 1)}, "parameter_slice"),
        ({"parameter_slice": slice(-1, 2)}, "parameter_slice"),
        ({"parameter_slice": slice(0, 5)}, "parameter_slice"),
        ({"parameter_slice": slice(0, 2, 2)}, "parameter_slice"),
        ({"source_block_size": 0}, "source_block_size"),
    ],
)
def test_adapter_rejects_invalid_requests(kwargs, match):
    meta = _metadata()
    adapter = SeriesProviderAdapter(_scalar_provider(meta, np.ones((4, 5))))
    parameter_slice = kwargs.get("parameter_slice", slice(0, 2))
    source_block_size = kwargs.get("source_block_size", 2)
    with pytest.raises(ValueError, match=match):
        list(adapter.iter_blocks(parameter_slice, source_block_size))

