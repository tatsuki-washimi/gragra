"""Public SI wrappers: call timing, ownership, collection and legacy parity."""

from __future__ import annotations

import importlib
import inspect
import weakref

import numpy as np
import pytest

from gragra.constants import G_SI
from gragra.gravity_sources import PointMassCloud
from gragra.gravity_sources.trajectory import PrescribedRigidTransform
from gragra.integrators.generic import ScalarSourceBlock
from gragra.integrators.rigid_motion import iter_rigid_motion_gfree_blocks
from gragra.observables import rigid_motion_observable, stream_rigid_motion_observable
from gragra.observables.trajectory import (
    rigid_transform_acceleration,
    rigid_transform_gravity_gradient,
    rigid_transform_potential,
)

COMPONENTS = {"potential": (), "acceleration": (3,), "gradient": (3, 3)}


def inputs():
    source = PointMassCloud(
        [2.0, 3.0, 4.0], [[-1.0, 0.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, -1.0]]
    )
    transform = PrescribedRigidTransform(
        [0.0, 1.0, 2.0],
        [[0.0, 0.0, 0.0], [0.5, -0.25, 0.75], [1.0, 0.0, 0.0]],
        [np.eye(3), [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], np.eye(3)],
    )
    return source, transform, np.array([[4.0, -3.0, 5.0], [5.0, 2.0, -4.0]])


def assembled(blocks, observable):
    result = np.empty((3, 2) + COMPONENTS[observable])
    coverage = np.zeros((3, 2), dtype=int)
    for block in blocks:
        assert (
            block.values.shape
            == (
                block.motion_slice.stop - block.motion_slice.start,
                block.target_slice.stop - block.target_slice.start,
            )
            + COMPONENTS[observable]
        )
        result[block.motion_slice, block.target_slice] = block.values
        coverage[block.motion_slice, block.target_slice] += 1
    np.testing.assert_array_equal(coverage, 1)
    return result


@pytest.mark.parametrize("observable", COMPONENTS)
def test_public_blocks_are_si_time_first_owned_readonly_and_dense_matches(observable):
    args = inputs()
    options = {
        "observable": observable,
        "motion_block_size": 2,
        "target_block_size": 1,
        "source_block_size": 2,
        "kernel_chunk_size": 1,
    }
    blocks = list(stream_rigid_motion_observable(*args, **options))
    core = list(iter_rigid_motion_gfree_blocks(*args, **options))
    assert len(blocks) == len(core)
    for public, internal in zip(blocks, core, strict=True):
        assert public.motion_slice == internal.motion_slice
        assert public.target_slice == internal.target_slice
        np.testing.assert_array_equal(public.values, G_SI * internal.values)
        assert public.values.dtype == np.float64
        assert public.values.flags.owndata
        assert not public.values.flags.writeable
        assert not np.shares_memory(public.values, internal.values)
        with pytest.raises(ValueError):
            public.values.flat[0] = 0.0
    for index, block in enumerate(blocks):
        for other in blocks[index + 1 :]:
            assert not np.shares_memory(block.values, other.values)
    dense = rigid_motion_observable(*args, **options)
    assert dense.shape == (3, 2) + COMPONENTS[observable]
    assert dense.flags.owndata
    assert not dense.flags.writeable
    np.testing.assert_array_equal(dense, assembled(blocks, observable))


@pytest.mark.parametrize("observable", COMPONENTS)
@pytest.mark.parametrize("tiles", [(1, 1, 1, 1), (2, 2, 2, 2), (8, 8, 8, 8)])
def test_all_four_partition_axes_preserve_observable(observable, tiles):
    args = inputs()
    expected = rigid_motion_observable(*args, observable=observable)
    motion, target, source, kernel = tiles
    actual = assembled(
        stream_rigid_motion_observable(
            *args,
            observable=observable,
            motion_block_size=motion,
            target_block_size=target,
            source_block_size=source,
            kernel_chunk_size=kernel,
        ),
        observable,
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0)


@pytest.mark.parametrize(
    "observable,legacy",
    [
        ("potential", rigid_transform_potential),
        ("acceleration", rigid_transform_acceleration),
        ("gradient", rigid_transform_gravity_gradient),
    ],
)
def test_additive_api_matches_existing_r6_wrapper_without_changing_signature(
    observable, legacy
):
    args = inputs()
    expected = legacy(*args, chunk_size=2, backend="numpy", softening_m=0.125)
    actual = rigid_motion_observable(*args, observable=observable, softening_m=0.125)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0)
    assert inspect.signature(legacy).parameters["backend"].default == "numpy"
    assert "backend" not in inspect.signature(rigid_motion_observable).parameters
    assert "backend" not in inspect.signature(stream_rigid_motion_observable).parameters


@pytest.mark.parametrize(
    "function", [stream_rigid_motion_observable, rigid_motion_observable]
)
@pytest.mark.parametrize(
    "options",
    [
        {"observable": "unknown"},
        {"softening_m": -1.0},
        {"softening_m": np.nan},
        {"softening_m": np.complex128(0.125 + 1j)},
        {"softening_m": np.array(True)},
        {"softening_m": np.array(False)},
        {"motion_block_size": 0},
        {"target_block_size": 0},
        {"source_block_size": 0},
        {"kernel_chunk_size": 0},
        {"memory_budget_bytes": 0},
        {"motion_block_size": True},
    ],
)
def test_option_errors_are_eager_even_with_empty_targets(function, options):
    source, transform, _ = inputs()
    kwargs = {"observable": "acceleration", **options}
    with pytest.raises((TypeError, ValueError, MemoryError)):
        function(source, transform, np.empty((0, 3)), **kwargs)


@pytest.mark.parametrize(
    "bad_targets",
    [np.zeros((2, 2)), np.array([[np.inf, 0.0, 0.0]]), np.zeros((1, 3), dtype=complex)],
)
def test_target_errors_happen_at_stream_call_time(bad_targets):
    source, transform, _ = inputs()
    with pytest.raises((TypeError, ValueError)):
        stream_rigid_motion_observable(
            source, transform, bad_targets, observable="potential"
        )


def test_stream_function_is_eager_but_collision_preflight_is_first_next():
    assert not inspect.isgeneratorfunction(stream_rigid_motion_observable)
    source = PointMassCloud([1.0], [[0.0, 0.0, 0.0]])
    _, transform, _ = inputs()
    stream = stream_rigid_motion_observable(
        source,
        transform,
        [[0.0, 0.0, 0.0]],
        observable="potential",
    )
    with pytest.raises(ValueError, match="(?i)colli|coincid"):
        next(stream)


@pytest.mark.parametrize("observable", COMPONENTS)
def test_zero_target_dense_shape_is_explicit_and_readonly(observable):
    source, transform, _ = inputs()
    targets = np.empty((0, 3))
    assert (
        list(
            stream_rigid_motion_observable(
                source, transform, targets, observable=observable
            )
        )
        == []
    )
    result = rigid_motion_observable(source, transform, targets, observable=observable)
    assert result.shape == (3, 0) + COMPONENTS[observable]
    assert result.dtype == np.float64
    assert not result.flags.writeable


@pytest.mark.parametrize("observable", COMPONENTS)
def test_single_target_vector_matches_one_row_for_dense_and_stream(observable):
    source, transform, targets = inputs()
    vector = targets[0]
    row = targets[:1]
    expected = rigid_motion_observable(source, transform, row, observable=observable)
    actual = rigid_motion_observable(source, transform, vector, observable=observable)
    assert actual.shape == (3, 1) + COMPONENTS[observable]
    np.testing.assert_array_equal(actual, expected)
    vector_blocks = list(
        stream_rigid_motion_observable(
            source,
            transform,
            vector,
            observable=observable,
        )
    )
    row_blocks = list(
        stream_rigid_motion_observable(
            source,
            transform,
            row,
            observable=observable,
        )
    )
    assert len(vector_blocks) == len(row_blocks)
    for vector_block, row_block in zip(vector_blocks, row_blocks, strict=True):
        assert vector_block.motion_slice == row_block.motion_slice
        assert vector_block.target_slice == row_block.target_slice
        assert vector_block.values.shape[1] == 1
        np.testing.assert_array_equal(vector_block.values, row_block.values)


@pytest.mark.parametrize("observable", COMPONENTS)
def test_list_of_numpy_target_rows_matches_dense_and_stream(observable):
    source, transform, targets = inputs()
    target_rows = [targets[0].copy(), targets[1].copy()]
    blocks = list(
        stream_rigid_motion_observable(
            source, transform, target_rows, observable=observable
        )
    )
    expected = np.empty((3, 2) + COMPONENTS[observable])
    for block in blocks:
        expected[block.motion_slice, block.target_slice] = block.values

    actual = rigid_motion_observable(
        source, transform, target_rows, observable=observable
    )
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


def test_call_time_snapshots_isolate_source_transform_and_targets():
    source, transform, targets = inputs()
    expected = rigid_motion_observable(
        source, transform, targets, observable="gradient"
    )
    stream = stream_rigid_motion_observable(
        source, transform, targets, observable="gradient"
    )
    # Owners can reopen write access; execution must own defensive snapshots.
    for value in (
        source.positions_m,
        source.masses_kg,
        transform.rotation_matrices,
        transform.translations_m,
        transform.time_s,
    ):
        value.setflags(write=True)
        value.flat[0] += 0.125
    targets[:] = 99.0
    np.testing.assert_array_equal(assembled(stream, "gradient"), expected)


class LateChangingProvider:
    kind = "scalar"
    parameter_count = 1
    source_count = 1
    replayable = True
    random_access = True
    resident_bytes = 0

    def __init__(self, change_after):
        self.calls = 0
        self.change_after = change_after

    def iter_blocks(self, parameter_slice, source_block_size):
        assert source_block_size >= 1
        self.calls += 1
        mass = 2.0 if self.calls > self.change_after else 1.0
        yield ScalarSourceBlock([[0.0, 0.0, 0.0]], [[mass]], 0, parameter_slice)


@pytest.mark.parametrize("observable", COMPONENTS)
@pytest.mark.parametrize("source_count", [0, 2])
def test_empty_and_all_zero_sources_collect_exact_zeros_for_every_observable(
    observable,
    source_count,
):
    class ZeroProvider(LateChangingProvider):
        def iter_blocks(self, parameter_slice, source_block_size):
            self.calls += 1
            for start in range(0, self.source_count, source_block_size):
                count = min(source_block_size, self.source_count - start)
                yield ScalarSourceBlock(
                    np.zeros((count, 3)), np.zeros((count, 1)), start, parameter_slice
                )

    source = ZeroProvider(float("inf"))
    source.source_count = source_count
    _, transform, _ = inputs()
    targets = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    actual = rigid_motion_observable(source, transform, targets, observable=observable)
    assert actual.shape == (3, 2) + COMPONENTS[observable]
    np.testing.assert_array_equal(actual, 0.0)
    assert source.calls >= 1


def test_dense_raises_on_late_replay_failure_instead_of_returning_partial_array():
    _, transform, targets = inputs()
    probe = LateChangingProvider(float("inf"))
    options = {
        "observable": "potential",
        "softening_m": 0.1,
        "motion_block_size": 1,
        "target_block_size": 1,
    }
    stream = stream_rigid_motion_observable(probe, transform, targets, **options)
    first = next(stream)
    cutoff = probe.calls
    assert np.isfinite(first.values).all()
    stream.close()
    source = LateChangingProvider(cutoff)
    with pytest.raises(ValueError, match="(?i)replay|digest|chang"):
        rigid_motion_observable(source, transform, targets, **options)
    assert source.calls > cutoff


def test_dense_budget_rejects_before_any_snapshot(monkeypatch):
    import gragra.integrators.rigid_motion as core

    snapshots = []

    def forbidden(*args, **kwargs):
        snapshots.append((args, kwargs))
        raise AssertionError("dense budget rejected after snapshot started")

    monkeypatch.setattr(core, "_snapshot_inputs", forbidden)
    with pytest.raises(MemoryError):
        rigid_motion_observable(*inputs(), observable="gradient", memory_budget_bytes=1)
    assert snapshots == []


@pytest.mark.parametrize(
    "function", [stream_rigid_motion_observable, rigid_motion_observable]
)
def test_targets_without_shape_metadata_are_rejected_before_materialization(function):
    class ShapeLessTargets:
        def __init__(self):
            self.array_calls = 0

        def __array__(self, dtype=None, copy=None):
            self.array_calls += 1
            return np.array([[4.0, -3.0, 5.0]], dtype=dtype, copy=copy)

    targets = ShapeLessTargets()
    source, transform, _ = inputs()
    with pytest.raises(TypeError, match="shape metadata"):
        function(
            source,
            transform,
            targets,
            observable="potential",
            memory_budget_bytes=1,
        )
    assert targets.array_calls == 0


@pytest.mark.parametrize(
    "function", [stream_rigid_motion_observable, rigid_motion_observable]
)
def test_negative_target_shape_is_rejected_before_materialization(
    function, monkeypatch
):
    import gragra.integrators.rigid_motion as core

    class NegativeShapeTargets:
        shape = (-1_000_000, 3)

        def __init__(self):
            self.array_calls = 0

        def __array__(self, dtype=None, copy=None):
            self.array_calls += 1
            return np.array([[4.0, -3.0, 5.0]], dtype=dtype, copy=copy)

    targets = NegativeShapeTargets()
    source, transform, _ = inputs()

    def forbidden_snapshot(*_args, **_kwargs):
        raise AssertionError("invalid target shape reached snapshot creation")

    monkeypatch.setattr(core, "_snapshot_inputs", forbidden_snapshot)
    with pytest.raises(
        ValueError, match="target shape dimensions must be non-negative"
    ):
        function(
            source,
            transform,
            targets,
            observable="potential",
            memory_budget_bytes=1,
        )
    assert targets.array_calls == 0


@pytest.mark.parametrize(
    "function", [stream_rigid_motion_observable, rigid_motion_observable]
)
def test_negative_target_shape_is_rejected_for_ndarray_subclasses_before_snapshot(
    function, monkeypatch
):
    import gragra.integrators.rigid_motion as core

    class NegativeShapeArray(np.ndarray):
        @property
        def shape(self):
            return (-1_000_000, 3)

    targets = np.array([[4.0, -3.0, 5.0]]).view(NegativeShapeArray)
    source, transform, _ = inputs()
    snapshots = []

    def forbidden_snapshot(*args, **kwargs):
        snapshots.append((args, kwargs))
        raise AssertionError("invalid target shape reached snapshot creation")

    monkeypatch.setattr(core, "_snapshot_inputs", forbidden_snapshot)
    with pytest.raises(
        ValueError, match="target shape dimensions must be non-negative"
    ):
        function(
            source,
            transform,
            targets,
            observable="potential",
            memory_budget_bytes=1,
        )
    assert snapshots == []


@pytest.mark.parametrize(
    "function", [stream_rigid_motion_observable, rigid_motion_observable]
)
def test_shape_bearing_array_like_without_dtype_is_rejected_before_materialization(
    function,
):
    class UnspecifiedDtypeTargets:
        shape = (1, 3)

        def __init__(self):
            self.array_calls = 0

        def __array__(self, dtype=None, copy=None):
            self.array_calls += 1
            return np.array([[1.0, 2.0, 3.0]], dtype=np.float64, copy=copy)

    targets = UnspecifiedDtypeTargets()
    source, transform, _ = inputs()
    with pytest.raises(TypeError, match="dtype metadata"):
        function(
            source,
            transform,
            targets,
            observable="potential",
            softening_m=0.125,
        )
    assert targets.array_calls == 0


@pytest.mark.parametrize(
    "function", [stream_rigid_motion_observable, rigid_motion_observable]
)
@pytest.mark.parametrize(
    "advertised_dtype,array_calls",
    [(np.dtype(np.complex128), 0), (np.dtype(np.float64), 1)],
)
def test_complex_shape_bearing_array_like_is_rejected_before_float_conversion(
    function, advertised_dtype, array_calls
):
    class ComplexTargets:
        shape = (1, 3)
        dtype = advertised_dtype

        def __init__(self):
            self.array_calls = 0

        def __array__(self, dtype=None, copy=None):
            self.array_calls += 1
            return np.array([[1.0 + 2.0j, 2.0, 3.0]], dtype=complex, copy=copy)

    targets = ComplexTargets()
    source, transform, _ = inputs()
    with pytest.raises(TypeError, match="complex values are rejected"):
        function(
            source,
            transform,
            targets,
            observable="potential",
            softening_m=0.125,
        )
    assert targets.array_calls == array_calls


@pytest.mark.parametrize(
    "function", [stream_rigid_motion_observable, rigid_motion_observable]
)
def test_wide_duck_target_dtype_is_budget_checked_before_materialization(
    function, monkeypatch
):
    import gragra.integrators.rigid_motion as core

    class WideStringTargets:
        shape = (64, 3)
        dtype = np.dtype("U1025")

        def __init__(self):
            self.array_calls = 0

        def __array__(self, dtype=None, copy=None):
            self.array_calls += 1
            rows = [
                [str(index + offset + 100) for offset in range(3)]
                for index in range(64)
            ]
            return np.array(rows, dtype=self.dtype, copy=copy)

    targets = WideStringTargets()
    source, transform, _ = inputs()

    def forbidden_snapshot(*_args, **_kwargs):
        raise AssertionError("wide target materialized before budget rejection")

    monkeypatch.setattr(core, "_snapshot_inputs", forbidden_snapshot)
    with pytest.raises(MemoryError):
        function(
            source,
            transform,
            targets,
            observable="potential",
            memory_budget_bytes=8192,
        )
    assert targets.array_calls == 0


@pytest.mark.parametrize(
    "function", [stream_rigid_motion_observable, rigid_motion_observable]
)
@pytest.mark.parametrize("dtype", [np.dtype("U"), np.dtype("S")])
def test_unsized_string_duck_dtype_is_rejected_before_materialization(function, dtype):
    class UnboundedStringTargets:
        shape = (1, 3)

        def __init__(self):
            self.dtype = dtype
            self.array_calls = 0

        def __array__(self, dtype=None, copy=None):
            self.array_calls += 1
            return np.array([["1", "2", "3"]], dtype=self.dtype, copy=copy)

    targets = UnboundedStringTargets()
    source, transform, _ = inputs()
    with pytest.raises(TypeError, match="fixed-width real or numeric-string dtype"):
        function(
            source,
            transform,
            targets,
            observable="potential",
            softening_m=0.125,
        )
    assert targets.array_calls == 0


def test_wide_duck_target_materialization_bytes_are_bounded(monkeypatch):
    import gragra.integrators.rigid_motion as core

    target_count = 64

    class WideStringTargets:
        shape = (target_count, 3)
        dtype = np.dtype("U1025")

        def __array__(self, dtype=None, copy=None):
            rows = [
                [str(index + offset + 100) for offset in range(3)]
                for index in range(target_count)
            ]
            return np.array(rows, dtype=self.dtype, copy=copy)

    references = []
    maximum_alive = 0
    maximum_live_bytes = 0

    class NumpyTracker:
        def __getattr__(self, name):
            return getattr(np, name)

        def _record(self, result):
            nonlocal maximum_alive, maximum_live_bytes
            if isinstance(result, np.ndarray) and result.shape == (target_count, 3):
                references.append(weakref.ref(result))
                live = [reference() for reference in references]
                live = [value for value in live if value is not None]
                maximum_alive = max(maximum_alive, len(live))
                maximum_live_bytes = max(
                    maximum_live_bytes, sum(value.nbytes for value in live)
                )
            return result

        def asarray(self, *args, **kwargs):
            return self._record(np.asarray(*args, **kwargs))

        def array(self, *args, **kwargs):
            return self._record(np.array(*args, **kwargs))

    monkeypatch.setattr(core, "np", NumpyTracker())
    source, transform, _ = inputs()
    list(
        stream_rigid_motion_observable(
            source,
            transform,
            WideStringTargets(),
            observable="potential",
            memory_budget_bytes=2 * 1024**2,
        )
    )
    assert maximum_alive == 2
    assert maximum_live_bytes <= target_count * 3 * (np.dtype("U1025").itemsize + 8)


@pytest.mark.parametrize("representation", ["integer", "numeric_string"])
def test_list_target_snapshot_stays_within_two_float64_copies(
    monkeypatch, representation
):
    import gragra.integrators.rigid_motion as core

    source, transform, _ = inputs()
    target_count = 64
    if representation == "integer":
        targets = [[index, index + 1, index + 2] for index in range(target_count)]
    else:
        padding = "0" * 1024
        targets = [
            [f"{index}.{padding}", f"{index + 1}.{padding}", f"{index + 2}.{padding}"]
            for index in range(target_count)
        ]
    references = []
    maximum_alive = 0
    maximum_live_bytes = 0

    class NumpyTracker:
        def __getattr__(self, name):
            return getattr(np, name)

        def _record(self, result):
            nonlocal maximum_alive, maximum_live_bytes
            if isinstance(result, np.ndarray) and result.shape == (target_count, 3):
                references.append(weakref.ref(result))
                maximum_alive = max(
                    maximum_alive,
                    sum(reference() is not None for reference in references),
                )
                maximum_live_bytes = max(
                    maximum_live_bytes,
                    sum(
                        reference().nbytes
                        for reference in references
                        if reference() is not None
                    ),
                )
            return result

        def asarray(self, *args, **kwargs):
            return self._record(np.asarray(*args, **kwargs))

        def array(self, *args, **kwargs):
            return self._record(np.array(*args, **kwargs))

    monkeypatch.setattr(core, "np", NumpyTracker())
    snapshots = core._snapshot_inputs(source, transform, targets)
    assert maximum_alive <= 2
    assert maximum_live_bytes <= 2 * snapshots[2].nbytes


def test_dense_output_reservation_rejects_a_budget_that_can_hold_stream_snapshots(
    monkeypatch,
):
    import gragra.integrators.rigid_motion as core

    source = PointMassCloud([1.0], [[0.0, 0.0, 0.0]])
    frames = 128
    transform = PrescribedRigidTransform(
        np.arange(frames),
        np.zeros((frames, 3)),
        np.repeat(np.eye(3)[None], frames, axis=0),
    )
    targets = np.tile([[1.0, 2.0, 3.0]], (128, 1))
    options = {
        "observable": "gradient",
        "memory_budget_bytes": 256 * 1024,
        "motion_block_size": 1,
        "target_block_size": 1,
        "source_block_size": 1,
        "kernel_chunk_size": 1,
    }
    stream = stream_rigid_motion_observable(source, transform, targets, **options)
    stream.close()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dense output reservation occurred after snapshots")

    monkeypatch.setattr(core, "_snapshot_inputs", forbidden)
    with pytest.raises(MemoryError):
        rigid_motion_observable(source, transform, targets, **options)


def test_dense_reservation_plus_minimum_working_set_rejects_before_snapshot(
    monkeypatch,
):
    import gragra.integrators.rigid_motion as core

    source, transform, targets = inputs()
    options = {
        "observable": "gradient",
        "motion_block_size": 1,
        "target_block_size": 1,
        "source_block_size": 1,
        "kernel_chunk_size": 1,
        "softening_m": 0.125,
    }
    cells = len(transform.time_s) * len(targets)
    reserved = cells * (9 * np.dtype(np.float64).itemsize + np.dtype(np.bool_).itemsize)

    def accepted(budget):
        try:
            rigid_motion_observable(
                source, transform, targets, memory_budget_bytes=budget, **options
            )
        except MemoryError:
            return False
        return True

    low, high = 1, 1 << 20
    assert not accepted(low)
    assert accepted(high)
    while low + 1 < high:
        middle = (low + high) // 2
        if accepted(middle):
            high = middle
        else:
            low = middle
    # Reservation alone fits; only its combination with snapshots/minimum
    # working buffers exceeds this exact one-byte-below-boundary budget.
    failing_budget = high - 1
    assert reserved < failing_budget

    def forbidden(*_args, **_kwargs):
        raise AssertionError("combined dense pre-copy minimum checked after snapshots")

    monkeypatch.setattr(core, "_snapshot_inputs", forbidden)
    with pytest.raises(MemoryError):
        rigid_motion_observable(
            source, transform, targets, memory_budget_bytes=failing_budget, **options
        )


def test_dense_collector_releases_completed_tiles_instead_of_retaining_a_list(
    monkeypatch,
):
    module = importlib.import_module("gragra.observables.rigid_motion")
    original = module._stream_rigid_motion
    references = []
    maximum_alive = 0

    def observed(*args, **kwargs):
        nonlocal maximum_alive
        for block in original(*args, **kwargs):
            references.append(weakref.ref(block.values))
            maximum_alive = max(
                maximum_alive, sum(ref() is not None for ref in references)
            )
            assert maximum_alive <= 2
            yield block

    monkeypatch.setattr(module, "_stream_rigid_motion", observed)
    result = module.rigid_motion_observable(
        *inputs(),
        observable="gradient",
        motion_block_size=1,
        target_block_size=1,
    )
    assert len(references) == 6
    assert result.shape == (3, 2, 3, 3)
    assert all(ref() is None for ref in references)
