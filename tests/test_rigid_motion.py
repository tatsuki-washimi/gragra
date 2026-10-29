"""Internal M3-A provider, publication, preflight and memory contracts."""

from __future__ import annotations

import numpy as np
import pytest

import gragra.integrators.rigid_motion as core
from gragra.gravity_sources import PointMassCloud
from gragra.gravity_sources.trajectory import PrescribedRigidTransform
from gragra.integrators.generic import ScalarSourceBlock


class StaticProvider:
    """Instrumented bounded static provider, independent of the M1 adapter."""

    kind = "scalar"
    parameter_count = 1
    replayable = True
    random_access = True
    resident_bytes = 0

    def __init__(self, positions, masses, *, widths=None, alter=None, parts=None):
        self.positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        self.masses = np.asarray(masses, dtype=np.float64).reshape(-1)
        self.source_count = len(self.masses)
        self.widths = widths
        self.alter = alter
        self.parts = parts
        self.calls = 0
        self.completed = 0
        self.requests = []

    def iter_blocks(self, parameter_slice, source_block_size):
        self.calls += 1
        replay = self.calls
        self.requests.append((parameter_slice, source_block_size))
        assert parameter_slice == slice(0, 1)
        width = source_block_size
        if self.widths:
            width = min(width, self.widths[(replay - 1) % len(self.widths)])
        parts = self.parts or [
            (i, min(i + width, self.source_count))
            for i in range(0, self.source_count, width)
        ]
        for start, stop in parts:
            block = ScalarSourceBlock(
                self.positions[start:stop],
                self.masses[start:stop, None],
                start,
                parameter_slice,
            )
            if self.alter:
                block = self.alter(block, replay)
            yield block
        self.completed += 1


def motion(frames=3):
    return PrescribedRigidTransform(
        np.arange(frames, dtype=np.float64),
        np.column_stack((np.arange(frames) * 0.25, np.zeros((frames, 2)))),
        np.repeat(np.eye(3)[None], frames, axis=0),
    )


def provider(**kwargs):
    return StaticProvider(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [-1.0, 0.0, 1.0]], [1.0, 2.0, 3.0], **kwargs
    )


def run(source, *, transform=None, targets=None, **kwargs):
    return core.iter_rigid_motion_gfree_blocks(
        source,
        motion() if transform is None else transform,
        np.array([[4.0, -3.0, 5.0], [5.0, 4.0, -2.0]]) if targets is None else targets,
        observable="acceleration",
        **kwargs,
    )


def collect(blocks, frames=3, targets=2):
    values = np.full((frames, targets, 3), np.nan)
    coverage = np.zeros((frames, targets), dtype=int)
    for block in blocks:
        assert isinstance(block, core.GFreeRigidMotionBlock)
        values[block.motion_slice, block.target_slice] = block.values
        coverage[block.motion_slice, block.target_slice] += 1
    np.testing.assert_array_equal(coverage, 1)
    return values


def test_partition_independent_digest_accepts_alternating_provider_block_widths():
    changing_parts = provider(widths=[1, 3, 2])
    actual = collect(
        run(
            changing_parts,
            source_block_size=3,
            motion_block_size=1,
            target_block_size=1,
            softening_m=0.1,
        )
    )
    # Positive softening changes physics, so compare equal-softening runs.
    expected = collect(run(provider(), source_block_size=1, softening_m=0.1))
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=0)
    assert changing_parts.calls > 1
    assert all(width <= 3 for _, width in changing_parts.requests)


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("kind", "displacement"),
        ("parameter_count", 2),
        ("replayable", False),
        ("random_access", False),
        ("source_count", -1),
        ("resident_bytes", -1),
    ],
)
def test_provider_metadata_is_rejected_at_call_time(attribute, value):
    source = provider()
    setattr(source, attribute, value)
    with pytest.raises((TypeError, ValueError)):
        run(source)
    assert source.calls == 0


@pytest.mark.parametrize(
    "parts", [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 1)], [(1, 3)]]
)
def test_provider_source_coverage_is_checked_before_any_publication(parts):
    stream = run(provider(parts=parts), source_block_size=3, softening_m=0.1)
    with pytest.raises(ValueError):
        next(stream)


def _raw_block(block, **changes):
    fields = {
        "positions_m": block.positions_m,
        "weights_kg": block.weights_kg,
        "source_start": block.source_start,
        "parameter_slice": block.parameter_slice,
    }
    fields.update(changes)
    # Keep the actual contract type while bypassing its constructor guards:
    # providers must be validated even if their block was corrupted afterwards.
    result = object.__new__(ScalarSourceBlock)
    for name, value in fields.items():
        object.__setattr__(result, name, value)
    return result


@pytest.mark.parametrize(
    "damage",
    [
        "complex",
        "negative",
        "nan",
        "float32",
        "writable",
        "wrong_shape",
        "wrong_slice",
        "positions32",
        "infinite_position",
        "oversized",
    ],
)
def test_provider_received_blocks_are_independently_validated(damage):
    def alter(block, _replay):
        weights = block.weights_kg.copy()
        positions = block.positions_m.copy()
        if damage == "complex":
            weights = weights.astype(complex)
        elif damage == "negative":
            weights[0, 0] = -1.0
        elif damage == "nan":
            weights[0, 0] = np.nan
        elif damage == "float32":
            weights = weights.astype(np.float32)
        elif damage == "wrong_shape":
            weights = weights[:, 0]
        elif damage == "wrong_slice":
            return _raw_block(block, parameter_slice=slice(1, 2))
        elif damage == "positions32":
            positions = positions.astype(np.float32)
        elif damage == "infinite_position":
            positions[0, 0] = np.inf
        elif damage == "oversized":
            positions = np.repeat(positions[:1], 4, axis=0)
            weights = np.ones((4, 1))
        positions.setflags(write=False)
        if damage != "writable":
            weights.setflags(write=False)
        return _raw_block(block, weights_kg=weights, positions_m=positions)

    with pytest.raises((TypeError, ValueError)):
        next(run(provider(alter=alter), source_block_size=3, softening_m=0.1))


@pytest.mark.parametrize("damage", ["position", "mass", "signed_zero", "zero_row"])
def test_canonical_digest_detects_content_and_signed_zero_changes(damage):
    def alter(block, replay):
        if replay == 1:
            return block
        positions, weights = block.positions_m.copy(), block.weights_kg.copy()
        if damage in {"position", "zero_row"}:
            positions[0, 0] += 0.25
        elif damage == "mass":
            weights[0, 0] += 1.0
        else:
            positions[0, 1] = -0.0
        return ScalarSourceBlock(
            positions, weights, block.source_start, block.parameter_slice
        )

    masses = [0.0, 2.0, 3.0] if damage == "zero_row" else [1.0, 2.0, 3.0]
    source = StaticProvider(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [-1.0, 0.0, 1.0]], masses, alter=alter
    )
    with pytest.raises(ValueError, match="(?i)replay|digest|chang"):
        next(run(source, softening_m=0.1))


def test_zero_mass_collision_is_excluded_but_positive_mass_contributes():
    source = StaticProvider([[4.0, -3.0, 5.0], [0.0, 0.0, 0.0]], [0.0, 2.0])
    actual = collect(run(source))
    expected = collect(run(PointMassCloud([2.0], [[0.0, 0.0, 0.0]])))
    np.testing.assert_array_equal(actual, expected)
    assert source.completed > 1


@pytest.mark.parametrize(
    "masses,positions", [([], []), ([0.0, 0.0], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])]
)
def test_empty_and_all_zero_provider_returns_exact_zero_and_reads_every_replay(
    masses, positions
):
    source = StaticProvider(positions, masses)
    values = collect(run(source))
    np.testing.assert_array_equal(values, 0.0)
    assert source.calls == source.completed
    assert source.calls >= 1


def test_zero_targets_still_validates_provider_baseline():
    source = provider()
    assert list(run(source, targets=np.empty((0, 3)))) == []
    assert source.completed >= 1
    bad = provider(parts=[(0, 1)])
    with pytest.raises(ValueError):
        list(run(bad, targets=np.empty((0, 3))))


def test_zero_softening_preflights_all_frames_before_first_yield_and_reads_all_blocks():
    # Only the last frame collides. Nothing from earlier frames may escape.
    source = StaticProvider([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], [1.0, 1.0])
    stream = run(source, targets=np.array([[0.5, 0.0, 0.0]]), source_block_size=1)
    assert source.calls == 0
    with pytest.raises(ValueError, match="(?i)colli|coincid"):
        next(stream)
    assert source.calls == source.completed
    assert source.completed >= 1


def test_collision_diagnostic_uses_lexicographic_global_indices():
    source = StaticProvider(
        [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [1.0, 1.0, 1.0]
    )
    with pytest.raises(ValueError) as error:
        next(
            run(
                source,
                transform=motion(1),
                targets=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
                source_block_size=1,
            )
        )
    message = str(error.value).lower()
    import re

    assert re.search(r"frame\D*0", message)
    assert re.search(r"target\D*0", message)
    assert re.search(r"source\D*1", message)
    assert source.calls == source.completed


def test_positive_softening_allows_exact_collision():
    result = collect(
        run(provider(), targets=np.array([[0.0, 0.0, 0.0]]), softening_m=0.1), targets=1
    )
    assert np.isfinite(result).all()


@pytest.mark.parametrize("observable", ["potential", "acceleration", "gradient"])
def test_tiny_noncoincident_underflow_raises_numeric_error_before_publication(
    observable,
):
    source = StaticProvider([[0.0, 0.0, 0.0]], [1.0])
    target = np.array([[1e-200, 0.0, 0.0]])
    assert target[0, 0] != source.positions[0, 0]
    stream = core.iter_rigid_motion_gfree_blocks(
        source,
        motion(1),
        target,
        observable=observable,
        softening_m=0.0,
    )
    with pytest.raises(FloatingPointError) as failure:
        next(stream)
    assert "collision" not in str(failure.value).lower()
    assert "coincid" not in str(failure.value).lower()


@pytest.mark.parametrize("observable", ["potential", "acceleration", "gradient"])
def test_finite_positive_softening_regularizes_tiny_noncoincident_separation(
    observable,
):
    source = StaticProvider([[0.0, 0.0, 0.0]], [1.0])
    stream = core.iter_rigid_motion_gfree_blocks(
        source,
        motion(1),
        [[1e-200, 0.0, 0.0]],
        observable=observable,
        softening_m=0.125,
    )
    blocks = list(stream)
    assert len(blocks) == 1
    actual = blocks[0].values
    assert np.isfinite(actual).all()
    # Here the d^2 term vanishes in binary64, while eps^2 = 1/64 is exact.
    # The tiny acceleration remains representable and must not become zero.
    expected = {
        "potential": np.array([[-8.0]]),
        "acceleration": np.array([[[-512.0 * 1e-200, 0.0, 0.0]]]),
        "gradient": -512.0 * np.eye(3)[None, None],
    }[observable]
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=0)


def test_gradient_denominator_overflow_raises_instead_of_returning_wrong_finite_value():
    source = StaticProvider([[0.0, 0.0, 0.0]], [1.0])
    stream = core.iter_rigid_motion_gfree_blocks(
        source,
        motion(1),
        [[1e70, 0.0, 0.0]],
        observable="gradient",
        softening_m=0.0,
    )
    with pytest.raises(FloatingPointError, match="fifth power"):
        next(stream)


@pytest.mark.parametrize("observable", ["potential", "acceleration", "gradient"])
def test_underflowing_positive_softening_at_exact_match_is_numeric_failure(observable):
    source = StaticProvider([[0.0, 0.0, 0.0]], [1.0])
    stream = core.iter_rigid_motion_gfree_blocks(
        source,
        motion(1),
        [[0.0, 0.0, 0.0]],
        observable=observable,
        softening_m=1e-200,
    )
    with pytest.raises(FloatingPointError) as failure:
        next(stream)
    assert "collision" not in str(failure.value).lower()
    assert "coincid" not in str(failure.value).lower()


def test_late_replay_error_keeps_previous_block_owned_readonly_and_unchanged():
    changed = False

    def alter(block, _replay):
        if changed:
            return ScalarSourceBlock(
                block.positions_m,
                block.weights_kg + 1.0,
                block.source_start,
                block.parameter_slice,
            )
        return block

    source = provider(alter=alter)
    stream = run(source, softening_m=0.1, motion_block_size=1, target_block_size=1)
    first = next(stream)
    before = first.values.copy()
    assert not first.values.flags.writeable
    assert first.values.flags.owndata
    changed = True
    with pytest.raises(ValueError, match="(?i)replay|digest|chang"):
        next(stream)
    np.testing.assert_array_equal(first.values, before)


def test_budget_rejects_before_snapshot_or_source_scan(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("snapshot started before rejecting impossible budget")

    monkeypatch.setattr(core, "_snapshot_inputs", forbidden)
    source = provider()
    with pytest.raises(MemoryError):
        run(source, memory_budget_bytes=1)
    assert calls == []
    assert source.calls == 0


def test_reported_provider_resident_bytes_are_reserved_before_snapshot(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("provider residency was ignored before snapshot")

    monkeypatch.setattr(core, "_snapshot_inputs", forbidden)
    source = provider()
    source.resident_bytes = 2**20
    with pytest.raises(MemoryError):
        run(source, memory_budget_bytes=2**19)
    assert source.calls == 0


def test_zero_mass_rows_are_excluded_from_potentially_overflowing_transforms():
    source = StaticProvider([[1.7e308, 1.7e308, 1.7e308], [0.0, 0.0, 0.0]], [0.0, 1.0])
    # A valid rotation overflows an inactive row; the active row stays finite.
    angle = np.pi / 4
    rotation = np.array(
        [
            [np.cos(angle), np.sin(angle), 0.0],
            [-np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform = PrescribedRigidTransform([0.0], [[0.0, 0.0, 0.0]], [rotation])
    actual = collect(run(source, transform=transform), frames=1)
    expected = collect(
        run(PointMassCloud([1.0], [[0.0, 0.0, 0.0]]), transform=transform), frames=1
    )
    np.testing.assert_array_equal(actual, expected)


def test_planner_minimum_budget_boundary_minus_one_exact_plus_one():
    source = StaticProvider([[0.0, 0.0, 0.0]], [1.0])
    kwargs = {
        "transform": motion(1),
        "targets": np.array([[1.0, 2.0, 3.0]]),
        "motion_block_size": 1,
        "target_block_size": 1,
        "source_block_size": 1,
        "kernel_chunk_size": 1,
        "softening_m": 0.1,
    }

    def accepted(budget):
        try:
            stream = run(source, memory_budget_bytes=budget, **kwargs)
        except MemoryError:
            return False
        stream.close()
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
    assert not accepted(high - 1)
    for budget in (high, high + 1):
        values = list(run(source, memory_budget_bytes=budget, **kwargs))
        assert len(values) == 1
        assert np.isfinite(values[0].values).all()
