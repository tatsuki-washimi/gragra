"""Bounded G-free executor for one prescribed rigid source motion."""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from gragra.gravity_sources import PointMassCloud
from gragra.gravity_sources.trajectory import (
    PrescribedRigidTransform,
    _reject_bool_or_complex,
)
from gragra.integrators.generic import ScalarSourceBlock, SourceProvider
from gragra.targets.points import TargetPoints

_OBSERVABLE_TAILS = {
    "potential": (),
    "acceleration": (3,),
    "gradient": (3, 3),
}
_DIGEST_DOMAIN = b"gragra:M3A:static-mass:v1\0"
_F64_LE = np.dtype("<f8")


@dataclass(frozen=True, slots=True)
class GFreeRigidMotionBlock:
    """One owned, read-only G-free motion/target tile."""

    motion_slice: slice
    target_slice: slice
    values: np.ndarray


@dataclass(frozen=True, slots=True)
class _ProviderMetadata:
    source_count: int
    parameter_count: int
    kind: str
    resident_bytes: int


@dataclass(frozen=True, slots=True)
class _ExecutionState:
    source: PointMassCloud | SourceProvider
    provider_metadata: _ProviderMetadata | None
    transform: PrescribedRigidTransform
    targets: np.ndarray
    observable: str
    softening_m: float
    motion_block_size: int
    target_block_size: int
    source_block_size: int
    kernel_chunk_size: int
    snapshot_bytes: int

    @property
    def source_count(self) -> int:
        if self.provider_metadata is not None:
            return self.provider_metadata.source_count
        return len(self.source.positions_m)

    @property
    def frame_count(self) -> int:
        return len(self.transform.time_s)

    @property
    def target_count(self) -> int:
        return len(self.targets)


def _positive_count(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_index(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a non-negative integer") from error
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _finite_nonnegative_scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        raise ValueError(f"{name} must be a finite non-negative real scalar")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite non-negative real scalar") from error
    if array.ndim != 0 or array.dtype.kind in "bc":
        raise ValueError(f"{name} must be a finite non-negative real scalar")
    try:
        result = float(array)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite non-negative real scalar") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real scalar")
    return result


def _provider_metadata(source: Any) -> _ProviderMetadata:
    if not isinstance(source, SourceProvider):
        raise TypeError("source must be a PointMassCloud or SourceProvider")
    kind = source.kind
    if kind != "scalar":
        raise ValueError("rigid motion requires a scalar source provider")
    if source.replayable is not True:
        raise ValueError("source provider must be replayable")
    if source.random_access is not True:
        raise ValueError("source provider must support random access")
    source_count = _nonnegative_index(source.source_count, "source.source_count")
    parameter_count = _positive_count(source.parameter_count, "source.parameter_count")
    if parameter_count != 1:
        raise ValueError("rigid motion requires parameter_count=1")
    resident_bytes = _nonnegative_index(source.resident_bytes, "source.resident_bytes")
    return _ProviderMetadata(
        source_count=source_count,
        parameter_count=parameter_count,
        kind=kind,
        resident_bytes=resident_bytes,
    )


def _assert_provider_metadata(state: _ExecutionState) -> None:
    if state.provider_metadata is None:
        return
    actual = _provider_metadata(state.source)
    if actual != state.provider_metadata:
        raise ValueError("source provider metadata changed during rigid motion")


def _shape_without_array_copy(value: Any) -> tuple[int, ...]:
    if np.isscalar(value):
        return ()
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        child_shapes = [_shape_without_array_copy(item) for item in value]
        if any(shape != child_shapes[0] for shape in child_shapes[1:]):
            raise ValueError("targets must be a rectangular numeric array")
        return (len(value),) + child_shapes[0]
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            dimensions = tuple(shape)
        except TypeError:
            pass
        else:
            if any(isinstance(item, (bool, np.bool_)) for item in dimensions):
                raise TypeError("target shape dimensions must be integers")
            try:
                normalized = tuple(operator.index(item) for item in dimensions)
            except (TypeError, ValueError):
                pass
            else:
                if any(dimension < 0 for dimension in normalized):
                    raise ValueError(
                        "target shape dimensions must be non-negative integers"
                    )
                return normalized
    raise TypeError("targets must expose non-owning shape metadata")


def _target_metadata(targets: Any) -> int:
    if isinstance(targets, TargetPoints):
        raw = targets.positions_m
    else:
        _reject_bool_or_complex(targets, "targets")
        raw = targets
    shape = _shape_without_array_copy(raw)
    if len(shape) == 1 and shape == (3,):
        return 1
    if len(shape) != 2 or shape[1] != 3:
        raise ValueError(f"targets must have shape (N,3), got {shape}")
    return int(shape[0])


def _target_materialization_metadata(
    targets: Any, target_count: int
) -> tuple[int, np.dtype | None]:
    """Return the dtype and extra bytes needed for duck-array materialization."""
    raw = targets.positions_m if isinstance(targets, TargetPoints) else targets
    if isinstance(raw, (np.ndarray, list, tuple)):
        return 0, None
    dtype_metadata = getattr(raw, "dtype", None)
    if dtype_metadata is None:
        raise TypeError("array-like targets must expose dtype metadata")
    try:
        dtype = np.dtype(dtype_metadata)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "array-like targets must expose valid dtype metadata"
        ) from error
    if dtype.kind == "b":
        raise TypeError("targets must not contain bool")
    if dtype.kind == "c":
        raise TypeError("targets must be real-valued; complex values are rejected")
    if dtype.kind not in "iufSU" or (dtype.kind in "SU" and dtype.itemsize == 0):
        raise TypeError(
            "array-like targets must have a fixed-width real or numeric-string dtype"
        )
    item_count = target_count * 3
    # The existing 2x-float64 reservation covers one materialized float64
    # array plus the owned snapshot. Reserve the excess for wider arraylikes.
    extra_bytes = item_count * max(0, dtype.itemsize - np.dtype(np.float64).itemsize)
    return extra_bytes, dtype


def _snapshot_inputs(
    source: Any,
    transform: Any,
    targets: Any,
    *,
    target_dtype: np.dtype | None = None,
):
    """Copy caller-owned geometry after the pre-copy budget check has passed."""
    if isinstance(source, PointMassCloud):
        source_snapshot = PointMassCloud(
            masses_kg=source.masses_kg,
            positions_m=source.positions_m,
        )
    else:
        source_snapshot = source

    transform_snapshot = PrescribedRigidTransform(
        time_s=transform.time_s,
        translations_m=transform.translations_m,
        rotation_matrices=transform.rotation_matrices,
    )

    if isinstance(targets, TargetPoints):
        raw_targets = targets.positions_m
    else:
        _reject_bool_or_complex(targets, "targets")
        raw_targets = targets
    try:
        if isinstance(raw_targets, (np.ndarray, list, tuple)):
            target_values = raw_targets
        else:
            # Preserve the provider's dtype long enough to reject complex and
            # bool values before float64 conversion can silently discard them.
            declared_dtype = (
                target_dtype
                if target_dtype is not None
                else np.dtype(raw_targets.dtype)
            )
            target_values = np.asarray(raw_targets)
            _reject_bool_or_complex(target_values, "targets")
            if target_values.dtype != declared_dtype:
                raise TypeError(
                    "array-like targets dtype metadata does not match materialized data"
                )
        target_snapshot = np.array(
            target_values, dtype=np.float64, order="C", copy=True
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"targets must be convertible to float64: {error}") from error
    if target_snapshot.ndim == 1 and target_snapshot.shape == (3,):
        target_snapshot = target_snapshot.reshape(1, 3)
    if target_snapshot.ndim != 2 or target_snapshot.shape[1] != 3:
        raise ValueError("targets must have shape (N,3)")
    if not np.isfinite(target_snapshot).all():
        raise ValueError("targets must contain only finite coordinates")
    target_snapshot.setflags(write=False)
    return source_snapshot, transform_snapshot, target_snapshot


def _snapshot_bytes(
    source: PointMassCloud | SourceProvider,
    provider_metadata: _ProviderMetadata | None,
    transform: PrescribedRigidTransform,
    targets: np.ndarray,
) -> int:
    total = int(targets.nbytes)
    total += int(transform.time_s.nbytes)
    total += int(transform.translations_m.nbytes)
    total += int(transform.rotation_matrices.nbytes)
    if provider_metadata is None:
        total += int(source.positions_m.nbytes)
        total += int(source.masses_kg.nbytes)
    return total


def _observable_components(observable: str) -> int:
    tail = _OBSERVABLE_TAILS[observable]
    result = 1
    for size in tail:
        result *= size
    return result


def _working_bytes(
    *,
    motion_size: int,
    target_size: int,
    source_size: int,
    kernel_size: int,
    source_count: int,
    observable: str,
) -> int:
    """Conservative live-array ledger for one scan or one result tile."""
    # Provider blocks contain positions plus one mass column. Reserve two such
    # blocks because an iterator may construct its next block before releasing
    # the previous one. The canonical hash copies and active index map are
    # additional library-owned arrays.
    if source_count:
        received_blocks = 2 * source_size * 4 * 8
        canonical = source_size * 4 * 8
        active_map = source_size * (1 + np.dtype(np.intp).itemsize)
    else:
        received_blocks = canonical = active_map = 0

    pairs = target_size * kernel_size if source_count and target_size else 0
    # 64 float64 slots per target/source pair covers the Newton intermediates,
    # including the 3x3 gradient, plus the collision comparison tile.
    kernel_workspace = pairs * 64 * 8
    # Gathered body positions/masses and one transformed source chunk. Matrix
    # multiplication and translation can briefly keep two coordinate arrays.
    source_chunk_workspace = (7 * kernel_size * 8) if source_count else 0
    components = _observable_components(observable)
    output = motion_size * target_size * components * 8
    # The public streaming wrapper owns the simultaneous SI-valued tile while
    # this executor's G-free tile remains live for conversion.
    converted_output = output
    finite_check = motion_size * target_size * components
    return (
        received_blocks
        + canonical
        + active_map
        + kernel_workspace
        + source_chunk_workspace
        + output
        + converted_output
        + finite_check
        + 256  # digest state and bounded iterator bookkeeping
    )


def _plan_tiles(
    *,
    frame_count: int,
    target_count: int,
    source_count: int,
    observable: str,
    memory_budget_bytes: int,
    reserved_bytes: int,
    snapshot_bytes: int,
    provider_resident_bytes: int,
    requested_motion: int,
    requested_target: int,
    requested_source: int,
    requested_kernel: int,
) -> tuple[int, int, int, int]:
    motion = min(requested_motion, frame_count)
    target = min(requested_target, target_count) if target_count else 0
    source = min(requested_source, source_count) if source_count else 1
    kernel = min(requested_kernel, source)
    available = (
        memory_budget_bytes - reserved_bytes - snapshot_bytes - provider_resident_bytes
    )

    def estimate(m: int, t: int, s: int, k: int) -> int:
        return _working_bytes(
            motion_size=m,
            target_size=t,
            source_size=s,
            kernel_size=k,
            source_count=source_count,
            observable=observable,
        )

    working = estimate(motion, target, source, kernel)
    while working > available:
        candidates: list[tuple[int, int, int, int, int, int]] = []
        # Tie order is target, source, motion, then kernel. Among candidates,
        # shrink the axis with the largest byte reduction.
        if target > 1:
            reduced = target // 2
            candidates.append(
                (
                    working - estimate(motion, reduced, source, kernel),
                    0,
                    motion,
                    reduced,
                    source,
                    kernel,
                )
            )
        if source > 1:
            reduced = source // 2
            candidates.append(
                (
                    working - estimate(motion, target, reduced, min(kernel, reduced)),
                    1,
                    motion,
                    target,
                    reduced,
                    min(kernel, reduced),
                )
            )
        if motion > 1:
            reduced = motion // 2
            candidates.append(
                (
                    working - estimate(reduced, target, source, kernel),
                    2,
                    reduced,
                    target,
                    source,
                    kernel,
                )
            )
        if kernel > 1:
            reduced = kernel // 2
            candidates.append(
                (
                    working - estimate(motion, target, source, reduced),
                    3,
                    motion,
                    target,
                    source,
                    reduced,
                )
            )
        improving = [candidate for candidate in candidates if candidate[0] > 0]
        if not improving:
            raise MemoryError("memory_budget_bytes is below the minimum working set")
        reduction, _, motion, target, source, kernel = max(
            improving, key=lambda candidate: (candidate[0], -candidate[1])
        )
        working -= reduction
    return motion, target, source, kernel


def _precopy_budget_check(
    *,
    memory_budget_bytes: int,
    reserved_bytes: int,
    snapshot_bytes: int,
    provider_resident_bytes: int,
    source_count: int,
    target_count: int,
    observable: str,
    snapshot_peak_extra_bytes: int = 0,
) -> None:
    minimum_working = _working_bytes(
        motion_size=1,
        target_size=1 if target_count else 0,
        source_size=1,
        kernel_size=1,
        source_count=source_count,
        observable=observable,
    )
    # Reserve a second snapshot's worth while defensive copies are formed,
    # covering conversion/copy temporaries conservatively.
    snapshot_peak = 2 * snapshot_bytes + snapshot_peak_extra_bytes
    if (
        reserved_bytes + snapshot_peak + provider_resident_bytes + minimum_working
        > memory_budget_bytes
    ):
        raise MemoryError(
            "memory_budget_bytes is below the minimum snapshot and working set"
        )


def _canonical_update(digest, values: np.ndarray) -> None:
    canonical = np.asarray(values, dtype=_F64_LE, order="C")
    digest.update(memoryview(canonical).cast("B"))


def _root_digest(
    metadata: _ProviderMetadata | None,
    source_count: int,
    positions_digest: bytes,
    masses_digest: bytes,
) -> bytes:
    header = {
        "kind": "scalar",
        "parameter_count": 1,
        "source_count": source_count,
        "positions_dtype": "<f8",
        "positions_shape": [source_count, 3],
        "mass_dtype": "<f8",
        "mass_shape": [source_count, 1],
    }
    if metadata is not None and (
        metadata.kind != "scalar"
        or metadata.parameter_count != 1
        or metadata.source_count != source_count
    ):
        raise ValueError("source provider metadata is inconsistent with digest header")
    encoded = json.dumps(
        header, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    root = hashlib.sha256()
    root.update(_DIGEST_DOMAIN)
    root.update(encoded)
    root.update(positions_digest)
    root.update(masses_digest)
    return root.digest()


def _iter_source_blocks(
    state: _ExecutionState, source_block_size: int
) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    _assert_provider_metadata(state)
    if state.provider_metadata is None:
        source = state.source
        source_count = len(source.positions_m)
        for start in range(0, source_count, source_block_size):
            stop = min(start + source_block_size, source_count)
            yield (
                source.positions_m[start:stop],
                source.masses_kg[start:stop, np.newaxis],
                start,
            )
        return

    expected_start = 0
    parameter_slice = slice(0, 1)
    for block in state.source.iter_blocks(parameter_slice, source_block_size):
        if not isinstance(block, ScalarSourceBlock):
            raise TypeError("source provider must yield scalar source blocks")
        block_parameters = block.parameter_slice
        if (
            block_parameters.start != 0
            or block_parameters.stop != 1
            or block_parameters.step not in (None, 1)
        ):
            raise ValueError("source provider parameter coverage mismatch")
        if block.source_start != expected_start:
            relation = (
                "gap" if block.source_start > expected_start else "overlap or reverse"
            )
            raise ValueError(
                f"source coverage {relation} at global source {expected_start}"
            )
        positions = block.positions_m
        masses = block.weights_kg
        width = len(positions)
        if width == 0 or width > source_block_size:
            raise ValueError("source block exceeds requested source_block_size")
        if expected_start + width > state.source_count:
            raise ValueError("source coverage exceeds source_count")
        if positions.shape != (width, 3) or positions.dtype != np.dtype(np.float64):
            raise TypeError("positions_m must be float64 with shape (Bs,3)")
        if positions.flags.writeable:
            raise ValueError("provider positions_m blocks must be read-only")
        if not np.isfinite(positions).all():
            raise ValueError("positions_m must contain only finite coordinates")
        if masses.shape != (width, 1) or masses.dtype != np.dtype(np.float64):
            raise TypeError("scalar weights_kg must be float64 with shape (Bs,1)")
        if masses.flags.writeable:
            raise ValueError("provider weights_kg blocks must be read-only")
        if not np.isfinite(masses).all() or np.any(masses < 0.0):
            raise ValueError("provider masses must be finite and non-negative")
        yield positions, masses, expected_start
        expected_start += width
    if expected_start != state.source_count:
        raise ValueError(
            f"source coverage missing tail beginning at global source {expected_start}"
        )
    _assert_provider_metadata(state)


def _world_chunk(
    body_positions: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    world = np.empty_like(body_positions)
    with np.errstate(over="ignore", invalid="ignore"):
        np.matmul(body_positions, rotation.T, out=world)
        world += translation
    if not np.isfinite(world).all():
        raise FloatingPointError("transformed source coordinates must be finite")
    return world


def _collision_in_block(
    state: _ExecutionState,
    positions: np.ndarray,
    masses: np.ndarray,
    source_start: int,
    best: tuple[int, int, int] | None,
) -> tuple[int, int, int] | None:
    active = np.flatnonzero(masses > 0.0)
    if not len(active) or not state.target_count:
        return best
    transform = state.transform
    for frame_index in range(state.frame_count):
        rotation = transform.rotation_matrices[frame_index]
        translation = transform.translations_m[frame_index]
        for source_offset in range(0, len(active), state.kernel_chunk_size):
            local_indices = active[
                source_offset : source_offset + state.kernel_chunk_size
            ]
            body_positions = positions[local_indices]
            world_positions = _world_chunk(body_positions, rotation, translation)
            for target_start in range(0, state.target_count, state.target_block_size):
                target_stop = min(
                    target_start + state.target_block_size, state.target_count
                )
                target_block = state.targets[target_start:target_stop]
                matches = np.all(
                    target_block[:, np.newaxis, :] == world_positions[np.newaxis, :, :],
                    axis=2,
                )
                flat_matches = np.flatnonzero(matches)
                if flat_matches.size:
                    target_local, source_local = np.unravel_index(
                        int(flat_matches[0]), matches.shape
                    )
                    candidate = (
                        frame_index,
                        target_start + int(target_local),
                        source_start + int(local_indices[source_local]),
                    )
                    if best is None or candidate < best:
                        best = candidate
    return best


def _scan_source(
    state: _ExecutionState,
    source_block_size: int,
    *,
    collision_preflight: bool,
    consume=None,
) -> tuple[bytes, tuple[int, int, int] | None]:
    positions_digest = hashlib.sha256()
    masses_digest = hashlib.sha256()
    best_collision = None
    for positions, masses_2d, source_start in _iter_source_blocks(
        state, source_block_size
    ):
        masses = masses_2d[:, 0]
        _canonical_update(positions_digest, positions)
        _canonical_update(masses_digest, masses_2d)
        if collision_preflight:
            best_collision = _collision_in_block(
                state, positions, masses, source_start, best_collision
            )
        if consume is not None:
            consume(positions, masses, source_start)
    _assert_provider_metadata(state)
    return (
        _root_digest(
            state.provider_metadata,
            state.source_count,
            positions_digest.digest(),
            masses_digest.digest(),
        ),
        best_collision,
    )


def _preflight_collisions(
    state: _ExecutionState, baseline_digest: bytes
) -> tuple[int, int, int] | None:
    """Run the complete bounded zero-softening collision scan."""
    if state.softening_m != 0.0 or state.target_count == 0:
        return None
    actual_digest, collision = _scan_source(
        state,
        state.source_block_size,
        collision_preflight=True,
    )
    if actual_digest != baseline_digest:
        raise ValueError("source provider changed during collision preflight")
    return collision


def _verify_source_replay(state: _ExecutionState, baseline_digest: bytes) -> None:
    """Exhaust one bounded replay when the result has no target tiles."""
    actual_digest, _ = _scan_source(
        state,
        state.source_block_size,
        collision_preflight=False,
    )
    if actual_digest != baseline_digest:
        raise ValueError("source provider replay differs from the validated digest")


def _accumulate_source_chunk(
    values: np.ndarray,
    frame_offset: int,
    target_block: np.ndarray,
    world_positions: np.ndarray,
    masses: np.ndarray,
    *,
    observable: str,
    softening_m: float,
    frame_index: int,
    target_start: int,
    source_indices: np.ndarray,
) -> None:
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        difference = world_positions[np.newaxis, :, :] - target_block[:, np.newaxis, :]
        squared_distance = np.sum(difference * difference, axis=2)
        if not np.isfinite(squared_distance).all():
            raise FloatingPointError("squared source-target distance is not finite")
        underflow = (squared_distance == 0.0) & np.any(difference != 0.0, axis=2)
        if softening_m == 0.0 and np.any(underflow):
            raise FloatingPointError(
                "squared source-target distance underflowed to zero"
            )
        effective_squared = squared_distance + softening_m * softening_m
        if not np.isfinite(effective_squared).all():
            raise FloatingPointError("effective squared distance is not finite")
        zero = effective_squared == 0.0
        if np.any(zero):
            if softening_m > 0.0:
                raise FloatingPointError(
                    "effective squared distance underflowed to zero"
                )
            target_local, source_local = np.argwhere(zero)[0]
            source_index = int(source_indices[source_local])
            raise ValueError(
                "Collision detected between frame "
                f"{frame_index}, target {target_start + int(target_local)}, source "
                f"{source_index} with zero softening."
            )
        root = np.sqrt(effective_squared)
        if observable == "potential":
            contribution = -1.0 / root
            values[frame_offset] += np.sum(contribution * masses[np.newaxis, :], axis=1)
            return

        distance_cubed = effective_squared * root
        if not np.isfinite(distance_cubed).all() or np.any(distance_cubed == 0.0):
            raise FloatingPointError(
                "source-target distance cubed is non-finite or underflowed to zero"
            )
        if observable == "acceleration":
            contribution = difference / distance_cubed[:, :, np.newaxis]
            values[frame_offset] += np.sum(
                contribution * masses[np.newaxis, :, np.newaxis], axis=1
            )
            return

        distance_fifth = distance_cubed * effective_squared
        if not np.isfinite(distance_fifth).all() or np.any(distance_fifth == 0.0):
            raise FloatingPointError(
                "source-target distance fifth power is invalid: non-finite or zero"
            )
        outer = difference[:, :, :, np.newaxis] * difference[:, :, np.newaxis, :]
        identity = np.eye(3, dtype=np.float64)
        contribution = (
            3.0 * outer / distance_fifth[:, :, np.newaxis, np.newaxis]
            - identity[np.newaxis, np.newaxis, :, :]
            / distance_cubed[:, :, np.newaxis, np.newaxis]
        )
        values[frame_offset] += np.sum(
            contribution * masses[np.newaxis, :, np.newaxis, np.newaxis], axis=1
        )


def _compute_tile(
    state: _ExecutionState,
    motion_slice: slice,
    target_slice: slice,
    baseline_digest: bytes,
) -> np.ndarray:
    tail = _OBSERVABLE_TAILS[state.observable]
    values = np.zeros(
        (motion_slice.stop - motion_slice.start, target_slice.stop - target_slice.start)
        + tail,
        dtype=np.float64,
    )
    target_block = state.targets[target_slice]

    def consume(positions: np.ndarray, masses: np.ndarray, source_start: int) -> None:
        active = np.flatnonzero(masses > 0.0)
        if not len(active):
            return
        for frame_offset, frame_index in enumerate(
            range(motion_slice.start, motion_slice.stop)
        ):
            rotation = state.transform.rotation_matrices[frame_index]
            translation = state.transform.translations_m[frame_index]
            for active_start in range(0, len(active), state.kernel_chunk_size):
                local_indices = active[
                    active_start : active_start + state.kernel_chunk_size
                ]
                body_positions = positions[local_indices]
                source_masses = masses[local_indices]
                world_positions = _world_chunk(body_positions, rotation, translation)
                _accumulate_source_chunk(
                    values,
                    frame_offset,
                    target_block,
                    world_positions,
                    source_masses,
                    observable=state.observable,
                    softening_m=state.softening_m,
                    frame_index=frame_index,
                    target_start=target_slice.start,
                    source_indices=source_start + local_indices,
                )

    actual_digest, _ = _scan_source(
        state,
        state.source_block_size,
        collision_preflight=False,
        consume=consume,
    )
    if actual_digest != baseline_digest:
        raise ValueError("source provider replay differs from the validated digest")
    if not np.isfinite(values).all():
        raise FloatingPointError("rigid motion execution result must be finite")
    values.setflags(write=False)
    return values


def _iterate_blocks(state: _ExecutionState) -> Iterator[GFreeRigidMotionBlock]:
    baseline_digest, _ = _scan_source(
        state,
        state.source_block_size,
        collision_preflight=False,
    )
    collision = _preflight_collisions(state, baseline_digest)
    if collision is not None:
        frame_index, target_index, source_index = collision
        raise ValueError(
            "Collision detected between frame "
            f"{frame_index}, target {target_index}, source {source_index} "
            "with zero softening."
        )
    if state.target_count == 0:
        _verify_source_replay(state, baseline_digest)
        return
    for motion_start in range(0, state.frame_count, state.motion_block_size):
        motion_slice = slice(
            motion_start,
            min(motion_start + state.motion_block_size, state.frame_count),
        )
        for target_start in range(0, state.target_count, state.target_block_size):
            target_slice = slice(
                target_start,
                min(target_start + state.target_block_size, state.target_count),
            )
            values = _compute_tile(state, motion_slice, target_slice, baseline_digest)
            yield GFreeRigidMotionBlock(motion_slice, target_slice, values)
            del values


def iter_rigid_motion_gfree_blocks(
    source,
    transform,
    targets,
    *,
    observable,
    softening_m=0.0,
    memory_budget_bytes=2 * 1024**3,
    motion_block_size=1,
    target_block_size=128,
    source_block_size=8192,
    kernel_chunk_size=1024,
    _reserved_bytes=0,
) -> Iterator[GFreeRigidMotionBlock]:
    """Return bounded, replay-validated G-free tiles for a rigid transform."""
    if observable not in _OBSERVABLE_TAILS:
        raise ValueError(
            "observable must be 'potential', 'acceleration', or 'gradient'"
        )
    softening = _finite_nonnegative_scalar(softening_m, "softening_m")
    budget = _positive_count(memory_budget_bytes, "memory_budget_bytes")
    requested_motion = _positive_count(motion_block_size, "motion_block_size")
    requested_target = _positive_count(target_block_size, "target_block_size")
    requested_source = _positive_count(source_block_size, "source_block_size")
    requested_kernel = _positive_count(kernel_chunk_size, "kernel_chunk_size")
    reserved_bytes = _nonnegative_index(_reserved_bytes, "_reserved_bytes")
    if not isinstance(transform, PrescribedRigidTransform):
        raise TypeError("transform must be a PrescribedRigidTransform")

    if isinstance(source, PointMassCloud):
        source_meta = None
        source_count = len(source.positions_m)
        provider_resident_bytes = 0
    else:
        source_meta = _provider_metadata(source)
        source_count = source_meta.source_count
        provider_resident_bytes = source_meta.resident_bytes

    target_count = _target_metadata(targets)
    snapshot_peak_extra_bytes, target_dtype = _target_materialization_metadata(
        targets, target_count
    )
    frame_count = len(transform.time_s)
    if frame_count <= 0:
        raise ValueError("transform must contain at least one frame")

    estimated_snapshot_bytes = (
        target_count * 3 * 8
        + int(transform.time_s.size) * 8
        + int(transform.translations_m.size) * 8
        + int(transform.rotation_matrices.size) * 8
    )
    if source_meta is None:
        estimated_snapshot_bytes += (
            int(source.positions_m.size) + int(source.masses_kg.size)
        ) * 8

    _precopy_budget_check(
        memory_budget_bytes=budget,
        reserved_bytes=reserved_bytes,
        snapshot_bytes=estimated_snapshot_bytes,
        provider_resident_bytes=provider_resident_bytes,
        source_count=source_count,
        target_count=target_count,
        observable=observable,
        snapshot_peak_extra_bytes=snapshot_peak_extra_bytes,
    )

    source_snapshot, transform_snapshot, target_snapshot = _snapshot_inputs(
        source, transform, targets, target_dtype=target_dtype
    )
    if source_meta is not None:
        source_meta_after = _provider_metadata(source_snapshot)
        if source_meta_after != source_meta:
            raise ValueError("source provider metadata changed during snapshot")
    actual_snapshot_bytes = _snapshot_bytes(
        source_snapshot,
        source_meta,
        transform_snapshot,
        target_snapshot,
    )
    if actual_snapshot_bytes != estimated_snapshot_bytes:
        raise ValueError("snapshot sizes differ from the pre-copy metadata")

    motion_tile, target_tile, source_tile, kernel_tile = _plan_tiles(
        frame_count=len(transform_snapshot.time_s),
        target_count=len(target_snapshot),
        source_count=source_count,
        observable=observable,
        memory_budget_bytes=budget,
        reserved_bytes=reserved_bytes,
        snapshot_bytes=actual_snapshot_bytes,
        provider_resident_bytes=provider_resident_bytes,
        requested_motion=requested_motion,
        requested_target=requested_target,
        requested_source=requested_source,
        requested_kernel=requested_kernel,
    )
    state = _ExecutionState(
        source=source_snapshot,
        provider_metadata=source_meta,
        transform=transform_snapshot,
        targets=target_snapshot,
        observable=observable,
        softening_m=softening,
        motion_block_size=motion_tile,
        target_block_size=target_tile,
        source_block_size=source_tile,
        kernel_chunk_size=kernel_tile,
        snapshot_bytes=actual_snapshot_bytes,
    )
    return _iterate_blocks(state)


__all__ = ["GFreeRigidMotionBlock", "iter_rigid_motion_gfree_blocks"]
