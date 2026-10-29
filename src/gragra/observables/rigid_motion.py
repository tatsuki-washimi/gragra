"""Streaming SI observables for a source under finite rigid motion."""

from __future__ import annotations

import operator
import sys
from dataclasses import dataclass

import numpy as np

from gragra.constants import G_SI
from gragra.gravity_sources.trajectory import PrescribedRigidTransform
from gragra.integrators.rigid_motion import (
    _finite_nonnegative_scalar,
    _target_metadata,
    iter_rigid_motion_gfree_blocks,
)

_OBSERVABLE_SHAPES = {
    "potential": (),
    "acceleration": (3,),
    "gradient": (3, 3),
}


@dataclass(frozen=True)
class RigidMotionBlock:
    """One immutable SI-valued tile, indexed by motion then target."""

    motion_slice: slice
    target_slice: slice
    values: np.ndarray


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_options(observable, softening_m, memory_budget_bytes, **sizes):
    if observable not in _OBSERVABLE_SHAPES:
        raise ValueError(
            "observable must be 'potential', 'acceleration', or 'gradient'"
        )
    softening = _finite_nonnegative_scalar(softening_m, "softening_m")
    budget = _positive_int("memory_budget_bytes", memory_budget_bytes)
    checked = {name: _positive_int(name, value) for name, value in sizes.items()}
    return softening, budget, checked


def _owned_si_values(values) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    result = np.empty(source.shape, dtype=np.float64, order="C")
    np.multiply(source, G_SI, out=result)
    result.setflags(write=False)
    return result


def _target_count_metadata(targets) -> int:
    """Return target count from shape metadata, without normalizing/copying."""
    return _target_metadata(targets)


def stream_rigid_motion_observable(
    source,
    transform,
    targets,
    *,
    observable: str,
    softening_m: float = 0.0,
    memory_budget_bytes: int = 2 * 1024**3,
    motion_block_size: int = 1,
    target_block_size: int = 128,
    source_block_size: int = 8192,
    kernel_chunk_size: int = 1024,
):
    """Return an iterator of SI blocks for a prescribed rigid transform.

    Option and input metadata validation, defensive snapshots, and the final
    memory plan are eager. Provider scans and zero-softening collision
    preflight run when iteration starts, before any public block can be yielded.
    Values use axes ``(frame, target)`` followed by components and have SI units.
    """
    return _stream_rigid_motion(
        source,
        transform,
        targets,
        observable=observable,
        softening_m=softening_m,
        memory_budget_bytes=memory_budget_bytes,
        motion_block_size=motion_block_size,
        target_block_size=target_block_size,
        source_block_size=source_block_size,
        kernel_chunk_size=kernel_chunk_size,
        reserved_bytes=0,
    )


def _stream_rigid_motion(
    source,
    transform,
    targets,
    *,
    observable: str,
    softening_m: float,
    memory_budget_bytes: int,
    motion_block_size: int,
    target_block_size: int,
    source_block_size: int,
    kernel_chunk_size: int,
    reserved_bytes: int,
):
    softening, budget, sizes = _validate_options(
        observable,
        softening_m,
        memory_budget_bytes,
        motion_block_size=motion_block_size,
        target_block_size=target_block_size,
        source_block_size=source_block_size,
        kernel_chunk_size=kernel_chunk_size,
    )
    # The executor is deliberately a normal function: its call performs input
    # validation, snapshots, and planning; its iterator performs provider scans.
    gfree_blocks = iter_rigid_motion_gfree_blocks(
        source,
        transform,
        targets,
        observable=observable,
        softening_m=softening,
        memory_budget_bytes=budget,
        _reserved_bytes=reserved_bytes,
        **sizes,
    )

    def publish():
        iterator = iter(gfree_blocks)
        while True:
            try:
                block = next(iterator)
            except StopIteration:
                return
            public_block = RigidMotionBlock(
                block.motion_slice,
                block.target_slice,
                _owned_si_values(block.values),
            )
            del block
            yield public_block
            del public_block

    return publish()


def rigid_motion_observable(
    source,
    transform,
    targets,
    *,
    observable: str,
    softening_m: float = 0.0,
    memory_budget_bytes: int = 2 * 1024**3,
    motion_block_size: int = 1,
    target_block_size: int = 128,
    source_block_size: int = 8192,
    kernel_chunk_size: int = 1024,
) -> np.ndarray:
    """Collect all SI tiles into a dense ``(frame, target, ...)`` result."""
    softening, budget, sizes = _validate_options(
        observable,
        softening_m,
        memory_budget_bytes,
        motion_block_size=motion_block_size,
        target_block_size=target_block_size,
        source_block_size=source_block_size,
        kernel_chunk_size=kernel_chunk_size,
    )
    # Read only shape metadata here. The executor performs authoritative type,
    # value, and shape validation before making any snapshots.
    if not isinstance(transform, PrescribedRigidTransform):
        raise TypeError("transform must be a PrescribedRigidTransform")
    frames = len(transform.time_s)
    target_count = _target_count_metadata(targets)
    components = int(np.prod(_OBSERVABLE_SHAPES[observable], dtype=np.int64)) or 1
    element_count = frames * target_count * components
    if element_count > sys.maxsize // np.dtype(np.float64).itemsize:
        raise MemoryError("dense rigid-motion result size overflows addressable memory")
    result_bytes = element_count * np.dtype(np.float64).itemsize
    coverage_count = frames * target_count
    if coverage_count > sys.maxsize:
        raise MemoryError(
            "dense rigid-motion coverage size overflows addressable memory"
        )
    reserved_bytes = result_bytes + coverage_count
    if reserved_bytes > sys.maxsize:
        raise MemoryError("dense rigid-motion reservation overflows addressable memory")
    # The executor sees the original total budget and includes this dense
    # reservation after inspecting all metadata and before taking snapshots.
    stream = _stream_rigid_motion(
        source,
        transform,
        targets,
        observable=observable,
        softening_m=softening,
        memory_budget_bytes=budget,
        **sizes,
        reserved_bytes=reserved_bytes,
    )
    shape = (frames, target_count) + _OBSERVABLE_SHAPES[observable]
    result = np.empty(shape, dtype=np.float64)
    covered = np.zeros((frames, target_count), dtype=np.bool_)
    # Motion-specific collection permits zero targets; it does not use the
    # legacy positive-count collector.
    iterator = iter(stream)
    while True:
        try:
            block = next(iterator)
        except StopIteration:
            break
        values = block.values
        if (
            values.ndim != 2 + len(_OBSERVABLE_SHAPES[observable])
            or values.dtype != np.float64
        ):
            raise ValueError("executor returned values with an invalid shape or dtype")
        ms, ts = block.motion_slice, block.target_slice
        expected = (ms.stop - ms.start, ts.stop - ts.start) + _OBSERVABLE_SHAPES[
            observable
        ]
        if block.values.shape != expected:
            raise ValueError("executor returned a block with inconsistent slice shape")
        if covered[ms, ts].any():
            raise ValueError("executor returned overlapping motion/target blocks")
        result[ms, ts] = block.values
        covered[ms, ts] = True
        del values, block
    if target_count and frames and not covered.all():
        raise ValueError("executor did not cover the complete motion/target result")
    result.setflags(write=False)
    return result


__all__ = [
    "RigidMotionBlock",
    "stream_rigid_motion_observable",
    "rigid_motion_observable",
]
