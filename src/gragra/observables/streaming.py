"""Observable wrappers for bounded source-series integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gragra.constants import G_SI
from gragra.integrators.streaming import (
    _bounded_slice,
    _positive_count,
    iter_g_free_blocks,
)


@dataclass(frozen=True)
class ObservableSeriesBlock:
    """A defensive, read-only ``(sample,target,...)`` result slab."""

    sample_slice: slice
    target_slice: slice
    values: np.ndarray

    def __post_init__(self) -> None:
        sample_slice = _bounded_slice(self.sample_slice, None, "sample_slice")
        target_slice = _bounded_slice(self.target_slice, None, "target_slice")
        raw = np.asarray(self.values)
        if raw.dtype.kind == "b" or raw.dtype.kind not in "iufc":
            raise TypeError("values must be a numeric non-boolean array")
        values = np.array(
            raw,
            dtype=np.complex128 if np.iscomplexobj(raw) else np.float64,
            copy=True,
        )
        expected = (
            sample_slice.stop - sample_slice.start,
            target_slice.stop - target_slice.start,
        )
        if values.ndim < 2 or values.shape[:2] != expected:
            raise ValueError(
                "values leading axes must match sample_slice and target_slice"
            )
        if values.shape[2:] not in {(), (3,), (3, 3)}:
            raise ValueError("values trailing shape must be (), (3,), or (3,3)")
        if not np.isfinite(values).all():
            raise ValueError("values must contain only finite values")
        values.setflags(write=False)
        object.__setattr__(self, "sample_slice", sample_slice)
        object.__setattr__(self, "target_slice", target_slice)
        object.__setattr__(self, "values", values)


def stream_observable(provider: Any, targets: Any, **kwargs):
    """Yield immutable observable blocks, applying ``G_SI`` exactly once."""
    for block in iter_g_free_blocks(provider, targets, **kwargs):
        yield ObservableSeriesBlock(
            block.sample_slice,
            block.target_slice,
            G_SI * block.values,
        )


def collect_observable_blocks(
    blocks,
    *,
    sample_count: int,
    target_count: int,
    axis_kind: str,
) -> np.ndarray:
    """Collect a complete block tiling into the package's public axis order."""
    samples = _positive_count(sample_count, "sample_count")
    targets = _positive_count(target_count, "target_count")
    if axis_kind not in {"time", "frequency"}:
        raise ValueError("axis_kind must be 'time' or 'frequency'")
    iterator = iter(blocks)
    first = next(iterator, None)
    if first is None:
        raise ValueError("cannot collect an empty block stream")
    if not isinstance(first, ObservableSeriesBlock):
        raise TypeError("blocks must contain ObservableSeriesBlock values")
    tail = first.values.shape[2:]
    dtype = first.values.dtype
    result = np.empty((samples, targets) + tail, dtype=dtype)
    coverage = np.zeros((samples, targets), dtype=np.bool_)

    def place(block: ObservableSeriesBlock) -> None:
        if not isinstance(block, ObservableSeriesBlock):
            raise TypeError("blocks must contain ObservableSeriesBlock values")
        sample_slice = _bounded_slice(block.sample_slice, samples, "sample_slice")
        target_slice = _bounded_slice(block.target_slice, targets, "target_slice")
        if block.values.shape[2:] != tail:
            raise ValueError("observable blocks have inconsistent trailing shapes")
        if block.values.dtype != dtype:
            raise TypeError("observable blocks have inconsistent dtypes")
        covered = coverage[sample_slice, target_slice]
        if covered.any():
            raise ValueError("observable block coverage overlaps")
        result[sample_slice, target_slice] = block.values
        covered[...] = True

    place(first)
    for block in iterator:
        place(block)
    if not coverage.all():
        missing = np.argwhere(~coverage)[0]
        raise ValueError(
            "observable block coverage is missing "
            f"sample {int(missing[0])}, target {int(missing[1])}"
        )
    if axis_kind == "frequency":
        result = np.swapaxes(result, 0, 1).copy()
    result.setflags(write=False)
    return result
