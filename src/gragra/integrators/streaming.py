"""Replayable, bounded, G-free integration of fixed-mesh source series."""

from __future__ import annotations

import hashlib
import math
import operator
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from gragra._arrays import as_xyz_table

DEFAULT_SAMPLE_BLOCK_SIZE = 1
DEFAULT_SOURCE_BLOCK_SIZE = 8192
DEFAULT_TARGET_BLOCK_SIZE = 128
DEFAULT_KERNEL_CHUNK_SIZE = 1024
DEFAULT_MEMORY_BUDGET_BYTES = 2 * 1024**3


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


def _finite_scalar(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        suffix = " non-negative" if nonnegative else ""
        raise ValueError(f"{name} must be a finite{suffix} real scalar")
    return result


def _numeric_copy(value: Any, name: str, *, complex_ok: bool = True) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind == "b" or raw.dtype.kind not in "iufc":
        raise TypeError(f"{name} must be a numeric non-boolean array")
    if np.iscomplexobj(raw) and not complex_ok:
        raise TypeError(f"{name} must be real-valued")
    result = np.array(
        raw, dtype=np.complex128 if np.iscomplexobj(raw) else np.float64, copy=True
    )
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _source_ids(value: Any, count: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("source_ids must be an ordered sequence of strings")
    try:
        result = tuple(value)
    except TypeError as error:
        raise TypeError("source_ids must be an ordered sequence of strings") from error
    if len(result) != count or any(
        not isinstance(item, str) or not item for item in result
    ):
        raise ValueError("source_ids must be non-empty strings matching source_count")
    if len(set(result)) != count:
        raise ValueError("source_ids must be ordered and unique")
    return result


def _bounded_slice(value: Any, total: int | None, name: str) -> slice:
    if not isinstance(value, slice) or value.step not in (None, 1):
        raise ValueError(f"{name} must be a forward contiguous slice")
    if value.start is None or value.stop is None:
        raise ValueError(f"{name} requires explicit bounds")
    start = _nonnegative_index(value.start, f"{name}.start")
    stop = _nonnegative_index(value.stop, f"{name}.stop")
    if start >= stop or (total is not None and stop > total):
        raise ValueError(f"{name} is out of range")
    return slice(start, stop)


@dataclass(frozen=True)
class SeriesMetadata:
    """Immutable geometry and sample-axis facts shared by every replay."""

    positions_m: np.ndarray
    source_ids: tuple[str, ...]
    coordinate_frame: str
    coordinate_origin_m: np.ndarray
    axis_kind: str
    axis_values: np.ndarray
    sample_count: int
    source_count: int
    t_ref_s: float | None = None
    phase_origin_m: np.ndarray | None = None
    quantity: str | None = None
    density_kind: str | None = None

    def __post_init__(self) -> None:
        sample_count = _positive_count(self.sample_count, "sample_count")
        source_count = _positive_count(self.source_count, "source_count")
        positions = as_xyz_table("positions_m", self.positions_m)
        if positions.shape != (source_count, 3):
            raise ValueError("positions_m must have shape (source_count,3)")
        axis = _numeric_copy(self.axis_values, "axis_values", complex_ok=False)
        if axis.shape != (sample_count,):
            raise ValueError("axis_values must have sample_count entries")
        if self.axis_kind not in {"time", "frequency"}:
            raise ValueError("axis_kind must be 'time' or 'frequency'")
        if np.any(np.diff(axis) <= 0.0):
            raise ValueError("axis_values must be strictly increasing")
        if self.axis_kind == "frequency" and np.any(axis <= 0.0):
            raise ValueError("frequency axis values must be positive")
        if (
            not isinstance(self.coordinate_frame, str)
            or not self.coordinate_frame.strip()
        ):
            raise ValueError("coordinate_frame must be a non-empty string")
        origin = _numeric_copy(
            self.coordinate_origin_m, "coordinate_origin_m", complex_ok=False
        )
        if origin.shape != (3,):
            raise ValueError("coordinate_origin_m must have shape (3,)")
        phase = None
        t_ref = None
        if self.axis_kind == "frequency":
            if self.t_ref_s is None:
                raise ValueError("frequency inputs require finite t_ref_s")
            t_ref = _finite_scalar(self.t_ref_s, "t_ref_s")
            if self.phase_origin_m is None:
                raise ValueError("frequency inputs require phase_origin_m")
            phase = _numeric_copy(
                self.phase_origin_m, "phase_origin_m", complex_ok=False
            )
            if phase.shape != (3,):
                raise ValueError("phase_origin_m must have shape (3,)")
        elif self.t_ref_s is not None or self.phase_origin_m is not None:
            raise ValueError("time inputs must not declare frequency phase metadata")
        if self.quantity not in {None, "density", "displacement", "surface_density"}:
            raise ValueError(
                "quantity must be density, displacement, or surface_density"
            )
        if self.density_kind not in {None, "rho", "delta_rho"}:
            raise ValueError("density_kind must be rho or delta_rho")
        if self.quantity == "density":
            if self.density_kind is None:
                raise ValueError("density quantity requires density_kind")
            if self.axis_kind == "frequency" and self.density_kind != "delta_rho":
                raise ValueError("frequency density requires density_kind='delta_rho'")
        elif self.density_kind is not None:
            raise ValueError("density_kind is valid only for density quantity")
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "source_count", source_count)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(
            self, "source_ids", _source_ids(self.source_ids, source_count)
        )
        object.__setattr__(self, "axis_values", axis)
        object.__setattr__(self, "coordinate_origin_m", origin)
        object.__setattr__(self, "t_ref_s", t_ref)
        object.__setattr__(self, "phase_origin_m", phase)


@dataclass(frozen=True)
class ScalarSourceSeriesBlock:
    """One sample/source slab of signed scalar mass weights."""

    positions_m: np.ndarray
    sample_slice: slice
    source_start: int
    weights_kg: np.ndarray
    source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        positions = as_xyz_table("positions_m", self.positions_m)
        sample_slice = _bounded_slice(self.sample_slice, None, "sample_slice")
        source_start = _nonnegative_index(self.source_start, "source_start")
        weights = _numeric_copy(self.weights_kg, "weights_kg")
        expected = (sample_slice.stop - sample_slice.start, len(positions))
        if weights.shape != expected:
            raise ValueError(f"weights_kg must have shape (Q,B)={expected}")
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "sample_slice", sample_slice)
        object.__setattr__(self, "source_start", source_start)
        object.__setattr__(self, "weights_kg", weights)
        object.__setattr__(
            self, "source_ids", _source_ids(self.source_ids, len(positions))
        )


@dataclass(frozen=True)
class DisplacementSourceSeriesBlock:
    """One sample/source slab with real mass separate from displacement."""

    positions_m: np.ndarray
    sample_slice: slice
    source_start: int
    mass_kg: np.ndarray
    displacement_m: np.ndarray
    source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        positions = as_xyz_table("positions_m", self.positions_m)
        sample_slice = _bounded_slice(self.sample_slice, None, "sample_slice")
        source_start = _nonnegative_index(self.source_start, "source_start")
        mass = _numeric_copy(self.mass_kg, "mass_kg", complex_ok=False)
        displacement = _numeric_copy(self.displacement_m, "displacement_m")
        if mass.shape != (len(positions),) or np.any(mass < 0.0):
            raise ValueError("mass_kg must be non-negative with shape (B,)")
        expected = (sample_slice.stop - sample_slice.start, len(positions), 3)
        if displacement.shape != expected:
            raise ValueError(f"displacement_m must have shape (Q,B,3)={expected}")
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "sample_slice", sample_slice)
        object.__setattr__(self, "source_start", source_start)
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "displacement_m", displacement)
        object.__setattr__(
            self, "source_ids", _source_ids(self.source_ids, len(positions))
        )


@dataclass(frozen=True)
class InternalGFreeBlock:
    sample_slice: slice
    target_slice: slice
    values: np.ndarray


class SeriesProvider:
    """Replayable factory plus immutable public metadata."""

    __slots__ = ("_metadata", "_factory", "_last_iterator", "_resident_bytes")

    def __init__(
        self, metadata: SeriesMetadata, factory: Callable[[slice, int], Iterator]
    ):
        if not isinstance(metadata, SeriesMetadata):
            raise TypeError("metadata must be SeriesMetadata")
        if isinstance(factory, Iterator) or not callable(factory):
            raise TypeError("factory must be callable and return a fresh iterator")
        self._metadata = metadata
        self._factory = factory
        self._last_iterator: Iterator | None = None
        self._resident_bytes = 0

    @property
    def metadata(self) -> SeriesMetadata:
        return self._metadata

    @property
    def resident_bytes(self) -> int:
        return self._resident_bytes

    def _set_resident_bytes(self, value: int) -> None:
        self._resident_bytes = _nonnegative_index(value, "resident_bytes")

    def __getattr__(self, name: str) -> Any:
        if name in SeriesMetadata.__dataclass_fields__:
            return getattr(self.metadata, name)
        raise AttributeError(name)

    def blocks(self, sample_slice: slice, source_block_size: int) -> Iterator:
        bounded = _bounded_slice(sample_slice, self.sample_count, "sample_slice")
        block_size = _positive_count(source_block_size, "source_block_size")
        result = self._factory(bounded, block_size)
        if not isinstance(result, Iterator):
            raise TypeError(
                "factory must return a fresh iterator, not a materialized iterable"
            )
        if result is self._last_iterator:
            raise TypeError("factory must return a fresh iterator for every replay")
        self._last_iterator = result
        return result


def _metadata_from_series(series: Any, quantity: str) -> SeriesMetadata:
    required = (
        "positions_m",
        "source_ids",
        "coordinate_frame",
        "coordinate_origin_m",
        "axis_kind",
        "axis_values",
        "sample_count",
        "source_count",
    )
    missing = [name for name in required if not hasattr(series, name)]
    if missing:
        raise TypeError(f"series is missing common attributes: {', '.join(missing)}")
    return SeriesMetadata(
        positions_m=series.positions_m,
        source_ids=series.source_ids,
        coordinate_frame=series.coordinate_frame,
        coordinate_origin_m=series.coordinate_origin_m,
        axis_kind=series.axis_kind,
        axis_values=series.axis_values,
        sample_count=series.sample_count,
        source_count=series.source_count,
        t_ref_s=getattr(series, "t_ref_s", None),
        phase_origin_m=getattr(series, "phase_origin_m", None),
        quantity=quantity,
        density_kind=getattr(series, "density_kind", None),
    )


def as_series_provider(
    series_or_metadata: Any, factory: Callable | None = None
) -> SeriesProvider:
    """Adapt accepted fixed-mesh series or explicit metadata/factory."""
    if factory is not None:
        return SeriesProvider(series_or_metadata, factory)
    if isinstance(series_or_metadata, SeriesMetadata):
        raise TypeError("explicit SeriesMetadata requires a callable factory")
    series = series_or_metadata
    if hasattr(series, "displacement_m") and hasattr(series, "mass_kg"):
        metadata = _metadata_from_series(series, "displacement")

        def series_factory(samples: slice, source_block_size: int) -> Iterator:
            for start in range(0, metadata.source_count, source_block_size):
                stop = min(start + source_block_size, metadata.source_count)
                yield DisplacementSourceSeriesBlock(
                    metadata.positions_m[start:stop],
                    samples,
                    start,
                    series.mass_kg[start:stop],
                    series.displacement_m[samples, start:stop],
                    metadata.source_ids[start:stop],
                )

    elif hasattr(series, "surface_density_kg_m2") and hasattr(series, "areas_m2"):
        metadata = _metadata_from_series(series, "surface_density")

        def series_factory(samples: slice, source_block_size: int) -> Iterator:
            for start in range(0, metadata.source_count, source_block_size):
                stop = min(start + source_block_size, metadata.source_count)
                weights = (
                    series.surface_density_kg_m2[samples, start:stop]
                    * series.areas_m2[start:stop]
                )
                yield ScalarSourceSeriesBlock(
                    metadata.positions_m[start:stop],
                    samples,
                    start,
                    weights,
                    metadata.source_ids[start:stop],
                )

    elif hasattr(series, "density_kg_m3") and hasattr(series, "volumes_m3"):
        metadata = _metadata_from_series(series, "density")

        def series_factory(samples: slice, source_block_size: int) -> Iterator:
            for start in range(0, metadata.source_count, source_block_size):
                stop = min(start + source_block_size, metadata.source_count)
                weights = (
                    series.density_kg_m3[samples, start:stop]
                    * series.volumes_m3[start:stop]
                )
                yield ScalarSourceSeriesBlock(
                    metadata.positions_m[start:stop],
                    samples,
                    start,
                    weights,
                    metadata.source_ids[start:stop],
                )

    else:
        raise TypeError("unsupported fixed-mesh series object")
    return SeriesProvider(metadata, series_factory)


def _hash_array(digest: Any, array: np.ndarray) -> None:
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(array).view(np.uint8))


def _validate_scan(
    blocks: Iterator,
    metadata: SeriesMetadata,
    sample_slice: slice,
    source_block_size: int,
    *,
    consume: Callable[[ScalarSourceSeriesBlock | DisplacementSourceSeriesBlock], None]
    | None = None,
) -> tuple[str, int]:
    expected_start = 0
    count = 0
    digest = hashlib.sha256()
    previous_type: type | None = None
    for block in blocks:
        if not isinstance(
            block, (ScalarSourceSeriesBlock, DisplacementSourceSeriesBlock)
        ):
            raise TypeError("factory yielded an unsupported source block")
        if previous_type is not None and type(block) is not previous_type:
            raise TypeError(
                "a provider scan must not mix scalar and displacement blocks"
            )
        previous_type = type(block)
        if block.sample_slice != sample_slice:
            raise ValueError("sample coverage mismatch")
        if block.source_start != expected_start:
            relation = (
                "gap" if block.source_start > expected_start else "overlap or reverse"
            )
            raise ValueError(
                f"source coverage {relation} at global source {expected_start}"
            )
        width = len(block.source_ids)
        if width == 0 or width > source_block_size:
            raise ValueError("source block exceeds requested source_block_size")
        stop = expected_start + width
        if stop > metadata.source_count:
            raise ValueError("source coverage exceeds source_count")
        if block.source_ids != metadata.source_ids[expected_start:stop]:
            raise ValueError(f"source ID mismatch at global source {expected_start}")
        if not np.array_equal(
            block.positions_m, metadata.positions_m[expected_start:stop]
        ):
            raise ValueError(
                f"source position mismatch at global source {expected_start}"
            )
        values = (
            block.weights_kg
            if isinstance(block, ScalarSourceSeriesBlock)
            else block.displacement_m
        )
        if metadata.axis_kind == "time" and np.iscomplexobj(values):
            raise TypeError("time series blocks must be real float64")
        if metadata.axis_kind == "frequency" and values.dtype != np.complex128:
            raise TypeError("frequency series blocks must be complex128")
        if (
            isinstance(block, ScalarSourceSeriesBlock)
            and metadata.quantity == "density"
            and metadata.density_kind == "rho"
            and np.any(block.weights_kg < 0.0)
        ):
            raise ValueError("rho density mass weights must be non-negative")
        digest.update(type(block).__name__.encode("ascii"))
        digest.update(
            f"{sample_slice.start}:{sample_slice.stop}:{expected_start}:{stop}".encode(
                "ascii"
            )
        )
        digest.update("\0".join(block.source_ids).encode("utf-8"))
        _hash_array(digest, block.positions_m)
        if isinstance(block, ScalarSourceSeriesBlock):
            _hash_array(digest, block.weights_kg)
        else:
            _hash_array(digest, block.mass_kg)
            _hash_array(digest, block.displacement_m)
        if consume is not None:
            consume(block)
        expected_start = stop
        count += 1
    if expected_start != metadata.source_count:
        raise ValueError(
            f"source coverage missing tail beginning at global source {expected_start}"
        )
    return digest.hexdigest(), count


def _observable_components(observable: str) -> tuple[int, tuple[int, ...]]:
    if observable == "potential":
        return 1, ()
    if observable == "acceleration":
        return 3, (3,)
    if observable == "gradient":
        return 9, (3, 3)
    raise ValueError("observable must be potential, acceleration, or gradient")


def estimate_streaming_working_bytes(
    *,
    sample_block_size: int,
    source_block_size: int,
    target_block_size: int,
    kernel_chunk_size: int,
    observable: str,
    complex_values: bool,
) -> int:
    """Conservative estimate including geometry, source and kernel temporaries."""
    q = _positive_count(sample_block_size, "sample_block_size")
    b = _positive_count(source_block_size, "source_block_size")
    r = _positive_count(target_block_size, "target_block_size")
    c = min(_positive_count(kernel_chunk_size, "kernel_chunk_size"), b)
    components, _ = _observable_components(observable)
    value_bytes = 16 if complex_values else 8
    output = q * r * components * value_bytes
    geometry = r * 3 * 8 + b * 3 * 8
    source_values = b * 8 + q * b * 3 * value_bytes
    collision = r * b * 4
    tensor_factor = (
        48 if observable == "gradient" else 14 if observable == "acceleration" else 8
    )
    kernel = r * c * tensor_factor * value_bytes
    return int(output + geometry + source_values + collision + kernel + 4096)


def _fit_working_sizes(
    q: int, b: int, r: int, c: int, observable: str, complex_values: bool, budget: int
) -> tuple[int, int, int, int]:
    while (
        estimate_streaming_working_bytes(
            sample_block_size=q,
            source_block_size=b,
            target_block_size=r,
            kernel_chunk_size=c,
            observable=observable,
            complex_values=complex_values,
        )
        > budget
    ):
        size, key = max((r, "r"), (c, "c"), (b, "b"), (q, "q"))
        if size <= 1:
            raise MemoryError("memory_budget_bytes is below the minimum working set")
        if key == "r":
            r = max(1, r // 2)
        elif key == "c":
            c = max(1, c // 2)
        elif key == "b":
            b = max(1, b // 2)
            c = min(c, b)
        else:
            q = max(1, q // 2)
    return q, b, r, c


def _collision_precheck(
    metadata: SeriesMetadata,
    targets: np.ndarray,
    source_block_size: int,
    target_block_size: int,
) -> None:
    for target_start in range(0, len(targets), target_block_size):
        target_stop = min(target_start + target_block_size, len(targets))
        for source_start in range(0, metadata.source_count, source_block_size):
            source_stop = min(source_start + source_block_size, metadata.source_count)
            equal = np.all(
                targets[target_start:target_stop, None, :]
                == metadata.positions_m[None, source_start:source_stop, :],
                axis=2,
            )
            if np.any(equal):
                local_target, local_source = divmod(
                    int(np.argmax(equal)), equal.shape[1]
                )
                target_index = target_start + int(local_target)
                source_index = source_start + int(local_source)
                raise ValueError(
                    "Collision detected at global sample 0, "
                    f"target {target_index}, source {source_index}, "
                    f"source_id {metadata.source_ids[source_index]!r} "
                    "with zero softening"
                )


def _accumulate_block(
    block: ScalarSourceSeriesBlock | DisplacementSourceSeriesBlock,
    *,
    total: np.ndarray,
    target_values: np.ndarray,
    observable: str,
    softening: float,
    kernel_chunk_size: int,
    backend: str,
) -> None:
    from gragra.integrators.generic import accumulate_series_block

    accumulate_series_block(
        block,
        total=total,
        target_values=target_values,
        observable=observable,
        softening=softening,
        kernel_chunk_size=kernel_chunk_size,
        backend=backend,
    )


def iter_g_free_blocks(
    provider: SeriesProvider,
    targets: Any,
    *,
    observable: str,
    sample_block_size: int = DEFAULT_SAMPLE_BLOCK_SIZE,
    source_block_size: int = DEFAULT_SOURCE_BLOCK_SIZE,
    target_block_size: int = DEFAULT_TARGET_BLOCK_SIZE,
    kernel_chunk_size: int = DEFAULT_KERNEL_CHUNK_SIZE,
    backend: str = "numpy",
    softening_m: float = 0.0,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
) -> Iterator[InternalGFreeBlock]:
    """Yield sample-major, target-minor G-free blocks after a full preflight."""
    if not isinstance(provider, SeriesProvider):
        raise TypeError("provider must be a SeriesProvider")
    _observable_components(observable)
    q = _positive_count(sample_block_size, "sample_block_size")
    b = _positive_count(source_block_size, "source_block_size")
    r = _positive_count(target_block_size, "target_block_size")
    c = _positive_count(kernel_chunk_size, "kernel_chunk_size")
    budget = _positive_count(memory_budget_bytes, "memory_budget_bytes")
    softening = _finite_scalar(softening_m, "softening_m", nonnegative=True)
    target_values = as_xyz_table("targets", targets)
    if len(target_values) == 0:
        raise ValueError("targets must not be empty")
    available_budget = budget - provider.resident_bytes
    if available_budget <= 0:
        raise MemoryError("provider resident memory exhausts memory_budget_bytes")
    q, b, r, c = _fit_working_sizes(
        q, b, r, c, observable, provider.axis_kind == "frequency", available_budget
    )
    c = min(c, b)
    if softening == 0.0:
        _collision_precheck(provider.metadata, target_values, b, r)

    expected_scans: dict[tuple[int, int], tuple[str, int]] = {}
    for sample_start in range(0, provider.sample_count, q):
        sample_slice = slice(sample_start, min(sample_start + q, provider.sample_count))
        expected_scans[(sample_slice.start, sample_slice.stop)] = _validate_scan(
            provider.blocks(sample_slice, b), provider.metadata, sample_slice, b
        )

    _, tail = _observable_components(observable)
    dtype = np.complex128 if provider.axis_kind == "frequency" else np.float64
    for sample_start in range(0, provider.sample_count, q):
        sample_slice = slice(sample_start, min(sample_start + q, provider.sample_count))
        for target_start in range(0, len(target_values), r):
            target_slice = slice(
                target_start, min(target_start + r, len(target_values))
            )
            total = np.zeros(
                (
                    sample_slice.stop - sample_slice.start,
                    target_slice.stop - target_slice.start,
                )
                + tail,
                dtype=dtype,
            )

            actual_scan = _validate_scan(
                provider.blocks(sample_slice, b),
                provider.metadata,
                sample_slice,
                b,
                consume=partial(
                    _accumulate_block,
                    total=total,
                    target_values=target_values[target_slice],
                    observable=observable,
                    softening=softening,
                    kernel_chunk_size=c,
                    backend=backend,
                ),
            )
            if actual_scan != expected_scans[(sample_slice.start, sample_slice.stop)]:
                raise ValueError(
                    "provider replay differs from validated preflight scan"
                )
            total.setflags(write=False)
            yield InternalGFreeBlock(sample_slice, target_slice, total)
