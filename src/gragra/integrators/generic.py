"""Internal parameterized source-provider contracts and adapters."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from gragra._arrays import as_xyz_table
from gragra.integrators.streaming import (
    DisplacementSourceSeriesBlock,
    ScalarSourceSeriesBlock,
    SeriesProvider,
    _bounded_slice,
    _finite_scalar,
    _hash_array,
    _nonnegative_index,
    _numeric_copy,
    _observable_components,
    _positive_count,
    _validate_scan,
    estimate_streaming_working_bytes,
)


@dataclass(frozen=True, slots=True)
class ScalarSourceBlock:
    """Immutable source-major scalar weights for one parameter slice."""

    positions_m: np.ndarray
    weights_kg: np.ndarray
    source_start: int
    parameter_slice: slice

    def __post_init__(self) -> None:
        positions = as_xyz_table("positions_m", self.positions_m)
        source_start = _nonnegative_index(self.source_start, "source_start")
        parameter_slice = _bounded_slice(
            self.parameter_slice, None, "parameter_slice"
        )
        weights = _numeric_copy(self.weights_kg, "weights_kg")
        expected = (len(positions), parameter_slice.stop - parameter_slice.start)
        if weights.shape != expected:
            raise ValueError(f"weights_kg must have shape (Bs,Bp)={expected}")
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "weights_kg", weights)
        object.__setattr__(self, "source_start", source_start)
        object.__setattr__(self, "parameter_slice", parameter_slice)

    @classmethod
    def _from_validated_series_block(
        cls, block: ScalarSourceSeriesBlock, parameter_slice: slice
    ) -> ScalarSourceBlock:
        """Reuse immutable series arrays while exposing their transposed view."""
        result = object.__new__(cls)
        object.__setattr__(result, "positions_m", block.positions_m)
        object.__setattr__(result, "weights_kg", block.weights_kg.T)
        object.__setattr__(result, "source_start", block.source_start)
        object.__setattr__(result, "parameter_slice", parameter_slice)
        return result


@dataclass(frozen=True, slots=True)
class DisplacementSourceBlock:
    """Immutable source-major mass and displacement for one parameter slice."""

    positions_m: np.ndarray
    mass_kg: np.ndarray
    displacement_m: np.ndarray
    source_start: int
    parameter_slice: slice

    def __post_init__(self) -> None:
        positions = as_xyz_table("positions_m", self.positions_m)
        source_start = _nonnegative_index(self.source_start, "source_start")
        parameter_slice = _bounded_slice(
            self.parameter_slice, None, "parameter_slice"
        )
        mass = _numeric_copy(self.mass_kg, "mass_kg", complex_ok=False)
        if mass.shape != (len(positions),) or np.any(mass < 0.0):
            raise ValueError("mass_kg must be non-negative with shape (Bs,)")
        displacement = _numeric_copy(self.displacement_m, "displacement_m")
        expected = (
            len(positions),
            parameter_slice.stop - parameter_slice.start,
            3,
        )
        if displacement.shape != expected:
            raise ValueError(
                f"displacement_m must have shape (Bs,Bp,3)={expected}"
            )
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "displacement_m", displacement)
        object.__setattr__(self, "source_start", source_start)
        object.__setattr__(self, "parameter_slice", parameter_slice)

    @classmethod
    def _from_validated_series_block(
        cls, block: DisplacementSourceSeriesBlock, parameter_slice: slice
    ) -> DisplacementSourceBlock:
        """Reuse immutable series arrays and expose source-major displacement."""
        result = object.__new__(cls)
        object.__setattr__(result, "positions_m", block.positions_m)
        object.__setattr__(result, "mass_kg", block.mass_kg)
        object.__setattr__(
            result, "displacement_m", block.displacement_m.transpose(1, 0, 2)
        )
        object.__setattr__(result, "source_start", block.source_start)
        object.__setattr__(result, "parameter_slice", parameter_slice)
        return result


@runtime_checkable
class SourceProvider(Protocol):
    """Repeatable, random-access provider of immutable source-major blocks.

    ``replayable`` is a provider contract across calls. Each adapter iterator
    checks its own complete preflight against the replay it yields; an
    executor combining multiple iterators must retain calculation-wide
    expectations until it can publish the completed result.

    Consumers that publish an accumulated result must keep it private until
    iteration finishes. A replay digest is final only after the iterator is
    exhausted; an adapter may raise at that point if the replay changed.
    """

    @property
    def source_count(self) -> int: ...

    @property
    def parameter_count(self) -> int: ...

    @property
    def kind(self) -> Literal["scalar", "displacement"]: ...

    @property
    def replayable(self) -> bool: ...

    @property
    def random_access(self) -> bool: ...

    @property
    def resident_bytes(self) -> int: ...

    def iter_blocks(
        self, parameter_slice: slice, source_block_size: int
    ) -> Iterator[ScalarSourceBlock | DisplacementSourceBlock]: ...


@dataclass(frozen=True, slots=True)
class SeriesProviderAdapter:
    """Adapt sample-major :class:`SeriesProvider` blocks to source-major data.

    Iteration first validates a complete preflight scan, then yields a bounded
    replay. The caller must not publish a result accumulated from yielded
    blocks until the iterator is exhausted, when its completed digest is
    checked against preflight. The expected digest lives only for that
    iterator; a multi-tile executor owns any calculation-wide digest state.
    """

    provider: SeriesProvider

    def __post_init__(self) -> None:
        if not isinstance(self.provider, SeriesProvider):
            raise TypeError("provider must be a SeriesProvider")
        if self.provider.quantity not in {
            "density",
            "displacement",
            "surface_density",
        }:
            raise ValueError(
                "SeriesProvider quantity must be density, displacement, "
                "or surface_density"
            )

    @property
    def source_count(self) -> int:
        return self.provider.source_count

    @property
    def parameter_count(self) -> int:
        return self.provider.sample_count

    @property
    def kind(self) -> Literal["scalar", "displacement"]:
        return "displacement" if self.provider.quantity == "displacement" else "scalar"

    @property
    def replayable(self) -> bool:
        return True

    @property
    def random_access(self) -> bool:
        return True

    @property
    def resident_bytes(self) -> int:
        return self.provider.resident_bytes

    def iter_blocks(
        self, parameter_slice: slice, source_block_size: int
    ) -> Iterator[ScalarSourceBlock | DisplacementSourceBlock]:
        parameters = _bounded_slice(
            parameter_slice, self.parameter_count, "parameter_slice"
        )
        block_size = _positive_count(source_block_size, "source_block_size")
        return self._iter_validated_blocks(parameters, block_size)

    def _iter_validated_blocks(
        self, parameter_slice: slice, source_block_size: int
    ) -> Iterator[ScalarSourceBlock | DisplacementSourceBlock]:
        expected = _validate_scan(
            self.provider.blocks(parameter_slice, source_block_size),
            self.provider.metadata,
            parameter_slice,
            source_block_size,
        )

        digest = hashlib.sha256()
        expected_start = 0
        scan_count = 0
        previous_type: type | None = None
        for block in self.provider.blocks(parameter_slice, source_block_size):
            self._validate_replay_block(
                block,
                parameter_slice,
                source_block_size,
                expected_start,
                previous_type,
            )
            previous_type = type(block)
            stop = expected_start + len(block.source_ids)
            digest.update(type(block).__name__.encode("ascii"))
            digest.update(
                (
                    f"{parameter_slice.start}:{parameter_slice.stop}:"
                    f"{expected_start}:{stop}"
                ).encode("ascii")
            )
            digest.update("\0".join(block.source_ids).encode("utf-8"))
            _hash_array(digest, block.positions_m)
            if isinstance(block, ScalarSourceSeriesBlock):
                _hash_array(digest, block.weights_kg)
                yield ScalarSourceBlock._from_validated_series_block(
                    block, parameter_slice
                )
            else:
                _hash_array(digest, block.mass_kg)
                _hash_array(digest, block.displacement_m)
                yield DisplacementSourceBlock._from_validated_series_block(
                    block, parameter_slice
                )
            expected_start = stop
            scan_count += 1

        if expected_start != self.source_count:
            raise ValueError(
                "source coverage missing tail beginning at global source "
                f"{expected_start}"
            )
        actual = (digest.hexdigest(), scan_count)
        if actual != expected:
            raise ValueError("provider replay differs from validated preflight scan")

    def _validate_replay_block(
        self,
        block: ScalarSourceSeriesBlock | DisplacementSourceSeriesBlock,
        parameter_slice: slice,
        source_block_size: int,
        expected_start: int,
        previous_type: type | None,
    ) -> None:
        if not isinstance(
            block, (ScalarSourceSeriesBlock, DisplacementSourceSeriesBlock)
        ):
            raise TypeError("factory yielded an unsupported source block")
        if previous_type is not None and type(block) is not previous_type:
            raise TypeError(
                "a provider scan must not mix scalar and displacement blocks"
            )
        if (self.kind == "scalar") != isinstance(block, ScalarSourceSeriesBlock):
            raise TypeError("source block kind does not match provider kind")
        if block.sample_slice != parameter_slice:
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
        if stop > self.source_count:
            raise ValueError("source coverage exceeds source_count")
        metadata = self.provider.metadata
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


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Immutable generic execution tiles and their conservative size estimate."""

    target_block_size: int
    parameter_block_size: int
    source_block_size: int
    kernel_chunk_size: int
    loop_order: tuple[str, str, str]
    strategy: Literal["fused", "blocked"]
    memory_budget_bytes: int
    observable: str
    output_shape: tuple[int, ...]
    result_dtype: np.dtype
    source_kind: Literal["scalar", "displacement"]
    target_count: int
    parameter_count: int
    source_count: int
    provider_resident_bytes: int
    backend_workspace_bytes: int
    strategy_scratch_bytes: int
    estimated_working_bytes: int
    estimated_total_bytes: int


def _nonnegative_bytes(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        result = int(value.__index__())
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a non-negative integer") from error
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _result_dtype(value: object) -> np.dtype:
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as error:
        raise TypeError("result_dtype must be float64 or complex128") from error
    if dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
        raise TypeError("result_dtype must be float64 or complex128")
    return dtype


def estimate_numpy_gradient_workspace_bytes(
    *,
    target_block_size: int,
    kernel_chunk_size: int,
    parameter_block_size: int,
    result_dtype: object,
) -> int:
    """Return a conservative NumPy parametric-gradient scratch bound.

    The parametric gradient kernel can retain at least 61 float64 geometry and
    kernel values per live target/source pair. Reserve 64 values for those
    arrays, then add the parameter-expanded 3x3 contribution and its reduced
    target/parameter result. The bound uses the full requested tiles, so it
    remains conservative if the execution planner later reduces a tile.
    """
    targets = _positive_count(target_block_size, "target_block_size")
    chunk = _positive_count(kernel_chunk_size, "kernel_chunk_size")
    parameters = _positive_count(parameter_block_size, "parameter_block_size")
    dtype = _result_dtype(result_dtype)
    result_itemsize = dtype.itemsize
    source_target_pairs = targets * chunk
    geometry_and_kernel = source_target_pairs * 64 * np.dtype(np.float64).itemsize
    parameterized_contribution = (
        source_target_pairs * 9 * parameters * result_itemsize
    )
    reduced_contribution = targets * parameters * 9 * result_itemsize
    return geometry_and_kernel + parameterized_contribution + reduced_contribution


def estimate_generic_working_bytes(
    *,
    target_count: int,
    target_block_size: int,
    parameter_block_size: int,
    source_block_size: int,
    kernel_chunk_size: int,
    parameter_count: int,
    observable: str,
    result_dtype: object,
    backend_workspace_bytes: int,
    source_kind: Literal["scalar", "displacement"] = "scalar",
    strategy: Literal["fused", "blocked"] = "fused",
) -> int:
    """Estimate one generic executor's bounded working set, excluding resident data.

    The legacy estimate deliberately treats source values as displacement-sized,
    which also bounds scalar blocks. Replay digest records are included because
    the full parameter stream is preflighted before any result is published.
    The blocked strategy includes its source-partial and contraction result
    tiles, a conservative geometry/kernel-tile workspace, and the weighted
    displacement tile when the source kind is displacement.
    Target normalization is whole-table work in ``as_xyz_table``; reserve four
    float64 slots per coordinate for conversion, copying, and the finite-check
    temporary. This also covers the internal real conversion of complex128
    array-like inputs accepted by that helper.
    """
    dtype = _result_dtype(result_dtype)
    targets = _positive_count(target_count, "target_count")
    q = _positive_count(parameter_block_size, "parameter_block_size")
    b = _positive_count(source_block_size, "source_block_size")
    r = _positive_count(target_block_size, "target_block_size")
    parameters = _positive_count(parameter_count, "parameter_count")
    workspace = _nonnegative_bytes(backend_workspace_bytes, "backend_workspace_bytes")
    if strategy not in {"fused", "blocked"}:
        raise ValueError("strategy must be 'fused' or 'blocked'")
    if source_kind not in {"scalar", "displacement"}:
        raise ValueError("source_kind must be 'scalar' or 'displacement'")
    kernel_chunk = min(
        _positive_count(kernel_chunk_size, "kernel_chunk_size"), b
    )
    baseline = estimate_streaming_working_bytes(
        sample_block_size=q,
        source_block_size=b,
        target_block_size=r,
        kernel_chunk_size=kernel_chunk_size,
        observable=observable,
        complex_values=dtype == np.dtype(np.complex128),
    )
    replay_records = (parameters + q - 1) // q
    target_normalization = targets * 3 * 4 * np.dtype(np.float64).itemsize
    strategy_scratch = estimate_generic_strategy_scratch_bytes(
        target_block_size=r,
        parameter_block_size=q,
        source_block_size=b,
        kernel_chunk_size=kernel_chunk,
        observable=observable,
        result_dtype=dtype,
        source_kind=source_kind,
        strategy=strategy,
    )
    return (
        baseline
        + workspace
        + 256 * replay_records
        + target_normalization
        + strategy_scratch
    )


def estimate_generic_strategy_scratch_bytes(
    *,
    target_block_size: int,
    parameter_block_size: int,
    source_block_size: int,
    kernel_chunk_size: int,
    observable: str,
    result_dtype: object,
    source_kind: Literal["scalar", "displacement"],
    strategy: Literal["fused", "blocked"],
) -> int:
    """Estimate strategy-owned scratch beyond the common tile accumulator."""
    if strategy == "fused":
        return 0
    if strategy != "blocked":
        raise ValueError("strategy must be 'fused' or 'blocked'")
    if source_kind not in {"scalar", "displacement"}:
        raise ValueError("source_kind must be 'scalar' or 'displacement'")
    targets = _positive_count(target_block_size, "target_block_size")
    parameters = _positive_count(parameter_block_size, "parameter_block_size")
    sources = _positive_count(source_block_size, "source_block_size")
    chunk = min(_positive_count(kernel_chunk_size, "kernel_chunk_size"), sources)
    dtype = _result_dtype(result_dtype)
    components, _ = _observable_components(observable)
    output_tile_bytes = targets * parameters * components * dtype.itemsize
    pairs = targets * chunk
    # Reserve 64 float64 values per pair for the geometry, kernel tile, and
    # construction temporaries. The largest built-in tile has 27 values per
    # pair (displacement gradient), leaving room for its live intermediates.
    kernel_workspace_bytes = pairs * 64 * np.dtype(np.float64).itemsize
    displacement_tile_bytes = 0
    if source_kind == "displacement":
        displacement_tile_bytes = chunk * parameters * 3 * dtype.itemsize
    return (
        kernel_workspace_bytes
        + 2 * output_tile_bytes
        + displacement_tile_bytes
    )


class ExecutionPlanner:
    """Create validated tile plans against resident plus working memory."""

    @staticmethod
    def plan(
        provider: SourceProvider,
        *,
        target_count: int,
        observable: str,
        result_dtype: object,
        backend_workspace_bytes: int,
        target_block_size: int = 128,
        parameter_block_size: int = 1,
        source_block_size: int = 8192,
        kernel_chunk_size: int = 1024,
        loop_order: tuple[str, str, str] = (
            "target",
            "parameter",
            "source",
        ),
        strategy: Literal["fused", "blocked"] = "fused",
        memory_budget_bytes: int = 2 * 1024**3,
    ) -> ExecutionPlan:
        if not isinstance(provider, SourceProvider):
            raise TypeError("provider must satisfy SourceProvider")
        if provider.kind not in {"scalar", "displacement"}:
            raise ValueError("provider.kind must be 'scalar' or 'displacement'")
        if not provider.replayable:
            raise ValueError("provider must be replayable")
        if not provider.random_access:
            raise ValueError("provider must support random parameter slices")

        targets = _positive_count(target_count, "target_count")
        parameters = _positive_count(provider.parameter_count, "parameter_count")
        sources = _positive_count(provider.source_count, "source_count")
        resident_bytes = _nonnegative_bytes(
            provider.resident_bytes, "provider.resident_bytes"
        )
        budget = _positive_count(memory_budget_bytes, "memory_budget_bytes")
        workspace = _nonnegative_bytes(
            backend_workspace_bytes, "backend_workspace_bytes"
        )
        dtype = _result_dtype(result_dtype)
        _, tail_shape = _observable_components(observable)

        order = tuple(loop_order)
        if order != ("target", "parameter", "source"):
            raise ValueError(
                "loop_order must be ('target', 'parameter', 'source')"
            )
        if strategy not in {"fused", "blocked"}:
            raise ValueError("strategy must be 'fused' or 'blocked'")

        target_tile = min(
            _positive_count(target_block_size, "target_block_size"), targets
        )
        parameter_tile = min(
            _positive_count(parameter_block_size, "parameter_block_size"),
            parameters,
        )
        source_tile = min(
            _positive_count(source_block_size, "source_block_size"), sources
        )
        kernel_tile = min(
            _positive_count(kernel_chunk_size, "kernel_chunk_size"), source_tile
        )
        if resident_bytes >= budget:
            raise MemoryError("provider resident memory exhausts memory_budget_bytes")
        available = budget - resident_bytes

        def estimate(
            target_size: int,
            parameter_size: int,
            source_size: int,
            kernel_size: int,
        ) -> int:
            return estimate_generic_working_bytes(
                target_count=targets,
                target_block_size=target_size,
                parameter_block_size=parameter_size,
                source_block_size=source_size,
                kernel_chunk_size=kernel_size,
                parameter_count=parameters,
                observable=observable,
                result_dtype=dtype,
                backend_workspace_bytes=workspace,
                source_kind=provider.kind,
                strategy=strategy,
            )

        working_bytes = estimate(
            target_tile, parameter_tile, source_tile, kernel_tile
        )
        while working_bytes > available:
            candidates = []
            if target_tile > 1:
                candidates.append(
                    (
                        estimate(
                            target_tile // 2,
                            parameter_tile,
                            source_tile,
                            kernel_tile,
                        ),
                        (target_tile // 2, parameter_tile, source_tile, kernel_tile),
                    )
                )
            if parameter_tile > 1:
                parameter_limit = parameter_tile - 1
                parameter_candidate = None
                parameter_size = 1
                # Within one ceil(P / q) bucket the estimate increases with q,
                # so evaluate only each bucket's left edge. Quotient jumps keep
                # this search O(sqrt(P)) instead of scanning every tile size.
                while parameter_size <= parameter_limit:
                    parameter_estimate = estimate(
                        target_tile,
                        parameter_size,
                        source_tile,
                        kernel_tile,
                    )
                    if (
                        parameter_candidate is None
                        or parameter_estimate < parameter_candidate[0]
                    ):
                        parameter_candidate = (
                            parameter_estimate,
                            (
                                target_tile,
                                parameter_size,
                                source_tile,
                                kernel_tile,
                            ),
                        )

                    digest_blocks = (parameters + parameter_size - 1) // parameter_size
                    if digest_blocks == 1:
                        next_parameter_size = parameter_limit + 1
                    else:
                        last_size = (parameters - 1) // (digest_blocks - 1)
                        next_parameter_size = min(parameter_limit, last_size) + 1
                    parameter_size = max(parameter_size + 1, next_parameter_size)

                if parameter_candidate is not None:
                    candidates.append(parameter_candidate)
            if source_tile > 1:
                reduced_source = source_tile // 2
                candidates.append(
                    (
                        estimate(
                            target_tile,
                            parameter_tile,
                            reduced_source,
                            min(kernel_tile, reduced_source),
                        ),
                        (
                            target_tile,
                            parameter_tile,
                            reduced_source,
                            min(kernel_tile, reduced_source),
                        ),
                    )
                )
            if kernel_tile > 1:
                candidates.append(
                    (
                        estimate(
                            target_tile,
                            parameter_tile,
                            source_tile,
                            kernel_tile // 2,
                        ),
                        (target_tile, parameter_tile, source_tile, kernel_tile // 2),
                    )
                )
            improving = [
                candidate for candidate in candidates if candidate[0] < working_bytes
            ]
            if not improving:
                raise MemoryError(
                    "memory_budget_bytes is below the minimum working set"
                )
            working_bytes, sizes = min(improving, key=lambda candidate: candidate[0])
            target_tile, parameter_tile, source_tile, kernel_tile = sizes

        return ExecutionPlan(
            target_block_size=target_tile,
            parameter_block_size=parameter_tile,
            source_block_size=source_tile,
            kernel_chunk_size=kernel_tile,
            loop_order=order,
            strategy=strategy,
            memory_budget_bytes=budget,
            observable=observable,
            output_shape=(targets, parameters) + tail_shape,
            result_dtype=dtype,
            source_kind=provider.kind,
            target_count=targets,
            parameter_count=parameters,
            source_count=sources,
            provider_resident_bytes=resident_bytes,
            backend_workspace_bytes=workspace,
            strategy_scratch_bytes=estimate_generic_strategy_scratch_bytes(
                target_block_size=target_tile,
                parameter_block_size=parameter_tile,
                source_block_size=source_tile,
                kernel_chunk_size=kernel_tile,
                observable=observable,
                result_dtype=dtype,
                source_kind=provider.kind,
                strategy=strategy,
            ),
            estimated_working_bytes=working_bytes,
            estimated_total_bytes=resident_bytes + working_bytes,
        )


def _scan_generic_provider(
    provider: SourceProvider,
    parameter_slice: slice,
    source_block_size: int,
    *,
    target_values: np.ndarray | None = None,
    target_block_size: int = 1,
    softening_m: float = 0.0,
    consume=None,
) -> tuple[tuple[str, int], str]:
    expected_start = 0
    count = 0
    digest = hashlib.sha256()
    geometry = hashlib.sha256()
    previous_type = None

    for block in provider.iter_blocks(parameter_slice, source_block_size):
        if not isinstance(block, (ScalarSourceBlock, DisplacementSourceBlock)):
            raise TypeError("provider yielded an unsupported source block")
        if previous_type is not None and type(block) is not previous_type:
            raise TypeError(
                "a provider scan must not mix scalar and displacement blocks"
            )
        previous_type = type(block)
        if (provider.kind == "scalar") != isinstance(block, ScalarSourceBlock):
            raise TypeError("source block kind does not match provider kind")
        if block.parameter_slice != parameter_slice:
            raise ValueError("parameter coverage mismatch")
        if block.source_start != expected_start:
            relation = (
                "gap" if block.source_start > expected_start else "overlap or reverse"
            )
            raise ValueError(
                f"source coverage {relation} at global source {expected_start}"
            )

        positions = block.positions_m
        width = len(positions)
        if width == 0 or width > source_block_size:
            raise ValueError("source block exceeds requested source_block_size")
        stop = expected_start + width
        if stop > provider.source_count:
            raise ValueError("source coverage exceeds source_count")
        if positions.shape != (width, 3) or positions.dtype != np.dtype(np.float64):
            raise TypeError("positions_m must be a float64 array with shape (Bs, 3)")
        if not np.isfinite(positions).all():
            raise ValueError("positions_m must contain only finite coordinates")
        if positions.flags.writeable:
            raise ValueError("provider source blocks must be read-only")
        if isinstance(block, ScalarSourceBlock):
            values = block.weights_kg
            if values.shape != (width, parameter_slice.stop - parameter_slice.start):
                raise ValueError("weights_kg must have shape (Bs,Bp)")
            if values.dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
                raise TypeError("weights_kg must have dtype float64 or complex128")
            if values.flags.writeable:
                raise ValueError("provider source blocks must be read-only")
            if not np.isfinite(values).all():
                raise ValueError("weights_kg must contain only finite values")
            if target_values is not None and softening_m == 0.0:
                _check_block_collisions(
                    target_values, positions, expected_start, target_block_size
                )
            digest.update(b"ScalarSourceBlock")
            digest.update(
                f"{parameter_slice.start}:{parameter_slice.stop}:{expected_start}:{stop}".encode(
                    "ascii"
                )
            )
            _hash_array(digest, positions)
            _hash_array(digest, values)
        else:
            mass = block.mass_kg
            values = block.displacement_m
            if mass.shape != (width,) or mass.dtype != np.dtype(np.float64):
                raise TypeError("mass_kg must be float64 with shape (Bs,)")
            if np.any(mass < 0.0) or not np.isfinite(mass).all():
                raise ValueError("mass_kg must be finite and non-negative")
            if values.shape != (
                width,
                parameter_slice.stop - parameter_slice.start,
                3,
            ):
                raise ValueError("displacement_m must have shape (Bs,Bp,3)")
            if values.dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
                raise TypeError(
                    "displacement_m must have dtype float64 or complex128"
                )
            if mass.flags.writeable or values.flags.writeable:
                raise ValueError("provider source blocks must be read-only")
            if not np.isfinite(values).all():
                raise ValueError("displacement_m must contain only finite values")
            if target_values is not None and softening_m == 0.0:
                _check_block_collisions(
                    target_values, positions, expected_start, target_block_size
                )
            digest.update(b"DisplacementSourceBlock")
            digest.update(
                f"{parameter_slice.start}:{parameter_slice.stop}:{expected_start}:{stop}".encode(
                    "ascii"
                )
            )
            _hash_array(digest, positions)
            _hash_array(digest, mass)
            _hash_array(digest, values)

        geometry.update(np.ascontiguousarray(positions).view(np.uint8))
        if consume is not None:
            consume(block)
        expected_start = stop
        count += 1

    if expected_start != provider.source_count:
        raise ValueError(
            "source coverage missing tail beginning at global source "
            f"{expected_start}"
        )
    return (digest.hexdigest(), count), geometry.hexdigest()


def _check_block_collisions(
    targets: np.ndarray,
    sources: np.ndarray,
    source_start: int,
    target_block_size: int,
) -> None:
    for target_start in range(0, len(targets), target_block_size):
        target_stop = min(target_start + target_block_size, len(targets))
        equal = np.all(
            targets[target_start:target_stop, None, :] == sources[None, :, :],
            axis=2,
        )
        if np.any(equal):
            local_target, local_source = divmod(
                int(np.argmax(equal)), equal.shape[1]
            )
            raise ValueError(
                "Collision detected at global target "
                f"{target_start + local_target}, source {source_start + local_source} "
                "with zero softening"
            )


class GenericExecutor:
    """Stream G-free result tiles from a replayable source provider."""

    @staticmethod
    def iter_blocks(
        provider: SourceProvider,
        targets,
        plan: ExecutionPlan,
        backend,
        *,
        softening_m: float = 0.0,
    ) -> Iterator[tuple[slice, slice, np.ndarray]]:
        from gragra.kernels.generic_backend import (
            BlockedContractionBackend,
            ContractionBackend,
        )

        if not isinstance(provider, SourceProvider):
            raise TypeError("provider must satisfy SourceProvider")
        if not isinstance(backend, ContractionBackend):
            raise TypeError("backend must satisfy ContractionBackend")
        if not isinstance(plan, ExecutionPlan):
            raise TypeError("plan must be an ExecutionPlan")
        if plan.strategy == "blocked" and not isinstance(
            backend, BlockedContractionBackend
        ):
            raise TypeError("blocked strategy requires BlockedContractionBackend")
        if not provider.replayable or not provider.random_access:
            raise ValueError("provider must be replayable and support random slices")
        if (
            provider.kind != plan.source_kind
            or provider.source_count != plan.source_count
            or provider.parameter_count != plan.parameter_count
            or provider.resident_bytes != plan.provider_resident_bytes
        ):
            raise ValueError("ExecutionPlan does not match provider metadata")

        target_values = as_xyz_table("targets", targets)
        if len(target_values) != plan.target_count:
            raise ValueError("targets length does not match ExecutionPlan")
        softening = _finite_scalar(softening_m, "softening_m", nonnegative=True)
        _, tail_shape = _observable_components(plan.observable)
        if plan.output_shape != (
            plan.target_count,
            plan.parameter_count,
        ) + tail_shape:
            raise ValueError("ExecutionPlan output shape is inconsistent")

        expected: list[tuple[str, int]] = []
        geometry_digest = None
        for parameter_start in range(
            0, plan.parameter_count, plan.parameter_block_size
        ):
            parameter_slice = slice(
                parameter_start,
                min(
                    parameter_start + plan.parameter_block_size,
                    plan.parameter_count,
                ),
            )
            scan, geometry = _scan_generic_provider(
                provider,
                parameter_slice,
                plan.source_block_size,
                target_values=target_values,
                target_block_size=plan.target_block_size,
                softening_m=softening,
            )
            if geometry_digest is None:
                geometry_digest = geometry
            elif geometry != geometry_digest:
                raise ValueError("source geometry differs between parameter slices")
            expected.append(scan)

        method_names = {
            ("scalar", "potential"): "potential_contract",
            ("scalar", "acceleration"): "acceleration_contract",
            ("scalar", "gradient"): "gradient_contract",
            ("displacement", "potential"): "displacement_potential_contract",
            ("displacement", "acceleration"): "dipole_contract",
            ("displacement", "gradient"): "displacement_gradient_contract",
        }
        method_name = method_names[(plan.source_kind, plan.observable)]
        if plan.strategy == "blocked":
            method_name += "_blocked"

        for target_start in range(0, plan.target_count, plan.target_block_size):
            target_slice = slice(
                target_start,
                min(target_start + plan.target_block_size, plan.target_count),
            )
            for parameter_index, parameter_start in enumerate(
                range(0, plan.parameter_count, plan.parameter_block_size)
            ):
                parameter_slice = slice(
                    parameter_start,
                    min(
                        parameter_start + plan.parameter_block_size,
                        plan.parameter_count,
                    ),
                )
                total = np.zeros(
                    (
                        target_slice.stop - target_slice.start,
                        parameter_slice.stop - parameter_slice.start,
                    )
                    + tail_shape,
                    dtype=plan.result_dtype,
                )
                source_partial = (
                    np.zeros_like(total) if plan.strategy == "blocked" else None
                )

                def accumulate(
                    block,
                    accumulator=total,
                    partial=source_partial,
                    tile_targets=target_values[target_slice],
                ):
                    backend_out = accumulator
                    if partial is not None:
                        # Blocked mode isolates each source block's contribution
                        # before adding it to the private tile total. Reuse this
                        # bounded scratch array and clear it for each block.
                        partial.fill(0.0)
                        backend_out = partial
                    if isinstance(block, ScalarSourceBlock):
                        getattr(backend, method_name)(
                            block.positions_m,
                            tile_targets,
                            block.weights_kg,
                            softening_m=softening,
                            chunk_size=plan.kernel_chunk_size,
                            out=backend_out,
                        )
                    else:
                        getattr(backend, method_name)(
                            block.positions_m,
                            tile_targets,
                            block.mass_kg,
                            block.displacement_m,
                            softening_m=softening,
                            chunk_size=plan.kernel_chunk_size,
                            out=backend_out,
                        )
                    if partial is not None:
                        accumulator += partial

                actual, _ = _scan_generic_provider(
                    provider,
                    parameter_slice,
                    plan.source_block_size,
                    consume=accumulate,
                )
                if actual != expected[parameter_index]:
                    raise ValueError(
                        "provider replay differs from validated preflight scan"
                    )
                if not np.isfinite(total).all():
                    raise ValueError("execution result must contain only finite values")
                total.setflags(write=False)
                yield target_slice, parameter_slice, total
                # Iterators suspend at yield. Drop the prior tile and its
                # closure-held accumulators before the next tile allocation.
                del accumulate, source_partial, total


def _validate_dispatch_output(
    out: np.ndarray,
    sources_m: np.ndarray,
    targets_m: np.ndarray,
    observable: str,
    value_arrays: tuple[np.ndarray, ...],
) -> None:
    if not isinstance(out, np.ndarray):
        raise TypeError("out must be a numpy array")
    targets = np.asarray(targets_m)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("targets_m must have shape (M, 3)")
    if not value_arrays or any(values.ndim < 2 for values in value_arrays):
        raise ValueError("contraction values must have a parameter axis")
    parameter_count = value_arrays[-1].shape[1]
    _, tail_shape = _observable_components(observable)
    expected_shape = (len(targets), parameter_count) + tail_shape
    if out.shape != expected_shape:
        raise ValueError(f"out must have shape {expected_shape}, got {out.shape}")
    if out.dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
        raise TypeError("out must have dtype float64 or complex128")
    requires_complex = any(np.iscomplexobj(values) for values in value_arrays)
    if requires_complex and out.dtype != np.dtype(np.complex128):
        raise TypeError("complex inputs require a complex128 out accumulator")
    if not out.flags.writeable:
        raise ValueError("out must be writeable")
    if not np.isfinite(out).all():
        raise ValueError("out must contain only finite values")
    inputs = (np.asarray(sources_m), targets, *value_arrays)
    if any(np.shares_memory(out, values) for values in inputs):
        raise ValueError("out must not share memory with contraction inputs")


class SeriesDispatchBackend:
    """ContractionBackend adapter that preserves direct-point dispatch sites."""

    def __init__(self, backend: str) -> None:
        self.backend = backend

    def _scalar(
        self,
        observable: str,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        weights_kg: np.ndarray,
        *,
        softening_m: float,
        chunk_size: int | None,
        out: np.ndarray,
    ) -> None:
        from gragra.gravity_sources import WeightedPointSource
        from gragra.integrators.point_direct import direct_point_sum

        _validate_dispatch_output(
            out,
            sources_m,
            targets_m,
            observable,
            (weights_kg,),
        )
        for parameter in range(weights_kg.shape[1]):
            out[:, parameter] += direct_point_sum(
                WeightedPointSource(sources_m, weights_kg[:, parameter]),
                targets_m,
                observable=observable,
                softening_m=softening_m,
                chunk_size=chunk_size,
                backend=self.backend,
            )

    def _displacement(
        self,
        observable: str,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        mass_kg: np.ndarray,
        displacement_m: np.ndarray,
        *,
        softening_m: float,
        chunk_size: int | None,
        out: np.ndarray,
    ) -> None:
        from gragra.integrators.point_direct import direct_point_displacement_sum

        _validate_dispatch_output(
            out,
            sources_m,
            targets_m,
            observable,
            (mass_kg[:, None], displacement_m),
        )
        for parameter in range(displacement_m.shape[1]):
            out[:, parameter] += direct_point_displacement_sum(
                mass_kg,
                displacement_m[:, parameter, :],
                sources_m,
                targets_m,
                observable=observable,
                softening_m=softening_m,
                chunk_size=chunk_size,
                backend=self.backend,
            )

    def potential_contract(
        self,
        sources_m,
        targets_m,
        weights_kg,
        *,
        softening_m=0.0,
        chunk_size=None,
        out,
    ):
        self._scalar(
            "potential",
            sources_m,
            targets_m,
            weights_kg,
            softening_m=softening_m,
            chunk_size=chunk_size,
            out=out,
        )

    def acceleration_contract(
        self,
        sources_m,
        targets_m,
        weights_kg,
        *,
        softening_m=0.0,
        chunk_size=None,
        out,
    ):
        self._scalar(
            "acceleration",
            sources_m,
            targets_m,
            weights_kg,
            softening_m=softening_m,
            chunk_size=chunk_size,
            out=out,
        )

    def gradient_contract(
        self,
        sources_m,
        targets_m,
        weights_kg,
        *,
        softening_m=0.0,
        chunk_size=None,
        out,
    ):
        self._scalar(
            "gradient",
            sources_m,
            targets_m,
            weights_kg,
            softening_m=softening_m,
            chunk_size=chunk_size,
            out=out,
        )

    def displacement_potential_contract(
        self,
        sources_m,
        targets_m,
        mass_kg,
        displacement_m,
        *,
        softening_m=0.0,
        chunk_size=None,
        out,
    ):
        self._displacement(
            "potential",
            sources_m,
            targets_m,
            mass_kg,
            displacement_m,
            softening_m=softening_m,
            chunk_size=chunk_size,
            out=out,
        )

    def dipole_contract(
        self,
        sources_m,
        targets_m,
        mass_kg,
        displacement_m,
        *,
        softening_m=0.0,
        chunk_size=None,
        out,
    ):
        self._displacement(
            "acceleration",
            sources_m,
            targets_m,
            mass_kg,
            displacement_m,
            softening_m=softening_m,
            chunk_size=chunk_size,
            out=out,
        )

    def displacement_gradient_contract(
        self,
        sources_m,
        targets_m,
        mass_kg,
        displacement_m,
        *,
        softening_m=0.0,
        chunk_size=None,
        out,
    ):
        self._displacement(
            "gradient",
            sources_m,
            targets_m,
            mass_kg,
            displacement_m,
            softening_m=softening_m,
            chunk_size=chunk_size,
            out=out,
        )


def accumulate_series_block(
    block: ScalarSourceSeriesBlock | DisplacementSourceSeriesBlock,
    *,
    total: np.ndarray,
    target_values: np.ndarray,
    observable: str,
    softening: float,
    kernel_chunk_size: int,
    backend: str,
) -> None:
    """Add one validated legacy block through the generic backend contract."""
    target_array = np.asarray(target_values)
    if target_array.ndim != 2 or target_array.shape[1] != 3:
        raise ValueError("target_values must have shape (M, 3)")
    if not isinstance(total, np.ndarray):
        raise TypeError("total must be a numpy array")

    _, tail_shape = _observable_components(observable)
    sample_count = block.sample_slice.stop - block.sample_slice.start
    expected_shape = (sample_count, len(target_array)) + tail_shape
    if total.shape != expected_shape:
        raise ValueError(f"total must have shape {expected_shape}, got {total.shape}")

    if isinstance(block, ScalarSourceSeriesBlock):
        value_arrays = (block.weights_kg,)
    else:
        value_arrays = (block.mass_kg, block.displacement_m)
    if total.dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
        raise TypeError("total must have dtype float64 or complex128")
    requires_complex = any(np.iscomplexobj(values) for values in value_arrays)
    if requires_complex and total.dtype != np.dtype(np.complex128):
        raise TypeError("complex inputs require a complex128 out accumulator")
    if not total.flags.writeable:
        raise ValueError("total must be writeable")
    if not np.isfinite(total).all():
        raise ValueError("total must contain only finite values")
    input_arrays = (block.positions_m, target_array, *value_arrays)
    if any(np.shares_memory(total, values) for values in input_arrays):
        raise ValueError("total must not share memory with contraction inputs")

    backend_adapter = SeriesDispatchBackend(backend)
    target_first = total.swapaxes(0, 1)
    if isinstance(block, ScalarSourceSeriesBlock):
        method_name = {
            "potential": "potential_contract",
            "acceleration": "acceleration_contract",
            "gradient": "gradient_contract",
        }[observable]
        getattr(backend_adapter, method_name)(
            block.positions_m,
            target_values,
            block.weights_kg.T,
            softening_m=softening,
            chunk_size=kernel_chunk_size,
            out=target_first,
        )
    else:
        method_name = {
            "potential": "displacement_potential_contract",
            "acceleration": "dipole_contract",
            "gradient": "displacement_gradient_contract",
        }[observable]
        getattr(backend_adapter, method_name)(
            block.positions_m,
            target_values,
            block.mass_kg,
            block.displacement_m.transpose(1, 0, 2),
            softening_m=softening,
            chunk_size=kernel_chunk_size,
            out=target_first,
        )

