"""Execution planning and streaming tests for generic source providers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gragra.constants import G_SI
from gragra.integrators.generic import ScalarSourceBlock
from gragra.integrators.streaming import (
    ScalarSourceSeriesBlock,
    SeriesMetadata,
    SeriesProvider,
    estimate_streaming_working_bytes,
)
from gragra.kernels.generic_backend import NumpyContractionBackend
from gragra.observables.streaming import stream_observable


class _Provider:
    def __init__(self, positions, weights, *, resident_bytes=0, collision=None):
        self.positions_m = np.asarray(positions, dtype=np.float64).copy()
        self.weights_kg = np.asarray(weights)
        self.source_count = len(self.positions_m)
        self.parameter_count = self.weights_kg.shape[1]
        self.kind = "scalar"
        self.replayable = True
        self.random_access = True
        self.resident_bytes = resident_bytes
        self.collision = collision
        self.scans = []
        self.widths = []

    def iter_blocks(self, parameter_slice, source_block_size):
        self.scans.append((parameter_slice.start, parameter_slice.stop))
        values = self.weights_kg
        if (
            self.collision is not None
            and parameter_slice.start == self.collision[0]
        ):
            values = values.copy()
        for start in range(0, self.source_count, source_block_size):
            stop = min(start + source_block_size, self.source_count)
            self.widths.append(stop - start)
            positions = self.positions_m[start:stop]
            if (
                self.collision is not None
                and parameter_slice.start == self.collision[0]
            ):
                positions = positions.copy()
                positions[0] = self.collision[1]
            yield ScalarSourceBlock(
                positions,
                values[start:stop, parameter_slice],
                start,
                parameter_slice,
            )


def _make_provider(*, source_count=7, parameter_count=4, resident_bytes=0):
    positions = np.column_stack(
        (np.arange(1.0, source_count + 1.0), np.zeros((source_count, 2)))
    )
    values = np.arange(1.0, source_count * parameter_count + 1.0).reshape(
        source_count, parameter_count
    ) / 10.0
    return _Provider(positions, values, resident_bytes=resident_bytes)


def _make_plan(
    provider,
    *,
    targets,
    observable="potential",
    result_dtype=np.float64,
    memory_budget_bytes=2_000_000,
    target_block_size=2,
    parameter_block_size=2,
    source_block_size=3,
    kernel_chunk_size=2,
    backend_workspace_bytes=256,
    strategy="fused",
):
    from gragra.integrators.generic import ExecutionPlanner

    return ExecutionPlanner.plan(
        provider,
        target_count=len(targets),
        observable=observable,
        result_dtype=result_dtype,
        backend_workspace_bytes=backend_workspace_bytes,
        target_block_size=target_block_size,
        parameter_block_size=parameter_block_size,
        source_block_size=source_block_size,
        kernel_chunk_size=kernel_chunk_size,
        strategy=strategy,
        memory_budget_bytes=memory_budget_bytes,
    )


def test_execution_planner_accounts_for_resident_workspace_and_fits_tiles():
    from gragra.integrators.generic import ExecutionPlan, ExecutionPlanner

    provider = _make_provider(resident_bytes=900)
    targets = np.column_stack(
        (np.arange(20.0, 28.0), np.ones((8, 2)))
    )
    plan = ExecutionPlanner.plan(
        provider,
        target_count=len(targets),
        observable="gradient",
        result_dtype=np.complex128,
        backend_workspace_bytes=1_200,
        target_block_size=8,
        parameter_block_size=4,
        source_block_size=7,
        kernel_chunk_size=7,
        strategy="blocked",
        memory_budget_bytes=18_000,
    )

    assert isinstance(plan, ExecutionPlan)
    assert plan.output_shape == (8, 4, 3, 3)
    assert plan.result_dtype == np.dtype(np.complex128)
    assert plan.strategy == "blocked"
    assert plan.loop_order == ("target", "parameter", "source")
    assert plan.provider_resident_bytes == 900
    assert plan.backend_workspace_bytes == 1_200
    assert plan.estimated_total_bytes <= plan.memory_budget_bytes
    assert plan.estimated_working_bytes + plan.provider_resident_bytes == (
        plan.estimated_total_bytes
    )
    assert 1 <= plan.target_block_size <= 8
    assert 1 <= plan.parameter_block_size <= 4
    assert 1 <= plan.source_block_size <= 7
    assert plan.kernel_chunk_size <= plan.source_block_size
    assert plan.target_block_size < 8 or plan.source_block_size < 7

    baseline = estimate_streaming_working_bytes(
        sample_block_size=plan.parameter_block_size,
        source_block_size=plan.source_block_size,
        target_block_size=plan.target_block_size,
        kernel_chunk_size=plan.kernel_chunk_size,
        observable="gradient",
        complex_values=True,
    )
    assert plan.estimated_working_bytes >= baseline + plan.backend_workspace_bytes


def test_execution_planner_counts_full_target_normalization_with_tiny_tiles():
    from gragra.integrators.generic import ExecutionPlanner

    provider = _make_provider(source_count=1, parameter_count=1)
    common = {
        "observable": "potential",
        "result_dtype": np.float64,
        "backend_workspace_bytes": 0,
        "target_block_size": 1,
        "parameter_block_size": 1,
        "source_block_size": 1,
        "kernel_chunk_size": 1,
        "memory_budget_bytes": 100_000,
    }

    small_plan = ExecutionPlanner.plan(provider, target_count=1, **common)
    assert small_plan.target_block_size == 1
    fitting_large_plan = ExecutionPlanner.plan(
        provider,
        target_count=2_500,
        **{**common, "memory_budget_bytes": 260_000},
    )
    assert fitting_large_plan.estimated_total_bytes <= 260_000
    assert fitting_large_plan.estimated_working_bytes >= 2_500 * 3 * 4 * 8
    with pytest.raises(MemoryError, match="minimum working set"):
        ExecutionPlanner.plan(provider, target_count=2_500, **common)


def test_execution_planner_shrinks_axis_that_reduces_total_estimate():
    from gragra.integrators.generic import ExecutionPlanner

    provider = _make_provider(source_count=1, parameter_count=1_000_000)
    plan = ExecutionPlanner.plan(
        provider,
        target_count=2,
        observable="potential",
        result_dtype=np.float64,
        backend_workspace_bytes=0,
        target_block_size=2,
        parameter_block_size=1_000,
        source_block_size=1,
        kernel_chunk_size=1,
        memory_budget_bytes=296_000,
    )

    assert plan.estimated_total_bytes <= 296_000
    assert plan.target_block_size == 1
    assert plan.parameter_block_size == 1_000


def test_execution_planner_searches_parameter_digest_breakpoints():
    from gragra.integrators.generic import ExecutionPlanner

    provider = _make_provider(source_count=1, parameter_count=1_000)
    plan = ExecutionPlanner.plan(
        provider,
        target_count=1,
        observable="potential",
        result_dtype=np.float64,
        backend_workspace_bytes=0,
        target_block_size=1,
        parameter_block_size=100,
        source_block_size=1,
        kernel_chunk_size=1,
        memory_budget_bytes=10_060,
    )

    assert plan.parameter_block_size == 91
    assert plan.estimated_total_bytes <= 10_060


@pytest.mark.parametrize(
    "observable,tail",
    [("potential", ()), ("acceleration", (3,)), ("gradient", (3, 3))],
)
@pytest.mark.parametrize("complex_values", [False, True])
def test_generic_executor_streams_target_first_readonly_blocks_without_mutation(
    observable, tail, complex_values
):
    from gragra.integrators.generic import ExecutionPlanner, GenericExecutor

    provider = _make_provider(source_count=7, parameter_count=4)
    if complex_values:
        provider.weights_kg = (
            provider.weights_kg + 0.25j * provider.weights_kg[::-1]
        ).astype(np.complex128)
    targets = np.array([[0.0, -1.0, 0.5], [8.0, 1.0, -2.0], [12.0, 0.5, 1.0]])
    original_targets = targets.copy()
    original_positions = provider.positions_m.copy()
    original_weights = provider.weights_kg.copy()
    plan = ExecutionPlanner.plan(
        provider,
        target_count=len(targets),
        observable=observable,
        result_dtype=np.complex128 if complex_values else np.float64,
        backend_workspace_bytes=512,
        target_block_size=1,
        parameter_block_size=2,
        source_block_size=3,
        kernel_chunk_size=2,
        memory_budget_bytes=100_000,
    )
    blocks = list(
        GenericExecutor.iter_blocks(
            provider, targets, plan, NumpyContractionBackend()
        )
    )

    assert [(target.start, parameter.start) for target, parameter, _ in blocks] == [
        (target, parameter)
        for target in range(0, len(targets), plan.target_block_size)
        for parameter in range(0, provider.parameter_count, plan.parameter_block_size)
    ]
    assert all(values.shape[2:] == tail for _, _, values in blocks)
    assert all(not values.flags.writeable for _, _, values in blocks)
    assert np.array_equal(targets, original_targets)
    assert np.array_equal(provider.positions_m, original_positions)
    assert np.array_equal(provider.weights_kg, original_weights)
    assert max(provider.widths) <= plan.source_block_size
    assert plan.source_block_size < provider.source_count

    result = np.empty(
        (len(targets), provider.parameter_count) + tail,
        dtype=np.complex128 if complex_values else np.float64,
    )
    for target_slice, parameter_slice, values in blocks:
        result[
            target_slice,
            parameter_slice,
        ] = values
    expected = np.zeros_like(result)
    for target_index, target in enumerate(targets):
        for parameter in range(provider.parameter_count):
            for source_index, source in enumerate(provider.positions_m):
                delta = source - target
                r2 = float(delta @ delta)
                weight = original_weights[source_index, parameter]
                if observable == "potential":
                    term = -1.0 / math.sqrt(r2)
                elif observable == "acceleration":
                    term = delta / r2**1.5
                else:
                    term = 3.0 * np.outer(delta, delta) / r2**2.5 - np.eye(3) / r2**1.5
                expected[target_index, parameter] += weight * term
    np.testing.assert_allclose(result, expected, rtol=3e-14, atol=3e-14)
    assert result.dtype == (np.complex128 if complex_values else np.float64)


def test_generic_executor_never_sends_a_full_source_array_to_the_backend():
    from gragra.integrators.generic import GenericExecutor

    class RecordingBackend(NumpyContractionBackend):
        def __init__(self):
            self.source_widths = []

        def potential_contract(self, sources_m, targets_m, weights_kg, **kwargs):
            self.source_widths.append(len(sources_m))
            return super().potential_contract(
                sources_m, targets_m, weights_kg, **kwargs
            )

    provider = _make_provider(source_count=257, parameter_count=3)
    targets = np.column_stack((np.arange(400.0, 405.0), np.ones((5, 2))))
    plan = _make_plan(
        provider,
        targets=targets,
        target_block_size=2,
        parameter_block_size=1,
        source_block_size=16,
        kernel_chunk_size=8,
    )
    backend = RecordingBackend()

    blocks = list(GenericExecutor.iter_blocks(provider, targets, plan, backend))

    assert blocks
    assert plan.source_block_size == 16 < provider.source_count
    assert max(provider.widths) <= plan.source_block_size
    assert max(backend.source_widths) <= plan.source_block_size
    assert all(
        values.shape[0] <= plan.target_block_size for _, _, values in blocks
    )
    assert plan.estimated_total_bytes <= plan.memory_budget_bytes


def test_generic_executor_allows_complex_result_dtype_for_real_weights():
    from gragra.integrators.generic import ExecutionPlanner, GenericExecutor

    provider = _make_provider(source_count=3, parameter_count=2)
    targets = np.array([[0.0, -2.0, 0.0], [7.0, 1.0, 0.0]])
    plan = ExecutionPlanner.plan(
        provider,
        target_count=len(targets),
        observable="potential",
        result_dtype=np.complex128,
        backend_workspace_bytes=0,
        target_block_size=1,
        parameter_block_size=2,
        source_block_size=2,
        kernel_chunk_size=2,
        memory_budget_bytes=100_000,
    )

    blocks = list(
        GenericExecutor.iter_blocks(provider, targets, plan, NumpyContractionBackend())
    )
    assert blocks
    assert all(values.dtype == np.dtype(np.complex128) for _, _, values in blocks)
    assert all(np.all(values.imag == 0.0) for _, _, values in blocks)


@pytest.mark.parametrize("observable", ["potential", "acceleration", "gradient"])
def test_generic_executor_dispatches_displacement_blocks(observable):
    from gragra.integrators.generic import (
        DisplacementSourceBlock,
        ExecutionPlanner,
        GenericExecutor,
    )
    from gragra.integrators.point_direct import direct_point_displacement_sum

    class DisplacementProvider:
        def __init__(self):
            self.positions_m = np.array(
                [[1.0, 0.2, 0.1], [2.0, -0.3, 0.4], [-1.0, 1.2, 0.6]]
            )
            self.mass_kg = np.array([1.5, 0.0, 2.0])
            self.displacement_m = np.arange(18.0).reshape(3, 2, 3) / 20.0
            self.source_count = len(self.positions_m)
            self.parameter_count = 2
            self.kind = "displacement"
            self.replayable = True
            self.random_access = True
            self.resident_bytes = 64

        def iter_blocks(self, parameter_slice, source_block_size):
            for start in range(0, self.source_count, source_block_size):
                stop = min(start + source_block_size, self.source_count)
                yield DisplacementSourceBlock(
                    self.positions_m[start:stop],
                    self.mass_kg[start:stop],
                    self.displacement_m[start:stop, parameter_slice, :],
                    start,
                    parameter_slice,
                )

    provider = DisplacementProvider()
    targets = np.array([[0.0, -0.5, 0.2], [4.0, 1.0, -1.0]])
    plan = ExecutionPlanner.plan(
        provider,
        target_count=len(targets),
        observable=observable,
        result_dtype=np.float64,
        backend_workspace_bytes=512,
        target_block_size=1,
        parameter_block_size=1,
        source_block_size=2,
        kernel_chunk_size=1,
        memory_budget_bytes=100_000,
    )
    blocks = list(
        GenericExecutor.iter_blocks(
            provider, targets, plan, NumpyContractionBackend()
        )
    )
    actual = np.empty(plan.output_shape)
    for target_slice, parameter_slice, values in blocks:
        actual[target_slice, parameter_slice] = values
    expected = np.stack(
        [
            direct_point_displacement_sum(
                provider.mass_kg,
                provider.displacement_m[:, parameter, :],
                provider.positions_m,
                targets,
                observable=observable,
                chunk_size=1,
            )
            for parameter in range(provider.parameter_count)
        ],
        axis=1,
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)
    assert plan.source_kind == "displacement"
    assert all(not values.flags.writeable for _, _, values in blocks)


def test_generic_executor_preflights_every_parameter_tile_before_first_yield():
    from gragra.integrators.generic import GenericExecutor

    provider = _make_provider(source_count=4, parameter_count=4)
    targets = np.array([[0.0, 1.0, 0.0], [7.0, 2.0, 1.0]])
    plan = _make_plan(
        provider,
        targets=targets,
        target_block_size=1,
        parameter_block_size=1,
        source_block_size=2,
        kernel_chunk_size=2,
    )
    blocks = GenericExecutor.iter_blocks(
        provider, targets, plan, NumpyContractionBackend()
    )

    first = next(blocks)
    assert first[0] == slice(0, 1)
    assert provider.scans[:4] == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert provider.scans[4] == (0, 1)  # first calculation replay


def test_generic_executor_rejects_late_collision_before_any_publication():
    from gragra.integrators.generic import GenericExecutor

    collision_target = np.array([50.0, 0.0, 0.0])
    provider = _make_provider(source_count=5, parameter_count=4)
    provider.collision = (3, collision_target)
    targets = np.array([[0.0, 0.0, 0.0], collision_target])
    plan = _make_plan(
        provider,
        targets=targets,
        target_block_size=1,
        parameter_block_size=1,
        source_block_size=2,
        kernel_chunk_size=1,
    )
    blocks = GenericExecutor.iter_blocks(
        provider, targets, plan, NumpyContractionBackend()
    )

    with pytest.raises(ValueError, match="Collision detected.*zero softening"):
        next(blocks)
    assert provider.scans == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_generic_executor_checks_each_replay_before_publishing_its_tile():
    from gragra.integrators.generic import GenericExecutor

    class ChangingProvider(_Provider):
        def __init__(self):
            super().__init__(
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                np.ones((2, 1)),
            )
            self.calls = 0

        def iter_blocks(self, parameter_slice, _source_block_size):
            self.calls += 1
            values = self.weights_kg
            if self.calls >= 3:
                values = values * 2.0
            yield ScalarSourceBlock(
                self.positions_m,
                values[:, parameter_slice],
                0,
                parameter_slice,
            )

    provider = ChangingProvider()
    targets = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    plan = _make_plan(
        provider,
        targets=targets,
        target_block_size=1,
        parameter_block_size=1,
        source_block_size=2,
        kernel_chunk_size=2,
    )
    blocks = GenericExecutor.iter_blocks(
        provider, targets, plan, NumpyContractionBackend()
    )

    first = next(blocks)
    assert first[0] == slice(0, 1)
    with pytest.raises(ValueError, match="replay"):
        next(blocks)


def test_stream_observable_applies_g_once_after_g_free_streaming():
    positions = np.array([[2.0, 0.0, 0.0]])
    metadata = SeriesMetadata(
        positions_m=positions,
        source_ids=("source",),
        coordinate_frame="ENU",
        coordinate_origin_m=np.zeros(3),
        axis_kind="time",
        axis_values=np.array([0.0]),
        sample_count=1,
        source_count=1,
        quantity="density",
        density_kind="delta_rho",
    )

    def factory(sample_slice, _source_block_size):
        yield ScalarSourceSeriesBlock(
            positions,
            sample_slice,
            0,
            np.array([[3.0]]),
            ("source",),
        )

    provider = SeriesProvider(metadata, factory)
    block = next(
        stream_observable(
            provider,
            [[0.0, 0.0, 0.0]],
            observable="potential",
        )
    )
    assert block.values[0, 0] == pytest.approx(-3.0 * G_SI / 2.0)


def test_series_dispatch_rejects_scalar_output_alias_without_mutating_targets():
    from gragra.integrators.generic import (
        SeriesDispatchBackend,
        accumulate_series_block,
    )

    positions = np.array([[4.0, 0.0, 0.0], [5.0, 1.0, 0.0]])
    block = ScalarSourceSeriesBlock(
        positions,
        slice(0, 3),
        0,
        np.ones((3, 2)),
        ("s0", "s1"),
    )
    targets = np.array([[0.0, 2.0, 0.0]])
    original_targets = targets.copy()
    aliased_total = targets.reshape(3, 1)

    with pytest.raises(ValueError, match="must not share memory"):
        accumulate_series_block(
            block,
            total=aliased_total,
            target_values=targets,
            observable="potential",
            softening=0.0,
            kernel_chunk_size=1,
            backend="numpy",
        )

    np.testing.assert_array_equal(targets, original_targets)
    with pytest.raises(ValueError, match="must not share memory"):
        SeriesDispatchBackend("numpy").potential_contract(
            positions,
            targets,
            block.weights_kg.T,
            out=targets,
        )
    np.testing.assert_array_equal(targets, original_targets)

    weights = np.ones((2, 3))
    original_weights = weights.copy()
    with pytest.raises(ValueError, match="must not share memory"):
        SeriesDispatchBackend("numpy").potential_contract(
            positions,
            np.array([[0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]),
            weights,
            out=weights,
        )
    np.testing.assert_array_equal(weights, original_weights)


def test_series_dispatch_rejects_displacement_output_alias_without_mutating_targets():
    from gragra.integrators.generic import (
        SeriesDispatchBackend,
        accumulate_series_block,
    )
    from gragra.integrators.streaming import DisplacementSourceSeriesBlock

    positions = np.array([[4.0, 0.0, 0.0], [5.0, 1.0, 0.0]])
    block = DisplacementSourceSeriesBlock(
        positions,
        slice(0, 1),
        0,
        np.ones(2),
        np.ones((1, 2, 3)),
        ("s0", "s1"),
    )
    targets = np.array(
        [[0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 4.0, 0.0]]
    )
    original_targets = targets.copy()
    aliased_total = targets.reshape(1, 3, 3)

    with pytest.raises(ValueError, match="must not share memory"):
        accumulate_series_block(
            block,
            total=aliased_total,
            target_values=targets,
            observable="acceleration",
            softening=0.0,
            kernel_chunk_size=1,
            backend="numpy",
        )

    np.testing.assert_array_equal(targets, original_targets)
    with pytest.raises(ValueError, match="must not share memory"):
        SeriesDispatchBackend("numpy").dipole_contract(
            positions,
            targets,
            block.mass_kg,
            block.displacement_m.transpose(1, 0, 2),
            out=targets.reshape(3, 1, 3),
        )
    np.testing.assert_array_equal(targets, original_targets)

    displacement = np.ones((2, 1, 3))
    original_displacement = displacement.copy()
    with pytest.raises(ValueError, match="must not share memory"):
        SeriesDispatchBackend("numpy").dipole_contract(
            positions,
            np.array([[0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]),
            np.ones(2),
            displacement,
            out=displacement,
        )
    np.testing.assert_array_equal(displacement, original_displacement)


@pytest.mark.parametrize("invalid_kind", ["shape", "dtype"])
def test_series_dispatch_validates_scalar_output_before_accumulation(invalid_kind):
    from gragra.integrators.generic import (
        SeriesDispatchBackend,
        accumulate_series_block,
    )

    positions = np.array([[4.0, 0.0, 0.0], [5.0, 1.0, 0.0]])
    block = ScalarSourceSeriesBlock(
        positions,
        slice(0, 2),
        0,
        np.ones((2, 2)),
        ("s0", "s1"),
    )
    targets = np.array([[0.0, 2.0, 0.0]])
    output = (
        np.zeros((2, 2), dtype=np.float64)
        if invalid_kind == "shape"
        else np.zeros((2, 1), dtype=np.float32)
    )

    with pytest.raises((TypeError, ValueError), match=invalid_kind):
        accumulate_series_block(
            block,
            total=output,
            target_values=targets,
            observable="potential",
            softening=0.0,
            kernel_chunk_size=1,
            backend="numpy",
        )

    weights = np.ones((2, 2))
    direct_output = (
        np.zeros((2, 2), dtype=np.float64)
        if invalid_kind == "shape"
        else np.zeros((1, 2), dtype=np.float32)
    )
    with pytest.raises((TypeError, ValueError), match=invalid_kind):
        SeriesDispatchBackend("numpy").potential_contract(
            positions,
            targets,
            weights,
            out=direct_output,
        )


@pytest.mark.parametrize("invalid_kind", ["shape", "dtype"])
def test_series_dispatch_validates_displacement_output_before_accumulation(
    invalid_kind,
):
    from gragra.integrators.generic import (
        SeriesDispatchBackend,
        accumulate_series_block,
    )
    from gragra.integrators.streaming import DisplacementSourceSeriesBlock

    positions = np.array([[4.0, 0.0, 0.0], [5.0, 1.0, 0.0]])
    displacement = np.ones((1, 2, 3))
    block = DisplacementSourceSeriesBlock(
        positions,
        slice(0, 1),
        0,
        np.ones(2),
        displacement,
        ("s0", "s1"),
    )
    targets = np.array([[0.0, 2.0, 0.0], [0.0, 3.0, 0.0]])
    total = (
        np.zeros((1, 1, 3), dtype=np.float64)
        if invalid_kind == "shape"
        else np.zeros((1, 2, 3), dtype=np.float32)
    )

    with pytest.raises((TypeError, ValueError), match=invalid_kind):
        accumulate_series_block(
            block,
            total=total,
            target_values=targets,
            observable="acceleration",
            softening=0.0,
            kernel_chunk_size=1,
            backend="numpy",
        )

    direct_output = (
        np.zeros((2, 2, 3), dtype=np.float64)
        if invalid_kind == "shape"
        else np.zeros((2, 1, 3), dtype=np.float32)
    )
    with pytest.raises((TypeError, ValueError), match=invalid_kind):
        SeriesDispatchBackend("numpy").dipole_contract(
            positions,
            targets,
            np.ones(2),
            displacement.transpose(1, 0, 2),
            out=direct_output,
        )


def test_series_dispatch_allows_complex_output_for_real_inputs():
    from gragra.integrators.generic import (
        SeriesDispatchBackend,
        accumulate_series_block,
    )

    positions = np.array([[2.0, 0.0, 0.0]])
    targets = np.array([[0.0, 0.0, 0.0]])
    block = ScalarSourceSeriesBlock(
        positions,
        slice(0, 2),
        0,
        np.array([[3.0], [4.0]]),
        ("s0",),
    )
    total = np.zeros((2, 1), dtype=np.complex128)

    accumulate_series_block(
        block,
        total=total,
        target_values=targets,
        observable="potential",
        softening=0.0,
        kernel_chunk_size=1,
        backend="numpy",
    )
    np.testing.assert_allclose(total[:, 0], [-1.5, -2.0])
    assert np.all(total.imag == 0.0)

    direct_out = np.zeros((1, 2), dtype=np.complex128)
    SeriesDispatchBackend("numpy").potential_contract(
        positions,
        targets,
        block.weights_kg.T,
        out=direct_out,
    )
    np.testing.assert_allclose(direct_out[0], [-1.5, -2.0])
    assert np.all(direct_out.imag == 0.0)
