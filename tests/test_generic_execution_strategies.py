"""Numerical and memory contracts for generic execution strategies."""

from __future__ import annotations

import math
import weakref

import numpy as np
import pytest

from gragra.integrators.generic import (
    ExecutionPlanner,
    GenericExecutor,
    ScalarSourceBlock,
)
from gragra.kernels.generic_backend import NumpyContractionBackend

# Declared before comparisons. The independent reference evaluates point
# kernels in longdouble and sums weighted terms with math.fsum. This matches
# T04's multi-source reference and leaves room for float64 kernel rounding and
# different source-block grouping while rejecting axis/sign errors.
NUMERICAL_BUDGET = {
    "reference": "longdouble point terms plus math.fsum, as in T04",
    "strategy_rtol": 3e-13,
    "strategy_atol": 3e-14,
    "reference_rtol": 3e-13,
    "reference_atol": 3e-14,
}


class _Provider:
    kind = "scalar"
    replayable = True
    random_access = True
    resident_bytes = 0

    def __init__(self):
        self.positions_m = np.array(
            [
                [1.0, 0.2, -0.3],
                [-0.7, 2.1, 0.4],
                [2.2, -1.0, 0.8],
                [0.3, 1.5, -1.2],
                [-2.1, -0.4, 1.3],
                [1.7, -2.3, 0.6],
                [-1.4, 0.7, 2.2],
            ],
            dtype=np.float64,
        )
        self.weights_kg = np.array(
            [
                [0.6, -1.2, 0.4],
                [-0.5, 0.7, 1.1],
                [1.3, -0.2, -0.8],
                [-0.9, 0.3, 0.5],
                [0.2, 1.4, -0.6],
                [0.75, -0.45, 0.9],
                [-1.1, 0.25, 0.35],
            ],
            dtype=np.float64,
        )
        self.source_count = len(self.positions_m)
        self.parameter_count = self.weights_kg.shape[1]

    def iter_blocks(self, parameter_slice, source_block_size):
        for source_start in range(0, self.source_count, source_block_size):
            source_stop = min(source_start + source_block_size, self.source_count)
            yield ScalarSourceBlock(
                self.positions_m[source_start:source_stop],
                self.weights_kg[source_start:source_stop, parameter_slice],
                source_start,
                parameter_slice,
            )


def _reference_kernel(source, target, observable, softening):
    delta = np.asarray(source, dtype=np.longdouble) - np.asarray(
        target, dtype=np.longdouble
    )
    squared = np.sum(delta * delta, dtype=np.longdouble) + np.longdouble(softening) ** 2
    if observable == "potential":
        return np.asarray(-1 / np.sqrt(squared), dtype=np.longdouble)
    if observable == "acceleration":
        return delta / squared ** np.longdouble(1.5)
    return 3 * np.outer(delta, delta) / squared ** np.longdouble(2.5) - np.eye(
        3, dtype=np.longdouble
    ) / squared ** np.longdouble(1.5)


def _reference(provider, targets, observable, softening):
    tail = {"potential": (), "acceleration": (3,), "gradient": (3, 3)}[observable]
    result = np.zeros(
        (len(targets), provider.parameter_count) + tail,
        dtype=provider.weights_kg.dtype,
    )
    components = list(np.ndindex(tail)) if tail else [()]
    for target_index, target in enumerate(targets):
        kernels = [
            _reference_kernel(source, target, observable, softening)
            for source in provider.positions_m
        ]
        for parameter in range(provider.parameter_count):
            for component in components:
                terms = []
                for source_index, kernel in enumerate(kernels):
                    entry = kernel[component] if component else kernel
                    terms.append(
                        provider.weights_kg[source_index, parameter] * float(entry)
                    )
                if np.iscomplexobj(provider.weights_kg):
                    result[(target_index, parameter) + component] = complex(
                        math.fsum(value.real for value in terms),
                        math.fsum(value.imag for value in terms),
                    )
                else:
                    result[(target_index, parameter) + component] = math.fsum(terms)
    return result


def _run_strategy(provider, targets, observable, strategy):
    plan = ExecutionPlanner.plan(
        provider,
        target_count=len(targets),
        observable=observable,
        result_dtype=provider.weights_kg.dtype,
        backend_workspace_bytes=512,
        target_block_size=2,
        parameter_block_size=2,
        source_block_size=3,
        kernel_chunk_size=2,
        strategy=strategy,
        memory_budget_bytes=100_000,
    )
    result = np.empty(plan.output_shape, dtype=plan.result_dtype)
    blocks = list(
        GenericExecutor.iter_blocks(
            provider, targets, plan, NumpyContractionBackend(), softening_m=0.17
        )
    )
    for target_slice, parameter_slice, values in blocks:
        result[target_slice, parameter_slice] = values
    return plan, result, blocks


@pytest.mark.parametrize("observable", ["potential", "acceleration", "gradient"])
@pytest.mark.parametrize("complex_values", [False, True])
def test_fused_and_blocked_agree_with_each_other_and_t04_reference(
    observable, complex_values
):
    provider = _Provider()
    if complex_values:
        provider.weights_kg = (
            provider.weights_kg + 0.5j * provider.weights_kg[::-1]
        ).astype(np.complex128)
    targets = np.array(
        [[0.1, -1.4, 0.2], [3.3, 0.8, -2.0], [-1.5, 2.5, 1.4]],
        dtype=np.float64,
    )
    positions_before = provider.positions_m.copy()
    weights_before = provider.weights_kg.copy()
    targets_before = targets.copy()

    fused_plan, fused, fused_blocks = _run_strategy(
        provider, targets, observable, "fused"
    )
    blocked_plan, blocked, blocked_blocks = _run_strategy(
        provider, targets, observable, "blocked"
    )
    expected = _reference(provider, targets, observable, softening=0.17)
    budget = NUMERICAL_BUDGET

    assert budget["reference"] == "longdouble point terms plus math.fsum, as in T04"
    np.testing.assert_allclose(
        fused, blocked, rtol=budget["strategy_rtol"], atol=budget["strategy_atol"]
    )
    np.testing.assert_allclose(
        fused, expected, rtol=budget["reference_rtol"], atol=budget["reference_atol"]
    )
    np.testing.assert_allclose(
        blocked,
        expected,
        rtol=budget["reference_rtol"],
        atol=budget["reference_atol"],
    )
    assert fused_plan.strategy == "fused"
    assert blocked_plan.strategy == "blocked"
    assert blocked_plan.estimated_working_bytes > fused_plan.estimated_working_bytes
    assert fused_plan.estimated_total_bytes <= fused_plan.memory_budget_bytes
    assert blocked_plan.estimated_total_bytes <= blocked_plan.memory_budget_bytes
    assert all(
        not values.flags.writeable for _, _, values in fused_blocks + blocked_blocks
    )
    np.testing.assert_array_equal(provider.positions_m, positions_before)
    np.testing.assert_array_equal(provider.weights_kg, weights_before)
    np.testing.assert_array_equal(targets, targets_before)


def test_blocked_strategy_accounts_for_its_second_accumulator_under_budget():
    provider = _Provider()
    targets = np.array(
        [[0.1, -1.4, 0.2], [3.3, 0.8, -2.0], [-1.5, 2.5, 1.4]],
        dtype=np.float64,
    )
    common = {
        "target_count": len(targets),
        "observable": "gradient",
        "result_dtype": np.float64,
        "backend_workspace_bytes": 512,
        "target_block_size": 3,
        "parameter_block_size": 3,
        "source_block_size": 7,
        "kernel_chunk_size": 4,
        "memory_budget_bytes": 11_600,
    }

    fused = ExecutionPlanner.plan(provider, strategy="fused", **common)
    blocked = ExecutionPlanner.plan(provider, strategy="blocked", **common)

    assert fused.estimated_total_bytes <= common["memory_budget_bytes"]
    assert blocked.estimated_total_bytes <= common["memory_budget_bytes"]
    assert blocked.target_block_size * blocked.parameter_block_size < (
        fused.target_block_size * fused.parameter_block_size
    )


def test_blocked_uses_a_private_source_partial_before_adding_to_tile():
    class RecordingBackend(NumpyContractionBackend):
        def __init__(self):
            self.outputs = []
            self.initial_values = []

        def potential_contract_blocked(
            self, sources_m, targets_m, weights_kg, **kwargs
        ):
            self.outputs.append(kwargs["out"])
            self.initial_values.append(kwargs["out"].copy())
            return super().potential_contract_blocked(
                sources_m, targets_m, weights_kg, **kwargs
            )

    provider = _Provider()
    targets = np.array([[0.1, -1.4, 0.2]])
    backend = RecordingBackend()
    plan = ExecutionPlanner.plan(
        provider,
        target_count=1,
        observable="potential",
        result_dtype=np.float64,
        backend_workspace_bytes=0,
        target_block_size=1,
        parameter_block_size=provider.parameter_count,
        source_block_size=3,
        kernel_chunk_size=2,
        strategy="blocked",
        memory_budget_bytes=20_000,
    )

    list(GenericExecutor.iter_blocks(provider, targets, plan, backend))

    # A complete replay has three source blocks. The blocked path contracts
    # each into a private partial before adding it to the tile accumulator.
    first_replay_outputs = backend.outputs[:3]
    assert len(first_replay_outputs) == 3
    assert len({id(output) for output in first_replay_outputs}) == 1
    assert all(not np.any(values) for values in backend.initial_values[:3])
    assert all(
        output.shape == (1, provider.parameter_count)
        for output in first_replay_outputs
    )


def test_blocked_dispatches_to_a_kernel_tile_contraction(monkeypatch):
    import gragra.kernels.generic_backend as generic_backend

    assert hasattr(NumpyContractionBackend, "potential_contract_blocked")

    provider = _Provider()
    targets = np.array([[0.1, -1.4, 0.2]])
    backend = NumpyContractionBackend()
    plan = ExecutionPlanner.plan(
        provider,
        target_count=1,
        observable="potential",
        result_dtype=np.float64,
        backend_workspace_bytes=0,
        target_block_size=1,
        parameter_block_size=provider.parameter_count,
        source_block_size=3,
        kernel_chunk_size=2,
        strategy="blocked",
        memory_budget_bytes=20_000,
    )
    kernel_shapes = []
    original = generic_backend._scalar_kernel_tile

    def record_kernel_tile(*args, **kwargs):
        kernel = original(*args, **kwargs)
        kernel_shapes.append(kernel.shape)
        return kernel

    monkeypatch.setattr(generic_backend, "_scalar_kernel_tile", record_kernel_tile)
    list(GenericExecutor.iter_blocks(provider, targets, plan, backend))

    # Each source/target geometry tile is built once for the full parameter
    # slice, so the seven sources need five kernel tiles, not fifteen.
    assert kernel_shapes == [(1, 2), (1, 1), (1, 2), (1, 1), (1, 1)]


def test_blocked_plan_records_kernel_and_contraction_scratch():
    provider = _Provider()
    plan = ExecutionPlanner.plan(
        provider,
        target_count=1,
        observable="gradient",
        result_dtype=np.float64,
        backend_workspace_bytes=0,
        target_block_size=1,
        parameter_block_size=provider.parameter_count,
        source_block_size=3,
        kernel_chunk_size=2,
        strategy="blocked",
        memory_budget_bytes=20_000,
    )

    partial_bytes = 1 * provider.parameter_count * 9 * np.dtype(np.float64).itemsize
    kernel_workspace_bytes = 1 * 2 * 64 * np.dtype(np.float64).itemsize
    contraction_bytes = partial_bytes
    assert plan.strategy_scratch_bytes == (
        partial_bytes + kernel_workspace_bytes + contraction_bytes
    )


def test_previous_tile_arrays_are_released_before_next_tile_allocation(monkeypatch):
    import gragra.integrators.generic as generic_module

    class RecordingBackend(NumpyContractionBackend):
        def __init__(self):
            self.output_refs = []

        def potential_contract_blocked(
            self, sources_m, targets_m, weights_kg, **kwargs
        ):
            self.output_refs.append(weakref.ref(kwargs["out"]))
            return super().potential_contract_blocked(
                sources_m, targets_m, weights_kg, **kwargs
            )

    provider = _Provider()
    targets = np.array([[0.1, -1.4, 0.2], [3.3, 0.8, -2.0]])
    backend = RecordingBackend()
    plan = ExecutionPlanner.plan(
        provider,
        target_count=len(targets),
        observable="potential",
        result_dtype=np.float64,
        backend_workspace_bytes=0,
        target_block_size=1,
        parameter_block_size=3,
        source_block_size=3,
        kernel_chunk_size=2,
        strategy="blocked",
        memory_budget_bytes=20_000,
    )
    previous_refs = []
    live_at_next_tile_allocation = []
    original_zeros = generic_module.np.zeros

    def inspect_tile_allocation(shape, *args, **kwargs):
        if shape == (1, 3) and previous_refs:
            live_at_next_tile_allocation.append(
                tuple(reference() is not None for reference in previous_refs)
            )
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(generic_module.np, "zeros", inspect_tile_allocation)
    blocks = GenericExecutor.iter_blocks(provider, targets, plan, backend)

    first = next(blocks)
    previous_refs[:] = [weakref.ref(first[2]), backend.output_refs[-1]]
    del first
    next(blocks)

    assert live_at_next_tile_allocation == [(False, False)]
    blocks.close()


def test_numpy_gradient_workspace_bound_covers_live_parameterized_arrays():
    from gragra.integrators.generic import estimate_numpy_gradient_workspace_bytes

    target_tile = 32
    kernel_chunk = 512
    parameter_tile = 4
    result_dtype = np.dtype(np.float64)
    workspace_bytes = estimate_numpy_gradient_workspace_bytes(
        target_block_size=target_tile,
        kernel_chunk_size=kernel_chunk,
        parameter_block_size=parameter_tile,
        result_dtype=result_dtype,
    )

    source_target_pairs = target_tile * kernel_chunk
    documented_geometry_bytes = source_target_pairs * 61 * np.dtype(np.float64).itemsize
    parameterized_contribution_bytes = (
        source_target_pairs * 9 * parameter_tile * result_dtype.itemsize
    )
    reduction_bytes = target_tile * parameter_tile * 9 * result_dtype.itemsize

    assert workspace_bytes >= (
        documented_geometry_bytes + parameterized_contribution_bytes + reduction_bytes
    )
