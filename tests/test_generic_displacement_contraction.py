"""Independent checks for parameterized source-displacement contractions."""

from __future__ import annotations

import numpy as np
import pytest

from gragra.integrators.generic import (
    DisplacementSourceBlock,
    ExecutionPlanner,
    GenericExecutor,
)
from gragra.kernels import inverse_square
from gragra.kernels.generic_backend import NumpyContractionBackend

# Declare the central-difference step and comparison budget before comparing
# the implementation. The fixed refinement sequence moves each source by at
# most 1e-4, 5e-5, then 2.5e-5 of the nearest target distance; the budget covers
# central-difference truncation and floating-point cancellation after
# normalization by the observable's characteristic mass * displacement /
# distance**power scale.
FINITE_DIFFERENCE_STEP_RATIOS = (1e-4, 5e-5, 2.5e-5)
FINITE_DIFFERENCE_BUDGET = {
    "reference": "central difference of scalar source-shifted kernels",
    "rtol": 5e-8,
    "atol": 3e-10,
}

CHUNK_BUDGET = {"rtol": 3e-13, "atol": 3e-14}

CONTRACTIONS = (
    ("displacement_potential_contract", "potential_contract", (), 2),
    ("dipole_contract", "acceleration_contract", (3,), 3),
    ("displacement_gradient_contract", "gradient_contract", (3, 3), 4),
)
BACKEND_CONTRACTIONS = tuple(
    (method, tail) for method, _, tail, _ in CONTRACTIONS
)
BLOCKED_CONTRACTIONS = tuple(
    (f"{method}_blocked", method, tail) for method, _, tail, _ in CONTRACTIONS
)

SOURCES = np.array(
    [
        [0.2, -0.4, 0.7],
        [-1.1, 0.6, 0.3],
        [0.8, 1.2, -0.9],
        [-0.5, -1.3, 0.6],
    ],
    dtype=np.float64,
)
TARGETS = np.array([[2.0, -0.3, 0.4], [-0.6, -1.4, 1.7]], dtype=np.float64)
MASS = np.array([1.5, 0.7, 2.1, 0.9], dtype=np.float64)
DISPLACEMENT = np.array(
    [
        [[0.3, -0.2, 0.5], [-0.2, 0.7, 0.1]],
        [[-0.1, 0.4, 0.2], [0.5, -0.3, 0.8]],
        [[0.6, 0.1, -0.3], [0.2, -0.6, 0.4]],
        [[-0.3, 0.2, 0.4], [0.1, 0.2, -0.5]],
    ],
    dtype=np.float64,
)


def _out_shape(targets, displacement, tail):
    return (len(targets), displacement.shape[1]) + tail


def _call(
    backend, name, sources, targets, mass, displacement, *, softening, chunk, out
):
    return getattr(backend, name)(
        sources,
        targets,
        mass,
        displacement,
        softening_m=softening,
        chunk_size=chunk,
        out=out,
    )


@pytest.mark.parametrize(
    "method,field_name,tail,distance_power", CONTRACTIONS
)
def test_source_shift_central_difference_fixes_each_derivative_sign(
    method, field_name, tail, distance_power
):
    backend = NumpyContractionBackend()
    softening = 0.17
    out = np.zeros(_out_shape(TARGETS, DISPLACEMENT, tail), dtype=np.float64)
    returned = _call(
        backend,
        method,
        SOURCES,
        TARGETS,
        MASS,
        DISPLACEMENT,
        softening=softening,
        chunk=2,
        out=out,
    )

    nearest_distance = np.linalg.norm(
        SOURCES[:, np.newaxis, :] - TARGETS[np.newaxis, :, :], axis=-1
    ).min()
    largest_displacement = np.linalg.norm(DISPLACEMENT, axis=-1).max()
    field = getattr(inverse_square, field_name)
    scale = (
        MASS.sum()
        * largest_displacement
        / nearest_distance**distance_power
    )
    budget = FINITE_DIFFERENCE_BUDGET
    finite_differences = []
    scaled_errors = []

    for step_ratio in FINITE_DIFFERENCE_STEP_RATIOS:
        step = step_ratio * nearest_distance / largest_displacement
        finite_difference = np.empty_like(out)
        for parameter in range(DISPLACEMENT.shape[1]):
            direction = DISPLACEMENT[:, parameter, :]
            plus = field(
                SOURCES + step * direction,
                MASS,
                TARGETS,
                softening_m=softening,
                chunk_size=2,
            )
            minus = field(
                SOURCES - step * direction,
                MASS,
                TARGETS,
                softening_m=softening,
                chunk_size=2,
            )
            finite_difference[:, parameter] = (plus - minus) / (2.0 * step)
        finite_differences.append(finite_difference)
        scaled_errors.append(np.max(np.abs(out / scale - finite_difference / scale)))

    assert budget["reference"] == "central difference of scalar source-shifted kernels"
    assert returned is None
    assert scaled_errors[-1] < scaled_errors[0], scaled_errors
    np.testing.assert_allclose(
        out / scale,
        finite_differences[-1] / scale,
        rtol=budget["rtol"],
        atol=budget["atol"],
    )


@pytest.mark.parametrize("method,tail", BACKEND_CONTRACTIONS)
@pytest.mark.parametrize("complex_values", [False, True])
def test_displacement_contracts_add_into_out_across_source_chunks(
    method, tail, complex_values
):
    backend = NumpyContractionBackend()
    displacement = DISPLACEMENT.copy()
    if complex_values:
        displacement = (displacement + 0.35j * displacement[::-1]).astype(
            np.complex128
        )
    sources_before = SOURCES.copy()
    targets_before = TARGETS.copy()
    mass_before = MASS.copy()
    displacement_before = displacement.copy()
    output_dtype = np.complex128 if complex_values else np.float64
    softening = 0.21

    reference = np.zeros(_out_shape(TARGETS, displacement, tail), dtype=output_dtype)
    _call(
        backend,
        method,
        SOURCES,
        TARGETS,
        MASS,
        displacement,
        softening=softening,
        chunk=None,
        out=reference,
    )

    for chunk in (1, 2, 3, 8):
        initial = np.linspace(-0.2, 0.3, reference.size).reshape(reference.shape)
        if complex_values:
            initial = (initial + 0.2j * initial[::-1]).astype(np.complex128)
        out = initial.copy()
        returned = _call(
            backend,
            method,
            SOURCES,
            TARGETS,
            MASS,
            displacement,
            softening=softening,
            chunk=chunk,
            out=out,
        )
        np.testing.assert_allclose(
            out,
            initial + reference,
            rtol=CHUNK_BUDGET["rtol"],
            atol=CHUNK_BUDGET["atol"],
        )
        assert returned is None
        assert out.dtype == output_dtype

    np.testing.assert_array_equal(SOURCES, sources_before)
    np.testing.assert_array_equal(TARGETS, targets_before)
    np.testing.assert_array_equal(MASS, mass_before)
    np.testing.assert_array_equal(displacement, displacement_before)


@pytest.mark.parametrize(
    "blocked_method,direct_method,tail", BLOCKED_CONTRACTIONS
)
@pytest.mark.parametrize("complex_values", [False, True])
def test_blocked_displacement_contractions_match_fused(
    blocked_method, direct_method, tail, complex_values
):
    backend = NumpyContractionBackend()
    displacement = DISPLACEMENT.copy()
    if complex_values:
        displacement = (displacement + 0.35j * displacement[::-1]).astype(
            np.complex128
        )
    dtype = np.complex128 if complex_values else np.float64
    blocked = np.zeros(_out_shape(TARGETS, displacement, tail), dtype=dtype)
    direct = np.zeros_like(blocked)

    _call(
        backend,
        direct_method,
        SOURCES,
        TARGETS,
        MASS,
        displacement,
        softening=0.17,
        chunk=2,
        out=direct,
    )
    _call(
        backend,
        blocked_method,
        SOURCES,
        TARGETS,
        MASS,
        displacement,
        softening=0.17,
        chunk=2,
        out=blocked,
    )

    np.testing.assert_allclose(
        blocked,
        direct,
        rtol=CHUNK_BUDGET["rtol"],
        atol=CHUNK_BUDGET["atol"],
    )


@pytest.mark.parametrize("observable", ["potential", "acceleration", "gradient"])
def test_generic_executor_uses_blocked_displacement_contracts(observable):
    class Provider:
        kind = "displacement"
        replayable = True
        random_access = True
        resident_bytes = 0

        def __init__(self):
            self.positions_m = SOURCES.copy()
            self.mass_kg = MASS.copy()
            self.displacement_m = DISPLACEMENT.copy()
            self.source_count = len(self.positions_m)
            self.parameter_count = self.displacement_m.shape[1]

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

    provider = Provider()
    plans = {
        strategy: ExecutionPlanner.plan(
            provider,
            target_count=len(TARGETS),
            observable=observable,
            result_dtype=np.float64,
            backend_workspace_bytes=0,
            target_block_size=2,
            parameter_block_size=2,
            source_block_size=3,
            kernel_chunk_size=2,
            strategy=strategy,
            memory_budget_bytes=100_000,
        )
        for strategy in ("fused", "blocked")
    }
    blocked_plan = plans["blocked"]
    output_components = {"potential": 1, "acceleration": 3, "gradient": 9}[
        observable
    ]
    blocked_output_tile = (
        blocked_plan.target_block_size
        * blocked_plan.parameter_block_size
        * output_components
        * np.dtype(np.float64).itemsize
    )
    blocked_geometry_tile = (
        blocked_plan.target_block_size
        * min(blocked_plan.source_block_size, blocked_plan.kernel_chunk_size)
        * 64
        * np.dtype(np.float64).itemsize
    )
    blocked_displacement_tile = (
        min(blocked_plan.source_block_size, blocked_plan.kernel_chunk_size)
        * blocked_plan.parameter_block_size
        * 3
        * np.dtype(np.float64).itemsize
    )
    assert blocked_plan.strategy_scratch_bytes == (
        2 * blocked_output_tile
        + blocked_geometry_tile
        + blocked_displacement_tile
    )
    results = {}
    for strategy, plan in plans.items():
        result = np.empty(plan.output_shape, dtype=plan.result_dtype)
        for target_slice, parameter_slice, values in GenericExecutor.iter_blocks(
            provider, TARGETS, plan, NumpyContractionBackend(), softening_m=0.17
        ):
            result[target_slice, parameter_slice] = values
        results[strategy] = result

    np.testing.assert_allclose(
        results["blocked"],
        results["fused"],
        rtol=CHUNK_BUDGET["rtol"],
        atol=CHUNK_BUDGET["atol"],
    )


@pytest.mark.parametrize("method,tail", BACKEND_CONTRACTIONS)
def test_complex_displacement_is_linear_without_conjugation(method, tail):
    backend = NumpyContractionBackend()
    complex_displacement = DISPLACEMENT + 0.4j * DISPLACEMENT[::-1]
    complex_out = np.zeros(
        _out_shape(TARGETS, complex_displacement, tail), dtype=np.complex128
    )
    real_out = np.zeros(_out_shape(TARGETS, DISPLACEMENT, tail), dtype=np.float64)
    imaginary_out = np.zeros_like(real_out)

    _call(
        backend,
        method,
        SOURCES,
        TARGETS,
        MASS,
        complex_displacement,
        softening=0.12,
        chunk=2,
        out=complex_out,
    )
    _call(
        backend,
        method,
        SOURCES,
        TARGETS,
        MASS,
        DISPLACEMENT,
        softening=0.12,
        chunk=2,
        out=real_out,
    )
    _call(
        backend,
        method,
        SOURCES,
        TARGETS,
        MASS,
        0.4 * DISPLACEMENT[::-1],
        softening=0.12,
        chunk=2,
        out=imaginary_out,
    )

    np.testing.assert_allclose(
        complex_out,
        real_out + 1j * imaginary_out,
        rtol=CHUNK_BUDGET["rtol"],
        atol=CHUNK_BUDGET["atol"],
    )


@pytest.mark.parametrize("method,tail", BACKEND_CONTRACTIONS)
def test_displacement_contract_collision_is_indexed_and_preflighted(method, tail):
    backend = NumpyContractionBackend()
    sources = np.array(
        [[4.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 1.0, 0.0]], dtype=np.float64
    )
    targets = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    mass = np.ones(3, dtype=np.float64)
    displacement = np.ones((3, 2, 3), dtype=np.float64)
    out = np.full(_out_shape(targets, displacement, tail), 7.0, dtype=np.float64)
    initial = out.copy()

    with pytest.raises(ValueError, match=r"target point 1.*source point 1"):
        _call(
            backend,
            method,
            sources,
            targets,
            mass,
            displacement,
            softening=0.0,
            chunk=1,
            out=out,
        )

    np.testing.assert_array_equal(out, initial)


INVALID_INPUTS = (
    ("complex_mass", TypeError, "real-valued"),
    ("boolean_mass", TypeError, "boolean"),
    ("negative_mass", ValueError, "non-negative"),
    ("infinite_mass", ValueError, "finite"),
    ("wrong_mass_shape", ValueError, "shape"),
    ("wrong_displacement_shape", ValueError, "shape"),
    ("nonfinite_displacement", ValueError, "finite"),
    ("boolean_displacement", TypeError, "boolean"),
    ("bad_softening", ValueError, "non-negative"),
    ("nan_softening", ValueError, "finite"),
    ("boolean_chunk", ValueError, "positive integer"),
    ("fractional_chunk", ValueError, "positive integer"),
    ("wrong_output_shape", ValueError, "shape"),
    ("wrong_output_dtype", TypeError, "float64"),
    ("nonfinite_output", ValueError, "finite"),
    ("readonly_output", ValueError, "writable"),
    ("aliased_output", ValueError, "share memory"),
)


@pytest.mark.parametrize("method,tail", BACKEND_CONTRACTIONS)
@pytest.mark.parametrize("case,exception,match", INVALID_INPUTS)
def test_displacement_contracts_validate_inputs(
    method, tail, case, exception, match
):
    backend = NumpyContractionBackend()
    sources = SOURCES.copy()
    targets = TARGETS.copy()
    mass = MASS.copy()
    displacement = DISPLACEMENT.copy()
    softening = 0.0
    chunk = 1
    out = np.zeros(_out_shape(targets, displacement, tail), dtype=np.float64)

    if case == "complex_mass":
        mass = mass.astype(np.complex128) + 1j
    elif case == "boolean_mass":
        mass = np.ones(len(sources), dtype=bool)
    elif case == "negative_mass":
        mass[1] = -1.0
    elif case == "infinite_mass":
        mass[1] = np.inf
    elif case == "wrong_mass_shape":
        mass = mass.reshape(2, 2)
    elif case == "wrong_displacement_shape":
        displacement = displacement[:, :, :2]
    elif case == "nonfinite_displacement":
        displacement[0, 0, 1] = np.nan
    elif case == "boolean_displacement":
        displacement = np.ones(displacement.shape, dtype=bool)
    elif case == "bad_softening":
        softening = -0.1
    elif case == "nan_softening":
        softening = np.nan
    elif case == "boolean_chunk":
        chunk = True
    elif case == "fractional_chunk":
        chunk = 1.5
    elif case == "wrong_output_shape":
        out = np.zeros((1, 1), dtype=np.float64)
    elif case == "wrong_output_dtype":
        out = np.zeros(_out_shape(targets, displacement, tail), dtype=np.float32)
    elif case == "nonfinite_output":
        out.flat[0] = np.nan
    elif case == "readonly_output":
        out.setflags(write=False)
    elif case == "aliased_output":
        backing = np.ones(max(out.size, displacement.size), dtype=np.float64)
        displacement = backing[: displacement.size].reshape(displacement.shape)
        out = backing[: out.size].reshape(out.shape)

    with pytest.raises(exception, match=match):
        _call(
            backend,
            method,
            sources,
            targets,
            mass,
            displacement,
            softening=softening,
            chunk=chunk,
            out=out,
        )


@pytest.mark.parametrize("method,tail", BACKEND_CONTRACTIONS)
def test_complex_displacement_requires_complex_accumulator(method, tail):
    backend = NumpyContractionBackend()
    displacement = DISPLACEMENT.astype(np.complex128) * (1.0 + 1.0j)
    with pytest.raises(TypeError, match="complex128"):
        _call(
            backend,
            method,
            SOURCES,
            TARGETS,
            MASS,
            displacement,
            softening=0.0,
            chunk=1,
            out=np.zeros(_out_shape(TARGETS, displacement, tail), dtype=np.float64),
        )
