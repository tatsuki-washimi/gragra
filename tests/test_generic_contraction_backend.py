"""Independent checks for the generic NumPy scalar contraction backend."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gragra.kernels.generic_backend import (
    ContractionBackend,
    NumpyContractionBackend,
)

# Declared before running comparisons. The single-source reference evaluates
# the closed-form kernels directly; the multi-source reference evaluates each
# geometric term in longdouble and sums terms with math.fsum. The budgets allow
# double-precision kernel rounding and the changed summation grouping, while
# remaining small enough to reject sign, axis, or tensor-index errors.
REFERENCE_BUDGET = {
    "single_point": {
        "reference": "closed-form point kernel",
        "rtol": 3e-14,
        "atol": 3e-15,
    },
    "multi_source": {
        "reference": "longdouble point terms plus math.fsum",
        "rtol": 3e-13,
        "atol": 3e-14,
    },
}

CONTRACTIONS = (
    ("potential_contract", "potential", ()),
    ("acceleration_contract", "acceleration", (3,)),
    ("gradient_contract", "gradient", (3, 3)),
)


def _reference_kernel(source, target, observable, softening):
    delta = np.asarray(source, dtype=np.longdouble) - np.asarray(
        target, dtype=np.longdouble
    )
    q = np.sum(delta * delta, dtype=np.longdouble) + np.longdouble(softening) ** 2
    if observable == "potential":
        return np.asarray(-1 / np.sqrt(q), dtype=np.longdouble)
    if observable == "acceleration":
        return delta / (q**np.longdouble(1.5))
    identity = np.eye(3, dtype=np.longdouble)
    return (
        3 * np.outer(delta, delta) / (q**np.longdouble(2.5))
        - identity / (q**np.longdouble(1.5))
    )


def _single_point_reference(sources, targets, weights, observable, softening):
    tail = {"potential": (), "acceleration": (3,), "gradient": (3, 3)}[
        observable
    ]
    result = np.empty((len(targets), weights.shape[1]) + tail, dtype=weights.dtype)
    for target_index, target in enumerate(targets):
        kernel = _reference_kernel(sources[0], target, observable, softening)
        for parameter in range(weights.shape[1]):
            result[target_index, parameter] = weights[0, parameter] * kernel
    return result


def _fsum(values):
    values = [complex(value) for value in values]
    real = math.fsum(value.real for value in values)
    imag = math.fsum(value.imag for value in values)
    return complex(real, imag) if any(value.imag for value in values) else real


def _multi_source_fsum_reference(
    sources, targets, weights, observable, softening
):
    tail = {"potential": (), "acceleration": (3,), "gradient": (3, 3)}[
        observable
    ]
    result = np.zeros((len(targets), weights.shape[1]) + tail, dtype=weights.dtype)
    component_indices = list(np.ndindex(tail)) if tail else [()]
    for target_index, target in enumerate(targets):
        kernels = [
            _reference_kernel(source, target, observable, softening)
            for source in sources
        ]
        for parameter in range(weights.shape[1]):
            for component in component_indices:
                terms = [
                    weights[source_index, parameter]
                    * float(kernel[component] if component else kernel)
                    for source_index, kernel in enumerate(kernels)
                ]
                result[(target_index, parameter) + component] = _fsum(terms)
    return result


def _output_shape(targets, weights, tail):
    return (len(targets), weights.shape[1]) + tail


def _call(backend, name, sources, targets, weights, *, softening, chunk, out):
    return getattr(backend, name)(
        sources,
        targets,
        weights,
        softening_m=softening,
        chunk_size=chunk,
        out=out,
    )


def _data(*, complex_values=False):
    sources = np.array(
        [
            [1.0, 0.2, -0.3],
            [-0.7, 2.1, 0.4],
            [2.2, -1.0, 0.8],
            [0.3, 1.5, -1.2],
            [-2.1, -0.4, 1.3],
        ],
        dtype=np.float64,
    )
    targets = np.array([[0.1, -1.4, 0.2], [3.3, 0.8, -2.0]], dtype=np.float64)
    weights = np.array(
        [
            [0.6, -1.2, 0.4],
            [-0.5, 0.7, 1.1],
            [1.3, -0.2, -0.8],
            [-0.9, 0.3, 0.5],
            [0.2, 1.4, -0.6],
        ],
        dtype=np.float64,
    )
    if complex_values:
        weights = (weights + 1j * weights[::-1]).astype(np.complex128)
    return sources, targets, weights


def test_numpy_backend_satisfies_scalar_contraction_protocol():
    assert isinstance(NumpyContractionBackend(), ContractionBackend)


@pytest.mark.parametrize("method,observable,tail", CONTRACTIONS)
@pytest.mark.parametrize("complex_values", [False, True])
def test_single_source_matches_closed_form_and_adds_to_out(
    method, observable, tail, complex_values
):
    backend = NumpyContractionBackend()
    sources = np.array([[1.75, -0.5, 2.25]], dtype=np.float64)
    targets = np.array([[0.25, 1.0, -0.75], [-2.0, 0.5, 0.5]], dtype=np.float64)
    weights = np.array([[2.5, -0.75]], dtype=np.float64)
    if complex_values:
        weights = (weights + 0.5j * weights[:, ::-1]).astype(np.complex128)
    softening = 0.2
    initial = np.full(_output_shape(targets, weights, tail), 0.125, dtype=weights.dtype)
    out = initial.copy()

    returned = _call(
        backend,
        method,
        sources,
        targets,
        weights,
        softening=softening,
        chunk=1,
        out=out,
    )
    expected = _single_point_reference(
        sources, targets, weights, observable, softening
    )
    budget = REFERENCE_BUDGET["single_point"]

    assert budget["reference"] == "closed-form point kernel"
    assert returned is None
    assert out.dtype == weights.dtype
    np.testing.assert_allclose(
        out,
        initial + expected,
        rtol=budget["rtol"],
        atol=budget["atol"],
    )


@pytest.mark.parametrize("method,observable,tail", CONTRACTIONS)
@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("chunk", [1, 2, 4, None])
def test_multi_source_matches_fsum_reference_across_chunks(
    method, observable, tail, complex_values, chunk
):
    backend = NumpyContractionBackend()
    sources, targets, weights = _data(complex_values=complex_values)
    sources_before = sources.copy()
    targets_before = targets.copy()
    weights_before = weights.copy()
    softening = 0.17
    initial = np.linspace(
        -0.2,
        0.3,
        num=np.prod(_output_shape(targets, weights, tail)),
        dtype=np.float64,
    ).reshape(_output_shape(targets, weights, tail))
    if complex_values:
        initial = (initial + 0.25j * initial[::-1]).astype(np.complex128)
    out = initial.copy()

    returned = _call(
        backend,
        method,
        sources,
        targets,
        weights,
        softening=softening,
        chunk=chunk,
        out=out,
    )
    expected = _multi_source_fsum_reference(
        sources, targets, weights, observable, softening
    )
    budget = REFERENCE_BUDGET["multi_source"]

    assert budget["reference"] == "longdouble point terms plus math.fsum"
    assert returned is None
    assert out.dtype == weights.dtype
    np.testing.assert_allclose(
        out,
        initial + expected,
        rtol=budget["rtol"],
        atol=budget["atol"],
    )
    np.testing.assert_array_equal(sources, sources_before)
    np.testing.assert_array_equal(targets, targets_before)
    np.testing.assert_array_equal(weights, weights_before)


@pytest.mark.parametrize("method,_observable,tail", CONTRACTIONS)
def test_zero_softening_collision_reports_global_indices_and_keeps_out(
    method, _observable, tail
):
    backend = NumpyContractionBackend()
    sources = np.array(
        [[4.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 1.0, 0.0]], dtype=np.float64
    )
    targets = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    weights = np.ones((3, 2), dtype=np.float64)
    out = np.full(_output_shape(targets, weights, tail), 7.0, dtype=np.float64)
    initial = out.copy()

    with pytest.raises(ValueError, match=r"target point 1.*source point 1"):
        _call(
            backend,
            method,
            sources,
            targets,
            weights,
            softening=0.0,
            chunk=1,
            out=out,
        )

    np.testing.assert_array_equal(out, initial)


@pytest.mark.parametrize(
    "changes,exception,match",
    [
        ({"sources": np.array([1.0, 2.0, 3.0])}, ValueError, "shape"),
        ({"sources": np.array([[1.0 + 1.0j, 0.0, 0.0]])}, TypeError, "real"),
        ({"sources": np.array([[np.nan, 0.0, 0.0]])}, ValueError, "finite"),
        ({"targets": np.array([[0.0, np.inf, 0.0]])}, ValueError, "finite"),
        ({"weights": np.array([1.0])}, ValueError, "shape"),
        ({"weights": np.array([[True]])}, TypeError, "boolean"),
        ({"weights": np.array([[np.inf]])}, ValueError, "finite"),
        ({"softening": -1.0}, ValueError, "non-negative"),
        ({"softening": np.nan}, ValueError, "finite"),
        ({"chunk": True}, ValueError, "positive integer"),
        ({"chunk": 1.5}, ValueError, "positive integer"),
        ({"out": np.zeros((1, 2), dtype=np.float64)}, ValueError, "shape"),
        ({"out": np.zeros((1, 1), dtype=np.float32)}, TypeError, "float64"),
        ({"out": np.array([[np.nan]], dtype=np.float64)}, ValueError, "finite"),
        ({"readonly_out": True}, ValueError, "writable"),
    ],
)
def test_potential_contract_rejects_invalid_inputs(changes, exception, match):
    backend = NumpyContractionBackend()
    sources = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    targets = np.array([[0.0, 2.0, 0.0]], dtype=np.float64)
    weights = np.array([[1.0]], dtype=np.float64)
    kwargs = {"softening": 0.0, "chunk": 1}
    out = np.zeros((1, 1), dtype=np.float64)

    sources = changes.get("sources", sources)
    targets = changes.get("targets", targets)
    weights = changes.get("weights", weights)
    kwargs["softening"] = changes.get("softening", kwargs["softening"])
    kwargs["chunk"] = changes.get("chunk", kwargs["chunk"])
    out = changes.get("out", out)
    if changes.get("readonly_out"):
        out.setflags(write=False)

    with pytest.raises(exception, match=match):
        _call(
            backend,
            "potential_contract",
            sources,
            targets,
            weights,
            softening=kwargs["softening"],
            chunk=kwargs["chunk"],
            out=out,
        )


def test_complex_weights_require_complex_accumulator():
    backend = NumpyContractionBackend()
    with pytest.raises(TypeError, match="complex128"):
        _call(
            backend,
            "potential_contract",
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[0.0, 1.0, 0.0]]),
            np.array([[1.0 + 1.0j]], dtype=np.complex128),
            softening=0.0,
            chunk=1,
            out=np.zeros((1, 1), dtype=np.float64),
        )
