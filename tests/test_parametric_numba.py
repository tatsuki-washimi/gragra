"""Numba parametric kernel tests: agreement with the numpy reference.

These verify that ``kernels._numba_parametric`` reproduces the numpy
``kernels.parametric`` reference across observables, dtypes, and weight
parametric shapes, with matching collision/validation contracts.
"""

import numpy as np
import pytest

pytest.importorskip("numba")

from gragra.kernels import _numba_parametric as nb  # noqa: E402
from gragra.kernels import parametric as ref  # noqa: E402

CONTRACTS = [
    ("potential", ref.potential_contract_parametric, nb.potential_contract_parametric),
    (
        "acceleration",
        ref.acceleration_contract_parametric,
        nb.acceleration_contract_parametric,
    ),
    ("gradient", ref.gradient_contract_parametric, nb.gradient_contract_parametric),
]
CONTRACT_IDS = [c[0] for c in CONTRACTS]
REF_NB = [(c[1], c[2]) for c in CONTRACTS]  # (ref_fn, nb_fn) pairs
NB_ONLY = [c[2] for c in CONTRACTS]  # nb_fn only

# Parametric weight shapes P (excluding the leading source axis N).
P_SHAPES = [(), (4,), (3, 2)]


def _make_case(rng, n, m, p_shape, complex_weights):
    sources = rng.random((n, 3)) * 10.0
    targets = rng.random((m, 3)) * 10.0 + 50.0  # well separated from sources
    w_shape = (n, *p_shape)
    w = rng.random(w_shape)
    if complex_weights:
        w = w + 1j * rng.random(w_shape)
    return sources, w, targets


@pytest.mark.parametrize("ref_fn,nb_fn", REF_NB, ids=CONTRACT_IDS)
@pytest.mark.parametrize("p_shape", P_SHAPES)
@pytest.mark.parametrize("complex_weights", [False, True])
def test_numba_matches_numpy(ref_fn, nb_fn, p_shape, complex_weights):
    rng = np.random.default_rng(0)
    sources, w, targets = _make_case(rng, 7, 5, p_shape, complex_weights)

    expected = ref_fn(sources, w, targets, softening_m=0.3)
    actual = nb_fn(sources, w, targets, softening_m=0.3)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
    assert actual.flags.writeable is False


@pytest.mark.parametrize("nb_fn", NB_ONLY, ids=CONTRACT_IDS)
def test_numba_dtype_follows_weights(nb_fn):
    rng = np.random.default_rng(1)
    sources, w_real, targets = _make_case(rng, 4, 3, (2,), False)
    sources, w_cplx, targets = _make_case(rng, 4, 3, (2,), True)
    assert nb_fn(sources, w_real, targets, softening_m=0.1).dtype == np.float64
    assert nb_fn(sources, w_cplx, targets, softening_m=0.1).dtype == np.complex128


@pytest.mark.parametrize("nb_fn", NB_ONLY, ids=CONTRACT_IDS)
def test_numba_chunk_invariance(nb_fn):
    rng = np.random.default_rng(2)
    sources, w, targets = _make_case(rng, 9, 4, (3,), True)
    base = nb_fn(sources, w, targets, softening_m=0.2, chunk_size=None)
    for cs in (1, 3, 9, 20):
        out = nb_fn(sources, w, targets, softening_m=0.2, chunk_size=cs)
        np.testing.assert_allclose(out, base, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("ref_fn,nb_fn", REF_NB, ids=CONTRACT_IDS)
def test_numba_empty_source_zeros(ref_fn, nb_fn):
    sources = np.zeros((0, 3))
    targets = [[1.0, 2.0, 3.0]]
    w = np.zeros((0, 4))
    expected = ref_fn(sources, w, targets)
    actual = nb_fn(sources, w, targets)
    assert actual.shape == expected.shape
    assert np.all(actual == 0.0)


@pytest.mark.parametrize("nb_fn", NB_ONLY, ids=CONTRACT_IDS)
def test_numba_collision_matches(nb_fn):
    # Two targets collide; the lowest target index / lowest source index wins.
    sources = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    targets = np.array([[5.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # target 1 hits source 1
    w = np.ones((2, 2))
    with pytest.raises(ValueError, match=r"target point 1.*source point 1.*zero soft"):
        nb_fn(sources, w, targets, softening_m=0.0)


@pytest.mark.parametrize("nb_fn", NB_ONLY, ids=CONTRACT_IDS)
def test_numba_invalid_inputs(nb_fn):
    sources = np.array([[0.0, 0.0, 0.0]])
    targets = np.array([[1.0, 0.0, 0.0]])
    w = np.ones((1, 2))
    with pytest.raises(ValueError):
        nb_fn(sources, w, targets, softening_m=-1.0)
    with pytest.raises(ValueError):
        nb_fn(sources, w, targets, chunk_size=0)
    with pytest.raises(ValueError):
        nb_fn(sources, np.ones((2, 2)), targets)  # first-axis mismatch
    with pytest.raises(ValueError):
        nb_fn(sources, np.array([np.inf, 1.0]).reshape(1, 2), targets)
