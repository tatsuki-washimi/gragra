"""C++ parametric kernel tests: 3-backend agreement and OpenMP determinism.

These verify that ``kernels._cpp_parametric`` reproduces the numpy
``kernels.parametric`` reference (and the numba backend when available) across
observables, dtypes, and weight parametric shapes, with matching
collision/validation contracts and thread-count-independent results.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("gragra._cpp_backend")

from gragra.kernels import _cpp_parametric as cpp  # noqa: E402
from gragra.kernels import parametric as ref  # noqa: E402

CONTRACTS = [
    ("potential", ref.potential_contract_parametric, cpp.potential_contract_parametric),
    (
        "acceleration",
        ref.acceleration_contract_parametric,
        cpp.acceleration_contract_parametric,
    ),
    ("gradient", ref.gradient_contract_parametric, cpp.gradient_contract_parametric),
]
CONTRACT_IDS = [c[0] for c in CONTRACTS]
REF_CPP = [(c[1], c[2]) for c in CONTRACTS]  # (ref_fn, cpp_fn) pairs
CPP_ONLY = [c[2] for c in CONTRACTS]  # cpp_fn only

P_SHAPES = [(), (4,), (3, 2)]


def _make_case(rng, n, m, p_shape, complex_weights):
    sources = rng.random((n, 3)) * 10.0
    targets = rng.random((m, 3)) * 10.0 + 50.0
    w_shape = (n, *p_shape)
    w = rng.random(w_shape)
    if complex_weights:
        w = w + 1j * rng.random(w_shape)
    return sources, w, targets


@pytest.mark.parametrize("ref_fn,cpp_fn", REF_CPP, ids=CONTRACT_IDS)
@pytest.mark.parametrize("p_shape", P_SHAPES)
@pytest.mark.parametrize("complex_weights", [False, True])
def test_cpp_matches_numpy(ref_fn, cpp_fn, p_shape, complex_weights):
    rng = np.random.default_rng(0)
    sources, w, targets = _make_case(rng, 7, 5, p_shape, complex_weights)

    expected = ref_fn(sources, w, targets, softening_m=0.3)
    actual = cpp_fn(sources, w, targets, softening_m=0.3)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
    assert actual.flags.writeable is False


@pytest.mark.parametrize("name", [c[0] for c in CONTRACTS])
def test_cpp_matches_numba(name):
    nb = pytest.importorskip("gragra.kernels._numba_parametric")
    nb_fn = getattr(nb, f"{name}_contract_parametric")
    cpp_fn = getattr(cpp, f"{name}_contract_parametric")
    rng = np.random.default_rng(3)
    sources, w, targets = _make_case(rng, 6, 4, (3,), True)
    np.testing.assert_allclose(
        cpp_fn(sources, w, targets, softening_m=0.2),
        nb_fn(sources, w, targets, softening_m=0.2),
        rtol=1e-12,
        atol=1e-15,
    )


@pytest.mark.parametrize("cpp_fn", CPP_ONLY, ids=CONTRACT_IDS)
def test_cpp_chunk_invariance(cpp_fn):
    rng = np.random.default_rng(2)
    sources, w, targets = _make_case(rng, 9, 4, (3,), True)
    base = cpp_fn(sources, w, targets, softening_m=0.2, chunk_size=None)
    for cs in (1, 3, 9, 20):
        out = cpp_fn(sources, w, targets, softening_m=0.2, chunk_size=cs)
        np.testing.assert_allclose(out, base, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("ref_fn,cpp_fn", REF_CPP, ids=CONTRACT_IDS)
def test_cpp_empty_inputs(ref_fn, cpp_fn):
    # N=0
    sources = np.zeros((0, 3))
    targets = [[1.0, 2.0, 3.0]]
    w = np.zeros((0, 4))
    expected = ref_fn(sources, w, targets)
    actual = cpp_fn(sources, w, targets)
    assert actual.shape == expected.shape
    assert np.all(actual == 0.0)
    # M=0
    sources2 = np.array([[0.0, 0.0, 0.0]])
    w2 = np.ones((1, 4))
    out_m = cpp_fn(sources2, w2, np.zeros((0, 3)))
    assert out_m.shape[0] == 0


@pytest.mark.parametrize("cpp_fn", CPP_ONLY, ids=CONTRACT_IDS)
def test_cpp_collision_matches(cpp_fn):
    sources = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    targets = np.array([[5.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # target 1 hits source 1
    w = np.ones((2, 2))
    with pytest.raises(ValueError, match=r"target point 1.*source point 1.*zero soft"):
        cpp_fn(sources, w, targets, softening_m=0.0)


@pytest.mark.parametrize("cpp_fn", CPP_ONLY, ids=CONTRACT_IDS)
def test_cpp_invalid_inputs(cpp_fn):
    sources = np.array([[0.0, 0.0, 0.0]])
    targets = np.array([[1.0, 0.0, 0.0]])
    w = np.ones((1, 2))
    with pytest.raises(ValueError):
        cpp_fn(sources, w, targets, softening_m=-1.0)
    with pytest.raises(ValueError):
        cpp_fn(sources, w, targets, chunk_size=0)
    with pytest.raises(ValueError):
        cpp_fn(sources, np.ones((2, 2)), targets)  # first-axis mismatch


# OpenMP determinism: the result must not depend on the thread count.
_DETERMINISM_SNIPPET = """
import numpy as np
from gragra.kernels import _cpp_parametric as cpp
rng = np.random.default_rng(7)
n, m = 40, 12
sources = rng.random((n, 3)) * 10.0
targets = rng.random((m, 3)) * 10.0 + 50.0
w = rng.random((n, 3, 2)) + 1j * rng.random((n, 3, 2))
out = cpp.gradient_contract_parametric(sources, w, targets, softening_m=0.1)
np.save(OUT_PATH, out)
"""


def test_cpp_openmp_determinism(tmp_path):
    out2 = tmp_path / "t2.npy"
    out4 = tmp_path / "t4.npy"
    for threads, out_path in (("2", out2), ("4", out4)):
        env = {**os.environ, "OMP_NUM_THREADS": threads}
        script = f"OUT_PATH = {str(out_path)!r}\n" + _DETERMINISM_SNIPPET
        subprocess.run([sys.executable, "-c", script], env=env, check=True)
    a = np.load(out2)
    b = np.load(out4)
    np.testing.assert_array_equal(a, b)
