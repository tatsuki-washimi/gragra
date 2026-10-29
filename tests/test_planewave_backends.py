"""Tests for plane-wave weight generation backends (numpy, numba, cpp)."""

import os
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest

from gragra.coherent.weights import plane_wave_mass_weights


def _available_backends():
    backends = ["numpy"]
    try:
        import numba  # noqa: F401

        backends.append("numba")
    except ImportError:
        pass
    try:
        import gragra._cpp_backend  # noqa: F401

        backends.append("cpp")
    except ImportError:
        pass
    return backends


@pytest.mark.parametrize("backend", _available_backends())
@pytest.mark.parametrize("complex_weights", [False, True])
@pytest.mark.parametrize("sign", [-1, 1])
def test_planewave_backends_equivalence(backend, complex_weights, sign):
    rng = np.random.default_rng(42)
    n = 10
    ndir = 4
    nf = 5

    if complex_weights:
        base_weights = rng.random(n) + 1j * rng.random(n)
    else:
        base_weights = rng.random(n)

    positions = rng.random((n, 3)) * 10.0
    directions = rng.random((ndir, 3))
    # normalize directions
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    wavenumbers = rng.random(nf) * 2.0 + 0.1
    phase0 = 0.35

    # Numpy reference
    w_ref = plane_wave_mass_weights(
        base_weights,
        positions,
        directions,
        wavenumbers,
        phase0=phase0,
        sign=sign,
        backend="numpy",
    )

    # Selected backend
    w_test = plane_wave_mass_weights(
        base_weights,
        positions,
        directions,
        wavenumbers,
        phase0=phase0,
        sign=sign,
        backend=backend,
    )

    assert w_test.shape == (n, ndir, nf)
    assert w_test.dtype == np.complex128
    assert not w_test.flags.writeable
    np.testing.assert_allclose(w_test, w_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("backend", _available_backends())
@pytest.mark.parametrize("n, ndir, nf", [(0, 3, 4), (5, 0, 4), (5, 3, 0), (0, 0, 0)])
def test_planewave_backends_empty_shapes(backend, n, ndir, nf):
    base_weights = np.ones(n, dtype=np.float64)
    positions = np.zeros((n, 3), dtype=np.float64)
    directions = np.zeros((ndir, 3), dtype=np.float64)
    if ndir > 0:
        directions[:, 0] = 1.0  # unit vector
    wavenumbers = np.ones(nf, dtype=np.float64)

    w_test = plane_wave_mass_weights(
        base_weights,
        positions,
        directions,
        wavenumbers,
        phase0=0.0,
        sign=1,
        backend=backend,
    )
    assert w_test.shape == (n, ndir, nf)
    assert w_test.dtype == np.complex128
    assert not w_test.flags.writeable
    if w_test.size > 0:
        assert np.all(w_test == 0)


def test_planewave_backend_validation_errors():
    base_weights = np.array([1.5, 2.5])
    positions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    directions = np.array([[1.0, 0.0, 0.0]])
    wavenumber = 0.1

    # Unknown backend
    err_msg = "Unsupported backend: must be 'numpy', 'numba', 'cpp', or 'cupy'"
    with pytest.raises(ValueError, match=err_msg):
        plane_wave_mass_weights(
            base_weights,
            positions,
            directions,
            wavenumber,
            backend="unknown",
        )

    # Invalid sign types and values
    for invalid_sign in [True, False, 0, 2, -2, [1]]:
        with pytest.raises((TypeError, ValueError)):
            plane_wave_mass_weights(
                base_weights,
                positions,
                directions,
                wavenumber,
                sign=invalid_sign,
            )

    # Invalid phase0 types and values
    for invalid_phase0 in [True, False, np.nan, np.inf, -np.inf, "0.1"]:
        with pytest.raises((TypeError, ValueError)):
            plane_wave_mass_weights(
                base_weights,
                positions,
                directions,
                wavenumber,
                phase0=invalid_phase0,
            )


_DETERMINISM_SNIPPET = """
import numpy as np
from gragra.coherent.weights import plane_wave_mass_weights
rng = np.random.default_rng(42)
n = 100
ndir = 10
nf = 20
base_weights = rng.random(n)
positions = rng.random((n, 3))
directions = rng.random((ndir, 3))
directions /= np.linalg.norm(directions, axis=1, keepdims=True)
wavenumbers = rng.random(nf)
w = plane_wave_mass_weights(
    base_weights, positions, directions, wavenumbers, backend="cpp"
)
np.save(OUT_PATH, w)
"""


@pytest.mark.skipif(
    "cpp" not in _available_backends(), reason="C++ backend not available"
)
def test_cpp_planewave_openmp_determinism(tmp_path, monkeypatch):
    # Ensure OMP_NUM_THREADS in the current process is not mutated
    monkeypatch.setenv("OMP_NUM_THREADS", "1")

    out1 = tmp_path / "t1.npy"
    out4 = tmp_path / "t4.npy"

    for threads, out_path in (("1", out1), ("4", out4)):
        env = {**os.environ, "OMP_NUM_THREADS": threads}
        script = f"OUT_PATH = {str(out_path)!r}\n" + _DETERMINISM_SNIPPET
        subprocess.run([sys.executable, "-c", script], env=env, check=True)

    w_t1 = np.load(out1)
    w_t4 = np.load(out4)
    np.testing.assert_array_equal(w_t1, w_t4)


def test_missing_backend_importerror():
    base_weights = np.array([1.5])
    positions = np.array([[1.0, 0.0, 0.0]])
    directions = np.array([[1.0, 0.0, 0.0]])
    wavenumber = 0.1

    # Remove modules from sys.modules cache to force re-importing inside dispatch
    sys.modules.pop("gragra.kernels._numba_planewave", None)
    sys.modules.pop("numba", None)

    # Mock numba import failure
    err_numba = "numba is not installed in the current environment"
    with (
        mock.patch.dict(sys.modules, {"numba": None}),
        pytest.raises(ImportError, match=err_numba),
    ):
        plane_wave_mass_weights(
            base_weights, positions, directions, wavenumber, backend="numba"
        )

    sys.modules.pop("gragra.kernels._cpp_planewave", None)
    sys.modules.pop("gragra._cpp_backend", None)

    # Mock C++ backend import failure
    err_cpp = r"the gragra C\+\+ backend is not built"
    with (
        mock.patch.dict(sys.modules, {"gragra._cpp_backend": None}),
        pytest.raises(ImportError, match=err_cpp),
    ):
        plane_wave_mass_weights(
            base_weights, positions, directions, wavenumber, backend="cpp"
        )
