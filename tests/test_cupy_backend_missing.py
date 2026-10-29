"""Tests for the cupy backend under missing/unavailable conditions.

Covers the two failure modes of the error contract
(gpu_backend_adr.md §5): cupy not installed -> ImportError with an
install hint, and cupy installed but no usable CUDA device ->
RuntimeError. These tests run in every environment (no GPU needed);
the RuntimeError tests additionally require cupy to be installed.
"""

import subprocess
import sys

import pytest

from gragra.coherent.weights import plane_wave_mass_weights
from gragra.gravity_sources import PointMass
from gragra.observables import point_potential
from gragra.observables.coherent import coherent_point_acceleration


@pytest.fixture
def hide_cupy(monkeypatch):
    """Simulate an environment where cupy is not installed.

    Order-independent: also drops the wrapper module from the import
    cache and the parent-package attribute, so a wrapper imported by an
    earlier test cannot leak through (mirrors the cpp missing fixture).
    """
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.delitem(
        sys.modules, "gragra.kernels._cupy_inverse_square", raising=False
    )
    kernels_pkg = sys.modules.get("gragra.kernels")
    if kernels_pkg is not None and hasattr(kernels_pkg, "_cupy_inverse_square"):
        monkeypatch.delattr(kernels_pkg, "_cupy_inverse_square")


def test_cupy_backend_missing_raises_import_error(hide_cupy):  # noqa: ARG001
    """backend='cupy' without cupy installed raises a clean ImportError."""
    source = PointMass(1.0, [0.0, 0.0, 0.0])
    target = [[1.0, 0.0, 0.0]]

    with pytest.raises(ImportError) as excinfo:
        point_potential(source, target, backend="cupy")

    msg = str(excinfo.value)
    assert "cupy is not installed" in msg
    assert "cupy-cuda12x" in msg  # install hint with wheel selection


def test_unknown_backend_message_lists_cupy():
    """The point-direct unknown-backend ValueError lists all four backends."""
    source = PointMass(1.0, [0.0, 0.0, 0.0])
    target = [[1.0, 0.0, 0.0]]

    with pytest.raises(
        ValueError,
        match=r"must be 'numpy', 'numba', 'cpp', or 'cupy'",
    ):
        point_potential(source, target, backend="bogus")


def test_coherent_cupy_missing_raises_import_error(hide_cupy):  # noqa: ARG001
    """coherent path backend='cupy' without cupy raises ImportError."""
    with pytest.raises(ImportError) as excinfo:
        coherent_point_acceleration(
            [[0.0, 0.0, 0.0]],
            [[1.0, 2.0]],
            [[1.0, 0.0, 0.0]],
            backend="cupy",
        )
    assert "cupy is not installed" in str(excinfo.value)


def test_plane_wave_weights_cupy_missing_raises_import_error(hide_cupy):  # noqa: ARG001
    """plane-wave weights backend='cupy' without cupy raises ImportError."""
    with pytest.raises(ImportError) as excinfo:
        plane_wave_mass_weights(
            [1.0],
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [1.0],
            backend="cupy",
        )
    assert "cupy is not installed" in str(excinfo.value)


def test_cupy_lazy_load_for_numpy_backend():
    """cupy is never imported on the default numpy path (core purity)."""
    script = (
        "import sys; "
        "from gragra.gravity_sources import PointMass; "
        "from gragra.observables import point_potential; "
        "point_potential(PointMass(1.0, [0.0, 0.0, 0.0]), [[1.0, 0.0, 0.0]]); "
        "assert 'cupy' not in sys.modules, 'cupy leaked into the numpy path'; "
        "assert 'gragra.kernels._cupy_inverse_square' not in sys.modules"
    )
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def _reset_device_check_cache():
    from gragra.kernels import _cupy_inverse_square as mod

    mod._ensure_cuda_device.cache_clear()


def test_cupy_device_unavailable_raises_runtime_error(monkeypatch):
    """cupy installed but CUDA runtime init fails -> RuntimeError (chained)."""
    cupy = pytest.importorskip("cupy")

    def fake_get_device_count():
        # cudaErrorInsufficientDriver (35): driver/runtime mismatch
        raise cupy.cuda.runtime.CUDARuntimeError(35)

    _reset_device_check_cache()
    monkeypatch.setattr(cupy.cuda.runtime, "getDeviceCount", fake_get_device_count)
    try:
        with pytest.raises(RuntimeError, match="no usable CUDA device"):
            point_potential(
                PointMass(1.0, [0.0, 0.0, 0.0]),
                [[1.0, 0.0, 0.0]],
                backend="cupy",
            )
    finally:
        # Do not leave the simulated failure state cached for later tests
        _reset_device_check_cache()


def test_cupy_unexpected_probe_error_is_not_misattributed(monkeypatch):
    """A non-CUDA failure while probing must not claim 'no usable device'."""
    cupy = pytest.importorskip("cupy")

    def fake_get_device_count():
        raise AttributeError("simulated cupy API drift")

    _reset_device_check_cache()
    monkeypatch.setattr(cupy.cuda.runtime, "getDeviceCount", fake_get_device_count)
    try:
        with pytest.raises(RuntimeError, match="unexpected error"):
            point_potential(
                PointMass(1.0, [0.0, 0.0, 0.0]),
                [[1.0, 0.0, 0.0]],
                backend="cupy",
            )
    finally:
        _reset_device_check_cache()


def test_cupy_input_validation_precedes_device_check():
    """Host-side input validation raises before any device probing.

    Runs with cupy installed but needs no GPU: the shared
    ``_validate_inputs`` rejects bad inputs before ``_ensure_cuda_device``.
    """
    pytest.importorskip("cupy")

    source = PointMass(1.0, [0.0, 0.0, 0.0])
    target = [[1.0, 0.0, 0.0]]

    with pytest.raises(ValueError, match="softening_m must be non-negative"):
        point_potential(source, target, softening_m=-0.1, backend="cupy")

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        point_potential(source, target, chunk_size=0, backend="cupy")


def test_cupy_zero_devices_raises_runtime_error(monkeypatch):
    """cupy installed but zero visible devices -> RuntimeError."""
    cupy = pytest.importorskip("cupy")

    _reset_device_check_cache()
    monkeypatch.setattr(cupy.cuda.runtime, "getDeviceCount", lambda: 0)
    try:
        with pytest.raises(RuntimeError, match="no CUDA device is visible"):
            point_potential(
                PointMass(1.0, [0.0, 0.0, 0.0]),
                [[1.0, 0.0, 0.0]],
                backend="cupy",
            )
    finally:
        _reset_device_check_cache()
