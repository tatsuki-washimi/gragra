# ruff: noqa: E402, N806, E501
"""Tests for the plane-wave CuPy backend (Phase C-3).

C-3 cancellation-sensitive phase measurements are recorded in
docs/design/cupy_backend.md §8.5 (the sole source of record — this module
used to embed its own copy of that table in this docstring; the copy was
removed in round-3 review to stop dual-maintenance drift). Each measurement
test below carries an inline registry-id-tagged comment mapping it to its
row (see tests/test_cupy_registry_contract.py for the machine check).
"""

import numpy as np
import pytest

try:
    import cupy as cp

    HAS_CUPY = True
    try:
        HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        HAS_DEVICE = False
except ImportError:
    HAS_CUPY = False
    HAS_DEVICE = False

# Module-level skip if CuPy or GPU is missing
pytestmark = pytest.mark.skipif(
    not (HAS_CUPY and HAS_DEVICE), reason="cupy or GPU device not available"
)

from gragra.coherent.weights import plane_wave_mass_weights
from gragra.waves.phase import plane_wave_complex_factor


class ShapeWrapper:
    """Wrapper that delegates to an object but overrides shape and len."""

    def __init__(self, obj, shape):
        self._obj = obj
        self.shape = shape
        self.dtype = obj.dtype

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def __len__(self):
        return self.shape[0]

    def __le__(self, other):
        return self._obj <= other

    def __lt__(self, other):
        return self._obj < other

    def __ge__(self, other):
        return self._obj >= other

    def __gt__(self, other):
        return self._obj > other

    def __eq__(self, other):
        return self._obj == other

    def __ne__(self, other):
        return self._obj != other


def test_cupy_planewave_equivalence():
    """Verify that plane_wave_mass_weights on GPU matches NumPy reference."""
    rng = np.random.default_rng(12345)

    # Check both real and complex weights
    for is_complex in [False, True]:
        n, n_dir, n_f = 20, 5, 4

        pos = rng.uniform(-10.0, 10.0, (n, 3))
        dirs = rng.uniform(-1.0, 1.0, (n_dir, 3))
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        wavenumber = rng.uniform(0.1, 5.0, (n_f,))

        if is_complex:
            w = rng.uniform(1.0, 100.0, (n,)) + 1j * rng.uniform(1.0, 100.0, (n,))
        else:
            w = rng.uniform(1.0, 100.0, (n,))

        phase0 = 0.5
        sign = -1

        # Check various chunk_size configurations (2, 5, and None)
        for chunk in [2, 5, None]:
            w_ref = plane_wave_mass_weights(
                w, pos, dirs, wavenumber, phase0, sign, backend="numpy"
            )

            w_cupy = plane_wave_mass_weights(
                w,
                pos,
                dirs,
                wavenumber,
                phase0,
                sign,
                chunk_size=chunk,
                backend="cupy",
            )

            # Check read-only
            assert not w_cupy.flags.writeable
            assert w_cupy.dtype == np.complex128

            # Equivalence comparison
            # registry-id: C3-planewave-phase-only
            # (chunk_size is accepted but has no effect on this path's cupy
            # execution — docs/design/cupy_backend.md §8.4/§8.5 — so this
            # sweep's three chunk_size values are recorded as one registry
            # row rather than three near-duplicates.)
            np.testing.assert_allclose(w_ref, w_cupy, rtol=1e-12, atol=1e-15)


def test_cupy_planewave_phase_only_equivalence():
    """Pin on-device phase factor directly against host plane_wave_complex_factor."""
    rng = np.random.default_rng(54321)

    n, n_dir, n_f = 15, 6, 3
    pos = rng.uniform(-5.0, 5.0, (n, 3))
    dirs = rng.uniform(-1.0, 1.0, (n_dir, 3))
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    wavenumber = rng.uniform(0.5, 3.0, (n_f,))
    phase0 = -0.2
    sign = 1

    # 1. Host phase complex factor
    factor_ref = plane_wave_complex_factor(
        positions_m=pos,
        k_hat=dirs,
        wavenumber_rad_m=wavenumber,
        phase0=phase0,
        sign=sign,
    )

    # 2. Get device version
    w_unit = np.ones(n, dtype=np.float64)
    factor_got = plane_wave_mass_weights(
        w_unit, pos, dirs, wavenumber, phase0, sign, backend="cupy"
    )

    # Compute error separately
    phase_error = np.max(np.abs(factor_ref - factor_got))
    assert phase_error <= 1e-12
    np.testing.assert_allclose(factor_ref, factor_got, rtol=1e-12, atol=1e-15)


def test_cupy_planewave_empty_axes():
    """Verify empty directions or frequencies are handled correctly without crash."""
    w = np.array([1.0, 2.0])
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    # Empty directions: Ndir=0
    dirs = np.zeros((0, 3))
    wavenumber = np.array([1.0])
    w_got = plane_wave_mass_weights(w, pos, dirs, wavenumber, backend="cupy")
    assert w_got.shape == (2, 0, 1)
    assert w_got.dtype == np.complex128
    assert not w_got.flags.writeable

    # Empty frequencies: Nf=0
    dirs = np.array([[1.0, 0.0, 0.0]])
    wavenumber = np.zeros((0,))
    w_got = plane_wave_mass_weights(w, pos, dirs, wavenumber, backend="cupy")
    assert w_got.shape == (2, 1, 0)
    assert w_got.dtype == np.complex128
    assert not w_got.flags.writeable


def test_cupy_planewave_empty_axis_skips_dot(monkeypatch):
    """Verify the empty-axis short-circuit returns before cp.dot allocates the
    (N, Ndir) intermediate (the VRAM-preflight blind spot fixed here: Nf=0 used
    to drive the estimate's dominant term to 0 and let the dot allocation slip
    through unaccounted)."""
    w = np.array([1.0, 2.0, 3.0])
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, -1.0, 0.5]])
    dirs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    wavenumber = np.zeros((0,))  # Nf=0

    def fail_dot(*_args, **_kwargs):
        raise AssertionError("cp.dot should not be called when an axis is empty")

    monkeypatch.setattr(cp, "dot", fail_dot)

    w_got = plane_wave_mass_weights(w, pos, dirs, wavenumber, backend="cupy")
    assert w_got.shape == (3, 2, 0)
    assert w_got.dtype == np.complex128
    assert not w_got.flags.writeable


def test_cupy_planewave_empty_axis_still_requires_cuda_device(monkeypatch):
    """Verify the empty-axis short-circuit still calls _ensure_cuda_device first.

    Regression guard for the device-required contract: an empty-axis call must
    not silently return a zero array without checking CUDA availability.
    """
    import gragra.kernels._cupy_planewave as cupy_planewave_mod

    orig_ensure_cuda_device = cupy_planewave_mod._ensure_cuda_device
    calls = []

    def wrapped_ensure_cuda_device(*args, **kwargs):
        calls.append((args, kwargs))
        return orig_ensure_cuda_device(*args, **kwargs)

    monkeypatch.setattr(
        cupy_planewave_mod, "_ensure_cuda_device", wrapped_ensure_cuda_device
    )

    w = np.array([1.0, 2.0])
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    dirs = np.zeros((0, 3))  # Ndir=0
    wavenumber = np.array([1.0])

    w_got = plane_wave_mass_weights(w, pos, dirs, wavenumber, backend="cupy")
    assert w_got.shape == (2, 0, 1)
    assert len(calls) == 1


def test_cupy_planewave_memory_preflight(monkeypatch):
    """Verify VRAM preflight memory check triggers MemoryError and does not allocate."""
    w = np.ones(10, dtype=np.float64)
    pos = np.zeros((10, 3))
    dirs = np.ones((5, 3))
    wavenumber = np.ones(2)

    class MockDevice:
        @property
        def mem_info(self):
            return (100, 1000000000)

    monkeypatch.setattr(cp.cuda, "Device", MockDevice)

    # Spy to ensure no CuPy allocation occurs
    allocated = False

    def spy_alloc(*_args, **_kwargs):
        nonlocal allocated
        allocated = True
        raise RuntimeError("Allocation occurred before preflight check!")

    monkeypatch.setattr(cp, "asarray", spy_alloc)
    monkeypatch.setattr(cp, "zeros", spy_alloc)
    monkeypatch.setattr(cp, "empty", spy_alloc)

    with pytest.raises(MemoryError, match="Insufficient VRAM"):
        plane_wave_mass_weights(w, pos, dirs, wavenumber, backend="cupy")

    assert not allocated


def test_cupy_planewave_overflow_preflight(monkeypatch):
    """Verify preflight handles python-int overflow dimensions and raises MemoryError."""
    w = ShapeWrapper(np.ones(10), (10**18,))
    pos = ShapeWrapper(np.ones((10, 3)), (10**18, 3))
    dirs = ShapeWrapper(np.ones((5, 3)), (5, 3))
    wavenumber = ShapeWrapper(np.ones(2), (2,))

    monkeypatch.setattr(
        "gragra.coherent.weights._prepare_planewave_arrays",
        lambda *_args, **_kwargs: (w, pos, dirs, wavenumber, 0.0, 1),
    )

    class MockDevice:
        @property
        def mem_info(self):
            return (8 * 10**9, 8 * 10**9)

    monkeypatch.setattr(cp.cuda, "Device", MockDevice)

    allocated = False

    def spy_alloc(*_args, **_kwargs):
        nonlocal allocated
        allocated = True
        raise RuntimeError("Allocation occurred before preflight check!")

    monkeypatch.setattr(cp, "asarray", spy_alloc)
    monkeypatch.setattr(cp, "zeros", spy_alloc)
    monkeypatch.setattr(cp, "empty", spy_alloc)

    with pytest.raises(MemoryError, match="Insufficient VRAM"):
        plane_wave_mass_weights(
            np.ones(10),
            np.ones((10, 3)),
            np.ones((5, 3)),
            np.ones(2),
            backend="cupy",
        )

    assert not allocated
