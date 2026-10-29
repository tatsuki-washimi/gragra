"""Correctness tests for the CuPy (GPU) accelerated backend.

These tests ensure the 'cupy' backend is numerically equivalent to the
default 'numpy' reference backend (and 'numba' / 'cpp' when available).
They require an installed CuPy wheel *and* a usable CUDA device; both
conditions skip at module level otherwise (turned into a failure by
``GRAGRA_REQUIRE_CUPY=1``, see conftest).
"""

import numpy as np
import pytest

# Two-stage module-level skip: cupy importable, then a usable device.
cupy = pytest.importorskip("cupy")


def _cuda_available() -> bool:
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        # CUDARuntimeError on driver/runtime mismatch or missing driver
        return False


if not _cuda_available():
    pytest.skip("no usable CUDA device", allow_module_level=True)

# E402: these imports must come after the module-level device-availability
# skip above (two-stage skip: cupy importable, then a usable CUDA device).
from gragra.constants import G_SI  # noqa: E402
from gragra.gravity_sources import PointMass, WeightedPointSource  # noqa: E402
from gragra.observables import (  # noqa: E402
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)
from gragra.observables.projection import point_projected_acceleration  # noqa: E402

# Equivalence gate per gpu_backend_adr.md §7: initial value matching the
# cpp gate; GPU in-chunk summation is a non-associative parallel
# reduction, so this is validated (and if needed relaxed with rationale)
# by local GPU measurement rather than guaranteed by summation-order
# identity as for cpp.
RTOL = 1e-12
ATOL = 1e-15


def test_numpy_cupy_equivalence_real():
    """numpy and cupy backends agree on all four point-direct observables."""
    source = WeightedPointSource(
        positions_m=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        weights_kg=[10.0, 20.0],
    )
    targets = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]

    v_numpy = point_potential(source, targets, backend="numpy")
    v_cupy = point_potential(source, targets, backend="cupy")
    assert v_cupy.dtype == np.float64  # real weights -> float64 (kind kept)
    assert np.allclose(v_numpy, v_cupy, rtol=RTOL, atol=ATOL)

    a_numpy = point_acceleration(source, targets, backend="numpy")
    a_cupy = point_acceleration(source, targets, backend="cupy")
    assert a_cupy.dtype == np.float64
    assert np.allclose(a_numpy, a_cupy, rtol=RTOL, atol=ATOL)

    direction = [1.0, 1.0, 0.0]
    ap_numpy = point_projected_acceleration(source, targets, direction, backend="numpy")
    ap_cupy = point_projected_acceleration(source, targets, direction, backend="cupy")
    assert np.allclose(ap_numpy, ap_cupy, rtol=RTOL, atol=ATOL)

    g_numpy = point_gravity_gradient(source, targets, backend="numpy")
    g_cupy = point_gravity_gradient(source, targets, backend="cupy")
    assert np.allclose(g_numpy, g_cupy, rtol=RTOL, atol=ATOL)


def test_numpy_cupy_equivalence_complex():
    """Complex weights: equivalence and complex128 output dtype."""
    source = WeightedPointSource(
        positions_m=[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        weights_kg=[10.0 + 5.0j, -20.0 + 3.0j],
    )
    targets = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]

    v_numpy = point_potential(source, targets, backend="numpy")
    v_cupy = point_potential(source, targets, backend="cupy")
    assert v_cupy.dtype == np.complex128
    assert np.allclose(v_numpy, v_cupy, rtol=RTOL, atol=ATOL)

    a_numpy = point_acceleration(source, targets, backend="numpy")
    a_cupy = point_acceleration(source, targets, backend="cupy")
    assert a_cupy.dtype == np.complex128
    assert np.allclose(a_numpy, a_cupy, rtol=RTOL, atol=ATOL)

    g_numpy = point_gravity_gradient(source, targets, backend="numpy")
    g_cupy = point_gravity_gradient(source, targets, backend="cupy")
    assert g_cupy.dtype == np.complex128
    assert np.allclose(g_numpy, g_cupy, rtol=RTOL, atol=ATOL)


def test_cupy_tensor_sign_convention():
    """1kg at origin, target (1,0,0): T = G_SI * diag(2, -1, -1)."""
    source = PointMass(1.0, [0.0, 0.0, 0.0])
    target = [[1.0, 0.0, 0.0]]

    t_tensor = point_gravity_gradient(source, target, softening_m=0.0, backend="cupy")

    assert t_tensor.shape == (1, 3, 3)
    expected = G_SI * np.diag([2.0, -1.0, -1.0])
    assert np.allclose(t_tensor[0], expected, rtol=RTOL, atol=ATOL)


def test_cupy_trace_semantics():
    """tr(T) ~ 0 only for softening=0; softened tensor is finite."""
    source = PointMass(2.0, [0.0, 0.0, 0.0])
    target = [[1.0, 2.0, -0.5]]

    t_zero = point_gravity_gradient(source, target, softening_m=0.0, backend="cupy")
    assert abs(np.trace(t_zero[0])) < 1e-9 * G_SI

    t_soft = point_gravity_gradient(source, target, softening_m=0.3, backend="cupy")
    assert np.all(np.isfinite(t_soft))


def test_multi_backend_equivalence():
    """numpy == cupy == (numba, cpp when installed) on complex weights."""
    backends = ["numpy", "cupy"]
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

    source = WeightedPointSource(
        positions_m=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        weights_kg=[10.0 + 3.0j, 20.0 - 5.0j],
    )
    targets = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]

    reference = point_potential(source, targets, backend="numpy")
    for backend in backends[1:]:
        val = point_potential(source, targets, backend=backend)
        assert np.allclose(reference, val, rtol=RTOL, atol=ATOL), backend


def test_cupy_chunk_invariance_small_n():
    """chunk_size=None vs small chunks agree within the equivalence gate.

    On the GPU, chunk invariance is a tolerance-based guarantee only
    (changing chunk_size changes the reduction grouping; small chunks
    add sequential accumulation error), so the test is restricted to
    small N where the accumulated difference stays well below the gate
    (gpu_backend_adr.md §6).
    """
    rng = np.random.default_rng(42)
    source = WeightedPointSource(
        positions_m=rng.random((10, 3)),
        weights_kg=rng.random(10),
    )
    targets = rng.random((5, 3)) + 2.0  # keep away from sources

    v_none = point_potential(source, targets, chunk_size=None, backend="cupy")
    v_two = point_potential(source, targets, chunk_size=2, backend="cupy")
    v_seven = point_potential(source, targets, chunk_size=7, backend="cupy")
    assert np.allclose(v_none, v_two, rtol=RTOL, atol=ATOL)
    assert np.allclose(v_none, v_seven, rtol=RTOL, atol=ATOL)

    g_none = point_gravity_gradient(source, targets, chunk_size=None, backend="cupy")
    g_seven = point_gravity_gradient(source, targets, chunk_size=7, backend="cupy")
    assert np.allclose(g_none, g_seven, rtol=RTOL, atol=ATOL)


def test_cupy_collision_detection():
    """ValueError on target-source collision, message matching numpy."""
    source = WeightedPointSource(
        positions_m=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        weights_kg=[10.0, 20.0],
    )
    targets = [[1.0, 1.0, 1.0]]  # Collides with source index 1

    with pytest.raises(
        ValueError,
        match=r"(?i)collision.*target.*0.*source.*1.*zero softening",
    ):
        point_potential(source, targets, softening_m=0.0, backend="cupy")


def test_cupy_collision_across_chunks():
    """Global source index is recovered across chunk boundaries."""
    source = WeightedPointSource(
        positions_m=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        weights_kg=[1.0, 1.0, 1.0],
    )
    targets = [[5.0, 5.0, 5.0], [2.0, 2.0, 2.0]]  # target 1 hits source 2

    with pytest.raises(
        ValueError,
        match=r"(?i)collision.*target.*1.*source.*2.*zero softening",
    ):
        point_potential(source, targets, softening_m=0.0, chunk_size=1, backend="cupy")


def test_cupy_invalid_inputs():
    """Standard input validation errors are raised (host-side, shared)."""
    source = PointMass(1.0, [0.0, 0.0, 0.0])
    target = [[1.0, 0.0, 0.0]]

    with pytest.raises(ValueError, match="softening_m must be non-negative"):
        point_potential(source, target, softening_m=-0.1, backend="cupy")

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        point_potential(source, target, chunk_size=0, backend="cupy")


def test_cupy_read_only_returns():
    """Returned host arrays are read-only."""
    source = PointMass(1.0, [0.0, 0.0, 0.0])
    target = [[1.0, 0.0, 0.0]]

    v = point_potential(source, target, backend="cupy")
    assert not v.flags.writeable

    a = point_acceleration(source, target, backend="cupy")
    assert not a.flags.writeable

    ap = point_projected_acceleration(source, target, [1.0, 0.0, 0.0], backend="cupy")
    assert not ap.flags.writeable

    g = point_gravity_gradient(source, target, backend="cupy")
    assert not g.flags.writeable


def test_cupy_empty_inputs():
    """N=0 and M=0 are handled gracefully (zeros / empty outputs)."""
    source_empty = WeightedPointSource(positions_m=np.zeros((0, 3)), weights_kg=[])
    targets = [[1.0, 2.0, 3.0]]
    v_empty_n = point_potential(source_empty, targets, backend="cupy")
    assert v_empty_n.shape == (1,)
    assert np.all(v_empty_n == 0.0)

    a_empty_n = point_acceleration(source_empty, targets, backend="cupy")
    assert a_empty_n.shape == (1, 3)
    assert np.all(a_empty_n == 0.0)

    g_empty_n = point_gravity_gradient(source_empty, targets, backend="cupy")
    assert g_empty_n.shape == (1, 3, 3)
    assert np.all(g_empty_n == 0.0)

    source = WeightedPointSource(positions_m=[[0.0, 0.0, 0.0]], weights_kg=[10.0])
    targets_empty = np.zeros((0, 3))
    v_empty_m = point_potential(source, targets_empty, backend="cupy")
    assert v_empty_m.shape == (0,)

    a_empty_m = point_acceleration(source, targets_empty, backend="cupy")
    assert a_empty_m.shape == (0, 3)

    g_empty_m = point_gravity_gradient(source, targets_empty, backend="cupy")
    assert g_empty_m.shape == (0, 3, 3)


def test_cupy_medium_random_equivalence():
    """Medium-size random case (complex weights, with/without softening)."""
    rng = np.random.default_rng(42)
    sources_xyz = rng.random((200, 3))
    weights = rng.random(200) + 1j * rng.random(200)
    targets_xyz = rng.random((64, 3)) + 5.0  # keep away from sources

    source = WeightedPointSource(sources_xyz, weights)

    for softening in (0.0, 0.05):
        for fn in (point_potential, point_acceleration, point_gravity_gradient):
            ref = fn(source, targets_xyz, softening_m=softening, backend="numpy")
            got = fn(source, targets_xyz, softening_m=softening, backend="cupy")
            assert np.allclose(ref, got, rtol=RTOL, atol=ATOL), (fn.__name__, softening)
