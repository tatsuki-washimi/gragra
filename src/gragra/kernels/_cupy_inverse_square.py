"""CuPy (GPU) accelerated gravity kernels.

Computes potential, acceleration, and gravity gradient contractions on a
CUDA device via CuPy. All kernel contractions are G-free (do not multiply
by G_SI); only the observables layer multiplies by G_SI.

This module is a thin device wrapper: the inverse-square formulas live in
``gragra.kernels.inverse_square`` as xp-agnostic helpers and are shared
with the numpy reference backend (single-source; see
``docs/architecture/gpu_backend_adr.md`` §3). Coordinates are float64 and
amplitudes/weights complex128 (or float64); the output dtype follows the
weights. Complex outputs represent linear complex amplitudes or
transfer-function-like responses, not directly real-valued time-domain
gravitational fields.

Chunking semantics on GPU
-------------------------
Unlike the numba/cpp backends (which accept but ignore ``chunk_size``),
on the GPU ``chunk_size`` is an effective device-memory tiling parameter:
``chunk_size=None`` materializes the full ``(M, N, 3)`` (and, for the
gradient, ``(M, N, 3, 3)``) intermediate in VRAM at once — beware of
out-of-memory for large N — while ``chunk_size=C`` bounds the
intermediate to ``(M, C, 3)``. Chunk invariance is a tolerance-based
guarantee only: in-chunk summation uses a parallel (non-associative)
reduction, so changing ``chunk_size`` changes the reduction grouping
(see ``gpu_backend_adr.md`` §6). This differs from nn-sus, which guards
the division with an additive ``r^3 + eps`` term instead of Plummer
``r_eff^2 = r^2 + eps^2``.

On the ``softening_m == 0`` path, collision detection synchronizes the
device once per chunk. CuPy's memory pool retains device allocations
after the call returns; releasing it (``free_all_blocks``) is the
caller's responsibility.
"""

import functools
from collections.abc import Callable

# A plain import, like the numba wrapper: the user-facing install hint for
# the normal backend="cupy" path lives in one place only
# (integrators/point_direct.py:_import_cupy_kernels), which always runs
# before this module is imported.
import cupy as cp
import numpy as np

from gragra.array_types import ComplexArray, FloatArray
from gragra.kernels.inverse_square import (
    _acceleration_impl,
    _dipole_impl,
    _gradient_impl,
    _potential_impl,
    _validate_dipole_inputs,
    _validate_inputs,
)


@functools.lru_cache(maxsize=1)
def _ensure_cuda_device() -> bool:
    """Verify that a usable CUDA device is present (cached on success).

    CuPy imports successfully without a GPU and only fails at device
    initialization time, so this check runs at the start of every
    contract call. ``functools.lru_cache`` does not cache exceptions,
    so a failure (no GPU, driver/runtime mismatch) is re-checked on the
    next call and recovery (e.g. after fixing ``CUDA_VISIBLE_DEVICES``)
    is not blocked. The cache is per-process and not fork-safe: CUDA
    contexts do not survive ``fork``, so worker processes must perform
    their own first call.
    """
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as e:
        raise RuntimeError(
            "cupy is installed but no usable CUDA device was found "
            "(no GPU, or a driver/runtime version mismatch). The cupy "
            "backend requires a working CUDA device at runtime; check "
            "`nvidia-smi` and your driver installation."
        ) from e
    except Exception as e:
        # Not a CUDA runtime failure — do not misdirect the user to
        # driver checks; surface the chained original error instead.
        raise RuntimeError(
            "unexpected error while probing the CUDA runtime via cupy; "
            "see the chained exception for the original cause."
        ) from e
    if count == 0:
        raise RuntimeError(
            "cupy is installed but no CUDA device is visible "
            "(getDeviceCount() == 0). Check `nvidia-smi` and the "
            "CUDA_VISIBLE_DEVICES environment variable."
        )
    return True


def _run_on_device(
    impl: Callable[..., None],
    out_shape_tail: tuple[int, ...],
    sources: FloatArray,
    w: FloatArray | ComplexArray,
    targets: FloatArray,
    eps: float,
    c_size: int,
) -> FloatArray | ComplexArray:
    """Transfer inputs H→D once, run the shared impl, transfer D→H once.

    The output dtype follows the weights (complex128 for complex weights,
    float64 otherwise) and is decided host-side; the returned host array
    is read-only.
    """
    _ensure_cuda_device()

    s_dev = cp.asarray(sources)
    w_dev = cp.asarray(w)
    t_dev = cp.asarray(targets)

    out_dtype = np.complex128 if np.iscomplexobj(w) else np.float64
    out_dev = cp.zeros((len(targets), *out_shape_tail), dtype=out_dtype)

    impl(cp, s_dev, w_dev, t_dev, eps, c_size, out_dev)

    out = cp.asnumpy(out_dev)
    out.setflags(write=False)
    return out


def _run_dipole_on_device(
    sources: FloatArray,
    mass: FloatArray,
    disp: FloatArray | ComplexArray,
    targets: FloatArray,
    eps: float,
    c_size: int,
) -> FloatArray | ComplexArray:
    """Transfer inputs H→D once, run the shared _dipole_impl, transfer D→H once.

    Output shape is (M, 3). The output dtype follows displacement_m (complex128 if
    displacement_m is complex, else float64) and is decided host-side; the returned
    host array is read-only.
    """
    _ensure_cuda_device()

    s_dev = cp.asarray(sources)
    m_dev = cp.asarray(mass)
    d_dev = cp.asarray(disp)
    t_dev = cp.asarray(targets)

    out_dtype = np.complex128 if np.iscomplexobj(disp) else np.float64
    out_dev = cp.zeros((len(targets), 3), dtype=out_dtype)

    _dipole_impl(cp, s_dev, m_dev, d_dev, t_dev, eps, c_size, out_dev)

    out = cp.asnumpy(out_dev)
    out.setflags(write=False)
    return out


def potential_contract(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> FloatArray | ComplexArray:
    """Compute the potential contraction on the GPU via CuPy.

    K_Φ = -1 / sqrt(r^2 + ε^2). This contraction is purely geometric and
    G-free (does not multiply by G_SI). The returned array is a read-only
    host (numpy) array of shape (M,); the output dtype follows the
    weights. Complex outputs represent linear complex amplitudes or
    transfer-function-like responses, not real-valued time-domain fields.

    On the GPU, ``chunk_size`` bounds the ``(M, C)``-sized intermediates
    in VRAM (``None`` = all sources at once; see module docstring).
    Raises RuntimeError if no usable CUDA device is present.
    """
    sources, w, targets, eps, c_size = _validate_inputs(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )
    return _run_on_device(_potential_impl, (), sources, w, targets, eps, c_size)


def acceleration_contract(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> FloatArray | ComplexArray:
    """Compute the acceleration contraction on the GPU via CuPy.

    K_a = Δ / (r^2 + ε^2)^(3/2). This contraction is purely geometric and
    G-free (does not multiply by G_SI). The returned array is a read-only
    host (numpy) array of shape (M, 3) — the component axis is trailing —
    and the output dtype follows the weights. Complex outputs represent
    linear complex amplitudes or transfer-function-like responses, not
    real-valued time-domain fields.

    On the GPU, ``chunk_size`` bounds the ``(M, C, 3)``-sized
    intermediates in VRAM (``None`` = all sources at once; see module
    docstring). Raises RuntimeError if no usable CUDA device is present.
    """
    sources, w, targets, eps, c_size = _validate_inputs(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )
    return _run_on_device(_acceleration_impl, (3,), sources, w, targets, eps, c_size)


def gradient_contract(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> FloatArray | ComplexArray:
    """Compute the gravity gradient contraction on the GPU via CuPy.

    K_T = 3 ΔΔ / (r^2 + ε^2)^(5/2) - I / (r^2 + ε^2)^(3/2). This
    contraction is purely geometric and G-free (does not multiply by
    G_SI). The returned array is a read-only host (numpy) array of shape
    (M, 3, 3) — the tensor axes are trailing — and the output dtype
    follows the weights. Complex outputs represent linear complex
    amplitudes or transfer-function-like responses, not real-valued
    time-domain fields.

    Note that tr(T) = 0 holds only for softening_m = 0; the softened
    tensor (softening_m > 0) is an approximation without strict
    harmonicity.

    On the GPU, ``chunk_size`` bounds the ``(M, C, 3, 3)``-sized
    intermediates in VRAM (``None`` = all sources at once; see module
    docstring). Raises RuntimeError if no usable CUDA device is present.
    """
    sources, w, targets, eps, c_size = _validate_inputs(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )
    return _run_on_device(_gradient_impl, (3, 3), sources, w, targets, eps, c_size)


def dipole_contract(
    sources_xyz: list | tuple | np.ndarray,
    mass_kg: list | tuple | np.ndarray,
    displacement_m: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> FloatArray | ComplexArray:
    """Compute the displacement-transfer (dipole/total) contraction on the GPU via CuPy.

    The returned array is a read-only host (numpy) array of shape (M, 3);
    the output dtype follows displacement_m (complex128 if displacement_m is
    complex, else float64).

    On the GPU, ``chunk_size`` bounds the intermediates in VRAM.
    Raises RuntimeError if no usable CUDA device is present.
    """
    sources, mass, disp, targets, eps, c_size = _validate_dipole_inputs(
        sources_xyz, mass_kg, displacement_m, targets_xyz, softening_m, chunk_size
    )
    return _run_dipole_on_device(sources, mass, disp, targets, eps, c_size)
