"""CuPy (GPU) accelerated parametric gravity kernels.

Computes parametric acceleration and gravity gradient contractions on a
CUDA device via CuPy. All kernel contractions are G-free (do not multiply
by G_SI); only the observables layer multiplies by G_SI.
"""

from collections.abc import Callable

import numpy as np

from gragra.array_types import ComplexArray, FloatArray
from gragra.kernels._cupy_inverse_square import _ensure_cuda_device
from gragra.kernels.parametric import (
    _acceleration_impl_parametric,
    _gradient_impl_parametric,
    _validate_inputs_parametric,
)


def _run_parametric_on_device(
    impl: Callable[..., None],
    out_shape_tail: tuple[int, ...],
    sources: FloatArray,
    w: FloatArray | ComplexArray,
    targets: FloatArray,
    eps: float,
    c_size: int,
    p_shape: tuple[int, ...],
) -> FloatArray | ComplexArray:
    """Transfer inputs H→D once, run the shared impl, transfer D→H once.

    The returned host array is read-only.
    """
    _ensure_cuda_device()
    import cupy as cp

    s_dev = cp.asarray(sources)
    w_dev = cp.asarray(w)
    t_dev = cp.asarray(targets)

    out_dtype = np.complex128 if np.iscomplexobj(w) else np.float64
    out_dev = cp.zeros((len(targets),) + p_shape + out_shape_tail, dtype=out_dtype)

    impl(cp, s_dev, w_dev, t_dev, eps, c_size, p_shape, out_dev)

    out = cp.asnumpy(out_dev)
    out.setflags(write=False)
    return out


def acceleration_contract_parametric(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> FloatArray | ComplexArray:
    """Calculate G-free acceleration contraction with parametric weights on GPU."""
    # Validate host inputs before any device work
    sources, w, targets, eps, c_size = _validate_inputs_parametric(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )
    p_shape = w.shape[1:]
    return _run_parametric_on_device(
        _acceleration_impl_parametric,
        (3,),
        sources,
        w,
        targets,
        eps,
        c_size,
        p_shape,
    )


def gradient_contract_parametric(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> FloatArray | ComplexArray:
    """Calculate G-free gravity gradient contraction with parametric weights on GPU."""
    # Validate host inputs before any device work
    sources, w, targets, eps, c_size = _validate_inputs_parametric(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )
    p_shape = w.shape[1:]
    return _run_parametric_on_device(
        _gradient_impl_parametric,
        (3, 3),
        sources,
        w,
        targets,
        eps,
        c_size,
        p_shape,
    )
