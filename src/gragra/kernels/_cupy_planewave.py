"""CuPy (GPU) accelerated plane-wave mass weights kernels.

Computes plane-wave coherent mass weights on a CUDA device via CuPy.
"""

import math

import numpy as np

from gragra.kernels._cupy_inverse_square import _ensure_cuda_device


def _estimate_planewave_vram_bytes(
    n: int,
    n_dir: int,
    n_f: int,
    itemsize: int,
    extra_bytes: int = 0,
) -> int:
    """Estimate peak VRAM for plane-wave computation.

    Parameters
    ----------
    n : int
        Number of positions.
    n_dir : int
        Number of directions.
    n_f : int
        Number of frequencies.
    itemsize : int
        Weight dtype itemsize in bytes.
    extra_bytes : int, optional
        Additional allocation bytes for derived operations.

    Returns
    -------
    int
        Estimated peak VRAM bytes including 1.1x safety margin.

    Notes
    -----
    Includes the ``(N, Ndir)`` float64 intermediate allocated by
    ``cp.dot(pos_dev, directions_dev.T)`` (the ``8 * n * n_dir`` term below),
    in addition to the ``(N, Ndir, Nf)`` complex128 phase/factor buffers.
    """
    base_bytes = (
        4 * 16 * n * n_dir * n_f
        + 8 * n * n_dir
        + itemsize * n
        + 8 * (3 * n + 3 * n_dir + n_f)
    )
    total_bytes = base_bytes + extra_bytes
    return math.ceil(1.1 * total_bytes)


def plane_wave_weight_factor(
    w: np.ndarray,
    pos: np.ndarray,
    directions: np.ndarray,
    wavenumber_rad_m: np.ndarray,
    phase0: float,
    sign: int,
    chunk_size: int | None = None,  # noqa: ARG001
) -> np.ndarray:
    """Calculate plane-wave weight factor on GPU.

    ``chunk_size`` is accepted for signature compatibility but ignored: this
    kernel does not tile; it only enforces an up-front VRAM safety limit
    before allocating.

    Returns a complex128 NumPy array of shape (N, Ndir, Nf) which is read-only.
    """
    _ensure_cuda_device()
    import cupy as cp

    # Ensure all dimension products are python ints to prevent numpy overflow
    n = int(w.shape[0])
    n_dir = int(directions.shape[0])
    n_f = int(wavenumber_rad_m.shape[0])

    # Short-circuit on any empty axis: cp.dot(pos_dev, directions_dev.T) would
    # still materialize an (N, Ndir) intermediate even when the result is
    # empty, and there is nothing to transfer to device either way. This must
    # stay *after* _ensure_cuda_device() so an empty-axis call on a
    # GPU-less/misconfigured device still raises instead of silently
    # returning a zero array.
    if n == 0 or n_dir == 0 or n_f == 0:
        res_host = np.zeros((n, n_dir, n_f), dtype=np.complex128)
        res_host.setflags(write=False)
        return res_host

    itemsize = int(w.dtype.itemsize)
    peak_bytes = _estimate_planewave_vram_bytes(n, n_dir, n_f, itemsize)

    free_vram, _ = cp.cuda.Device().mem_info
    if peak_bytes > 0.5 * free_vram:
        raise MemoryError(
            f"Insufficient VRAM for plane-wave mass weights: estimated "
            f"{peak_bytes} bytes, free VRAM is {free_vram} bytes."
        )

    # Transfer inputs to device
    w_dev = cp.asarray(w)
    pos_dev = cp.asarray(pos)
    directions_dev = cp.asarray(directions)
    wavenumber_dev = cp.asarray(wavenumber_rad_m)

    # Compute phase and complex factor on-device (hand-duplicated formula)
    # dot product: (N, Ndir)
    dot_dev = cp.dot(pos_dev, directions_dev.T)
    # phase: (N, Ndir, Nf)
    phase_dev = (
        dot_dev[:, :, cp.newaxis] * wavenumber_dev[cp.newaxis, cp.newaxis, :] + phase0
    )
    # complex factor: (N, Ndir, Nf)
    factor_dev = cp.exp(1j * sign * phase_dev)

    # Apply base weights
    res_dev = w_dev[:, cp.newaxis, cp.newaxis] * factor_dev

    # Output cast to complex128 and transfer back to host. copy=False avoids
    # a redundant full-array copy when res_dev is already complex128 (w
    # complex128 * factor_dev complex128 already yields complex128).
    res_dev = res_dev.astype(cp.complex128, copy=False)
    res_host = cp.asnumpy(res_dev)
    res_host.setflags(write=False)
    return res_host
