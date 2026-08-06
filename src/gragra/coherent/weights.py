"""Coherent mass weights utilities."""

import numpy as np

from gragra._arrays import as_weights, as_xyz_table
from gragra.waves.directions import normalize_directions


def _resolve_plane_wave_weight_factor(backend: str):
    if backend == "numpy":
        from gragra.waves.phase import plane_wave_complex_factor

        def numpy_contract(
            w,
            pos,
            directions,
            wavenumber_rad_m,
            phase0,
            sign,
            chunk_size=None,  # noqa: ARG001
        ):
            factor = plane_wave_complex_factor(
                positions_m=pos,
                k_hat=directions,
                wavenumber_rad_m=wavenumber_rad_m,
                phase0=phase0,
                sign=sign,
            )
            res = w[:, np.newaxis, np.newaxis] * factor
            res = np.asarray(res, dtype=np.complex128)
            res = res.copy()
            res.setflags(write=False)
            return res

        return numpy_contract

    if backend == "numba":
        try:
            import numba  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "numba is not installed in the current environment. "
                "Please install numba (or use `pip install gragra[numba]`) "
                "to use the numba backend."
            ) from e
        from gragra.kernels._numba_planewave import plane_wave_weight_factor

        return plane_wave_weight_factor

    if backend == "cpp":
        try:
            from gragra.kernels._cpp_planewave import plane_wave_weight_factor
        except ImportError:
            raise
        return plane_wave_weight_factor

    if backend == "cupy":
        try:
            import cupy  # noqa: F401
        except ImportError as e:

            class CupyImportValueError(ImportError, ValueError):
                pass

            raise CupyImportValueError(
                "cupy is not installed in the current environment. Install the "
                "CuPy wheel matching your CUDA/ROCm runtime (e.g. `pip install "
                "'cupy-cuda12x[ctk]'` for CUDA 12.x without a system CUDA "
                "Toolkit, or plain `cupy-cuda12x` when one is installed) to use "
                "the cupy backend. Note that a CUDA-capable GPU and driver are "
                "required at runtime; see docs/design/cupy_backend.md."
            ) from e
        from gragra.kernels._cupy_planewave import plane_wave_weight_factor

        return plane_wave_weight_factor

    raise ValueError(
        f"Unsupported backend: must be 'numpy', 'numba', 'cpp', or 'cupy', "
        f"got '{backend}'"
    )


def _prepare_planewave_arrays(
    base_weights_kg: list | tuple | np.ndarray,
    positions_m: list | tuple | np.ndarray,
    directions: list | tuple | np.ndarray,
    wavenumber_rad_m: float | int | list | tuple | np.ndarray,
    phase0: float | int = 0.0,
    sign: int = 1,
):
    if isinstance(sign, bool):
        raise TypeError("sign must not be bool")
    if sign not in (1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")

    if isinstance(phase0, bool):
        raise TypeError("phase0 must not be bool")
    if not isinstance(phase0, (int, float, np.integer, np.floating)):
        raise TypeError("phase0 must be numeric")
    if not np.isfinite(phase0):
        raise ValueError("phase0 must be finite")
    phase0 = float(phase0)

    # Validate wavenumber
    k_val = np.asarray(wavenumber_rad_m)
    if k_val.dtype.kind == "b" or isinstance(wavenumber_rad_m, bool):
        raise TypeError("wavenumber_rad_m must be numeric, not bool")

    try:
        k_val = np.asarray(wavenumber_rad_m, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError(f"wavenumber_rad_m must be numeric: {e}") from e

    if k_val.ndim == 0:
        k_val = k_val[np.newaxis]
    elif k_val.ndim == 1:
        pass
    else:
        raise ValueError(f"wavenumber_rad_m must be 0D or 1D, got shape {k_val.shape}")

    if not np.isfinite(k_val).all():
        raise ValueError("wavenumber_rad_m must contain only finite numbers")

    if (k_val < 0.0).any():
        raise ValueError("wavenumber_rad_m must be non-negative")

    w = as_weights("base_weights_kg", base_weights_kg)
    pos = as_xyz_table("positions_m", positions_m)

    if len(w) != len(pos):
        raise ValueError(
            "Length of base_weights_kg and positions_m must match: "
            f"{len(w)} != {len(pos)}"
        )

    k_norm = normalize_directions(directions)

    return w, pos, k_norm, k_val, phase0, sign


def plane_wave_mass_weights(
    base_weights_kg: list | tuple | np.ndarray,
    positions_m: list | tuple | np.ndarray,
    directions: list | tuple | np.ndarray,
    wavenumber_rad_m: float | int | list | tuple | np.ndarray,
    phase0: float | int = 0.0,
    sign: int = 1,
    *,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> np.ndarray:
    """Calculate plane-wave coherent mass weights: base_weights * complex_factor.

    Complex outputs represent linear complex amplitudes or transfer-function-like
    responses, not directly real-valued time-domain gravitational fields.

    ``backend`` selects the implementation (``"numpy"`` default,
    ``"numba"``, ``"cpp"``, or ``"cupy"``). Raises ImportError if the selected
    compiled backend ("numba", "cpp", or "cupy") is not installed or available.
    The ``"cupy"`` backend estimates peak memory and enforces a VRAM safety limit
    before allocating device memory.

    Returns a complex128 array of shape (N, Ndir, Nf) which is read-only.
    """
    w, pos, k_norm, k_val, p0, sg = _prepare_planewave_arrays(
        base_weights_kg=base_weights_kg,
        positions_m=positions_m,
        directions=directions,
        wavenumber_rad_m=wavenumber_rad_m,
        phase0=phase0,
        sign=sign,
    )

    contract = _resolve_plane_wave_weight_factor(backend)
    if backend in ("numpy", "cupy"):
        return contract(w, pos, k_norm, k_val, p0, sg, chunk_size=chunk_size)
    return contract(w, pos, k_norm, k_val, p0, sg)
