"""C++ implementation wrapper of plane-wave weight factor."""

import numpy as np

try:
    import gragra._cpp_backend as _backend
except ImportError as e:
    raise ImportError(
        "the gragra C++ backend is not built. Install a wheel that "
        "includes it, or reinstall from source with "
        "--config-settings=cmake.define.GRAGRA_BUILD_CPP=ON"
    ) from e

from gragra.kernels._cpp_required import PLANEWAVE_REQUIRED as _REQUIRED

_missing = [s for s in _REQUIRED if not hasattr(_backend, s)]
if _missing:
    raise ImportError(
        "the gragra C++ backend is built but is stale/incomplete "
        f"(missing {', '.join(_missing)}). Reinstall from source with "
        "--config-settings=cmake.define.GRAGRA_BUILD_CPP=ON"
    )


def plane_wave_weight_factor(
    w: np.ndarray,
    pos: np.ndarray,
    k_hat: np.ndarray,
    k: np.ndarray,
    phase0: float,
    sign: int,
) -> np.ndarray:
    """C++ accelerated plane-wave weights contraction.

    Note: Inputs are assumed to be validated by the public interface.
    """
    # Ensure C-contiguous complex128 for weights to match C++ signature
    # std::complex<double>
    w_c = np.ascontiguousarray(w, dtype=np.complex128)
    pos_c = np.ascontiguousarray(pos, dtype=np.float64)
    k_hat_c = np.ascontiguousarray(k_hat, dtype=np.float64)
    k_c = np.ascontiguousarray(k, dtype=np.float64)

    res = _backend.plane_wave_weight_factor(
        w_c,
        pos_c,
        k_hat_c,
        k_c,
        float(phase0),
        int(sign),
    )

    res = np.ascontiguousarray(res, dtype=np.complex128)
    res.setflags(write=False)
    return res
