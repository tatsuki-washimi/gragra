"""C++ accelerated parametric inverse-square gravity kernels.

Computes potential, acceleration, and gravity gradient contractions with
multi-dimensional parametric weights of shape ``(N, *P)``. All kernel
contractions are G-free (do not multiply by ``G_SI``).

The weight tensor ``(N, *P)`` is flattened to ``(N, P_flat)`` and the compiled
backend operates on that fixed 2-D interface; outputs are reshaped back to
``(M, *P, ...)``. Like the point-direct C++ backend, ``chunk_size`` is accepted
and validated for streaming transparency but is not used in the computation;
results are chunk-size independent. The returned arrays are read-only and the
output dtype follows the weights.
"""

import numpy as np

try:
    import gragra._cpp_backend as _cpp_backend
except ImportError as e:
    raise ImportError(
        "the gragra C++ backend is not built. Install a wheel that "
        "includes it, or reinstall from source with "
        "--config-settings=cmake.define.GRAGRA_BUILD_CPP=ON"
    ) from e

from gragra.kernels._collision import _check_collisions
from gragra.kernels._cpp_required import PARAMETRIC_REQUIRED as _REQUIRED
from gragra.kernels.parametric import _flatten_weights, _validate_inputs_parametric

_missing = [s for s in _REQUIRED if not hasattr(_cpp_backend, s)]
if _missing:
    raise ImportError(
        "the gragra C++ backend is built but is stale/incomplete "
        f"(missing {', '.join(_missing)}). Reinstall from source with "
        "--config-settings=cmake.define.GRAGRA_BUILD_CPP=ON"
    )


def potential_contract_parametric(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> np.ndarray:
    """C++ parametric potential contraction. K_Φ = -1 / sqrt(r^2 + ε^2)."""
    sources, w, targets, eps, _ = _validate_inputs_parametric(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )
    n = len(sources)
    m = len(targets)
    w2d, p_shape = _flatten_weights(w, n)
    collision_source_index = np.full(m, -1, dtype=np.int64)
    eps2 = eps**2

    if np.iscomplexobj(w):
        out2d = _cpp_backend.potential_contract_parametric_complex(
            sources, w2d, targets, eps2, collision_source_index
        )
    else:
        out2d = _cpp_backend.potential_contract_parametric_real(
            sources, w2d, targets, eps2, collision_source_index
        )

    _check_collisions(collision_source_index, eps)
    out = out2d.reshape((m,) + p_shape)
    out.setflags(write=False)
    return out


def acceleration_contract_parametric(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> np.ndarray:
    """C++ parametric acceleration contraction. K_a = Δ / (r^2 + ε^2)^(3/2)."""
    sources, w, targets, eps, _ = _validate_inputs_parametric(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )
    n = len(sources)
    m = len(targets)
    w2d, p_shape = _flatten_weights(w, n)
    collision_source_index = np.full(m, -1, dtype=np.int64)
    eps2 = eps**2

    if np.iscomplexobj(w):
        out3d = _cpp_backend.acceleration_contract_parametric_complex(
            sources, w2d, targets, eps2, collision_source_index
        )
    else:
        out3d = _cpp_backend.acceleration_contract_parametric_real(
            sources, w2d, targets, eps2, collision_source_index
        )

    _check_collisions(collision_source_index, eps)
    out = out3d.reshape((m,) + p_shape + (3,))
    out.setflags(write=False)
    return out


def gradient_contract_parametric(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> np.ndarray:
    """C++ parametric gravity-gradient contraction.

    K_T = 3 ΔΔ / (r^2 + ε^2)^(5/2) - I / (r^2 + ε^2)^(3/2).

    Note that for softening_m > 0, the softened tensor does not satisfy tr(T) = 0.
    """
    sources, w, targets, eps, _ = _validate_inputs_parametric(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )
    n = len(sources)
    m = len(targets)
    w2d, p_shape = _flatten_weights(w, n)
    collision_source_index = np.full(m, -1, dtype=np.int64)
    eps2 = eps**2

    if np.iscomplexobj(w):
        out4d = _cpp_backend.gradient_contract_parametric_complex(
            sources, w2d, targets, eps2, collision_source_index
        )
    else:
        out4d = _cpp_backend.gradient_contract_parametric_real(
            sources, w2d, targets, eps2, collision_source_index
        )

    _check_collisions(collision_source_index, eps)
    out = out4d.reshape((m,) + p_shape + (3, 3))
    out.setflags(write=False)
    return out
