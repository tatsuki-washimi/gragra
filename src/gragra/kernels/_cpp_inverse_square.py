"""C++ accelerated gravity kernels.

Computes potential, acceleration, and gravity gradient contractions.
All kernel contractions are G-free (do not multiply by G_SI).
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

from gragra.array_types import ComplexArray, FloatArray
from gragra.kernels._collision import _check_collisions
from gragra.kernels._cpp_required import INVERSE_SQUARE_REQUIRED as _REQUIRED
from gragra.kernels.inverse_square import _validate_dipole_inputs, _validate_inputs

_missing = [s for s in _REQUIRED if not hasattr(_cpp_backend, s)]
if _missing:
    raise ImportError(
        "the gragra C++ backend is built but is stale/incomplete "
        f"(missing {', '.join(_missing)}). Reinstall from source with "
        "--config-settings=cmake.define.GRAGRA_BUILD_CPP=ON"
    )


def potential_contract(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> FloatArray | ComplexArray:
    """Compute potential contraction using the C++ backend.

    This contraction is purely geometric and G-free (does not multiply by G_SI).
    The returned array is read-only and has shape (M,). The output dtype
    follows the weights.

    Note that chunk_size is accepted and validated for streaming transparency,
    but is not used in the actual C++ computation.
    """
    sources, w, targets, eps, _ = _validate_inputs(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )
    m = len(targets)
    collision_source_index = np.full(m, -1, dtype=np.int64)
    eps2 = eps**2

    if np.iscomplexobj(w):
        out = _cpp_backend.potential_contract_complex(
            sources, w, targets, eps2, collision_source_index
        )
    else:
        out = _cpp_backend.potential_contract_real(
            sources, w, targets, eps2, collision_source_index
        )

    _check_collisions(collision_source_index, eps)
    out.setflags(write=False)
    return out


def acceleration_contract(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> FloatArray | ComplexArray:
    """Compute acceleration contraction using the C++ backend.

    This contraction is purely geometric and G-free (does not multiply by G_SI).
    The returned array is read-only and has shape (M, 3). The output dtype
    follows the weights.

    Note that chunk_size is accepted and validated for streaming transparency,
    but is not used in the actual C++ computation.
    """
    sources, w, targets, eps, _ = _validate_inputs(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )
    m = len(targets)
    collision_source_index = np.full(m, -1, dtype=np.int64)
    eps2 = eps**2

    if np.iscomplexobj(w):
        out = _cpp_backend.acceleration_contract_complex(
            sources, w, targets, eps2, collision_source_index
        )
    else:
        out = _cpp_backend.acceleration_contract_real(
            sources, w, targets, eps2, collision_source_index
        )

    _check_collisions(collision_source_index, eps)
    out.setflags(write=False)
    return out


def gradient_contract(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> FloatArray | ComplexArray:
    """Compute gravity gradient contraction using the C++ backend.

    This contraction is purely geometric and G-free (does not multiply by G_SI).
    The returned array is read-only and has shape (M, 3, 3). The output dtype
    follows the weights.

    Note that chunk_size is accepted and validated for streaming transparency,
    but is not used in the actual C++ computation.

    Note that for softening_m > 0, the softened tensor does not satisfy tr(T) = 0.
    """
    sources, w, targets, eps, _ = _validate_inputs(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )
    m = len(targets)
    collision_source_index = np.full(m, -1, dtype=np.int64)
    eps2 = eps**2

    if np.iscomplexobj(w):
        out = _cpp_backend.gradient_contract_complex(
            sources, w, targets, eps2, collision_source_index
        )
    else:
        out = _cpp_backend.gradient_contract_real(
            sources, w, targets, eps2, collision_source_index
        )

    _check_collisions(collision_source_index, eps)
    out.setflags(write=False)
    return out


def dipole_contract(
    sources_xyz: list | tuple | np.ndarray,
    mass_kg: list | tuple | np.ndarray,
    displacement_m: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> FloatArray | ComplexArray:
    """Compute the vector-weight (dipole/total) contraction using the C++ backend.

    G-free. The returned array is read-only and has shape (M, 3). Output
    dtype follows displacement_m (mass_kg is always real per kernel_contract.md
    Section 2.1.1).

    Note that chunk_size is accepted and validated for streaming transparency,
    but is not used in the actual C++ computation.
    """
    sources, mass, disp, targets, eps, _ = _validate_dipole_inputs(
        sources_xyz, mass_kg, displacement_m, targets_xyz, softening_m, chunk_size
    )
    m = len(targets)
    collision_source_index = np.full(m, -1, dtype=np.int64)
    eps2 = eps**2

    if np.iscomplexobj(disp):
        out = _cpp_backend.dipole_contract_complex(
            sources, mass, disp, targets, eps2, collision_source_index
        )
    else:
        out = _cpp_backend.dipole_contract_real(
            sources, mass, disp, targets, eps2, collision_source_index
        )

    _check_collisions(collision_source_index, eps)
    out.setflags(write=False)
    return out
