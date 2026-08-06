"""Displacement transfer acceleration observable."""

import numpy as np

from gragra.array_types import ComplexArray, FloatArray
from gragra.constants import G_SI
from gragra.integrators.point_direct import direct_point_dipole_sum
from gragra.targets import TargetPoints


def displacement_transfer_acceleration(
    sources_xyz: list | tuple | np.ndarray,
    mass_kg: list | tuple | np.ndarray,
    displacement_m: list | tuple | np.ndarray,
    targets: TargetPoints | list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> FloatArray | ComplexArray:
    """Calculate the G_SI-scaled displacement-transfer (dipole/total) acceleration.

    a_i = G_SI * (-sum_n mass_n * sum_j K_T_ij(r_n) * displacement_n[j])

    See ``gragra.kernels.inverse_square.dipole_contract`` for the G-free
    contract this wraps (K_T is the gravity gradient kernel) and
    ``kernel_contract.md`` Section 2.2 for the six preconditions under which
    this is equivalent to the bulk+surface divergence path — a sub-volume
    embedded in surrounding material (e.g. rock) is NOT equivalent.

    Unlike ``point_acceleration``, this does not take a ``WeightedPointSource``:
    ``mass_kg`` (real, finite, non-negative background mass elements) and
    ``displacement_m`` (real or complex displacement vectors, shape (N, 3))
    are a two-argument vector-weight composition, not a ``weights_kg``-only
    quantity (source-types.md CRITICAL).

    Complex outputs represent linear complex amplitudes or transfer-function-like
    responses, not directly real-valued time-domain gravitational fields.

    Parameters
    ----------
    sources_xyz : array-like, shape (N, 3)
    mass_kg : array-like, shape (N,), real, finite, non-negative
    displacement_m : array-like, shape (N, 3), real or complex
    targets : TargetPoints or array-like
    softening_m : float, optional
        Plummer softening parameter in meters. Defaults to 0.0. Note that for
        softening_m > 0, the underlying K_T tensor does not satisfy tr(T) = 0
        (kernel_contract.md §4), so this response inherits the same
        non-trace-free caveat (see ``dipole_contract``).
    chunk_size : int, optional
    backend : str, optional
        'numpy' (default), 'numba', 'cpp', or 'cupy' (GPU; requires a CuPy
        wheel matching the local CUDA/ROCm runtime and a working CUDA device).

    Returns
    -------
    FloatArray or ComplexArray, shape (M, 3)
    """
    g_free = direct_point_dipole_sum(
        mass_kg,
        displacement_m,
        sources_xyz,
        targets,
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend=backend,
    )
    res = G_SI * g_free
    res.setflags(write=False)
    return res
