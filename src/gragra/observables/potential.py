"""Potential observable."""

import numpy as np

from gragra.array_types import ComplexArray, FloatArray
from gragra.constants import G_SI
from gragra.integrators.point_direct import direct_point_sum
from gragra.sources import PointMass, PointMassCloud, WeightedPointSource
from gragra.targets import TargetPoints


def point_potential(
    source: WeightedPointSource | PointMass | PointMassCloud,
    targets: TargetPoints | list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> FloatArray | ComplexArray:
    """Calculate the gravitational potential at target points.

    Potential Φ = G_SI * sum_j (w_j * K_Φ)

    Complex outputs represent linear complex amplitudes or transfer-function-like
    responses, not directly real-valued time-domain gravitational fields.

    Parameters
    ----------
    source : WeightedPointSource, PointMass, or PointMassCloud
        The source of gravitational perturbation.
    targets : TargetPoints or array-like
        The target observation points.
    softening_m : float, optional
        Plummer softening parameter. Defaults to 0.0.
    chunk_size : int, optional
        Chunk size for source-axis loop. Defaults to None.
    backend : str, optional
        The calculation backend: 'numpy' (default), 'numba', 'cpp', or
        'cupy' (GPU, point-direct only; requires a CuPy wheel matching the
        local CUDA/ROCm runtime and a working CUDA device).

    Returns
    -------
    FloatArray or ComplexArray
        Gravitational potential of shape (M,).
    """
    g_free = direct_point_sum(
        source,
        targets,
        observable="potential",
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend=backend,
    )
    res = G_SI * g_free
    res.setflags(write=False)
    return res
