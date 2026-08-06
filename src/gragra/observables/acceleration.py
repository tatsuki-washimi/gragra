"""Acceleration observables."""

import operator
from typing import TYPE_CHECKING

import numpy as np

from gragra.array_types import ComplexArray, FloatArray
from gragra.constants import G_SI
from gragra.integrators.point_direct import direct_point_dipole_sum, direct_point_sum
from gragra.sources import PointMass, PointMassCloud, WeightedPointSource
from gragra.targets import TargetPoints

if TYPE_CHECKING:
    from gragra.adapters.specfem.reader import SpecfemGLLDataset


def point_acceleration(
    source: WeightedPointSource | PointMass | PointMassCloud,
    targets: TargetPoints | list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> FloatArray | ComplexArray:
    """Calculate the gravitational acceleration vector at target points.

    Acceleration a = G_SI * sum_j (w_j * K_a)

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
        Gravitational acceleration of shape (M, 3).
    """
    g_free = direct_point_sum(
        source,
        targets,
        observable="acceleration",
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend=backend,
    )
    res = G_SI * g_free
    res.setflags(write=False)
    return res


def displacement_field_acceleration(
    dataset: "SpecfemGLLDataset",
    targets: TargetPoints | list | tuple | np.ndarray,
    *,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> FloatArray:
    """Calculate acceleration time series from a SPECFEM GLL displacement field.

    Every GLL row contributes the background mass
    ``density_kg_m3 * quadrature_volume_m3``.  Its displacement is contracted
    with the existing G-free dipole kernel, then this Observable applies
    ``G_SI`` exactly once. The rc1 reader rejects PML input and this wrapper
    fixes Plummer softening at zero. Targets must lie in a vacuum region
    (inside a cavity or above the free surface), because a rock-interior local
    term is outside the rc1 contract.

    rc1 keeps the closed dataset's complete displacement array in memory, but
    computes the result one time index at a time. It does not stream HDF5
    snapshots. Returns a read-only array of shape ``(Nt, M, 3)``.
    """
    if not hasattr(dataset, "iter_blocks") or not hasattr(dataset, "displacement_m"):
        raise TypeError("dataset must be a SpecfemGLLDataset")
    if dataset.displacement_m is None:  # defensive for future static readers
        raise ValueError("dataset does not contain a time-sampled displacement field")

    if chunk_size is None:
        element_block_size = None
    else:
        if isinstance(chunk_size, (bool, np.bool_)):
            raise ValueError("chunk_size must be a positive integer")
        try:
            chunk_size = operator.index(chunk_size)
        except TypeError as error:
            raise ValueError("chunk_size must be a positive integer") from error
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        ngll = dataset.coordinates_m.shape[1]
        element_block_size = max(1, chunk_size // ngll)

    result: np.ndarray | None = None
    for time_index in range(dataset.time_s.size):
        total: np.ndarray | None = None
        for block in dataset.iter_blocks(element_block_size=element_block_size):
            assert block.displacement_m is not None
            contribution = direct_point_dipole_sum(
                block.mass_kg.reshape(-1),
                block.displacement_m[time_index].reshape(-1, 3),
                block.coordinates_m.reshape(-1, 3),
                targets,
                softening_m=0.0,
                chunk_size=chunk_size,
                backend=backend,
            )
            total = contribution if total is None else total + contribution
        assert total is not None  # reader rejects zero-element files
        acceleration = G_SI * total
        if result is None:
            result = np.empty(
                (dataset.time_s.size, *acceleration.shape), dtype=np.float64
            )
        result[time_index] = acceleration

    assert result is not None
    result.setflags(write=False)
    return result
