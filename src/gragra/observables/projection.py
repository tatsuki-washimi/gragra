"""Projection utilities and projected acceleration observable."""

import numpy as np

from gragra._arrays import as_xyz_vector
from gragra.array_types import ComplexArray, FloatArray
from gragra.observables.acceleration import point_acceleration
from gragra.sources import PointMass, PointMassCloud, WeightedPointSource
from gragra.targets import TargetPoints


def project_vectors(
    vectors: list | tuple | np.ndarray,
    direction: list | tuple | np.ndarray,
) -> FloatArray | ComplexArray:
    """Calculate the raw dot product projection of vectors along a direction.

    This is a raw algebraic dot product (sum of component-wise multiplication)
    without normalization of the direction vector.
    The caller is responsible for normalization.

    Parameters
    ----------
    vectors : array-like
        Array of vectors with shape (..., 3).
    direction : array-like
        Direction vector with shape (3,).

    Returns
    -------
    FloatArray or ComplexArray
        Projected scalar array of shape (...).
    """
    vecs = np.asarray(vectors)
    dir_vec = as_xyz_vector("direction", direction)

    if vecs.shape[-1] != 3:
        raise ValueError(
            f"vectors must end with a component axis of size 3, got shape {vecs.shape}"
        )

    res = np.sum(vecs * dir_vec, axis=-1)
    res.setflags(write=False)
    return res


def point_projected_acceleration(
    source: WeightedPointSource | PointMass | PointMassCloud,
    targets: TargetPoints | list | tuple | np.ndarray,
    direction: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> FloatArray | ComplexArray:
    """Calculate projected gravitational acceleration along a normalized direction.

    The direction vector is internally normalized to a unit vector.

    Complex outputs represent linear complex amplitudes or transfer-function-like
    responses, not directly real-valued time-domain gravitational fields.

    Parameters
    ----------
    source : WeightedPointSource, PointMass, or PointMassCloud
        The source of gravitational perturbation.
    targets : TargetPoints or array-like
        The target observation points.
    direction : array-like
        Direction vector of shape (3,). Must be non-zero and finite.
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
        Projected acceleration of shape (M,).
    """
    dir_vec = as_xyz_vector("direction", direction)
    norm = np.linalg.norm(dir_vec)
    if norm == 0.0:
        raise ValueError("direction vector cannot be a zero vector")

    unit_dir = dir_vec / norm

    # Calculate full 3D acceleration
    acc = point_acceleration(
        source,
        targets,
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend=backend,
    )

    # Project along unit direction
    return project_vectors(acc, unit_dir)
