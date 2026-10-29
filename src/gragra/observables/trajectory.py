"""Observables for prescribed point and rigid-body motion."""

import numpy as np

from gragra.constants import G_SI
from gragra.gravity_sources import PointMassCloud
from gragra.gravity_sources.trajectory import (
    PrescribedPointTrajectory,
    PrescribedRigidTransform,
)
from gragra.integrators.trajectory_direct import (
    direct_rigid_transform_sum,
    direct_trajectory_sum,
)


def _trajectory_observable(
    trajectory: PrescribedPointTrajectory,
    targets,
    *,
    observable: str,
    softening_m: float,
    chunk_size: int | None,
    backend: str,
) -> np.ndarray:
    g_free = direct_trajectory_sum(
        trajectory,
        targets,
        observable=observable,
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend=backend,
    )
    result = G_SI * g_free
    result.setflags(write=False)
    return result


def _rigid_observable(
    source: PointMassCloud,
    transform: PrescribedRigidTransform,
    targets,
    *,
    observable: str,
    softening_m: float,
    chunk_size: int | None,
    backend: str,
) -> np.ndarray:
    g_free = direct_rigid_transform_sum(
        source,
        transform,
        targets,
        observable=observable,
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend=backend,
    )
    result = G_SI * g_free
    result.setflags(write=False)
    return result


def trajectory_potential(
    trajectory: PrescribedPointTrajectory,
    targets,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> np.ndarray:
    """Return potential ``(P, M)`` for a prescribed point trajectory."""
    return _trajectory_observable(
        trajectory,
        targets,
        observable="potential",
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend=backend,
    )


def trajectory_acceleration(
    trajectory: PrescribedPointTrajectory,
    targets,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> np.ndarray:
    """Return acceleration ``(P, M, 3)`` for a prescribed point trajectory."""
    return _trajectory_observable(
        trajectory,
        targets,
        observable="acceleration",
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend=backend,
    )


def trajectory_gravity_gradient(
    trajectory: PrescribedPointTrajectory,
    targets,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> np.ndarray:
    """Return gravity gradient ``(P, M, 3, 3)`` for a point trajectory."""
    return _trajectory_observable(
        trajectory,
        targets,
        observable="gradient",
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend=backend,
    )


def rigid_transform_potential(
    source: PointMassCloud,
    transform: PrescribedRigidTransform,
    targets,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> np.ndarray:
    """Return potential ``(P, M)`` for a transformed point-mass cloud."""
    return _rigid_observable(
        source,
        transform,
        targets,
        observable="potential",
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend=backend,
    )


def rigid_transform_acceleration(
    source: PointMassCloud,
    transform: PrescribedRigidTransform,
    targets,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> np.ndarray:
    """Return acceleration ``(P, M, 3)`` for a transformed point cloud."""
    return _rigid_observable(
        source,
        transform,
        targets,
        observable="acceleration",
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend=backend,
    )


def rigid_transform_gravity_gradient(
    source: PointMassCloud,
    transform: PrescribedRigidTransform,
    targets,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> np.ndarray:
    """Return gradient ``(P, M, 3, 3)`` for a transformed point cloud."""
    return _rigid_observable(
        source,
        transform,
        targets,
        observable="gradient",
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend=backend,
    )


__all__ = [
    "trajectory_potential",
    "trajectory_acceleration",
    "trajectory_gravity_gradient",
    "rigid_transform_potential",
    "rigid_transform_acceleration",
    "rigid_transform_gravity_gradient",
]
