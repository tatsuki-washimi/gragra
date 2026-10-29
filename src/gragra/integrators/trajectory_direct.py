"""G-free direct integration for prescribed source motion."""

import operator

import numpy as np

from gragra.gravity_sources import PointMass, PointMassCloud, WeightedPointSource
from gragra.gravity_sources.trajectory import (
    PrescribedPointTrajectory,
    PrescribedRigidTransform,
    _reject_bool_or_complex,
)
from gragra.integrators.point_direct import _normalize_targets, direct_point_sum

_BACKENDS = ("numpy", "numba", "cpp", "cupy")


def _validate_options(
    targets, softening_m: float, chunk_size: int | None, backend: str
) -> tuple[np.ndarray, float, int | None, str]:
    """Validate shared options and targets once, before any frame is evaluated."""
    _reject_bool_or_complex(targets, "targets")
    target_array = _normalize_targets(targets)

    if isinstance(softening_m, (bool, np.bool_)):
        raise ValueError("softening_m must be a finite non-negative scalar")
    try:
        softening = float(np.asarray(softening_m))
    except (TypeError, ValueError) as error:
        raise ValueError("softening_m must be a finite non-negative scalar") from error
    if not np.isfinite(softening) or softening < 0.0:
        raise ValueError(
            f"softening_m must be a finite non-negative scalar, got {softening_m}"
        )

    if chunk_size is not None:
        if isinstance(chunk_size, (bool, np.bool_)):
            raise ValueError("chunk_size must be a positive integer")
        try:
            chunk_size = operator.index(chunk_size)
        except TypeError as error:
            raise ValueError("chunk_size must be a positive integer") from error
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

    if backend not in _BACKENDS:
        raise ValueError(
            "Unsupported backend: must be 'numpy', 'numba', 'cpp', or 'cupy', "
            f"got '{backend}'"
        )
    return target_array, softening, chunk_size, backend


def _find_collision(
    source_positions: np.ndarray, target_positions: np.ndarray
) -> tuple[int, int] | None:
    """Find the lexicographically earliest target/source exact coordinate match."""
    block_size = 65536
    for target_index, target in enumerate(target_positions):
        for start in range(0, len(source_positions), block_size):
            block = source_positions[start : start + block_size]
            matches = np.all(block == target, axis=1)
            if np.any(matches):
                return target_index, start + int(np.flatnonzero(matches)[0])
    return None


def _call_frame(
    source: WeightedPointSource | PointMass | PointMassCloud,
    targets: np.ndarray,
    *,
    observable: str,
    softening_m: float,
    chunk_size: int | None,
    backend: str,
    frame_index: int,
    source_positions: np.ndarray,
) -> np.ndarray:
    """Call the established direct path and refine only collision failures."""
    try:
        return direct_point_sum(
            source,
            targets,
            observable=observable,
            softening_m=softening_m,
            chunk_size=chunk_size,
            backend=backend,
        )
    except ValueError as error:
        if softening_m == 0.0:
            collision = _find_collision(source_positions, targets)
            if collision is not None:
                target_index, source_index = collision
                raise ValueError(
                    "Collision detected between frame "
                    f"{frame_index}, target {target_index}, source {source_index} "
                    "with zero softening."
                ) from error
        raise


def direct_trajectory_sum(
    trajectory: PrescribedPointTrajectory,
    targets,
    *,
    observable: str,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> np.ndarray:
    """Evaluate one G-free observable at every prescribed point frame."""
    if not isinstance(trajectory, PrescribedPointTrajectory):
        raise TypeError("trajectory must be a PrescribedPointTrajectory")
    target_array, softening, chunk_size, backend = _validate_options(
        targets, softening_m, chunk_size, backend
    )
    frame_count = len(trajectory.time_s)
    target_count = len(target_array)
    shape = {
        "potential": (frame_count, target_count),
        "acceleration": (frame_count, target_count, 3),
        "gradient": (frame_count, target_count, 3, 3),
    }.get(observable)
    if shape is None:
        raise ValueError(
            "observable must be 'potential', 'acceleration', or 'gradient'"
        )
    result = np.zeros(shape, dtype=np.float64)
    for frame_index, position in enumerate(trajectory.positions_m):
        source = PointMass(trajectory.mass_kg, position)
        result[frame_index] = _call_frame(
            source,
            target_array,
            observable=observable,
            softening_m=softening,
            chunk_size=chunk_size,
            backend=backend,
            frame_index=frame_index,
            source_positions=source.position_m[np.newaxis, :],
        )
    result.setflags(write=False)
    return result


def direct_rigid_transform_sum(
    source: PointMassCloud,
    transform: PrescribedRigidTransform,
    targets,
    *,
    observable: str,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> np.ndarray:
    """Evaluate one G-free observable at every rigid-transform frame."""
    if not isinstance(source, PointMassCloud):
        raise TypeError("source must be a PointMassCloud")
    if not isinstance(transform, PrescribedRigidTransform):
        raise TypeError("transform must be a PrescribedRigidTransform")
    target_array, softening, chunk_size, backend = _validate_options(
        targets, softening_m, chunk_size, backend
    )
    frame_count = len(transform.time_s)
    target_count = len(target_array)
    shape = {
        "potential": (frame_count, target_count),
        "acceleration": (frame_count, target_count, 3),
        "gradient": (frame_count, target_count, 3, 3),
    }.get(observable)
    if shape is None:
        raise ValueError(
            "observable must be 'potential', 'acceleration', or 'gradient'"
        )
    result = np.zeros(shape, dtype=np.float64)
    for frame_index, (translation, rotation) in enumerate(
        zip(transform.translations_m, transform.rotation_matrices, strict=True)
    ):
        world_positions = source.positions_m @ rotation.T + translation
        frame_source = PointMassCloud(source.masses_kg, world_positions)
        result[frame_index] = _call_frame(
            frame_source,
            target_array,
            observable=observable,
            softening_m=softening,
            chunk_size=chunk_size,
            backend=backend,
            frame_index=frame_index,
            source_positions=world_positions,
        )
    result.setflags(write=False)
    return result
