"""Continuous-time response of one rigid body sampled from a pose function.

This private module turns a single rigid point-mass cloud and a continuous
body pose into a :data:`~gragra._continuous_contract.ResponseFunction`. The
response accepts arbitrary relative times (any order, duplicates allowed) and
returns time-first SI arrays that match the M3-A rigid-motion layout:
potential ``(Ntime, Ntarget)``, acceleration ``(Ntime, Ntarget, 3)``, and
gradient ``(Ntime, Ntarget, 3, 3)``.

Method:

1. Validate the request times with :func:`normalize_relative_times`.
2. Sort the times and remove duplicates. The sorted unique times define the
   sampled motion, because :class:`PrescribedRigidTransform` requires strictly
   increasing ``time_s``.
3. Evaluate the pose function once at the unique times and build the sampled
   transform ``x = R(t) x_body + c(t)``.
4. Evaluate the transform through the private M3-A streaming executor. The
   gravitational constant is applied once inside that path, not here.
5. Restore the caller's order, including duplicates.

Target frames: a framed high-level ``TargetPoints`` is converted once, at
construction, into the computation frame ``frame`` with
``targets.frame.transform_points_m``. A missing frame path raises the
frame module's ``ValueError``. A numerical ``TargetPoints`` has no frame, so
its positions are read as already expressed in ``frame``. Vector and tensor
components of the output are in the basis of ``frame``, the same basis the
pose uses.

Memory: the arrays this module holds (pose arrays, sorted times, the dense
unique-time result, and the final reordered copy) are estimated before any
work starts. The estimate is compared with the managed-array budget, and the
streaming executor receives the amount it must reserve in its own planning.
Results are bounded by the budget, not by process RSS.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u

from gragra._continuous_contract import (
    ResponseFunction,
    check_memory_budget,
    component_shape,
    normalize_relative_times,
    validate_observable,
)
from gragra.geometry._frame import Frame
from gragra.gravity_sources import PointMassCloud
from gragra.gravity_sources.trajectory import PrescribedRigidTransform
from gragra.observables.rigid_motion import _stream_rigid_motion
from gragra.targets._high_points import TargetPoints as FramedTargetPoints
from gragra.targets.points import TargetPoints

_FLOAT_BYTES = np.dtype(np.float64).itemsize
_INDEX_BYTES = np.dtype(np.intp).itemsize

# Physical units of the time-first responses, indexed by observable name.
_OBSERVABLE_UNITS: dict[str, u.UnitBase] = {
    "potential": u.m**2 / u.s**2,
    "acceleration": u.m / u.s**2,
    "gradient": u.s**-2,
}

# Tile sizes match the public M3-A rigid-motion observable defaults.
_MOTION_BLOCK_SIZE = 1
_TARGET_BLOCK_SIZE = 128
_SOURCE_BLOCK_SIZE = 8192
_KERNEL_CHUNK_SIZE = 1024


def _estimate_managed_bytes(
    *, n_input: int, n_unique: int, n_target: int, n_component: int
) -> int:
    """Estimate the managed-array bytes of one response call.

    Counts the pose arrays (rotations and translations), the sorted unique
    times, the input-to-unique index map, the dense unique-time result, the
    per-(time, target) coverage bytes used by the M3-A executor, and the final
    reordered result copy. Process RSS is not included.
    """
    pose_bytes = n_unique * (9 + 3) * _FLOAT_BYTES
    time_bytes = n_unique * _FLOAT_BYTES
    index_bytes = n_input * _INDEX_BYTES
    dense_bytes = n_unique * n_target * n_component * _FLOAT_BYTES
    coverage_bytes = n_unique * n_target
    output_bytes = n_input * n_target * n_component * _FLOAT_BYTES
    return int(
        pose_bytes
        + time_bytes
        + index_bytes
        + dense_bytes
        + coverage_bytes
        + output_bytes
    )


def _target_positions_in_frame(targets: object, frame: Frame) -> np.ndarray:
    """Return read-only target positions (m) expressed in ``frame``.

    A framed high-level ``TargetPoints`` is transformed with its own frame. A
    numerical ``TargetPoints`` carries no frame, so its positions are taken as
    already expressed in ``frame``.
    """
    if isinstance(targets, FramedTargetPoints):
        positions = targets.frame.transform_points_m(targets.positions_m, frame)
    elif isinstance(targets, TargetPoints):
        positions = targets.positions_m
    else:
        raise TypeError("targets must be a TargetPoints or a framed TargetPoints")
    result = np.array(positions, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _validate_pose_output(pose_output: object, n_unique: int):
    """Check the ``(R, c)`` pair returned by the pose and return plain arrays."""
    if not isinstance(pose_output, tuple) or len(pose_output) != 2:
        raise TypeError("pose must return a (Rotation, Quantity) pair")
    rotation, translation = pose_output
    if not isinstance(translation, u.Quantity):
        raise TypeError("pose translations must be an astropy Quantity in metres")
    try:
        translations_m = np.asarray(translation.to_value(u.m), dtype=np.float64)
    except (u.UnitsError, TypeError) as exc:
        raise ValueError("pose translations must have length units") from exc
    if translations_m.shape != (n_unique, 3):
        raise ValueError(
            f"pose translations must have shape ({n_unique}, 3), "
            f"got {translations_m.shape}"
        )
    if not hasattr(rotation, "as_matrix"):
        raise TypeError("pose must return a scipy.spatial.transform.Rotation")
    matrices = np.asarray(rotation.as_matrix(), dtype=np.float64).reshape(-1, 3, 3)
    if matrices.shape[0] != n_unique:
        raise ValueError(
            f"pose rotations must have {n_unique} entries, got {matrices.shape[0]}"
        )
    return matrices, translations_m


def make_rigid_pose_response(
    source: PointMassCloud,
    pose,
    targets: TargetPoints | FramedTargetPoints,
    *,
    frame: Frame,
    memory_budget_bytes: int,
    softening_m: float = 0.0,
) -> ResponseFunction:
    """Return a response function for one rigid body with a continuous pose.

    Parameters
    ----------
    source:
        Positive-mass point cloud in the body frame. Only the public
        ``PointMassCloud`` type is accepted.
    pose:
        Callable ``pose(times) -> (Rotation, Quantity[m] (N, 3))`` that maps the
        body into ``frame`` as ``x = R x_body + c``. ``times`` is a Quantity in
        seconds relative to the GPS epoch of the caller.
    targets:
        Observation points. A numerical ``TargetPoints`` has no frame, and its
        positions are read as expressed in ``frame``. A high-level
        ``TargetPoints`` carries its own frame and is converted into ``frame``
        at construction.
    frame:
        Computation frame of the pose. Vector and tensor components of every
        returned observable are in the basis of this frame.
    memory_budget_bytes:
        Budget for managed arrays. Exceeding it raises ``MemoryError``.
    softening_m:
        Plummer softening length in metres. Zero with a collision raises
        ``ValueError`` from the kernel path.

    Returns
    -------
    ResponseFunction
        ``response(times, observable)`` returning a time-first SI Quantity.
        Output order and duplicates follow the input ``times``. The result is
        read-only.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If times, the pose output, or a collision (with zero softening) is invalid,
        or if a framed target has no frame path to ``frame``.
    MemoryError
        If the managed arrays exceed ``memory_budget_bytes``.

    Notes
    -----
    The gravitational constant is applied once, by the M3-A path. Use the
    public point observables to cross-check results at a fixed pose.
    This response depends on the private M3-A executor
    ``gragra.observables.rigid_motion._stream_rigid_motion``. That dependency
    is a documented exception; the public streaming API reserves no bytes.
    """
    if not isinstance(source, PointMassCloud):
        raise TypeError("source must be a PointMassCloud")
    if not isinstance(frame, Frame):
        raise TypeError("frame must be a Frame")
    if not callable(pose):
        raise TypeError("pose must be callable")
    # Validate the budget type and sign eagerly; zero estimate never exceeds it.
    check_memory_budget(0, memory_budget_bytes)
    # Frame conversion happens once here, so a missing frame path fails early.
    target_positions_m = _target_positions_in_frame(targets, frame)

    def response(times, observable: str) -> u.Quantity:
        """Return the SI response at ``times`` in the caller's order."""
        name = validate_observable(observable)
        seconds = normalize_relative_times(times).to_value(u.s)
        unique_s, inverse = np.unique(seconds, return_inverse=True)
        inverse = np.asarray(inverse, dtype=np.intp).reshape(-1)
        n_unique = int(unique_s.size)
        n_target = int(target_positions_m.shape[0])
        components = component_shape(name)
        n_component = int(np.prod(components, dtype=np.int64)) or 1

        estimate = _estimate_managed_bytes(
            n_input=int(seconds.size),
            n_unique=n_unique,
            n_target=n_target,
            n_component=n_component,
        )
        check_memory_budget(estimate, memory_budget_bytes)

        matrices, translations_m = _validate_pose_output(pose(unique_s * u.s), n_unique)
        transform = PrescribedRigidTransform(
            time_s=unique_s,
            translations_m=translations_m,
            rotation_matrices=matrices,
        )

        # Bytes this module holds while the executor streams: pose arrays,
        # sorted times, index map, the dense result, and executor coverage.
        # The final reordered copy is inside the estimate checked above.
        dense = np.empty((n_unique, n_target, *components), dtype=np.float64)
        reserved = (
            matrices.nbytes
            + translations_m.nbytes
            + unique_s.nbytes
            + inverse.nbytes
            + dense.nbytes
            + n_unique * n_target
        )
        blocks = _stream_rigid_motion(
            source,
            transform,
            target_positions_m,
            observable=name,
            softening_m=softening_m,
            memory_budget_bytes=memory_budget_bytes,
            motion_block_size=_MOTION_BLOCK_SIZE,
            target_block_size=_TARGET_BLOCK_SIZE,
            source_block_size=_SOURCE_BLOCK_SIZE,
            kernel_chunk_size=_KERNEL_CHUNK_SIZE,
            reserved_bytes=reserved,
        )
        for block in blocks:
            dense[block.motion_slice, block.target_slice] = block.values

        # Fancy indexing restores the caller's order and duplicates.
        ordered = dense[inverse]
        ordered.setflags(write=False)
        return u.Quantity(ordered, _OBSERVABLE_UNITS[name], copy=False)

    return response


__all__ = ["make_rigid_pose_response"]
