"""Parametric inverse square gravity kernels.

Computes potential, acceleration, and gravity gradient contractions with
multi-dimensional parametric weights.
All kernel contractions are G-free (do not multiply by G_SI).
"""

import numpy as np

from gragra._arrays import as_xyz_table
from gragra.array_types import ComplexArray, FloatArray


def _validate_chunk_size(chunk_size: int | None) -> None:
    """Validate a source-axis chunk_size: None or a positive int.

    Extracted from ``_validate_inputs_parametric`` so callers that need to
    fail fast on ``chunk_size`` before doing other (e.g. GPU) work can reuse
    the same check without duplicating the error message.
    """
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")


def _validate_inputs_parametric(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    softening_m: float,
    chunk_size: int | None,
) -> tuple[FloatArray, np.ndarray, FloatArray, float, int]:
    """Validate and normalize inputs for parametric kernel contractions."""
    if softening_m < 0:
        raise ValueError(f"softening_m must be non-negative, got {softening_m}")

    _validate_chunk_size(chunk_size)

    sources = as_xyz_table("sources_xyz", sources_xyz)
    targets = as_xyz_table("targets_xyz", targets_xyz)

    # Validate weights
    w_arr = np.asarray(weights)
    dtype = np.complex128 if np.iscomplexobj(w_arr) else np.float64
    try:
        w = np.asarray(weights, dtype=dtype)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"weights must be numeric convertible to float64 or complex128: {e}"
        ) from e

    if w.ndim < 1:
        raise ValueError(f"weights must have at least 1 dimension, got {w.ndim}")

    if len(sources) != w.shape[0]:
        raise ValueError(
            "Length of sources_xyz and weights first axis must match: "
            f"{len(sources)} != {w.shape[0]}"
        )

    if not np.isfinite(w).all():
        raise ValueError("weights must contain only finite numbers")

    w = w.copy()
    w.setflags(write=False)

    # Use max(n_sources, 1) so that N=0 (empty source) yields a zero-iteration
    # loop returning zeros, matching the numba/cpp backends, rather than raising
    # range() step-zero ValueError.
    n_sources = len(sources)
    actual_chunk_size = chunk_size if chunk_size is not None else max(n_sources, 1)

    return sources, w, targets, float(softening_m), actual_chunk_size


def _flatten_weights(
    w: np.ndarray, n_sources: int
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Flatten ``(N, *P)`` weights to a contiguous ``(N, P_flat)`` 2-D array.

    Returns the flattened array and the trailing parametric shape ``P`` so the
    compiled backends can operate on a fixed 2-D interface and reshape the
    result back to ``(M, *P, ...)``. ``P_flat = prod(P)`` and is 1 when ``P`` is
    empty (scalar weights ``(N,)``).
    """
    p_shape = w.shape[1:]
    p_flat = int(np.prod(p_shape)) if p_shape else 1
    return np.ascontiguousarray(w).reshape(n_sources, p_flat), p_shape


def _check_chunk_collision_parametric(xp, r2, start: int) -> None:
    """Raise ValueError if any pairwise distance in this chunk is zero."""
    zero_mask = r2 == 0.0
    if bool(zero_mask.any()):
        zero_idx = xp.nonzero(zero_mask)
        target_idx = int(zero_idx[0][0])
        source_idx = start + int(zero_idx[1][0])
        raise ValueError(
            "Collision detected between target point "
            f"{target_idx} and source point {source_idx} "
            "with zero softening."
        )


def _acceleration_impl_parametric(
    xp,
    sources: FloatArray,
    w: FloatArray | ComplexArray,
    targets: FloatArray,
    eps: float,
    c_size: int,
    p_shape: tuple[int, ...],
    out: FloatArray | ComplexArray,
) -> None:
    """xp-agnostic chunk loop for the parametric acceleration contraction."""
    n_sources = len(sources)
    eps2 = eps**2

    for start in range(0, n_sources, c_size):
        end = min(start + c_size, n_sources)
        s_chunk = sources[start:end]
        w_chunk = w[start:end]

        diff = s_chunk[xp.newaxis, :, :] - targets[:, xp.newaxis, :]
        r2 = xp.sum(diff**2, axis=-1)

        if eps == 0.0:
            _check_chunk_collision_parametric(xp, r2, start)

        r_eff = xp.sqrt(r2 + eps2)
        r_eff3 = (r2 + eps2) * r_eff

        term = diff / r_eff3[:, :, xp.newaxis]

        term_expanded = term[
            (slice(None), slice(None)) + (xp.newaxis,) * len(p_shape) + (slice(None),)
        ]
        w_expanded = w_chunk[xp.newaxis, ..., xp.newaxis]

        out += xp.sum(term_expanded * w_expanded, axis=1)


def _gradient_impl_parametric(
    xp,
    sources: FloatArray,
    w: FloatArray | ComplexArray,
    targets: FloatArray,
    eps: float,
    c_size: int,
    p_shape: tuple[int, ...],
    out: FloatArray | ComplexArray,
) -> None:
    """xp-agnostic chunk loop for the parametric gradient contraction."""
    n_sources = len(sources)
    eps2 = eps**2
    eye3 = xp.eye(3)

    for start in range(0, n_sources, c_size):
        end = min(start + c_size, n_sources)
        s_chunk = sources[start:end]
        w_chunk = w[start:end]

        diff = s_chunk[xp.newaxis, :, :] - targets[:, xp.newaxis, :]
        r2 = xp.sum(diff**2, axis=-1)

        if eps == 0.0:
            _check_chunk_collision_parametric(xp, r2, start)

        r_eff = xp.sqrt(r2 + eps2)
        r_eff3 = (r2 + eps2) * r_eff
        r_eff5 = r_eff3 * (r2 + eps2)

        diff_outer = diff[:, :, :, xp.newaxis] * diff[:, :, xp.newaxis, :]

        term = (
            3.0 * diff_outer / r_eff5[:, :, xp.newaxis, xp.newaxis]
            - eye3[xp.newaxis, xp.newaxis, :, :] / r_eff3[:, :, xp.newaxis, xp.newaxis]
        )

        term_expanded = term[
            (slice(None), slice(None))
            + (xp.newaxis,) * len(p_shape)
            + (slice(None), slice(None))
        ]
        w_expanded = w_chunk[xp.newaxis, ..., xp.newaxis, xp.newaxis]

        out += xp.sum(term_expanded * w_expanded, axis=1)


def potential_contract_parametric(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Calculate the G-free potential contraction with parametric weights.

    K_Φ = -1 / sqrt(r^2 + ε^2)

    Parameters
    ----------
    sources_xyz : array-like
        Source coordinates table of shape (N, 3) in meters.
    weights : array-like
        Weights vector of shape (N, *P) in kilograms (real or complex).
    targets_xyz : array-like
        Target coordinates table of shape (M, 3) in meters.
    softening_m : float, optional
        Plummer softening parameter in meters. Defaults to 0.0.
    chunk_size : int, optional
        Chunk size for source-axis loop to manage memory. Defaults to None.

    Returns
    -------
    np.ndarray
        Calculated potential of shape (M, *P).
    """
    sources, w, targets, eps, c_size = _validate_inputs_parametric(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )

    m_targets = len(targets)
    n_sources = len(sources)
    p_shape = w.shape[1:]

    out_dtype = np.complex128 if np.iscomplexobj(w) else np.float64
    out = np.zeros((m_targets,) + p_shape, dtype=out_dtype)

    eps2 = eps**2

    for start in range(0, n_sources, c_size):
        end = min(start + c_size, n_sources)
        s_chunk = sources[start:end]
        w_chunk = w[start:end]

        diff = s_chunk[np.newaxis, :, :] - targets[:, np.newaxis, :]
        r2 = np.sum(diff**2, axis=-1)

        if eps == 0.0:
            zero_idx = np.where(r2 == 0.0)
            if len(zero_idx[0]) > 0:
                target_idx = zero_idx[0][0]
                source_idx = start + zero_idx[1][0]
                raise ValueError(
                    "Collision detected between target point "
                    f"{target_idx} and source point {source_idx} "
                    "with zero softening."
                )

        r_eff = np.sqrt(r2 + eps2)
        term = -1.0 / r_eff

        term_expanded = term[(slice(None), slice(None)) + (np.newaxis,) * len(p_shape)]
        w_expanded = w_chunk[np.newaxis, ...]

        out += np.sum(term_expanded * w_expanded, axis=1)

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
    """Calculate G-free acceleration contraction with parametric weights.

    K_a = Δ / (r^2 + ε^2)^(3/2)
    where Δ = r'_j(source) - r_i(target)

    Parameters
    ----------
    sources_xyz : array-like
        Source coordinates table of shape (N, 3) in meters.
    weights : array-like
        Weights vector of shape (N, *P) in kilograms (real or complex).
    targets_xyz : array-like
        Target coordinates table of shape (M, 3) in meters.
    softening_m : float, optional
        Plummer softening parameter in meters. Defaults to 0.0.
    chunk_size : int, optional
        Chunk size for source-axis loop to manage memory. Defaults to None.

    Returns
    -------
    np.ndarray
        Calculated acceleration of shape (M, *P, 3).
    """
    sources, w, targets, eps, c_size = _validate_inputs_parametric(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )

    m_targets = len(targets)
    p_shape = w.shape[1:]

    out_dtype = np.complex128 if np.iscomplexobj(w) else np.float64
    out = np.zeros((m_targets,) + p_shape + (3,), dtype=out_dtype)

    _acceleration_impl_parametric(np, sources, w, targets, eps, c_size, p_shape, out)

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
    """Calculate G-free gravity gradient contraction with parametric weights.

    K_T = 3 * ΔΔ / (r^2 + ε^2)^(5/2) - I / (r^2 + ε^2)^(3/2)
    where Δ = r'_j(source) - r_i(target)
    and I is the 3x3 identity matrix.

    Note that for softening_m > 0, the softened tensor does not satisfy
    tr(T) = 0.

    Parameters
    ----------
    sources_xyz : array-like
        Source coordinates table of shape (N, 3) in meters.
    weights : array-like
        Weights vector of shape (N, *P) in kilograms (real or complex).
    targets_xyz : array-like
        Target coordinates table of shape (M, 3) in meters.
    softening_m : float, optional
        Plummer softening parameter in meters. Defaults to 0.0.
    chunk_size : int, optional
        Chunk size for source-axis loop to manage memory. Defaults to None.

    Returns
    -------
    np.ndarray
        Calculated gravity gradient tensor of shape (M, *P, 3, 3).
    """
    sources, w, targets, eps, c_size = _validate_inputs_parametric(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )

    m_targets = len(targets)
    p_shape = w.shape[1:]

    out_dtype = np.complex128 if np.iscomplexobj(w) else np.float64
    out = np.zeros((m_targets,) + p_shape + (3, 3), dtype=out_dtype)

    _gradient_impl_parametric(np, sources, w, targets, eps, c_size, p_shape, out)

    out.setflags(write=False)
    return out
