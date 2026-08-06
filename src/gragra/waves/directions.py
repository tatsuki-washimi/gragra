"""Wave directions utilities."""

import numpy as np


def normalize_directions(directions: list | tuple | np.ndarray) -> np.ndarray:
    """Normalize direction vectors.

    Expects shape (3,) or (Ndir, 3).
    Returns a read-only float64 array of shape (Ndir, 3).
    Raises ValueError if any vector is zero or non-finite.
    """
    arr = np.asarray(directions)

    if arr.dtype.kind == "b" or isinstance(directions, bool):
        raise TypeError("directions must be numeric, not bool")

    try:
        arr = np.asarray(directions, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError(f"directions must be numeric: {e}") from e

    if arr.ndim == 1:
        if arr.shape != (3,):
            raise ValueError(f"1D direction must have shape (3,), got {arr.shape}")
        arr = arr[np.newaxis, :]
    elif arr.ndim == 2:
        if arr.shape[1] != 3:
            raise ValueError(
                f"2D directions must have shape (Ndir, 3), got {arr.shape}"
            )
    else:
        raise ValueError(f"directions must be 1D or 2D, got shape {arr.shape}")

    if not np.isfinite(arr).all():
        raise ValueError("directions must contain only finite numbers")

    lengths = np.sqrt(np.sum(arr**2, axis=-1, keepdims=True))
    if (lengths == 0.0).any():
        raise ValueError("directions must not contain zero vectors")

    normalized = arr / lengths
    normalized = normalized.copy()
    normalized.setflags(write=False)
    return normalized


def orthonormal_basis_from_direction(
    k_hat: list | tuple | np.ndarray,
    reference_axis: list | tuple | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build orthonormal basis (e1, e2) orthogonal to k_hat.

    k_hat is expected to be a single 3D vector of shape (3,) or (1, 3).
    If reference_axis is provided and not parallel (absolute dot product <= 0.99)
    to k_hat, e1 is defined by projecting reference_axis onto the plane
    orthogonal to k_hat. Otherwise, a stable deterministic reference axis is chosen.
    e2 is defined as k_hat_norm x e1 to complete a right-handed system.

    Returns tuple (e1, e2) of shape (3,).
    """
    k = np.asarray(k_hat, dtype=np.float64).squeeze()
    if k.shape != (3,):
        raise ValueError(
            f"k_hat must represent a single 3D vector, got shape {k.shape}"
        )

    if not np.isfinite(k).all():
        raise ValueError("k_hat must contain only finite numbers")

    k_len = np.linalg.norm(k)
    if k_len == 0.0:
        raise ValueError("k_hat must not be a zero vector")
    k_norm = k / k_len

    # Select reference axis
    if reference_axis is not None:
        ref = np.asarray(reference_axis, dtype=np.float64).squeeze()
        if ref.shape != (3,):
            raise ValueError(f"reference_axis must have shape (3,), got {ref.shape}")
        if not np.isfinite(ref).all():
            raise ValueError("reference_axis must contain only finite numbers")
        ref_len = np.linalg.norm(ref)
        if ref_len == 0.0:
            raise ValueError("reference_axis must not be a zero vector")
        ref_norm = ref / ref_len
    else:
        ref_norm = None

    # If reference_axis is not parallel to k_hat, choose a stable deterministic one
    is_parallel = True
    if ref_norm is not None:
        dot_val = np.abs(np.dot(k_norm, ref_norm))
        if dot_val <= 0.99:
            is_parallel = False

    if is_parallel:
        abs_k = np.abs(k_norm)
        min_idx = np.argmin(abs_k)
        ref_norm = np.zeros(3)
        ref_norm[min_idx] = 1.0

    # Project ref_norm onto the plane orthogonal to k_norm
    e1_raw = ref_norm - np.dot(ref_norm, k_norm) * k_norm
    e1 = e1_raw / np.linalg.norm(e1_raw)

    # e2 is the cross product of k_norm and e1
    e2 = np.cross(k_norm, e1)

    e1 = e1.copy()
    e2 = e2.copy()
    e1.setflags(write=False)
    e2.setflags(write=False)

    return e1, e2
