"""Internal array normalization and validation helpers."""

import numpy as np

from gragra.array_types import ComplexArray, FloatArray


def as_xyz_vector(name: str, value: list | tuple | np.ndarray) -> FloatArray:
    """Normalize and validate a single 3D vector.

    Expects a value convertable to a float64 array of shape (3,).
    Sets the writeable flag of the returned array to False (read-only).

    Parameters
    ----------
    name : str
        Parameter name for error messages.
    value : array-like
        The input coordinates, e.g., [x, y, z].

    Returns
    -------
    FloatArray
        A read-only float64 array of shape (3,).
    """
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError(f"{name} must be numeric convertible to float64: {e}") from e

    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")

    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite numbers")

    # Defensive copy and set to read-only
    arr = arr.copy()
    arr.setflags(write=False)
    return arr


def as_xyz_table(name: str, value: list | tuple | np.ndarray) -> FloatArray:
    """Normalize and validate a table of 3D vectors.

    Expects a value convertable to a float64 array of shape (N, 3).
    Strictly rejects shape (3,).
    Sets the writeable flag of the returned array to False (read-only).

    Parameters
    ----------
    name : str
        Parameter name for error messages.
    value : array-like
        The input coordinates table, e.g., [[x1, y1, z1], [x2, y2, z2], ...].

    Returns
    -------
    FloatArray
        A read-only float64 array of shape (N, 3).
    """
    is_1d = False
    try:
        val_arr = np.asarray(value)
        if val_arr.ndim == 1:
            is_1d = True
    except (ValueError, TypeError):
        pass

    if is_1d:
        raise ValueError(f"{name} must be a 2D table of shape (N, 3), not a 1D vector")

    try:
        arr = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError(f"{name} must be numeric convertible to float64: {e}") from e

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {arr.shape}")

    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite numbers")

    arr = arr.copy()
    arr.setflags(write=False)
    return arr


def as_weights(
    name: str, value: list | tuple | np.ndarray
) -> FloatArray | ComplexArray:
    """Normalize and validate weight vectors (real or complex).

    Expects a value convertable to float64 or complex128 array of shape (N,).
    Keeps real inputs as float64 and complex inputs as complex128.
    Sets the writeable flag of the returned array to False (read-only).

    Parameters
    ----------
    name : str
        Parameter name for error messages.
    value : array-like
        The weights array, e.g. [w1, w2, ...].

    Returns
    -------
    FloatArray or ComplexArray
        A read-only array of shape (N,).
    """
    raw_arr = np.asarray(value)

    # Determine appropriate dtype (preserve complex if input is complex)
    dtype = np.complex128 if np.iscomplexobj(raw_arr) else np.float64

    try:
        arr = np.asarray(value, dtype=dtype)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"{name} must be convertible to float64 or complex128: {e}"
        ) from e

    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a 1D array of shape (N,), got shape {arr.shape}"
        )

    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite numbers")

    arr = arr.copy()
    arr.setflags(write=False)
    return arr


def as_index_table(
    name: str, value: list | tuple | np.ndarray, n_max: int
) -> np.ndarray:
    """Normalize and validate an integer connectivity table of shape (Nt, 3).

    Expects integer vertex indices in the half-open range ``[0, n_max)``.
    Returns a read-only ``np.intp`` array of shape (Nt, 3), independent of the
    input integer dtype, so callers obtain a canonical connectivity table.

    This consolidates the integer-connectivity validation that is otherwise
    duplicated across the field/adapter helpers; it rejects bool, non-integer
    dtype, wrong shape, and both lower (``< 0``) and upper (``>= n_max``)
    out-of-range indices.

    Parameters
    ----------
    name : str
        Parameter name for error messages.
    value : array-like
        The connectivity table, e.g. [[i0, i1, i2], ...].
    n_max : int
        Exclusive upper bound for indices (typically the number of vertices).

    Returns
    -------
    np.ndarray
        A read-only ``np.intp`` array of shape (Nt, 3).
    """
    _check_no_bool(value, name)

    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{name} must be of integer type, got {raw.dtype}")

    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f"{name} must have shape (Nt, 3), got {raw.shape}")

    arr = raw.astype(np.intp, copy=True)
    if arr.size > 0 and ((arr < 0).any() or (arr >= n_max).any()):
        raise ValueError(f"{name} indices must be within [0, {n_max})")

    arr.setflags(write=False)
    return arr


def _check_no_bool(val, name):
    """Helper to raise TypeError if any boolean value is found in inputs."""
    if isinstance(val, (bool, np.bool_)):
        raise TypeError(f"{name} must not contain bool")
    if hasattr(val, "dtype"):
        if val.dtype.kind == "b":
            raise TypeError(f"{name} must not contain bool")
        if val.dtype.kind in "fiuc":
            return
    if isinstance(val, np.ndarray):
        for x in val.flat:
            _check_no_bool(x, name)
    elif isinstance(val, (list, tuple)):
        for x in val:
            _check_no_bool(x, name)
