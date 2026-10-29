"""Shared scalar/array input validation helpers for the `analytic/` subpackage.

Extracted from `cavity_sphere.py` so that `cavity_coupling.py`,
`cavity_cuboid.py` and `cavity_cylinder.py` share one validation
implementation instead of each carrying a module-private copy.
"""

import numpy as np


def validate_scalar_input(val: float, name: str) -> float:
    """Validate positive finite real scalar input."""
    if isinstance(val, (bool, np.bool_)):
        raise TypeError(f"{name} must not be boolean")
    if isinstance(val, complex) or np.iscomplexobj(val):
        raise TypeError(f"{name} must be real numeric, not complex")
    if not isinstance(val, (int, float, np.number)):
        raise TypeError(f"{name} must be a numeric scalar")
    val_float = float(val)
    if not np.isfinite(val_float):
        raise ValueError(f"{name} must be finite")
    if val_float <= 0.0:
        raise ValueError(f"{name} must be positive")
    return val_float


def validate_nonnegative_scalar_input(val: float, name: str) -> float:
    """Validate non-negative finite real scalar input."""
    if isinstance(val, (bool, np.bool_)):
        raise TypeError(f"{name} must not be boolean")
    if isinstance(val, complex) or np.iscomplexobj(val):
        raise TypeError(f"{name} must be real numeric, not complex")
    if not isinstance(val, (int, float, np.number)):
        raise TypeError(f"{name} must be a numeric scalar")
    val_float = float(val)
    if not np.isfinite(val_float):
        raise ValueError(f"{name} must be finite")
    if val_float < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return val_float


def _has_bool(x: object) -> bool:
    """Check if x is a boolean or contains boolean elements in a list/tuple/array."""
    if isinstance(x, (bool, np.bool_)):
        return True
    if isinstance(x, (list, tuple)):
        return any(_has_bool(item) for item in x)
    if isinstance(x, np.ndarray) and x.dtype.kind == "b":
        return True
    return False


def _finite_array(val: float | list | tuple | np.ndarray, name: str) -> np.ndarray:
    """Cast to float64 and reject non-finite values.

    Shared by legacy validators `validate_array_input` and `validate_coordinate_input`.
    Preserves permissive casting (e.g. numeric conversion) for backwards compatibility.
    """
    arr = np.asarray(val, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def validate_array_input(
    val: float | list | tuple | np.ndarray, name: str
) -> np.ndarray:
    """Validate non-negative finite array-like input (legacy permissive cast)."""
    arr = _finite_array(val, name)
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return arr


def validate_coordinate_input(
    val: float | list | tuple | np.ndarray, name: str
) -> np.ndarray:
    """Validate finite scalar/1-D coordinate input (legacy permissive cast)."""
    arr = _finite_array(val, name)
    if arr.ndim > 1:
        raise ValueError(f"{name} must be a scalar or 1-D array, got shape {arr.shape}")
    return arr


def validate_strict_nonnegative_array(
    val: float | list | tuple | np.ndarray, name: str
) -> np.ndarray:
    """Validate non-negative finite real array (strictly rejecting boolean and complex)."""
    if _has_bool(val):
        raise TypeError(f"{name} must not be boolean")
    raw = np.asarray(val)
    if raw.dtype.kind == "c" or np.iscomplexobj(val):
        raise TypeError(f"{name} must be real numeric, not complex")
    if raw.dtype.kind == "O" or not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    arr = np.asarray(val, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return arr


def validate_real_xyz_vector(val: list | tuple | np.ndarray, name: str) -> np.ndarray:
    """Validate a finite real 3-vector (strictly rejecting boolean and complex)."""
    if _has_bool(val):
        raise TypeError(f"{name} must not be boolean")
    raw = np.asarray(val)
    if raw.dtype.kind == "c" or np.iscomplexobj(val):
        raise TypeError(f"{name} must be real numeric, not complex")
    if raw.dtype.kind == "O" or not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    arr = np.asarray(val, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    arr = arr.copy()
    arr.setflags(write=False)
    return arr


def validate_real_xyz_table(val: list | tuple | np.ndarray, name: str) -> np.ndarray:
    """Validate a table of finite real 3-vectors (strictly rejecting boolean and complex)."""
    if _has_bool(val):
        raise TypeError(f"{name} must not be boolean")
    raw = np.asarray(val)
    if raw.dtype.kind == "c" or np.iscomplexobj(val):
        raise TypeError(f"{name} must be real numeric, not complex")
    if raw.dtype.kind == "O" or not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    arr = np.asarray(val, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    arr = arr.copy()
    arr.setflags(write=False)
    return arr
