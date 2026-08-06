"""Shared scalar/array input validation helpers for the `analytic/` subpackage.

Extracted from `cavity_sphere.py` so that `cavity_coupling.py`,
`cavity_cuboid.py` and `cavity_cylinder.py` share one validation
implementation instead of each carrying a module-private copy.
"""

import numpy as np


def validate_scalar_input(val: float, name: str) -> float:
    """Validate positive finite real scalar input."""
    if isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
        raise TypeError(f"{name} must be a numeric scalar")
    val_float = float(val)
    if not np.isfinite(val_float):
        raise ValueError(f"{name} must be finite")
    if val_float <= 0.0:
        raise ValueError(f"{name} must be positive")
    return val_float


def _finite_array(val: float | list | tuple | np.ndarray, name: str) -> np.ndarray:
    """Cast to float64 and reject non-finite values.

    Shared by `validate_array_input` and `validate_coordinate_input`, which
    otherwise duplicated this exact cast-and-check.
    """
    arr = np.asarray(val, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def validate_array_input(
    val: float | list | tuple | np.ndarray, name: str
) -> np.ndarray:
    """Validate non-negative finite array-like input."""
    arr = _finite_array(val, name)
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return arr


def validate_coordinate_input(
    val: float | list | tuple | np.ndarray, name: str
) -> np.ndarray:
    """Validate finite scalar/1-D coordinate input (negative/zero allowed).

    Unlike `validate_array_input` (non-negative -- meant for wavenumbers,
    radii, etc.), coordinate values are signed by nature (position relative
    to an origin/center). Added for `cavity_cylinder.py`'s
    on-axis `z_m` array, which is not a `(3,)`/`(N,3)` vector/table (so
    `gragra._arrays.as_xyz_vector`/`as_xyz_table` do not apply) but still
    must permit negative values.

    Restricted to ndim <= 1 (scalar or shape (M,)): this validator has no
    `(N,3)`-table counterpart, so a caller accidentally passing an (M,3)
    array of xyz points here (rather than a single-axis coordinate array)
    would otherwise be silently accepted and broadcast element-wise instead
    of raising.
    """
    arr = _finite_array(val, name)
    if arr.ndim > 1:
        raise ValueError(f"{name} must be a scalar or 1-D array, got shape {arr.shape}")
    return arr
