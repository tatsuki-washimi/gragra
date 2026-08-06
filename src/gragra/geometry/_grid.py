"""Private grid generation and validation helpers for geometry discretization."""

from typing import Any

import numpy as np

from gragra.array_types import FloatArray


def _validate_positive_scalar(name: str, value: Any) -> float:
    """Validate that a value is a positive finite numeric scalar.

    Rejects bool, non-scalar arrays, non-finite values, and negative/zero values.
    """
    if isinstance(value, bool) or (hasattr(value, "dtype") and value.dtype == bool):
        raise TypeError(f"{name} must be numeric, not bool")
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be numeric, not string/bytes")
    try:
        arr = np.asarray(value)
        if arr.ndim > 0:
            raise TypeError(f"{name} must be a scalar, got shape {arr.shape}")
        if arr.dtype.kind in "SUOb":
            raise TypeError(f"{name} must not be string, bytes, object, or bool")
        raw_val = arr.item()
        if isinstance(raw_val, (bool, np.bool_)):
            raise TypeError(f"{name} must be numeric, not bool")
        val = float(raw_val)
    except (ValueError, TypeError) as e:
        raise TypeError(f"{name} must be a numeric scalar: {e}") from e

    if not np.isfinite(val):
        raise ValueError(f"{name} must be finite")
    if val <= 0:
        raise ValueError(f"{name} must be positive, got {val}")
    return val


def _validate_lengths(name: str, value: Any) -> FloatArray:
    """Validate that value is a 3-element vector of positive numeric scalars."""
    if isinstance(value, bool) or (hasattr(value, "dtype") and value.dtype == bool):
        raise TypeError(f"{name} must be numeric, not bool")
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be numeric, not string/bytes")

    # Reject bool/string elements in iterable
    if hasattr(value, "__iter__"):
        if isinstance(value, np.ndarray) and value.dtype.kind in "SUOb":
            raise TypeError(f"{name} must not be string, bytes, object, or bool dtype")
        for item in value:
            if isinstance(item, (bool, np.bool_)):
                raise TypeError(f"Elements of {name} must not be bool")
            if isinstance(item, (str, bytes)):
                raise TypeError(f"Elements of {name} must not be string/bytes")
            if hasattr(item, "dtype") and item.dtype.kind in "SUOb":
                raise TypeError(
                    f"Elements of {name} must not be string/bytes/object/bool"
                )

    try:
        arr = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError(f"{name} must be convertible to float64 array: {e}") from e

    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")

    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite numbers")
    if (arr <= 0).any():
        raise ValueError(f"{name} must contain only positive values")

    arr = arr.copy()
    arr.setflags(write=False)
    return arr


def _validate_density(value: Any) -> float | complex:
    """Validate density_kg_m3 to ensure it is a finite scalar.

    Rejects bool, non-scalar arrays, and non-finite values.
    Preserves float/complex type.
    """
    if isinstance(value, bool) or (hasattr(value, "dtype") and value.dtype == bool):
        raise TypeError("density_kg_m3 must not be bool")
    try:
        arr = np.asarray(value)
    except (ValueError, TypeError) as e:
        raise TypeError(f"density_kg_m3 must be convertible to numpy array: {e}") from e

    if arr.ndim > 0:
        raise ValueError(f"density_kg_m3 must be a scalar, got shape {arr.shape}")

    if arr.dtype == object:
        raise TypeError("density_kg_m3 must be a numeric type")

    try:
        val = arr.item()
    except (ValueError, TypeError) as e:
        raise TypeError(f"density_kg_m3 must be a numeric scalar: {e}") from e

    if not isinstance(val, (int, float, complex, np.number)):
        raise TypeError(f"density_kg_m3 must be numeric, got {type(val)}")

    if not np.isfinite(val):
        raise ValueError("density_kg_m3 must be finite")

    if np.iscomplexobj(val):
        return complex(val)
    return float(val)


def _validate_max_points(value: Any) -> int | None:
    """Validate max_points to ensure it is None or a positive integer.

    Rejects bool.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("max_points must be an integer, not bool")
    if isinstance(value, (str, bytes)):
        raise TypeError("max_points must be an integer, not string/bytes")
    try:
        arr = np.asarray(value)
        if arr.ndim > 0:
            raise TypeError(f"max_points must be a scalar, got shape {arr.shape}")
        if arr.dtype.kind in "SUOb":
            raise TypeError("max_points must not be string, bytes, object, or bool")
        val = arr.item()
        if isinstance(val, (bool, np.bool_)):
            raise TypeError("max_points must be an integer, not bool")
    except (ValueError, TypeError) as e:
        raise TypeError(f"max_points must be an integer: {e}") from e

    if not isinstance(val, (int, np.integer)):
        raise TypeError(f"max_points must be an integer, got {type(val)}")

    if val <= 0:
        raise ValueError("max_points must be positive")
    return int(val)


def _generate_grid(
    center_m: FloatArray,
    bbox_lengths: FloatArray,
    spacing_m: float,
    max_points: int | None,
) -> tuple[FloatArray, float]:
    """Generate Cartesian midpoint grid coordinates and the cell volume.

    Parameters
    ----------
    center_m : FloatArray
        Shape (3,) center of the bounding box.
    bbox_lengths : FloatArray
        Shape (3,) lengths of the bounding box.
    spacing_m : float
        Grid spacing (maximum cell width).
    max_points : int or None
        Maximum allowed grid points.

    Returns
    -------
    tuple[FloatArray, float]
        - A read-only float64 array of shape (K, 3) containing grid cell centers.
        - The cell volume dV (dx * dy * dz) in cubic meters.
    """
    lx, ly, lz = bbox_lengths

    n_x = int(np.ceil(lx / spacing_m))
    n_y = int(np.ceil(ly / spacing_m))
    n_z = int(np.ceil(lz / spacing_m))

    total_bbox_points = n_x * n_y * n_z
    if max_points is not None and total_bbox_points > max_points:
        raise ValueError(
            f"bbox would require {total_bbox_points} cells, "
            f"exceeds max_points={max_points}"
        )

    x_min, x_max = center_m[0] - lx / 2.0, center_m[0] + lx / 2.0
    x_bounds = np.linspace(x_min, x_max, n_x + 1)
    x_centers = (x_bounds[:-1] + x_bounds[1:]) / 2.0
    dx = lx / n_x

    y_min, y_max = center_m[1] - ly / 2.0, center_m[1] + ly / 2.0
    y_bounds = np.linspace(y_min, y_max, n_y + 1)
    y_centers = (y_bounds[:-1] + y_bounds[1:]) / 2.0
    dy = ly / n_y

    z_min, z_max = center_m[2] - lz / 2.0, center_m[2] + lz / 2.0
    z_bounds = np.linspace(z_min, z_max, n_z + 1)
    z_centers = (z_bounds[:-1] + z_bounds[1:]) / 2.0
    dz = lz / n_z

    x_mesh, y_mesh, z_mesh = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    grid_points = np.stack([x_mesh, y_mesh, z_mesh], axis=-1).reshape(-1, 3)
    grid_points.setflags(write=False)

    dv = dx * dy * dz
    return grid_points, dv
