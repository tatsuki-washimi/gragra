"""GWexpy adapter implementation."""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import gwexpy


def _import_gwexpy():
    """Lazy import helper for gwexpy with a clear error message."""
    try:
        import gwexpy

        return gwexpy
    except ImportError as e:
        raise ImportError(
            "gwexpy is not installed in the current environment. "
            "Please install gwexpy to use this adapter."
        ) from e


def _validate_grid_shape(grid_shape: tuple[int, int, int]) -> None:
    """Validate that grid_shape is a 3-tuple of positive integers (excluding bool)."""
    if not isinstance(grid_shape, tuple):
        raise TypeError("grid_shape must be a tuple")
    if len(grid_shape) != 3:
        raise ValueError("grid_shape must be a 3-tuple (nx, ny, nz)")
    for n in grid_shape:
        if isinstance(n, bool) or not isinstance(n, numbers.Integral):
            raise TypeError("grid_shape elements must be positive integers")
        if n <= 0:
            raise ValueError("grid_shape elements must be positive integers")


def _validate_ts(t_s: float) -> float:
    """Validate that t_s is a finite real number (excluding bool and complex)."""
    if isinstance(t_s, bool):
        raise TypeError("t_s must be a finite real number")
    if not isinstance(t_s, (numbers.Real, np.integer, np.floating)):
        # This also rejects complex types (numbers.Complex but not numbers.Real).
        raise TypeError("t_s must be a finite real number")
    try:
        val = float(t_s)
    except (TypeError, ValueError) as e:
        raise TypeError("t_s must be a finite real number") from e
    if not np.isfinite(val):
        raise ValueError("t_s must be a finite real number")
    return val


def _validate_positions(positions_m: np.ndarray | None, n_points: int) -> None:
    """Validate that positions_m has shape (n_points, 3) and is finite and real."""
    if positions_m is None:
        return
    if not isinstance(positions_m, np.ndarray):
        raise TypeError("positions_m must be a numpy ndarray")
    if positions_m.ndim != 2 or positions_m.shape != (n_points, 3):
        raise ValueError(f"positions_m must have shape ({n_points}, 3)")
    if np.iscomplexobj(positions_m):
        raise TypeError("positions_m must be real-valued")
    if not np.isfinite(positions_m).all():
        raise ValueError("positions_m must contain only finite numbers")


def _validate_common(
    n_points: int,
    t_s: float,
    positions_m: np.ndarray | None,
    grid_shape: tuple[int, int, int] | None,
) -> float:
    """Validate shared arguments and return the validated t_s as a float.

    Validates t_s (finite real, not bool/complex), positions_m shape/dtype/finiteness,
    and grid_shape consistency with n_points when provided.
    """
    t_s = _validate_ts(t_s)
    _validate_positions(positions_m, n_points)
    if grid_shape is not None:
        _validate_grid_shape(grid_shape)
        nx, ny, nz = grid_shape
        if nx * ny * nz != n_points:
            raise ValueError(
                f"grid_shape product {nx * ny * nz} must match data points {n_points}"
            )
    return t_s


def _prepare_axes(
    n_points: int, grid_shape: tuple[int, int, int] | None, t_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, str]]:
    """Prepare time and spatial axes coordinate arrays and metadata."""
    axis0 = np.array([t_s], dtype=np.float64)
    if grid_shape is None:
        axis1 = np.arange(n_points, dtype=np.float64)
        axis2 = np.array([0.0], dtype=np.float64)
        axis3 = np.array([0.0], dtype=np.float64)
        axis_names = ["time", "point", "dummy_y", "dummy_z"]
        space_domain = {"point": "real", "dummy_y": "real", "dummy_z": "real"}
    else:
        nx, ny, nz = grid_shape
        axis1 = np.arange(nx, dtype=np.float64)
        axis2 = np.arange(ny, dtype=np.float64)
        axis3 = np.arange(nz, dtype=np.float64)
        axis_names = ["time", "x", "y", "z"]
        space_domain = {"x": "real", "y": "real", "z": "real"}
    return axis0, axis1, axis2, axis3, axis_names, space_domain


def _prepare_4d(
    comp_data: np.ndarray, grid_shape: tuple[int, int, int] | None
) -> np.ndarray:
    """Reshape a component data array of shape (n_points,) to 4D."""
    if grid_shape is None:
        return comp_data[np.newaxis, :, np.newaxis, np.newaxis]

    nx, ny, nz = grid_shape
    reshaped = comp_data.reshape(nx, ny, nz)
    return reshaped[np.newaxis, ...]


def _make_scalar_field(gwexpy, arr4d: np.ndarray, unit, axes):
    """Build a gwexpy ScalarField from a 4D array and prepared axes.

    Shared by the scalar/vector/tensor converters so the ScalarField
    construction (axis wiring, axis0_domain="time") lives in one place.
    ``axes`` is the tuple returned by :func:`_prepare_axes`.
    """
    axis0, axis1, axis2, axis3, axis_names, space_domain = axes
    return gwexpy.fields.ScalarField(
        arr4d,
        unit=unit,
        axis0=axis0,
        axis1=axis1,
        axis2=axis2,
        axis3=axis3,
        axis_names=axis_names,
        axis0_domain="time",
        space_domain=space_domain,
    )


def to_gwexpy_scalar_field(
    data: np.ndarray,
    *,
    positions_m: np.ndarray | None = None,
    grid_shape: tuple[int, int, int] | None = None,
    unit=None,
    t_s: float = 0.0,
) -> gwexpy.fields.ScalarField:
    """Convert gragra scalar data to a gwexpy ScalarField.

    positions_m is currently validated only. It is not encoded into gwexpy axis
    metadata in flat point mode.
    """
    gwexpy = _import_gwexpy()

    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy ndarray")
    if data.ndim != 1:
        raise ValueError("Scalar data must have shape (M,)")

    n_points = data.shape[0]
    t_s = _validate_common(n_points, t_s, positions_m, grid_shape)

    axes = _prepare_axes(n_points, grid_shape, t_s)
    arr4d = _prepare_4d(data, grid_shape)

    return _make_scalar_field(gwexpy, arr4d, unit, axes)


def to_gwexpy_vector_field(
    data: np.ndarray,
    *,
    positions_m: np.ndarray | None = None,
    grid_shape: tuple[int, int, int] | None = None,
    unit=None,
    t_s: float = 0.0,
) -> gwexpy.fields.VectorField:
    """Convert gragra vector data to a gwexpy VectorField.

    positions_m is currently validated only. It is not encoded into gwexpy axis
    metadata in flat point mode.
    """
    gwexpy = _import_gwexpy()

    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy ndarray")
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("Vector data must have shape (M, 3)")

    n_points = data.shape[0]
    t_s = _validate_common(n_points, t_s, positions_m, grid_shape)

    axes = _prepare_axes(n_points, grid_shape, t_s)

    components = {}
    comps_keys = ["x", "y", "z"]
    for i, key in enumerate(comps_keys):
        comp_data = data[:, i]
        arr4d = _prepare_4d(comp_data, grid_shape)
        components[key] = _make_scalar_field(gwexpy, arr4d, unit, axes)

    return gwexpy.fields.VectorField(components)


def to_gwexpy_tensor_field(
    data: np.ndarray,
    *,
    positions_m: np.ndarray | None = None,
    grid_shape: tuple[int, int, int] | None = None,
    unit=None,
    t_s: float = 0.0,
) -> gwexpy.fields.TensorField:
    """Convert gragra tensor data to a gwexpy TensorField.

    positions_m is currently validated only. It is not encoded into gwexpy axis
    metadata in flat point mode.
    """
    gwexpy = _import_gwexpy()

    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy ndarray")
    if data.ndim != 3 or data.shape[1:] != (3, 3):
        raise ValueError("Tensor data must have shape (M, 3, 3)")

    n_points = data.shape[0]
    t_s = _validate_common(n_points, t_s, positions_m, grid_shape)

    axes = _prepare_axes(n_points, grid_shape, t_s)

    components = {}
    for i in range(3):
        for j in range(3):
            comp_data = data[:, i, j]
            arr4d = _prepare_4d(comp_data, grid_shape)
            components[(i, j)] = _make_scalar_field(gwexpy, arr4d, unit, axes)

    return gwexpy.fields.TensorField(components)
