"""HDF5 native field importer for gragra."""

import operator
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from gragra._arrays import _check_no_bool, as_xyz_table
from gragra.array_types import FloatArray
from gragra.fields.regular_grid import DisplacementGrid
from gragra.fields.unstructured import UnstructuredDisplacementField


def _snapshot_index(p, num_snapshots: int) -> int:
    """Validate and coerce a single requested snapshot index.

    Uses ``operator.index`` (not ``int()``) so that non-integer numeric
    values (e.g. ``1.9``) are rejected instead of being silently truncated,
    which would otherwise associate a returned weight/slice with the wrong
    requested snapshot. ``bool`` is rejected explicitly since
    ``operator.index(True)`` would otherwise silently accept it as ``1``.
    """
    if isinstance(p, (bool, np.bool_)):
        raise ValueError(f"Snapshot index must not be bool, got {p!r}")
    try:
        p_idx = operator.index(p)
    except TypeError as e:
        raise ValueError(
            f"Snapshot index {p!r} must be an integer, got {type(p).__name__}"
        ) from e
    if p_idx < 0 or p_idx >= num_snapshots:
        raise ValueError(
            f"Snapshot index {p_idx} out of range for batch of size {num_snapshots}"
        )
    return p_idx


@dataclass(frozen=True)
class DiscreteBackgroundDensity:
    """Discrete background density model for unstructured fields.

    Parameters
    ----------
    centroids_m : FloatArray
        Centroids of the mesh cells, shape (Nc, 3).
    rho0_kg_m3 : FloatArray
        Background density values at the centroids, shape (Nc,).
    """

    centroids_m: FloatArray
    rho0_kg_m3: FloatArray

    def __post_init__(self):
        # Validate centroids_m
        _check_no_bool(self.centroids_m, "centroids_m")
        centroids_val = as_xyz_table("centroids_m", self.centroids_m)
        nc = len(centroids_val)

        # Validate rho0_kg_m3
        _check_no_bool(self.rho0_kg_m3, "rho0_kg_m3")
        if np.iscomplexobj(self.rho0_kg_m3):
            raise TypeError("rho0_kg_m3 must be real-valued")
        try:
            rho = np.asarray(self.rho0_kg_m3, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise TypeError(f"rho0_kg_m3 must be numeric: {e}") from e

        if rho.ndim != 1 or len(rho) != nc:
            raise ValueError(
                "rho0_kg_m3 must be a 1D array matching centroids: "
                f"{rho.shape} != ({nc},)"
            )
        if not np.isfinite(rho).all():
            raise ValueError("rho0_kg_m3 must be finite")
        if (rho < 0.0).any():
            raise ValueError("rho0_kg_m3 must be non-negative")

        centroids_val = centroids_val.copy()
        centroids_val.setflags(write=False)
        rho_copy = rho.copy()
        rho_copy.setflags(write=False)

        object.__setattr__(self, "centroids_m", centroids_val)
        object.__setattr__(self, "rho0_kg_m3", rho_copy)

    def __call__(self, points_m: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        _check_no_bool(points_m, "points_m")
        points = np.asarray(points_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_m must be a 2D coordinate array of shape (N, 3)")
        if points.shape[0] != self.centroids_m.shape[0]:
            raise ValueError(
                f"points_m count does not match background density elements: "
                f"{points.shape[0]} != {self.centroids_m.shape[0]}"
            )
        # Assumes the same centroid formula (p0+p1+p2+p3)/4 as the
        # unstructured.py helper.
        if not np.allclose(points, self.centroids_m, rtol=1e-9, atol=0.0):
            raise ValueError(
                "points_m does not match stored centroids within tolerance"
            )
        return self.rho0_kg_m3


@dataclass(frozen=True)
class DifferentiableDiscreteBackgroundDensity:
    """Differentiable discrete background density model for unstructured fields.

    Parameters
    ----------
    centroids_m : FloatArray
        Centroids of the mesh cells, shape (Nc, 3).
    rho0_kg_m3 : FloatArray
        Background density values at the centroids, shape (Nc,).
    grad_rho0_kg_m4 : FloatArray
        Spatial gradient of background density at the centroids, shape (Nc, 3).
    """

    centroids_m: FloatArray
    rho0_kg_m3: FloatArray
    grad_rho0_kg_m4: FloatArray

    def __post_init__(self):
        # Validate centroids_m
        _check_no_bool(self.centroids_m, "centroids_m")
        centroids_val = as_xyz_table("centroids_m", self.centroids_m)
        nc = len(centroids_val)

        # Validate rho0_kg_m3
        _check_no_bool(self.rho0_kg_m3, "rho0_kg_m3")
        if np.iscomplexobj(self.rho0_kg_m3):
            raise TypeError("rho0_kg_m3 must be real-valued")
        try:
            rho = np.asarray(self.rho0_kg_m3, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise TypeError(f"rho0_kg_m3 must be numeric: {e}") from e

        if rho.ndim != 1 or len(rho) != nc:
            raise ValueError(
                "rho0_kg_m3 must be a 1D array matching centroids: "
                f"{rho.shape} != ({nc},)"
            )
        if not np.isfinite(rho).all():
            raise ValueError("rho0_kg_m3 must be finite")
        if (rho < 0.0).any():
            raise ValueError("rho0_kg_m3 must be non-negative")

        # Validate grad_rho0_kg_m4
        _check_no_bool(self.grad_rho0_kg_m4, "grad_rho0_kg_m4")
        if np.iscomplexobj(self.grad_rho0_kg_m4):
            raise TypeError("grad_rho0_kg_m4 must be real-valued")
        try:
            grad = np.asarray(self.grad_rho0_kg_m4, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise TypeError(f"grad_rho0_kg_m4 must be numeric: {e}") from e

        if grad.shape != (nc, 3):
            raise ValueError(
                f"grad_rho0_kg_m4 must have shape ({nc}, 3), got {grad.shape}"
            )
        if not np.isfinite(grad).all():
            raise ValueError("grad_rho0_kg_m4 must be finite")

        centroids_val = centroids_val.copy()
        centroids_val.setflags(write=False)
        rho_copy = rho.copy()
        rho_copy.setflags(write=False)
        grad_copy = grad.copy()
        grad_copy.setflags(write=False)

        object.__setattr__(self, "centroids_m", centroids_val)
        object.__setattr__(self, "rho0_kg_m3", rho_copy)
        object.__setattr__(self, "grad_rho0_kg_m4", grad_copy)

    def __call__(self, points_m: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        _check_no_bool(points_m, "points_m")
        points = np.asarray(points_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_m must be a 2D coordinate array of shape (N, 3)")
        if points.shape[0] != self.centroids_m.shape[0]:
            raise ValueError(
                f"points_m count does not match background density elements: "
                f"{points.shape[0]} != {self.centroids_m.shape[0]}"
            )
        # Assumes the same centroid formula (p0+p1+p2+p3)/4 as the
        # unstructured.py helper.
        if not np.allclose(points, self.centroids_m, rtol=1e-9, atol=0.0):
            raise ValueError(
                "points_m does not match stored centroids within tolerance"
            )
        return self.rho0_kg_m3

    def gradient(self, points_m: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        _check_no_bool(points_m, "points_m")
        points = np.asarray(points_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_m must be a 2D coordinate array of shape (N, 3)")
        if points.shape[0] != self.centroids_m.shape[0]:
            raise ValueError(
                f"points_m count does not match background density elements: "
                f"{points.shape[0]} != {self.centroids_m.shape[0]}"
            )
        # Assumes the same centroid formula (p0+p1+p2+p3)/4 as the
        # unstructured.py helper.
        if not np.allclose(points, self.centroids_m, rtol=1e-9, atol=0.0):
            raise ValueError(
                "points_m does not match stored centroids within tolerance"
            )
        return self.grad_rho0_kg_m4


def _validate_length_scale(length_scale_m: float) -> None:
    if isinstance(length_scale_m, bool):
        raise TypeError("length_scale_m must not be bool")
    if not isinstance(length_scale_m, (int, float, np.integer, np.floating)):
        raise TypeError("length_scale_m must be numeric")
    if not np.isfinite(length_scale_m):
        raise ValueError("length_scale_m must be finite")
    if length_scale_m <= 0.0:
        raise ValueError("length_scale_m must be positive")


def from_hdf5_unstructured_field(
    path: str, *, length_scale_m: float = 1.0
) -> UnstructuredDisplacementField:
    """Import an UnstructuredDisplacementField from an HDF5 file.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    length_scale_m : float, default 1.0
        Length scale conversion factor to convert coordinates to meters.

    Returns
    -------
    UnstructuredDisplacementField
        The imported unstructured field helper.
    """
    _validate_length_scale(length_scale_m)

    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for HDF5 field features. "
            'Please install it using: pip install "gragra[field-io]"'
        ) from e

    with h5py.File(path, "r") as f:
        # Check mesh_type attribute
        mesh_type = f.attrs.get("mesh_type")
        if mesh_type is None:
            raise ValueError("Missing 'mesh_type' attribute in HDF5 file")
        # Ensure it is decoded to string if bytes
        if isinstance(mesh_type, bytes):
            mesh_type = mesh_type.decode("utf-8")
        if mesh_type != "unstructured":
            raise ValueError(f"Expected mesh_type 'unstructured', got '{mesh_type}'")

        # Check required datasets
        required = ["vertices_m", "connectivity", "displacement_m"]
        for key in required:
            if key not in f:
                raise ValueError(f"Missing required dataset '{key}' in HDF5 file")

        vertices_m = f["vertices_m"][()]
        connectivity = f["connectivity"][()]
        displacement_m = f["displacement_m"][()]

    scaled_vertices = np.asarray(vertices_m) * float(length_scale_m)
    scaled_displacement = np.asarray(displacement_m) * float(length_scale_m)

    return UnstructuredDisplacementField(
        vertices_m=scaled_vertices,
        tetra=np.asarray(connectivity),
        displacement_m=scaled_displacement,
    )


def from_hdf5_displacement_grid(
    path: str, *, length_scale_m: float = 1.0
) -> DisplacementGrid:
    """Import a DisplacementGrid from an HDF5 file.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    length_scale_m : float, default 1.0
        Length scale conversion factor to convert coordinates to meters.

    Returns
    -------
    DisplacementGrid
        The imported displacement grid helper.
    """
    _validate_length_scale(length_scale_m)

    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for HDF5 field features. "
            'Please install it using: pip install "gragra[field-io]"'
        ) from e

    with h5py.File(path, "r") as f:
        # Check mesh_type attribute
        mesh_type = f.attrs.get("mesh_type")
        if mesh_type is None:
            raise ValueError("Missing 'mesh_type' attribute in HDF5 file")
        if isinstance(mesh_type, bytes):
            mesh_type = mesh_type.decode("utf-8")
        if mesh_type != "structured":
            raise ValueError(f"Expected mesh_type 'structured', got '{mesh_type}'")

        # Check required datasets
        required = ["x_m", "y_m", "z_m", "displacement_m"]
        for key in required:
            if key not in f:
                raise ValueError(f"Missing required dataset '{key}' in HDF5 file")

        x_m = f["x_m"][()]
        y_m = f["y_m"][()]
        z_m = f["z_m"][()]
        displacement_m = f["displacement_m"][()]

    scaled_x = np.asarray(x_m) * float(length_scale_m)
    scaled_y = np.asarray(y_m) * float(length_scale_m)
    scaled_z = np.asarray(z_m) * float(length_scale_m)
    scaled_displacement = np.asarray(displacement_m) * float(length_scale_m)

    return DisplacementGrid(
        x_m=scaled_x,
        y_m=scaled_y,
        z_m=scaled_z,
        displacement_m=scaled_displacement,
    )


def from_hdf5_field(
    path: str, *, length_scale_m: float = 1.0
) -> UnstructuredDisplacementField | DisplacementGrid:
    """Import a displacement field (unstructured or structured) from HDF5.

    The mesh_type attribute in the HDF5 file determines the concrete return type.

    Note: This only reads the displacement field geometry/data. Background density
    is read separately via read_hdf5_background_density().

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    length_scale_m : float, default 1.0
        Length scale conversion factor to convert coordinates to meters.

    Returns
    -------
    UnstructuredDisplacementField or DisplacementGrid
        The imported field helper.
    """
    _validate_length_scale(length_scale_m)

    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for HDF5 field features. "
            'Please install it using: pip install "gragra[field-io]"'
        ) from e

    with h5py.File(path, "r") as f:
        mesh_type = f.attrs.get("mesh_type")
        if mesh_type is None:
            raise ValueError("Missing 'mesh_type' attribute in HDF5 file")
        if isinstance(mesh_type, bytes):
            mesh_type = mesh_type.decode("utf-8")

        if mesh_type == "unstructured":
            return from_hdf5_unstructured_field(path, length_scale_m=length_scale_m)
        if mesh_type == "structured":
            return from_hdf5_displacement_grid(path, length_scale_m=length_scale_m)
        raise ValueError(f"Invalid mesh_type attribute in HDF5 file: '{mesh_type}'")


def read_hdf5_background_density(
    path: str, *, length_scale_m: float = 1.0
) -> (
    None
    | float
    | DiscreteBackgroundDensity
    | DifferentiableDiscreteBackgroundDensity
    | np.ndarray
):
    """Read background density data from an HDF5 file.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    length_scale_m : float, default 1.0
        Length scale conversion factor (must match the scale factor
        used to import the field).

    Returns
    -------
    None, float, DiscreteBackgroundDensity,
    DifferentiableDiscreteBackgroundDensity, or np.ndarray
        For unstructured meshes:
          - None: if rho0_kg_m3 is missing.
          - float: if rho0_kg_m3 is a scalar.
          - DiscreteBackgroundDensity: if rho0_kg_m3 is cell-wise.
          - DifferentiableDiscreteBackgroundDensity: if grad_rho0_kg_m4 is also present.
        For structured meshes:
          - None: if rho0_kg_m3 is missing.
          - float: if rho0_kg_m3 is a scalar.
          - np.ndarray: raw (Nx, Ny, Nz) array representing spatially varying density.
    """
    _validate_length_scale(length_scale_m)

    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for HDF5 field features. "
            'Please install it using: pip install "gragra[field-io]"'
        ) from e

    with h5py.File(path, "r") as f:
        # Check mesh_type
        mesh_type = f.attrs.get("mesh_type")
        if mesh_type is None:
            raise ValueError("Missing 'mesh_type' attribute in HDF5 file")
        if isinstance(mesh_type, bytes):
            mesh_type = mesh_type.decode("utf-8")
        if mesh_type not in ("unstructured", "structured"):
            raise ValueError(f"Invalid mesh_type attribute in HDF5 file: '{mesh_type}'")

        if "rho0_kg_m3" not in f:
            return None

        rho0_ds = f["rho0_kg_m3"]
        if rho0_ds.ndim == 0:
            rho0_kg_m3 = rho0_ds[()]
            has_grad = "grad_rho0_kg_m4" in f
            grad_rho0_kg_m4 = f["grad_rho0_kg_m4"][()] if has_grad else None
            vertices_m = None
            connectivity = None
        else:
            if mesh_type == "unstructured":
                for key in ["vertices_m", "connectivity"]:
                    if key not in f:
                        raise ValueError(
                            f"Missing required dataset '{key}' to compute centroids "
                            "for unstructured background density"
                        )
                vertices_m = f["vertices_m"][()]
                connectivity = f["connectivity"][()]
            else:
                vertices_m = None
                connectivity = None
            rho0_kg_m3 = rho0_ds[()]
            has_grad = "grad_rho0_kg_m4" in f
            grad_rho0_kg_m4 = f["grad_rho0_kg_m4"][()] if has_grad else None

        if mesh_type == "structured" and rho0_ds.ndim > 0:
            # For structured, we can use x_m, y_m, z_m or displacement_m
            # to check dimensions
            grid_shape = None
            if "displacement_m" in f and len(f["displacement_m"].shape) == 4:
                grid_shape = f["displacement_m"].shape[:3]
            elif all(k in f for k in ["x_m", "y_m", "z_m"]):
                grid_shape = (
                    len(f["x_m"]),
                    len(f["y_m"]),
                    len(f["z_m"]),
                )
            elif "displacement_m" in f and len(f["displacement_m"].shape) == 5:
                grid_shape = f["displacement_m"].shape[1:4]
            else:
                raise ValueError(
                    "Missing grid dimension indicators to validate "
                    "structured background density"
                )

    if mesh_type == "structured" and has_grad:
        raise ValueError(
            "grad_rho0_kg_m4 is not supported for structured meshes; "
            "the gradient is computed internally via np.gradient"
        )

    # Do scaling and validation outside the h5py context block
    rho0_arr = np.asarray(rho0_kg_m3)
    if rho0_arr.ndim == 0:
        if has_grad:
            raise ValueError("grad_rho0_kg_m4 requires non-scalar rho0_kg_m3")
        rho0_val = float(rho0_arr)
        if not np.isfinite(rho0_val):
            raise ValueError("rho0_kg_m3 must be finite")
        if rho0_val < 0.0:
            raise ValueError("rho0_kg_m3 must be non-negative")
        return rho0_val

    if mesh_type == "unstructured":
        scaled_vertices = np.asarray(vertices_m) * float(length_scale_m)
        conn = np.asarray(connectivity)

        # Compute centroids
        p0 = scaled_vertices[conn[:, 0]]
        p1 = scaled_vertices[conn[:, 1]]
        p2 = scaled_vertices[conn[:, 2]]
        p3 = scaled_vertices[conn[:, 3]]
        centroids_m = (p0 + p1 + p2 + p3) / 4.0

        if has_grad:
            grad = np.asarray(grad_rho0_kg_m4)
            # Differentiable discrete density checks are also run in __post_init__
            return DifferentiableDiscreteBackgroundDensity(
                centroids_m=centroids_m,
                rho0_kg_m3=rho0_arr,
                grad_rho0_kg_m4=grad,
            )
        return DiscreteBackgroundDensity(
            centroids_m=centroids_m,
            rho0_kg_m3=rho0_arr,
        )

    # Structured grid path
    nx, ny, nz = grid_shape
    if rho0_arr.shape != (nx, ny, nz):
        raise ValueError(
            f"rho0_kg_m3 array must match grid dimensions ({nx},{ny},{nz}), "
            f"got {rho0_arr.shape}"
        )
    if np.iscomplexobj(rho0_arr):
        raise TypeError("rho0_kg_m3 must be real-valued")
    if not np.isfinite(rho0_arr).all():
        raise ValueError("rho0_kg_m3 must be finite")
    if (rho0_arr < 0.0).any():
        raise ValueError("rho0_kg_m3 must be non-negative")

    rho_copy = rho0_arr.copy().astype(np.float64)
    rho_copy.setflags(write=False)
    return rho_copy


def iter_hdf5_field_snapshots(
    path: str,
    *,
    length_scale_m: float = 1.0,
    snapshots: Sequence[int] | None = None,
) -> Iterator[DisplacementGrid | UnstructuredDisplacementField]:
    """Iterate over batched HDF5 field snapshots.

    Opens the HDF5 file, reads the mesh metadata once, and yields each snapshot
    as a DisplacementGrid or UnstructuredDisplacementField helper.

    Memory Strategy:
    ----------------
    To avoid materializing the entire (P, ...) displacement dataset in memory,
    this generator reads the displacement slice-by-slice from the HDF5 dataset.

    Disclaimer on Complex Outputs (Linear Harmonic Amplitudes):
    -----------------------------------------------------------
    If the stored displacement dataset contains complex values, the yielded
    field helpers will contain complex-valued displacement_m. These represent
    linear complex amplitudes rather than real-time fields.
    """
    _validate_length_scale(length_scale_m)

    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for HDF5 field features. "
            'Please install it using: pip install "gragra[field-io]"'
        ) from e

    with h5py.File(path, "r") as f:
        # Check mesh_type attribute
        mesh_type = f.attrs.get("mesh_type")
        if mesh_type is None:
            raise ValueError("Missing 'mesh_type' attribute in HDF5 file")
        if isinstance(mesh_type, bytes):
            mesh_type = mesh_type.decode("utf-8")
        if mesh_type not in ("unstructured", "structured"):
            raise ValueError(f"Invalid mesh_type attribute in HDF5 file: '{mesh_type}'")

        if "displacement_m" not in f:
            raise ValueError("Missing required dataset 'displacement_m' in HDF5 file")

        disp_ds = f["displacement_m"]
        disp_ndim = disp_ds.ndim

        # Batched check
        if mesh_type == "structured":
            if disp_ndim == 4:
                raise ValueError(
                    "Single snapshot structured grid detected. "
                    "Use from_hdf5_field instead."
                )
            if disp_ndim != 5:
                raise ValueError(
                    f"Expected 5-D displacement_m for batched structured grid, "
                    f"got ndim={disp_ndim}"
                )
        else:
            if disp_ndim == 2:
                raise ValueError(
                    "Single snapshot unstructured field detected. "
                    "Use from_hdf5_field instead."
                )
            if disp_ndim != 3:
                raise ValueError(
                    f"Expected 3-D displacement_m for batched unstructured field, "
                    f"got ndim={disp_ndim}"
                )

        num_snapshots = disp_ds.shape[0]
        if num_snapshots == 0:
            raise ValueError(
                "Batched displacement_m has a leading snapshot axis of size 0; "
                "the batch contract requires at least one snapshot (P >= 1)."
            )

        # Determine snapshots to yield
        if snapshots is not None:
            selected_p = [_snapshot_index(p, num_snapshots) for p in snapshots]
        else:
            selected_p = list(range(num_snapshots))

        # Read coordinates/mesh once
        if mesh_type == "structured":
            for key in ["x_m", "y_m", "z_m"]:
                if key not in f:
                    raise ValueError(f"Missing structured grid coordinate axis '{key}'")
            x_m = f["x_m"][()] * float(length_scale_m)
            y_m = f["y_m"][()] * float(length_scale_m)
            z_m = f["z_m"][()] * float(length_scale_m)
        else:
            for key in ["vertices_m", "connectivity"]:
                if key not in f:
                    raise ValueError(f"Missing unstructured mesh dataset '{key}'")
            vertices_m = f["vertices_m"][()] * float(length_scale_m)
            connectivity = f["connectivity"][()]

        # Yield snapshots slice-by-slice inside h5py File context
        for p in selected_p:
            disp_p = disp_ds[p][()] * float(length_scale_m)
            if mesh_type == "structured":
                yield DisplacementGrid(
                    x_m=x_m,
                    y_m=y_m,
                    z_m=z_m,
                    displacement_m=disp_p,
                )
            else:
                yield UnstructuredDisplacementField(
                    vertices_m=vertices_m,
                    tetra=connectivity,
                    displacement_m=disp_p,
                )


def from_hdf5_batched_weights(
    path: str,
    rho0_kg_m3=None,
    *,
    length_scale_m: float = 1.0,
    include_rho0_gradient_term: bool = False,
    snapshots: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute batched weights directly from a batched HDF5 file.

    Parameters
    ----------
    path : str
        Path to the batched HDF5 file.
    rho0_kg_m3 : float, ndarray, BackgroundDensityModel, or None
        Precedence:
        1. If provided (non-None), this density is used and file datasets are ignored.
        2. If None, falls back to the 'rho0_kg_m3' dataset inside the HDF5 file.
        3. If None and no dataset exists in the file, raises ValueError.
    length_scale_m : float, default 1.0
        Length scale conversion factor to convert coordinates to meters.
    include_rho0_gradient_term : bool, default False
        Whether to include background density gradient term.
        Only valid for unstructured meshes.
    snapshots : Sequence[int], optional
        Sequence of snapshot indices to import. Defaults to all snapshots.

    Returns
    -------
    positions_m : (N, 3) ndarray
        Read-only source positions.
    weights : (N, P) ndarray
        Read-only computed batched weights.
    """
    _validate_length_scale(length_scale_m)

    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for HDF5 field features. "
            'Please install it using: pip install "gragra[field-io]"'
        ) from e

    with h5py.File(path, "r") as f:
        mesh_type = f.attrs.get("mesh_type")
        if mesh_type is None:
            raise ValueError("Missing 'mesh_type' attribute in HDF5 file")
        if isinstance(mesh_type, bytes):
            mesh_type = mesh_type.decode("utf-8")
        if mesh_type not in ("unstructured", "structured"):
            raise ValueError(f"Invalid mesh_type attribute in HDF5 file: '{mesh_type}'")

        if "displacement_m" not in f:
            raise ValueError("Missing required dataset 'displacement_m' in HDF5 file")

        disp_ds = f["displacement_m"]
        num_snapshots_file = disp_ds.shape[0]

        # snapshots validation
        if snapshots is not None:
            selected_indices = [
                _snapshot_index(p, num_snapshots_file) for p in snapshots
            ]
        else:
            selected_indices = list(range(num_snapshots_file))

        # structured + gradient validation
        if mesh_type == "structured" and include_rho0_gradient_term:
            raise ValueError(
                "include_rho0_gradient_term=True is not supported for structured grids."
            )

        # Precedence check for rho0
        if rho0_kg_m3 is None:
            # We will read this from file outside this block
            pass
        else:
            # Validation on the provided rho0_kg_m3
            if (
                mesh_type == "structured"
                and isinstance(rho0_kg_m3, np.ndarray)
                and rho0_kg_m3.ndim != 0
            ):
                if rho0_kg_m3.ndim == 4:
                    raise ValueError(
                        "Batch-dependent background density is not supported."
                    )
                if "x_m" in f and "y_m" in f and "z_m" in f:
                    expected_shape = (len(f["x_m"]), len(f["y_m"]), len(f["z_m"]))
                else:
                    expected_shape = disp_ds.shape[1:4]
                if rho0_kg_m3.shape != expected_shape:
                    raise ValueError(
                        f"rho0_kg_m3 array shape {rho0_kg_m3.shape} "
                        f"does not match grid shape {expected_shape}"
                    )

    if rho0_kg_m3 is None:
        rho0 = read_hdf5_background_density(path, length_scale_m=length_scale_m)
        if rho0 is None:
            raise ValueError(
                "rho0_kg_m3 is None, and no 'rho0_kg_m3' dataset was "
                "found in the HDF5 file."
            )
    else:
        rho0 = rho0_kg_m3

    # iter_hdf5_field_snapshots with streaming
    all_weights = []
    positions = None

    for idx, field_p in enumerate(
        iter_hdf5_field_snapshots(
            path, length_scale_m=length_scale_m, snapshots=selected_indices
        )
    ):
        p = selected_indices[idx]
        try:
            if mesh_type == "structured":
                perturbed_p = field_p.as_density_perturbation_grid(rho0)
                vol_p = perturbed_p.as_volume_element_source()
                weighted_p = vol_p.as_weighted_source()
            else:
                perturbed_p = field_p.as_density_perturbation_source(
                    rho0, include_rho0_gradient_term=include_rho0_gradient_term
                )
                weighted_p = perturbed_p.as_weighted_source()
        except ValueError as e:
            raise ValueError(f"snapshot p={p}: {e}") from e
        except TypeError as e:
            raise TypeError(f"snapshot p={p}: {e}") from e

        if idx == 0:
            positions = weighted_p.positions_m

        all_weights.append(weighted_p.weights_kg)

    if not all_weights:
        raise ValueError("No snapshots processed")

    weights = np.column_stack(all_weights)
    positions = positions.copy()
    positions.setflags(write=False)
    weights.setflags(write=False)
    return positions, weights
