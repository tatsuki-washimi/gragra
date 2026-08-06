"""Structured regular-grid field helpers."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from gragra._arrays import _check_no_bool
from gragra.array_types import ComplexArray, FloatArray
from gragra.sources.volume import VolumeElementSource


def _validate_axis(
    name: str, coords: np.ndarray, expected_len: int, min_len: int = 1
) -> np.ndarray:
    """Validate coordinate axis features.

    Must be 1D, match expected length, finite, real, and strictly increasing.
    """
    _check_no_bool(coords, name)
    if np.iscomplexobj(coords):
        raise TypeError(f"{name} must be real-valued")
    try:
        arr = np.asarray(coords, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError(f"{name} must be numeric: {e}") from e

    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got ndim {arr.ndim}")
    if len(arr) != expected_len:
        raise ValueError(
            f"Length of {name} must match data dimension: {len(arr)} != {expected_len}"
        )
    if len(arr) < min_len:
        raise ValueError(f"{name} length must be >= {min_len}, got {len(arr)}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite numbers")

    if len(arr) >= 2 and not np.all(np.diff(arr) > 0):
        raise ValueError(f"{name} must be strictly increasing")

    # Defensive copy and make read-only
    arr = arr.copy()
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class DensityPerturbationGrid:
    """Helper for bulk density perturbation fields on regular/rectilinear grids.

    This helper facilitates importing density perturbation data defined on a
    structured regular/rectilinear grid (where coordinate spacings along axes
    can be non-uniform) and converting it into a gragra source representation.

    Parameters
    ----------
    x_m : array-like
        1D coordinates of the grid points along the x-axis in meters.
        Spacings between elements can be non-uniform (rectilinear).
    y_m : array-like
        1D coordinates of the grid points along the y-axis in meters.
        Spacings between elements can be non-uniform (rectilinear).
    z_m : array-like
        1D coordinates of the grid points along the z-axis in meters.
        Spacings between elements can be non-uniform (rectilinear).
    delta_density_kg_m3 : array-like
        3D array of shape (Nx, Ny, Nz) containing density perturbations in kg/m^3.
        Can be real (float64) or complex (complex128 for harmonic amplitude).
    cell_size_m : float, tuple, or array-like, optional
        Uniform cell dimensions (dx, dy, dz) in meters.
        If None, cell sizes are implicitly estimated from local coordinate spacing
        via np.gradient (requires Nx, Ny, Nz >= 2).
    """

    x_m: FloatArray
    y_m: FloatArray
    z_m: FloatArray
    delta_density_kg_m3: FloatArray | ComplexArray
    cell_size_m: float | tuple[float, float, float] | FloatArray | None = None

    def __post_init__(self):
        # Validate data array first
        _check_no_bool(self.delta_density_kg_m3, "delta_density_kg_m3")
        is_complex = np.iscomplexobj(self.delta_density_kg_m3)
        dtype = np.complex128 if is_complex else np.float64
        try:
            data = np.asarray(self.delta_density_kg_m3, dtype=dtype)
        except (ValueError, TypeError) as e:
            raise TypeError(f"delta_density_kg_m3 must be numeric: {e}") from e

        if data.ndim != 3:
            raise ValueError(
                f"delta_density_kg_m3 must be a 3D array, got shape {data.shape}"
            )
        if not np.isfinite(data).all():
            raise ValueError("delta_density_kg_m3 must contain only finite numbers")

        nx, ny, nz = data.shape

        # Validate axes
        x_validated = _validate_axis("x_m", self.x_m, nx)
        y_validated = _validate_axis("y_m", self.y_m, ny)
        z_validated = _validate_axis("z_m", self.z_m, nz)

        # Validate and parse cell_size_m
        if self.cell_size_m is not None:
            _check_no_bool(self.cell_size_m, "cell_size_m")
            try:
                cs = np.asarray(self.cell_size_m, dtype=np.float64)
            except (ValueError, TypeError) as e:
                raise TypeError(f"cell_size_m must be numeric: {e}") from e

            if cs.ndim == 0:
                if cs.item() <= 0.0 or not np.isfinite(cs.item()):
                    raise ValueError("cell_size_m must be positive and finite")
                sizes = np.full(3, cs.item())
            elif cs.ndim == 1 and len(cs) == 3:
                if (cs <= 0.0).any() or not np.isfinite(cs).all():
                    raise ValueError("cell_size_m must be positive and finite")
                sizes = cs.copy()
            else:
                msg = (
                    f"cell_size_m must be a scalar or a 3-element array, "
                    f"got shape {cs.shape}"
                )
                raise ValueError(msg)
            sizes.setflags(write=False)
        else:
            # Check length constraint if implicitly estimated
            for name, val in [
                ("x_m", x_validated),
                ("y_m", y_validated),
                ("z_m", z_validated),
            ]:
                if len(val) < 2:
                    msg = (
                        f"Axis length must be >= 2 to estimate cell size when "
                        f"cell_size_m is None. Got length {len(val)} for {name}."
                    )
                    raise ValueError(msg)
            sizes = None

        # Set frozen fields
        data_copy = data.copy()
        data_copy.setflags(write=False)

        object.__setattr__(self, "x_m", x_validated)
        object.__setattr__(self, "y_m", y_validated)
        object.__setattr__(self, "z_m", z_validated)
        object.__setattr__(self, "delta_density_kg_m3", data_copy)
        object.__setattr__(self, "cell_size_m", sizes)

    def as_volume_element_source(self) -> VolumeElementSource:
        """Convert the regular grid representation to a VolumeElementSource.

        Each grid point represents the center of a volume element.

        Returns
        -------
        VolumeElementSource
            The volume element source.
        """
        # Determine cell widths along each axis
        if self.cell_size_m is not None:
            dx = np.full(len(self.x_m), self.cell_size_m[0])
            dy = np.full(len(self.y_m), self.cell_size_m[1])
            dz = np.full(len(self.z_m), self.cell_size_m[2])
        else:
            dx = np.gradient(self.x_m)
            dy = np.gradient(self.y_m)
            dz = np.gradient(self.z_m)

        # Compute cell volumes (Nx, Ny, Nz)
        volumes_3d = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]

        # Flat coordinate arrays
        x_grid, y_grid, z_grid = np.meshgrid(
            self.x_m, self.y_m, self.z_m, indexing="ij"
        )
        positions = np.stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], axis=-1)

        volumes = volumes_3d.ravel()
        densities = self.delta_density_kg_m3.ravel()

        # Create defensive copies
        positions = positions.copy()
        positions.setflags(write=False)

        volumes = volumes.copy()
        volumes.setflags(write=False)

        densities = densities.copy()
        densities.setflags(write=False)

        return VolumeElementSource(
            positions_m=positions,
            volumes_m3=volumes,
            density_kg_m3=densities,
        )


@dataclass(frozen=True)
class DisplacementGrid:
    """Helper representing a bulk displacement field on a regular or rectilinear grid.

    This helper calculates density perturbations from displacement vectors
    on a structured regular/rectilinear grid (where coordinate spacings along
    axes can be non-uniform) using linearized mass conservation.

    Parameters
    ----------
    x_m : array-like
        1D coordinates of the grid points along the x-axis in meters.
        Spacings between elements can be non-uniform (rectilinear).
    y_m : array-like
        1D coordinates of the grid points along the y-axis in meters.
        Spacings between elements can be non-uniform (rectilinear).
    z_m : array-like
        1D coordinates of the grid points along the z-axis in meters.
        Spacings between elements can be non-uniform (rectilinear).
    displacement_m : array-like
        4D array of shape (Nx, Ny, Nz, 3) containing displacement vectors.
        Can be real (float64) or complex (complex128 for harmonic amplitude).
    """

    x_m: FloatArray
    y_m: FloatArray
    z_m: FloatArray
    displacement_m: FloatArray | ComplexArray

    def __post_init__(self):
        # Validate displacement array first
        _check_no_bool(self.displacement_m, "displacement_m")
        is_complex = np.iscomplexobj(self.displacement_m)
        dtype = np.complex128 if is_complex else np.float64
        try:
            data = np.asarray(self.displacement_m, dtype=dtype)
        except (ValueError, TypeError) as e:
            raise TypeError(f"displacement_m must be numeric: {e}") from e

        if data.ndim != 4:
            raise ValueError(
                f"displacement_m must be a 4D array, got shape {data.shape}"
            )
        if data.shape[3] != 3:
            raise ValueError(
                f"displacement_m last dimension must be 3, got shape {data.shape}"
            )
        if not np.isfinite(data).all():
            raise ValueError("displacement_m must contain only finite numbers")

        nx, ny, nz, _ = data.shape

        # Validate axes (min_len=3 is required for np.gradient edge_order=2)
        x_validated = _validate_axis("x_m", self.x_m, nx, min_len=3)
        y_validated = _validate_axis("y_m", self.y_m, ny, min_len=3)
        z_validated = _validate_axis("z_m", self.z_m, nz, min_len=3)

        # Set frozen fields
        data_copy = data.copy()
        data_copy.setflags(write=False)

        object.__setattr__(self, "x_m", x_validated)
        object.__setattr__(self, "y_m", y_validated)
        object.__setattr__(self, "z_m", z_validated)
        object.__setattr__(self, "displacement_m", data_copy)

    def as_density_perturbation_grid(
        self,
        rho0_kg_m3: float | npt.NDArray[np.float64],
    ) -> DensityPerturbationGrid:
        """Compute the density perturbation grid from the displacement field.

        Calculated via delta_density = -rho0 * div(u) - u . grad(rho0),
        using second-order central finite differences.

        Parameters
        ----------
        rho0_kg_m3 : float or (Nx, Ny, Nz) ndarray
            float: uniform background density.
            ndarray (Nx, Ny, Nz): spatially varying background density.
                Full delta_rho = -rho0*div(u) - u.grad(rho0) is computed
                using np.gradient (edge_order=2) for both div(u) and grad(rho0).

        Returns
        -------
        DensityPerturbationGrid
            The computed density perturbation grid helper.

        Notes
        -----
        Limitation: When ``rho0_kg_m3`` is a single uniform scalar, this evaluates
        only the bulk term ``delta_rho = -rho0 * div(u)``. Spatially varying background
        densities on unstructured meshes are handled differently.
        """
        # Validate rho0_kg_m3
        _check_no_bool(rho0_kg_m3, "rho0_kg_m3")
        raw_arr = np.asarray(rho0_kg_m3)
        if np.iscomplexobj(raw_arr):
            raise TypeError("rho0_kg_m3 must be real-valued")
        if raw_arr.dtype.kind not in "iuf":
            raise TypeError(f"rho0_kg_m3 must be numeric, got {raw_arr.dtype}")

        # Extract component fields (Nx, Ny, Nz)
        ux = self.displacement_m[..., 0]
        uy = self.displacement_m[..., 1]
        uz = self.displacement_m[..., 2]

        # Compute divergence components using np.gradient (edge_order=2)
        dux_dx = np.gradient(ux, self.x_m, axis=0, edge_order=2)
        duy_dy = np.gradient(uy, self.y_m, axis=1, edge_order=2)
        duz_dz = np.gradient(uz, self.z_m, axis=2, edge_order=2)

        div_u = dux_dx + duy_dy + duz_dz

        if raw_arr.ndim == 0:
            rho0_val = float(raw_arr)
            if not np.isfinite(rho0_val):
                raise ValueError("rho0_kg_m3 must be finite")
            if rho0_val < 0.0:
                raise ValueError("rho0_kg_m3 must be non-negative")

            # delta_density = -rho0 * div(u)
            delta_density = -rho0_val * div_u

        elif raw_arr.ndim == 3:
            rho0_arr = raw_arr.astype(np.float64)

            # Shape verification
            nx, ny, nz, _ = self.displacement_m.shape
            if rho0_arr.shape != (nx, ny, nz):
                raise ValueError(
                    f"rho0_kg_m3 array must match grid dimensions "
                    f"({nx},{ny},{nz}), got {rho0_arr.shape}"
                )

            if not np.isfinite(rho0_arr).all():
                raise ValueError("rho0_kg_m3 must be finite")
            if (rho0_arr < 0.0).any():
                raise ValueError("rho0_kg_m3 must be non-negative")

            # Compute background density gradients
            drho_dx = np.gradient(rho0_arr, self.x_m, axis=0, edge_order=2)
            drho_dy = np.gradient(rho0_arr, self.y_m, axis=1, edge_order=2)
            drho_dz = np.gradient(rho0_arr, self.z_m, axis=2, edge_order=2)

            # Full delta_rho = -rho0 * div(u) - u . grad(rho0)
            delta_density = -rho0_arr * div_u - (
                ux * drho_dx + uy * drho_dy + uz * drho_dz
            )

        else:
            raise ValueError(
                f"rho0_kg_m3 array must be 3-D (Nx,Ny,Nz), got ndim={raw_arr.ndim}"
            )

        return DensityPerturbationGrid(
            x_m=self.x_m,
            y_m=self.y_m,
            z_m=self.z_m,
            delta_density_kg_m3=delta_density,
            cell_size_m=None,
        )
