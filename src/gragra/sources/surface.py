"""Surface mass sheet source representation."""

from dataclasses import dataclass

import numpy as np

from gragra._arrays import _check_no_bool, as_xyz_table
from gragra.array_types import ComplexArray, FloatArray
from gragra.sources.weighted_points import WeightedPointSource


@dataclass(frozen=True)
class SurfaceMassSheetSource:
    """Surface mass sheet source representation.

    Represents displacement-induced surface mass perturbations.
    Note that weights_kg returned by as_weighted_source() are mass elements only
    (e.g., δm = δρ * u_n * dS) in kilograms. Transfer factors or coherent phase
    terms must be handled by a separate abstraction.

    Parameters
    ----------
    positions_m : array-like
        Coordinates of shape (N, 3) in meters.
    normals : array-like
        Normal vectors of shape (N, 3) (will be unit normalized internally).
    areas_m2 : array-like
        Area elements of shape (N,) in square meters. Must be positive and finite.
    normal_displacement_m : float, complex, or array-like
        Normal displacement in meters. Can be a scalar or a vector of shape (N,).
    density_kg_m3 : float or complex
        Mass density of the medium in kg/m^3 (must be a scalar).
    """

    positions_m: FloatArray
    normals: FloatArray
    areas_m2: FloatArray
    normal_displacement_m: float | complex | FloatArray | ComplexArray
    density_kg_m3: float | complex

    def __post_init__(self):
        # Validate positions_m
        positions = as_xyz_table("positions_m", self.positions_m)
        n_elements = len(positions)

        # Validate normals
        _check_no_bool(self.normals, "normals")
        try:
            norm_arr = np.asarray(self.normals, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"normals must be numeric convertible to float64: {e}"
            ) from e

        if norm_arr.shape != positions.shape:
            raise ValueError(
                "normals shape must match positions_m: "
                f"{norm_arr.shape} != {positions.shape}"
            )
        if not np.isfinite(norm_arr).all():
            raise ValueError("normals must contain only finite numbers")

        lengths = np.linalg.norm(norm_arr, axis=-1)
        if (lengths == 0.0).any():
            raise ValueError("normals must not contain zero vectors")

        normalized_normals = norm_arr / lengths[:, np.newaxis]
        normalized_normals.setflags(write=False)

        # Validate areas_m2
        _check_no_bool(self.areas_m2, "areas_m2")
        try:
            areas = np.asarray(self.areas_m2, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise TypeError(f"areas_m2 must be numeric: {e}") from e

        if areas.ndim != 1:
            raise ValueError(f"areas_m2 must be a 1D array, got shape {areas.shape}")
        if len(areas) != n_elements:
            raise ValueError(
                "Length of areas_m2 must match positions_m: "
                f"{len(areas)} != {n_elements}"
            )
        if not np.isfinite(areas).all():
            raise ValueError("areas_m2 must contain only finite numbers")
        if (areas <= 0.0).any():
            raise ValueError("areas_m2 must contain only positive values")

        # Validate normal_displacement_m
        _check_no_bool(self.normal_displacement_m, "normal_displacement_m")
        raw_disp = np.asarray(self.normal_displacement_m)
        if raw_disp.ndim == 0:
            if not np.isfinite(raw_disp):
                raise ValueError("normal_displacement_m must be finite")
            if np.iscomplexobj(raw_disp):
                disp_val = complex(raw_disp.item())
            else:
                disp_val = float(raw_disp.item())
        elif raw_disp.ndim == 1:
            if len(raw_disp) != n_elements:
                raise ValueError(
                    "Length of normal_displacement_m must match positions_m: "
                    f"{len(raw_disp)} != {n_elements}"
                )
            if not np.isfinite(raw_disp).all():
                raise ValueError(
                    "normal_displacement_m must contain only finite numbers"
                )
            dtype = np.complex128 if np.iscomplexobj(raw_disp) else np.float64
            disp_val = np.asarray(self.normal_displacement_m, dtype=dtype)
            disp_val = disp_val.copy()
            disp_val.setflags(write=False)
        else:
            raise ValueError(
                "normal_displacement_m must be a scalar or a 1D array, "
                f"got shape {raw_disp.shape}"
            )

        # Validate density_kg_m3 (strictly scalar)
        _check_no_bool(self.density_kg_m3, "density_kg_m3")
        raw_dens = np.asarray(self.density_kg_m3)
        if raw_dens.ndim != 0:
            raise ValueError(
                f"density_kg_m3 must be a scalar, got shape {raw_dens.shape}"
            )
        if not np.isfinite(raw_dens):
            raise ValueError("density_kg_m3 must be finite")
        if np.iscomplexobj(raw_dens):
            dens_val = complex(raw_dens.item())
        else:
            dens_val = float(raw_dens.item())

        # Freeze arrays
        areas = areas.copy()
        areas.setflags(write=False)

        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "normals", normalized_normals)
        object.__setattr__(self, "areas_m2", areas)
        object.__setattr__(self, "normal_displacement_m", disp_val)
        object.__setattr__(self, "density_kg_m3", dens_val)

    def as_weighted_source(self) -> WeightedPointSource:
        """Convert to the canonical WeightedPointSource format.

        weights = density * displacement * area.
        """
        is_complex = (
            np.iscomplexobj(self.normal_displacement_m)
            or isinstance(self.normal_displacement_m, complex)
            or isinstance(self.density_kg_m3, complex)
        )
        dtype = np.complex128 if is_complex else np.float64
        weights = np.asarray(
            self.density_kg_m3 * self.normal_displacement_m * self.areas_m2,
            dtype=dtype,
        )
        return WeightedPointSource(positions_m=self.positions_m, weights_kg=weights)
