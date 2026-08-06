"""Volume element source representation."""

from dataclasses import dataclass

import numpy as np

from gragra._arrays import _check_no_bool, as_xyz_table
from gragra.array_types import ComplexArray, FloatArray
from gragra.sources.weighted_points import WeightedPointSource


@dataclass(frozen=True)
class VolumeElementSource:
    """Volume element source representing discretized 3D volumes.

    Note that weights_kg returned by as_weighted_source() are mass elements only
    (e.g., δm = δρ * dV) in kilograms. Transfer factors or coherent phase terms
    must be handled by a separate abstraction.

    Parameters
    ----------
    positions_m : array-like
        Coordinates of shape (N, 3) in meters.
    volumes_m3 : array-like
        Volume elements of shape (N,) in cubic meters. Must be positive and finite.
    density_kg_m3 : float, complex, or array-like
        Mass density in kg/m^3. Can be a scalar or a vector of shape (N,).
    """

    positions_m: FloatArray
    volumes_m3: FloatArray
    density_kg_m3: float | complex | FloatArray | ComplexArray

    def __post_init__(self):
        # Validate positions
        positions = as_xyz_table("positions_m", self.positions_m)

        # Validate volumes
        _check_no_bool(self.volumes_m3, "volumes_m3")
        try:
            volumes = np.asarray(self.volumes_m3, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise TypeError(f"volumes_m3 must be numeric: {e}") from e

        if volumes.ndim != 1:
            raise ValueError(
                f"volumes_m3 must be a 1D array, got shape {volumes.shape}"
            )
        if len(volumes) != len(positions):
            raise ValueError(
                "Length of positions_m and volumes_m3 must match: "
                f"{len(positions)} != {len(volumes)}"
            )
        if not np.isfinite(volumes).all():
            raise ValueError("volumes_m3 must contain only finite numbers")
        if (volumes <= 0.0).any():
            raise ValueError("volumes_m3 must contain only positive values")

        # Validate density
        _check_no_bool(self.density_kg_m3, "density_kg_m3")
        density = self.density_kg_m3

        raw_dens = np.asarray(density)
        if raw_dens.ndim == 0:
            if not np.isfinite(raw_dens):
                raise ValueError("density_kg_m3 must be finite")
            if np.iscomplexobj(raw_dens):
                dens_val = complex(raw_dens.item())
            else:
                dens_val = float(raw_dens.item())
        elif raw_dens.ndim == 1:
            if len(raw_dens) != len(positions):
                raise ValueError(
                    "Length of density_kg_m3 must match positions_m: "
                    f"{len(raw_dens)} != {len(positions)}"
                )
            if not np.isfinite(raw_dens).all():
                raise ValueError("density_kg_m3 must contain only finite numbers")
            dtype = np.complex128 if np.iscomplexobj(raw_dens) else np.float64
            dens_val = np.asarray(density, dtype=dtype)
            dens_val = dens_val.copy()
            dens_val.setflags(write=False)
        else:
            raise ValueError(
                "density_kg_m3 must be a scalar or a 1D array, "
                f"got shape {raw_dens.shape}"
            )

        # Defensive copies of inputs and write=False
        volumes = volumes.copy()
        volumes.setflags(write=False)

        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "volumes_m3", volumes)
        object.__setattr__(self, "density_kg_m3", dens_val)

    def as_weighted_source(self) -> WeightedPointSource:
        """Convert to the canonical WeightedPointSource format.

        weights = density * volume.
        """
        is_complex = np.iscomplexobj(self.density_kg_m3) or isinstance(
            self.density_kg_m3, complex
        )
        dtype = np.complex128 if is_complex else np.float64
        weights = np.asarray(self.density_kg_m3 * self.volumes_m3, dtype=dtype)
        return WeightedPointSource(positions_m=self.positions_m, weights_kg=weights)
