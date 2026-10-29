"""Small fixed-mesh builders for mass/displacement evaluations.

These builders make one explicit, already-sampled source representation. They
do not interpolate, extrapolate, or add the mass-displacement and
bulk/interface representations together. The resulting metadata is carried
along so a field evaluation can check IDs, frame, origin, and phase before it
is handed to an existing source or series API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gragra._arrays import _check_no_bool, as_xyz_table, as_xyz_vector
from gragra.fields.displacement_series import FixedMeshDisplacementSeries
from gragra.fields.surface import SurfaceDisplacementField
from gragra.gravity_sources import (
    SurfaceMassSheetSource,
    VolumeElementSource,
    WeightedPointSource,
)
from gragra.waves import plane_wave_displacement


def _source_ids(source_ids, count: int) -> tuple[str, ...]:
    if source_ids is None:
        raise TypeError("source_ids must be explicitly provided as N strings")
    if isinstance(source_ids, (str, bytes)):
        raise TypeError("source_ids must be N strings")
    ids = tuple(source_ids)
    if len(ids) != count:
        raise ValueError("source_ids must contain one ID per source")
    if any(not isinstance(item, str) for item in ids):
        raise TypeError("source_ids must be N strings")
    if any(not item for item in ids):
        raise ValueError("source_ids must be non-empty strings")
    if len(set(ids)) != count:
        raise ValueError("source_ids must be unique")
    return ids


def _frame(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("coordinate_frame must be a non-empty string")
    return value


def _positive_vector(name: str, value, count: int) -> np.ndarray:
    _check_no_bool(value, name)
    raw = np.asarray(value)
    if raw.dtype.kind in "bOc" or not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"{name} must be a finite real array")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError(f"{name} must contain only positive finite values")
    result = result.copy()
    result.setflags(write=False)
    return result


def _density(value) -> float:
    _check_no_bool(value, "density_kg_m3")
    raw = np.asarray(value)
    if (
        raw.ndim != 0
        or raw.dtype.kind in "bOc"
        or not np.issubdtype(raw.dtype, np.number)
    ):
        raise TypeError("density_kg_m3 must be a real scalar")
    result = float(raw)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("density_kg_m3 must be non-negative and finite")
    return result


def _displacement(value, count: int) -> np.ndarray:
    _check_no_bool(value, "displacement_m")
    raw = np.asarray(value)
    if raw.dtype.kind in "bO" or not np.issubdtype(raw.dtype, np.number):
        raise TypeError("displacement_m must be numeric")
    if raw.shape != (count, 3):
        raise ValueError(f"displacement_m must have shape ({count}, 3)")
    dtype = np.complex128 if np.iscomplexobj(raw) else np.float64
    result = np.asarray(value, dtype=dtype).copy()
    if not np.isfinite(result).all():
        raise ValueError("displacement_m must contain only finite values")
    result.setflags(write=False)
    return result


def _finite_phase(value: float, name: str) -> float:
    _check_no_bool(value, name)
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class FixedMeshDisplacementSources:
    """One fixed mesh's SI background mass and sampled displacement field.

    ``mass_source`` and ``volume_source`` are alternative views of the same
    background mass. They are not meant to be summed with one another or with
    a surface representation.
    """

    positions_m: np.ndarray
    volumes_m3: np.ndarray
    density_kg_m3: float
    mass_kg: np.ndarray
    displacement_m: np.ndarray
    source_ids: tuple[str, ...]
    coordinate_frame: str
    coordinate_origin_m: np.ndarray
    phase_origin_m: np.ndarray
    phase0_rad: float
    unit_system: str = "SI"

    @property
    def volume_source(self) -> VolumeElementSource:
        """Return the existing density-times-volume source representation."""
        return VolumeElementSource(
            self.positions_m, self.volumes_m3, self.density_kg_m3
        )

    @property
    def mass_source(self) -> WeightedPointSource:
        """Return the existing background-mass point representation."""
        return WeightedPointSource(self.positions_m, self.mass_kg)

    def as_frequency_series(
        self, frequency_hz: float, *, t_ref_s: float = 0.0
    ) -> FixedMeshDisplacementSeries:
        """Wrap this one sampled field in the existing one-sample series API."""
        frequency = _finite_phase(frequency_hz, "frequency_hz")
        if frequency <= 0.0:
            raise ValueError("frequency_hz must be positive")
        return FixedMeshDisplacementSeries(
            self.positions_m,
            self.mass_kg,
            self.displacement_m[None, :, :],
            np.array([frequency], dtype=np.float64),
            "frequency",
            self.coordinate_frame,
            self.coordinate_origin_m,
            self.source_ids,
            {"phase0_rad": self.phase0_rad, "unit_system": self.unit_system},
            t_ref_s=t_ref_s,
            phase_origin_m=self.phase_origin_m,
        )


@dataclass(frozen=True)
class FixedMeshSurfaceSources:
    """One fixed triangle mesh's SI surface sheet plus shared metadata."""

    source: SurfaceMassSheetSource
    source_ids: tuple[str, ...]
    coordinate_frame: str
    coordinate_origin_m: np.ndarray
    phase_origin_m: np.ndarray
    phase0_rad: float
    unit_system: str = "SI"


def _build_displacement_sources(
    positions_m,
    volumes_m3,
    density_kg_m3,
    displacement_m,
    *,
    source_ids,
    coordinate_frame,
    coordinate_origin_m,
    phase_origin_m,
    phase0_rad,
) -> FixedMeshDisplacementSources:
    positions = as_xyz_table("positions_m", positions_m)
    count = len(positions)
    volumes = _positive_vector("volumes_m3", volumes_m3, count)
    density = _density(density_kg_m3)
    displacement = _displacement(displacement_m, count)
    ids = _source_ids(source_ids, count)
    frame = _frame(coordinate_frame)
    origin = as_xyz_vector("coordinate_origin_m", coordinate_origin_m)
    phase_origin = as_xyz_vector("phase_origin_m", phase_origin_m)
    phase0 = _finite_phase(phase0_rad, "phase0_rad")
    mass = np.asarray(density * volumes, dtype=np.float64)
    mass.setflags(write=False)
    return FixedMeshDisplacementSources(
        positions,
        volumes,
        density,
        mass,
        displacement,
        ids,
        frame,
        origin,
        phase_origin,
        phase0,
    )


def build_fixed_mesh_displacement_sources(
    positions_m,
    volumes_m3,
    density_kg_m3,
    displacement_m,
    *,
    source_ids,
    coordinate_frame: str = "world",
    coordinate_origin_m=(0.0, 0.0, 0.0),
    phase_origin_m=(0.0, 0.0, 0.0),
    phase0_rad: float = 0.0,
) -> FixedMeshDisplacementSources:
    """Build fixed-volume mass and displacement sources without interpolation."""
    return _build_displacement_sources(
        positions_m,
        volumes_m3,
        density_kg_m3,
        displacement_m,
        source_ids=source_ids,
        coordinate_frame=coordinate_frame,
        coordinate_origin_m=coordinate_origin_m,
        phase_origin_m=phase_origin_m,
        phase0_rad=phase0_rad,
    )


def build_plane_wave_fixed_mesh_sources(
    positions_m,
    volumes_m3,
    density_kg_m3,
    *,
    k_hat,
    wavenumber_rad_m: float,
    wave_type: str,
    amplitude_m: float = 1.0,
    phase0: float = 0.0,
    phase_origin_m=(0.0, 0.0, 0.0),
    source_ids,
    coordinate_frame: str = "world",
    coordinate_origin_m=(0.0, 0.0, 0.0),
) -> FixedMeshDisplacementSources:
    """Sample an existing P/S plane-wave constructor on a fixed mesh once."""
    phase_origin = as_xyz_vector("phase_origin_m", phase_origin_m)
    displacement = plane_wave_displacement(
        np.asarray(positions_m, dtype=np.float64) - phase_origin,
        k_hat,
        wavenumber_rad_m,
        wave_type,
        amplitude_m=amplitude_m,
        phase0=phase0,
    )
    if displacement.shape[1:] != (1, 1, 3):
        raise ValueError("plane-wave builder requires one direction and one frequency")
    return _build_displacement_sources(
        positions_m,
        volumes_m3,
        density_kg_m3,
        displacement[:, 0, 0, :],
        source_ids=source_ids,
        coordinate_frame=coordinate_frame,
        coordinate_origin_m=coordinate_origin_m,
        phase_origin_m=phase_origin,
        phase0_rad=phase0,
    )


def build_fixed_mesh_surface_sources(
    vertices_m,
    triangles,
    displacement_m,
    density_jump_kg_m3: float,
    *,
    source_ids,
    coordinate_frame: str = "world",
    coordinate_origin_m=(0.0, 0.0, 0.0),
    phase_origin_m=(0.0, 0.0, 0.0),
    phase0_rad: float = 0.0,
) -> FixedMeshSurfaceSources:
    """Build surface density-times-area sources from one sampled triangle field."""
    field = SurfaceDisplacementField(vertices_m, triangles, displacement_m)
    source = field.as_surface_mass_source(density_jump_kg_m3)
    ids = _source_ids(source_ids, len(source.positions_m))
    return FixedMeshSurfaceSources(
        source,
        ids,
        _frame(coordinate_frame),
        as_xyz_vector("coordinate_origin_m", coordinate_origin_m),
        as_xyz_vector("phase_origin_m", phase_origin_m),
        _finite_phase(phase0_rad, "phase0_rad"),
    )
