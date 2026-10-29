"""GWexpy exchange for fixed-mesh surface-density series."""

from gragra.adapters._gwexpy_series import (
    GwexpySeriesBundle,
    _companion,
    _current_scalar,
    _scalar,
)
from gragra.fields.surface_density_series import FixedMeshSurfaceDensitySeries


def to_gwexpy_surface_density_series(series):
    if not isinstance(series, FixedMeshSurfaceDensitySeries):
        raise TypeError("series must be a FixedMeshSurfaceDensitySeries")
    geometry = {
        "areas_m2": series.areas_m2,
        "normals": series.normals,
        "face_metadata": series.face_metadata,
    }
    return GwexpySeriesBundle(
        _scalar(series.surface_density_kg_m2, series, "kg/m^2"),
        _companion(series, "surface_density", geometry),
    )


def from_gwexpy_surface_density_series(bundle):
    c = bundle.companion
    return FixedMeshSurfaceDensitySeries(
        c["positions_m"],
        c["areas_m2"],
        c["normals"],
        _current_scalar(bundle, "kg/m^2", "surface_density"),
        c["axis_values"],
        c["axis_kind"],
        c["face_metadata"],
        c["coordinate_frame"],
        c["coordinate_origin_m"],
        c["source_ids"],
        c["source_metadata"],
        t_ref_s=c.get("t_ref_s"),
        phase_origin_m=c.get("phase_origin_m"),
    )
