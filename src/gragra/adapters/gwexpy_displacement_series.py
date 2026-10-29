"""GWexpy exchange for fixed-mesh displacement series."""

from gragra.adapters._gwexpy_series import (
    GwexpySeriesBundle,
    _companion,
    _current_vector,
    _vector,
)
from gragra.fields.displacement_series import FixedMeshDisplacementSeries


def to_gwexpy_displacement_series(series):
    if not isinstance(series, FixedMeshDisplacementSeries):
        raise TypeError("series must be a FixedMeshDisplacementSeries")
    return GwexpySeriesBundle(
        _vector(series.displacement_m, series),
        _companion(series, "displacement", {"mass_kg": series.mass_kg}),
    )


def from_gwexpy_displacement_series(bundle):
    c = bundle.companion
    return FixedMeshDisplacementSeries(
        c["positions_m"],
        c["mass_kg"],
        _current_vector(bundle),
        c["axis_values"],
        c["axis_kind"],
        c["coordinate_frame"],
        c["coordinate_origin_m"],
        c["source_ids"],
        c["source_metadata"],
        t_ref_s=c.get("t_ref_s"),
        phase_origin_m=c.get("phase_origin_m"),
    )
