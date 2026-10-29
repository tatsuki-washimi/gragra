"""GWexpy exchange for density time series and phasors."""

from gragra.adapters._gwexpy_series import (
    GwexpySeriesBundle,
    _companion,
    _current_scalar,
    _scalar,
)
from gragra.fields.density_series import DensityPhasors, DensityTimeSeries


def to_gwexpy_density_series(series):
    if not isinstance(series, (DensityTimeSeries, DensityPhasors)):
        raise TypeError("series must be a DensityTimeSeries or DensityPhasors")
    return GwexpySeriesBundle(
        _scalar(series.density_kg_m3, series, "kg/m^3"),
        _companion(
            series,
            "density",
            {"volumes_m3": series.volumes_m3, "density_kind": series.density_kind},
        ),
    )


def from_gwexpy_density_series(bundle):
    c = bundle.companion
    values = _current_scalar(bundle, "kg/m^3", "density")
    args = (c["positions_m"], c["volumes_m3"], values, c["axis_values"])
    common = (
        c["coordinate_frame"],
        c["coordinate_origin_m"],
        c["source_ids"],
        c["source_metadata"],
    )
    if c["axis_kind"] == "time":
        return DensityTimeSeries(*args, c["density_kind"], *common)
    if c.get("density_kind") != "delta_rho":
        raise ValueError("frequency density_kind must be 'delta_rho'")
    return DensityPhasors(*args, c["t_ref_s"], c["phase_origin_m"], *common)


def to_gwexpy_density_time_series(series: DensityTimeSeries):
    """Convert only a real time-series density input to native GWexpy."""
    if not isinstance(series, DensityTimeSeries):
        raise TypeError("series must be a DensityTimeSeries")
    return to_gwexpy_density_series(series)


def from_gwexpy_density_time_series(bundle):
    """Restore a real density time series after validating its native layout."""
    series = from_gwexpy_density_series(bundle)
    if not isinstance(series, DensityTimeSeries):
        raise ValueError("bundle does not contain a density time series")
    return series


def to_gwexpy_density_phasors(series: DensityPhasors):
    """Convert only complex density phasors to native GWexpy."""
    if not isinstance(series, DensityPhasors):
        raise TypeError("series must be a DensityPhasors")
    return to_gwexpy_density_series(series)


def from_gwexpy_density_phasors(bundle):
    """Restore density phasors after validating their native layout."""
    series = from_gwexpy_density_series(bundle)
    if not isinstance(series, DensityPhasors):
        raise ValueError("bundle does not contain density phasors")
    return series
