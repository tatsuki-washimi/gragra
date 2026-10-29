"""GWexpy TimeSeriesMatrix construction with exact GPS-axis checks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from astropy import units as u
from gwexpy.timeseries import TimeSeriesMatrix

from gragra.results._coords import normalize_gps
from gragra.results._time import TimeSampling


def canonical_time_axis(matrix: TimeSeriesMatrix) -> tuple[tuple[int, int], float, int]:
    """Read canonical metadata and validate GWexpy's displayed x-index."""
    if not isinstance(matrix, TimeSeriesMatrix):
        raise TypeError("payload must be a gwexpy TimeSeriesMatrix")
    try:
        epoch_gps = normalize_gps(matrix.epoch)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("TimeSeriesMatrix epoch must be canonical GPS") from exc
    try:
        interval = np.asarray(matrix.dt.to_value(u.s))
    except (AttributeError, TypeError, ValueError, u.UnitConversionError) as exc:
        raise ValueError("TimeSeriesMatrix dt must convert to seconds") from exc
    if interval.shape != () or np.iscomplexobj(interval):
        raise ValueError("TimeSeriesMatrix dt must be a real scalar")
    dt_seconds = float(interval)
    if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError("TimeSeriesMatrix dt must be positive and finite")
    shape = getattr(matrix, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 3:
        raise ValueError("TimeSeriesMatrix must have shape (N, C, T)")
    count = shape[2]
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
        raise ValueError("TimeSeriesMatrix sample count must be an integer")
    if int(count) <= 0:
        raise ValueError("TimeSeriesMatrix sample count must be positive")
    sample_count = int(count)
    _validate_display_xindex(matrix, epoch_gps, dt_seconds, sample_count)
    return epoch_gps, dt_seconds, sample_count


def _display_times_seconds(
    epoch_gps: tuple[int, int], dt_seconds: float, count: int
) -> np.ndarray:
    """Return the absolute float-second coordinates GWexpy can display."""
    epoch_seconds = epoch_gps[0] + epoch_gps[1] * 1e-9
    with np.errstate(over="ignore", invalid="ignore"):
        return epoch_seconds + np.arange(count, dtype=np.float64) * dt_seconds


def _validate_display_xindex(
    matrix: TimeSeriesMatrix,
    epoch_gps: tuple[int, int],
    dt_seconds: float,
    count: int,
) -> None:
    """Check display coordinates without treating their float deltas as truth."""
    try:
        raw = np.asarray(matrix.xindex.to_value(u.s))
    except (AttributeError, TypeError, ValueError, u.UnitConversionError) as exc:
        raise ValueError("TimeSeriesMatrix xindex must convert to seconds") from exc
    if raw.shape != (count,) or np.iscomplexobj(raw):
        raise ValueError(
            "TimeSeriesMatrix xindex must be a real vector of sample count"
        )
    try:
        display = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("TimeSeriesMatrix xindex must contain real seconds") from exc
    expected = _display_times_seconds(epoch_gps, dt_seconds, count)
    if (
        not np.isfinite(display).all()
        or not np.isfinite(expected).all()
        or not np.array_equal(display, expected)
    ):
        raise ValueError(
            "TimeSeriesMatrix xindex must match the displayed canonical time axis"
        )


def validate_time_axis(
    matrix: TimeSeriesMatrix,
    sampling: TimeSampling,
    *,
    observable: str,
) -> None:
    """Require the payload's exact canonical axis to match its context."""
    if not isinstance(sampling, TimeSampling):
        raise TypeError("sampling must be a TimeSampling")
    epoch_gps, dt_seconds, count = canonical_time_axis(matrix)
    if (
        epoch_gps != sampling.epoch_gps
        or dt_seconds != sampling.dt_seconds
        or count != sampling.count
    ):
        raise ValueError(
            f"{observable} TimeSeriesMatrix time axis must exactly match "
            "context TimeSampling (epoch, dt, count)"
        )


def make_time_matrix(
    values: np.ndarray,
    *,
    sampling: TimeSampling,
    unit: u.UnitBase,
    rows: Sequence[str],
    cols: Sequence[str],
    name: str,
) -> TimeSeriesMatrix:
    """Build one native SI matrix from canonical expected coordinates."""
    display_times = (
        _display_times_seconds(sampling.epoch_gps, sampling.dt_seconds, sampling.count)
        * u.s
    )
    matrix = TimeSeriesMatrix(
        values,
        xindex=display_times,
        unit=unit,
        rows=list(rows),
        cols=list(cols),
        name=name,
        epoch=sampling.epoch,
    )
    # GWexpy stores its display index as float seconds. At GPS-sized epochs a
    # sufficiently small dt can round away there, so the exact cadence remains
    # separately available through this canonical metadata field.
    matrix.epoch = sampling.epoch
    matrix._dx = sampling.dt
    # GWexpy treats dimensionless-scale Units as false in its constructor path;
    # assign the physical unit to every cell explicitly so 1/s² stays intact.
    for row in range(len(rows)):
        for column in range(len(cols)):
            matrix.meta[row, column].unit = unit
    return matrix


__all__ = [
    "TimeSeriesMatrix",
    "canonical_time_axis",
    "validate_time_axis",
    "make_time_matrix",
]
