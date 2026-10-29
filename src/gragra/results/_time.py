"""Immutable expected coordinates for a uniform GPS time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from gwpy.time import LIGOTimeGPS

from gragra.results._coords import normalize_gps


@dataclass(frozen=True, slots=True, init=False)
class TimeSampling:
    """Expected epoch, positive sample interval, and sample count.

    The epoch is stored as an integer GPS seconds/nanoseconds pair. Accessing
    ``epoch`` returns a fresh ``LIGOTimeGPS`` wrapper, so callers cannot mutate
    the coordinate retained by this immutable contract.
    """

    epoch_gps: tuple[int, int]
    dt_seconds: float
    count: int

    def __init__(self, *, epoch: object, dt: u.Quantity, count: int) -> None:
        epoch_gps = normalize_gps(epoch)
        if not isinstance(dt, u.Quantity):
            raise TypeError("dt must be an astropy Quantity with time units")
        raw = np.asarray(dt.to_value(u.s))
        if raw.shape != ():
            raise ValueError("dt must be a scalar quantity")
        if np.iscomplexobj(raw) or np.issubdtype(raw.dtype, np.bool_):
            raise TypeError("dt must be a real scalar")
        try:
            dt_seconds = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("dt must be a real scalar") from exc
        if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
            raise ValueError("dt must be positive and finite")
        if isinstance(count, (bool, np.bool_)) or not isinstance(
            count, (int, np.integer)
        ):
            raise TypeError("count must be a positive integer, not bool")
        if int(count) <= 0:
            raise ValueError("count must be a positive integer")

        object.__setattr__(self, "epoch_gps", epoch_gps)
        object.__setattr__(self, "dt_seconds", dt_seconds)
        object.__setattr__(self, "count", int(count))

    @property
    def epoch(self) -> LIGOTimeGPS:
        """Return a new GPS wrapper for the canonical epoch."""
        return LIGOTimeGPS(*self.epoch_gps)

    @property
    def dt(self) -> u.Quantity:
        """Return the sample interval as a fresh seconds quantity."""
        return self.dt_seconds * u.s


__all__ = ["TimeSampling"]
