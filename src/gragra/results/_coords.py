"""Shared GPS and frequency normalization without model-specific checks."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
from astropy import units as u
from astropy.time import Time


def normalize_gps(value: object) -> tuple[int, int]:
    """Return an integral GPS second/nanosecond pair without float precision loss."""
    if isinstance(value, tuple) and len(value) == 2:
        seconds, nanoseconds = value
    elif hasattr(value, "gpsSeconds") and hasattr(value, "gpsNanoSeconds"):
        seconds, nanoseconds = value.gpsSeconds, value.gpsNanoSeconds
    elif isinstance(value, Time):
        decimal_seconds = value.to_value("gps", subfmt="decimal")
        seconds = int(decimal_seconds)
        nanoseconds = int(
            ((decimal_seconds - Decimal(seconds)) * 1_000_000_000).to_integral_value()
        )
    else:
        raise TypeError(
            "reference_time must be a GPS (seconds, nanoseconds) tuple or LIGOTimeGPS"
        )
    if any(
        isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
        for x in (seconds, nanoseconds)
    ):
        raise TypeError("GPS seconds and nanoseconds must be integers")
    carry, nanos = divmod(int(nanoseconds), 1_000_000_000)
    return int(seconds) + carry, nanos


def gps_offset_seconds(later: tuple[int, int], earlier: tuple[int, int]) -> float:
    """Subtract GPS parts before float conversion for phasor reparameterization."""
    return float(later[0] - earlier[0]) + float(later[1] - earlier[1]) * 1e-9


def _frequencies_hz(value: u.Quantity) -> np.ndarray:
    """Validate and convert an ordered physical frequency axis to Hz."""
    if not isinstance(value, u.Quantity):
        raise TypeError("frequencies must be an astropy Quantity")
    raw = np.asarray(value.to_value(u.Hz))
    if np.iscomplexobj(raw):
        raise ValueError("frequencies must be real")
    frequencies = np.asarray(raw, dtype=np.float64)
    if (
        frequencies.ndim != 1
        or not len(frequencies)
        or not np.isfinite(frequencies).all()
        or np.any(frequencies <= 0)
        or np.any(np.diff(frequencies) <= 0)
    ):
        raise ValueError(
            "frequencies must be positive, finite, nonempty, and strictly increasing"
        )
    return frequencies


def one_sided_frequencies_from_quantity(value: u.Quantity) -> tuple[float, ...]:
    """Validate a one-sided axis including DC (nonnegative, strict increase)."""
    if not isinstance(value, u.Quantity):
        raise TypeError("frequencies must be an astropy Quantity")
    raw = np.asarray(value.to_value(u.Hz), dtype=np.float64)
    if (
        raw.ndim != 1
        or not len(raw)
        or not np.isfinite(raw).all()
        or np.any(raw < 0.0)
        or np.any(np.diff(raw) <= 0.0)
    ):
        raise ValueError(
            "one-sided frequencies must be finite, nonnegative, and strictly increasing"
        )
    return tuple(float(item) for item in raw)


def one_sided_frequencies_hz(count: int, dt_seconds: float) -> np.ndarray:
    """Return ``k/(M dt)`` for ``k=0..M//2`` (includes DC and Nyquist when even)."""
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
        raise TypeError("count must be a positive integer")
    sample_count = int(count)
    if sample_count <= 0:
        raise ValueError("count must be positive")
    if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError("dt_seconds must be positive and finite")
    m = sample_count
    df = 1.0 / (m * float(dt_seconds))
    return np.arange(m // 2 + 1, dtype=np.float64) * df
