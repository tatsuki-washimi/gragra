"""Phase-speed normalization for spherical-cavity and spherical-shell inputs.

This module defines the per-bin P/S phase-speed contract for those two finite
spherical wave models. It does not define material dispersion for P/SV or
Rayleigh models.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.units import UnitConversionError
from gwexpy.frequencyseries import FrequencySeries


def _real_finite_values(values: object, name: str) -> np.ndarray:
    """Copy real numeric values to float64 and reject nonfinite or non-real data."""
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        copied = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(copied).all():
        raise ValueError(f"{name} must be finite")
    return copied


def _values_in_unit(
    values: object, source_unit: u.UnitBase, target_unit: u.UnitBase, name: str
) -> np.ndarray:
    """Convert real values between compatible units and return an owned array."""
    copied = _real_finite_values(values, name)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            converted = u.Quantity(copied, source_unit, copy=True).to_value(target_unit)
    except UnitConversionError as exc:
        raise ValueError(
            f"{name} must have units compatible with {target_unit}"
        ) from exc
    return _real_finite_values(converted, name)


def _evaluation_frequencies_hz(frequencies: object) -> np.ndarray:
    if not isinstance(frequencies, u.Quantity):
        raise TypeError("frequencies must be an Astropy Quantity")
    raw = _real_finite_values(frequencies.value, "frequencies")
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("frequencies must be a nonempty one-dimensional axis")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            hz = u.Quantity(raw, frequencies.unit, copy=True).to_value(u.Hz)
    except UnitConversionError as exc:
        raise ValueError("frequencies must have units compatible with Hz") from exc
    hz = _real_finite_values(hz, "frequencies")
    if np.any(hz <= 0) or np.any(np.diff(hz) <= 0):
        raise ValueError("frequencies must be positive and strictly increasing")
    return hz


def _series_frequencies_hz(series: FrequencySeries) -> np.ndarray:
    coordinates = series.frequencies
    return _values_in_unit(
        np.asarray(coordinates), coordinates.unit, u.Hz, "phase-speed frequencies"
    )


def _spherical_phase_speed_to_wavenumbers(
    phase_speed: u.Quantity | FrequencySeries,
    frequencies: u.Quantity,
    *,
    mode: str,
) -> u.Quantity:
    """Normalize a spherical cavity/shell P or S phase speed to ``k(f)``.

    ``phase_speed`` is either a real, finite, positive scalar velocity Quantity,
    constant over the supplied frequency axis, or a coordinate-bearing GWexpy
    ``FrequencySeries`` with real, finite, positive values. Series coordinates
    must exactly equal the evaluation frequencies after conversion to Hz; this
    helper never interpolates, extrapolates, or fills bins. ``mode`` is
    deliberately explicit and limited to P or S.

    Returns an owned, read-only Quantity in rad/m with one value per evaluation
    frequency, computed independently as ``2*pi*f/v_phase``.
    """
    if not isinstance(mode, str) or mode not in ("P", "S"):
        raise ValueError("mode must be explicitly 'P' or 'S' for spherical models")
    frequency_hz = _evaluation_frequencies_hz(frequencies)

    if isinstance(phase_speed, FrequencySeries):
        values = _values_in_unit(
            phase_speed.value,
            phase_speed.unit,
            u.m / u.s,
            "phase_speed",
        )
        if values.ndim != 1 or values.size == 0:
            raise ValueError("phase_speed FrequencySeries must be one-dimensional")
        coordinates_hz = _series_frequencies_hz(phase_speed)
        if coordinates_hz.shape != frequency_hz.shape or not np.array_equal(
            coordinates_hz, frequency_hz
        ):
            raise ValueError(
                "phase_speed FrequencySeries coordinates must exactly match "
                "evaluation frequencies after unit normalization"
            )
        if np.any(values <= 0):
            raise ValueError("phase_speed values must be positive")
        speeds_m_s = values
    elif isinstance(phase_speed, u.Quantity):
        values = _values_in_unit(
            phase_speed.value,
            phase_speed.unit,
            u.m / u.s,
            "phase_speed",
        )
        if values.ndim != 0:
            raise ValueError("scalar phase_speed Quantity must be zero-dimensional")
        if values <= 0:
            raise ValueError("phase_speed must be positive")
        speeds_m_s = float(values)
    else:
        raise TypeError("phase_speed must be a scalar Quantity or FrequencySeries")

    frequency_mantissa, frequency_exponent = np.frexp(frequency_hz)
    speed_mantissa, speed_exponent = np.frexp(speeds_m_s)
    scaled_wavenumber = (frequency_mantissa / speed_mantissa) * (2.0 * np.pi)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        wavenumber = np.ldexp(scaled_wavenumber, frequency_exponent - speed_exponent)
    wavenumber = _real_finite_values(wavenumber, "derived wavenumbers")
    if np.any(wavenumber <= 0):
        raise ValueError("derived wavenumbers must be positive")

    result = u.Quantity(np.array(wavenumber, dtype=np.float64, copy=True), u.rad / u.m)
    result.base.setflags(write=False)
    result.setflags(write=False)
    return result


__all__ = ["_spherical_phase_speed_to_wavenumbers"]
