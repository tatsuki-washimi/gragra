"""SI stochastic acoustic acceleration spectra for Gen1."""

from __future__ import annotations

import numpy as np

from gragra.analytic.stochastic_acoustic import (
    centered_cavern_isotropic_acoustic_acceleration_csd_g_free,
    surface_isotropic_acoustic_acceleration_csd_g_free,
    underground_isotropic_acoustic_acceleration_csd_g_free,
)
from gragra.constants import G_SI

ArrayLike = list | tuple | np.ndarray


def _to_si(g_free: np.ndarray, label: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(g_free, dtype=np.float64) * (G_SI**2)
    if not np.isfinite(result).all():
        raise ValueError(f"SI {label} acceleration CSD is non-finite")
    result = result.copy()
    result.setflags(write=False)
    return result


def surface_isotropic_acoustic_acceleration_csd(
    *,
    sound_speed_m_s: float,
    frequency_hz: float | ArrayLike,
    surface_pressure_psd_pa2_hz: float | ArrayLike,
) -> np.ndarray:
    """Return SI A1 one-sided acceleration CSD."""
    return _to_si(
        surface_isotropic_acoustic_acceleration_csd_g_free(
            sound_speed_m_s=sound_speed_m_s,
            frequency_hz=frequency_hz,
            surface_pressure_psd_pa2_hz=surface_pressure_psd_pa2_hz,
        ),
        "surface acoustic",
    )


def underground_isotropic_acoustic_acceleration_csd(
    *,
    depth_m: float,
    sound_speed_m_s: float,
    frequency_hz: float | ArrayLike,
    surface_pressure_psd_pa2_hz: float | ArrayLike,
) -> np.ndarray:
    """Return SI A2 one-sided acceleration CSD.

    Public acceptance is limited to the observable tier in
    ``gragra.analytic.stochastic_acoustic``; development-only diagnostics are
    not formal public acceptance.
    """
    return _to_si(
        underground_isotropic_acoustic_acceleration_csd_g_free(
            depth_m=depth_m,
            sound_speed_m_s=sound_speed_m_s,
            frequency_hz=frequency_hz,
            surface_pressure_psd_pa2_hz=surface_pressure_psd_pa2_hz,
        ),
        "underground acoustic",
    )


def centered_cavern_isotropic_acoustic_acceleration_csd(
    *,
    radius_m: float,
    sound_speed_m_s: float,
    frequency_hz: float | ArrayLike,
    cavern_pressure_psd_pa2_hz: float | ArrayLike,
) -> np.ndarray:
    """Return SI A3 one-sided acceleration CSD."""
    return _to_si(
        centered_cavern_isotropic_acoustic_acceleration_csd_g_free(
            radius_m=radius_m,
            sound_speed_m_s=sound_speed_m_s,
            frequency_hz=frequency_hz,
            cavern_pressure_psd_pa2_hz=cavern_pressure_psd_pa2_hz,
        ),
        "centered cavern acoustic",
    )


__all__ = [
    "surface_isotropic_acoustic_acceleration_csd",
    "underground_isotropic_acoustic_acceleration_csd",
    "centered_cavern_isotropic_acoustic_acceleration_csd",
]
