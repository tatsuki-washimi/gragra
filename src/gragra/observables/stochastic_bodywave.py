"""SI stochastic body-wave acceleration spectra for Gen1."""

from __future__ import annotations

import numpy as np

from gragra.analytic.stochastic_bodywave import (
    fullspace_p_isotropic_acceleration_csd_g_free,
    spherical_cavity_ps_isotropic_acceleration_csd_g_free,
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


def fullspace_p_isotropic_acceleration_csd(
    *,
    density_kg_m3: float,
    frequency_hz: float | ArrayLike,
    p_displacement_psd_m2_hz: float | ArrayLike,
) -> np.ndarray:
    """Return SI one-sided acceleration CSD for isotropic full-space P waves."""
    return _to_si(
        fullspace_p_isotropic_acceleration_csd_g_free(
            density_kg_m3=density_kg_m3,
            frequency_hz=frequency_hz,
            p_displacement_psd_m2_hz=p_displacement_psd_m2_hz,
        ),
        "full-space P",
    )


def spherical_cavity_ps_isotropic_acceleration_csd(
    *,
    radius_m: float,
    density_kg_m3: float,
    v_p_m_s: float,
    v_s_m_s: float,
    frequency_hz: float | ArrayLike,
    p_displacement_psd_m2_hz: float | ArrayLike,
    s_displacement_psd_m2_hz: float | ArrayLike,
) -> np.ndarray:
    """Return SI centered spherical-cavity isotropic P/S acceleration CSD."""
    return _to_si(
        spherical_cavity_ps_isotropic_acceleration_csd_g_free(
            radius_m=radius_m,
            density_kg_m3=density_kg_m3,
            v_p_m_s=v_p_m_s,
            v_s_m_s=v_s_m_s,
            frequency_hz=frequency_hz,
            p_displacement_psd_m2_hz=p_displacement_psd_m2_hz,
            s_displacement_psd_m2_hz=s_displacement_psd_m2_hz,
        ),
        "spherical-cavity P/S",
    )


__all__ = [
    "fullspace_p_isotropic_acceleration_csd",
    "spherical_cavity_ps_isotropic_acceleration_csd",
]
