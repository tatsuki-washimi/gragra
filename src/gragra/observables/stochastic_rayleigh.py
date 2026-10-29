"""SI stochastic Rayleigh-wave acceleration spectra."""

from __future__ import annotations

import numpy as np

from gragra.analytic.stochastic_rayleigh import (
    rayleigh_isotropic_acceleration_csd_g_free,
)
from gragra.constants import G_SI

ArrayLike = list | tuple | np.ndarray


def rayleigh_isotropic_acceleration_csd(
    targets_m: ArrayLike,
    *,
    density_kg_m3: float,
    v_p_m_s: float,
    v_s_m_s: float,
    frequency_hz: float | ArrayLike,
    surface_vertical_displacement_psd_m2_hz: float | ArrayLike,
) -> np.ndarray:
    """Return the SI one-sided acceleration CSD for isotropic Rayleigh noise.

    This is the SI boundary for the G-free stochastic Rayleigh kernel. The
    underlying acceleration response contains one factor of G_SI; the
    second-order CSD therefore contains G_SI**2.

    The read-only float64 result is in (m/s^2)^2/Hz and follows the same
    scalar/series shape contract as the G-free function.
    """
    g_free = rayleigh_isotropic_acceleration_csd_g_free(
        targets_m,
        density_kg_m3=density_kg_m3,
        v_p_m_s=v_p_m_s,
        v_s_m_s=v_s_m_s,
        frequency_hz=frequency_hz,
        surface_vertical_displacement_psd_m2_hz=(
            surface_vertical_displacement_psd_m2_hz
        ),
    )
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(g_free, dtype=np.float64) * (G_SI**2)
    if not np.isfinite(result).all():
        raise ValueError("SI Rayleigh acceleration CSD is non-finite")
    result = result.copy()
    result.setflags(write=False)
    return result


__all__ = ["rayleigh_isotropic_acceleration_csd"]
