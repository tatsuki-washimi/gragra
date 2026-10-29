"""SI uniform-advection stochastic temperature gravity for Gen1."""

from __future__ import annotations

import numpy as np

from gragra.analytic.stochastic_temperature import (
    uniform_advection_temperature_acceleration_psd_g_free,
)
from gragra.constants import G_SI

ArrayLike = list | tuple | np.ndarray


def uniform_advection_temperature_acceleration_psd(
    *,
    density_kg_m3: float,
    temperature_k: float,
    structure_coefficient_k_m_neg_p_over_2: float,
    structure_exponent: float,
    wind_speed_m_s: float,
    minimum_approach_m: float,
    wind_axis_angle_rad: float,
    frequency_hz: float | ArrayLike,
) -> np.ndarray | float:
    """Return the SI T1 one-sided projected acceleration PSD."""
    g_free = uniform_advection_temperature_acceleration_psd_g_free(
        density_kg_m3=density_kg_m3,
        temperature_k=temperature_k,
        structure_coefficient_k_m_neg_p_over_2=(
            structure_coefficient_k_m_neg_p_over_2
        ),
        structure_exponent=structure_exponent,
        wind_speed_m_s=wind_speed_m_s,
        minimum_approach_m=minimum_approach_m,
        wind_axis_angle_rad=wind_axis_angle_rad,
        frequency_hz=frequency_hz,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(g_free, dtype=np.float64) * (G_SI**2)
    if not np.isfinite(result).all():
        raise ValueError("SI temperature acceleration PSD is non-finite")
    if result.ndim == 0:
        return float(result)
    result = result.copy()
    result.setflags(write=False)
    return result


__all__ = ["uniform_advection_temperature_acceleration_psd"]
