"""SI entry points for the literature-sourced body-wave half-space gravity."""

from __future__ import annotations

from typing import Literal

import numpy as np

from gragra.analytic.bodywave_literature import (
    halfspace_reflected_p_above_observables_g_free,
    halfspace_single_bodywave_observables_g_free,
)
from gragra.observables.analytic import GravityObservables, to_gravity_observables

ArrayLike = list | tuple | np.ndarray


def halfspace_single_bodywave_observables(
    targets_m: ArrayLike,
    *,
    wave_type: Literal["P", "SV"],
    density_kg_m3: float,
    v_p_m_s: float,
    v_s_m_s: float,
    frequency_hz: float | ArrayLike,
    incidence_angle_rad: float,
    azimuth_rad: float,
    displacement_m: float | complex | ArrayLike,
    phase_origin_m: ArrayLike = (0.0, 0.0, 0.0),
) -> GravityObservables:
    """SI potential/acceleration/gradient of a single upward-propagating P or SV wave.

    Multiplies
    :func:`gragra.analytic.bodywave_literature.halfspace_single_bodywave_observables_g_free`
    by ``G_SI`` exactly once via
    :func:`gragra.observables.analytic.to_gravity_observables`. See that
    G-free function's docstring for the literature mapping (H19 Eq. 86-88),
    evidence classification, and parameter/shape contract.

    Returns
    -------
    GravityObservables
        SI observables (potential m^2/s^2, acceleration m/s^2, gradient s^-2).
    """
    g_free = halfspace_single_bodywave_observables_g_free(
        targets_m,
        wave_type=wave_type,
        density_kg_m3=density_kg_m3,
        v_p_m_s=v_p_m_s,
        v_s_m_s=v_s_m_s,
        frequency_hz=frequency_hz,
        incidence_angle_rad=incidence_angle_rad,
        azimuth_rad=azimuth_rad,
        displacement_m=displacement_m,
        phase_origin_m=phase_origin_m,
    )
    return to_gravity_observables(g_free)


def halfspace_reflected_p_above_observables(
    targets_m: ArrayLike,
    *,
    density_kg_m3: float,
    v_p_m_s: float,
    v_s_m_s: float,
    frequency_hz: float | ArrayLike,
    incidence_angle_rad: float,
    azimuth_rad: float,
    incident_displacement_m: float | complex | ArrayLike,
    phase_origin_m: ArrayLike = (0.0, 0.0, 0.0),
) -> GravityObservables:
    """SI potential/acceleration/gradient of incident + reflected P + converted SV.

    Multiplies
    :func:`gragra.analytic.bodywave_literature.halfspace_reflected_p_above_observables_g_free`
    by ``G_SI`` exactly once via
    :func:`gragra.observables.analytic.to_gravity_observables`. See that
    G-free function's docstring for the literature mapping (H19 Eq. 89/90),
    evidence classification, and parameter/shape contract.

    Returns
    -------
    GravityObservables
        SI observables (potential m^2/s^2, acceleration m/s^2, gradient s^-2).
    """
    g_free = halfspace_reflected_p_above_observables_g_free(
        targets_m,
        density_kg_m3=density_kg_m3,
        v_p_m_s=v_p_m_s,
        v_s_m_s=v_s_m_s,
        frequency_hz=frequency_hz,
        incidence_angle_rad=incidence_angle_rad,
        azimuth_rad=azimuth_rad,
        incident_displacement_m=incident_displacement_m,
        phase_origin_m=phase_origin_m,
    )
    return to_gravity_observables(g_free)
