"""SI entry point for the literature-sourced Rayleigh half-space gravity."""

from __future__ import annotations

import numpy as np

from gragra.analytic.rayleigh_literature import (
    rayleigh_halfspace_allspace_observables_g_free,
)
from gragra.observables.analytic import GravityObservables, to_gravity_observables

ArrayLike = list | tuple | np.ndarray


def rayleigh_halfspace_allspace_observables(
    targets_m: ArrayLike,
    *,
    density_kg_m3: float,
    v_p_m_s: float,
    v_s_m_s: float,
    frequency_hz: float | ArrayLike,
    vertical_surface_amplitude_m: float | complex | ArrayLike,
    direction: ArrayLike = (1.0, 0.0, 0.0),
    phase_origin_m: ArrayLike = (0.0, 0.0, 0.0),
) -> GravityObservables:
    """SI potential/acceleration/gradient of a Rayleigh wave above and below ground.

    Multiplies
    :func:`gragra.analytic.rayleigh_literature.rayleigh_halfspace_allspace_observables_g_free`
    by ``G_SI`` exactly once via
    :func:`gragra.observables.analytic.to_gravity_observables`. See that
    G-free function's docstring for the literature mapping (H19 Eq. 96-98),
    evidence classification, and parameter/shape contract.

    Returns
    -------
    GravityObservables
        SI observables (potential m^2/s^2, acceleration m/s^2, gradient s^-2).
    """
    g_free = rayleigh_halfspace_allspace_observables_g_free(
        targets_m,
        density_kg_m3=density_kg_m3,
        v_p_m_s=v_p_m_s,
        v_s_m_s=v_s_m_s,
        frequency_hz=frequency_hz,
        vertical_surface_amplitude_m=vertical_surface_amplitude_m,
        direction=direction,
        phase_origin_m=phase_origin_m,
    )
    return to_gravity_observables(g_free)
