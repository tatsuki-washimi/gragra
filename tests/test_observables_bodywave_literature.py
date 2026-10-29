"""Tests for the SI entry points of LIT-B (G_SI multiplication only)."""

from __future__ import annotations

import numpy as np

from gragra.analytic.bodywave_literature import (
    halfspace_reflected_p_above_observables_g_free,
    halfspace_single_bodywave_observables_g_free,
)
from gragra.constants import G_SI
from gragra.observables.bodywave_literature import (
    halfspace_reflected_p_above_observables,
    halfspace_single_bodywave_observables,
)

RHO = 2700.0
V_P, V_S = 5800.0, 3200.0
FREQ = 5.0
TARGETS = [[1.0, 2.0, 3.0], [4.0, -5.0, -6.0]]


def test_single_mode_si_matches_g_free_times_g_si():
    kwargs = {
        "wave_type": "P",
        "density_kg_m3": RHO,
        "v_p_m_s": V_P,
        "v_s_m_s": V_S,
        "frequency_hz": FREQ,
        "incidence_angle_rad": 0.5,
        "azimuth_rad": 0.2,
        "displacement_m": 1e-9 + 0.3e-9j,
    }
    si = halfspace_single_bodywave_observables(TARGETS, **kwargs)
    g_free = halfspace_single_bodywave_observables_g_free(TARGETS, **kwargs)
    np.testing.assert_allclose(si.potential, G_SI * g_free.potential, rtol=1e-14)
    np.testing.assert_allclose(si.acceleration, G_SI * g_free.acceleration, rtol=1e-14)
    np.testing.assert_allclose(si.gradient, G_SI * g_free.gradient, rtol=1e-14)


def test_reflected_p_si_matches_g_free_times_g_si():
    kwargs = {
        "density_kg_m3": RHO,
        "v_p_m_s": V_P,
        "v_s_m_s": V_S,
        "frequency_hz": FREQ,
        "incidence_angle_rad": 0.6,
        "azimuth_rad": 0.2,
        "incident_displacement_m": 1e-9,
    }
    ground_targets = [[1.0, 2.0, 3.0], [4.0, -5.0, 6.0]]
    si = halfspace_reflected_p_above_observables(ground_targets, **kwargs)
    g_free = halfspace_reflected_p_above_observables_g_free(ground_targets, **kwargs)
    np.testing.assert_allclose(si.potential, G_SI * g_free.potential, rtol=1e-14)
    np.testing.assert_allclose(si.acceleration, G_SI * g_free.acceleration, rtol=1e-14)
    np.testing.assert_allclose(si.gradient, G_SI * g_free.gradient, rtol=1e-14)


def test_readonly_output_arrays():
    si = halfspace_single_bodywave_observables(
        TARGETS,
        wave_type="SV",
        density_kg_m3=RHO,
        v_p_m_s=V_P,
        v_s_m_s=V_S,
        frequency_hz=FREQ,
        incidence_angle_rad=0.4,
        azimuth_rad=0.1,
        displacement_m=1e-9,
    )
    for arr in (si.potential, si.acceleration, si.gradient):
        assert arr.flags.writeable is False
