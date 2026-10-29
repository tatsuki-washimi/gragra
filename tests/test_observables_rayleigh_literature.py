"""Tests for the SI entry point of the literature-sourced Rayleigh gravity."""

from __future__ import annotations

import numpy as np

from gragra.analytic.rayleigh_literature import (
    rayleigh_halfspace_allspace_observables_g_free,
)
from gragra.constants import G_SI
from gragra.observables.rayleigh_literature import (
    rayleigh_halfspace_allspace_observables,
)


def test_si_matches_g_free_times_g_si():
    targets = np.array([[3.0, -1.0, 2.0], [3.0, -1.0, -2.0]])
    kwargs = {
        "targets_m": targets,
        "density_kg_m3": 2300.0,
        "v_p_m_s": np.sqrt(3.0) * 1500.0,
        "v_s_m_s": 1500.0,
        "frequency_hz": 2.5,
        "vertical_surface_amplitude_m": 1e-9,
    }
    g_free = rayleigh_halfspace_allspace_observables_g_free(**kwargs)
    si = rayleigh_halfspace_allspace_observables(**kwargs)
    np.testing.assert_allclose(si.potential, G_SI * g_free.potential, rtol=1e-13)
    np.testing.assert_allclose(si.acceleration, G_SI * g_free.acceleration, rtol=1e-13)
    np.testing.assert_allclose(si.gradient, G_SI * g_free.gradient, rtol=1e-13)


def test_si_output_is_gravity_observables_and_readonly():
    from gragra.observables.analytic import GravityObservables

    out = rayleigh_halfspace_allspace_observables(
        [[1.0, 0.0, 1.0]],
        density_kg_m3=2300.0,
        v_p_m_s=np.sqrt(3.0) * 1500.0,
        v_s_m_s=1500.0,
        frequency_hz=2.5,
        vertical_surface_amplitude_m=1e-9,
    )
    assert isinstance(out, GravityObservables)
    for arr in (out.potential, out.acceleration, out.gradient):
        assert arr.flags.writeable is False
