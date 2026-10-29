"""SI boundary for general P/SV half-space gravity."""

from gragra.analytic.halfspace_psv import halfspace_psv_observables_g_free
from gragra.observables.analytic import GravityObservables, to_gravity_observables


def halfspace_psv_observables(
    targets_m,
    *,
    wave_type,
    density_kg_m3,
    v_p_m_s,
    v_s_m_s,
    frequency_hz,
    incidence_angle_rad,
    azimuth_rad,
    incident_displacement_m,
    phase_origin_m=(0.0, 0.0, 0.0),
) -> GravityObservables:
    """Return SI potential (m²/s²), acceleration (m/s²), gradient (s⁻²).

    Parameters and shapes follow
    :func:`gragra.analytic.halfspace_psv.halfspace_psv_observables_g_free`.
    The shared conversion multiplies by G exactly once at this boundary.
    """
    return to_gravity_observables(
        halfspace_psv_observables_g_free(
            targets_m,
            wave_type=wave_type,
            density_kg_m3=density_kg_m3,
            v_p_m_s=v_p_m_s,
            v_s_m_s=v_s_m_s,
            frequency_hz=frequency_hz,
            incidence_angle_rad=incidence_angle_rad,
            azimuth_rad=azimuth_rad,
            incident_displacement_m=incident_displacement_m,
            phase_origin_m=phase_origin_m,
        )
    )
