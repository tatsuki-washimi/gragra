"""Scientific and contract tests for Gen1 stochastic body-wave CSD kernels."""

from __future__ import annotations

import numpy as np
import pytest

from gragra.analytic.spherical_cavity_series import (
    spherical_cavity_observables_g_free,
)
from gragra.analytic.stochastic_bodywave import (
    fullspace_p_isotropic_acceleration_csd_g_free,
    spherical_cavity_ps_isotropic_acceleration_csd_g_free,
)
from gragra.constants import G_SI
from gragra.observables.stochastic_bodywave import (
    fullspace_p_isotropic_acceleration_csd,
    spherical_cavity_ps_isotropic_acceleration_csd,
)


RHO = 2500.0
VP = 5800.0
VS = 3200.0
RADIUS = 5.0


def _outer_sum(vectors: list[np.ndarray], weight: float) -> np.ndarray:
    total = np.zeros((3, 3), dtype=np.complex128)
    for vector in vectors:
        total += weight * np.outer(vector, np.conjugate(vector))
    return total


def test_fullspace_p_scalar_formula_shape_and_read_only() -> None:
    psd = 3.0e-18
    out = fullspace_p_isotropic_acceleration_csd_g_free(
        density_kg_m3=RHO,
        frequency_hz=2.0,
        p_displacement_psd_m2_hz=psd,
    )
    expected = (4.0 * np.pi * RHO) ** 2 * psd * np.eye(3)
    assert out.shape == (3, 3)
    assert out.dtype == np.float64
    assert not out.flags.writeable
    np.testing.assert_allclose(out, expected, rtol=0.0, atol=0.0)


def test_fullspace_p_frequency_axis_and_psd_linearity() -> None:
    freqs = np.array([0.5, 2.0, 7.0])
    psd = np.array([1.0e-18, 2.0e-18, 4.0e-18])
    out = fullspace_p_isotropic_acceleration_csd_g_free(
        density_kg_m3=RHO,
        frequency_hz=freqs,
        p_displacement_psd_m2_hz=psd,
    )
    assert out.shape == (3, 3, 3)
    for index, frequency in enumerate(freqs):
        scalar = fullspace_p_isotropic_acceleration_csd_g_free(
            density_kg_m3=RHO,
            frequency_hz=float(frequency),
            p_displacement_psd_m2_hz=float(psd[index]),
        )
        np.testing.assert_allclose(out[index], scalar, rtol=0.0, atol=0.0)

    doubled = fullspace_p_isotropic_acceleration_csd_g_free(
        density_kg_m3=RHO,
        frequency_hz=freqs,
        p_displacement_psd_m2_hz=2.0 * psd,
    )
    np.testing.assert_allclose(doubled, 2.0 * out, rtol=0.0, atol=0.0)


def test_spherical_cavity_p_and_s_limits_against_deterministic_center() -> None:
    frequency = 3.0
    psd = 2.5e-18
    kp = 2.0 * np.pi * frequency / VP
    ks = 2.0 * np.pi * frequency / VS

    p_vectors = []
    for direction in np.eye(3):
        response = spherical_cavity_observables_g_free(
            [[0.0, 0.0, 0.0]],
            radius_m=RADIUS,
            density_kg_m3=RHO,
            wavenumber_rad_m=kp,
            displacement_m=1.0,
            propagation_direction=direction,
            polarization=direction,
            wave_type="P",
            center_m=(0.0, 0.0, 0.0),
            phase_origin_m=(0.0, 0.0, 0.0),
        )["total"].acceleration[0]
        p_vectors.append(response)

    # Three mutually transverse S modes whose polarization vectors span x,y,z.
    s_modes = (
        (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    )
    s_vectors = []
    for direction, polarization in s_modes:
        response = spherical_cavity_observables_g_free(
            [[0.0, 0.0, 0.0]],
            radius_m=RADIUS,
            density_kg_m3=RHO,
            wavenumber_rad_m=ks,
            displacement_m=1.0,
            propagation_direction=direction,
            polarization=polarization,
            wave_type="S",
            center_m=(0.0, 0.0, 0.0),
            phase_origin_m=(0.0, 0.0, 0.0),
        )["total"].acceleration[0]
        s_vectors.append(response)

    expected_p = spherical_cavity_ps_isotropic_acceleration_csd_g_free(
        radius_m=RADIUS,
        density_kg_m3=RHO,
        v_p_m_s=VP,
        v_s_m_s=VS,
        frequency_hz=frequency,
        p_displacement_psd_m2_hz=psd,
        s_displacement_psd_m2_hz=0.0,
    )
    expected_s = spherical_cavity_ps_isotropic_acceleration_csd_g_free(
        radius_m=RADIUS,
        density_kg_m3=RHO,
        v_p_m_s=VP,
        v_s_m_s=VS,
        frequency_hz=frequency,
        p_displacement_psd_m2_hz=0.0,
        s_displacement_psd_m2_hz=psd,
    )

    direct_p = _outer_sum(p_vectors, psd)
    direct_s = _outer_sum(s_vectors, psd)
    np.testing.assert_allclose(direct_p.imag, 0.0, atol=1e-13 * np.max(expected_p))
    np.testing.assert_allclose(direct_s.imag, 0.0, atol=1e-13 * np.max(expected_s))
    np.testing.assert_allclose(direct_p.real, expected_p, rtol=2e-11, atol=0.0)
    np.testing.assert_allclose(direct_s.real, expected_s, rtol=2e-11, atol=0.0)


def test_spherical_cavity_ps_adds_incoherent_powers() -> None:
    p_psd = 1.3e-18
    s_psd = 2.1e-18
    total = spherical_cavity_ps_isotropic_acceleration_csd_g_free(
        radius_m=RADIUS,
        density_kg_m3=RHO,
        v_p_m_s=VP,
        v_s_m_s=VS,
        frequency_hz=5.0,
        p_displacement_psd_m2_hz=p_psd,
        s_displacement_psd_m2_hz=s_psd,
    )
    p_only = spherical_cavity_ps_isotropic_acceleration_csd_g_free(
        radius_m=RADIUS,
        density_kg_m3=RHO,
        v_p_m_s=VP,
        v_s_m_s=VS,
        frequency_hz=5.0,
        p_displacement_psd_m2_hz=p_psd,
        s_displacement_psd_m2_hz=0.0,
    )
    s_only = spherical_cavity_ps_isotropic_acceleration_csd_g_free(
        radius_m=RADIUS,
        density_kg_m3=RHO,
        v_p_m_s=VP,
        v_s_m_s=VS,
        frequency_hz=5.0,
        p_displacement_psd_m2_hz=0.0,
        s_displacement_psd_m2_hz=s_psd,
    )
    np.testing.assert_allclose(total, p_only + s_only, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(total, np.diag(np.diag(total)), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.diag(total),
        np.full(3, total[0, 0]),
        rtol=0.0,
        atol=0.0,
    )


def test_spherical_cavity_small_ka_limit() -> None:
    frequency = 1.0e-5
    p_psd = 1.7e-18
    s_psd = 2.3e-18
    out = spherical_cavity_ps_isotropic_acceleration_csd_g_free(
        radius_m=1.0,
        density_kg_m3=RHO,
        v_p_m_s=VP,
        v_s_m_s=VS,
        frequency_hz=frequency,
        p_displacement_psd_m2_hz=p_psd,
        s_displacement_psd_m2_hz=s_psd,
    )
    expected_component = (
        (4.0 * np.pi * RHO / 3.0) ** 2
        * (4.0 * p_psd + s_psd)
    )
    np.testing.assert_allclose(
        np.diag(out),
        np.full(3, expected_component),
        rtol=2e-13,
        atol=0.0,
    )


def test_si_wrappers_are_exact_g_squared_scaling() -> None:
    p_kwargs = dict(
        density_kg_m3=RHO,
        frequency_hz=np.array([1.0, 4.0]),
        p_displacement_psd_m2_hz=np.array([1e-18, 2e-18]),
    )
    p_gfree = fullspace_p_isotropic_acceleration_csd_g_free(**p_kwargs)
    p_si = fullspace_p_isotropic_acceleration_csd(**p_kwargs)
    np.testing.assert_allclose(p_si, p_gfree * G_SI**2, rtol=0.0, atol=0.0)
    assert not p_si.flags.writeable

    cavity_kwargs = dict(
        radius_m=RADIUS,
        density_kg_m3=RHO,
        v_p_m_s=VP,
        v_s_m_s=VS,
        frequency_hz=np.array([1.0, 4.0]),
        p_displacement_psd_m2_hz=np.array([1e-18, 2e-18]),
        s_displacement_psd_m2_hz=np.array([3e-18, 4e-18]),
    )
    c_gfree = spherical_cavity_ps_isotropic_acceleration_csd_g_free(**cavity_kwargs)
    c_si = spherical_cavity_ps_isotropic_acceleration_csd(**cavity_kwargs)
    np.testing.assert_allclose(c_si, c_gfree * G_SI**2, rtol=0.0, atol=0.0)
    assert not c_si.flags.writeable


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: fullspace_p_isotropic_acceleration_csd_g_free(
                density_kg_m3=RHO,
                frequency_hz=0.0,
                p_displacement_psd_m2_hz=1e-18,
            ),
            "positive",
        ),
        (
            lambda: fullspace_p_isotropic_acceleration_csd_g_free(
                density_kg_m3=RHO,
                frequency_hz=1.0,
                p_displacement_psd_m2_hz=-1e-18,
            ),
            "non-negative",
        ),
        (
            lambda: spherical_cavity_ps_isotropic_acceleration_csd_g_free(
                radius_m=0.0,
                density_kg_m3=RHO,
                v_p_m_s=VP,
                v_s_m_s=VS,
                frequency_hz=1.0,
                p_displacement_psd_m2_hz=1e-18,
                s_displacement_psd_m2_hz=1e-18,
            ),
            "positive",
        ),
        (
            lambda: spherical_cavity_ps_isotropic_acceleration_csd_g_free(
                radius_m=RADIUS,
                density_kg_m3=RHO,
                v_p_m_s=VS,
                v_s_m_s=VP,
                frequency_hz=1.0,
                p_displacement_psd_m2_hz=1e-18,
                s_displacement_psd_m2_hz=1e-18,
            ),
            "requires",
        ),
    ],
)
def test_rejects_invalid_inputs(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
