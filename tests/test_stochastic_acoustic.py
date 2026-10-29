"""Scientific and contract tests for Gen1 stochastic acoustic CSD kernels."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from gragra.analytic.stochastic_acoustic import (
    centered_cavern_isotropic_acoustic_acceleration_csd_g_free,
    isotropic_acoustic_depth_factor,
    surface_isotropic_acoustic_acceleration_csd_g_free,
    underground_isotropic_acoustic_acceleration_csd_g_free,
)
from gragra.constants import G_SI
from gragra.observables.stochastic_acoustic import (
    centered_cavern_isotropic_acoustic_acceleration_csd,
    surface_isotropic_acoustic_acceleration_csd,
    underground_isotropic_acoustic_acceleration_csd,
)


CS = 343.0


def _depth_factor_reference(x: float) -> float:
    value, _ = quad(
        lambda theta: 0.5
        * np.sin(theta) ** 3
        * np.exp(-x * np.sin(theta)),
        0.0,
        np.pi / 2.0,
        epsabs=2e-14,
        epsrel=2e-14,
        limit=200,
    )
    return value


def test_a1_surface_tensor_and_frequency_scaling() -> None:
    psd = 4.0e-4
    frequency = 2.0
    out = surface_isotropic_acoustic_acceleration_csd_g_free(
        sound_speed_m_s=CS,
        frequency_hz=frequency,
        surface_pressure_psd_pa2_hz=psd,
    )
    base = psd / (CS * frequency) ** 2
    expected = base * np.diag([1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0])
    assert out.shape == (3, 3)
    assert not out.flags.writeable
    np.testing.assert_allclose(out, expected, rtol=0.0, atol=0.0)

    high = surface_isotropic_acoustic_acceleration_csd_g_free(
        sound_speed_m_s=CS,
        frequency_hz=2.0 * frequency,
        surface_pressure_psd_pa2_hz=psd,
    )
    np.testing.assert_allclose(high, out / 4.0, rtol=2e-15, atol=0.0)


@pytest.mark.parametrize("x", [0.0, 0.1, 0.7, 1.0, 5.0, 10.0, 20.0, 50.0])
def test_a2_depth_factor_against_independent_angular_quadrature(x: float) -> None:
    actual = float(isotropic_acoustic_depth_factor(x))
    expected = _depth_factor_reference(x)
    # This is an implementation-conditioning test, not the final science
    # acceptance budget. The current analytic branch set is intentionally
    # quadrature-free and will receive a tighter source-pinned error policy.
    np.testing.assert_allclose(actual, expected, rtol=2.0e-4, atol=2.0e-14)


def test_a2_surface_limit_and_tensor_ratio() -> None:
    psd = 2.0e-4
    freqs = np.array([0.5, 2.0, 8.0])
    surface = surface_isotropic_acoustic_acceleration_csd_g_free(
        sound_speed_m_s=CS,
        frequency_hz=freqs,
        surface_pressure_psd_pa2_hz=psd,
    )
    underground_zero = underground_isotropic_acoustic_acceleration_csd_g_free(
        depth_m=0.0,
        sound_speed_m_s=CS,
        frequency_hz=freqs,
        surface_pressure_psd_pa2_hz=psd,
    )
    np.testing.assert_allclose(underground_zero, surface, rtol=2e-15, atol=0.0)

    underground = underground_isotropic_acoustic_acceleration_csd_g_free(
        depth_m=100.0,
        sound_speed_m_s=CS,
        frequency_hz=freqs,
        surface_pressure_psd_pa2_hz=psd,
    )
    np.testing.assert_allclose(
        underground[..., 0, 0], underground[..., 1, 1], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        underground[..., 2, 2], 2.0 * underground[..., 0, 0], rtol=0.0, atol=0.0
    )
    assert np.all(underground[..., 0, 0] < surface[..., 0, 0])


def test_a2_large_x_leading_asymptotic() -> None:
    x = 80.0
    actual = float(isotropic_acoustic_depth_factor(x))
    leading = 3.0 / x**4
    assert actual > leading
    np.testing.assert_allclose(actual / leading, 1.0, rtol=0.01, atol=0.0)


def test_a3_small_kr_limit_and_isotropy() -> None:
    radius = 1.0e-4
    frequency = 0.1
    pressure_psd = 3.0e-4
    out = centered_cavern_isotropic_acoustic_acceleration_csd_g_free(
        radius_m=radius,
        sound_speed_m_s=CS,
        frequency_hz=frequency,
        cavern_pressure_psd_pa2_hz=pressure_psd,
    )
    x = 2.0 * np.pi * frequency * radius / CS
    expected_component = (
        (2.0 / (CS * frequency)) ** 2
        * pressure_psd
        * x**4
        / 108.0
    )
    np.testing.assert_allclose(
        np.diag(out),
        np.full(3, expected_component),
        rtol=2e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(out, np.diag(np.diag(out)), rtol=0.0, atol=0.0)


def test_a3_frequency_series_matches_scalar() -> None:
    freqs = np.array([0.5, 3.0, 10.0])
    psd = np.array([1e-4, 2e-4, 4e-4])
    out = centered_cavern_isotropic_acoustic_acceleration_csd_g_free(
        radius_m=20.0,
        sound_speed_m_s=CS,
        frequency_hz=freqs,
        cavern_pressure_psd_pa2_hz=psd,
    )
    assert out.shape == (3, 3, 3)
    for index, frequency in enumerate(freqs):
        scalar = centered_cavern_isotropic_acoustic_acceleration_csd_g_free(
            radius_m=20.0,
            sound_speed_m_s=CS,
            frequency_hz=float(frequency),
            cavern_pressure_psd_pa2_hz=float(psd[index]),
        )
        np.testing.assert_allclose(out[index], scalar, rtol=0.0, atol=0.0)


def test_acoustic_si_wrappers_are_exact_g_squared_scaling() -> None:
    surface_kwargs = dict(
        sound_speed_m_s=CS,
        frequency_hz=np.array([1.0, 4.0]),
        surface_pressure_psd_pa2_hz=np.array([1e-4, 2e-4]),
    )
    surface_g = surface_isotropic_acoustic_acceleration_csd_g_free(**surface_kwargs)
    surface_si = surface_isotropic_acoustic_acceleration_csd(**surface_kwargs)
    np.testing.assert_allclose(surface_si, surface_g * G_SI**2, rtol=0.0, atol=0.0)

    underground_kwargs = dict(depth_m=100.0, **surface_kwargs)
    underground_g = underground_isotropic_acoustic_acceleration_csd_g_free(
        **underground_kwargs
    )
    underground_si = underground_isotropic_acoustic_acceleration_csd(
        **underground_kwargs
    )
    np.testing.assert_allclose(
        underground_si, underground_g * G_SI**2, rtol=0.0, atol=0.0
    )

    cavern_kwargs = dict(
        radius_m=20.0,
        sound_speed_m_s=CS,
        frequency_hz=np.array([1.0, 4.0]),
        cavern_pressure_psd_pa2_hz=np.array([1e-4, 2e-4]),
    )
    cavern_g = centered_cavern_isotropic_acoustic_acceleration_csd_g_free(
        **cavern_kwargs
    )
    cavern_si = centered_cavern_isotropic_acoustic_acceleration_csd(**cavern_kwargs)
    np.testing.assert_allclose(cavern_si, cavern_g * G_SI**2, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: surface_isotropic_acoustic_acceleration_csd_g_free(
                sound_speed_m_s=0.0,
                frequency_hz=1.0,
                surface_pressure_psd_pa2_hz=1e-4,
            ),
            "positive",
        ),
        (
            lambda: underground_isotropic_acoustic_acceleration_csd_g_free(
                depth_m=-1.0,
                sound_speed_m_s=CS,
                frequency_hz=1.0,
                surface_pressure_psd_pa2_hz=1e-4,
            ),
            "non-negative",
        ),
        (
            lambda: centered_cavern_isotropic_acoustic_acceleration_csd_g_free(
                radius_m=-1.0,
                sound_speed_m_s=CS,
                frequency_hz=1.0,
                cavern_pressure_psd_pa2_hz=1e-4,
            ),
            "non-negative",
        ),
        (
            lambda: surface_isotropic_acoustic_acceleration_csd_g_free(
                sound_speed_m_s=CS,
                frequency_hz=1.0,
                surface_pressure_psd_pa2_hz=-1e-4,
            ),
            "non-negative",
        ),
    ],
)
def test_rejects_invalid_acoustic_inputs(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
