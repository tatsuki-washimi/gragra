"""Disjoint development checks for temperature PSD evaluation range."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import kve

from gragra.analytic.stochastic_temperature import (
    uniform_advection_temperature_acceleration_psd_g_free,
)


@pytest.mark.parametrize("density", [1.3e140, 1.3e160])
def test_large_advection_exponent_preserves_representable_psd(density):
    # p=1 makes Gamma(p+2)=2 and sin(pi*p/2)=1. These inputs are
    # development values, not a formal fixture or an oracle import.
    wind, radius, temperature, coefficient = 13.0, 2.0, 281.0, 0.07
    theta = 0.31
    frequencies = (
        np.array([338.0, 344.0, 348.0, 389.0, 397.0]) * wind / (2 * np.pi * radius)
    )
    expected = []
    for frequency in frequencies:
        omega = 2 * np.pi * frequency
        x = omega * radius / wind
        # Independent tail integral, shifted to retain its exponential scale:
        # The parallel and perpendicular tail integrals use K0² and K1².
        integral, _ = quad(
            lambda s, x=x: (
                (x + s)
                * (
                    np.cos(theta) ** 2 * kve(0, x + s) ** 2
                    + 0.5 * np.sin(theta) ** 2 * kve(1, x + s) ** 2
                )
                * np.exp(-2 * s)
            ),
            0,
            np.inf,
            epsabs=1e-13,
            epsrel=2e-12,
        )
        scaled_geometry = 2 * integral / x**2
        with localcontext() as dc:
            dc.prec = 100
            prefactor = (
                16
                * Decimal.from_float(np.pi) ** 2
                * (
                    Decimal.from_float(density)
                    * Decimal(radius)
                    * Decimal.from_float(coefficient)
                    / Decimal(temperature)
                )
                ** 2
                * Decimal(wind)
            )
            value = (
                prefactor
                / Decimal.from_float(omega) ** 2
                * (-2 * Decimal.from_float(x)).exp()
                * Decimal.from_float(scaled_geometry)
            )
            expected.append(float(value))
    actual = uniform_advection_temperature_acceleration_psd_g_free(
        density_kg_m3=density,
        temperature_k=temperature,
        structure_coefficient_k_m_neg_p_over_2=coefficient,
        structure_exponent=1.0,
        wind_speed_m_s=wind,
        minimum_approach_m=radius,
        wind_axis_angle_rad=theta,
        frequency_hz=frequencies,
    )
    assert np.all(actual > 0)
    assert not actual.flags.writeable
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=0)
    scalar = uniform_advection_temperature_acceleration_psd_g_free(
        density_kg_m3=density,
        temperature_k=temperature,
        structure_coefficient_k_m_neg_p_over_2=coefficient,
        structure_exponent=1.0,
        wind_speed_m_s=wind,
        minimum_approach_m=radius,
        wind_axis_angle_rad=theta,
        frequency_hz=float(frequencies[-1]),
    )
    assert isinstance(scalar, float)
    np.testing.assert_allclose(scalar, expected[-1], rtol=2e-11, atol=0)
