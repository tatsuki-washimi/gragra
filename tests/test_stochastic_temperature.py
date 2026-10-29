"""Scientific and contract tests for Gen1 stochastic temperature PSD."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import kv

from gragra.analytic.stochastic_temperature import (
    uniform_advection_geometry_factor,
    uniform_advection_temperature_acceleration_psd_g_free,
)
from gragra.constants import G_SI
from gragra.observables.stochastic_temperature import (
    uniform_advection_temperature_acceleration_psd,
)


BASE = dict(
    density_kg_m3=1.2,
    temperature_k=293.0,
    structure_coefficient_k_m_neg_p_over_2=0.1,
    structure_exponent=2.0 / 3.0,
    wind_speed_m_s=10.0,
    minimum_approach_m=5.0,
    wind_axis_angle_rad=0.0,
)


def _tail_reference(x: float, order: int) -> float:
    value, _ = quad(
        lambda t: t * kv(order, t) ** 2,
        x,
        np.inf,
        epsabs=1e-13,
        epsrel=2e-12,
        limit=300,
    )
    return value


@pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 3.0, 10.0])
def test_geometry_factor_matches_independent_bessel_tail_integrals(x: float) -> None:
    d0 = 2.0 * _tail_reference(x, 0) / x**2
    d1 = 2.0 * _tail_reference(x, 1) / x**2

    parallel = float(uniform_advection_geometry_factor(x, 0.0))
    transverse = float(uniform_advection_geometry_factor(x, np.pi / 2.0))
    np.testing.assert_allclose(parallel, d0, rtol=3e-11, atol=1e-14)
    np.testing.assert_allclose(transverse, 0.5 * d1, rtol=3e-11, atol=1e-14)


def test_geometry_factor_large_x_asymptotic() -> None:
    x = 50.0
    common = np.pi / (2.0 * x**2) * np.exp(-2.0 * x)
    parallel = float(uniform_advection_geometry_factor(x, 0.0))
    transverse = float(uniform_advection_geometry_factor(x, np.pi / 2.0))
    np.testing.assert_allclose(parallel / common, 1.0, rtol=0.02, atol=0.0)
    np.testing.assert_allclose(
        transverse / (0.5 * common), 1.0, rtol=0.04, atol=0.0
    )


def test_temperature_psd_scalings_and_series_shape() -> None:
    frequencies = np.array([0.2, 0.5, 1.0])
    out = uniform_advection_temperature_acceleration_psd_g_free(
        **BASE, frequency_hz=frequencies
    )
    assert out.shape == (3,)
    assert not out.flags.writeable
    assert np.all(out >= 0.0)

    doubled_ct = uniform_advection_temperature_acceleration_psd_g_free(
        **{**BASE, "structure_coefficient_k_m_neg_p_over_2": 0.2},
        frequency_hz=frequencies,
    )
    np.testing.assert_allclose(doubled_ct, 4.0 * out, rtol=2e-15, atol=0.0)

    doubled_rho = uniform_advection_temperature_acceleration_psd_g_free(
        **{**BASE, "density_kg_m3": 2.4},
        frequency_hz=frequencies,
    )
    np.testing.assert_allclose(doubled_rho, 4.0 * out, rtol=2e-15, atol=0.0)


def test_temperature_scalar_and_si_g_squared_scaling() -> None:
    g_free = uniform_advection_temperature_acceleration_psd_g_free(
        **BASE, frequency_hz=0.5
    )
    si = uniform_advection_temperature_acceleration_psd(
        **BASE, frequency_hz=0.5
    )
    assert isinstance(g_free, float)
    assert isinstance(si, float)
    np.testing.assert_allclose(si, g_free * G_SI**2, rtol=0.0, atol=0.0)


def test_parallel_and_transverse_wind_axes_are_distinct() -> None:
    parallel = uniform_advection_temperature_acceleration_psd_g_free(
        **BASE, frequency_hz=0.5
    )
    transverse = uniform_advection_temperature_acceleration_psd_g_free(
        **{**BASE, "wind_axis_angle_rad": np.pi / 2.0},
        frequency_hz=0.5,
    )
    assert parallel > 0.0
    assert transverse > 0.0
    assert parallel != transverse


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"temperature_k": 0.0}, "positive"),
        ({"structure_exponent": 0.0}, "0 < p < 2"),
        ({"structure_exponent": 2.0}, "0 < p < 2"),
        ({"wind_speed_m_s": 0.0}, "positive"),
        ({"minimum_approach_m": 0.0}, "positive"),
        ({"structure_coefficient_k_m_neg_p_over_2": -0.1}, "non-negative"),
        ({"frequency_hz": 0.0}, "positive"),
    ],
)
def test_rejects_invalid_temperature_inputs(overrides, message: str) -> None:
    kwargs = {**BASE, "frequency_hz": 0.5}
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        uniform_advection_temperature_acceleration_psd_g_free(**kwargs)
