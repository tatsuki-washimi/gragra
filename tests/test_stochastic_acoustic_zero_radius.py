"""Centered-cavern zero-radius limit and public tensor transport."""

import numpy as np
import pytest

from gragra.analytic.stochastic_acoustic import (
    centered_cavern_isotropic_acoustic_acceleration_csd_g_free,
)
from gragra.constants import G_SI
from gragra.observables.stochastic_acoustic import (
    centered_cavern_isotropic_acoustic_acceleration_csd,
)

ENDPOINTS = (
    centered_cavern_isotropic_acoustic_acceleration_csd_g_free,
    centered_cavern_isotropic_acoustic_acceleration_csd,
)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("pressure", [0.0, 0.0037])
def test_zero_radius_is_readonly_whole_tensor_zero(endpoint, vector, pressure):
    result = endpoint(
        radius_m=0.0,
        sound_speed_m_s=329.0,
        frequency_hz=[0.83, 2.19, 7.31] if vector else 2.19,
        cavern_pressure_psd_pa2_hz=pressure,
    )
    assert result.shape == ((3, 3, 3) if vector else (3, 3))
    assert result.dtype == np.float64
    assert not result.flags.writeable
    np.testing.assert_array_equal(result, np.zeros_like(result))


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_negative_radius_is_rejected(endpoint):
    with pytest.raises(ValueError, match="non-negative"):
        endpoint(
            radius_m=-0.23,
            sound_speed_m_s=329.0,
            frequency_hz=2.19,
            cavern_pressure_psd_pa2_hz=0.0037,
        )


def test_positive_radius_scalar_vector_and_si_transport():
    kwargs = {
        "radius_m": 6.7,
        "sound_speed_m_s": 329.0,
        "cavern_pressure_psd_pa2_hz": [0.0037, 0.0, 0.0019],
    }
    frequencies = [0.83, 2.19, 7.31]
    gf = ENDPOINTS[0](frequency_hz=frequencies, **kwargs)
    si = ENDPOINTS[1](frequency_hz=frequencies, **kwargs)
    assert not gf.flags.writeable and not si.flags.writeable
    np.testing.assert_allclose(si, gf * G_SI**2, rtol=2e-15, atol=0)
    np.testing.assert_array_equal(gf[1], np.zeros((3, 3)))
    assert gf[0, 0, 0] > 0 and gf[2, 0, 0] > 0
    for i, frequency in enumerate(frequencies):
        scalar = ENDPOINTS[0](
            radius_m=6.7,
            sound_speed_m_s=329.0,
            frequency_hz=frequency,
            cavern_pressure_psd_pa2_hz=kwargs["cavern_pressure_psd_pa2_hz"][i],
        )
        np.testing.assert_array_equal(scalar, gf[i])
        assert not scalar.flags.writeable
