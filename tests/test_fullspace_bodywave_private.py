"""Private fullspace transfer: independent phase, derivatives and domain checks."""

import numpy as np
import pytest

from gragra.analytic._fullspace_bodywave import (
    fullspace_bodywave_observables_g_free,
)
from gragra.analytic.stochastic_bodywave import (
    fullspace_p_isotropic_acceleration_csd_g_free,
)


def fields(points, **changes):
    params = {
        "density_kg_m3": 1937.0,
        "wavenumber_rad_m": 0.23,
        "displacement_m": 2.3e-8 - 4.1e-9j,
        "propagation_direction": [0, 0, 2],
        "polarization": [0, 0, 7],
        "wave_type": "P",
    }
    return fullspace_bodywave_observables_g_free(points, **{**params, **changes})


def test_axis_plane_poisson_solution_phase_and_derivatives():
    points = np.array([[0, 0, 0], [1, 2, 3], [-4, 5, 7]])
    out = fields(points)
    # Fourier-Poisson independently at k along z: Phi=i*4pi*rho*xi/k.
    rho, k, amplitude = 1937.0, 0.23, 2.3e-8 - 4.1e-9j
    response = 4 * np.pi * rho * amplitude * np.exp(1j * k * points[:, 2])
    np.testing.assert_allclose(out.potential, 1j * response / k, rtol=3e-15)
    np.testing.assert_allclose(out.acceleration[:, 2], response, rtol=3e-15)
    np.testing.assert_allclose(out.gradient[:, 2, 2], 1j * k * response, rtol=3e-15)
    assert np.count_nonzero(out.acceleration[:, :2]) == 0
    assert np.count_nonzero(out.gradient[:, :2, :]) == 0
    assert not out.potential.flags.writeable
    # a=-grad(Phi), and acceleration gradient by centered target derivatives.
    delta = 1e-3
    shifted = points + np.array([0, 0, delta])
    back = points - np.array([0, 0, delta])
    np.testing.assert_allclose(
        -(fields(shifted).potential - fields(back).potential) / (2 * delta),
        out.acceleration[:, 2],
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        (fields(shifted).acceleration - fields(back).acceleration) / (2 * delta),
        out.gradient[:, :, 2],
        rtol=1e-8,
    )


def test_frame_covariance_phase_origin_and_antiparallel_p():
    points = np.array([[1, 2, 3], [-2, 0, 1]])
    origin = np.array([0.2, -0.3, 0.7])
    direction = np.array([1, 2, 2])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    a = fields(
        points,
        propagation_direction=direction,
        polarization=-direction,
        phase_origin_m=origin,
    )
    b = fields(
        points @ rotation.T,
        propagation_direction=rotation @ direction,
        polarization=-rotation @ direction,
        phase_origin_m=rotation @ origin,
    )
    np.testing.assert_allclose(a.potential, b.potential, rtol=2e-15)
    np.testing.assert_allclose(a.acceleration @ rotation.T, b.acceleration, rtol=2e-15)
    np.testing.assert_allclose(
        rotation @ a.gradient @ rotation.T, b.gradient, rtol=2e-15, atol=1e-24
    )
    shifted = fields(
        points - origin, propagation_direction=direction, polarization=-direction
    )
    np.testing.assert_allclose(a.potential, shifted.potential, rtol=2e-15)


@pytest.mark.parametrize("polarization", [[2, -1, 0], [2, 4, -5]])
def test_transverse_generic_two_polarizations_whole_field_null(polarization):
    out = fields(
        [[0, 0, 0], [2, 3, -1]],
        propagation_direction=[1, 2, 2],
        polarization=polarization,
        wave_type="S",
    )
    for quantity in out:
        assert np.count_nonzero(quantity) == 0


def test_shared_transfer_connects_first_order_and_stochastic_csd():
    rho, psd = 1937.0, 2.3e-18
    tensor = fullspace_p_isotropic_acceleration_csd_g_free(
        density_kg_m3=rho, frequency_hz=2.3, p_displacement_psd_m2_hz=psd
    )
    column = fields([[0, 0, 0]], displacement_m=np.sqrt(psd)).acceleration[0]
    np.testing.assert_allclose(
        np.diag(tensor), np.full(3, abs(column[2]) ** 2), rtol=4e-16
    )


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"density_kg_m3": True}, TypeError),
        ({"density_kg_m3": -1}, ValueError),
        ({"wavenumber_rad_m": 0}, ValueError),
        ({"wavenumber_rad_m": complex(1, 0)}, TypeError),
        ({"displacement_m": False}, TypeError),
        ({"displacement_m": complex(np.nan, 0)}, ValueError),
        ({"propagation_direction": [0, 0, 0]}, ValueError),
        ({"polarization": [0, 0, 0]}, ValueError),
        ({"polarization": [True, 0, 0]}, TypeError),
        ({"phase_origin_m": [0, np.inf, 0]}, ValueError),
        ({"wave_type": "Q"}, ValueError),
        ({"wave_type": "S"}, ValueError),
        ({"polarization": [1, 0, 0]}, ValueError),
    ],
)
def test_invalid_source_domain(changes, error):
    with pytest.raises(error):
        fields([[0, 0, 0]], **changes)


@pytest.mark.parametrize(
    "targets,error",
    [
        (True, TypeError),
        ([[0, False, 0]], TypeError),
        ([[0, 0, complex(1, 0)]], TypeError),
        ([0, 0, 0], ValueError),
        ([[0, np.nan, 0]], ValueError),
    ],
)
def test_invalid_target_domain(targets, error):
    with pytest.raises(error):
        fields(targets)


def test_first_order_si_conversion_uses_g_once():
    from gragra.constants import G_SI
    from gragra.observables.analytic import to_gravity_observables

    gf = fields([[1, 2, 3]])
    si = to_gravity_observables(gf)
    for source, result in zip(gf, si, strict=True):
        np.testing.assert_array_equal(result, G_SI * source)
    with pytest.raises(TypeError, match="already"):
        to_gravity_observables(si)


def test_zero_density_amplitude_empty_and_large_raw_norms():
    for changes in ({"density_kg_m3": 0}, {"displacement_m": 0}):
        for value in fields([[1, 2, 3]], **changes):
            assert np.count_nonzero(value) == 0
    empty = fields(np.empty((0, 3)))
    assert empty.potential.shape == (0,)
    huge = fields(
        [[1, 2, 3]], propagation_direction=[0, 0, 1e300], polarization=[0, 0, 1e-300]
    )
    normal = fields([[1, 2, 3]])
    for a, b in zip(huge, normal, strict=True):
        np.testing.assert_array_equal(a, b)


def test_nonfinite_derived_fields_rejected():
    with pytest.raises(ValueError, match="derived"):
        fields([[1, 2, 3]], density_kg_m3=1e308)


@pytest.mark.parametrize(
    "changes",
    [
        {
            "wave_type": "S",
            "polarization": [2, -1, 0],
            "propagation_direction": [1, 2, 2],
            "wavenumber_rad_m": 1e-320,
        },
        {
            "wave_type": "S",
            "polarization": [2, -1, 0],
            "propagation_direction": [1, 2, 2],
            "density_kg_m3": 1e308,
        },
        {"density_kg_m3": 0, "wavenumber_rad_m": 1e308},
        {"displacement_m": 0, "wavenumber_rad_m": 1e-320, "density_kg_m3": 1e308},
    ],
)
def test_exact_physical_null_precedes_derived_overflow(changes):
    out = fields([[1e308, 1e308, 1e308]], **changes)
    for quantity in out:
        assert np.count_nonzero(quantity) == 0
        assert not quantity.flags.writeable


def test_physical_null_still_validates_every_input():
    with pytest.raises(ValueError, match="shape"):
        fields([0, 0, 0], displacement_m=0)
    with pytest.raises(ValueError, match="phase_origin"):
        fields([[0, 0, 0]], density_kg_m3=0, phase_origin_m=[np.inf, 0, 0])
