"""Scientific and contract tests for stochastic Rayleigh acceleration CSD."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gragra.analytic.rayleigh_literature import (
    rayleigh_halfspace_allspace_observables_g_free,
)
from gragra.analytic.stochastic_rayleigh import (
    rayleigh_isotropic_acceleration_csd_g_free,
)
from gragra.constants import G_SI
from gragra.observables.stochastic_rayleigh import (
    rayleigh_isotropic_acceleration_csd,
)

MATERIAL = {
    "density_kg_m3": 2300.0,
    "v_p_m_s": 5800.0,
    "v_s_m_s": 3200.0,
}


def test_scalar_shape_symmetry_ratio_and_read_only() -> None:
    out = rayleigh_isotropic_acceleration_csd_g_free(
        [[3.0, -1.0, 10.0]],
        frequency_hz=2.5,
        surface_vertical_displacement_psd_m2_hz=4e-18,
        **MATERIAL,
    )
    assert out.shape == (1, 3, 3)
    assert out.dtype == np.float64
    assert not out.flags.writeable
    np.testing.assert_array_equal(out[..., 0, 1], 0.0)
    np.testing.assert_array_equal(out[..., 0, 2], 0.0)
    np.testing.assert_array_equal(out[..., 1, 2], 0.0)
    np.testing.assert_allclose(out[..., 0, 0], out[..., 1, 1], rtol=0, atol=0)
    np.testing.assert_allclose(out[..., 2, 2], 2.0 * out[..., 0, 0], rtol=0, atol=0)


def test_frequency_series_matches_scalar_and_psd_is_linear() -> None:
    targets = [[0.0, 0.0, 5.0], [2.0, 3.0, 12.0]]
    freqs = np.array([0.5, 1.5, 4.0])
    psd = np.array([1e-18, 3e-18, 2e-18])
    out = rayleigh_isotropic_acceleration_csd_g_free(
        targets,
        frequency_hz=freqs,
        surface_vertical_displacement_psd_m2_hz=psd,
        **MATERIAL,
    )
    assert out.shape == (3, 2, 3, 3)
    for index, frequency in enumerate(freqs):
        scalar = rayleigh_isotropic_acceleration_csd_g_free(
            targets,
            frequency_hz=float(frequency),
            surface_vertical_displacement_psd_m2_hz=float(psd[index]),
            **MATERIAL,
        )
        np.testing.assert_allclose(out[index], scalar, rtol=1e-14, atol=0.0)

    doubled = rayleigh_isotropic_acceleration_csd_g_free(
        targets,
        frequency_hz=freqs,
        surface_vertical_displacement_psd_m2_hz=2.0 * psd,
        **MATERIAL,
    )
    np.testing.assert_allclose(doubled, 2.0 * out, rtol=1e-15, atol=0.0)


def test_uniform_azimuth_average_matches_deterministic_rayleigh() -> None:
    target = np.array([[2.3, -0.7, 8.0]])
    frequency = 2.0
    input_psd = 7e-18
    directions = 32
    outer_sum = np.zeros((3, 3), dtype=np.complex128)
    directional_outer_products = np.empty((directions, 3, 3), dtype=np.complex128)
    for index in range(directions):
        angle = 2.0 * np.pi * index / directions
        response = rayleigh_halfspace_allspace_observables_g_free(
            target,
            frequency_hz=frequency,
            vertical_surface_amplitude_m=1.0,
            direction=(np.cos(angle), np.sin(angle), 0.0),
            **MATERIAL,
        ).acceleration[0]
        outer_product = np.outer(response, np.conjugate(response))
        directional_outer_products[index] = outer_product
        outer_sum += outer_product
    averaged = input_psd * outer_sum / directions

    # Compare sequential accumulation with a compensated sum of the same
    # float64 directional products. gamma_n models only the sequential route's
    # 31 additions and final PSD multiplication; division by 32 is exact.
    unit_roundoff = np.finfo(np.float64).eps / 2.0
    gamma_n = directions * unit_roundoff / (1.0 - directions * unit_roundoff)
    directional_magnitude = np.sum(np.abs(directional_outer_products), axis=0)
    summation_bound = (
        gamma_n
        * (input_psd / directions)
        * directional_magnitude
    )
    compensated_sum = np.empty((3, 3), dtype=np.complex128)
    for row in range(3):
        for column in range(3):
            values = directional_outer_products[:, row, column]
            compensated_sum[row, column] = complex(
                math.fsum(float(value.real) for value in values),
                math.fsum(float(value.imag) for value in values),
            )
    compensated_average = input_psd * compensated_sum / directions

    expected = rayleigh_isotropic_acceleration_csd_g_free(
        target,
        frequency_hz=frequency,
        surface_vertical_displacement_psd_m2_hz=input_psd,
        **MATERIAL,
    )[0]
    off_diagonal = ~np.eye(3, dtype=bool)
    assert np.all(np.abs(averaged - compensated_average) <= summation_bound)

    # This is a scale-aware software tolerance for the directional average;
    # gamma_n does not bound fsum or direction/response/product errors. It is
    # not a rigorous bound on exact isotropic zero or scientific/oracle error.
    software_allowance = summation_bound
    assert np.all(
        np.abs(compensated_average[off_diagonal])
        <= software_allowance[off_diagonal]
    )
    assert np.all(
        np.abs(averaged - np.conjugate(averaged.T)) <= software_allowance
    )
    assert np.all(np.abs(np.diag(averaged.imag)) <= np.diag(software_allowance))
    np.testing.assert_allclose(
        np.diag(averaged.real), np.diag(expected), rtol=2e-14, atol=0.0
    )


def test_si_boundary_is_exact_g_squared_scaling() -> None:
    kwargs = {
        **MATERIAL,
        "frequency_hz": np.array([1.0, 3.0]),
        "surface_vertical_displacement_psd_m2_hz": np.array([1e-18, 2e-18]),
    }
    g_free = rayleigh_isotropic_acceleration_csd_g_free(
        [[0.0, 0.0, 4.0]], **kwargs
    )
    si = rayleigh_isotropic_acceleration_csd([[0.0, 0.0, 4.0]], **kwargs)
    assert not si.flags.writeable
    np.testing.assert_allclose(si, g_free * G_SI**2, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"frequency_hz": 0.0}, ValueError, "positive"),
        ({"frequency_hz": [1.0, 0.0]}, ValueError, "positive"),
        (
            {"surface_vertical_displacement_psd_m2_hz": -1.0},
            ValueError,
            "non-negative",
        ),
        (
            {"surface_vertical_displacement_psd_m2_hz": [1.0, 2.0]},
            ValueError,
            "shape",
        ),
    ],
)
def test_rejects_invalid_frequency_or_psd(overrides, error, message) -> None:
    kwargs = {
        **MATERIAL,
        "frequency_hz": 1.0,
        "surface_vertical_displacement_psd_m2_hz": 1e-18,
    }
    kwargs.update(overrides)
    with pytest.raises(error, match=message):
        rayleigh_isotropic_acceleration_csd_g_free(
            [[0.0, 0.0, 1.0]], **kwargs
        )


@pytest.mark.parametrize("z", [0.0, -1.0])
def test_rejects_surface_and_underground_targets(z) -> None:
    with pytest.raises(ValueError, match="z>0"):
        rayleigh_isotropic_acceleration_csd_g_free(
            [[0.0, 0.0, z]],
            frequency_hz=1.0,
            surface_vertical_displacement_psd_m2_hz=1e-18,
            **MATERIAL,
        )


def test_rejects_complex_psd() -> None:
    with pytest.raises(TypeError, match="real"):
        rayleigh_isotropic_acceleration_csd_g_free(
            [[0.0, 0.0, 1.0]],
            frequency_hz=1.0,
            surface_vertical_displacement_psd_m2_hz=1e-18 + 0j,
            **MATERIAL,
        )
