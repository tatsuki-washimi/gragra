"""Checks for the independent general-point spherical-cavity reference."""

import numpy as np
import pytest

from scripts.research_wave6_spherical_cavity_general_point import (
    _cavity_observables_at_order,
    _plane_wave_parameters,
    cavity_observables_g_free,
    center_closed_forms,
)

RADIUS_M = 1.0
DENSITY_KG_M3 = 2700.0
WAVENUMBER_RAD_M = 0.8
DISPLACEMENT_M = 1.0e-6 + 0.4e-6j
PROPAGATION_DIRECTION = np.array((0.0, 0.0, 1.0))
CENTER = np.array(((0.0, 0.0, 0.0),))


def _case(
    wave_type: str,
    polarization: np.ndarray,
    targets: np.ndarray,
    order: int,
    *,
    radius_m: float = RADIUS_M,
    wavenumber_rad_m: float = WAVENUMBER_RAD_M,
):
    return _cavity_observables_at_order(
        targets,
        radius_m=radius_m,
        density_kg_m3=DENSITY_KG_M3,
        wavenumber_rad_m=wavenumber_rad_m,
        displacement_m=DISPLACEMENT_M,
        propagation_direction=PROPAGATION_DIRECTION,
        polarization=polarization,
        wave_type=wave_type,
        order=order,
    )


def test_center_p_and_s_match_existing_closed_forms():
    closed = center_closed_forms(
        radius_m=RADIUS_M,
        density_kg_m3=DENSITY_KG_M3,
        k=WAVENUMBER_RAD_M,
        displacement_m=DISPLACEMENT_M,
    )
    p = _case("P", np.array((0.0, 0.0, 1.0)), CENTER, order=64)
    np.testing.assert_allclose(
        p["bulk"]["acceleration"][0],
        [0.0, 0.0, closed["bulk_p"]],
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        p["wall"]["acceleration"][0],
        [0.0, 0.0, closed["wall_p"]],
        atol=1.0e-14,
    )
    s = _case("S", np.array((1.0, 0.0, 0.0)), CENTER, order=64)
    np.testing.assert_allclose(s["bulk"]["acceleration"], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        s["wall"]["acceleration"][0],
        [closed["wall_s"], 0.0, 0.0],
        atol=1.0e-14,
    )


def test_center_reference_uses_solid_angle_not_radius_scaled_weights():
    radius = 2.0
    wavenumber = 0.4
    closed = center_closed_forms(
        radius_m=radius,
        density_kg_m3=DENSITY_KG_M3,
        k=wavenumber,
        displacement_m=DISPLACEMENT_M,
    )
    p = _case(
        "P",
        np.array((0.0, 0.0, 1.0)),
        CENTER,
        order=48,
        radius_m=radius,
        wavenumber_rad_m=wavenumber,
    )
    np.testing.assert_allclose(
        p["total"]["acceleration"][0, 2],
        closed["bulk_p"] + closed["wall_p"],
        atol=1.0e-13,
    )


def test_general_point_trace_obeys_laplace_or_poisson_away_from_wall():
    targets = np.array(((0.0, 0.0, 0.0), (0.2, -0.1, 0.1), (1.4, 0.2, -0.1)))
    p = _case("P", np.array((0.0, 0.0, 1.0)), targets, order=64)
    trace = np.trace(p["total"]["gradient"], axis1=1, axis2=2)
    expected = np.zeros(3, dtype=np.complex128)
    expected[2] = 4.0j * np.pi * DENSITY_KG_M3 * WAVENUMBER_RAD_M * DISPLACEMENT_M
    expected[2] *= np.exp(1j * WAVENUMBER_RAD_M * targets[2, 2])
    np.testing.assert_allclose(trace, expected, rtol=1.0e-7, atol=1.0e-10)
    np.testing.assert_allclose(
        np.trace(p["wall"]["gradient"], axis1=1, axis2=2), 0.0, atol=1.0e-10
    )

    s = _case("S", np.array((1.0, 0.0, 0.0)), targets, order=64)
    np.testing.assert_allclose(
        np.trace(s["total"]["gradient"], axis1=1, axis2=2), 0.0, atol=1.0e-10
    )


def test_checked_reference_accepts_well_conditioned_targets():
    targets = np.array(((0.0, 0.0, 0.0), (0.2, -0.1, 0.1), (1.4, 0.2, -0.1)))
    result = cavity_observables_g_free(
        targets,
        radius_m=RADIUS_M,
        density_kg_m3=DENSITY_KG_M3,
        wavenumber_rad_m=WAVENUMBER_RAD_M,
        displacement_m=DISPLACEMENT_M,
        propagation_direction=PROPAGATION_DIRECTION,
        polarization=np.array((0.0, 0.0, 1.0)),
        wave_type="P",
        order=32,
    )
    assert result["total"]["gradient"].flags.writeable is False


def test_checked_reference_rejects_underresolved_near_wall_target():
    targets = np.array(((0.0, 0.0, 0.0), (0.0, 0.0, 0.99)))
    with pytest.raises(ValueError, match=r"target_indices.*1"):
        cavity_observables_g_free(
            targets,
            radius_m=RADIUS_M,
            density_kg_m3=DENSITY_KG_M3,
            wavenumber_rad_m=WAVENUMBER_RAD_M,
            displacement_m=DISPLACEMENT_M,
            propagation_direction=PROPAGATION_DIRECTION,
            polarization=np.array((0.0, 0.0, 1.0)),
            wave_type="P",
            order=32,
        )


def test_plane_wave_rejects_incompatible_polarization():
    with pytest.raises(ValueError, match="parallel"):
        _plane_wave_parameters(
            "P",
            WAVENUMBER_RAD_M,
            DISPLACEMENT_M,
            PROPAGATION_DIRECTION,
            np.array((1.0, 0.0, 0.0)),
        )
    with pytest.raises(ValueError, match="transverse"):
        _plane_wave_parameters(
            "S",
            WAVENUMBER_RAD_M,
            DISPLACEMENT_M,
            PROPAGATION_DIRECTION,
            np.array((0.0, 0.0, 1.0)),
        )


def test_plane_wave_normalization_does_not_mutate_inputs_and_rejects_zero():
    direction = np.array((0.0, 0.0, 2.0))
    polarization = np.array((0.0, 0.0, 4.0))
    direction_before = direction.copy()
    polarization_before = polarization.copy()
    _plane_wave_parameters(
        "P",
        WAVENUMBER_RAD_M,
        DISPLACEMENT_M,
        direction,
        polarization,
    )
    np.testing.assert_array_equal(direction, direction_before)
    np.testing.assert_array_equal(polarization, polarization_before)

    with pytest.raises(ValueError, match="non-zero"):
        _plane_wave_parameters(
            "P",
            WAVENUMBER_RAD_M,
            DISPLACEMENT_M,
            np.zeros(3),
            np.array((0.0, 0.0, 1.0)),
        )


def test_checked_reference_rejects_nonfinite_target():
    with pytest.raises(ValueError, match="finite"):
        cavity_observables_g_free(
            np.array(((np.nan, 0.0, 0.0),)),
            radius_m=RADIUS_M,
            density_kg_m3=DENSITY_KG_M3,
            wavenumber_rad_m=WAVENUMBER_RAD_M,
            displacement_m=DISPLACEMENT_M,
            propagation_direction=PROPAGATION_DIRECTION,
            polarization=np.array((0.0, 0.0, 1.0)),
            wave_type="P",
            order=32,
        )
