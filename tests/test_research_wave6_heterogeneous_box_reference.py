"""Regression checks for the independent Wave 6-M heterogeneous reference."""

import numpy as np

from scripts.research_wave6_heterogeneous_box_reference import (
    _bounds_density,
    _contrast,
    _far_field_quadrupole_attribution,
    _far_field_residuals,
    _multipole_moments,
    _reference_order_convergence,
    _translation,
    run_case,
)
from scripts.research_wave456_rotation import PASSING_TARGETS_M


def test_heterogeneous_mesh_converges_to_two_box_reference():
    times = np.linspace(-2.0, 2.0, 5)
    coarse = run_case((2, 1, 1), times_s=times, order=16)
    fine = run_case((8, 4, 4), times_s=times, order=16)

    assert coarse["total_mass_kg"] == 5400.0
    assert fine["total_mass_kg"] == 5400.0
    assert coarse["density_ratio"] == 4.0
    assert fine["point_mass_count"] > coarse["point_mass_count"]
    for observable in ("potential", "acceleration", "gradient"):
        coarse_error = coarse["relative_l2_per_target"][observable]
        fine_error = fine["relative_l2_per_target"][observable]
        assert max(fine_error) < max(coarse_error)
        assert all(np.isfinite(fine_error))


def test_ratio_two_pilot_remains_available_as_an_additional_case():
    result = run_case(
        (2, 1, 1), np.linspace(-2.0, 2.0, 3), case_name="additional_ratio_2"
    )
    assert result["total_mass_kg"] == 5400.0
    assert result["density_ratio"] == 2.0


def test_heterogeneous_contrast_separates_com_shift_from_internal_distribution():
    times = np.linspace(-2.0, 2.0, 5)
    result = _contrast(times, PASSING_TARGETS_M, order=16)

    for observable in ("potential", "acceleration", "gradient"):
        fixed = result[observable]["fixed_occupancy_vs_uniform"]
        aligned = result[observable]["com_aligned_vs_uniform"]
        assert max(aligned) < min(fixed)
        assert min(aligned) > 1.0e-5


def test_independent_box_reference_is_converged_and_moments_are_explicit():
    convergence = _reference_order_convergence()
    for observable in ("potential", "acceleration", "gradient"):
        assert max(convergence[observable]["order24_vs_36"]) < 1.0e-10

    moments = _multipole_moments(_bounds_density("original_ratio_4"))
    assert moments["mass_kg"] == 5400.0
    np.testing.assert_allclose(moments["center_m"], [0.3, 0.0, 0.0])
    np.testing.assert_allclose(moments["dipole_about_com"], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        moments["quadrupole_about_com"],
        np.diag([1728.0, -864.0, -864.0]),
        atol=1.0e-12,
    )


def test_com_aligned_far_field_residual_decreases_with_distance():
    residuals = _far_field_residuals()
    for values in residuals.values():
        assert values[0] > values[1] > values[2]


def test_quadrupole_prediction_is_compared_with_observed_far_field_difference():
    result = _far_field_quadrupole_attribution()

    np.testing.assert_allclose(
        result["delta_quadrupole_kg_m2"],
        np.diag([-972.0, 486.0, 486.0]),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result["delta_second_moment_kg_m2"],
        np.diag([-486.0, 0.0, 0.0]),
        atol=1.0e-12,
    )

    distances = np.asarray(result["distances_m"])
    assert distances.tolist() == [50.0, 100.0, 200.0]
    maximum_far_field_error = {
        "potential": 1.0e-2,
        "acceleration": 6.0e-2,
        "gradient": 8.0e-2,
    }
    for observable in ("potential", "acceleration", "gradient"):
        observed = np.asarray(result["observed_difference_g_si"][observable])
        predicted = np.asarray(result["quadrupole_prediction_g_si"][observable])
        relative_error = np.asarray(result["relative_error"][observable])
        assert observed.shape == predicted.shape
        assert observed.shape[0] == len(distances)
        assert np.isfinite(observed).all()
        assert np.isfinite(predicted).all()
        assert np.isfinite(relative_error).all()
        assert relative_error[-1] < relative_error[0]
        assert relative_error[-1] < maximum_far_field_error[observable]


def test_quadrupole_comparison_is_invariant_to_common_world_translation():
    base = _far_field_quadrupole_attribution()
    common_center = _translation(0.0)
    np.testing.assert_allclose(base["common_center_m"], common_center)
    np.testing.assert_allclose(base["expansion_center_m"], common_center)

    offset = np.array([7.0, -3.0, 2.0])
    translated = _far_field_quadrupole_attribution(world_offset_m=offset)
    np.testing.assert_allclose(translated["common_center_m"], common_center + offset)
    np.testing.assert_allclose(translated["expansion_center_m"], common_center + offset)

    for observable in ("potential", "acceleration", "gradient"):
        for field in (
            "observed_difference_g_si",
            "quadrupole_prediction_g_si",
            "relative_error",
        ):
            np.testing.assert_allclose(
                translated[field][observable],
                base[field][observable],
                rtol=1.0e-10,
                atol=1.0e-20,
            )
