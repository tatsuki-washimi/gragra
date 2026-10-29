"""Focused scientific-contract tests for the Wave 4--6 finite-body scripts."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import gragra
from scripts.research_wave4_finite_body import (
    AFFINE_A,
    AFFINE_B_M,
    AFFINE_CENTER_M,
    HOST_BOUNDS_M,
    INNER_BOUNDS_M,
    a_oracle_status,
    build_cartesian_six_tetra_mesh,
    evaluate_static_observables_g_free,
)
from scripts.research_wave5_finite_body import (
    affine_displacement,
    finite_motion_taylor_study,
    free_surface_column_study,
)
from scripts.research_wave6_finite_body_cavity import (
    _independent_oracle,
    _load_oracle_module,
    fixed_p1_subcell_quadrature_study,
    prescribed_p_wave_displacement,
    wave6_case_matrix,
)
from scripts.research_wave6_finite_body_oracle_retry import (
    retry_unresolved_oracle,
)

ROOT = Path(__file__).resolve().parents[1]


def test_imports_lane_worktree_source():
    assert Path(gragra.__file__).resolve().is_relative_to((ROOT / "src").resolve())


def test_cartesian_six_tetra_mesh_preserves_exact_box_and_cavity_geometry():
    mesh = build_cartesian_six_tetra_mesh(
        HOST_BOUNDS_M,
        4.0,
        inner_bounds_m=INNER_BOUNDS_M,
        host_density_kg_m3=2300.0,
        inner_density_kg_m3=0.0,
    )
    assert mesh.tetra.shape == (6 * 10 * 8 * 6, 4)
    assert mesh.cell_density_kg_m3.shape == (6 * 10 * 8 * 6,)
    assert mesh.host_volume_m3 == 40.0 * 32.0 * 24.0
    assert mesh.inner_volume_m3 == 12.0 * 8.0 * 8.0
    assert mesh.material_volume_m3 == mesh.host_volume_m3 - mesh.inner_volume_m3
    assert mesh.material_mass_kg == 2300.0 * mesh.material_volume_m3
    assert np.count_nonzero(mesh.cell_density_kg_m3 == 0.0) == 6 * 3 * 2 * 2
    for value in (mesh.vertices_m, mesh.tetra, mesh.cell_density_kg_m3):
        assert value.flags.writeable is False

    scaled_bounds = HOST_BOUNDS_M.copy()
    scaled_bounds[:2] *= 1.5
    scaled_bounds[2, 0] *= 1.5
    scaled = build_cartesian_six_tetra_mesh(
        scaled_bounds,
        4.0,
        inner_bounds_m=INNER_BOUNDS_M,
        host_density_kg_m3=2300.0,
        inner_density_kg_m3=0.0,
    )
    # The unscaled cavity faces are explicit vertices even though x=-4 and 8
    # are not on a uniform h=4 grid beginning at the scaled x lower bound.
    assert {-4.0, 8.0}.issubset(set(scaled.vertices_m[:, 0]))
    assert scaled.inner_volume_m3 == 12.0 * 8.0 * 8.0
    assert scaled.nominal_h_m == 4.0
    np.testing.assert_array_less(scaled.axis_max_cell_width_m, 4.0 + 1e-14)


def test_static_observables_have_trailing_components_and_readonly_dtype_contract():
    position = np.array([[1.0, 0.0, 0.0]])
    targets = np.array([[0.0, 0.0, 0.0]])
    real = evaluate_static_observables_g_free(position, np.array([2.0]), targets)
    assert real.potential.shape == (1,)
    assert real.acceleration.shape == (1, 3)
    assert real.gradient.shape == (1, 3, 3)
    assert real.potential.dtype == np.float64
    np.testing.assert_allclose(real.potential, [-2.0])
    np.testing.assert_allclose(real.acceleration, [[2.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        real.gradient, [[[4.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, -2.0]]]
    )

    complex_result = evaluate_static_observables_g_free(
        position, np.array([2.0 + 3.0j]), targets
    )
    assert complex_result.potential.dtype == np.complex128
    for value in complex_result:
        assert value.flags.writeable is False


def test_affine_finite_motion_fixed_mass_matches_jacobian_density_and_is_second_order():
    mesh = build_cartesian_six_tetra_mesh(
        np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]),
        1.0,
        host_density_kg_m3=2.0,
    )
    study = finite_motion_taylor_study(
        mesh,
        np.array([[3.0, -2.5, 2.25]]),
        alphas=np.array([1 / 16, 1 / 32, 1 / 64, 1 / 128]),
        b_m=AFFINE_B_M,
        matrix=AFFINE_A,
        center_m=AFFINE_CENTER_M,
    )
    assert study["jacobian_mass_relative_difference"] <= 32 * np.finfo(float).eps
    for observable in ("potential", "acceleration", "gradient"):
        record = study["taylor"][observable]
        assert len(record["physical_scale_g_free_by_target"]) == 1
        assert record["physical_scale_g_free_by_target"][0] > 0.0
        assert record["reference_noise_source"] == "deterministic_base_reevaluation"
        assert len(record["samples"]) == 4
        assert "raw_difference_g_free" in record["samples"][0]["positive"]
        assert record["second_order_confirmed"]
        for sign in ("positive", "negative"):
            target = record["order_assessment"][sign][0]
            assert target["status"] == "CONFIRMED"
            assert target["accepted_orders"]
            assert all(1.8 <= value <= 2.2 for value in target["accepted_orders"])


def test_free_surface_volume_and_sheet_are_compared_not_added_with_controls():
    study = free_surface_column_study(
        alphas_m=np.array([-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]),
        targets_m=np.array([[0.2, -0.1, 2.5]]),
        order=10,
    )
    assert study["combination_policy"] == "compare_only_do_not_add"
    assert study["controls"]["zero_displacement_exact_zero"]
    assert study["controls"]["zero_density_jump_exact_zero"]
    assert study["controls"]["sign_reversal_linear_sheet"]
    assert study["mass_change_zero_assumed"] is False
    for observable in ("potential", "acceleration", "gradient"):
        errors = np.asarray(study["relative_error_by_abs_alpha"][observable])
        assert errors[-1] < errors[0]


def test_prescribed_p_wave_and_case_matrix_are_fixed_and_separate_sweeps():
    points = np.array([[0.0, 0.0, 0.0], [5.0, 2.0, -1.0]])
    wave = prescribed_p_wave_displacement(points, wavelength_m=20.0)
    assert wave.shape == (2, 3)
    assert wave.dtype == np.complex128
    np.testing.assert_array_equal(wave[:, 1:], 0.0)
    assert wave.flags.writeable is False

    cases = wave6_case_matrix()
    assert len(cases) == 15
    assert {case["wavelength_m"] for case in cases} == {20.0, 40.0, 80.0}
    assert sum(case["sweep"] == "mesh" for case in cases) == 9
    assert sum(case["sweep"] == "extent" for case in cases) == 6
    assert all(case["amplitude_m"] == 1e-9 for case in cases)
    assert all(case["vp_m_s"] == 2800.0 for case in cases)


def test_oracle_file_loader_supports_dataclasses_with_postponed_annotations(tmp_path):
    module_path = tmp_path / "oracle.py"
    module_path.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Evidence:\n"
        "    value: float\n"
    )
    module = _load_oracle_module(module_path)
    assert module.Evidence(2.5).value == 2.5
    assert (
        module.__source_sha256__ == hashlib.sha256(module_path.read_bytes()).hexdigest()
    )


def test_fixed_p1_field_subcell_quadrature_is_separate_from_mesh_refinement():
    mesh = build_cartesian_six_tetra_mesh(
        np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]),
        1.0,
        host_density_kg_m3=2.0,
    )
    displacement = affine_displacement(
        mesh.vertices_m,
        b_m=np.array([0.2, -0.1, 0.05]),
        matrix=np.zeros((3, 3)),
        center_m=np.zeros(3),
    )
    study = fixed_p1_subcell_quadrature_study(
        mesh,
        displacement,
        np.array([[5.0, 3.0, 4.0]]),
        subcells=(1, 2, 4),
        gauss_order=2,
        tetra_batch_size=8,
        face_batch_size=16,
    )
    assert study["field_mesh_frozen"]
    assert study["tetra_count"] == len(mesh.tetra)
    assert [item["subcells_per_duffy_axis"] for item in study["curves"]] == [1, 2, 4]
    for observable in ("potential", "acceleration", "gradient"):
        residual = [
            max(item["observables"][observable]["representation_normalized_residual"])
            for item in study["curves"]
        ]
        assert residual[-1] < residual[0]
        quadrature_error = [
            max(item["observables"][observable]["mass_vs_finest_normalized_error"])
            for item in study["curves"]
        ]
        assert quadrature_error[-1] == 0.0
        assert quadrature_error[1] < quadrature_error[0]


def test_independent_oracle_budget_covers_current_a_panel_contract():
    class Recorder:
        kwargs = None

        @classmethod
        def adaptive_displacement_box_observables_g_free(cls, *_args, **kwargs):
            cls.kwargs = kwargs
            return SimpleNamespace(), SimpleNamespace()

    _independent_oracle(Recorder, HOST_BOUNDS_M, 20.0)
    assert Recorder.kwargs["max_evaluations"] == 4096


def test_a_oracle_status_requires_both_strict_convergence_and_parent_acceptance():
    convergence = {"status": "CONVERGED", "normalized_difference": 5e-12}
    assert (
        a_oracle_status(convergence, parent_accepted=False) == "CONVERGED_PROVISIONAL"
    )
    assert a_oracle_status(convergence, parent_accepted=True) == "CONVERGED_ACCEPTED"
    convergence["normalized_difference"] = 2e-11
    assert a_oracle_status(convergence, parent_accepted=True) == "ORACLE_UNRESOLVED"


def test_oracle_retry_preserves_initial_attempt_and_stops_at_strict_convergence():
    class Recorder:
        budgets = []

        @classmethod
        def adaptive_displacement_box_observables_g_free(cls, *_args, **kwargs):
            budget = kwargs["max_evaluations"]
            cls.budgets.append(budget)
            status = "CONVERGED" if budget >= 8192 else "ORACLE_UNRESOLVED"
            convergence = SimpleNamespace(
                status=status,
                normalized_difference=5e-12 if status == "CONVERGED" else 2e-5,
                low_order=10,
                high_order=12,
                subbox_count=1400,
                max_depth_reached=0,
                evaluations=budget - 20,
            )
            value = SimpleNamespace(
                potential=np.ones(4),
                acceleration=np.ones((4, 3)),
                gradient=np.ones((4, 3, 3)),
            )
            return value, convergence

    initial = {
        "status": "ORACLE_UNRESOLVED",
        "normalized_difference": 1e-4,
        "low_order": 10,
        "high_order": 12,
        "subbox_count": 1377,
        "max_depth_reached": 0,
        "evaluations": 3696,
    }
    _, convergence, attempts = retry_unresolved_oracle(
        Recorder,
        HOST_BOUNDS_M,
        20.0,
        initial_convergence=initial,
        retry_budgets=(8192, 16384),
    )
    assert [attempt["max_evaluations"] for attempt in attempts] == [4096, 8192]
    assert attempts[0]["convergence"] == initial
    assert convergence.status == "CONVERGED"
    assert Recorder.budgets == [8192]
