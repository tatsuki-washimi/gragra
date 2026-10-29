"""Acceptance tests for the finite smooth water-surface research fixture."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gragra.constants import G_SI
from gragra.gravity_sources import WeightedPointSource
from gragra.integrators.point_direct import direct_point_sum
from scripts.research_water_surface_models import (
    ALPHAS_M,
    ETA_REF_M,
    GRADIENT_COMPONENT_ORDER,
    SOURCE_MODEL_STATUS,
    TARGETS_M,
    _classify_volume_block,
    _oracle_module,
    _select_amplitude_record,
    composite_gl2_surface,
    csd_transfer_error_bound,
    evaluate_surface,
    make_surface_series,
    phasor_samples,
    propagate_csd,
    run_deterministic_validation,
    write_validation,
)


def test_composite_gl2_surface_preserves_area_and_series_contract():
    positions, areas, normals, source_ids = composite_gl2_surface(16)
    assert positions.shape == (4 * 16**2, 3)
    assert areas.shape == (4 * 16**2,)
    assert normals.shape == positions.shape
    assert len(source_ids) == len(positions) == len(set(source_ids))
    np.testing.assert_allclose(positions[:, 2], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        normals,
        np.broadcast_to((0.0, 0.0, 1.0), normals.shape),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(np.sum(areas), 4.0, rtol=0.0, atol=1e-14)

    series = make_surface_series(16, ETA_REF_M)
    assert series.surface_density_kg_m2.shape == (1, 4 * 16**2)
    assert series.axis_kind == "time"
    assert series.t_ref_s is None
    assert series.phase_origin_m is None
    assert series.positions_m.flags.writeable is False
    assert series.areas_m2.flags.writeable is False
    assert series.surface_density_kg_m2.flags.writeable is False


@pytest.mark.parametrize(
    ("observable", "tail"),
    [("potential", ()), ("acceleration", (3,)), ("gradient", (3, 3))],
)
def test_surface_observables_have_expected_shapes_and_g_once(observable, tail):
    series = make_surface_series(16, ETA_REF_M)
    result = evaluate_surface(16, ETA_REF_M)[observable]
    assert result.shape == (len(TARGETS_M), *tail)
    assert result.flags.writeable is False
    expected_g_free = direct_point_sum(
        WeightedPointSource(
            series.positions_m,
            series.areas_m2 * series.surface_density_kg_m2[0],
        ),
        TARGETS_M,
        observable=observable,
        softening_m=0.0,
        chunk_size=8192,
        backend="numpy",
    )
    np.testing.assert_array_equal(result, G_SI * expected_g_free)


def test_linear_sheet_zero_sign_reflection_trace_and_complex_linearity():
    positive = evaluate_surface(16, ETA_REF_M)
    negative = evaluate_surface(16, -ETA_REF_M)
    zero = evaluate_surface(16, 0.0)
    complex_factor = 1.0 + 2.0j
    complex_result = evaluate_surface(
        16,
        ETA_REF_M * complex_factor,
        frequency_hz=3.0,
        t_ref_s=0.0,
        phase_origin_m=np.zeros(3),
    )

    for observable in positive:
        np.testing.assert_array_equal(negative[observable], -positive[observable])
        np.testing.assert_array_equal(zero[observable], 0.0)
        np.testing.assert_allclose(
            complex_result[observable],
            complex_factor * positive[observable],
            rtol=2e-15,
            atol=0.0,
        )

    # TARGETS_M is (+x,+y), (-x,+y), (+x,-y), (-x,-y).
    reflection_x = np.diag((-1.0, 1.0, 1.0))
    reflection_y = np.diag((1.0, -1.0, 1.0))
    reflection_xy = reflection_y @ reflection_x
    np.testing.assert_allclose(positive["potential"], positive["potential"][0])
    np.testing.assert_allclose(
        positive["acceleration"][1], reflection_x @ positive["acceleration"][0]
    )
    np.testing.assert_allclose(
        positive["acceleration"][2], reflection_y @ positive["acceleration"][0]
    )
    np.testing.assert_allclose(
        positive["gradient"][1],
        reflection_x @ positive["gradient"][0] @ reflection_x,
    )
    np.testing.assert_allclose(
        positive["gradient"][2],
        reflection_y @ positive["gradient"][0] @ reflection_y,
    )
    np.testing.assert_allclose(
        positive["acceleration"][3],
        reflection_xy @ positive["acceleration"][0],
    )
    np.testing.assert_allclose(
        positive["gradient"][3],
        reflection_xy @ positive["gradient"][0] @ reflection_xy,
    )
    np.testing.assert_allclose(
        positive["gradient"],
        np.swapaxes(positive["gradient"], -1, -2),
        rtol=0.0,
        atol=1e-27,
    )
    trace = np.trace(positive["gradient"], axis1=-2, axis2=-1)
    trace_floor = (
        128.0
        * np.finfo(np.float64).eps
        * np.linalg.norm(positive["gradient"], axis=(-2, -1))
    )
    assert np.all(np.abs(trace) <= trace_floor)


def test_nonzero_height_with_zero_density_contrast_is_zero_for_all_observables():
    zero_contrast = evaluate_surface(
        16,
        ETA_REF_M,
        density_jump_kg_m3=0.0,
    )
    for observable in ("potential", "acceleration", "gradient"):
        np.testing.assert_array_equal(zero_contrast[observable], 0.0)


def test_frequency_metadata_and_phasor_time_origin_are_explicit():
    phase_origin = np.array((1.0, 2.0, 3.0))
    series = make_surface_series(
        16,
        ETA_REF_M * (1.0 + 1.0j),
        frequency_hz=2.0,
        t_ref_s=0.25,
        phase_origin_m=phase_origin,
    )
    assert series.axis_kind == "frequency"
    assert series.t_ref_s == 0.25
    np.testing.assert_array_equal(series.phase_origin_m, phase_origin)
    assert series.fourier_convention == "Re[q_hat exp(-2pi i f(t-t_ref_s))]"

    times = np.arange(256) / 256.0
    q_hat = np.array((1.0, 1.0j))
    samples = phasor_samples(q_hat, 7.0, times, t_ref_s=0.0)
    fft_bin = np.fft.rfft(samples, axis=0)[7] / len(times)
    np.testing.assert_allclose(fft_bin, np.conjugate(q_hat) / 2.0, atol=2e-15)
    np.testing.assert_allclose(np.mean(samples**2, axis=0), 0.5, atol=2e-15)


def test_csd_propagation_matches_literal_real_and_complex_fixtures():
    h_real = np.array(((1.0, 2.0), (3.0, -1.0)))
    independent = np.diag((2.0, 3.0))
    np.testing.assert_allclose(
        propagate_csd(h_real, independent),
        np.array(((14.0, 0.0), (0.0, 21.0))),
        rtol=0.0,
        atol=0.0,
    )

    q_hat = np.array((1.0, 1.0j))
    line_csd = np.outer(q_hat, np.conjugate(q_hat)) / 2.0
    np.testing.assert_allclose(
        propagate_csd(h_real, line_csd),
        np.array(((2.5, 0.5 + 3.5j), (0.5 - 3.5j, 5.0))),
        rtol=0.0,
        atol=0.0,
    )

    h_complex = np.array(((1.0, 1.0j), (1.0, -1.0j)))
    np.testing.assert_allclose(
        propagate_csd(h_complex, independent),
        np.array(((5.0, -1.0), (-1.0, 5.0))),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(h_complex @ q_hat, np.array((0.0, 2.0)))
    np.testing.assert_allclose(
        propagate_csd(h_complex, line_csd),
        np.array(((0.0, 0.0), (0.0, 2.0))),
        rtol=0.0,
        atol=0.0,
    )


def test_csd_transfer_error_bound_uses_operator_norms():
    h_reference = np.array(((1.0, 2.0), (3.0, -1.0)))
    input_csd = np.diag((2.0, 3.0))
    e_h = 0.125
    expected = (
        2.0 * np.linalg.norm(h_reference, ord=2) * e_h + e_h**2
    ) * np.linalg.norm(input_csd, ord=2)
    assert csd_transfer_error_bound(h_reference, e_h, input_csd) == expected


def test_source_models_remain_blocked_without_imputation():
    assert SOURCE_MODEL_STATUS["status"] == "SOURCE_INPUT_BLOCKED"
    assert SOURCE_MODEL_STATUS["model_arrays_generated"] is False
    assert "model1_covariance" in SOURCE_MODEL_STATUS["missing_inputs"]
    assert "corrected_downstream_spectrum" in SOURCE_MODEL_STATUS["missing_inputs"]


@pytest.fixture(scope="module")
def deterministic_report():
    pytest.importorskip("scipy")
    return run_deterministic_validation()


@pytest.mark.slow
def test_surface_accuracy_full_gate(deterministic_report):
    report = deterministic_report
    assert report["surface_accuracy"]["status"] == "VERIFIED"
    assert report["surface_accuracy"]["adopted_n"] in (64, 128, 256)
    assert tuple(report["surface_accuracy"]["evaluated_n"])[:3] == (16, 32, 64)
    assert len(report["surface_accuracy"]["targets"]) == 4
    for target in report["surface_accuracy"]["targets"]:
        for observable in ("potential", "acceleration", "gradient"):
            block = target["observables"][observable]
            assert block["reference_status"] == "CONVERGED"
            assert block["norm_pass"]
            assert all(block["component_pass"])
            assert block["reference_evaluations"] <= 2_000_000
            assert block["reference_uncertainty"] <= block["norm_budget"] / 10.0
            assert "production_value" in block
            assert "reference_component_error" in block
            assert all("production_value" in item for item in block["trend"])


@pytest.mark.slow
def test_volume_amplitude_and_u_budget_full_gate(deterministic_report):
    report = deterministic_report
    amplitude = report["amplitude_convergence"]
    assert amplitude["status"] == "VERIFIED"
    assert tuple(amplitude["evaluated_alphas_m"]) == ALPHAS_M
    assert amplitude["adopted_m"] in (32, 64)
    assert amplitude["adopted_k"] in (16, 32)
    assert amplitude["adopted_alpha_m"] in ALPHAS_M
    assert len(amplitude["targets"]) == 4
    for target in amplitude["targets"]:
        for observable in ("potential", "acceleration", "gradient"):
            block = target["observables"][observable]
            assert block["status"] == "VERIFIED"
            assert block["failure_causes"] == []
            assert block["u_m_status"] == "VERIFIED"
            assert block["u_k_status"] == "VERIFIED"
            assert block["u_vol_status"] == "VERIFIED"
            assert block["roundoff_status"] == "VERIFIED"
            assert block["linearization_status"] == "WITHIN_TOTAL_BUDGET"
            assert block["u_ref"] <= block["budget"] / 10.0
            assert block["u_vol"] <= block["budget"] / 10.0
            assert block["u_round"] <= block["budget"] / 10.0
            assert block["total_error"] <= block["budget"]
            assert "q_plus" in block and "q_minus" in block
            assert "symmetric_derivative" in block
    first_alpha_failures = [
        cause
        for target in amplitude["alpha_records"][0]["targets"]
        for block in target["observables"].values()
        for cause in block["failure_causes"]
    ]
    assert "LINEARIZATION_NOT_RESOLVED" in first_alpha_failures
    diagnostics = amplitude["order_diagnostics"]
    assert diagnostics["required_for_acceptance"] is False
    assert diagnostics["measured_count"] > 0
    assert diagnostics["excluded_count"] >= 0


@pytest.mark.slow
def test_adaptive_reference_enforces_kernel_evaluation_cap():
    pytest.importorskip("scipy")
    from tests.oracles.water_surface_reference import (
        ReferenceEvaluationLimitError,
        adaptive_surface_reference,
    )

    with pytest.raises(ReferenceEvaluationLimitError, match="cap exceeded"):
        adaptive_surface_reference(
            TARGETS_M[0],
            "potential",
            tolerance=1e-10,
            evaluation_cap=1,
        )


@pytest.mark.slow
def test_common_spectra_and_fft_gate(deterministic_report):
    spectra = deterministic_report["common_spectra"]
    assert spectra["status"] == "VERIFIED"
    assert spectra["fft_positive_bin_matches_conjugate_phasor"]
    assert spectra["scipy_csd_imaginary_sign_matches"]
    assert spectra["literal_propagation_fixtures_pass"]
    assert spectra["transfer_source"] == "independent_continuous_reference"
    assert spectra["input_csd"]["value"] == [[1.0]]
    assert spectra["input_csd"]["eta_units"] == "m"
    assert spectra["input_csd"]["units"] == "m^2/Hz"
    assert spectra["input_csd"]["spectral_measure"] == "one_sided_Hz_density"
    assert spectra["literal_line_fixture"]["spectral_measure"] == (
        "bin_integrated_line_power"
    )
    assert spectra["scipy_csd_fixture"]["scaling"] == "spectrum"
    assert spectra["scipy_csd_fixture"]["spectral_measure"] == (
        "bin_integrated_line_power"
    )
    for observable, component_count in (
        ("potential", 1),
        ("acceleration", 3),
        ("gradient", 9),
    ):
        block = spectra["single_basis_output_csd"][observable]
        assert block["shape"] == [4, component_count, component_count]
        assert block["hermitian"]
        assert block["psd_within_roundoff"]
        assert max(block["numerical_rank_by_target"]) <= 1
        assert len(block["rank_tolerance_by_target"]) == len(TARGETS_M)
        assert block["fixture_rank_limit"] == 1
        assert (
            block["output_units"]
            == {
                "potential": "m^4/s^4/Hz",
                "acceleration": "m^2/s^4/Hz",
                "gradient": "1/s^4/Hz",
            }[observable]
        )
    assert (
        spectra["structural_rank_limits"]["general_gradient_stf_single_point_support"]
        == 5
    )


@pytest.mark.slow
def test_raw_report_declares_observable_transfer_and_csd_units(deterministic_report):
    units = deterministic_report["units"]
    assert units["observable_q"] == {
        "potential": "m^2/s^2",
        "acceleration": "m/s^2",
        "gradient": "1/s^2",
    }
    assert units["transfer_q_per_eta"] == {
        "potential": "m/s^2",
        "acceleration": "1/s^2",
        "gradient": "1/(m*s^2)",
    }
    assert units["input_csd"] == "m^2/Hz"
    assert units["output_csd"] == {
        "potential": "m^4/s^4/Hz",
        "acceleration": "m^2/s^4/Hz",
        "gradient": "1/s^4/Hz",
    }
    assert units["gradient_component_order"] == list(GRADIENT_COMPONENT_ORDER)


def test_top_level_records_reference_cap_failure(monkeypatch, tmp_path):
    oracle = _oracle_module()

    def fail_reference(*_args, **_kwargs):
        raise oracle.ReferenceEvaluationLimitError("injected cap failure")

    monkeypatch.setattr(oracle, "adaptive_surface_reference_pair", fail_reference)
    result_path, manifest_path = write_validation(tmp_path)
    report = json.loads(result_path.read_text())
    assert manifest_path.exists()
    assert report["overall_executable_status"] == "FAILED"
    assert report["surface_accuracy"]["status"] == "REFERENCE_NOT_CONVERGED"
    assert report["surface_accuracy"]["failure_causes"] == [
        "REFERENCE_EVALUATION_CAP_EXCEEDED"
    ]
    assert report["amplitude_convergence"]["status"] == (
        "NOT_RUN_REFERENCE_NOT_CONVERGED"
    )
    assert report["common_spectra"]["status"] == "NOT_RUN_REFERENCE_NOT_CONVERGED"


@pytest.mark.parametrize(
    ("failed_gate", "expected_cause"),
    [
        ("reference", "REFERENCE_NOT_CONVERGED"),
        ("quadrature", "QUADRATURE_U_VOL_NOT_CONVERGED"),
        ("roundoff", "ROUNDOFF_NOT_RESOLVED"),
    ],
)
def test_volume_block_does_not_assess_linearization_after_numerical_failure(
    failed_gate, expected_cause
):
    flags = {
        "reference_ok": True,
        "u_m_ok": True,
        "u_k_ok": True,
        "u_vol_ok": True,
        "roundoff_ok": True,
    }
    if failed_gate == "reference":
        flags["reference_ok"] = False
    elif failed_gate == "quadrature":
        flags["u_vol_ok"] = False
    else:
        flags["roundoff_ok"] = False
    classification = _classify_volume_block(
        **flags,
        total_error=2.0,
        budget=1.0,
    )
    assert expected_cause in classification["failure_causes"]
    assert classification["linearization_status"] == "NOT_ASSESSED"
    assert "LINEARIZATION_NOT_RESOLVED" not in classification["failure_causes"]


def test_no_passing_alpha_uses_neutral_aggregate_status_for_numerical_failure():
    records = [
        {
            "alpha_m": 0.005,
            "targets": [
                {
                    "observables": {
                        "potential": {
                            "pass": False,
                            "failure_causes": ["REFERENCE_NOT_CONVERGED"],
                        }
                    }
                }
            ],
        }
    ]
    adopted, status = _select_amplitude_record(records)
    assert adopted is records[-1]
    assert status == "NOT_VERIFIED"


@pytest.mark.slow
def test_direct_script_entrypoint_loads_independent_oracle(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "research_water_surface_models.py"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads((tmp_path / "water-surface-results.json").read_text())
    manifest = json.loads(
        (tmp_path / "water-surface-reproduction-manifest.json").read_text()
    )
    assert result["overall_executable_status"] == "VERIFIED"
    assert manifest["result"]["path"].endswith("water-surface-results.json")
