"""Tests for faithful, additive frozen refinement trace replay."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")

SCRIPT = Path(__file__).with_name("record_frozen_refinement_traces.py")
SPEC = importlib.util.spec_from_file_location("gen1_refinement_trace_recorder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RECORDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECORDER)


def _plane_row():
    fixtures = RECORDER.oracle_module("noncavity_fixtures").noncavity_fixtures()
    fixture = next(item for item in fixtures if item["family"] == "temperature_plane")
    reference, metadata = RECORDER.oracle_module(
        "plane_static_reference"
    ).static_plane_reference(fixture)
    return {
        "fixture": RECORDER._jsonable(fixture),
        "reference": RECORDER._jsonable(reference),
        "metadata": RECORDER._jsonable(metadata),
    }


def test_capture_preserves_mp_mantissas_and_explicit_decimal_digits() -> None:
    with mp.workdps(100):
        value = mp.mpf("1.23456789012345678901234567890123456789e-73")
        captured = RECORDER._capture_number(value, 100)

    assert captured["mpf"] == list(value._mpf_)
    assert len(captured["decimal"].replace(".", "").replace("e", "")) > 80


def test_captured_numpy_longdouble_restores_dtype_before_float64_rounding() -> None:
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("platform longdouble does not exceed float64 precision")

    lower = np.float64("1.000000000000001")
    upper = np.nextafter(lower, np.inf)
    original = (np.longdouble(lower) + np.longdouble(upper)) / np.longdouble(2)
    captured = RECORDER._capture_number(original, 80)

    assert captured["dtype"] == str(np.dtype(np.longdouble))
    restored = np.longdouble(captured["decimal"])
    assert restored == original
    assert float(captured["decimal"]) != float(original)
    assert RECORDER._captured_real(captured, 60, "gradient") == float(original)


def test_change_trend_does_not_treat_numeric_zero_as_a_certificate() -> None:
    with mp.workdps(80):
        unresolved = RECORDER._change_trend(mp.mpf(0), mp.mpf(0), None)
        explained = RECORDER._change_trend(
            mp.mpf("1e-30"), mp.mpf("1e-40"), mp.mpf("1e-25")
        )

    assert unresolved["classification"] == "unresolved_zero_change"
    assert explained["classification"] == "decreasing"


def test_original_plane_entrypoint_replays_metadata_and_records_captured_steps() -> (
    None
):
    row = _plane_row()
    module = RECORDER.oracle_module("plane_static_reference")
    original_callback = module.observable_values

    trace = RECORDER.record_row("plane", row)

    assert module.observable_values is original_callback
    assert trace["reproduction"]["passed"] is True, trace["reproduction"]
    assert trace["original_entrypoint"]["capture_call_count"] == 4
    calls = trace["original_entrypoint"]["captured_calls"]
    assert all(call["kernel"] == "plane_observable_values" for call in calls)
    assert all(call["inputs"]["step_m"] for call in calls)
    assert (
        trace["observable_traces"]["potential"]["derivative_trend"]["classification"]
        == "not_applicable"
    )
    assert (
        trace["observable_traces"]["potential"]["quadrature_trend"]["classification"]
        == "not_applicable"
    )
    assert trace["observable_traces"]["acceleration"]["frozen_component_checks"][
        "derivative"
    ]["matches_frozen_float"]


def test_noncavity_point_force_retains_temporal_quadrature_refinement() -> None:
    fixtures = RECORDER.oracle_module("noncavity_fixtures").noncavity_fixtures()
    fixture = next(item for item in fixtures if item["family"] == "point_force")
    reference, metadata = RECORDER.oracle_module(
        "noncavity_reference"
    ).noncavity_reference(fixture)
    row = {
        "fixture": RECORDER._jsonable(fixture),
        "reference": RECORDER._jsonable(reference),
        "metadata": RECORDER._jsonable(metadata),
    }

    trace = RECORDER.record_row("noncavity", row, source_row_index=0)

    assert trace["reproduction"]["passed"] is True
    RECORDER._validate_trace_record(trace, "noncavity", 0, row)
    potential = trace["observable_traces"]["potential"]
    assert potential["derivative_trend"]["classification"] == "not_applicable"
    assert potential["evaluation_inputs"]["quadrature_orders"] == [16, 24, 32]
    assert potential["quadrature_differences_si"] is not None
    assert potential["quadrature_trend"]["classification"] != "not_applicable"
    assert potential["frozen_component_checks"]["quadrature"]["matches_frozen_float"]
    assert trace["original_entrypoint"]["actual_reference"] == row["reference"]


def test_schema_record_is_json_serializable_without_touching_frozen_row() -> None:
    row = _plane_row()
    before = json.dumps(row, sort_keys=True)

    record = RECORDER.record_row("plane", row)
    encoded = json.dumps(record, allow_nan=False)

    assert json.loads(encoded)["schema"] == RECORDER.SCHEMA
    assert json.dumps(row, sort_keys=True) == before


def test_source_surface_callbacks_are_captured_and_replayed_without_rebuilding_u() -> (
    None
):
    fixture = next(
        item
        for item in RECORDER.oracle_module("source_fixtures").source_fixtures()
        if item["family"] == "finite_temperature" and item["shape"] == "cuboid"
    )
    reference, metadata = RECORDER.oracle_module("source_reference").source_reference(
        fixture
    )
    row = {
        "fixture": RECORDER._jsonable(fixture),
        "reference": RECORDER._jsonable(reference),
        "metadata": RECORDER._jsonable(metadata),
    }

    trace = RECORDER.record_row("source", row, source_row_index=0)

    assert trace["reproduction"]["passed"] is True
    RECORDER._validate_trace_record(trace, "source", 0, row)
    assert trace["original_entrypoint"]["capture_call_count"] == 3
    assert all(
        call["kernel"] == "source_surface_values"
        for call in trace["original_entrypoint"]["captured_calls"]
    )
    assert trace["observable_traces"]["potential"]["quadrature_trend"][
        "classification"
    ] in {
        "decreasing",
        "nondecreasing_explained_by_independent_floor",
        "unresolved_zero_change",
        "unresolved_nondecreasing",
    }


def test_all_pinned_oracle_module_hashes_match_the_copied_sources() -> None:
    hashes = RECORDER._module_hashes()

    assert hashes == RECORDER.EXPECTED_ORACLE_SHA256


def test_dataset_selector_pins_accepted_frozen_sphere_and_plane_sets() -> None:
    assert RECORDER.DATASETS["sphere"] == "sphere-three-quadrature-20261001"
    assert RECORDER.EXPECTED_REFERENCE_SHA256["sphere"] == (
        "a5da1f55c0c91e84c1241d00325354af1aa79acd4f13258ee9984ef857ff2783"
    )
    assert RECORDER.DATASETS["plane"] == "plane-static-frozen-20261001"
    assert RECORDER.EXPECTED_REFERENCE_SHA256["plane"] == (
        "8076a91f1f7462770518b717fb3a204ee4be29e0906a618658aff132c97b063e"
    )


def test_resume_rejects_a_changed_observer_hash(tmp_path: Path) -> None:
    upstream_path = tmp_path / "accepted-manifest.json"
    upstream_path.write_text("{}\n", encoding="utf-8")
    manifest = {
        "frozen_reference_sha256": "reference",
        "observer_sha256": "observer-current",
        "oracle_module_sha256": {"oracle.py": "oracle"},
        "zero_certificate_sha256": "zero-certificates",
        "polynomial_certificate_sha256": "polynomial-certificates",
        "upstream_provenance": {
            "manifest": str(upstream_path),
            "manifest_sha256": RECORDER._sha256(upstream_path),
        },
    }
    checkpoint = {
        **manifest,
        "dataset_manifest_pin_sha256": RECORDER._manifest_pin_sha256(manifest),
    }

    with pytest.raises(RuntimeError, match="observer_sha256 changed; use a new run id"):
        RECORDER._validate_resume_pins(
            checkpoint,
            manifest,
            reference_sha256="reference",
            observer_sha256="observer-updated",
            oracle_hashes={"oracle.py": "oracle"},
            zero_certificate_sha256="zero-certificates",
            polynomial_certificate_sha256="polynomial-certificates",
        )

    with pytest.raises(
        RuntimeError,
        match="polynomial_certificate_sha256 changed; use a new run id",
    ):
        RECORDER._validate_resume_pins(
            checkpoint,
            manifest,
            reference_sha256="reference",
            observer_sha256="observer-current",
            oracle_hashes={"oracle.py": "oracle"},
            zero_certificate_sha256="zero-certificates",
            polynomial_certificate_sha256="updated-polynomial-certificates",
        )


def test_production_import_status_comes_from_loaded_modules(monkeypatch) -> None:
    unexpected_name = "gragra._refinement_observer_test_marker"
    monkeypatch.setitem(sys.modules, unexpected_name, types.ModuleType(unexpected_name))

    loaded = RECORDER._loaded_production_modules()
    state = RECORDER._process_independence(loaded)

    assert unexpected_name in loaded
    assert state["production_imported"] is True
    assert unexpected_name in state["production_module_names"]
    assert state["independence_check_passed"] is False


def test_worker_records_production_modules_loaded_during_replay(monkeypatch) -> None:
    unexpected_name = "gragra._refinement_worker_test_marker"

    def replay_with_unexpected_import(dataset, _row, **_kwargs):
        monkeypatch.setitem(
            sys.modules, unexpected_name, types.ModuleType(unexpected_name)
        )
        return {"schema": RECORDER.SCHEMA, "dataset": dataset, "status": "reproduced"}

    monkeypatch.setattr(RECORDER, "record_row", replay_with_unexpected_import)
    _, record, error = RECORDER._worker(("sphere", 0, {"fixture": {}}))

    assert error is None
    assert unexpected_name in record["process_independence"]["modules_after"]
    assert record["process_independence"]["production_imported"] is True
    assert record["process_independence"]["independence_check_passed"] is False


def test_floor_allowlist_excludes_truncation_and_crosscheck_discrepancies() -> None:
    uncertainty = {
        "components": [
            ["precision_propagation", "1e-20"],
            ["float_conversion", "2e-20"],
            ["explicit_truncation_tail", "1e-8"],
            ["alternate_potential_quadrature", "1e-7"],
            ["independent_potential_quadrature", "1e-6"],
            ["derivative_refinement", "1e-12"],
            ["quadrature_refinement", "1e-12"],
        ]
    }

    candidates = RECORDER._floor_candidates(uncertainty, observable="potential")
    with mp.workdps(80):
        trend = RECORDER._change_trend(mp.mpf("1e-12"), mp.mpf("2e-12"), None)

    assert set(candidates) == {"precision_propagation", "float_conversion"}
    assert trend["classification"] == "unresolved_nondecreasing"


def test_zero_certificate_requires_full_identity() -> None:
    fixture = {"family": "gaussian", "target_m": [0.0, 0.0, 0.0]}
    uncertainty = {
        "uncertainty_kind": "estimate",
        "used_uncertainty_si": 0.0,
    }
    observable_record = {
        "uncertainty": uncertainty,
        "values_si": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    }
    certificate = {
        "dataset": RECORDER.DATASETS["noncavity"],
        "fixture": fixture,
        "observable": "acceleration",
        "certificate": "even radial center gradient",
        "original_uncertainty_kind": "estimate",
        "original_used_uncertainty_si": 0.0,
        "frozen_labels_and_values_unchanged": True,
    }

    matched = RECORDER._matching_zero_certificate(
        "noncavity", fixture, "acceleration", observable_record, [certificate]
    )
    changed_fixture = {**fixture, "target_m": [0.0, 1e-100, 0.0]}
    wrong_fixture = RECORDER._matching_zero_certificate(
        "noncavity",
        changed_fixture,
        "acceleration",
        observable_record,
        [certificate],
    )
    nonzero_record = {
        **observable_record,
        "values_si": [[1e-300, 0.0], [0.0, 0.0], [0.0, 0.0]],
    }
    no_zero = RECORDER._matching_zero_certificate(
        "noncavity", fixture, "acceleration", nonzero_record, [certificate]
    )

    assert matched == certificate
    assert wrong_fixture is None
    assert no_zero is None


def _polynomial_certificate(dataset, source_row_index, fixture):
    if dataset == "noncavity":
        certificate_kind = "retarded_cubic_pulse_polynomial_panels"
    else:
        certificate_kind = "uniform_Sphere_radial_polynomial_moments"
    return {
        "dataset": dataset,
        "source_row_index": source_row_index,
        "fixture": fixture,
        "certificate_kind": certificate_kind,
        "covered_observables": ["potential", "acceleration", "gradient"],
        "covered_original_stencils": (
            "all original loop steps and axis/diagonal seven-point evaluations"
        ),
        "certified_error": "zero quadrature truncation in ideal real arithmetic only",
        "frozen_values_uncertainty_labels_unchanged": True,
    }


def test_polynomial_certificate_requires_dataset_index_fixture_and_observable() -> None:
    force_fixture = {
        "family": "point_force",
        "target_m": [1.2, 0.3, -0.4],
        "elapsed_s": 0.0,
        "vp_m_s": 2.3,
        "direction": [1.0, 0.0, 0.0],
        "knots_s": [0.0, 1.0, 2.0],
        "coefficients_N": [[0.0, 0.0, 3.0, -2.0], [1.0, 0.0, -3.0, 2.0]],
    }
    certificate = _polynomial_certificate("noncavity", 19, force_fixture)

    matched = RECORDER._matching_polynomial_certificate(
        "noncavity", 19, force_fixture, "potential", [certificate]
    )
    wrong_index = RECORDER._matching_polynomial_certificate(
        "noncavity", 20, force_fixture, "potential", [certificate]
    )
    outside_certificate_scope = RECORDER._matching_polynomial_certificate(
        "noncavity", 46, force_fixture, "potential", [certificate]
    )
    changed_fixture = {**force_fixture, "elapsed_s": 0.1}
    wrong_fixture = RECORDER._matching_polynomial_certificate(
        "noncavity", 19, changed_fixture, "potential", [certificate]
    )
    wrong_dataset = RECORDER._matching_polynomial_certificate(
        "source", 19, force_fixture, "potential", [certificate]
    )
    wrong_observable = RECORDER._matching_polynomial_certificate(
        "noncavity", 19, force_fixture, "temperature", [certificate]
    )
    changed_certificate = {
        **certificate,
        "fixture": {**force_fixture, "elapsed_s": 0.1},
    }
    wrong_certificate_fixture = RECORDER._matching_polynomial_certificate(
        "noncavity", 19, force_fixture, "potential", [changed_certificate]
    )

    assert matched == certificate
    assert wrong_index is None
    assert outside_certificate_scope is None
    assert wrong_fixture is None
    assert wrong_dataset is None
    assert wrong_observable is None
    assert wrong_certificate_fixture is None

    sphere_fixture = {
        "family": "finite_temperature",
        "shape": "sphere",
        "geometry": {"radius_m": 1.1},
        "target_m": [0.0, 0.0, 0.0],
    }
    sphere_certificate = _polynomial_certificate("source", 2, sphere_fixture)
    assert (
        RECORDER._matching_polynomial_certificate(
            "source", 2, sphere_fixture, "gradient", [sphere_certificate]
        )
        == sphere_certificate
    )
    assert (
        RECORDER._matching_polynomial_certificate(
            "source", 6, sphere_fixture, "gradient", [sphere_certificate]
        )
        is None
    )


def test_polynomial_certificate_adjudicates_only_zero_zero_quadrature_trend(
    monkeypatch,
) -> None:
    fixture = {"family": "point_force", "target_m": [1.2, 0.3, -0.4]}
    certificate = _polynomial_certificate("noncavity", 19, fixture)
    values = [[1.0, 0.0]]
    reference = {
        name: {
            "values_si": values,
            "uncertainty": {
                "uncertainty_kind": "estimate",
                "used_uncertainty_si": 1.0,
                "method": "pinned estimate",
                "components": [["quadrature_refinement", 0.0]],
            },
        }
        for name in RECORDER.OBSERVABLE_SLICES
    }
    row = {"fixture": fixture, "reference": reference, "metadata": {}}

    def zero_trace(*_args, **_kwargs):
        return {
            "status": "numerical",
            "route": "test-mpmath",
            "evaluation_inputs": {"quadrature_orders": [16, 24, 32]},
            "quadrature_trend": {
                "classification": "unresolved_zero_change",
                "nondecreasing": True,
                "coarse_to_middle_si": "0.0",
                "middle_to_fine_si": "0.0",
                "independent_floor_si": None,
            },
            "quadrature_differences_si": {
                "coarse_to_middle": "0.0",
                "middle_to_fine": "0.0",
            },
            "derivative_trend": {
                "classification": "not_applicable",
                "reason": "test observable has no derivative refinement",
            },
            "frozen_component_checks": {
                "derivative": None,
                "quadrature": {
                    "component_name": "quadrature_refinement",
                    "formula_float_si": 0.0,
                    "formula_value_si": "0.0",
                    "frozen_component_si": 0.0,
                    "matches_frozen_float": True,
                },
            },
            "captured_final_values_match_frozen": True,
            "captured_final_values_si": RECORDER._capture_number(mp.mpc(1, 0), 60),
        }

    monkeypatch.setattr(RECORDER, "_mp_trace_for_observable", zero_trace)
    record = RECORDER._case_trace(
        "noncavity",
        row,
        [{"kernel": "test", "inputs": {}, "output": [1.0]}],
        reference,
        {},
        source_row_index=19,
        polynomial_certificates=[certificate],
        polynomial_certificate_source="pinned-certificate.json",
        polynomial_certificate_sha256=RECORDER.EXPECTED_POLYNOMIAL_CERTIFICATES_SHA256,
        polynomial_verifier_script_sha256=(
            RECORDER.EXPECTED_POLYNOMIAL_VERIFIER_SCRIPT_SHA256
        ),
    )

    potential = record["observable_traces"]["potential"]
    adjudication = potential["polynomial_quadrature_certificate_adjudication"]
    assert potential["quadrature_trend"]["classification"] == "unresolved_zero_change"
    assert potential["quadrature_differences_si"] == {
        "coarse_to_middle": "0.0",
        "middle_to_fine": "0.0",
    }
    assert adjudication["raw_quadrature_trend_classification"] == (
        "unresolved_zero_change"
    )
    assert adjudication["certified_error"] == (
        "zero quadrature truncation in ideal real arithmetic only"
    )
    assert adjudication["temporal_quadrature_remains_applicable"] is True
    assert adjudication["total_numerical_error_certified"] is False
    assert adjudication["supplement_sha256"] == (
        RECORDER.EXPECTED_POLYNOMIAL_CERTIFICATES_SHA256
    )
    assert record["status"] == "reproduced"
    assert record["raw_unresolved_trends"] == [
        f"{name}/quadrature_trend" for name in RECORDER.OBSERVABLE_SLICES
    ]
    assert record["unresolved_trends"] == []
    RECORDER._validate_trace_record(
        record,
        "noncavity",
        19,
        row,
        polynomial_certificates=[certificate],
        polynomial_certificate_sha256=RECORDER.EXPECTED_POLYNOMIAL_CERTIFICATES_SHA256,
    )

    unreproduced = json.loads(json.dumps(record))
    unreproduced["original_entrypoint"]["actual_reference"]["potential"]["values_si"][
        0
    ][0] = 2.0
    unreproduced["reproduction"][
        "reference_matches_full_values_uncertainty_methods_components"
    ] = False
    unreproduced["reproduction"]["passed"] = False
    unreproduced["reproduction"]["failed_checks"] = [
        "actual original-entrypoint values/uncertainties differ from frozen reference"
    ]
    unreproduced["status"] = "replay_mismatch"
    with pytest.raises(ValueError, match="unreproduced polynomial row"):
        RECORDER._validate_trace_record(
            unreproduced,
            "noncavity",
            19,
            row,
            polynomial_certificates=[certificate],
            polynomial_certificate_sha256=(
                RECORDER.EXPECTED_POLYNOMIAL_CERTIFICATES_SHA256
            ),
        )


def test_polynomial_certificate_does_not_adjudicate_nonzero_trend(monkeypatch) -> None:
    fixture = {"family": "point_force", "target_m": [1.2, 0.3, -0.4]}
    certificate = _polynomial_certificate("noncavity", 19, fixture)
    values = [[1.0, 0.0]]
    reference = {
        name: {
            "values_si": values,
            "uncertainty": {
                "uncertainty_kind": "estimate",
                "used_uncertainty_si": 1.0,
                "method": "pinned estimate",
                "components": [],
            },
        }
        for name in RECORDER.OBSERVABLE_SLICES
    }
    row = {"fixture": fixture, "reference": reference, "metadata": {}}

    def increasing_trace(*_args, **_kwargs):
        return {
            "status": "numerical",
            "quadrature_trend": {
                "classification": "unresolved_nondecreasing",
                "nondecreasing": True,
                "coarse_to_middle_si": "1.0",
                "middle_to_fine_si": "2.0",
                "independent_floor_si": None,
            },
            "quadrature_differences_si": {
                "coarse_to_middle": "1.0",
                "middle_to_fine": "2.0",
            },
            "derivative_trend": {"classification": "not_applicable"},
            "frozen_component_checks": {},
            "captured_final_values_match_frozen": True,
            "captured_final_values_si": values,
        }

    monkeypatch.setattr(RECORDER, "_mp_trace_for_observable", increasing_trace)
    record = RECORDER._case_trace(
        "noncavity",
        row,
        [],
        reference,
        {},
        source_row_index=19,
        polynomial_certificates=[certificate],
        polynomial_certificate_source="pinned-certificate.json",
        polynomial_certificate_sha256=RECORDER.EXPECTED_POLYNOMIAL_CERTIFICATES_SHA256,
        polynomial_verifier_script_sha256=(
            RECORDER.EXPECTED_POLYNOMIAL_VERIFIER_SCRIPT_SHA256
        ),
    )

    potential = record["observable_traces"]["potential"]
    assert "polynomial_quadrature_certificate_adjudication" not in potential
    assert potential["quadrature_differences_si"]["middle_to_fine"] == "2.0"
    assert record["unresolved_trends"] == [
        f"{name}/quadrature_trend" for name in RECORDER.OBSERVABLE_SLICES
    ]


def _indexed_plane_trace():
    row = _plane_row()
    record = RECORDER.record_row("plane", row)
    record["source_row_index"] = 0
    record["process_independence"] = _worker_evidence()
    return row, record


def _trace_bytes(*records):
    return b"".join(
        (json.dumps(record, allow_nan=False, sort_keys=True) + "\n").encode()
        for record in records
    )


def _checkpoint_for_trace(data, *, status, completed, selected):
    digest = RECORDER._sha256_bytes(data)
    checkpoint = {
        "status": status,
        "selected_source_row_indices": selected,
        "completed_source_row_indices": completed,
        "committed_trace_bytes": len(data),
        "committed_trace_sha256": digest,
    }
    if status != "running":
        records = RECORDER._decode_trace_bytes(data, description="test trace")
        mismatch_count = sum(
            record.get("status") in {"replay_mismatch", "replay_error"}
            for record in records
        )
        unresolved_count = sum(
            bool(record.get("unresolved_trends")) for record in records
        )
        process_names = sorted(
            {
                name
                for record in records
                for name in record["process_independence"]["production_module_names"]
            }
        )
        checkpoint["trace_bytes"] = len(data)
        checkpoint["traces_jsonl_sha256"] = digest
        checkpoint["record_count"] = len(completed)
        checkpoint["case_count"] = len(selected)
        checkpoint["mismatch_count"] = mismatch_count
        checkpoint["unresolved_trend_case_count"] = unresolved_count
        checkpoint["production_module_names"] = process_names
        checkpoint["production_imported"] = bool(process_names)
        checkpoint["independence_check_passed"] = not process_names
    return checkpoint


def test_resume_rejects_deleted_committed_trace_row(tmp_path: Path) -> None:
    row, record = _indexed_plane_trace()
    original_bytes = _trace_bytes(record)
    path = tmp_path / "traces.jsonl"
    path.write_bytes(b"")
    checkpoint = _checkpoint_for_trace(
        original_bytes, status="complete", completed=[0], selected=[0]
    )

    with pytest.raises(RuntimeError, match="shorter than committed trace prefix"):
        RECORDER._verify_trace_file(
            path,
            checkpoint,
            dataset="plane",
            selected_rows={0: row},
            selected_indices=[0],
        )


def test_resume_rejects_altered_committed_returned_value(tmp_path: Path) -> None:
    row, record = _indexed_plane_trace()
    original_bytes = _trace_bytes(record)
    changed = json.loads(original_bytes)
    changed["original_entrypoint"]["actual_reference"]["potential"]["values_si"][0][
        0
    ] += 1.0
    path = tmp_path / "traces.jsonl"
    path.write_bytes(_trace_bytes(changed))
    checkpoint = _checkpoint_for_trace(
        original_bytes, status="complete", completed=[0], selected=[0]
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "shorter than committed trace prefix|"
            "committed trace prefix SHA-256 mismatch"
        ),
    ):
        RECORDER._verify_trace_file(
            path,
            checkpoint,
            dataset="plane",
            selected_rows={0: row},
            selected_indices=[0],
        )
    with pytest.raises(ValueError, match="inconsistent value-match flag"):
        RECORDER._validate_trace_record(changed, "plane", 0, row)


def test_resume_rejects_duplicate_and_out_of_range_trace_rows(tmp_path: Path) -> None:
    row, record = _indexed_plane_trace()
    prefix = _trace_bytes(record)
    duplicate_path = tmp_path / "duplicate.jsonl"
    duplicate_path.write_bytes(prefix + prefix)
    checkpoint = _checkpoint_for_trace(
        prefix + prefix, status="running", completed=[0], selected=[0, 1]
    )

    with pytest.raises(ValueError, match="duplicate trace source row index"):
        RECORDER._verify_trace_file(
            duplicate_path,
            checkpoint,
            dataset="plane",
            selected_rows={0: row, 1: row},
            selected_indices=[0, 1],
        )

    with pytest.raises(
        ValueError, match="selected source row indices contain duplicates"
    ):
        RECORDER._validated_selected_indices([0, 0], {0})
    with pytest.raises(ValueError, match="outside the frozen reference"):
        RECORDER._validated_selected_indices([2], {0, 1})

    out_of_range = json.loads(prefix)
    out_of_range["source_row_index"] = 2
    out_of_range["dataset"] = "plane"
    out_of_range["fixture"] = row["fixture"]
    out_of_range["frozen_reference"] = {
        "reference": row["reference"],
        "metadata": row["metadata"],
    }
    tail_path = tmp_path / "out-of-range.jsonl"
    tail_path.write_bytes(prefix + _trace_bytes(out_of_range))
    tail_checkpoint = _checkpoint_for_trace(
        prefix, status="running", completed=[0], selected=[0, 1]
    )
    state = RECORDER._verify_trace_file(
        tail_path,
        tail_checkpoint,
        dataset="plane",
        selected_rows={0: row, 1: row},
        selected_indices=[0, 1],
    )
    assert state["completed_source_row_indices"] == {0}
    assert state["uncommitted_tail_bytes"] == _trace_bytes(out_of_range)


def test_resume_rejects_a_trace_with_different_fixture_identity() -> None:
    row, record = _indexed_plane_trace()
    altered = json.loads(json.dumps(record))
    altered["fixture"]["target_m"][0] += 1e-100

    with pytest.raises(ValueError, match="fixture identity differs"):
        RECORDER._validate_trace_record(altered, "plane", 0, row)


def test_resume_recovers_one_flushed_row_missing_from_checkpoint(
    tmp_path: Path,
) -> None:
    row, record = _indexed_plane_trace()
    path = tmp_path / "traces.jsonl"
    path.write_bytes(_trace_bytes(record))
    checkpoint = _checkpoint_for_trace(
        b"", status="running", completed=[], selected=[0]
    )

    state = RECORDER._verify_trace_file(
        path,
        checkpoint,
        dataset="plane",
        selected_rows={0: row},
        selected_indices=[0],
    )

    assert state["completed_source_row_indices"] == set()
    assert state["recovered_source_row_indices"] == set()
    assert state["records"] == []
    assert state["uncommitted_tail_bytes"] == _trace_bytes(record)
    assert state["trace_sha256"] == RECORDER._sha256_bytes(path.read_bytes())


def test_trace_validator_requires_all_observables_and_capture_evidence() -> None:
    row, record = _indexed_plane_trace()
    missing_observable = json.loads(json.dumps(record))
    del missing_observable["observable_traces"]["gradient"]
    with pytest.raises(ValueError, match="exactly the frozen observable set"):
        RECORDER._validate_trace_record(missing_observable, "plane", 0, row)

    wrong_capture_count = json.loads(json.dumps(record))
    wrong_capture_count["original_entrypoint"]["capture_call_count"] += 1
    with pytest.raises(ValueError, match="capture count differs"):
        RECORDER._validate_trace_record(wrong_capture_count, "plane", 0, row)

    missing_capture_payload = json.loads(json.dumps(record))
    missing_capture_payload["original_entrypoint"]["captured_calls"][0].pop("output")
    with pytest.raises(ValueError, match="malformed captured call"):
        RECORDER._validate_trace_record(missing_capture_payload, "plane", 0, row)


def test_trace_validator_rejects_missing_worker_independence_proof() -> None:
    row, record = _indexed_plane_trace()
    record.pop("process_independence")

    with pytest.raises(ValueError, match="missing worker process evidence"):
        RECORDER._validate_trace_record(
            record, "plane", 0, row, require_worker_evidence=True
        )


def test_replay_error_keeps_a_consistent_capture_list_and_worker_proof() -> None:
    row, record = _indexed_plane_trace()
    for key in (
        "original_entrypoint",
        "reproduction",
        "observable_traces",
        "raw_unresolved_trends",
        "unresolved_trends",
    ):
        record.pop(key, None)
    record["status"] = "replay_error"
    record["error"] = "RuntimeError: injected failure"
    record["captured_calls_before_error"] = []
    record["capture_call_count_before_error"] = 0
    RECORDER._validate_trace_record(
        record, "plane", 0, row, require_worker_evidence=True
    )

    record["capture_call_count_before_error"] = 1
    with pytest.raises(ValueError, match="capture count differs"):
        RECORDER._validate_trace_record(
            record, "plane", 0, row, require_worker_evidence=True
        )


def test_trace_validator_rejects_reproduced_status_with_honest_output_mismatch() -> (
    None
):
    row, record = _indexed_plane_trace()
    record["original_entrypoint"]["actual_reference"]["potential"]["values_si"][0][
        0
    ] += 1.0
    record["reproduction"][
        "reference_matches_full_values_uncertainty_methods_components"
    ] = False
    record["reproduction"]["passed"] = False
    record["reproduction"]["failed_checks"] = [
        "actual original-entrypoint values/uncertainties differ from frozen reference"
    ]

    with pytest.raises(
        ValueError,
        match="status is inconsistent with replay evidence",
    ):
        RECORDER._validate_trace_record(record, "plane", 0, row)


def _worker_evidence():
    return {
        "pid": os.getpid(),
        "modules_before": [],
        "modules_after": [],
        "production_imported": False,
        "production_module_names": [],
        "independence_check_passed": True,
    }


def _temporary_plane_runner(monkeypatch, tmp_path: Path):
    row = _plane_row()
    reference_path = tmp_path / "plane.jsonl"
    reference_path.write_text(
        json.dumps(row, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    upstream_manifest = tmp_path / "upstream-manifest.json"
    upstream_manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setitem(
        RECORDER.EXPECTED_REFERENCE_SHA256, "plane", RECORDER._sha256(reference_path)
    )
    monkeypatch.setattr(
        RECORDER,
        "_dataset_paths",
        lambda _root, _dataset: (reference_path, upstream_manifest),
    )
    monkeypatch.setattr(
        RECORDER, "_module_hashes", lambda: RECORDER.EXPECTED_ORACLE_SHA256
    )
    monkeypatch.setattr(RECORDER, "_loaded_production_modules", lambda: [])
    monkeypatch.setattr(
        RECORDER,
        "_load_zero_certificates",
        lambda _root: {
            "path": str(tmp_path / "zero-certificates.json"),
            "sha256": "zero-pin",
            "data": {"classification": "test fixture", "certificates": []},
        },
    )
    monkeypatch.setattr(
        RECORDER, "_load_polynomial_quadrature_certificates", lambda *_args: None
    )
    valid_record = RECORDER.record_row(
        "plane",
        row,
        source_row_index=0,
        zero_certificates=(),
        zero_certificate_sha256="zero-pin",
    )
    valid_record["process_independence"] = _worker_evidence()
    worker_calls = []

    def fake_worker(payload):
        index = payload[1]
        worker_calls.append(index)
        record = json.loads(json.dumps(valid_record))
        record["source_row_index"] = index
        record["process_independence"] = _worker_evidence()
        return index, record, None

    monkeypatch.setattr(RECORDER, "_worker", fake_worker)
    return row, valid_record, worker_calls


@pytest.mark.parametrize(
    "tail_mutation",
    [
        "valid_row",
        "missing_observables",
        "missing_worker_proof",
        "bad_status",
        "partial",
    ],
)
def test_resume_quarantines_uncommitted_tail_and_recomputes_pending_row(
    monkeypatch, tmp_path: Path, tail_mutation: str
) -> None:
    _row, valid_record, worker_calls = _temporary_plane_runner(monkeypatch, tmp_path)
    output_root = tmp_path / "runs"
    run_id = "resume-tail"
    target = output_root / run_id / "plane"
    real_atomic_json = RECORDER._atomic_json

    def fail_after_trace_append(path, value):
        if (
            Path(path) == target / "checkpoint.json"
            and value.get("status") == "running"
            and value.get("completed_source_row_indices") == [0]
        ):
            raise RuntimeError("simulated interruption after trace fsync")
        return real_atomic_json(path, value)

    monkeypatch.setattr(RECORDER, "_atomic_json", fail_after_trace_append)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        RECORDER.run_dataset(
            "plane", tmp_path, output_root, run_id, workers=1, indices=[0]
        )
    assert worker_calls == [0]

    trace_path = target / "traces.jsonl"
    original_tail = trace_path.read_bytes()
    if tail_mutation == "missing_observables":
        damaged = json.loads(original_tail)
        damaged.pop("observable_traces")
        trace_path.write_bytes(_trace_bytes(damaged))
        original_tail = trace_path.read_bytes()
    elif tail_mutation == "missing_worker_proof":
        damaged = json.loads(original_tail)
        damaged.pop("process_independence")
        trace_path.write_bytes(_trace_bytes(damaged))
        original_tail = trace_path.read_bytes()
    elif tail_mutation == "bad_status":
        damaged = json.loads(original_tail)
        damaged["reproduction"]["passed"] = False
        damaged["reproduction"]["failed_checks"] = ["injected mismatch"]
        trace_path.write_bytes(_trace_bytes(damaged))
        original_tail = trace_path.read_bytes()
    elif tail_mutation == "partial":
        original_tail = original_tail[: len(original_tail) // 2]
        trace_path.write_bytes(original_tail)

    monkeypatch.setattr(RECORDER, "_atomic_json", real_atomic_json)
    summary = RECORDER.run_dataset(
        "plane",
        tmp_path,
        output_root,
        run_id,
        workers=1,
        resume=True,
        indices=[0],
    )

    assert worker_calls == [0, 0]
    assert summary["completed_now"] == 1
    assert summary["status"] == "complete"
    checkpoint = json.loads((target / "checkpoint.json").read_text(encoding="utf-8"))
    assert len(checkpoint["quarantined_uncommitted_tails"]) == 1
    quarantine = checkpoint["quarantined_uncommitted_tails"][0]
    quarantined_bytes = (target / quarantine["path"]).read_bytes()
    assert quarantined_bytes == original_tail
    assert quarantine["byte_count"] == len(original_tail)
    assert quarantine["sha256"] == RECORDER._sha256_bytes(original_tail)
    final_record = json.loads(trace_path.read_text(encoding="utf-8"))
    assert final_record["observable_traces"] == valid_record["observable_traces"]
    assert final_record["process_independence"]["independence_check_passed"] is True


def test_resume_rejects_semantically_bad_row_in_finished_complete_run(
    monkeypatch, tmp_path: Path
) -> None:
    _row, _valid_record, worker_calls = _temporary_plane_runner(monkeypatch, tmp_path)
    output_root = tmp_path / "runs"
    run_id = "finished-semantic-corruption"
    summary = RECORDER.run_dataset(
        "plane", tmp_path, output_root, run_id, workers=1, indices=[0]
    )
    assert summary["status"] == "complete"
    assert worker_calls == [0]

    target = output_root / run_id / "plane"
    trace_path = target / "traces.jsonl"
    record = json.loads(trace_path.read_text(encoding="utf-8"))
    record["original_entrypoint"]["actual_reference"]["potential"]["values_si"][0][
        0
    ] += 1.0
    record["reproduction"][
        "reference_matches_full_values_uncertainty_methods_components"
    ] = False
    record["reproduction"]["passed"] = False
    record["reproduction"]["failed_checks"] = [
        "actual original-entrypoint values/uncertainties differ from frozen reference"
    ]
    changed_bytes = _trace_bytes(record)
    trace_path.write_bytes(changed_bytes)
    checkpoint_path = target / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint.update(
        {
            "committed_trace_bytes": len(changed_bytes),
            "committed_trace_sha256": RECORDER._sha256_bytes(changed_bytes),
            "trace_bytes": len(changed_bytes),
            "traces_jsonl_sha256": RECORDER._sha256_bytes(changed_bytes),
            "record_count": 1,
        }
    )
    checkpoint_path.write_text(
        json.dumps(checkpoint, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="status is inconsistent with replay evidence",
    ):
        RECORDER.run_dataset(
            "plane",
            tmp_path,
            output_root,
            run_id,
            workers=1,
            resume=True,
            indices=[0],
        )
    assert worker_calls == [0]


def test_finished_checkpoint_requires_every_selected_case(tmp_path: Path) -> None:
    row, _record = _indexed_plane_trace()
    path = tmp_path / "traces.jsonl"
    path.write_bytes(b"")
    checkpoint = _checkpoint_for_trace(
        b"", status="complete", completed=[], selected=[0]
    )

    with pytest.raises(ValueError, match="final trace set is incomplete"):
        RECORDER._verify_trace_file(
            path,
            checkpoint,
            dataset="plane",
            selected_rows={0: row},
            selected_indices=[0],
        )


def test_resume_rejects_empty_selected_set_before_claiming_independence(
    monkeypatch, tmp_path: Path
) -> None:
    _temporary_plane_runner(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="at least one frozen source row"):
        RECORDER.run_dataset(
            "plane",
            tmp_path,
            tmp_path / "runs",
            "empty-selection",
            workers=1,
            indices=[],
        )


def test_trace_row_rejects_unexpected_supplement_pins() -> None:
    row, record = _indexed_plane_trace()
    record["zero_certificate_supplement_sha256"] = "unexpected-zero-pin"
    record["polynomial_certificate_supplement_sha256"] = "unexpected-poly-pin"

    with pytest.raises(ValueError, match="different polynomial certificate pin"):
        RECORDER._validate_trace_record(record, "plane", 0, row)


def test_capture_context_restores_wrappers_if_setup_import_fails(monkeypatch) -> None:
    captures = []

    def original_evaluate(_engine, _order):
        return [1.0, 2.0]

    class FakeSurfaceNewton:
        evaluate = staticmethod(original_evaluate)

    fake_surface = types.SimpleNamespace(SurfaceNewton=FakeSurfaceNewton)

    def fake_oracle_module(name):
        if name == "surface_reference":
            return fake_surface
        if name == "surface_high_precision":
            raise ImportError("injected second import failure")
        raise AssertionError(name)

    monkeypatch.setattr(RECORDER, "oracle_module", fake_oracle_module)

    with (
        pytest.raises(ImportError, match="injected second import failure"),
        RECORDER.capture_context("cuboid", captures),
    ):
        pytest.fail("context setup should fail before yield")

    assert FakeSurfaceNewton.evaluate is original_evaluate
    engine = types.SimpleNamespace(target=[0.0, 0.0, 0.0], fixture={})
    assert FakeSurfaceNewton.evaluate(engine, 8) == [1.0, 2.0]
    assert captures == []


def test_capture_context_restores_wrappers_after_entrypoint_exception(
    monkeypatch,
) -> None:
    captures = []

    def original(_engine, _point, _step):
        raise RuntimeError("injected wrapped callback failure")

    fake_module = types.SimpleNamespace(observable_values=original)
    monkeypatch.setattr(RECORDER, "oracle_module", lambda _name: fake_module)

    with (
        pytest.raises(RuntimeError, match="injected wrapped callback failure"),
        RECORDER.capture_context("plane", captures),
    ):
        fake_module.observable_values(None, None, None)

    assert fake_module.observable_values is original
    assert captures[0]["_error"] == "RuntimeError: injected wrapped callback failure"


def test_noncavity_quadrature_family_replays_both_order_differences() -> None:
    fixture = next(
        item
        for item in RECORDER.oracle_module("noncavity_fixtures").noncavity_fixtures()
        if item["family"] == "gaussian"
    )
    reference, metadata = RECORDER.oracle_module(
        "noncavity_reference"
    ).noncavity_reference(fixture)
    row = {
        "fixture": RECORDER._jsonable(fixture),
        "reference": RECORDER._jsonable(reference),
        "metadata": RECORDER._jsonable(metadata),
    }

    trace = RECORDER.record_row("noncavity", row)

    assert trace["reproduction"]["passed"] is True
    assert (
        trace["observable_traces"]["potential"]["quadrature_differences_si"] is not None
    )
    assert trace["observable_traces"]["potential"]["frozen_component_checks"][
        "quadrature"
    ]["matches_frozen_float"]


def test_surface_entrypoint_captures_original_longdouble_order_triplet() -> None:
    fixture = next(RECORDER.oracle_module("cavity_fixtures").cuboid_fixtures())
    reference, metadata = RECORDER.oracle_module("surface_reference").surface_reference(
        fixture
    )
    row = {
        "fixture": RECORDER._jsonable(fixture),
        "reference": RECORDER._jsonable(reference),
        "metadata": RECORDER._jsonable(metadata),
    }

    trace = RECORDER.record_row("cuboid", row, source_row_index=0)

    assert trace["reproduction"]["passed"] is True, trace["reproduction"]
    RECORDER._validate_trace_record(trace, "cuboid", 0, row)
    assert trace["original_entrypoint"]["capture_call_count"] >= 3
    assert trace["observable_traces"]["acceleration"]["route"] == "numpy.longdouble"
    assert trace["observable_traces"]["acceleration"]["frozen_component_checks"][
        "quadrature"
    ]["matches_frozen_float"]
