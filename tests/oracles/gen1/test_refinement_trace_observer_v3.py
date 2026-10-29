"""Regression probes for the versioned refinement trace observer."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

V3_SCRIPT = Path(__file__).with_name("record_frozen_refinement_traces_v3.py")
V3_SPEC = importlib.util.spec_from_file_location("gen1_recorder_v3", V3_SCRIPT)
assert V3_SPEC is not None and V3_SPEC.loader is not None
V3 = importlib.util.module_from_spec(V3_SPEC)
V3_SPEC.loader.exec_module(V3)

AUDIT_V3_SCRIPT = Path(__file__).with_name("audit_frozen_refinement_traces_v3.py")
AUDIT_V3_SPEC = importlib.util.spec_from_file_location(
    "gen1_trace_auditor_v3", AUDIT_V3_SCRIPT
)
assert AUDIT_V3_SPEC is not None and AUDIT_V3_SPEC.loader is not None
AUDIT_V3 = importlib.util.module_from_spec(AUDIT_V3_SPEC)
AUDIT_V3_SPEC.loader.exec_module(AUDIT_V3)

PROVENANCE_V3_SCRIPT = Path(__file__).with_name("refinement_trace_provenance_v3.py")
PROVENANCE_V3_SPEC = importlib.util.spec_from_file_location(
    "gen1_trace_provenance_v3", PROVENANCE_V3_SCRIPT
)
assert PROVENANCE_V3_SPEC is not None and PROVENANCE_V3_SPEC.loader is not None
PROVENANCE_V3 = importlib.util.module_from_spec(PROVENANCE_V3_SPEC)
PROVENANCE_V3_SPEC.loader.exec_module(PROVENANCE_V3)

RUNNER_V3_SCRIPT = Path(__file__).with_name("run_refinement_trace_validation.py")
RUNNER_V3_SPEC = importlib.util.spec_from_file_location(
    "gen1_trace_runner_v3", RUNNER_V3_SCRIPT
)
assert RUNNER_V3_SPEC is not None and RUNNER_V3_SPEC.loader is not None
RUNNER_V3 = importlib.util.module_from_spec(RUNNER_V3_SPEC)
RUNNER_V3_SPEC.loader.exec_module(RUNNER_V3)


def _plane_row():
    fixtures = V3.oracle_module("noncavity_fixtures").noncavity_fixtures()
    fixture = next(item for item in fixtures if item["family"] == "temperature_plane")
    reference, metadata = V3.oracle_module(
        "plane_static_reference"
    ).static_plane_reference(fixture)
    return {
        "fixture": V3._jsonable(fixture),
        "reference": V3._jsonable(reference),
        "metadata": V3._jsonable(metadata),
    }


def _baseline_record():
    row = _plane_row()
    record = V3.record_row("plane", row, source_row_index=0)
    record["process_independence"] = {
        "pid": os.getpid(),
        "modules_before": [],
        "modules_after": [],
        "production_module_names": [],
        "production_imported": False,
        "independence_check_passed": True,
    }
    return row, record


def _audit(record, row):
    candidate = json.loads(json.dumps(record))
    candidate["zero_certificate_supplement_sha256"] = AUDIT_V3.ZERO_SHA256
    return AUDIT_V3.audit_record("plane", 0, row, candidate)


def test_v3_auditor_pins_the_versioned_observer_source(tmp_path):
    assert AUDIT_V3.sha256(V3_SCRIPT) == AUDIT_V3.OBSERVER_SHA256

    with pytest.raises(ValueError, match="frozen v3 observer"):
        AUDIT_V3.main(
            [
                "--l6-root",
                str(tmp_path),
                "--replay-root",
                str(tmp_path),
                "--observer-source",
                str(V3_SCRIPT.with_name("record_frozen_refinement_traces.py")),
                "--output",
                str(tmp_path / "audit.json"),
            ]
        )


def test_validator_rejects_missing_required_component_slot():
    row, record = _baseline_record()
    del record["observable_traces"]["acceleration"]["frozen_component_checks"][
        "derivative"
    ]

    with pytest.raises(ValueError, match="component-check slot"):
        V3._validate_trace_record(record, "plane", 0, row)


def test_validator_rejects_arbitrary_not_applicable_trend():
    row, record = _baseline_record()
    record["observable_traces"]["acceleration"]["derivative_trend"] = {
        "classification": "not_applicable",
        "reason": "caller supplied explanation",
    }

    with pytest.raises(ValueError, match="not_applicable"):
        V3._validate_trace_record(record, "plane", 0, row)


def test_validator_and_auditor_require_canonical_structural_na_reason():
    row, record = _baseline_record()
    record["observable_traces"]["potential"]["derivative_trend"][
        "reason"
    ] = "caller supplied explanation"

    with pytest.raises(ValueError, match="not_applicable reason"):
        V3._validate_trace_record(record, "plane", 0, row)
    with pytest.raises(ValueError, match="N/A reason mismatch"):
        _audit(record, row)


def test_validator_rejects_trend_that_disagrees_with_captured_callbacks():
    row, record = _baseline_record()
    trend = record["observable_traces"]["acceleration"]["derivative_trend"]
    trend.update(
        {
            "classification": "decreasing",
            "coarse_to_middle_si": "1e20",
            "middle_to_fine_si": "1e10",
            "nondecreasing": False,
        }
    )

    with pytest.raises(ValueError, match="captured|callback|pair"):
        V3._validate_trace_record(record, "plane", 0, row)


@pytest.mark.parametrize(
    "mutation",
    [
        "passed_false",
        "failed_checks_on_success",
        "actual_value_drift",
        "status",
        "capture_count",
        "final_vector_short",
        "final_match_flag",
        "empty_observable",
    ],
)
def test_validator_and_auditor_reject_reproduction_or_status_contradictions(mutation):
    row, record = _baseline_record()
    if mutation == "passed_false":
        record["reproduction"]["passed"] = False
    elif mutation == "failed_checks_on_success":
        record["reproduction"]["failed_checks"] = ["injected failure"]
    elif mutation == "actual_value_drift":
        record["original_entrypoint"]["actual_reference"]["potential"][
            "values_si"
        ][0][0] += 1.0
    elif mutation == "status":
        record["status"] = "unresolved_trend"
    elif mutation == "capture_count":
        record["original_entrypoint"]["capture_call_count"] += 1
    elif mutation == "final_vector_short":
        record["observable_traces"]["gradient"]["captured_final_values_si"].pop()
    elif mutation == "final_match_flag":
        trace = record["observable_traces"]["acceleration"]
        trace["captured_final_values_match_frozen"] = not trace[
            "captured_final_values_match_frozen"
        ]
    else:
        record["observable_traces"]["acceleration"] = {}

    with pytest.raises(ValueError):
        V3._validate_trace_record(record, "plane", 0, row)
    with pytest.raises(ValueError):
        _audit(record, row)


def test_trace_decoder_rejects_duplicate_json_keys():
    _row, record = _baseline_record()
    baseline = _trace_bytes(record)
    duplicate = baseline.replace(
        b'"observable_traces": {',
        b'"observable_traces": {"gradient": {},',
        1,
    )
    assert duplicate != baseline

    with pytest.raises(RuntimeError, match="duplicate"):
        V3._decode_trace_bytes(duplicate, description="duplicate-key probe")


def test_independent_auditor_accepts_v3_baseline():
    row, record = _baseline_record()

    result = _audit(record, row)

    assert result["status"] == "reproduced"
    assert result["semantic_validation"] == "SEMANTIC_VALID"
    assert result["provenance_validation"]["status"] == "PROVENANCE_UNVERIFIED"
    assert result["scientific_admission"] == "NOT_ADMITTED"


def test_v3_auditor_does_not_import_or_trust_recorder_verdict():
    auditor_source = AUDIT_V3_SCRIPT.read_text(encoding="utf-8")
    assert "record_frozen_refinement_traces_v3" not in auditor_source
    row, record = _baseline_record()
    record["semantic_validation"] = "SEMANTIC_INVALID"
    record["provenance_validation"] = "PROVENANCE_VERIFIED"
    record["scientific_admission"] = "ADMITTED_WITH_DECLARED_SCOPE"

    result = _audit(record, row)

    assert result["semantic_validation"] == "SEMANTIC_VALID"
    assert result["provenance_validation"]["status"] == "PROVENANCE_UNVERIFIED"
    assert result["scientific_admission"] == "NOT_ADMITTED"


def test_v3_historical_format_inspection_is_read_only_and_not_a_v3_verdict():
    inspected = AUDIT_V3.inspect_historical_record(
        {"schema": "gragra.gen1.refinement-trace-observer.v1", "status": "reproduced"}
    )

    assert inspected["historical_format"] == "V1"
    assert inspected["format_readable"] is True
    assert inspected["semantic_validation"] == "SEMANTIC_NOT_RUN"
    assert inspected["provenance_validation"] == "PROVENANCE_UNVERIFIED"
    assert inspected["scientific_admission"] == "NOT_ADMITTED"
    assert inspected["v3_verdict_applied_retroactively"] is False


def test_v3_semantics_do_not_imply_provenance_or_admission():
    row, record = _baseline_record()
    V3._validate_trace_record(record, "plane", 0, row)
    result = PROVENANCE_V3.admission_decision(
        execution_state="complete",
        semantic_status="SEMANTIC_VALID",
        provenance_status="PROVENANCE_UNVERIFIED",
        independent_audit_valid=True,
        certificate_validation_valid=True,
        trend_validation_complete=True,
    )

    assert result == "NOT_ADMITTED"


def test_unresolved_trend_blocks_admission_even_with_verified_provenance():
    result = PROVENANCE_V3.admission_decision(
        execution_state="complete",
        semantic_status="SEMANTIC_VALID",
        provenance_status="PROVENANCE_VERIFIED",
        independent_audit_valid=True,
        certificate_validation_valid=True,
        trend_validation_complete=False,
    )

    assert result == "NOT_ADMITTED"


@pytest.mark.parametrize(
    ("execution_state", "semantic_status", "provenance_status", "expected"),
    [
        ("running", "SEMANTIC_INVALID", "PROVENANCE_NOT_RUN", "NOT_ADMITTED"),
        ("complete", "SEMANTIC_NOT_RUN", "PROVENANCE_MISMATCH", "NOT_ADMITTED"),
        ("interrupted", "SEMANTIC_NOT_RUN", "PROVENANCE_NOT_RUN", "INCOMPLETE_EVIDENCE"),
    ],
)
def test_invalid_or_mismatched_evidence_precedes_incomplete_state(
    execution_state, semantic_status, provenance_status, expected
):
    result = PROVENANCE_V3.admission_decision(
        execution_state=execution_state,
        semantic_status=semantic_status,
        provenance_status=provenance_status,
        independent_audit_valid=False,
        certificate_validation_valid=False,
        trend_validation_complete=False,
    )

    assert result == expected


def test_v3_canonical_runner_selects_v3_and_fails_closed_without_anchor(
    monkeypatch, tmp_path
):
    assert RUNNER_V3.DEFAULT_RECORDER.name == "record_frozen_refinement_traces_v3.py"
    assert RUNNER_V3.DEFAULT_AUDITOR.name == "audit_frozen_refinement_traces_v3.py"
    assert RUNNER_V3.DEFAULT_SCHEMA == V3.SCHEMA
    monkeypatch.setattr(
        RUNNER_V3.RECORDER,
        "run_dataset",
        lambda dataset, *_args, **_kwargs: {
            "dataset": dataset,
            "status": "complete",
            "execution_state": "complete",
            "semantic_validation": "SEMANTIC_VALID",
            "provenance_validation": "PROVENANCE_UNVERIFIED",
            "scientific_admission": "NOT_ADMITTED",
        },
    )
    monkeypatch.setattr(RUNNER_V3.AUDITOR, "certificate_inputs", lambda _root: ([], []))
    monkeypatch.setattr(
        RUNNER_V3.AUDITOR,
        "audit_dataset",
        lambda dataset, *_args, **_kwargs: {
            "dataset": dataset,
            "semantic_validation": "SEMANTIC_VALID",
            "provenance_validation": "PROVENANCE_UNVERIFIED",
            "scientific_admission": "NOT_ADMITTED",
            "errors": [],
            "status": "reproduced",
        },
    )
    output_root = tmp_path / "output"
    status = RUNNER_V3.main(
        [
            "--l6-root", str(tmp_path / "assets"),
            "--output-root", str(output_root),
            "--run-id", "v3-route-probe",
            "--datasets", "plane",
            "--indices", "0",
        ]
    )
    report = json.loads(
        (output_root / "v3-route-probe" / "validation-report-v3.json").read_text()
    )

    assert status == 3
    assert report["selected_observer"] == "V3"
    assert report["semantic_validation"] == "SEMANTIC_VALID"
    assert report["provenance_validation"] == "PROVENANCE_UNVERIFIED"
    assert report["scientific_admission"] == "NOT_ADMITTED"


@pytest.mark.parametrize("digits", [15, 30, 50, 80, 100])
def test_v3_precision_round_trip_and_semantic_pair_check(digits):
    import mpmath as mp

    with mp.workdps(digits + 20):
        delta = mp.power(10, -(digits - 8))
        values = [mp.mpf(1) + 2 * delta, mp.mpf(1) + delta, mp.mpf(1)]
        encoded = [V3._capture_number(value, digits) for value in values]
        expected_tuple = [tuple(item["mpf"]) for item in encoded]

    previous_dps = mp.mp.dps
    try:
        mp.mp.dps = 15
        recorder_values = [V3._decode_callback_value(item) for item in encoded]
        auditor_values = [AUDIT_V3.decode(item) for item in encoded]
    finally:
        mp.mp.dps = previous_dps

    assert [tuple(value._mpf_) for value in recorder_values] == expected_tuple
    assert [tuple(value._mpf_) for value in auditor_values] == expected_tuple
    with mp.workdps(digits + 20):
        recorder_pair = V3._callback_pair(
            [(value,) for value in recorder_values], "potential", longdouble=False
        )
        auditor_pair = tuple(
            AUDIT_V3.vector_difference((left,), (right,))
            for left, right in zip(
                recorder_values[:-1], recorder_values[1:], strict=True
            )
        )
        assert recorder_pair == auditor_pair
        trace = {
            "derivative_differences_si": {
                "4h_to_2h": V3._exact_mp_string(recorder_pair[0]),
                "2h_to_h": V3._exact_mp_string(recorder_pair[1]),
                "ten_max": V3._exact_mp_string(10 * max(recorder_pair)),
            }
        }
        V3._require_saved_pair(
            trace,
            "derivative_differences_si",
            recorder_pair,
            ("4h_to_2h", "2h_to_h"),
            longdouble=False,
            precision=digits + 20,
            index=0,
            observable="potential",
        )
        auditor_trend = {
            "classification": "unresolved_nondecreasing",
            "nondecreasing": True,
            "coarse_to_middle_si": V3._exact_mp_string(recorder_pair[0]),
            "middle_to_fine_si": V3._exact_mp_string(recorder_pair[1]),
            "independent_floor_si": None,
        }
        AUDIT_V3.audit_trend(auditor_trend, auditor_pair, None)
        trace["derivative_differences_si"]["2h_to_h"] = mp.nstr(
            recorder_pair[1] * 2, digits + 30
        )
        with pytest.raises(ValueError, match="differs from captured callbacks"):
            V3._require_saved_pair(
                trace,
                "derivative_differences_si",
                recorder_pair,
                ("4h_to_2h", "2h_to_h"),
                longdouble=False,
                precision=digits + 20,
                index=0,
                observable="potential",
            )
        auditor_trend["middle_to_fine_si"] = trace[
            "derivative_differences_si"
        ]["2h_to_h"]
        with pytest.raises(ValueError, match="trend/named pair mismatch"):
            AUDIT_V3.audit_trend(auditor_trend, auditor_pair, None)


def test_v3_provenance_requires_trusted_external_root():
    payload = {"schema": "test", "callback_sha256": "a" * 64, "run_id": "r"}
    expected = PROVENANCE_V3.commitment_document(payload)

    class Provider:
        def __init__(self, root_verified, document):
            self.root_verified = root_verified
            self.document = document

        def verify_commitment(self, _lookup_key, _expected):
            return {
                "trust_root_verified": self.root_verified,
                "document": self.document,
                "anchor_id": "test-only",
            }

    untrusted = PROVENANCE_V3.verify_external_commitment(
        expected, Provider(False, expected), lookup_key="run/r"
    )
    verified = PROVENANCE_V3.verify_external_commitment(
        expected, Provider(True, expected), lookup_key="run/r"
    )
    mismatch_document = PROVENANCE_V3.commitment_document(
        {**payload, "callback_sha256": "b" * 64}
    )
    mismatch = PROVENANCE_V3.verify_external_commitment(
        expected, Provider(True, mismatch_document), lookup_key="run/r"
    )
    unavailable = PROVENANCE_V3.verify_external_commitment(
        expected, None, lookup_key="run/r"
    )
    incomplete = PROVENANCE_V3.verify_external_commitment(
        expected,
        Provider(True, {"schema": PROVENANCE_V3.SCHEMA}),
        lookup_key="run/incomplete",
    )

    assert untrusted["status"] == "PROVENANCE_UNVERIFIED"
    assert verified["status"] == "PROVENANCE_VERIFIED"
    assert mismatch["status"] == "PROVENANCE_MISMATCH"
    assert unavailable["status"] == "PROVENANCE_UNVERIFIED"
    assert incomplete["status"] == "PROVENANCE_MISMATCH"
    assert "malformed" in incomplete["reason"]


def test_v3_callback_commitment_binds_order_and_source_identities():
    row, record = _baseline_record()
    binding = {
        "run_id": "unit-run",
        "dataset": "plane",
        "source_row_index": 0,
        "observer_sha256": "1" * 64,
        "auditor_sha256": "2" * 64,
        "oracle_module_sha256": {"plane_static_reference.py": "3" * 64},
        "frozen_reference_sha256": "4" * 64,
        "fixture_sha256": "5" * 64,
        "certificate_sha256": {"zero": "6" * 64, "polynomial": None},
        "environment": {"python": "unit-test"},
        "numeric_representation": {"digits": [80]},
    }
    payload = PROVENANCE_V3.callback_payload(record, binding)
    original_digest = PROVENANCE_V3.commitment_document(payload)[
        "commitment_sha256"
    ]
    changed_source = PROVENANCE_V3.commitment_document(
        PROVENANCE_V3.callback_payload(
            record, {**binding, "oracle_module_sha256": {"other.py": "7" * 64}}
        )
    )["commitment_sha256"]
    reversed_record = json.loads(json.dumps(record))
    reversed_record["original_entrypoint"]["captured_calls"].reverse()
    changed_order = PROVENANCE_V3.commitment_document(
        PROVENANCE_V3.callback_payload(reversed_record, binding)
    )["commitment_sha256"]

    assert payload["binding"]["fixture_sha256"] == binding["fixture_sha256"]
    assert payload["callbacks"] == record["original_entrypoint"]["captured_calls"]
    assert changed_source != original_digest
    assert changed_order != original_digest


def test_a3_50_digit_fallback_positive_and_perturbed_negative_control():
    import mpmath as mp

    digits = 50
    with mp.workdps(80):
        delta = mp.mpf("1e-42")
        vectors = [
            [mp.mpf(1) + 2 * delta] + [mp.mpf(0)] * 12,
            [mp.mpf(1) + delta] + [mp.mpf(0)] * 12,
            [mp.mpf(1)] + [mp.mpf(0)] * 12,
        ]
        calls = [
            {
                "kernel": "surface_high_precision_evaluate",
                "inputs": {"digits": digits, "order": order},
                "output": [
                    [V3._capture_number(value, digits) for value in vector],
                    [V3._capture_number(mp.mpf(1), digits) for _ in range(13)],
                    1,
                ],
            }
            for order, vector in zip((16, 24, 32), vectors, strict=True)
        ]
        expected_pair, _ = V3._surface_callback_pair(
            calls, "cuboid", {}, {}, {"route": "mpmath_fallback"}, "potential"
        )
        auditor_pair, _auditor_vectors, _lookup = AUDIT_V3.high_precision_callback_pair(
            calls, "potential"
        )
        assert expected_pair == auditor_pair
        saved = {
            "quadrature_differences_si": {
                "coarse_to_middle": V3._exact_mp_string(expected_pair[0]),
                "middle_to_fine": V3._exact_mp_string(expected_pair[1]),
                "ten_max": V3._exact_mp_string(10 * max(expected_pair)),
            }
        }
        V3._require_saved_pair(
            saved,
            "quadrature_differences_si",
            expected_pair,
            ("coarse_to_middle", "middle_to_fine"),
            longdouble=False,
            precision=80,
            index=0,
            observable="potential",
        )
        trend = {
            "classification": "unresolved_nondecreasing",
            "nondecreasing": True,
            "coarse_to_middle_si": V3._exact_mp_string(expected_pair[0]),
            "middle_to_fine_si": V3._exact_mp_string(expected_pair[1]),
            "independent_floor_si": None,
        }
        AUDIT_V3.audit_trend(trend, expected_pair, None)

        calls[0]["output"][0][0] = V3._capture_number(mp.mpf(1) + 3 * delta, digits)
        perturbed_pair, _ = V3._surface_callback_pair(
            calls, "cuboid", {}, {}, {"route": "mpmath_fallback"}, "potential"
        )
        perturbed_auditor_pair, _values, _map = AUDIT_V3.high_precision_callback_pair(
            calls, "potential"
        )
        assert perturbed_pair == perturbed_auditor_pair
        assert perturbed_pair != expected_pair
        with pytest.raises(ValueError, match="differs from captured callbacks"):
            V3._require_saved_pair(
                saved,
                "quadrature_differences_si",
                perturbed_pair,
                ("coarse_to_middle", "middle_to_fine"),
                longdouble=False,
                precision=80,
                index=0,
                observable="potential",
            )
        with pytest.raises(ValueError, match="trend/named pair mismatch"):
            AUDIT_V3.audit_trend(trend, perturbed_auditor_pair, None)


def test_v3_a3_fallback_positive_control_uses_pinned_high_precision_oracle_route():
    fixture = next(
        item
        for item in V3.oracle_module("cavity_fixtures").cuboid_fixtures()
        if item.target_kind == "exterior" and item.lengths_m == (1.0, 1.0, 1.0)
    )
    fixture_json = V3._jsonable(fixture)
    _initial, metadata = V3._invoke_original_entrypoint("cuboid", fixture_json)
    acceptance = V3.oracle_module("acceptance")
    unresolved = {
        name: {
            "values_si": (),
            "uncertainty": acceptance.OracleUncertainty(
                None, None, None, None, "synthetic unresolved input", ()
            ),
        }
        for name in V3.OBSERVABLE_SLICES
    }
    captures = []
    with V3.capture_context("cuboid", captures):
        output, fallback_metadata = V3.oracle_module(
            "surface_high_precision"
        ).high_precision_reference(
            V3._surface_fixture("cuboid", fixture_json), 6.67430e-11, unresolved
        )
    metadata.update(fallback_metadata)
    row = {
        "fixture": fixture_json,
        "reference": V3._jsonable(output),
        "metadata": V3._jsonable(metadata),
    }
    record = V3._case_trace(
        "cuboid",
        row,
        captures,
        output,
        metadata,
        source_row_index=0,
        zero_certificate_sha256=V3.EXPECTED_ZERO_CERTIFICATES_SHA256,
        polynomial_certificate_sha256=None,
    )
    record["process_independence"] = _worker_evidence()

    V3._validate_trace_record(
        record,
        "cuboid",
        0,
        row,
        zero_certificate_sha256=V3.EXPECTED_ZERO_CERTIFICATES_SHA256,
        polynomial_certificate_sha256=None,
    )
    audited = AUDIT_V3.audit_record("cuboid", 0, row, record)

    assert {trace.get("route") for trace in record["observable_traces"].values()} == {
        "mpmath_fallback"
    }
    assert all(
        call["inputs"].get("digits") in {50, 80}
        for call in captures
        if call["kernel"] == "surface_high_precision_evaluate"
    )
    assert record["status"] == audited["status"]
    assert audited["provenance_validation"]["status"] == "PROVENANCE_UNVERIFIED"
    assert audited["scientific_admission"] == "NOT_ADMITTED"


def test_coordinated_callback_rewrite_is_not_scientifically_admitted():
    row, record = _baseline_record()
    trace = record["observable_traces"]["acceleration"]
    calls = record["original_entrypoint"]["captured_calls"]
    events = [
        call
        for call in calls
        if call["kernel"] == "plane_observable_values"
        and (call["inputs"].get("digits") or call["inputs"].get("mp_dps_at_call"))
        == 80
    ][-3:]
    final = V3._decode_callback_value(events[-1]["output"])["acceleration"]
    component_float = trace["frozen_component_checks"]["derivative"][
        "frozen_component_si"
    ]
    with V3.mp.workdps(100):
        magnitude = V3.mp.mpf(str(component_float)) / 10
        middle = (final[0] + magnitude / 2, final[1], final[2])
        coarse = (middle[0] + magnitude, middle[1], middle[2])
        for event, vector in zip(events[:2], (coarse, middle), strict=True):
            event["output"]["acceleration"] = [
                V3._capture_number(value, 80) for value in vector
            ]
    with V3.mp.workdps(80):
        pair = V3._mp_callback_pairs(calls, "plane", row["fixture"], "acceleration")[0]
        formula = 10 * max(pair)
        assert float(formula) == component_float
        trace["derivative_differences_si"].update(
            {
                "4h_to_2h": V3._exact_mp_string(pair[0]),
                "2h_to_h": V3._exact_mp_string(pair[1]),
                "ten_max": V3._exact_mp_string(formula),
            }
        )
        check = trace["frozen_component_checks"]["derivative"]
        check["formula_value_si"] = V3._exact_mp_string(formula)
        check["formula_float_si"] = float(formula)
        check["matches_frozen_float"] = check["formula_float_si"] == component_float
        trace["derivative_trend"].update(
            V3._change_trend(
                *pair, V3.mp.mpf(trace["derivative_trend"]["independent_floor_si"])
            )
        )
    raw, unresolved, status = V3._derived_trace_status(
        record["observable_traces"], record["reproduction"]["failed_checks"]
    )
    record.update(raw_unresolved_trends=raw, unresolved_trends=unresolved, status=status)

    V3._validate_trace_record(record, "plane", 0, row)
    audit_result = _audit(record, row)
    provenance = PROVENANCE_V3.verify_external_commitment(
        PROVENANCE_V3.commitment_document({"callback_sha256": "baseline"}),
        None,
        lookup_key="missing-anchor",
    )
    admission = PROVENANCE_V3.admission_decision(
        execution_state="complete",
        semantic_status="SEMANTIC_VALID",
        provenance_status=provenance["status"],
        independent_audit_valid=audit_result["status"] in {"reproduced", "unresolved_trend"},
        certificate_validation_valid=True,
        trend_validation_complete=not record.get("unresolved_trends"),
    )

    assert audit_result["status"] == record["status"]
    assert audit_result["semantic_validation"] == "SEMANTIC_VALID"
    assert audit_result["provenance_validation"]["status"] == "PROVENANCE_UNVERIFIED"
    assert audit_result["scientific_admission"] == "NOT_ADMITTED"
    assert provenance["status"] == "PROVENANCE_UNVERIFIED"
    assert admission == "NOT_ADMITTED"


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            lambda record: record["observable_traces"]["acceleration"][
                "frozen_component_checks"
            ].pop("derivative"),
            "MP component slots incomplete",
        ),
        (
            lambda record: record["observable_traces"]["acceleration"].__setitem__(
                "derivative_trend",
                {"classification": "not_applicable", "reason": "caller supplied"},
            ),
            "structural N/A trend mismatch",
        ),
        (
            lambda record: record["observable_traces"]["acceleration"][
                "derivative_trend"
            ].update(
                {
                    "classification": "decreasing",
                    "coarse_to_middle_si": "1e20",
                    "middle_to_fine_si": "1e10",
                    "nondecreasing": False,
                }
            ),
            "captured/named pair mismatch|trend/named pair mismatch",
        ),
    ],
    ids=("missing-component-slot", "arbitrary-na", "callback-trend-mismatch"),
)
def test_independent_auditor_rejects_semantic_mutations(mutation, expected_error):
    row, record = _baseline_record()
    mutation(record)

    with pytest.raises(ValueError, match=expected_error):
        _audit(record, row)


def test_independent_auditor_rejects_duplicate_json_keys(tmp_path):
    _row, record = _baseline_record()
    baseline = _trace_bytes(record)
    duplicate = baseline.replace(
        b'"observable_traces": {',
        b'"observable_traces": {"gradient": {},',
        1,
    )
    assert duplicate != baseline
    path = tmp_path / "duplicate.jsonl"
    path.write_bytes(duplicate)

    with pytest.raises(ValueError, match="duplicate"):
        AUDIT_V3.read_rows(path)


def _worker_evidence():
    return {
        "pid": os.getpid(),
        "modules_before": [],
        "modules_after": [],
        "production_module_names": [],
        "production_imported": False,
        "independence_check_passed": True,
    }


def _trace_bytes(record):
    return (json.dumps(record, allow_nan=False, sort_keys=True) + "\n").encode()


def _temporary_plane_runner(monkeypatch, tmp_path):
    row = _plane_row()
    reference_path = tmp_path / "plane.jsonl"
    reference_path.write_bytes(_trace_bytes(row))
    upstream_manifest = tmp_path / "upstream-manifest.json"
    upstream_manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setitem(
        V3.EXPECTED_REFERENCE_SHA256, "plane", V3._sha256(reference_path)
    )
    monkeypatch.setattr(
        V3, "_dataset_paths", lambda _root, _dataset: (reference_path, upstream_manifest)
    )
    monkeypatch.setattr(V3, "_module_hashes", lambda: V3.EXPECTED_ORACLE_SHA256)
    monkeypatch.setattr(V3, "_loaded_production_modules", lambda: [])
    monkeypatch.setattr(
        V3,
        "_load_zero_certificates",
        lambda _root: {
            "path": str(tmp_path / "zero-certificates.json"),
            "sha256": "zero-pin",
            "data": {"classification": "test fixture", "certificates": []},
        },
    )
    monkeypatch.setattr(V3, "_load_polynomial_quadrature_certificates", lambda *_: None)
    record = V3.record_row(
        "plane", row, source_row_index=0, zero_certificate_sha256="zero-pin"
    )
    record["process_independence"] = _worker_evidence()
    worker_calls = []

    def fake_worker(payload):
        index = payload[1]
        worker_calls.append(index)
        candidate = json.loads(json.dumps(record))
        candidate["source_row_index"] = index
        candidate["process_independence"] = _worker_evidence()
        return index, candidate, None

    monkeypatch.setattr(V3, "_worker", fake_worker)
    return row, tmp_path / "runs", "v3-resume", worker_calls


def _finish_plane_run(monkeypatch, tmp_path):
    row, output_root, run_id, worker_calls = _temporary_plane_runner(
        monkeypatch, tmp_path
    )
    summary = V3.run_dataset(
        "plane", tmp_path, output_root, run_id, workers=1, indices=[0]
    )
    assert summary["status"] == "complete"
    assert worker_calls == [0]
    return row, output_root, run_id, worker_calls


def _rewrite_committed_records(target, records, *, completed_indices=None):
    payload = b"".join(_trace_bytes(record) for record in records)
    (target / "traces.jsonl").write_bytes(payload)
    checkpoint_path = target / "checkpoint.json"
    checkpoint = V3._strict_json_file(checkpoint_path)
    digest = V3._sha256_bytes(payload)
    indices = [record["source_row_index"] for record in records]
    if completed_indices is None:
        completed_indices = sorted(set(indices))
    checkpoint.update(
        {
            "committed_trace_bytes": len(payload),
            "committed_trace_sha256": digest,
            "trace_bytes": len(payload),
            "traces_jsonl_sha256": digest,
            "record_count": len(records),
            "completed_source_row_indices": completed_indices,
            "mismatch_count": sum(
                record.get("status") in {"replay_mismatch", "replay_error"}
                for record in records
            ),
            "unresolved_trend_case_count": sum(
                bool(record.get("unresolved_trends")) for record in records
            ),
        }
    )
    V3._atomic_json(checkpoint_path, checkpoint)


def test_v3_resume_rejects_missing_committed_record_after_digest_rehash(
    monkeypatch, tmp_path
):
    _row, output_root, run_id, worker_calls = _finish_plane_run(monkeypatch, tmp_path)
    target = output_root / run_id / "plane"
    _rewrite_committed_records(target, [], completed_indices=[])

    with pytest.raises(ValueError, match="final trace set is incomplete"):
        V3.run_dataset(
            "plane", tmp_path, output_root, run_id, workers=1, resume=True, indices=[0]
        )
    assert worker_calls == [0]


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("duplicate", "duplicate trace source row index"),
        ("out_of_range", "is not selected"),
        ("fixture", "fixture identity differs from frozen row"),
        ("worker_evidence", "malformed worker module observations"),
    ],
)
def test_v3_resume_rejects_rehashed_record_identity_or_worker_corruption(
    mutation, expected_error, monkeypatch, tmp_path
):
    _row, output_root, run_id, worker_calls = _finish_plane_run(monkeypatch, tmp_path)
    target = output_root / run_id / "plane"
    original = V3._decode_trace_bytes(
        (target / "traces.jsonl").read_bytes(), description="baseline V3 trace"
    )[0]
    if mutation == "duplicate":
        records = [original, json.loads(json.dumps(original))]
        completed_indices = [0]
    else:
        changed = json.loads(json.dumps(original))
        if mutation == "out_of_range":
            changed["source_row_index"] = 1
        elif mutation == "fixture":
            changed["fixture"]["injected_identity_change"] = True
        else:
            changed["process_independence"]["modules_before"] = None
        records = [changed]
        completed_indices = [0]
    _rewrite_committed_records(
        target, records, completed_indices=completed_indices
    )

    with pytest.raises((RuntimeError, ValueError), match=expected_error):
        V3.run_dataset(
            "plane", tmp_path, output_root, run_id, workers=1, resume=True, indices=[0]
        )
    assert worker_calls == [0]


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("observer_sha256", "0" * 64),
        ("oracle_module_sha256", {"changed_oracle.py": "0" * 64}),
    ],
)
def test_v3_resume_rejects_changed_source_identity_pins(
    field, changed_value, monkeypatch, tmp_path
):
    _row, output_root, run_id, worker_calls = _finish_plane_run(monkeypatch, tmp_path)
    target = output_root / run_id / "plane"
    checkpoint_path = target / "checkpoint.json"
    checkpoint = V3._strict_json_file(checkpoint_path)
    checkpoint[field] = changed_value
    V3._atomic_json(checkpoint_path, checkpoint)

    with pytest.raises(RuntimeError, match=f"{field} changed"):
        V3.run_dataset(
            "plane", tmp_path, output_root, run_id, workers=1, resume=True, indices=[0]
        )
    assert worker_calls == [0]


def _mutated_trace_bytes(original, mutation):
    if mutation == "duplicate_key":
        damaged = original.replace(
            b'"observable_traces": {',
            b'"observable_traces": {"gradient": {},',
            1,
        )
        assert damaged != original
        return damaged

    record = V3._decode_trace_bytes(original, description="test trace")[0]
    if mutation == "missing_component_slot":
        del record["observable_traces"]["acceleration"][
            "frozen_component_checks"
        ]["derivative"]
    elif mutation == "arbitrary_na":
        record["observable_traces"]["acceleration"]["derivative_trend"] = {
            "classification": "not_applicable",
            "reason": "caller supplied",
        }
    elif mutation == "invented_trend_pair":
        record["observable_traces"]["acceleration"]["derivative_trend"].update(
            {
                "classification": "decreasing",
                "coarse_to_middle_si": "1e20",
                "middle_to_fine_si": "1e10",
                "nondecreasing": False,
            }
        )
    else:
        raise AssertionError(f"unknown semantic mutation {mutation!r}")
    return _trace_bytes(record)


def _expected_semantic_rejection(mutation):
    return {
        "duplicate_key": (RuntimeError, "duplicate"),
        "missing_component_slot": (ValueError, "component-check slots"),
        "arbitrary_na": (ValueError, "not_applicable"),
        "invented_trend_pair": (ValueError, "callback|captured|pair"),
    }[mutation]


def test_v3_normal_resume_does_not_replay_completed_row(monkeypatch, tmp_path):
    _row, output_root, run_id, worker_calls = _finish_plane_run(monkeypatch, tmp_path)

    summary = V3.run_dataset(
        "plane", tmp_path, output_root, run_id, workers=1, resume=True, indices=[0]
    )

    assert summary["status"] == "complete"
    assert summary["completed_now"] == 0
    assert worker_calls == [0]
    assert summary["execution_state"] == "complete"
    assert summary["semantic_validation"] == "SEMANTIC_VALID"
    assert summary["provenance_validation"] == "PROVENANCE_UNVERIFIED"
    assert summary["scientific_admission"] == "NOT_ADMITTED"


def test_v3_mock_trusted_anchor_resume_is_verified_but_not_self_admitted(
    monkeypatch, tmp_path
):
    _row, output_root, run_id, worker_calls = _temporary_plane_runner(
        monkeypatch, tmp_path
    )

    class MockAnchor:
        def verify_commitment(self, _lookup_key, expected_document):
            return {
                "trust_root_verified": True,
                "document": expected_document,
                "anchor_id": "unit-test-only-mock",
            }

    provider = MockAnchor()
    first = V3.run_dataset(
        "plane", tmp_path, output_root, run_id, workers=1, indices=[0],
        provenance_provider=provider,
    )
    resumed = V3.run_dataset(
        "plane", tmp_path, output_root, run_id, workers=1, resume=True,
        indices=[0], provenance_provider=provider,
    )

    assert first["provenance_validation"] == "PROVENANCE_VERIFIED"
    assert resumed["provenance_validation"] == "PROVENANCE_VERIFIED"
    assert resumed["scientific_admission"] == "NOT_ADMITTED"
    assert resumed["completed_now"] == 0
    assert worker_calls == [0]


def _coordinated_callback_mutation(record, row):
    trace = record["observable_traces"]["acceleration"]
    calls = record["original_entrypoint"]["captured_calls"]
    baseline_commitment = record["provenance_commitment"]
    events = [
        call
        for call in calls
        if call["kernel"] == "plane_observable_values"
        and (call["inputs"].get("digits") or call["inputs"].get("mp_dps_at_call"))
        == 80
    ][-3:]
    final = V3._decode_callback_value(events[-1]["output"])["acceleration"]
    component_float = trace["frozen_component_checks"]["derivative"][
        "frozen_component_si"
    ]
    with V3.mp.workdps(100):
        magnitude = V3.mp.mpf(str(component_float)) / 10
        middle = (final[0] + magnitude / 2, final[1], final[2])
        coarse = (middle[0] + magnitude, middle[1], middle[2])
        for event, vector in zip(events[:2], (coarse, middle), strict=True):
            event["output"]["acceleration"] = [
                V3._capture_number(value, 80) for value in vector
            ]
    with V3.mp.workdps(80):
        pair = V3._mp_callback_pairs(calls, "plane", row["fixture"], "acceleration")[0]
        formula = 10 * max(pair)
        check = trace["frozen_component_checks"]["derivative"]
        check["formula_value_si"] = V3._exact_mp_string(formula)
        check["formula_float_si"] = float(formula)
        check["matches_frozen_float"] = check["formula_float_si"] == component_float
        trace["derivative_differences_si"].update(
            {
                "4h_to_2h": V3._exact_mp_string(pair[0]),
                "2h_to_h": V3._exact_mp_string(pair[1]),
                "ten_max": V3._exact_mp_string(formula),
            }
        )
        trace["derivative_trend"].update(
            V3._change_trend(
                *pair, V3.mp.mpf(trace["derivative_trend"]["independent_floor_si"])
            )
        )
    raw, unresolved, status = V3._derived_trace_status(
        record["observable_traces"], record["reproduction"]["failed_checks"]
    )
    record.update(raw_unresolved_trends=raw, unresolved_trends=unresolved, status=status)
    V3._validate_trace_record(
        record,
        "plane",
        0,
        row,
        zero_certificate_sha256=record["zero_certificate_supplement_sha256"],
        polynomial_certificate_sha256=record[
            "polynomial_certificate_supplement_sha256"
        ],
    )
    binding = baseline_commitment["payload"]["binding"]
    record["provenance_commitment"] = PROVENANCE_V3.commitment_document(
        PROVENANCE_V3.callback_payload(record, binding)
    )
    return baseline_commitment


@pytest.mark.parametrize("trusted_anchor", [False, True])
def test_v3_resume_keeps_coordinated_mutation_unadmitted_or_rejects_anchor_mismatch(
    trusted_anchor, monkeypatch, tmp_path
):
    _row, output_root, run_id, worker_calls = _finish_plane_run(monkeypatch, tmp_path)
    target = output_root / run_id / "plane"
    trace_path = target / "traces.jsonl"
    record = V3._decode_trace_bytes(
        trace_path.read_bytes(), description="baseline V3 trace"
    )[0]
    baseline_commitment = _coordinated_callback_mutation(record, _row)
    modified = _trace_bytes(record)
    trace_path.write_bytes(modified)
    digest = V3._sha256_bytes(modified)
    checkpoint_path = target / "checkpoint.json"
    checkpoint = V3._strict_json_file(checkpoint_path)
    manifest_path = target / "manifest.json"
    manifest = V3._strict_json_file(manifest_path)
    manifest["callback_commitment_index"]["0"] = record["provenance_commitment"][
        "commitment_sha256"
    ]
    V3._atomic_json(manifest_path, manifest)
    checkpoint.update(
        {
            "committed_trace_bytes": len(modified),
            "committed_trace_sha256": digest,
            "trace_bytes": len(modified),
            "traces_jsonl_sha256": digest,
            "record_count": 1,
            "dataset_manifest_pin_sha256": V3._manifest_pin_sha256(manifest),
        }
    )
    V3._atomic_json(checkpoint_path, checkpoint)
    run_evidence_path = target / "run-evidence.json"
    run_evidence = V3._strict_json_file(run_evidence_path)
    run_evidence["traces_jsonl_sha256"] = digest
    run_evidence["dataset_manifest_sha256"] = V3._sha256(manifest_path)
    V3._atomic_json(run_evidence_path, run_evidence)
    invocation_path = target / "run-invocations.jsonl"
    invocation_rows = [
        json.loads(line)
        for line in invocation_path.read_text(encoding="utf-8").splitlines()
    ]
    for item in invocation_rows:
        if item["event"] == "finished":
            item["traces_jsonl_sha256"] = digest
    invocation_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in invocation_rows),
        encoding="utf-8",
    )
    run_evidence["run_invocations_jsonl_sha256"] = V3._sha256(invocation_path)
    V3._atomic_json(run_evidence_path, run_evidence)

    provider = None
    if trusted_anchor:
        class BaselineAnchor:
            def verify_commitment(self, _lookup_key, _expected):
                return {
                    "trust_root_verified": True,
                    "document": baseline_commitment,
                    "anchor_id": "synthetic-test-only",
                }

        provider = BaselineAnchor()

    if trusted_anchor:
        with pytest.raises(RuntimeError, match="trusted external provenance commitment mismatch"):
            V3.run_dataset(
                "plane", tmp_path, output_root, run_id, workers=1, resume=True,
                indices=[0], provenance_provider=provider,
            )
    else:
        summary = V3.run_dataset(
            "plane", tmp_path, output_root, run_id, workers=1, resume=True,
            indices=[0],
        )
        assert summary["status"] == "complete"
        assert summary["execution_state"] == "complete"
        assert summary["provenance_validation"] == "PROVENANCE_UNVERIFIED"
        assert summary["scientific_admission"] == "NOT_ADMITTED"
    assert worker_calls == [0]


def test_v3_resume_quarantines_and_recomputes_valid_uncommitted_tail(
    monkeypatch, tmp_path
):
    _row, output_root, run_id, worker_calls = _temporary_plane_runner(
        monkeypatch, tmp_path
    )
    target = output_root / run_id / "plane"
    trace_path = target / "traces.jsonl"
    real_atomic_json = V3._atomic_json

    def interrupt_after_trace_append(path, value):
        if (
            Path(path) == target / "checkpoint.json"
            and value.get("status") == "running"
            and value.get("completed_source_row_indices") == [0]
        ):
            raise RuntimeError("simulated interruption after trace fsync")
        return real_atomic_json(path, value)

    monkeypatch.setattr(V3, "_atomic_json", interrupt_after_trace_append)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        V3.run_dataset("plane", tmp_path, output_root, run_id, workers=1, indices=[0])
    assert worker_calls == [0]
    uncommitted_tail = trace_path.read_bytes()

    monkeypatch.setattr(V3, "_atomic_json", real_atomic_json)
    summary = V3.run_dataset(
        "plane", tmp_path, output_root, run_id, workers=1, resume=True, indices=[0]
    )

    assert worker_calls == [0, 0]
    assert summary["status"] == "complete"
    assert summary["completed_now"] == 1
    checkpoint = V3._strict_json_file(target / "checkpoint.json")
    assert len(checkpoint["quarantined_uncommitted_tails"]) == 1
    quarantine = checkpoint["quarantined_uncommitted_tails"][0]
    quarantined_bytes = (target / quarantine["path"]).read_bytes()
    assert quarantined_bytes == uncommitted_tail
    assert quarantine["byte_count"] == len(uncommitted_tail)
    assert quarantine["sha256"] == V3._sha256_bytes(uncommitted_tail)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_key",
        "missing_component_slot",
        "arbitrary_na",
        "invented_trend_pair",
    ],
)
def test_v3_resume_rejects_rehashed_semantic_corruption(
    mutation, monkeypatch, tmp_path
):
    _row, output_root, run_id, worker_calls = _finish_plane_run(monkeypatch, tmp_path)
    target = output_root / run_id / "plane"
    trace_path = target / "traces.jsonl"
    original = trace_path.read_bytes()
    damaged = _mutated_trace_bytes(original, mutation)
    trace_path.write_bytes(damaged)
    checkpoint_path = target / "checkpoint.json"
    checkpoint = V3._strict_json_file(checkpoint_path)
    digest = V3._sha256_bytes(damaged)
    checkpoint.update(
        {
            "committed_trace_bytes": len(damaged),
            "committed_trace_sha256": digest,
            "trace_bytes": len(damaged),
            "traces_jsonl_sha256": digest,
            "record_count": 1,
        }
    )
    V3._atomic_json(checkpoint_path, checkpoint)

    assert checkpoint["committed_trace_sha256"] == V3._sha256_bytes(
        trace_path.read_bytes()
    )
    error_type, error_match = _expected_semantic_rejection(mutation)
    with pytest.raises(error_type, match=error_match):
        V3.run_dataset(
            "plane", tmp_path, output_root, run_id, workers=1, resume=True, indices=[0]
        )
    assert worker_calls == [0]


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_key",
        "missing_component_slot",
        "arbitrary_na",
        "invented_trend_pair",
    ],
)
def test_v3_trace_verifier_rejects_rehashed_semantic_corruption(
    mutation, monkeypatch, tmp_path
):
    row, output_root, run_id, _worker_calls = _finish_plane_run(monkeypatch, tmp_path)
    target = output_root / run_id / "plane"
    trace_path = target / "traces.jsonl"
    original = trace_path.read_bytes()
    damaged = _mutated_trace_bytes(original, mutation)
    trace_path.write_bytes(damaged)
    checkpoint = V3._strict_json_file(target / "checkpoint.json")
    digest = V3._sha256_bytes(damaged)
    checkpoint.update(
        {
            "committed_trace_bytes": len(damaged),
            "committed_trace_sha256": digest,
            "trace_bytes": len(damaged),
            "traces_jsonl_sha256": digest,
            "record_count": 1,
        }
    )
    error_type, error_match = _expected_semantic_rejection(mutation)
    with pytest.raises(error_type, match=error_match):
        V3._verify_trace_file(
            trace_path,
            checkpoint,
            dataset="plane",
            selected_rows={0: row},
            selected_indices=[0],
            zero_certificate_sha256=checkpoint["zero_certificate_sha256"],
            polynomial_certificate_sha256=checkpoint[
                "polynomial_certificate_sha256"
            ],
        )


def test_v3_exact_zero_certificate_remains_admissible():
    fixture = next(
        item
        for item in V3.oracle_module("noncavity_fixtures").noncavity_fixtures()
        if item["family"] == "gaussian" and item["target_kind"] == "center"
    )
    reference, metadata = V3.oracle_module(
        "noncavity_reference"
    ).noncavity_reference(fixture)
    row = {
        "fixture": V3._jsonable(fixture),
        "reference": V3._jsonable(reference),
        "metadata": V3._jsonable(metadata),
    }
    certificate = {
        "dataset": V3.DATASETS["noncavity"],
        "fixture": row["fixture"],
        "observable": "acceleration",
        "certificate": "even radial center gradient",
        "original_uncertainty_kind": "estimate",
        "original_used_uncertainty_si": 0.0,
        "frozen_labels_and_values_unchanged": True,
    }
    record = V3.record_row(
        "noncavity",
        row,
        source_row_index=0,
        zero_certificates=[certificate],
        zero_certificate_source="frozen-zero-estimate-certificates-integrated.json",
        zero_certificate_sha256=V3.EXPECTED_ZERO_CERTIFICATES_SHA256,
        polynomial_certificate_sha256=V3.EXPECTED_POLYNOMIAL_CERTIFICATES_SHA256,
    )
    record["process_independence"] = _worker_evidence()

    V3._validate_trace_record(
        record,
        "noncavity",
        0,
        row,
        zero_certificates=[certificate],
        zero_certificate_sha256=V3.EXPECTED_ZERO_CERTIFICATES_SHA256,
        polynomial_certificate_sha256=V3.EXPECTED_POLYNOMIAL_CERTIFICATES_SHA256,
    )
    result = AUDIT_V3.audit_record(
        "noncavity", 0, row, record, zero=[certificate]
    )

    assert record["observable_traces"]["acceleration"][
        "zero_certificate_adjudication"
    ]["adjudicated_zero_change_trends"] == [
        "derivative_trend",
        "quadrature_trend",
    ]
    assert result["status"] in {"reproduced", "unresolved_trend"}

    out_of_scope = json.loads(json.dumps(certificate))
    out_of_scope["observable"] = "potential"
    with pytest.raises(ValueError, match="invalid zero certificate"):
        V3._validate_trace_record(
            record,
            "noncavity",
            0,
            row,
            zero_certificates=[out_of_scope],
            zero_certificate_sha256=V3.EXPECTED_ZERO_CERTIFICATES_SHA256,
            polynomial_certificate_sha256=V3.EXPECTED_POLYNOMIAL_CERTIFICATES_SHA256,
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        AUDIT_V3.audit_record(
            "noncavity", 0, row, record, zero=[out_of_scope]
        )

    wrong_condition = json.loads(json.dumps(certificate))
    wrong_condition["original_used_uncertainty_si"] = 1.0
    with pytest.raises(ValueError, match="invalid zero certificate"):
        V3._validate_trace_record(
            record,
            "noncavity",
            0,
            row,
            zero_certificates=[wrong_condition],
            zero_certificate_sha256=V3.EXPECTED_ZERO_CERTIFICATES_SHA256,
            polynomial_certificate_sha256=V3.EXPECTED_POLYNOMIAL_CERTIFICATES_SHA256,
        )
    with pytest.raises(ValueError, match="zero annotation field mismatch"):
        AUDIT_V3.audit_record(
            "noncavity", 0, row, record, zero=[wrong_condition]
        )

    wrong_digest = json.loads(json.dumps(record))
    wrong_digest["observable_traces"]["acceleration"][
        "zero_certificate_adjudication"
    ]["supplement_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="unpinned zero certificate"):
        V3._validate_trace_record(
            wrong_digest,
            "noncavity",
            0,
            row,
            zero_certificates=[certificate],
            zero_certificate_sha256=V3.EXPECTED_ZERO_CERTIFICATES_SHA256,
            polynomial_certificate_sha256=V3.EXPECTED_POLYNOMIAL_CERTIFICATES_SHA256,
        )
    with pytest.raises(ValueError, match="annotation field mismatch"):
        AUDIT_V3.audit_record(
            "noncavity", 0, row, wrong_digest, zero=[certificate]
        )


def test_v3_exact_whole_case_certificate_positive_and_scope_negative_control():
    fixture = next(
        item
        for item in V3.oracle_module(
            "sphere_high_degree_fixtures"
        ).sphere_high_degree_fixtures()
        if item.target_kind == "center"
    )
    fixture_json = V3._jsonable(fixture)
    output, metadata = V3._invoke_original_entrypoint("high_sphere", fixture_json)
    row = {
        "fixture": fixture_json,
        "reference": V3._jsonable(output),
        "metadata": V3._jsonable(metadata),
    }
    record = V3.record_row(
        "high_sphere",
        row,
        source_row_index=0,
        zero_certificate_sha256=V3.EXPECTED_ZERO_CERTIFICATES_SHA256,
        polynomial_certificate_sha256=None,
    )
    record["process_independence"] = _worker_evidence()

    V3._validate_trace_record(
        record,
        "high_sphere",
        0,
        row,
        zero_certificate_sha256=V3.EXPECTED_ZERO_CERTIFICATES_SHA256,
        polynomial_certificate_sha256=None,
    )
    audit_record = json.loads(json.dumps(record))
    audit_record["zero_certificate_supplement_sha256"] = AUDIT_V3.ZERO_SHA256
    audit = AUDIT_V3.audit_record("high_sphere", 0, row, audit_record)

    assert metadata["exact_zero_certificate"]
    assert record["original_entrypoint"]["captured_calls"] == []
    assert all(
        trace["status"] == "exact_whole_case_certificate"
        for trace in record["observable_traces"].values()
    )
    assert audit["semantic_validation"] == "SEMANTIC_VALID"
    assert audit["status"] == "reproduced"
    assert audit["scientific_admission"] == "NOT_ADMITTED"

    binding = {
        "run_id": "unit-whole-case-run",
        "dataset": "high_sphere",
        "source_row_index": 0,
        "observer_sha256": AUDIT_V3.OBSERVER_SHA256,
        "auditor_sha256": AUDIT_V3.sha256(AUDIT_V3_SCRIPT),
        "oracle_module_sha256": AUDIT_V3.EXPECTED_ORACLE_SHA256,
        "frozen_reference_sha256": AUDIT_V3.EXPECTED_REFERENCE_SHA256[
            "high_sphere"
        ],
        "fixture_sha256": V3._sha256_bytes(
            V3._canonical_json(row["fixture"]).encode("utf-8")
        ),
        "certificate_sha256": {"zero": AUDIT_V3.ZERO_SHA256, "polynomial": None},
        "environment": {"kind": "unit-test-only"},
        "numeric_representation": {"routes": {"potential": "exact_whole_case_certificate"}},
    }

    class MockAnchor:
        def verify_commitment(self, _lookup_key, expected_document):
            return {
                "trust_root_verified": True,
                "document": expected_document,
                "anchor_id": "unit-test-only-mock",
            }

    trusted_audit = AUDIT_V3.audit_record(
        "high_sphere",
        0,
        row,
        audit_record,
        provenance_provider=MockAnchor(),
        provenance_binding=binding,
    )
    assert trusted_audit["provenance_validation"]["status"] == "PROVENANCE_VERIFIED"
    assert trusted_audit["scientific_admission"] == "ADMITTED_WITH_DECLARED_SCOPE"

    invalid_scope = json.loads(json.dumps(record))
    invalid_scope["observable_traces"]["gradient"]["certificate"] = (
        "certificate for a different fixture"
    )
    with pytest.raises(ValueError, match="unpinned whole-case certificate"):
        V3._validate_trace_record(
            invalid_scope,
            "high_sphere",
            0,
            row,
            zero_certificate_sha256=V3.EXPECTED_ZERO_CERTIFICATES_SHA256,
            polynomial_certificate_sha256=None,
        )
    with pytest.raises(ValueError, match="whole-case proof mismatch"):
        AUDIT_V3.audit_record("high_sphere", 0, row, invalid_scope)

def test_v3_rigorous_bound_override_remains_admissible():
    fixtures = list(V3.oracle_module("source_fixtures").source_fixtures())
    fixture = next(
        item
        for item in fixtures
        if item["family"] == "finite_temperature"
        and item["shape"] == "cuboid"
        and item["target_kind"] == "center"
    )
    reference, metadata = V3.oracle_module("source_reference").source_reference(
        fixture
    )
    row = {
        "fixture": V3._jsonable(fixture),
        "reference": V3._jsonable(reference),
        "metadata": V3._jsonable(metadata),
    }
    record = V3.record_row(
        "source",
        row,
        source_row_index=12,
        zero_certificate_sha256=V3.EXPECTED_ZERO_CERTIFICATES_SHA256,
        polynomial_certificate_sha256=V3.EXPECTED_POLYNOMIAL_CERTIFICATES_SHA256,
    )
    record["process_independence"] = _worker_evidence()

    V3._validate_trace_record(
        record,
        "source",
        12,
        row,
        zero_certificate_sha256=V3.EXPECTED_ZERO_CERTIFICATES_SHA256,
        polynomial_certificate_sha256=V3.EXPECTED_POLYNOMIAL_CERTIFICATES_SHA256,
    )
    result = AUDIT_V3.audit_record("source", 12, row, record)

    assert record["observable_traces"]["acceleration"]["status"] == (
        "certified_override"
    )
    assert result["routes"]["acceleration"] == "certified_override"

    wrong_reason = json.loads(json.dumps(record))
    wrong_reason["observable_traces"]["acceleration"]["derivative_trend"][
        "reason"
    ] = "the frozen output is an exact symmetry/parity certificate"
    with pytest.raises(ValueError, match="not_applicable reason"):
        V3._validate_trace_record(
            wrong_reason,
            "source",
            12,
            row,
            zero_certificate_sha256=V3.EXPECTED_ZERO_CERTIFICATES_SHA256,
            polynomial_certificate_sha256=V3.EXPECTED_POLYNOMIAL_CERTIFICATES_SHA256,
        )
    with pytest.raises(ValueError, match="N/A reason mismatch"):
        AUDIT_V3.audit_record("source", 12, row, wrong_reason)


def test_v3_polynomial_quadrature_certificate_remains_admissible():
    fixture = list(V3.oracle_module("noncavity_fixtures").noncavity_fixtures())[19]
    reference, metadata = V3.oracle_module(
        "noncavity_reference"
    ).noncavity_reference(fixture)
    row = {
        "fixture": V3._jsonable(fixture),
        "reference": V3._jsonable(reference),
        "metadata": V3._jsonable(metadata),
    }
    certificate = {
        "dataset": "noncavity",
        "source_row_index": 19,
        "fixture": row["fixture"],
        "certificate_kind": "retarded_cubic_pulse_polynomial_panels",
        "covered_observables": ["potential", "acceleration", "gradient"],
        "covered_original_stencils": V3.POLYNOMIAL_CERTIFICATE_COVERED_STENCILS,
        "certified_error": V3.POLYNOMIAL_CERTIFICATE_CERTIFIED_ERROR,
        "frozen_values_uncertainty_labels_unchanged": True,
    }
    record = V3.record_row(
        "noncavity",
        row,
        source_row_index=19,
        zero_certificate_sha256=V3.EXPECTED_ZERO_CERTIFICATES_SHA256,
        polynomial_certificates=[certificate],
        polynomial_certificate_source="polynomial-quadrature-certificates-integrated.json",
        polynomial_certificate_sha256=V3.EXPECTED_POLYNOMIAL_CERTIFICATES_SHA256,
        polynomial_verifier_script_sha256=V3.EXPECTED_POLYNOMIAL_VERIFIER_SCRIPT_SHA256,
    )
    record["process_independence"] = _worker_evidence()

    V3._validate_trace_record(
        record,
        "noncavity",
        19,
        row,
        zero_certificate_sha256=V3.EXPECTED_ZERO_CERTIFICATES_SHA256,
        polynomial_certificates=[certificate],
        polynomial_certificate_sha256=V3.EXPECTED_POLYNOMIAL_CERTIFICATES_SHA256,
    )
    result = AUDIT_V3.audit_record(
        "noncavity", 19, row, record, polynomial=[certificate]
    )

    adjudications = [
        trace.get("polynomial_quadrature_certificate_adjudication")
        for trace in record["observable_traces"].values()
    ]
    assert any(adjudications)
    assert result["status"] == record["status"]
    assert result["unresolved_trends"] == record["unresolved_trends"]
    for name, trace in record["observable_traces"].items():
        if trace.get("polynomial_quadrature_certificate_adjudication"):
            assert f"{name}/quadrature_trend" not in result["unresolved_trends"]

    out_of_scope = json.loads(json.dumps(certificate))
    out_of_scope["source_row_index"] = 18
    with pytest.raises(ValueError, match="out-of-scope polynomial certificate"):
        V3._validate_trace_record(
            record,
            "noncavity",
            19,
            row,
            zero_certificate_sha256=V3.EXPECTED_ZERO_CERTIFICATES_SHA256,
            polynomial_certificates=[out_of_scope],
            polynomial_certificate_sha256=V3.EXPECTED_POLYNOMIAL_CERTIFICATES_SHA256,
        )
    with pytest.raises(ValueError, match="certificate identity mismatch"):
        AUDIT_V3.audit_record(
            "noncavity", 19, row, record, polynomial=[out_of_scope]
        )

    wrong_condition = json.loads(json.dumps(certificate))
    wrong_condition["certified_error"] = "total numerical error fully certified"
    with pytest.raises(ValueError, match="out-of-scope polynomial certificate"):
        V3._validate_trace_record(
            record,
            "noncavity",
            19,
            row,
            zero_certificate_sha256=V3.EXPECTED_ZERO_CERTIFICATES_SHA256,
            polynomial_certificates=[wrong_condition],
            polynomial_certificate_sha256=V3.EXPECTED_POLYNOMIAL_CERTIFICATES_SHA256,
        )
    with pytest.raises(ValueError, match="annotation field mismatch"):
        AUDIT_V3.audit_record(
            "noncavity", 19, row, record, polynomial=[wrong_condition]
        )

    wrong_digest = json.loads(json.dumps(record))
    for trace in wrong_digest["observable_traces"].values():
        annotation = trace.get("polynomial_quadrature_certificate_adjudication")
        if annotation is not None:
            annotation["supplement_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="invalid polynomial certificate"):
        V3._validate_trace_record(
            wrong_digest,
            "noncavity",
            19,
            row,
            zero_certificate_sha256=V3.EXPECTED_ZERO_CERTIFICATES_SHA256,
            polynomial_certificates=[certificate],
            polynomial_certificate_sha256=V3.EXPECTED_POLYNOMIAL_CERTIFICATES_SHA256,
        )
    with pytest.raises(ValueError, match="annotation field mismatch"):
        AUDIT_V3.audit_record(
            "noncavity", 19, row, wrong_digest, polynomial=[certificate]
        )
