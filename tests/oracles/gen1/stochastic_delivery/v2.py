"""Selected v2 delivery: six accepted comparisons plus an A2 closure record.

This profile preserves the historical v1 reader while representing the closed,
resource-limited A2 state without manufacturing a comparison or transfer.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import a2, reader
from .a2 import verify_native_reference
from .archive import EvidenceError, validate_pin

SCHEMA = "gen1-stochastic-selected-delivery-v2"
FAMILIES = ("B1", "R1", "B2", "A1", "A3", "T1")
GENERIC_ARTIFACTS = {
    "budget",
    "decision_kernel",
    "dependency_closure",
    "environment",
    "family_procedure",
    "fixture",
    "freeze",
    "procedure",
    "production_source_admission",
    "purpose",
    "records",
    "reference_admission",
    "seal",
    "source_scales",
}
DESCRIPTOR_FIELDS = {
    "schema",
    "accepted_families",
    "history",
    "final_identity_boundary",
    "a2_closure",
}
CLOSURE_FIELDS = {
    "decision",
    "decision_seal",
    "reference",
    "preflight",
    "ledger_snapshot",
    "predecessor",
    "preserved_predecessor",
    "preserved_seals",
}
REFERENCE_FIELDS = {"admission", "root_receipt", "artifact_map", "canonical_pins"}
PREFLIGHT_FIELDS = {"artifact", "snapshot"}
SNAPSHOT_FIELDS = {
    "aggregate_seconds",
    "candidate",
    "candidate_artifacts",
    "candidate_keys",
    "candidate_pins",
    "candidate_status_fields",
    "command_packet",
    "completion_snapshot",
    "failed_closure_runner_receipt",
    "issued_spec",
    "native_completed",
    "native_summary",
    "no_finalization_repeated",
    "observer_completed",
    "original_reservation",
    "production_api_calls",
    "production_api_calls_basis",
    "remaining_seconds",
    "schema_version",
    "scientific_acceptance",
    "scope",
    "trusted_span",
}
SNAPSHOT_PIN_KEYS = {
    "budget_sha256",
    "comparator_commit",
    "comparator_tree",
    "decision_kernel_sha256",
    "dependency_closure_sha256",
    "environment_sha256",
    "family",
    "family_procedure_sha256",
    "fixture_sha256",
    "freeze_sha256",
    "policy_sha256",
    "procedure_sha256",
    "production_commit",
    "production_source_admission_sha256",
    "production_tree",
    "purpose_sha256",
    "records_sha256",
    "reference_admission_sha256",
    "seal_sha256",
    "source_scales_sha256",
}
SNAPSHOT_CANDIDATE_KEYS = ["evidence", "family", "phase", "pins", "schema", "status"]
SNAPSHOT_CANDIDATE_ARTIFACT_KIND = "a2_preflight_candidate_artifact"
HUMAN_CLOSURE_FIELDS = {
    "schema",
    "created_at_utc",
    "decision",
    "authority",
    "predecessor",
    "preserved_predecessor",
    "preserved_seals",
    "family",
    "family_status",
    "resource_status",
    "reference",
    "preflight",
    "production_comparison",
    "production_acceptance",
    "resource_accounting",
    "conditioning_failure",
    "physics_failure",
    "reason",
    "frozen_contract",
    "immutable_scope",
    "ledger_snapshot_at_closure",
    "accepted_families_preserved",
    "accepted_family_recalculation_for_a2_closure",
    "campaign_a2_work_closed",
    "future_unfinished_work",
    "continuation",
    "creation_rule",
}
REFERENCE_COUNTS = {
    "auxiliary": 30,
    "duplicate": 0,
    "expected": 84,
    "missing": 0,
    "resolved": 84,
    "science": 54,
}
CAMPAIGN_WORK_CLOSED = [
    "resource-ledger refinement",
    "additional 39-candidate admission",
    "71-liability reconstruction",
    "new-budget search",
    "production comparison",
]


def _fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != expected:
        raise EvidenceError(label + " schema mismatch")


def _same_identity(actual, embedded, label):
    if (
        not isinstance(embedded, dict)
        or set(embedded)
        not in (
            {"path", "sha256"},
            {"path", "sha256", "kind"},
            {"path", "sha256", "preserved"},
        )
        or (actual["path"], actual["sha256"]) != (embedded["path"], embedded["sha256"])
    ):
        raise EvidenceError(label + " raw pin identity differs")


def _typed_pin(pin, label):
    try:
        validate_pin(pin)
    except EvidenceError as error:
        raise EvidenceError(label + " pin invalid") from error
    return pin


def _snapshot_ref(value, label):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise EvidenceError(label + " reference schema mismatch")
    if (
        not isinstance(value["path"], str)
        or not Path(value["path"]).is_absolute()
        or not isinstance(value["sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
    ):
        raise EvidenceError(label + " reference invalid")


def _verify_external_seal(store, pin, artifact_pin, *, current=False):
    seal = store.json(pin)
    fields = {
        "schema",
        "artifact",
        "sha256",
        "filesystem_reread",
        "creation_rule",
    }
    fields.add(
        "predecessor_and_predecessor_seal_verification"
        if current
        else "known_answer_vectors"
    )
    if not current:
        fields.add("note")
    if (
        set(seal) != fields
        or seal["schema"] != "gragra.external-record-seal.v1"
        or seal["artifact"] != artifact_pin["path"]
        or seal["sha256"] != artifact_pin["sha256"]
        or not isinstance(seal["filesystem_reread"], str)
        or not seal["filesystem_reread"]
    ):
        raise EvidenceError("external decision seal subject differs")
    if current and (
        seal["filesystem_reread"] != "PASS"
        or seal["predecessor_and_predecessor_seal_verification"] != "PASS"
    ):
        raise EvidenceError("current closure decision seal is incomplete")
    return seal


def _verify_decision_and_history(store, closure):
    decision_pin = _typed_pin(closure["decision"], "A2 closure decision")
    seal_pin = _typed_pin(closure["decision_seal"], "A2 closure seal")
    predecessor_pin = _typed_pin(closure["predecessor"], "A2 STOP-2 predecessor")
    preserved_pin = _typed_pin(
        closure["preserved_predecessor"], "preserved A2 STOP-2 predecessor"
    )
    preserved_seals = closure["preserved_seals"]
    if not isinstance(preserved_seals, list) or len(preserved_seals) != 2:
        raise EvidenceError("both preserved A2 STOP-2 seals required")
    for pin in preserved_seals:
        _typed_pin(pin, "preserved A2 STOP-2 seal")
    decision = store.json(decision_pin)
    _fields(decision, HUMAN_CLOSURE_FIELDS, "A2 human closure decision")
    if (
        decision["schema"] != "gragra.a2.resource-human-decision.v2"
        or decision["decision"] != "HUMAN_APPROVED_OPTION_A"
        or decision["family"] != "A2"
        or decision["family_status"] != "PARTIAL"
        or decision["resource_status"] != "RESOURCE_LIMITED"
        or decision["production_comparison"] != "PRODUCTION_COMPARISON_NOT_RUN"
        or decision["production_acceptance"] != "NOT_ESTABLISHED"
        or decision["conditioning_failure"] != "NOT_CLAIMED"
        or decision["physics_failure"] != "NOT_CLAIMED"
        or not isinstance(decision["reason"], str)
        or not decision["reason"]
    ):
        raise EvidenceError("A2 closure decision status differs")
    authority = decision["authority"]
    _fields(
        authority,
        {"source", "previous_paseo_session", "quote", "scope"},
        "A2 closure authority",
    )
    if (
        authority["source"] != "Current human instruction in takeover session"
        or not isinstance(authority["previous_paseo_session"], str)
        or not authority["previous_paseo_session"]
        or authority["quote"] != "Option A を承認します。"
        or authority["scope"]
        != (
            "Close A2 under existing frozen contracts and proceed autonomously "
            "through final PR #159 completion gates."
        )
    ):
        raise EvidenceError("A2 Option A authority scope differs")
    _same_identity(predecessor_pin, decision["predecessor"], "A2 STOP-2 predecessor")
    _same_identity(
        preserved_pin,
        decision["preserved_predecessor"],
        "preserved A2 STOP-2 predecessor",
    )

    for field in ("reference", "preflight", "ledger_snapshot_at_closure"):
        if field == "reference":
            _fields(
                decision[field],
                {
                    "status",
                    "admission",
                    "root_receipt",
                    "counts",
                    "new_reference_execution",
                },
                "A2 closure reference",
            )
        elif field == "preflight":
            _fields(
                decision[field],
                {"status", "artifact", "snapshot", "qualification"},
                "A2 closure preflight",
            )
    _same_identity(
        closure["ledger_snapshot"],
        decision["ledger_snapshot_at_closure"],
        "A2 ledger snapshot",
    )

    seal = _verify_external_seal(store, seal_pin, decision_pin, current=True)
    predecessor = store.json(predecessor_pin)
    preserved = store.json(preserved_pin)
    for record in (predecessor, preserved):
        if (
            record.get("schema") != "gragra.a2.resource-human-decision.v1"
            or record.get("decision") != "HUMAN_DECISION_REQUIRED"
            or record.get("stop_rule") != "STOP-2"
            or record.get("execution_hold") != "ALL_NEW_CAMPAIGN_SUBPROCESSES_STOPPED"
        ):
            raise EvidenceError("historical A2 STOP-2 record identity differs")
    if (
        predecessor.get("predecessor", {}).get("path") != preserved_pin["path"]
        or predecessor.get("predecessor", {}).get("sha256") != preserved_pin["sha256"]
        or predecessor.get("predecessor", {}).get("preserved") is not True
    ):
        raise EvidenceError("STOP-2 v2 predecessor does not preserve v1")
    expected_seals = decision["preserved_seals"]
    if not isinstance(expected_seals, list) or len(expected_seals) != 2:
        raise EvidenceError("closure must retain both historical STOP-2 seals")
    records = (preserved_pin, predecessor_pin)
    if any(
        (pin["path"], pin["sha256"]) != (expected["path"], expected["sha256"])
        for pin, expected in zip(preserved_seals, expected_seals, strict=True)
    ):
        raise EvidenceError("preserved STOP-2 seal pins differ")
    for seal_pin_item, record_pin in zip(preserved_seals, records, strict=True):
        _verify_external_seal(store, seal_pin_item, record_pin)
        store.read(record_pin)
    store.read(closure["ledger_snapshot"])
    return decision, seal


def _verify_reference(store, closure, decision):
    reference = closure["reference"]
    _fields(reference, REFERENCE_FIELDS, "A2 reference descriptor")
    for name, pin in reference.items():
        _typed_pin(pin, "A2 reference " + name)
    _same_identity(
        reference["admission"],
        decision["reference"]["admission"],
        "A2 reference admission",
    )
    _same_identity(
        reference["root_receipt"],
        decision["reference"]["root_receipt"],
        "A2 reference receipt",
    )

    admission = store.json(reference["admission"])
    _fields(
        admission,
        {"schema", "family", "decision", "pins", "reviews"},
        "A2 reference admission",
    )
    if (
        admission["schema"] != "gen1-stochastic-reference-admission-v1"
        or admission["family"] != "A2"
        or admission["decision"] != "PASS"
        or decision["reference"]["status"]
        != "AVAILABLE_SEALED_AS_ESTABLISHED_BY_EXISTING_EVIDENCE"
        or decision["reference"]["counts"] != REFERENCE_COUNTS
        or decision["reference"]["new_reference_execution"] is not False
        or decision["preflight"]["status"]
        != "COMPLETED_AS_ESTABLISHED_BY_EXISTING_EVIDENCE"
        or decision["preflight"]["qualification"]
        != (
            "Completed metadata preflight does not imply production authorization "
            "or acceptance."
        )
    ):
        raise EvidenceError("A2 retained reference admission differs")

    receipt = store.json(reference["root_receipt"])
    receipt_fields = {
        "admission",
        "artifacts",
        "canonical_pins",
        "controller_count_clarification",
        "controller_script",
        "counts",
        "decision",
        "gate",
        "new_native_driver_must_independently_validate_before_production_authorization",
        "production_acceptance",
        "production_comparison_authorized",
        "source_projection",
    }
    _fields(receipt, receipt_fields, "A2 reference root receipt")
    if (
        receipt["admission"] != reference["admission"]
        or receipt["artifacts"] != reference["artifact_map"]
        or receipt["canonical_pins"] != reference["canonical_pins"]
        or receipt["decision"] != "ACCEPTED_FROZEN_REFERENCE_SET"
        or receipt["counts"] != REFERENCE_COUNTS
        or receipt["production_acceptance"] != "NOT_ESTABLISHED"
        or receipt["production_comparison_authorized"] is not False
    ):
        raise EvidenceError("A2 reference root receipt linkage/status differs")
    _typed_pin(receipt["controller_script"], "A2 reference receipt controller")
    store.read(receipt["controller_script"])

    artifacts = store.json(reference["artifact_map"])
    _fields(
        artifacts,
        {
            "budget",
            "environment",
            "family_procedure",
            "fixture",
            "freeze",
            "records",
            "seal",
        },
        "native A2 reference artifact map",
    )
    for name, pin in artifacts.items():
        _typed_pin(pin, "native A2 reference " + name)
        store.read(pin)
    canonical_pins = store.json(reference["canonical_pins"])
    if canonical_pins != admission["pins"]:
        raise EvidenceError("A2 canonical reference pins differ from admission")

    reviews = admission["reviews"]
    if not isinstance(reviews, dict) or set(reviews) != {
        "science",
        "numerical",
        "provenance",
    }:
        raise EvidenceError("three A2 reference review roles required")
    reviewers = set()
    for role, pin in reviews.items():
        _typed_pin(pin, "A2 reference review")
        review = store.json(pin)
        _fields(
            review,
            {
                "schema",
                "family",
                "role",
                "reviewer_id",
                "decision",
                "pins",
                "original_review",
            },
            "A2 native reference review",
        )
        if (
            review["schema"] != "gen1-stochastic-reference-review-v1"
            or review["family"] != "A2"
            or review["role"] != role
            or review["decision"] != "PASS"
            or review["pins"] != admission["pins"]
            or not isinstance(review["reviewer_id"], str)
            or not review["reviewer_id"]
            or review["reviewer_id"] in reviewers
        ):
            raise EvidenceError("A2 reference review scope differs")
        reviewers.add(review["reviewer_id"])
        _typed_pin(review["original_review"], "A2 original reference review")
        store.read(review["original_review"])

    native_summary = verify_native_reference(store, artifacts)
    if native_summary != admission["pins"] or native_summary != canonical_pins:
        raise EvidenceError("sealed A2 native reference differs from admitted pins")
    return native_summary


def _verify_preflight(store, closure, decision, reference_summary):
    preflight = closure["preflight"]
    _fields(preflight, PREFLIGHT_FIELDS, "A2 preflight descriptor")
    for name, pin in preflight.items():
        _typed_pin(pin, "A2 preflight " + name)
    _same_identity(
        preflight["artifact"],
        decision["preflight"]["artifact"],
        "A2 preflight artifact",
    )
    _same_identity(
        preflight["snapshot"],
        decision["preflight"]["snapshot"],
        "A2 preflight snapshot",
    )

    artifact = store.json(preflight["artifact"])
    _fields(
        artifact, {"pins", "production_api_calls", "status"}, "A2 metadata preflight"
    )
    if (
        artifact["status"] != "PASS_NO_COMPUTATION"
        or type(artifact["production_api_calls"]) is not int
        or artifact["production_api_calls"] != 0
    ):
        raise EvidenceError("A2 metadata preflight is not a no-computation result")
    if (
        not isinstance(artifact["pins"], dict)
        or set(artifact["pins"]) != SNAPSHOT_PIN_KEYS
    ):
        raise EvidenceError("A2 metadata preflight candidate pin set differs")

    snapshot = a2.operational_json(store.read(preflight["snapshot"]))
    _fields(snapshot, SNAPSHOT_FIELDS, "A2 completed candidate snapshot")
    if (
        snapshot["schema_version"] != 1
        or snapshot["scope"]
        != (
            "actual metadata artifact and completed accounting snapshot; "
            "no scientific acceptance"
        )
        or snapshot["scientific_acceptance"] is not False
        or type(snapshot["production_api_calls"]) is not int
        or snapshot["production_api_calls"] != 0
        or snapshot["native_completed"] is not True
        or snapshot["observer_completed"] is not True
        or snapshot["no_finalization_repeated"] is not True
        or snapshot["candidate_status_fields"] != {"status": "REVIEW_CANDIDATE"}
        or snapshot["candidate_keys"] != SNAPSHOT_CANDIDATE_KEYS
        or snapshot["candidate_pins"] != artifact["pins"]
        or snapshot["production_api_calls_basis"]
        != (
            "exact command packet declares zero; executed reviewed builder is "
            "metadata-only; this is not a scientific result or dynamic API "
            "instrumentation claim"
        )
    ):
        raise EvidenceError("A2 completed metadata snapshot status/pins differ")
    _snapshot_ref(snapshot["candidate"], "A2 candidate")
    candidate_artifacts = snapshot["candidate_artifacts"]
    if not isinstance(candidate_artifacts, list) or not candidate_artifacts:
        raise EvidenceError("A2 candidate artifact list required")
    seen = set()
    for item in candidate_artifacts:
        _snapshot_ref(item, "A2 candidate artifact")
        identity = (item["path"], item["sha256"])
        if identity in seen:
            raise EvidenceError("duplicate A2 candidate artifact")
        seen.add(identity)
    if (snapshot["candidate"]["path"], snapshot["candidate"]["sha256"]) not in seen:
        raise EvidenceError("A2 candidate is absent from completed artifact snapshot")
    if (preflight["artifact"]["path"], preflight["artifact"]["sha256"]) not in seen:
        raise EvidenceError(
            "actual A2 preflight is absent from completed artifact snapshot"
        )
    # Retain each direct candidate-artifact member as opaque historical bytes.
    # Other snapshot references are not recursively followed or interpreted.
    for index, item in enumerate(candidate_artifacts):
        candidate_pin = {
            "path": item["path"],
            "sha256": item["sha256"],
            "kind": SNAPSHOT_CANDIDATE_ARTIFACT_KIND,
        }
        _typed_pin(candidate_pin, f"A2 historical candidate artifact {index}")
        store.read(candidate_pin)

    candidate_pins = artifact["pins"]
    frozen = reference_summary["frozen_artifacts_sha256"]
    if (
        candidate_pins["family"] != "A2"
        or candidate_pins["reference_admission_sha256"]
        != closure["reference"]["admission"]["sha256"]
        or candidate_pins["records_sha256"] != reference_summary["records_sha256"]
        or candidate_pins["seal_sha256"] != reference_summary["seal_sha256"]
        or candidate_pins["environment_sha256"]
        != reference_summary["environment_sha256"]
        or candidate_pins["budget_sha256"] != frozen["budget"]
        or candidate_pins["family_procedure_sha256"] != frozen["family_procedure"]
        or candidate_pins["fixture_sha256"] != frozen["fixture"]
        or candidate_pins["freeze_sha256"] != frozen["freeze"]
    ):
        raise EvidenceError("A2 preflight candidate/native reference pins differ")
    # aggregate_seconds and remaining_seconds are parsed only as finite
    # operational metadata. They are not an admitted budget calculation.
    return {
        "status": artifact["status"],
        "production_api_calls": 0,
        "snapshot_scope": snapshot["scope"],
        "scientific_acceptance": False,
        "native_completed": True,
        "observer_completed": True,
        "original_finalization_repeated": False,
        "historical_candidate_artifact_raw_retrieval_count": len(candidate_artifacts),
        "candidate_artifact_scope": "DIRECT_SNAPSHOT_REFERENCES_RAW_BYTES_ONLY",
    }


def verify_a2_closure(store, descriptor):
    """Verify an A2 resource-limited closure, never a production comparison."""
    _fields(descriptor, CLOSURE_FIELDS, "A2 closure descriptor")
    for name in (
        "decision",
        "decision_seal",
        "ledger_snapshot",
        "predecessor",
        "preserved_predecessor",
    ):
        _typed_pin(descriptor[name], "A2 closure " + name)
    decision, _decision_seal = _verify_decision_and_history(store, descriptor)
    reference_summary = _verify_reference(store, descriptor, decision)
    preflight_summary = _verify_preflight(
        store, descriptor, decision, reference_summary
    )

    resource_accounting = decision["resource_accounting"]
    _fields(
        resource_accounting,
        {
            "conservative_remaining_budget",
            "budget_exhaustion",
            "unresolved_candidates",
            "unresolved_liability_population",
            "unresolved_values_assumed_zero",
            "reviewed_local_bounds_seconds",
            "local_bounds_adopted_into_ledger_by_this_decision",
            "production_reservation_seconds",
            "production_reservation",
        },
        "A2 resource closure accounting",
    )
    if (
        resource_accounting["conservative_remaining_budget"] != "NOT_ESTABLISHED"
        or resource_accounting["budget_exhaustion"] != "NOT_PROVEN"
        or resource_accounting["unresolved_candidates"] != 39
        or resource_accounting["unresolved_liability_population"] != 71
        or resource_accounting["unresolved_values_assumed_zero"] is not False
        or resource_accounting["reviewed_local_bounds_seconds"] != ["0.4", "0.7"]
        or resource_accounting["local_bounds_adopted_into_ledger_by_this_decision"]
        is not False
        or resource_accounting["production_reservation_seconds"] != 720
        or resource_accounting["production_reservation"] != "NOT_AUTHORIZED"
    ):
        raise EvidenceError("A2 resource closure accounting status differs")
    _fields(
        decision["frozen_contract"],
        {"aggregate_subprocess_wall_seconds", "science_changed", "resource_changed"},
        "A2 frozen contract",
    )
    if (
        decision["frozen_contract"]["aggregate_subprocess_wall_seconds"] != 21600
        or decision["frozen_contract"]["science_changed"] is not False
        or decision["frozen_contract"]["resource_changed"] is not False
        or decision["accepted_families_preserved"] != list(FAMILIES)
        or decision["accepted_family_recalculation_for_a2_closure"] is not False
        or decision["campaign_a2_work_closed"] != CAMPAIGN_WORK_CLOSED
        or decision["future_unfinished_work"].get("status")
        != "DEFERRED_OUTSIDE_CURRENT_PR159_CAMPAIGN"
        or decision["continuation"].get(
            "a2_production_acceptance_on_completion_critical_path"
        )
        is not False
    ):
        raise EvidenceError("A2 closure campaign/scope differs")
    return {
        "status": "A2_RESOURCE_LIMITED_CLOSURE_VERIFIED",
        "verification_scope": "SOFTWARE_EVIDENCE_GRAPH_ONLY",
        "family_status": "PARTIAL",
        "resource_status": "RESOURCE_LIMITED",
        "production_comparison": "PRODUCTION_COMPARISON_NOT_RUN",
        "production_acceptance": "NOT_ESTABLISHED",
        "conservative_remaining_budget": "NOT_ESTABLISHED",
        "budget_exhaustion": "NOT_PROVEN",
        "conditioning_failure": "NOT_CLAIMED",
        "physics_failure": "NOT_CLAIMED",
        "reference_counts": dict(REFERENCE_COUNTS),
        "reference_admission_verified": True,
        "reference_execution_repeated": False,
        "preflight": preflight_summary,
        "immutable_ledger_snapshot_verified": True,
        "historical_stop2_records_verified": True,
        "decision_seal_verified": True,
        "included_in_accepted_families": False,
        # The snapshot may carry historical operational time fields. This
        # profile deliberately makes no remaining-budget projection from them.
        "remaining_budget_seconds": None,
    }


def verify_selected_delivery_v2(store, descriptor):
    """Verify six accepted graphs plus the separate non-accepting A2 closure."""
    _fields(descriptor, DESCRIPTOR_FIELDS, "selected delivery v2 descriptor")
    if (
        descriptor["schema"] != SCHEMA
        or descriptor["final_identity_boundary"]
        != "EXTERNAL_ATTESTATION_AFTER_TRACKED_COMMIT"
    ):
        raise EvidenceError("selected delivery v2 identity/schema mismatch")
    if not isinstance(descriptor["accepted_families"], list) or [
        row.get("family") if isinstance(row, dict) else None
        for row in descriptor["accepted_families"]
    ] != list(FAMILIES):
        raise EvidenceError("exact six ordered accepted families required")
    if not isinstance(descriptor["history"], list):
        raise EvidenceError("selected delivery history must be a pin list")
    for historical in descriptor["history"]:
        _typed_pin(historical, "selected delivery history")
        store.read(historical)

    results = []
    for family in descriptor["accepted_families"]:
        _fields(family, {"family", "comparison", "transfer"}, "selected family")
        if family["family"] not in FAMILIES:
            raise EvidenceError("A2 closure cannot be an accepted family graph")
        for stage in family["comparison"].get("stages", []):
            policy = store.json(stage["policy"])
            if (
                family["family"] != "B1"
                and set(policy["artifacts"]) != GENERIC_ARTIFACTS
            ):
                raise EvidenceError("complete admitted generic input domains required")
        comparisons = reader.verify_selected_comparison(
            store, family["comparison"], family=family["family"]
        )
        pins = comparisons[-1]["pins"]
        actual_p = {
            "commit": pins["production_commit"],
            "tree": pins["production_tree"],
        }
        final_policy = store.json(family["comparison"]["stages"][-1]["policy"])
        actual_closure = final_policy["artifacts"]["dependency_closure"]
        if actual_closure["sha256"] != pins["dependency_closure_sha256"]:
            raise EvidenceError("actual comparison closure raw binding differs")
        transfer = reader.verify_transfer(
            store,
            family["transfer"],
            actual_p=actual_p,
            actual_closure=actual_closure,
        )
        results.append(
            {
                "family": family["family"],
                "actual_comparison_P": actual_p,
                "comparison_summary": comparisons[-1]["summary"],
                "phase_summaries": [
                    {"phase": result["phase"], "summary": result["summary"]}
                    for result in comparisons
                ],
                "transfer": transfer,
            }
        )
    a2_closure = verify_a2_closure(store, descriptor["a2_closure"])
    return {
        "status": "SIX_FAMILY_BYTE_GRAPH_VERIFIED",
        "families": results,
        "accepted_family_count": 6,
        "science_acceptance": {
            "accepted_families": list(FAMILIES),
            "a2_production_acceptance_established": False,
        },
        "a2_closure": a2_closure,
        "history_in_accepted_counts": False,
        "H_final_assigned": False,
    }
