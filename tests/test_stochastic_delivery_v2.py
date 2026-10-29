"""Selected v2 graphs keep the accepted six separate from A2 closure evidence."""

import hashlib
import json

import pytest

from tests.oracles.gen1.stochastic_delivery import reader, v2
from tests.oracles.gen1.stochastic_delivery.archive import EvidenceError
from tests.test_stochastic_delivery_evidence import MemoryEvidence

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


def _hex(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _pin(store, value):
    return store.add(value)


def _family_graphs(store):
    result = []
    for family in FAMILIES:
        closure_hash = _hex(f"{family} closure")
        policy = {
            "artifacts": {name: {"sha256": _hex(name)} for name in GENERIC_ARTIFACTS}
        }
        policy["artifacts"]["dependency_closure"] = {"sha256": closure_hash}
        result.append(
            {
                "family": family,
                "comparison": {"stages": [{"policy": _pin(store, policy)}]},
                "transfer": {"synthetic": family},
                "_expected_closure": closure_hash,
            }
        )
    return result


def _closure_record(
    *,
    admission_pin,
    receipt_pin,
    preflight_pin,
    snapshot_pin,
    ledger_pin,
    predecessor_pin,
    preserved_predecessor_pin,
    preserved_seals,
    changes=None,
):
    record = {
        "schema": "gragra.a2.resource-human-decision.v2",
        "created_at_utc": "2026-10-04T12:08:45.979402+00:00",
        "decision": "HUMAN_APPROVED_OPTION_A",
        "authority": {
            "source": "Current human instruction in takeover session",
            "previous_paseo_session": "retained-session-id",
            "quote": "Option A を承認します。",
            "scope": (
                "Close A2 under existing frozen contracts and proceed autonomously "
                "through final PR #159 completion gates."
            ),
        },
        "predecessor": {key: predecessor_pin[key] for key in ("path", "sha256")},
        "preserved_predecessor": {
            key: preserved_predecessor_pin[key] for key in ("path", "sha256")
        },
        "preserved_seals": [
            {key: pin[key] for key in ("path", "sha256")} for pin in preserved_seals
        ],
        "family": "A2",
        "family_status": "PARTIAL",
        "resource_status": "RESOURCE_LIMITED",
        "reference": {
            "status": "AVAILABLE_SEALED_AS_ESTABLISHED_BY_EXISTING_EVIDENCE",
            "admission": {key: admission_pin[key] for key in ("path", "sha256")},
            "root_receipt": {key: receipt_pin[key] for key in ("path", "sha256")},
            "counts": {
                "auxiliary": 30,
                "duplicate": 0,
                "expected": 84,
                "missing": 0,
                "resolved": 84,
                "science": 54,
            },
            "new_reference_execution": False,
        },
        "preflight": {
            "status": "COMPLETED_AS_ESTABLISHED_BY_EXISTING_EVIDENCE",
            "artifact": {key: preflight_pin[key] for key in ("path", "sha256")},
            "snapshot": {key: snapshot_pin[key] for key in ("path", "sha256")},
            "qualification": (
                "Completed metadata preflight does not imply production "
                "authorization or acceptance."
            ),
        },
        "production_comparison": "PRODUCTION_COMPARISON_NOT_RUN",
        "production_acceptance": "NOT_ESTABLISHED",
        "resource_accounting": {
            "conservative_remaining_budget": "NOT_ESTABLISHED",
            "budget_exhaustion": "NOT_PROVEN",
            "unresolved_candidates": 39,
            "unresolved_liability_population": 71,
            "unresolved_values_assumed_zero": False,
            "reviewed_local_bounds_seconds": ["0.4", "0.7"],
            "local_bounds_adopted_into_ledger_by_this_decision": False,
            "production_reservation_seconds": 720,
            "production_reservation": "NOT_AUTHORIZED",
        },
        "conditioning_failure": "NOT_CLAIMED",
        "physics_failure": "NOT_CLAIMED",
        "reason": (
            "The frozen resource/accounting contract did not establish a conservative "
            "remaining budget, so the 720-second production reservation could not be "
            "authorized. No A2 production comparison failure occurred and actual "
            "budget exhaustion is not proven."
        ),
        "frozen_contract": {
            "aggregate_subprocess_wall_seconds": 21600,
            "science_changed": False,
            "resource_changed": False,
        },
        "immutable_scope": ["frozen science contract", "resource ledger history"],
        "ledger_snapshot_at_closure": {
            key: ledger_pin[key] for key in ("path", "sha256")
        },
        "accepted_families_preserved": list(FAMILIES),
        "accepted_family_recalculation_for_a2_closure": False,
        "campaign_a2_work_closed": [
            "resource-ledger refinement",
            "additional 39-candidate admission",
            "71-liability reconstruction",
            "new-budget search",
            "production comparison",
        ],
        "future_unfinished_work": {
            "status": "DEFERRED_OUTSIDE_CURRENT_PR159_CAMPAIGN",
            "sequence": [
                "new resource/accounting procedure revision",
                "independent review",
                "bounded resumption",
            ],
            "prerequisite": (
                "Positive-liability enclosure or a lawful accounting procedure "
                "explicitly defined by the new revision."
            ),
        },
        "continuation": {
            "a2_production_acceptance_on_completion_critical_path": False,
            "authorized_steps": [
                "family-state finalization",
                "focused stochastic regression",
            ],
            "stop_conditions": ["existing H1-H5"],
            "pr_state_required": ["Open", "Draft", "NOT MERGED"],
        },
        "creation_rule": (
            "New immutable successor record. Existing records are never overwritten; "
            "later corrections require another successor."
        ),
    }
    record.update(changes or {})
    return record


def _seal(store, artifact_pin, *, current=False):
    seal = {
        "schema": "gragra.external-record-seal.v1",
        "artifact": artifact_pin["path"],
        "sha256": artifact_pin["sha256"],
        "creation_rule": "Exact immutable bytes retained.",
        "filesystem_reread": "PASS" if current else "NOT_RUN_AFTER_OS_HOLD",
    }
    if current:
        seal["predecessor_and_predecessor_seal_verification"] = "PASS"
    else:
        seal["known_answer_vectors"] = "SHA-256 empty and abc PASS"
        seal["note"] = "Preserved historical STOP-2 seal."
    return _pin(store, seal)


def _delivery_fixture(
    *,
    decision_changes=None,
    admission_changes=None,
    receipt_changes=None,
    canonical_changes=None,
    preflight_changes=None,
    snapshot_changes=None,
    accounting_changes=None,
):
    store = MemoryEvidence()
    families = _family_graphs(store)

    artifact_map = {}
    for name in (
        "budget",
        "environment",
        "family_procedure",
        "fixture",
        "freeze",
        "records",
        "seal",
    ):
        pin = _pin(store, {"reference_artifact": name})
        pin["kind"] = "native_reference_provenance"
        artifact_map[name] = pin
    artifact_map_pin = _pin(store, artifact_map)
    native_summary = {
        "family": "A2",
        "E_commit": "60bfc4045b48276c111e62bcd30159f071627d05",
        "E_tree": "7ffb8b23bd9224caf43074138c59fe0aa19d4413",
        "records_sha256": artifact_map["records"]["sha256"],
        "seal_sha256": artifact_map["seal"]["sha256"],
        "environment_sha256": artifact_map["environment"]["sha256"],
        "expected_keys_sha256": _hex("84 frozen A2 keys"),
        "frozen_artifacts_sha256": {
            name: artifact_map[name]["sha256"]
            for name in ("freeze", "fixture", "budget", "family_procedure")
        },
        "source_pin": {
            "head": "60bfc4045b48276c111e62bcd30159f071627d05",
            "tree": "7ffb8b23bd9224caf43074138c59fe0aa19d4413",
            "files_sha256": {
                "fixture": {"A2-fixtures-v1.json": artifact_map["fixture"]["sha256"]}
            },
        },
        "artifact_hashes": {
            "authorization.json": _hex("authorization"),
            "family-freeze.json": artifact_map["freeze"]["sha256"],
            "policy.json": artifact_map["fixture"]["sha256"],
            "purpose-records.json": _hex("purpose records"),
            "records.raw.jsonl": artifact_map["records"]["sha256"],
            "resource-ledger.json": _hex("native reference ledger"),
            "resource-result.json": _hex("resource result"),
            "stderr.jsonl": _hex("stderr"),
            "stdout.jsonl": artifact_map["records"]["sha256"],
        },
    }
    canonical = dict(native_summary)
    canonical.update(canonical_changes or {})
    canonical_pin = _pin(store, canonical)
    reviews = {}
    for role in ("science", "numerical", "provenance"):
        original_review = _pin(store, {"report": role})
        reviews[role] = _pin(
            store,
            {
                "schema": "gen1-stochastic-reference-review-v1",
                "family": "A2",
                "role": role,
                "reviewer_id": "/reviewer/" + role,
                "decision": "PASS",
                "pins": dict(native_summary),
                "original_review": original_review,
            },
        )
    admission = {
        "schema": "gen1-stochastic-reference-admission-v1",
        "family": "A2",
        "decision": "PASS",
        "pins": dict(native_summary),
        "reviews": reviews,
    }
    admission.update(admission_changes or {})
    admission_pin = _pin(store, admission)
    controller_script = _pin(store, {"builder": "reference metadata only"})
    controller_script["kind"] = "root_native_reference_admission_orchestration"
    receipt = {
        "admission": admission_pin,
        "artifacts": artifact_map_pin,
        "canonical_pins": canonical_pin,
        "controller_count_clarification": "retained existing controller evidence",
        "controller_script": controller_script,
        "counts": {
            "auxiliary": 30,
            "duplicate": 0,
            "expected": 84,
            "missing": 0,
            "resolved": 84,
            "science": 54,
        },
        "decision": "ACCEPTED_FROZEN_REFERENCE_SET",
        "gate": "Previously reviewed actual-reference pin maps",
        (
            "new_native_driver_must_independently_validate_"
            "before_production_authorization"
        ): True,
        "production_acceptance": "NOT_ESTABLISHED",
        "production_comparison_authorized": False,
        "source_projection": "Projection of retained E60 authorization",
    }
    receipt.update(receipt_changes or {})
    receipt_pin = _pin(store, receipt)

    candidate_pins = {
        "budget_sha256": artifact_map["budget"]["sha256"],
        "comparator_commit": "ad096eb09a41bffc132bda48f101d99f8446102c",
        "comparator_tree": "554b02da00a8c6a49bef96d3ca4f9d9c0a494b11",
        "decision_kernel_sha256": _hex("decision kernel"),
        "dependency_closure_sha256": _hex("dependency closure"),
        "environment_sha256": artifact_map["environment"]["sha256"],
        "family": "A2",
        "family_procedure_sha256": artifact_map["family_procedure"]["sha256"],
        "fixture_sha256": artifact_map["fixture"]["sha256"],
        "freeze_sha256": artifact_map["freeze"]["sha256"],
        "policy_sha256": artifact_map["fixture"]["sha256"],
        "procedure_sha256": _hex("execution procedure"),
        "production_commit": "377b40949cde0d3545199eb6f2d5b4e55824f78f",
        "production_source_admission_sha256": _hex("production source admission"),
        "production_tree": "8844ffade95fb97bd5006c8ea1dcf366416e14e1",
        "purpose_sha256": _hex("purpose"),
        "records_sha256": native_summary["records_sha256"],
        "reference_admission_sha256": admission_pin["sha256"],
        "seal_sha256": native_summary["seal_sha256"],
        "source_scales_sha256": _hex("source scales"),
    }
    preflight = {
        "status": "PASS_NO_COMPUTATION",
        "pins": dict(candidate_pins),
        "production_api_calls": 0,
    }
    preflight.update(preflight_changes or {})
    preflight_pin = _pin(store, preflight)
    candidate_path = "/retained/candidate/authorization-candidate.json"
    candidate_bytes = b"retained metadata candidate bytes\n"
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    store.raw[candidate_path] = candidate_bytes
    snapshot = {
        "schema_version": 1,
        "scope": (
            "actual metadata artifact and completed accounting snapshot; "
            "no scientific acceptance"
        ),
        "candidate": {
            "path": candidate_path,
            "sha256": candidate_sha256,
        },
        "candidate_artifacts": [
            {
                "path": candidate_path,
                "sha256": candidate_sha256,
            },
            {"path": preflight_pin["path"], "sha256": preflight_pin["sha256"]},
        ],
        "candidate_keys": ["evidence", "family", "phase", "pins", "schema", "status"],
        "candidate_pins": dict(candidate_pins),
        "candidate_status_fields": {"status": "REVIEW_CANDIDATE"},
        "command_packet": {"path": "/retained/command.json", "sha256": _hex("command")},
        "completion_snapshot": {
            "path": "/retained/completion.json",
            "sha256": _hex("completion"),
        },
        "failed_closure_runner_receipt": {
            "path": "/retained/runner.json",
            "sha256": _hex("runner"),
        },
        "issued_spec": {"path": "/retained/spec.json", "sha256": _hex("spec")},
        "native_completed": True,
        "native_summary": {
            "path": "/retained/native-summary.json",
            "sha256": _hex("native summary"),
        },
        "no_finalization_repeated": True,
        "observer_completed": True,
        "original_reservation": {
            "path": "/retained/reservation.json",
            "sha256": _hex("reservation"),
        },
        "production_api_calls": 0,
        "production_api_calls_basis": (
            "exact command packet declares zero; executed reviewed builder is "
            "metadata-only; this is not a scientific result or dynamic API "
            "instrumentation claim"
        ),
        "trusted_span": {
            "path": "/retained/trusted-span.json",
            "sha256": _hex("trusted span"),
        },
        # These finite operational values are deliberately not budget evidence.
        "aggregate_seconds": 21198.0,
        "remaining_seconds": 401.0,
        "scientific_acceptance": False,
    }
    snapshot.update(snapshot_changes or {})
    snapshot_raw = (
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    snapshot_pin = store.add(snapshot_raw, raw=True)

    ledger_pin = store.add(
        b"immutable previous A2 resource ledger snapshot\n", raw=True
    )
    preserved_predecessor_pin = _pin(
        store,
        {
            "schema": "gragra.a2.resource-human-decision.v1",
            "decision": "HUMAN_DECISION_REQUIRED",
            "stop_rule": "STOP-2",
            "execution_hold": "ALL_NEW_CAMPAIGN_SUBPROCESSES_STOPPED",
        },
    )
    predecessor_pin = _pin(
        store,
        {
            "schema": "gragra.a2.resource-human-decision.v1",
            "decision": "HUMAN_DECISION_REQUIRED",
            "stop_rule": "STOP-2",
            "execution_hold": "ALL_NEW_CAMPAIGN_SUBPROCESSES_STOPPED",
            "predecessor": {
                "path": preserved_predecessor_pin["path"],
                "sha256": preserved_predecessor_pin["sha256"],
                "preserved": True,
            },
        },
    )
    predecessor_seal = _seal(store, predecessor_pin)
    preserved_seal = _seal(store, preserved_predecessor_pin)

    decision = _closure_record(
        admission_pin=admission_pin,
        receipt_pin=receipt_pin,
        preflight_pin=preflight_pin,
        snapshot_pin=snapshot_pin,
        ledger_pin=ledger_pin,
        predecessor_pin=predecessor_pin,
        preserved_predecessor_pin=preserved_predecessor_pin,
        preserved_seals=[preserved_seal, predecessor_seal],
        changes=decision_changes,
    )
    if accounting_changes:
        decision["resource_accounting"].update(accounting_changes)
    decision_pin = _pin(store, decision)
    decision_seal = _seal(store, decision_pin, current=True)

    descriptor = {
        "schema": "gen1-stochastic-selected-delivery-v2",
        "accepted_families": [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in families
        ],
        "history": [],
        "final_identity_boundary": "EXTERNAL_ATTESTATION_AFTER_TRACKED_COMMIT",
        "a2_closure": {
            "decision": decision_pin,
            "decision_seal": decision_seal,
            "reference": {
                "admission": admission_pin,
                "root_receipt": receipt_pin,
                "artifact_map": artifact_map_pin,
                "canonical_pins": canonical_pin,
            },
            "preflight": {"artifact": preflight_pin, "snapshot": snapshot_pin},
            "ledger_snapshot": ledger_pin,
            "predecessor": predecessor_pin,
            "preserved_predecessor": preserved_predecessor_pin,
            "preserved_seals": [preserved_seal, predecessor_seal],
        },
    }
    return store, descriptor, native_summary


def _patch_accepted_graphs(monkeypatch, native_summary):
    seen = []
    native_calls = []

    def comparison(_store, descriptor, *, family):
        assert family in FAMILIES
        seen.append(family)
        policy = _store.json(descriptor["stages"][-1]["policy"])
        dependency = policy["artifacts"]["dependency_closure"]
        return [
            {
                "phase": "comparison",
                "pins": {
                    "production_commit": _hex(family + " P commit"),
                    "production_tree": _hex(family + " P tree"),
                    "dependency_closure_sha256": dependency["sha256"],
                },
                "summary": {"family": family},
            }
        ]

    def transfer(_store, descriptor, *, actual_p, actual_closure):
        return {
            "descriptor": descriptor,
            "actual_p": actual_p,
            "actual_closure": actual_closure,
            "archived_bytes_verified": True,
        }

    monkeypatch.setattr(reader, "verify_selected_comparison", comparison)
    monkeypatch.setattr(reader, "verify_transfer", transfer)

    def native_reference(_store, artifacts):
        native_calls.append(artifacts)
        assert set(artifacts) == {
            "budget",
            "environment",
            "family_procedure",
            "fixture",
            "freeze",
            "records",
            "seal",
        }
        return native_summary

    monkeypatch.setattr(v2, "verify_native_reference", native_reference)
    return seen, native_calls


def test_v2_verifies_six_accepted_graphs_and_separate_a2_resource_limited_closure(
    monkeypatch,
):
    store, descriptor, native_summary = _delivery_fixture()
    seen, native_calls = _patch_accepted_graphs(monkeypatch, native_summary)

    result = reader.verify_selected_delivery_v2(store, descriptor)

    assert seen == list(FAMILIES)
    assert len(native_calls) == 1
    assert [row["family"] for row in result["families"]] == list(FAMILIES)
    assert result["accepted_family_count"] == 6
    assert result["science_acceptance"]["accepted_families"] == list(FAMILIES)
    assert result["science_acceptance"]["a2_production_acceptance_established"] is False
    assert "production_api_calls" not in result
    assert result["a2_closure"]["status"] == "A2_RESOURCE_LIMITED_CLOSURE_VERIFIED"
    assert set(result["a2_closure"]) == {
        "status",
        "verification_scope",
        "family_status",
        "resource_status",
        "production_comparison",
        "production_acceptance",
        "conservative_remaining_budget",
        "budget_exhaustion",
        "conditioning_failure",
        "physics_failure",
        "reference_counts",
        "reference_admission_verified",
        "reference_execution_repeated",
        "preflight",
        "immutable_ledger_snapshot_verified",
        "historical_stop2_records_verified",
        "decision_seal_verified",
        "included_in_accepted_families",
        "remaining_budget_seconds",
    }
    assert "actual_comparison_P" not in result["a2_closure"]
    assert (
        result["a2_closure"]["preflight"][
            "historical_candidate_artifact_raw_retrieval_count"
        ]
        == 2
    )
    assert (
        result["a2_closure"]["preflight"]["candidate_artifact_scope"]
        == "DIRECT_SNAPSHOT_REFERENCES_RAW_BYTES_ONLY"
    )
    assert (
        result["a2_closure"]["production_comparison"] == "PRODUCTION_COMPARISON_NOT_RUN"
    )
    assert result["a2_closure"]["production_acceptance"] == "NOT_ESTABLISHED"
    assert result["a2_closure"]["conservative_remaining_budget"] == "NOT_ESTABLISHED"
    assert result["a2_closure"]["budget_exhaustion"] == "NOT_PROVEN"
    assert result["a2_closure"]["conditioning_failure"] == "NOT_CLAIMED"
    assert result["a2_closure"]["physics_failure"] == "NOT_CLAIMED"
    assert result["a2_closure"]["included_in_accepted_families"] is False
    assert result["a2_closure"]["remaining_budget_seconds"] is None


def test_v1_entrypoint_remains_strict_and_rejects_v2_descriptor(monkeypatch):
    store, descriptor, native_summary = _delivery_fixture()
    _patch_accepted_graphs(monkeypatch, native_summary)

    with pytest.raises(EvidenceError, match="schema mismatch"):
        reader.verify_selected_delivery(store, descriptor)


@pytest.mark.parametrize(
    "change",
    [
        {"family_status": "COMPLETE"},
        {"production_comparison": "PASS"},
        {"production_acceptance": "PASS"},
        {"conditioning_failure": "CLAIMED"},
        {"physics_failure": "CLAIMED"},
        {"accepted_family_recalculation_for_a2_closure": True},
    ],
)
def test_v2_rejects_a2_acceptance_or_failure_promotion(monkeypatch, change):
    store, descriptor, native_summary = _delivery_fixture(decision_changes=change)
    _patch_accepted_graphs(monkeypatch, native_summary)

    with pytest.raises(EvidenceError):
        v2.verify_selected_delivery_v2(store, descriptor)


@pytest.mark.parametrize(
    ("decision_change", "accounting_change"),
    [
        ({}, {"conservative_remaining_budget": "ESTABLISHED"}),
        ({}, {"budget_exhaustion": "PROVEN"}),
        ({}, {"production_reservation": "AUTHORIZED"}),
        ({"accepted_families_preserved": [*FAMILIES, "A2"]}, {}),
    ],
)
def test_v2_rejects_budget_or_a2_count_promotion(
    monkeypatch, decision_change, accounting_change
):
    store, descriptor, native_summary = _delivery_fixture(
        decision_changes=decision_change,
        accounting_changes=accounting_change,
    )
    _patch_accepted_graphs(monkeypatch, native_summary)

    with pytest.raises(EvidenceError):
        v2.verify_selected_delivery_v2(store, descriptor)


def test_v2_rejects_a2_comparison_shape_and_a2_accepted_family(monkeypatch):
    store, descriptor, native_summary = _delivery_fixture()
    _patch_accepted_graphs(monkeypatch, native_summary)
    descriptor["accepted_families"].append(
        {"family": "A2", "comparison": {}, "transfer": {}}
    )
    with pytest.raises(EvidenceError):
        v2.verify_selected_delivery_v2(store, descriptor)

    store, descriptor, native_summary = _delivery_fixture()
    _patch_accepted_graphs(monkeypatch, native_summary)
    descriptor["a2_closure"]["comparison"] = {}
    with pytest.raises(EvidenceError):
        v2.verify_selected_delivery_v2(store, descriptor)


def test_v2_rejects_reference_receipt_authorization_and_pin_mismatch(monkeypatch):
    store, descriptor, native_summary = _delivery_fixture(
        receipt_changes={"production_comparison_authorized": True}
    )
    _patch_accepted_graphs(monkeypatch, native_summary)
    with pytest.raises(EvidenceError):
        v2.verify_selected_delivery_v2(store, descriptor)

    store, descriptor, native_summary = _delivery_fixture()
    _patch_accepted_graphs(monkeypatch, native_summary)
    descriptor["a2_closure"]["reference"]["root_receipt"]["sha256"] = _hex(
        "wrong receipt"
    )
    with pytest.raises(EvidenceError):
        v2.verify_selected_delivery_v2(store, descriptor)


def test_v2_rejects_native_reference_summary_or_admission_mismatch(monkeypatch):
    changed = {"expected_keys_sha256": _hex("wrong 84-key digest")}
    store, descriptor, native_summary = _delivery_fixture(canonical_changes=changed)
    _patch_accepted_graphs(monkeypatch, native_summary)
    with pytest.raises(EvidenceError):
        v2.verify_selected_delivery_v2(store, descriptor)

    store, descriptor, native_summary = _delivery_fixture(
        admission_changes={"decision": "REVIEW_CANDIDATE"}
    )
    _patch_accepted_graphs(monkeypatch, native_summary)
    with pytest.raises(EvidenceError):
        v2.verify_selected_delivery_v2(store, descriptor)


@pytest.mark.parametrize(
    ("preflight_changes", "snapshot_changes"),
    [
        ({"status": "AUTHORIZED"}, {}),
        ({"production_api_calls": 1}, {}),
        ({}, {"scientific_acceptance": True}),
        ({}, {"production_api_calls": 1}),
    ],
)
def test_v2_rejects_preflight_authorization_or_scientific_acceptance(
    monkeypatch, preflight_changes, snapshot_changes
):
    store, descriptor, native_summary = _delivery_fixture(
        preflight_changes=preflight_changes, snapshot_changes=snapshot_changes
    )
    _patch_accepted_graphs(monkeypatch, native_summary)
    with pytest.raises(EvidenceError):
        v2.verify_selected_delivery_v2(store, descriptor)


@pytest.mark.parametrize("failure", ["missing", "tampered"])
def test_v2_rejects_missing_or_tampered_direct_candidate_artifact_bytes(
    monkeypatch, failure
):
    store, descriptor, native_summary = _delivery_fixture()
    _patch_accepted_graphs(monkeypatch, native_summary)
    item = descriptor["a2_closure"]["preflight"]
    snapshot = v2.a2.operational_json(store.read(item["snapshot"]))
    candidate = snapshot["candidate_artifacts"][0]

    if failure == "missing":
        original_read = store.read

        def missing_read(pin):
            if (pin["path"], pin["sha256"]) == (
                candidate["path"],
                candidate["sha256"],
            ):
                assert pin["kind"] == v2.SNAPSHOT_CANDIDATE_ARTIFACT_KIND
                raise EvidenceError("missing selected retrieval pin")
            return original_read(pin)

        monkeypatch.setattr(store, "read", missing_read)
    else:
        store.raw[candidate["path"]] = b"tampered candidate bytes\n"

    with pytest.raises(EvidenceError):
        v2.verify_selected_delivery_v2(store, descriptor)


def test_v2_rejects_changed_ledger_or_stop2_pins(monkeypatch):
    store, descriptor, native_summary = _delivery_fixture()
    _patch_accepted_graphs(monkeypatch, native_summary)
    descriptor["a2_closure"]["ledger_snapshot"]["sha256"] = _hex("wrong ledger")
    with pytest.raises(EvidenceError):
        v2.verify_selected_delivery_v2(store, descriptor)

    store, descriptor, native_summary = _delivery_fixture()
    _patch_accepted_graphs(monkeypatch, native_summary)
    descriptor["a2_closure"]["preserved_seals"][0]["sha256"] = _hex("wrong STOP-2 seal")
    with pytest.raises(EvidenceError):
        v2.verify_selected_delivery_v2(store, descriptor)


def test_v2_operational_snapshot_accepts_float_metadata_without_treating_it_as_budget(
    monkeypatch,
):
    store, descriptor, native_summary = _delivery_fixture()
    _patch_accepted_graphs(monkeypatch, native_summary)

    result = v2.verify_selected_delivery_v2(store, descriptor)

    assert result["a2_closure"]["remaining_budget_seconds"] is None
    assert result["a2_closure"]["conservative_remaining_budget"] == "NOT_ESTABLISHED"
