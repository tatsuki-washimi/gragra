"""Finite B1 offline profile: admitted 70 connections precede 30 science rows.

Only archived bytes are read. The two admitted pure arithmetic sources execute
from their fixed raw hashes; production and reference generators never load.
A current binding supplement is distinct from the original execution and from
this offline replay. None of these checks establishes final source transfer.
"""

from __future__ import annotations

import hashlib
import re
from decimal import Decimal
from fractions import Fraction
from types import ModuleType

from .archive import EvidenceError, decode_json, validate_pin

ROLES = {"science", "numerical", "provenance"}
KEY_FIELDS = (
    "family",
    "fixture_id",
    "frequency_hz",
    "target_id",
    "observable",
    "representation",
    "set",
)
STAGE_FIELDS = {
    "phase",
    "policy",
    "candidate",
    "authorization",
    "authorization_reviews",
    "attempt",
    "resource_supervisor",
    "result_admission",
    "result_reviews",
    "local_validator_proof",
    "excluded_reviewer_ids",
}
AUTHOR_IDS = {"/root/b1_exec", "/root/seismic_exec", "/root/draft"}
# This historical R_aux subject has one known reference writer. Preserve the
# actual issued science-review alias as a finite literal mapping, not a general
# alias heuristic. This roster does not certify unknown historical R_B1 authors.
R_AUX_AUTHOR_IDS = {"/root/b1_exec"}
R_AUX_REVIEWER_ALIASES = {"seismic_exec_independent_B1aux": "/root/seismic_exec"}
R_AUX_REVIEWER_IDS = {
    "science": "/root/seismic_exec",
    "numerical": "/root/temperature_exec",
    "provenance": "/root/acoustic_exec",
}

ARITHMETIC_SHA256 = {
    "kernel": "09dbe97ceaf635b0fd0eafdf2cb3f11ac994af276e318a684c45c00a5548be68",
    "adapter": "3d13f6e509f3727cd0d2045a6e8eeab5f609987425ec5156386efd39f5006aea",
}
ADAPTER_VERSIONS = {
    ARITHMETIC_SHA256["adapter"]: "original",
    "16a0e035b0f502010a8e91d67d1799d617ae53097a6f465e255559caead6829e": "formatted",
}
# The reviewed style revision changes bytes, not the admitted rational mapping.
# These literal evidence pins belong to this one revision, not a policy registry.
_STYLE_ROOT = (
    "/home/washimi/.local/state/gragra/gen1-production/"
    "20261004-pr159-final/reviews/protocol/"
)
ADAPTER_STYLE_REVIEWS = {
    "science": {
        "path": _STYLE_ROOT + "style-science-source-companion-v1.json",
        "sha256": "3a92b37d776a69407f685b5db4497b89214baf617e22ce5e54e4a8d3a6473113",
        "kind": "source_review",
    },
    "numerical": {
        "path": _STYLE_ROOT + "style-numerical-companion-v1.json",
        "sha256": "195ff99af2548575ba73f2bf566eb9d39c97fbfd31eda95fa6a97452e5b34063",
        "kind": "source_review",
    },
    "provenance": {
        "path": _STYLE_ROOT + "style-provenance-companion-v1.json",
        "sha256": "ccfc1e7e00d79a9c61ddf24506b922ec9b93490f0a78b8548166e67162cfc9e8",
        "kind": "source_review",
    },
}
TRANSPORT_SHA256 = "1238027f4e5049df586a979ab62aebb8e093a1a2ce71d2c69a6caa4eb3df133a"
BASE_PINS = {
    "family",
    "production_commit",
    "production_tree",
    "comparator_commit",
    "comparator_tree",
    "fixture_sha256",
    "budget_sha256",
    "procedure_sha256",
    "f_common_sha256",
    "f_b1_aux_sha256",
    "decision_kernel_sha256",
    "dependency_closure_sha256",
    "r_b1_records_sha256",
    "r_b1_seal_sha256",
    "r_b1_review_sha256",
    "r_aux_records_sha256",
    "r_aux_seal_sha256",
    "r_aux_admission_sha256",
    "receipt_expectations_sha256",
    "bound_receipts_sha256",
    "policy_sha256",
}
CONNECTION_GATES = {
    "units",
    "shape",
    "frequency_axis",
    "input_normalization",
    "polarization",
    "spatial_phase",
    "frame",
    "source_applicability",
    "reference_resolved",
    "numeric_inequalities",
    "precision_crosscheck",
}


def _need(condition, label):
    if not condition:
        raise EvidenceError(label)


def _fields(value, names, label):
    _need(
        isinstance(value, dict) and set(value) == set(names), label + " schema mismatch"
    )


def _key(key):
    _fields(key, KEY_FIELDS, "B1 key")
    _need(all(isinstance(key[n], str) and key[n] for n in KEY_FIELDS), "invalid B1 key")
    return tuple(key[n] for n in KEY_FIELDS)


def _keys(keys):
    result = [_key(key) for key in keys]
    _need(len(result) == len(set(result)), "duplicate B1 key")
    return set(result)


def _same(left, right):
    return all(left.get(n) == right.get(n) for n in ("path", "sha256"))


def _canonical(value):
    import json

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def _fraction(value):
    _need(isinstance(value, str), "scientific decimal strings required")
    try:
        number = Decimal(value)
        _need(number.is_finite(), "nonfinite scientific number")
        return Fraction(number)
    except (ValueError, ArithmeticError) as error:
        raise EvidenceError("invalid scientific number") from error


def _jsonl(store, pin):
    raw = store.read(pin)
    _need(
        raw.endswith(b"\n") and raw and not any(not x for x in raw.splitlines()),
        "complete raw JSONL required",
    )
    return [decode_json(line) for line in raw.splitlines()]


def _load_arithmetic(store, pin, kind):
    """Execute only literal admitted CAS source bytes, without path imports."""
    raw = store.read(pin)
    _need(
        kind in ARITHMETIC_SHA256
        and hashlib.sha256(raw).hexdigest() == pin["sha256"]
        and (
            pin["sha256"] in ADAPTER_VERSIONS
            if kind == "adapter"
            else pin["sha256"] == ARITHMETIC_SHA256[kind]
        ),
        "unadmitted B1 arithmetic source",
    )
    version = ADAPTER_VERSIONS[pin["sha256"]] if kind == "adapter" else "original"
    if version == "formatted":
        for role, review_pin in ADAPTER_STYLE_REVIEWS.items():
            validate_pin(review_pin)
            review = decode_json(store.read(review_pin))
            _need(
                review.get("schema")
                == "gen1-stochastic-family-comparator-source-review-v1"
                and review.get("role") == role
                and review.get("decision") == "PASS"
                and review.get("commit") == "fab8e060d408d1a34a1a8378dd0ad62a41ac17bf"
                and review.get("source_files", {}).get(
                    "tests/oracles/gen1/b1_exact_record_adapter.py"
                )
                == pin["sha256"],
                "formatted adapter equivalence evidence differs",
            )
            store.read(review["source_candidate"])
            store.read(review["original_review"])
    module = ModuleType("offline_b1_" + kind)
    exec(compile(raw, "<admitted-B1-" + kind + ">", "exec"), module.__dict__)
    module.admitted_source = {
        "sha256": pin["sha256"],
        "version": version,
        "equivalence_reviews": ADAPTER_STYLE_REVIEWS if version == "formatted" else {},
    }
    return module


def _reviews(store, reviews, *, phase, subject, pins, result, excluded):
    _fields(reviews, ROLES, "three reviews")
    identities = set()
    for role, pin in reviews.items():
        review = store.json(pin)
        fields = {"schema", "role", "reviewer_id", "decision", "pins"} | (
            {"family", "phase", "seal_sha256", "original_review"}
            if result
            else {"candidate_sha256"}
        )
        _fields(review, fields, "result review" if result else "authorization review")
        if result:
            expected_schema = (
                "gen1-stochastic-connection-result-review-v1"
                if phase == "connection"
                else "gen1-stochastic-comparison-result-review-v1"
            )
            _need(
                review["schema"] == expected_schema
                and review["family"] == "B1"
                and review["phase"] == phase
                and review["seal_sha256"] == subject["sha256"],
                "B1 result review subject differs",
            )
            store.read(review["original_review"])
            _need(
                isinstance(review["reviewer_id"], str)
                and re.fullmatch(r"/root/[a-z0-9_]+", review["reviewer_id"])
                is not None,
                "canonical result reviewer identity required",
            )
        else:
            _need(
                review["schema"] == "gen1-stochastic-review-v1"
                and review["candidate_sha256"] == subject["sha256"],
                "B1 authorization review subject differs",
            )
        ident = review["reviewer_id"]
        _need(
            isinstance(ident, str)
            and ident
            and ident not in identities
            and ident not in excluded
            and review["role"] == role
            and review["decision"] == "PASS"
            and review["pins"] == pins,
            "independent exact-subject PASS required",
        )
        identities.add(ident)


def _closure(store, policy, run, attempt):
    closure = store.json(policy["artifacts"]["dependency_closure"])
    _need(
        closure["production_commit"] == run["production_commit"]
        and closure["production_tree"] == run["production_tree"],
        "whole planned closure P differs",
    )

    # All planned nodes, including unloaded modules, remain mandatory CAS bytes.
    def read_pins(value):
        if isinstance(value, dict):
            if {"path", "sha256", "kind"} <= set(value):
                store.read({n: value[n] for n in ("path", "sha256", "kind")})
            for item in value.values():
                read_pins(item)
        elif isinstance(value, list):
            for item in value:
                read_pins(item)

    read_pins(closure)
    runtime = store.json(attempt["artifacts"]["runtime_dependencies"])
    modules = {m["name"]: m for m in closure["modules"]}
    _need(
        len(modules) == len(closure["modules"])
        and runtime.get("status") == "MATCH"
        and runtime.get("planned_sha256") == run["dependency_closure_sha256"]
        and runtime.get("python") == closure["python"]
        and any(m["name"].startswith("gragra.") for m in runtime["observed_modules"])
        and all(modules.get(m["name"]) == m for m in runtime["observed_modules"])
        and all(e in closure["import_edges"] for e in runtime["observed_edges"])
        and all(
            n in closure["native_backends"] for n in runtime["observed_native_backends"]
        ),
        "actual finite runtime closure mismatch",
    )
    return closure


def _binding_proof(store, stage, pins):
    proof = store.json(stage["local_validator_proof"])
    _fields(
        proof,
        {
            "schema",
            "family",
            "phase",
            "profile",
            "decision",
            "subjects",
            "validator",
            "argv",
            "pins",
            "original_proof",
        },
        "B1 supplemental binding proof",
    )
    subjects = {
        n: stage[n]
        for n in (
            "policy",
            "candidate",
            "authorization",
            "attempt",
            "resource_supervisor",
        )
    }
    phase = stage["phase"]
    _need(
        proof["schema"] == "gen1-stochastic-local-validator-proof-v1"
        and proof["family"] == "B1"
        and proof["phase"] == phase
        and proof["profile"] == "b1-v1"
        and proof["decision"] == "PASS"
        and proof["subjects"] == subjects
        and proof["pins"] == pins,
        "B1 supplemental proof exact binding differs",
    )
    _fields(proof["validator"], {"source", "qualified_function"}, "actual B1 helper")
    expected = "tests.oracles.gen1.stochastic_formal.common.transport." + (
        "validate_connection_pass"
        if phase == "connection"
        else "require_science_authority"
    )
    _need(
        proof["validator"]["qualified_function"] == expected
        and proof["validator"]["source"]["sha256"] == TRANSPORT_SHA256,
        "unknown B1 actual binding helper",
    )
    store.read(proof["validator"]["source"])
    _need(
        isinstance(proof["argv"], list)
        and proof["argv"]
        and all(isinstance(s, str) and s for s in proof["argv"]),
        "actual supplement argv required",
    )
    original = store.json(proof["original_proof"])
    _need(
        original.get("schema") == "gen1-root-B1-result-binding-verification-v1"
        and original.get("phase") == phase
        and original.get("scope") == "SUPPLEMENTAL_CURRENT_AUTHORITY_BINDING_ONLY"
        and isinstance(original.get("created_at_utc"), str)
        and original["created_at_utc"]
        and original.get("subjects") == subjects
        and original.get("pins") == pins
        and original.get("result") == {"decision": "PASS", "pins": pins}
        and original.get("production_API_calls") == 0
        and original.get("reference_generations") == 0
        and original.get("original_execution_repeated") is False
        and original.get("argv") == proof["argv"],
        "supplement cannot invent historical result validation",
    )
    store.read(original["orchestration_source"])
    store.read(original["preflight_source"])


def _admission(store, stage, run):
    admission = store.json(stage["result_admission"])
    _fields(
        admission,
        {
            "schema",
            "family",
            "phase",
            "decision",
            "pins",
            "attempt",
            "reviews",
            "original_admission",
        },
        "B1 result admission wrapper",
    )
    phase = stage["phase"]
    _need(
        admission["schema"] == "gen1-stochastic-result-admission-v1"
        and admission["family"] == "B1"
        and admission["phase"] == phase
        and admission["decision"] == "PASS"
        and admission["pins"] == run
        and admission["attempt"] == stage["attempt"]
        and admission["reviews"] == stage["result_reviews"],
        "B1 actual admission wrapper differs",
    )
    original = store.json(admission["original_admission"])
    _need(
        original.get("family") == "B1"
        and original.get("pins") == run
        and original.get("reviews") == stage["result_reviews"],
        "root original result review selection differs",
    )
    if phase == "connection":
        _need(
            original.get("schema") == "gen1-stochastic-connection-result-admission-v1"
            and original.get("decision") == "PASS"
            and _same(original["seal"], stage["attempt"])
            and original.get("counts")
            == {
                "independent_first_order_reference": 56,
                "production_consistency_only": 14,
            }
            and original.get("source_connections") is True
            and original.get("science_comparison_authorized") is False,
            "original root 70 admission differs",
        )
    else:
        _need(
            original.get("schema") == "gen1-stochastic-family-acceptance-v1"
            and original.get("status") == "VERIFIED_BOUNDED"
            and original.get("accepted") == 30
            and original.get("science_records") == 30
            and original.get("failed") == 0
            and original.get("unresolved") == 0
            and original.get("P_family") == run["production_commit"]
            and _same(original["comparison_seal"], stage["attempt"])
            and original.get("connection_reference_gates") == 56
            and original.get("connection_consistency_gates") == 14
            and _same(original["supervisor_seal"], stage["resource_supervisor"]),
            "original root 30 acceptance differs",
        )
        store.read(original["connection_admission"])
    return original


def _bindings(store, policy, fixture, run):
    artifacts = policy["artifacts"]
    declarations = store.json(artifacts["receipt_expectations"])["receipts"]
    bundle = store.json(artifacts["bound_receipts"])
    _fields(bundle, {"schema", "status", "pins", "receipts"}, "BOUND receipt bundle")
    _need(
        bundle["schema"] == "gen1-stochastic-bound-receipts-v1"
        and bundle["status"] == "BOUND"
        and bundle["pins"]
        == {
            n: run[n]
            for n in ("production_commit", "production_tree", "f_b1_aux_sha256")
        },
        "bindings cannot replace executions",
    )
    _need(
        len(declarations) == 70
        and len(bundle["receipts"]) == 70
        and _keys([r["key"] for r in declarations])
        == _keys(fixture["expected_auxiliary_keys"]),
        "70 declared binding keys required",
    )
    expected = {_key(r["key"]): r for r in declarations}
    seen = set()
    ids = set()
    dependencies = set()
    independent = _keys(fixture["expected_reference_auxiliary_keys"])
    for row in bundle["receipts"]:
        identity = _key(row["key"])
        _need(
            identity in expected
            and identity not in seen
            and row["receipt_id"] not in ids,
            "duplicate or unknown binding receipt",
        )
        seen.add(identity)
        ids.add(row["receipt_id"])
        _fields(
            row,
            set(expected[identity]) | {"status", "content_sha256"},
            "binding receipt",
        )
        declared = {
            n: v for n, v in row.items() if n not in {"status", "content_sha256"}
        }
        _need(
            row["status"] == "BOUND"
            and declared == expected[identity]
            and row["content_sha256"]
            == hashlib.sha256(
                _canonical({n: v for n, v in row.items() if n != "content_sha256"})
            ).hexdigest()
            and row["kind"]
            == (
                "independent_reference"
                if identity in independent
                else "production_consistency"
            ),
            "binding endpoint/rule/hash/partition differs",
        )
        store.read(row["source"])
        store.read(row["destination"])
        dependencies |= _keys(row["science_keys"])
    _need(
        dependencies == _keys(fixture["expected_keys"]),
        "all 30 science dependency keys required",
    )


def _references(store, policy, fixtures):
    a = policy["artifacts"]
    _need(
        not _same(a["r_b1_records"], a["r_aux_records"]), "R_aux cannot substitute R_B1"
    )
    aux = _jsonl(store, a["r_aux_records"])
    _need(
        len(aux) == 56
        and _keys([r["key"] for r in aux])
        == _keys(fixtures["expected_reference_auxiliary_keys"])
        and all(
            r["status"] == "resolved"
            and r["procedure_hash"] == a["f_b1_aux"]["sha256"]
            and r["record_kind"] == "first_order_connection_reference"
            for r in aux
        ),
        "independent R_aux exact 56 identity differs",
    )
    auxseal = store.json(a["r_aux_seal"])
    _need(
        auxseal.get("records_sha256") == a["r_aux_records"]["sha256"]
        and auxseal.get("f_b1_aux_sha256") == a["f_b1_aux"]["sha256"]
        and auxseal.get("mode") == "formal"
        and auxseal.get("exit_code") == 0
        and auxseal.get("coverage_valid") is True
        and auxseal.get("cleanup_confirmed") is True,
        "R_aux formal seal differs",
    )
    auxadmit = store.json(a["r_aux_admission"])
    _need(
        auxadmit.get("schema") == "gen1-stochastic-reference-admission-v1"
        and auxadmit.get("family") == "B1_aux"
        and auxadmit.get("decision") == "PASS"
        and auxadmit.get("records_sha256") == a["r_aux_records"]["sha256"]
        and auxadmit.get("seal_sha256") == a["r_aux_seal"]["sha256"]
        and auxadmit.get("f_b1_aux_sha256") == a["f_b1_aux"]["sha256"],
        "R_aux independent admission differs",
    )
    _fields(auxadmit["reviews"], ROLES, "R_aux reviews")
    ids = set()
    for role, pin in auxadmit["reviews"].items():
        r = store.json(pin)
        identity = R_AUX_REVIEWER_ALIASES.get(
            r.get("reviewer_id"), r.get("reviewer_id")
        )
        _need(
            identity == R_AUX_REVIEWER_IDS[role]
            and identity not in R_AUX_AUTHOR_IDS
            and r.get("decision") == "PASS"
            and r.get("role") == role
            and r.get("pins") == auxadmit["pins"]
            and r.get("reviewer_id")
            and identity not in ids,
            "R_aux exact independent reference review differs",
        )
        ids.add(identity)
        # These literal envelope shapes are historical issued review bytes:
        # science is itself a detailed supplement retaining its prior report;
        # numerical/provenance wrappers retain their original report.
        original_field = (
            "prior_science_review" if role == "science" else "original_review"
        )
        original_pin = dict(r[original_field])
        if set(original_pin) == {"path", "sha256"}:
            # Add only a typed retrieval role to the literal historical
            # two-field pin. The original review/record bytes remain unchanged.
            original_pin["kind"] = "legacy_R_aux_" + original_field
        validate_pin(original_pin)
        store.read(original_pin)
    science = store.json(a["r_b1_review"])
    _need(
        science.get("R_B1") == "ACCEPTED_FROZEN_REFERENCE_SET"
        and science.get("attempt", {}).get("records_sha256")
        == a["r_b1_records"]["sha256"]
        and science["attempt"].get("seal_sha256") == a["r_b1_seal"]["sha256"],
        "R_B1 accepted reference identity differs",
    )
    rbseal = store.json(a["r_b1_seal"])
    _need(
        rbseal.get("artifacts_sha256", {}).get("records.raw.jsonl")
        == a["r_b1_records"]["sha256"],
        "R_B1 sealed raw bytes differ",
    )
    return science


def _stage(store, stage):
    _fields(stage, STAGE_FIELDS, "B1 stage")
    phase = stage["phase"]
    excluded = stage["excluded_reviewer_ids"]
    _need(
        isinstance(excluded, list)
        and len(set(excluded)) == len(excluded)
        and set(excluded) >= AUTHOR_IDS,
        "actual B1 author exclusions required",
    )
    policy, candidate, authority, attempt = [
        store.json(stage[n])
        for n in ("policy", "candidate", "authorization", "attempt")
    ]
    _need(
        policy.get("schema") == "gen1-stochastic-B1-comparison-policy-v2"
        and policy.get("family") == "B1"
        and policy.get("phase") == phase,
        "B1 policy phase differs",
    )
    _fields(
        candidate,
        {"schema", "family", "phase", "status", "pins", "evidence"},
        "B1 candidate",
    )
    _fields(
        authority,
        {"schema", "family", "phase", "status", "pins", "candidate", "reviews"},
        "B1 authority",
    )
    for obj, status in ((candidate, "REVIEW_CANDIDATE"), (authority, "AUTHORIZED")):
        _need(
            obj["schema"] == "gen1-stochastic-authorization-v2"
            and obj["family"] == "B1"
            and obj["phase"] == phase
            and obj["status"] == status,
            "B1 candidate or authority phase differs",
        )
    expected_names = BASE_PINS | (
        {"connection_pass_sha256"} if phase == "comparison" else set()
    )
    _fields(candidate["pins"], expected_names, "B1 authority pins")
    _need(
        _same(authority["candidate"], stage["candidate"])
        and authority["reviews"] == stage["authorization_reviews"]
        and authority["pins"] == candidate["pins"]
        and candidate["pins"]["policy_sha256"] == stage["policy"]["sha256"]
        and _same(candidate["evidence"].get("policy"), stage["policy"]),
        "stale B1 authority binding",
    )
    _need(
        {**policy["pins"], "family": "B1", "policy_sha256": stage["policy"]["sha256"]}
        == candidate["pins"],
        "literal policy/preflight pins differ",
    )
    for artifact in candidate["evidence"].values():
        store.read(artifact)
    for name, artifact in policy["artifacts"].items():
        store.read(artifact)
        if name + "_sha256" in candidate["pins"]:
            _need(
                candidate["pins"][name + "_sha256"] == artifact["sha256"],
                "B1 artifact raw pin differs",
            )
    _reviews(
        store,
        stage["authorization_reviews"],
        phase=phase,
        subject=stage["candidate"],
        pins=candidate["pins"],
        result=False,
        excluded=set(),
    )
    run = attempt["pins"]
    _fields(
        run,
        set(candidate["pins"]) | {"authorization_sha256", "supervisor_launch_sha256"},
        "B1 run pins",
    )
    _need(
        attempt.get("schema") == "gen1-stochastic-execution-seal-v1"
        and attempt.get("family") == "B1"
        and attempt.get("phase") == phase
        and attempt.get("decision") == "PASS"
        and attempt.get("exit_code") == 0
        and attempt.get("exception") is None
        and all(run[n] == v for n, v in candidate["pins"].items())
        and run["authorization_sha256"] == stage["authorization"]["sha256"],
        "B1 actual run not bound PASS",
    )
    for artifact in attempt["artifacts"].values():
        store.read(artifact)
    fixture = store.json(policy["artifacts"]["fixture"])
    expected = (
        fixture["expected_auxiliary_keys"]
        if phase == "connection"
        else fixture["expected_keys"]
    )
    count = 70 if phase == "connection" else 30
    launch = store.json(attempt["artifacts"]["launch"])
    _need(
        len(expected) == count
        and len(_keys(expected)) == count
        and _keys(attempt["expected_keys"]) == _keys(expected)
        and attempt["completed"] == count
        and launch["pins"] == run
        and _keys(launch["expected_keys"]) == _keys(expected),
        "B1 frozen complete coverage differs",
    )
    rows = _jsonl(store, attempt["artifacts"]["records"])
    _need(
        len(rows) == count
        and _keys([r["key"] for r in rows]) == _keys(expected)
        and all(
            r.get("status") == "resolved" and r.get("passed") is True for r in rows
        ),
        "B1 omitted or failed actual row",
    )
    closure = _closure(store, policy, run, attempt)
    supervisor = store.json(stage["resource_supervisor"])
    for artifact in supervisor["artifacts"].values():
        store.read(artifact)
    cleanup = supervisor.get("cleanup", {})
    sl = supervisor["artifacts"]["launch"]
    _need(
        supervisor.get("schema") == "gen1-stochastic-supervisor-seal-v1"
        and supervisor.get("outcome") == "CHILD_EXITED"
        and supervisor.get("exit_code") == 0
        and supervisor.get("cleanup_confirmed") is True
        and cleanup.get("leader_reaped") is True
        and cleanup.get("process_group_empty") is True
        and cleanup.get("status") == "confirmed_stopped"
        and cleanup.get("wait_errors") == []
        and run["supervisor_launch_sha256"] == sl["sha256"],
        "B1 terminal resource cleanup differs",
    )
    supervisor_launch = store.json(sl)
    _need(
        supervisor_launch.get("mode") == "FORMAL"
        and supervisor_launch.get("policy_sha256") == stage["policy"]["sha256"]
        and _fraction(supervisor["elapsed_seconds"])
        <= supervisor_launch["timeout_seconds"],
        "B1 actual launch or timeout differs",
    )
    original = _admission(store, stage, run)
    _reviews(
        store,
        stage["result_reviews"],
        phase=phase,
        subject=stage["attempt"],
        pins=run,
        result=True,
        excluded=set(excluded),
    )
    _binding_proof(store, stage, run)
    _bindings(store, policy, fixture, run)
    reference_review = _references(store, policy, fixture)
    return (
        {"phase": phase, "pins": run, "records": rows},
        policy,
        fixture,
        closure,
        original,
        reference_review,
    )


def _connections(result, policy, fixture):
    reference = _keys(fixture["expected_reference_auxiliary_keys"])
    consistency = _keys(fixture["expected_consistency_auxiliary_keys"])
    _need(
        len(reference) == 56
        and len(consistency) == 14
        and not reference & consistency
        and reference | consistency == _keys(fixture["expected_auxiliary_keys"]),
        "frozen 56/14 partition differs",
    )
    for row in result["records"]:
        independent = _key(row["key"]) in reference
        label = (
            "independent_first_order_reference"
            if independent
            else "production_consistency_only"
        )
        _fields(row["gate_results"], CONNECTION_GATES, "11 connection gates")
        assessment = row["assessment"]
        _need(
            all(v is True for v in row["gate_results"].values())
            and row["evidence_class"] == label
            and assessment.get("evidence_class") == label
            and assessment.get("passed") is True
            and assessment.get("precision_crosscheck") is True
            and row["production_pin"]
            == {
                "commit": result["pins"]["production_commit"],
                "tree": result["pins"]["production_tree"],
            }
            and row["input_sha256"]
            == hashlib.sha256(_canonical(row["inputs"])).hexdigest(),
            "B1 connection execution evidence differs",
        )
        _need(
            _same(row["reference_pin"], policy["artifacts"]["r_aux_records"])
            if independent
            else row["reference_pin"] is None,
            "R_aux only supplies independent first-order evidence",
        )
        _need(
            [c["precision"] for c in assessment["conditions"]] == [120, 180],
            "both frozen decision precisions required",
        )
        for condition in assessment["conditions"]:
            d, u, b = [
                _fraction(condition[n])
                for n in ("difference_upper", "uncertainty_upper", "allowance_lower")
            ]
            _need(
                min(d, u, b) >= 0
                and d + u <= b
                and 10 * u <= b
                and all(
                    condition[n] is True
                    for n in (
                        "difference_plus_uncertainty",
                        "oracle_fraction",
                        "passed",
                    )
                ),
                "recorded full-field connection inequality fails",
            )
    result["summary"] = {
        "resolved": 70,
        "independent_first_order_reference": 56,
        "production_consistency_only": 14,
        "source_connections": True,
        "science_comparison_authorized": False,
    }


def _science(store, result, policy, fixture, closure, reference_review):
    kernel = _load_arithmetic(store, policy["artifacts"]["decision_kernel"], "kernel")
    adapter_pins = [p for p in closure["artifacts"] if p["sha256"] in ADAPTER_VERSIONS]
    _need(len(adapter_pins) == 1, "one exact admitted adapter CAS pin required")
    adapter = _load_arithmetic(store, adapter_pins[0], "adapter")
    zeros = [
        k
        for k in fixture["expected_keys"]
        if k["fixture_id"] in {"zero_density", "zero_psd"}
    ]
    _need(len(zeros) == 12, "frozen 12 whole-zero partition differs")
    references = adapter.adapt_b1_records(
        store.read(policy["artifacts"]["r_b1_records"]),
        reference_review["uncertainty_and_oracle_fraction"]["per_record"],
        fixture["expected_keys"],
        zeros,
    )
    budgets = store.json(policy["artifacts"]["budget"])
    fixtures = {f["fixture_id"]: f for f in fixture["science"]}
    classification = {"rigorous_bound": 0, "exact_zero": 0}
    ratios = []
    for row in result["records"]:
        ref = references[_key(row["key"])]
        f = fixtures[row["key"]["fixture_id"]]
        _need(
            row.get("reference_central_mapping") == "central*I3"
            and row.get("reference_uncertainty_mapping") == "3*u_up"
            and ref["u_matrix_exact"] == 3 * ref["u_up_exact"]
            and ref["u_matrix_kernel"] == ref["u_matrix_exact"],
            "public central*I3 / U_matrix=3*u_up changed",
        )
        tolerance = adapter.fraction_to_exact_text(
            adapter.parse_exact_fraction(budgets["structural_rate"])
            * ref["source_scale"]
        )
        structure = kernel.structure_gates(
            row["production_quantity"], symmetry="identity", tolerance=tolerance
        )
        expected_units = (
            "kg^2 m^-4 Hz^-1"
            if row["key"]["representation"] == "Gfree"
            else "m^2 s^-4 Hz^-1"
        )
        contract = {
            "units": ref["units"] == expected_units,
            "shape": row["production_quantity"]["shape"] == [3, 3],
            "frequency_axis": f["frequency_hz"].count(row["key"]["frequency_hz"]) == 1,
            "input_normalization": f["input_assumptions"] == "exact_input"
            and f["input_convention"] in {"component", "total"}
            and len(f["displacement_psd_m2_hz"]) == len(f["frequency_hz"]),
            "source_connections": True,
            "resolved_reference": ref["classification"] in classification,
            "sealed_exact_coverage": True,
        }
        recomputed = kernel.assess(
            row["production_quantity"],
            ref["kernel_quantity"],
            adapter.fraction_to_exact_text(ref["source_scale"]),
            budgets["absolute_rate"],
            budgets["relative_rate"],
            ref["kernel_uncertainty"],
            structure,
            contract,
        )
        _need(
            structure == row["structure_conditions"]
            and contract == row["contract_conditions"]
            and recomputed == row["assessment"]
            and recomputed["accepted"] is True
            and row["exact_zero_pass"] is True,
            "retained pure science arithmetic or gates differ",
        )
        if ref["is_exact_zero"]:
            _need(
                all(
                    _fraction(x[n]) == 0
                    for x in row["production_quantity"]["data"]
                    for n in ("real", "imag")
                ),
                "whole-zero production quantity not zero",
            )
        classification[ref["classification"]] += 1
        b = _fraction(recomputed["budget_lower"])
        ratios.append(
            {
                "key": row["key"],
                "D_over_B": str(_fraction(recomputed["difference_upper"]) / b)
                if b
                else None,
                "U_over_B": str(_fraction(recomputed["uncertainty_upper"]) / b)
                if b
                else None,
                "whole_zero": ref["is_exact_zero"],
            }
        )
    _need(
        classification == {"rigorous_bound": 18, "exact_zero": 12},
        "R_B1 18/12 classifications changed",
    )
    result["summary"] = {
        "science_count": 30,
        "arithmetic_sources": {
            "kernel": kernel.admitted_source,
            "adapter": adapter.admitted_source,
        },
        "science_passed": 30,
        "rigorous_bound": 18,
        "exact_whole_zero": 12,
        "reference_central_mapping": "central*I3",
        "reference_uncertainty_mapping": "3*u_up",
        "ratios": ratios,
        "claim": (
            "accepted on frozen fixture matrix at recorded P only; "
            "final closure transfer remains separate"
        ),
    }


def verify_b1_comparison(store, descriptor):
    """Verify two explicit archived stages; never import/call production or Git."""
    _fields(descriptor, {"profile", "stages"}, "B1 comparison descriptor")
    _need(
        descriptor["profile"] == "b1-v1"
        and isinstance(descriptor["stages"], list)
        and len(descriptor["stages"]) == 2
        and all(isinstance(s, dict) for s in descriptor["stages"])
        and [s.get("phase") for s in descriptor["stages"]]
        == ["connection", "comparison"],
        "B1 requires ordered complete 70 then 30 stages",
    )
    try:
        connection, cp, cf, _, _, _ = _stage(store, descriptor["stages"][0])
        _connections(connection, cp, cf)
        # Nothing from the science stage is read before complete reviewed 70 PASS.
        science, sp, sf, closure, original, review = _stage(
            store, descriptor["stages"][1]
        )
        connection_pin = descriptor["stages"][0]["attempt"]
        _need(
            _same(sp["artifacts"]["connection_seal"], connection_pin)
            and sp["connection_pins"] == connection["pins"]
            and science["pins"]["connection_pass_sha256"] == connection_pin["sha256"],
            "BOUND or R_aux cannot replace literal actual 70 PASS",
        )
        shared = BASE_PINS - {"policy_sha256"}
        _need(
            all(science["pins"][n] == connection["pins"][n] for n in shared),
            "B1 science changed F/P/C/R after reviewed 70",
        )
        _need(
            all(sp["artifacts"][n] == cp["artifacts"][n] for n in cp["artifacts"]),
            "R_aux or R_B1 artifact identity changed",
        )
        original_connection = store.json(original["connection_admission"])
        _need(
            original_connection["pins"] == connection["pins"]
            and _same(original_connection["seal"], connection_pin),
            "science root receipt inherits different 70 admission",
        )
        _science(store, science, sp, sf, closure, review)
        return [connection, science]
    except EvidenceError:
        raise
    except (KeyError, TypeError, ValueError, ArithmeticError) as error:
        raise EvidenceError("invalid B1 retained evidence") from error
