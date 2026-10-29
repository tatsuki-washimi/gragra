"""Offline interpretation of the finite admitted A2 managed-resource profile.

Only retained CAS bytes are read. This module does not import the execution
adapter, launch children, run production or regenerate reference quantities.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePosixPath

from .archive import EvidenceError

PROFILE = "a2-ledger-v1"
BUDGET_SHA256 = "f46f41dad65c76be2d255d1596b70ba9b43e47b164c6781dd58fd1aa788506b9"
RUNTIME_PROFILE_SOURCES = {
    BUDGET_SHA256: (
        "bd7af6bcbbb441a5f7339d4c5b05760bad3e897d4730414d372a8acdd1e6ebad",
    ),
    "a61ed66ad19291fe1a00fa87c5b266c7633e4e17d52802cb24b41709690d90a8": (
        "1cf51b42dfd4c8bcd4d00aae10ba19980e5bebcca523c6ae1e734947cf39df96",
        "e249760aaab011dba787263ec654ed1e6104633d7d2a3c778a1b136f811cc015",
    ),
}
LEDGER_PATH = (
    "/home/washimi/.local/state/gragra/gen1-production/"
    "20261004-pr159-final/a2/resource-ledger.json"
)
RESOURCE_TOKEN = "GRAGRA_STOCHASTIC_FAMILY_RESOURCE_LAUNCH"
COMPLETION_FIELDS = {
    "schema",
    "issuer",
    "decision",
    "managed_profile",
    "aggregate_budget_reservation",
    "controller_id",
    "resource_launch",
    "supervisor_seal",
    "attempt_seal",
    "postwait_ledger_snapshot",
    "root_external_span",
}


def _native_pin(pin):
    if (
        not isinstance(pin, dict)
        or set(pin)
        not in (
            {"path", "sha256"},
            {"path", "sha256", "kind"},
            {"path", "sha256", "decision"},
        )
        or "decision" in pin
        and pin["decision"] != "PASS"
    ):
        raise EvidenceError("unknown native A2 provenance pin")
    return {
        "path": pin["path"],
        "sha256": pin["sha256"],
        "kind": pin.get("kind", "native_reference_provenance"),
    }


def verify_native_reference(store, artifacts):
    """Project the actual native seal/authorization, without synthetic E fields."""
    from .graph import key_id

    seal = store.json(artifacts["seal"])
    fixture = store.json(artifacts["fixture"])
    expected = fixture["expected_keys"] + fixture["expected_scalar_keys"]
    identities = [key_id(key) for key in expected]
    if (
        len(identities) != 84
        or len(set(identities)) != 84
        or sum(key["set"] == "science" for key in expected) != 54
    ):
        raise EvidenceError("A2 frozen 84-key/54science/30AUX partition differs")
    root = PurePosixPath(artifacts["seal"]["path"]).parent
    hashes = seal.get("artifacts_sha256", {})
    required = {
        "authorization.json",
        "family-freeze.json",
        "policy.json",
        "records.raw.jsonl",
        "resource-ledger.json",
        "resource-result.json",
        "purpose-records.json",
        "stderr.jsonl",
        "stdout.jsonl",
    }
    if set(hashes) != required:
        raise EvidenceError("native A2 sealed artifact closure differs")
    sealed = {
        name: {
            "path": str(root / name),
            "sha256": digest,
            "kind": "native_reference_provenance",
        }
        for name, digest in hashes.items()
    }
    for pin in sealed.values():
        store.read(pin)
    auth = store.json(sealed["authorization.json"])
    execution = seal.get("execution", {})
    coverage = seal.get("execution_coverage", {})
    provenance = seal.get("provenance", {})
    if (
        seal.get("format") != "gen1-stochastic-attempt-seal-v2"
        or execution.get("status") != "success"
        or execution.get("exit_code") != 0
        or execution.get("cleanup", {}).get("status") != "confirmed_stopped"
        or coverage.get("status") != "accepted"
        or coverage.get("record_count") != 84
        or len(coverage.get("expected_keys", [])) != 84
        or {key_id(key) for key in coverage.get("expected_keys", [])} != set(identities)
        or any(
            coverage.get(name)
            for name in (
                "malformed_lines",
                "partial_final_line",
                "reasons",
                "duplicate_keys",
                "duplicate_expected_keys",
                "unknown_keys",
                "unresolved_keys",
                "not_generated_keys",
            )
        )
        or provenance.get("family") != "A2"
        or provenance.get("production_imported") is not False
        or provenance.get("purpose") != "formal-science"
        or auth.get("family") != "A2"
        or auth.get("authorized_by") != "root"
        or auth.get("formal_execution") != "AUTHORIZED"
        or auth.get("E_commit") != provenance.get("E_commit")
        or auth.get("F_sha256") != provenance.get("F_sha256")
        or auth.get("F_sha256") != artifacts["freeze"]["sha256"]
        or auth.get("fixture_sha256") != provenance.get("fixture_sha256")
        or auth.get("fixture_sha256") != artifacts["fixture"]["sha256"]
        or auth.get("procedure_sha256") != artifacts["family_procedure"]["sha256"]
    ):
        raise EvidenceError("native A2 generation identity/execution/coverage differs")
    raw = store.read(artifacts["records"])
    environment = artifacts["environment"]
    inventory_pin = auth["reference_runtime_inventory_pin"]
    if (
        PurePosixPath(artifacts["records"]["path"]).parent != root
        or not raw.endswith(b"\n")
        or hashes["records.raw.jsonl"] != artifacts["records"]["sha256"]
        or hashes["stdout.jsonl"] != artifacts["records"]["sha256"]
        or hashes["family-freeze.json"] != artifacts["freeze"]["sha256"]
        or hashes["policy.json"] != artifacts["fixture"]["sha256"]
        or inventory_pin["path"] != environment["path"]
        or inventory_pin["sha256"] != environment["sha256"]
    ):
        raise EvidenceError("native A2 records/F/environment transport differs")
    store.read(_native_pin(inventory_pin))
    result = store.json(sealed["resource-result.json"])
    ledger = operational_json(store.read(sealed["resource-ledger.json"]))
    entries = [
        row for row in ledger["entries"] if row.get("id") == result.get("ledger_id")
    ]
    if len(entries) != 1:
        raise EvidenceError("native A2 terminal worker identity missing/duplicate")
    entry = entries[0]
    authorization_path = entry["command"][-1]
    authorization_pin = {
        "path": authorization_path,
        "sha256": hashes["authorization.json"],
        "kind": "native_reference_provenance",
    }
    if (
        result.get("status") != "success"
        or result.get("exit_code") != 0
        or result.get("cleanup_confirmed") is not True
        or result.get("launch_state") != "started"
        or entry.get("kind") != "formal"
        or entry.get("status") != "success"
        or entry.get("exit_code") != 0
        or entry.get("cleanup_confirmed") is not True
        or type(entry.get("pid")) is not int
        or entry["pid"] <= 0
        or not 0 <= entry.get("elapsed_seconds", -1) <= 3600
        or entry["command"][-2] != "--authorization"
    ):
        raise EvidenceError("native A2 terminal duration/cleanup/argv differs")
    store.read(authorization_pin)
    admission = store.json(_native_pin(auth["root_source_admission"]))
    durable = store.json(_native_pin(auth["durable_E"]))
    integration = store.json(_native_pin(auth["root_E_integration_admission"]))
    if (
        admission.get("decision") != "PASS_FOR_REVIEWED_REFERENCE_SOURCE_INTEGRATION"
        or integration.get("decision") != "PASS_REVIEWED_E_INTEGRATED"
        or durable.get("E_commit") != auth["E_commit"]
        or durable.get("E_tree") != auth["E_tree"]
        or integration.get("E_commit") != auth["E_commit"]
        or integration.get("E_tree") != auth["E_tree"]
        or durable.get("restored_tree") != auth["E_tree"]
        or durable.get("clean_E") is not True
        or durable.get("reviewed_source_hashes_by_group")
        != auth["reviewed_source_hashes_by_group"]
    ):
        raise EvidenceError("native A2 source/durable E admission differs")
    store.read(_native_pin(durable["bundle"]))
    reviewers = []
    for role in ("science", "numerical", "provenance"):
        review = store.json(_native_pin(auth["reviews"][role]))
        if (
            review.get("role") != role
            or review.get("decision") != "PASS"
            or review.get("scope") != "reference implementation"
            or review.get("families") != ["A2"]
            or review.get("reviewed_commit") != admission["reviewed_commit"]
            or review.get("file_hashes_by_group")
            != auth["reviewed_source_hashes_by_group"]
        ):
            raise EvidenceError("native A2 actual implementation review differs")
        for original in review["original_reviews"]:
            store.read(_native_pin(original))
        reviewers.append(review["reviewer_id"])
    if len(set(reviewers)) != 3:
        raise EvidenceError("native A2 source reviewers not distinct")
    canonical = json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n"
    return {
        "family": "A2",
        "E_commit": auth["E_commit"],
        "E_tree": auth["E_tree"],
        "records_sha256": artifacts["records"]["sha256"],
        "seal_sha256": artifacts["seal"]["sha256"],
        "environment_sha256": environment["sha256"],
        "expected_keys_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "frozen_artifacts_sha256": {
            name: artifacts[name]["sha256"]
            for name in ("freeze", "fixture", "budget", "family_procedure")
        },
        "source_pin": {
            "head": auth["E_commit"],
            "tree": auth["E_tree"],
            "files_sha256": auth["file_hashes_by_group"],
        },
        "artifact_hashes": hashes,
    }


def operational_json(raw):
    """Clock JSON is finite operational metadata, never scientific input JSON."""

    def pairs(items):
        result = {}
        for name, value in items:
            if name in result:
                raise EvidenceError("duplicate A2 operational metadata field")
            result[name] = value
        return result

    def nonfinite(_value):
        raise EvidenceError("nonfinite A2 operational metadata")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise EvidenceError("nonfinite A2 operational clock")
        return number

    return json.loads(
        raw, object_pairs_hook=pairs, parse_constant=nonfinite, parse_float=finite_float
    )


def _owned_entries(state, controller_pid):
    owners, selected, pending = {controller_pid}, [], list(state["entries"])
    while True:
        found = [row for row in pending if row.get("parent_pid") in owners]
        if not found:
            return selected
        for row in found:
            selected.append(row)
            pending.remove(row)
            if type(row.get("pid")) is int:
                owners.add(row["pid"])


def verify_managed_completion(
    store,
    *,
    closure,
    resource_pin,
    resource,
    supervisor_pin,
    supervisor,
    attempt_pin,
    completion_pin,
    partial=False,
):
    """Verify the retained exact completion profile without local reexecution.

    The conditions follow the admitted execution adapter's
    verify_result_profile. A partial scientific result is deliberately not
    accepted by the default PASS-only completion interpretation. The partial
    mode reads lifecycle evidence only; a separate issued eligibility packet
    and the original three reviews are required before any partial summary.
    """
    if completion_pin is None:
        raise EvidenceError("A2 requires actual root controller completion")
    sources = {row["path"]: row["sha256"] for row in closure["modules"]}
    for name, module_name in (
        (
            "managed_child_adapter",
            "tests.oracles.gen1.stochastic_formal.common.a2_resource_adapter",
        ),
        ("normative_budget_source", "tests.oracles.gen1.stochastic_formal.a2.budget"),
    ):
        pin = resource[name]
        if sources.get(pin["path"]) != pin["sha256"] or not any(
            row["name"] == module_name
            and row["path"] == pin["path"]
            and row["sha256"] == pin["sha256"]
            for row in closure["modules"]
        ):
            raise EvidenceError("A2 operational source absent from actual closure")
        store.read(pin)
    proof = store.json(completion_pin)
    if (
        set(proof) != COMPLETION_FIELDS
        or proof["schema"] != "gen1-stochastic-a2-managed-controller-completion-v1"
        or proof["issuer"] != "root"
        or proof["decision"] != "CONTROLLER_AND_MANAGED_CHILD_COMPLETE"
        or any(
            value.get("managed_profile") != PROFILE
            for value in (proof, resource, supervisor)
        )
        or proof["aggregate_budget_reservation"]
        != resource["aggregate_budget_reservation"]
        or supervisor["aggregate_budget_reservation"]
        != resource["aggregate_budget_reservation"]
        or proof["controller_id"] != resource["controller_id"]
        or proof["controller_id"] != supervisor["controller_id"]
        or proof["resource_launch"] != resource_pin
        or proof["supervisor_seal"] != supervisor_pin
        or proof["attempt_seal"] != attempt_pin
        or resource["aggregate_budget_reservation"]["ledger_path"] != LEDGER_PATH
        or resource["managed_child_adapter"]["sha256"]
        not in RUNTIME_PROFILE_SOURCES.get(
            resource["normative_budget_source"]["sha256"], ()
        )
    ):
        raise EvidenceError("A2 completion/actual resource/attempt/source differs")
    store.read(resource["managed_environment_executable"])
    terminal = operational_json(
        store.read(supervisor["artifacts"]["managed-ledger-snapshot.json"])
    )
    later_state = operational_json(store.read(proof["postwait_ledger_snapshot"]))
    span = operational_json(store.read(proof["root_external_span"]))
    worker_id = supervisor["managed_ledger_id"]
    rows = [row for row in terminal["entries"] if row.get("id") == worker_id]
    later = [row for row in later_state["entries"] if row.get("id") == worker_id]
    controllers = [
        row
        for row in later_state.get("controllers", [])
        if row.get("id") == proof["controller_id"]
    ]
    if len(rows) != 1 or later != rows or len(controllers) != 1:
        raise EvidenceError("A2 worker/controller terminal identity missing/changed")
    row, controller = rows[0], controllers[0]
    if (
        (not partial and (row.get("status") != "success" or row.get("exit_code") != 0))
        or partial
        and (
            not isinstance(row.get("status"), str)
            or not row["status"]
            or type(row.get("exit_code")) is not int
            or row["exit_code"] != supervisor.get("exit_code")
        )
        or row.get("cleanup_confirmed") is not True
        or row.get("kind") != "repair_validation"
        or type(row.get("elapsed_seconds")) not in (int, float)
        or not 0 <= row["elapsed_seconds"] <= 3600
        or row.get("parent_pid") != resource["supervisor_pid"]
        or row.get("pid") != supervisor["cleanup"]["pid"]
        or str(row.get("elapsed_seconds")) != supervisor["elapsed_seconds"]
        or controller.get("status") != "completed"
        or type(controller.get("deadline_seconds")) is not int
        or not 1 <= controller["deadline_seconds"] <= 3600
        or type(controller.get("wall_bound_seconds")) is not int
        or controller["wall_bound_seconds"] != controller["deadline_seconds"] + 100
        or controller["wall_bound_seconds"] > 3700
        or (not partial and controller.get("outcome") != "success")
        or (partial and controller.get("outcome") not in {"success", "error"})
        or controller.get("pid") != resource["supervisor_pid"]
        or controller.get("external_span_pin")
        != {
            "path": proof["root_external_span"]["path"],
            "sha256": proof["root_external_span"]["sha256"],
        }
        or span.get("issuer") != "root"
        or span.get("decision") != "CONTROLLER_EXTERNAL_SPAN_ATTESTED"
        or span.get("controller_id") != proof["controller_id"]
        or span.get("controller_argv") != controller["command"]
        or span.get("started_monotonic") != controller["started_monotonic"]
        or span.get("elapsed_seconds") != controller["external_span_upper_seconds"]
        or not 0 <= span["elapsed_seconds"] <= controller["wall_bound_seconds"]
    ):
        raise EvidenceError("A2 actual post-wait cleanup/duration differs")
    token = (
        RESOURCE_TOKEN
        + "="
        + json.dumps(resource_pin, sort_keys=True, separators=(",", ":"))
    )
    if (
        row["command"][:3] != ["/usr/bin/env", "-i", "--"]
        or token not in row["command"]
    ):
        raise EvidenceError("A2 terminal exact managed command is unbound")
    if any(
        "elapsed_seconds" not in child or child.get("cleanup_confirmed") is not True
        for child in _owned_entries(later_state, resource["supervisor_pid"])
    ):
        raise EvidenceError("A2 controller descendants live/unaccounted")
    return proof


def replay_a2_arithmetic(store, stage, *, partial=False):
    """Replay exact scalar/CSD arithmetic; keep missing and rejected keys visible."""
    from fractions import Fraction

    from .arithmetic import _binary64, jsonl, load_kernel
    from .graph import assessment_ratios, key_id

    policy, attempt = store.json(stage["policy"]), store.json(stage["attempt"])
    artifacts = policy["artifacts"]
    fixture, budget, scales = [
        store.json(artifacts[name]) for name in ("fixture", "budget", "source_scales")
    ]
    expected = fixture["expected_keys"] + fixture["expected_scalar_keys"]
    ids = [key_id(key) for key in expected]
    if len(ids) != 84 or len(set(ids)) != 84 or any(key[0] != "A2" for key in ids):
        raise EvidenceError("A2 arithmetic requires exact 84 frozen keys")
    references = jsonl(store, artifacts["records"])
    comparison_pin = attempt["artifacts"]["records"]
    comparisons = (
        {}
        if partial and store.read(comparison_pin) == b""
        else jsonl(store, comparison_pin)
    )
    scale_ids = [key_id(row["key"]) for row in scales["rows"]]
    if (
        scales["schema"] != "gen1-stochastic-source-scales-v1"
        or scales["family"] != "A2"
        or len(scale_ids) != len(set(scale_ids))
        or set(scale_ids) != set(ids)
        or set(references) != set(ids)
        or not set(comparisons) <= set(ids)
        or not partial
        and set(comparisons) != set(ids)
    ):
        raise EvidenceError("A2 retained arithmetic frozen coverage differs")
    kernel = load_kernel(store, artifacts["decision_kernel"])
    source_scales = {key_id(row["key"]): row["value"] for row in scales["rows"]}
    counts = {"science": 0, "auxiliary": 0}
    passed = {"science": 0, "auxiliary": 0}
    retained_passed = {"science": 0, "auxiliary": 0}
    retained_failed = []
    failed, unresolved, maximum = [], [], {"D_over_B": None, "U_over_B": None}
    unreplayed_units = []
    scalar_count, csd_count = 0, 0
    for key, identity in zip(expected, ids, strict=True):
        counts[key["set"]] += 1
        reference, reference_hash = references[identity]
        scalar = key["observable"] == "isotropic_depth_factor"
        scalar_count += scalar
        csd_count += not scalar
        if (
            reference["uncertainty"]["kind"] != "estimate"
            or reference["status"] != "resolved"
        ):
            raise EvidenceError(
                "A2 requires the actual resolved uncertainty-estimate R"
            )
        if identity not in comparisons:
            unresolved.append(key)
            continue
        row = comparisons[identity][0]
        if row.get("status") != "resolved":
            if (
                row.get("status") != "unresolved"
                or not partial
                or row.get("passed") is True
            ):
                raise EvidenceError("unresolved A2 record cannot claim PASS")
            unresolved.append(key)
            continue
        quantity = row["production_quantity"]
        if (
            row["reference_record_sha256"] != reference_hash
            or row["reference_purpose"] != "formal-science"
            or row["reference_set"] != key["set"]
            or type(row.get("passed")) is not bool
            or quantity["shape"] != ([] if scalar else [3, 3])
            or reference["quantity"]["shape"] != quantity["shape"]
            or scalar
            and (key["representation"] != "Gfree" or key["set"] != "auxiliary")
        ):
            raise EvidenceError("A2 actual reference/scalar/physical binding differs")
        if row["passed"]:
            retained_passed[key["set"]] += 1
        else:
            retained_failed.append(key)
        for value in quantity["data"]:
            _binary64(value["real"], kernel)
            _binary64(value["imag"], kernel)
        g_squared = True
        if not scalar:
            g = float(reference["inputs"]["g_si"])
            if not math.isfinite(g) or g <= 0:
                raise EvidenceError("A2 frozen G is not finite positive")
            gf_key = key_id({**key, "representation": "Gfree"})
            si_key = key_id({**key, "representation": "SI"})
            if gf_key not in comparisons or si_key not in comparisons:
                if not partial:
                    raise EvidenceError("A2 CSD Gfree/SI counterpart missing")
                unreplayed_units.append(key)
                continue
            gf, si = comparisons[gf_key][0], comparisons[si_key][0]
            if gf.get("status") != "resolved" or si.get("status") != "resolved":
                if not partial:
                    raise EvidenceError("A2 CSD counterpart unresolved")
                unreplayed_units.append(key)
                continue
            for p, q in zip(
                gf["production_quantity"]["data"],
                si["production_quantity"]["data"],
                strict=True,
            ):
                for part in ("real", "imag"):
                    actual_si = _binary64(p[part], kernel) * g**2
                    g_squared &= kernel.decimal_fraction(
                        kernel.promote_binary64(actual_si)
                    ) == kernel.decimal_fraction(q[part])
        connection = row["connection_assessment"]
        if (
            set(connection) != {"gates", "evidence"}
            or set(connection["gates"]) != set(policy["connection_requirements"])
            or any(type(value) is not bool for value in connection["gates"].values())
        ):
            raise EvidenceError("A2 actual connection obligations missing/unknown")
        tolerance = kernel.fraction_decimal(
            kernel.decimal_fraction(budget["structural_rate"])
            * kernel.decimal_fraction(source_scales[identity]),
            180,
            downward=True,
        )
        structures = kernel.structure_gates(
            quantity, "none" if scalar else "horizontal_vertical_2", tolerance
        )
        contracts = {
            "units": reference["units"]
            == ("1" if scalar else fixture["units"][key["representation"]])
            and g_squared,
            "shape": quantity["shape"] == reference["quantity"]["shape"],
            "frequency_axis": reference["key"]["frequency_hz"] == key["frequency_hz"],
            "input_normalization": True,
            "source_connections": all(connection["gates"].values()),
            "resolved_reference": True,
            "sealed_exact_coverage": True,
        }
        assessment = kernel.assess(
            quantity,
            reference["quantity"],
            source_scales[identity],
            budget["absolute_rate"],
            budget["relative_rate"],
            reference["uncertainty"],
            structures,
            contracts,
        )
        if (
            assessment != row["assessment"]
            or structures != row["structure_conditions"]
            or contracts != row["contract_conditions"]
            or row["passed"] != assessment["accepted"]
        ):
            raise EvidenceError("A2 retained arithmetic/structural/units differs")
        if assessment["accepted"]:
            passed[key["set"]] += 1
        else:
            failed.append(key)
        ratios = assessment_ratios(assessment, whole_zero=False)
        for name in maximum:
            if ratios[name] is not None:
                ratio = Fraction(ratios[name])
                if maximum[name] is None or ratio > Fraction(maximum[name]["value"]):
                    maximum[name] = {
                        "value": str(ratio),
                        "display_upper": kernel.fraction_decimal(
                            ratio, 12, upward=True
                        ),
                        "key": key,
                    }
    if counts != {"science": 54, "auxiliary": 30} or (csd_count, scalar_count) != (
        56,
        28,
    ):
        raise EvidenceError("A2 exact physical/scalar partition differs")
    if not partial and (failed or unresolved):
        raise EvidenceError("A2 bounded adoption cannot contain failed/unresolved keys")
    return {
        "family": "A2",
        "status": "PARTIAL_RETAINED_EVIDENCE"
        if partial
        else "RETAINED_ARITHMETIC_REPLAYED",
        "science_count": 54,
        "auxiliary_count": 30,
        "physical_csd_count": 56,
        "mathematical_scalar_auxiliary_count": 28,
        "passed_counts": passed,
        "passed_counts_scope": "independently replayed frozen assessments only",
        "retained_claimed_passed_counts": retained_passed,
        "retained_claimed_failed_count": len(retained_failed),
        "retained_claimed_failed_keys": retained_failed,
        "unreplayed_resolved_count": len(unreplayed_units),
        "failed_count": len(failed),
        "unresolved_count": len(unresolved),
        "failed_keys": failed,
        "unresolved_keys": unresolved,
        "max_ratios": maximum,
        "unreplayed_units_counterpart_keys": unreplayed_units,
        "max_ratios_scope": "available independently replayed serialized quantities",
        "uncertainty_kind_counts": {
            "estimate": 84,
            "rigorous_bound": 0,
            "exact_zero": 0,
        },
        "numeric_delivery_gates_pass": not failed
        and not unresolved
        and not unreplayed_units,
        "independent_result_admission_verified": False,
        "final_transfer_established": False,
        "original_execution_repeated": False,
        "production_api_calls": 0,
    }


def verify_a2_comparison(store, descriptor):
    """Admit bounded A2 only through its native managed profile, never generic."""
    from .graph import verify_stage

    if (
        not isinstance(descriptor, dict)
        or set(descriptor)
        not in ({"profile", "stages"}, {"profile", "stages", "partial_eligibility"})
        or descriptor["profile"] != PROFILE
        or not isinstance(descriptor["stages"], list)
        or len(descriptor["stages"]) != 1
        or descriptor["stages"][0].get("phase") != "comparison"
    ):
        raise EvidenceError("exact one-stage native A2 managed profile required")
    stage = descriptor["stages"][0]
    if "partial_eligibility" in descriptor:
        return [
            verify_partial_evidence(store, stage, descriptor["partial_eligibility"])
        ]
    result = verify_stage(store, stage, family="A2", profile=PROFILE)
    result["summary"] = replay_a2_arithmetic(store, stage)
    result["summary"]["independent_result_admission_verified"] = True
    return [result]


def verify_partial_evidence(store, stage, eligibility_pin):
    """Read a reviewed frozen-budget partial outcome, never relabel it bounded.

    The root issues this *only* if the ordinary three result reviewers establish
    the previously frozen partial eligibility. No packet exists merely because
    a run failed. This parser cannot authorize or execute another attempt.
    """
    from .graph import (
        ROLES,
        STAGE_FIELDS,
        _fields,
        _keys,
        _reviews,
        key_id,
        verify_input_admission,
    )

    _fields(stage, STAGE_FIELDS | {"managed_controller_completion"}, "partial A2 stage")
    policy, candidate, authority, attempt = [
        store.json(stage[name])
        for name in ("policy", "candidate", "authorization", "attempt")
    ]
    eligibility = store.json(eligibility_pin)
    _fields(
        candidate,
        {"schema", "family", "phase", "status", "pins", "evidence"},
        "partial candidate",
    )
    _fields(
        authority,
        {"schema", "family", "phase", "status", "pins", "candidate", "reviews"},
        "partial authority",
    )
    fields = {
        "schema",
        "issuer_id",
        "issued_at",
        "family",
        "decision",
        "reason",
        "pins",
        "attempt",
        "resource_supervisor",
        "frozen_artifacts_sha256",
        "records_sha256",
        "counts",
        "reviews",
        "excluded_reviewer_ids",
        "frozen_budget_exhaustion_evidence",
    }
    artifacts = policy["artifacts"]
    frozen = {
        name: artifacts[name]["sha256"]
        for name in ("fixture", "budget", "family_procedure", "freeze")
    }
    if (
        set(eligibility) != fields
        or eligibility["schema"] != "gen1-root-a2-partial-eligibility-v1"
        or eligibility["family"] != "A2"
        or eligibility["issuer_id"] != "/root"
        or not isinstance(eligibility["issued_at"], str)
        or not eligibility["issued_at"]
        or eligibility["decision"] != "PARTIAL_NUMERICAL_CONDITIONING_UNRESOLVED"
        or eligibility["reason"]
        != "FROZEN_RESOURCE_BUDGET_EXHAUSTED_NUMERICAL_CONDITIONING_UNRESOLVED"
        or eligibility["attempt"] != stage["attempt"]
        or eligibility["resource_supervisor"] != stage["resource_supervisor"]
        or eligibility["pins"] != attempt["pins"]
        or eligibility["frozen_artifacts_sha256"] != frozen
        or eligibility["records_sha256"] != attempt["artifacts"]["records"]["sha256"]
        or eligibility["reviews"] != stage["result_reviews"]
        or stage["result_admission"] != eligibility_pin
        or stage["local_validator_proof"] is not None
        or eligibility["excluded_reviewer_ids"] != stage["excluded_reviewer_ids"]
        or not eligibility["excluded_reviewer_ids"]
        or set(eligibility["reviews"]) != ROLES
    ):
        raise EvidenceError("actual root A2 partial eligibility binding differs")
    store.read(eligibility["frozen_budget_exhaustion_evidence"])
    if (
        stage["phase"] != "comparison"
        or policy["family"] != "A2"
        or candidate.get("schema") != "gen1-stochastic-authorization-v2"
        or candidate.get("family") != "A2"
        or candidate.get("phase") != "comparison"
        or candidate.get("status") != "REVIEW_CANDIDATE"
        or authority.get("schema") != "gen1-stochastic-authorization-v2"
        or authority.get("family") != "A2"
        or authority.get("phase") != "comparison"
        or authority.get("status") != "AUTHORIZED"
        or authority.get("candidate") != stage["candidate"]
        or authority.get("reviews") != stage["authorization_reviews"]
        or authority.get("pins") != candidate["pins"]
        or candidate["pins"].get("policy_sha256") != stage["policy"]["sha256"]
        or candidate["evidence"].get("policy") != stage["policy"]
        or any(
            attempt["pins"].get(name) != value
            for name, value in candidate["pins"].items()
        )
        or attempt["pins"].get("authorization_sha256")
        != stage["authorization"]["sha256"]
        or attempt.get("schema") != "gen1-stochastic-execution-seal-v1"
        or attempt.get("family") != "A2"
        or attempt.get("phase") != "comparison"
        or attempt.get("decision") == "PASS"
        or set(attempt["pins"])
        != set(candidate["pins"])
        | {"authorization_sha256", "supervisor_launch_sha256", "resource_launch_sha256"}
    ):
        raise EvidenceError("partial A2 actual immutable authority/run differs")
    for pin in [
        *candidate["evidence"].values(),
        *artifacts.values(),
        *attempt["artifacts"].values(),
    ]:
        store.read(pin)
    runtime = store.json(attempt["artifacts"]["runtime_dependencies"])
    if (
        runtime.get("status") != "MATCH"
        or runtime.get("planned_sha256") != attempt["pins"]["dependency_closure_sha256"]
        or runtime["planned_sha256"] != artifacts["dependency_closure"]["sha256"]
    ):
        raise EvidenceError(
            "partial conditioning cannot waive an actual dependency mismatch"
        )
    _reviews(
        store,
        stage["authorization_reviews"],
        family="A2",
        phase="comparison",
        subject=stage["candidate"],
        subject_pins=candidate["pins"],
        excluded=[],
        result=False,
    )
    reference = verify_input_admission(
        store,
        artifacts["reference_admission"],
        family="A2",
        source=False,
        expected_bindings={
            "records_sha256": artifacts["records"]["sha256"],
            "seal_sha256": artifacts["seal"]["sha256"],
            "frozen_artifacts_sha256": frozen,
        },
        authorship_pin=stage["input_authorship"]["reference"],
    )
    if verify_native_reference(store, artifacts) != reference:
        raise EvidenceError("partial A2 immutable native R differs")
    verify_input_admission(
        store,
        artifacts["production_source_admission"],
        family="A2",
        source=True,
        expected_bindings={
            "production_commit": candidate["pins"]["production_commit"],
            "production_tree": candidate["pins"]["production_tree"],
            "reference_records_sha256": reference["records_sha256"],
            "reference_seal_sha256": reference["seal_sha256"],
            "frozen_artifacts_sha256": frozen,
        },
        authorship_pin=stage["input_authorship"]["production_source"],
    )
    _reviews(
        store,
        stage["result_reviews"],
        family="A2",
        phase="comparison",
        subject=stage["attempt"],
        subject_pins=attempt["pins"],
        excluded=stage["excluded_reviewer_ids"],
        result=True,
    )
    if eligibility["issuer_id"] in {
        store.json(pin)["reviewer_id"] for pin in stage["result_reviews"].values()
    }:
        raise EvidenceError("root partial issuer cannot self-review eligibility")
    supervisor = store.json(stage["resource_supervisor"])
    for pin in supervisor["artifacts"].values():
        store.read(pin)
    if (
        supervisor.get("cleanup_confirmed") is not True
        or supervisor.get("cleanup", {}).get("status") != "confirmed_stopped"
    ):
        raise EvidenceError("partial A2 has not established actual worker cleanup")
    resource_artifact = supervisor["artifacts"]["resource-launch.json"]
    resource_pin = {**resource_artifact, "kind": "family_resource_launch"}
    resource = store.json(resource_artifact)
    if (
        supervisor.get("schema") != "gen1-stochastic-family-supervisor-seal-v1"
        or supervisor.get("family") != "A2"
        or supervisor.get("policy_sha256") != stage["policy"]["sha256"]
        or resource.get("schema") != "gen1-stochastic-family-resource-launch-v1"
        or resource.get("policy_sha256") != stage["policy"]["sha256"]
        or attempt["pins"].get("resource_launch_sha256") != resource_pin["sha256"]
        or attempt["pins"].get("supervisor_launch_sha256")
        != supervisor["artifacts"]["launch.json"]["sha256"]
    ):
        raise EvidenceError("partial A2 actual managed launch binding differs")
    verify_managed_completion(
        store,
        closure=store.json(artifacts["dependency_closure"]),
        resource_pin=resource_pin,
        resource=resource,
        supervisor_pin=stage["resource_supervisor"],
        supervisor=supervisor,
        attempt_pin=stage["attempt"],
        completion_pin=stage["managed_controller_completion"],
        partial=True,
    )
    fixture = store.json(artifacts["fixture"])
    expected = fixture["expected_keys"] + fixture["expected_scalar_keys"]
    launch = store.json(attempt["artifacts"]["launch"])
    if (
        _keys(attempt["expected_keys"]) != _keys(expected)
        or _keys(launch["expected_keys"]) != _keys(expected)
        or launch["pins"] != attempt["pins"]
    ):
        raise EvidenceError("partial A2 changed its frozen keyset/launch")
    summary = replay_a2_arithmetic(store, stage, partial=True)
    counts = {
        name: summary[name]
        for name in (
            "science_count",
            "auxiliary_count",
            "passed_counts",
            "failed_count",
            "unresolved_count",
        )
    }
    # The issued eligibility describes actual retained row dispositions, not
    # arithmetic acceptance. Unpaired rows remain visible but unreplayed.
    counts["passed_counts"] = summary["retained_claimed_passed_counts"]
    counts["failed_count"] = summary["retained_claimed_failed_count"]
    if eligibility["counts"] != counts or not (
        counts["failed_count"] or counts["unresolved_count"]
    ):
        raise EvidenceError("root partial counts differ from actual retained records")
    summary.update(
        status="PARTIAL_NUMERICAL_CONDITIONING_UNRESOLVED",
        numeric_delivery_gates_pass=False,
        independent_partial_evidence_review_verified=True,
        independent_result_admission_verified=False,
    )
    raw = store.read(attempt["artifacts"]["records"])
    from .archive import decode_json

    rows = [decode_json(line) for line in raw.splitlines()]
    if len(rows) != attempt["completed"] or len(
        {key_id(row["key"]) for row in rows}
    ) != len(rows):
        raise EvidenceError("partial actual completed count differs")
    return {
        "phase": "comparison",
        "pins": attempt["pins"],
        "records": rows,
        "summary": summary,
        "partial_eligibility": eligibility_pin,
    }
