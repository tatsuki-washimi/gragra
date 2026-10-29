"""Verify a selected comparison byte graph without importing production modules."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

from .archive import EvidenceError, decode_json

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
    "input_authorship",
}


def key_id(key):
    if (
        not isinstance(key, dict)
        or set(key) != set(KEY_FIELDS)
        or any(not isinstance(key[n], str) or not key[n] for n in KEY_FIELDS)
    ):
        raise EvidenceError("invalid complete frozen key")
    return tuple(key[n] for n in KEY_FIELDS)


def _keys(keys):
    values = [key_id(key) for key in keys]
    if len(values) != len(set(values)):
        raise EvidenceError("duplicate expected/observed key")
    return set(values)


def _fields(value, fields, label):
    if not isinstance(value, dict) or set(value) != fields:
        raise EvidenceError(label + " schema mismatch")


def verify_authorship_attestation(
    store, authorship, *, schema, original_schema, bindings
):
    """Bind a projected author roster to an actually issued scoped attestation."""
    fields = {"schema", "authors", "original_attestation"} | set(bindings)
    _fields(authorship, fields, "subject authorship")
    if authorship["schema"] != schema or any(
        authorship[name] != value for name, value in bindings.items()
    ):
        raise EvidenceError("subject authorship binding differs")
    authors = set()
    if not isinstance(authorship["authors"], list) or not authorship["authors"]:
        raise EvidenceError("finite nonempty actual author roster required")
    for author in authorship["authors"]:
        _fields(author, {"reviewer_id", "source_artifacts"}, "subject author")
        identity = author["reviewer_id"]
        if (
            not isinstance(identity, str)
            or not identity
            or identity in authors
            or not isinstance(author["source_artifacts"], list)
            or not author["source_artifacts"]
        ):
            raise EvidenceError("actual author/source identities required")
        authors.add(identity)
        for pin in author["source_artifacts"]:
            store.read(pin)
    original = store.json(authorship["original_attestation"])
    _fields(
        original,
        fields - {"original_attestation"} | {"issuer_id", "issued_at"},
        "original authorship attestation",
    )
    if (
        original["schema"] != original_schema
        or any(original[name] != value for name, value in bindings.items())
        or original["authors"] != authorship["authors"]
        or not isinstance(original["issuer_id"], str)
        or not original["issuer_id"]
        or original["issuer_id"] in authors
        or not isinstance(original["issued_at"], str)
        or not original["issued_at"]
    ):
        raise EvidenceError("issued original authorship subjects/roster differ")
    return authors


def _reviews(store, pins, *, family, phase, subject, subject_pins, excluded, result):
    if not isinstance(pins, dict) or set(pins) != ROLES:
        raise EvidenceError("three independent review roles required")
    ids = set()
    for role, pin in pins.items():
        review = store.json(pin)
        if result:
            _fields(
                review,
                {
                    "schema",
                    "family",
                    "phase",
                    "role",
                    "reviewer_id",
                    "decision",
                    "pins",
                    "seal_sha256",
                    "original_review",
                },
                "result review",
            )
            if (
                review["schema"] != "gen1-stochastic-comparison-result-review-v1"
                or review["family"] != family
                or review["phase"] != phase
                or review["seal_sha256"] != subject["sha256"]
            ):
                raise EvidenceError("result review subject differs")
            store.read(review["original_review"])
        else:
            _fields(
                review,
                {
                    "schema",
                    "role",
                    "reviewer_id",
                    "decision",
                    "candidate_sha256",
                    "pins",
                },
                "authorization review",
            )
            if (
                review["schema"] != "gen1-stochastic-review-v1"
                or review["candidate_sha256"] != subject["sha256"]
            ):
                raise EvidenceError("authorization review subject differs")
        reviewer = review["reviewer_id"]
        if (
            not isinstance(reviewer, str)
            or not reviewer
            or reviewer in ids
            or reviewer in excluded
            or review["role"] != role
            or review["decision"] != "PASS"
            or review["pins"] != subject_pins
        ):
            raise EvidenceError("independent exact-subject PASS review required")
        ids.add(reviewer)


def verify_local_proof(store, proof_pin, *, family, phase, profile, subjects, run_pins):
    """Verify retained mechanical proof binding; never claim local reexecution."""
    proof = store.json(proof_pin)
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
        "local validator proof",
    )
    if (
        proof["schema"] != "gen1-stochastic-local-validator-proof-v1"
        or proof["family"] != family
        or proof["phase"] != phase
        or proof["profile"] != profile
        or proof["decision"] != "PASS"
        or proof["subjects"] != subjects
        or proof["pins"] != run_pins
    ):
        raise EvidenceError("actual-environment validator proof binding differs")
    _fields(
        proof["validator"], {"source", "qualified_function"}, "local validator identity"
    )
    known = {
        "a2-ledger-v1": {
            "tests.oracles.gen1.stochastic_formal.common.family_supervisor.verify_family_result"
        },
        "generic-v1": {
            "tests.oracles.gen1.stochastic_formal.common.family_supervisor.verify_family_result"
        },
        "b1-v1": {
            "tests.oracles.gen1.stochastic_formal.common.transport.validate_connection_pass"
        }
        if phase == "connection"
        else {"__main__"},
    }
    if (
        proof["validator"]["qualified_function"] not in known[profile]
        or not isinstance(proof["argv"], list)
        or not proof["argv"]
        or any(not isinstance(x, str) or not x for x in proof["argv"])
    ):
        raise EvidenceError("unknown execution validator identity/argv")
    store.read(proof["validator"]["source"])
    original = store.json(proof["original_proof"])
    if (
        original.get("schema") != "gen1-root-family-result-mechanical-verification-v1"
        or original.get("result", {}).get("decision") != "PASS"
        or original["result"].get("pins") != run_pins
    ):
        raise EvidenceError("retained local mechanical proof differs")
    if (
        original["result"].get("attempt") != subjects["attempt"]
        or original["result"].get("resource_supervisor")
        != subjects["resource_supervisor"]
        or original.get("authorization") != subjects["authorization"]
        or original.get("authorization_candidate") != subjects["candidate"]
        or original.get("comparator_source") != proof["validator"]["source"]
        or original.get("production_api_calls_during_verification") != 0
        or len(proof["argv"]) < 2
        or original.get("script", {}).get("path") != proof["argv"][1]
    ):
        raise EvidenceError("original validator actual subjects/source/argv differ")
    store.read(original["script"])
    policy = store.json(subjects["policy"])
    closure = store.json(policy["artifacts"]["dependency_closure"])
    validator = proof["validator"]["source"]
    if not any(
        module["name"]
        == "tests.oracles.gen1.stochastic_formal.common.family_supervisor"
        and module["path"] == validator["path"]
        and module["sha256"] == validator["sha256"]
        for module in closure["modules"]
    ):
        raise EvidenceError(
            "validator source is outside actual admitted planned closure"
        )


def assessment_ratios(assessment, *, whole_zero=False):
    """Retain exact rational bounds; display rounding is a caller decision."""
    names = ("difference_upper", "uncertainty_upper", "budget_lower")
    try:
        values = [Decimal(assessment[n]) for n in names]
    except (KeyError, ValueError, TypeError) as error:
        raise EvidenceError("missing serialized assessment bounds") from error
    if any(not x.is_finite() or x < 0 for x in values):
        raise EvidenceError("invalid finite nonnegative assessment bound")
    difference, uncertainty, budget = map(Fraction, values)
    if not budget:
        if not whole_zero or difference or uncertainty:
            raise EvidenceError("zero budget requires certified whole-zero result")
        return {
            "D_over_B": None,
            "U_over_B": None,
            "reason": "CERTIFIED_WHOLE_ZERO_BUDGET",
        }
    return {
        "D_over_B": str(difference / budget),
        "U_over_B": str(uncertainty / budget),
        "reason": None,
    }


def verify_stage(store, stage, *, family, profile):
    stage_fields = (
        STAGE_FIELDS | {"managed_controller_completion"}
        if profile == "a2-ledger-v1"
        else STAGE_FIELDS
    )
    _fields(stage, stage_fields, "comparison stage")
    standard = profile in {"generic-v1", "a2-ledger-v1"}
    if (profile == "a2-ledger-v1") != (family == "A2"):
        raise EvidenceError("managed A2 profile cannot admit another family")
    phase = stage["phase"]
    if phase not in {"connection", "comparison"} or standard and phase != "comparison":
        raise EvidenceError("unknown comparison phase")
    excluded = stage["excluded_reviewer_ids"]
    if (
        not isinstance(excluded, list)
        or any(not isinstance(x, str) or not x for x in excluded)
        or len(excluded) != len(set(excluded))
    ):
        raise EvidenceError("explicit excluded author identities required")
    policy, candidate, authority, attempt = [
        store.json(stage[name])
        for name in ("policy", "candidate", "authorization", "attempt")
    ]
    _fields(
        candidate,
        {"schema", "family", "phase", "status", "pins", "evidence"},
        "candidate",
    )
    _fields(
        authority,
        {"schema", "family", "phase", "status", "pins", "candidate", "reviews"},
        "authority",
    )
    for subject, status in ((candidate, "REVIEW_CANDIDATE"), (authority, "AUTHORIZED")):
        if (
            subject["schema"] != "gen1-stochastic-authorization-v2"
            or subject["family"] != family
            or subject["phase"] != phase
            or subject["status"] != status
        ):
            raise EvidenceError("candidate/authority phase binding differs")
    if (
        authority["candidate"] != stage["candidate"]
        or authority["reviews"] != stage["authorization_reviews"]
        or authority["pins"] != candidate["pins"]
        or candidate["pins"].get("policy_sha256") != stage["policy"]["sha256"]
    ):
        raise EvidenceError("stale policy/candidate authorization")
    if candidate["evidence"].get("policy") != stage["policy"]:
        raise EvidenceError("candidate must bind actual policy artifact")
    for pin in candidate["evidence"].values():
        store.read(pin)
    for pin in policy["artifacts"].values():
        store.read(pin)
    if standard:
        artifacts = policy["artifacts"]
        if (
            "reference_admission" in artifacts
            or "production_source_admission" in artifacts
        ):
            _fields(
                stage["input_authorship"],
                {"reference", "production_source"},
                "input authorship pins",
            )
            frozen_hashes = {
                name: artifacts[name]["sha256"]
                for name in ("fixture", "budget", "family_procedure", "freeze")
            }
            reference = verify_input_admission(
                store,
                artifacts["reference_admission"],
                family=family,
                source=False,
                expected_bindings={
                    "records_sha256": artifacts["records"]["sha256"],
                    "seal_sha256": artifacts["seal"]["sha256"],
                    "frozen_artifacts_sha256": frozen_hashes,
                },
                authorship_pin=stage["input_authorship"]["reference"],
            )
            if profile == "a2-ledger-v1":
                from .a2 import verify_native_reference

                if verify_native_reference(store, artifacts) != reference:
                    raise EvidenceError(
                        "native A2 reference projection differs from admitted pins"
                    )
            verify_input_admission(
                store,
                artifacts["production_source_admission"],
                family=family,
                source=True,
                expected_bindings={
                    "production_commit": candidate["pins"]["production_commit"],
                    "production_tree": candidate["pins"]["production_tree"],
                    "reference_records_sha256": reference["records_sha256"],
                    "reference_seal_sha256": reference["seal_sha256"],
                    "frozen_artifacts_sha256": frozen_hashes,
                },
                authorship_pin=stage["input_authorship"]["production_source"],
            )
    _reviews(
        store,
        stage["authorization_reviews"],
        family=family,
        phase=phase,
        subject=stage["candidate"],
        subject_pins=candidate["pins"],
        excluded=[],
        result=False,
    )
    run_pins = attempt["pins"]
    if (
        attempt.get("schema") != "gen1-stochastic-execution-seal-v1"
        or attempt.get("family") != family
        or attempt.get("phase") != phase
        or attempt.get("decision") != "PASS"
        or attempt.get("exit_code") != 0
        or attempt.get("exception") is not None
    ):
        raise EvidenceError("actual attempt not completely successful")
    if (
        any(run_pins.get(n) != value for n, value in candidate["pins"].items())
        or run_pins.get("authorization_sha256") != stage["authorization"]["sha256"]
    ):
        raise EvidenceError("actual run has stale immutable P/authority pins")
    if standard and set(run_pins) != set(candidate["pins"]) | {
        "authorization_sha256",
        "supervisor_launch_sha256",
        "resource_launch_sha256",
    }:
        raise EvidenceError("unknown material actual run pin")
    for pin in attempt["artifacts"].values():
        store.read(pin)
    launch = store.json(attempt["artifacts"]["launch"])
    expected = _keys(attempt["expected_keys"])
    if standard:
        fixture = store.json(policy["artifacts"]["fixture"])
        frozen = _keys(
            fixture["expected_keys"]
            + fixture.get("expected_auxiliary_keys", [])
            + (
                fixture.get("expected_scalar_keys", [])
                if profile == "a2-ledger-v1"
                else []
            )
        )
        if expected != frozen:
            raise EvidenceError("actual coverage is not full immutable frozen keyset")
    if (
        not expected
        or launch["pins"] != run_pins
        or _keys(launch["expected_keys"]) != expected
        or attempt["completed"] != len(expected)
    ):
        raise EvidenceError("attempt launch/expected coverage differs")
    raw_records = store.read(attempt["artifacts"]["records"])
    if not raw_records.endswith(b"\n"):
        raise EvidenceError("complete raw record line termination required")
    rows = [decode_json(line) for line in raw_records.splitlines()]
    if (
        len(rows) != len(expected)
        or _keys([r["key"] for r in rows]) != expected
        or any(r["key"]["family"] != family for r in rows)
    ):
        raise EvidenceError("omitted/duplicate/unknown frozen result key")
    for row in rows:
        if (
            row.get("status") != "resolved"
            or row.get("passed") is not True
            or phase == "comparison"
            and row.get("assessment", {}).get("accepted") is not True
        ):
            raise EvidenceError("failed/unresolved row cannot be admitted")
        if standard and (
            row.get("reference_purpose") != "formal-science"
            or row.get("reference_set") != row["key"]["set"]
            or row["key"]["set"] not in {"science", "auxiliary"}
        ):
            raise EvidenceError("calibration/history recount forbidden")
    runtime = store.json(attempt["artifacts"]["runtime_dependencies"])
    if runtime.get("status") != "MATCH" or runtime.get(
        "planned_sha256"
    ) != run_pins.get("dependency_closure_sha256"):
        raise EvidenceError("actual runtime closure mismatch")
    supervisor = store.json(stage["resource_supervisor"])
    for pin in supervisor["artifacts"].values():
        store.read(pin)
    if standard:
        resource = supervisor["artifacts"]["resource-launch.json"]
        if (
            supervisor.get("schema") != "gen1-stochastic-family-supervisor-seal-v1"
            or supervisor.get("family") != family
            or supervisor.get("outcome") != "CHILD_EXITED"
            or supervisor.get("exit_code") != 0
            or supervisor.get("cleanup_confirmed") is not True
            or supervisor.get("policy_sha256") != stage["policy"]["sha256"]
            or run_pins["resource_launch_sha256"] != resource["sha256"]
            or run_pins["supervisor_launch_sha256"]
            != supervisor["artifacts"]["launch.json"]["sha256"]
        ):
            raise EvidenceError("terminal resource supervisor binding differs")
        if profile == "a2-ledger-v1":
            from .a2 import verify_managed_completion

            closure = store.json(policy["artifacts"]["dependency_closure"])
            verify_managed_completion(
                store,
                closure=closure,
                resource_pin={**resource, "kind": "family_resource_launch"},
                resource=store.json(resource),
                supervisor_pin=stage["resource_supervisor"],
                supervisor=supervisor,
                attempt_pin=stage["attempt"],
                completion_pin=stage["managed_controller_completion"],
            )
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
        "result admission",
    )
    if (
        admission["schema"] != "gen1-stochastic-result-admission-v1"
        or admission["family"] != family
        or admission["phase"] != phase
        or admission["decision"] != "PASS"
        or admission["pins"] != run_pins
        or admission["attempt"] != stage["attempt"]
        or admission["reviews"] != stage["result_reviews"]
    ):
        raise EvidenceError("result admission subject differs")
    store.read(admission["original_admission"])
    original_admission = store.json(admission["original_admission"])
    if standard and (
        original_admission.get("schema") != "gen1-root-family-production-acceptance-v1"
        or original_admission.get("decision") != "VERIFIED_BOUNDED"
        or original_admission.get("family") != family
        or original_admission.get("attempt") != stage["attempt"]
        or original_admission.get("resource_supervisor") != stage["resource_supervisor"]
        or original_admission.get("production_comparison_pins") != run_pins
        or original_admission.get("required_result_reviews") != stage["result_reviews"]
        or original_admission.get("excluded_reviewer_ids") != excluded
        or not excluded
    ):
        raise EvidenceError("actual root result/author roster binding differs")
    _reviews(
        store,
        stage["result_reviews"],
        family=family,
        phase=phase,
        subject=stage["attempt"],
        subject_pins=run_pins,
        excluded=excluded,
        result=True,
    )
    subjects = {
        name: stage[name]
        for name in (
            "policy",
            "candidate",
            "authorization",
            "attempt",
            "resource_supervisor",
        )
    }
    verify_local_proof(
        store,
        stage["local_validator_proof"],
        family=family,
        phase=phase,
        profile=profile,
        subjects=subjects,
        run_pins=run_pins,
    )
    return {"phase": phase, "pins": run_pins, "records": rows}


def verify_comparison(store, descriptor, *, family):
    _fields(descriptor, {"profile", "stages"}, "final comparison")
    if family in {"A2", "B1"}:
        raise EvidenceError(
            "family execution profile pending exact reviewed implementation"
        )
    profile = descriptor["profile"]
    if profile not in {"generic-v1", "b1-v1"}:
        raise EvidenceError("unknown/unreviewed comparison profile")
    phases = [stage["phase"] for stage in descriptor["stages"]]
    if phases != (
        ["connection", "comparison"] if family == "B1" else ["comparison"]
    ) or (family == "B1") != (profile == "b1-v1"):
        raise EvidenceError("B1 requires separate ordered 70 then 30 phases")
    results = [
        verify_stage(store, stage, family=family, profile=profile)
        for stage in descriptor["stages"]
    ]
    if family == "B1":
        if len(results[0]["records"]) != 70 or len(results[1]["records"]) != 30:
            raise EvidenceError("B1 requires complete 70/30 coverage")
        connection = descriptor["stages"][0]["attempt"]
        # Science must inherit the exact independently admitted connection seal.
        science_policy = store.json(descriptor["stages"][1]["policy"])
        if science_policy["artifacts"].get("connection_seal") != connection:
            raise EvidenceError(
                "BOUND or R_aux cannot substitute for actual 70-PASS seal"
            )
    return results


def verify_input_admission(
    store, admission_pin, *, family, source, expected_bindings, authorship_pin
):
    """Independent exact-subject input review graph, never a numerical PASS claim."""
    admission = store.json(admission_pin)
    _fields(
        admission,
        {"schema", "family", "decision", "pins", "reviews"},
        "input admission",
    )
    label = "production-source" if source else "reference"
    if (
        admission["schema"] != f"gen1-stochastic-{label}-admission-v1"
        or admission["family"] != family
        or admission["decision"] != "PASS"
    ):
        raise EvidenceError("input admission family/type differs")
    subject = admission["pins"]
    if any(subject.get(name) != value for name, value in expected_bindings.items()):
        raise EvidenceError("input admission actual immutable bindings differ")
    authorship = store.json(authorship_pin)
    if authorship.get("schema") == "gen1-stochastic-input-authorship-legacy-v1":
        from .authorship import verify_legacy_input_authorship

        excluded = verify_legacy_input_authorship(
            store,
            authorship,
            family=family,
            source=source,
            admission_pin=admission_pin,
            admission=admission,
        )
    else:
        excluded = verify_authorship_attestation(
            store,
            authorship,
            schema="gen1-stochastic-input-authorship-v1",
            original_schema="gen1-root-input-authorship-attestation-v1",
            bindings={
                "family": family,
                "subject_kind": "production_source" if source else "reference",
                "admission": admission_pin,
                "pins": subject,
            },
        )
    if set(admission["reviews"]) != ROLES:
        raise EvidenceError("input admission requires three roles")
    identities = set()
    for role, pin in admission["reviews"].items():
        review = store.json(pin)
        fields = {
            "schema",
            "family",
            "role",
            "reviewer_id",
            "decision",
            "pins",
            "original_review",
        } | ({"claim_attestations"} if source else set())
        _fields(review, fields, "input review")
        reviewer = review["reviewer_id"]
        if (
            review["schema"] != f"gen1-stochastic-{label}-review-v1"
            or review["family"] != family
            or review["role"] != role
            or review["decision"] != "PASS"
            or review["pins"] != subject
            or not isinstance(reviewer, str)
            or not reviewer
            or reviewer in identities
            or reviewer in excluded
        ):
            raise EvidenceError("independent exact input review required")
        identities.add(reviewer)
        store.read(review["original_review"])
        if source:
            for claim in review["claim_attestations"].values():
                _fields(claim, {"decision", "scope", "proof_report"}, "source claim")
                if (
                    claim["decision"] != "PASS"
                    or claim["scope"] != "SOURCE_MODEL_BINDING_ONLY"
                ):
                    raise EvidenceError(
                        "source binding cannot claim numerical execution"
                    )
                store.read(claim["proof_report"])
    return subject
