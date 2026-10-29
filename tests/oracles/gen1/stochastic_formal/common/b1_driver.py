"""Versioned two-stage B1 CLI. BOUND receipts authorize connections, not science.

Controller supplies exact policy, authorization and three reviewer digests.
Preflight loads serialized references and pure decision arithmetic only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.oracles.gen1 import compare_stochastic_production as legacy
from tests.oracles.gen1.b1_exact_record_adapter import adapt_b1_records, key_tuple

from . import b1_connections as connections
from .dependencies import verify_formal_closure
from .execution import run_reviewed_phase
from .transport import (
    GateError,
    artifact_pin,
    canonical_bytes,
    exact_keys,
    key_id,
    strict_json,
    validate_bound_receipts,
    verify_artifact,
)

POLICY_SCHEMA = "gen1-stochastic-B1-comparison-policy-v2"
REFERENCE_ADMISSION_SCHEMA = "gen1-stochastic-reference-admission-v1"
BASE_ARTIFACTS = {
    "f_common",
    "fixture",
    "budget",
    "procedure",
    "f_b1_aux",
    "aux_freeze",
    "r_b1_records",
    "r_b1_seal",
    "r_b1_review",
    "r_aux_records",
    "r_aux_seal",
    "r_aux_admission",
    "dependency_closure",
    "bound_receipts",
    "receipt_expectations",
    "decision_kernel",
}


def auxiliary_admission_pins(records_pin: dict, seal_pin: dict) -> dict:
    """Derive reviewed identities from the sealed retained evidence, never reports."""
    seal = strict_json(verify_artifact(seal_pin))
    root = Path(seal_pin["path"]).parent
    fields = (
        "E_aux_SHA",
        "E_aux_tree",
        "f_b1_aux_sha256",
        "family_procedure_sha256",
        "fixtures_sha256",
        "generator_sha256",
        "runner_sha256",
        "freeze_declaration_sha256",
        "integration_head",
        "admission_sha256",
        "expected_environment_sha256",
    )
    pins = {name: seal[name] for name in fields}
    if seal["records_sha256"] != records_pin["sha256"]:
        raise GateError("R_aux retained records mismatch")
    pins.update(records_sha256=records_pin["sha256"], seal_sha256=seal_pin["sha256"])
    retained = []
    for name, field in (
        ("expected-environment.json", "expected_environment_sha256"),
        ("supervisor-environment.json", "supervisor_environment_sha256"),
        ("worker-environment.json", "worker_environment_sha256"),
    ):
        pin = artifact_pin(root / name, "reference_environment")
        if seal["artifact_hashes"].get(name) != pin["sha256"]:
            raise GateError("R_aux retained environment byte mismatch")
        if field == "expected_environment_sha256" and pins[field] != pin["sha256"]:
            raise GateError("R_aux expected environment mismatch")
        pins[field] = pin["sha256"]
        retained.append(verify_artifact(pin))
    if len(set(retained)) != 1:
        raise GateError("R_aux supervisor/worker environment differs")
    environment = strict_json(retained[0])
    if environment.get("production_modules") != []:
        raise GateError("R_aux reference imported production modules")
    for name in ("generator_sha256", "runner_sha256"):
        if environment.get(name) != pins[name]:
            raise GateError("R_aux retained generator/runner identity differs")
    for name in ("python_executable_sha256", "python_implementation", "python_version"):
        if not isinstance(environment.get(name), str) or not environment[name]:
            raise GateError("R_aux retained Python identity incomplete")
        pins[name] = environment[name]
    if not environment.get("loaded_modules"):
        raise GateError("R_aux retained loaded dependency inventory missing")
    return pins


def reviewed_auxiliary_admission(
    admission: dict,
    records_pin: dict,
    seal_pin: dict,
    f_aux: str,
    expected_review_sha: dict,
) -> None:
    if (
        set(admission)
        != {
            "schema",
            "decision",
            "family",
            "f_b1_aux_sha256",
            "records_sha256",
            "seal_sha256",
            "pins",
            "reviews",
        }
        or admission["schema"] != REFERENCE_ADMISSION_SCHEMA
        or admission["decision"] != "PASS"
        or admission["family"] != "B1_aux"
        or admission["f_b1_aux_sha256"] != f_aux
        or admission["records_sha256"] != records_pin["sha256"]
        or admission["seal_sha256"] != seal_pin["sha256"]
    ):
        raise GateError("R_aux reviewed admission mismatch")
    if admission["pins"] != auxiliary_admission_pins(records_pin, seal_pin):
        raise GateError("R_aux admission pins differ from actual sealed environment")
    roles = {"science", "numerical", "provenance"}
    if set(admission["reviews"]) != roles or set(expected_review_sha) != roles:
        raise GateError("R_aux three reviews required")
    reviewer_ids = set()
    for role, pin in admission["reviews"].items():
        if pin["sha256"] != expected_review_sha[role]:
            raise GateError("R_aux review digest differs from trusted policy")
        review = strict_json(verify_artifact(pin))
        if (
            review.get("decision") != "PASS"
            or review.get("role") != role
            or review.get("records_sha256") != records_pin["sha256"]
            or review.get("seal_sha256") != seal_pin["sha256"]
            or review.get("f_b1_aux_sha256") != f_aux
            or review.get("pins") != admission["pins"]
            or not review.get("reviewer_id")
            or review["reviewer_id"] in reviewer_ids
        ):
            raise GateError("R_aux reviewer identity/byte pins mismatch")
        reviewer_ids.add(review["reviewer_id"])


def preflight(policy_pin: dict) -> dict:
    """Validate all endpoint bytes, admitted references and typed planned closure."""
    policy = strict_json(verify_artifact(policy_pin))
    fields = {
        "schema",
        "family",
        "phase",
        "pins",
        "production_repo",
        "comparator_repo",
        "artifacts",
        "endpoints",
        "r_aux_review_sha256",
        "connection_pins",
    }
    if (
        set(policy) != fields
        or policy["schema"] != POLICY_SCHEMA
        or policy["family"] != "B1"
        or policy["phase"] not in {"connection", "comparison"}
    ):
        raise GateError("B1 v2 policy schema mismatch")
    phase = policy["phase"]
    pins = policy["pins"]
    artifacts = policy["artifacts"]
    required = BASE_ARTIFACTS | (
        {"connection_seal"} if phase == "comparison" else set()
    )
    if set(artifacts) != required:
        raise GateError("B1 policy artifact closure incomplete/unknown")
    for artifact in artifacts.values():
        verify_artifact(artifact)
    legacy.verify_clean_git_pin(
        Path(policy["production_repo"]),
        pins["production_commit"],
        pins["production_tree"],
    )
    legacy.verify_clean_git_pin(
        Path(policy["comparator_repo"]),
        pins["comparator_commit"],
        pins["comparator_tree"],
    )
    f_aux = artifacts["f_b1_aux"]["sha256"]
    if pins.get("f_b1_aux_sha256") != f_aux:
        raise GateError("named F_B1_aux hash differs across authority")
    freeze = strict_json(verify_artifact(artifacts["aux_freeze"]))
    if freeze.get("status") != "FROZEN" or freeze.get("f_b1_aux_sha256") != f_aux:
        raise GateError("F_B1_aux is not exactly frozen")
    for name, pin_field in [
        ("f_common", "f_common_sha256"),
        ("fixture", "fixture_sha256"),
        ("budget", "budget_sha256"),
        ("procedure", "procedure_sha256"),
        ("r_b1_records", "r_b1_records_sha256"),
        ("r_b1_seal", "r_b1_seal_sha256"),
        ("r_b1_review", "r_b1_review_sha256"),
        ("r_aux_records", "r_aux_records_sha256"),
        ("r_aux_seal", "r_aux_seal_sha256"),
        ("r_aux_admission", "r_aux_admission_sha256"),
        ("dependency_closure", "dependency_closure_sha256"),
        ("bound_receipts", "bound_receipts_sha256"),
        ("receipt_expectations", "receipt_expectations_sha256"),
        ("decision_kernel", "decision_kernel_sha256"),
    ]:
        if pins.get(pin_field) != artifacts[name]["sha256"]:
            raise GateError(f"authority artifact mismatch: {name}")
    fixtures = strict_json(verify_artifact(artifacts["fixture"]))
    budget = strict_json(verify_artifact(artifacts["budget"]))
    procedure = strict_json(verify_artifact(artifacts["procedure"]))
    if procedure["connection_evaluator_contract"]["gate_fields"] != list(
        connections.CONNECTION_GATES
    ):
        raise GateError("frozen 11-gate contract differs")
    if (
        len(exact_keys(fixtures["expected_keys"])) != 30
        or len(exact_keys(fixtures["expected_reference_auxiliary_keys"])) != 56
        or len(exact_keys(fixtures["expected_consistency_auxiliary_keys"])) != 14
        or exact_keys(fixtures["expected_auxiliary_keys"])
        != exact_keys(fixtures["expected_reference_auxiliary_keys"])
        | exact_keys(fixtures["expected_consistency_auxiliary_keys"])
    ):
        raise GateError("frozen science/reference/consistency partitions invalid")
    expectations = strict_json(verify_artifact(artifacts["receipt_expectations"]))
    receipt_bundle = strict_json(verify_artifact(artifacts["bound_receipts"]))
    binding_pins = {
        "production_commit": pins["production_commit"],
        "production_tree": pins["production_tree"],
        "f_b1_aux_sha256": f_aux,
    }
    validate_bound_receipts(
        receipt_bundle,
        expectations["receipts"],
        fixtures["expected_keys"],
        binding_pins,
    )
    if {key_id(r["key"]) for r in expectations["receipts"]} != exact_keys(
        fixtures["expected_auxiliary_keys"]
    ):
        raise GateError("receipt keys differ from frozen auxiliary keys")
    for row in expectations["receipts"]:
        expected_kind = (
            "independent_reference"
            if key_id(row["key"])
            in exact_keys(fixtures["expected_reference_auxiliary_keys"])
            else "production_consistency"
        )
        if row["kind"] != expected_kind:
            raise GateError("receipt partition differs from frozen science")
    closure = strict_json(verify_artifact(artifacts["dependency_closure"]))
    verify_formal_closure(closure)
    if (
        closure["production_commit"] != pins["production_commit"]
        or closure["production_tree"] != pins["production_tree"]
    ):
        raise GateError("typed closure does not bind actual production P")
    kernel = legacy.load_pinned_kernel(
        policy["production_repo"], commit=pins["production_commit"]
    )
    if artifacts["decision_kernel"]["sha256"] != legacy.KERNEL_SHA256:
        raise GateError("admitted common arithmetic helper changed")
    r_aux_seal = strict_json(verify_artifact(artifacts["r_aux_seal"]))
    if (
        r_aux_seal.get("f_b1_aux_sha256") != f_aux
        or r_aux_seal.get("records_sha256") != artifacts["r_aux_records"]["sha256"]
        or r_aux_seal.get("mode") != "formal"
        or r_aux_seal.get("exit_code") != 0
        or r_aux_seal.get("cleanup_confirmed") is not True
        or r_aux_seal.get("coverage_valid") is not True
    ):
        raise GateError("R_aux formal seal admission incomplete")
    reviewed_auxiliary_admission(
        strict_json(verify_artifact(artifacts["r_aux_admission"])),
        artifacts["r_aux_records"],
        artifacts["r_aux_seal"],
        f_aux,
        policy["r_aux_review_sha256"],
    )
    aux_rows = [
        strict_json(line)
        for line in verify_artifact(artifacts["r_aux_records"]).splitlines()
    ]
    conn_fixtures = {row["fixture_id"]: row for row in fixtures["connection_fixtures"]}
    auxiliary = connections.validate_auxiliary_records(
        aux_rows,
        fixtures["expected_reference_auxiliary_keys"],
        f_aux,
        conn_fixtures,
        budget["connection_source_scales"],
        kernel,
    )
    review = strict_json(verify_artifact(artifacts["r_b1_review"]))
    if (
        review.get("R_B1") != "ACCEPTED_FROZEN_REFERENCE_SET"
        or review.get("attempt", {}).get("records_sha256")
        != artifacts["r_b1_records"]["sha256"]
        or review.get("attempt", {}).get("seal_sha256")
        != artifacts["r_b1_seal"]["sha256"]
    ):
        raise GateError("R_B1 accepted reference byte identity mismatch")
    zero_keys = [
        key
        for key in fixtures["expected_keys"]
        if key["fixture_id"] in {"zero_density", "zero_psd"}
    ]
    if len(zero_keys) != 12:
        raise GateError("B1 whole-zero partition changed")
    science = adapt_b1_records(
        verify_artifact(artifacts["r_b1_records"]),
        review["uncertainty_and_oracle_fraction"]["per_record"],
        fixtures["expected_keys"],
        zero_keys,
    )
    connections.validate_endpoint_map(
        policy["endpoints"], fixtures["expected_auxiliary_keys"]
    )
    # Both reference sets remain distinct: R_aux can only veto source gates.
    # It never supplies the 30 science centers or U_matrix=3*u_up.
    if pins.get("family", "B1") != "B1":
        raise GateError("policy authority family differs from B1")
    auth_pins = {**pins, "family": "B1", "policy_sha256": policy_pin["sha256"]}
    if (
        phase == "comparison"
        and pins.get("connection_pass_sha256") != artifacts["connection_seal"]["sha256"]
    ):
        raise GateError("science authority does not pin exact connection seal")
    return {
        "policy": policy,
        "pins": auth_pins,
        "fixtures": fixtures,
        "budget": budget,
        "kernel": kernel,
        "closure": closure,
        "auxiliary": auxiliary,
        "science": science,
        "zero_keys": zero_keys,
        "connection_fixtures": conn_fixtures,
    }


def science_factory(prepared):
    """Build only after reviewed 70 PASS and exact science authority verified."""
    import importlib

    kernel = prepared["kernel"]
    fixtures = prepared["fixtures"]
    analytic = importlib.import_module("gragra.analytic.stochastic_bodywave")
    observable = importlib.import_module("gragra.observables.stochastic_bodywave")
    science = {row["fixture_id"]: row for row in fixtures["science"]}

    def evaluate(key):
        fixture = science[key["fixture_id"]]
        inputs = legacy.build_production_inputs(key, fixture)
        api = (
            observable.fullspace_p_isotropic_acceleration_csd
            if key["representation"] == "SI"
            else analytic.fullspace_p_isotropic_acceleration_csd_g_free
        )
        result = legacy.assess_one(
            key=key,
            fixture=fixture,
            reference=prepared["science"][key_tuple(key)],
            production_matrix=api(**inputs["kwargs"]),
            expected_units=fixtures["units"][key["representation"]],
            source_connections_passed=True,
            exact_coverage_passed=True,
            budget={**prepared["budget"], "units": fixtures["units"]},
            kernel=kernel,
        )
        return {
            "key": key,
            "status": "resolved",
            "passed": result["accepted"],
            "assessment": result["assessment"],
            "production_quantity": result["production_quantity"],
            "structure_conditions": result["structure_conditions"],
            "contract_conditions": result["contract_conditions"],
            "exact_zero_pass": result["exact_zero_pass"],
            "inputs": inputs["evidence"],
            "reference_central_mapping": "central*I3",
            "reference_uncertainty_mapping": "3*u_up",
        }

    return evaluate


def execute(prepared, *, authorization: dict, reviews: dict, output: str | Path):
    policy = prepared["policy"]
    for name in ("production_repo", "comparator_repo"):
        if Path(output).resolve().is_relative_to(Path(policy[name]).resolve()):
            raise GateError("formal comparison output must be outside source trees")
    phase = policy["phase"]
    artifacts = policy["artifacts"]
    fixture = prepared["fixtures"]
    if phase == "connection":

        def factory():
            return connections.production_factory(
                policy["endpoints"],
                prepared["connection_fixtures"],
                {r["fixture_id"]: r for r in fixture["science"]},
                {
                    key_id(r["key"]): r
                    for r in prepared["budget"]["connection_source_scales"]
                },
                prepared["kernel"],
                prepared["auxiliary"],
                {
                    "commit": policy["pins"]["production_commit"],
                    "tree": policy["pins"]["production_tree"],
                },
                artifacts["r_aux_records"],
                prepared["budget"]["connection_budgets"],
            )

        expected = fixture["expected_auxiliary_keys"]
    else:

        def factory():
            return science_factory(prepared)

        expected = fixture["expected_keys"]
    return run_reviewed_phase(
        authorization=authorization,
        pins=prepared["pins"],
        reviews=reviews,
        phase=phase,
        family="B1",
        expected_keys=expected,
        output=output,
        evaluator_factory=factory,
        dependency_closure=prepared["closure"],
        connection_seal=artifacts.get("connection_seal"),
        connection_keys=fixture["expected_auxiliary_keys"],
        connection_pins=policy["connection_pins"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    for role in ("science", "numerical", "provenance"):
        parser.add_argument(f"--{role}-review-sha256", required=True)
    parser.add_argument("--output")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    prepared = preflight(
        {
            "path": str(Path(args.policy).absolute()),
            "sha256": args.policy_sha256,
            "kind": "policy",
        }
    )
    auth = {
        "path": str(Path(args.authorization).absolute()),
        "sha256": args.authorization_sha256,
        "kind": "authorization",
    }
    reviews = {
        role: getattr(args, f"{role}_review_sha256")
        for role in ("science", "numerical", "provenance")
    }
    if not args.execute:
        from .transport import require_science_authority, verify_authorization

        if prepared["policy"]["phase"] == "comparison":
            require_science_authority(
                auth,
                prepared["pins"],
                reviews,
                connection_seal=prepared["policy"]["artifacts"]["connection_seal"],
                connection_keys=prepared["fixtures"]["expected_auxiliary_keys"],
                connection_pins=prepared["policy"]["connection_pins"],
            )
        else:
            verify_authorization(auth, prepared["pins"], "connection", reviews)
        print(
            canonical_bytes(
                {
                    "status": "PREFLIGHT_PASS",
                    "phase": prepared["policy"]["phase"],
                    "production_calls": False,
                }
            ).decode(),
            end="",
        )
        return 0
    if not args.output:
        parser.error("--output required with --execute")
    print(execute(prepared, authorization=auth, reviews=reviews, output=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
