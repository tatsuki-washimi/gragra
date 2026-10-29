"""Frozen-family admission and lazy production comparison; no oracle imports."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from ...common.b1_driver import legacy
from ...common.dependencies import verify_formal_closure
from ...common.family_adapter import call_plan, source_scale_table
from ...common.transport import (
    GateError,
    artifact_pin,
    canonical_bytes,
    exact_keys,
    key_id,
    strict_json,
    verify_artifact,
    verify_authorization,
)

FAMILIES = {"R1", "B2", "A1", "A2", "A3", "T1"}
REFERENCE_REVIEW_SCHEMA = "gen1-stochastic-reference-review-v1"
REFERENCE_ADMISSION_SCHEMA = "gen1-stochastic-reference-admission-v1"
POLICY_SCHEMA = "gen1-stochastic-family-policy-v1"


def frozen_expected_keys(fixtures: dict) -> list[dict]:
    """Preserve frozen CSD and separately declared A2 nominal scalar AUX keys."""
    main = fixtures["expected_keys"]
    scalar = fixtures.get("expected_scalar_keys", [])
    if scalar and fixtures.get("family") != "A2":
        raise GateError("scalar key partition is only frozen for A2")
    if fixtures.get("family") == "A2":
        count = fixtures["coverage_counts"]["mathematical_scalar_auxiliary"]
        if len(scalar) != count or any(
            k["family"] != "A2"
            or k["observable"] != "isotropic_depth_factor"
            or k["representation"] != "Gfree"
            or k["target_id"] != "nominal_x"
            or k["set"] != "auxiliary"
            for k in scalar
        ):
            raise GateError("frozen A2 scalar AUX partition invalid")
    expected = [*main, *scalar]
    exact_keys(expected)
    return expected


def _json(pin):
    return strict_json(verify_artifact(pin))


def derive_reference_pins(artifacts: dict) -> dict:
    """Derive finite reviewer pins from actual frozen/sealed retained bytes.

    This does not issue a review decision or admit scientific reference records.
    """
    required = {
        "freeze",
        "fixture",
        "budget",
        "family_procedure",
        "records",
        "seal",
        "environment",
    }
    if set(artifacts) != required:
        raise GateError("reference pin input fields mismatch")
    documents = {n: _json(p) for n, p in artifacts.items() if n != "records"}
    freeze, fixtures, seal = (documents[n] for n in ("freeze", "fixture", "seal"))
    family = fixtures["family"]
    if (
        family not in FAMILIES
        or freeze.get("family") != family
        or freeze.get("status") != "FROZEN"
    ):
        raise GateError("exact family frozen declaration required")
    for name, field in [
        ("fixture", "fixture_sha256"),
        ("budget", "budget_sha256"),
        ("family_procedure", "procedure_sha256"),
    ]:
        if (
            freeze.get(field) != artifacts[name]["sha256"]
            or documents[name].get("family") != family
        ):
            raise GateError("family frozen triplet byte mismatch")
    if family == "A2":
        from .resource_adapter import derive_native_reference_pins

        return derive_native_reference_pins(artifacts, documents, fixtures, seal)
    raw = verify_artifact(artifacts["records"])
    if not raw or not raw.endswith(b"\n"):
        raise GateError("nonempty terminated reference records required")
    root = Path(artifacts["seal"]["path"]).parent
    if (
        root.stat().st_mode & 0o222
        or Path(artifacts["seal"]["path"]).stat().st_mode & 0o222
    ):
        raise GateError("reference attempt and terminal seal must be read-only")
    if seal.get("format") == "gen1-stochastic-attempt-seal-v2":
        execution, coverage = seal["execution"], seal["execution_coverage"]
        if (
            execution.get("exit_code") != 0
            or execution.get("status") != "success"
            or execution.get("cleanup", {}).get("process_group_empty") is not True
            or execution.get("cleanup", {}).get("leader_reaped") is not True
            or coverage.get("status") != "accepted"
        ):
            raise GateError(
                "reference execution/coverage/cleanup not sealed successful"
            )
        source = seal["provenance"]["source_pin"]
        e_commit, e_tree = source["head"], source["tree"]
        hashes = seal["artifacts_sha256"]
        expected = coverage["expected_keys"]
    elif family == "T1" and seal.get("family") == "T1":
        if (
            seal.get("mode") != "formal"
            or seal.get("outcome") != "COMPLETE"
            or seal.get("exit_code") != 0
            or seal.get("coverage_valid") is not True
            or seal.get("cleanup_confirmed") is not True
        ):
            raise GateError("T1 formal reference seal unresolved")
        source = seal["pins"]
        for name, field in [
            ("fixture", "fixtures_sha256"),
            ("budget", "budget_sha256"),
            ("family_procedure", "procedure_sha256"),
            ("freeze", "freeze_declaration_sha256"),
            ("environment", "expected_environment_sha256"),
        ]:
            if source.get(field) != artifacts[name]["sha256"]:
                raise GateError(
                    "T1 sealed input/environment differs from actual frozen pins"
                )
        e_commit, e_tree = source["E_commit"], source["E_tree"]
        hashes = seal["artifact_hashes"]
        expected = seal["expected_keys"]
    else:
        raise GateError("unsupported sealed reference envelope")
    if exact_keys(expected) != exact_keys(frozen_expected_keys(fixtures)):
        raise GateError("reference seal differs from frozen key partition")
    for name in ("records", "environment"):
        path = Path(artifacts[name]["path"])
        embedded = (
            path.parent == root and hashes.get(path.name) == artifacts[name]["sha256"]
        )
        external_environment = (
            name == "environment"
            and seal.get("format") == "gen1-stochastic-attempt-seal-v2"
            and seal["provenance"].get("environment_inventory")
            == {"path": artifacts[name]["path"], "sha256": artifacts[name]["sha256"]}
        )
        if not embedded and not external_environment:
            raise GateError("reference records/environment not bound by actual seal")
        if path.is_symlink() or path.stat().st_mode & 0o222:
            raise GateError(
                "reference records/environment must be retained read-only bytes"
            )
    for name, digest in hashes.items():
        path = root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise GateError("actual sealed reference artifact byte mismatch")
        if path.stat().st_mode & 0o222:
            raise GateError("sealed reference artifacts remain writable")
    for value in (e_commit, e_tree):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
            raise GateError("reference exact E/tree missing")
    env = documents["environment"]
    if env.get("production_modules") not in (None, []) or any(
        n == "gragra" or n.startswith("gragra.") for n in env["loaded_modules"]
    ):
        raise GateError("independent reference environment loaded production modules")
    return {
        "family": family,
        "frozen_artifacts_sha256": {
            n: artifacts[n]["sha256"]
            for n in ("freeze", "fixture", "budget", "family_procedure")
        },
        "records_sha256": artifacts["records"]["sha256"],
        "seal_sha256": artifacts["seal"]["sha256"],
        "artifact_hashes": hashes,
        "E_commit": e_commit,
        "E_tree": e_tree,
        "source_pin": source,
        "environment_sha256": artifacts["environment"]["sha256"],
        "expected_keys_sha256": hashlib.sha256(
            canonical_bytes(frozen_expected_keys(fixtures))
        ).hexdigest(),
    }


def require_reference_admission(admission: dict, artifacts: dict) -> dict:
    pins = derive_reference_pins(artifacts)
    if (
        set(admission) != {"schema", "family", "decision", "pins", "reviews"}
        or admission["schema"] != REFERENCE_ADMISSION_SCHEMA
        or admission["family"] != pins["family"]
        or admission["decision"] != "PASS"
        or admission["pins"] != pins
        or set(admission["reviews"]) != {"science", "numerical", "provenance"}
    ):
        raise GateError("actual sealed reference admission mismatch")
    ids = set()
    for role, pin in admission["reviews"].items():
        review = _json(pin)
        if (
            set(review)
            != {
                "schema",
                "family",
                "role",
                "reviewer_id",
                "decision",
                "pins",
                "original_review",
            }
            or review["schema"] != REFERENCE_REVIEW_SCHEMA
            or review["family"] != pins["family"]
            or review["role"] != role
            or review["decision"] != "PASS"
            or review["pins"] != pins
            or not isinstance(review["reviewer_id"], str)
            or not review["reviewer_id"]
            or review["reviewer_id"] in ids
        ):
            raise GateError("independent reference review pin/role/decision mismatch")
        verify_artifact(review["original_review"])
        ids.add(review["reviewer_id"])
    return pins


def execution_resources(procedure: dict, family: str) -> dict:
    """Normalize existing frozen fields; introduce no resource allowance."""
    if family in {"R1", "B2"}:
        path, rss_path = ["numerical_procedure", "formal_timeout_seconds"], None
    elif family == "T1":
        path, rss_path = (
            ["resources", "timeout_seconds_per_attempt"],
            ["resources", "rss_bytes_per_process"],
        )
    elif family == "A2":
        path, rss_path = (
            ["resources", "max_attempt_seconds"],
            ["resources", "rss_bytes_per_process"],
        )
    else:
        path, rss_path = ["resources", "timeout_seconds"], ["resources", "rss_bytes"]

    def get(keys):
        item = procedure
        for key in keys:
            item = item[key]
        if type(item) is not int or item <= 0:
            raise GateError("frozen resource must be a positive integer")
        return item

    return {
        "schema": "gen1-stochastic-execution-resources-v1",
        "family": family,
        "resources": {"timeout_seconds": get(path)},
        "timeout_source_path": path,
        "rss_bytes_per_process": get(rss_path) if rss_path else None,
        "rss_source_path": rss_path,
    }


def _records(pin):
    raw = verify_artifact(pin)
    if not raw.endswith(b"\n"):
        raise GateError("unterminated reference/metadata JSONL")
    if raw.lstrip().startswith(b"["):
        result = strict_json(raw)
    else:
        result = [strict_json(line) for line in raw.splitlines()]
    if not isinstance(result, list) or not result:
        raise GateError("nonempty reference/metadata records required")
    return result


def _purpose_records(pin, family):
    if family != "A2":
        return _records(pin)
    from tests.oracles.gen1.stochastic_f_common_v2 import (
        check_calibration_formal_separation,
    )

    bundle = _json(pin)
    if not isinstance(bundle, dict) or set(bundle) != {
        "formal",
        "calibration",
        "decision",
    }:
        raise GateError("native A2 purpose bundle shape differs")
    formal, calibration = bundle["formal"], bundle["calibration"]
    if (
        not isinstance(formal, list)
        or not isinstance(calibration, list)
        or len(formal) != 84
        or len(calibration) != 4
    ):
        raise GateError("native A2 purpose bundle counts differ")
    try:
        exact_keys([row["key"] for row in formal])
        exact_keys([row["key"] for row in calibration])
        decision = check_calibration_formal_separation(calibration, formal)
    except (KeyError, TypeError, ValueError) as error:
        raise GateError("native A2 purpose rows invalid") from error
    if decision.get("admitted") is not True or bundle["decision"] != decision:
        raise GateError("native A2 purpose separation decision differs")
    return formal


def frozen_input_rows(fixtures: dict) -> dict:
    rows = fixtures.get("science", fixtures.get("fixtures"))
    if not isinstance(rows, list) or not rows:
        raise GateError("family frozen fixture rows unavailable")
    result = {}
    for row in rows:
        name = row.get("fixture_id", row.get("key_base", {}).get("fixture_id"))
        if not name or name in result:
            raise GateError("duplicate/missing frozen fixture id")
        result[name] = row
    return result


def _descriptor(spec: dict, closure: dict):
    if set(spec) != {"module", "symbol", "source"}:
        raise GateError("family connection assessor descriptor mismatch")
    if not isinstance(spec["symbol"], str) or not spec["symbol"].isidentifier():
        raise GateError("unknown connection assessor symbol")
    if spec["module"] != __package__ + ".family_connections":
        raise GateError(
            "production connection assessor must be reviewed comparison code"
        )
    verify_artifact(spec["source"])
    planned = {p["name"]: p for p in closure["modules"]}
    pin = planned.get(spec["module"])
    if (
        pin is None
        or pin["path"] != spec["source"]["path"]
        or pin["sha256"] != spec["source"]["sha256"]
    ):
        raise GateError("connection assessor absent from actual typed closure")


def prepare_reference_contracts(artifacts, fixtures, kernel):
    """Actual frozen reference metadata path; no P computation or source approval."""
    family = fixtures["family"]
    records = _records(artifacts["records"])
    record_hashes = {
        key_id(strict_json(line)["key"]): hashlib.sha256(line + b"\n").hexdigest()
        for line in verify_artifact(artifacts["records"]).splitlines()
    }
    coverage = kernel.admit_records(records, frozen_expected_keys(fixtures))
    if not coverage["coverage_valid"]:
        raise GateError("reference envelope/resolved exact coverage mismatch")
    seal = _json(artifacts["seal"])
    hashes = seal.get("artifacts_sha256", seal.get("artifact_hashes"))
    purpose_path = Path(artifacts["purpose"]["path"])
    if (
        purpose_path.parent != Path(artifacts["seal"]["path"]).parent
        or hashes.get(purpose_path.name) != artifacts["purpose"]["sha256"]
    ):
        raise GateError("formal purpose sidecar not bound to actual reference seal")
    metadata = _purpose_records(artifacts["purpose"], family)
    if exact_keys([m["key"] for m in metadata]) != exact_keys(
        frozen_expected_keys(fixtures)
    ):
        raise GateError("formal purpose exact coverage mismatch")
    purposes = {key_id(m["key"]): m for m in metadata}
    frozen = frozen_input_rows(fixtures)
    by_key, plans = {}, {}
    for record in records:
        key = record["key"]
        identity = key_id(key)
        row = frozen[key["fixture_id"]]
        scalar = key["observable"] == "isotropic_depth_factor"
        expected = (
            {
                "nominal_exact_x": row["nominal_exact_x"],
                "frequency_hz": key["frequency_hz"],
            }
            if scalar
            else row["inputs"]
            if "inputs" in row
            else {
                **row,
                "frequency_hz": key["frequency_hz"],
                "g_si": fixtures.get("g_si", row.get("g_si")),
            }
        )
        units = "1" if scalar else fixtures["units"][key["representation"]]
        if (
            record["inputs"] != expected
            or record["procedure_hash"] != artifacts["family_procedure"]["sha256"]
            or record["units"] != units
        ):
            raise GateError("reference input/procedure/units differs from exact F")
        m = purposes[identity]
        if (
            set(m) != {"purpose", "set", "key", "scientific_input"}
            or m["purpose"] != "formal-science"
            or m["set"] != key["set"]
        ):
            raise GateError("reference acceptance purpose invalid")
        scientific = m["scientific_input"]
        expected_scientific_inputs = dict(record["inputs"])
        if family in {"R1", "B2"}:
            metadata_fields = {
                "fixture_id",
                "frame",
                "psd_convention",
                "input_assumptions",
                "set",
                "target_id",
                "eta_P",
            }
            expected_scientific_inputs = {
                n: v
                for n, v in expected_scientific_inputs.items()
                if n not in metadata_fields
            }
        elif family in {"A1", "A3"}:
            expected_scientific_inputs["pressure_convention"] = row[
                "pressure_convention"
            ]
        frame, convention = (
            ("dimensionless scalar x", "dimensionless scalar")
            if scalar
            else (row["frame"], row["psd_convention"])
        )
        if (
            set(scientific)
            != {
                "frequency_hz",
                "frame",
                "psd_convention",
                "source_pins",
                "procedure_hash",
                "inputs",
            }
            or scientific["frequency_hz"] != key["frequency_hz"]
            or scientific["procedure_hash"] != record["procedure_hash"]
            or scientific["source_pins"] != record["source_pins"]
            or scientific["frame"] != frame
            or scientific["psd_convention"] != convention
            or scientific["inputs"] != expected_scientific_inputs
        ):
            raise GateError("reference scientific input preimage differs from F/record")
        inputs = {**record["inputs"]}
        if family == "A2" and "nominal_exact_x" in row:
            inputs["nominal_exact_x"] = row["nominal_exact_x"]
        plans[identity] = call_plan(key, inputs)
        by_key[identity] = record
    return by_key, plans, record_hashes, hashes


def preflight(policy_pin: dict) -> dict:
    """No production imports/calls; all P pins derive from exact reader state."""
    policy = _json(policy_pin)
    fields = {
        "schema",
        "family",
        "pins",
        "production_repo",
        "comparator_repo",
        "integration_repo",
        "artifacts",
        "connection_assessor",
        "connection_requirements",
    }
    if (
        set(policy) != fields
        or policy["schema"] != POLICY_SCHEMA
        or policy["family"] not in FAMILIES
    ):
        raise GateError("family policy schema mismatch")
    family, artifacts, pins = policy["family"], policy["artifacts"], policy["pins"]
    required = {
        "freeze",
        "fixture",
        "budget",
        "family_procedure",
        "records",
        "seal",
        "environment",
        "reference_admission",
        "production_source_admission",
        "purpose",
        "source_scales",
        "dependency_closure",
        "procedure",
        "decision_kernel",
    }
    if set(artifacts) != required or pins.get("family") != family:
        raise GateError("family policy artifact set/pins mismatch")
    for name, pin in artifacts.items():
        verify_artifact(pin)
        if pins.get(name + "_sha256") != pin["sha256"]:
            raise GateError("family authority artifact mismatch: " + name)
    a2_resources = None
    if family == "A2":
        from . import resource_adapter as a2_resources

    for prefix in ("production", "comparator"):
        (
            a2_resources.verify_clean_git_pin
            if a2_resources
            else legacy.verify_clean_git_pin
        )(
            Path(policy[prefix + "_repo"]),
            pins[prefix + "_commit"],
            pins[prefix + "_tree"],
        )
    reference_artifacts = {
        n: artifacts[n]
        for n in (
            "freeze",
            "fixture",
            "budget",
            "family_procedure",
            "records",
            "seal",
            "environment",
        )
    }
    r_pins = require_reference_admission(
        _json(artifacts["reference_admission"]), reference_artifacts
    )
    if a2_resources:
        a2_resources.require_ancestor(policy["integration_repo"], r_pins["E_commit"])
    else:
        ancestor = subprocess.run(
            [
                "git",
                "-C",
                policy["integration_repo"],
                "merge-base",
                "--is-ancestor",
                r_pins["E_commit"],
                "HEAD",
            ],
            capture_output=True,
        )
        if ancestor.returncode:
            raise GateError(
                "sealed reference E is not durable in actual integration lineage"
            )
    if artifacts["decision_kernel"]["sha256"] != legacy.KERNEL_SHA256:
        raise GateError("admitted comparison arithmetic bytes differ")
    kernel = (
        a2_resources.load_pinned_kernel(
            policy["production_repo"], commit=pins["production_commit"], legacy=legacy
        )
        if a2_resources
        else legacy.load_pinned_kernel(
            policy["production_repo"], commit=pins["production_commit"]
        )
    )
    fixtures, budget, procedure = (
        _json(artifacts[n]) for n in ("fixture", "budget", "family_procedure")
    )
    requirements = procedure.get(
        "required_connections", procedure.get("connection_checks")
    )
    if requirements is None:
        requirements = (
            ["domain", "whole_path", "normalization", "scaling_contract"]
            if family == "T1"
            else ["numerical_procedure", "actual_x_transport", "SI", "coverage"]
        )
        if any(n not in procedure for n in requirements):
            raise GateError(
                "family connection contract mapping lacks frozen source fields"
            )
    if (
        not requirements
        or len(requirements) != len(set(requirements))
        or policy["connection_requirements"] != requirements
    ):
        raise GateError("family connection gates differ from frozen obligations")
    resources = execution_resources(procedure, family)
    resources["family_procedure"] = artifacts["family_procedure"]
    if _json(artifacts["procedure"]) != resources:
        raise GateError("execution-resource receipt differs from frozen fields")
    closure = _json(artifacts["dependency_closure"])
    (a2_resources.verify_formal_closure if a2_resources else verify_formal_closure)(
        closure
    )
    if (
        closure["production_commit"] != pins["production_commit"]
        or closure["production_tree"] != pins["production_tree"]
    ):
        raise GateError("family closure fails actual P binding")
    _descriptor(policy["connection_assessor"], closure)
    # New generic source modules must themselves occur in the finite runtime set.
    planned = {p["name"]: p for p in closure["modules"]}
    for name, path in [
        (__name__, __file__),
        (
            "tests.oracles.gen1.stochastic_formal.common.family_adapter",
            str(Path(__file__).parents[2] / "common/family_adapter.py"),
        ),
        (
            __package__ + ".family_execution",
            str(Path(__file__).with_name("family_execution.py")),
        ),
    ]:
        pin = planned.get(name)
        if (
            pin is None
            or Path(pin["path"]).resolve() != Path(path).resolve()
            or pin["sha256"] != artifact_pin(path, "comparison_source")["sha256"]
        ):
            raise GateError("actual family adapter/driver absent from closure")
    by_key, plans, record_hashes, hashes = prepare_reference_contracts(
        artifacts, fixtures, kernel
    )
    scale_table = _json(artifacts["source_scales"])
    if (
        scale_table["pins"]["fixture"] != artifacts["fixture"]
        or scale_table["pins"]["budget"] != artifacts["budget"]
        or scale_table["pins"]["family_procedure"] != artifacts["family_procedure"]
    ):
        raise GateError("source-scale exact F input pins differ")
    verify_artifact(scale_table["pins"]["builder_source"])
    if scale_table["rounding"] != {
        "arithmetic_significant_digits": [220, 260],
        "serialization_significant_digits": 180,
        "direction": "downward",
        "reference_or_production_outputs_used": False,
    }:
        raise GateError("source-scale conservative rounding contract differs")
    scales = source_scale_table(scale_table, frozen_expected_keys(fixtures), kernel)
    for row in scale_table["rows"]:
        if (
            row["input_sha256"]
            != hashlib.sha256(
                canonical_bytes(by_key[key_id(row["key"])]["inputs"])
            ).hexdigest()
        ):
            raise GateError("source-scale complete frozen input hash differs")
    from .family_source_review import require_source_admission
    from .reader_contract import resolve_source_binding_requirements

    source_binding_requirements = resolve_source_binding_requirements(
        family, requirements
    )

    source_review = require_source_admission(
        artifacts["production_source_admission"],
        family=family,
        production_repo=policy["production_repo"],
        production_commit=pins["production_commit"],
        production_tree=pins["production_tree"],
        frozen_artifacts=artifacts,
        reference_pins=r_pins,
        plans=plans,
        closure=closure,
        requirements=source_binding_requirements,
    )
    reference_artifacts = {
        name: artifact_pin(
            Path(artifacts["seal"]["path"]).parent / name, "sealed_reference_metadata"
        )
        for name in hashes
    }
    science_count = sum(k["set"] == "science" for k in frozen_expected_keys(fixtures))
    declared_science = fixtures.get(
        "expected_science_count",
        fixtures.get("coverage_counts", {}).get("physical_science"),
    )
    declared_aux = fixtures.get(
        "expected_auxiliary_count",
        fixtures.get("coverage_counts", {}).get("mathematical_scalar_auxiliary", 0)
        + fixtures.get("coverage_counts", {}).get("physical_surface_auxiliary", 0),
    )
    if (
        science_count != declared_science
        or sum(k["set"] == "auxiliary" for k in frozen_expected_keys(fixtures))
        != declared_aux
    ):
        raise GateError("frozen science/AUX partition count mismatch")
    return {
        "policy": policy,
        "pins": {**pins, "policy_sha256": policy_pin["sha256"]},
        "reference_pins": r_pins,
        "fixtures": fixtures,
        "budget": budget,
        "procedure": procedure,
        "resources": resources,
        "source_review": source_review,
        "reference_artifacts": reference_artifacts,
        "records": by_key,
        "record_hashes": record_hashes,
        "plans": plans,
        "scales": scales,
        "kernel": kernel,
        "closure": closure,
    }


def execute(prepared: dict, *, authorization: dict, reviews: dict, output: str | Path):
    """Construct production functions only inside the admitted guarded worker."""
    from ...common.family_adapter import production_evaluator
    from .family_execution import run_family_phase
    from .family_supervisor import verify_family_supervision

    resource_launch = verify_family_supervision(prepared)

    def factory():
        verify_authorization(authorization, prepared["pins"], "comparison", reviews)
        return production_evaluator(prepared)

    return run_family_phase(
        authorization=authorization,
        pins=prepared["pins"],
        reviews=reviews,
        phase="comparison",
        family=prepared["policy"]["family"],
        expected_keys=frozen_expected_keys(prepared["fixtures"]),
        output=output,
        evaluator_factory=factory,
        dependency_closure=prepared["closure"],
        resource_launch=resource_launch,
        prepared=prepared,
    )


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    for role in ("science", "numerical", "provenance"):
        parser.add_argument("--" + role + "-review-sha256", required=True)
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
    authority = {
        "path": str(Path(args.authorization).absolute()),
        "sha256": args.authorization_sha256,
        "kind": "authorization",
    }
    reviews = {
        role: getattr(args, role + "_review_sha256")
        for role in ("science", "numerical", "provenance")
    }
    if args.execute:
        if args.output is None:
            raise GateError("fresh external output required")
        execute(prepared, authorization=authority, reviews=reviews, output=args.output)
    else:
        verify_authorization(authority, prepared["pins"], "comparison", reviews)
        print(
            canonical_bytes(
                {
                    "status": "PASS_NO_COMPUTATION",
                    "family": prepared["policy"]["family"],
                    "pins": prepared["pins"],
                    "production_api_calls": 0,
                }
            ).decode()
        )


if __name__ == "__main__":
    main()
