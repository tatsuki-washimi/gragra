"""Finite source-model review admission; never grants numerical execution PASS."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from .transport import GateError, canonical_bytes, strict_json, verify_artifact

SCHEMA = "gen1-stochastic-production-source-review-v1"
ADMISSION = "gen1-stochastic-production-source-admission-v1"


def require_source_admission(
    pin,
    *,
    family,
    production_repo,
    production_commit,
    production_tree,
    frozen_artifacts,
    reference_pins,
    plans,
    closure,
    requirements,
):
    packet = strict_json(verify_artifact(pin))
    if (
        set(packet) != {"schema", "family", "decision", "pins", "reviews"}
        or packet["schema"] != ADMISSION
        or packet["family"] != family
        or packet["decision"] != "PASS"
        or set(packet["reviews"]) != {"science", "numerical", "provenance"}
    ):
        raise GateError("production source admission fields/decision mismatch")
    pins = packet["pins"]
    fields = {
        "family",
        "production_commit",
        "production_tree",
        "frozen_artifacts_sha256",
        "reference_seal_sha256",
        "reference_records_sha256",
        "expected_keys_sha256",
        "source_files",
    }
    expected = {
        "family": family,
        "production_commit": production_commit,
        "production_tree": production_tree,
        "frozen_artifacts_sha256": {
            n: frozen_artifacts[n]["sha256"]
            for n in ("freeze", "fixture", "budget", "family_procedure")
        },
        "reference_seal_sha256": reference_pins["seal_sha256"],
        "reference_records_sha256": reference_pins["records_sha256"],
        "expected_keys_sha256": reference_pins["expected_keys_sha256"],
    }
    if (
        set(pins) != fields
        or any(pins[n] != v for n, v in expected.items())
        or not isinstance(pins["source_files"], dict)
        or not pins["source_files"]
    ):
        raise GateError("production source proof actual P/F/R pins mismatch")
    repo = Path(production_repo).resolve(strict=True)
    for relative, digest in pins["source_files"].items():
        path = repo / relative
        if (
            not relative.startswith("src/gragra/")
            or path.is_symlink()
            or not path.resolve(strict=True).is_relative_to(repo / "src/gragra")
        ):
            raise GateError("source proof contains unknown or escaping production path")
        blob = subprocess.run(
            ["git", "-C", str(repo), "show", production_commit + ":" + relative],
            check=True,
            capture_output=True,
        ).stdout
        if (
            hashlib.sha256(blob).hexdigest() != digest
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise GateError("source proof stale actual Git/current production bytes")
    modules = {p["name"]: p for p in closure["modules"]}
    for name in {p["module"] for p in plans.values()} | {"gragra.constants"}:
        module = modules.get(name)
        if module is None or not Path(module["path"]).is_relative_to(repo):
            raise GateError("actual endpoint missing from typed source closure")
        relative = str(Path(module["path"]).relative_to(repo))
        if pins["source_files"].get(relative) != module["sha256"]:
            raise GateError("source proof omits actual endpoint/constants bytes")
    reviewers = set()
    for role, review_pin in packet["reviews"].items():
        review = strict_json(verify_artifact(review_pin))
        if (
            set(review)
            != {
                "schema",
                "family",
                "role",
                "reviewer_id",
                "decision",
                "pins",
                "claim_attestations",
                "original_review",
            }
            or review["schema"] != SCHEMA
            or review["family"] != family
            or review["role"] != role
            or review["decision"] != "PASS"
            or review["pins"] != pins
            or not isinstance(review["reviewer_id"], str)
            or not review["reviewer_id"]
            or review["reviewer_id"] in reviewers
        ):
            raise GateError("production source independent review mismatch")
        reviewers.add(review["reviewer_id"])
        verify_artifact(review["original_review"])
        claims = review["claim_attestations"]
        if set(claims) != (set(requirements) if role == "science" else set()):
            raise GateError("source model claim set differs from frozen obligations")
        for claim in claims.values():
            if (
                set(claim) != {"decision", "scope", "proof_report"}
                or claim["decision"] != "PASS"
                or claim["scope"] != "SOURCE_MODEL_BINDING_ONLY"
            ):
                raise GateError(
                    "source claim cannot imply numerical execution acceptance"
                )
            verify_artifact(claim["proof_report"])
    # Keep the exact reader-derived binding separate from candidate policy pins.
    return {
        "pins": pins,
        "packet": {"admission": pin, "reviews": packet["reviews"]},
        "expected_pins_sha256": hashlib.sha256(canonical_bytes(pins)).hexdigest(),
    }
