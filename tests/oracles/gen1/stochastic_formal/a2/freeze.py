"""Raw-byte semantic checks for the uniform new-family F sidecar."""

import hashlib
import json
import re
from pathlib import Path

from review import reject_noncanonical_number, unique_object

REQUIRED = {
    "schema_version",
    "freeze_id",
    "family",
    "status",
    "reviewed_commit",
    "procedure_file",
    "procedure_sha256",
    "fixture_file",
    "fixture_sha256",
    "budget_file",
    "budget_sha256",
    "artifact_hashes",
    "reviews",
    "root_concurrence",
    "baseline_pins",
    "formal_execution",
}
OPTIONAL = {
    "scope",
    "integrity",
    "frozen_determination_utc",
    "historical_review_pins",
    "immutable_author_proposals",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repo_file(root, path):
    if not isinstance(path, str) or Path(path).is_absolute():
        raise ValueError("relative F path required")
    result = (root / path).resolve()
    if not result.is_relative_to(root) or not result.is_file():
        raise ValueError("F path missing or escapes repository")
    return result


def pinned_json(pin):
    if not re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]):
        raise ValueError("SHA256 pin required")
    path = Path(pin["path"])
    if sha(path) != pin["sha256"]:
        raise ValueError("actual sidecar/review/receipt bytes mismatch")
    value = json.loads(
        path.read_bytes(),
        object_pairs_hook=unique_object,
        parse_float=reject_noncanonical_number,
        parse_constant=reject_noncanonical_number,
    )
    if not isinstance(value, dict):
        raise ValueError("typed review/receipt object required")
    return value


def verify_freeze(root, pin, fixture_path, procedure_hash):
    root = Path(root).resolve()
    freeze = pinned_json(pin)
    if not set(freeze) >= REQUIRED or set(freeze) - REQUIRED - OPTIONAL:
        raise ValueError("uniform family sidecar fields required")
    if (
        type(freeze["schema_version"]) is not int
        or freeze["schema_version"] != 1
        or freeze["family"] != "A2"
        or freeze["freeze_id"] != "gen1-stochastic-F_A2-v1"
        or freeze["status"] != "FROZEN"
        or not freeze["formal_execution"].startswith("HOLD")
    ):
        raise ValueError("A2 reviewed frozen sidecar required")
    if not re.fullmatch(r"[0-9a-f]{40}", freeze["reviewed_commit"]):
        raise ValueError("protocol candidate commit required")
    artifacts = {
        freeze[f"{kind}_file"]: freeze[f"{kind}_sha256"]
        for kind in ("fixture", "budget", "procedure")
    }
    if len(artifacts) != 3 or artifacts != freeze["artifact_hashes"]:
        raise ValueError("exact three-artifact direct/map pin equality required")
    for path, expected in artifacts.items():
        if (
            not re.fullmatch(r"[0-9a-f]{64}", expected)
            or sha(repo_file(root, path)) != expected
        ):
            raise ValueError("actual F artifact mismatch")
    if Path(fixture_path).resolve() != repo_file(root, freeze["fixture_file"]):
        raise ValueError("worker fixtures must be exact frozen tracked artifact")
    if procedure_hash != freeze["procedure_sha256"]:
        raise ValueError("actual procedure pin mismatch")
    baseline = freeze["baseline_pins"]
    if not isinstance(baseline, dict) or not baseline:
        raise ValueError("baseline actual-byte pins required")
    procedure = json.loads(repo_file(root, freeze["procedure_file"]).read_bytes())
    if baseline != procedure.get("baseline_pins"):
        raise ValueError("sidecar/procedure baseline pins mismatch")
    for path, expected in baseline.items():
        if sha(repo_file(root, path)) != expected:
            raise ValueError("frozen baseline mismatch")
    reviews = freeze["reviews"]
    if set(reviews) != {"science", "numerical", "provenance"}:
        raise ValueError("three protocol review pins required")
    reviewer_ids = set()
    for role, review in reviews.items():
        if (
            set(review) != {"path", "sha256", "decision", "scope"}
            or review["decision"] != "PASS"
            or review["scope"] != "protocol freeze"
        ):
            raise ValueError("typed protocol PASS pin required")
        actual = pinned_json(review)
        if (
            actual.get("decision") != "PASS"
            or actual.get("family") != "A2"
            or actual.get("scope") != "protocol freeze"
            or actual.get("role") != role
            or not isinstance(actual.get("reviewer_id"), str)
            or not actual.get("reviewer_id")
            or actual.get("reviewer_id") in reviewer_ids
            or set(actual)
            != {
                "schema_version",
                "role",
                "reviewer_id",
                "decision",
                "scope",
                "family",
                "reviewed_commit",
                "artifact_hashes",
                "original_reviews",
            }
            or type(actual.get("schema_version")) is not int
            or actual["schema_version"] != 1
        ):
            raise ValueError("actual protocol review family/decision mismatch")
        reviewer_ids.add(actual["reviewer_id"])
        if actual.get("reviewed_commit") != freeze["reviewed_commit"]:
            raise ValueError("actual protocol candidate mismatch")
        if actual.get("artifact_hashes") != artifacts:
            raise ValueError("actual review missing exact F artifact pins")
        originals = actual.get("original_reviews")
        if not isinstance(originals, list) or not originals:
            raise ValueError("original independent review bytes required")
        for original in originals:
            if (
                set(original) != {"path", "sha256"}
                or sha(original["path"]) != original["sha256"]
            ):
                raise ValueError("original review actual bytes mismatch")
    expected_decision = "PASS_FOR_F_A2_V1_FREEZE"
    root_pin = freeze["root_concurrence"]
    if (
        set(root_pin) != {"path", "sha256", "decision"}
        or root_pin["decision"] != expected_decision
    ):
        raise ValueError("typed root concurrence required")
    receipt = pinned_json(root_pin)
    if receipt.get("family") != "A2" or receipt.get("decision") != expected_decision:
        raise ValueError("actual root receipt family/decision mismatch")
    if receipt.get("artifact_hashes") != artifacts or receipt.get("review_hashes") != {
        kind: review["sha256"] for kind, review in reviews.items()
    }:
        raise ValueError("actual root receipt artifact/review mismatch")
    if not receipt.get("formal_execution", "").startswith("HOLD"):
        raise ValueError("protocol freeze alone must not authorize execution")
    return freeze
