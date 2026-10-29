"""Canonical pre-E implementation reviews and actual runtime source closure."""

import hashlib
import json
import sys
from pathlib import Path

from budget import BoundedGit

REQUIRED_ADMISSION_INFRASTRUCTURE = {
    "tests/oracles/gen1/stochastic_attempt.py",
    "tests/oracles/gen1/run_stochastic_local.py",
}

PREFIX = "tests/oracles/gen1/stochastic_formal/a2/"
REVIEW_PATHS = {
    "generator": {PREFIX + "oracle.py"},
    "runner": {
        PREFIX + name
        for name in ("runner.py", "worker.py", "freeze.py", "purpose.py", "review.py")
    },
    "sealer_helpers": {
        PREFIX + "budget.py",
        "tests/oracles/gen1/stochastic_attempt.py",
        "tests/oracles/gen1/stochastic_acceptance.py",
        "tests/oracles/gen1/stochastic_f_common_v2.py",
    },
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def reject_noncanonical_number(_value):
    raise ValueError("canonical scientific decimal strings required")


def inventory():
    result = {}
    for name, module in sorted(sys.modules.items()):
        if name == "gragra" or name.startswith("gragra."):
            raise ValueError("production loaded in reference process")
        path = getattr(module, "__file__", None)
        if path and Path(path).is_file():
            result[name] = {"path": str(Path(path).resolve()), "sha256": sha(path)}
    return result


def verify_implementation_reviews(auth, root, e_commit):
    """Bind independent pre-E reviews to actual candidate, E and working bytes."""
    git = BoundedGit(auth["resource_ledger_path"])
    groups = auth["reviewed_source_hashes_by_group"]
    if set(groups) != set(REVIEW_PATHS) or any(
        set(groups[group]) != paths for group, paths in REVIEW_PATHS.items()
    ):
        raise ValueError("exact A2 three-group reviewed source closure required")
    admission = auth["file_hashes_by_group"]
    if set(admission) != {"policy", "fixture", "generator", "source", "infrastructure"}:
        raise ValueError("frozen five-group admission boundary required")
    if any(not isinstance(group, dict) or not group for group in admission.values()):
        raise ValueError("nonempty frozen admission group maps required")
    if not REQUIRED_ADMISSION_INFRASTRUCTURE.issubset(admission["infrastructure"]):
        raise ValueError("frozen required infrastructure files omitted")
    admitted = {}
    for group in admission.values():
        for path, expected in group.items():
            actual_path = (root / path).resolve()
            if (
                Path(path).is_absolute()
                or ".." in Path(path).parts
                or not Path(path).parts
                or not actual_path.is_relative_to(root)
            ):
                raise ValueError("relative admitted source path required")
            if path in admitted:
                raise ValueError("duplicate admitted source path")
            admitted[path] = expected
            blob = git.check_output(
                ["git", "-C", str(root), "show", e_commit + ":" + path]
            )
            if (
                sha(actual_path) != expected
                or hashlib.sha256(blob).hexdigest() != expected
            ):
                raise ValueError(
                    "admitted source differs from committed E/working bytes"
                )
    reviewed = {path: sha for group in groups.values() for path, sha in group.items()}
    if any(admitted.get(path) != sha for path, sha in reviewed.items()):
        raise ValueError("reviewed source omitted from five-group admission closure")
    fields = {
        "schema_version",
        "role",
        "reviewer_id",
        "decision",
        "scope",
        "reviewed_commit",
        "families",
        "file_hashes_by_group",
        "original_reviews",
    }
    if set(auth.get("reviews", {})) != {"science", "numerical", "provenance"}:
        raise ValueError("all independent reviews required")
    reviewer_ids = set()
    for role, pin in auth["reviews"].items():
        if pin.get("decision") != "PASS" or sha(pin["path"]) != pin["sha256"]:
            raise ValueError("implementation review bytes/verdict mismatch")
        actual = json.loads(
            Path(pin["path"]).read_bytes(),
            object_pairs_hook=unique_object,
            parse_float=reject_noncanonical_number,
            parse_constant=reject_noncanonical_number,
        )
        if (
            set(actual) != fields
            or type(actual["schema_version"]) is not int
            or actual["schema_version"] != 1
            or actual["role"] != role
            or actual["decision"] != "PASS"
            or actual["scope"] != "reference implementation"
            or actual["families"] != ["A2"]
            or actual["file_hashes_by_group"] != groups
            or not isinstance(actual["reviewer_id"], str)
            or not actual["reviewer_id"]
            or actual["reviewer_id"] in reviewer_ids
            or not isinstance(actual["reviewed_commit"], str)
            or len(actual["reviewed_commit"]) != 40
            or any(c not in "0123456789abcdef" for c in actual["reviewed_commit"])
        ):
            raise ValueError(
                "implementation review semantic source/role binding mismatch"
            )
        reviewer_ids.add(actual["reviewer_id"])
        candidate = actual["reviewed_commit"]
        ancestry = git.run(
            [
                "git",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                candidate,
                e_commit,
            ],
            capture_output=True,
        )
        if ancestry.returncode:
            raise ValueError("reviewed source candidate is not an ancestor of E")
        for path, expected_sha in reviewed.items():
            blob = git.check_output(
                ["git", "-C", str(root), "show", candidate + ":" + path]
            )
            if hashlib.sha256(blob).hexdigest() != expected_sha:
                raise ValueError(
                    "reviewed source differs from candidate committed blob"
                )
        originals = actual["original_reviews"]
        if not isinstance(originals, list) or not originals:
            raise ValueError("original implementation reports required")
        for original in originals:
            if (
                set(original) != {"path", "sha256"}
                or sha(original["path"]) != original["sha256"]
            ):
                raise ValueError("original implementation report byte mismatch")
