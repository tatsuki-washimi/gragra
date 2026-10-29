"""Check recorded Git-blob pins against their declared revisions (read-only).

Usage: python scripts/check_git_blob_pins.py [--rev REV]

Explicit HEAD pins are compared with REV. Historical pins are compared with
their recorded commit or tree. A path-bound pin without a recorded revision is
reported as UNVERIFIABLE; it is never compared with the current HEAD by
default. Pathless pins are checked for object presence only.

Exit status is 0 when every binding is verified, 1 for a missing or differing
object/path, and 2 when any path-bound pin cannot be verified.
"""

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parent.parent
HEX40 = re.compile(r"^[0-9a-f]{40}$")
SELF = "docs/design/gen1/pin-reconciliation-v1.json"
RUFF_IDENTITY = "docs/design/gen1/ruff-exclusion-identity-v1.json"


@dataclass(frozen=True)
class Pin:
    manifest: str
    pointer: str
    path: str | None
    blob: str
    scope: str
    revisions: tuple[tuple[str, str], ...] = ()


def git_at(repo, *args, stdin=None):
    return subprocess.run(
        ["git", "-C", str(repo), *args], input=stdin, capture_output=True, check=False
    )


def git(*args, stdin=None):
    return git_at(REPO, *args, stdin=stdin)


def is_repo_path(value):
    if not isinstance(value, str) or "/" not in value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts)
        and not any(character in value for character in "*?[]")
        and not value.endswith("/")
    )


def _revision_candidates(value):
    if isinstance(value, str) and HEX40.fullmatch(value):
        return (("commit", value),)
    if not isinstance(value, dict):
        return ()
    candidates = []
    for kind in ("commit", "tree"):
        revision = value.get(kind)
        if isinstance(revision, str) and HEX40.fullmatch(revision):
            candidates.append((kind, revision))
    return tuple(candidates)


def _nearest_revision(ancestors, names):
    for node in reversed(ancestors):
        for name in names:
            if name in node:
                candidates = _revision_candidates(node[name])
                if candidates:
                    return candidates
    return ()


def _local_observation_revision(ancestors):
    for node in reversed(ancestors):
        commit = node.get("locally_observed_tracking_ref_commit")
        tree = node.get("locally_observed_tree")
        candidates = []
        if isinstance(commit, str) and HEX40.fullmatch(commit):
            candidates.append(("commit", commit))
        if isinstance(tree, str) and HEX40.fullmatch(tree):
            candidates.append(("tree", tree))
        if candidates:
            return tuple(candidates)
    return ()


def classify_pin(manifest, pointer, key, ancestors):
    """Return a pin scope and recorded revisions without guessing HEAD."""
    if "historical_identity_at_baseline" in pointer:
        revisions = _nearest_revision(ancestors, ("identity_baseline",))
        return ("historical", revisions) if revisions else ("unverifiable", ())

    if key.endswith("_at_head") or (
        manifest == RUFF_IDENTITY
        and key == "git_blob_sha1"
        and "/files[" in pointer
        and "historical_identity_at_baseline" not in pointer
    ):
        return "head", ()

    if "locally_observed_production_blobs" in pointer:
        revisions = _local_observation_revision(ancestors)
        return ("historical", revisions) if revisions else ("unverifiable", ())

    if "at_tested_revision" in key:
        revisions = _nearest_revision(ancestors, ("tested_revision",))
        return ("historical", revisions) if revisions else ("unverifiable", ())

    if "at_reviewed_candidate" in key:
        revisions = _nearest_revision(
            ancestors,
            ("reviewed_candidate_revision", "candidate_revision", "reviewed_candidate"),
        )
        return ("historical", revisions) if revisions else ("unverifiable", ())

    if "/tested_candidate/production_blobs/" in pointer:
        revisions = _nearest_revision(ancestors, ("tested_candidate",))
        return ("historical", revisions) if revisions else ("unverifiable", ())

    revisions = _nearest_revision(
        ancestors,
        ("tested_revision", "reviewed_candidate_revision", "candidate_revision"),
    )
    return ("historical", revisions) if revisions else ("unverifiable", ())


def walk(node, pointer, pins, manifest, ancestors=()):
    if isinstance(node, dict):
        context = ancestors + (node,)
        sibling = None
        for context_node in reversed(context):
            for key in ("path", "file", "original_path"):
                if is_repo_path(context_node.get(key)):
                    sibling = context_node[key]
                    break
            if sibling is not None:
                break
        for key, value in node.items():
            if not isinstance(value, str) or not HEX40.fullmatch(value):
                continue
            if isinstance(key, str) and "blob" in key.lower():
                scope, revisions = classify_pin(
                    manifest, f"{pointer}/{key}", key, context
                )
                pins.append(
                    Pin(manifest, f"{pointer}/{key}", sibling, value, scope, revisions)
                )
            elif is_repo_path(key):
                scope, revisions = classify_pin(
                    manifest, f"{pointer}/{key}", key, context
                )
                pins.append(
                    Pin(manifest, f"{pointer}/{key}", key, value, scope, revisions)
                )
        for key, value in node.items():
            walk(value, f"{pointer}/{key}", pins, manifest, context)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            walk(item, f"{pointer}[{index}]", pins, manifest, ancestors)


def verify_binding(
    repo,
    path,
    blob,
    *,
    scope,
    revision=None,
    current_rev="HEAD",
):
    """Return HEAD_EQUAL, HISTORICAL_EQUAL, FAIL, or UNVERIFIABLE."""
    if not is_repo_path(path) or not HEX40.fullmatch(blob):
        return "FAIL"
    if scope == "head":
        binding_revision = current_rev
        success = "HEAD_EQUAL"
    elif scope == "historical":
        if revision is None:
            return "UNVERIFIABLE"
        candidates = (
            (("auto", revision),) if isinstance(revision, str) else tuple(revision)
        )
        binding_revision = None
        commit_revision = None
        tree_revision = None
        recorded_tree = None
        for candidate in candidates:
            if not isinstance(candidate, tuple) or len(candidate) != 2:
                continue
            kind, value = candidate
            if kind not in {"auto", "commit", "tree"} or not HEX40.fullmatch(value):
                continue
            if kind == "tree":
                recorded_tree = value
            object_type = git_at(repo, "cat-file", "-t", value)
            found_type = object_type.stdout.decode().strip()
            if (
                object_type.returncode == 0
                and found_type in {"commit", "tree"}
                and (kind == "auto" or found_type == kind)
            ):
                if found_type == "commit":
                    commit_revision = value
                else:
                    tree_revision = value
        if commit_revision is not None:
            if recorded_tree is not None:
                actual_tree = git_at(
                    repo,
                    "rev-parse",
                    "--verify",
                    "--quiet",
                    f"{commit_revision}^{{tree}}",
                )
                if (
                    actual_tree.returncode != 0
                    or actual_tree.stdout.decode().strip() != recorded_tree
                ):
                    return "FAIL"
            binding_revision = commit_revision
        elif tree_revision is not None:
            binding_revision = tree_revision
        if binding_revision is None:
            return "UNVERIFIABLE"
        success = "HISTORICAL_EQUAL"
    else:
        return "UNVERIFIABLE"

    actual = git_at(
        repo, "rev-parse", "--verify", "--quiet", f"{binding_revision}:{path}"
    )
    if actual.returncode != 0:
        return "FAIL"
    return success if actual.stdout.decode().strip() == blob else "FAIL"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rev", default="HEAD")
    args = parser.parse_args()

    pins = []
    listing_result = git("ls-tree", "-r", "--name-only", args.rev)
    if listing_result.returncode != 0:
        print(f"FAIL could not read tree {args.rev}")
        return 1
    tree_names = listing_result.stdout.decode().splitlines()
    for name in tree_names:
        if not name.endswith(".json") or name == SELF:
            continue
        raw = git("show", f"{args.rev}:{name}").stdout
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        walk(data, "", pins, name)

    listing = "\n".join(pin.blob for pin in pins) + "\n"
    batch = git("cat-file", "--batch-check", stdin=listing.encode()).stdout.decode()
    present = [not line.endswith("missing") for line in batch.splitlines()]
    counts = Counter()
    failures = []
    unverifiable = []
    for pin, object_present in zip(pins, present, strict=True):
        counts["present" if object_present else "missing"] += 1
        if not object_present:
            failures.append(f"object missing: {pin.manifest}{pin.pointer}")
            continue
        if pin.path is None:
            counts["presence_only"] += 1
            continue
        if pin.scope == "unverifiable":
            counts["unverifiable"] += 1
            unverifiable.append(
                f"no recorded commit/tree: {pin.manifest}{pin.pointer} ({pin.path})"
            )
            continue
        status = verify_binding(
            REPO,
            pin.path,
            pin.blob,
            scope=pin.scope,
            revision=pin.revisions,
            current_rev=args.rev,
        )
        if status == "HEAD_EQUAL":
            counts["head_equal"] += 1
        elif status == "HISTORICAL_EQUAL":
            counts["historical_equal"] += 1
        elif status == "UNVERIFIABLE":
            counts["unverifiable"] += 1
            unverifiable.append(
                f"recorded revision unavailable: {pin.manifest}{pin.pointer} "
                f"({pin.path})"
            )
        else:
            counts["bound_differs"] += 1
            failures.append(
                f"bound pin differs at {pin.scope}: {pin.manifest}{pin.pointer} "
                f"({pin.path})"
            )

    print(
        f"pins={len(pins)} present={counts['present']} missing={counts['missing']} "
        f"presence_only={counts['presence_only']} head_equal={counts['head_equal']} "
        f"historical_equal={counts['historical_equal']} "
        f"unverifiable={counts['unverifiable']} bound_differs={counts['bound_differs']}"
    )
    for failure in failures:
        print("FAIL", failure)
    for item in unverifiable:
        print("UNVERIFIABLE", item)
    if failures:
        return 1
    if unverifiable:
        return 2
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
