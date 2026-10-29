"""Verify docs/design/gen1/pin-reconciliation-v1.json against this repository.

Usage: python scripts/check_pin_reconciliation.py [--record PATH] [--state-root DIR]

Checks, all read-only:
  1. The record is valid JSON, has no absolute home-directory path, and its
     summary counts equal its entries.
  2. Every resolved_by_git_history entry: the commit exists in this clone and
     the blob at commit:path hashes to the pinned sha256.
  3. Every unresolved entry: no version of the path in any ref of this clone
     hashes to the pin.
  4. Every resolved_by_local_run_snapshot entry: checked only when --state-root
     is given (those bytes are outside git); otherwise reported as not checked.
  5. Append-only: no manifest named by the record differs between the base
     commit and HEAD.

Exit status is 0 when every check passes and 1 otherwise.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RECORD = "docs/design/gen1/pin-reconciliation-v1.json"


def git(*args):
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, check=False
    )


def blob_digest(commit, path):
    result = git("rev-parse", "--verify", "--quiet", f"{commit}:{path}")
    if result.returncode != 0:
        return None
    blob = result.stdout.decode().strip()
    if not blob:
        return None
    content = git("cat-file", "blob", blob)
    if content.returncode != 0:
        return None
    return hashlib.sha256(content.stdout).hexdigest()


def digests_in_history(path, failures):
    seen = set()
    history = git("log", "--all", "--format=%H", "--", path)
    if history.returncode != 0:
        failures.append(
            f"history scan failed for {path}: git log exited {history.returncode}"
        )
        return None
    for commit in history.stdout.decode().split():
        digest = blob_digest(commit, path)
        if digest is None:
            failures.append(f"historical blob unavailable for {path} at {commit[:12]}")
            continue
        seen.add(digest)
    return seen


def check_summary(record, failures):
    summary = record["summary"]["repository_path_status"]
    expected = {
        "resolved_git_history": len(record["resolved_by_git_history"]),
        "resolved_local_run_snapshot": len(record["resolved_by_local_run_snapshot"]),
        "unresolved": len(record["unresolved"]),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            failures.append(f"summary {key}={summary.get(key)} but entries={value}")


def check_history(record, failures):
    for entry in record["resolved_by_git_history"]:
        commit, path = entry["matching_commit"], entry["path"]
        if git("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
            failures.append(f"history commit missing from clone: {commit[:12]}")
            continue
        digest = blob_digest(commit, path)
        if digest is None:
            failures.append(f"historical blob unavailable for {path} at {commit[:12]}")
        elif digest != entry["pinned_sha256"]:
            failures.append(f"history mismatch {path} at {commit[:12]}")


def check_unresolved(record, failures):
    for entry in record["unresolved"]:
        if entry["status"] != "UNRESOLVED" or not entry.get("dependency_note"):
            failures.append(f"unresolved entry malformed: {entry['pointer']}")
        history_digests = digests_in_history(entry["path"], failures)
        if history_digests is not None and entry["pinned_sha256"] in history_digests:
            failures.append(f"unresolved pin found in history: {entry['path']}")


def check_snapshots(record, state_root, failures):
    """Return how many snapshot entries were not checked (no state root given)."""
    if state_root is None:
        return len(record["resolved_by_local_run_snapshot"])
    for entry in record["resolved_by_local_run_snapshot"]:
        local = Path(state_root) / entry["retained_relative_path"]
        if not local.is_file():
            failures.append(f"retained copy missing: {entry['retained_relative_path']}")
        elif hashlib.sha256(local.read_bytes()).hexdigest() != entry["pinned_sha256"]:
            failures.append(
                f"retained copy hash differs: {entry['retained_relative_path']}"
            )
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", default=DEFAULT_RECORD)
    parser.add_argument("--state-root", default=None)
    args = parser.parse_args()

    failures = []
    text = (REPO / args.record).read_text(encoding="utf-8")
    if "/home/" in text:
        failures.append("record contains an absolute home-directory path")
    record = json.loads(text)

    check_summary(record, failures)
    check_history(record, failures)
    check_unresolved(record, failures)
    not_checked = check_snapshots(record, args.state_root, failures)

    manifests = sorted(
        {
            entry["manifest"]
            for entry in record["resolved_by_git_history"]
            + record["resolved_by_local_run_snapshot"]
            + record["unresolved"]
        }
    )
    changed = git(
        "diff", "--name-only", record["base_commit"], "HEAD", "--", *manifests
    )
    if changed.returncode != 0:
        detail = changed.stderr.decode(errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        failures.append(
            f"manifest diff command failed (exit {changed.returncode}){suffix}"
        )
        changed_names = []
    else:
        changed_names = changed.stdout.decode().split()
    if changed.returncode == 0 and changed_names:
        failures.append(f"manifests changed since base: {changed_names}")

    print(
        f"history_checked={len(record['resolved_by_git_history'])} "
        f"unresolved_checked={len(record['unresolved'])} "
        f"snapshot_entries={len(record['resolved_by_local_run_snapshot'])} "
        f"snapshot_not_checked={not_checked}"
    )
    print(f"manifests_compared={len(manifests)} changed={len(changed_names)}")
    for failure in failures:
        print("FAIL", failure)
    if not failures:
        print("PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
