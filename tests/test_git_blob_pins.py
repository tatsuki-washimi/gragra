"""Regression tests for current and revision-bound Git blob pins."""

import subprocess
from pathlib import Path

import pytest

from scripts import check_git_blob_pins


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture
def git_history(tmp_path: Path) -> dict[str, str | Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Pin Test")
    _git(repo, "config", "user.email", "pin-test@example.invalid")
    source = repo / "src" / "module.py"
    source.parent.mkdir()
    source.write_text("VALUE = 'old'\n", encoding="utf-8")
    _git(repo, "add", "src/module.py")
    _git(repo, "commit", "-qm", "old source")
    old_commit = _git(repo, "rev-parse", "HEAD")
    old_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    old_blob = _git(repo, "rev-parse", "HEAD:src/module.py")

    source.write_text("VALUE = 'new'\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "new source")
    head = _git(repo, "rev-parse", "HEAD")
    current_blob = _git(repo, "rev-parse", "HEAD:src/module.py")
    return {
        "repo": repo,
        "old_commit": old_commit,
        "old_tree": old_tree,
        "old_blob": old_blob,
        "head": head,
        "current_blob": current_blob,
    }


def test_current_head_pin_is_checked_at_requested_head(git_history):
    result = check_git_blob_pins.verify_binding(
        git_history["repo"],
        "src/module.py",
        git_history["current_blob"],
        scope="head",
        current_rev=git_history["head"],
    )

    assert result == "HEAD_EQUAL"


@pytest.mark.parametrize(
    ("manifest", "pointer", "key"),
    [
        (
            "docs/design/gen1/external-pin-inventory-v1.json",
            "/targets[0]/git_blob_sha1_at_head",
            "git_blob_sha1_at_head",
        ),
        (
            "docs/design/gen1/ruff-exclusion-identity-v1.json",
            "/files[0]/git_blob_sha1",
            "git_blob_sha1",
        ),
    ],
)
def test_head_pin_field_is_classified_as_current(manifest, pointer, key):
    scope, revisions = check_git_blob_pins.classify_pin(
        manifest,
        pointer,
        key,
        ({},),
    )

    assert scope == "head"
    assert revisions == ()


def test_ruff_historical_baseline_blob_uses_a9_revision(git_history):
    record = {
        "identity_baseline": {
            "commit": git_history["old_commit"],
            "tree": git_history["old_tree"],
        },
        "files": [
            {
                "path": "src/module.py",
                "git_blob_sha1": git_history["current_blob"],
                "historical_identity_at_baseline": {
                    "git_blob_sha1": git_history["old_blob"]
                },
            }
        ],
    }
    pins = []
    check_git_blob_pins.walk(record, "", pins, check_git_blob_pins.RUFF_IDENTITY)

    historical_pin = next(
        pin for pin in pins if "historical_identity_at_baseline" in pin.pointer
    )
    assert historical_pin.path == "src/module.py"
    assert historical_pin.scope == "historical"
    assert historical_pin.revisions == (
        ("commit", git_history["old_commit"]),
        ("tree", git_history["old_tree"]),
    )
    assert (
        check_git_blob_pins.verify_binding(
            git_history["repo"],
            historical_pin.path,
            historical_pin.blob,
            scope=historical_pin.scope,
            revision=historical_pin.revisions,
            current_rev=git_history["head"],
        )
        == "HISTORICAL_EQUAL"
    )


def test_ruff_historical_baseline_without_revision_is_unverifiable():
    record = {
        "files": [
            {
                "path": "src/module.py",
                "historical_identity_at_baseline": {"git_blob_sha1": "0" * 40},
            }
        ]
    }
    pins = []
    check_git_blob_pins.walk(record, "", pins, check_git_blob_pins.RUFF_IDENTITY)

    historical_pin = next(
        pin for pin in pins if "historical_identity_at_baseline" in pin.pointer
    )
    assert historical_pin.path == "src/module.py"
    assert historical_pin.scope == "unverifiable"
    assert historical_pin.revisions == ()


@pytest.mark.parametrize("revision_key", ["old_commit", "old_tree"])
def test_historical_pin_is_checked_at_recorded_revision(git_history, revision_key):
    result = check_git_blob_pins.verify_binding(
        git_history["repo"],
        "src/module.py",
        git_history["old_blob"],
        scope="historical",
        revision=git_history[revision_key],
        current_rev=git_history["head"],
    )

    assert result == "HISTORICAL_EQUAL"


def test_tested_revision_pin_uses_ancestor_commit_and_tree(git_history):
    context = {
        "tested_revision": {
            "commit": git_history["old_commit"],
            "tree": git_history["old_tree"],
        }
    }

    scope, revisions = check_git_blob_pins.classify_pin(
        "docs/design/gen1/subsequent-model-validation-v1.json",
        "/validations[0]/production_source[0]/git_blob_at_tested_revision",
        "git_blob_at_tested_revision",
        (context,),
    )

    assert scope == "historical"
    assert revisions == (
        ("commit", git_history["old_commit"]),
        ("tree", git_history["old_tree"]),
    )


def test_local_observation_pin_uses_recorded_tracking_commit(git_history):
    context = {
        "locally_observed_tracking_ref_commit": git_history["old_commit"],
        "locally_observed_tree": git_history["old_tree"],
    }

    scope, revisions = check_git_blob_pins.classify_pin(
        "docs/design/gen1/cylinder-q2-validation-v1.json",
        "/source_identity/pr159_context_at_recording/"
        "locally_observed_production_blobs/src/module.py",
        "src/module.py",
        (context,),
    )

    assert scope == "historical"
    assert revisions == (
        ("commit", git_history["old_commit"]),
        ("tree", git_history["old_tree"]),
    )


def test_modified_historical_blob_fails_against_recorded_revision(git_history):
    result = check_git_blob_pins.verify_binding(
        git_history["repo"],
        "src/module.py",
        git_history["current_blob"],
        scope="historical",
        revision=git_history["old_commit"],
        current_rev=git_history["head"],
    )

    assert result == "FAIL"


def test_historical_pin_rejects_tree_that_disagrees_with_recorded_commit(git_history):
    result = check_git_blob_pins.verify_binding(
        git_history["repo"],
        "src/module.py",
        git_history["old_blob"],
        scope="historical",
        revision=(
            ("commit", git_history["old_commit"]),
            ("tree", _git(git_history["repo"], "rev-parse", "HEAD^{tree}")),
        ),
        current_rev=git_history["head"],
    )

    assert result == "FAIL"


def test_historical_pin_without_revision_is_unverifiable(git_history):
    result = check_git_blob_pins.verify_binding(
        git_history["repo"],
        "src/module.py",
        git_history["old_blob"],
        scope="historical",
        revision=None,
        current_rev=git_history["head"],
    )

    assert result == "UNVERIFIABLE"


def test_missing_recorded_revision_is_not_replaced_with_head():
    scope, revisions = check_git_blob_pins.classify_pin(
        "docs/design/gen1/subsequent-model-validation-v1.json",
        "/validations[1]/production_source[0]/git_blob_at_tested_revision",
        "git_blob_at_tested_revision",
        ({},),
    )

    assert scope == "unverifiable"
    assert revisions == ()


@pytest.mark.parametrize("scope", ["head", "historical"])
def test_wrong_path_fails_even_when_blob_exists(git_history, scope):
    result = check_git_blob_pins.verify_binding(
        git_history["repo"],
        "src/missing.py",
        git_history["current_blob"],
        scope=scope,
        revision=git_history["old_commit"] if scope == "historical" else None,
        current_rev=git_history["head"],
    )

    assert result == "FAIL"
