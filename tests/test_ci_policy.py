"""Tests for public CI change classification and workflow policy."""

import re
from pathlib import Path

from scripts.ci.classify_changes import classify_paths

ROOT = Path(__file__).resolve().parents[1]


def test_docs_only_requires_non_empty_added_or_modified_documentation() -> None:
    docs_only, reason = classify_paths(
        "A\tdocs_site/docs/index.md\nM\tREADME.md\nM\tdocs_site/docs/guide.mdx\n"
    )

    assert docs_only is True
    assert "docs-only" in reason


def test_source_test_build_dependency_workflow_script_and_unknown_are_full() -> None:
    for record in (
        "M\tsrc/gragra/core.py\n",
        "A\ttests/test_core.py\n",
        "M\tpyproject.toml\n",
        "M\t.github/workflows/ci.yml\n",
        "M\tscripts/ci/check.py\n",
        "M\tsrc/guide.md\n",
        "M\tguide.mdx\n",
        "M\tunknown/file.txt\n",
    ):
        docs_only, reason = classify_paths(record)
        assert docs_only is False
        assert "full" in reason


def test_empty_input_is_full() -> None:
    assert classify_paths("") == (False, "full: empty diff")


def test_malformed_records_are_full() -> None:
    docs_only, reason = classify_paths("M\nA\t\nthis is not a status record\n")

    assert docs_only is False
    assert "malformed" in reason


def test_rename_copy_delete_and_other_statuses_are_full() -> None:
    for record in (
        "R100\told.md\tnew.md\n",
        "C100\told.md\tnew.md\n",
        "D\told.md\n",
        "T\tdocs_site/docs/index.md\n",
        "M\tdocs_site/mkdocs.yml\n",
    ):
        docs_only, reason = classify_paths(record)
        assert docs_only is False
        assert "full" in reason


def test_multiple_status_records_are_classified_as_one_diff() -> None:
    docs_only, _ = classify_paths("A\tdocs_site/docs/index.md\nM\tsrc/x.py\n")

    assert docs_only is False


def test_name_status_records_must_have_exactly_one_path() -> None:
    for record in (
        "A\tdocs_site/docs/index.md\tunexpected\n",
        "M\tdocs_site/docs/index.md\n\n",
        "A docs_site/docs/index.md\n",
    ):
        docs_only, reason = classify_paths(record)
        assert docs_only is False
        assert "full" in reason


def test_public_workflow_has_scoped_policy_shape() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert "pull_request:" in workflow
    assert "push:" in workflow
    assert "public-ci-${{ github.event.pull_request.number || github.ref }}" in workflow
    assert "cancel-in-progress: true" in workflow


def test_every_public_job_is_guarded_before_runner_allocation() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    expected_jobs = {
        "policy",
        "changes",
        "docs",
        "reference",
        "numba-parity",
        "cpp-parity",
        "package-smoke",
        "required",
    }
    guard = "github.repository == 'tatsuki-washimi/gragra'"
    visibility = "github.event.repository.visibility == 'public'"
    for job in expected_jobs:
        match = re.search(
            rf"^  {re.escape(job)}:\n    if: (?P<condition>[^\n]+)\n",
            workflow,
            re.MULTILINE,
        )
        assert match, f"missing active job-level if for {job}"
        condition = match.group("condition")
        assert guard in condition
        assert visibility in condition
        assert re.search(
            rf"^  {re.escape(job)}:\n    if: [^\n]+\n    runs-on: ubuntu-latest\n"
            rf"    timeout-minutes: \d+$",
            workflow,
            re.MULTILINE,
        )

    required_condition = (
        "always() && github.repository == 'tatsuki-washimi/gragra' && "
        "github.event.repository.visibility == 'public'"
    )
    assert re.search(
        rf"^  required:\n    if: {re.escape(required_condition)}$",
        workflow,
        re.MULTILINE,
    )


def test_workflow_is_public_only_and_actions_are_sha_pinned() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert re.search(r"push:\n\s+branches: \[main\]", workflow)
    assert re.search(r"pull_request:\n\s+branches: \[main\]", workflow)
    for forbidden in (
        "schedule:",
        "workflow_run:",
        "repository_dispatch:",
        "pull_request_target:",
    ):
        assert forbidden not in workflow
    assert "permissions:\n  contents: read" in workflow
    for uses in re.findall(r"uses:\s+([^\s]+)", workflow):
        assert re.fullmatch(r"actions/[a-z-]+@[0-9a-f]{40}", uses), uses


def test_changes_and_final_gate_propagate_selection_without_fallback() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    needs = (
        "needs: [policy, changes, docs, reference, numba-parity, cpp-parity, "
        "package-smoke]"
    )

    assert "needs: policy" in workflow
    assert "fetch-depth: 0" in workflow
    assert "github.event.pull_request.base.sha" in workflow
    assert "github.event.pull_request.head.sha" in workflow
    assert "github.event.before" in workflow
    assert "github.sha" in workflow
    assert "0000000000000000000000000000000000000000" in workflow
    assert "git diff --name-status --find-renames" in workflow
    assert "GITHUB_OUTPUT" in workflow
    assert "if: always()" in workflow
    assert needs in workflow
    assert "failure" in workflow
    assert "cancelled" in workflow
    assert "skipped" in workflow
    assert "docs_only" in workflow


def test_full_checks_cover_references_backends_and_fresh_package_environments() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert "python-version: ['3.11', '3.12']" in workflow
    assert "GRAGRA_SPECFEM_BACKEND=numba" in workflow
    assert "GRAGRA_SPECFEM_BACKEND=cpp" in workflow
    assert "source_copy=$(mktemp -d)" in workflow
    assert 'cp -a "$GITHUB_WORKSPACE"/. "$source_copy"/' in workflow
    assert "from gragra.kernels._cpp_required import ALL_REQUIRED" in workflow
    assert "missing required C++ symbols" in workflow
    assert "python -m build --wheel --sdist" in workflow
    assert 'python -m venv "$venv_dir"' in workflow
    assert "checkout == imported or checkout in imported.parents" in workflow


def test_public_workflow_has_no_publication_or_private_scope_steps() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text().lower()

    for forbidden in (
        "upload-artifact",
        "twine upload",
        "gh release",
        "private",
        "deploy",
    ):
        assert forbidden not in workflow
