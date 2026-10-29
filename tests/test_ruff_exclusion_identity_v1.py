"""Keep the approved Ruff exclusions bound to one reviewed byte sequence."""

import hashlib
import json
import shutil
import subprocess
import tomllib
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = "docs/design/gen1/ruff-exclusion-identity-v1.json"
APPROVAL_PATH = "docs/design/gen1/lint-exemptions-v1.json"
EXPECTED_SCHEMA = "gragra-gen1-ruff-exclusion-identity-v1"
EXPECTED_BASELINE_COMMIT = "6ea5221f68a3d2eeb30a0c3c8d3bd7838107665b"
EXPECTED_BASELINE_TREE = "f6f4b83aee91759611d4ac975ed58cf648ae7b25"
EXPECTED_Q2_COMMIT = "2942809b7b456b88c96e4b5edbac91314e9ba541"
EXPECTED_Q2_TREE = "fb05afa6587623d992b79888f443657f3b9ae021"
EXPECTED_Q2_SOURCE_COMMIT = "d0abad235527019915b913b09ae20faaa11f3c3a"
Q2_NUMERICS_PATH = "src/gragra/analytic/atmospheric/_cavity/_numerics.py"
EXPECTED_NUMERICS_A9_SHA256 = (
    "01c4b40bb45e1bf07de5f22b8fd9d4ee8824999616417f7170c3be9313ee269f"
)
EXPECTED_NUMERICS_A9_BLOB = "54dbdc6ce28cb8a94baccbe2265d849ad41c1c06"
EXPECTED_COUNTS = {"lint_and_format": 30, "format_only": 14}
A2_DEPTH_PATH = "src/gragra/analytic/stochastic_acoustic.py"
# Bytes before the Watson-coefficient / exponential-sum correction (Issue #180).
# They remain the bytes at the A9 baseline and the Q2 integration candidate.
EXPECTED_A2_DEPTH_PRE_FIX_SHA256 = (
    "4322306dd617e6e395e9ca960672fc16581840a67162bbf75b158a55165f01b3"
)
EXPECTED_A2_DEPTH_PRE_FIX_BLOB = "b189b5fee5dd277e8eaed9de77fe3afa1f890082"
HEX_DIGITS = set("0123456789abcdef")


def _valid_hex(value, length):
    return isinstance(value, str) and len(value) == length and set(value) <= HEX_DIGITS


def _git_blob_sha1(raw):
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def _safe_relative_file_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts)
        and not any(character in value for character in "*?[]")
        and not value.endswith("/")
    )


def _read_json(root, relative_path, failures, label):
    path = root / relative_path
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        failures.append(f"{label} is missing: {relative_path}")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        failures.append(f"{label} is invalid: {relative_path}: {exc}")
    return None


def _identity_errors(root, *, verify_git=False):
    """Return all contract violations without depending on external assets."""
    failures = []
    identity = _read_json(root, IDENTITY_PATH, failures, "identity index")
    if identity is None:
        return failures

    if identity.get("schema") != EXPECTED_SCHEMA:
        failures.append("identity index schema differs")
    baseline = identity.get("identity_baseline", {})
    if baseline.get("commit") != EXPECTED_BASELINE_COMMIT:
        failures.append("identity baseline commit differs")
    if baseline.get("tree") != EXPECTED_BASELINE_TREE:
        failures.append("identity baseline tree differs")
    if baseline.get("contains_this_index") is not False:
        failures.append("identity baseline must predate this index")
    if baseline.get("scientific_provenance_claim") is not False:
        failures.append("identity baseline makes an unsupported provenance claim")
    q2_candidate = identity.get("q2_integration_candidate", {})
    if q2_candidate.get("commit") != EXPECTED_Q2_COMMIT:
        failures.append("Q2 integration candidate commit differs")
    if q2_candidate.get("tree") != EXPECTED_Q2_TREE:
        failures.append("Q2 integration candidate tree differs")
    if q2_candidate.get("validation_source_commit") != EXPECTED_Q2_SOURCE_COMMIT:
        failures.append("Q2 validation source commit differs")
    if q2_candidate.get("scientific_provenance_claim") is not False:
        failures.append(
            "Q2 integration candidate makes an unsupported provenance claim"
        )
    if (
        identity.get("identity_fields", {}).get(
            "git_blob_sha1_is_not_commit_or_tree_id"
        )
        is not True
    ):
        failures.append("Git blob identity is not distinguished from commit/tree IDs")
    if not identity.get("historical_claim"):
        failures.append("identity index lacks its historical scope statement")

    approval = identity.get("approval_record", {})
    if approval.get("path") != APPROVAL_PATH:
        failures.append("approval record path differs")
    if approval.get("schema") != "gragra-gen1-lint-exemptions-v1":
        failures.append("approval record schema differs")
    approval_path = root / APPROVAL_PATH
    approval_bytes = approval_path.read_bytes() if approval_path.is_file() else None
    if approval_bytes is None:
        failures.append(f"approval record is missing: {APPROVAL_PATH}")
        approval_record = None
        approval_sha256 = None
    else:
        approval_sha256 = hashlib.sha256(approval_bytes).hexdigest()
        if approval.get("sha256") != approval_sha256:
            failures.append("approval record SHA256 differs")
        try:
            approval_record = json.loads(approval_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            failures.append(f"approval record is invalid: {exc}")
            approval_record = None

    approved_by_group = {"lint_and_format": {}, "format_only": {}}
    if approval_record is not None:
        for group, approval_key in (
            ("lint_and_format", "exempt_lint_and_format"),
            ("format_only", "exempt_format_only"),
        ):
            entries = approval_record.get(approval_key, [])
            for entry in entries:
                path = entry.get("path")
                if path in approved_by_group[group]:
                    failures.append(f"duplicate approval path: {path}")
                approved_by_group[group][path] = entry
        if approval_record.get("schema") != "gragra-gen1-lint-exemptions-v1":
            failures.append("approval record schema content differs")

    scope = identity.get("scope", {})
    if scope != EXPECTED_COUNTS | {"total": 44}:
        failures.append("identity index scope counts differ")

    rows = identity.get("files")
    if not isinstance(rows, list):
        failures.append("identity index files must be a list")
        rows = []

    paths_by_group = {group: [] for group in EXPECTED_COUNTS}
    rows_by_path = {}
    for row in rows:
        if not isinstance(row, dict):
            failures.append("identity index file entry must be an object")
            continue
        path = row.get("path")
        group = row.get("exemption_class")
        if not _safe_relative_file_path(path):
            failures.append(f"unsafe or non-file identity path: {path!r}")
            continue
        if group not in EXPECTED_COUNTS:
            failures.append(f"unknown exclusion class for {path}: {group!r}")
            continue
        if path in rows_by_path:
            failures.append(f"duplicate identity path: {path}")
        rows_by_path[path] = row
        paths_by_group[group].append(path)

        if not _valid_hex(row.get("sha256"), 64):
            failures.append(f"invalid SHA256 in identity index: {path}")
        if not _valid_hex(row.get("git_blob_sha1"), 40):
            failures.append(f"invalid Git blob SHA1 in identity index: {path}")
        if path == Q2_NUMERICS_PATH:
            if row.get("historical_identity_at_baseline") != {
                "sha256": EXPECTED_NUMERICS_A9_SHA256,
                "git_blob_sha1": EXPECTED_NUMERICS_A9_BLOB,
            }:
                failures.append(f"historical A9 identity differs: {path}")
        elif path == A2_DEPTH_PATH:
            if row.get("historical_identity_at_baseline") != {
                "sha256": EXPECTED_A2_DEPTH_PRE_FIX_SHA256,
                "git_blob_sha1": EXPECTED_A2_DEPTH_PRE_FIX_BLOB,
            }:
                failures.append(f"historical pre-fix identity differs: {path}")
        elif "historical_identity_at_baseline" in row:
            failures.append(f"unexpected historical baseline override: {path}")

        reference = row.get("approval_record_reference", {})
        if reference != {
            "record_sha256": approval_sha256,
            "entry_group": (
                "exempt_lint_and_format"
                if group == "lint_and_format"
                else "exempt_format_only"
            ),
            "entry_path": path,
        }:
            failures.append(f"approval record reference differs: {path}")

        if path not in approved_by_group[group]:
            failures.append(f"identity classification lacks approval: {path}")
        elif approval_record is not None:
            approved = approved_by_group[group][path]
            recorded_sha256 = approved.get("current_sha256")
            if recorded_sha256 is not None and row.get("sha256") != recorded_sha256:
                failures.append(
                    f"identity differs from existing approval record: {path}"
                )
            expected_prior_identity = (
                "recorded_in_lint_exemptions_v1"
                if recorded_sha256 is not None
                else "not_recorded_in_lint_exemptions_v1"
            )
            if row.get("prior_identity_status") != expected_prior_identity:
                failures.append(f"prior identity status differs: {path}")
            recorded_blob = approved.get("git_blob_sha1_at_head")
            if recorded_blob is not None and row.get("git_blob_sha1") != recorded_blob:
                failures.append(
                    f"Git blob identity differs from approval record: {path}"
                )

        file_path = root / path
        if not file_path.is_file():
            failures.append(f"excluded file is missing: {path}")
            continue
        raw = file_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row.get("sha256"):
            failures.append(f"excluded file SHA256 mismatch: {path}")
        if _git_blob_sha1(raw) != row.get("git_blob_sha1"):
            failures.append(f"excluded file Git blob SHA1 mismatch: {path}")

    for group, expected_count in EXPECTED_COUNTS.items():
        if len(paths_by_group[group]) != expected_count:
            failures.append(f"identity index {group} count differs")
        if len(set(paths_by_group[group])) != len(paths_by_group[group]):
            failures.append(f"identity index {group} contains duplicate paths")
        if set(paths_by_group[group]) != set(approved_by_group[group]):
            failures.append(f"identity index {group} paths differ from approval record")
    if set(paths_by_group["lint_and_format"]) & set(paths_by_group["format_only"]):
        failures.append("identity index exclusion classes overlap")
    if len(rows) != 44 or len(rows_by_path) != 44:
        failures.append("identity index must contain exactly 44 unique files")
    if set(rows_by_path) != set(approved_by_group["lint_and_format"]) | set(
        approved_by_group["format_only"]
    ):
        failures.append("identity index paths do not match the approved file set")

    pyproject = _read_toml(root / "pyproject.toml", failures)
    if pyproject is not None:
        ruff = pyproject.get("tool", {}).get("ruff", {})
        lint_paths = ruff.get("extend-exclude", [])
        format_paths = ruff.get("format", {}).get("exclude", [])
        exclusion_fields = _collect_exclusion_fields(ruff)
        expected_fields = {
            ("extend-exclude",),
            ("format", "exclude"),
        }
        if set(exclusion_fields) != expected_fields:
            failures.append("Ruff has an unregistered exclusion setting")
        for label, paths in (("lint", lint_paths), ("format", format_paths)):
            if not isinstance(paths, list) or any(
                not isinstance(path, str) for path in paths
            ):
                failures.append(f"Ruff {label} exclusions must be a string list")
                continue
            if len(paths) != len(set(paths)):
                failures.append(f"Ruff {label} exclusions contain duplicates")
        if (
            isinstance(lint_paths, list)
            and all(isinstance(path, str) for path in lint_paths)
            and set(lint_paths) != set(paths_by_group["lint_and_format"])
        ):
            failures.append("Ruff lint exclusion paths differ from identity index")
        if (
            isinstance(format_paths, list)
            and all(isinstance(path, str) for path in format_paths)
            and set(format_paths) != set(paths_by_group["format_only"])
        ):
            failures.append("Ruff format exclusion paths differ from identity index")

    if verify_git:
        failures.extend(_git_revision_errors(root, identity, rows_by_path))

    return failures


def _git_revision_errors(root, identity, rows_by_path):
    """Check each identity against its declared commit and preserve the A9 bytes."""
    failures = []
    baseline = identity["identity_baseline"]
    candidate = identity["q2_integration_candidate"]
    revisions = (
        ("A9 identity baseline", baseline["commit"], baseline["tree"]),
        ("Q2 integration candidate", candidate["commit"], candidate["tree"]),
    )
    valid_commits = set()
    for label, commit, expected_tree in revisions:
        actual_commit = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "rev-parse",
                "--verify",
                "--quiet",
                f"{commit}^{{commit}}",
            ],
            capture_output=True,
            check=False,
        )
        if actual_commit.returncode != 0:
            failures.append(f"{label} commit is unavailable: {commit}")
            continue
        valid_commits.add(commit)
        actual_tree = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "rev-parse",
                "--verify",
                "--quiet",
                f"{commit}^{{tree}}",
            ],
            capture_output=True,
            check=False,
        )
        if (
            actual_tree.returncode != 0
            or actual_tree.stdout.decode().strip() != expected_tree
        ):
            failures.append(f"{label} tree differs or is unavailable: {commit}")

    source_commit = candidate["validation_source_commit"]
    source_result = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{source_commit}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if source_result.returncode != 0:
        failures.append(f"Q2 validation source commit is unavailable: {source_commit}")

    def check_path(commit, path, expected_sha256, expected_blob, label):
        blob_result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "rev-parse",
                "--verify",
                "--quiet",
                f"{commit}:{path}",
            ],
            capture_output=True,
            check=False,
        )
        if blob_result.returncode != 0:
            failures.append(f"{label} blob is unavailable: {path} at {commit}")
            return
        blob = blob_result.stdout.decode().strip()
        if blob != expected_blob:
            failures.append(f"{label} Git blob differs: {path} at {commit}")
        content = subprocess.run(
            ["git", "-C", str(root), "cat-file", "blob", blob],
            capture_output=True,
            check=False,
        )
        if content.returncode != 0:
            failures.append(f"{label} blob content is unavailable: {path} at {commit}")
        elif hashlib.sha256(content.stdout).hexdigest() != expected_sha256:
            failures.append(f"{label} SHA256 differs: {path} at {commit}")

    for path, row in rows_by_path.items():
        current_sha256 = row.get("sha256")
        current_blob = row.get("git_blob_sha1")
        if candidate["commit"] in valid_commits:
            # The A2 depth source changed after the Q2 candidate; the candidate
            # still holds the pre-fix bytes recorded as the historical identity.
            at_candidate = (
                row.get("historical_identity_at_baseline", {})
                if path == A2_DEPTH_PATH
                else {}
            )
            check_path(
                candidate["commit"],
                path,
                at_candidate.get("sha256", current_sha256),
                at_candidate.get("git_blob_sha1", current_blob),
                "Q2 integration candidate",
            )
        if baseline["commit"] in valid_commits:
            historical = row.get("historical_identity_at_baseline", {})
            check_path(
                baseline["commit"],
                path,
                historical.get("sha256", current_sha256),
                historical.get("git_blob_sha1", current_blob),
                "A9 identity baseline",
            )
        if path == Q2_NUMERICS_PATH and source_result.returncode == 0:
            check_path(
                source_commit,
                path,
                current_sha256,
                current_blob,
                "Q2 validation source",
            )

    return failures


def _read_toml(path, failures):
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        failures.append("pyproject.toml is missing")
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        failures.append(f"pyproject.toml is invalid: {exc}")
    return None


def _collect_exclusion_fields(node, prefix=()):
    fields = {}
    if not isinstance(node, dict):
        return fields
    for key, value in node.items():
        path = prefix + (key,)
        if "exclude" in key:
            fields[path] = value
        if isinstance(value, dict):
            fields.update(_collect_exclusion_fields(value, path))
    return fields


def _copy_contract_tree(destination):
    identity = json.loads((REPO_ROOT / IDENTITY_PATH).read_text(encoding="utf-8"))
    paths = {
        IDENTITY_PATH,
        APPROVAL_PATH,
        "pyproject.toml",
        *(row["path"] for row in identity["files"]),
    }
    for relative_path in paths:
        source = REPO_ROOT / relative_path
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return json.loads((destination / IDENTITY_PATH).read_text(encoding="utf-8"))


def _write_identity(root, identity):
    (root / IDENTITY_PATH).write_text(
        json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _move_ruff_exclusion(root, path, source_group, destination_group):
    pyproject_path = root / "pyproject.toml"
    content = pyproject_path.read_text(encoding="utf-8")
    line = f'    "{path}",\n'
    format_marker = "[tool.ruff.format]"
    lint_content, format_content = content.split(format_marker, maxsplit=1)
    if source_group == "lint_and_format":
        lint_content = lint_content.replace(line, "", 1)
    elif source_group == "format_only":
        format_content = format_content.replace(line, "", 1)
    if destination_group == "lint_and_format":
        lint_content = lint_content.replace(
            "extend-exclude = [\n", f"extend-exclude = [\n{line}", 1
        )
    elif destination_group == "format_only":
        format_content = format_content.replace(
            "exclude = [\n", f"exclude = [\n{line}", 1
        )
    pyproject_path.write_text(
        lint_content + format_marker + format_content, encoding="utf-8"
    )


def test_ruff_exclusions_match_the_fixed_44_file_identity():
    failures = _identity_errors(REPO_ROOT, verify_git=True)
    assert not failures, "\n".join(failures)


def test_a9_baseline_and_q2_integration_identity_are_revision_scoped():
    identity = json.loads((REPO_ROOT / IDENTITY_PATH).read_text(encoding="utf-8"))
    row = next(
        item
        for item in identity["files"]
        if item["path"] == "src/gragra/analytic/atmospheric/_cavity/_numerics.py"
    )

    assert identity["identity_baseline"]["commit"] == EXPECTED_BASELINE_COMMIT
    candidate = identity["q2_integration_candidate"]
    assert candidate["commit"] == "2942809b7b456b88c96e4b5edbac91314e9ba541"
    assert candidate["tree"] == "fb05afa6587623d992b79888f443657f3b9ae021"
    assert row["historical_identity_at_baseline"] == {
        "sha256": "01c4b40bb45e1bf07de5f22b8fd9d4ee8824999616417f7170c3be9313ee269f",
        "git_blob_sha1": "54dbdc6ce28cb8a94baccbe2265d849ad41c1c06",
    }
    assert row["git_blob_sha1"] == "0da2efddeb9eb50de149a0996079ce69fe0ad72b"


def test_wrong_q2_identity_revision_fails(tmp_path):
    identity = _copy_contract_tree(tmp_path)
    identity["q2_integration_candidate"] = {
        "commit": "0" * 40,
        "tree": "fb05afa6587623d992b79888f443657f3b9ae021",
        "validation_source_commit": "d0abad235527019915b913b09ae20faaa11f3c3a",
        "meaning": "Q2 source identity in the reviewed integration candidate.",
    }
    _write_identity(tmp_path, identity)

    failures = _identity_errors(tmp_path)

    assert "Q2 integration candidate commit differs" in failures


def test_git_blob_identity_uses_git_object_encoding():
    raw = b"identity probe\n"
    result = subprocess.run(
        ["git", "hash-object", "--stdin"],
        cwd=REPO_ROOT,
        input=raw,
        capture_output=True,
        check=True,
    )

    assert _git_blob_sha1(raw) == result.stdout.decode("ascii").strip()


def test_one_byte_file_change_fails_even_when_index_and_ruff_config_agree(tmp_path):
    identity = _copy_contract_tree(tmp_path)
    row = identity["files"][0]
    path = tmp_path / row["path"]
    raw = bytearray(path.read_bytes())
    raw[0] ^= 1
    path.write_bytes(raw)

    failures = _identity_errors(tmp_path)

    assert f"excluded file SHA256 mismatch: {row['path']}" in failures
    assert f"excluded file Git blob SHA1 mismatch: {row['path']}" in failures


def test_unregistered_ruff_exclusion_fails(tmp_path):
    _copy_contract_tree(tmp_path)
    pyproject_path = tmp_path / "pyproject.toml"
    content = pyproject_path.read_text(encoding="utf-8")
    content = content.replace(
        "extend-exclude = [\n",
        'extend-exclude = [\n    "tests/unregistered_exclusion.py",\n',
        1,
    )
    pyproject_path.write_text(content, encoding="utf-8")

    failures = _identity_errors(tmp_path)

    assert "Ruff lint exclusion paths differ from identity index" in failures


def test_new_ruff_exclusion_setting_fails(tmp_path):
    _copy_contract_tree(tmp_path)
    pyproject_path = tmp_path / "pyproject.toml"
    content = pyproject_path.read_text(encoding="utf-8")
    content = content.replace(
        "[tool.ruff]\n",
        '[tool.ruff]\nexclude = ["tests/unregistered.py"]\n',
        1,
    )
    pyproject_path.write_text(content, encoding="utf-8")

    failures = _identity_errors(tmp_path)

    assert "Ruff has an unregistered exclusion setting" in failures


def test_ruff_exclusion_classification_change_fails(tmp_path):
    identity = _copy_contract_tree(tmp_path)
    row = next(
        item
        for item in identity["files"]
        if item["exemption_class"] == "lint_and_format"
    )
    row["exemption_class"] = "format_only"
    row["approval_record_reference"]["entry_group"] = "exempt_format_only"
    _move_ruff_exclusion(tmp_path, row["path"], "lint_and_format", "format_only")
    _write_identity(tmp_path, identity)

    failures = _identity_errors(tmp_path)

    assert f"identity classification lacks approval: {row['path']}" in failures
    assert "identity index lint_and_format count differs" in failures
    assert "identity index format_only count differs" in failures
    assert "Ruff lint exclusion paths differ from identity index" not in failures
    assert "Ruff format exclusion paths differ from identity index" not in failures


def test_missing_identity_entry_fails(tmp_path):
    identity = _copy_contract_tree(tmp_path)
    identity["files"].pop(0)
    _write_identity(tmp_path, identity)

    failures = _identity_errors(tmp_path)

    assert "identity index must contain exactly 44 unique files" in failures
    assert "Ruff lint exclusion paths differ from identity index" in failures


def test_missing_ruff_exclusion_fails(tmp_path):
    identity = _copy_contract_tree(tmp_path)
    row = next(
        item
        for item in identity["files"]
        if item["exemption_class"] == "lint_and_format"
    )
    pyproject_path = tmp_path / "pyproject.toml"
    content = pyproject_path.read_text(encoding="utf-8")
    content = content.replace(f'    "{row["path"]}",\n', "", 1)
    pyproject_path.write_text(content, encoding="utf-8")

    failures = _identity_errors(tmp_path)

    assert "Ruff lint exclusion paths differ from identity index" in failures


def test_duplicate_identity_path_fails(tmp_path):
    identity = _copy_contract_tree(tmp_path)
    identity["files"].append(identity["files"][0].copy())
    _write_identity(tmp_path, identity)

    failures = _identity_errors(tmp_path)

    assert f"duplicate identity path: {identity['files'][0]['path']}" in failures
    assert "identity index must contain exactly 44 unique files" in failures


def test_missing_excluded_file_fails(tmp_path):
    identity = _copy_contract_tree(tmp_path)
    path = identity["files"][0]["path"]
    (tmp_path / path).unlink()

    failures = _identity_errors(tmp_path)

    assert f"excluded file is missing: {path}" in failures


def test_wrong_git_blob_identity_fails(tmp_path):
    identity = _copy_contract_tree(tmp_path)
    row = identity["files"][0]
    row["git_blob_sha1"] = "0" * 40
    _write_identity(tmp_path, identity)

    failures = _identity_errors(tmp_path)

    assert f"excluded file Git blob SHA1 mismatch: {row['path']}" in failures


def test_existing_approval_record_bytes_are_pinned(tmp_path):
    _copy_contract_tree(tmp_path)
    path = tmp_path / APPROVAL_PATH
    raw = bytearray(path.read_bytes())
    raw[0] ^= 1
    path.write_bytes(raw)

    failures = _identity_errors(tmp_path)

    assert "approval record SHA256 differs" in failures


def test_conftest_identity_is_new_at_the_baseline_not_historical_evidence():
    identity = json.loads((REPO_ROOT / IDENTITY_PATH).read_text(encoding="utf-8"))
    row = next(
        item for item in identity["files"] if item["path"] == "tests/conftest.py"
    )

    assert row["prior_identity_status"] == "not_recorded_in_lint_exemptions_v1"
    assert _valid_hex(row["sha256"], 64)
    assert _valid_hex(row["git_blob_sha1"], 40)


def test_a2_depth_source_records_pre_fix_identity_alongside_current_bytes():
    identity = json.loads((REPO_ROOT / IDENTITY_PATH).read_text(encoding="utf-8"))
    row = next(item for item in identity["files"] if item["path"] == A2_DEPTH_PATH)

    assert row["historical_identity_at_baseline"] == {
        "sha256": EXPECTED_A2_DEPTH_PRE_FIX_SHA256,
        "git_blob_sha1": EXPECTED_A2_DEPTH_PRE_FIX_BLOB,
    }
    assert row["sha256"] != EXPECTED_A2_DEPTH_PRE_FIX_SHA256
    assert row["git_blob_sha1"] != EXPECTED_A2_DEPTH_PRE_FIX_BLOB
