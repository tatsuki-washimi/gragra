"""Cold validation of the exact retained A2 reference envelope.

This module reads captured pinned artifact bytes. It does not import the
production package, start a process, consult Git, or generate reference values.
It reads the sealed historical resource ledger as evidence and does not mutate
or access the current operational diagnostic ledger. Ancestor directory
identity is not claimed; validation binds the bytes captured from each opened
file descriptor to their immutable hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any


class RetainedReferenceError(ValueError):
    """A retained A2 reference pin, record, or review does not match."""


RETAINED_SCHEMA = "gragra.a2.v2.retained-reference-manifest.v1"
ADMISSION_SCHEMA = "gen1-stochastic-reference-admission-v1"
REVIEW_SCHEMA = "gen1-stochastic-reference-review-v1"
SCIENCE_REPORT_SCHEMA = "gen1-stochastic-reference-science-report-v1"
SCIENCE_AUDIT_KIND = "science_retained_record_audit"

E_COMMIT = "60bfc4045b48276c111e62bcd30159f071627d05"
E_TREE = "7ffb8b23bd9224caf43074138c59fe0aa19d4413"
ATTEMPT_ROOT = "/var/tmp/gragra-gen1-production-20261004/a2/attempts/a2-reference-001"
RECORDS_PATH = ATTEMPT_ROOT + "/records.raw.jsonl"
SEAL_PATH = ATTEMPT_ROOT + "/seal.json"
ADMISSION_PATH = "/home/washimi/.local/state/gragra/gen1-production/20261004-pr159-final/a2/reference-admission-001-v1.json"

RECORDS_SHA256 = "1b30be2a4a1ebebd6b5baac1192c4e564ba4790cfcb299beff6e6bee930fe80a"
SEAL_SHA256 = "f9f788ce67901e7a4a23af64f69fd06f2b0ad452fd75f0a6568aad4b90b81ff5"
ADMISSION_SHA256 = "e5bb957ce4abfb543f0ba0058962e325f8ce7f2461b1266a97730c47b35c58c8"
AUDIT_SHA256 = "b80670f136f408933fb59dccf2f9b757ee2bbc31b726843d0a98df153a3ef990"

FROZEN_ARTIFACTS = {
    "fixture": (
        "docs/design/gen1/stochastic/A2-fixtures-v1.json",
        "8e5c5ba6f739bd978310b3087a12f966447d4e8e94427f14c776e6f4b1f020b7",
    ),
    "budget": (
        "docs/design/gen1/stochastic/A2-budget-v1.json",
        "21ae08e8056ab4ebbc645807a35a6ecbc2bdd4729c7d6b76fc70533f7d97ab81",
    ),
    "family_procedure": (
        "docs/design/gen1/stochastic/A2-procedure-v1.json",
        "c33ff1d2f6d9c811d1e229fbcc9d90d10bc17e0357dd2520f8cc8b83b6be5995",
    ),
    "freeze": (
        "docs/design/gen1/stochastic/A2-freeze-v1.json",
        "97042caf874364eaf6003f8dcdc4d160d07bb03ab91bb2e45d1df732ec2fb5a3",
    ),
}

UNCERTAINTY_CAUSES = frozenset(
    {
        "input",
        "root",
        "endpoint",
        "quadrature",
        "tail",
        "formulation",
        "precision",
        "roundoff",
        "serialization",
    }
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_NATIVE_KEY_FIELD_ORDER = (
    "family",
    "fixture_id",
    "frequency_hz",
    "target_id",
    "observable",
    "representation",
    "set",
)
_NATIVE_KEY_FIELDS = frozenset(_NATIVE_KEY_FIELD_ORDER)
_RECORD_FIELDS = frozenset(
    {
        "admission_status",
        "input_assumptions",
        "input_convention",
        "inputs",
        "key",
        "procedure_hash",
        "quantity",
        "schema_version",
        "source_pins",
        "status",
        "uncertainty",
        "units",
    }
)
_ATTEMPT_ARTIFACTS = frozenset(
    {
        "authorization.json",
        "family-freeze.json",
        "policy.json",
        "records.raw.jsonl",
        "resource-ledger.json",
        "resource-result.json",
        "purpose-records.json",
        "stderr.jsonl",
        "stdout.jsonl",
    }
)


def _fail(message: str) -> None:
    raise RetainedReferenceError(message)


def _reject_production_modules() -> None:
    if any(name == "gragra" or name.startswith("gragra.") for name in sys.modules):
        _fail("production module is loaded in the independent retained verifier")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            _fail("duplicate JSON object field")
        result[name] = value
    return result


def _finite_json_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise RetainedReferenceError("malformed JSON number") from exc
    if not math.isfinite(number):
        _fail("nonfinite JSON number")
    return number


def _reject_json_constant(value: str) -> None:
    _fail("nonfinite JSON constant: " + value)


def _strict_json(raw: bytes) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetainedReferenceError("malformed UTF-8 JSON artifact") from exc


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise RetainedReferenceError("value is not canonical JSON") from exc


def _canonical_bytes(value: Any) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _canonical_native_key(key: Any) -> str:
    if (
        not isinstance(key, dict)
        or set(key) != _NATIVE_KEY_FIELDS
        or any(not isinstance(value, str) or not value for value in key.values())
    ):
        _fail("native A2 key must retain exactly seven nonempty string fields")
    return _canonical_json(key)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_regular(path: str, *, label: str, read_only: bool = False) -> bytes:
    if not isinstance(path, str) or not Path(path).is_absolute():
        _fail(label + " path must be absolute")
    target = Path(path)
    if not hasattr(os, "O_NOFOLLOW"):
        _fail("O_NOFOLLOW is unavailable for pinned artifact reads")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = -1
    try:
        fd = os.open(target, flags)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            _fail(label + " is not a regular file")
        if read_only and before.st_mode & 0o222:
            _fail(label + " remains writable")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        raw = b"".join(chunks)
        if identity_before != identity_after or len(raw) != after.st_size:
            _fail(label + " changed while its pinned bytes were captured")
        if read_only and after.st_mode & 0o222:
            _fail(label + " became writable while its pinned bytes were captured")
        return raw
    except OSError as exc:
        raise RetainedReferenceError(label + " cannot be read") from exc
    finally:
        if fd >= 0:
            os.close(fd)


def _check_sha(raw: bytes, expected: Any, *, label: str) -> str:
    if not isinstance(expected, str) or not _HEX64.fullmatch(expected):
        _fail(label + " SHA256 pin is malformed")
    actual = _sha256(raw)
    if actual != expected:
        _fail(label + " bytes differ from the pinned SHA256")
    return actual


def _manifest_pin(
    value: Any,
    *,
    path: str,
    sha256: str,
    label: str,
    read_only: bool = False,
) -> bytes:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        _fail(label + " manifest pin fields differ")
    if value.get("path") != path or value.get("sha256") != sha256:
        _fail(label + " manifest pin differs from the retained schema")
    raw = _read_regular(path, label=label, read_only=read_only)
    _check_sha(raw, sha256, label=label)
    return raw


def _native_bytes(pin: Any, *, label: str, read_only: bool = False) -> bytes:
    allowed = (
        {"path", "sha256"},
        {"path", "sha256", "kind"},
        {"path", "sha256", "decision"},
    )
    if not isinstance(pin, dict) or set(pin) not in allowed:
        _fail(label + " native pin fields differ")
    if "decision" in pin and pin["decision"] != "PASS":
        _fail(label + " native pin decision is not PASS")
    if "kind" in pin and (not isinstance(pin["kind"], str) or not pin["kind"]):
        _fail(label + " native pin kind is malformed")
    raw = _read_regular(pin["path"], label=label, read_only=read_only)
    _check_sha(raw, pin["sha256"], label=label)
    return raw


def _strict_pin_document(pin: Any, *, label: str) -> dict[str, Any]:
    value = _strict_json(_native_bytes(pin, label=label))
    if not isinstance(value, dict):
        _fail(label + " must contain a JSON object")
    return value


def _require_readonly_root(path: str) -> Path:
    if not isinstance(path, str) or path != ATTEMPT_ROOT:
        _fail("retained attempt root differs from the native envelope")
    root = Path(path)
    if not hasattr(os, "O_NOFOLLOW"):
        _fail("O_NOFOLLOW is unavailable for retained root inspection")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    fd = -1
    try:
        fd = os.open(root, flags)
        opened = os.fstat(fd)
        if not stat.S_ISDIR(opened.st_mode) or opened.st_mode & 0o222:
            _fail("native retained attempt root is not a read-only directory")
    except OSError as exc:
        raise RetainedReferenceError("native retained attempt root is unavailable") from exc
    finally:
        if fd >= 0:
            os.close(fd)
    return root


def _load_frozen(manifest: dict[str, Any], retained: dict[str, Any]) -> tuple[dict[str, bytes], dict[str, Any]]:
    paths = manifest.get("paths")
    if (
        not isinstance(paths, dict)
        or not isinstance(paths.get("worktree_root"), str)
        or not Path(paths["worktree_root"]).is_absolute()
    ):
        _fail("manifest paths.worktree_root must be absolute for frozen artifact pins")
    try:
        worktree = Path(paths["worktree_root"]).resolve(strict=True)
    except OSError as exc:
        raise RetainedReferenceError("candidate worktree root is unavailable") from exc
    if not worktree.is_dir():
        _fail("candidate worktree root is not a directory")

    supplied = retained.get("frozen_artifacts")
    if not isinstance(supplied, dict) or set(supplied) != set(FROZEN_ARTIFACTS):
        _fail("retained frozen_artifacts must pin fixture, budget, procedure, and freeze")

    raw_files: dict[str, bytes] = {}
    documents: dict[str, Any] = {}
    for name, (relative, digest) in FROZEN_ARTIFACTS.items():
        expected_path = str(worktree.joinpath(*PurePosixPath(relative).parts))
        raw = _manifest_pin(supplied[name], path=expected_path, sha256=digest, label=name)
        raw_files[name] = raw
        document = _strict_json(raw)
        if not isinstance(document, dict) or document.get("family") != "A2":
            _fail(name + " is not the pinned A2 frozen artifact")
        documents[name] = document

    freeze = documents["freeze"]
    if (
        freeze.get("status") != "FROZEN"
        or freeze.get("fixture_sha256") != FROZEN_ARTIFACTS["fixture"][1]
        or freeze.get("budget_sha256") != FROZEN_ARTIFACTS["budget"][1]
        or freeze.get("procedure_sha256") != FROZEN_ARTIFACTS["family_procedure"][1]
        or documents["family_procedure"].get("fixture_sha256")
        != FROZEN_ARTIFACTS["fixture"][1]
        or documents["family_procedure"].get("budget_sha256")
        != FROZEN_ARTIFACTS["budget"][1]
    ):
        _fail("frozen A2 fixture/budget/procedure/freeze linkage differs")
    return raw_files, documents


def _decimal(value: Any, *, label: str, nonnegative: bool = False) -> Decimal:
    if not isinstance(value, str):
        _fail(label + " must be a retained decimal string")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise RetainedReferenceError(label + " is not a decimal string") from exc
    if not number.is_finite() or (nonnegative and number < 0):
        _fail(label + " is nonfinite or negative")
    return number


def _check_record_payload(record: dict[str, Any], entry: dict[str, Any], *, source_pins: list[Any]) -> None:
    key = entry["native_key"]
    kind = entry["record_kind"]
    if kind == "physical_csd":
        expected_inputs = entry["inputs"]
    elif kind == "nominal_scalar":
        expected_inputs = {
            "frequency_hz": key["frequency_hz"],
            "nominal_exact_x": entry["nominal_exact_x"],
        }
    else:
        _fail("retained map contains an unknown record kind")
    if set(record) != _RECORD_FIELDS:
        _fail("retained record fields differ from the native A2 record schema")
    if (
        type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or record.get("status") != "resolved"
        or record.get("admission_status") != "NOT_FORMALLY_ADMITTED"
        or record.get("procedure_hash") != FROZEN_ARTIFACTS["family_procedure"][1]
        or record.get("key") != key
        or record.get("inputs") != expected_inputs
        or record.get("source_pins") != source_pins
    ):
        _fail("retained record key, kind-specific frozen inputs, or native status differs")

    if kind == "physical_csd":
        if (
            record.get("input_assumptions")
            != "Exact decimal physical model tuple; one-sided per-Hz total surface pressure PSD; ideal rigid atmospheric halfspace; numerical U is estimated."
            or record.get("input_convention") != "surface_total_one_sided_per_hz"
            or record.get("units")
            != ("kg^2 m^-4 Hz^-1" if key["representation"] == "Gfree" else "m^2 s^-4 Hz^-1")
        ):
            _fail("physical retained record convention or units differ")
        expected_shape = [3, 3]
        expected_length = 9
    elif kind == "nominal_scalar":
        if (
            record.get("input_assumptions")
            != "Exact nominal dimensionless x mathematical AUX; Gfree tag means no G factor; no SI representation."
            or record.get("input_convention") != "dimensionless_exact_nominal_x"
            or record.get("units") != "1"
        ):
            _fail("nominal scalar retained record convention or units differ")
        expected_shape = []
        expected_length = 1
    else:
        _fail("retained map contains an unknown record kind")

    quantity = record.get("quantity")
    if (
        not isinstance(quantity, dict)
        or set(quantity) != {"data", "shape"}
        or quantity.get("shape") != expected_shape
        or not isinstance(quantity.get("data"), list)
        or len(quantity["data"]) != expected_length
    ):
        _fail("retained quantity shape differs from its frozen native key")
    for cell in quantity["data"]:
        if not isinstance(cell, dict) or set(cell) != {"real", "imag"}:
            _fail("retained quantity cell is not an exact real/imaginary pair")
        _decimal(cell["real"], label="quantity.real")
        _decimal(cell["imag"], label="quantity.imag")

    uncertainty = record.get("uncertainty")
    if (
        not isinstance(uncertainty, dict)
        or set(uncertainty) != {"kind", "components", "certificate"}
        or uncertainty.get("kind") != "estimate"
        or uncertainty.get("certificate") is not None
        or not isinstance(uncertainty.get("components"), dict)
        or set(uncertainty["components"]) != UNCERTAINTY_CAUSES
    ):
        _fail("retained uncertainty must contain the nine frozen estimate causes")
    for cause, component in uncertainty["components"].items():
        if (
            not isinstance(component, dict)
            or set(component) != {"method", "value"}
            or not isinstance(component.get("method"), str)
            or not component["method"].strip()
        ):
            _fail("retained uncertainty component metadata is malformed")
        _decimal(component["value"], label="uncertainty." + cause, nonnegative=True)


def _verify_records(raw: bytes, entries: list[dict[str, Any]], source_pins: list[Any]) -> dict[str, Any]:
    if not raw.endswith(b"\n") or raw.count(b"\n") != 84:
        _fail("retained JSONL must contain 84 terminated native records")
    expected = {entry["canonical_key_json"]: entry for entry in entries}
    if len(expected) != 84:
        _fail("frozen A2 coverage map is not a unique 84-key map")

    seen: set[str] = set()
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n") or line in (b"\n", b"\r\n") or b"\r" in line:
            _fail("retained JSONL contains a blank or non-LF record")
        record = _strict_json(line)
        if not isinstance(record, dict):
            _fail("retained JSONL row is not an object")
        encoded = _canonical_native_key(record.get("key"))
        entry = expected.get(encoded)
        if entry is None or encoded in seen:
            _fail("retained records contain an unknown or duplicate native key")
        seen.add(encoded)
        _check_record_payload(record, entry, source_pins=source_pins)
    if seen != set(expected):
        _fail("retained records are not a complete 84-key bijection")
    counts = {"physical_csd": 0, "nominal_scalar": 0, "science": 0, "auxiliary": 0}
    for entry in entries:
        counts[entry["record_kind"]] += 1
        counts[entry["acceptance_role"]] += 1
    if counts != {"physical_csd": 56, "nominal_scalar": 28, "science": 54, "auxiliary": 30}:
        _fail("retained two-axis A2 coverage counts differ")
    return {"record_count": 84, "unique_native_keys": 84, **counts}


def validate_adequacy_audit(
    audit: dict[str, Any],
    entries: list[dict[str, Any]],
    *,
    records_sha256: str,
    seal_sha256: str,
    procedure_sha256: str,
) -> dict[str, Any]:
    """Validate all reported ratios in the exact pinned historical audit.

    This checks the audit's 84-key set and its stated per-key bounds. It does
    not recompute the nine-component sum or theta/mu discrepancy.
    """
    if not isinstance(audit, dict):
        _fail("adequacy audit must be an object")
    required = {
        "decision",
        "E",
        "seal_sha256",
        "records_sha256",
        "procedure_sha256",
        "physical_csd",
        "nominal_scalar_aux",
        "counts",
        "no_production_import",
        "checks",
    }
    metadata = {"reviewer_id", "role", "scope"}
    if set(audit) not in (required, required | metadata):
        _fail("adequacy audit fields differ")
    if (
        audit.get("decision") != "PASS"
        or audit.get("E") != E_COMMIT
        or audit.get("records_sha256") != records_sha256
        or audit.get("seal_sha256") != seal_sha256
        or audit.get("procedure_sha256") != procedure_sha256
        or type(audit.get("physical_csd")) is not int
        or audit["physical_csd"] != 56
        or type(audit.get("nominal_scalar_aux")) is not int
        or audit["nominal_scalar_aux"] != 28
        or audit.get("counts") != {"science": 54, "auxiliary": 30}
        or audit.get("no_production_import") is not True
    ):
        _fail("adequacy audit decision, pins, counts, or import claim differs")
    if metadata <= set(audit) and (
        not isinstance(audit.get("reviewer_id"), str)
        or not audit["reviewer_id"]
        or audit.get("role") != "science"
        or not isinstance(audit.get("scope"), str)
        or not audit["scope"]
    ):
        _fail("adequacy audit reviewer metadata is malformed")

    if not isinstance(entries, list) or len(entries) != 84:
        _fail("adequacy audit requires the complete frozen 84-key map")
    expected: dict[str, dict[str, Any]] = {}
    entry_counts = {"physical_csd": 0, "nominal_scalar": 0, "science": 0, "auxiliary": 0}
    for entry in entries:
        if not isinstance(entry, dict):
            _fail("frozen coverage entry is malformed")
        key = entry.get("native_key")
        encoded = _canonical_native_key(key)
        if entry.get("canonical_key_json") != encoded or encoded in expected:
            _fail("frozen coverage entry keys are aliased or duplicated")
        if entry.get("record_kind") not in {"physical_csd", "nominal_scalar"}:
            _fail("frozen coverage entry record kind is unknown")
        if entry.get("acceptance_role") not in {"science", "auxiliary"}:
            _fail("frozen coverage entry acceptance role is unknown")
        if key["family"] != "A2" or key["set"] not in {"science", "auxiliary"}:
            _fail("frozen coverage entry family/set differs")
        kind = entry["record_kind"]
        role = entry["acceptance_role"]
        if key["observable"] == "acceleration_csd":
            if (
                kind != "physical_csd"
                or role != key["set"]
                or key["representation"] not in {"Gfree", "SI"}
                or key["target_id"] != "underground_origin"
            ):
                _fail("physical CSD kind/role differs from its native key")
        elif (
            key["observable"] != "isotropic_depth_factor"
            or kind != "nominal_scalar"
            or role != "auxiliary"
            or key["set"] != "auxiliary"
            or key["representation"] != "Gfree"
            or key["target_id"] != "nominal_x"
        ):
            _fail("nominal scalar kind/role differs from its native key")
        entry_counts[kind] += 1
        entry_counts[role] += 1
        expected[encoded] = entry
    if len(expected) != 84:
        _fail("frozen coverage map is not bijective")
    if entry_counts != {"physical_csd": 56, "nominal_scalar": 28, "science": 54, "auxiliary": 30}:
        _fail("frozen coverage map kind/role counts differ")

    checks = audit.get("checks")
    if not isinstance(checks, list) or len(checks) != 84:
        _fail("adequacy audit does not contain one check per native key")
    seen: set[str] = set()
    maximum = Decimal(0)
    threshold = Decimal("1e-35")
    for row in checks:
        if not isinstance(row, dict) or set(row) not in (
            {"key", "public_plus_coordinate_over_norm"},
            {"key", "public_plus_coordinate_over_norm", "nominal_scalar_aux"},
        ):
            _fail("adequacy check fields differ")
        encoded = _canonical_native_key(row.get("key"))
        entry = expected.get(encoded)
        if entry is None or encoded in seen:
            _fail("adequacy checks contain an unknown or duplicate native key")
        seen.add(encoded)
        if "nominal_scalar_aux" in row:
            if row["nominal_scalar_aux"] is not (entry["record_kind"] == "nominal_scalar"):
                _fail("adequacy scalar classification differs from the frozen map")
        ratio = _decimal(
            row.get("public_plus_coordinate_over_norm"),
            label="public_plus_coordinate_over_norm",
            nonnegative=True,
        )
        if not ratio < threshold:
            _fail("retained public adequacy is not strictly below 1e-35")
        maximum = max(maximum, ratio)
    if seen != set(expected):
        _fail("adequacy audit is not a complete 84-key bijection")
    return {
        "validation_scope": "pinned_historical_audit_reported_ratios_only",
        "checked_keys": 84,
        "maximum_ratio": str(maximum),
        "strict_threshold": "1e-35",
    }


def _verify_native_attempt(
    root: Path,
    records_raw: bytes,
    seal_raw: bytes,
    frozen: dict[str, Any],
    entries: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    seal = _strict_json(seal_raw)
    if not isinstance(seal, dict) or seal.get("format") != "gen1-stochastic-attempt-seal-v2":
        _fail("native A2 seal format differs")
    hashes = seal.get("artifacts_sha256")
    if not isinstance(hashes, dict) or set(hashes) != _ATTEMPT_ARTIFACTS:
        _fail("native A2 sealed artifact closure differs")
    sealed_artifact_bytes: dict[str, bytes] = {}
    for name, digest in hashes.items():
        path = root / name
        artifact_raw = _read_regular(str(path), label="sealed " + name, read_only=True)
        _check_sha(artifact_raw, digest, label="sealed " + name)
        sealed_artifact_bytes[name] = artifact_raw

    fixture_hash = FROZEN_ARTIFACTS["fixture"][1]
    freeze_hash = FROZEN_ARTIFACTS["freeze"][1]
    procedure_hash = FROZEN_ARTIFACTS["family_procedure"][1]
    execution = seal.get("execution", {})
    coverage = seal.get("execution_coverage", {})
    provenance = seal.get("provenance", {})
    expected_keys = [entry["native_key"] for entry in entries]
    expected_by_key = {_canonical_native_key(key) for key in expected_keys}
    sealed_keys = coverage.get("expected_keys") if isinstance(coverage, dict) else None
    if not isinstance(sealed_keys, list) or len(sealed_keys) != 84:
        _fail("native seal expected-key inventory is malformed")
    sealed_by_key = {_canonical_native_key(key) for key in sealed_keys}
    if (
        len(sealed_by_key) != 84
        or sealed_by_key != expected_by_key
        or execution.get("status") != "success"
        or execution.get("exit_code") != 0
        or execution.get("cleanup", {}).get("status") != "confirmed_stopped"
        or coverage.get("status") != "accepted"
        or coverage.get("record_count") != 84
        or coverage.get("key_fields") != list(_NATIVE_KEY_FIELD_ORDER)
        or coverage.get("record_key_path") != ["key"]
        or any(
            coverage.get(name)
            for name in (
                "malformed_lines",
                "partial_final_line",
                "reasons",
                "duplicate_keys",
                "duplicate_expected_keys",
                "unknown_keys",
                "unresolved_keys",
                "not_generated_keys",
            )
        )
        or provenance.get("family") != "A2"
        or provenance.get("production_imported") is not False
        or provenance.get("purpose") != "formal-science"
    ):
        _fail("native A2 execution, cleanup, coverage, or provenance seal differs")

    authorization_raw = sealed_artifact_bytes["authorization.json"]
    authorization = _strict_json(authorization_raw)
    if not isinstance(authorization, dict):
        _fail("native A2 authorization is not a JSON object")
    if (
        authorization.get("family") != "A2"
        or authorization.get("authorized_by") != "root"
        or authorization.get("formal_execution") != "AUTHORIZED"
        or authorization.get("E_commit") != provenance.get("E_commit")
        or authorization.get("E_commit") != E_COMMIT
        or authorization.get("E_tree") != E_TREE
        or authorization.get("F_sha256") != freeze_hash
        or authorization.get("F_sha256") != provenance.get("F_sha256")
        or authorization.get("fixture_sha256") != fixture_hash
        or authorization.get("fixture_sha256") != provenance.get("fixture_sha256")
        or authorization.get("procedure_sha256") != procedure_hash
        or hashes.get("records.raw.jsonl") != RECORDS_SHA256
        or hashes.get("records.raw.jsonl") != _sha256(records_raw)
        or hashes.get("stdout.jsonl") != _sha256(records_raw)
        or hashes.get("family-freeze.json") != freeze_hash
        or hashes.get("policy.json") != fixture_hash
    ):
        _fail("native A2 authorization or frozen artifact identity differs")
    family_freeze_raw = _native_bytes(authorization.get("family_freeze"), label="native family freeze")
    if _sha256(family_freeze_raw) != freeze_hash:
        _fail("native authorization family-freeze bytes differ")

    if (root / "records.raw.jsonl") != Path(RECORDS_PATH):
        _fail("native records path differs from the retained envelope")
    environment_pin = authorization.get("reference_runtime_inventory_pin")
    environment_raw = _native_bytes(
        environment_pin,
        label="native runtime inventory",
        read_only=True,
    )
    if (
        not isinstance(environment_pin, dict)
        or set(environment_pin) != {"path", "sha256"}
        or environment_pin.get("sha256") != "2a4ec38d1fc3ece3c1abcabf0a943bb233782b338f854c308e0459f4d55634d8"
        or hashes.get("records.raw.jsonl") != RECORDS_SHA256
    ):
        _fail("native runtime inventory pin differs")
    environment = _strict_json(environment_raw)
    if (
        not isinstance(environment, dict)
        or environment.get("production_modules") not in (None, [])
        or not isinstance(environment.get("loaded_modules"), list)
        or any(
            not isinstance(name, str) or name == "gragra" or name.startswith("gragra.")
            for name in environment["loaded_modules"]
        )
    ):
        _fail("native reference runtime imported a production module")

    expected_source_pins = frozen["family_procedure"].get("record_source_pins")
    fixture_source_pins = frozen["fixture"].get("source_pins")
    procedure_source_pins = frozen["family_procedure"].get("source_pins")
    source_artifacts = authorization.get("source_artifacts")
    if (
        not isinstance(expected_source_pins, list)
        or len(expected_source_pins) != 1
        or expected_source_pins != fixture_source_pins
        or expected_source_pins != procedure_source_pins
        or not isinstance(source_artifacts, list)
        or len(source_artifacts) != 1
    ):
        _fail("frozen native source citation pins differ")
    source_pin = expected_source_pins[0]
    source_pin_path = source_artifacts[0].get("path") if isinstance(source_artifacts[0], dict) else None
    source_pin_hash = source_artifacts[0].get("sha256") if isinstance(source_artifacts[0], dict) else None
    if (
        not isinstance(source_pin, dict)
        or set(source_pin) != {"equations", "pages", "sha256"}
        or source_pin.get("equations") != ["131", "132", "160"]
        or source_pin.get("pages") != [69, 70, 82]
        or source_pin.get("sha256") != source_pin_hash
    ):
        _fail("native source citation does not match the frozen source pin")
    source_raw = _read_regular(source_pin_path, label="native source artifact")
    _check_sha(source_raw, source_pin_hash, label="native source artifact")

    # Verify the actual managed terminal result and the ledger witness exactly
    # as the native A2 adapter did, without opening its controller path.
    result = _strict_json(sealed_artifact_bytes["resource-result.json"])
    ledger = _strict_json(sealed_artifact_bytes["resource-ledger.json"])
    if not isinstance(result, dict) or not isinstance(ledger, dict):
        _fail("native A2 terminal result/ledger is malformed")
    ledger_entries = ledger.get("entries")
    if not isinstance(ledger_entries, list):
        _fail("native A2 ledger entries are missing")
    matches = [item for item in ledger_entries if isinstance(item, dict) and item.get("id") == result.get("ledger_id")]
    if len(matches) != 1:
        _fail("native A2 terminal ledger identity is missing or duplicated")
    entry = matches[0]
    command = entry.get("command")
    if (
        result.get("status") != "success"
        or result.get("exit_code") != 0
        or result.get("cleanup_confirmed") is not True
        or result.get("launch_state") != "started"
        or entry.get("kind") != "formal"
        or entry.get("status") != "success"
        or entry.get("exit_code") != 0
        or entry.get("cleanup_confirmed") is not True
        or type(entry.get("pid")) is not int
        or entry["pid"] <= 0
        or type(entry.get("elapsed_seconds")) not in (int, float)
        or not 0 <= entry["elapsed_seconds"] <= 3600
        or not isinstance(command, list)
        or len(command) < 2
        or command[-2] != "--authorization"
        or not isinstance(command[-1], str)
    ):
        _fail("native A2 managed terminal result or command differs")
    if command[-1] != str(root / "authorization.json"):
        _fail("managed authorization argument path differs from the sealed native artifact")
    command_authorization = sealed_artifact_bytes["authorization.json"]
    _check_sha(command_authorization, hashes["authorization.json"], label="managed authorization argument")

    # The three original E/dependency documents and archived bundle are reached
    # only through the path pins carried by the native authorization/seal.
    source_admission = _strict_pin_document(
        authorization.get("root_source_admission"), label="root source admission"
    )
    durable = _strict_pin_document(authorization.get("durable_E"), label="durable E manifest")
    integration = _strict_pin_document(
        authorization.get("root_E_integration_admission"), label="root E integration admission"
    )
    reviewed_groups = authorization.get("reviewed_source_hashes_by_group")
    if (
        source_admission.get("decision") != "PASS_FOR_REVIEWED_REFERENCE_SOURCE_INTEGRATION"
        or integration.get("decision") != "PASS_REVIEWED_E_INTEGRATED"
        or durable.get("E_commit") != E_COMMIT
        or durable.get("E_tree") != E_TREE
        or integration.get("E_commit") != E_COMMIT
        or integration.get("E_tree") != E_TREE
        or durable.get("restored_tree") != E_TREE
        or durable.get("clean_E") is not True
        or durable.get("reviewed_source_hashes_by_group") != reviewed_groups
    ):
        _fail("native E source admission, integration, or durable tree differs")
    _native_bytes(durable.get("bundle"), label="archived durable E bundle")

    implementation_reviewers: list[str] = []
    authorization_reviews = authorization.get("reviews")
    if not isinstance(authorization_reviews, dict) or set(authorization_reviews) != {
        "science",
        "numerical",
        "provenance",
    }:
        _fail("native E implementation review roles differ")
    for role in ("science", "numerical", "provenance"):
        review = _strict_pin_document(authorization_reviews[role], label="native E " + role + " review")
        if (
            review.get("role") != role
            or review.get("decision") != "PASS"
            or review.get("scope") != "reference implementation"
            or review.get("families") != ["A2"]
            or review.get("reviewed_commit") != source_admission.get("reviewed_commit")
            or review.get("file_hashes_by_group") != reviewed_groups
            or not isinstance(review.get("reviewer_id"), str)
            or not review["reviewer_id"]
            or not isinstance(review.get("original_reviews"), list)
        ):
            _fail("native E implementation review subject differs")
        for original in review["original_reviews"]:
            _native_bytes(original, label="original E " + role + " review")
        implementation_reviewers.append(review["reviewer_id"])
    if len(set(implementation_reviewers)) != 3:
        _fail("native E source reviewers are not distinct")

    pins = {
        "family": "A2",
        "E_commit": E_COMMIT,
        "E_tree": E_TREE,
        "records_sha256": RECORDS_SHA256,
        "seal_sha256": SEAL_SHA256,
        "environment_sha256": environment_pin["sha256"],
        "expected_keys_sha256": _sha256(
            _canonical_bytes(
                [
                    *frozen["fixture"]["expected_keys"],
                    *frozen["fixture"]["expected_scalar_keys"],
                ]
            )
        ),
        "frozen_artifacts_sha256": {
            name: FROZEN_ARTIFACTS[name][1]
            for name in ("freeze", "fixture", "budget", "family_procedure")
        },
        "source_pin": {
            "head": E_COMMIT,
            "tree": E_TREE,
            "files_sha256": authorization.get("file_hashes_by_group"),
        },
        "artifact_hashes": hashes,
    }
    return seal, authorization, pins, environment_pin["sha256"]


def _verify_admission(
    admission: dict[str, Any],
    pins: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        set(admission) != {"schema", "family", "decision", "pins", "reviews"}
        or admission.get("schema") != ADMISSION_SCHEMA
        or admission.get("family") != "A2"
        or admission.get("decision") != "PASS"
        or admission.get("pins") != pins
    ):
        _fail("pinned native reference admission does not match derived evidence")
    reviews = admission.get("reviews")
    if not isinstance(reviews, dict) or set(reviews) != {"science", "numerical", "provenance"}:
        _fail("reference admission review roles differ")

    reviewer_ids: set[str] = set()
    review_documents: dict[str, dict[str, Any]] = {}
    original_documents: dict[str, dict[str, Any]] = {}
    for role in ("science", "numerical", "provenance"):
        review = _strict_pin_document(reviews[role], label="retained " + role + " companion")
        if (
            set(review)
            != {"schema", "family", "role", "reviewer_id", "decision", "pins", "original_review"}
            or review.get("schema") != REVIEW_SCHEMA
            or review.get("family") != "A2"
            or review.get("role") != role
            or review.get("decision") != "PASS"
            or review.get("pins") != pins
            or not isinstance(review.get("reviewer_id"), str)
            or not review["reviewer_id"]
            or review["reviewer_id"] in reviewer_ids
        ):
            _fail("retained reference review role, reviewer, or exact subject differs")
        reviewer_ids.add(review["reviewer_id"])
        original_raw = _native_bytes(review.get("original_review"), label="original " + role + " report")
        original = _strict_json(original_raw)
        if not isinstance(original, dict):
            _fail("original " + role + " review is not a JSON report")
        review_documents[role] = review
        original_documents[role] = original

    if len(reviewer_ids) != 3:
        _fail("retained reference reviewers are not distinct")
    science = original_documents["science"]
    if (
        science.get("schema") != SCIENCE_REPORT_SCHEMA
        or science.get("family") != "A2"
        or science.get("role") != "science"
        or science.get("decision") != "PASS"
        or science.get("pins") != pins
        or science.get("reviewer_id") != review_documents["science"].get("reviewer_id")
    ):
        _fail("original science review does not pin the exact retained reference")
    audit_pin = science.get("retained_record_audit")
    if (
        not isinstance(audit_pin, dict)
        or set(audit_pin) != {"kind", "path", "sha256"}
        or audit_pin.get("kind") != SCIENCE_AUDIT_KIND
        or audit_pin.get("sha256") != AUDIT_SHA256
    ):
        _fail("science report retained-record audit pin differs")
    audit = _strict_json(_native_bytes(audit_pin, label="science retained-record audit"))
    if not isinstance(audit, dict):
        _fail("science retained-record audit is not a JSON object")
    if (
        audit.get("reviewer_id") != science.get("reviewer_id")
        or audit.get("role") != "science"
        or audit.get("scope")
        != "Retained reference arithmetic and scientific normalization audit only, no new angular integral or production call"
    ):
        _fail("science retained-record audit subject differs from its pinned report")
    return audit, science


def inspect_retained(manifest: dict[str, Any]) -> dict[str, Any]:
    """Read, bind, and independently check the one authorized retained A2 R.

    ``manifest.retained_reference`` uses the schema documented in the A-retained
    support handoff.  This function is read-only and returns zero science/old
    credit even when all retained checks pass.
    """
    _reject_production_modules()
    try:
        if not isinstance(manifest, dict):
            _fail("manifest must be an object")
        retained = manifest.get("retained_reference")
        expected_fields = {
            "schema",
            "attempt_root",
            "records",
            "seal",
            "admission",
            "frozen_artifacts",
        }
        if (
            not isinstance(retained, dict)
            or set(retained) != expected_fields
            or retained.get("schema") != RETAINED_SCHEMA
        ):
            _fail("manifest.retained_reference schema or field set differs")
        root = _require_readonly_root(retained.get("attempt_root"))
        records_raw = _manifest_pin(
            retained.get("records"),
            path=RECORDS_PATH,
            sha256=RECORDS_SHA256,
            label="retained records",
        )
        seal_raw = _manifest_pin(
            retained.get("seal"),
            path=SEAL_PATH,
            sha256=SEAL_SHA256,
            label="retained seal",
            read_only=True,
        )
        admission_raw = _manifest_pin(
            retained.get("admission"),
            path=ADMISSION_PATH,
            sha256=ADMISSION_SHA256,
            label="retained admission",
        )
        if root / "records.raw.jsonl" != Path(RECORDS_PATH) or root / "seal.json" != Path(SEAL_PATH):
            _fail("retained records and seal are not direct native attempt artifacts")

        frozen_raw, frozen = _load_frozen(manifest, retained)
        # Import only the same independent, stdlib-only v2 coverage mapper used
        # by the diagnostic contract.  No v1 adapter or production module is
        # part of this import closure.
        from .coverage import CoverageError, build_map

        try:
            fixture_map = build_map(frozen_raw["fixture"])
        except (CoverageError, KeyError, TypeError, ValueError) as exc:
            raise RetainedReferenceError("frozen A2 coverage map is invalid") from exc
        entries = fixture_map.get("entries")
        if not isinstance(entries, list) or len(entries) != 84:
            _fail("independent frozen coverage map is not the complete 84-key map")

        # Parse the same captured bytes already checked against their pins.
        # Reopening a path here would weaken the snapshot binding.
        seal, authorization, pins, environment_sha256 = _verify_native_attempt(
            root, records_raw, seal_raw, frozen, entries
        )
        admission = _strict_json(admission_raw)
        if not isinstance(admission, dict):
            _fail("retained admission artifact is not an object")
        audit, _science_report = _verify_admission(admission, pins)
        source_pins = frozen["family_procedure"].get("record_source_pins")
        record_summary = _verify_records(records_raw, entries, source_pins)
        historical_audit_validation = validate_adequacy_audit(
            audit,
            entries,
            records_sha256=RECORDS_SHA256,
            seal_sha256=SEAL_SHA256,
            procedure_sha256=FROZEN_ARTIFACTS["family_procedure"][1],
        )
        if (
            seal.get("provenance", {}).get("E_commit") != E_COMMIT
            or authorization.get("E_tree") != E_TREE
            or environment_sha256 != "2a4ec38d1fc3ece3c1abcabf0a943bb233782b338f854c308e0459f4d55634d8"
        ):
            _fail("final retained source/environment pins changed")
        _reject_production_modules()
        return {
            "schema": "gragra.a2.v2.retained-reference-result.v1",
            "decision": "PASS",
            "scope": "retained_reference_validation_only",
            "family": "A2",
            "E_commit": E_COMMIT,
            "E_tree": E_TREE,
            "records_sha256": RECORDS_SHA256,
            "seal_sha256": SEAL_SHA256,
            "procedure_sha256": FROZEN_ARTIFACTS["family_procedure"][1],
            "coverage": record_summary,
            "historical_audit_validation": historical_audit_validation,
            "science_acceptance": False,
            "science_credit": 0,
            "old_credit": 0,
            "old_credit_seconds": 0,
        }
    finally:
        _reject_production_modules()
