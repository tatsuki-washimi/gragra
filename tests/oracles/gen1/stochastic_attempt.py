"""Strict, append-free sealing for Gen1 stochastic reference attempts.

This module knows only the key fields supplied by the frozen policy boundary.
It does not interpret scientific values or compute reference/acceptance results.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

SEAL_FILENAME = "seal.json"
RECORDS_FILENAME = "records.raw.jsonl"
EXECUTION_STATUSES = {
    "success",
    "error",
    "exception",
    "interruption",
    "timeout",
    "not_started",
}


class AdmissionError(ValueError):
    """The supplied frozen key boundary or attempt record is malformed."""


class AttemptPendingError(RuntimeError):
    """Attempt cleanup is unresolved, so its artifacts cannot be sealed."""

    def __init__(self, attempt_dir: str | Path, detail: str) -> None:
        self.attempt_dir = Path(attempt_dir)
        super().__init__(f"attempt remains pending at {self.attempt_dir}: {detail}")


class DuplicateJSONPropertyError(ValueError):
    """A JSON object repeats a property name."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize canonical JSON with compact sorted keys and UTF-8."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise DuplicateJSONPropertyError(f"duplicate JSON property: {name}")
        result[name] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


def _key_object(
    value: Any,
    fields: tuple[str, ...],
    label: str,
    *,
    allow_extra_fields: bool = False,
) -> dict[str, str]:
    if isinstance(value, Mapping):
        if allow_extra_fields:
            if not set(fields).issubset(value):
                raise AdmissionError(f"{label} is missing configured key fields")
        elif set(value) != set(fields):
            raise AdmissionError(
                f"{label} must contain exactly the configured key fields"
            )
        parts = [value[name] for name in fields]
    elif isinstance(value, (tuple, list)):
        if len(value) != len(fields):
            raise AdmissionError(f"{label} must have {len(fields)} key values")
        parts = list(value)
    else:
        raise AdmissionError(f"{label} must be a key mapping or tuple")

    if any(not isinstance(part, str) or not part for part in parts):
        raise AdmissionError(f"{label} key values must be nonempty strings")
    return dict(zip(fields, parts, strict=True))


def _key_token(value: Mapping[str, str], fields: tuple[str, ...]) -> bytes:
    return canonical_json_bytes({name: value[name] for name in fields})


def _canonical_keys(
    key_fields: Sequence[str], expected_keys: Sequence[Any]
) -> tuple[tuple[str, ...], list[dict[str, str]], dict[bytes, dict[str, str]]]:
    if isinstance(key_fields, (str, bytes)) or isinstance(expected_keys, (str, bytes)):
        raise AdmissionError(
            "key fields and expected keys must be sequences, not strings"
        )
    fields = tuple(key_fields)
    if len(fields) != 7 or any(
        not isinstance(name, str) or not name for name in fields
    ):
        raise AdmissionError(
            "the frozen key boundary must supply seven nonempty field names"
        )
    if len(set(fields)) != len(fields):
        raise AdmissionError("the frozen key boundary repeats a field name")

    normalized = [_key_object(item, fields, "expected key") for item in expected_keys]
    unique = {_key_token(item, fields): item for item in normalized}
    return fields, normalized, unique


def _canonical_record_key_path(record_key_path: Sequence[str]) -> tuple[str, ...]:
    if isinstance(record_key_path, (str, bytes)):
        raise AdmissionError("record key path must be a sequence of mapping names")
    path = tuple(record_key_path)
    if not path or any(not isinstance(part, str) or not part for part in path):
        raise AdmissionError("record key path must contain nonempty mapping names")
    return path


def validate_key_boundary(
    key_fields: Sequence[str],
    expected_keys: Sequence[Any],
    record_key_path: Sequence[str],
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """Check supplied key fields, extraction path, and keys before generation."""
    _canonical_record_key_path(record_key_path)
    fields, normalized, unique = _canonical_keys(key_fields, expected_keys)
    if not normalized:
        raise AdmissionError("frozen expected keyset must not be empty")
    if len(unique) != len(normalized):
        raise AdmissionError("frozen expected keyset contains duplicate keys")
    return fields, normalized


def _safe_artifact_path(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name:
        raise ValueError(f"unsafe attempt artifact path: {name!r}")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in name
    ):
        raise ValueError(f"unsafe attempt artifact path: {name!r}")
    if path.as_posix() == SEAL_FILENAME:
        raise ValueError("seal.json is reserved for the final manifest")
    return path


def _write_artifact(attempt_dir: Path, name: str, contents: bytes) -> None:
    relative = _safe_artifact_path(name)
    target = attempt_dir.joinpath(*relative.parts)
    parent = attempt_dir
    for part in relative.parts[:-1]:
        parent = parent / part
        if parent.is_symlink():
            raise ValueError(f"attempt artifact path traverses a symlink: {name}")
        parent.mkdir(exist_ok=True)
    if target.exists() or target.is_symlink():
        if (
            not target.is_file()
            or target.is_symlink()
            or target.read_bytes() != contents
        ):
            raise FileExistsError(
                f"attempt artifact already exists with different bytes: {name}"
            )
        return
    with target.open("xb") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())


def _read_jsonl(
    raw: bytes, fields: tuple[str, ...], record_key_path: tuple[str, ...]
) -> tuple[list[dict[str, str]], list[dict[str, Any]], bool, list[str]]:
    records: list[dict[str, str]] = []
    malformed_lines: list[dict[str, Any]] = []
    reasons: list[str] = []
    partial_final_line = bool(raw) and not raw.endswith(b"\n")
    if partial_final_line:
        reasons.append("unterminated_final_line")

    chunks = raw.split(b"\n")
    if raw.endswith(b"\n"):
        chunks.pop()
    if raw == b"":
        chunks = []

    for line_number, line in enumerate(chunks, start=1):
        if not line:
            malformed_lines.append({"line": line_number, "reason": "blank_line"})
            continue
        try:
            parsed = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=_pairs_without_duplicates,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            DuplicateJSONPropertyError,
            ValueError,
        ) as exc:
            malformed_lines.append(
                {"line": line_number, "reason": f"{type(exc).__name__}: {exc}"}
            )
            continue

        if not isinstance(parsed, dict):
            malformed_lines.append(
                {"line": line_number, "reason": "record_is_not_an_object"}
            )
            continue
        try:
            key_source: Any = parsed
            for part in record_key_path:
                if not isinstance(key_source, Mapping) or part not in key_source:
                    raise AdmissionError(
                        "record does not match the configured key extraction path"
                    )
                key_source = key_source[part]
            if not isinstance(key_source, Mapping):
                raise AdmissionError(
                    "record key at configured path must be an object"
                )
            key = _key_object(
                key_source,
                fields,
                f"record at line {line_number}",
            )
        except AdmissionError as exc:
            malformed_lines.append({"line": line_number, "reason": str(exc)})
            continue

        records.append(key)
        try:
            canonical = canonical_json_bytes(parsed)
        except (TypeError, ValueError) as exc:
            reasons.append("noncanonical_jsonl")
            malformed_lines.append(
                {"line": line_number, "reason": f"not_json_serializable: {exc}"}
            )
            continue
        if canonical != line:
            reasons.append("noncanonical_jsonl")

    if malformed_lines:
        reasons.append("malformed_jsonl")
    return records, malformed_lines, partial_final_line, reasons


def _artifact_digests(attempt_dir: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    for path in sorted(attempt_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(
                f"symbolic link is not an admissible attempt artifact: {path}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"non-regular attempt artifact: {path}")
        relative = path.relative_to(attempt_dir).as_posix()
        if relative == SEAL_FILENAME:
            continue
        digests[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(digests.items()))


def _publish_manifest(path: Path, manifest_bytes: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".seal-", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        with suppress(FileNotFoundError):
            temporary.unlink()
        raise


def _make_readonly(attempt_dir: Path) -> None:
    for path in sorted(attempt_dir.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    attempt_dir.chmod(0o555)


def seal_attempt(
    attempt_dir: str | Path,
    *,
    key_fields: Sequence[str],
    expected_keys: Sequence[Any],
    record_key_path: Sequence[str],
    records_bytes: bytes | None,
    execution: Mapping[str, Any],
    artifacts: Mapping[str, bytes],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal one attempt once, preserving every artifact byte and admission defect.

    The caller must allocate a fresh attempt directory before invoking this
    function. Existing artifact paths may be supplied again only with identical
    bytes; finalization never overwrites them.
    """
    root = Path(attempt_dir)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("attempt directory must already exist and be a real directory")
    manifest_path = root / SEAL_FILENAME
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(f"attempt is already sealed: {root}")

    record_path = _canonical_record_key_path(record_key_path)
    fields, expected, expected_by_token = _canonical_keys(key_fields, expected_keys)
    if records_bytes is not None:
        if not isinstance(records_bytes, bytes):
            raise TypeError("records_bytes must be bytes")
        _write_artifact(root, RECORDS_FILENAME, records_bytes)
    elif not (root / RECORDS_FILENAME).is_file():
        _write_artifact(root, RECORDS_FILENAME, b"")

    for name, contents in artifacts.items():
        if not isinstance(contents, bytes):
            raise TypeError(f"artifact {name!r} must be bytes")
        _write_artifact(root, name, contents)

    raw = (root / RECORDS_FILENAME).read_bytes()
    parsed_keys, malformed_lines, partial_final_line, reasons = _read_jsonl(
        raw, fields, record_path
    )
    expected_counts = Counter(_key_token(item, fields) for item in expected)
    observed_counts = Counter(_key_token(item, fields) for item in parsed_keys)
    duplicate_expected = [
        expected_by_token[token]
        for token, count in expected_counts.items()
        if count != 1
    ]
    observed_by_token = {_key_token(item, fields): item for item in parsed_keys}
    duplicate_keys = [
        {"key": observed_by_token[token], "count": count}
        for token, count in observed_counts.items()
        if count > 1
    ]
    expected_tokens = set(expected_by_token)
    unknown_keys = [
        {"key": observed_by_token[token], "count": count}
        for token, count in observed_counts.items()
        if token not in expected_tokens
    ]
    generated = [
        expected_by_token[token]
        for token in expected_tokens
        if observed_counts[token] == 1
    ]
    not_generated = [
        expected_by_token[token]
        for token in expected_tokens
        if observed_counts[token] == 0
    ]
    unresolved = [
        expected_by_token[token]
        for token in expected_tokens
        if observed_counts[token] != 1
    ]

    def key_sort(item: Mapping[str, str]) -> bytes:
        return _key_token(item, fields)

    expected = sorted(expected, key=key_sort)
    generated = sorted(generated, key=key_sort)
    not_generated = sorted(not_generated, key=key_sort)
    unresolved = sorted(unresolved, key=key_sort)
    duplicate_expected = sorted(duplicate_expected, key=key_sort)
    duplicate_keys.sort(key=lambda item: _key_token(item["key"], fields))
    unknown_keys.sort(key=lambda item: _key_token(item["key"], fields))

    if duplicate_expected:
        reasons.append("duplicate_expected_keys")
    if not expected:
        reasons.append("empty_expected_keys")
    if duplicate_keys:
        reasons.append("duplicate_keys")
    if unknown_keys:
        reasons.append("unknown_keys")
    if not_generated:
        reasons.append("missing_expected_keys")

    execution_record = dict(execution)
    if execution_record.get("launch_state") == "started":
        cleanup = execution_record.get("cleanup")
        if not isinstance(cleanup, Mapping) or cleanup.get("status") != "confirmed_stopped":
            raise AttemptPendingError(root, "child or writer process remains active")
    status = execution_record.get("status")
    exit_code = execution_record.get("exit_code")
    if status not in EXECUTION_STATUSES:
        raise ValueError(f"unknown execution status: {status!r}")
    if exit_code is not None and (
        isinstance(exit_code, bool) or not isinstance(exit_code, int)
    ):
        raise TypeError("execution exit_code must be an integer or null")
    execution_record["exit_code_status"] = (
        "generated" if exit_code is not None else "not_generated"
    )
    if isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code < 0:
        execution_record["termination"] = {
            "kind": "signal",
            "signal_number": -exit_code,
        }
        reasons.append("process_terminated_by_signal")
    if status != "success":
        reasons.append(f"execution_{status}")
    if exit_code != 0:
        reasons.append("nonzero_or_missing_exitcode")

    reasons = sorted(set(reasons))
    execution_coverage = {
        "status": "accepted" if not reasons else "rejected",
        "scope": "process_exit_and_exact_key_coverage_only",
        "reasons": reasons,
        "key_fields": list(fields),
        "record_key_path": list(record_path),
        "expected_keys": expected,
        "generated_keys": generated,
        "not_generated_keys": not_generated,
        "unresolved_keys": unresolved,
        "duplicate_expected_keys": duplicate_expected,
        "duplicate_keys": duplicate_keys,
        "unknown_keys": unknown_keys,
        "record_count": len(parsed_keys),
        "malformed_lines": malformed_lines,
        "partial_final_line": partial_final_line,
    }

    # Validate metadata serializability before publishing the one-way seal.
    canonical_json_bytes(execution_record)
    canonical_json_bytes(provenance)
    manifest = {
        "format": "gen1-stochastic-attempt-seal-v2",
        "attempt_id": root.name,
        "sealed_at_utc": datetime.now(UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z"),
        "execution": execution_record,
        "execution_coverage": execution_coverage,
        "scientific_assessment": {"status": "not_assessed"},
        "formal_acceptance": {"status": "not_established"},
        "provenance": dict(provenance),
        "artifacts_sha256": _artifact_digests(root),
    }
    manifest_bytes = canonical_json_bytes(manifest) + b"\n"
    _publish_manifest(manifest_path, manifest_bytes)
    _make_readonly(root)
    return manifest
