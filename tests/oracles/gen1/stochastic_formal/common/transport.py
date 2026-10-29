"""Actual-byte bindings and reviewed phase authorization, without production imports.

These helpers do not establish physics or change any frozen acceptance rule.
A caller must additionally apply its family's frozen reference/admission contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

AUTH_SCHEMA = "gen1-stochastic-authorization-v2"
REVIEW_SCHEMA = "gen1-stochastic-review-v1"
RECEIPT_SCHEMA = "gen1-stochastic-bound-receipts-v1"
SEAL_SCHEMA = "gen1-stochastic-execution-seal-v1"
KEY_FIELDS = (
    "family",
    "fixture_id",
    "frequency_hz",
    "target_id",
    "observable",
    "representation",
    "set",
)
CONNECTION_GATES = (
    "units",
    "shape",
    "frequency_axis",
    "input_normalization",
    "polarization",
    "spatial_phase",
    "frame",
    "source_applicability",
    "reference_resolved",
    "numeric_inequalities",
    "precision_crosscheck",
)
HEX = re.compile(r"[0-9a-f]{64}\Z")


class GateError(ValueError):
    """An admission condition was absent, stale, ambiguous, or false."""


def canonical_bytes(value: Any) -> bytes:
    def check(item):
        if isinstance(item, float):
            raise GateError("scientific JSON floats are forbidden")
        if isinstance(item, dict):
            if any(not isinstance(k, str) for k in item):
                raise GateError("string object keys required")
            for v in item.values():
                check(v)
        elif isinstance(item, (list, tuple)):
            for v in item:
                check(v)

    check(value)
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def strict_json(data: bytes) -> Any:
    def pairs(items):
        obj = {}
        for k, v in items:
            if k in obj:
                raise GateError(f"duplicate JSON field: {k}")
            obj[k] = v
        return obj

    try:
        value = json.loads(
            data,
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(GateError("nonfinite JSON")),
        )
        canonical_bytes(value)
        return value
    except (UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise GateError("invalid JSON") from exc


def read_json(path: str | Path) -> Any:
    return strict_json(Path(path).read_bytes())


def artifact_pin(path: str | Path, kind: str) -> dict[str, str]:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise GateError(f"regular artifact missing: {path}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "kind": kind,
    }


def verify_artifact(pin: dict[str, str]) -> bytes:
    if not isinstance(pin, dict) or set(pin) != {"path", "sha256", "kind"}:
        raise GateError("artifact pin schema mismatch")
    if not isinstance(pin["sha256"], str) or not HEX.fullmatch(pin["sha256"]):
        raise GateError("invalid artifact SHA256")
    if not isinstance(pin["kind"], str) or not pin["kind"]:
        raise GateError("artifact kind missing")
    path = Path(pin["path"])
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise GateError("artifact path is not a regular absolute file")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != pin["sha256"]:
        raise GateError("artifact bytes/hash mismatch")
    return data


def require_external_directory(path: str | Path) -> None:
    """Reject every Git working/bare tree before allocating formal evidence."""
    parent = Path(path).resolve()
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir():
        parent = parent.parent
    environment = dict(os.environ)
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR"):
        environment.pop(name, None)
    result = subprocess.run(
        ["git", "-C", str(parent), "rev-parse", "--absolute-git-dir"],
        capture_output=True,
        check=False,
        env=environment,
    )
    if result.returncode == 0:
        raise GateError("formal evidence must be outside every Git repository")


def key_id(key: dict[str, str]) -> tuple[str, ...]:
    if not isinstance(key, dict) or set(key) != set(KEY_FIELDS):
        raise GateError("key schema mismatch")
    if any(not isinstance(key[k], str) or not key[k] for k in KEY_FIELDS):
        raise GateError("empty key field")
    return tuple(key[k] for k in KEY_FIELDS)


def exact_keys(keys: list[dict[str, str]]) -> set[tuple[str, ...]]:
    if not isinstance(keys, list) or not keys:
        raise GateError("expected keys missing")
    result = [key_id(k) for k in keys]
    if len(result) != len(set(result)):
        raise GateError("duplicate key")
    return set(result)


def bind_receipts(expected_rows: list[dict], pins: dict) -> dict:
    """Bind trusted endpoint/mapping declarations, with no future PASS claim."""
    rows = []
    for expected in expected_rows:
        row = strict_json(canonical_bytes(expected))
        verify_artifact(row["source"])
        verify_artifact(row["destination"])
        row["status"] = "BOUND"
        row["content_sha256"] = content_hash(row)
        rows.append(row)
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "BOUND",
        "pins": strict_json(canonical_bytes(pins)),
        "receipts": rows,
    }


def validate_bound_receipts(
    bundle: dict, expected_rows: list[dict], science_keys: list[dict], pins: dict
) -> dict:
    if (
        set(bundle) != {"schema", "status", "pins", "receipts"}
        or bundle["schema"] != RECEIPT_SCHEMA
        or bundle["status"] != "BOUND"
        or bundle["pins"] != pins
    ):
        raise GateError("BOUND receipt envelope/pin mismatch")
    if len(expected_rows) != 70 or len(bundle["receipts"]) != 70:
        raise GateError("70 receipt keys required")
    expected = {key_id(row["key"]): row for row in expected_rows}
    if len(expected) != 70:
        raise GateError("duplicate expected receipt key")
    if sum(r["kind"] == "independent_reference" for r in expected_rows) != 56:
        raise GateError("56 independent bindings required")
    if sum(r["kind"] == "production_consistency" for r in expected_rows) != 14:
        raise GateError("14 consistency bindings required")
    science = exact_keys(science_keys)
    if len(science) != 30:
        raise GateError("30 science dependencies required")
    seen = set()
    ids = set()
    dependencies = set()
    for row in bundle["receipts"]:
        rid = key_id(row["key"])
        if rid in seen or rid not in expected:
            raise GateError("duplicate/unknown receipt key")
        seen.add(rid)
        if row.get("receipt_id") in ids:
            raise GateError("duplicate receipt ID")
        ids.add(row.get("receipt_id"))
        if row.get("status") != "BOUND":
            raise GateError("binding cannot claim execution PASS")
        if set(row) != set(expected[rid]) | {"status", "content_sha256"}:
            raise GateError("receipt fields differ from trusted schema")
        declaration = {
            k: v for k, v in row.items() if k not in {"status", "content_sha256"}
        }
        if declaration != expected[rid]:
            raise GateError("receipt endpoints/mapping/rule mismatch")
        if row["content_sha256"] != content_hash(
            {k: v for k, v in row.items() if k != "content_sha256"}
        ):
            raise GateError("receipt content hash mismatch")
        verify_artifact(row["source"])
        verify_artifact(row["destination"])
        mapped = exact_keys(row["science_keys"])
        if not mapped <= science:
            raise GateError("unknown science dependency")
        dependencies |= mapped
    if seen != set(expected) or dependencies != science:
        raise GateError("missing receipt/dependency coverage")
    return {
        "status": "BOUND",
        "independent": 56,
        "consistency": 14,
        "science_dependencies": 30,
    }


def verify_authorization(
    pin: dict, expected_pins: dict, phase: str, expected_reviews: dict[str, str]
) -> dict:
    """Verify externally trusted three-review hash chain for exactly one phase."""
    if phase not in {"connection", "comparison"}:
        raise GateError("unknown authorization phase")
    binding = strict_json(verify_artifact(pin))
    fields = {"schema", "family", "phase", "status", "pins", "candidate", "reviews"}
    if (
        set(binding) != fields
        or binding["schema"] != AUTH_SCHEMA
        or binding["status"] != "AUTHORIZED"
        or binding["phase"] != phase
        or binding["pins"] != expected_pins
        or binding["family"] != expected_pins.get("family")
    ):
        raise GateError("authorization phase/status/pin mismatch")
    roles = {"science", "numerical", "provenance"}
    if set(binding["reviews"]) != roles or set(expected_reviews) != roles:
        raise GateError("three independent reviews required")
    candidate = strict_json(verify_artifact(binding["candidate"]))
    if (
        set(candidate) != {"schema", "family", "phase", "status", "pins", "evidence"}
        or any(
            candidate[k] != binding[k] for k in ("schema", "family", "phase", "pins")
        )
        or candidate["status"] != "REVIEW_CANDIDATE"
        or not isinstance(candidate["evidence"], dict)
        or not candidate["evidence"]
    ):
        raise GateError("authorization candidate mismatch")
    for artifact in candidate["evidence"].values():
        verify_artifact(artifact)
    reviewers = set()
    for role, review_pin in binding["reviews"].items():
        if review_pin["sha256"] != expected_reviews[role]:
            raise GateError("review differs from external trusted digest")
        review = strict_json(verify_artifact(review_pin))
        if (
            set(review)
            != {"schema", "role", "reviewer_id", "decision", "candidate_sha256", "pins"}
            or review["schema"] != REVIEW_SCHEMA
            or review["role"] != role
            or review["decision"] != "PASS"
            or review["candidate_sha256"] != binding["candidate"]["sha256"]
            or review["pins"] != expected_pins
            or not isinstance(review["reviewer_id"], str)
            or not review["reviewer_id"]
            or review["reviewer_id"] in reviewers
        ):
            raise GateError("independent review decision/pin mismatch")
        reviewers.add(review["reviewer_id"])
    return binding


def _write_new(path: Path, data: bytes) -> None:
    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class SealedAttempt:
    """Append durable records and seal every outcome in a fresh external directory."""

    def __init__(
        self,
        root: str | Path,
        *,
        family: str,
        phase: str,
        pins: dict,
        expected_keys: list[dict],
    ):
        self.root = Path(root)
        if self.root.is_symlink() or not self.root.is_dir() or any(self.root.iterdir()):
            raise GateError("fresh empty attempt directory required")
        self.expected = exact_keys(expected_keys)
        self.rows = []
        self.pins = strict_json(canonical_bytes(pins))
        self.family = family
        self.phase = phase
        _write_new(
            self.root / "launch.json",
            canonical_bytes(
                {
                    "schema": SEAL_SCHEMA,
                    "family": family,
                    "phase": phase,
                    "pins": self.pins,
                    "expected_keys": expected_keys,
                }
            ),
        )
        _write_new(self.root / "records.jsonl", b"")

    def __enter__(self):
        return self

    def append(self, record: dict) -> None:
        rid = key_id(record["key"])
        if rid not in self.expected or any(key_id(r["key"]) == rid for r in self.rows):
            raise GateError("unknown/duplicate attempt record")
        data = canonical_bytes(record)
        fd = os.open(
            self.root / "records.jsonl",
            os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
        )
        with os.fdopen(fd, "ab") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        self.rows.append(record)

    def __exit__(self, exc_type, exc, traceback):
        covered = {key_id(r["key"]) for r in self.rows}
        passed = (
            exc is None
            and covered == self.expected
            and all(
                r.get("status") == "resolved" and r.get("passed") is True
                for r in self.rows
            )
        )
        seal = {
            "schema": SEAL_SCHEMA,
            "family": self.family,
            "phase": self.phase,
            "pins": self.pins,
            "decision": "PASS"
            if passed
            else "FAILED_EXECUTION"
            if exc
            else "FAIL_OR_UNRESOLVED",
            "exit_code": 0 if exc is None else 1,
            "expected_keys": read_json(self.root / "launch.json")["expected_keys"],
            "completed": len(self.rows),
            "exception": None
            if exc is None
            else {"type": exc_type.__name__, "message": str(exc)},
            "artifacts": {
                "launch": artifact_pin(self.root / "launch.json", "launch"),
                "records": artifact_pin(self.root / "records.jsonl", "records"),
            },
        }
        runtime = self.root / "runtime-dependencies.json"
        if runtime.exists():
            seal["artifacts"]["runtime_dependencies"] = artifact_pin(
                runtime, "runtime_dependencies"
            )
        _write_new(self.root / "seal.json", canonical_bytes(seal))
        for path in self.root.iterdir():
            if path.is_file():
                path.chmod(0o444)
        self.root.chmod(0o555)
        return False


def validate_connection_pass(
    pin: dict, expected_keys: list[dict], expected_pins: dict
) -> dict:
    """A sealed result, never a BOUND receipt, supplies the 70 execution veto."""
    seal = strict_json(verify_artifact(pin))
    if (
        seal.get("schema") != SEAL_SCHEMA
        or seal.get("family") != "B1"
        or seal.get("phase") != "connection"
        or seal.get("decision") != "PASS"
        or seal.get("exit_code") != 0
        or seal.get("pins") != expected_pins
    ):
        raise GateError("connection seal decision/pin mismatch")
    runtime_pin = seal.get("artifacts", {}).get("runtime_dependencies")
    if runtime_pin is None:
        raise GateError("connection seal lacks mandatory runtime dependency evidence")
    runtime = strict_json(verify_artifact(runtime_pin))
    if (
        runtime.get("status") != "MATCH"
        or runtime.get("planned_sha256")
        != expected_pins.get("dependency_closure_sha256")
        or not any(
            p.get("name", "").startswith("gragra")
            for p in runtime.get("observed_modules", [])
        )
    ):
        raise GateError("connection runtime snapshot does not bind actual P closure")
    expected = exact_keys(expected_keys)
    if (
        len(expected) != 70
        or exact_keys(seal["expected_keys"]) != expected
        or seal.get("completed") != 70
    ):
        raise GateError("connection seal exact coverage mismatch")
    launch = strict_json(verify_artifact(seal["artifacts"]["launch"]))
    if (
        launch.get("pins") != expected_pins
        or exact_keys(launch["expected_keys"]) != expected
    ):
        raise GateError("connection launch mismatch")
    raw = verify_artifact(seal["artifacts"]["records"])
    rows = [strict_json(line) for line in raw.splitlines()]
    if len(rows) != 70 or exact_keys([r["key"] for r in rows]) != expected:
        raise GateError("connection records exact coverage mismatch")
    for row in rows:
        if (
            row.get("status") != "resolved"
            or row.get("passed") is not True
            or not isinstance(row.get("gate_results"), dict)
            or set(row["gate_results"]) != set(CONNECTION_GATES)
            or any(v is not True for v in row["gate_results"].values())
        ):
            raise GateError("connection assessment lacks complete 11-gate PASS")
    return {"decision": "PASS", "resolved": 70, "seal_sha256": pin["sha256"]}


def require_science_authority(
    pin: dict,
    pins: dict,
    reviews: dict,
    *,
    connection_seal: dict | None,
    connection_keys: list[dict],
    connection_pins: dict,
) -> dict:
    """B1 science cannot start from connection-only auth or unrevised PASS evidence."""
    if connection_seal is None:
        raise GateError(
            "reviewed 70 PASS connection seal required before science launch"
        )
    if pins.get("connection_pass_sha256") != connection_seal["sha256"]:
        raise GateError("science authority does not bind exact 70 PASS seal")
    validate_connection_pass(connection_seal, connection_keys, connection_pins)
    return verify_authorization(pin, pins, "comparison", reviews)
