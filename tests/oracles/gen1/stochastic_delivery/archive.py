"""Exact-byte content archive; original sealed JSON paths are never rewritten."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


class EvidenceError(ValueError):
    """Missing, unknown or inconsistent delivery evidence."""


def validate_pin(pin: dict) -> None:
    if (
        not isinstance(pin, dict)
        or set(pin) != {"path", "sha256", "kind"}
        or not isinstance(pin["path"], str)
        or not Path(pin["path"]).is_absolute()
        or not isinstance(pin["kind"], str)
        or not pin["kind"]
        or not isinstance(pin["sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", pin["sha256"])
    ):
        raise EvidenceError("invalid raw artifact pin")


def _read(path: Path, expected_hash: str) -> bytes:
    if path.is_symlink() or path.resolve() != path.absolute():
        raise EvidenceError("symlink/aliased evidence path")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise EvidenceError("missing evidence bytes") from error
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise EvidenceError("evidence raw hash differs")
    return data


def decode_json(data: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise EvidenceError("duplicate JSON key")
            result[key] = value
        return result

    def forbidden(_value):
        raise EvidenceError("nonfinite or floating scientific JSON token")

    try:
        return json.loads(
            data,
            object_pairs_hook=pairs,
            parse_float=forbidden,
            parse_constant=forbidden,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError("invalid JSON evidence") from error


def archive_selected(selection: list[dict], root: Path) -> dict:
    """Copy only explicit selected artifacts, preserving exact bytes and pins."""
    root = Path(root).absolute()
    if root.exists():
        raise EvidenceError("fresh archive root required")
    rows = []
    seen = set()
    for entry in selection:
        if set(entry) != {"original", "role"} or not entry["role"]:
            raise EvidenceError("explicit original pin/graph role required")
        pin = entry["original"]
        validate_pin(pin)
        identity = (pin["path"], pin["sha256"], pin["kind"], entry["role"])
        if identity in seen:
            raise EvidenceError("duplicate retrieval entry")
        seen.add(identity)
        rows.append((pin, entry["role"], _read(Path(pin["path"]), pin["sha256"])))
    (root / "sha256").mkdir(parents=True, exist_ok=False)
    entries = []
    for pin, role, raw in rows:
        relative = "sha256/" + pin["sha256"]
        path = root / relative
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(raw)
            path.chmod(0o444)
        _read(path, pin["sha256"])
        entries.append(
            {"original": dict(pin), "role": role, "archive_relative_path": relative}
        )
    return {"schema": "gen1-stochastic-retrieval-map-v1", "entries": entries}


class EvidenceStore:
    """Resolve original artifact identities to raw CAS bytes, without relocation."""

    def __init__(self, root: Path, mapping: dict):
        if (
            set(mapping) != {"schema", "entries"}
            or mapping["schema"] != "gen1-stochastic-retrieval-map-v1"
        ):
            raise EvidenceError("retrieval schema mismatch")
        self.root = Path(root).absolute()
        self.entries = {}
        seen = set()
        for row in mapping["entries"]:
            if set(row) != {"original", "role", "archive_relative_path"}:
                raise EvidenceError("retrieval entry schema mismatch")
            pin = row["original"]
            validate_pin(pin)
            if row["archive_relative_path"] != "sha256/" + pin["sha256"]:
                raise EvidenceError("CAS path must be exact raw hash")
            if not isinstance(row["role"], str) or not row["role"]:
                raise EvidenceError("retrieval graph role required")
            identity = (pin["path"], pin["sha256"], pin["kind"])
            full = (*identity, row["role"])
            if full in seen:
                raise EvidenceError("duplicate retrieval mapping")
            seen.add(full)
            self.entries.setdefault(identity, {})[row["role"]] = row[
                "archive_relative_path"
            ]

    def read(self, pin: dict, *, role: str | None = None) -> bytes:
        validate_pin(pin)
        identity = (pin["path"], pin["sha256"], pin["kind"])
        entries = self.entries.get(identity)
        if entries is None:
            raise EvidenceError("missing selected retrieval pin")
        if role is not None and role not in entries:
            raise EvidenceError("retrieval graph role differs")
        relative = next(iter(entries.values())) if role is None else entries[role]
        path = self.root / relative
        if path.exists() and path.stat().st_mode & 0o222:
            raise EvidenceError("CAS evidence remains writable")
        return _read(path, pin["sha256"])

    def json(self, pin: dict, *, role: str | None = None):
        return decode_json(self.read(pin, role=role))


def materialize_checkout(checkout_root: Path, mapping: dict, destination: Path) -> dict:
    """Restore exact Git-stored CAS bytes into a fresh read-only external archive.

    Git preserves executable bits, not 0444. Writable checkout objects are only
    transport inputs here; they are never accepted as immutable evidence by
    EvidenceStore. No original absolute evidence path is accessed or rewritten.
    """
    source = EvidenceStore(checkout_root, mapping)  # Validate the finite mapping.
    destination = Path(destination).absolute()
    if destination.exists():
        raise EvidenceError("fresh materialization destination required")
    if destination.resolve() != destination:
        raise EvidenceError("aliased materialization destination")
    for ancestor in (destination, *destination.parents):
        if (ancestor / ".git").exists():
            raise EvidenceError("materialization must be outside Git")
    rows = []
    for identity, roles in source.entries.items():
        relative = next(iter(roles.values()))
        rows.append((relative, identity[1], _read(source.root / relative, identity[1])))
    (destination / "sha256").mkdir(parents=True, exist_ok=False)
    for relative, digest, raw in rows:
        output = destination / relative
        if not output.exists():
            with output.open("xb") as stream:
                stream.write(raw)
            output.chmod(0o444)
        _read(output, digest)
    restored = EvidenceStore(destination, mapping)
    for row in mapping["entries"]:
        restored.read(row["original"], role=row["role"])
    return {
        "schema": "gen1-stochastic-cas-materialization-v1",
        "source": str(source.root),
        "destination": str(destination),
        "objects": len({row[1] for row in rows}),
        "status": "EXACT_BYTES_READONLY_MATERIALIZED",
        "original_execution_repeated": False,
    }
