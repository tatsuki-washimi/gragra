"""Resolve finite planned metadata/configuration origins into normal CAS objects."""

from __future__ import annotations

import os
from pathlib import Path

from .archive import EvidenceError, _read, decode_json, validate_pin
from .closure import _validate


def _identity(pin):
    validate_pin(pin)
    return tuple(pin[name] for name in ("path", "sha256", "kind"))


def _resolution(path):
    """Measure logical ancestors and their literal link targets, without caching."""
    path = Path(path)
    links = []
    for ancestor in reversed((path, *path.parents)):
        if ancestor.is_symlink():
            links.append({"path": str(ancestor), "target": os.readlink(ancestor)})
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise EvidenceError("dependency origin cannot be resolved") from error
    if not links or resolved == path:
        raise EvidenceError("explicit aliased dependency origin required")
    return resolved, links


def archive_with_dependency_resolutions(selection, root, resolution_pin):
    """Copy a finite graph with only declared planned metadata/config aliases.

    Ordinary artifacts still use the strict original-path reader. The exception
    is confined to package_record and import_configuration pins present in the
    exact selected dependency closures. This records current retrieval rather
    than historical admission.
    The immutable closure, original logical pin and raw bytes remain unchanged.
    """
    _identity(resolution_pin)
    profile = decode_json(_read(Path(resolution_pin["path"]), resolution_pin["sha256"]))
    if (
        not isinstance(profile, dict)
        or set(profile) != {"schema", "dependency_closures", "entries"}
        or profile["schema"] != "gen1-stochastic-dependency-resolution-v1"
        or not isinstance(profile["dependency_closures"], list)
        or not profile["dependency_closures"]
        or not isinstance(profile["entries"], list)
        or not profile["entries"]
    ):
        raise EvidenceError("finite dependency resolution profile required")
    closure_ids, allowed = set(), set()
    for pin in profile["dependency_closures"]:
        identity = _identity(pin)
        if identity in closure_ids:
            raise EvidenceError("duplicate resolution closure")
        closure_ids.add(identity)
        closure = decode_json(_read(Path(pin["path"]), pin["sha256"]))
        _validate(closure)
        for package in closure["packages"]:
            artifact = package["artifact"]
            if artifact is not None and artifact["kind"] == "package_record":
                allowed.add(_identity(artifact))
        for artifact in closure["artifacts"]:
            if artifact["kind"] == "import_configuration":
                allowed.add(_identity(artifact))
    resolutions = {}
    for entry in profile["entries"]:
        if not isinstance(entry, dict) or set(entry) != {
            "original",
            "declared_path",
            "resolved_path",
            "symlinks",
        }:
            raise EvidenceError("unknown dependency resolution fields")
        identity = _identity(entry["original"])
        if identity not in allowed or entry["declared_path"] != identity[0]:
            raise EvidenceError(
                "resolution is not an exact planned metadata/config pin"
            )
        if identity in resolutions:
            raise EvidenceError("duplicate dependency resolution")
        resolved, links = _resolution(identity[0])
        if entry["resolved_path"] != str(resolved) or entry["symlinks"] != links:
            raise EvidenceError("declared dependency resolution differs")
        resolutions[identity] = _read(resolved, identity[1])
        if _resolution(identity[0]) != (resolved, links):
            raise EvidenceError("dependency resolution changed during retrieval")
    root = Path(root).absolute()
    if root.exists() or root.resolve() != root:
        raise EvidenceError("fresh unaliased archive root required")
    rows, seen, selected_ids = [], set(), set()
    for entry in selection:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"original", "role"}
            or not isinstance(entry["role"], str)
            or not entry["role"]
        ):
            raise EvidenceError("explicit original pin/graph role required")
        pin, role = entry["original"], entry["role"]
        identity = _identity(pin)
        full = (*identity, role)
        if full in seen:
            raise EvidenceError("duplicate retrieval entry")
        seen.add(full)
        selected_ids.add(identity)
        raw = resolutions.get(identity)
        if raw is None:
            raw = _read(Path(pin["path"]), pin["sha256"])
        rows.append((pin, role, raw))
    required = closure_ids | set(resolutions) | {_identity(resolution_pin)}
    if not required <= selected_ids:
        raise EvidenceError("resolution proof/closure/dependency not selected")
    # All graph/proof bytes have passed before allocating an archive directory.
    (root / "sha256").mkdir(parents=True, exist_ok=False)
    entries = []
    for pin, role, raw in rows:
        relative = "sha256/" + pin["sha256"]
        target = root / relative
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(raw)
            target.chmod(0o444)
        _read(target, pin["sha256"])
        entries.append(
            {"original": dict(pin), "role": role, "archive_relative_path": relative}
        )
    return {"schema": "gen1-stochastic-retrieval-map-v1", "entries": entries}
