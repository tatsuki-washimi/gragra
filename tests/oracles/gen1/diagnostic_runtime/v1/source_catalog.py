"""Stdlib-only source, interpreter, and immutable-pin checks for both models."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any


class SourceCatalogError(ValueError):
    """A pinned file, interpreter, or candidate source inventory differs."""


_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_SUFFIXES = {".py", ".pyi", ".pth", ".so"}


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _regular_descriptor(path: str | Path, *, label: str) -> tuple[int, os.stat_result]:
    target = Path(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(target, flags)
        info = os.fstat(descriptor)
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise SourceCatalogError(f"{label} cannot be opened without following a symlink") from exc
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise SourceCatalogError(f"{label} is not a regular file")
    return descriptor, info


def _same_file_state(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def sha256_file(path: str | Path) -> str:
    """Hash one regular file through a stable O_NOFOLLOW descriptor."""
    descriptor, before = _regular_descriptor(path, label=f"pinned file {path}")
    digest = hashlib.sha256()
    byte_count = 0
    try:
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            byte_count += len(block)
        after = os.fstat(descriptor)
        if not _same_file_state(before, after) or byte_count != after.st_size:
            raise SourceCatalogError(f"pinned file changed during descriptor hashing: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def read_pinned_bytes(path: str | Path, expected_sha256: str, *, label: str) -> bytes:
    """Capture and hash the exact bytes read through a no-follow descriptor."""
    target = Path(path)
    if not target.is_absolute() or not _HEX64.fullmatch(expected_sha256):
        raise SourceCatalogError(f"{label} path/hash pin is malformed")
    descriptor, before = _regular_descriptor(target, label=label)
    chunks = []
    try:
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        captured = b"".join(chunks)
        if not _same_file_state(before, after) or len(captured) != after.st_size:
            raise SourceCatalogError(f"{label} changed while its pinned bytes were captured")
    finally:
        os.close(descriptor)
    if sha256_bytes(captured) != expected_sha256:
        raise SourceCatalogError(f"{label} captured bytes differ from the pinned SHA256")
    return captured


def strict_json(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise SourceCatalogError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise SourceCatalogError(f"nonfinite JSON constant: {value}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceCatalogError("malformed UTF-8 JSON bytes") from exc


def verify_pin(pin: dict[str, Any], *, label: str) -> dict[str, str]:
    if (
        not isinstance(pin, dict)
        or set(pin) != {"path", "sha256"}
        or not isinstance(pin.get("path"), str)
        or not Path(pin["path"]).is_absolute()
        or not isinstance(pin.get("sha256"), str)
        or not _HEX64.fullmatch(pin["sha256"])
    ):
        raise SourceCatalogError(f"{label} pin is malformed")
    actual = sha256_file(pin["path"])
    if actual != pin["sha256"]:
        raise SourceCatalogError(f"{label} bytes differ from pinned SHA256")
    return {"path": str(Path(pin["path"]).resolve(strict=True)), "sha256": actual}


def verify_interpreter(python_pin: dict[str, Any]) -> dict[str, Any]:
    """Bind the currently running isolated interpreter without subprocesses."""
    executable = python_pin.get("executable")
    if not isinstance(executable, dict):
        raise SourceCatalogError("manifest Python executable pin is malformed")
    expected_path = executable.get("path")
    expected_hash = executable.get("sha256")
    if (
        not isinstance(expected_path, str)
        or not Path(expected_path).is_absolute()
        or not isinstance(expected_hash, str)
        or not _HEX64.fullmatch(expected_hash)
    ):
        raise SourceCatalogError("manifest Python path/hash is malformed")
    actual_path = str(Path(sys.executable).resolve(strict=True))
    if actual_path != str(Path(expected_path).resolve(strict=True)):
        raise SourceCatalogError("sys.executable path differs from the frozen interpreter")
    actual_hash = sha256_file(actual_path)
    if actual_hash != expected_hash:
        raise SourceCatalogError("Python executable bytes differ from the frozen hash")
    if sys.flags.isolated != 1 or sys.flags.no_site != 1:
        raise SourceCatalogError("Python must be launched with -I -S")
    if any(os.environ.get(name) for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE")):
        raise SourceCatalogError("ambient Python path variables are forbidden")
    return {
        "executable": actual_path,
        "sha256": actual_hash,
        "version": list(sys.version_info[:3]),
        "isolated": sys.flags.isolated,
        "no_site": sys.flags.no_site,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
    }


def _safe_relative(raw: Any) -> PurePosixPath:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise SourceCatalogError("source catalog path must be a nonempty POSIX relative path")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SourceCatalogError("source catalog path escapes or aliases its root")
    return path


def _normalize_scan_root(raw: Any) -> str:
    if raw == ".":
        return ""
    return _safe_relative(raw).as_posix()


def _scan_root_matches(name: str, raw_root: Any) -> bool:
    normalized = _normalize_scan_root(raw_root)
    return not normalized or name == normalized or name.startswith(normalized + "/")


def _source_inventory(root: Path, scan_roots: list[str]) -> set[str]:
    found: set[str] = set()

    def walk_error(exc: OSError) -> None:
        raise SourceCatalogError(f"candidate source walk failed: errno={exc.errno}") from exc

    for raw_root in scan_roots:
        normalized = _normalize_scan_root(raw_root)
        candidate = root if not normalized else root.joinpath(*PurePosixPath(normalized).parts)
        if candidate.is_symlink():
            raise SourceCatalogError(f"source scan root is a symlink: {raw_root}")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise SourceCatalogError(f"source scan root is absent or escapes: {raw_root}") from exc
        if not resolved.is_dir():
            raise SourceCatalogError(f"source scan root is not a directory: {raw_root}")
        for directory, child_dirs, child_files in os.walk(
            resolved, topdown=True, followlinks=False, onerror=walk_error
        ):
            base = Path(directory)
            for name in list(child_dirs):
                child = base / name
                if child.is_symlink():
                    raise SourceCatalogError(f"symlink appears in candidate source scan: {child}")
                if name == "__pycache__":
                    child_dirs.remove(name)
            for name in child_files:
                path = base / name
                if path.is_symlink():
                    raise SourceCatalogError(f"symlink appears in candidate source scan: {path}")
                if path.suffix in _SOURCE_SUFFIXES and "__pycache__" not in path.parts:
                    relative = path.relative_to(root).as_posix()
                    if _scan_root_matches(relative, raw_root):
                        found.add(relative)
    return found


def verify_source_catalog(root: str | Path, catalog: dict[str, Any]) -> dict[str, Any]:
    """Require exact pinned bytes and a complete importable-source inventory."""
    candidate_root = Path(root).resolve(strict=True)
    if not candidate_root.is_dir() or not isinstance(catalog, dict):
        raise SourceCatalogError("candidate root/catalog is malformed")
    files = catalog.get("files")
    inventory = catalog.get("inventory")
    if not isinstance(files, dict) or not files or not isinstance(inventory, list):
        raise SourceCatalogError("source catalog lacks a complete hash map/inventory")
    if (
        any(not isinstance(name, str) for name in inventory)
        or inventory != sorted(inventory)
        or len(inventory) != len(set(inventory))
        or set(inventory) != set(files)
    ):
        raise SourceCatalogError("source catalog inventory is unsorted, duplicated or incomplete")
    observed_hashes = {}
    for raw_name in inventory:
        relative = _safe_relative(raw_name)
        path = candidate_root.joinpath(*relative.parts)
        if path.is_symlink():
            raise SourceCatalogError(f"catalogued source is a symlink: {raw_name}")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(candidate_root)
        except (OSError, ValueError) as exc:
            raise SourceCatalogError(f"catalogued source is absent or escapes root: {raw_name}") from exc
        expected = files[raw_name]
        if not isinstance(expected, str) or not _HEX64.fullmatch(expected):
            raise SourceCatalogError(f"catalogued SHA256 is malformed: {raw_name}")
        actual = sha256_file(resolved)
        if actual != expected:
            raise SourceCatalogError(f"candidate source bytes differ: {raw_name}")
        observed_hashes[raw_name] = actual
    scan_roots = catalog.get("scan_roots", ["."])
    if not isinstance(scan_roots, list) or not scan_roots:
        raise SourceCatalogError("source catalog has no importable-source scan roots")
    normalized_roots = [_normalize_scan_root(raw) for raw in scan_roots]
    if len(normalized_roots) != len(set(normalized_roots)):
        raise SourceCatalogError("source scan roots contain duplicate aliases")
    expected_sources = {
        name
        for name in inventory
        if Path(name).suffix in _SOURCE_SUFFIXES and "__pycache__" not in Path(name).parts
    }
    observed_sources = _source_inventory(candidate_root, scan_roots)
    expected_scanned = {
        name
        for name in expected_sources
        if any(_scan_root_matches(name, raw) for raw in scan_roots)
    }
    if observed_sources != expected_scanned:
        raise SourceCatalogError("candidate source inventory contains unlisted or missing importable bytes")
    catalog_raw = json.dumps(
        {"files": observed_hashes, "inventory": inventory, "scan_roots": scan_roots},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    return {
        "root": str(candidate_root),
        "file_count": len(observed_hashes),
        "source_file_count": len(observed_sources),
        "catalog_sha256": sha256_bytes(catalog_raw),
        "files": observed_hashes,
        "inventory": inventory,
        "scan_roots": scan_roots,
        "dirty_candidate_refused": False,
    }
