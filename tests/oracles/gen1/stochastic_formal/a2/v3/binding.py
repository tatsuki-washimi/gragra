"""A2 v3 diagnostic binding: the v2 binding with the six location rule changed.

The only semantic change from v2 is the location of the pinned ``six`` module. v2
pins one absolute path from a historical environment. v3 accepts the ``six.py``
that sits directly in the running interpreter's purelib, and keeps the v2 version
and SHA256 pins. Every other check is the v2 check.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import stat
import sys
import sysconfig
import types
from pathlib import Path, PurePosixPath
from typing import Any


class BindingError(ValueError):
    """Candidate source, interpreter, environment or import origin differs."""


_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_SUFFIXES = {".py", ".pyi", ".pth", ".so"}
_SIX_MODULE = "six"
_SIX_VERSION = "1.17.0"
_SIX_LOCATION = "sysconfig-purelib/six.py"
_SIX_SHA256 = "c51c91f703d3d4b3696c923cb5fec213e05e75d9215393befac7f2fa6a3904df"
SIX_IMPORTER_PIN = {
    "module": _SIX_MODULE,
    "version": _SIX_VERSION,
    "location": _SIX_LOCATION,
    "sha256": _SIX_SHA256,
}


def expected_six_path() -> str:
    """Return the resolved ``six.py`` path in the running interpreter's purelib."""
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve(strict=True)
    try:
        return str((purelib / "six.py").resolve(strict=True))
    except OSError as exc:
        raise BindingError(
            "six.py is absent from the active interpreter purelib"
        ) from exc


def _normalize_scan_root(raw: Any) -> str:
    """Normalize a catalog scan root; the empty normalized value means `.`."""
    if raw == ".":
        return ""
    return _safe_relative(raw).as_posix()


def _scan_root_matches(name: str, raw_root: Any) -> bool:
    normalized = _normalize_scan_root(raw_root)
    return not normalized or name == normalized or name.startswith(normalized + "/")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise BindingError(
            f"pinned source cannot be opened without following a symlink: {target}"
        ) from exc
    digest = hashlib.sha256()
    byte_count = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BindingError(f"pinned source is not a regular file: {target}")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            byte_count += len(block)
        after = os.fstat(descriptor)
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
        if identity_before != identity_after or byte_count != after.st_size:
            raise BindingError(
                f"pinned source changed during descriptor hashing: {target}"
            )
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def read_pinned_bytes(path: str | Path, expected_sha256: str, *, label: str) -> bytes:
    """Capture one pinned regular file through a single no-follow descriptor."""
    target = Path(path)
    if not target.is_absolute() or not _HEX64.fullmatch(expected_sha256):
        raise BindingError(f"{label} pin path/hash is malformed")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise BindingError(
            f"{label} cannot be opened as a pinned regular file"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BindingError(f"{label} pin is not a regular file")
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
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
        if identity_before != identity_after or sum(map(len, chunks)) != after.st_size:
            raise BindingError(f"{label} changed while its pinned bytes were captured")
    finally:
        os.close(descriptor)
    captured = b"".join(chunks)
    if sha256_bytes(captured) != expected_sha256:
        raise BindingError(f"{label} captured bytes differ from the pinned SHA256")
    return captured


def _safe_relative(raw: Any) -> PurePosixPath:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise BindingError("source catalog path must be a nonempty POSIX relative path")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise BindingError("source catalog path escapes or aliases its root")
    return path


def _source_inventory(root: Path, scan_roots: list[str]) -> set[str]:
    found: set[str] = set()
    for raw_root in scan_roots:
        normalized = _normalize_scan_root(raw_root)
        candidate = (
            root if not normalized else root.joinpath(*PurePosixPath(normalized).parts)
        )
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise BindingError(
                f"source scan root is absent or escapes the worktree: {raw_root}"
            ) from exc
        if not resolved.is_dir():
            raise BindingError(f"source scan root is not a directory: {raw_root}")
        for path in resolved.rglob("*"):
            if path.is_symlink():
                raise BindingError(f"symlink appears in candidate source scan: {path}")
            if not path.is_file() or path.suffix not in _SOURCE_SUFFIXES:
                continue
            if "__pycache__" in path.parts:
                continue
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError as exc:
                raise BindingError("source scan escaped candidate root") from exc
            if _scan_root_matches(rel, raw_root):
                found.add(rel)
    return found


def verify_source_catalog(root: str | Path, catalog: dict[str, Any]) -> dict[str, Any]:
    """Require exact catalog bytes and reject added/removed importable source files."""
    candidate_root = Path(root).resolve(strict=True)
    if not candidate_root.is_dir() or not isinstance(catalog, dict):
        raise BindingError("candidate root/catalog is malformed")
    files = catalog.get("files")
    inventory = catalog.get("inventory")
    if not isinstance(files, dict) or not files or not isinstance(inventory, list):
        raise BindingError("source catalog lacks a complete hash map/inventory")
    if (
        any(not isinstance(name, str) for name in inventory)
        or inventory != sorted(inventory)
        or len(inventory) != len(set(inventory))
        or set(inventory) != set(files)
    ):
        raise BindingError(
            "source catalog inventory is unsorted, duplicated or incomplete"
        )
    observed_hashes = {}
    for raw_name in inventory:
        relative = _safe_relative(raw_name)
        path = candidate_root.joinpath(*relative.parts)
        try:
            if path.is_symlink():
                raise BindingError(f"catalogued source is a symlink: {raw_name}")
            resolved = path.resolve(strict=True)
            resolved.relative_to(candidate_root)
        except (OSError, ValueError) as exc:
            raise BindingError(
                f"catalogued source is absent or escapes root: {raw_name}"
            ) from exc
        expected = files[raw_name]
        if not isinstance(expected, str) or not _HEX64.fullmatch(expected):
            raise BindingError(f"catalogued SHA256 is malformed: {raw_name}")
        actual = sha256_file(resolved)
        if actual != expected:
            raise BindingError(f"candidate source bytes differ: {raw_name}")
        observed_hashes[raw_name] = actual
    scan_roots = catalog.get("scan_roots", ["."])
    if not isinstance(scan_roots, list) or not scan_roots:
        raise BindingError("source catalog has no importable-source scan roots")
    normalized_roots = [_normalize_scan_root(raw) for raw in scan_roots]
    if len(normalized_roots) != len(set(normalized_roots)):
        raise BindingError("source scan roots contain duplicate aliases")
    source_expected = {
        name
        for name in inventory
        if Path(name).suffix in _SOURCE_SUFFIXES
        and "__pycache__" not in Path(name).parts
    }
    source_observed = _source_inventory(candidate_root, scan_roots)
    expected_scanned = {
        name
        for name in source_expected
        if any(_scan_root_matches(name, raw) for raw in scan_roots)
    }
    if source_observed != expected_scanned:
        raise BindingError(
            "candidate source inventory contains unlisted or missing importable bytes"
        )
    catalog_bytes = (
        json.dumps(
            {
                "files": observed_hashes,
                "inventory": inventory,
                "scan_roots": scan_roots,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    return {
        "root": str(candidate_root),
        "file_count": len(observed_hashes),
        "source_file_count": len(source_observed),
        "catalog_sha256": sha256_bytes(catalog_bytes),
        "files": observed_hashes,
        "inventory": inventory,
        "scan_roots": scan_roots,
        "dirty_candidate_refused": False,
    }


def strict_json(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise BindingError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise BindingError(f"nonfinite JSON constant: {value}")

    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BindingError("malformed UTF-8 JSON bytes") from exc


def load_source_catalog(manifest: dict[str, Any]) -> dict[str, Any]:
    """Load the A2 pinned canonical catalog map without invoking Git or subprocesses."""
    raw = manifest.get("source_catalog")
    if isinstance(raw, dict) and set(raw) == {"path", "sha256"}:
        if not isinstance(raw.get("path"), str) or not isinstance(
            raw.get("sha256"), str
        ):
            raise BindingError("A2 source catalog pin is malformed")
        captured = read_pinned_bytes(
            raw["path"], raw["sha256"], label="A2 source catalog"
        )
        raw = strict_json(captured)
    if (
        not isinstance(raw, dict)
        or not set(raw).issubset({"files", "inventory", "scan_roots"})
        or not isinstance(raw.get("files"), dict)
        or not raw["files"]
    ):
        raise BindingError("A2 source catalog must be a nonempty pinned files map")
    files = raw["files"]
    inventory = sorted(files)
    if raw.get("inventory", inventory) != inventory:
        raise BindingError("A2 source catalog inventory contradicts its exact file map")
    scan_roots = raw.get("scan_roots", ["."])
    if (
        not isinstance(scan_roots, list)
        or not scan_roots
        or any(not isinstance(item, str) for item in scan_roots)
        or scan_roots != sorted(set(scan_roots))
    ):
        raise BindingError(
            "A2 source catalog scan roots are malformed or nondeterministic"
        )
    return {
        "files": dict(files),
        "inventory": inventory,
        "scan_roots": list(scan_roots),
    }


def verify_loaded_candidate_origins(
    root: str | Path, catalog: dict[str, Any]
) -> dict[str, Any]:
    """Bind each loaded gragra module to a catalogued candidate source file."""
    candidate_root = Path(root).resolve(strict=True)
    files = catalog.get("files") if isinstance(catalog, dict) else None
    if not isinstance(files, dict):
        raise BindingError(
            "candidate source map is missing during import-origin verification"
        )
    observed: dict[str, dict[str, str]] = {}
    for name, module in sorted(sys.modules.items()):
        if name != "gragra" and not name.startswith("gragra."):
            continue
        spec = getattr(module, "__spec__", None)
        origins = [getattr(module, "__file__", None), getattr(spec, "origin", None)]
        if any(
            not isinstance(value, str) or not Path(value).is_absolute()
            for value in origins
        ):
            raise BindingError(
                f"loaded production module lacks absolute file/spec origin: {name}"
            )
        resolved = [str(Path(value).resolve(strict=True)) for value in origins]
        if resolved[0] != resolved[1]:
            raise BindingError(
                f"loaded production module file/spec origins disagree: {name}"
            )
        path = Path(resolved[0])
        try:
            relative = path.relative_to(candidate_root).as_posix()
        except ValueError as exc:
            raise BindingError(
                f"production module escaped candidate worktree: {name}"
            ) from exc
        digest = sha256_file(path)
        if files.get(relative) != digest:
            raise BindingError(
                f"production module is absent from or differs from "
                f"source catalog: {name}"
            )
        observed[name] = {"path": str(path), "sha256": digest}
    if (
        "gragra" not in observed
        or "gragra.analytic.stochastic_acoustic" not in observed
    ):
        raise BindingError(
            "loaded candidate origin set omits the A2 production package "
            "or implementation"
        )
    return {"modules": observed, "module_count": len(observed)}


def verify_pin(pin: dict[str, Any], *, label: str) -> dict[str, str]:
    if (
        not isinstance(pin, dict)
        or set(pin) != {"path", "sha256"}
        or not isinstance(pin.get("path"), str)
        or not Path(pin["path"]).is_absolute()
        or not isinstance(pin.get("sha256"), str)
        or not _HEX64.fullmatch(pin["sha256"])
    ):
        raise BindingError(f"{label} pin is malformed")
    actual = sha256_file(pin["path"])
    if actual != pin["sha256"]:
        raise BindingError(f"{label} bytes differ from pinned SHA256")
    return {"path": str(Path(pin["path"]).resolve(strict=True)), "sha256": actual}


def verify_interpreter(python_pin: dict[str, Any]) -> dict[str, Any]:
    """Bind this exact isolated interpreter without conda/Git subprocesses."""
    executable = python_pin.get("executable")
    if not isinstance(executable, dict):
        raise BindingError("manifest Python executable pin is malformed")
    expected_path = executable.get("path")
    expected_hash = executable.get("sha256")
    if (
        not isinstance(expected_path, str)
        or not Path(expected_path).is_absolute()
        or not isinstance(expected_hash, str)
        or not _HEX64.fullmatch(expected_hash)
    ):
        raise BindingError("manifest Python path/hash is malformed")
    actual_path = str(Path(sys.executable).resolve(strict=True))
    if actual_path != str(Path(expected_path).resolve(strict=True)):
        raise BindingError("sys.executable path differs from the frozen interpreter")
    actual_hash = sha256_file(actual_path)
    if actual_hash != expected_hash:
        raise BindingError("Python executable bytes differ from the frozen hash")
    if sys.flags.isolated != 1 or sys.flags.no_site != 1:
        raise BindingError("Python must be launched with -I -S")
    if any(
        os.environ.get(name) for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE")
    ):
        raise BindingError("ambient Python path variables are forbidden")
    return {
        "executable": actual_path,
        "sha256": actual_hash,
        "version": list(sys.version_info[:3]),
        "isolated": sys.flags.isolated,
        "no_site": sys.flags.no_site,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
    }


def explicit_import_path(
    *,
    worktree_root: str | Path,
    site_packages: list[str],
) -> list[str]:
    """Construct import roots by list insertion only; never process .pth files."""
    root = Path(worktree_root).resolve(strict=True)
    requested = [str(root), str(root / "src")]
    for raw in site_packages:
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            raise BindingError("site-packages entries must be absolute paths")
        path = Path(raw).resolve(strict=True)
        if not path.is_dir():
            raise BindingError("explicit site-packages path is not a directory")
        requested.append(str(path))
    if len(requested) != len(set(requested)):
        raise BindingError("explicit import roots contain duplicates")
    # Preserve the interpreter's isolated stdlib search roots ahead of all
    # candidate and site-package roots. Under -I -S this is the original
    # interpreter-controlled sys.path; editable .pth/finder paths are absent.
    interpreter_roots = [item for item in sys.path if item not in requested]
    sys.path[:] = [*interpreter_roots, *requested]
    return list(sys.path)


def verify_import_state(
    expected_origins: dict[str, dict[str, str]],
    *,
    expected_versions: dict[str, str] | None = None,
    six_pin: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check actual loaded module origins/hashes and optional distribution versions."""
    if six_pin is not None:
        six_origin = expected_origins.get("six")
        if (
            six_origin != {"path": expected_six_path(), "sha256": _SIX_SHA256}
            or six_pin != SIX_IMPORTER_PIN
            or (expected_versions or {}).get("six") != _SIX_VERSION
            or "six" not in sys.modules
        ):
            raise BindingError(
                "post-import six exception is not present in the loaded "
                "immutable origin/version pins"
            )
    origins: dict[str, dict[str, str]] = {}
    for name, pin in expected_origins.items():
        module = sys.modules.get(name)
        if module is None:
            raise BindingError(f"expected module is not loaded: {name}")
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str):
            spec = getattr(module, "__spec__", None)
            origin = getattr(spec, "origin", None)
        if not isinstance(origin, str) or not Path(origin).is_absolute():
            raise BindingError(f"module has no absolute origin: {name}")
        resolved = str(Path(origin).resolve(strict=True))
        digest = sha256_file(resolved)
        if (
            resolved != str(Path(pin["path"]).resolve(strict=True))
            or digest != pin["sha256"]
        ):
            raise BindingError(f"module origin/bytes differ from pin: {name}")
        origins[name] = {"path": resolved, "sha256": digest}
    metadata_versions: dict[str, str] = {}
    if expected_versions and "iniconfig" in expected_versions:
        if manifest is None:
            raise BindingError(
                "iniconfig version verification requires pinned distribution metadata"
            )
        metadata_versions = _verified_distribution_versions(
            verify_test_framework_inventory(manifest)
        )
    versions: dict[str, str] = {}
    for name, expected in (expected_versions or {}).items():
        module = sys.modules.get(name) or importlib.import_module(name)
        actual = (
            metadata_versions.get(name)
            if name == "iniconfig"
            else getattr(module, "__version__", None)
        )
        if not isinstance(actual, str) or actual != expected:
            raise BindingError(f"dependency version differs: {name}")
        versions[name] = actual
    meta_names = validate_meta_path(sys.meta_path, six_pin=six_pin)
    return {
        "origins": origins,
        "versions": versions,
        "sys_path": list(sys.path),
        "meta_path_types": sorted(meta_names),
    }


def validate_meta_path(
    finders: list[Any] | tuple[Any, ...],
    *,
    trusted_finders: tuple[Any, ...] | None = None,
    six_pin: dict[str, Any] | None = None,
) -> list[str]:
    """Require the standard finders plus one pinned-six post-import finder."""
    if trusted_finders is None:
        from importlib.machinery import BuiltinImporter, FrozenImporter, PathFinder

        trusted_finders = (BuiltinImporter, FrozenImporter, PathFinder)
    elif six_pin is not None:
        raise BindingError(
            "verified six exception cannot use caller-supplied "
            "standard finder identities"
        )
    names = {finder.__name__: finder for finder in trusted_finders}
    if (
        set(names) != {"BuiltinImporter", "FrozenImporter", "PathFinder"}
        or len(names) != 3
    ):
        raise BindingError("trusted import machinery identity inventory is malformed")
    six_importer = None
    if six_pin is not None:
        six_importer, _ = _load_verified_six_importer(six_pin)
    observed_names = []
    for finder in finders:
        matched = next((name for name, known in names.items() if finder is known), None)
        if matched is None and six_importer is not None and finder is six_importer:
            matched = "six._importer"
        if matched is None:
            raise BindingError("editable or third-party meta_path finder is installed")
        observed_names.append(matched)
    expected_names = set(names) | (
        {"six._importer"} if six_importer is not None else set()
    )
    if (
        len(observed_names) != len(expected_names)
        or set(observed_names) != expected_names
    ):
        raise BindingError("standard meta_path finder set is incomplete or duplicated")
    return observed_names


def _load_verified_six_importer(pin: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Bind the already-loaded six module and its single finder to immutable bytes."""
    if not isinstance(pin, dict) or pin != SIX_IMPORTER_PIN:
        raise BindingError("six pin differs from the reviewed immutable dependency pin")
    module = sys.modules.get(_SIX_MODULE)
    if module is None or getattr(module, "__name__", None) != _SIX_MODULE:
        raise BindingError("pinned six module is not already present in sys.modules")
    namespace = getattr(module, "__dict__", None)
    if not isinstance(namespace, dict) or namespace.get("__name__") != _SIX_MODULE:
        raise BindingError("loaded six module namespace identity is malformed")
    source_path = expected_six_path()
    module_file = getattr(module, "__file__", None)
    spec = getattr(module, "__spec__", None)
    spec_origin = getattr(spec, "origin", None)
    if (
        not isinstance(module_file, str)
        or not isinstance(spec_origin, str)
        or not Path(module_file).is_absolute()
        or not Path(spec_origin).is_absolute()
        or str(Path(module_file).resolve(strict=True)) != source_path
        or str(Path(spec_origin).resolve(strict=True)) != source_path
    ):
        raise BindingError(
            "six __file__/__spec__.origin differ from the reviewed source path"
        )
    actual_hash = sha256_file(source_path)
    if actual_hash != _SIX_SHA256:
        raise BindingError("six source bytes differ from the reviewed immutable SHA256")
    if namespace.get("__version__") != _SIX_VERSION:
        raise BindingError(
            "six module version differs from the reviewed dependency pin"
        )

    importer_class = namespace.get("_SixMetaPathImporter")
    importer = namespace.get("_importer")
    if (
        not isinstance(importer_class, type)
        or importer_class.__module__ != _SIX_MODULE
        or importer_class.__qualname__ != "_SixMetaPathImporter"
        or importer is None
        or type(importer) is not importer_class
        or getattr(importer, "name", None) != _SIX_MODULE
    ):
        raise BindingError("six finder is not the exact module-owned singleton")
    method_sources: dict[str, str] = {}
    for method_name in ("find_module", "find_spec"):
        method = importer_class.__dict__.get(method_name)
        if (
            not isinstance(method, types.FunctionType)
            or method.__globals__ is not namespace
            or method.__module__ != _SIX_MODULE
            or not isinstance(method.__code__.co_filename, str)
            or not Path(method.__code__.co_filename).is_absolute()
        ):
            raise BindingError(f"six finder method binding differs: {method_name}")
        method_path = str(Path(method.__code__.co_filename).resolve(strict=True))
        if method_path != source_path:
            raise BindingError(
                f"six finder method source differs from pinned file: {method_name}"
            )
        method_sources[method_name] = method_path
    evidence = {
        "module": _SIX_MODULE,
        "version": _SIX_VERSION,
        "path": source_path,
        "sha256": actual_hash,
        "module_file_matches": True,
        "spec_origin_matches": True,
        "singleton_attribute": "_importer",
        "singleton_name": "six",
        "class_qualname": "_SixMetaPathImporter",
        "finder_methods": {
            name: {"source_path": path, "globals_are_six_module": True}
            for name, path in sorted(method_sources.items())
        },
        "meta_path_singleton_identity_verified": True,
    }
    return importer, evidence


def verify_six_importer(pin: dict[str, Any]) -> dict[str, Any]:
    """Return evidence for the exact loaded six singleton; this never imports six."""
    _, evidence = _load_verified_six_importer(pin)
    return evidence


def reject_production_modules() -> None:
    if any(name == "gragra" or name.startswith("gragra.") for name in sys.modules):
        raise BindingError("independent retained worker imported production modules")


def verify_test_framework_inventory(manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify pytest's pinned source tree and direct runtime dependencies."""
    external = manifest.get("external_pins")
    pin = (
        external.get("pytest_framework_sources") if isinstance(external, dict) else None
    )
    if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
        raise BindingError("pytest framework source inventory pin is missing")
    inventory_raw = read_pinned_bytes(
        pin["path"], pin["sha256"], label="pytest framework inventory"
    )
    inventory = strict_json(inventory_raw)
    expected_names = ["iniconfig", "packaging", "pluggy", "pygments", "pytest"]
    if (
        not isinstance(inventory, dict)
        or set(inventory) != {"schema", "package_root", "distributions", "files"}
        or inventory.get("schema") != "gragra.a2.test-framework-source-inventory.v1"
        or not isinstance(inventory.get("package_root"), str)
        or not Path(inventory["package_root"]).is_absolute()
        or not isinstance(inventory.get("distributions"), list)
        or not isinstance(inventory.get("files"), list)
    ):
        raise BindingError("pytest framework source inventory schema is malformed")
    distributions = inventory["distributions"]
    names = [
        item.get("name") if isinstance(item, dict) else None for item in distributions
    ]
    if names != expected_names or len(names) != len(distributions):
        raise BindingError(
            "pytest operational distribution inventory is incomplete or reordered"
        )
    dependency_pins = manifest.get("dependency_inventory")
    if not isinstance(dependency_pins, list):
        raise BindingError(
            "dependency inventory is missing while checking pytest sources"
        )
    pins_by_module = {
        item.get("module"): item for item in dependency_pins if isinstance(item, dict)
    }
    expected_module_paths = {
        "iniconfig": "iniconfig/__init__.py",
        "packaging": "packaging/__init__.py",
        "pluggy": "pluggy/__init__.py",
        "pygments": "pygments/__init__.py",
        "pytest": "pytest/__init__.py",
    }
    for distribution in distributions:
        name = distribution["name"]
        if set(distribution) != {
            "name",
            "version",
            "module",
            "module_path",
            "module_sha256",
            "metadata_path",
            "metadata_sha256",
            "requires_dist",
        }:
            raise BindingError(f"pytest distribution inventory fields differ: {name}")
        dependency = pins_by_module.get(name)
        expected_module_path = str(
            Path(inventory["package_root"]) / expected_module_paths[name]
        )
        if (
            distribution["module"] != name
            or distribution["module_path"] != expected_module_path
            or not isinstance(distribution["version"], str)
            or not isinstance(distribution["module_sha256"], str)
            or dependency is None
            or dependency.get("path") != distribution["module_path"]
            or dependency.get("sha256") != distribution["module_sha256"]
            or dependency.get("version") != distribution["version"]
        ):
            raise BindingError(
                f"pytest dependency pin differs from framework inventory: {name}"
            )
    files = inventory["files"]
    file_paths = [item.get("path") for item in files if isinstance(item, dict)]
    if file_paths != sorted(set(file_paths)) or len(file_paths) != len(files):
        raise BindingError(
            "pytest framework file inventory is duplicated or nondeterministic"
        )
    package_root = Path(inventory["package_root"]).resolve(strict=True)
    observed_files: dict[str, str] = {}
    for item in files:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256"}
            or not isinstance(item.get("path"), str)
            or not Path(item["path"]).is_absolute()
            or not isinstance(item.get("sha256"), str)
            or not _HEX64.fullmatch(item["sha256"])
        ):
            raise BindingError("pytest framework source file pin is malformed")
        path = Path(item["path"])
        try:
            path.relative_to(package_root)
        except ValueError as exc:
            raise BindingError(
                "pytest framework source file escapes the pinned package root"
            ) from exc
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise BindingError(f"pytest framework source differs from its pin: {path}")
        observed_files[str(path)] = actual
    package_dirs = ["pytest", "_pytest", "pluggy", "packaging", "iniconfig", "pygments"]
    expected_files: set[str] = set()
    for name in package_dirs:
        directory = package_root / name
        if directory.is_symlink() or not directory.is_dir():
            raise BindingError(
                f"pytest framework package directory is absent or a symlink: {name}"
            )
        expected_files.update(
            str(path) for path in directory.rglob("*.py") if path.is_file()
        )
    for distribution in distributions:
        expected_files.add(distribution["metadata_path"])
    if set(observed_files) != expected_files:
        raise BindingError(
            "pytest framework source inventory has missing or "
            "unlisted Python/metadata files"
        )
    observed_distributions = []
    for distribution in distributions:
        metadata_raw = read_pinned_bytes(
            distribution["metadata_path"],
            distribution["metadata_sha256"],
            label=f"{distribution['name']} distribution metadata",
        )
        try:
            lines = metadata_raw.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise BindingError("pytest distribution metadata is not UTF-8") from exc
        versions = [
            line.partition(":")[2].strip()
            for line in lines
            if line.startswith("Version:")
        ]
        requirements = [
            line.partition(":")[2].strip()
            for line in lines
            if line.startswith("Requires-Dist:")
        ]
        if (
            versions != [distribution["version"]]
            or requirements != distribution["requires_dist"]
            or distribution["metadata_path"] not in observed_files
            or observed_files[distribution["metadata_path"]]
            != distribution["metadata_sha256"]
        ):
            raise BindingError(
                "pytest distribution version/dependency metadata "
                f"differs: {distribution['name']}"
            )
        observed_distributions.append(
            {
                "name": distribution["name"],
                "version": distribution["version"],
                "metadata_sha256": distribution["metadata_sha256"],
            }
        )
    return {
        "inventory_path": str(Path(pin["path"]).resolve(strict=True)),
        "inventory_sha256": pin["sha256"],
        "file_count": len(observed_files),
        "distributions": observed_distributions,
        "files": observed_files,
    }


def _verified_distribution_versions(framework: dict[str, Any]) -> dict[str, str]:
    """Extract versions only from hash-checked distribution metadata."""
    distributions = (
        framework.get("distributions") if isinstance(framework, dict) else None
    )
    if not isinstance(distributions, list):
        raise BindingError("verified test-framework distribution evidence is missing")
    versions: dict[str, str] = {}
    for item in distributions:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("version"), str)
            or not isinstance(item.get("metadata_sha256"), str)
            or not _HEX64.fullmatch(item["metadata_sha256"])
            or item["name"] in versions
        ):
            raise BindingError(
                "verified test-framework distribution evidence is malformed"
            )
        versions[item["name"]] = item["version"]
    return versions


def environment_report(manifest: dict[str, Any]) -> dict[str, Any]:
    """Charged environment leaf: import pinned dependencies and record origins."""
    python = verify_interpreter(manifest["python"])
    site_packages = manifest["python"].get("site_packages")
    if not isinstance(site_packages, list):
        raise BindingError("explicit site-packages inventory is missing")
    explicit_import_path(
        worktree_root=manifest["paths"]["worktree_root"], site_packages=site_packages
    )
    pre_import_meta_path = validate_meta_path(sys.meta_path)
    test_framework = verify_test_framework_inventory(manifest)
    metadata_versions = _verified_distribution_versions(test_framework)
    dependency_pins = manifest.get("dependency_inventory")
    if not isinstance(dependency_pins, list) or not dependency_pins:
        raise BindingError("pinned dependency inventory is missing")
    dependencies = {}
    six_pin = None
    for pin in dependency_pins:
        if not isinstance(pin, dict) or set(pin) != {
            "module",
            "version",
            "path",
            "sha256",
        }:
            raise BindingError("dependency pin fields differ")
        if pin.get("module") == "six":
            if six_pin is not None:
                raise BindingError("dependency inventory duplicates the six pin")
            six_pin = pin
        module = importlib.import_module(pin["module"])
        path = str(Path(module.__file__).resolve(strict=True))
        actual = sha256_file(path)
        module_version = (
            metadata_versions.get(pin["module"])
            if pin["module"] == "iniconfig"
            else getattr(module, "__version__", None)
        )
        if (
            module_version != pin["version"]
            or path != str(Path(pin["path"]).resolve(strict=True))
            or actual != pin["sha256"]
        ):
            raise BindingError(
                f"dependency origin/version/hash differs: {pin['module']}"
            )
        dependencies[pin["module"]] = {
            "version": module_version,
            "path": path,
            "sha256": actual,
        }
    if six_pin != SIX_IMPORTER_PIN:
        raise BindingError(
            "immutable dependency inventory must include the reviewed six pin"
        )
    six_importer = verify_six_importer(six_pin)
    post_import_meta_path = validate_meta_path(sys.meta_path, six_pin=six_pin)
    try:
        status_raw = Path("/proc/self/status").read_bytes()
        task_ids = sorted(
            int(entry.name) for entry in Path("/proc/self/task").iterdir()
        )
    except (OSError, ValueError) as exc:
        raise BindingError("current process thread evidence is unavailable") from exc
    threads_lines = [
        line.split(b":", 1)[1].strip()
        for line in status_raw.splitlines()
        if line.startswith(b"Threads:")
    ]
    if len(threads_lines) != 1 or not threads_lines[0].isdigit():
        raise BindingError("current process Threads field is absent or malformed")
    thread_count = int(threads_lines[0])
    if thread_count <= 0 or thread_count != len(task_ids):
        raise BindingError("current process status/task thread evidence contradicts")
    if thread_count != 1:
        raise BindingError(
            "environment admission requires exactly one observed process thread"
        )
    maps_path = Path("/proc/self/maps")
    maps = maps_path.read_bytes()
    return {
        "python": python,
        "dependencies": dependencies,
        "sys_path": list(sys.path),
        "pre_import_meta_path_types": pre_import_meta_path,
        "meta_path_types": sorted(post_import_meta_path),
        "six_importer": six_importer,
        "test_framework": test_framework,
        "native_maps_sha256": sha256_bytes(maps),
        "native_maps_base64": __import__("base64").b64encode(maps).decode("ascii"),
        "thread_count": thread_count,
        "thread_count_limit": 1,
        "thread_status_sha256": sha256_bytes(status_raw),
        "thread_status_base64": __import__("base64")
        .b64encode(status_raw)
        .decode("ascii"),
        "task_ids": task_ids,
        "science_credit": 0,
    }
