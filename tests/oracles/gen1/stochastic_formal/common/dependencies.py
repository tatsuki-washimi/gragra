"""Finite planned/runtime dependency closure; no recursive operating-system audit."""

from __future__ import annotations

import builtins
import hashlib
import importlib
import importlib.metadata
import importlib.util
import platform
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from .transport import GateError, artifact_pin, canonical_bytes, verify_artifact

SCHEMA = "gen1-stochastic-dependencies-v1"


def python_pin() -> dict:
    return {
        "implementation": platform.python_implementation(),
        "version": sys.version,
        "executable": artifact_pin(Path(sys.executable).resolve(), "python_executable"),
    }


def module_pin(
    module: ModuleType,
    *,
    package: str,
    root: str | Path,
    repo: str | Path | None = None,
    commit: str | None = None,
) -> dict:
    origin = getattr(module, "__file__", None)
    if not origin:
        raise GateError(f"file-backed module required: {module.__name__}")
    path = Path(origin).resolve(strict=True)
    root = Path(root).resolve(strict=True)
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise GateError("module lies outside declared package root") from exc
    blob = None
    if repo is not None:
        if commit is None:
            raise GateError("Git-backed module requires exact commit")
        try:
            git_relative = path.relative_to(Path(repo).resolve()).as_posix()
            result = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", f"{commit}:{git_relative}"],
                check=True,
                capture_output=True,
                text=True,
            )
            blob = result.stdout.strip()
            source = subprocess.run(
                ["git", "-C", str(repo), "show", f"{commit}:{git_relative}"],
                check=True,
                capture_output=True,
            ).stdout
            if source != path.read_bytes():
                raise GateError("module working bytes differ from pinned Git blob")
        except (subprocess.CalledProcessError, ValueError) as exc:
            raise GateError("module Git pin unavailable") from exc
    return {
        "name": module.__name__,
        "package": package,
        "path": str(path),
        "root": str(root),
        "relative_path": relative,
        "git_blob": blob,
        "git_blob_missing_reason": "module is outside the tracked repository"
        if blob is None
        else None,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "kind": "python"
        if path.suffix == ".py"
        else "extension"
        if any(str(path).endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES)
        else "bytecode",
    }


def native_paths() -> list[str]:
    """Identify actual mapped BLAS/LAPACK paths without caching byte identity."""
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        raise GateError("runtime native backend map unavailable")
    paths = set()
    for line in maps.read_text().splitlines():
        path = line.split(maxsplit=5)[-1]
        lower = path.lower()
        if path.startswith("/") and any(
            token in lower for token in ("blas", "lapack", "mkl", "blis")
        ):
            if path.endswith(" (deleted)"):
                raise GateError("runtime native backend bytes deleted")
            paths.add(str(Path(path).resolve(strict=True)))
    return sorted(paths)


def native_inventory() -> list[dict]:
    """Hash actually mapped BLAS/LAPACK libraries in this Linux worker."""
    return [artifact_pin(path, "native_backend") for path in native_paths()]


def verify_runtime_closure(
    manifest: dict, modules: dict[str, ModuleType], observed_edges: list[dict]
) -> dict:
    """Require every loaded protected module and observed edge to be planned."""
    required = {
        "schema",
        "role",
        "production_commit",
        "production_tree",
        "modules",
        "import_edges",
        "packages",
        "native_backends",
        "artifacts",
        "python",
        "protected_packages",
    }
    if (
        set(manifest) != required
        or manifest["schema"] != SCHEMA
        or manifest["role"] not in {"production", "reference"}
    ):
        raise GateError("dependency closure schema/role mismatch")
    if manifest["python"] != python_pin():
        raise GateError("Python runtime differs from planned closure")
    verify_artifact(manifest["python"]["executable"])
    protected = manifest["protected_packages"]
    if (
        not isinstance(protected, list)
        or not protected
        or len(protected) != len(set(protected))
    ):
        raise GateError("protected package list invalid")
    planned = {}
    for pin in manifest["modules"]:
        fields = {
            "name",
            "package",
            "path",
            "root",
            "relative_path",
            "git_blob",
            "git_blob_missing_reason",
            "sha256",
            "kind",
        }
        if not isinstance(pin, dict) or set(pin) != fields or pin["name"] in planned:
            raise GateError("planned module schema/duplicate mismatch")
        if pin["package"] not in protected or not (
            pin["name"] == pin["package"]
            or pin["name"].startswith(pin["package"] + ".")
        ):
            raise GateError("planned module protected package mismatch")
        if pin["kind"] not in {"python", "extension", "bytecode"}:
            raise GateError("unknown module kind")
        if pin["git_blob"] is None and not pin["git_blob_missing_reason"]:
            raise GateError("missing module provenance reason")
        if pin["package"] == "gragra" and pin["git_blob"] is None:
            # Test/runtime source-only modules may explicitly describe non-Git origins;
            # formal preflight additionally requires Git pins for every gragra source.
            pass
        path = Path(pin["path"]).resolve(strict=True)
        if (
            path.relative_to(Path(pin["root"]).resolve()).as_posix()
            != pin["relative_path"]
        ):
            raise GateError("planned relative origin mismatch")
        verify_artifact(
            {"path": pin["path"], "sha256": pin["sha256"], "kind": pin["kind"]}
        )
        planned[pin["name"]] = pin
    actual = {}
    for name, module in modules.items():
        if not any(name == p or name.startswith(p + ".") for p in protected):
            continue
        if name not in planned:
            raise GateError(f"unknown runtime module: {name}")
        pin = planned[name]
        origin = getattr(module, "__file__", None)
        if not origin or Path(origin).resolve(strict=True) != Path(pin["path"]).resolve(
            strict=True
        ):
            raise GateError(f"runtime origin mismatch: {name}")
        verify_artifact(
            {"path": pin["path"], "sha256": pin["sha256"], "kind": pin["kind"]}
        )
        actual[name] = pin
    edges = manifest["import_edges"]
    if any(
        not isinstance(e, dict) or set(e) != {"from", "to"}
        for e in edges + observed_edges
    ):
        raise GateError("import edge schema mismatch")
    allowed = {(e["from"], e["to"]) for e in edges}
    if len(allowed) != len(edges):
        raise GateError("duplicate planned import edge")
    if any((e["from"], e["to"]) not in allowed for e in observed_edges):
        raise GateError("unknown runtime import edge")
    for package in manifest["packages"]:
        if set(package) != {
            "name",
            "version",
            "location",
            "artifact",
            "artifact_missing_reason",
        }:
            raise GateError("package provenance schema mismatch")
        if package["artifact"] is None:
            if not package["artifact_missing_reason"]:
                raise GateError("package artifact hash omission unexplained")
        else:
            verify_artifact(package["artifact"])
        if importlib.metadata.version(package["name"]) != package["version"]:
            raise GateError("runtime distribution version mismatch")
        location = str(
            Path(
                importlib.metadata.distribution(package["name"]).locate_file("")
            ).resolve()
        )
        if location != package["location"]:
            raise GateError("runtime distribution origin mismatch")
    for backend in manifest["native_backends"]:
        verify_artifact(backend)
    planned_native = {p["path"]: p for p in manifest["native_backends"]}
    observed_native = (
        native_inventory() if modules and {"numpy", "scipy"} & set(protected) else []
    )
    if any(planned_native.get(p["path"]) != p for p in observed_native):
        raise GateError("unknown runtime native backend")
    for artifact in manifest["artifacts"]:
        verify_artifact(artifact)
    return {
        "status": "MATCH",
        "observed_modules": list(actual.values()),
        "observed_edges": observed_edges,
        "observed_native_backends": observed_native,
        "planned_sha256": hashlib.sha256(canonical_bytes(manifest)).hexdigest(),
        "python": python_pin(),
    }


class ImportGuard:
    """Observe protected imports and revalidate loaded bytes before computation.

    Install only in a single-threaded dedicated comparison child. This context is
    not an OS sandbox; its planned protected packages delimit the scientific audit.
    """

    def __init__(self, manifest: dict):
        self.manifest = manifest
        self.edges = []
        self._import = None
        self._import_module = None
        self._checking = False
        self.allowed = {(e["from"], e["to"]) for e in manifest["import_edges"]}
        self.protected = tuple(manifest["protected_packages"])
        self.planned = {p["name"]: p for p in manifest["modules"]}
        self.checked_modules = set()
        self.checked_native = set()

    def verify_import_state(self):
        """Check new loaded identities without repeatedly hashing cold imports.

        This incremental check never replaces verify(): complete byte identities
        are rehashed at each evaluation boundary and on phase exit.
        """
        self._checking = True
        try:
            for name, module in tuple(sys.modules.items()):
                if not any(
                    name == p or name.startswith(p + ".") for p in self.protected
                ):
                    continue
                pin = self.planned.get(name)
                if pin is None:
                    raise GateError(f"unknown runtime module: {name}")
                origin = getattr(module, "__file__", None)
                if not origin or Path(origin).resolve(strict=True) != Path(
                    pin["path"]
                ).resolve(strict=True):
                    raise GateError(f"runtime origin mismatch: {name}")
                if name not in self.checked_modules:
                    verify_artifact(
                        {
                            "path": pin["path"],
                            "sha256": pin["sha256"],
                            "kind": pin["kind"],
                        }
                    )
                    self.checked_modules.add(name)
            if {"numpy", "scipy"} & set(self.protected):
                planned_native = {
                    p["path"]: p for p in self.manifest["native_backends"]
                }
                # Inspect mapped identities on every import; hash each newly
                # mapped library once here and all mapped bytes at boundaries.
                for path in native_paths():
                    if path not in planned_native:
                        raise GateError("unknown runtime native backend")
                    if path not in self.checked_native:
                        verify_artifact(planned_native[path])
                        self.checked_native.add(path)
        finally:
            self._checking = False

    def _edge(self, origin: str, target: str):
        def protected(name):
            return any(name == p or name.startswith(p + ".") for p in self.protected)

        if protected(origin) or protected(target):
            edge = (origin, target)
            if edge not in self.allowed:
                raise GateError(f"unknown import edge: {origin} -> {target}")
            row = {"from": origin, "to": target}
            if row not in self.edges:
                self.edges.append(row)

    def __enter__(self):
        self._import = builtins.__import__
        self._import_module = importlib.import_module

        def guarded(name, globals_map=None, locals_map=None, fromlist=(), level=0):
            if self._checking:
                return self._import(name, globals_map, locals_map, fromlist, level)
            origin = (globals_map or {}).get("__name__", "<external>")
            target = name
            if level:
                package = (globals_map or {}).get("__package__", "")
                target = importlib.util.resolve_name("." * level + name, package)
            self._edge(origin, target)
            result = self._import(name, globals_map, locals_map, fromlist, level)
            self.verify_import_state()
            return result

        def dynamic(name, package=None):
            origin = sys._getframe(1).f_globals.get("__name__", "<external>")
            target = (
                importlib.util.resolve_name(name, package)
                if name.startswith(".")
                else name
            )
            self._edge(origin, target)
            result = self._import_module(name, package)
            self.verify_import_state()
            return result

        builtins.__import__ = guarded
        importlib.import_module = dynamic
        return self

    def verify(self):
        self._checking = True
        try:
            return verify_runtime_closure(self.manifest, dict(sys.modules), self.edges)
        finally:
            self._checking = False

    def __exit__(self, exc_type, exc, traceback):
        builtins.__import__ = self._import
        importlib.import_module = self._import_module
        if exc is None:
            self.verify()
        return False


class ImportRecorder:
    """Discover actual import edges in an isolated import-only process.

    This recorder is never admitted for comparison execution. Its finite output
    must pass a separate cold ImportGuard verification before candidate review.
    """

    def __init__(self, protected: list[str]):
        self.protected = tuple(protected)
        self.edges = []

    def _record(self, origin, target):
        if any(
            n == p or n.startswith(p + ".")
            for n in (origin, target)
            for p in self.protected
        ):
            edge = {"from": origin, "to": target}
            if edge not in self.edges:
                self.edges.append(edge)

    def __enter__(self):
        self.original = builtins.__import__
        self.dynamic_original = importlib.import_module

        def guarded(name, globals_map=None, locals_map=None, fromlist=(), level=0):
            origin = (globals_map or {}).get("__name__", "<external>")
            target = name
            if level:
                target = importlib.util.resolve_name(
                    "." * level + name, (globals_map or {}).get("__package__", "")
                )
            self._record(origin, target)
            return self.original(name, globals_map, locals_map, fromlist, level)

        def dynamic(name, package=None):
            origin = sys._getframe(1).f_globals.get("__name__", "<external>")
            target = (
                importlib.util.resolve_name(name, package)
                if name.startswith(".")
                else name
            )
            self._record(origin, target)
            return self.dynamic_original(name, package)

        builtins.__import__ = guarded
        importlib.import_module = dynamic
        return self

    def __exit__(self, exc_type, exc, traceback):
        builtins.__import__ = self.original
        importlib.import_module = self.dynamic_original
        return False


def verify_formal_closure(manifest: dict) -> dict:
    """Require exact P/tree and Git blobs for formal gragra dependencies."""
    import re

    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("production_commit", "")):
        raise GateError("actual P commit missing from dependency closure")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("production_tree", "")):
        raise GateError("actual P tree missing from dependency closure")
    required = {
        "gragra",
        "numpy",
        "scipy",
        "math",
        "fractions",
        "decimal",
        "_decimal",
        "tests.oracles.gen1.stochastic_formal.common",
    }
    if not required <= set(manifest["protected_packages"]):
        raise GateError("formal closure omits mandatory protected scientific packages")
    if not required <= {p["package"] for p in manifest["modules"]}:
        raise GateError("formal closure omits mandatory actual scientific sources")
    if not any(p["kind"] == "import_configuration" for p in manifest["artifacts"]):
        raise GateError("formal closure omits material import configuration")
    for pin in manifest["modules"]:
        if pin["package"] != "gragra":
            continue
        if not isinstance(pin["git_blob"], str) or not re.fullmatch(
            r"[0-9a-f]{40}", pin["git_blob"]
        ):
            raise GateError("formal gragra module lacks exact Git blob")
        repo = Path(pin["root"])
        commit = manifest["production_commit"]
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", f"{commit}^{{tree}}"],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip() != manifest["production_tree"]:
            raise GateError("P tree mismatch")
        blob = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", f"{commit}:{pin['relative_path']}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        source = subprocess.run(
            ["git", "-C", str(repo), "show", f"{commit}:{pin['relative_path']}"],
            check=True,
            capture_output=True,
        ).stdout
        if (
            blob != pin["git_blob"]
            or hashlib.sha256(source).hexdigest() != pin["sha256"]
        ):
            raise GateError("planned gragra blob bytes differ from actual P")
    return verify_runtime_closure(manifest, {}, [])
