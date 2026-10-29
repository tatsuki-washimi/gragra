"""Import-only planned closure builder; this module never calls production APIs."""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import subprocess
import sys
from pathlib import Path

from .dependencies import (
    SCHEMA,
    ImportGuard,
    ImportRecorder,
    module_pin,
    native_inventory,
    python_pin,
    verify_formal_closure,
)
from .transport import GateError, artifact_pin, canonical_bytes


def source_edges(paths: list[tuple[str, Path]]) -> list[dict]:
    """Collect finite declared imports and constant dynamic import edges."""
    edges = set()
    for name, path in paths:
        if path.suffix != ".py":
            continue
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                edges.update((name, item.name) for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                target = node.module or ""
                if node.level:
                    target = importlib.util.resolve_name(
                        "." * node.level + target, package
                    )
                edges.add((name, target))
            elif isinstance(node, ast.Call) and node.args:
                function = node.func
                dynamic = (
                    isinstance(function, ast.Attribute)
                    and function.attr == "import_module"
                )
                value = node.args[0]
                if (
                    dynamic
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                ):
                    edges.add((name, value.value))
    return [{"from": a, "to": b} for a, b in sorted(edges)]


def build_inventory(
    *,
    production_repo: Path,
    modules: list[str],
    configurations: list[Path],
    comparator_repo: Path,
) -> dict:
    """Import listed modules for origins without any comparison calls."""
    repo = production_repo.resolve(strict=True)

    def git(expression):
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", expression],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    commit, tree = git("HEAD"), git("HEAD^{tree}")
    protected = [
        "gragra",
        "numpy",
        "scipy",
        "math",
        "fractions",
        "decimal",
        "_decimal",
        "tests.oracles.gen1.stochastic_formal.common",
    ]
    if any(
        name == p or name.startswith(p + ".")
        for name in sys.modules
        for p in ("gragra", "numpy", "scipy")
    ):
        raise GateError(
            "cold import-only discovery requires unloaded production/numeric packages"
        )
    with ImportRecorder(protected) as recorder:
        for name in modules + ["math", "fractions", "decimal", "_decimal"]:
            importlib.import_module(name)
    pins = []
    paths = []
    for name, module in sorted(sys.modules.items()):
        package = next(
            (p for p in protected if name == p or name.startswith(p + ".")), None
        )
        if package is None or getattr(module, "__file__", None) is None:
            continue
        path = Path(module.__file__).resolve()
        pin = module_pin(
            module,
            package=package,
            root=repo if package == "gragra" else path.parent,
            repo=repo if package == "gragra" else None,
            commit=commit if package == "gragra" else None,
        )
        # A C extension can advertise a public alias (__name__='decimal') while
        # its actual import key is '_decimal'. Audit the runtime import key.
        pin["name"] = name
        pins.append(pin)
        paths.append((name, path))
    # Every tracked gragra source is a finite planned dependency, even when an
    # endpoint imports it lazily. This does not import or execute those sources.
    known = {p["name"] for p in pins}
    from types import ModuleType

    for path in sorted((repo / "src/gragra").rglob("*.py")):
        relative = path.relative_to(repo / "src").with_suffix("")
        parts = relative.parts[:-1] if path.name == "__init__.py" else relative.parts
        name = ".".join(parts)
        paths.append((name, path))
        if name not in known:
            module = ModuleType(name)
            module.__file__ = str(path)
            pins.append(
                module_pin(
                    module, package="gragra", root=repo, repo=repo, commit=commit
                )
            )
    artifacts = [artifact_pin(p, "import_configuration") for p in configurations]
    if not artifacts:
        raise GateError("explicit material import configuration required")
    common = comparator_repo / "tests/oracles/gen1/stochastic_formal/common"
    for path in sorted(common.glob("*.py")):
        name = "tests.oracles.gen1.stochastic_formal.common." + path.stem
        paths.append((name, path))
        artifacts.append(artifact_pin(path, "comparator_source"))
        module_name = (
            "tests.oracles.gen1.stochastic_formal.common"
            if path.name == "__init__.py"
            else name
        )
        if module_name not in known:
            module = ModuleType(module_name)
            module.__file__ = str(path)
            pins.append(
                module_pin(
                    module,
                    package="tests.oracles.gen1.stochastic_formal.common",
                    root=comparator_repo,
                )
            )
    for relative in (
        "tests/oracles/gen1/compare_stochastic_production.py",
        "tests/oracles/gen1/b1_exact_record_adapter.py",
        "tests/oracles/gen1/run_stochastic_local.py",
        "tests/oracles/gen1/stochastic_attempt.py",
    ):
        path = comparator_repo / relative
        artifacts.append(artifact_pin(path, "comparator_source"))
        paths.append((relative.removesuffix(".py").replace("/", "."), path))
    packages = []
    for name in ("numpy", "scipy"):
        distribution = importlib.metadata.distribution(name)
        record = next(
            (
                p
                for p in distribution.files or []
                if str(p).endswith(".dist-info/RECORD")
            ),
            None,
        )
        packages.append(
            {
                "name": name,
                "version": distribution.version,
                "location": str(Path(distribution.locate_file("")).resolve()),
                "artifact": artifact_pin(
                    distribution.locate_file(record), "package_record"
                )
                if record
                else None,
                "artifact_missing_reason": None
                if record
                else "distribution has no RECORD inventory",
            }
        )
    edges = source_edges(paths)
    selected = [
        {"from": "tests.oracles.gen1.stochastic_formal.common.b1_connections", "to": n}
        for n in modules
    ]
    # Constant endpoint module names are reviewed declarations supplied to the
    # driver, not AST literals in its dynamic endpoint selection.
    selected += [
        {"from": "tests.oracles.gen1.stochastic_formal.common.b1_driver", "to": n}
        for n in modules
    ]
    edge_map = {(e["from"], e["to"]): e for e in edges + selected + recorder.edges}
    return {
        "schema": SCHEMA,
        "role": "production",
        "production_commit": commit,
        "production_tree": tree,
        "modules": sorted(pins, key=lambda p: p["name"]),
        "import_edges": [edge_map[e] for e in sorted(edge_map)],
        "packages": packages,
        "native_backends": native_inventory(),
        "artifacts": artifacts,
        "python": python_pin(),
        "protected_packages": protected,
    }


def verify_cold_imports(manifest: dict, modules: list[str]) -> dict:
    """Strict second-process check; import APIs only, never call production."""
    if any(
        name == p or name.startswith(p + ".")
        for name in sys.modules
        for p in ("gragra", "numpy", "scipy")
    ):
        raise GateError("strict verification requires cold production/numeric packages")
    verify_formal_closure(manifest)
    with ImportGuard(manifest) as guard:
        for name in modules + ["math", "fractions", "decimal", "_decimal"]:
            importlib.import_module(name)
        return guard.verify()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-repo", type=Path, required=True)
    parser.add_argument("--comparator-repo", type=Path, required=True)
    parser.add_argument("--module", action="append", required=True)
    parser.add_argument(
        "--import-configuration", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise GateError("fresh inventory output required")
    result = build_inventory(
        production_repo=args.production_repo,
        modules=args.module,
        configurations=args.import_configuration,
        comparator_repo=args.comparator_repo,
    )
    args.output.write_bytes(canonical_bytes(result))


if __name__ == "__main__":
    main()
