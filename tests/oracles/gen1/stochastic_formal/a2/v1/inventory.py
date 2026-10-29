"""A2-only planned source inventory layered on the generic family inventory."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

from ...common.dependencies import module_pin
from ...common.family_inventory import build_family_inventory
from ...common.inventory import source_edges
from ...common.transport import GateError, artifact_pin

PACKAGE = "tests.oracles.gen1.stochastic_formal.a2.v1"


def build_a2_family_inventory(
    *, production_repo, comparator_repo, configurations, connection_module
):
    """Add only A2 versioned sources to a generic, complete planned closure."""
    if connection_module != PACKAGE + ".family_connections":
        raise GateError("versioned A2 connection module required")
    comparator_repo = Path(comparator_repo).resolve(strict=True)
    versioned_root = (
        comparator_repo / "tests/oracles/gen1/stochastic_formal/a2/v1"
    )
    budget_root = comparator_repo / "tests/oracles/gen1/stochastic_formal/a2"
    if not versioned_root.is_dir() or not (budget_root / "budget.py").is_file():
        raise GateError("versioned A2 implementation sources are incomplete")

    # The generic builder admits only a reviewed common-family connection plug-in.
    # The A2-specific plug-in is added below under its versioned module identity.
    manifest = build_family_inventory(
        family="A2",
        production_repo=production_repo,
        comparator_repo=comparator_repo,
        configurations=configurations,
        connection_module="tests.oracles.gen1.stochastic_formal.common.family_connections_acoustic",
    )
    planned = {pin["name"]: pin for pin in manifest["modules"]}
    artifacts = list(manifest["artifacts"])
    source_paths = []

    for path in sorted(versioned_root.glob("*.py")):
        name = PACKAGE if path.name == "__init__.py" else PACKAGE + "." + path.stem
        module = ModuleType(name)
        module.__file__ = str(path)
        planned[name] = module_pin(module, package=PACKAGE, root=versioned_root)
        artifacts.append(artifact_pin(path, "comparator_source"))
        source_paths.append((name, path))

    budget_path = budget_root / "budget.py"
    budget_module = ModuleType(PACKAGE + ".budget")
    budget_module.__file__ = str(budget_path)
    planned[budget_module.__name__] = module_pin(
        budget_module,
        package=PACKAGE,
        root=comparator_repo,
    )
    artifacts.append(artifact_pin(budget_path, "comparator_source"))
    source_paths.append((budget_module.__name__, budget_path))

    edges = {(edge["from"], edge["to"]): edge for edge in manifest["import_edges"]}
    for edge in source_edges(source_paths):
        edges[(edge["from"], edge["to"])] = edge
    common_family_adapter = (
        "tests.oracles.gen1.stochastic_formal.common.family_adapter"
    )
    generic_acoustic_plugin = (
        "tests.oracles.gen1.stochastic_formal.common.family_connections_acoustic"
    )
    edges.pop((common_family_adapter, generic_acoustic_plugin), None)
    for origin, target in (
        (common_family_adapter, PACKAGE + ".family_connections"),
        (PACKAGE + ".family_driver", PACKAGE + ".family_execution"),
        (PACKAGE + ".family_driver", PACKAGE + ".family_connections"),
        (PACKAGE + ".family_driver", PACKAGE + ".resource_adapter"),
        (PACKAGE + ".family_execution", PACKAGE + ".resource_adapter"),
        (PACKAGE + ".family_supervisor", PACKAGE + ".resource_adapter"),
        (PACKAGE + ".resource_adapter", PACKAGE + ".budget"),
    ):
        edges[(origin, target)] = {"from": origin, "to": target}

    manifest["modules"] = sorted(planned.values(), key=lambda pin: pin["name"])
    manifest["protected_packages"] = sorted(
        set(manifest["protected_packages"]) | {PACKAGE}
    )
    manifest["artifacts"] = artifacts
    manifest["import_edges"] = [edges[key] for key in sorted(edges)]
    return manifest
