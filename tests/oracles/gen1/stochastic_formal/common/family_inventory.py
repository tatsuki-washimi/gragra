"""Finite family endpoint/native dependency mapping; no scientific evaluations."""

from __future__ import annotations

import ast
from pathlib import Path

from .family_adapter import _API
from .inventory import build_inventory
from .transport import GateError


def build_family_inventory(
    *, family, production_repo, comparator_repo, configurations, connection_module
):
    if family not in _API or not connection_module.startswith(
        "tests.oracles.gen1.stochastic_formal.common.family_connections_"
    ):
        raise GateError("finite known family and reviewed connection module required")
    package, _ = _API[family]
    modules = [
        "numpy",
        "scipy.special",
        "gragra.constants",
        "gragra.analytic.stochastic_" + package,
        "gragra.observables.stochastic_" + package,
    ]
    manifest = build_inventory(
        production_repo=Path(production_repo),
        comparator_repo=Path(comparator_repo),
        modules=modules,
        configurations=configurations,
    )
    planned = {p["name"]: p for p in manifest["modules"]}
    if connection_module not in planned:
        raise GateError("reviewed family plugin source absent from comparator tree")
    edges = {(e["from"], e["to"]): e for e in manifest["import_edges"]}
    origin = "tests.oracles.gen1.stochastic_formal.common.family_adapter"
    for target in modules + [connection_module]:
        edges[(origin, target)] = {"from": origin, "to": target}
    # NumPy ndarray reduction dispatch is lazy in native code. Declare the
    # finite actual caller->implementation edge at pinned AST callsites.
    target = "numpy._core._methods"
    if target not in planned:
        raise GateError("NumPy reduction implementation not inventoried")
    for name, pin in planned.items():
        path = Path(pin["path"])
        if (
            pin["package"]
            not in {"gragra", "tests.oracles.gen1.stochastic_formal.common"}
            or path.suffix != ".py"
        ):
            continue
        calls = [
            n
            for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr
            in {"all", "any", "max", "min", "sum", "mean", "prod", "std", "var", "ptp"}
        ]
        if calls:
            edges[(name, target)] = {"from": name, "to": target}
    manifest["import_edges"] = [edges[e] for e in sorted(edges)]
    return manifest
