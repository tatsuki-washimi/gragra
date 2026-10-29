"""SciPy is a required runtime dependency, confined to the analytic models.

Decision S1 in docs/architecture/numerical_dependency_adr.md: a module-level
``scipy`` import is allowed only under ``analytic/``. Imports inside functions
(lazy imports) are allowed where an optional extra needs them, such as
``adapters/lebedev.py``.
"""

import ast
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "gragra"


class _ModuleLevelScipy(ast.NodeVisitor):
    def __init__(self):
        self.depth = 0
        self.lines = []

    def visit_FunctionDef(self, node):
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_Import(self, node):
        self._record(node, [alias.name for alias in node.names])

    def visit_ImportFrom(self, node):
        if node.level == 0 and node.module:
            self._record(node, [node.module])

    def _record(self, node, names):
        if self.depth == 0 and any(_root(name) == "scipy" for name in names):
            self.lines.append(node.lineno)


def _root(name):
    return name.split(".")[0]


def _module_level_scipy_sites():
    sites = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        scan = _ModuleLevelScipy()
        scan.visit(ast.parse(path.read_text(encoding="utf-8")))
        if scan.lines:
            sites[path.relative_to(SRC_ROOT).as_posix()] = scan.lines
    return sites


def test_module_level_scipy_import_is_confined_to_analytic_models():
    sites = _module_level_scipy_sites()
    offenders = {
        rel: lines for rel, lines in sites.items() if not rel.startswith("analytic/")
    }
    assert offenders == {}


def test_analytic_special_function_models_use_scipy_at_module_level():
    sites = _module_level_scipy_sites()
    assert sites, "expected the analytic special-function models to import scipy"
    assert all(rel.startswith("analytic/") for rel in sites)


def test_scipy_is_a_required_runtime_dependency():
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = pyproject["project"]["dependencies"]
    assert any(dep.startswith("scipy") for dep in dependencies)
