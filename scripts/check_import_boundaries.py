#!/usr/bin/env python3
"""Check static imports across gragra's numerical and adapter layers.

Pass a Python file for an edit hook, a directory for a local scan, or no argument
to scan this checkout's complete ``src/gragra`` tree.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "gragra"
HIGH_LEVEL = frozenset({"astropy", "pandas", "gwexpy", "gwpy", "matplotlib"})
FORBIDDEN_EVERYWHERE = frozenset({"nn_sus", "nnsus"})

# These are the only numerical-tree modules allowed to import high-level
# libraries. Keep the list at file granularity: exempting all of ``geometry``
# or ``analytic`` would hide imports from their existing numerical modules.
HIGH_LEVEL_SOURCE_PATHS = frozenset(
    {
        "_evaluation_setup.py",
        "_analytic_evaluation.py",
        "_spectral_from_timeseries.py",
        "_time_matrix.py",
        "_rayleigh_contract.py",
        "_rayleigh_evaluation_setup.py",
        "_phasor_matrix.py",
        "_rayleigh_response_setup.py",
        "_continuous_contract.py",
        "_continuous_rigid_motion.py",
        "_periodic_fourier.py",
        "_finite_window_fourier.py",
        "_continuous_fourier.py",
        "geometry/_appearance.py",
        "geometry/_cuboid.py",
        "geometry/_cylinder.py",
        "geometry/_frame.py",
        "geometry/_halfspace.py",
        "geometry/_sphere.py",
        "geometry/_spherical_shell.py",
        "targets/_high_points.py",
        "analytic/objects/__init__.py",
        "analytic/objects/_cuboid_model.py",
        "analytic/objects/_cylinder_model.py",
        "analytic/objects/_uniform_sphere.py",
        "analytic/objects/_gen1_motion.py",
        "analytic/seismic/__init__.py",
        "analytic/seismic/_gen1_point_force.py",
        "analytic/seismic/_gen1_sca_wall.py",
        "analytic/atmospheric/__init__.py",
        "analytic/atmospheric/_acoustic.py",
        "analytic/atmospheric/_temperature.py",
        "analytic/atmospheric/_cavity/_model.py",
        "analytic/atmospheric/_cavity/_cuboid.py",
        "analytic/atmospheric/_cavity/_cylinder.py",
        "analytic/seismic/_halfspace_psv_model.py",
        "analytic/seismic/_rayleigh.py",
        "analytic/seismic/_rayleigh_response.py",
        "analytic/seismic/_phase_speed.py",
        "analytic/seismic/_spherical_cavity_model.py",
        "analytic/seismic/_spherical_shell_model.py",
        "gravity_sources/_discretize.py",
        "gravity_sources/_force_history.py",
        "gravity_sources/_models.py",
        "gravity_sources/_rigid_motion_model.py",
        "results/_time_grid.py",
        "fields/_temperature.py",
        "evaluation.py",
        "results/_evaluation.py",
        "results/_response.py",
        "results/_comparison.py",
        "results/_binding.py",
        "results/_context.py",
        "results/_coords.py",
        "results/_time.py",
        "results/_protocols.py",
        "_plotting/__init__.py",
        "_plotting/_setup.py",
        "_plotting/_setup_shapes.py",
        "_plotting/_rayleigh_setup.py",
        "_plotting/_appearance.py",
        "_plotting/_field.py",
        "_plotting/_diagnostics.py",
        "_plotting/_hist.py",
        "_plotting/_map.py",
        "_plotting/_contribution.py",
        "_plotting/_result_field.py",
        "_plotting/_formula_latex.py",
        "plotting/__init__.py",
    }
)


def _module_for_source_path(relative: Path) -> str:
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(("gragra", *parts))


HIGH_LEVEL_MODULES = frozenset(
    _module_for_source_path(Path(path)) for path in HIGH_LEVEL_SOURCE_PATHS
)

HIGH_LEVEL_PUBLIC_IMPORTS = {
    "gragra": frozenset({"TargetPoints"}),
    "gragra.geometry": frozenset(
        {
            "Appearance",
            "Cuboid",
            "Cylinder",
            "Frame",
            "HalfSpace",
            "Sphere",
            "SphericalShell",
        }
    ),
    "gragra.targets": frozenset({"TargetPoints"}),
    "gragra.gravity_sources": frozenset(
        {
            "discretize",
            "point_sum",
            "rigid_motion",
            "ForceHistory",
            "smooth_pulse",
            "smooth_ramp_to_hold",
        }
    ),
    "gragra.fields": frozenset(
        {
            "TemperatureFourierMode",
            "TemperatureGaussian",
            "TemperaturePerturbationGrid",
        }
    ),
    "gragra.results": frozenset(
        {
            "EvaluationResult",
            "compare",
            "GravityResponse",
            "ResponseContract",
            "EvaluationContext",
            "StaticRepresentation",
            "TimeSeriesRepresentation",
            "TimeSampling",
            "ModelSpec",
            "FieldEvaluation",
            "EvaluationRecord",
        }
    ),
    "gragra.analytic.objects": frozenset(
        {
            "uniform_cuboid",
            "uniform_cylinder",
            "uniform_sphere",
            "constant_velocity",
            "translational_oscillation",
            "fixed_axis_rotation",
        }
    ),
    "gragra.analytic.seismic": frozenset(
        {
            "halfspace_psv",
            "rayleigh_halfspace",
            "rayleigh_response",
            "spherical_cavity",
            "spherical_shell",
            "small_cavity_wall",
            "point_force",
        }
    ),
    "gragra.analytic.atmospheric": frozenset(
        {
            "acoustic_halfspace",
            "isobaric_temperature",
            "acoustic_cavity_mode",
            "SphereAcousticMode",
            "CuboidAcousticMode",
            "CylinderAcousticMode",
            "AcousticCavityNumerics",
        }
    ),
}

# Package roots may route these imports only from their module-level lazy
# ``__getattr__``. Exact files, modules, and names keep ordinary numerical
# imports from gaining access to the high-level layer.
LAZY_EXPORT_IMPORTS = {
    "__init__.py": {
        "gragra.targets._high_points": frozenset({"TargetPoints"}),
    },
    "geometry/__init__.py": {
        "gragra.geometry._appearance": frozenset({"Appearance"}),
        "gragra.geometry._cuboid": frozenset({"Cuboid"}),
        "gragra.geometry._cylinder": frozenset({"Cylinder"}),
        "gragra.geometry._frame": frozenset({"Frame"}),
        "gragra.geometry._halfspace": frozenset({"HalfSpace"}),
        "gragra.geometry._sphere": frozenset({"Sphere"}),
        "gragra.geometry._spherical_shell": frozenset({"SphericalShell"}),
    },
    "targets/__init__.py": {
        "gragra.targets._high_points": frozenset({"TargetPoints"}),
    },
    "gravity_sources/__init__.py": {
        "gragra.gravity_sources._discretize": frozenset({"discretize"}),
        "gragra.gravity_sources._force_history": frozenset(
            {"ForceHistory", "smooth_pulse", "smooth_ramp_to_hold"}
        ),
        "gragra.gravity_sources._models": frozenset({"point_sum"}),
        "gragra.gravity_sources._rigid_motion_model": frozenset({"rigid_motion"}),
    },
    "fields/__init__.py": {
        "gragra.fields._temperature": frozenset(
            {
                "TemperatureFourierMode",
                "TemperatureGaussian",
                "TemperaturePerturbationGrid",
            }
        ),
    },
    "analytic/seismic/__init__.py": {
        "gragra.analytic.seismic._gen1_sca_wall": frozenset({"small_cavity_wall"}),
        "gragra.analytic.seismic._gen1_point_force": frozenset({"point_force"}),
    },
    "analytic/objects/__init__.py": {
        "gragra.analytic.objects._gen1_motion": frozenset(
            {
                "constant_velocity",
                "translational_oscillation",
                "fixed_axis_rotation",
            }
        ),
    },
    "analytic/atmospheric/__init__.py": {
        "gragra.analytic.atmospheric._acoustic": frozenset({"acoustic_halfspace"}),
        "gragra.analytic.atmospheric._temperature": frozenset({"isobaric_temperature"}),
        "gragra.analytic.atmospheric._cavity._model": frozenset(
            {
                "acoustic_cavity_mode",
                "SphereAcousticMode",
                "CuboidAcousticMode",
                "CylinderAcousticMode",
                "AcousticCavityNumerics",
            }
        ),
    },
    "results/__init__.py": {
        "gragra.results._evaluation": frozenset({"EvaluationResult"}),
        "gragra.results._comparison": frozenset({"compare"}),
        "gragra.results._response": frozenset({"GravityResponse", "ResponseContract"}),
        "gragra.results._context": frozenset({"EvaluationContext"}),
        "gragra.results._representation": frozenset(
            {"StaticRepresentation", "TimeSeriesRepresentation"}
        ),
        "gragra.results._time": frozenset({"TimeSampling"}),
        "gragra.results._protocols": frozenset({"ModelSpec", "FieldEvaluation"}),
        "gragra.results._record": frozenset({"EvaluationRecord"}),
        "gragra.exceptions": frozenset({"NumericalDiagnostic"}),
    },
}


def _source_relative(path: Path) -> Path | None:
    """Return a path beneath any ``src/gragra`` checkout, including fixtures."""
    parts = path.resolve().parts
    for index in range(len(parts) - 1):
        if parts[index : index + 2] == ("src", "gragra"):
            return Path(*parts[index + 2 :])
    return None


def _import_from_module(node: ast.ImportFrom, relative: Path) -> str:
    if not node.level:
        return node.module or ""
    package = ["gragra", *relative.parent.parts]
    if node.level > len(package):
        return ""
    parent = package[: len(package) - node.level + 1]
    if node.module:
        return ".".join([*parent, *node.module.split(".")])
    return ".".join(parent)


def _is_adapter(module: str) -> bool:
    return module == "gragra.adapters" or module.startswith("gragra.adapters.")


class _ImportVisitor(ast.NodeVisitor):
    def __init__(self, relative: Path) -> None:
        self.relative = relative
        self.numerical = relative.parts[0] != "adapters"
        self.high_level_source = relative.as_posix() in HIGH_LEVEL_SOURCE_PATHS
        self.direct_type_checking: set[str] = set()
        self.typing_names: set[str] = set()
        self.scope_stack: list[tuple[str, str]] = []
        self.in_type_checking = False
        self.violations: list[tuple[int, str]] = []

    def _unbind(self, name: str) -> None:
        self.direct_type_checking.discard(name)
        self.typing_names.discard(name)

    def _is_type_checking_test(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.direct_type_checking
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "TYPE_CHECKING"
            and isinstance(node.value, ast.Name)
            and node.value.id in self.typing_names
        )

    def _check(
        self,
        module: str,
        line: int,
        *,
        high_level_symbol: bool = False,
        lazy_export: bool = False,
    ) -> None:
        root = module.split(".", 1)[0]
        if (
            root in FORBIDDEN_EVERYWHERE
            or self.numerical
            and root in HIGH_LEVEL
            and not self.high_level_source
            or self.numerical
            and not self.in_type_checking
            and _is_adapter(module)
            or self.numerical
            and not self.high_level_source
            and not self.in_type_checking
            and not lazy_export
            and (module in HIGH_LEVEL_MODULES or high_level_symbol)
        ):
            self.violations.append((line, module))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check(alias.name, node.lineno)
            bound = alias.asname or alias.name.split(".", 1)[0]
            self._unbind(bound)
            if alias.name in {"typing", "typing_extensions"}:
                self.typing_names.add(bound)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = _import_from_module(node, self.relative)
        aliases = frozenset(alias.name for alias in node.names)
        allowed_lazy_names = LAZY_EXPORT_IMPORTS.get(self.relative.as_posix(), {}).get(
            module, frozenset()
        )
        lazy_export = (
            self.scope_stack == [("function", "__getattr__")]
            and bool(aliases)
            and aliases <= allowed_lazy_names
        )
        self._check(module, node.lineno, lazy_export=lazy_export)
        if module == "gragra":
            for alias in node.names:
                if alias.name == "adapters":
                    self._check("gragra.adapters", node.lineno)
        public_imports = HIGH_LEVEL_PUBLIC_IMPORTS.get(module, frozenset())
        for alias in node.names:
            imported_module = f"{module}.{alias.name}" if module else ""
            if imported_module in HIGH_LEVEL_MODULES:
                self._check(imported_module, node.lineno)
            elif alias.name in public_imports:
                self._check(
                    imported_module,
                    node.lineno,
                    high_level_symbol=True,
                )
            bound = alias.asname or alias.name
            self._unbind(bound)
            if (
                module in {"typing", "typing_extensions"}
                and alias.name == "TYPE_CHECKING"
            ):
                self.direct_type_checking.add(bound)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._unbind(node.id)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type:
            self.visit(node.type)
        if node.name:
            self._unbind(node.name)
        for item in node.body:
            self.visit(item)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.pattern:
            self.visit(node.pattern)
        if node.name:
            self._unbind(node.name)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name:
            self._unbind(node.name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        self.generic_visit(node)
        if node.rest:
            self._unbind(node.rest)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            node.attr == "TYPE_CHECKING"
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and isinstance(node.value, ast.Name)
        ):
            self.typing_names.discard(node.value.id)
        self.generic_visit(node)

    def _visit_scoped_body(
        self, body: list[ast.stmt], parameters: set[str] | None = None
    ) -> None:
        outer_direct = self.direct_type_checking
        outer_typing = self.typing_names
        self.direct_type_checking = outer_direct.copy()
        self.typing_names = outer_typing.copy()
        for name in parameters or ():
            self._unbind(name)
        for item in body:
            self.visit(item)
        self.direct_type_checking = outer_direct
        self.typing_names = outer_typing

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._unbind(node.name)
        parameters = {
            arg.arg
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        }
        if node.args.vararg:
            parameters.add(node.args.vararg.arg)
        if node.args.kwarg:
            parameters.add(node.args.kwarg.arg)
        self.scope_stack.append(("function", node.name))
        self._visit_scoped_body(node.body, parameters)
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._unbind(node.name)
        self.scope_stack.append(("class", node.name))
        self._visit_scoped_body(node.body)
        self.scope_stack.pop()

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        was_type_checking = self.in_type_checking
        before_direct = self.direct_type_checking.copy()
        before_typing = self.typing_names.copy()
        guarded = self._is_type_checking_test(node.test)
        negated = (
            isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and self._is_type_checking_test(node.test.operand)
        )
        self.in_type_checking = was_type_checking or guarded
        for item in node.body:
            self.visit(item)
        body_direct = self.direct_type_checking.copy()
        body_typing = self.typing_names.copy()
        self.direct_type_checking = before_direct
        self.typing_names = before_typing
        self.in_type_checking = was_type_checking or negated
        for item in node.orelse:
            self.visit(item)
        self.direct_type_checking.intersection_update(body_direct)
        self.typing_names.intersection_update(body_typing)
        self.in_type_checking = was_type_checking


def check_file(path: Path) -> list[str]:
    """Return ``path:line: module`` diagnostics for a single source file."""
    relative = _source_relative(path)
    if relative is None or path.suffix != ".py":
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        return [f"{path}:{error.lineno}: syntax error: {error.msg}"]
    visitor = _ImportVisitor(relative)
    visitor.visit(tree)
    return [
        f"{path}:{line}: forbidden import {module}"
        for line, module in visitor.violations
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=SRC_ROOT)
    args = parser.parse_args()
    target: Path = args.path
    if not target.exists():
        parser.error(f"path does not exist: {target}")
    paths = sorted(target.rglob("*.py")) if target.is_dir() else [target]
    violations = [diagnostic for path in paths for diagnostic in check_file(path)]
    for diagnostic in violations:
        print(diagnostic)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
