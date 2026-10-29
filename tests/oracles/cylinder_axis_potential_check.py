"""Run high-precision cylinder regressions in a disposable Python process.

The mpmath oracle can leave allocator arenas resident in the pytest parent.
That unrelated lifetime peak can contaminate resource-budget tests later in the
same test session, so each independent-reference case runs in a fresh process.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def run_isolated_reference_case(scenario: str, *arguments: object) -> bool:
    """Run one reference regression in a fresh interpreter.

    Return ``False`` when the optional mpmath dependency is unavailable.
    """
    if importlib.util.find_spec("mpmath") is None:
        return False

    script_path = Path(__file__).resolve()
    repository_root = script_path.parents[2]
    environment = os.environ.copy()
    pythonpath = [str(repository_root / "src"), str(repository_root / "tests")]
    inherited_pythonpath = environment.get("PYTHONPATH")
    if inherited_pythonpath:
        pythonpath.append(inherited_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)

    completed = subprocess.run(
        [sys.executable, str(script_path), scenario, *(str(v) for v in arguments)],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"isolated cylinder reference case {scenario!r} failed "
            f"(exit {completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return True


def _verify_import_source(repository_root: Path) -> None:
    sys.path.insert(0, str(repository_root / "tests"))
    sys.path.insert(0, str(repository_root / "src"))

    import gragra
    from gragra.analytic import cavity_cylinder

    expected_package = repository_root / "src" / "gragra" / "__init__.py"
    expected_cylinder = (
        repository_root / "src" / "gragra" / "analytic" / "cavity_cylinder.py"
    )
    if Path(gragra.__file__).resolve() != expected_package.resolve():
        raise RuntimeError(f"unexpected gragra import: {gragra.__file__}")
    if Path(cavity_cylinder.__file__).resolve() != expected_cylinder.resolve():
        raise RuntimeError(f"unexpected cylinder import: {cavity_cylinder.__file__}")


def _thin_axis(z: float) -> None:
    import numpy as np
    from oracles.cylinder_axis_potential_reference import volume_potential

    from gragra.analytic import (
        cylinder_axis_observables_g_free,
        cylinder_potential_axis_g_free,
    )

    reference, _ = volume_potential(1, 1e-8, 1800, z)
    expected = float(reference)
    actual = cylinder_potential_axis_g_free([0, 0, 0], 1, 1e-8, 1800, [[0, 0, z]])
    np.testing.assert_allclose(actual, expected, rtol=16 * np.finfo(float).eps, atol=0)
    combined = cylinder_axis_observables_g_free([0, 0, 0], 1, 1e-8, 1800, [[0, 0, z]])
    np.testing.assert_array_equal(combined.potential, actual)
    assert actual.dtype == np.float64
    assert actual.shape == (1,)
    assert not actual.flags.writeable


def _height_sweep(height: float) -> None:
    import numpy as np
    from oracles.cylinder_axis_potential_reference import (
        center_potential,
        volume_potential,
    )

    from gragra.analytic import cylinder_potential_axis_g_free

    z_values = np.array(
        [
            0,
            height / 4,
            -height / 4,
            height / 2 * (1 - 1e-6),
            height / 2 * (1 + 1e-6),
            2 + height,
            1e6 + height,
        ]
    )
    targets = np.column_stack((np.zeros((len(z_values), 2)), z_values))
    actual = cylinder_potential_axis_g_free([0, 0, 0], 1, height, 1800, targets)
    for z, value in zip(z_values, actual, strict=True):
        low, _ = volume_potential(1, height, 1800, float(z), dps=100)
        high, _ = volume_potential(1, height, 1800, float(z), dps=100, refinement=1)
        scale = 1800 + abs(high)
        assert abs(low - high) / scale <= 1e-11
        assert abs(float(value) - high) / scale <= 1e-10
        if z == 0:
            assert abs(center_potential(1, height, 1800) - high) / scale <= 1e-11


def _center_refinement() -> None:
    from oracles.cylinder_axis_potential_reference import (
        center_axial_potential,
        center_potential,
        volume_potential,
    )

    previous = None
    for dps in (100, 160, 240):
        low, _ = volume_potential(1, 1e-8, 1800, 0, dps=dps)
        high, _ = volume_potential(1, 1e-8, 1800, 0, dps=dps, refinement=1)
        closed = center_potential(1, 1e-8, 1800, dps=dps)
        axial = center_axial_potential(1, 1e-8, 1800, dps=dps)
        axial_high = center_axial_potential(1, 1e-8, 1800, dps=dps, refinement=1)
        scale = 1800 + abs(high)
        assert abs(low - high) / scale <= 1e-11
        assert abs(closed - high) / scale <= 1e-11
        assert abs(axial - axial_high) / scale <= 1e-11
        assert abs(axial_high - high) / scale <= 1e-11
        if previous is not None:
            assert abs(previous - high) / scale <= 1e-11
        previous = high


def _shared_observables() -> None:
    import numpy as np
    from oracles.cylinder_axis_potential_reference import center_potential

    from gragra.analytic import (
        cylinder_analytic_observables_g_free,
        cylinder_axis_observables_g_free,
        cylinder_observables_g_free,
    )

    expected = float(center_potential(1, 1e-8, 1800))
    for api in (
        cylinder_analytic_observables_g_free,
        cylinder_axis_observables_g_free,
        cylinder_observables_g_free,
    ):
        result = api([0, 0, 0], 1, 1e-8, 1800, [[0, 0, 0]])
        np.testing.assert_allclose(
            result.potential, expected, rtol=16 * np.finfo(float).eps, atol=0
        )
        assert result.potential.dtype == np.float64
        assert result.potential.shape == (1,)
        assert not result.potential.flags.writeable


def _main() -> None:
    script_path = Path(__file__).resolve()
    repository_root = script_path.parents[2]
    _verify_import_source(repository_root)

    if len(sys.argv) < 2:
        raise SystemExit("expected a cylinder reference scenario")
    scenario = sys.argv[1]
    if scenario == "thin-axis" and len(sys.argv) == 3:
        _thin_axis(float(sys.argv[2]))
    elif scenario == "height-sweep" and len(sys.argv) == 3:
        _height_sweep(float(sys.argv[2]))
    elif scenario == "center-refinement" and len(sys.argv) == 2:
        _center_refinement()
    elif scenario == "shared-observables" and len(sys.argv) == 2:
        _shared_observables()
    else:
        raise SystemExit(f"invalid cylinder reference invocation: {sys.argv[1:]}")


if __name__ == "__main__":
    _main()
