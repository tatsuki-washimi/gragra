"""Lazy-import contracts for ``gragra.plotting``, checked in fresh interpreters.

Each check runs in a subprocess with ``-I`` so that this test session's already
imported modules cannot hide an eager import.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CODE_IMPORT_ONLY = """
import json
import sys

import gragra.plotting  # noqa: F401

heavy = sorted(m for m in sys.modules if m.split(".")[0] in ("matplotlib", "gwexpy"))
print(json.dumps(heavy))
"""

CODE_IMPORT_HIST = """
import json
import sys

from gragra.plotting import hist, plot_map  # noqa: F401
import gragra._plotting._hist  # noqa: F401
import gragra._plotting._map  # noqa: F401
import gragra._plotting._diagnostics  # noqa: F401

names = ("matplotlib", "matplotlib.pyplot", "gwexpy")
print(json.dumps({name: name in sys.modules for name in names}))
"""

CODE_IMPORT_SOURCE_DIAGNOSTICS = """
import json
import sys

from gragra.plotting import (
    ContributionFrequencyStatistic,
    ContributionRMSStatistics,
    ContributionStatistics,
    ContributionValidityMask,
    DirectionalContributionStatistics,
    DynamicRadialHistogramResult,
    GravityGradientRadialHistogramResult,
    RadialHistogramResult,
    SourceContributionResult,
    SourceFieldResult,
    contribution_statistics,
    directional_contribution_statistics,
    dynamic_radial_histogram,
    gravity_gradient_radial_histogram,
    radial_histogram,
    source_contributions,
    source_field,
    source_field_result,
)

names = ("matplotlib", "matplotlib.pyplot", "gwexpy")
after_exports = {name: name in sys.modules for name in names}

import numpy as np
from gragra.fields.density_series import DensityPhasors

phasors = DensityPhasors(
    np.array([[0.0, 0.0, 0.0]]),
    np.array([1.0]),
    np.array([[1.0 + 0.5j]]),
    np.array([1.0]),
    0.0,
    np.zeros(3),
    "root",
    np.zeros(3),
    ("s0",),
)
result = source_contributions(phasors, "acceleration", [[0.0, 0.0, 2.0]])
after_contribution_call = {name: name in sys.modules for name in names}
assert isinstance(result, SourceContributionResult)
assert callable(contribution_statistics)
assert callable(directional_contribution_statistics)
assert callable(dynamic_radial_histogram)
assert callable(gravity_gradient_radial_histogram)
assert callable(radial_histogram)
assert callable(source_field)
assert callable(source_field_result)
assert all(
    isinstance(value, type)
    for value in (
        ContributionFrequencyStatistic,
        ContributionRMSStatistics,
        ContributionStatistics,
        ContributionValidityMask,
        DirectionalContributionStatistics,
        DynamicRadialHistogramResult,
        GravityGradientRadialHistogramResult,
        RadialHistogramResult,
        SourceFieldResult,
    )
)
print(json.dumps({
    "after_exports": after_exports,
    "after_contribution_call": after_contribution_call,
}))
"""

CODE_HIST = """
import json
import sys

NAMES = ("matplotlib", "matplotlib.pyplot", "gwexpy")


def loaded():
    return {name: name in sys.modules for name in NAMES}


from gragra.plotting import hist

after_import = loaded()

import numpy as np
from astropy import units as u

from gragra.geometry import Frame
from gragra.gravity_sources import WeightedPointSource
from gragra.gravity_sources._models import point_sum
from gragra.results import Contribution, GravityFieldDefinition, PotentialReference
from gragra.targets import TargetPoints

root = Frame("root")
field = GravityFieldDefinition.source_field(
    "s", Contribution("total"), PotentialReference.ZERO_AT_INFINITY
)
setup = point_sum(
    WeightedPointSource([[0.0, 0.0, 0.0]], [1.0]), frame=root, field_definition=field
).at(TargetPoints(np.array([[3.0, 4.0, 0.0]]) * u.m, ("A",), root))

after_setup = loaded()
result = hist(setup, "distance", bins=[0.0, 10.0])
after_call = loaded()
print(json.dumps({
    "after_import": after_import,
    "after_setup": after_setup,
    "after_call": after_call,
    "values": [float(v) for v in result.values.value],
}))
"""

CODE_PLOT_FIELD = """
import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
from astropy import units as u

from gragra.fields.regular_grid import DensityPerturbationGrid
from gragra.geometry import Frame
from gragra.plotting import plot_field

root = Frame("root")
frame = Frame("lab", np.zeros(3) * u.m, np.eye(3), root)
grid = DensityPerturbationGrid(
    np.array([0.0, 1.0]), np.array([0.0, 2.0]), np.array([0.0, 3.0]), np.ones((2, 2, 2))
)
fig = plot_field(grid, frame=frame, plane="xy", index=0)
print(json.dumps({"figure": type(fig).__name__}))
"""


def _run(code: str) -> dict | list:
    env = {**os.environ, "MPLBACKEND": "Agg"}
    source_root = REPO_ROOT / "src"
    bootstrap = f"""
import pathlib
import sys
import sysconfig

for key in ("purelib", "platlib"):
    dependency_path = sysconfig.get_paths()[key]
    if dependency_path not in sys.path:
        sys.path.insert(0, dependency_path)
sys.path.insert(0, {str(source_root)!r})
import gragra
assert pathlib.Path(gragra.__file__).resolve().is_relative_to(
    pathlib.Path({str(source_root)!r}).resolve()
), gragra.__file__
{code}
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", bootstrap],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=300,
    )
    # Surface the child's traceback so a failure names its cause.
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_importing_gragra_plotting_loads_no_matplotlib_or_gwexpy() -> None:
    assert _run(CODE_IMPORT_ONLY) == []


def test_importing_hist_modules_loads_no_pyplot_or_gwexpy() -> None:
    loaded = _run(CODE_IMPORT_HIST)

    assert loaded == {
        "matplotlib": False,
        "matplotlib.pyplot": False,
        "gwexpy": False,
    }


def test_source_diagnostics_exports_and_numeric_call_stay_cold() -> None:
    result = _run(CODE_IMPORT_SOURCE_DIAGNOSTICS)
    expected = {
        "matplotlib": False,
        "matplotlib.pyplot": False,
        "gwexpy": False,
    }
    assert result == {
        "after_exports": expected,
        "after_contribution_call": expected,
    }


def test_hist_call_is_where_gwexpy_and_pyplot_are_loaded() -> None:
    result = _run(CODE_HIST)

    # Building the setup loads matplotlib (not pyplot) through gragra.results,
    # and it does not load gwexpy or pyplot.
    assert result["after_import"] == {
        "matplotlib": False,
        "matplotlib.pyplot": False,
        "gwexpy": False,
    }
    assert result["after_setup"]["matplotlib.pyplot"] is False
    assert result["after_setup"]["gwexpy"] is False
    # The documented, accepted fact: hist() itself loads GWexpy, whose Histogram
    # import reads Matplotlib pyplot.
    assert result["after_call"]["gwexpy"] is True
    assert result["after_call"]["matplotlib.pyplot"] is True
    assert result["values"] == [1.0]


def test_plot_field_still_works_after_lazy_export() -> None:
    result = _run(CODE_PLOT_FIELD)

    assert result == {"figure": "Figure"}
