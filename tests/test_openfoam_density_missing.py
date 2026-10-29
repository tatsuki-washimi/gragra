# ruff: noqa: E501, I001

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if type(finder).__name__ != "ScikitBuildRedirectingFinder"
]
sys.path.insert(0, str(_ROOT / "src"))

from gragra.adapters.openfoam_density import load_openfoam_density  # noqa: E402
from gragra.observables import (  # noqa: E402
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)
from tests._openfoam_synthetic import write_two_hex  # noqa: E402
from tests.test_openfoam_density import KW  # noqa: E402

import gragra  # noqa: E402
import gragra.adapters.openfoam_density as _openfoam_density  # noqa: E402

assert gragra.__file__.startswith(str(_ROOT / "src"))
assert _openfoam_density.__file__.startswith(str(_ROOT / "src"))

requires_meshio = pytest.mark.skipif(
    importlib.util.find_spec("meshio") is None,
    reason="requires the optional gragra[mesh] dependency",
)


@requires_meshio
def test_rejects_wrong_suffix_units_and_id_permutation(tmp_path):
    p = write_two_hex(tmp_path / "bad.vtk", ids=(0, 0))
    with pytest.raises(ValueError, match="filename"):
        load_openfoam_density(p, **KW)
    p = write_two_hex(tmp_path / "w6_0.vtk", ids=(0, 0))
    with pytest.raises(ValueError, match="permutation"):
        load_openfoam_density(p, **KW)
    with pytest.raises(ValueError, match="canonical"):
        load_openfoam_density(p, **{**KW, "density_unit": "g/cm3"})


def test_lazy_meshio_message_when_dependency_missing(monkeypatch, tmp_path):
    p = write_two_hex(tmp_path / "w6_0.vtk")
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
    with pytest.raises(ImportError, match="gragra\\[mesh\\]"):
        load_openfoam_density(p, **KW)


def test_subprocess_without_meshio_imports_adapters_and_reports_loader_extra(tmp_path):
    child = """
import importlib.util
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
vtk = Path(sys.argv[2])
sys.meta_path[:] = [x for x in sys.meta_path if type(x).__name__ != 'ScikitBuildRedirectingFinder']
sys.path.insert(0, str(root / 'src'))
original = importlib.util.find_spec
importlib.util.find_spec = lambda name, *a, **k: None if name == 'meshio' else original(name, *a, **k)
import gragra
import gragra.adapters
from gragra.adapters.openfoam_density import load_openfoam_density
try:
    load_openfoam_density(vtk, field_name='delta_rho', time_s=0.0, time_name='0', coordinate_frame='case', coordinate_origin_m=[0, 0, 0], density_unit='kg/m^3', coordinate_unit='m', volume_unit='m^3', time_unit='s', provenance={'distribution':'OpenFOAM Foundation','version':'14','patch':'20260724','solver':'manufactured-no-solver','compressibility':'not-applicable','region':'region0','reconstructed':True,'exporter':'foamToVTK','export_command':'foamToVTK'})
except ImportError as exc:
    assert 'gragra[mesh]' in str(exc)
    print(json.dumps({'gragra': gragra.__file__, 'adapters': gragra.adapters.__file__, 'loader': load_openfoam_density.__module__, 'meshio_loaded': 'meshio' in sys.modules}))
else:
    raise AssertionError('loader did not require gragra[mesh]')
"""
    vtk = tmp_path / "blocked_0.vtk"
    vtk.write_text(
        "# vtk DataFile Version 3.0\nblocked\nASCII\nDATASET UNSTRUCTURED_GRID\n",
        encoding="ascii",
    )
    result = subprocess.run(
        [sys.executable, "-c", child, str(_ROOT), str(vtk)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    origins = json.loads(result.stdout)
    assert origins["gragra"].startswith(str(_ROOT / "src"))
    assert origins["adapters"].startswith(str(_ROOT / "src"))
    assert origins["loader"] == "gragra.adapters.openfoam_density"
    assert origins["meshio_loaded"] is False


def test_rejects_mixed_boolean_origin_and_non_header_ascii(tmp_path):
    p = write_two_hex(tmp_path / "w6_0.vtk")
    with pytest.raises(TypeError, match="bool"):
        load_openfoam_density(p, **{**KW, "coordinate_origin_m": [True, 0, 0]})
    p.write_bytes(b"not a VTK header; ASCII UNSTRUCTURED_GRID in a comment\n")
    with pytest.raises(ValueError, match="legacy ASCII"):
        load_openfoam_density(p, **KW)


@pytest.mark.parametrize("field", ["C", "Vc", "delta_rho"])
@requires_meshio
def test_rejects_each_missing_required_cell_field(tmp_path, field):
    p = write_two_hex(tmp_path / "w6_0.vtk")
    text = p.read_text(encoding="ascii")
    marker = f"VECTORS {field} float" if field == "C" else f"SCALARS {field} float 1"
    renamed = f"VECTORS missing_{field} float" if field == "C" else f"SCALARS missing_{field} float 1"
    p.write_text(text.replace(marker, renamed), encoding="ascii")
    with pytest.raises(ValueError, match="missing required"):
        load_openfoam_density(p, **KW)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda text: text.replace("-1.0\n2.0", "nan\n2.0"),
            "finite",
        ),
        (
            lambda text: text.replace(
                "1.0 1 1\n0.0 1 1", "1.0 1 1.25\n0.0 1 1"
            ),
            "nonaffine",
        ),
        (
            lambda text: text.replace(
                "0.0 0 1\n1.0 0 1\n1.0 1 1\n0.0 1 1",
                "0.0 0 0\n1.0 0 0\n1.0 1 0\n0.0 1 0",
            ),
            "degenerate",
        ),
        (
            lambda text: text.replace("0.5 0.5 0.5\n1.5 0.5 0.5", "0.6 0.5 0.5\n1.5 0.5 0.5"),
            "geometry tolerance",
        ),
    ],
)
@requires_meshio
def test_rejects_bad_native_shape_or_geometry(tmp_path, mutate, error):
    p = write_two_hex(tmp_path / "w6_0.vtk")
    p.write_text(mutate(p.read_text(encoding="ascii")), encoding="ascii")
    with pytest.raises((TypeError, ValueError), match=error):
        load_openfoam_density(p, **KW)


@requires_meshio
def test_rejects_scalar_native_centres(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        load_openfoam_density(
            write_two_hex(tmp_path / "w6_0.vtk", scalar_centres=True), **KW
        )


@pytest.mark.parametrize("field_name", ["rho", "delta_rho"])
@requires_meshio
def test_rejects_vector_density_payload(tmp_path, field_name):
    p = write_two_hex(tmp_path / "w6_0.vtk")
    values = "999 0 0\n1002 0 0" if field_name == "rho" else "-1 0 0\n2 0 0"
    scalar = (
        f"SCALARS {field_name} float 1\nLOOKUP_TABLE default\n"
        + ("999\n1002" if field_name == "rho" else "-1.0\n2.0")
    )
    p.write_text(
        p.read_text(encoding="ascii").replace(
            scalar, f"VECTORS {field_name} float\n{values}"
        ),
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="shape"):
        load_openfoam_density(p, **{**KW, "field_name": field_name})


@requires_meshio
def test_rejects_negative_rho_and_accepts_exact_zero_delta(tmp_path):
    rho = write_two_hex(tmp_path / "rho_0.vtk")
    rho.write_text(
        rho.read_text(encoding="ascii").replace("999\n1002", "-1\n1002"),
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="density"):
        load_openfoam_density(rho, **{**KW, "field_name": "rho"})
    zero = load_openfoam_density(
        write_two_hex(tmp_path / "zero_0.vtk", density=(0.0, 0.0)), **KW
    )
    source = zero.as_volume_element_source().as_weighted_source()
    targets = np.array([[3.0, 0.7, -0.4]])
    for observable in (point_potential, point_acceleration, point_gravity_gradient):
        assert np.array_equal(observable(source, targets), np.zeros_like(observable(source, targets)))


@pytest.mark.parametrize(
    "provenance",
    [
        {**KW["provenance"], "reconstructed": 1},
        {**KW["provenance"], "unexpected": "key"},
    ],
)
@requires_meshio
def test_rejects_invalid_profile_assertions(tmp_path, provenance):
    with pytest.raises(ValueError, match="reconstructed|exactly"):
        load_openfoam_density(write_two_hex(tmp_path / "w6_0.vtk"), **{**KW, "provenance": provenance})


@requires_meshio
def test_rejects_time_mismatch_and_successor_mesh_origin_or_kind_change(tmp_path):
    first = load_openfoam_density(write_two_hex(tmp_path / "w6_0.vtk"), **KW)
    with pytest.raises(ValueError, match="time_s"):
        load_openfoam_density(
            write_two_hex(tmp_path / "time_0.vtk"), **{**KW, "time_s": 0.1}
        )
    shifted = load_openfoam_density(
        write_two_hex(tmp_path / "shifted_0.125.vtk", x_offset=1.0),
        **{**KW, "time_s": 0.125, "time_name": "0.125"},
    )
    with pytest.raises(ValueError, match="geometry|mesh"):
        first.assert_compatible_successor(shifted)
    changed_origin = load_openfoam_density(
        write_two_hex(tmp_path / "origin_0.125.vtk", density=(2.0, -1.0)),
        **{
            **KW,
            "time_s": 0.125,
            "time_name": "0.125",
            "coordinate_origin_m": [1.0, 0.0, 0.0],
        },
    )
    with pytest.raises(ValueError, match="geometry"):
        first.assert_compatible_successor(changed_origin)
    changed_kind = load_openfoam_density(
        write_two_hex(tmp_path / "rho_0.125.vtk", density=(2.0, -1.0)),
        **{**KW, "field_name": "rho", "time_s": 0.125, "time_name": "0.125"},
    )
    with pytest.raises(ValueError, match="geometry|profile"):
        first.assert_compatible_successor(changed_kind)
