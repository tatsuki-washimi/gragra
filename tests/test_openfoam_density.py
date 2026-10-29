# ruff: noqa: I001

import importlib.util
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
from gragra.analytic import cuboid_filled_internal_field  # noqa: E402
from gragra.observables import (  # noqa: E402
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)
from gragra.gravity_sources.volume import VolumeElementSource  # noqa: E402
from tests._openfoam_synthetic import write_two_hex  # noqa: E402

import gragra  # noqa: E402
import gragra.adapters.openfoam_density as _openfoam_density  # noqa: E402

assert gragra.__file__.startswith(str(_ROOT / "src"))
assert _openfoam_density.__file__.startswith(str(_ROOT / "src"))

requires_meshio = pytest.mark.skipif(
    importlib.util.find_spec("meshio") is None,
    reason="requires the optional gragra[mesh] dependency",
)

PROV = {
    "distribution": "OpenFOAM Foundation",
    "version": "14",
    "patch": "20260724",
    "solver": "manufactured-no-solver",
    "compressibility": "not-applicable",
    "region": "region0",
    "reconstructed": True,
    "exporter": "foamToVTK",
    "export_command": "foamToVTK -ascii",
}
KW = {
    "field_name": "delta_rho",
    "time_s": 0.0,
    "time_name": "0",
    "coordinate_frame": "case",
    "coordinate_origin_m": np.array([2.0, 3.0, 4.0]),
    "density_unit": "kg/m^3",
    "coordinate_unit": "m",
    "volume_unit": "m^3",
    "time_unit": "s",
    "provenance": PROV,
}


@requires_meshio
def test_load_preserves_native_rows_and_translates_origin(tmp_path):
    p = write_two_hex(tmp_path / "w6_0.vtk")
    s = load_openfoam_density(p, **KW)
    assert s.input_path == str(p.resolve())
    assert np.array_equal(s.positions_m, [[2.5, 3.5, 4.5], [3.5, 3.5, 4.5]])
    assert np.array_equal(s.cell_ids, [0, 1])
    assert (
        s.positions_m.dtype == np.dtype("float64") and not s.positions_m.flags.writeable
    )
    assert s.provenance["density_dtype"] == "float32"
    assert np.array_equal(
        s.as_volume_element_source().as_weighted_source().weights_kg, [-1.0, 2.0]
    )


@requires_meshio
def test_successor_requires_same_geometry_and_strict_time(tmp_path):
    a = load_openfoam_density(write_two_hex(tmp_path / "w6_0.vtk"), **KW)
    b = load_openfoam_density(
        write_two_hex(tmp_path / "w6_0.125.vtk", density=(2.0, -1.0)),
        **{**KW, "time_s": 0.125, "time_name": "0.125"},
    )
    a.assert_compatible_successor(b)
    with pytest.raises(ValueError, match="strictly"):
        a.assert_compatible_successor(a)


@requires_meshio
def test_public_snapshot_constructor_rejects_invented_snapshot(tmp_path):
    snapshot = load_openfoam_density(write_two_hex(tmp_path / "w6_0.vtk"), **KW)
    with pytest.raises(TypeError, match="load_openfoam_density"):
        type(snapshot)(
            input_path="/invented/input_0.vtk",
            provenance={"input_sha256": "0" * 64},
        )


@requires_meshio
def test_loader_result_is_defensively_readonly(tmp_path):
    snapshot = load_openfoam_density(write_two_hex(tmp_path / "w6_0.vtk"), **KW)
    with pytest.raises(ValueError):
        snapshot.positions_m[0, 0] = 99.0
    with pytest.raises(TypeError):
        snapshot.provenance["solver"] = "invented"


def test_frozen_box_refinement_and_chunk_invariance():
    target = np.array([[3.0, 0.7, -0.4]])
    exact = 6.67430e-11 * cuboid_filled_internal_field([2.0, 1.0, 1.0], target, 3.0)
    errors = []
    for nx, ny, nz in ((2, 1, 1), (4, 2, 2), (8, 4, 4)):
        axes = [
            np.linspace(-1, 1, nx + 1),
            np.linspace(-0.5, 0.5, ny + 1),
            np.linspace(-0.5, 0.5, nz + 1),
        ]
        centres = (
            np.array(np.meshgrid(*[(a[:-1] + a[1:]) / 2 for a in axes], indexing="ij"))
            .reshape(3, -1)
            .T
        )
        source = VolumeElementSource(
            centres, np.full(len(centres), 2 / nx / ny / nz), 3.0
        ).as_weighted_source()
        got = point_acceleration(source, target)
        assert np.allclose(
            got, point_acceleration(source, target, chunk_size=1), rtol=5e-13, atol=0
        )
        assert np.allclose(
            got, point_acceleration(source, target, chunk_size=7), rtol=5e-13, atol=0
        )
        errors.append(np.linalg.norm(got - exact) / np.linalg.norm(exact))
    assert errors[2] < errors[0]
    assert errors[2] / errors[1] <= 0.60
    assert errors[2] <= 0.02


def test_moving_density_lump_changes_acceleration():
    target = np.array([[3.0, 0.0, 0.0]])
    left = VolumeElementSource([[-0.5, 0, 0]], [1.0], [3.0]).as_weighted_source()
    right = VolumeElementSource([[0.5, 0, 0]], [1.0], [3.0]).as_weighted_source()
    assert not np.array_equal(
        point_acceleration(left, target), point_acceleration(right, target)
    )


@requires_meshio
def test_extra_fields_leave_physics_exactly_unchanged(tmp_path):
    plain = load_openfoam_density(write_two_hex(tmp_path / "plain_0.vtk"), **KW)
    extra = load_openfoam_density(
        write_two_hex(tmp_path / "extra_0.vtk", extra=True), **KW
    )
    for name in ("positions_m", "volumes_m3", "density_kg_m3", "cell_ids"):
        assert np.array_equal(getattr(plain, name), getattr(extra, name))
    targets = np.array([[3.0, 0.7, -0.4], [-2.0, 1.0, 2.0]])
    for observable in (point_potential, point_acceleration, point_gravity_gradient):
        assert np.array_equal(
            observable(plain.as_volume_element_source().as_weighted_source(), targets),
            observable(extra.as_volume_element_source().as_weighted_source(), targets),
        )
