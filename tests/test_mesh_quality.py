"""Tests for the trimesh-backed mesh quality adapter (Phase 10-f).

trimesh は optional 依存（``gragra[mesh-quality]`` extra）のため importorskip で
保護する。fixture はすべてテスト内生成（バイナリ非コミット）。cube fixture は
conftest の Phase 10-c 系と同一メッシュを共有する。
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from gragra.adapters import check_mesh_quality, load_trimesh  # noqa: E402


def _mesh_object(points, triangles=None, quads=None, extra_cells=()):
    """meshio.Mesh 相当の duck-typed オブジェクト（meshio 非依存）。"""
    cells = list(extra_cells)
    if triangles is not None:
        cells.append(SimpleNamespace(type="triangle", data=np.asarray(triangles)))
    if quads is not None:
        cells.append(SimpleNamespace(type="quad", data=np.asarray(quads)))
    return SimpleNamespace(points=np.asarray(points, dtype=np.float64), cells=cells)


def test_watertight_cube_passes(cube_vertices, cube_triangles):
    # 一辺 2 の立方体: V = 8, A = 24, Euler 数 = 2（球と同相）。
    mesh = _mesh_object(cube_vertices, triangles=cube_triangles)
    q = check_mesh_quality(mesh, expected_volume_m3=8.0)
    assert q["pass"] is True
    assert q["is_watertight"] is True
    assert q["is_winding_consistent"] is True
    np.testing.assert_allclose(q["volume_m3"], 8.0, rtol=1e-9)
    np.testing.assert_allclose(q["area_m2"], 24.0, rtol=1e-9)
    assert q["euler_number"] == 2
    assert q["n_faces"] == 12
    assert q["n_vertices"] == 8
    assert q["n_duplicate_vertices"] == 0
    assert q["volume_rel_error"] < 1e-9
    # エッジ長: 面内辺 2.0 と面対角線 2√2 のみ
    np.testing.assert_allclose(q["min_edge_length_m"], 2.0, rtol=1e-12)
    np.testing.assert_allclose(q["max_edge_length_m"], 2.0 * np.sqrt(2.0), rtol=1e-12)


def test_missing_face_not_watertight(cube_vertices, cube_triangles):
    mesh = _mesh_object(cube_vertices, triangles=cube_triangles[:-1])
    q = check_mesh_quality(mesh)
    assert q["is_watertight"] is False
    assert q["pass"] is False


def test_stl_style_duplicate_vertices_detected(cube_vertices, cube_triangles):
    # STL 風: 面ごとに頂点を非共有展開（36 頂点、実座標は 8 種 → 28 重複）。
    # 位置ベースの merge 後は watertight 判定が回復する。
    exploded_points = cube_vertices[cube_triangles].reshape(-1, 3)
    exploded_tris = np.arange(36).reshape(12, 3)
    mesh = _mesh_object(exploded_points, triangles=exploded_tris)
    q = check_mesh_quality(mesh, expected_volume_m3=8.0)
    assert q["n_vertices"] == 36
    assert q["n_duplicate_vertices"] == 28
    assert q["is_watertight"] is True
    assert q["pass"] is True
    np.testing.assert_allclose(q["volume_m3"], 8.0, rtol=1e-9)


def test_volume_gate_five_percent(cube_vertices, cube_triangles):
    mesh = _mesh_object(cube_vertices, triangles=cube_triangles)
    q_ok = check_mesh_quality(mesh, expected_volume_m3=8.0 * 1.02)
    assert q_ok["volume_rel_error"] < 0.05
    assert q_ok["pass"] is True
    q_ng = check_mesh_quality(mesh, expected_volume_m3=8.0 * 1.06)
    assert q_ng["volume_rel_error"] > 0.05
    assert q_ng["pass"] is False


def test_quad_cells_are_split(cube_vertices):
    # 外向き巻きの 6 quad 面（conftest の三角形分割と同じ向き規約）。
    quads = np.array(
        [
            [0, 3, 2, 1],  # bottom (n = -z)
            [4, 5, 6, 7],  # top    (n = +z)
            [0, 1, 5, 4],  # front  (n = -y)
            [3, 7, 6, 2],  # back   (n = +y)
            [0, 4, 7, 3],  # left   (n = -x)
            [1, 2, 6, 5],  # right  (n = +x)
        ],
        dtype=np.int64,
    )
    q = check_mesh_quality(_mesh_object(cube_vertices, quads=quads))
    assert q["n_faces"] == 12  # quad → 2 三角形
    assert q["is_watertight"] is True
    np.testing.assert_allclose(q["volume_m3"], 8.0, rtol=1e-9)


def test_unsupported_cells_only_raises(cube_vertices):
    tetra = SimpleNamespace(type="tetra", data=np.array([[0, 1, 2, 6]]))
    mesh = _mesh_object(cube_vertices, extra_cells=[tetra])
    with pytest.raises(ValueError, match="triangle or quad"):
        load_trimesh(mesh)


def test_path_input_roundtrip(tmp_path, cube_vertices, cube_triangles):
    # ファイル経路入力: STL をテスト内生成（コミットしない）して再読込。
    path = tmp_path / "cube.stl"
    trimesh.Trimesh(vertices=cube_vertices, faces=cube_triangles, process=False).export(
        path
    )
    q = check_mesh_quality(path, expected_volume_m3=8.0)
    assert q["pass"] is True
    # STL は面ごとに頂点を展開して保存するため重複頂点が報告される
    assert q["n_duplicate_vertices"] > 0
    np.testing.assert_allclose(q["volume_m3"], 8.0, rtol=1e-9)


@pytest.mark.parametrize("bad", [0.0, -1e-8, np.nan, np.inf])
def test_invalid_rounding_value_raises(bad, cube_vertices, cube_triangles):
    mesh = _mesh_object(cube_vertices, triangles=cube_triangles)
    with pytest.raises(ValueError, match="duplicate_vertex_rounding"):
        check_mesh_quality(mesh, duplicate_vertex_rounding=bad)


@pytest.mark.parametrize("bad", [True, "1e-8", None])
def test_invalid_rounding_type_raises(bad, cube_vertices, cube_triangles):
    # 型違いは TypeError（adapters の TypeError/ValueError 分離規約）
    mesh = _mesh_object(cube_vertices, triangles=cube_triangles)
    with pytest.raises(TypeError, match="duplicate_vertex_rounding"):
        check_mesh_quality(mesh, duplicate_vertex_rounding=bad)


@pytest.mark.parametrize("bad", [0.0, -8.0, np.nan, np.inf])
def test_invalid_expected_volume_value_raises(bad, cube_vertices, cube_triangles):
    mesh = _mesh_object(cube_vertices, triangles=cube_triangles)
    with pytest.raises(ValueError, match="expected_volume_m3"):
        check_mesh_quality(mesh, expected_volume_m3=bad)


@pytest.mark.parametrize("bad", [True, "8.0"])
def test_invalid_expected_volume_type_raises(bad, cube_vertices, cube_triangles):
    # bool は float(True)=1.0 に化けて 5% ゲートを狂わせるため明示拒否
    mesh = _mesh_object(cube_vertices, triangles=cube_triangles)
    with pytest.raises(TypeError, match="expected_volume_m3"):
        check_mesh_quality(mesh, expected_volume_m3=bad)


def test_nan_vertex_raises(cube_vertices, cube_triangles):
    # NaN 頂点のサイレント伝播（watertight=True + volume=nan）を禁止
    bad_points = cube_vertices.copy()
    bad_points[0, 0] = np.nan
    mesh = _mesh_object(bad_points, triangles=cube_triangles)
    with pytest.raises(ValueError, match="finite"):
        check_mesh_quality(mesh)


@pytest.mark.parametrize("bad_index", [-1, 8])
def test_out_of_range_cell_index_raises(bad_index, cube_vertices, cube_triangles):
    # 負 index は numpy fancy indexing でサイレントに折り返すため明示拒否
    bad_tris = cube_triangles.copy()
    bad_tris[0, 2] = bad_index
    mesh = _mesh_object(cube_vertices, triangles=bad_tris)
    with pytest.raises(ValueError, match="out of range"):
        check_mesh_quality(mesh)


def test_empty_triangle_block_no_edges_raises(cube_vertices):
    mesh = _mesh_object(cube_vertices, triangles=np.empty((0, 3), dtype=np.int64))
    with pytest.raises(ValueError, match="no edges"):
        check_mesh_quality(mesh)


def test_missing_trimesh_raises_actionable_error(
    monkeypatch, cube_vertices, cube_triangles
):
    # sys.modules に None を差すと `import trimesh` が ImportError になる
    monkeypatch.setitem(sys.modules, "trimesh", None)
    mesh = _mesh_object(cube_vertices, triangles=cube_triangles)
    with pytest.raises(ImportError, match="mesh-quality"):
        check_mesh_quality(mesh)
