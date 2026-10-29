"""Tests for triangle surface orientation diagnostics (Phase 10-c-f).

Diagnostic-only winding/topology check. Analytic-first per testing.md: the cube
fixture (shared via conftest with the surface-field sign-pinning tests) has a
known signed volume and known winding, so the diagnostic can be pinned exactly.
"""

import dataclasses
import subprocess
import sys

import numpy as np
import pytest

from gragra.fields.surface_orientation import (
    OrientationReport,
    check_triangle_orientation,
    reorient_triangles,
)

# ---------------------------------------------------------------------------
# 1. Closed, consistent, outward cube
# ---------------------------------------------------------------------------


def test_outward_cube_is_consistent_closed_outward(
    cube_vertices, cube_triangles, cube_half
):
    rep = check_triangle_orientation(cube_vertices, cube_triangles)
    assert rep.is_consistent is True
    assert rep.is_closed is True
    assert rep.is_outward is True
    edge = 2.0 * cube_half
    np.testing.assert_allclose(rep.signed_volume_m3, edge**3, rtol=1e-12)
    assert rep.inconsistent_edge_count == 0
    assert rep.boundary_edge_count == 0
    assert rep.nonmanifold_edge_count == 0
    assert rep.inconsistent_triangle_indices.size == 0


# ---------------------------------------------------------------------------
# 2. Fully reversed winding -> inward (still consistent/closed)
# ---------------------------------------------------------------------------


def test_full_reversal_is_inward(cube_vertices, cube_triangles, cube_half):
    reversed_tri = cube_triangles[:, ::-1].copy()
    rep = check_triangle_orientation(cube_vertices, reversed_tri)
    assert rep.is_consistent is True
    assert rep.is_closed is True
    assert rep.is_outward is False
    edge = 2.0 * cube_half
    np.testing.assert_allclose(rep.signed_volume_m3, -(edge**3), rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. One face reversed -> inconsistent
# ---------------------------------------------------------------------------


def test_single_face_reversal_is_inconsistent(cube_vertices, cube_triangles):
    tri = cube_triangles.copy()
    tri[0] = tri[0][::-1]  # reverse winding of triangle 0 only
    rep = check_triangle_orientation(cube_vertices, tri)
    assert rep.is_consistent is False
    # The reversed triangle's 3 edges each clash with their neighbor.
    assert rep.inconsistent_edge_count == 3
    assert rep.is_outward is None
    # Reversed face (index 0) plus its 3 edge-neighbors = 4 unique triangles.
    idx = rep.inconsistent_triangle_indices
    assert idx.dtype == np.intp
    assert len(idx) == 4
    assert 0 in idx.tolist()


# ---------------------------------------------------------------------------
# 4. Open surface (boundary edges)
# ---------------------------------------------------------------------------


def test_single_triangle_is_open():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    tri = np.array([[0, 1, 2]], dtype=np.int64)
    rep = check_triangle_orientation(vertices, tri)
    assert rep.is_closed is False
    assert rep.boundary_edge_count == 3
    assert rep.is_outward is None
    assert rep.is_consistent is True


# ---------------------------------------------------------------------------
# 5. Non-manifold edge (shared by three triangles)
# ---------------------------------------------------------------------------


def test_nonmanifold_edge():
    # Edge (0, 1) shared by three triangles fanning out of it.
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    tri = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int64)
    rep = check_triangle_orientation(vertices, tri)
    assert rep.nonmanifold_edge_count >= 1
    assert rep.is_closed is False
    assert rep.is_outward is None
    # Non-manifold and winding consistency are independent: the shared edge has
    # count >= 3 and is excluded from the opposite-direction check, and the
    # remaining edges are boundary, so no co-directed shared edge exists.
    assert rep.is_consistent is True


# ---------------------------------------------------------------------------
# 5b. Multi-component closed mesh: is_outward is ambiguous (documented limit)
# ---------------------------------------------------------------------------


def test_multicomponent_closed_mesh_outward_is_ambiguous(
    cube_vertices, cube_triangles, cube_half
):
    # Two disjoint closed cubes: cube A wound outward, cube B wound inward and
    # twice the linear size (8x the volume). The mesh is consistent and closed
    # per component, yet a single is_outward cannot certify per-component
    # outwardness -- here signed_volume = V_A - V_B < 0 despite A being outward.
    big = cube_vertices * 2.0 + np.array([10.0, 0.0, 0.0])  # translate apart
    vertices = np.concatenate([cube_vertices, big], axis=0)
    nv = len(cube_vertices)
    tri_a = cube_triangles  # outward
    tri_b = cube_triangles[:, ::-1] + nv  # inward, reindexed to the 2nd cube
    triangles = np.concatenate([tri_a, tri_b], axis=0)

    rep = check_triangle_orientation(vertices, triangles)
    assert rep.is_consistent is True
    assert rep.is_closed is True
    edge = 2.0 * cube_half
    # V_A = edge**3 (outward, +), V_B = (2*edge)**3 (inward, -).
    np.testing.assert_allclose(
        rep.signed_volume_m3, edge**3 - (2.0 * edge) ** 3, rtol=1e-12
    )
    # is_outward reports the sign of the (ambiguous) total signed volume.
    assert rep.is_outward is False  # even though component A is outward


# ---------------------------------------------------------------------------
# 6. Empty mesh
# ---------------------------------------------------------------------------


def test_empty_mesh(cube_vertices):
    tri = np.zeros((0, 3), dtype=np.int64)
    rep = check_triangle_orientation(cube_vertices, tri)
    assert rep.is_consistent is True
    assert rep.is_closed is False
    assert rep.is_outward is None
    assert rep.signed_volume_m3 == 0.0
    assert rep.inconsistent_edge_count == 0
    assert rep.boundary_edge_count == 0
    assert rep.nonmanifold_edge_count == 0
    assert rep.inconsistent_triangle_indices.dtype == np.intp
    assert rep.inconsistent_triangle_indices.size == 0


# ---------------------------------------------------------------------------
# 7. Method-path equality + dtype independence (array C1)
# ---------------------------------------------------------------------------


def _assert_reports_equal(r, ref):
    assert r.is_consistent == ref.is_consistent
    assert r.is_closed == ref.is_closed
    assert r.is_outward == ref.is_outward
    np.testing.assert_allclose(r.signed_volume_m3, ref.signed_volume_m3)
    assert r.inconsistent_edge_count == ref.inconsistent_edge_count
    assert r.boundary_edge_count == ref.boundary_edge_count
    assert r.nonmanifold_edge_count == ref.nonmanifold_edge_count
    assert r.inconsistent_triangle_indices.dtype == np.intp
    np.testing.assert_array_equal(
        r.inconsistent_triangle_indices, ref.inconsistent_triangle_indices
    )


def test_method_paths_and_dtype_agree(cube_vertices, cube_triangles):
    from gragra.adapters.meshio_adapter import TriangleSurfaceMesh
    from gragra.fields.surface import SurfaceDisplacementField

    ref = check_triangle_orientation(cube_vertices, cube_triangles)

    # SurfaceDisplacementField stores triangles without dtype coercion; feed it
    # int32 to confirm the report is independent of the input integer dtype.
    disp = np.zeros_like(cube_vertices)
    sdf = SurfaceDisplacementField(cube_vertices, cube_triangles.astype(np.int32), disp)
    # TriangleSurfaceMesh coerces to int64.
    tsm = TriangleSurfaceMesh(cube_vertices, cube_triangles)

    _assert_reports_equal(sdf.check_orientation(), ref)
    _assert_reports_equal(tsm.check_orientation(), ref)
    # Direct function with a different integer dtype must also agree.
    _assert_reports_equal(
        check_triangle_orientation(cube_vertices, cube_triangles.astype(np.int32)),
        ref,
    )


# ---------------------------------------------------------------------------
# 8. Immutability / read-only
# ---------------------------------------------------------------------------


def test_report_is_immutable(cube_vertices, cube_triangles):
    rep = check_triangle_orientation(cube_vertices, cube_triangles)
    assert rep.inconsistent_triangle_indices.flags.writeable is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        rep.is_consistent = False


def test_inconsistent_indices_read_only_when_present(cube_vertices, cube_triangles):
    tri = cube_triangles.copy()
    tri[0] = tri[0][::-1]
    rep = check_triangle_orientation(cube_vertices, tri)
    assert rep.inconsistent_triangle_indices.size > 0
    assert rep.inconsistent_triangle_indices.flags.writeable is False


# ---------------------------------------------------------------------------
# 9. Input validation
# ---------------------------------------------------------------------------


def test_noninteger_triangles_raise(cube_vertices):
    with pytest.raises(TypeError):
        check_triangle_orientation(cube_vertices, np.array([[0.0, 1.0, 2.0]]))


def test_bad_shape_raises(cube_vertices):
    with pytest.raises(ValueError):
        check_triangle_orientation(cube_vertices, np.array([0, 1, 2]))


def test_out_of_range_index_raises(cube_vertices):
    n = len(cube_vertices)
    with pytest.raises(ValueError):
        check_triangle_orientation(cube_vertices, np.array([[0, 1, n]]))
    with pytest.raises(ValueError):
        check_triangle_orientation(cube_vertices, np.array([[0, 1, -1]]))


def test_repeated_index_triangle_raises(cube_vertices):
    with pytest.raises(ValueError):
        check_triangle_orientation(cube_vertices, np.array([[0, 1, 1]]))


def test_bool_triangles_raise(cube_vertices):
    with pytest.raises(TypeError):
        check_triangle_orientation(cube_vertices, np.array([[False, True, True]]))


# ---------------------------------------------------------------------------
# 10. Regression tie: diagnostic underpins the mass-conservation sign
# ---------------------------------------------------------------------------


def test_diagnostic_underpins_mass_conservation_cube(
    cube_vertices, cube_triangles, cube_half
):
    # The same outward cube fixture that fixes the bulk/surface mass-conservation
    # sign in test_surface_field must be reported as outward by the diagnostic.
    rep = check_triangle_orientation(cube_vertices, cube_triangles)
    assert rep.is_outward is True
    edge = 2.0 * cube_half
    np.testing.assert_allclose(rep.signed_volume_m3, edge**3, rtol=1e-12)


# ---------------------------------------------------------------------------
# 11. Core purity: module and package import must not pull the adapters layer
# ---------------------------------------------------------------------------


def test_import_does_not_load_adapters_or_optional_backends():
    for target in ("gragra.fields.surface_orientation", "gragra.fields"):
        code = (
            "import sys\n"
            f"import {target}\n"
            "leaked = [m for m in ('gragra.adapters', 'meshio', 'gwexpy', "
            "'scipy', 'h5py') if m in sys.modules]\n"
            "assert not leaked, leaked\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_orientation_report_is_dataclass():
    assert dataclasses.is_dataclass(OrientationReport)


# ---------------------------------------------------------------------------
# 12. Triangle Surface Orientation Correction (Phase 10-c-g)
# ---------------------------------------------------------------------------


def test_reorient_already_consistent(cube_vertices, cube_triangles, cube_half):
    # Already consistent mesh should remain consistent.
    # Note that the orientation might be flipped or shifted, but its volume
    # magnitude (enclosed volume) must be preserved, and check_orientation
    # must confirm it is consistent.
    res = reorient_triangles(cube_triangles)
    assert res.dtype == np.intp
    assert res.flags.writeable is False

    rep = check_triangle_orientation(cube_vertices, res)
    assert rep.is_consistent is True
    assert rep.is_closed is True
    edge = 2.0 * cube_half
    np.testing.assert_allclose(abs(rep.signed_volume_m3), edge**3, rtol=1e-12)


def test_reorient_inconsistent_mesh(cube_vertices, cube_triangles, cube_half):
    # Reverse winding of one face to make it inconsistent.
    inconsistent_tri = cube_triangles.copy()
    inconsistent_tri[0] = inconsistent_tri[0][::-1]

    rep_before = check_triangle_orientation(cube_vertices, inconsistent_tri)
    assert rep_before.is_consistent is False

    # Correction
    corrected_tri = reorient_triangles(inconsistent_tri)
    assert corrected_tri.dtype == np.intp
    assert corrected_tri.flags.writeable is False

    rep_after = check_triangle_orientation(cube_vertices, corrected_tri)
    assert rep_after.is_consistent is True
    assert rep_after.is_closed is True
    edge = 2.0 * cube_half
    np.testing.assert_allclose(abs(rep_after.signed_volume_m3), edge**3, rtol=1e-12)


def test_reorient_input_is_preserved():
    # Verify the input array is not mutated and a copy is returned.
    tri = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64)
    orig_copy = tri.copy()

    res = reorient_triangles(tri)
    np.testing.assert_array_equal(tri, orig_copy)
    assert res is not tri


def test_reorient_empty_mesh():
    tri = np.zeros((0, 3), dtype=np.int64)
    res = reorient_triangles(tri)
    assert res.size == 0
    assert res.dtype == np.intp
    assert res.flags.writeable is False


def test_reorient_validation_errors():
    # Self-loops
    with pytest.raises(ValueError, match="repeated vertex indices"):
        reorient_triangles([[0, 1, 1]])

    # Negative indices
    with pytest.raises(ValueError, match="must be non-negative"):
        reorient_triangles([[0, 1, -1]])

    # Bool inputs
    with pytest.raises(TypeError, match="must not contain bool"):
        reorient_triangles([[False, True, True]])

    # Non-integer type
    with pytest.raises(TypeError, match="must be of integer type"):
        reorient_triangles([[0.0, 1.0, 2.0]])

    # Bad shape
    with pytest.raises(ValueError, match="must have shape"):
        reorient_triangles([0, 1, 2])


def test_reorient_nonmanifold_raises():
    # Edge (0, 1) shared by three triangles
    tri = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int64)
    with pytest.raises(ValueError, match="mesh is non-manifold"):
        reorient_triangles(tri)


def test_reorient_nonorientable_raises():
    # Mobius strip (6 triangles)
    # Winding consistency cannot be resolved topologically
    mobius_tri = np.array(
        [
            [0, 1, 2],
            [2, 1, 3],
            [2, 3, 4],
            [4, 3, 5],
            [4, 5, 1],
            [1, 5, 0],
        ],
        dtype=np.int64,
    )
    with pytest.raises(ValueError, match="mesh is non-orientable"):
        reorient_triangles(mobius_tri)


def test_reorient_seed_face_reversed_yields_inward(
    cube_vertices, cube_triangles, cube_half
):
    # tri 0 is the component's minimum index (the seed).
    # Reversing tri 0 winding must propagate to the whole component, making
    # the entire reoriented mesh inward (since the seed winding is preserved).
    inconsistent_tri = cube_triangles.copy()
    inconsistent_tri[0] = inconsistent_tri[0][::-1]

    res = reorient_triangles(inconsistent_tri)
    rep = check_triangle_orientation(cube_vertices, res)
    assert rep.is_consistent is True
    assert rep.is_closed is True

    # Volume must be negative (inward) because the seed was reversed to inward.
    edge = 2.0 * cube_half
    np.testing.assert_allclose(rep.signed_volume_m3, -(edge**3), rtol=1e-12)
    assert rep.is_outward is False


def test_reorient_full_reversal_preserves_inward(
    cube_vertices, cube_triangles, cube_half
):
    # Fully reversed cube is already consistent but inward.
    # Winding consistency correction must not force outward orientation
    # and must preserve the inward orientation of the seed (tri 0).
    inward_tri = cube_triangles[:, ::-1].copy()

    res = reorient_triangles(inward_tri)
    np.testing.assert_array_equal(res, inward_tri)  # topological no-op

    rep = check_triangle_orientation(cube_vertices, res)
    assert rep.is_consistent is True
    assert rep.is_closed is True
    edge = 2.0 * cube_half
    np.testing.assert_allclose(rep.signed_volume_m3, -(edge**3), rtol=1e-12)
    assert rep.is_outward is False


def test_reorient_multicomponent_preserves_each_seed_orientation(
    cube_vertices, cube_triangles, cube_half
):
    # Two disjoint cubes:
    # Component A (vertices 0-7): outward, but we reverse its tri 0 -> seed is inward.
    # Component B (vertices 8-15): inward, but we reverse its tri 0 -> seed is outward.
    # We index Component B with a vertex offset of 8.
    nv = len(cube_vertices)
    edge = 2.0 * cube_half

    tri_a = cube_triangles.copy()
    tri_a[0] = tri_a[0][::-1]  # seed is now inward

    tri_b = (cube_triangles[:, ::-1] + nv).copy()
    tri_b[0] = tri_b[0][::-1]  # seed is now outward

    combined_tri = np.concatenate([tri_a, tri_b], axis=0)

    res = reorient_triangles(combined_tri)

    # Validate Component A (triangles 0 to 11)
    res_a = res[0:12]
    rep_a = check_triangle_orientation(cube_vertices, res_a)
    assert rep_a.is_consistent is True
    np.testing.assert_allclose(rep_a.signed_volume_m3, -(edge**3), rtol=1e-12)

    # Validate Component B (triangles 12 to 23, remove offset nv)
    res_b = res[12:24] - nv
    rep_b = check_triangle_orientation(cube_vertices, res_b)
    assert rep_b.is_consistent is True
    np.testing.assert_allclose(rep_b.signed_volume_m3, edge**3, rtol=1e-12)


def test_reorient_idempotent(cube_triangles):
    inconsistent_tri = cube_triangles.copy()
    inconsistent_tri[0] = inconsistent_tri[0][::-1]

    res1 = reorient_triangles(inconsistent_tri)
    res2 = reorient_triangles(res1)
    np.testing.assert_array_equal(res1, res2)


def test_methods_with_consistent_orientation(cube_vertices, cube_triangles):
    from gragra.adapters.meshio_adapter import TriangleSurfaceMesh
    from gragra.fields.surface import SurfaceDisplacementField

    inconsistent_tri = cube_triangles.copy()
    inconsistent_tri[0] = inconsistent_tri[0][::-1]

    expected = reorient_triangles(inconsistent_tri)

    # 1. SurfaceDisplacementField with int32 triangles
    disp = np.zeros_like(cube_vertices)
    sdf = SurfaceDisplacementField(
        cube_vertices, inconsistent_tri.astype(np.int32), disp
    )

    sdf_corrected = sdf.with_consistent_orientation()
    assert isinstance(sdf_corrected, SurfaceDisplacementField)
    np.testing.assert_array_equal(sdf_corrected.triangles, expected)
    assert sdf_corrected.triangles.dtype == np.intp
    assert sdf_corrected.triangles.flags.writeable is False
    assert sdf_corrected is not sdf

    # 2. TriangleSurfaceMesh
    tsm = TriangleSurfaceMesh(cube_vertices, inconsistent_tri)

    tsm_corrected = tsm.with_consistent_orientation()
    assert isinstance(tsm_corrected, TriangleSurfaceMesh)
    np.testing.assert_array_equal(tsm_corrected.triangles, expected)
    assert tsm_corrected.triangles.dtype == np.intp
    assert tsm_corrected.triangles.flags.writeable is False
    assert tsm_corrected is not tsm
