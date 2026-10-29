"""Tests for gragra.adapters.specfem.specfem_bin.

Covers: normal-path reading of the synthetic decompose_mesh tree (Fortran
record auto-detection, ibool/coordinate/rho/Database/part_array parsing),
element_id recovery round-trip + partition-independence (bin level), the
shared-face nglob reduction, and the destructive/negative paths (corrupted
marker, out-of-range ibool, truncated/excess EOF, endianness auto-detection,
missing displacement component, inconsistent 'it' set, non-positive rho,
material-count/mat1 mismatch).
"""

import numpy as np
import pytest
from _specfem_synthetic import (
    _fortran_record_bytes,
    _int_payload,
    _write_records_file,
    _write_single_record,
    write_displacement_snapshot_files,
    write_synthetic_specfem_tree,
)

from gragra.adapters.specfem import specfem_bin

# ===========================================================================
# 1. Normal-path: reading the synthetic tree
# ===========================================================================


def test_read_ibool_coords_rho_and_database_match_fixture(tmp_path):
    info = write_synthetic_specfem_tree(tmp_path / "tree", partition_variant="a")

    for rank in range(info.nproc):
        db = specfem_bin.parse_database(info.database_path(rank))
        expected_global_ids = info.elements_for_rank(rank)
        assert db.nspec_local == len(expected_global_ids)
        assert len(db.materials) == 2
        for mat in db.materials:
            assert mat.domain_id == 2.0  # IDOMAIN_ELASTIC

        ibool, layout, integer_kind = specfem_bin.read_ibool(
            info.bin_path(rank, "ibool"), info.ngll
        )
        assert ibool.shape == (db.nspec_local, info.ngll)
        assert layout.endianness == "<"
        assert integer_kind == 4
        assert int(ibool.min()) == 0  # zero-based after conversion

        nglob = int(ibool.max()) + 1
        x, _ = specfem_bin.read_coords(info.bin_path(rank, "x"), nglob)
        y, _ = specfem_bin.read_coords(info.bin_path(rank, "y"), nglob)
        z, _ = specfem_bin.read_coords(info.bin_path(rank, "z"), nglob)
        xyz = np.stack([x, y, z], axis=-1)
        elem_coords = specfem_bin.gather_element_coordinates(ibool, xyz)
        assert elem_coords.shape == (db.nspec_local, info.ngll, 3)

        rho, _ = specfem_bin.read_rho(
            info.bin_path(rank, "rho"), db.nspec_local, info.ngll
        )
        assert rho.shape == (db.nspec_local, info.ngll)

        # Cross-check rho against the fixture's ground truth per element
        # (local element k <-> global_ids_this_rank[k], ascending, matching
        # the Database's local element numbering convention).
        for k, global_id in enumerate(expected_global_ids):
            expected_rho = info.elements[global_id].rho_profile
            # material 1 (rho=2000) is stored exactly; material 2 (rho=3000)
            # is 1-ulp-perturbed float32 -- rtol accommodates both regimes.
            assert rho[k] == pytest.approx(expected_rho, rel=1e-6)

        # Database node/element block round-trip: local node iloc -> row.
        row_of_iloc = {int(v): i for i, v in enumerate(db.node_iloc)}
        for k, global_id in enumerate(expected_global_ids):
            elem = db.elements[k]
            assert elem.mat1 == info.elements[global_id].material_id
            assert elem.mat2 == 0
            corner_rows = [row_of_iloc[nid] for nid in elem.node_loc_ids]
            corners_from_db = db.node_xyz[corner_rows]
            assert np.array_equal(corners_from_db, info.elements[global_id].corners8)


def test_a1_style_axis_direction_check_on_e0(tmp_path):
    """A minimal, bin-level analogue of Tier A1's direction-fixed check:
    E0 (the 100x200x400 m axis-aligned anchor element) must have its
    GLL-grid end-to-end displacement along each index axis match the
    expected physical vector -- this is only possible if ibool/coordinate
    indexing is not transposed.
    """
    info = write_synthetic_specfem_tree(tmp_path / "tree", partition_variant="a")
    rank = info.rank_of(0)  # E0's rank
    local_index = info.elements_for_rank(rank).index(0)

    ibool, _, _ = specfem_bin.read_ibool(info.bin_path(rank, "ibool"), info.ngll)
    nglob = int(ibool.max()) + 1
    x, _ = specfem_bin.read_coords(info.bin_path(rank, "x"), nglob)
    y, _ = specfem_bin.read_coords(info.bin_path(rank, "y"), nglob)
    z, _ = specfem_bin.read_coords(info.bin_path(rank, "z"), nglob)
    xyz = np.stack([x, y, z], axis=-1)
    elem_coords = specfem_bin.gather_element_coordinates(ibool, xyz)[local_index]
    grid = elem_coords.reshape((5, 5, 5, 3))  # (k, j, i, 3)

    origin = grid[0, 0, 0]
    end_i = grid[0, 0, 4] - origin
    end_j = grid[0, 4, 0] - origin
    end_k = grid[4, 0, 0] - origin
    assert end_i == pytest.approx((100.0, 0.0, 0.0), abs=1e-9)
    assert end_j == pytest.approx((0.0, 200.0, 0.0), abs=1e-9)
    assert end_k == pytest.approx((0.0, 0.0, 400.0), abs=1e-9)


# ===========================================================================
# 2. element_id recovery round-trip + partition independence
# ===========================================================================


def test_recover_element_ids_round_trip(tmp_path):
    info = write_synthetic_specfem_tree(tmp_path / "tree", partition_variant="a")
    part = specfem_bin.read_part_array_vtk(info.part_array_path())

    all_recovered: list[int] = []
    for rank in range(info.nproc):
        recovered = specfem_bin.recover_element_ids(part, rank)
        expected = info.elements_for_rank(rank)
        assert list(recovered) == expected
        all_recovered.extend(int(v) for v in recovered)

        db = specfem_bin.parse_database(info.database_path(rank))
        row_of_iloc = {int(v): i for i, v in enumerate(db.node_iloc)}
        for k, global_id in enumerate(recovered):
            elem = db.elements[k]
            corner_rows = [row_of_iloc[nid] for nid in elem.node_loc_ids]
            corners_from_db = db.node_xyz[corner_rows]
            assert np.array_equal(
                corners_from_db, info.elements[int(global_id)].corners8
            )

    assert sorted(all_recovered) == list(range(info.n_elements_total))


def test_element_id_partition_independence_bin_level(tmp_path):
    info_a = write_synthetic_specfem_tree(tmp_path / "variant_a", partition_variant="a")
    info_b = write_synthetic_specfem_tree(tmp_path / "variant_b", partition_variant="b")

    def corner_set_by_element_id(info) -> dict[int, frozenset]:
        part = specfem_bin.read_part_array_vtk(info.part_array_path())
        result: dict[int, frozenset] = {}
        for rank in range(info.nproc):
            recovered = specfem_bin.recover_element_ids(part, rank)
            db = specfem_bin.parse_database(info.database_path(rank))
            row_of_iloc = {int(v): i for i, v in enumerate(db.node_iloc)}
            for k, global_id in enumerate(recovered):
                elem = db.elements[k]
                corner_rows = [row_of_iloc[nid] for nid in elem.node_loc_ids]
                corners = db.node_xyz[corner_rows]
                result[int(global_id)] = frozenset(tuple(row) for row in corners)
        return result

    sets_a = corner_set_by_element_id(info_a)
    sets_b = corner_set_by_element_id(info_b)
    assert set(sets_a) == set(sets_b) == set(range(info_a.n_elements_total))
    for element_id in sets_a:
        assert sets_a[element_id] == sets_b[element_id]


# ===========================================================================
# 3. Shared-face effective check
# ===========================================================================


def test_shared_face_reduces_nglob_below_ngll_times_nspec(tmp_path):
    info = write_synthetic_specfem_tree(tmp_path / "tree", partition_variant="a")
    shared_rank = info.rank_of(0)
    assert info.rank_of(1) == shared_rank  # E0, E1 on the same rank in variant "a"

    ibool, _, _ = specfem_bin.read_ibool(info.bin_path(shared_rank, "ibool"), info.ngll)
    nspec_rank = ibool.shape[0]
    nglob_rank = int(ibool.max()) + 1
    assert nglob_rank < info.ngll * nspec_rank


# ===========================================================================
# 4. Destructive / negative paths
# ===========================================================================


def test_corrupted_negative_marker_raises_with_subrecord_hint(tmp_path):
    good_path = tmp_path / "good.bin"
    _write_single_record(good_path, np.arange(8, dtype="<i4").tobytes())
    data = bytearray(good_path.read_bytes())
    # Flip the leading marker to a negative value.
    marker = np.frombuffer(bytes(data[:4]), dtype="<i4")[0]
    data[:4] = np.array([-abs(int(marker)) - 1], dtype="<i4").tobytes()
    bad_path = tmp_path / "bad.bin"
    bad_path.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="subrecord"):
        specfem_bin.read_records(bad_path)


def test_ibool_zero_or_negative_raises(tmp_path):
    # ngll=1, nspec=3, one-based values [0, 1, 2] -- entry 0 is invalid.
    payload = np.array([0, 1, 2], dtype="<i4").tobytes()
    path = tmp_path / "ibool_nonpositive.bin"
    _write_single_record(path, payload)
    with pytest.raises(ValueError, match="non-positive"):
        specfem_bin.read_ibool(path, ngll=1)


def test_ibool_gap_is_not_a_surjection_raises(tmp_path):
    # ngll=1, nspec=3, one-based values [1, 2, 4] -- 3 is missing (gap).
    payload = np.array([1, 2, 4], dtype="<i4").tobytes()
    path = tmp_path / "ibool_gap.bin"
    _write_single_record(path, payload)
    with pytest.raises(ValueError, match="surjection"):
        specfem_bin.read_ibool(path, ngll=1)


def test_gather_element_coordinates_out_of_range_raises():
    ibool = np.array([[0, 1, 5]], dtype=np.int64)  # index 5 out of range
    xyz = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="nglob mismatch"):
        specfem_bin.gather_element_coordinates(ibool, xyz)


def test_truncated_file_raises(tmp_path):
    good_path = tmp_path / "good.bin"
    _write_single_record(good_path, np.arange(16, dtype="<i4").tobytes())
    truncated_path = tmp_path / "truncated.bin"
    truncated_path.write_bytes(good_path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="auto-detect"):
        specfem_bin.read_records(truncated_path)


def test_excess_trailing_bytes_raises(tmp_path):
    good_path = tmp_path / "good.bin"
    _write_single_record(good_path, np.arange(16, dtype="<i4").tobytes())
    excess_path = tmp_path / "excess.bin"
    excess_path.write_bytes(good_path.read_bytes() + b"\x00\x01\x02")
    with pytest.raises(ValueError, match="auto-detect"):
        specfem_bin.read_records(excess_path)


def test_big_endian_file_is_auto_detected(tmp_path):
    payload = np.array([10, 20, 30, 40], dtype=">i4").tobytes()
    data = _fortran_record_bytes(payload, endian=">", marker_bytes=4)
    path = tmp_path / "big_endian.bin"
    path.write_bytes(data)

    layout, payloads = specfem_bin.read_records(path)
    assert layout.endianness == ">"
    assert layout.record_marker_bytes == 4
    assert len(payloads) == 1
    values = np.frombuffer(payloads[0], dtype=">i4")
    assert values.tolist() == [10, 20, 30, 40]


def test_missing_displacement_component_raises(tmp_path):
    write_displacement_snapshot_files(
        tmp_path / "snaps",
        nspec=2,
        ngll=8,
        its=[1, 2, 3],
        omit_component_for_last_it="Y",
    )
    with pytest.raises(ValueError, match=r"missing displacement component"):
        specfem_bin.read_displacement_snapshots(tmp_path / "snaps", nspec=2, ngll=8)


def test_inconsistent_it_set_across_components_raises(tmp_path):
    snaps_dir = tmp_path / "snaps_it_mismatch"
    write_displacement_snapshot_files(snaps_dir, nspec=2, ngll=8, its=[1, 2])
    # Add an extra it=3 for X only (Y, Z absent) -- an inconsistent 'it' set.
    extra = np.arange(2 * 8, dtype="<f4")
    _write_single_record(snaps_dir / "displ_X_it000003.bin", extra.tobytes())
    with pytest.raises(ValueError, match=r"missing displacement component"):
        specfem_bin.read_displacement_snapshots(snaps_dir, nspec=2, ngll=8)


def test_displacement_snapshots_normal_path(tmp_path):
    snaps_dir = tmp_path / "snaps_ok"
    expected = write_displacement_snapshot_files(snaps_dir, nspec=2, ngll=8, its=[5, 7])
    displacement, its, layout = specfem_bin.read_displacement_snapshots(
        snaps_dir, nspec=2, ngll=8
    )
    assert its == [5, 7]
    assert layout.endianness == "<"
    assert displacement.shape == (2, 2, 8, 3)
    assert displacement == pytest.approx(expected)


def test_rho_non_positive_raises(tmp_path):
    payload = np.array([2000.0, 0.0, 3000.0], dtype="<f4").tobytes()
    path = tmp_path / "rho_bad.bin"
    _write_single_record(path, payload)
    with pytest.raises(ValueError, match="strictly positive"):
        specfem_bin.read_rho(path, nspec=1, ngll=3)


def test_database_mat1_out_of_range_raises(tmp_path):
    payloads: list[bytes] = []
    payloads.append(_int_payload(1))  # nnodes_loc = 1
    payloads.append(
        np.array([1], dtype="<i4").tobytes() + np.zeros(3, dtype="<f8").tobytes()
    )  # 1 node record
    payloads.append(np.array([1, 0], dtype="<i4").tobytes())  # ndef=1, nundef=0
    mat_values = np.zeros(17, dtype="<f8")
    mat_values[7] = 2.0
    payloads.append(mat_values.tobytes())  # 1 material record
    payloads.append(_int_payload(1))  # nspec_local = 1
    # element record: iloc=1, mat1=2 (invalid -- only 1 material defined),
    # mat2=0, 8 node ids all = 1.
    elem_values = [1, 2, 0] + [1] * 8
    payloads.append(np.array(elem_values, dtype="<i4").tobytes())

    path = tmp_path / "proc000000_Database"
    _write_records_file(path, payloads)

    with pytest.raises(ValueError, match="mat1"):
        specfem_bin.parse_database(path)


# ===========================================================================
# 5. read_part_array_vtk truncation guards (post-completion-review fix #1,
#    2026-08-04): every other parse function in this module raises
#    ``ValueError`` on malformed/truncated input, but
#    ``read_part_array_vtk``'s three block-reading loops (POINTS/CELLS/
#    CELL_DATA) previously indexed ``lines[i]`` without first checking
#    ``i < n_lines``, so a file whose declared count exceeds its actual
#    line count raised a raw ``IndexError`` instead -- an importer
#    contract violation (a corrupted/truncated solver output must be
#    rejected cleanly, not crash with an implementation-detail exception).
# ===========================================================================


def test_read_part_array_vtk_truncated_points_raises(tmp_path):
    # Declares 4 points (12 coordinate values) but only provides 2 point
    # rows (6 values) before the file ends.
    text = (
        "# vtk DataFile Version 3.0\n"
        "gragra truncated-POINTS negative-path fixture\n"
        "ASCII\n"
        "DATASET UNSTRUCTURED_GRID\n"
        "POINTS 4 double\n"
        "0.0 0.0 0.0\n"
        "1.0 0.0 0.0\n"
    )
    path = tmp_path / "truncated_points.vtk"
    path.write_text(text)
    with pytest.raises(ValueError, match="POINTS block truncated"):
        specfem_bin.read_part_array_vtk(path)


def test_read_part_array_vtk_truncated_cells_raises(tmp_path):
    # Declares 2 CELLS rows but only provides 1 before the file ends.
    text = (
        "# vtk DataFile Version 3.0\n"
        "gragra truncated-CELLS negative-path fixture\n"
        "ASCII\n"
        "DATASET UNSTRUCTURED_GRID\n"
        "POINTS 2 double\n"
        "0.0 0.0 0.0\n"
        "1.0 0.0 0.0\n"
        "CELLS 2 18\n"
        "8 0 1 2 3 4 5 6 7\n"
    )
    path = tmp_path / "truncated_cells.vtk"
    path.write_text(text)
    with pytest.raises(ValueError, match="CELLS block truncated"):
        specfem_bin.read_part_array_vtk(path)


def test_read_part_array_vtk_truncated_cell_data_raises(tmp_path):
    # Declares CELL_DATA 2 (2 elem_flag scalar values) but only provides 1
    # value before the file ends.
    text = (
        "# vtk DataFile Version 3.0\n"
        "gragra truncated-CELL_DATA negative-path fixture\n"
        "ASCII\n"
        "DATASET UNSTRUCTURED_GRID\n"
        "POINTS 1 double\n"
        "0.0 0.0 0.0\n"
        "CELLS 1 9\n"
        "8 0 0 0 0 0 0 0 0\n"
        "CELL_TYPES 1\n"
        "12\n"
        "CELL_DATA 2\n"
        "SCALARS elem_flag int 1\n"
        "LOOKUP_TABLE default\n"
        "0\n"
    )
    path = tmp_path / "truncated_cell_data.vtk"
    path.write_text(text)
    with pytest.raises(ValueError, match="CELL_DATA block truncated"):
        specfem_bin.read_part_array_vtk(path)
