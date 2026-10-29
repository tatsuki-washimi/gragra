from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from _specfem_synthetic import (
    write_rank_displacement_snapshot_files,
    write_synthetic_specfem_tree,
)

import gragra
from gragra.adapters.specfem import export, schema, specfem_bin
from gragra.adapters.specfem.streaming import SpecfemGLLStreamReader
from scripts import research_wave5_specfem_export as bounded

ROOT = Path(__file__).resolve().parents[1]


def test_imports_lane_source() -> None:
    assert Path(gragra.__file__).resolve() == ROOT / "src/gragra/__init__.py"


def _enable_time_output(info) -> None:
    text = info.par_file_path.read_text()
    info.par_file_path.write_text(
        text.replace(
            "SAVE_DISPLACEMENT               = .false.",
            "SAVE_DISPLACEMENT               = .true.",
        )
    )


def _config(info, output: Path) -> export.ExportConfig:
    return export.ExportConfig(
        specfem_dir=info.root,
        output_path=output,
        output_sampling_kind="time_sampled",
        ngll_shape=info.ngll_shape,
        build_identity={
            "build_kind": "package",
            "package_version": "bounded-test",
            "package_digest": "sha256:" + "ab" * 32,
            "build_configuration": {},
        },
        exporter_code_ref="bounded-test",
        f_resolved_estimate=1.0,
        t0=10.0,
        t0_source="test_fixture_injected",
        cadence={
            "f_max_analysis": 5.0,
            "f_min_analysis": 0.1,
            "minimum_cycles": 3.0,
            "sampling_oversampling_factor": 4.0,
            "antialias_filter_applied": False,
        },
        cadence_provenance="confirmed",
        chunk_element_size=2,
        gzip_level=1,
    )


def _structured_database(global_ids: list[int], *, length_m: int = 500):
    nx = length_m // 250
    xmin = 3000 - length_m // 2
    points: list[tuple[float, float, float]] = []
    elements = []
    for global_id in global_ids:
        iz, remainder = divmod(global_id, nx * nx)
        iy, ix = divmod(remainder, nx)
        x, y, z = xmin + 250 * ix, xmin + 250 * iy, -3000 + 250 * iz
        corners = (
            (x, y, z),
            (x + 250, y, z),
            (x + 250, y + 250, z),
            (x, y + 250, z),
            (x, y, z + 250),
            (x + 250, y, z + 250),
            (x + 250, y + 250, z + 250),
            (x, y + 250, z + 250),
        )
        first = len(points) + 1
        points.extend(corners)
        elements.append(SimpleNamespace(node_loc_ids=tuple(range(first, first + 8))))
    return SimpleNamespace(
        nnodes_loc=len(points),
        node_iloc=np.arange(1, len(points) + 1, dtype=np.int64),
        node_xyz=np.asarray(points, dtype=np.float64),
        elements=elements,
    )


def test_prepare_structured_part_array_recovers_exact_global_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_path = tmp_path / "DATABASES_MPI_RUN"
    local_path.mkdir()
    ids_by_rank = {
        rank: [global_id for global_id in range(48) if global_id % 4 == rank]
        for rank in range(4)
    }

    def parse_database(path: Path):
        rank = int(path.name[4:10])
        return _structured_database(ids_by_rank[rank])

    monkeypatch.setattr(specfem_bin, "parse_database", parse_database)
    report = bounded.prepare_structured_part_array(local_path, length_m=500, nproc=4)

    part = specfem_bin.read_part_array_vtk(local_path / "part_array.vtk")
    assert report["status"] == "PART_ARRAY_COMPLETE"
    assert report["rank_element_counts"] == [12, 12, 12, 12]
    np.testing.assert_array_equal(part.elem_flag, np.arange(48) % 4)
    assert part.cells[0].tolist() == [0, 1, 4, 3, 9, 10, 13, 12]
    assert report["output_sha256"] == bounded._file_sha256(
        local_path / "part_array.vtk"
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        bounded.prepare_structured_part_array(local_path, length_m=500, nproc=4)


def test_prepare_structured_part_array_retains_incomplete_on_bad_local_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_path = tmp_path / "DATABASES_MPI_RUN"
    local_path.mkdir()
    ids_by_rank = {
        rank: [global_id for global_id in range(48) if global_id % 4 == rank]
        for rank in range(4)
    }
    ids_by_rank[2][0], ids_by_rank[2][1] = ids_by_rank[2][1], ids_by_rank[2][0]

    def parse_database(path: Path):
        rank = int(path.name[4:10])
        return _structured_database(ids_by_rank[rank])

    monkeypatch.setattr(specfem_bin, "parse_database", parse_database)
    with pytest.raises(ValueError, match="not strictly increasing"):
        bounded.prepare_structured_part_array(local_path, length_m=500, nproc=4)

    assert not (local_path / "part_array.vtk").exists()
    failure = local_path / "part_array.vtk.incomplete.failure.json"
    assert failure.exists()
    assert "PART_ARRAY_INCOMPLETE_RETAINED" in failure.read_text()


def test_bounded_export_matches_eager_export(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    info = write_synthetic_specfem_tree(tmp_path / "tree", partition_variant="a")
    _enable_time_output(info)
    expected = write_rank_displacement_snapshot_files(info, its=[1, 2, 3])
    eager_path = tmp_path / "eager.h5"
    bounded_path = tmp_path / "bounded.h5"
    export.export_specfem_gll_hdf5(_config(info, eager_path))

    report = bounded.export_bounded_specfem_gll_hdf5(
        _config(info, bounded_path),
        expected_native_steps=(1, 2, 3),
        source_block_elements=2,
    )

    assert report["status"] == "CANONICAL_HDF5_COMPLETE"
    assert report["source_block_elements"] == 2
    assert report["samples_written"] == 3
    assert report["peak_input_elements"] == expected.shape[1]
    assert report["sample_frame_elements"] == expected.shape[1]
    assert report["sample_frame_bytes"] == expected.shape[1] * 125 * 3 * 8
    assert report["write_strategy"] == (
        "one_float64_sample_frame_then_contiguous_global_element_slabs"
    )
    assert report["readback_strategy"] == (
        "one_float64_raw_sample_frame_then_contiguous_global_element_slabs"
    )
    expected_global_blocks = 3 * ((expected.shape[1] + 1) // 2)
    assert report["write_blocks"] == expected_global_blocks
    assert report["readback_blocks"] == expected_global_blocks
    assert report["scientific_cpml_exclusion_claim"] is False
    assert not bounded_path.with_suffix(".h5.partial").exists()
    with h5py.File(eager_path, "r") as eager, h5py.File(bounded_path, "r") as actual:
        for dataset in schema.required_dataset_paths("time_sampled"):
            np.testing.assert_array_equal(actual[dataset][()], eager[dataset][()])
        np.testing.assert_array_equal(
            actual[schema.PATH_FIELD_DISPLACEMENT][()], expected
        )
        assert actual[schema.METADATA_GROUP].attrs["pml_exclusion_applied"]
        assert not actual[schema.PATH_MESH_IS_PML][()].any()
        assert (
            actual[schema.METADATA_GROUP].attrs["scientific_cpml_exclusion_claim"] == 0
        )


def test_bounded_export_retains_partial_on_missing_snapshot_component(
    tmp_path: Path,
) -> None:
    info = write_synthetic_specfem_tree(tmp_path / "tree", partition_variant="a")
    _enable_time_output(info)
    write_rank_displacement_snapshot_files(info, its=[1, 2])
    missing = info.local_path / "proc000001_displ_Z_it000002.bin"
    missing.unlink()
    output = tmp_path / "bounded.h5"

    with pytest.raises(ValueError, match="snapshot coverage"):
        bounded.export_bounded_specfem_gll_hdf5(
            _config(info, output),
            expected_native_steps=(1, 2),
            source_block_elements=2,
        )

    partial = output.with_suffix(".h5.partial")
    assert partial.exists()
    failure = output.with_suffix(".h5.partial.failure.json")
    assert failure.exists()
    assert "missing" in failure.read_text()


def test_bounded_export_records_interruption_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = write_synthetic_specfem_tree(tmp_path / "tree", partition_variant="a")
    _enable_time_output(info)
    write_rank_displacement_snapshot_files(info, its=[1])
    output = tmp_path / "bounded.h5"

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt("focused interruption probe")

    monkeypatch.setattr(bounded, "_snapshot_coverage", interrupt)
    with pytest.raises(KeyboardInterrupt, match="focused interruption probe"):
        bounded.export_bounded_specfem_gll_hdf5(
            _config(info, output), expected_native_steps=(1,)
        )

    assert not output.exists()
    assert output.with_suffix(".h5.partial").exists()
    failure = output.with_suffix(".h5.partial.failure.json")
    assert failure.exists()
    assert "KeyboardInterrupt" in failure.read_text()


def test_bounded_export_refuses_overwrite(tmp_path: Path) -> None:
    info = write_synthetic_specfem_tree(tmp_path / "tree", partition_variant="a")
    _enable_time_output(info)
    write_rank_displacement_snapshot_files(info, its=[1])
    output = tmp_path / "bounded.h5"
    output.write_text("preserve")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        bounded.export_bounded_specfem_gll_hdf5(
            _config(info, output), expected_native_steps=(1,)
        )
    assert output.read_text() == "preserve"


def test_bounded_export_rejects_finite_readback_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5py = pytest.importorskip("h5py")
    info = write_synthetic_specfem_tree(tmp_path / "tree", partition_variant="a")
    _enable_time_output(info)
    write_rank_displacement_snapshot_files(info, its=[1, 2])
    output = tmp_path / "bounded.h5"
    original = bounded._readback_displacement_identity

    def corrupt_then_readback(partial: Path, *args, **kwargs):
        with h5py.File(partial, "r+") as h5:
            field = h5[schema.PATH_FIELD_DISPLACEMENT]
            field[0, 0, 0, 0] = float(field[0, 0, 0, 0]) + 1.0
        return original(partial, *args, **kwargs)

    monkeypatch.setattr(
        bounded, "_readback_displacement_identity", corrupt_then_readback
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        bounded.export_bounded_specfem_gll_hdf5(
            _config(info, output),
            expected_native_steps=(1, 2),
            source_block_elements=2,
        )

    assert not output.exists()
    assert output.with_suffix(".h5.partial").exists()
    failure = output.with_suffix(".h5.partial.failure.json")
    assert failure.exists()
    assert "identity mismatch" in failure.read_text()


def test_bounded_export_publish_does_not_replace_racing_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("h5py")
    info = write_synthetic_specfem_tree(tmp_path / "tree", partition_variant="a")
    _enable_time_output(info)
    write_rank_displacement_snapshot_files(info, its=[1])
    output = tmp_path / "bounded.h5"
    original = bounded._readback_displacement_identity

    def race_after_readback(partial: Path, *args, **kwargs):
        report = original(partial, *args, **kwargs)
        output.write_text("racing publisher must be preserved")
        return report

    monkeypatch.setattr(bounded, "_readback_displacement_identity", race_after_readback)
    with pytest.raises(FileExistsError):
        bounded.export_bounded_specfem_gll_hdf5(
            _config(info, output), expected_native_steps=(1,)
        )

    assert output.read_text() == "racing publisher must be preserved"
    assert output.with_suffix(".h5.partial").exists()
    assert output.with_suffix(".h5.partial.failure.json").exists()


def test_bounded_export_records_log_derived_actual_dt(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    info = write_synthetic_specfem_tree(tmp_path / "tree", partition_variant="a")
    _enable_time_output(info)
    write_rank_displacement_snapshot_files(info, its=[1, 2])
    output = tmp_path / "bounded.h5"

    bounded.export_bounded_specfem_gll_hdf5(
        _config(info, output),
        expected_native_steps=(1, 2),
        source_block_elements=2,
        actual_dt_s=0.123456789,
    )

    with h5py.File(output, "r") as h5:
        assert h5[schema.METADATA_GROUP].attrs["dt_s"] == 0.123456789
        assert h5[schema.METADATA_GROUP].attrs["par_file_dt_s"] == 0.05
        assert h5[schema.METADATA_GROUP].attrs["actual_dt_source"] == (
            "accepted_output_solver_log"
        )


def _write_invalid_cadence_source(tmp_path: Path) -> Path:
    h5py = pytest.importorskip("h5py")
    info = write_synthetic_specfem_tree(tmp_path / "tree", partition_variant="a")
    _enable_time_output(info)
    write_rank_displacement_snapshot_files(info, its=[1, 2, 3])
    source = tmp_path / "source.h5"
    export.export_specfem_gll_hdf5(_config(info, source))
    with h5py.File(source, "r+") as h5:
        metadata = h5[schema.METADATA_GROUP].attrs
        metadata["f_min_analysis"] = 5.0
        metadata["f_max_analysis"] = 5.0
        metadata["minimum_cycles"] = 0.1
        h5[schema.PATH_MESH_MATERIAL_ID][:] = 1
    return source


def test_cadence_correction_preserves_all_datasets_and_original(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    source = _write_invalid_cadence_source(tmp_path)
    corrected = tmp_path / "corrected.h5"

    report = bounded.correct_canonical_cadence_no_overwrite(
        source,
        corrected,
        expected_source_f_min=5.0,
        corrected_f_min=4.0,
        data_block_elements=2,
    )

    assert report["status"] == "CANONICAL_CADENCE_CORRECTION_COMPLETE"
    assert report["all_datasets_bitwise_equal"] is True
    assert report["source_f_min_analysis"] == 5.0
    assert report["corrected_f_min_analysis"] == 4.0
    assert not corrected.with_suffix(".h5.partial").exists()
    with h5py.File(source, "r") as original, h5py.File(corrected, "r") as actual:
        assert original[schema.METADATA_GROUP].attrs["f_min_analysis"] == 5.0
        assert actual[schema.METADATA_GROUP].attrs["f_min_analysis"] == 4.0
        for dataset in schema.required_dataset_paths("time_sampled"):
            np.testing.assert_array_equal(actual[dataset][()], original[dataset][()])
    with SpecfemGLLStreamReader(
        corrected,
        profile_density_range=None,
        sample_slice=slice(0, 3),
        slab_elements=2,
        memory_budget_bytes=10_000_000,
    ) as reader:
        assert reader.provider.source_count == 6 * 125

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        bounded.correct_canonical_cadence_no_overwrite(
            source,
            corrected,
            expected_source_f_min=5.0,
            corrected_f_min=4.0,
            data_block_elements=2,
        )


@pytest.mark.parametrize("failed_staging_name", ["data", "report"])
def test_cadence_correction_cleanup_failure_keeps_published_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_staging_name: str,
) -> None:
    source = _write_invalid_cadence_source(tmp_path)
    corrected = tmp_path / "corrected.h5"
    partial = corrected.with_suffix(".h5.partial")
    report_path = corrected.with_suffix(".h5.cadence-correction.json")
    report_partial = report_path.with_suffix(".json.partial")
    failed_staging = {
        "data": partial,
        "report": report_partial,
    }[failed_staging_name]
    other_staging = report_partial if failed_staging == partial else partial
    original_unlink = Path.unlink

    def fail_selected_staging_cleanup(path: Path, *args, **kwargs) -> None:
        if path == failed_staging:
            raise PermissionError("synthetic staging cleanup failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_selected_staging_cleanup)

    report = bounded.correct_canonical_cadence_no_overwrite(
        source,
        corrected,
        expected_source_f_min=5.0,
        corrected_f_min=4.0,
        data_block_elements=2,
    )

    assert report["status"] == "CANONICAL_CADENCE_CORRECTION_COMPLETE"
    assert corrected.is_file()
    assert bounded._file_sha256(corrected) == report["output_sha256"]
    assert json.loads(report_path.read_text()) == report
    assert failed_staging.exists()
    assert not other_staging.exists()
    assert not corrected.with_suffix(".h5.partial.failure.json").exists()


def test_cadence_correction_report_race_preserves_competitor_and_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_invalid_cadence_source(tmp_path)
    corrected = tmp_path / "corrected.h5"
    report_path = corrected.with_suffix(".h5.cadence-correction.json")
    original_clone = bounded._clone_file_exclusive

    def clone_then_race(source_path: Path, partial: Path) -> str:
        strategy = original_clone(source_path, partial)
        report_path.write_text("competing report")
        return strategy

    monkeypatch.setattr(bounded, "_clone_file_exclusive", clone_then_race)
    with pytest.raises(FileExistsError):
        bounded.correct_canonical_cadence_no_overwrite(
            source,
            corrected,
            expected_source_f_min=5.0,
            corrected_f_min=4.0,
            data_block_elements=2,
        )

    assert not corrected.exists()
    assert corrected.with_suffix(".h5.partial").exists()
    assert corrected.with_suffix(".h5.partial.failure.json").exists()
    assert report_path.read_text() == "competing report"


def test_cadence_correction_binds_clone_to_preclone_source_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5py = pytest.importorskip("h5py")
    source = _write_invalid_cadence_source(tmp_path)
    expected_source_sha256 = bounded._file_sha256(source)
    corrected = tmp_path / "corrected.h5"
    original_clone = bounded._clone_file_exclusive

    def mutate_then_clone(source_path: Path, partial: Path) -> str:
        with h5py.File(source_path, "r+") as h5:
            density = h5[schema.PATH_MATERIAL_DENSITY]
            density[0, 0] = float(density[0, 0]) + 1.0
        return original_clone(source_path, partial)

    monkeypatch.setattr(bounded, "_clone_file_exclusive", mutate_then_clone)
    with pytest.raises(ValueError, match="clone SHA256"):
        bounded.correct_canonical_cadence_no_overwrite(
            source,
            corrected,
            expected_source_f_min=5.0,
            corrected_f_min=4.0,
            expected_source_sha256=expected_source_sha256,
            data_block_elements=2,
        )

    assert not corrected.exists()
    assert corrected.with_suffix(".h5.partial").exists()
    assert corrected.with_suffix(".h5.partial.failure.json").exists()


def test_cadence_correction_final_race_preserves_final_and_cleans_owned_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_invalid_cadence_source(tmp_path)
    corrected = tmp_path / "corrected.h5"
    partial = corrected.with_suffix(".h5.partial")
    report_path = corrected.with_suffix(".h5.cadence-correction.json")
    original_link = bounded.os.link

    def race_at_final(source_path: Path, destination: Path) -> None:
        if Path(source_path) == partial and Path(destination) == corrected:
            corrected.write_text("competing final")
        original_link(source_path, destination)

    monkeypatch.setattr(bounded.os, "link", race_at_final)
    with pytest.raises(FileExistsError):
        bounded.correct_canonical_cadence_no_overwrite(
            source,
            corrected,
            expected_source_f_min=5.0,
            corrected_f_min=4.0,
            data_block_elements=2,
        )

    assert corrected.read_text() == "competing final"
    assert partial.exists()
    assert not report_path.exists()
    assert report_path.with_suffix(".json.partial").exists()
    assert corrected.with_suffix(".h5.partial.failure.json").exists()


def test_protocol_export_uses_nonzero_width_validated_frequency_band(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        bounded,
        "_protocol_build_identity",
        lambda: {
            "build_kind": "source",
            "specfem_git_commit": "test",
            "source_tree_clean": True,
            "build_configuration": {},
            "compiler": {},
            "mpi": {},
        },
    )
    monkeypatch.setattr(bounded, "_runtime_entries", lambda: ())

    config = bounded.protocol_export_config(tmp_path, tmp_path / "output.h5")

    assert config.cadence["f_min_analysis"] == 2.4
    assert config.cadence["f_max_analysis"] == 2.5
    assert config.cadence["f_min_analysis"] < config.cadence["f_max_analysis"]
