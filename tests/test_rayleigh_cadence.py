"""Tests for the bounded Rayleigh movie-cadence research tool."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from _specfem_synthetic import _write_single_record


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "research_rayleigh_cadence.py"
    spec = importlib.util.spec_from_file_location("research_rayleigh_cadence", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_rank_snapshots(
    directory: Path,
    *,
    rank: int,
    nspec: int,
    ngll: int,
    steps: tuple[int, ...],
    offset: float = 0.0,
) -> None:
    base = np.arange(nspec * ngll, dtype=np.float32)
    for step in steps:
        for component_index, component in enumerate("XYZ"):
            values = base + step * 100.0 + component_index * 10.0 + offset
            path = directory / (f"proc{rank:06d}_displ_{component}_it{step:06d}.bin")
            _write_single_record(path, values.astype("<f4").tobytes())


def _source_case(tmp_path: Path) -> tuple[Path, Path]:
    case = tmp_path / "source-case"
    data = case / "DATA"
    static = case / "OUTPUT_FILES" / "DATABASES_MPI_RUN"
    data.mkdir(parents=True)
    static.mkdir(parents=True)
    (data / "Par_file").write_text(
        "\n".join(
            (
                "SUPPRESS_UTM_PROJECTION = .true.",
                "NPROC = 4",
                "NSTEP = 240",
                "DT = 0.005",
                "STACEY_ABSORBING_CONDITIONS = .true.",
                "STACEY_INSTEAD_OF_FREE_SURFACE = .false.",
                "BOTTOM_FREE_SURFACE = .false.",
                "MOVIE_VOLUME = .true.",
                "SAVE_DISPLACEMENT = .true.",
                "NTSTEP_BETWEEN_FRAMES = 1",
                "LOCAL_PATH = OUTPUT_FILES/DATABASES_MPI_RUN",
            )
        )
        + "\n"
    )
    (data / "CMTSOLUTION").write_text("fixture\n")
    (case / "OUTPUT_FILES" / "values_from_mesher.h").write_text("header\n")
    for rank in range(4):
        selected = static / f"proc{rank:06d}_external_mesh.bin"
        if rank == 0:
            target = case / "mesh-rank-0.bin"
            target.write_bytes(b"mesh-0")
            selected.symlink_to(target)
        else:
            selected.write_bytes(f"mesh-{rank}".encode())
    solver = tmp_path / "xspecfem3D"
    solver.write_text("fixture solver\n")
    solver.chmod(0o755)
    return case, solver


def test_prepare_cases_changes_only_movie_cadence(tmp_path):
    cadence = _load_script()
    source_case, solver = _source_case(tmp_path)

    prepared = cadence.prepare_cases(source_case, tmp_path / "W1", solver)

    m1 = prepared[1]
    m2 = prepared[2]
    par1 = (m1 / "DATA" / "Par_file").read_text()
    par2 = (m2 / "DATA" / "Par_file").read_text()
    assert "NSTEP = 8" in par1
    assert "NSTEP = 8" in par2
    assert "NTSTEP_BETWEEN_FRAMES = 1" in par1
    assert "NTSTEP_BETWEEN_FRAMES = 2" in par2
    assert par1.replace("FRAMES = 1", "FRAMES = 2") == par2
    assert (m1 / "DATA").resolve() != (m2 / "DATA").resolve()
    assert (m1 / "OUTPUT_FILES").resolve() != (m2 / "OUTPUT_FILES").resolve()
    assert (m1 / "OUTPUT_FILES" / "DATABASES_MPI_RUN").is_dir()
    assert (m2 / "OUTPUT_FILES" / "DATABASES_MPI_RUN").is_dir()
    assert not (
        m1 / "OUTPUT_FILES" / "DATABASES_MPI_RUN" / "proc000000_external_mesh.bin"
    ).is_symlink()
    assert (m1 / "run_rayleigh_hook.sh").stat().st_mode & 0o111

    manifest = json.loads((m1 / "input-manifest.json").read_text())
    assert manifest["cadence"] == 1
    assert manifest["fixed_conditions"]["nstep"] == 8
    assert len(manifest["static_inputs"]) == 5
    assert any(
        Path(row["destination"]).name == "values_from_mesher.h"
        for row in manifest["static_inputs"]
    )
    assert all(
        row["source_sha256_before"] == row["destination_sha256_before"]
        for row in manifest["static_inputs"]
    )

    (m1 / "DATA" / "CMTSOLUTION").write_text("changed destination\n")
    assert (source_case / "DATA" / "CMTSOLUTION").read_text() == "fixture\n"


def test_verify_inputs_records_unchanged_source_and_destination(tmp_path):
    cadence = _load_script()
    source_case, solver = _source_case(tmp_path)
    prepared = cadence.prepare_cases(source_case, tmp_path / "W1", solver)

    report = cadence.verify_inputs(prepared[1])

    assert report["all_sources_unchanged"]
    assert report["all_destinations_unchanged"]
    assert report["all_source_destination_pairs_equal"]


def test_compare_outputs_matches_rank_element_gll_component(tmp_path):
    cadence = _load_script()
    m1 = tmp_path / "m1"
    m2 = tmp_path / "m2"
    m1.mkdir()
    m2.mkdir()
    for rank in range(2):
        _write_rank_snapshots(m1, rank=rank, nspec=2, ngll=5, steps=tuple(range(1, 9)))
        _write_rank_snapshots(m2, rank=rank, nspec=2, ngll=5, steps=(2, 4, 6, 8))

    report = cadence.compare_outputs(m1, m2, ranks=2, nspec_rank=2, ngll_points=5)

    expected = 2 * 4 * 2 * 5 * 3
    assert report["expected_value_count"] == expected
    assert report["actual_value_count_m1"] == expected
    assert report["actual_value_count_m2"] == expected
    assert report["exact_match_count"] == expected
    assert report["maximum_absolute_difference"] == 0.0
    assert report["missing_logical_keys_m1"] == []
    assert report["missing_logical_keys_m2"] == []
    assert report["duplicate_logical_keys_m1"] == []
    assert report["duplicate_logical_keys_m2"] == []
    assert report["nonzero_value_count_m1"] > 0
    assert report["nonfinite_value_count_m1"] == 0
    assert report["time_mapping"] == [
        {
            "native_step": step,
            "array_index": step - 1,
            "physical_time_s": 10.0 + step * 0.005,
        }
        for step in (2, 4, 6, 8)
    ]


def test_compare_outputs_reports_changed_value(tmp_path):
    cadence = _load_script()
    m1 = tmp_path / "m1"
    m2 = tmp_path / "m2"
    m1.mkdir()
    m2.mkdir()
    _write_rank_snapshots(m1, rank=0, nspec=1, ngll=2, steps=(2, 4, 6, 8))
    _write_rank_snapshots(m2, rank=0, nspec=1, ngll=2, steps=(2, 4, 6, 8), offset=0.25)

    report = cadence.compare_outputs(m1, m2, ranks=1, nspec_rank=1, ngll_points=2)

    assert report["exact_match_count"] == 0
    assert report["maximum_absolute_difference"] == 0.25
