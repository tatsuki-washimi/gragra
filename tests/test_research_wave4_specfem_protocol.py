from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import gragra
from scripts import research_wave4_specfem_protocol as protocol

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_TEMPLATE_HASHES = {
    "DATA/Par_file": "df106c29ef147a2145e9079285331cac7de0ca3434b020e504ad6e70ebd8bce3",
    "DATA/CMTSOLUTION": (
        "1bd5e35a02eed2b6ced7434f4cbc5a0da182b65272ca3e5288e5c5640a452f26"
    ),
    "DATA/FORCESOLUTION": (
        "fc2ba397aa3c003354812f74f4850ef3f9743a777d1428dca96d0277d4c093a2"
    ),
    "DATA/STATIONS": "5874ea02cdf0e710da303465c899e8cc1a143bae0bbd912e520374df3baccbaa",
    "DATA/STATIONS_FILTERED": (
        "e72848ec42183ed99bb03675360288c6ae90e03b1c8fc1ea2b10d08ffe99718a"
    ),
    "DATA/observation_grid_to_use_for_gravity.txt": (
        "3764b554eda4ac1339384de575044daa767273d3299f52b0366f9407c3d5cc4f"
    ),
    "DATA/meshfem3D_files/Mesh_Par_file": (
        "58abb84f5cb7242fbbbeba7c7f75ad54846e4df353bead21d0f692d7deb37735"
    ),
    "DATA/meshfem3D_files/interfaces.txt": (
        "2e1b869e2c304835340ebf7a00e39ed4b30e6e8432f2a21080bed53afd3b75b0"
    ),
    "DATA/meshfem3D_files/interface1.txt": (
        "10378b2d420f200d38a7dffbfdcfb28a0694cf08842707b0bb8d4402bec34cb8"
    ),
}
SYNTHETIC_GENERATED_HASHES = {
    12_000: {
        "Mesh_Par_file": (
            "58abb84f5cb7242fbbbeba7c7f75ad54846e4df353bead21d0f692d7deb37735"
        ),
        "interfaces.txt": (
            "2e1b869e2c304835340ebf7a00e39ed4b30e6e8432f2a21080bed53afd3b75b0"
        ),
    },
    18_000: {
        "Mesh_Par_file": (
            "6d7dd451e790e3ac8dcb8458170cbea47f4760cec056265e9b0f8b09f81e0248"
        ),
        "interfaces.txt": (
            "945528b53431eb0f7c05698308b6158d9417d65d1d5c9d82d76bc0f872fef32d"
        ),
    },
}


@pytest.fixture
def synthetic_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    template = tmp_path / "synthetic-template"
    payloads = {
        relative: f"synthetic protocol fixture: {relative}\n".encode()
        for relative in protocol.INPUT_RELATIVE_PATHS
    }
    payloads["DATA/meshfem3D_files/Mesh_Par_file"] = b"""\
LATITUDE_MIN                    = -3000.0
LATITUDE_MAX                    = 9000.0
LONGITUDE_MIN                   = -3000.0
LONGITUDE_MAX                   = 9000.0
NEX_XI                          = 48
NEX_ETA                         = 48
1             48             1              48        1        12              1
"""
    payloads["DATA/meshfem3D_files/interfaces.txt"] = b"""\
1
 .true. 2 2 -3000.0 -3000.0 12000.0 12000.0
"""
    for relative, payload in payloads.items():
        target = template / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        target.chmod(0o640)
        assert protocol.sha256_file(target) == SYNTHETIC_TEMPLATE_HASHES[relative]

    provenance = tmp_path / "synthetic-provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "normal_12km_case": {
                    "input_files": [
                        {"path": relative, "sha256": digest}
                        for relative, digest in SYNTHETIC_TEMPLATE_HASHES.items()
                    ]
                },
                "solver": {
                    "compiler": "GNU Fortran 13.3.0",
                    "mpi": "Open MPI 4.1.6",
                },
            }
        )
    )
    monkeypatch.setattr(protocol, "PROVENANCE_INVENTORY", provenance)
    monkeypatch.setattr(protocol, "GENERATED_INPUT_HASHES", SYNTHETIC_GENERATED_HASHES)
    unavailable_template = tmp_path / "unavailable-external-template"
    assert not unavailable_template.exists()
    monkeypatch.setattr(protocol, "TEMPLATE_CASE", unavailable_template)
    return template


def test_imports_lane_source() -> None:
    assert Path(gragra.__file__).resolve() == ROOT / "src/gragra/__init__.py"


@pytest.mark.parametrize("length_m", [12_000, 18_000])
def test_generate_inputs_matches_pinned_hashes(
    tmp_path: Path, synthetic_template: Path, length_m: int
) -> None:
    case_root = tmp_path / f"L{length_m}"

    manifest = protocol.generate_case_inputs(synthetic_template, case_root, length_m)

    assert manifest["status"] == "INPUTS_VALIDATED"
    assert manifest["length_m"] == length_m
    assert len(manifest["files"]) == 9
    assert {item["relative_path"] for item in manifest["files"]} == set(
        protocol.INPUT_RELATIVE_PATHS
    )
    by_name = {item["relative_path"]: item["sha256"] for item in manifest["files"]}
    assert (
        by_name["DATA/meshfem3D_files/Mesh_Par_file"]
        == (protocol.GENERATED_INPUT_HASHES[length_m]["Mesh_Par_file"])
    )
    assert (
        by_name["DATA/meshfem3D_files/interfaces.txt"]
        == (protocol.GENERATED_INPUT_HASHES[length_m]["interfaces.txt"])
    )
    assert protocol.validate_case_inputs(case_root, length_m)["status"] == (
        "INPUTS_VALIDATED"
    )
    assert (case_root / "OUTPUT_FILES/DATABASES_MPI").is_dir()
    assert (case_root / "OUTPUT_FILES/DATABASES_MPI_RUN").is_dir()
    assert not (case_root / "OUTPUT_FILES/DATABASES_MPI_RUN").is_symlink()
    assert (case_root / "DATA/STATIONS_FILTERED").stat().st_mode & 0o200
    assert {item["relative_path"]: item["mode"] for item in manifest["files"]} == {
        relative: (synthetic_template / relative).stat().st_mode & 0o777
        for relative in protocol.INPUT_RELATIVE_PATHS
    }


def test_generate_inputs_rejects_nonempty_destination(
    tmp_path: Path, synthetic_template: Path
) -> None:
    destination = tmp_path / "occupied"
    destination.mkdir()
    (destination / "unexpected.txt").write_text("preserve me")

    with pytest.raises(FileExistsError, match="must be empty"):
        protocol.generate_case_inputs(synthetic_template, destination, 12_000)

    assert (destination / "unexpected.txt").read_text() == "preserve me"


def test_materialize_mesher_database_copies_static_files_without_symlink(
    tmp_path: Path,
) -> None:
    case = tmp_path / "case"
    source = case / "OUTPUT_FILES/DATABASES_MPI"
    target = case / "OUTPUT_FILES/DATABASES_MPI_RUN"
    source.mkdir(parents=True)
    target.mkdir()
    (source / "proc000000_Database").write_bytes(b"database")
    (source / "proc000000_mesh.vtk").write_bytes(b"mesh")

    report = protocol.materialize_mesher_database(case)

    assert report["status"] == "MESHER_DATABASE_MATERIALIZED"
    assert report["file_count"] == 2
    assert not target.is_symlink()
    assert (target / "proc000000_Database").read_bytes() == b"database"
    assert (target / "proc000000_mesh.vtk").read_bytes() == b"mesh"

    with pytest.raises(FileExistsError, match="must be empty"):
        protocol.materialize_mesher_database(case)


def test_gate_requires_parent_l_acceptance_even_with_capacity(tmp_path: Path) -> None:
    missing = tmp_path / "L-parent-acceptance.json"
    resources = protocol.ResourceSnapshot(
        free_bytes=protocol.PAIR_CAPACITY_GATE_BYTES + 1,
        mem_available_bytes=5 * 1024**3,
        swap_used_bytes=0,
    )

    report = protocol.evaluate_launch_gate(
        resources,
        resources,
        l_acceptance_path=missing,
    )

    assert report["status"] == "WAITING_L_ACCEPTANCE"
    assert report["capacity_status"] == "READY_CAPACITY"
    assert report["memory_status"] == "READY_MEMORY"
    assert report["swap_status"] == "READY_SWAP_STABLE"
    assert report["launch_allowed"] is False


def test_gate_accepts_only_parent_manifest_with_file_hashes(tmp_path: Path) -> None:
    acceptance = tmp_path / "L-parent-acceptance.json"
    acceptance.write_text(
        json.dumps(
            {
                "lane": "L",
                "status": "accepted",
                "base_commit": protocol.GRAGRA_BASE_COMMIT,
                "files": [
                    {"path": path, "sha256": "a" * 64}
                    for path in protocol.REQUIRED_L_FILES
                ],
            }
        )
    )
    resources = protocol.ResourceSnapshot(
        free_bytes=protocol.PAIR_CAPACITY_GATE_BYTES + 1,
        mem_available_bytes=5 * 1024**3,
        swap_used_bytes=0,
    )

    report = protocol.evaluate_launch_gate(
        resources,
        resources,
        l_acceptance_path=acceptance,
    )

    assert report["status"] == "READY_TO_LAUNCH"
    assert report["launch_allowed"] is True


def test_pinned_solver_checks_compiler_and_mpi_identities_portably(
    tmp_path: Path,
    synthetic_template: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert synthetic_template.is_dir()
    solver_root = tmp_path / "synthetic-solver"
    binary_hashes = {
        "xmeshfem3D": (
            "db1d4a3b7810556b95bc53c2c11f33545af04b76887d0e78592dccd1f3378c54"
        ),
        "xgenerate_databases": (
            "71467835de2d37bb9a59882fed31f9a1ceda0a7a24e991955de9235e1b31b729"
        ),
        "xspecfem3D": (
            "1b77bc1e8d120b283bbebdb387538c75b4d8560f1ae9af93f981822188f3955d"
        ),
    }
    for name, digest in binary_hashes.items():
        binary = solver_root / "bin" / name
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(f"synthetic executable: {name}\n".encode())
        assert protocol.sha256_file(binary) == digest

    def fake_run(argv, *, check, capture_output, text):
        assert check is True
        assert capture_output is True
        assert text is True
        command = tuple(str(value) for value in argv)
        if command[-2:] == ("rev-parse", "HEAD"):
            stdout = f"{protocol.SOLVER_COMMIT}\n"
        elif command[-3:] == ("status", "--porcelain", "--untracked-files=no"):
            stdout = ""
        elif command == ("/usr/bin/gfortran", "--version"):
            stdout = "GNU Fortran (Ubuntu) 13.3.0 13.3.0\n"
        elif command == ("/usr/bin/mpirun", "--version"):
            stdout = "mpirun (Open MPI) 4.1.6\n"
        else:
            raise AssertionError(f"unexpected command: {command}")
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(protocol, "SOLVER_ROOT", solver_root)
    monkeypatch.setattr(protocol, "BINARY_HASHES", binary_hashes)
    monkeypatch.setattr(protocol.subprocess, "run", fake_run)

    report = protocol.validate_pinned_solver()

    assert report["compiler"] == "GNU Fortran 13.3.0"
    assert report["mpi"] == "Open MPI 4.1.6"
    assert report["solver_root"] == str(solver_root)
    assert {item["sha256"] for item in report["binaries"]} == set(
        binary_hashes.values()
    )


def test_toolchain_identity_rejects_mismatched_version() -> None:
    with pytest.raises(RuntimeError, match="compiler identity"):
        protocol.validate_toolchain_identity(
            "GNU Fortran (Ubuntu) 14.1.0\n",
            "mpirun (Open MPI) 4.1.6\n",
            expected_compiler="GNU Fortran 13.3.0",
            expected_mpi="Open MPI 4.1.6",
        )


@pytest.mark.parametrize(
    ("before", "after", "status"),
    [
        (
            protocol.ResourceSnapshot(1, 5 * 1024**3, 0),
            protocol.ResourceSnapshot(1, 5 * 1024**3, 0),
            "WAITING_CAPACITY",
        ),
        (
            protocol.ResourceSnapshot(protocol.PAIR_CAPACITY_GATE_BYTES, 1, 0),
            protocol.ResourceSnapshot(protocol.PAIR_CAPACITY_GATE_BYTES, 1, 0),
            "WAITING_MEMORY",
        ),
        (
            protocol.ResourceSnapshot(
                protocol.PAIR_CAPACITY_GATE_BYTES, 5 * 1024**3, 0
            ),
            protocol.ResourceSnapshot(
                protocol.PAIR_CAPACITY_GATE_BYTES, 5 * 1024**3, 1
            ),
            "WAITING_SWAP_STABILITY",
        ),
    ],
)
def test_resource_gates_precede_l_acceptance(
    tmp_path: Path,
    before: protocol.ResourceSnapshot,
    after: protocol.ResourceSnapshot,
    status: str,
) -> None:
    report = protocol.evaluate_launch_gate(
        before,
        after,
        l_acceptance_path=tmp_path / "missing.json",
    )
    assert report["status"] == status
    assert report["launch_allowed"] is False


def _accepted_log(*, steps: range = range(1, 241), max_value: str = "1.0E-9") -> str:
    step_lines = "\n".join(
        f"Time step # {step}\nMax norm displacement vector U in all slices "
        f"(m) = {max_value}"
        for step in steps
    )
    return f"""
using single precision for the calculations
There is a total of 4 slices
number of elements elastic : 6912
total elastic elements : 27648
NSPEC_global_min = 6912
NSPEC_global_max = 6912
NSPEC_global_sum = 27648
number of steps between frames = 1
using Stacey first-order (P1) condition
scheme: Newmark
time step: 4.99999989E-03 s
number of time steps: 240
RAYLEIGH_DIAG rank= 0
GRAGRA_RAYLEIGH_PILOT t0 initialized and B1 acceleration computed
{step_lines}
End of the simulation
"""


def test_validate_solver_log_records_actual_dt_and_complete_steps() -> None:
    report = protocol.validate_solver_log_text(_accepted_log(), length_m=12_000)

    assert report["status"] == "SOLVER_LOG_ACCEPTED"
    assert report["actual_dt_s"] == pytest.approx(0.00499999989)
    assert report["observed_steps"] == [1, 240]
    assert report["effective_source_count"] == 0


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (_accepted_log(steps=range(1, 240)), "step coverage"),
        (_accepted_log(max_value="NaN"), "non-finite"),
        (_accepted_log().replace("scheme: Newmark", "scheme: leapfrog"), "Newmark"),
        (_accepted_log().replace("End of the simulation", ""), "required log string"),
    ],
)
def test_validate_solver_log_rejects_partial_or_nonfinite_output(
    text: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        protocol.validate_solver_log_text(text, length_m=12_000)


def test_raw_inventory_requires_all_eight_fields_and_all_steps(tmp_path: Path) -> None:
    for rank in range(4):
        for step in range(1, 241):
            for field in protocol.ELEMENT_LOCAL_FIELDS:
                (tmp_path / f"proc{rank:06d}_{field}_it{step:06d}.bin").touch()
            (tmp_path / f"proc{rank:06d}_div_glob_it{step:06d}.bin").touch()

    inventory = protocol.inventory_raw_outputs(tmp_path)
    assert inventory["status"] == "RAW_ALL_RETAINED_COMPLETE"
    assert inventory["file_count"] == 7_680
    assert inventory["displacement_file_count"] == 2_880

    (tmp_path / "proc000003_curl_Z_it000240.bin").unlink()
    with pytest.raises(ValueError, match="missing raw outputs"):
        protocol.inventory_raw_outputs(tmp_path)


def test_static_inventory_records_counts_without_dynamic_payloads(
    tmp_path: Path,
) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "mesh.bin").write_bytes(b"mesh")
    (tmp_path / "nested/database.bin").write_bytes(b"database")

    inventory = protocol.inventory_static_database(tmp_path)

    assert inventory["status"] == "STATIC_DATABASE_INVENTORIED"
    assert inventory["file_count"] == 2
    assert inventory["total_bytes"] == 12
    assert inventory["inventory_method"] == "stat-only"


def test_solver_output_directory_is_distinct_real_run_directory(
    tmp_path: Path,
) -> None:
    case = tmp_path / "case"
    (case / "OUTPUT_FILES/DATABASES_MPI").mkdir(parents=True)
    run_directory = case / "OUTPUT_FILES/DATABASES_MPI_RUN"
    run_directory.mkdir()

    assert protocol.solver_output_directory(case) == run_directory


def _prepared_case(tmp_path: Path, synthetic_template: Path) -> Path:
    case = tmp_path / "case"
    protocol.generate_case_inputs(synthetic_template, case, 12_000)
    return case


def test_interrupted_stage_is_recorded_before_and_after_runner(
    tmp_path: Path,
    synthetic_template: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _prepared_case(tmp_path, synthetic_template)
    monkeypatch.setattr(protocol, "validate_pinned_solver", lambda: {"status": "ok"})
    monkeypatch.setattr(
        protocol,
        "capture_launch_gate",
        lambda *_args, **_kwargs: {
            "status": "READY_TO_LAUNCH",
            "launch_allowed": True,
        },
    )

    def interrupted_runner(*_args, **_kwargs):
        started = json.loads((case / "run-manifest.json").read_text())
        assert started["stages"]["mesh"]["status"] == "STARTED_INCOMPLETE"
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        protocol.execute_stage(case, 12_000, "mesh", runner=interrupted_runner)

    retained = json.loads((case / "run-manifest.json").read_text())
    assert retained["status"] == "INTERRUPTED_RETAINED"
    assert retained["stages"]["mesh"]["status"] == "INTERRUPTED_RETAINED"
    assert retained["stages"]["mesh"]["exit_code"] is None
    assert retained["stages"]["mesh"]["error_type"] == "KeyboardInterrupt"


def test_solver_stage_rejects_partial_raw_before_runner(
    tmp_path: Path,
    synthetic_template: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _prepared_case(tmp_path, synthetic_template)
    local = case / "OUTPUT_FILES/DATABASES_MPI"
    local.mkdir(parents=True, exist_ok=True)
    (local / "proc000000_displ_X_it000001.bin").touch()
    (case / "run-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stages": {
                    "mesh": {"exit_code": 0},
                    "database": {"exit_code": 0},
                },
            }
        )
    )
    monkeypatch.setattr(protocol, "validate_pinned_solver", lambda: {"status": "ok"})
    monkeypatch.setattr(
        protocol,
        "capture_launch_gate",
        lambda *_args, **_kwargs: {
            "status": "READY_TO_LAUNCH",
            "launch_allowed": True,
        },
    )

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("runner must not be called")

    with pytest.raises(RuntimeError, match="dynamic .* already exists"):
        protocol.execute_stage(case, 12_000, "solver", runner=must_not_run)


def test_matched_actual_dt_requires_exact_pair_value() -> None:
    pair = protocol.validate_matched_actual_dt(
        {"length_m": 12_000, "actual_dt_s": 0.00499999989},
        {"length_m": 18_000, "actual_dt_s": 0.00499999989},
    )
    assert pair["status"] == "MATCHED_ACTUAL_DT"
    assert pair["physical_time_rule"] == "10 + native_it * actual_DT"

    with pytest.raises(ValueError, match="actual_DT mismatch"):
        protocol.validate_matched_actual_dt(
            {"length_m": 12_000, "actual_dt_s": 0.00499999989},
            {"length_m": 18_000, "actual_dt_s": 0.005},
        )
