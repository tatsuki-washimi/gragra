#!/usr/bin/env python3
"""Prepare and execute the pinned SPECFEM finite-domain protocol.

The driver intentionally exposes one solver stage per invocation.  It never
removes raw output and it refuses to launch until disk, memory, swap-trend,
solver provenance, input hashes, and the parent-reviewed L acceptance all
pass.  Scientific values in this file are the approved Proposed protocol;
resource readings and run records emitted by it are Observed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

GRAGRA_BASE_COMMIT: Final[str] = "bc502b025f070f8d0aab80d18e314b0a97c45482"
SOLVER_COMMIT: Final[str] = "2d4e94e5a2da5c2ac880a2aa026a208085d7942d"
PAIR_CAPACITY_GATE_BYTES: Final[int] = 179_484_805_430
MIN_MEM_AVAILABLE_BYTES: Final[int] = 4 * 1024**3
NPROC: Final[int] = 4
NSTEP: Final[int] = 240
NGLL: Final[int] = 125

SOLVER_ROOT: Final[Path] = Path("/home/washimi/work/specfem3d-rayleigh-init-20260907")
TEMPLATE_CASE: Final[Path] = (
    SOLVER_ROOT / "research_cases/rayleigh_b5_250m_12km_hookrun"
)
DEFAULT_EXTERNAL_LANE: Final[Path] = Path(
    "/home/washimi/work/gragra-research/2026-09-12-wave4-6/S"
)
DEFAULT_RUNS_ROOT: Final[Path] = DEFAULT_EXTERNAL_LANE / "runs"
DEFAULT_L_ACCEPTANCE: Final[Path] = Path(
    "/home/washimi/work/gragra-research/2026-09-12-wave4-6/verification/"
    "L-parent-acceptance.json"
)
PROTOCOL_DIR: Final[Path] = Path(
    "/home/washimi/work/gragra-research/2026-09-11-third-wave/specfem-protocol"
)
PROVENANCE_INVENTORY: Final[Path] = PROTOCOL_DIR / "provenance-inventory.json"
PROTOCOL_MANIFEST: Final[Path] = PROTOCOL_DIR / "protocol-manifest.json"

CASE_NAMES: Final[dict[int, str]] = {
    # Earlier preserved paths contain failed attempts. Scientific execution
    # proceeds only in this fresh directory.
    12_000: "L12000-regenerated-attempt4",
    18_000: "L18000-attempt2",
}
DOMAIN_FACTS: Final[dict[int, dict[str, int]]] = {
    12_000: {"horizontal_elements": 48, "elements": 27_648, "local": 6_912},
    18_000: {"horizontal_elements": 72, "elements": 62_208, "local": 15_552},
}
GENERATED_INPUT_HASHES: Final[dict[int, dict[str, str]]] = {
    12_000: {
        "Mesh_Par_file": (
            "13458384ab68635fbe0ac4ba246a68a5bb371e85465c7edbee27763b6a9656cc"
        ),
        "interfaces.txt": (
            "345b427308b0815ba5b3764da1fba4432e8a5320e31c210beccc225b15435edb"
        ),
    },
    18_000: {
        "Mesh_Par_file": (
            "71edeccfb2b8f77b9582356584a7abca9dd06ddd9d47fd43305f178c74f0b4f9"
        ),
        "interfaces.txt": (
            "099461a100b691f32ef0fcd6ea940a9222d67d685212e48d4dcb87f9b1fe7955"
        ),
    },
}
INPUT_RELATIVE_PATHS: Final[tuple[str, ...]] = (
    "DATA/Par_file",
    "DATA/CMTSOLUTION",
    "DATA/FORCESOLUTION",
    "DATA/STATIONS",
    "DATA/STATIONS_FILTERED",
    "DATA/observation_grid_to_use_for_gravity.txt",
    "DATA/meshfem3D_files/Mesh_Par_file",
    "DATA/meshfem3D_files/interfaces.txt",
    "DATA/meshfem3D_files/interface1.txt",
)
REQUIRED_L_FILES: Final[tuple[str, ...]] = (
    "src/gragra/integrators/streaming.py",
    "src/gragra/observables/streaming.py",
    "src/gragra/adapters/specfem/_validation.py",
    "src/gragra/adapters/specfem/streaming.py",
)
ELEMENT_LOCAL_FIELDS: Final[tuple[str, ...]] = (
    "displ_X",
    "displ_Y",
    "displ_Z",
    "div",
    "curl_X",
    "curl_Y",
    "curl_Z",
)
DISPLACEMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {"displ_X", "displ_Y", "displ_Z"}
)
BINARY_HASHES: Final[dict[str, str]] = {
    "xmeshfem3D": "526ccb62be395feadcfa819aca6f3833395e769a37c350b631c9310610e021f3",
    "xgenerate_databases": (
        "75e4c19233925946d6306aa7993d807d07979d3ef68c6c4208c8c4f88746631f"
    ),
    "xspecfem3D": "4c5106c73b00f2e47a62f592a546d6e20b9f88b0c6eee0c6b1c1d72697acc8bb",
}
STAGE_ARGV: Final[dict[str, tuple[str, ...]]] = {
    "mesh": (
        "mpirun",
        "--allow-run-as-root",
        "-np",
        "4",
        str(SOLVER_ROOT / "bin/xmeshfem3D"),
    ),
    "database": (
        "mpirun",
        "--allow-run-as-root",
        "-np",
        "4",
        str(SOLVER_ROOT / "bin/xgenerate_databases"),
    ),
    "solver": (
        "mpirun",
        "--allow-run-as-root",
        "-np",
        "4",
        "env",
        "GRAGRA_RAYLEIGH_PILOT=1",
        str(SOLVER_ROOT / "bin/xspecfem3D"),
    ),
}


@dataclass(frozen=True)
class ResourceSnapshot:
    """Observed launch resources on the filesystem holding the run."""

    free_bytes: int
    mem_available_bytes: int
    swap_used_bytes: int


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any, *, readonly: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    if readonly:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _write_or_verify_readonly_json(path: Path, value: Any) -> None:
    """Publish immutable evidence once, or require an exact existing record."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise FileExistsError(f"existing immutable record differs: {path}")
        return
    _write_json(path, value, readonly=True)


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(
            f"{label}: expected exactly one source occurrence, found {count}"
        )
    return text.replace(old, new, 1)


def _generated_text(template_case: Path, length_m: int) -> dict[str, bytes]:
    if length_m not in CASE_NAMES:
        raise ValueError(
            f"length_m must be one of {sorted(CASE_NAMES)}, got {length_m}"
        )
    values = {
        relative: (template_case / relative).read_bytes()
        for relative in INPUT_RELATIVE_PATHS
    }
    mesh_name = "DATA/meshfem3D_files/Mesh_Par_file"
    interface_name = "DATA/meshfem3D_files/interfaces.txt"
    mesh = values[mesh_name].decode("utf-8")
    half = length_m // 2
    minimum = 3000 - half
    maximum = 3000 + half
    nex = length_m // 250
    replacements = (
        (
            "LATITUDE_MIN                    = -3000.0",
            f"LATITUDE_MIN                    = {minimum}.0",
            "latitude minimum",
        ),
        (
            "LATITUDE_MAX                    = 9000.0",
            f"LATITUDE_MAX                    = {maximum}.0",
            "latitude maximum",
        ),
        (
            "LONGITUDE_MIN                   = -3000.0",
            f"LONGITUDE_MIN                   = {minimum}.0",
            "longitude minimum",
        ),
        (
            "LONGITUDE_MAX                   = 9000.0",
            f"LONGITUDE_MAX                   = {maximum}.0",
            "longitude maximum",
        ),
        (
            "NEX_XI                          = 48",
            f"NEX_XI                          = {nex}",
            "NEX_XI",
        ),
        (
            "NEX_ETA                         = 48",
            f"NEX_ETA                         = {nex}",
            "NEX_ETA",
        ),
        (
            "1             48             1              48        1        "
            "12              1",
            f"1             {nex}             1              {nex}        "
            "1        12              1",
            "region bounds",
        ),
    )
    for old, new, label in replacements:
        mesh = _replace_once(mesh, old, new, label=label)
    interface = values[interface_name].decode("utf-8")
    interface = _replace_once(
        interface,
        " .true. 2 2 -3000.0 -3000.0 12000.0 12000.0",
        f" .true. 2 2 {minimum}.0 {minimum}.0 {length_m}.0 {length_m}.0",
        label="interfaces data row",
    )
    values[mesh_name] = mesh.encode("utf-8")
    values[interface_name] = interface.encode("utf-8")
    return values


def _expected_template_hashes() -> dict[str, str]:
    inventory = json.loads(PROVENANCE_INVENTORY.read_text())
    return {
        item["path"]: item["sha256"]
        for item in inventory["normal_12km_case"]["input_files"]
    }


def validate_case_inputs(case_root: Path, length_m: int) -> dict[str, Any]:
    """Validate all nine DATA files without loading any large field."""
    expected_original = _expected_template_hashes()
    files: list[dict[str, Any]] = []
    generated = GENERATED_INPUT_HASHES[length_m]
    for relative in INPUT_RELATIVE_PATHS:
        path = case_root / relative
        if not path.is_file():
            raise ValueError(f"missing input file: {path}")
        digest = sha256_file(path)
        name = Path(relative).name
        expected = generated.get(name, expected_original[relative])
        if digest != expected:
            raise ValueError(
                f"input hash mismatch for {relative}: observed {digest}, "
                f"expected {expected}"
            )
        files.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": digest,
                "mode": stat.S_IMODE(path.stat().st_mode),
            }
        )
    return {
        "schema_version": 1,
        "classification": "Observed generated-input validation",
        "status": "INPUTS_VALIDATED",
        "length_m": length_m,
        "case_root": str(case_root.resolve()),
        "file_count": len(files),
        "files": files,
    }


def generate_case_inputs(
    template_case: Path, case_root: Path, length_m: int
) -> dict[str, Any]:
    """Build one new case atomically; preserve any failed partial directory."""
    template_case = Path(template_case)
    case_root = Path(case_root)
    if case_root.exists() and any(case_root.iterdir()):
        raise FileExistsError(f"case destination must be empty: {case_root}")
    if case_root.exists():
        case_root.rmdir()
    partial = case_root.with_name(
        f"{case_root.name}.prepare-partial-{uuid.uuid4().hex}"
    )
    partial.mkdir(parents=True)
    try:
        generated = _generated_text(template_case, length_m)
        for relative, payload in generated.items():
            path = partial / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            path.chmod(stat.S_IMODE((template_case / relative).stat().st_mode))
        (partial / "OUTPUT_FILES" / "DATABASES_MPI").mkdir(parents=True)
        (partial / "OUTPUT_FILES" / "DATABASES_MPI_RUN").mkdir()
        report = validate_case_inputs(partial, length_m)
        _write_json(partial / "input-manifest.json", report, readonly=True)
        partial.rename(case_root)
        report["case_root"] = str(case_root.resolve())
        return report
    except Exception as error:
        _write_json(
            partial / "preparation-failure.json",
            {
                "classification": "Observed partial preparation failure",
                "status": "INPUT_PREPARATION_FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "retention": "partial directory retained; no deletion performed",
            },
            readonly=True,
        )
        raise


def resource_snapshot(path: Path) -> ResourceSnapshot:
    usage = shutil.disk_usage(Path(path).resolve())
    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        match = re.fullmatch(r"([^:]+):\s+(\d+)\s+kB", line)
        if match:
            meminfo[match.group(1)] = int(match.group(2)) * 1024
    for required in ("MemAvailable", "SwapTotal", "SwapFree"):
        if required not in meminfo:
            raise RuntimeError(f"/proc/meminfo lacks {required}")
    return ResourceSnapshot(
        free_bytes=usage.free,
        mem_available_bytes=meminfo["MemAvailable"],
        swap_used_bytes=meminfo["SwapTotal"] - meminfo["SwapFree"],
    )


def _validated_l_acceptance(path: Path) -> tuple[bool, str | None]:
    if not path.is_file():
        return False, "parent L acceptance manifest is absent"
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return False, f"parent L acceptance manifest is unreadable: {error}"
    accepted_status = str(value.get("status", "")).lower().startswith("accepted")
    if value.get("lane") != "L" or not accepted_status:
        return False, "parent L acceptance status is not accepted"
    if value.get("base_commit") != GRAGRA_BASE_COMMIT:
        return False, "parent L acceptance base commit does not match"
    accepted = {
        item.get("path"): item.get("sha256")
        for item in value.get("files", [])
        if isinstance(item, dict)
    }
    for required in REQUIRED_L_FILES:
        digest = accepted.get(required)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            return False, f"parent L acceptance lacks a valid hash for {required}"
    return True, None


def evaluate_launch_gate(
    before: ResourceSnapshot,
    after: ResourceSnapshot,
    *,
    l_acceptance_path: Path = DEFAULT_L_ACCEPTANCE,
) -> dict[str, Any]:
    capacity_ready = (
        min(before.free_bytes, after.free_bytes) >= PAIR_CAPACITY_GATE_BYTES
    )
    memory_ready = min(before.mem_available_bytes, after.mem_available_bytes) >= (
        MIN_MEM_AVAILABLE_BYTES
    )
    swap_stable = after.swap_used_bytes <= before.swap_used_bytes
    l_ready, l_reason = _validated_l_acceptance(Path(l_acceptance_path))
    if not capacity_ready:
        status = "WAITING_CAPACITY"
    elif not memory_ready:
        status = "WAITING_MEMORY"
    elif not swap_stable:
        status = "WAITING_SWAP_STABILITY"
    elif not l_ready:
        status = "WAITING_L_ACCEPTANCE"
    else:
        status = "READY_TO_LAUNCH"
    return {
        "classification": "Observed launch preflight",
        "status": status,
        "launch_allowed": status == "READY_TO_LAUNCH",
        "capacity_status": "READY_CAPACITY" if capacity_ready else "WAITING_CAPACITY",
        "memory_status": "READY_MEMORY" if memory_ready else "WAITING_MEMORY",
        "swap_status": "READY_SWAP_STABLE" if swap_stable else "WAITING_SWAP_STABILITY",
        "l_status": "ACCEPTED" if l_ready else "WAITING_L_ACCEPTANCE",
        "l_reason": l_reason,
        "required_free_bytes": PAIR_CAPACITY_GATE_BYTES,
        "required_mem_available_bytes": MIN_MEM_AVAILABLE_BYTES,
        "before": asdict(before),
        "after": asdict(after),
        "l_acceptance_path": str(Path(l_acceptance_path)),
    }


def capture_launch_gate(
    runs_root: Path,
    *,
    l_acceptance_path: Path = DEFAULT_L_ACCEPTANCE,
    interval_s: float = 1.0,
) -> dict[str, Any]:
    before = resource_snapshot(runs_root)
    time.sleep(interval_s)
    after = resource_snapshot(runs_root)
    report = evaluate_launch_gate(before, after, l_acceptance_path=l_acceptance_path)
    report["observed_utc"] = _utc_now()
    report["runs_root"] = str(Path(runs_root).resolve())
    return report


def validate_pinned_solver() -> dict[str, Any]:
    """Verify the exact tracked-clean solver revision and all three binaries."""
    inventory = json.loads(PROVENANCE_INVENTORY.read_text())
    expected_solver = inventory["solver"]
    head = subprocess.run(
        ["git", "-C", str(SOLVER_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(SOLVER_ROOT),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != SOLVER_COMMIT:
        raise RuntimeError(f"solver HEAD {head} does not equal pinned {SOLVER_COMMIT}")
    if dirty:
        raise RuntimeError("solver tracked files are not clean")
    compiler_output = subprocess.run(
        ["/usr/bin/gfortran", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    mpi_output = subprocess.run(
        ["/usr/bin/mpirun", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    toolchain = validate_toolchain_identity(
        compiler_output,
        mpi_output,
        expected_compiler=expected_solver["compiler"],
        expected_mpi=expected_solver["mpi"],
    )
    binaries = []
    for name, expected in BINARY_HASHES.items():
        path = SOLVER_ROOT / "bin" / name
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"binary hash mismatch for {path}")
        binaries.append({"name": name, "path": str(path), "sha256": observed})
    return {
        "classification": "Observed pinned solver provenance",
        "status": "PINNED_SOLVER_VALIDATED",
        "solver_root": str(SOLVER_ROOT),
        "git_head": head,
        "tracked_files_clean": True,
        **toolchain,
        "binaries": binaries,
    }


def validate_toolchain_identity(
    compiler_output: str,
    mpi_output: str,
    *,
    expected_compiler: str,
    expected_mpi: str,
) -> dict[str, str]:
    """Normalize and compare the pinned compiler and MPI version outputs."""
    compiler_match = re.search(
        r"GNU Fortran[^\n]*?\s(\d+\.\d+\.\d+)\s*$", compiler_output, re.MULTILINE
    )
    mpi_match = re.search(r"mpirun \(Open MPI\)\s+(\d+\.\d+\.\d+)", mpi_output)
    if compiler_match is None:
        raise RuntimeError("compiler identity could not be parsed")
    if mpi_match is None:
        raise RuntimeError("MPI identity could not be parsed")
    compiler = f"GNU Fortran {compiler_match.group(1)}"
    mpi = f"Open MPI {mpi_match.group(1)}"
    if compiler != expected_compiler:
        raise RuntimeError(
            f"compiler identity {compiler!r} does not match {expected_compiler!r}"
        )
    if mpi != expected_mpi:
        raise RuntimeError(f"MPI identity {mpi!r} does not match {expected_mpi!r}")
    return {"compiler": compiler, "mpi": mpi}


def validate_solver_log_text(text: str, *, length_m: int) -> dict[str, Any]:
    """Apply every textual acceptance condition that belongs to solver output."""
    facts = DOMAIN_FACTS[length_m]
    required = (
        "using single precision for the calculations",
        "RAYLEIGH_DIAG",
        "GRAGRA_RAYLEIGH_PILOT t0 initialized and B1 acceleration computed",
        "using Stacey first-order (P1) condition",
        "End of the simulation",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise ValueError(f"required log string(s) missing: {missing}")
    if re.search(r"scheme:\s+Newmark", text) is None:
        raise ValueError("solver log does not identify the Newmark scheme")

    def _integer(pattern: str, label: str) -> int:
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"solver log lacks {label}")
        return int(match.group(1))

    observed = {
        "nproc": _integer(r"There is a total of\s+(\d+)\s+slices", "slice count"),
        "global_elements": _integer(r"NSPEC_global_sum\s*=\s*(\d+)", "global elements"),
        "local_min": _integer(r"NSPEC_global_min\s*=\s*(\d+)", "local element minimum"),
        "local_max": _integer(r"NSPEC_global_max\s*=\s*(\d+)", "local element maximum"),
        "cadence": _integer(
            r"number of steps between frames\s*=\s*(\d+)", "snapshot cadence"
        ),
        "nstep": _integer(r"number of time steps:\s*(\d+)", "NSTEP"),
    }
    expected = {
        "nproc": NPROC,
        "global_elements": facts["elements"],
        "local_min": facts["local"],
        "local_max": facts["local"],
        "cadence": 1,
        "nstep": NSTEP,
    }
    if observed != expected:
        raise ValueError(
            f"solver log dimensions/numerics mismatch: {observed} != {expected}"
        )
    dt_match = re.search(r"time step:\s*([+\-0-9.Ee]+)\s+s", text)
    if dt_match is None:
        raise ValueError("solver log lacks actual time step")
    actual_dt = float(dt_match.group(1))
    if not math.isfinite(actual_dt) or actual_dt <= 0.0:
        raise ValueError("solver log actual time step is non-finite or non-positive")
    steps = [int(value) for value in re.findall(r"Time step #\s*(\d+)", text)]
    if steps != list(range(1, NSTEP + 1)):
        raise ValueError(
            "solver log step coverage is incomplete or unordered: "
            f"found {len(steps)} steps"
        )
    max_values = re.findall(
        r"Max norm displacement vector U in all slices \(m\)\s*=\s*(\S+)", text
    )
    try:
        finite = max_values and all(math.isfinite(float(value)) for value in max_values)
    except ValueError:
        finite = False
    if not finite:
        raise ValueError("solver log contains a non-finite maximum displacement")
    return {
        "classification": "Observed solver-log acceptance",
        "status": "SOLVER_LOG_ACCEPTED",
        "length_m": length_m,
        "actual_dt_s": actual_dt,
        "observed_steps": [steps[0], steps[-1]],
        "step_count": len(steps),
        "effective_source_count": 0,
        **observed,
    }


def inventory_raw_outputs(directory: Path) -> dict[str, Any]:
    """Inventory all 8 fields without reading their potentially large payloads."""
    directory = Path(directory)
    expected: set[str] = set()
    for rank in range(NPROC):
        for step in range(1, NSTEP + 1):
            expected.update(
                f"proc{rank:06d}_{field}_it{step:06d}.bin"
                for field in ELEMENT_LOCAL_FIELDS
            )
            expected.add(f"proc{rank:06d}_div_glob_it{step:06d}.bin")
    observed = {
        path.name: path for path in directory.glob("proc*_it*.bin") if path.is_file()
    }
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    if missing or extra:
        raise ValueError(
            f"missing raw outputs={missing[:5]} ({len(missing)} total); "
            f"unexpected raw outputs={extra[:5]} ({len(extra)} total)"
        )
    files = [observed[name] for name in sorted(expected)]
    return {
        "classification": "Observed stat-only raw inventory",
        "status": "RAW_ALL_RETAINED_COMPLETE",
        "directory": str(directory.resolve()),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "displacement_file_count": sum(
            any(f"_{field}_" in path.name for field in DISPLACEMENT_FIELDS)
            for path in files
        ),
        "retention": "all raw files retained; no deletion performed",
        "hash_policy": "stat-only for large raw fields",
    }


def inventory_static_database(directory: Path) -> dict[str, Any]:
    """Record a stat-only recursive inventory without reading static payloads."""
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"static database must be a real directory: {directory}")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"static database contains no files: {directory}")
    return {
        "classification": "Observed stat-only static database inventory",
        "status": "STATIC_DATABASE_INVENTORIED",
        "directory": str(directory.resolve()),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "inventory_method": "stat-only",
        "retention": "all static database files retained",
    }


def validate_matched_actual_dt(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    """Require the two accepted solver logs to report exactly one actual DT."""
    lengths = [int(first["length_m"]), int(second["length_m"])]
    if sorted(lengths) != [12_000, 18_000]:
        raise ValueError(f"matched pair must be L12000/L18000, got {lengths}")
    actual_first = float(first["actual_dt_s"])
    actual_second = float(second["actual_dt_s"])
    if not math.isfinite(actual_first) or not math.isfinite(actual_second):
        raise ValueError("actual_DT must be finite for both cases")
    if actual_first != actual_second:
        raise ValueError(
            f"actual_DT mismatch: L{lengths[0]}={actual_first!r}, "
            f"L{lengths[1]}={actual_second!r}"
        )
    return {
        "classification": "Observed matched solver time step",
        "status": "MATCHED_ACTUAL_DT",
        "lengths_m": lengths,
        "actual_dt_s": actual_first,
        "physical_time_rule": "10 + native_it * actual_DT",
        "native_step_origin": 1,
    }


def materialize_mesher_database(case_root: Path) -> dict[str, Any]:
    """Copy mesher output into Par_file's distinct real LOCAL_PATH directory."""
    case_root = Path(case_root)
    source = case_root / "OUTPUT_FILES/DATABASES_MPI"
    target = case_root / "OUTPUT_FILES/DATABASES_MPI_RUN"
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f"mesher database source must be a real directory: {source}")
    if target.is_symlink():
        raise ValueError(f"solver database target must not be a symlink: {target}")
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        raise FileExistsError(f"solver database target must be empty: {target}")
    source_files = sorted(path for path in source.iterdir() if path.is_file())
    if not source_files:
        raise ValueError(f"mesher database source contains no files: {source}")
    files = []
    for source_path in source_files:
        target_path = target / source_path.name
        shutil.copy2(source_path, target_path)
        source_hash = sha256_file(source_path)
        target_hash = sha256_file(target_path)
        if target_hash != source_hash:
            raise RuntimeError(
                f"static database copy hash mismatch for {source_path.name}"
            )
        files.append(
            {
                "name": source_path.name,
                "size_bytes": source_path.stat().st_size,
                "sha256": source_hash,
            }
        )
    report = {
        "classification": "Observed static mesher database materialization",
        "status": "MESHER_DATABASE_MATERIALIZED",
        "source": str(source.resolve()),
        "target": str(target.resolve()),
        "target_is_symlink": False,
        "file_count": len(files),
        "total_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
        "retention": "both mesher and solver database copies retained",
    }
    _write_json(
        case_root / "mesher-database-materialization.json", report, readonly=True
    )
    return report


def _load_run_manifest(case_root: Path) -> dict[str, Any]:
    path = case_root / "run-manifest.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {
        "schema_version": 1,
        "classification": "Observed staged external execution",
        "case_root": str(case_root.resolve()),
        "stages": {},
        "retention": "raw/static/canonical retained; no deletion authorized",
    }


def execute_stage(
    case_root: Path,
    length_m: int,
    stage: str,
    *,
    l_acceptance_path: Path = DEFAULT_L_ACCEPTANCE,
    gate_interval_s: float = 1.0,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Execute exactly one pinned MPI stage and persist success or failure."""
    if stage not in STAGE_ARGV:
        raise ValueError(f"stage must be one of {tuple(STAGE_ARGV)}, got {stage!r}")
    case_root = Path(case_root)
    validate_case_inputs(case_root, length_m)
    validate_pinned_solver()
    gate = capture_launch_gate(
        case_root,
        l_acceptance_path=l_acceptance_path,
        interval_s=gate_interval_s,
    )
    if not gate["launch_allowed"]:
        raise RuntimeError(f"launch gate is {gate['status']}")
    manifest = _load_run_manifest(case_root)
    prior = {"mesh": (), "database": ("mesh",), "solver": ("mesh", "database")}
    for needed in prior[stage]:
        if manifest["stages"].get(needed, {}).get("exit_code") != 0:
            raise RuntimeError(
                f"stage {stage} requires successful prior stage {needed}"
            )
    dynamic_outputs = sorted((case_root / "OUTPUT_FILES").glob("**/*_it*.bin"))
    if dynamic_outputs:
        raise RuntimeError("dynamic *_it*.bin output already exists; refusing new run")
    if stage in manifest["stages"]:
        raise RuntimeError(f"stage {stage} already has a recorded attempt")

    logs = case_root / "protocol-logs"
    logs.mkdir(exist_ok=True)
    stdout_path = logs / f"{stage}.stdout.log"
    stderr_path = logs / f"{stage}.stderr.log"
    time_path = logs / f"{stage}.time-v.txt"
    argv = list(STAGE_ARGV[stage])
    wrapped = ["/usr/bin/time", "-v", "-o", str(time_path), *argv]
    env = dict(os.environ)
    env.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        }
    )
    started = _utc_now()
    started_monotonic = time.monotonic()
    record = {
        "stage": stage,
        "status": "STARTED_INCOMPLETE",
        "fixed_argv": argv,
        "timed_argv": wrapped,
        "cwd": str(case_root.resolve()),
        "started_utc": started,
        "finished_utc": None,
        "wall_seconds": None,
        "exit_code": None,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "time_v_path": str(time_path),
        "launch_gate": gate,
    }
    manifest["stages"][stage] = record
    manifest["status"] = "STARTED_INCOMPLETE"
    _write_json(case_root / "run-manifest.json", manifest)
    post_stage: dict[str, Any] | None = None
    try:
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            completed = runner(
                wrapped,
                cwd=case_root,
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
            )
        if completed.returncode == 0 and stage == "mesh":
            post_stage = materialize_mesher_database(case_root)
        elif completed.returncode == 0 and stage == "database":
            post_stage = inventory_static_database(
                case_root / "OUTPUT_FILES/DATABASES_MPI_RUN"
            )
            _write_json(
                case_root / "static-database-inventory-after-database.json",
                post_stage,
                readonly=True,
            )
    except BaseException as error:
        interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
        record.update(
            {
                "status": (
                    "INTERRUPTED_RETAINED" if interrupted else "FAILED_RETAINED"
                ),
                "finished_utc": _utc_now(),
                "wall_seconds": time.monotonic() - started_monotonic,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        manifest["status"] = record["status"]
        _write_json(case_root / "run-manifest.json", manifest)
        raise
    record.update(
        {
            "status": ("SUCCEEDED" if completed.returncode == 0 else "FAILED_RETAINED"),
            "finished_utc": _utc_now(),
            "wall_seconds": time.monotonic() - started_monotonic,
            "exit_code": completed.returncode,
            "post_stage": post_stage,
        }
    )
    manifest["status"] = record["status"]
    _write_json(case_root / "run-manifest.json", manifest)
    if completed.returncode != 0:
        raise RuntimeError(f"{stage} failed with exit code {completed.returncode}")
    return record


def solver_output_directory(case_root: Path) -> Path:
    """Return the pinned Par_file LOCAL_PATH, never the mesher work directory."""
    directory = Path(case_root) / "OUTPUT_FILES/DATABASES_MPI_RUN"
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"solver output must be a real directory: {directory}")
    return directory


def accept_solver_run(case_root: Path, length_m: int) -> dict[str, Any]:
    """Validate successful commands, solver log, raw completeness, and retention."""
    case_root = Path(case_root)
    manifest = _load_run_manifest(case_root)
    for stage in STAGE_ARGV:
        if manifest["stages"].get(stage, {}).get("exit_code") != 0:
            raise ValueError(f"run manifest lacks successful {stage} stage")
    log_path = case_root / "OUTPUT_FILES/output_solver.txt"
    log = validate_solver_log_text(log_path.read_text(), length_m=length_m)
    local_path = solver_output_directory(case_root)
    raw = inventory_raw_outputs(local_path)
    report = {
        "classification": "Observed external solver acceptance",
        "status": "SOLVER_RUN_ACCEPTED_FOR_EXPORT",
        "length_m": length_m,
        "case_root": str(case_root.resolve()),
        "inputs": validate_case_inputs(case_root, length_m),
        "solver": validate_pinned_solver(),
        "log": log,
        "raw_inventory": raw,
        "limitations": [
            "execution acceptance is not a stability proof",
            "execution acceptance is not a domain-convergence claim",
            "no one-percent gate was applied",
        ],
    }
    _write_or_verify_readonly_json(case_root / "solver-acceptance.json", report)
    _write_or_verify_readonly_json(case_root / "raw-inventory.json", raw)
    return report


def prepare_all(runs_root: Path, template_case: Path = TEMPLATE_CASE) -> dict[str, Any]:
    reports = []
    for length_m, name in CASE_NAMES.items():
        reports.append(
            generate_case_inputs(template_case, Path(runs_root) / name, length_m)
        )
    return {
        "classification": "Observed input preparation",
        "status": "BOTH_CASES_INPUTS_VALIDATED",
        "runs_root": str(Path(runs_root).resolve()),
        "cases": reports,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    prepare.add_argument("--template-case", type=Path, default=TEMPLATE_CASE)
    dry = sub.add_parser("dry-run")
    dry.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    dry.add_argument("--l-acceptance", type=Path, default=DEFAULT_L_ACCEPTANCE)
    dry.add_argument("--interval-s", type=float, default=1.0)
    run = sub.add_parser("run-stage")
    run.add_argument("length_m", type=int, choices=sorted(CASE_NAMES))
    run.add_argument("stage", choices=tuple(STAGE_ARGV))
    run.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    run.add_argument("--l-acceptance", type=Path, default=DEFAULT_L_ACCEPTANCE)
    accept = sub.add_parser("accept")
    accept.add_argument("length_m", type=int, choices=sorted(CASE_NAMES))
    accept.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        report = prepare_all(args.runs_root, args.template_case)
        _write_json(DEFAULT_EXTERNAL_LANE / "input-preparation.json", report)
    elif args.command == "dry-run":
        report = {
            "classification": "Observed no-launch dry run",
            "status": "DRY_RUN",
            "launch_gate": capture_launch_gate(
                args.runs_root,
                l_acceptance_path=args.l_acceptance,
                interval_s=args.interval_s,
            ),
            "solver": validate_pinned_solver(),
            "cases": [
                validate_case_inputs(args.runs_root / name, length_m)
                for length_m, name in CASE_NAMES.items()
            ],
            "fixed_stage_argv": {key: list(value) for key, value in STAGE_ARGV.items()},
            "action": "no command executed",
        }
        _write_json(DEFAULT_EXTERNAL_LANE / "dry-run.json", report)
    elif args.command == "run-stage":
        report = execute_stage(
            args.runs_root / CASE_NAMES[args.length_m],
            args.length_m,
            args.stage,
            l_acceptance_path=args.l_acceptance,
        )
    else:
        report = accept_solver_run(
            args.runs_root / CASE_NAMES[args.length_m], args.length_m
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
