#!/usr/bin/env python3
"""Prepare and compare the bounded Rayleigh movie-cadence pilot runs.

This research helper copies the mesher header and four per-rank external mesh
databases.  It deliberately does not traverse or hash the mixed snapshot
directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# Do not let an editable-install redirector select another worktree.
_SCIENCE_ROOT = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if type(finder).__name__ != "ScikitBuildRedirectingFinder"
]
sys.path.insert(0, str(_SCIENCE_ROOT / "src"))

from gragra.adapters.specfem.specfem_bin import read_records  # noqa: E402

RANKS = 4
NSTEP = 8
DT_S = 0.005
T_INIT_S = 10.0
COMPARE_STEPS = (2, 4, 6, 8)
_SNAPSHOT_RE = re.compile(
    r"^proc(?P<rank>\d{6})_displ_(?P<component>[XYZ])_it(?P<step>\d{6})\.bin$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_assignment(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*).*$", re.MULTILINE)
    updated, count = pattern.subn(rf"\g<1>{value}", text)
    if count != 1:
        raise ValueError(f"Par_file must contain exactly one {key} assignment")
    return updated


def _copy_static(source: Path, destination: Path) -> None:
    subprocess.run(
        [
            "cp",
            "--reflink=auto",
            "--dereference",
            "--preserve=mode,timestamps",
            str(source),
            str(destination),
        ],
        check=True,
    )


def _fixed_conditions(par_file: str) -> dict[str, object]:
    required = {
        "NPROC": "4",
        "NSTEP": "8",
        "DT": "0.005",
        "STACEY_ABSORBING_CONDITIONS": ".true.",
        "STACEY_INSTEAD_OF_FREE_SURFACE": ".false.",
        "BOTTOM_FREE_SURFACE": ".false.",
        "MOVIE_VOLUME": ".true.",
        "SAVE_DISPLACEMENT": ".true.",
        "LOCAL_PATH": "OUTPUT_FILES/DATABASES_MPI_RUN",
    }
    for key, expected in required.items():
        match = re.search(rf"^\s*{key}\s*=\s*(\S+)", par_file, re.MULTILINE)
        if match is None or match.group(1).lower() != expected.lower():
            found = None if match is None else match.group(1)
            raise ValueError(f"fixed Par_file condition {key}={expected}, got {found}")
    return {
        "nproc": 4,
        "nstep": NSTEP,
        "dt_s": DT_S,
        "stacey_absorbing_conditions": True,
        "stacey_instead_of_free_surface": False,
        "bottom_free_surface": False,
        "movie_volume": True,
        "save_displacement": True,
        "local_path": "OUTPUT_FILES/DATABASES_MPI_RUN",
    }


def prepare_cases(
    source_case: str | Path, output_root: str | Path, solver: str | Path
) -> dict[int, Path]:
    """Create independent cadence-1 and cadence-2 cases and input manifests."""
    source_case = Path(source_case).resolve()
    output_root = Path(output_root).resolve()
    solver = Path(solver).resolve()
    if not solver.is_file() or not os.access(solver, os.X_OK):
        raise ValueError(f"solver is not an executable file: {solver}")
    source_data = source_case / "DATA"
    source_static = source_case / "OUTPUT_FILES" / "DATABASES_MPI_RUN"
    if not source_data.is_dir() or not source_static.is_dir():
        raise ValueError(
            f"source case lacks DATA or static database directory: {source_case}"
        )

    selected = [
        (
            source_static / f"proc{rank:06d}_external_mesh.bin",
            Path("OUTPUT_FILES")
            / "DATABASES_MPI_RUN"
            / f"proc{rank:06d}_external_mesh.bin",
        )
        for rank in range(RANKS)
    ]
    selected.append(
        (
            source_case / "OUTPUT_FILES" / "values_from_mesher.h",
            Path("OUTPUT_FILES") / "values_from_mesher.h",
        )
    )
    missing = [str(source) for source, _ in selected if not source.is_file()]
    if missing:
        raise ValueError(f"missing selected static inputs: {missing}")
    source_hashes = {source: _sha256(source) for source, _ in selected}
    prepared: dict[int, Path] = {}
    output_root.mkdir(parents=True, exist_ok=True)

    for cadence in (1, 2):
        case = output_root / f"cadence-m{cadence}"
        if case.exists():
            raise FileExistsError(f"refusing to overwrite prepared case: {case}")
        destination_static = case / "OUTPUT_FILES" / "DATABASES_MPI_RUN"
        destination_static.mkdir(parents=True)
        shutil.copytree(source_data, case / "DATA", copy_function=shutil.copy2)
        par_path = case / "DATA" / "Par_file"
        par_text = par_path.read_text()
        par_text = _replace_assignment(par_text, "NSTEP", str(NSTEP))
        par_text = _replace_assignment(par_text, "NTSTEP_BETWEEN_FRAMES", str(cadence))
        fixed = _fixed_conditions(par_text)
        par_path.write_text(par_text)

        static_rows = []
        for source, relative_destination in selected:
            destination = case / relative_destination
            destination.parent.mkdir(parents=True, exist_ok=True)
            _copy_static(source, destination)
            if destination.is_symlink():
                raise RuntimeError(
                    f"static input must not remain a symlink: {destination}"
                )
            static_rows.append(
                {
                    "source": str(source),
                    "destination": str(destination),
                    "size_bytes": source.stat().st_size,
                    "source_sha256_before": source_hashes[source],
                    "destination_sha256_before": _sha256(destination),
                }
            )
        changed_sources = [
            str(source)
            for source, _ in selected
            if _sha256(source) != source_hashes[source]
        ]
        if changed_sources:
            raise RuntimeError(
                f"source static inputs changed while copying: {changed_sources}"
            )

        launcher = case / "run_rayleigh_hook.sh"
        launcher.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"cd {case!s}\n"
            "exec mpirun --allow-run-as-root -np 4 "
            f"env GRAGRA_RAYLEIGH_PILOT=1 {solver!s}\n"
        )
        launcher.chmod(0o755)
        manifest = {
            "classification": "Prepared input",
            "source_case": str(source_case),
            "solver": str(solver),
            "solver_sha256": _sha256(solver),
            "cadence": cadence,
            "fixed_conditions": fixed,
            "static_input_selection": (
                "values_from_mesher.h plus four proc*_external_mesh.bin files; "
                "mixed time-series files excluded"
            ),
            "static_inputs": static_rows,
        }
        (case / "input-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        prepared[cadence] = case
    return prepared


def verify_inputs(case_path: str | Path) -> dict[str, object]:
    """Re-hash the selected source and destination files after a run."""
    case_path = Path(case_path)
    manifest = json.loads((case_path / "input-manifest.json").read_text())
    rows = []
    for original in manifest["static_inputs"]:
        source = Path(original["source"])
        destination = Path(original["destination"])
        source_after = _sha256(source)
        destination_after = _sha256(destination)
        rows.append(
            {
                **original,
                "source_sha256_after": source_after,
                "destination_sha256_after": destination_after,
                "source_unchanged": source_after == original["source_sha256_before"],
                "destination_unchanged": destination_after
                == original["destination_sha256_before"],
                "source_destination_equal": source_after == destination_after,
            }
        )
    return {
        "classification": "Observed input-integrity check",
        "case": str(case_path.resolve()),
        "all_sources_unchanged": all(row["source_unchanged"] for row in rows),
        "all_destinations_unchanged": all(row["destination_unchanged"] for row in rows),
        "all_source_destination_pairs_equal": all(
            row["source_destination_equal"] for row in rows
        ),
        "static_inputs": rows,
    }


def _logical_files(
    directory: Path,
) -> tuple[dict[tuple[int, int, str], Path], list[str]]:
    by_key: dict[tuple[int, int, str], Path] = {}
    duplicates: list[str] = []
    for path in sorted(directory.glob("proc*_displ_*_it*.bin")):
        match = _SNAPSHOT_RE.match(path.name)
        if match is None:
            continue
        key = (
            int(match.group("rank")),
            int(match.group("step")),
            match.group("component"),
        )
        if key in by_key:
            duplicates.append(f"rank={key[0]},step={key[1]},component={key[2]}")
        else:
            by_key[key] = path
    return by_key, duplicates


def _read_component(path: Path, expected_count: int) -> np.ndarray:
    layout, records = read_records(path)
    if len(records) != 1:
        raise ValueError(f"{path}: expected one Fortran record, found {len(records)}")
    payload = records[0]
    if len(payload) == expected_count * 4:
        dtype = np.dtype(f"{layout.endianness}f4")
    elif len(payload) == expected_count * 8:
        dtype = np.dtype(f"{layout.endianness}f8")
    else:
        raise ValueError(
            f"{path}: payload has {len(payload)} bytes; expected "
            f"{expected_count * 4} or {expected_count * 8}"
        )
    return np.frombuffer(payload, dtype=dtype, count=expected_count)


def compare_outputs(
    m1_directory: str | Path,
    m2_directory: str | Path,
    *,
    ranks: int = RANKS,
    nspec_rank: int = 3456,
    ngll_points: int = 125,
) -> dict[str, object]:
    """Compare common native steps by rank/local-element/GLL/component."""
    m1_directory = Path(m1_directory)
    m2_directory = Path(m2_directory)
    files1, duplicates1 = _logical_files(m1_directory)
    files2, duplicates2 = _logical_files(m2_directory)
    keys = [
        (rank, step, component)
        for rank in range(ranks)
        for step in COMPARE_STEPS
        for component in "XYZ"
    ]
    missing1 = [key for key in keys if key not in files1]
    missing2 = [key for key in keys if key not in files2]
    expected_per_component = nspec_rank * ngll_points
    actual1 = actual2 = exact = nonzero1 = nonzero2 = 0
    nonfinite1 = nonfinite2 = 0
    maximum = 0.0
    for key in keys:
        if key not in files1 or key not in files2:
            continue
        values1 = _read_component(files1[key], expected_per_component)
        values2 = _read_component(files2[key], expected_per_component)
        actual1 += values1.size
        actual2 += values2.size
        exact += int(np.count_nonzero(values1 == values2))
        nonzero1 += int(np.count_nonzero(values1))
        nonzero2 += int(np.count_nonzero(values2))
        nonfinite1 += int(np.count_nonzero(~np.isfinite(values1)))
        nonfinite2 += int(np.count_nonzero(~np.isfinite(values2)))
        if values1.size:
            maximum = max(maximum, float(np.max(np.abs(values1 - values2))))

    def encode(keys_to_encode: list[tuple[int, int, str]]) -> list[str]:
        return [
            f"rank={rank},step={step},component={component}"
            for rank, step, component in keys_to_encode
        ]

    return {
        "classification": "Observed native-binary cadence comparison",
        "logical_layout": "rank/local-element/GLL/component",
        "ranks": ranks,
        "nspec_rank": nspec_rank,
        "ngll_points_per_element": ngll_points,
        "compared_native_steps": list(COMPARE_STEPS),
        "expected_value_count": ranks
        * len(COMPARE_STEPS)
        * nspec_rank
        * ngll_points
        * 3,
        "actual_value_count_m1": actual1,
        "actual_value_count_m2": actual2,
        "exact_match_count": exact,
        "maximum_absolute_difference": maximum,
        "missing_logical_keys_m1": encode(missing1),
        "missing_logical_keys_m2": encode(missing2),
        "duplicate_logical_keys_m1": duplicates1,
        "duplicate_logical_keys_m2": duplicates2,
        "nonzero_value_count_m1": nonzero1,
        "nonzero_value_count_m2": nonzero2,
        "nonfinite_value_count_m1": nonfinite1,
        "nonfinite_value_count_m2": nonfinite2,
        "time_mapping": [
            {
                "native_step": step,
                "array_index": step - 1,
                "physical_time_s": T_INIT_S + step * DT_S,
            }
            for step in COMPARE_STEPS
        ],
        "interpretation": (
            "This compares common saved native steps only; it is not a "
            "solver finite-domain stability or convergence test."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("source_case", type=Path)
    prepare.add_argument("output_root", type=Path)
    prepare.add_argument("solver", type=Path)
    verify = subparsers.add_parser("verify-inputs")
    verify.add_argument("case", type=Path)
    compare = subparsers.add_parser("compare")
    compare.add_argument("m1_directory", type=Path)
    compare.add_argument("m2_directory", type=Path)
    compare.add_argument("--output-json", type=Path, required=True)
    compare.add_argument("--ranks", type=int, default=RANKS)
    compare.add_argument("--nspec-rank", type=int, default=3456)
    compare.add_argument("--ngll-points", type=int, default=125)
    args = parser.parse_args()

    if args.command == "prepare":
        result = {
            str(key): str(value)
            for key, value in prepare_cases(
                args.source_case, args.output_root, args.solver
            ).items()
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "verify-inputs":
        print(json.dumps(verify_inputs(args.case), indent=2, sort_keys=True))
    else:
        result = compare_outputs(
            args.m1_directory,
            args.m2_directory,
            ranks=args.ranks,
            nspec_rank=args.nspec_rank,
            ngll_points=args.ngll_points,
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
