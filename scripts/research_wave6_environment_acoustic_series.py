"""Exercise accepted G/L density-series streaming on the acoustic cases."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import resource
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from gragra.constants import G_SI
from scripts.research_wave6_environment_acoustic import (
    FREQUENCY_HZ,
    TARGETS_M,
    _complex_json,
    _difference,
    _load_reference,
    _sha256,
    acoustic_case_matrix,
    acoustic_density_phasor,
    acoustic_real_density_slab,
    case_representation,
    fit_phasor,
    midpoint_box_sources,
)


def _acceptance_hashes(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "ACCEPT" not in str(data.get("status", "")).upper():
        raise ValueError(f"dependency acceptance is not accepted: {path}")
    if isinstance(data.get("sha256"), dict):
        return {str(key): str(value) for key, value in data["sha256"].items()}
    files = data.get("files")
    if isinstance(files, list):
        return {str(item["path"]): str(item["sha256"]) for item in files}
    raise ValueError(f"dependency acceptance has no file hash mapping: {path}")


def _extend_package_path(package_name: str, directory: Path) -> None:
    package = importlib.import_module(package_name)
    resolved = str(directory.resolve())
    if resolved not in package.__path__:
        package.__path__.append(resolved)


def load_accepted_series_contract(
    *,
    g_root: Path,
    l_root: Path,
    g_acceptance: Path,
    l_acceptance: Path,
) -> dict[str, object]:
    """Load accepted G/L modules without copying them into Lane E."""
    g_hashes = _acceptance_hashes(g_acceptance)
    l_hashes = _acceptance_hashes(l_acceptance)
    required = {
        "G": (
            g_root,
            g_hashes,
            (
                "src/gragra/fields/_series.py",
                "src/gragra/fields/density_series.py",
            ),
        ),
        "L": (
            l_root,
            l_hashes,
            (
                "src/gragra/integrators/streaming.py",
                "src/gragra/observables/streaming.py",
                "src/gragra/adapters/field_series_hdf5.py",
            ),
        ),
    }
    files: dict[str, dict[str, str]] = {}
    for lane, (root, hashes, relative_paths) in required.items():
        for relative in relative_paths:
            path = root / relative
            observed = _sha256(path)
            expected = hashes.get(relative)
            if observed != expected:
                raise ValueError(
                    f"{lane} accepted source hash mismatch for {relative}: "
                    f"{observed} != {expected}"
                )
            files[f"{lane}:{relative}"] = {
                "path": str(path.resolve()),
                "sha256": observed,
            }
    _extend_package_path("gragra.fields", g_root / "src/gragra/fields")
    _extend_package_path("gragra.integrators", l_root / "src/gragra/integrators")
    _extend_package_path("gragra.observables", l_root / "src/gragra/observables")
    _extend_package_path("gragra.adapters", l_root / "src/gragra/adapters")
    density_module = importlib.import_module("gragra.fields.density_series")
    field_base_module = importlib.import_module("gragra.fields._series")
    integrator_module = importlib.import_module("gragra.integrators.streaming")
    observable_module = importlib.import_module("gragra.observables.streaming")
    hdf_module = importlib.import_module("gragra.adapters.field_series_hdf5")
    imported_modules = {
        "gragra.fields._series": (
            field_base_module,
            "G",
            "src/gragra/fields/_series.py",
        ),
        "gragra.fields.density_series": (
            density_module,
            "G",
            "src/gragra/fields/density_series.py",
        ),
        "gragra.integrators.streaming": (
            integrator_module,
            "L",
            "src/gragra/integrators/streaming.py",
        ),
        "gragra.observables.streaming": (
            observable_module,
            "L",
            "src/gragra/observables/streaming.py",
        ),
        "gragra.adapters.field_series_hdf5": (
            hdf_module,
            "L",
            "src/gragra/adapters/field_series_hdf5.py",
        ),
    }
    actual_imports = {}
    for name, (module, lane, relative) in imported_modules.items():
        actual_path = Path(module.__file__).resolve()
        observed = _sha256(actual_path)
        expected = required[lane][1][relative]
        if observed != expected:
            raise ImportError(
                f"{name} resolved to unaccepted source {actual_path}: "
                f"{observed} != {expected}"
            )
        actual_imports[name] = {
            "path": str(actual_path),
            "sha256": observed,
            "accepted_relative_path": relative,
        }
    return {
        "DensityPhasors": density_module.DensityPhasors,
        "SeriesMetadata": integrator_module.SeriesMetadata,
        "ScalarSourceSeriesBlock": integrator_module.ScalarSourceSeriesBlock,
        "as_series_provider": integrator_module.as_series_provider,
        "stream_observable": observable_module.stream_observable,
        "collect_observable_blocks": observable_module.collect_observable_blocks,
        "write_fixed_mesh_series_hdf5": hdf_module.write_fixed_mesh_series_hdf5,
        "read_fixed_mesh_series_hdf5": hdf_module.read_fixed_mesh_series_hdf5,
        "files": files,
        "actual_imports": actual_imports,
    }


@dataclass
class _LazyDensityValues:
    density_phasor: np.ndarray
    times_s: np.ndarray

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("density slabs require sample and source slices")
        return acoustic_real_density_slab(
            self.density_phasor,
            self.times_s,
            sample_slice=key[0],
            source_slice=key[1],
        )


@dataclass
class _LazyDensityTimeSeries:
    positions_m: np.ndarray
    volumes_m3: np.ndarray
    time_s: np.ndarray
    density_phasor: np.ndarray
    source_ids: tuple[str, ...]
    coordinate_frame: str = "acoustic-finite-air-box"
    density_kind: str = "delta_rho"

    def __post_init__(self) -> None:
        self.axis_kind = "time"
        self.axis_values = self.time_s
        self.sample_count = len(self.time_s)
        self.source_count = len(self.positions_m)
        self.coordinate_origin_m = np.zeros(3, dtype=np.float64)
        self.density_kg_m3 = _LazyDensityValues(self.density_phasor, self.time_s)


def _source_ids(count: int) -> tuple[str, ...]:
    return tuple(f"cell-{index}" for index in range(count))


def _time_provider(contract, positions, volumes, phasor, times, source_ids):
    metadata = contract["SeriesMetadata"](
        positions_m=positions,
        source_ids=source_ids,
        coordinate_frame="acoustic-finite-air-box",
        coordinate_origin_m=np.zeros(3),
        axis_kind="time",
        axis_values=times,
        sample_count=len(times),
        source_count=len(positions),
        quantity="density",
        density_kind="delta_rho",
    )

    def factory(samples: slice, source_block_size: int):
        for start in range(0, len(positions), source_block_size):
            stop = min(start + source_block_size, len(positions))
            density = acoustic_real_density_slab(
                phasor,
                times,
                sample_slice=samples,
                source_slice=slice(start, stop),
            )
            yield contract["ScalarSourceSeriesBlock"](
                positions[start:stop],
                samples,
                start,
                density * volumes[None, start:stop],
                source_ids[start:stop],
            )

    return contract["as_series_provider"](metadata, factory)


def _collect(contract, provider, observable: str) -> np.ndarray:
    blocks = contract["stream_observable"](
        provider,
        TARGETS_M,
        observable=observable,
        sample_block_size=8,
        source_block_size=4096,
        target_block_size=3,
        kernel_chunk_size=512,
        backend="numpy",
        memory_budget_bytes=512 * 1024**2,
    )
    return contract["collect_observable_blocks"](
        blocks,
        sample_count=provider.sample_count,
        target_count=len(TARGETS_M),
        axis_kind=provider.axis_kind,
    )


def _direct_phasor(reference, positions, volumes, density):
    value = reference.point_mass_observables_g_free(
        positions, density * volumes, TARGETS_M
    )
    return {
        name: np.asarray(G_SI * getattr(value, name), dtype=np.complex128)
        for name in ("potential", "acceleration", "gradient")
    }


def _integrate_provider(contract, provider) -> dict[str, np.ndarray]:
    return {
        name: _collect(contract, provider, name)
        for name in ("potential", "acceleration", "gradient")
    }


def run_series_study(
    *,
    output: Path,
    worktree: Path,
    reference_module: Path,
    reference_sha256: str,
    oracle_dir: Path,
    g_root: Path,
    l_root: Path,
    g_acceptance: Path,
    l_acceptance: Path,
) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source_path = Path(__import__("gragra").__file__).resolve()
    expected_source = worktree.resolve() / "src/gragra/__init__.py"
    if source_path != expected_source:
        raise RuntimeError(
            f"gragra imported from {source_path}, expected {expected_source}"
        )
    contract = load_accepted_series_contract(
        g_root=g_root.resolve(),
        l_root=l_root.resolve(),
        g_acceptance=g_acceptance.resolve(),
        l_acceptance=l_acceptance.resolve(),
    )
    reference = _load_reference(reference_module, reference_sha256)
    matrix = acoustic_case_matrix()
    representation = case_representation(matrix)
    output.mkdir(parents=True)
    started_at = datetime.now(UTC).isoformat()
    run_started = time.monotonic()
    times = np.arange(128, dtype=np.float64) / (64.0 * FREQUENCY_HZ)
    unique: dict[str, object] = {}
    hdf_record = None
    for case_index in representation["unique_case_indices"]:
        parameters = matrix[case_index]
        case_started = time.monotonic()
        positions, volumes = midpoint_box_sources(
            horizontal_half_width_wavelengths=float(
                parameters["horizontal_half_width_wavelengths"]
            ),
            height_wavelengths=float(parameters["height_wavelengths"]),
            cells_per_wavelength=int(parameters["cells_per_wavelength"]),
        )
        density = acoustic_density_phasor(positions)
        source_ids = _source_ids(len(positions))
        direct = _direct_phasor(reference, positions, volumes, density)
        phasors = contract["DensityPhasors"](
            positions,
            volumes,
            density[None, :],
            np.array([FREQUENCY_HZ]),
            0.0,
            np.zeros(3),
            "acoustic-finite-air-box",
            np.zeros(3),
            source_ids,
        )
        frequency_provider = contract["as_series_provider"](phasors)
        frequency_values = _integrate_provider(contract, frequency_provider)
        frequency_values = {
            name: value[:, 0] for name, value in frequency_values.items()
        }
        time_provider = _time_provider(
            contract, positions, volumes, density, times, source_ids
        )
        time_values = _integrate_provider(contract, time_provider)
        one_cycle = fit_phasor(time_values, times, sample_slice=slice(0, 64))
        two_cycles = fit_phasor(time_values, times, sample_slice=slice(0, 128))
        frequency_comparison = _difference(frequency_values, direct)
        one_cycle_comparison = _difference(one_cycle, direct)
        two_cycle_comparison = _difference(two_cycles, direct)
        worst_normalized_difference = max(
            max(comparison[name]["normalized"])
            for comparison in (
                frequency_comparison,
                one_cycle_comparison,
                two_cycle_comparison,
            )
            for name in ("potential", "acceleration", "gradient")
        )
        if worst_normalized_difference > 1e-11:
            raise AssertionError(
                "accepted provider acoustic result missed the 1e-11 target: "
                f"{worst_normalized_difference}"
            )
        if case_index == 1:
            lazy_series = _LazyDensityTimeSeries(
                positions, volumes, times, density, source_ids
            )
            hdf_path = output / "representative-base-density-time.h5"
            contract["write_fixed_mesh_series_hdf5"](
                hdf_path,
                lazy_series,
                sample_block_size=8,
                source_block_size=4096,
            )
            hdf_provider = contract["read_fixed_mesh_series_hdf5"](
                hdf_path,
                metadata_sample_block_size=8,
                metadata_source_block_size=4096,
            )
            hdf_values = _integrate_provider(contract, hdf_provider)
            hdf_equal = all(
                np.array_equal(time_values[name], hdf_values[name])
                for name in time_values
            )
            if not hdf_equal:
                raise AssertionError("HDF slab provider differs from factory provider")
            hdf_record = {
                "path": str(hdf_path),
                "sha256": _sha256(hdf_path),
                "bytes": hdf_path.stat().st_size,
                "representative_case_index": case_index,
                "factory_equal": hdf_equal,
            }
        oracle_case = oracle_dir / f"case-{case_index:02d}-{parameters['family']}.json"
        record = {
            "case_index": case_index,
            "parameters": parameters,
            "source_count": len(positions),
            "oracle_case": {
                "path": str(oracle_case.resolve()),
                "sha256": _sha256(oracle_case),
            },
            "density_path": (
                "bounded real slabs from Re[density_hat*exp(-2pi*i*f*t)]; "
                "no full time-by-source array"
            ),
            "direct_independent_phasor": {
                name: _complex_json(value) for name, value in direct.items()
            },
            "frequency_provider_vs_direct": frequency_comparison,
            "raw_frequency_provider": {
                name: _complex_json(value) for name, value in frequency_values.items()
            },
            "real_time_streaming": {
                "times_s": times.tolist(),
                "sample_count": len(times),
                "sample_block_size": 8,
                "source_block_size": 4096,
                "replay_validation": (
                    "stream_observable preflight and integration scans matched"
                ),
                "one_cycle_vs_direct": one_cycle_comparison,
                "two_cycles_vs_direct": two_cycle_comparison,
                "worst_normalized_difference": worst_normalized_difference,
                "target": 1e-11,
                "raw_gravity_series": {
                    name: {
                        "shape": list(value.shape),
                        "values": value.tolist(),
                    }
                    for name, value in time_values.items()
                },
                "one_cycle": {
                    name: _complex_json(value) for name, value in one_cycle.items()
                },
                "two_cycles": {
                    name: _complex_json(value) for name, value in two_cycles.items()
                },
            },
            "elapsed_s": time.monotonic() - case_started,
        }
        case_path = output / f"unique-case-{case_index:02d}.json"
        case_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        unique[str(case_index)] = {
            "path": str(case_path),
            "sha256": _sha256(case_path),
            "source_count": len(positions),
        }
    if hdf_record is None:
        raise AssertionError("representative HDF case was not executed")
    acoustic_script = Path(acoustic_density_phasor.__code__.co_filename).resolve()
    thermodynamic_function = acoustic_density_phasor.__globals__[
        "adiabatic_pressure_density_perturbation"
    ]
    thermodynamic_source = Path(thermodynamic_function.__code__.co_filename).resolve()
    manifest = {
        "status": "complete",
        "classification": (
            "observed accepted G/L provider, HDF5, and streaming execution"
        ),
        "started_at": started_at,
        "ended_at": datetime.now(UTC).isoformat(),
        "source": {
            "gragra": str(source_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "acoustic_script": {
                "path": str(acoustic_script),
                "sha256": _sha256(acoustic_script),
            },
            "thermodynamic": {
                "path": str(thermodynamic_source),
                "sha256": _sha256(thermodynamic_source),
            },
        },
        "execution": {
            "argv": list(sys.argv),
            "argv_shell": shlex.join(sys.argv),
            "elapsed_s": time.monotonic() - run_started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                )
            },
        },
        "dependencies": {
            "G_parent_acceptance": {
                "path": str(g_acceptance.resolve()),
                "sha256": _sha256(g_acceptance),
            },
            "L_parent_acceptance": {
                "path": str(l_acceptance.resolve()),
                "sha256": _sha256(l_acceptance),
            },
            "accepted_source_files": contract["files"],
            "actual_imported_modules": contract["actual_imports"],
            "A_reference_module": {
                "path": str(reference_module.resolve()),
                "sha256": _sha256(reference_module),
            },
            "oracle_directory": str(oracle_dir.resolve()),
        },
        "series": {
            "time_formula": "t_n=n/(64f), n=0..127",
            "density_formula": "Re[(p_hat/c_s^2) exp(+ikx) exp(-2pi*i*f*t)]",
            "all_nine_case_entries": representation,
            "unique_case_count": len(unique),
            "unique_cases": unique,
            "representative_hdf5": hdf_record,
        },
    }
    (output / "acoustic-series-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--reference-module", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--g-root", type=Path, required=True)
    parser.add_argument("--l-root", type=Path, required=True)
    parser.add_argument("--g-acceptance", type=Path, required=True)
    parser.add_argument("--l-acceptance", type=Path, required=True)
    args = parser.parse_args()
    run_series_study(
        output=args.output_dir,
        worktree=args.worktree,
        reference_module=args.reference_module,
        reference_sha256=args.reference_sha256,
        oracle_dir=args.oracle_dir,
        g_root=args.g_root,
        l_root=args.l_root,
        g_acceptance=args.g_acceptance,
        l_acceptance=args.l_acceptance,
    )


if __name__ == "__main__":
    main()
