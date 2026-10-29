"""Evaluate the approved finite-box acoustic gravity benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import resource
import shlex
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from gragra.constants import G_SI
from gragra.fields.thermodynamic import adiabatic_pressure_density_perturbation
from gragra.gravity_sources import WeightedPointSource
from gragra.observables import (
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)

SOUND_SPEED_M_S = 343.0
FREQUENCY_HZ = 10.0
PRESSURE_PA = 1.0
WAVELENGTH_M = SOUND_SPEED_M_S / FREQUENCY_HZ
REFERENCE_PANEL_WIDTH_M = WAVELENGTH_M / 2.0
TARGETS_M = np.array(
    [
        [0.0, 0.0, -0.25 * WAVELENGTH_M],
        [0.125 * WAVELENGTH_M, 0.0, -0.5 * WAVELENGTH_M],
        [0.0, 0.25 * WAVELENGTH_M, -WAVELENGTH_M],
    ],
    dtype=np.float64,
)


def acoustic_case_matrix() -> list[dict[str, int | float | str]]:
    result: list[dict[str, int | float | str]] = []
    for cells in (8, 12, 16):
        result.append(
            {
                "family": "mesh_width",
                "cells_per_wavelength": cells,
                "horizontal_half_width_wavelengths": 3,
                "height_wavelengths": 2,
            }
        )
    for half_width in (2, 3, 4):
        result.append(
            {
                "family": "horizontal_half_width",
                "cells_per_wavelength": 12,
                "horizontal_half_width_wavelengths": half_width,
                "height_wavelengths": 2,
            }
        )
    for height in (1, 2, 3):
        result.append(
            {
                "family": "height",
                "cells_per_wavelength": 12,
                "horizontal_half_width_wavelengths": 3,
                "height_wavelengths": height,
            }
        )
    return result


def acoustic_density_phasor(points_m: np.ndarray) -> np.ndarray:
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_m must be finite with shape (N,3)")
    pressure = PRESSURE_PA * np.exp(1j * 2.0 * np.pi * points[:, 0] / WAVELENGTH_M)
    density = np.asarray(
        adiabatic_pressure_density_perturbation(pressure, SOUND_SPEED_M_S),
        dtype=np.complex128,
    ).copy()
    density.setflags(write=False)
    return density


def acoustic_real_density_slab(
    density_phasor: np.ndarray,
    times_s: np.ndarray,
    *,
    sample_slice: slice,
    source_slice: slice,
) -> np.ndarray:
    """Materialize one bounded real density sample/source slab."""
    phasor = np.asarray(density_phasor, dtype=np.complex128)
    times = np.asarray(times_s, dtype=np.float64)
    if phasor.ndim != 1 or times.ndim != 1:
        raise ValueError("density_phasor and times_s must be one-dimensional")
    if not np.isfinite(phasor).all() or not np.isfinite(times).all():
        raise ValueError("density_phasor and times_s must be finite")
    selected_times = times[sample_slice]
    selected_phasor = phasor[source_slice]
    phase = np.exp(-2j * np.pi * FREQUENCY_HZ * selected_times[:, None])
    result = np.asarray(
        np.real(phase * selected_phasor[None, :]), dtype=np.float64
    ).copy()
    result.setflags(write=False)
    return result


def _case_key(parameters: dict[str, int | float | str]) -> tuple[int, float, float]:
    return (
        int(parameters["cells_per_wavelength"]),
        float(parameters["horizontal_half_width_wavelengths"]),
        float(parameters["height_wavelengths"]),
    )


def case_representation(
    matrix: list[dict[str, int | float | str]],
) -> dict[str, object]:
    """Map repeated sweep entries to the first identical physical case."""
    first_by_key: dict[tuple[int, float, float], int] = {}
    representative_by_case_index: dict[str, int] = {}
    for index, parameters in enumerate(matrix):
        key = _case_key(parameters)
        first_by_key.setdefault(key, index)
        representative_by_case_index[str(index)] = first_by_key[key]
    return {
        "unique_case_indices": sorted(set(representative_by_case_index.values())),
        "representative_by_case_index": representative_by_case_index,
    }


def study_status(oracle_statuses: list[str]) -> str:
    return (
        "complete"
        if oracle_statuses and all(value == "CONVERGED" for value in oracle_statuses)
        else "partial"
    )


def midpoint_box_sources(
    *,
    horizontal_half_width_wavelengths: float,
    height_wavelengths: float,
    cells_per_wavelength: int,
) -> tuple[np.ndarray, np.ndarray]:
    half_width = float(horizontal_half_width_wavelengths) * WAVELENGTH_M
    height = float(height_wavelengths) * WAVELENGTH_M
    cells = int(cells_per_wavelength)
    if half_width <= 0.0 or height <= 0.0 or cells <= 0:
        raise ValueError("box dimensions and cells_per_wavelength must be positive")
    nx = int(round(2.0 * half_width / WAVELENGTH_M * cells))
    nz = int(round(height / WAVELENGTH_M * cells))
    x_width = 2.0 * half_width / nx
    z_width = height / nz
    x = -half_width + (np.arange(nx, dtype=np.float64) + 0.5) * x_width
    y = x.copy()
    z = (np.arange(nz, dtype=np.float64) + 0.5) * z_width
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    positions = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    volumes = np.full(len(positions), x_width * x_width * z_width, dtype=np.float64)
    positions.setflags(write=False)
    volumes.setflags(write=False)
    return positions, volumes


def _partition_bounds(
    bounds_m: np.ndarray, *, maximum_width_m: float
) -> list[np.ndarray]:
    bounds = np.asarray(bounds_m, dtype=np.float64)
    maximum_width = float(maximum_width_m)
    if bounds.shape != (3, 2) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("bounds_m must have shape (3,2) with increasing endpoints")
    if (
        not np.isfinite(bounds).all()
        or not np.isfinite(maximum_width)
        or maximum_width <= 0
    ):
        raise ValueError("bounds_m and maximum_width_m must be finite and positive")
    intervals = []
    for lower, upper in bounds:
        count = max(1, int(np.ceil((upper - lower) / maximum_width - 1e-14)))
        edges = np.linspace(lower, upper, count + 1)
        intervals.append(tuple(zip(edges[:-1], edges[1:], strict=True)))
    panels = []
    for x_interval, y_interval, z_interval in itertools.product(*intervals):
        panel = np.asarray((x_interval, y_interval, z_interval), dtype=np.float64)
        panel.setflags(write=False)
        panels.append(panel)
    return panels


def synthesize_real_series(
    phasors: dict[str, np.ndarray], times_s: np.ndarray
) -> dict[str, np.ndarray]:
    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("times_s must be a finite one-dimensional array")
    phase_shape = (len(times),) + (1,) * max(
        np.asarray(value).ndim for value in phasors.values()
    )
    phase = np.exp(-2j * np.pi * FREQUENCY_HZ * times).reshape(phase_shape)
    result = {}
    for name, value in phasors.items():
        amplitude = np.asarray(value, dtype=np.complex128)
        local_phase = phase.reshape((len(times),) + (1,) * amplitude.ndim)
        series = np.asarray(
            np.real(local_phase * amplitude[None, ...]), dtype=np.float64
        )
        series.setflags(write=False)
        result[name] = series
    return result


def fit_phasor(
    series: dict[str, np.ndarray],
    times_s: np.ndarray,
    *,
    sample_slice: slice,
) -> dict[str, np.ndarray]:
    times = np.asarray(times_s, dtype=np.float64)[sample_slice]
    design = np.column_stack(
        (
            np.cos(2.0 * np.pi * FREQUENCY_HZ * times),
            np.sin(2.0 * np.pi * FREQUENCY_HZ * times),
        )
    )
    result = {}
    for name, value in series.items():
        samples = np.asarray(value, dtype=np.float64)[sample_slice]
        flat = samples.reshape(len(samples), -1)
        coefficients, _, _, _ = np.linalg.lstsq(design, flat, rcond=None)
        amplitude = (coefficients[0] + 1j * coefficients[1]).reshape(samples.shape[1:])
        amplitude = np.asarray(amplitude, dtype=np.complex128)
        amplitude.setflags(write=False)
        result[name] = amplitude
    return result


def _observables(positions: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    source = WeightedPointSource(positions, weights)
    return {
        "potential": point_potential(source, TARGETS_M),
        "acceleration": point_acceleration(source, TARGETS_M),
        "gradient": point_gravity_gradient(source, TARGETS_M),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _complex_json(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value, dtype=np.complex128)
    return {
        "shape": list(array.shape),
        "real": array.real.tolist(),
        "imag": array.imag.tolist(),
    }


def _norm_per_target(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1:
        return np.abs(array)
    return np.linalg.norm(array, axis=tuple(range(1, array.ndim)))


def _difference(
    observed: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> dict[str, object]:
    result = {}
    for name in ("potential", "acceleration", "gradient"):
        absolute = _norm_per_target(observed[name] - reference[name])
        scale = _norm_per_target(reference[name])
        result[name] = {
            "absolute_norm": absolute.tolist(),
            "reference_norm": scale.tolist(),
            "normalized": (absolute / np.maximum(scale, np.finfo(float).tiny)).tolist(),
        }
    return result


def _load_reference(path: Path, expected_sha256: str):
    path = Path(path).resolve()
    observed_sha256 = _sha256(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"A reference source hash mismatch: {observed_sha256} != {expected_sha256}"
        )
    specification = importlib.util.spec_from_file_location(
        "gragra_lane_a_wave456_reference", path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load A reference module from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _partitioned_finite_reference(
    reference,
    bounds_m: np.ndarray,
    *,
    order: int,
) -> dict[str, np.ndarray]:
    totals = {
        "potential": np.zeros(len(TARGETS_M), dtype=np.complex128),
        "acceleration": np.zeros((len(TARGETS_M), 3), dtype=np.complex128),
        "gradient": np.zeros((len(TARGETS_M), 3, 3), dtype=np.complex128),
    }
    for panel in _partition_bounds(bounds_m, maximum_width_m=REFERENCE_PANEL_WIDTH_M):
        value = reference.acoustic_finite_box_observables_g_free(
            panel,
            TARGETS_M,
            pressure_pa=PRESSURE_PA,
            sound_speed_m_s=SOUND_SPEED_M_S,
            frequency_hz=FREQUENCY_HZ,
            order=order,
        )
        for name in totals:
            totals[name] += getattr(value, name)
    for name, value in totals.items():
        totals[name] = np.asarray(G_SI * value, dtype=np.complex128)
        totals[name].setflags(write=False)
    return totals


def _maximum_normalized_difference(
    low: dict[str, np.ndarray], high: dict[str, np.ndarray]
) -> float:
    comparisons = _difference(low, high)
    return max(
        max(comparisons[name]["normalized"])
        for name in ("potential", "acceleration", "gradient")
    )


def run_study(
    output: Path,
    reference_module: Path,
    reference_sha256: str,
    reference_json: Path,
    g_acceptance: Path,
    worktree: Path,
) -> None:
    output = Path(output).resolve()
    reference_module = Path(reference_module).resolve()
    reference_json = Path(reference_json).resolve()
    g_acceptance = Path(g_acceptance).resolve()
    worktree = Path(worktree).resolve()
    if output.exists():
        raise FileExistsError(output)
    for dependency in (reference_module, reference_json, g_acceptance):
        if not dependency.is_file():
            raise FileNotFoundError(dependency)
    source_path = Path(__import__("gragra").__file__).resolve()
    expected_source = worktree / "src/gragra/__init__.py"
    if source_path != expected_source:
        raise RuntimeError(
            f"gragra imported from {source_path}, expected {expected_source}"
        )
    reference = _load_reference(reference_module, reference_sha256)
    output.mkdir(parents=True)
    started_at = _iso_now()
    run_started = time.monotonic()
    cases = []
    times = np.arange(128, dtype=np.float64) / (64.0 * FREQUENCY_HZ)
    for case_index, parameters in enumerate(acoustic_case_matrix()):
        started = time.monotonic()
        positions, volumes = midpoint_box_sources(
            horizontal_half_width_wavelengths=float(
                parameters["horizontal_half_width_wavelengths"]
            ),
            height_wavelengths=float(parameters["height_wavelengths"]),
            cells_per_wavelength=int(parameters["cells_per_wavelength"]),
        )
        density = acoustic_density_phasor(positions)
        discrete = _observables(positions, density * volumes)
        half_width = (
            float(parameters["horizontal_half_width_wavelengths"]) * WAVELENGTH_M
        )
        height = float(parameters["height_wavelengths"]) * WAVELENGTH_M
        bounds = np.array(
            [[-half_width, half_width], [-half_width, half_width], [0.0, height]],
            dtype=np.float64,
        )
        finite_low = _partitioned_finite_reference(reference, bounds, order=10)
        finite_high = _partitioned_finite_reference(reference, bounds, order=12)
        low_order, high_order = 10, 12
        normalized_quadrature_difference = _maximum_normalized_difference(
            finite_low, finite_high
        )
        convergence_attempts = [
            {
                "low_order": 10,
                "high_order": 12,
                "normalized_difference": normalized_quadrature_difference,
                "status": (
                    "CONVERGED"
                    if normalized_quadrature_difference <= 1e-11
                    else "FAILED_TARGET"
                ),
            }
        ]
        if normalized_quadrature_difference > 1e-11:
            finite_low = _partitioned_finite_reference(reference, bounds, order=16)
            finite_high = _partitioned_finite_reference(reference, bounds, order=24)
            low_order, high_order = 16, 24
            normalized_quadrature_difference = _maximum_normalized_difference(
                finite_low, finite_high
            )
            convergence_attempts.append(
                {
                    "low_order": 16,
                    "high_order": 24,
                    "normalized_difference": normalized_quadrature_difference,
                    "status": (
                        "CONVERGED"
                        if normalized_quadrature_difference <= 1e-11
                        else "FAILED_TARGET"
                    ),
                }
            )
        infinite = reference.acoustic_infinite_halfspace_observables_g_free(
            TARGETS_M,
            pressure_pa=PRESSURE_PA,
            sound_speed_m_s=SOUND_SPEED_M_S,
            frequency_hz=FREQUENCY_HZ,
        )
        infinite_physical = {
            name: G_SI * getattr(infinite, name)
            for name in ("potential", "acceleration", "gradient")
        }
        real_series = synthesize_real_series(discrete, times)
        one_cycle = fit_phasor(real_series, times, sample_slice=slice(0, 64))
        two_cycles = fit_phasor(real_series, times, sample_slice=slice(0, 128))
        case = {
            "case_index": case_index,
            **parameters,
            "bounds_m": bounds.tolist(),
            "mesh_width_m": WAVELENGTH_M / int(parameters["cells_per_wavelength"]),
            "source_count": len(positions),
            "density_model": {
                "pressure_hat_pa": PRESSURE_PA,
                "sound_speed_m_s": SOUND_SPEED_M_S,
                "density_hat_at_phase_origin_kg_m3": PRESSURE_PA / SOUND_SPEED_M_S**2,
                "spatial_phase": "exp(+i k x)",
                "temporal_phase": "exp(-i 2*pi*f*t)",
                "air_bulk_only": True,
                "fixed_ground_sheet_included": False,
            },
            "numerical_midpoint": {
                name: _complex_json(value) for name, value in discrete.items()
            },
            "independent_finite_box_low_order": {
                name: _complex_json(value) for name, value in finite_low.items()
            },
            "independent_finite_box_high_order": {
                name: _complex_json(value) for name, value in finite_high.items()
            },
            "independent_finite_box_convergence": {
                "status": (
                    "CONVERGED"
                    if normalized_quadrature_difference <= 1e-11
                    else "ORACLE_UNRESOLVED"
                ),
                "low_order": low_order,
                "high_order": high_order,
                "partition_maximum_width_m": REFERENCE_PANEL_WIDTH_M,
                "partition_count": len(
                    _partition_bounds(bounds, maximum_width_m=REFERENCE_PANEL_WIDTH_M)
                ),
                "normalized_difference": normalized_quadrature_difference,
                "target": 1e-11,
                "attempts": convergence_attempts,
            },
            "infinite_horizontal_halfspace": {
                name: _complex_json(value) for name, value in infinite_physical.items()
            },
            "comparisons": {
                "numerical_vs_same_finite_box_high_order": _difference(
                    discrete, finite_high
                ),
                "finite_box_low_vs_high": _difference(finite_low, finite_high),
                "finite_box_vs_infinite_horizontal_halfspace": _difference(
                    finite_high, infinite_physical
                ),
            },
            "real_time_fit": {
                "times_s": times.tolist(),
                "one_cycle_samples": [0, 64],
                "two_cycle_samples": [0, 128],
                "one_cycle": {
                    name: _complex_json(value) for name, value in one_cycle.items()
                },
                "two_cycles": {
                    name: _complex_json(value) for name, value in two_cycles.items()
                },
                "one_cycle_vs_input": _difference(one_cycle, discrete),
                "two_cycles_vs_input": _difference(two_cycles, discrete),
            },
            "elapsed_s": time.monotonic() - started,
        }
        cases.append(case)
        case_path = output / f"case-{case_index:02d}-{parameters['family']}.json"
        case_path.write_text(
            json.dumps(case, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    manifest = {
        "status": study_status(
            [case["independent_finite_box_convergence"]["status"] for case in cases]
        ),
        "classification": {
            "numerical_midpoint": "observed computation",
            "independent_finite_box": "inferred independent quadrature",
            "infinite_horizontal_halfspace": "inferred closed-form reference",
            "imputed_values": [],
        },
        "started_at": started_at,
        "ended_at": _iso_now(),
        "source": {
            "gragra": str(source_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "thermodynamic": {
                "path": str(
                    Path(
                        adiabatic_pressure_density_perturbation.__code__.co_filename
                    ).resolve()
                ),
                "sha256": _sha256(
                    Path(
                        adiabatic_pressure_density_perturbation.__code__.co_filename
                    ).resolve()
                ),
            },
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
        },
        "dependencies": {
            "A_reference_module": {
                "path": str(reference_module),
                "sha256": _sha256(reference_module),
            },
            "A_reference_values": {
                "path": str(reference_json),
                "sha256": _sha256(reference_json),
            },
            "G_parent_acceptance": {
                "path": str(g_acceptance),
                "sha256": _sha256(g_acceptance),
                "qualification": (
                    "accepted series/bridge contract; integration is parent-owned"
                ),
            },
        },
        "fixed_parameters": {
            "sound_speed_m_s": SOUND_SPEED_M_S,
            "frequency_hz": FREQUENCY_HZ,
            "wavelength_m": WAVELENGTH_M,
            "pressure_hat_pa": PRESSURE_PA,
            "targets_m": TARGETS_M.tolist(),
            "sample_formula": "t_n=n/(64f), n=0..127",
            "case_count": len(cases),
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
        "case_files": {
            f"case-{index:02d}-{case['family']}.json": {
                "sha256": _sha256(output / f"case-{index:02d}-{case['family']}.json")
            }
            for index, case in enumerate(cases)
        },
    }
    (output / "acoustic-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-module", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--g-acceptance", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    args = parser.parse_args()
    run_study(
        args.output_dir,
        args.reference_module,
        args.reference_sha256,
        args.reference_json,
        args.g_acceptance,
        args.worktree,
    )


if __name__ == "__main__":
    main()
