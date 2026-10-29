#!/usr/bin/env python3
"""Stream the saved 250 m Rayleigh case through the Wave 5 A/B/C paths.

A fits the actual solver displacement on the retained element-local GLL rows.
B evaluates the analytic displacement on those exact rows and masses. Both use
the public displacement observables, where ``G_SI`` is applied exactly once.
C comes from the accepted independent infinite-halfspace reference wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import resource
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [
    item
    for item in sys.meta_path
    if type(item).__name__ != "ScikitBuildRedirectingFinder"
]
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import gragra  # noqa: E402
from gragra.adapters.specfem import schema  # noqa: E402
from gragra.adapters.specfem.streaming import SpecfemGLLStreamReader  # noqa: E402
from gragra.observables.dipole import (  # noqa: E402
    displacement_transfer_acceleration,
    displacement_transfer_gravity_gradient,
    displacement_transfer_potential,
)
from gragra.observables.wave456_reference import (  # noqa: E402
    rayleigh_halfspace_observables,
)
from gragra.research.rayleigh_timefit import (  # noqa: E402
    fit_complex_amplitude,
    snapshot_physical_times,
)
from gragra.waves.rayleigh import rayleigh_displacement  # noqa: E402

if not Path(gragra.__file__).resolve().is_relative_to(ROOT / "src"):
    raise RuntimeError(
        "research script imported gragra outside its integration worktree"
    )

FREQUENCY_HZ = 2.5
OMEGA_RAD_S = 2.0 * np.pi * FREQUENCY_HZ
T_REF_S = 10.0
DT_S = 0.005
SAVED_SAMPLE_COUNT = 240
FRONT_STOP_S = 10.8
VP_M_S = 2800.0
VS_M_S = 1500.0
RHO_KG_M3 = 2300.0
SURFACE_VERTICAL_AMPLITUDE_M = 1e-9
WAVELENGTH_M = 556.3302242263422
RAYLEIGH_SPEED_M_S = FREQUENCY_HZ * WAVELENGTH_M
WAVENUMBER_RAD_M = 2.0 * np.pi / WAVELENGTH_M
DIRECTION = (1.0, 0.0, 0.0)
PHASE_ORIGIN_M = (0.0, 0.0, 0.0)
GLL_POINTS_PER_ELEMENT = 125
DEFAULT_SOURCE_BLOCK_ELEMENTS = 6912
DEFAULT_KERNEL_CHUNK_SIZE = 1024
DEFAULT_MEMORY_BUDGET_BYTES = 2 * 1024**3
REAL_MEMORY_PREFLIGHT_MAX_ELEMENTS = 512
MEMORY_PLANNING_FRACTION = 0.80
MEMORY_PREFLIGHT_TARGET_FRACTION = 0.90

_OBSERVABLE_SHAPES = {
    "potential": (),
    "acceleration": (3,),
    "gradient": (3, 3),
}
_NORM_NAMES = {
    "potential": "complex_scalar_absolute",
    "acceleration": "complex_vector_2_norm",
    "gradient": "complex_tensor_frobenius_norm",
}


def canonical_targets_m() -> np.ndarray:
    """Return the three prescribed targets without exposing mutable storage."""
    targets = np.array(
        [
            [3000.0, 3000.0, 0.25 * WAVELENGTH_M],
            [3000.0, 3000.0, 0.50 * WAVELENGTH_M],
            [3000.0, 3000.0, 0.75 * WAVELENGTH_M],
        ],
        dtype=np.float64,
    )
    targets.setflags(write=False)
    return targets


def _positive_integer(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result != value or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def rayleigh_windows(
    raw_times_s: np.ndarray, time_step_indices: np.ndarray
) -> dict[str, Any]:
    """Validate and return the two canonical windows and corrected time map."""
    raw = np.asarray(raw_times_s)
    steps_raw = np.asarray(time_step_indices)
    if raw.ndim != 1 or steps_raw.ndim != 1:
        raise ValueError("saved raw times and time_step must be one-dimensional")
    if len(raw) != SAVED_SAMPLE_COUNT or len(steps_raw) != SAVED_SAMPLE_COUNT:
        raise ValueError("the saved 250 m case requires exactly 240 samples")
    if raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError("saved raw times must be finite real values")
    if steps_raw.dtype.kind not in "iu" or steps_raw.dtype.kind == "b":
        raise ValueError("time_step must contain integer indices")
    steps = np.asarray(steps_raw, dtype=np.int64)
    expected_steps = np.arange(SAVED_SAMPLE_COUNT, dtype=np.int64)
    if not np.array_equal(steps, expected_steps):
        raise ValueError("saved time_step must be the exact indices 0..239")
    raw = np.asarray(raw, dtype=np.float64)
    expected_raw = T_REF_S + expected_steps.astype(np.float64) * DT_S
    if not np.array_equal(raw, expected_raw):
        raise ValueError(
            "saved raw times must equal 10 s + time_step * 0.005 s exactly"
        )
    physical = snapshot_physical_times(steps, t_init_s=T_REF_S, dt_s=DT_S)
    front = np.flatnonzero(physical <= FRONT_STOP_S + 1e-12)

    def record(indices: np.ndarray) -> dict[str, Any]:
        return {
            "input_sample_indices": indices.tolist(),
            "time_step_indices": steps[indices].tolist(),
            "raw_times_s": raw[indices].tolist(),
            "physical_times_s": physical[indices].tolist(),
        }

    return {
        "front": record(front),
        "full": record(np.arange(SAVED_SAMPLE_COUNT, dtype=np.int64)),
        "fit_contract": {
            "frequency_hz": FREQUENCY_HZ,
            "omega_rad_s": OMEGA_RAD_S,
            "t_ref_s": T_REF_S,
            "time_domain_convention": "Re[C exp(-i omega (t-t_ref_s))]",
            "physical_time_mapping": "10 + (time_step + 1) * 0.005 s",
            "phase_origin_m": list(PHASE_ORIGIN_M),
        },
    }


def _three_observables(
    positions_m: np.ndarray,
    mass_kg: np.ndarray,
    displacement_m: np.ndarray,
    targets_m: np.ndarray,
    *,
    kernel_chunk_size: int,
) -> dict[str, np.ndarray]:
    return {
        "potential": displacement_transfer_potential(
            positions_m,
            mass_kg,
            displacement_m,
            targets_m,
            chunk_size=kernel_chunk_size,
        ),
        "acceleration": displacement_transfer_acceleration(
            positions_m,
            mass_kg,
            displacement_m,
            targets_m,
            chunk_size=kernel_chunk_size,
        ),
        "gradient": displacement_transfer_gravity_gradient(
            positions_m,
            mass_kg,
            displacement_m,
            targets_m,
            chunk_size=kernel_chunk_size,
        ),
    }


def _norm_by_target(value: np.ndarray, observable: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.complex128)
    if observable == "potential":
        return np.abs(array)
    if observable == "acceleration":
        return np.linalg.norm(array, axis=-1)
    return np.sqrt(np.sum(np.abs(array) ** 2, axis=(-2, -1)))


def _complex_record(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.complex128)
    if not np.isfinite(array.real).all() or not np.isfinite(array.imag).all():
        raise ValueError("complex observable output must be finite")
    return {
        "encoding": "complex128_real_imag_last_axis",
        "shape": list(array.shape),
        "values": np.stack((array.real, array.imag), axis=-1).tolist(),
    }


def _encoded_observables(values: dict[str, np.ndarray]) -> dict[str, Any]:
    return {name: _complex_record(values[name]) for name in _OBSERVABLE_SHAPES}


def _window_result(
    window_contract: dict[str, Any],
    a_values: dict[str, np.ndarray],
    b_values: dict[str, np.ndarray],
    c_values: dict[str, np.ndarray],
) -> dict[str, Any]:
    differences = {
        "A_minus_B": {name: a_values[name] - b_values[name] for name in a_values},
        "B_minus_C": {name: b_values[name] - c_values[name] for name in b_values},
        "A_minus_C": {name: a_values[name] - c_values[name] for name in a_values},
    }
    metrics: dict[str, Any] = {}
    for name in _OBSERVABLE_SHAPES:
        scale = _norm_by_target(c_values[name], name)
        if not np.isfinite(scale).all() or np.any(scale <= 0.0):
            raise ValueError(f"C {name} norm must be finite and nonzero per target")
        observable_metrics = {
            "norm": _NORM_NAMES[name],
            "normalization": "per-target C norm; no component-wise division",
            "C_norm_by_target": scale.tolist(),
        }
        for comparison, difference in differences.items():
            difference_norm = _norm_by_target(difference[name], name)
            normalized = difference_norm / scale
            observable_metrics[f"{comparison}_norm_by_target"] = (
                difference_norm.tolist()
            )
            observable_metrics[f"{comparison}_normalized_by_C"] = normalized.tolist()
            observable_metrics[f"{comparison}_normalized_by_C_statistics"] = {
                "mean": float(np.mean(normalized)),
                "median": float(np.median(normalized)),
                "standard_deviation": float(np.std(normalized)),
            }
        metrics[name] = observable_metrics
    return {
        **window_contract,
        "paths": {
            "A_solver_fitted": _encoded_observables(a_values),
            "B_analytic_same_gll": _encoded_observables(b_values),
            "C_infinite_halfspace": _encoded_observables(c_values),
        },
        "differences": {
            comparison: _encoded_observables(values)
            for comparison, values in differences.items()
        },
        "metrics_by_target": metrics,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _sha256_source_ids(source_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for source_id in source_ids:
        encoded = source_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _source_file_identity() -> dict[str, Any]:
    paths = {
        "crosspath_script": Path(__file__).resolve(),
        "streaming_reader": ROOT / "src/gragra/adapters/specfem/streaming.py",
        "time_fit": ROOT / "src/gragra/research/rayleigh_timefit.py",
        "rayleigh_kinematics": ROOT / "src/gragra/waves/rayleigh.py",
        "displacement_observables": ROOT / "src/gragra/observables/dipole.py",
        "independent_reference_wrapper": (
            ROOT / "src/gragra/observables/wave456_reference.py"
        ),
        "independent_reference_analytic": (
            ROOT / "src/gragra/analytic/wave456_reference.py"
        ),
    }
    return {
        name: {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for name, path in paths.items()
    }


def _input_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"input is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _selected_element_slice(
    element_start: int, element_stop: int | None
) -> slice | None:
    if (
        isinstance(element_start, (bool, np.bool_))
        or int(element_start) != element_start
    ):
        raise ValueError("element_start must be a non-negative integer")
    start = int(element_start)
    if start < 0:
        raise ValueError("element_start must be a non-negative integer")
    if element_stop is None:
        if start != 0:
            raise ValueError("element_stop is required when element_start is nonzero")
        return None
    if isinstance(element_stop, (bool, np.bool_)) or int(element_stop) != element_stop:
        raise ValueError("element_stop must be a positive integer")
    stop = int(element_stop)
    if stop <= start:
        raise ValueError("element_stop must be greater than element_start")
    return slice(start, stop)


def _resource_plan(
    *,
    reader_resident_bytes: int,
    requested_block_elements: int,
    selected_elements: int,
    memory_budget_bytes: int,
) -> dict[str, Any]:
    # This allocation count is an analytic first bound only.  HDF5 and LAPACK
    # allocator peaks are implementation-dependent, so the final block choice
    # is governed by the isolated actual-reader preflight below.
    per_element_live = (
        3 * SAVED_SAMPLE_COUNT * GLL_POINTS_PER_ELEMENT * 3 * 8
        + 2 * GLL_POINTS_PER_ELEMENT * 3 * 16
        + GLL_POINTS_PER_ELEMENT * 4 * 8
    )
    planning_limit = int(memory_budget_bytes * MEMORY_PLANNING_FRACTION)
    available = planning_limit - reader_resident_bytes
    maximum_from_budget = available // per_element_live
    if maximum_from_budget < 1:
        raise MemoryError("memory budget cannot hold one 240-sample GLL source block")
    effective_elements = min(
        requested_block_elements, selected_elements, int(maximum_from_budget)
    )
    estimated_block_live = effective_elements * per_element_live
    estimated_peak = reader_resident_bytes + estimated_block_live
    return {
        "memory_budget_bytes": memory_budget_bytes,
        "memory_planning_fraction": MEMORY_PLANNING_FRACTION,
        "memory_planning_limit_bytes": planning_limit,
        "reader_estimated_resident_bytes": reader_resident_bytes,
        "requested_source_block_elements": requested_block_elements,
        "effective_source_block_elements": effective_elements,
        "effective_source_block_count": effective_elements * GLL_POINTS_PER_ELEMENT,
        "source_block_adjusted_for_budget": (
            effective_elements < min(requested_block_elements, selected_elements)
        ),
        "per_element_analytic_live_bytes": per_element_live,
        "estimated_block_live_bytes": estimated_block_live,
        "estimated_peak_working_bytes": estimated_peak,
        "estimate_scope": (
            "analytic reader estimate plus two live 240-sample displacement buffers, "
            "one full np.linalg.lstsq RHS copy, fitted and analytic complex "
            "phasors, and source geometry; actual-reader preflight governs selection"
        ),
    }


def _metadata_resource_probe(
    hdf5_path: str | Path,
    *,
    element_start: int,
    element_stop: int | None,
    requested_block_elements: int,
    memory_budget_bytes: int,
) -> dict[str, Any]:
    """Read HDF metadata only and reproduce the reader's resident estimate."""
    try:
        import h5py
    except ImportError as error:
        raise ImportError("h5py is required for Rayleigh memory preflight") from error

    path = Path(hdf5_path).expanduser().resolve()
    element_slice = _selected_element_slice(element_start, element_stop)
    with h5py.File(path, "r") as h5:
        coordinate_shape = h5[schema.PATH_MESH_COORDINATES].shape
        displacement = h5[schema.PATH_FIELD_DISPLACEMENT]
        if len(coordinate_shape) != 3:
            raise ValueError("/mesh/coordinates must have shape (element, gll, 3)")
        n_elements, ngll, _ = coordinate_shape
        if ngll != GLL_POINTS_PER_ELEMENT:
            raise ValueError("Rayleigh preflight requires 125 GLL points per element")
        selected_start = 0 if element_slice is None else element_slice.start
        selected_stop = n_elements if element_slice is None else element_slice.stop
        if selected_stop > n_elements:
            raise ValueError("element_slice stop exceeds the available elements")
        selected_elements = selected_stop - selected_start
        chunk_sizes = []
        for dataset_path in schema.required_dataset_paths("time_sampled"):
            dataset = h5[dataset_path]
            if dataset.chunks is not None:
                chunk_sizes.append(
                    int(
                        np.prod(dataset.chunks, dtype=np.int64) * dataset.dtype.itemsize
                    )
                )
        cache_resident = max(chunk_sizes, default=0) * len(chunk_sizes)
        chunk_shape = (
            None
            if displacement.chunks is None
            else [int(value) for value in displacement.chunks]
        )
        field_shape = [int(value) for value in displacement.shape]
        n_time = field_shape[0]

    source_count = selected_elements * ngll
    validation_slab_elements = min(128, selected_elements)
    layout_bytes = n_elements * 8 + n_time * 3 * 8 + 4096
    reader_estimate = int(
        cache_resident
        + source_count * (3 * 8 + 72)
        + layout_bytes
        + n_time * 8
        + validation_slab_elements * ngll * 160
        + 4096
    )
    plan = _resource_plan(
        reader_resident_bytes=reader_estimate,
        requested_block_elements=_positive_integer(
            requested_block_elements, "source_block_elements"
        ),
        selected_elements=selected_elements,
        memory_budget_bytes=memory_budget_bytes,
    )
    return {
        "selected_start": selected_start,
        "selected_stop": selected_stop,
        "selected_elements": selected_elements,
        "total_elements": n_elements,
        "field_shape": field_shape,
        "field_chunk_shape": chunk_shape,
        "reader_estimated_resident_bytes": reader_estimate,
        "validation_slab_elements": validation_slab_elements,
        "budget_plan": plan,
    }


def _memory_preflight_worker(connection: Any, block_elements: int) -> None:
    """Measure synthetic numerical buffers in an isolated process."""
    try:
        source_count = block_elements * GLL_POINTS_PER_ELEMENT
        displacement = np.full(
            (SAVED_SAMPLE_COUNT, source_count, 3), 1.0, dtype=np.float64
        )
        provider_copy = displacement.copy()
        physical_times = snapshot_physical_times(
            np.arange(SAVED_SAMPLE_COUNT), t_init_s=T_REF_S, dt_s=DT_S
        )
        fitted = fit_complex_amplitude(
            provider_copy, physical_times, OMEGA_RAD_S, t_ref_s=T_REF_S
        )
        # Keep both field buffers live through the fit, as the HDF generator does.
        if displacement.shape != provider_copy.shape or fitted.shape != (
            source_count,
            3,
        ):  # pragma: no cover
            raise RuntimeError("memory preflight buffer shape changed")
        current_rss_bytes = _current_rss_bytes()
        connection.send(
            {
                "status": "ok",
                "measurement": "isolated worker /proc/self/status VmHWM",
                "current_rss_bytes": current_rss_bytes,
                "peak_rss_bytes": _peak_rss_bytes(),
            }
        )
    except BaseException as error:
        connection.send(
            {"status": "error", "type": type(error).__name__, "message": str(error)}
        )
    finally:
        connection.close()


def _measure_memory_preflight(block_elements: int) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_memory_preflight_worker, args=(child, block_elements)
    )
    process.start()
    child.close()
    process.join()
    try:
        result = parent.recv() if parent.poll() else None
    except EOFError:
        result = None
    if result is None:
        result = {
            "status": "error",
            "type": "WorkerExit",
            "message": (
                "memory preflight worker exited with code "
                f"{process.exitcode} without a result"
            ),
        }
    parent.close()
    process.close()
    return result


def _real_memory_preflight_worker(
    connection: Any,
    hdf5_path: str,
    selected_start: int,
    selected_stop: int,
    block_elements: int,
    memory_budget_bytes: int,
) -> None:
    """Measure two real reader/provider/fit blocks in an isolated process."""
    try:
        preflight_stop = min(selected_stop, selected_start + 2 * block_elements)
        preflight_elements = preflight_stop - selected_start
        with SpecfemGLLStreamReader(
            hdf5_path,
            profile_density_range=(RHO_KG_M3, RHO_KG_M3),
            element_slice=slice(selected_start, preflight_stop),
            slab_elements=min(128, preflight_elements),
            memory_budget_bytes=memory_budget_bytes,
        ) as reader:
            layout = reader._layout
            if layout is None:  # pragma: no cover - guarded by reader context
                raise RuntimeError("validated SPECFEM layout is unavailable")
            windows = rayleigh_windows(layout.time_s, layout.time_step)
            physical_times = np.asarray(
                windows["full"]["physical_times_s"], dtype=np.float64
            )
            blocks_processed = 0
            for block in reader.provider.blocks(
                slice(0, SAVED_SAMPLE_COUNT),
                block_elements * GLL_POINTS_PER_ELEMENT,
            ):
                fitted = fit_complex_amplitude(
                    block.displacement_m,
                    physical_times,
                    OMEGA_RAD_S,
                    t_ref_s=T_REF_S,
                )
                if fitted.shape != (len(block.source_ids), 3):  # pragma: no cover
                    raise RuntimeError("real memory preflight fit shape changed")
                blocks_processed += 1
                if blocks_processed == 2:
                    break
            connection.send(
                {
                    "status": "ok",
                    "measurement": "isolated worker /proc/self/status VmHWM",
                    "measurement_scope": (
                        "actual HDF reader plus two 240-sample provider blocks "
                        "and harmonic fits"
                    ),
                    "selected_elements": preflight_elements,
                    "blocks_processed": blocks_processed,
                    "reader_estimated_resident_bytes": reader.estimated_resident_bytes,
                    "current_rss_bytes": _current_rss_bytes(),
                    "peak_rss_bytes": _peak_rss_bytes(),
                }
            )
    except BaseException as error:
        connection.send(
            {"status": "error", "type": type(error).__name__, "message": str(error)}
        )
    finally:
        connection.close()


def _measure_real_memory_preflight(
    hdf5_path: str | Path,
    *,
    selected_start: int,
    selected_stop: int,
    block_elements: int,
    memory_budget_bytes: int,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_real_memory_preflight_worker,
        args=(
            child,
            str(Path(hdf5_path).expanduser().resolve()),
            selected_start,
            selected_stop,
            block_elements,
            memory_budget_bytes,
        ),
    )
    process.start()
    child.close()
    process.join()
    try:
        result = parent.recv() if parent.poll() else None
    except EOFError:
        result = None
    if result is None:
        result = {
            "status": "error",
            "type": "WorkerExit",
            "message": (
                "real memory preflight worker exited with code "
                f"{process.exitcode} without a result"
            ),
        }
    parent.close()
    process.close()
    return result


def _choose_source_block(
    probe: dict[str, Any],
    *,
    hdf5_path: str | Path,
    requested_block_elements: int,
    memory_budget_bytes: int,
) -> dict[str, Any]:
    """Shrink a block until the isolated worker's interval peak RSS is safe."""
    requested = _positive_integer(requested_block_elements, "source_block_elements")
    plan = _resource_plan(
        reader_resident_bytes=probe["reader_estimated_resident_bytes"],
        requested_block_elements=requested,
        selected_elements=probe["selected_elements"],
        memory_budget_bytes=memory_budget_bytes,
    )
    candidate = min(
        plan["effective_source_block_elements"], REAL_MEMORY_PREFLIGHT_MAX_ELEMENTS
    )
    target = int(memory_budget_bytes * MEMORY_PREFLIGHT_TARGET_FRACTION)
    attempts = []
    while True:
        measured = _measure_real_memory_preflight(
            hdf5_path,
            selected_start=probe["selected_start"],
            selected_stop=probe["selected_stop"],
            block_elements=candidate,
            memory_budget_bytes=memory_budget_bytes,
        )
        attempt = {"source_block_elements": candidate, **measured}
        if measured["status"] == "ok":
            reader_adjustment = max(
                0,
                int(probe["reader_estimated_resident_bytes"])
                - int(
                    measured.get(
                        "reader_estimated_resident_bytes",
                        probe["reader_estimated_resident_bytes"],
                    )
                ),
            )
            combined = int(measured["peak_rss_bytes"]) + reader_adjustment
            attempt["full_reader_estimate_adjustment_bytes"] = reader_adjustment
            attempt["adjusted_peak_rss_bytes"] = combined
            attempts.append(attempt)
            if combined <= target:
                break
        else:
            attempts.append(attempt)
        if candidate == 1:
            return {
                "method": (
                    "isolated actual HDF reader plus two provider/fit blocks; "
                    "Linux VmHWM interval peak adjusted for the full reader estimate"
                ),
                "target_fraction": MEMORY_PREFLIGHT_TARGET_FRACTION,
                "target_bytes": target,
                "attempts": attempts,
                "effective_source_block_elements": None,
                "passed": False,
            }
        candidate = max(1, candidate // 2)
    return {
        "method": (
            "isolated actual HDF reader plus two provider/fit blocks; Linux VmHWM "
            "interval peak adjusted for the full reader estimate"
        ),
        "target_fraction": MEMORY_PREFLIGHT_TARGET_FRACTION,
        "target_bytes": target,
        "attempts": attempts,
        "effective_source_block_elements": candidate,
        "passed": True,
    }


def _field_io_plan(
    dataset: Any,
    *,
    selected_start: int,
    selected_stop: int,
    source_block_elements: int,
) -> dict[str, Any]:
    selected_elements = selected_stop - selected_start
    evaluation_samples = 160 + SAVED_SAMPLE_COUNT
    itemsize = int(dataset.dtype.itemsize)
    logical_evaluation = (
        evaluation_samples * selected_elements * GLL_POINTS_PER_ELEMENT * 3 * itemsize
    )
    logical_validation = int(np.prod(dataset.shape, dtype=np.int64) * itemsize)
    if dataset.chunks is None:
        return {
            "field_storage": "contiguous",
            "field_chunk_shape": None,
            "estimated_evaluation_logical_field_bytes": logical_evaluation,
            "estimated_evaluation_decoded_chunk_bytes": logical_evaluation,
            "estimated_validation_logical_field_bytes": logical_validation,
            "estimated_validation_decoded_chunk_bytes": logical_validation,
            "chunk_decode_estimate_assumption": (
                "contiguous dataset; decoded-chunk amplification does not apply"
            ),
        }

    chunk_shape = tuple(int(value) for value in dataset.chunks)
    chunk_bytes = int(np.prod(chunk_shape, dtype=np.int64) * itemsize)
    time_chunks = sum(
        math.ceil(sample_count / chunk_shape[0])
        for sample_count in (160, SAVED_SAMPLE_COUNT)
    )
    gll_chunks = math.ceil(GLL_POINTS_PER_ELEMENT / chunk_shape[2])
    component_chunks = math.ceil(3 / chunk_shape[3])
    element_chunk_intersections = 0
    for block_start in range(selected_start, selected_stop, source_block_elements):
        block_stop = min(block_start + source_block_elements, selected_stop)
        element_chunk_intersections += (
            (block_stop - 1) // chunk_shape[1] - block_start // chunk_shape[1] + 1
        )
    evaluation_chunk_reads = (
        time_chunks * element_chunk_intersections * gll_chunks * component_chunks
    )
    full_element_chunks = math.ceil(dataset.shape[1] / chunk_shape[1])
    validation_chunk_reads = (
        SAVED_SAMPLE_COUNT * full_element_chunks * gll_chunks * component_chunks
    )
    return {
        "field_storage": "chunked",
        "field_chunk_shape": list(chunk_shape),
        "field_chunk_uncompressed_bytes": chunk_bytes,
        "source_blocks_per_window": math.ceil(
            selected_elements / source_block_elements
        ),
        "estimated_evaluation_logical_field_bytes": logical_evaluation,
        "estimated_evaluation_chunk_reads": evaluation_chunk_reads,
        "estimated_evaluation_decoded_chunk_bytes": (
            evaluation_chunk_reads * chunk_bytes
        ),
        "estimated_validation_logical_field_bytes": logical_validation,
        "estimated_validation_chunk_reads": validation_chunk_reads,
        "estimated_validation_decoded_chunk_bytes": (
            validation_chunk_reads * chunk_bytes
        ),
        "chunk_decode_estimate_assumption": (
            "each source-block selection decodes every intersecting HDF5 field "
            "chunk once; validation counts one pass over each sample and field chunk"
        ),
    }


def evaluate_saved_rayleigh(
    hdf5_path: str | Path,
    *,
    element_start: int = 0,
    element_stop: int | None = None,
    source_block_elements: int = DEFAULT_SOURCE_BLOCK_ELEMENTS,
    kernel_chunk_size: int = DEFAULT_KERNEL_CHUNK_SIZE,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
    _known_input_identity: dict[str, Any] | None = None,
    _known_source_files: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the saved case without materializing all sources and times."""
    source_block_elements = _positive_integer(
        source_block_elements,
        "source_block_elements",
    )
    kernel_chunk_size = _positive_integer(kernel_chunk_size, "kernel_chunk_size")
    memory_budget_bytes = _positive_integer(memory_budget_bytes, "memory_budget_bytes")
    element_slice = _selected_element_slice(element_start, element_stop)
    input_path = Path(hdf5_path)
    input_identity = _known_input_identity or _input_file_identity(input_path)
    source_files = _known_source_files or _source_file_identity()
    targets = canonical_targets_m()
    slab_elements = (
        128
        if element_slice is None
        else min(128, element_slice.stop - element_slice.start)
    )

    with SpecfemGLLStreamReader(
        input_path,
        profile_density_range=(RHO_KG_M3, RHO_KG_M3),
        element_slice=element_slice,
        slab_elements=slab_elements,
        memory_budget_bytes=memory_budget_bytes,
    ) as reader:
        provider = reader.provider
        layout = reader._layout
        if layout is None:  # pragma: no cover - guarded by reader.provider
            raise RuntimeError("validated SPECFEM layout is unavailable")
        windows = rayleigh_windows(layout.time_s, layout.time_step)
        selected_elements = provider.source_count // GLL_POINTS_PER_ELEMENT
        selected_start = 0 if element_slice is None else element_slice.start
        selected_stop = selected_start + selected_elements
        is_full = selected_start == 0 and selected_stop == layout.n_elements
        resources = _resource_plan(
            reader_resident_bytes=reader.estimated_resident_bytes,
            requested_block_elements=source_block_elements,
            selected_elements=selected_elements,
            memory_budget_bytes=memory_budget_bytes,
        )
        block_sources = resources["effective_source_block_count"]
        h5 = reader._h5
        if h5 is None:  # pragma: no cover - guarded by reader.provider
            raise RuntimeError("open SPECFEM HDF5 handle is unavailable")
        resources.update(
            _field_io_plan(
                h5[schema.PATH_FIELD_DISPLACEMENT],
                selected_start=selected_start,
                selected_stop=selected_stop,
                source_block_elements=resources["effective_source_block_elements"],
            )
        )
        c_reference = rayleigh_halfspace_observables(
            targets,
            density_kg_m3=RHO_KG_M3,
            v_p_m_s=VP_M_S,
            v_s_m_s=VS_M_S,
            frequency_hz=FREQUENCY_HZ,
            vertical_surface_amplitude_m=SURFACE_VERTICAL_AMPLITUDE_M,
            direction=DIRECTION,
            phase_origin_m=PHASE_ORIGIN_M,
        )
        c_values = {
            "potential": c_reference.potential,
            "acceleration": c_reference.acceleration,
            "gradient": c_reference.gradient,
        }
        mass_digest = hashlib.sha256()
        output_windows: dict[str, Any] = {}
        resources["observed_parent_block_peak_rss_bytes"] = _rss_bytes()

        def enforce_parent_budget(stage: str) -> None:
            peak = _rss_bytes()
            resources["observed_parent_block_peak_rss_bytes"] = max(
                resources["observed_parent_block_peak_rss_bytes"], peak
            )
            if peak > memory_budget_bytes:
                raise MemoryError(
                    "actual peak RSS exceeded memory budget during "
                    f"{stage}: {peak} > {memory_budget_bytes} bytes"
                )

        for window_name in ("front", "full"):
            contract = windows[window_name]
            sample_indices = contract["input_sample_indices"]
            sample_slice = slice(sample_indices[0], sample_indices[-1] + 1)
            physical_times = np.asarray(contract["physical_times_s"], dtype=np.float64)
            a_values = {
                name: np.zeros((len(targets), *tail), dtype=np.complex128)
                for name, tail in _OBSERVABLE_SHAPES.items()
            }
            b_values = {name: np.zeros_like(value) for name, value in a_values.items()}
            expected_source_start = 0
            for block in provider.blocks(sample_slice, block_sources):
                enforce_parent_budget(f"{window_name} source-block read")
                if block.source_start != expected_source_start:
                    raise RuntimeError(
                        "provider returned non-contiguous source coverage"
                    )
                expected_source_start += len(block.source_ids)
                if window_name == "front":
                    mass = np.ascontiguousarray(block.mass_kg, dtype=np.float64)
                    mass_digest.update(memoryview(mass).cast("B"))
                fitted = fit_complex_amplitude(
                    block.displacement_m,
                    physical_times,
                    OMEGA_RAD_S,
                    t_ref_s=T_REF_S,
                )
                enforce_parent_budget(f"{window_name} harmonic fit")
                analytic = rayleigh_displacement(
                    block.positions_m,
                    DIRECTION,
                    WAVENUMBER_RAD_M,
                    VP_M_S,
                    VS_M_S,
                    amplitude_m=SURFACE_VERTICAL_AMPLITUDE_M,
                )
                for name, value in _three_observables(
                    block.positions_m,
                    block.mass_kg,
                    fitted,
                    targets,
                    kernel_chunk_size=kernel_chunk_size,
                ).items():
                    a_values[name] += value
                for name, value in _three_observables(
                    block.positions_m,
                    block.mass_kg,
                    analytic,
                    targets,
                    kernel_chunk_size=kernel_chunk_size,
                ).items():
                    b_values[name] += value
                enforce_parent_budget(f"{window_name} observable accumulation")
            if expected_source_start != provider.source_count:
                raise RuntimeError("provider omitted sources from the selected case")
            output_windows[window_name] = _window_result(
                contract, a_values, b_values, c_values
            )

        return {
            "study": "Wave 5 saved-Rayleigh three-observable A/B/C cross-path",
            "saved_case_identity": (
                "old saved 250 m case; distinct from the new S 12/18 km solver pair"
            ),
            "case_scope": {
                "label": (
                    "FULL_SAVED_CASE" if is_full else "SUBSET_ONLY_NOT_WHOLE_SAVED_CASE"
                ),
                "is_full_saved_case": is_full,
                "selected_element_range": [selected_start, selected_stop],
                "selected_element_count": selected_elements,
                "total_element_count": layout.n_elements,
            },
            "classification": {
                "A_solver_fitted": "Observed solver field, then inferred harmonic fit",
                "B_analytic_same_gll": (
                    "Inferred analytic field on observed GLL geometry and mass"
                ),
                "C_infinite_halfspace": "Inferred independent analytic reference",
                "imputed_values": [],
            },
            "configuration": {
                "frequency_hz": FREQUENCY_HZ,
                "t_ref_s": T_REF_S,
                "dt_s": DT_S,
                "density_kg_m3": RHO_KG_M3,
                "v_p_m_s": VP_M_S,
                "v_s_m_s": VS_M_S,
                "surface_vertical_amplitude_m": SURFACE_VERTICAL_AMPLITUDE_M,
                "wavelength_m": WAVELENGTH_M,
                "rayleigh_speed_m_s": RAYLEIGH_SPEED_M_S,
                "wavenumber_rad_m": WAVENUMBER_RAD_M,
                "direction": list(DIRECTION),
                "phase_origin_m": list(PHASE_ORIGIN_M),
                "kernel_chunk_size": kernel_chunk_size,
                "G_boundary": "public displacement observables apply G_SI exactly once",
                "comparison_policy": "metrics only; no 1 percent acceptance gate",
            },
            "targets_m": targets.tolist(),
            "fit_contract": windows["fit_contract"],
            "input_file": {
                **input_identity,
                "validated_hdf5_provenance": _json_ready(layout.provenance),
            },
            "source_files": source_files,
            "discrete_sources": {
                "source_count": provider.source_count,
                "gll_points_per_element": GLL_POINTS_PER_ELEMENT,
                "representation": (
                    "all element-local GLL rows retained; coincident coordinates "
                    "not merged"
                ),
                "first_source_id": provider.source_ids[0],
                "last_source_id": provider.source_ids[-1],
                "positions_float64_sha256": _sha256_array(provider.positions_m),
                "mass_float64_sha256": mass_digest.hexdigest(),
                "source_ids_utf8_length_prefixed_sha256": _sha256_source_ids(
                    provider.source_ids
                ),
            },
            "resource_estimate": resources,
            "windows": output_windows,
        }


def _rss_bytes() -> int:
    """Return this process's lifetime high-water RSS from ``ru_maxrss``."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _proc_status_memory_bytes(field: str) -> int:
    """Read a Linux ``/proc/self/status`` RSS field in bytes."""
    status_path = Path("/proc/self/status")
    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(
            "current/peak RSS preflight measurement requires /proc/self/status"
        ) from error
    for line in lines:
        if line.startswith(f"{field}:"):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB":
                break
            try:
                return int(fields[1]) * 1024
            except ValueError:
                break
    raise RuntimeError(f"/proc/self/status does not contain a valid {field} value")


def _current_rss_bytes() -> int:
    """Return current resident bytes, not the process-lifetime high-water mark."""
    return _proc_status_memory_bytes("VmRSS")


def _peak_rss_bytes() -> int:
    """Return the isolated process's interval high-water RSS from ``VmHWM``.

    The preflight workers use ``spawn``.  On Linux, ``ru_maxrss`` can retain
    the parent's lifetime high-water value in that child, while procfs
    ``VmHWM`` starts at the child's own process lifetime.  Reading ``VmHWM``
    after the reader/fit interval therefore rejects transient allocations
    without treating an unrelated parent peak as the worker's peak.
    """
    return _proc_status_memory_bytes("VmHWM")


def _write_json(path: Path, payload: dict[str, Any], *, exclusive: bool) -> None:
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(
    hdf5_path: str | Path,
    output_path: str | Path,
    *,
    element_start: int = 0,
    element_stop: int | None = None,
    source_block_elements: int = DEFAULT_SOURCE_BLOCK_ELEMENTS,
    kernel_chunk_size: int = DEFAULT_KERNEL_CHUNK_SIZE,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
    argv: list[str] | None = None,
) -> Path:
    """Write a create-new result, retaining ``.partial`` after any failure."""
    output = Path(output_path).expanduser().resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("research output must be outside the repository")
    partial = Path(f"{output}.partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")
    started_utc = datetime.now(UTC).isoformat()
    invocation = list(sys.argv if argv is None else argv)
    initial = {
        "status": "incomplete",
        "exit": None,
        "started_utc": started_utc,
        "argv": invocation,
        "input_path": str(Path(hdf5_path).expanduser().resolve()),
    }
    _write_json(partial, initial, exclusive=True)
    input_identity: dict[str, Any] | None = None
    source_files: dict[str, Any] | None = None
    memory_preflight: dict[str, Any] | None = None
    try:
        memory_budget_bytes = _positive_integer(
            memory_budget_bytes, "memory_budget_bytes"
        )
        probe = _metadata_resource_probe(
            hdf5_path,
            element_start=element_start,
            element_stop=element_stop,
            requested_block_elements=source_block_elements,
            memory_budget_bytes=memory_budget_bytes,
        )
        memory_preflight = _choose_source_block(
            probe,
            hdf5_path=hdf5_path,
            requested_block_elements=source_block_elements,
            memory_budget_bytes=memory_budget_bytes,
        )
        if not memory_preflight["passed"]:
            raise MemoryError("measured memory preflight cannot satisfy the RSS budget")
        effective_source_block_elements = memory_preflight[
            "effective_source_block_elements"
        ]
        input_identity = _input_file_identity(Path(hdf5_path))
        source_files = _source_file_identity()
        _write_json(
            partial,
            {
                **initial,
                "metadata_only_preflight_probe": probe,
                "measured_memory_preflight": memory_preflight,
                "input_file": input_identity,
                "source_files": source_files,
            },
            exclusive=False,
        )
        result = evaluate_saved_rayleigh(
            hdf5_path,
            element_start=element_start,
            element_stop=element_stop,
            source_block_elements=effective_source_block_elements,
            kernel_chunk_size=kernel_chunk_size,
            memory_budget_bytes=memory_budget_bytes,
            _known_input_identity=input_identity,
            _known_source_files=source_files,
        )
        result["resource_estimate"].update(
            {
                "cli_requested_source_block_elements": source_block_elements,
                "metadata_only_preflight_probe": probe,
                "measured_memory_preflight": memory_preflight,
            }
        )
        peak_rss_bytes = _rss_bytes()
        if peak_rss_bytes > memory_budget_bytes:
            raise MemoryError(
                "actual peak RSS exceeded memory budget: "
                f"{peak_rss_bytes} > {memory_budget_bytes} bytes"
            )
        payload = {
            "status": "complete",
            "exit": 0,
            "started_utc": started_utc,
            "completed_utc": datetime.now(UTC).isoformat(),
            "argv": invocation,
            "peak_rss_bytes": peak_rss_bytes,
            **result,
        }
        _write_json(partial, payload, exclusive=False)
        reopened = json.loads(partial.read_text(encoding="utf-8"))
        if reopened != payload:
            raise RuntimeError("serialized result differs on reopened readback")
        try:
            os.link(partial, output)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite {output}") from error
        partial.unlink()
        return output
    except Exception as error:
        failure = {
            **initial,
            "exit": 1,
            "failed_utc": datetime.now(UTC).isoformat(),
            "peak_rss_bytes": _rss_bytes(),
            "failure": {"type": type(error).__name__, "message": str(error)},
        }
        if input_identity is not None:
            failure["input_file"] = input_identity
        if source_files is not None:
            failure["source_files"] = source_files
        if memory_preflight is not None:
            failure["measured_memory_preflight"] = memory_preflight
        _write_json(partial, failure, exclusive=False)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--element-start", type=int, default=0)
    parser.add_argument("--element-stop", type=int)
    parser.add_argument(
        "--source-block-elements", type=int, default=DEFAULT_SOURCE_BLOCK_ELEMENTS
    )
    parser.add_argument(
        "--kernel-chunk-size", type=int, default=DEFAULT_KERNEL_CHUNK_SIZE
    )
    parser.add_argument(
        "--memory-budget-bytes", type=int, default=DEFAULT_MEMORY_BUDGET_BYTES
    )
    args = parser.parse_args()
    print(
        run(
            args.hdf5,
            args.output,
            element_start=args.element_start,
            element_stop=args.element_stop,
            source_block_elements=args.source_block_elements,
            kernel_chunk_size=args.kernel_chunk_size,
            memory_budget_bytes=args.memory_budget_bytes,
        )
    )


if __name__ == "__main__":
    main()
