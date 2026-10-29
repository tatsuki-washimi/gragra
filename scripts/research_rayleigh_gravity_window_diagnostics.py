#!/usr/bin/env python3
"""Stage-5 (5-D) spatial decomposition of the Rayleigh front/full gravity gap.

``docs/research/wave4-6-I.md`` (stage 4) found that the front/full harmonic
fit *of the raw displacement array* has a large residual (0.34/0.39) for the
saved 6 km solver waveform, but that this figure carries no mass, GLL
quadrature volume, or gravity-kernel weighting -- it is a displacement
fit-quality figure, not a gravity-domain error.

This module computes the actual gravity-domain front/full gap directly:

    Q_hat_front - Q_hat_full

for potential, acceleration, and gravity gradient, and decomposes it by
source region. ``fit_complex_amplitude`` (linear per source, over time) and
``displacement_transfer_{potential,acceleration,gravity_gradient}`` (linear
per time sample, over sources) act on independent axes of the same
``(time, source)`` displacement array, so they commute: masking sources into
disjoint regions before applying the same window-A path from
``research_wave5_rayleigh_crosspaths`` (fit per window, then apply the
gravity kernel) and summing the complex region contributions reproduces the
un-masked total exactly. That reproduction is the consistency check this
module performs before reporting which region dominates the gap.

Regions are defined by distance to the case's domain boundaries in
wavelength units, using the domain extents confirmed directly from the
input HDF5 (``/mesh/coordinates`` range): x,y in [0, 6000] m, z in
[-3000, 0] m for the saved 250 m case. Region assignment is mutually
exclusive by priority: lateral boundary first, then bottom, then free
surface, then interior.
"""

from __future__ import annotations

import argparse
import json
import os
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
from gragra.adapters.specfem.streaming import SpecfemGLLStreamReader  # noqa: E402
from gragra.observables.dipole import (  # noqa: E402
    displacement_transfer_acceleration,
    displacement_transfer_gravity_gradient,
    displacement_transfer_potential,
)
from gragra.research.rayleigh_timefit import fit_complex_amplitude  # noqa: E402
from scripts import research_wave5_rayleigh_crosspaths as study  # noqa: E402

if not Path(gragra.__file__).resolve().is_relative_to(ROOT / "src"):
    raise RuntimeError(
        "research script imported gragra outside its integration worktree"
    )

DEFAULT_SOURCE_BLOCK_ELEMENTS = 512
DEFAULT_MEMORY_BUDGET_BYTES = 2 * 1024**3
DEFAULT_KERNEL_CHUNK_SIZE = 1024

# Confirmed 2026-09-14 directly from /mesh/coordinates in
# /tmp/gragra-rayleigh-b5-250m-240.h5: x,y in [0, 6000] m, z in [-3000, 0] m.
SAVED_CASE_DOMAIN_BOUNDS_M = ((0.0, 6000.0), (0.0, 6000.0), (-3000.0, 0.0))

REGION_NAMES = (
    "lateral_boundary_near",
    "bottom_boundary_near",
    "surface_near",
    "interior",
)

_OBSERVABLE_FUNCTIONS = {
    "potential": displacement_transfer_potential,
    "acceleration": displacement_transfer_acceleration,
    "gradient": displacement_transfer_gravity_gradient,
}


def region_masks(
    positions_m: np.ndarray,
    *,
    domain_bounds_m: tuple[tuple[float, float], ...],
    boundary_distance_m: float,
) -> dict[str, np.ndarray]:
    """Partition sources into four mutually exclusive, exhaustive regions.

    Priority (a source matching an earlier rule is not reassigned by a later
    one): lateral boundary, bottom boundary, free surface, interior. This
    keeps the four boolean masks disjoint and their union the full set, which
    is required for the region-sum consistency check.
    """
    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_m must have shape (N, 3)")
    if boundary_distance_m <= 0.0 or not np.isfinite(boundary_distance_m):
        raise ValueError("boundary_distance_m must be positive and finite")
    (x0, x1), (y0, y1), (z0, z1) = domain_bounds_m
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    d_lateral = np.minimum(np.minimum(x - x0, x1 - x), np.minimum(y - y0, y1 - y))
    d_bottom = z - z0
    d_surface = z1 - z
    lateral = d_lateral < boundary_distance_m
    bottom = (~lateral) & (d_bottom < boundary_distance_m)
    surface = (~lateral) & (~bottom) & (d_surface < boundary_distance_m)
    interior = ~(lateral | bottom | surface)
    masks = {
        "lateral_boundary_near": lateral,
        "bottom_boundary_near": bottom,
        "surface_near": surface,
        "interior": interior,
    }
    stacked = np.stack(list(masks.values()))
    if not np.array_equal(stacked.sum(axis=0), np.ones(len(positions), dtype=np.int64)):
        raise RuntimeError("region masks are not a partition (overlap or gap found)")
    return masks


def _empty_observable_accumulators() -> dict[str, np.ndarray]:
    return {
        name: np.zeros((len(study.canonical_targets_m()), *tail), dtype=np.complex128)
        for name, tail in study._OBSERVABLE_SHAPES.items()
    }


def region_gravity_window_gap(
    hdf5_path: str | Path,
    *,
    boundary_distance_wavelengths: float = 1.0,
    domain_bounds_m: tuple[tuple[float, float], ...] = SAVED_CASE_DOMAIN_BOUNDS_M,
    element_start: int = 0,
    element_stop: int | None = None,
    source_block_elements: int = DEFAULT_SOURCE_BLOCK_ELEMENTS,
    kernel_chunk_size: int = DEFAULT_KERNEL_CHUNK_SIZE,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
) -> dict[str, Any]:
    """Compute the region-decomposed front/full gravity gap for path A.

    Reuses the exact window-A pattern from
    ``research_wave5_rayleigh_crosspaths`` (fit the solver displacement per
    window, then apply the G-free-then-G_SI-once gravity kernel), but
    accumulates the result separately per source region in addition to the
    unmasked total, so the two can be cross-checked against each other.
    """
    boundary_distance_m = float(boundary_distance_wavelengths) * study.WAVELENGTH_M
    targets = study.canonical_targets_m()
    element_slice = study._selected_element_slice(element_start, element_stop)
    slab_elements = (
        128 if element_slice is None else min(128, element_slice.stop - element_slice.start)
    )
    block_sources = source_block_elements * study.GLL_POINTS_PER_ELEMENT
    with SpecfemGLLStreamReader(
        hdf5_path,
        profile_density_range=(study.RHO_KG_M3, study.RHO_KG_M3),
        element_slice=element_slice,
        slab_elements=slab_elements,
        memory_budget_bytes=memory_budget_bytes,
    ) as reader:
        provider = reader.provider
        layout = reader._layout
        if layout is None:  # pragma: no cover - guarded by reader.provider
            raise RuntimeError("validated SPECFEM layout is unavailable")
        windows = study.rayleigh_windows(layout.time_s, layout.time_step)

        region_values: dict[str, dict[str, dict[str, np.ndarray]]] = {
            region: {"front": _empty_observable_accumulators(), "full": _empty_observable_accumulators()}
            for region in REGION_NAMES
        }
        total_values: dict[str, dict[str, np.ndarray]] = {
            "front": _empty_observable_accumulators(),
            "full": _empty_observable_accumulators(),
        }
        region_source_counts = {region: 0 for region in REGION_NAMES}

        for window_name in ("front", "full"):
            contract = windows[window_name]
            sample_indices = contract["input_sample_indices"]
            sample_slice = slice(sample_indices[0], sample_indices[-1] + 1)
            physical_times = np.asarray(contract["physical_times_s"], dtype=np.float64)
            for block in provider.blocks(sample_slice, block_sources):
                fitted = fit_complex_amplitude(
                    block.displacement_m,
                    physical_times,
                    study.OMEGA_RAD_S,
                    t_ref_s=study.T_REF_S,
                )
                masks = region_masks(
                    block.positions_m,
                    domain_bounds_m=domain_bounds_m,
                    boundary_distance_m=boundary_distance_m,
                )
                for name, fn in _OBSERVABLE_FUNCTIONS.items():
                    total_values[window_name][name] += fn(
                        block.positions_m,
                        block.mass_kg,
                        fitted,
                        targets,
                        chunk_size=kernel_chunk_size,
                    )
                for region_name, mask in masks.items():
                    if window_name == "front":
                        region_source_counts[region_name] += int(np.count_nonzero(mask))
                    if not np.any(mask):
                        continue
                    sub_positions = block.positions_m[mask]
                    sub_mass = block.mass_kg[mask]
                    sub_fitted = fitted[mask]
                    for name, fn in _OBSERVABLE_FUNCTIONS.items():
                        region_values[region_name][window_name][name] += fn(
                            sub_positions,
                            sub_mass,
                            sub_fitted,
                            targets,
                            chunk_size=kernel_chunk_size,
                        )

        def _gap(values: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
            return {
                name: values["front"][name] - values["full"][name]
                for name in _OBSERVABLE_FUNCTIONS
            }

        total_gap = _gap(total_values)
        region_gap = {region: _gap(region_values[region]) for region in REGION_NAMES}
        region_sum_gap = {
            name: sum(region_gap[region][name] for region in REGION_NAMES)
            for name in _OBSERVABLE_FUNCTIONS
        }
        consistency = {
            name: {
                "max_absolute_difference": float(
                    np.max(np.abs(region_sum_gap[name] - total_gap[name]))
                ),
                "max_total_norm": float(np.max(study._norm_by_target(total_gap[name], name))),
            }
            for name in _OBSERVABLE_FUNCTIONS
        }

        def _encode(values: dict[str, np.ndarray]) -> dict[str, Any]:
            return {name: study._complex_record(value) for name, value in values.items()}

        region_norm_share = {
            region: {
                name: (
                    study._norm_by_target(region_gap[region][name], name)
                    / np.maximum(
                        study._norm_by_target(total_gap[name], name), np.finfo(float).tiny
                    )
                ).tolist()
                for name in _OBSERVABLE_FUNCTIONS
            }
            for region in REGION_NAMES
        }

        return {
            "study": "Stage-5 (5-D) Rayleigh gravity front/full gap, spatial decomposition",
            "input_file": str(Path(hdf5_path).expanduser().resolve()),
            "boundary_distance_wavelengths": boundary_distance_wavelengths,
            "boundary_distance_m": boundary_distance_m,
            "domain_bounds_m": [list(axis) for axis in domain_bounds_m],
            "region_source_counts": region_source_counts,
            "targets_m": targets.tolist(),
            "total_gap_front_minus_full": _encode(total_gap),
            "region_gap_front_minus_full": {
                region: _encode(region_gap[region]) for region in REGION_NAMES
            },
            "region_sum_gap_front_minus_full": _encode(region_sum_gap),
            "consistency_check_region_sum_vs_total": consistency,
            "region_norm_share_of_total_gap": region_norm_share,
        }


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
    boundary_distance_wavelengths: float = 1.0,
    source_block_elements: int = DEFAULT_SOURCE_BLOCK_ELEMENTS,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
) -> Path:
    """Write a create-new result, retaining ``.partial`` on failure."""
    output = Path(output_path).expanduser().resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("research output must be outside the repository")
    partial = Path(f"{output}.partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")
    started_utc = datetime.now(UTC).isoformat()
    _write_json(partial, {"status": "incomplete", "started_utc": started_utc}, exclusive=True)
    try:
        result = region_gravity_window_gap(
            hdf5_path,
            boundary_distance_wavelengths=boundary_distance_wavelengths,
            source_block_elements=source_block_elements,
            memory_budget_bytes=memory_budget_bytes,
        )
        payload = {
            "status": "complete",
            "started_utc": started_utc,
            "completed_utc": datetime.now(UTC).isoformat(),
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
        _write_json(
            partial,
            {
                "status": "failed",
                "started_utc": started_utc,
                "failed_utc": datetime.now(UTC).isoformat(),
                "failure": {"type": type(error).__name__, "message": str(error)},
            },
            exclusive=False,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boundary-distance-wavelengths", type=float, default=1.0)
    parser.add_argument(
        "--source-block-elements", type=int, default=DEFAULT_SOURCE_BLOCK_ELEMENTS
    )
    parser.add_argument(
        "--memory-budget-bytes", type=int, default=DEFAULT_MEMORY_BUDGET_BYTES
    )
    args = parser.parse_args()
    print(
        run(
            args.hdf5,
            args.output,
            boundary_distance_wavelengths=args.boundary_distance_wavelengths,
            source_block_elements=args.source_block_elements,
            memory_budget_bytes=args.memory_budget_bytes,
        )
    )


if __name__ == "__main__":
    main()
