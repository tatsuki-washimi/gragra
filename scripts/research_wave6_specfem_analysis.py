#!/usr/bin/env python3
"""Bounded two-window A/B/C analysis for the matched SPECFEM domain pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from gragra.adapters.specfem import schema
from gragra.adapters.specfem.streaming import SpecfemGLLStreamReader
from gragra.observables.dipole import displacement_transfer_acceleration
from gragra.waves.rayleigh import rayleigh_displacement
from scripts.research_wave4_specfem_protocol import (
    CASE_NAMES,
    DEFAULT_RUNS_ROOT,
    validate_matched_actual_dt,
)

VP_M_S = 2800.0
VS_M_S = 1500.0
FREQUENCY_HZ = 2.5
OMEGA_RAD_S = 15.707963267948966
WAVENUMBER_RAD_M = 0.01129398517924721
AMPLITUDE_M = 1.0e-9
T_REF_S = 10.0
DIRECTION = np.array([1.0, 0.0, 0.0], dtype=np.float64)
TARGET_LABELS = ("0.25_lambda", "0.5_lambda", "0.75_lambda")
TARGETS_M = np.array(
    [
        [3000.0, 3000.0, 139.08255605658556],
        [3000.0, 3000.0, 278.1651121131711],
        [3000.0, 3000.0, 417.2476681697567],
    ],
    dtype=np.float64,
)
FIXED_C_M_S2 = np.array(
    [
        [
            complex(-1.0007377423002946e-16, -1.2488145963543198e-16),
            complex(-0.0, 0.0),
            complex(1.2488145963543198e-16, -1.0007377423002946e-16),
        ],
        [
            complex(-2.0803293790760316e-17, -2.5960304923078368e-17),
            complex(-0.0, 0.0),
            complex(2.5960304923078368e-17, -2.0803293790760316e-17),
        ],
        [
            complex(-4.324579899923688e-18, -5.3966171893461266e-18),
            complex(-0.0, 0.0),
            complex(5.3966171893461266e-18, -4.324579899923688e-18),
        ],
    ],
    dtype=np.complex128,
)
FIXED_NORM_C_M_S2 = np.array(
    [
        2.2631897511838873e-16,
        4.704709266774927e-17,
        9.78012969230675e-18,
    ],
    dtype=np.float64,
)
WINDOW_ENDS = (160, 240)
SOURCE_BLOCK_ELEMENTS = 512
NGLL_PER_ELEMENT = 125
KERNEL_CHUNK_SOURCES = 65_536
MEMORY_BUDGET_BYTES = 2 * 1024**3


def physical_times(native_steps: np.ndarray, actual_dt_s: float) -> np.ndarray:
    """Construct physical time exactly once from one-based native steps."""
    steps = np.asarray(native_steps)
    if steps.ndim != 1 or steps.dtype.kind not in "iu":
        raise TypeError("native_steps must be a one-dimensional integer array")
    dt = np.float64(actual_dt_s)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("actual_dt_s must be finite and positive")
    result = np.float64(T_REF_S) + steps.astype(np.float64) * dt
    result.setflags(write=False)
    return result


def fit_two_windows(
    values: np.ndarray,
    native_steps: np.ndarray,
    *,
    actual_dt_s: float,
    window_ends: Sequence[int] = WINDOW_ENDS,
) -> dict[int, np.ndarray]:
    """Fit ``coef0 + i*coef1`` with the protocol's two exact matrices."""
    y = np.asarray(values)
    steps = np.asarray(native_steps)
    ends = tuple(int(value) for value in window_ends)
    if y.dtype.kind not in "iuf" or np.iscomplexobj(y):
        raise TypeError("time-domain values must be real numeric data")
    y = np.asarray(y, dtype=np.float64)
    if y.ndim < 1 or y.shape[0] != len(steps):
        raise ValueError("values axis 0 must match native_steps")
    if not ends or tuple(sorted(set(ends))) != ends or ends[0] < 3:
        raise ValueError("window_ends must be unique, increasing, and at least 3")
    if ends[-1] > len(steps):
        raise ValueError("largest fit window exceeds the available samples")
    expected_steps = np.arange(1, len(steps) + 1, dtype=steps.dtype)
    if not np.array_equal(steps, expected_steps):
        raise ValueError("native steps must be exactly ordered one-based samples")
    if not np.isfinite(y).all():
        raise ValueError("time-domain values must be finite")
    times = physical_times(steps, actual_dt_s)
    fitted: dict[int, np.ndarray] = {}
    for stop in ends:
        theta = np.float64(OMEGA_RAD_S) * (times[:stop] - np.float64(T_REF_S))
        design = np.column_stack((np.cos(theta), np.sin(theta)))
        if np.linalg.matrix_rank(design) != 2:
            raise ValueError(f"fit window 1..{stop} has a rank-deficient design")
        coefficients, *_ = np.linalg.lstsq(
            design, y[:stop].reshape(stop, -1), rcond=None
        )
        amplitude = np.asarray(
            (coefficients[0] + 1j * coefficients[1]).reshape(y.shape[1:]),
            dtype=np.complex128,
        )
        amplitude.setflags(write=False)
        fitted[stop] = amplitude
    return fitted


def _analytic_displacement(positions_m: np.ndarray) -> np.ndarray:
    return rayleigh_displacement(
        positions_m,
        DIRECTION,
        WAVENUMBER_RAD_M,
        VP_M_S,
        VS_M_S,
        amplitude_m=AMPLITUDE_M,
    )


def _window_result(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    norm_c: np.ndarray,
) -> dict[str, np.ndarray]:
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    c = np.asarray(c, dtype=np.complex128)
    differences = {
        "A_minus_B": a - b,
        "B_minus_C": b - c,
        "A_minus_C": a - c,
    }
    result: dict[str, np.ndarray] = {"A": a, "B": b, "C": c, **differences}
    for name, difference in differences.items():
        norms = np.linalg.norm(difference, axis=1).astype(np.float64)
        result[f"norm_{name}"] = norms
        result[f"normalized_norm_{name}"] = norms / norm_c
    return result


def analyze_case(
    canonical_path: str | Path,
    *,
    length_m: int,
    actual_dt_s: float,
    targets_m: np.ndarray = TARGETS_M,
    fixed_c: np.ndarray = FIXED_C_M_S2,
    fixed_norm_c: np.ndarray = FIXED_NORM_C_M_S2,
    window_ends: Sequence[int] = WINDOW_ENDS,
    source_block_elements: int = SOURCE_BLOCK_ELEMENTS,
    max_blocks: int | None = None,
) -> dict[str, Any]:
    """Fit A and integrate A/B in bounded element blocks via lane-L reader."""
    import h5py

    path = Path(canonical_path).resolve()
    targets = np.asarray(targets_m, dtype=np.float64)
    c = np.asarray(fixed_c, dtype=np.complex128)
    norm_c = np.asarray(fixed_norm_c, dtype=np.float64)
    ends = tuple(int(value) for value in window_ends)
    if targets.ndim != 2 or targets.shape[1:] != (3,):
        raise ValueError("targets_m must have shape (target,3)")
    if c.shape != targets.shape or norm_c.shape != (len(targets),):
        raise ValueError("fixed C and norm(C) must match the target axis")
    if not np.isfinite(targets).all() or not np.isfinite(c).all():
        raise ValueError("targets and fixed C must be finite")
    if not np.isfinite(norm_c).all() or np.any(norm_c <= 0.0):
        raise ValueError("fixed norm(C) values must be finite and positive")
    if source_block_elements <= 0:
        raise ValueError("source_block_elements must be positive")
    if max_blocks is not None and max_blocks <= 0:
        raise ValueError("max_blocks must be positive when supplied")
    with h5py.File(path, "r") as h5:
        time_step = np.asarray(h5[schema.PATH_TIME_STEP][: ends[-1]])
        total_samples = int(h5[schema.PATH_FIELD_DISPLACEMENT].shape[0])
    expected_zero_based = np.arange(ends[-1], dtype=np.int64)
    if total_samples < ends[-1] or not np.array_equal(time_step, expected_zero_based):
        raise ValueError(
            "canonical time_step must cover ordered native steps 1..window"
        )
    native_steps = expected_zero_based + 1
    source_block_size = source_block_elements * NGLL_PER_ELEMENT
    a_totals = {stop: np.zeros((len(targets), 3), dtype=np.complex128) for stop in ends}
    b_total = np.zeros((len(targets), 3), dtype=np.complex128)
    block_count = 0
    processed_sources = 0
    analytic_diagnostic: dict[str, Any] | None = None
    started = time.perf_counter()
    with SpecfemGLLStreamReader(
        path,
        profile_density_range=None,
        sample_slice=slice(0, ends[-1]),
        slab_elements=source_block_elements,
        memory_budget_bytes=MEMORY_BUDGET_BYTES,
    ) as reader:
        provider = reader.provider
        for block in provider.blocks(slice(0, ends[-1]), source_block_size):
            if block.source_start != processed_sources:
                raise ValueError("lane-L provider source order is not contiguous")
            if len(block.source_ids) % NGLL_PER_ELEMENT:
                raise ValueError("lane-L provider split an element-local GLL group")
            fitted = fit_two_windows(
                block.displacement_m,
                native_steps,
                actual_dt_s=actual_dt_s,
                window_ends=ends,
            )
            analytic = _analytic_displacement(block.positions_m)
            for stop in ends:
                a_totals[stop] += displacement_transfer_acceleration(
                    block.positions_m,
                    block.mass_kg,
                    fitted[stop],
                    targets,
                    chunk_size=KERNEL_CHUNK_SOURCES,
                )
            b_total += displacement_transfer_acceleration(
                block.positions_m,
                block.mass_kg,
                analytic,
                targets,
                chunk_size=KERNEL_CHUNK_SOURCES,
            )
            if analytic_diagnostic is None:
                diagnostic_sources = min(NGLL_PER_ELEMENT, len(analytic))
                times = physical_times(native_steps, actual_dt_s)
                samples = np.real(
                    analytic[None, :diagnostic_sources, :]
                    * np.exp(
                        -1j
                        * np.float64(OMEGA_RAD_S)
                        * (times - np.float64(T_REF_S))[:, None, None]
                    )
                )
                fitted_analytic = fit_two_windows(
                    samples,
                    native_steps,
                    actual_dt_s=actual_dt_s,
                    window_ends=ends,
                )
                analytic_diagnostic = {
                    "representative_sources": diagnostic_sources,
                    "components_checked": 3,
                    "max_abs_error_by_window": {
                        stop: float(
                            np.max(
                                np.abs(
                                    fitted_analytic[stop]
                                    - analytic[:diagnostic_sources]
                                )
                            )
                        )
                        for stop in ends
                    },
                    "design_matrix_shared_with_A": True,
                }
            processed_sources += len(block.source_ids)
            block_count += 1
            if max_blocks is not None and block_count >= max_blocks:
                break
        reader_resident_bytes = reader.estimated_resident_bytes
        full_source_count = provider.source_count
    if processed_sources == 0:
        raise ValueError("lane-L provider yielded no sources")
    full_case = processed_sources == full_source_count
    if max_blocks is None and not full_case:
        raise ValueError("lane-L provider did not cover the full source set")
    windows = {
        stop: _window_result(a_totals[stop], b_total, c, norm_c) for stop in ends
    }
    return {
        "classification": "Observed bounded SPECFEM two-window analysis",
        "status": (
            "CASE_ANALYSIS_COMPLETE" if full_case else "REPRESENTATIVE_BLOCK_COMPLETE"
        ),
        "canonical_path": str(path),
        "length_m": int(length_m),
        "reader_class": "SpecfemGLLStreamReader",
        "actual_dt_s": float(actual_dt_s),
        "physical_time_rule": "10 + native_it * actual_DT",
        "native_step_origin": 1,
        "window_ends_inclusive": list(ends),
        "source_block_elements": source_block_elements,
        "source_block_size": source_block_size,
        "source_count": full_source_count,
        "processed_source_count": processed_sources,
        "processed_element_count": processed_sources // NGLL_PER_ELEMENT,
        "block_count": block_count,
        "full_case": full_case,
        "shared_nodes_deduplicated": False,
        "reader_estimated_resident_bytes": reader_resident_bytes,
        "process_peak_rss_bytes_cumulative": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "wall_seconds_including_reader_and_fit": time.perf_counter() - started,
        "analytic_fit_diagnostic": analytic_diagnostic,
        "targets_m": targets,
        "target_labels": list(TARGET_LABELS),
        "fixed_norm_C_m_s2": norm_c,
        "windows": windows,
        "one_percent_gate": False,
        "domain_convergence_claim": False,
    }


def _element_keys(
    h5: Any, *, block_elements: int
) -> tuple[list[tuple[float, ...]], int]:
    coordinates = h5[schema.PATH_MESH_COORDINATES]
    keys: list[tuple[float, ...]] = []
    for start in range(0, coordinates.shape[0], block_elements):
        stop = min(start + block_elements, coordinates.shape[0])
        slab = np.asarray(coordinates[start:stop])
        minima = np.min(slab, axis=1)
        keys.extend(tuple(float(value) for value in row) for row in minima)
    return keys, len(keys) - len(set(keys))


def _indexed_rows(dataset: Any, indices: Sequence[int]) -> np.ndarray:
    order = np.argsort(np.asarray(indices, dtype=np.int64))
    sorted_indices = np.asarray(indices, dtype=np.int64)[order]
    sorted_values = np.asarray(dataset[sorted_indices])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return sorted_values[inverse]


def _hash_array(digest: Any, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))


def _bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare dtype, shape, and every stored byte, including signed zero."""
    if left.shape != right.shape or left.dtype.str != right.dtype.str:
        return False
    left_bytes = np.ascontiguousarray(left).view(np.uint8)
    right_bytes = np.ascontiguousarray(right).view(np.uint8)
    return bool(np.array_equal(left_bytes, right_bytes))


def _element_bitwise_equal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return a byte-identity mask along the leading element axis."""
    if left.shape != right.shape or left.dtype.str != right.dtype.str:
        return np.zeros(left.shape[0], dtype=np.bool_)
    left_bytes = np.ascontiguousarray(left).view(np.uint8).reshape(left.shape[0], -1)
    right_bytes = np.ascontiguousarray(right).view(np.uint8).reshape(right.shape[0], -1)
    return np.all(left_bytes == right_bytes, axis=1)


def compare_nested_overlap(
    inner_path: str | Path,
    outer_path: str | Path,
    *,
    block_elements: int = SOURCE_BLOCK_ELEMENTS,
    targets_m: np.ndarray = TARGETS_M,
    analytic_displacement: Callable[[np.ndarray], np.ndarray] = _analytic_displacement,
) -> dict[str, Any]:
    """Require exact nested static fields and re-add B elements identically."""
    import h5py

    if block_elements <= 0:
        raise ValueError("block_elements must be positive")
    inner_path = Path(inner_path).resolve()
    outer_path = Path(outer_path).resolve()
    targets = np.asarray(targets_m, dtype=np.float64)
    result: dict[str, Any] = {
        "classification": "Observed conservative nested-overlap comparison",
        "inner_path": str(inner_path),
        "outer_path": str(outer_path),
        "coordinate_and_weight_comparison": "bitwise",
        "shared_nodes_deduplicated": False,
        "block_elements": block_elements,
        "domain_size_difference_interpretable": False,
    }
    with h5py.File(inner_path, "r") as inner, h5py.File(outer_path, "r") as outer:
        inner_keys, inner_duplicates = _element_keys(
            inner, block_elements=block_elements
        )
        outer_keys, outer_duplicates = _element_keys(
            outer, block_elements=block_elements
        )
        outer_map = {key: index for index, key in enumerate(outer_keys)}
        missing = [key for key in inner_keys if key not in outer_map]
        bounds_min = np.min(np.asarray(inner_keys), axis=0)
        bounds_max = np.max(np.asarray(inner_keys), axis=0)
        inner_key_set = set(inner_keys)
        outer_in_overlap = {
            key
            for key in outer_keys
            if np.all(np.asarray(key) >= bounds_min)
            and np.all(np.asarray(key) <= bounds_max)
        }
        extras = outer_in_overlap - inner_key_set
        result.update(
            {
                "overlap_element_count": len(inner_keys),
                "missing_key_count": len(missing),
                "duplicate_key_count": inner_duplicates + outer_duplicates,
                "extra_overlap_key_count": len(extras),
            }
        )
        if missing or inner_duplicates or outer_duplicates or extras:
            result.update(
                {
                    "status": "NESTED_OVERLAP_BITWISE_REJECTED",
                    "all_static_fields_bitwise_equal": False,
                    "b_element_contributions_bitwise_equal": False,
                    "b_element_contributions_status": "NOT_EVALUATED",
                    "b_element_contributions_evaluated_element_count": 0,
                    "b_overlap_inner_same_order": None,
                    "b_overlap_outer_same_order": None,
                    "first_mismatch_key": list(
                        missing[0] if missing else next(iter(extras), inner_keys[0])
                    ),
                }
            )
            return result

        fields = (
            ("coordinates", schema.PATH_MESH_COORDINATES),
            ("quadrature_volume", schema.PATH_MESH_QUADRATURE_VOLUME),
            ("material_id", schema.PATH_MESH_MATERIAL_ID),
            ("domain_id", schema.PATH_MESH_DOMAIN_ID),
            ("density", schema.PATH_MATERIAL_DENSITY),
            ("is_pml", schema.PATH_MESH_IS_PML),
        )
        exact_by_field = {name: True for name, _ in fields}
        inner_field_digests = {name: hashlib.sha256() for name, _ in fields}
        outer_field_digests = {name: hashlib.sha256() for name, _ in fields}
        max_coordinate = 0.0
        max_quadrature = 0.0
        first_mismatch: tuple[float, ...] | None = None
        inner_b = np.zeros((len(targets), 3), dtype=np.complex128)
        outer_b = np.zeros((len(targets), 3), dtype=np.complex128)
        inner_digest = hashlib.sha256()
        outer_digest = hashlib.sha256()
        contributions_equal = True
        contribution_elements_evaluated = 0
        for start in range(0, len(inner_keys), block_elements):
            stop = min(start + block_elements, len(inner_keys))
            outer_indices = [outer_map[key] for key in inner_keys[start:stop]]
            inner_values: dict[str, np.ndarray] = {}
            outer_values: dict[str, np.ndarray] = {}
            for name, path in fields:
                left = np.asarray(inner[path][start:stop])
                right = _indexed_rows(outer[path], outer_indices)
                inner_values[name] = left
                outer_values[name] = right
                _hash_array(inner_field_digests[name], left)
                _hash_array(outer_field_digests[name], right)
                equal = _bitwise_equal(left, right)
                exact_by_field[name] &= equal
                if not equal and first_mismatch is None:
                    element_equal = _element_bitwise_equal(left, right)
                    local = int(np.nonzero(~element_equal)[0][0])
                    first_mismatch = inner_keys[start + local]
                if name == "coordinates":
                    max_coordinate = max(
                        max_coordinate, float(np.max(np.abs(left - right)))
                    )
                elif name == "quadrature_volume":
                    max_quadrature = max(
                        max_quadrature, float(np.max(np.abs(left - right)))
                    )
            if not all(exact_by_field.values()):
                continue
            for local in range(stop - start):
                left_positions = inner_values["coordinates"][local]
                right_positions = outer_values["coordinates"][local]
                left_mass = (
                    inner_values["density"][local]
                    * inner_values["quadrature_volume"][local]
                )
                right_mass = (
                    outer_values["density"][local]
                    * outer_values["quadrature_volume"][local]
                )
                left_contribution = displacement_transfer_acceleration(
                    left_positions,
                    left_mass,
                    analytic_displacement(left_positions),
                    targets,
                    chunk_size=KERNEL_CHUNK_SOURCES,
                )
                right_contribution = displacement_transfer_acceleration(
                    right_positions,
                    right_mass,
                    analytic_displacement(right_positions),
                    targets,
                    chunk_size=KERNEL_CHUNK_SOURCES,
                )
                contributions_equal &= np.array_equal(
                    left_contribution, right_contribution
                )
                _hash_array(inner_digest, left_contribution)
                _hash_array(outer_digest, right_contribution)
                inner_b += left_contribution
                outer_b += right_contribution
                contribution_elements_evaluated += 1
        all_static = all(exact_by_field.values())
        field_identity_sha256 = {
            name: {
                "inner": inner_field_digests[name].hexdigest(),
                "outer": outer_field_digests[name].hexdigest(),
            }
            for name, _ in fields
        }
        complete_contribution_coverage = (
            all_static and contribution_elements_evaluated == len(inner_keys)
        )
        contribution_hash_equal = (
            complete_contribution_coverage
            and inner_digest.digest() == outer_digest.digest()
        )
        contributions_equal &= (
            contribution_hash_equal
            and complete_contribution_coverage
            and np.array_equal(inner_b, outer_b)
        )
        accepted = all_static and contributions_equal
        contribution_status = (
            "COMPLETE"
            if complete_contribution_coverage
            else (
                "NOT_EVALUATED"
                if contribution_elements_evaluated == 0
                else "INCOMPLETE"
            )
        )
        result.update(
            {
                "status": (
                    "NESTED_OVERLAP_BITWISE_ACCEPTED"
                    if accepted
                    else "NESTED_OVERLAP_BITWISE_REJECTED"
                ),
                "exact_by_field": exact_by_field,
                "field_identity_sha256": field_identity_sha256,
                "all_static_fields_bitwise_equal": all_static,
                "b_element_contributions_bitwise_equal": bool(contributions_equal),
                "b_element_contributions_status": contribution_status,
                "b_element_contributions_evaluated_element_count": (
                    contribution_elements_evaluated
                ),
                "b_element_contributions_expected_element_count": len(inner_keys),
                "b_element_contribution_inner_sha256": (
                    inner_digest.hexdigest() if complete_contribution_coverage else None
                ),
                "b_element_contribution_outer_sha256": (
                    outer_digest.hexdigest() if complete_contribution_coverage else None
                ),
                "b_element_contribution_partial_inner_sha256": (
                    inner_digest.hexdigest()
                    if not complete_contribution_coverage
                    else None
                ),
                "b_element_contribution_partial_outer_sha256": (
                    outer_digest.hexdigest()
                    if not complete_contribution_coverage
                    else None
                ),
                "b_overlap_inner_same_order": (
                    inner_b if complete_contribution_coverage else None
                ),
                "b_overlap_outer_same_order": (
                    outer_b if complete_contribution_coverage else None
                ),
                "b_overlap_inner_same_order_partial": (
                    inner_b
                    if 0 < contribution_elements_evaluated < len(inner_keys)
                    else None
                ),
                "b_overlap_outer_same_order_partial": (
                    outer_b
                    if 0 < contribution_elements_evaluated < len(inner_keys)
                    else None
                ),
                "first_mismatch_key": (
                    None if first_mismatch is None else list(first_mismatch)
                ),
                "max_abs_coordinate_difference": max_coordinate,
                "max_abs_quadrature_volume_difference": max_quadrature,
                "domain_size_difference_interpretable": accepted,
                "same_element_addition_order": True,
            }
        )
    return result


def _complex_from_pairs(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape[-1] != 2:
        raise ValueError("complex JSON values must end in [real,imag]")
    return np.asarray(array[..., 0] + 1j * array[..., 1], dtype=np.complex128)


def _has_exact_passing_rss_budget_record(result: dict[str, Any]) -> bool:
    """Accept only the exact JSON integer budget record emitted by this lane."""
    budget = result.get("rss_budget_bytes")
    peak = result.get("process_peak_rss_bytes_cumulative")
    return (
        result.get("rss_budget_passed") is True
        and type(budget) is int
        and budget == MEMORY_BUDGET_BYTES
        and type(peak) is int
        and 0 <= peak <= MEMORY_BUDGET_BYTES
    )


def pair_comparison(
    result_12: dict[str, Any],
    result_18: dict[str, Any],
    overlap: dict[str, Any],
) -> dict[str, Any]:
    """Build signed adjacent-domain vectors without a tolerance gate."""
    expected_source_counts = {
        12_000: 48 * 48 * 12 * NGLL_PER_ELEMENT,
        18_000: 72 * 72 * 12 * NGLL_PER_ELEMENT,
    }
    for expected_length, result in ((12_000, result_12), (18_000, result_18)):
        if result.get("status") != "CASE_ANALYSIS_COMPLETE" or not result.get(
            "full_case", False
        ):
            raise ValueError(
                f"L{expected_length} result is not a complete full-case analysis"
            )
        if result.get("length_m") != expected_length:
            raise ValueError(
                f"expected L{expected_length} result, "
                f"got length_m={result.get('length_m')!r}"
            )
        if result.get("processed_source_count") != result.get("source_count"):
            raise ValueError(f"L{expected_length} result has partial source coverage")
        if result.get("source_count") != expected_source_counts[expected_length]:
            raise ValueError(f"L{expected_length} result has unexpected source count")
        if not _has_exact_passing_rss_budget_record(result):
            raise ValueError(
                f"L{expected_length} result lacks a passing final RSS check"
            )
        if result.get("window_ends_inclusive") != list(WINDOW_ENDS):
            raise ValueError(f"L{expected_length} result has wrong fit windows")
        if set(result.get("windows", {})) != {str(stop) for stop in WINDOW_ENDS}:
            raise ValueError(f"L{expected_length} result window set is incomplete")
        if result.get("target_labels") != list(TARGET_LABELS):
            raise ValueError(f"L{expected_length} result target labels do not match")
        targets = np.asarray(result.get("targets_m"), dtype=np.float64)
        norm_c = np.asarray(result.get("fixed_norm_C_m_s2"), dtype=np.float64)
        if not _bitwise_equal(targets, TARGETS_M):
            raise ValueError(f"L{expected_length} target coordinates do not match")
        if not _bitwise_equal(norm_c, FIXED_NORM_C_M_S2):
            raise ValueError(f"L{expected_length} norm(C) does not match")
        for stop in WINDOW_ENDS:
            window = result["windows"][str(stop)]
            for name in ("A", "B", "C"):
                value = _complex_from_pairs(window[name])
                if value.shape != FIXED_C_M_S2.shape or not np.isfinite(value).all():
                    raise ValueError(
                        f"L{expected_length} window {stop} {name} is malformed"
                    )
            c_pairs = np.asarray(window["C"], dtype=np.float64)
            expected_c_pairs = np.stack((FIXED_C_M_S2.real, FIXED_C_M_S2.imag), axis=-1)
            if not _bitwise_equal(c_pairs, expected_c_pairs):
                raise ValueError(f"L{expected_length} window {stop} C does not match")
    dt_12 = result_12.get("actual_dt_s")
    dt_18 = result_18.get("actual_dt_s")
    if (
        isinstance(dt_12, bool)
        or isinstance(dt_18, bool)
        or not isinstance(dt_12, (int, float))
        or not isinstance(dt_18, (int, float))
        or not np.isfinite(dt_12)
        or not np.isfinite(dt_18)
        or dt_12 <= 0.0
        or dt_12 != dt_18
    ):
        raise ValueError("case result actual_DT values do not match exactly")
    rows = {}
    for stop in WINDOW_ENDS:
        key = str(stop)
        left = result_12["windows"][key]
        right = result_18["windows"][key]
        a_difference = _complex_from_pairs(right["A"]) - _complex_from_pairs(left["A"])
        b_difference = _complex_from_pairs(right["B"]) - _complex_from_pairs(left["B"])
        rows[key] = {
            "A_L18000_minus_A_L12000": a_difference,
            "B_L18000_minus_B_L12000": b_difference,
            "norm_A_L18000_minus_A_L12000": np.linalg.norm(a_difference, axis=1),
            "norm_B_L18000_minus_B_L12000": np.linalg.norm(b_difference, axis=1),
            "normalized_norm_A_L18000_minus_A_L12000": np.linalg.norm(
                a_difference, axis=1
            )
            / FIXED_NORM_C_M_S2,
            "normalized_norm_B_L18000_minus_B_L12000": np.linalg.norm(
                b_difference, axis=1
            )
            / FIXED_NORM_C_M_S2,
        }
    return {
        "classification": "Observed matched finite-domain comparison",
        "status": "PAIR_COMPARISON_COMPLETE",
        "actual_dt_s": result_12["actual_dt_s"],
        "physical_time_rule": "10 + native_it * actual_DT",
        "windows": rows,
        "nested_overlap": overlap,
        "domain_size_difference_interpretable": overlap[
            "domain_size_difference_interpretable"
        ],
        "one_percent_gate": False,
        "domain_convergence_claim": False,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return np.stack((value.real, value.imag), axis=-1).tolist()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_no_overwrite(path: Path, value: Any) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    failure = path.with_suffix(path.suffix + ".partial.failure.json")
    if path.exists() or partial.exists() or failure.exists():
        raise FileExistsError(f"refusing to overwrite analysis artifact for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = (
            json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        )
        with partial.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        observed_payload = partial.read_text(encoding="utf-8")
        if observed_payload != payload or json.loads(observed_payload) != json.loads(
            payload
        ):
            raise ValueError("analysis JSON reopened readback mismatch")
        os.link(partial, path)
        partial.unlink()
    except BaseException as error:
        if not failure.exists():
            with failure.open("x", encoding="utf-8") as stream:
                json.dump(
                    {
                        "classification": "Observed incomplete analysis output",
                        "status": "ANALYSIS_OUTPUT_INCOMPLETE_RETAINED",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "partial": str(partial),
                    },
                    stream,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                stream.write("\n")
        raise


def _matched_actual_dt(runs_root: Path) -> float:
    reports = []
    for length_m in (12_000, 18_000):
        path = runs_root / CASE_NAMES[length_m] / "solver-acceptance.json"
        value = json.loads(path.read_text())
        reports.append(value["log"])
    return float(validate_matched_actual_dt(*reports)["actual_dt_s"])


def _canonical_path(runs_root: Path, length_m: int) -> Path:
    return runs_root / CASE_NAMES[length_m] / "canonical" / f"specfem-L{length_m}.h5"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _execution_manifest(
    *,
    argv: Sequence[str],
    started_utc: str,
    input_paths: Sequence[Path],
    exit_status: int,
) -> dict[str, Any]:
    source = Path(__file__).resolve()
    return {
        "classification": "Observed analysis execution provenance",
        "argv": [str(source), *(str(value) for value in argv)],
        "started_utc": started_utc,
        "completed_utc": datetime.now(UTC).isoformat(),
        "exit_status": exit_status,
        "source": {"path": str(source), "sha256": _file_sha256(source)},
        "inputs": [
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in input_paths
        ],
    }


def _validate_rss_preflight(
    report: dict[str, Any], *, length_m: int, source_block_elements: int
) -> None:
    if report.get("status") != "RSS_PREFLIGHT_ACCEPTED":
        raise ValueError("full analysis requires an accepted RSS preflight")
    if report.get("length_m") != length_m:
        raise ValueError("RSS preflight length does not match full analysis")
    if report.get("source_block_elements") != source_block_elements:
        raise ValueError("RSS preflight block size does not match full analysis")
    if report.get("processed_element_count") != source_block_elements:
        raise ValueError("RSS preflight did not process exactly one requested block")
    if not _has_exact_passing_rss_budget_record(report):
        raise ValueError("RSS preflight lacks an exact passing 2 GiB budget record")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("rss-preflight", "analyze"):
        item = sub.add_parser(command)
        item.add_argument("length_m", type=int, choices=(12_000, 18_000))
        item.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
        item.add_argument("--output-json", type=Path, required=True)
        item.add_argument(
            "--source-block-elements", type=int, default=SOURCE_BLOCK_ELEMENTS
        )
        item.add_argument("--canonical-path", type=Path, default=None)
        if command == "analyze":
            item.add_argument("--rss-preflight-json", type=Path, required=True)
    pair = sub.add_parser("pair")
    pair.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    pair.add_argument("--case-12-json", type=Path, required=True)
    pair.add_argument("--case-18-json", type=Path, required=True)
    pair.add_argument("--output-json", type=Path, required=True)
    pair.add_argument("--canonical-12", type=Path, default=None)
    pair.add_argument("--canonical-18", type=Path, default=None)
    return parser


def _execute_command(
    args: argparse.Namespace, *, command_argv: Sequence[str], started_utc: str
) -> int:
    if args.command in {"rss-preflight", "analyze"}:
        actual_dt = _matched_actual_dt(args.runs_root)
        canonical_path = args.canonical_path or _canonical_path(
            args.runs_root, args.length_m
        )
        if args.command == "analyze":
            rss_report = json.loads(args.rss_preflight_json.read_text())
            _validate_rss_preflight(
                rss_report,
                length_m=args.length_m,
                source_block_elements=args.source_block_elements,
            )
        result = analyze_case(
            canonical_path,
            length_m=args.length_m,
            actual_dt_s=actual_dt,
            source_block_elements=args.source_block_elements,
            max_blocks=1 if args.command == "rss-preflight" else None,
        )
        peak_rss = result["process_peak_rss_bytes_cumulative"]
        budget_passed = peak_rss <= MEMORY_BUDGET_BYTES
        result["rss_budget_bytes"] = MEMORY_BUDGET_BYTES
        result["rss_budget_passed"] = budget_passed
        if args.command == "rss-preflight":
            exact_block = (
                result["processed_element_count"] == args.source_block_elements
            )
            result["status"] = (
                "RSS_PREFLIGHT_ACCEPTED"
                if exact_block and budget_passed
                else "RSS_PREFLIGHT_REJECTED_SHRINK_BLOCK_AND_RETRY"
            )
            exit_status = 0 if exact_block and budget_passed else 2
        else:
            result["status"] = (
                "CASE_ANALYSIS_COMPLETE"
                if budget_passed
                else "FULL_ANALYSIS_RSS_BUDGET_REJECTED"
            )
            exit_status = 0 if budget_passed else 2
        inputs = [canonical_path]
        if args.command == "analyze":
            inputs.append(args.rss_preflight_json)
        result["execution"] = _execution_manifest(
            argv=command_argv,
            started_utc=started_utc,
            input_paths=inputs,
            exit_status=exit_status,
        )
        _write_json_no_overwrite(args.output_json, result)
    else:
        result_12 = json.loads(args.case_12_json.read_text())
        result_18 = json.loads(args.case_18_json.read_text())
        canonical_12 = args.canonical_12 or _canonical_path(args.runs_root, 12_000)
        canonical_18 = args.canonical_18 or _canonical_path(args.runs_root, 18_000)
        overlap = compare_nested_overlap(
            canonical_12,
            canonical_18,
        )
        result = pair_comparison(result_12, result_18, overlap)
        result["execution"] = _execution_manifest(
            argv=command_argv,
            started_utc=started_utc,
            input_paths=(
                args.case_12_json,
                args.case_18_json,
                canonical_12,
                canonical_18,
            ),
            exit_status=0,
        )
        _write_json_no_overwrite(args.output_json, result)
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return exit_status if args.command in {"rss-preflight", "analyze"} else 0


def _input_paths_for_args(args: argparse.Namespace) -> list[Path]:
    if args.command in {"rss-preflight", "analyze"}:
        paths = [args.canonical_path or _canonical_path(args.runs_root, args.length_m)]
        if args.command == "analyze":
            paths.append(args.rss_preflight_json)
        return paths
    return [
        args.case_12_json,
        args.case_18_json,
        args.canonical_12 or _canonical_path(args.runs_root, 12_000),
        args.canonical_18 or _canonical_path(args.runs_root, 18_000),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    command_argv = list(sys.argv[1:] if argv is None else argv)
    started_utc = datetime.now(UTC).isoformat()
    args = _parser().parse_args(command_argv)
    try:
        return _execute_command(
            args, command_argv=command_argv, started_utc=started_utc
        )
    except BaseException as error:
        failure_path = args.output_json.with_suffix(
            args.output_json.suffix + ".execution.failure.json"
        )
        if not failure_path.exists():
            existing_inputs = [
                path for path in _input_paths_for_args(args) if path.exists()
            ]
            failure_report = {
                "classification": "Observed failed analysis execution",
                "status": "ANALYSIS_EXECUTION_FAILED_RETAINED",
                "error_type": type(error).__name__,
                "error": str(error),
                "execution": _execution_manifest(
                    argv=command_argv,
                    started_utc=started_utc,
                    input_paths=existing_inputs,
                    exit_status=1,
                ),
            }
            _write_json_no_overwrite(failure_path, failure_report)
        raise


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    raise SystemExit(main())
