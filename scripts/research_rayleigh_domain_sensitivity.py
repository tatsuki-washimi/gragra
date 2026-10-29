#!/usr/bin/env python3
"""Evaluate bounded horizontal Rayleigh finite-box truncation sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Do not let an editable-install redirector select another worktree.
_SCIENCE_ROOT = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if type(finder).__name__ != "ScikitBuildRedirectingFinder"
]
sys.path.insert(0, str(_SCIENCE_ROOT / "src"))

from gragra.observables.dipole import (  # noqa: E402
    displacement_transfer_acceleration,
)
from gragra.waves.rayleigh import rayleigh_displacement  # noqa: E402

VP_M_S = 2800.0
VS_M_S = 1500.0
DENSITY_KG_M3 = 2300.0
FREQUENCY_HZ = 2.5
AMPLITUDE_M = 1.0e-9
RAYLEIGH_SPEED_M_S = 1390.8255605658555
WAVELENGTH_M = 556.3302242263422
WAVENUMBER_RAD_M = 2.0 * np.pi / WAVELENGTH_M
CENTER_XY_M = (3000.0, 3000.0)
DEPTH_M = 3000.0
ELEMENT_SIZE_M = 250.0
NGLL = 5
LENGTHS_M = (6000.0, 9000.0, 12000.0, 15000.0, 18000.0)
TARGETS_M = np.array(
    [
        [3000.0, 3000.0, 0.25 * WAVELENGTH_M],
        [3000.0, 3000.0, 0.50 * WAVELENGTH_M],
        [3000.0, 3000.0, 0.75 * WAVELENGTH_M],
    ],
    dtype=np.float64,
)


class BoxSpec:
    """Structured element box with retained element-local GLL rows."""

    __slots__ = ("length_m", "depth_m", "element_size_m", "center_xy_m", "ngll")

    def __init__(
        self,
        length_m: float,
        depth_m: float,
        element_size_m: float,
        center_xy_m: tuple[float, float] = CENTER_XY_M,
        ngll: int = NGLL,
    ) -> None:
        values = np.asarray(
            [length_m, depth_m, element_size_m, *center_xy_m], dtype=np.float64
        )
        if not np.all(np.isfinite(values)) or np.any(values[:3] <= 0.0):
            raise ValueError(
                "box dimensions, spacing, and center must be finite and positive"
            )
        if ngll < 2:
            raise ValueError("ngll must be at least 2")
        for extent in (length_m, depth_m):
            count = extent / element_size_m
            if not math.isclose(count, round(count), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    "box extents must be integer multiples of element_size_m"
                )
        self.length_m = float(length_m)
        self.depth_m = float(depth_m)
        self.element_size_m = float(element_size_m)
        self.center_xy_m = (float(center_xy_m[0]), float(center_xy_m[1]))
        self.ngll = int(ngll)

    @property
    def element_shape(self) -> tuple[int, int, int]:
        horizontal = round(self.length_m / self.element_size_m)
        vertical = round(self.depth_m / self.element_size_m)
        return horizontal, horizontal, vertical

    @property
    def element_count(self) -> int:
        return math.prod(self.element_shape)

    @property
    def source_count(self) -> int:
        return self.element_count * self.ngll**3

    @property
    def retained_shared_node_representations(self) -> int:
        nx, ny, nz = self.element_shape
        unique = (
            (nx * (self.ngll - 1) + 1)
            * (ny * (self.ngll - 1) + 1)
            * (nz * (self.ngll - 1) + 1)
        )
        return self.source_count - unique


class IntegrationResult(NamedTuple):
    values: np.ndarray
    source_count: int
    nonfinite_count: int
    peak_rss_bytes: int
    elapsed_s: float


def _gll_nodes_weights(ngll: int) -> tuple[np.ndarray, np.ndarray]:
    polynomial = np.polynomial.legendre.Legendre.basis(ngll - 1)
    interior = polynomial.deriv().roots()
    nodes = np.concatenate(([-1.0], interior, [1.0])).astype(np.float64)
    weights = 2.0 / (ngll * (ngll - 1) * polynomial(nodes) ** 2)
    return nodes, weights


def _element_origins(box: BoxSpec, start: int, stop: int) -> np.ndarray:
    nx, ny, nz = box.element_shape
    flat = np.arange(start, stop, dtype=np.int64)
    ix, iy, iz = np.unravel_index(flat, (nx, ny, nz), order="C")
    x0 = box.center_xy_m[0] - box.length_m / 2.0
    y0 = box.center_xy_m[1] - box.length_m / 2.0
    return np.column_stack(
        (
            x0 + ix * box.element_size_m,
            y0 + iy * box.element_size_m,
            -box.depth_m + iz * box.element_size_m,
        )
    )


def _element_points(origins: np.ndarray, box: BoxSpec) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights_1d = _gll_nodes_weights(box.ngll)
    local = (nodes + 1.0) * (box.element_size_m / 2.0)
    gx, gy, gz = np.meshgrid(local, local, local, indexing="ij")
    local_xyz = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
    wx, wy, wz = np.meshgrid(weights_1d, weights_1d, weights_1d, indexing="ij")
    jacobian = (box.element_size_m / 2.0) ** 3
    local_weights = (wx * wy * wz).ravel() * jacobian
    coordinates = (origins[:, None, :] + local_xyz[None, :, :]).reshape(-1, 3)
    quadrature = (
        np.broadcast_to(local_weights, (origins.shape[0], local_weights.size))
        .reshape(-1)
        .copy()
    )
    return coordinates, quadrature


def iter_box_blocks(box: BoxSpec, max_sources: int = 256000):
    """Yield complete element blocks; coincident shared nodes stay duplicated."""
    sources_per_element = box.ngll**3
    elements_per_block = max_sources // sources_per_element
    if elements_per_block < 1:
        raise ValueError(
            f"max_sources must hold one element ({sources_per_element} sources)"
        )
    for start in range(0, box.element_count, elements_per_block):
        stop = min(start + elements_per_block, box.element_count)
        yield _element_points(_element_origins(box, start, stop), box)


def compare_nested_overlap(
    inner: BoxSpec, outer: BoxSpec, *, max_sources: int = 256000
) -> dict[str, object]:
    """Verify that every inner element uses identical GLL positions/weights."""
    if (
        inner.depth_m != outer.depth_m
        or inner.element_size_m != outer.element_size_m
        or inner.ngll != outer.ngll
        or inner.center_xy_m != outer.center_xy_m
        or inner.length_m > outer.length_m
    ):
        raise ValueError(
            "boxes must be centered, nested, and use identical depth/quadrature"
        )
    offset_elements = (outer.length_m - inner.length_m) / (2.0 * inner.element_size_m)
    if not offset_elements.is_integer():
        raise ValueError("horizontal box boundaries must align to the element grid")
    coordinate_match = True
    weight_match = True
    sources = 0
    for inner_coordinates, inner_weights in iter_box_blocks(
        inner, max_sources=max_sources
    ):
        # The outer mesh evaluates the same mathematical element rule at every
        # overlapping origin; reconstructing it independently catches changes
        # in coordinates or quadrature without materializing the outer box.
        element_count = inner_coordinates.shape[0] // inner.ngll**3
        reshaped = inner_coordinates.reshape(element_count, inner.ngll**3, 3)
        origins = reshaped[:, 0, :]
        outer_coordinates, outer_weights = _element_points(origins, outer)
        coordinate_match &= np.array_equal(inner_coordinates, outer_coordinates)
        weight_match &= np.array_equal(inner_weights, outer_weights)
        sources += inner_coordinates.shape[0]
    return {
        "coordinate_exact_match": bool(coordinate_match),
        "quadrature_weight_exact_match": bool(weight_match),
        "overlap_element_count": inner.element_count,
        "overlap_source_count": sources,
        "shared_nodes_deduplicated": False,
    }


def integrate_box(
    box: BoxSpec,
    *,
    targets: np.ndarray = TARGETS_M,
    source_block: int = 256000,
    kernel_chunk: int = 65536,
) -> IntegrationResult:
    """Integrate the direct analytic displacement phasor over one box."""
    started = time.perf_counter()
    total = np.zeros((len(targets), 3), dtype=np.complex128)
    nonfinite = 0
    source_count = 0
    for coordinates, quadrature in iter_box_blocks(box, max_sources=source_block):
        displacement = rayleigh_displacement(
            coordinates,
            (1.0, 0.0, 0.0),
            WAVENUMBER_RAD_M,
            VP_M_S,
            VS_M_S,
            amplitude_m=AMPLITUDE_M,
        )
        contribution = displacement_transfer_acceleration(
            coordinates,
            DENSITY_KG_M3 * quadrature,
            displacement,
            targets,
            chunk_size=kernel_chunk,
        )
        nonfinite += int(np.count_nonzero(~np.isfinite(contribution)))
        total += contribution
        source_count += coordinates.shape[0]
    total.setflags(write=False)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return IntegrationResult(
        total,
        source_count,
        nonfinite,
        int(peak_rss),
        time.perf_counter() - started,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_saved_solver_results(
    a6_path: str | Path,
    a12_path: str | Path,
    *,
    analytic_reference: np.ndarray | None = None,
) -> dict[str, object]:
    """Record saved A(6 km)/A(12 km) provenance without reinterpreting it."""

    def summarize(path_like: str | Path) -> dict[str, object]:
        path = Path(path_like).resolve()
        payload = json.loads(path.read_text())
        rows = payload.get("results", [])
        target_rows = [row for row in rows if "target_xyz_m" in row]
        target_match = None
        if target_rows:
            observed_targets = {
                tuple(np.asarray(row["target_xyz_m"], dtype=np.float64))
                for row in target_rows
            }
            expected_targets = {tuple(row) for row in TARGETS_M}
            target_match = observed_targets == expected_targets
        material = payload.get("material")
        rayleigh = payload.get("rayleigh")
        explicit_physics_match = None
        if material is not None and rayleigh is not None:
            explicit_physics_match = all(
                (
                    material.get("vp_m_s") == VP_M_S,
                    material.get("vs_m_s") == VS_M_S,
                    material.get("rho_kg_m3") == DENSITY_KG_M3,
                    rayleigh.get("f_hz") == FREQUENCY_HZ,
                    rayleigh.get("amplitude_m") == AMPLITUDE_M,
                    rayleigh.get("lambda_R_m") == WAVELENGTH_M,
                )
            )
        time_match = None
        if "dt_s" in payload and "t_init_s" in payload:
            time_match = (
                payload["dt_s"] == 0.005
                and payload["t_init_s"] == 10.0
                and payload.get("time_rule")
                == "t_physical = t_init + (time_step + 1)*DT"
            )
        reference_match = None
        reference_max_relative_difference = None
        reference_rows = [
            row for row in target_rows if "C_analytic_infinite_halfspace" in row
        ]
        if analytic_reference is not None and reference_rows:
            reference_match = True
            relative_differences = []
            for row in reference_rows:
                target = np.asarray(row["target_xyz_m"], dtype=np.float64)
                matches = np.nonzero(np.all(target == TARGETS_M, axis=1))[0]
                if matches.size != 1:
                    reference_match = False
                    break
                saved = _decode_complex_vectors([row["C_analytic_infinite_halfspace"]])[
                    0
                ]
                current = analytic_reference[int(matches[0])]
                relative_differences.append(
                    float(np.linalg.norm(saved - current) / np.linalg.norm(current))
                )
                reference_match &= np.allclose(saved, current, rtol=1e-14, atol=0.0)
            if relative_differences:
                reference_max_relative_difference = max(relative_differences)
        return {
            "path": str(path),
            "sha256": _sha256(path),
            "source_hdf5": payload.get("source_hdf5"),
            "result_count": len(rows),
            "conditions_checked": {
                "target_coordinates_match": target_match,
                "explicit_material_rayleigh_metadata_match": explicit_physics_match,
                "explicit_time_metadata_match": time_match,
                "saved_C_matches_current_analytic_reference_rtol_1e-14": (
                    reference_match
                ),
                "saved_C_maximum_relative_difference": (
                    reference_max_relative_difference
                ),
                "null_means_metadata_not_recorded": True,
            },
        }

    return {
        "classification": "Observed saved-result reference",
        "a6": summarize(a6_path),
        "a12": summarize(a12_path),
        "not_a_solver_convergence_test": True,
        "revision_and_configuration_confounds": [
            "The saved JSON files do not record an exact solver commit.",
            "A6 and A12 name different source HDF5 products; the raw /tmp files "
            "were not reopened in W1.",
            "A12 is a hook-run product while A6 is the earlier 6 km product, so "
            "their difference mixes horizontal extent with revision/configuration.",
            "The saved A fields are retained only as observed references, not as "
            "a solver convergence comparison.",
        ],
    }


def _load_b6_reference() -> np.ndarray:
    path = Path(__file__).with_name("research_rayleigh_b6.py")
    spec = importlib.util.spec_from_file_location("_w1_existing_rayleigh_b6", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load existing B6 calculation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    expected = {
        "VP": VP_M_S,
        "VS": VS_M_S,
        "RHO": DENSITY_KG_M3,
        "FREQUENCY_HZ": FREQUENCY_HZ,
        "AMPLITUDE_M": AMPLITUDE_M,
        "LAMBDA_R": WAVELENGTH_M,
    }
    for name, value in expected.items():
        if not math.isclose(float(getattr(module, name)), value, rel_tol=1e-14):
            raise ValueError(f"existing B6 {name} no longer matches W1 conditions")
    if not np.array_equal(module.TARGETS, TARGETS_M):
        raise ValueError("existing B6 targets no longer match W1 targets")
    return np.asarray(module._analytic_rayleigh_gravity(), dtype=np.complex128)


def _complex_vectors(values: np.ndarray) -> list[list[list[float]]]:
    return [
        [[float(value.real), float(value.imag)] for value in row]
        for row in np.asarray(values, dtype=np.complex128)
    ]


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(right))
    if denominator == 0.0:
        raise ValueError("relative L2 reference must be nonzero")
    return float(np.linalg.norm(left - right) / denominator)


def targetwise_relative_l2(left: np.ndarray, right: np.ndarray) -> list[float]:
    """Return one vector relative error per target, each with its own scale."""
    return targetwise_difference_over_reference(left, right, right)


def targetwise_difference_over_reference(
    left: np.ndarray, right: np.ndarray, reference: np.ndarray
) -> list[float]:
    """Return ``||left-right||`` per target, scaled by each reference row."""
    left = np.asarray(left)
    right = np.asarray(right)
    reference = np.asarray(reference)
    if left.shape != right.shape or left.shape != reference.shape or left.ndim != 2:
        raise ValueError(
            "left, right, and reference must share (targets, components) shape"
        )
    denominators = np.linalg.norm(reference, axis=1)
    if np.any(denominators == 0.0):
        raise ValueError("each target reference vector must be nonzero")
    return [
        float(value) for value in np.linalg.norm(left - right, axis=1) / denominators
    ]


def _decode_complex_vectors(values: list[list[list[float]]]) -> np.ndarray:
    pairs = np.asarray(values, dtype=np.float64)
    if pairs.ndim != 3 or pairs.shape[-1] != 2:
        raise ValueError(
            "complex vectors must use (target, component, real/imag) layout"
        )
    return pairs[..., 0] + 1j * pairs[..., 1]


def augment_targetwise_metrics(report: dict[str, object]) -> dict[str, object]:
    """Add per-height metrics to an existing result without recomputing B(L)."""
    analytic = _decode_complex_vectors(report["C_infinite_halfspace_m_s2"])
    previous: np.ndarray | None = None
    for row in report["boxes"]:
        current = _decode_complex_vectors(row["B_finite_box_m_s2"])
        row["B_minus_C_relative_l2_by_target"] = targetwise_relative_l2(
            current, analytic
        )
        adjacent = row["adjacent_B_difference"]
        if adjacent is not None:
            assert previous is not None
            adjacent["l2_norm_m_s2_by_target"] = [
                float(value) for value in np.linalg.norm(current - previous, axis=1)
            ]
            adjacent["relative_to_current_l2_by_target"] = targetwise_relative_l2(
                previous, current
            )
            adjacent["relative_to_C_l2_by_target"] = (
                targetwise_difference_over_reference(current, previous, analytic)
            )
        previous = current
    report["primary_diagnostic"] = (
        "Per-target adjacent B(L) complex-vector differences, with each relative "
        "value normalized by that target's C vector."
    )
    return report


def run_analysis(
    *,
    a6_path: Path,
    a12_path: Path,
    source_block: int,
    kernel_chunk: int,
) -> dict[str, object]:
    if source_block > 256000:
        raise ValueError("source_block must be <= 256000")
    if kernel_chunk > 65536:
        raise ValueError("kernel_chunk must be <= 65536")
    analytic = _load_b6_reference()
    rows = []
    previous_values: np.ndarray | None = None
    previous_length: float | None = None
    for length in LENGTHS_M:
        box = BoxSpec(length, DEPTH_M, ELEMENT_SIZE_M)
        result = integrate_box(
            box,
            targets=TARGETS_M,
            source_block=source_block,
            kernel_chunk=kernel_chunk,
        )
        adjacent = None
        if previous_values is not None:
            adjacent = {
                "from_length_m": previous_length,
                "complex_vector_difference": _complex_vectors(
                    result.values - previous_values
                ),
                "l2_norm_m_s2": float(np.linalg.norm(result.values - previous_values)),
                "relative_to_current_l2": _relative_l2(previous_values, result.values),
            }
        rows.append(
            {
                "length_m": length,
                "element_shape": list(box.element_shape),
                "element_count": box.element_count,
                "source_count": result.source_count,
                "B_finite_box_m_s2": _complex_vectors(result.values),
                "B_minus_C_relative_l2": _relative_l2(result.values, analytic),
                "adjacent_B_difference": adjacent,
                "nonfinite_count": result.nonfinite_count,
                "elapsed_s": result.elapsed_s,
                "peak_rss_bytes": result.peak_rss_bytes,
                "output_array_bytes": result.values.nbytes,
            }
        )
        previous_values = result.values
        previous_length = length
    overlap = [
        {
            "inner_length_m": inner,
            "outer_length_m": outer,
            **compare_nested_overlap(
                BoxSpec(inner, DEPTH_M, ELEMENT_SIZE_M),
                BoxSpec(outer, DEPTH_M, ELEMENT_SIZE_M),
                max_sources=source_block,
            ),
        }
        for inner, outer in zip(LENGTHS_M[:-1], LENGTHS_M[1:], strict=True)
    ]
    report = {
        "classification": "Observed deterministic analytic calculation",
        "scope": (
            "Horizontal finite-box truncation sensitivity at fixed 3000 m depth, "
            "250 m elements, and NGLL=5. B-C also contains fixed-depth truncation "
            "and quadrature error; adjacent B differences are the primary diagnostic."
        ),
        "not_a_solver_stability_or_convergence_test": True,
        "conditions": {
            "lengths_m": list(LENGTHS_M),
            "center_xy_m": list(CENTER_XY_M),
            "z_range_m": [-DEPTH_M, 0.0],
            "element_size_m": ELEMENT_SIZE_M,
            "ngll": NGLL,
            "rho_kg_m3": DENSITY_KG_M3,
            "vp_m_s": VP_M_S,
            "vs_m_s": VS_M_S,
            "frequency_hz": FREQUENCY_HZ,
            "amplitude_m": AMPLITUDE_M,
            "wavelength_m": WAVELENGTH_M,
            "targets_m": TARGETS_M.tolist(),
            "source_block": source_block,
            "kernel_chunk": kernel_chunk,
            "shared_gll_nodes_deduplicated": False,
        },
        "C_infinite_halfspace_m_s2": _complex_vectors(analytic),
        "boxes": rows,
        "nested_overlap_checks": overlap,
        "saved_solver_reference": summarize_saved_solver_results(
            a6_path, a12_path, analytic_reference=analytic
        ),
    }
    return augment_targetwise_metrics(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a6-json", type=Path, required=True)
    parser.add_argument("--a12-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--source-block", type=int, default=256000)
    parser.add_argument("--kernel-chunk", type=int, default=65536)
    parser.add_argument("--minimum-free-gib", type=float, default=10.0)
    args = parser.parse_args()
    free_bytes = shutil.disk_usage(args.output_json.parent).free
    if free_bytes < args.minimum_free_gib * 1024**3:
        raise RuntimeError(
            f"free disk {free_bytes / 1024**3:.3f} GiB is below required "
            f"{args.minimum_free_gib:.3f} GiB"
        )
    result = run_analysis(
        a6_path=args.a6_json,
        a12_path=args.a12_json,
        source_block=args.source_block,
        kernel_chunk=args.kernel_chunk,
    )
    result["free_disk_bytes_before_write"] = free_bytes
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
