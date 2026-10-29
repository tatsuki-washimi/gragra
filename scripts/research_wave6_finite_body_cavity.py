#!/usr/bin/env python3
"""Wave 6 prescribed P-wave response of a finite host containing a cavity."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
import types
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from gragra.constants import G_SI
from gragra.fields.finite_body_p1 import (
    compare_tetra_p1_density_representations,
    piecewise_tetra_p1_density_sources,
)
from gragra.fields.unstructured import UnstructuredDisplacementField
from gragra.integrators.point_direct import direct_point_displacement_sum
from scripts.research_wave4_finite_body import (
    CAVITY_TARGETS_M,
    HOST_BOUNDS_M,
    HOST_DENSITY_KG_M3,
    INNER_BOUNDS_M,
    OBSERVABLES,
    _array_record,
    a_oracle_status,
    build_cartesian_six_tetra_mesh,
    evaluate_static_observables_g_free,
    normalized_error,
    response_norm,
)

VP_M_S = 2800.0
AMPLITUDE_M = 1e-9
WAVELENGTHS_M = (20.0, 40.0, 80.0)


def prescribed_p_wave_displacement(
    points_m, *, wavelength_m: float, amplitude_m: float = AMPLITUDE_M
) -> np.ndarray:
    """Return +x P-wave phasor ``amplitude * exp(i k x) e_x``."""
    points = np.asarray(points_m, dtype=np.float64)
    wavelength = float(wavelength_m)
    amplitude = float(amplitude_m)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or wavelength <= 0.0
        or not np.isfinite(wavelength)
        or not np.isfinite(amplitude)
    ):
        raise ValueError(
            "finite (N,3) points, positive wavelength, and finite amplitude required"
        )
    result = np.zeros((len(points), 3), dtype=np.complex128)
    result[:, 0] = amplitude * np.exp(1j * (2.0 * np.pi / wavelength) * points[:, 0])
    result.setflags(write=False)
    return result


def _scaled_host(scale: float) -> np.ndarray:
    bounds = HOST_BOUNDS_M.copy()
    bounds[:2] *= scale
    bounds[2, 0] *= scale
    return bounds


def wave6_case_matrix() -> list[dict]:
    cases = []
    for wavelength in WAVELENGTHS_M:
        for h in (4.0, 2.0, 1.0):
            cases.append(
                {
                    "case_id": f"mesh-lambda{wavelength:g}-h{h:g}",
                    "sweep": "mesh",
                    "wavelength_m": wavelength,
                    "h_m": h,
                    "extent_scale": 1.0,
                    "amplitude_m": AMPLITUDE_M,
                    "vp_m_s": VP_M_S,
                }
            )
        for scale in (1.5, 2.0):
            cases.append(
                {
                    "case_id": f"extent-lambda{wavelength:g}-s{scale:g}",
                    "sweep": "extent",
                    "wavelength_m": wavelength,
                    "h_m": 4.0,
                    "extent_scale": scale,
                    "amplitude_m": AMPLITUDE_M,
                    "vp_m_s": VP_M_S,
                }
            )
    return cases


def _subcell_gauss_unit_interval(
    subcells: int, order: int
) -> tuple[np.ndarray, np.ndarray]:
    if subcells < 1 or order < 1:
        raise ValueError("subcells and order must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    all_nodes = []
    all_weights = []
    for index in range(subcells):
        lo = index / subcells
        hi = (index + 1) / subcells
        all_nodes.append(0.5 * (hi - lo) * nodes + 0.5 * (hi + lo))
        all_weights.append(0.5 * (hi - lo) * weights)
    return np.concatenate(all_nodes), np.concatenate(all_weights)


def _duffy_tetra_rule(subcells: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = _subcell_gauss_unit_interval(subcells, order)
    a, b, c = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    wa, wb, wc = np.meshgrid(weights, weights, weights, indexing="ij")
    barycentric = np.stack(
        (
            (1.0 - a) * (1.0 - b) * (1.0 - c),
            a,
            (1.0 - a) * b,
            (1.0 - a) * (1.0 - b) * c,
        ),
        axis=-1,
    ).reshape(-1, 4)
    rule_weights = (wa * wb * wc * (1.0 - a) ** 2 * (1.0 - b)).reshape(-1)
    return barycentric, rule_weights


def _duffy_triangle_rule(subcells: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = _subcell_gauss_unit_interval(subcells, order)
    a, b = np.meshgrid(nodes, nodes, indexing="ij")
    wa, wb = np.meshgrid(weights, weights, indexing="ij")
    barycentric = np.stack(((1.0 - a) * (1.0 - b), a, (1.0 - a) * b), axis=-1).reshape(
        -1, 3
    )
    rule_weights = (wa * wb * (1.0 - a)).reshape(-1)
    return barycentric, rule_weights


def _mutable_observables(target_count: int, dtype) -> dict[str, np.ndarray]:
    return {
        "potential": np.zeros(target_count, dtype=dtype),
        "acceleration": np.zeros((target_count, 3), dtype=dtype),
        "gradient": np.zeros((target_count, 3, 3), dtype=dtype),
    }


def _decode_array_record(record: dict) -> np.ndarray:
    values = np.asarray(record["values"], dtype=np.float64)
    if record["encoding"] == "real_imag_last_axis":
        return np.asarray(values[..., 0] + 1j * values[..., 1], dtype=np.complex128)
    if record["encoding"] != "plain":
        raise ValueError(f"unknown array encoding {record['encoding']!r}")
    return values


def add_fixed_quadrature_error_curves(study: dict) -> None:
    """Derive fixed-field errors relative to the stored finest quadrature."""
    curves = study["curves"]
    for name in OBSERVABLES:
        finest_mass = _decode_array_record(
            curves[-1]["observables"][name]["mass_displacement_physical"]
        )
        finest_combined = _decode_array_record(
            curves[-1]["observables"][name]["bulk_plus_interface_physical"]
        )
        for curve in curves:
            observable = curve["observables"][name]
            mass = _decode_array_record(observable["mass_displacement_physical"])
            combined = _decode_array_record(observable["bulk_plus_interface_physical"])
            observable["mass_vs_finest_normalized_error"] = normalized_error(
                mass, finest_mass, name
            ).tolist()
            observable["bulk_interface_vs_finest_normalized_error"] = normalized_error(
                combined, finest_combined, name
            ).tolist()


def _accumulate_static(
    accumulator: dict[str, np.ndarray], positions, weights, targets
) -> None:
    value = evaluate_static_observables_g_free(positions, weights, targets)
    for name in OBSERVABLES:
        accumulator[name] += getattr(value, name)


def fixed_p1_subcell_quadrature_study(
    mesh,
    nodal_displacement_m,
    targets_m,
    *,
    subcells=(1, 2, 4),
    gauss_order: int = 2,
    tetra_batch_size: int = 128,
    face_batch_size: int = 256,
) -> dict:
    """Refine only independent quadrature on one frozen P1 tetra field.

    Duffy-coordinate subcells vary the integration points while connectivity,
    nodal displacement, material interfaces, and interpolation remain fixed.
    This isolates quadrature from the separate h mesh/interpolation sweep.
    """
    displacement = np.asarray(nodal_displacement_m)
    dtype = np.complex128 if np.iscomplexobj(displacement) else np.float64
    if (
        displacement.shape != mesh.vertices_m.shape
        or not np.isfinite(displacement).all()
    ):
        raise ValueError("nodal_displacement_m must be finite and match vertices")
    targets = np.asarray(targets_m, dtype=np.float64)
    field = UnstructuredDisplacementField(mesh.vertices_m, mesh.tetra, displacement)
    sources = piecewise_tetra_p1_density_sources(
        field, np.repeat(mesh.cell_density_kg_m3[:, None], 4, axis=1)
    )
    tetra_nodes = mesh.vertices_m[mesh.tetra]
    tetra_u = displacement[mesh.tetra]
    edges = np.stack(
        (
            tetra_nodes[:, 1] - tetra_nodes[:, 0],
            tetra_nodes[:, 2] - tetra_nodes[:, 0],
            tetra_nodes[:, 3] - tetra_nodes[:, 0],
        ),
        axis=-1,
    )
    determinant = np.abs(np.linalg.det(edges))
    inverse = np.linalg.inv(edges)
    displacement_edges = tetra_u[:, 1:] - tetra_u[:, :1]
    divergence = np.einsum("nki,nki->n", displacement_edges, inverse)
    faces = sources.interface_triangles
    face_nodes = mesh.vertices_m[faces]
    face_u = displacement[faces]
    cross = np.cross(
        face_nodes[:, 1] - face_nodes[:, 0], face_nodes[:, 2] - face_nodes[:, 0]
    )
    cross_norm = np.linalg.norm(cross, axis=1)
    normals = cross / cross_norm[:, None]
    jumps = np.mean(sources.interface_density_jump_kg_m3, axis=1)
    curves = []
    for subcell_count in subcells:
        tetra_bary, tetra_weights = _duffy_tetra_rule(int(subcell_count), gauss_order)
        triangle_bary, triangle_weights = _duffy_triangle_rule(
            int(subcell_count), gauss_order
        )
        mass_accumulator = _mutable_observables(len(targets), dtype)
        bulk_accumulator = _mutable_observables(len(targets), dtype)
        interface_accumulator = _mutable_observables(len(targets), dtype)
        for start in range(0, len(mesh.tetra), tetra_batch_size):
            stop = min(start + tetra_batch_size, len(mesh.tetra))
            position = np.einsum(
                "ql,nli->nqi", tetra_bary, tetra_nodes[start:stop]
            ).reshape(-1, 3)
            u_value = np.einsum("ql,nli->nqi", tetra_bary, tetra_u[start:stop]).reshape(
                -1, 3
            )
            volume_weight = (
                determinant[start:stop, None] * tetra_weights[None, :]
            ).reshape(-1)
            rho = np.repeat(mesh.cell_density_kg_m3[start:stop], len(tetra_weights))
            mass = rho * volume_weight
            for name in OBSERVABLES:
                mass_accumulator[name] += direct_point_displacement_sum(
                    mass,
                    u_value,
                    position,
                    targets,
                    observable=name,
                    chunk_size=8192,
                    backend="numpy",
                )
            bulk_density = np.repeat(
                -mesh.cell_density_kg_m3[start:stop] * divergence[start:stop],
                len(tetra_weights),
            )
            _accumulate_static(
                bulk_accumulator, position, bulk_density * volume_weight, targets
            )
        for start in range(0, len(faces), face_batch_size):
            stop = min(start + face_batch_size, len(faces))
            position = np.einsum(
                "ql,fli->fqi", triangle_bary, face_nodes[start:stop]
            ).reshape(-1, 3)
            u_value = np.einsum("ql,fli->fqi", triangle_bary, face_u[start:stop])
            normal_u = np.einsum("fqi,fi->fq", u_value, normals[start:stop])
            weights = (
                jumps[start:stop, None]
                * normal_u
                * cross_norm[start:stop, None]
                * triangle_weights[None, :]
            ).reshape(-1)
            _accumulate_static(interface_accumulator, position, weights, targets)
        observables = {}
        for name in OBSERVABLES:
            combined = bulk_accumulator[name] + interface_accumulator[name]
            denominator = response_norm(mass_accumulator[name], name)
            nonzero_scale = max(float(np.max(denominator)), np.finfo(float).tiny)
            residual = response_norm(
                combined - mass_accumulator[name], name
            ) / np.maximum(denominator, np.finfo(float).eps * nonzero_scale)
            observables[name] = {
                "mass_displacement_physical": _array_record(
                    G_SI * mass_accumulator[name]
                ),
                "bulk_physical": _array_record(G_SI * bulk_accumulator[name]),
                "interface_physical": _array_record(G_SI * interface_accumulator[name]),
                "bulk_plus_interface_physical": _array_record(G_SI * combined),
                "representation_normalized_residual": residual.tolist(),
            }
        curves.append(
            {
                "subcells_per_duffy_axis": int(subcell_count),
                "gauss_order_per_subcell_axis": int(gauss_order),
                "tetra_points_per_cell": int(len(tetra_weights)),
                "triangle_points_per_face": int(len(triangle_weights)),
                "observables": observables,
            }
        )
    result = {
        "classification": "Observed independent quadrature",
        "field_mesh_frozen": True,
        "separation": (
            "only Duffy quadrature subcells change; P1 nodes/connectivity/"
            "interpolation and material interfaces are unchanged"
        ),
        "tetra_count": len(mesh.tetra),
        "face_count": len(faces),
        "nominal_h_m": mesh.nominal_h_m,
        "curves": curves,
    }
    add_fixed_quadrature_error_curves(result)
    return result


def _six_host_regions(host_bounds):
    regions = []
    for axis in range(3):
        for side in range(2):
            box = np.asarray(host_bounds, dtype=np.float64).copy()
            for prior in range(axis):
                box[prior] = INNER_BOUNDS_M[prior]
            box[axis] = (
                (host_bounds[axis, 0], INNER_BOUNDS_M[axis, 0])
                if side == 0
                else (INNER_BOUNDS_M[axis, 1], host_bounds[axis, 1])
            )
            regions.append(box)
    return regions


def _load_oracle_module(path: Path):
    source = path.read_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    module_name = "lane_a_wave456_reference_" + source_sha256[:12]
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__source_sha256__ = source_sha256
    # Dataclasses with postponed annotations resolve their defining globals
    # through sys.modules while the module body executes.
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _independent_oracle(
    module, host_bounds, wavelength, *, max_evaluations: int = 4096
):
    def displacement(points):
        return prescribed_p_wave_displacement(points, wavelength_m=wavelength)

    regions = [
        (bounds, HOST_DENSITY_KG_M3, displacement)
        for bounds in _six_host_regions(host_bounds)
    ]
    value, convergence = module.adaptive_displacement_box_observables_g_free(
        regions,
        CAVITY_TARGETS_M,
        low_order=10,
        high_order=12,
        escalation_orders=(16, 24),
        rtol=1e-11,
        max_depth=8,
        max_evaluations=max_evaluations,
    )
    return value, convergence


def run_wave6(
    output_dir: Path,
    oracle_module_path: Path,
    *,
    case_ids: set[str] | None = None,
    a_parent_accepted_sha256: str | None = None,
) -> Path:
    started = datetime.now(UTC)
    module = _load_oracle_module(oracle_module_path)
    selected = [
        case
        for case in wave6_case_matrix()
        if case_ids is None or case["case_id"] in case_ids
    ]
    if not selected:
        raise ValueError("no Wave 6 cases selected")
    oracle_sha256 = module.__source_sha256__
    parent_accepted = a_parent_accepted_sha256 == oracle_sha256
    result = {
        "schema_version": 1,
        "classification": {
            "dimensions": "Proposed",
            "computed_matrices": "Observed",
            "interpretations": "Inferred",
            "imputed_values": [],
        },
        "model_scope": (
            "prescribed +x P-wave over a finite host containing a fixed cavity"
        ),
        "limitations": [
            "no elastic scattering solution",
            "no infinite-ground extrapolation",
            "mesh and extent sweeps are separate",
            "A oracle acceptance requires normalized difference <= 1e-11 "
            "without tolerance relaxation",
        ],
        "oracle_module": {
            "path": str(oracle_module_path),
            "sha256": oracle_sha256,
            "parent_acceptance": "accepted_exact_bytes"
            if parent_accepted
            else "PROVISIONAL_PARENT_UNACCEPTED",
        },
        "targets_m": CAVITY_TARGETS_M.tolist(),
        "cases": [],
    }
    oracle_cache = {}
    for case in selected:
        tick = time.perf_counter()
        host_bounds = _scaled_host(case["extent_scale"])
        mesh = build_cartesian_six_tetra_mesh(
            host_bounds,
            case["h_m"],
            inner_bounds_m=INNER_BOUNDS_M,
            host_density_kg_m3=HOST_DENSITY_KG_M3,
            inner_density_kg_m3=0.0,
        )
        displacement = prescribed_p_wave_displacement(
            mesh.vertices_m, wavelength_m=case["wavelength_m"]
        )
        field = UnstructuredDisplacementField(mesh.vertices_m, mesh.tetra, displacement)
        sources = piecewise_tetra_p1_density_sources(
            field, np.repeat(mesh.cell_density_kg_m3[:, None], 4, axis=1)
        )
        oracle_key = (case["extent_scale"], case["wavelength_m"])
        if oracle_key not in oracle_cache:
            oracle_cache[oracle_key] = _independent_oracle(
                module, host_bounds, case["wavelength_m"]
            )
        oracle, convergence = oracle_cache[oracle_key]
        observables = {}
        for name in OBSERVABLES:
            report = compare_tetra_p1_density_representations(
                sources,
                CAVITY_TARGETS_M,
                observable=name,
                chunk_size=8192,
                backend="numpy",
            )
            reference = getattr(oracle, name)
            observables[name] = {
                "unit": {
                    "potential": "m^2 s^-2",
                    "acceleration": "m s^-2",
                    "gradient": "s^-2",
                }[name],
                "mass_displacement": _array_record(
                    G_SI * report.mass_displacement_response
                ),
                "bulk": _array_record(G_SI * report.bulk_response),
                "interface": _array_record(G_SI * report.interface_response),
                "bulk_plus_interface": _array_record(
                    G_SI * report.bulk_interface_response
                ),
                "independent_A": _array_record(G_SI * reference),
                "mass_vs_A_normalized_error": normalized_error(
                    report.mass_displacement_response, reference, name
                ).tolist(),
                "bulk_interface_vs_A_normalized_error": normalized_error(
                    report.bulk_interface_response, reference, name
                ).tolist(),
                "representation_normalized_residual": report.rel_residual.tolist(),
                "comparison_policy": (
                    "mass-displacement and bulk+interface compared; never double-added"
                ),
            }
        result["cases"].append(
            {
                **case,
                "frequency_hz": case["vp_m_s"] / case["wavelength_m"],
                "host_bounds_m": host_bounds.tolist(),
                "fixed_cavity_bounds_m": INNER_BOUNDS_M.tolist(),
                "mesh": {
                    "vertices": len(mesh.vertices_m),
                    "tetra": len(mesh.tetra),
                    "nominal_h_m": mesh.nominal_h_m,
                    "axis_cell_counts": list(mesh.axis_cell_counts),
                    "axis_max_cell_width_m": mesh.axis_max_cell_width_m.tolist(),
                    "material_mass_kg": mesh.material_mass_kg,
                    "geometry_volume_abs_error_m3": abs(
                        mesh.host_volume_m3 - float(np.sum(mesh.tetra_volume_m3))
                    ),
                    "estimated_persistent_array_bytes": int(
                        mesh.vertices_m.nbytes
                        + mesh.tetra.nbytes
                        + mesh.cell_density_kg_m3.nbytes
                        + mesh.tetra_volume_m3.nbytes
                    ),
                },
                "A_convergence": asdict(convergence),
                "A_numerical_status": a_oracle_status(
                    asdict(convergence), parent_accepted=parent_accepted
                ),
                "A_parent_acceptance": "accepted_exact_bytes"
                if parent_accepted
                else "PROVISIONAL_PARENT_UNACCEPTED",
                "observables": observables,
                "elapsed_s": time.perf_counter() - tick,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024.0,
            }
        )
    result["fixed_mesh_quadrature_studies"] = []
    for wavelength in sorted({case["wavelength_m"] for case in selected}):
        fixed_mesh = build_cartesian_six_tetra_mesh(
            HOST_BOUNDS_M,
            4.0,
            inner_bounds_m=INNER_BOUNDS_M,
            host_density_kg_m3=HOST_DENSITY_KG_M3,
            inner_density_kg_m3=0.0,
        )
        fixed_displacement = prescribed_p_wave_displacement(
            fixed_mesh.vertices_m, wavelength_m=wavelength
        )
        study = fixed_p1_subcell_quadrature_study(
            fixed_mesh,
            fixed_displacement,
            CAVITY_TARGETS_M,
            subcells=(1, 2, 4),
            gauss_order=2,
        )
        study["wavelength_m"] = wavelength
        result["fixed_mesh_quadrature_studies"].append(study)
    result["started_utc"] = started.isoformat()
    result["finished_utc"] = datetime.now(UTC).isoformat()
    result["elapsed_s"] = (datetime.now(UTC) - started).total_seconds()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "wave6_finite_body_cavity_results.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oracle-module", type=Path, required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--a-parent-accepted-sha256")
    args = parser.parse_args()
    print(
        run_wave6(
            args.output_dir,
            args.oracle_module,
            case_ids=None if args.case_id is None else set(args.case_id),
            a_parent_accepted_sha256=args.a_parent_accepted_sha256,
        )
    )


if __name__ == "__main__":
    main()
