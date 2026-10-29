#!/usr/bin/env python3
"""Wave 6 stage-4 quadrature order/local-refinement study for the fixed cavity field.

Stage 3 (``docs/research/wave4-6-F.md``) found that Duffy subcell refinement
alone (subcells 1, 2, 4 at Gauss order 2, see
:func:`scripts.research_wave6_finite_body_cavity.fixed_p1_subcell_quadrature_study`)
does not converge the cavity-near gradient. This module isolates two further
independent variables on the same frozen P1 field: (1) Gauss order itself at
fixed subcells, and (2) refining only the quadrature near the cavity boundary
rather than uniformly over the whole mesh.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from gragra.constants import G_SI
from gragra.fields.finite_body_p1 import piecewise_tetra_p1_density_sources
from gragra.fields.unstructured import UnstructuredDisplacementField
from gragra.integrators.point_direct import direct_point_displacement_sum
from scripts.research_wave4_finite_body import (
    CAVITY_TARGETS_M,
    HOST_BOUNDS_M,
    HOST_DENSITY_KG_M3,
    INNER_BOUNDS_M,
    OBSERVABLES,
    _array_record,
    build_cartesian_six_tetra_mesh,
    normalized_error,
)
from scripts.research_wave6_finite_body_cavity import (
    WAVELENGTHS_M,
    _accumulate_static,
    _decode_array_record,
    _duffy_tetra_rule,
    _duffy_triangle_rule,
    _mutable_observables,
    fixed_p1_subcell_quadrature_study,
    prescribed_p_wave_displacement,
)


def order_sweep_quadrature_study(
    mesh,
    nodal_displacement_m,
    targets_m,
    *,
    orders: tuple[int, ...] = (2, 4, 6),
    subcells: int = 4,
    tetra_batch_size: int = 128,
    face_batch_size: int = 256,
) -> dict:
    """Compare Duffy quadrature at fixed subcells but increasing Gauss order.

    Isolates order from subcell count (the stage-3 sweep in
    :func:`fixed_p1_subcell_quadrature_study` varies only ``subcells`` at
    ``gauss_order=2``). Reuses that function unchanged, calling it once per
    order with a single-element ``subcells`` tuple.
    """
    if len(orders) < 1:
        raise ValueError("orders must be non-empty")
    curves = []
    for order in orders:
        study = fixed_p1_subcell_quadrature_study(
            mesh,
            nodal_displacement_m,
            targets_m,
            subcells=(subcells,),
            gauss_order=int(order),
            tetra_batch_size=tetra_batch_size,
            face_batch_size=face_batch_size,
        )
        curves.append(study["curves"][0])
    finest = curves[-1]
    for name in OBSERVABLES:
        finest_mass = _decode_array_record(
            finest["observables"][name]["mass_displacement_physical"]
        )
        finest_combined = _decode_array_record(
            finest["observables"][name]["bulk_plus_interface_physical"]
        )
        for curve in curves:
            observable = curve["observables"][name]
            mass = _decode_array_record(observable["mass_displacement_physical"])
            combined = _decode_array_record(
                observable["bulk_plus_interface_physical"]
            )
            observable["mass_vs_finest_order_normalized_error"] = normalized_error(
                mass, finest_mass, name
            ).tolist()
            observable[
                "bulk_interface_vs_finest_order_normalized_error"
            ] = normalized_error(combined, finest_combined, name).tolist()
    return {
        "classification": "Observed independent quadrature (order sweep)",
        "field_mesh_frozen": True,
        "separation": (
            "only Gauss order changes at fixed subcells; complements the "
            "stage-3 subcell-only sweep"
        ),
        "subcells_per_duffy_axis": int(subcells),
        "orders": [int(order) for order in orders],
        "curves": curves,
    }


def _distance_to_box_m(points_m: np.ndarray, bounds_m: np.ndarray) -> np.ndarray:
    """Euclidean distance from each point to the closest surface of a box.

    Zero when the point lies inside or on the box.
    """
    lower = bounds_m[:, 0]
    upper = bounds_m[:, 1]
    outside = np.maximum(np.maximum(lower - points_m, points_m - upper), 0.0)
    return np.linalg.norm(outside, axis=-1)


def local_refinement_quadrature_study(
    mesh,
    nodal_displacement_m,
    targets_m,
    *,
    near_distance_m: float,
    inner_bounds_m: np.ndarray = INNER_BOUNDS_M,
    coarse_subcells: int = 4,
    coarse_order: int = 2,
    fine_subcells: int = 4,
    fine_order: int = 6,
    tetra_batch_size: int = 128,
    face_batch_size: int = 256,
) -> dict:
    """Compare uniform-coarse, uniform-fine, and cavity-local-refined quadrature.

    Elements (tetra by centroid, interface faces by centroid) within
    ``near_distance_m`` of the fixed cavity boundary use the fine rule;
    farther elements keep the coarse rule. Reports each configuration's total
    quadrature-point count and its normalized error against the uniform-fine
    configuration, so that "does local refinement reach near-uniform-fine
    accuracy for fewer total points" is a direct read of the output.
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

    inner_bounds = np.asarray(inner_bounds_m, dtype=np.float64)
    tetra_near = (
        _distance_to_box_m(tetra_nodes.mean(axis=1), inner_bounds) <= near_distance_m
    )
    face_near = (
        _distance_to_box_m(face_nodes.mean(axis=1), inner_bounds) <= near_distance_m
    )

    def _tetra_pass(indices, bary, weights, mass_acc, bulk_acc) -> int:
        for start in range(0, len(indices), tetra_batch_size):
            idx = indices[start : start + tetra_batch_size]
            position = np.einsum("ql,nli->nqi", bary, tetra_nodes[idx]).reshape(-1, 3)
            u_value = np.einsum("ql,nli->nqi", bary, tetra_u[idx]).reshape(-1, 3)
            volume_weight = (determinant[idx, None] * weights[None, :]).reshape(-1)
            rho = np.repeat(mesh.cell_density_kg_m3[idx], len(weights))
            mass = rho * volume_weight
            for name in OBSERVABLES:
                mass_acc[name] += direct_point_displacement_sum(
                    mass,
                    u_value,
                    position,
                    targets,
                    observable=name,
                    chunk_size=8192,
                    backend="numpy",
                )
            bulk_density = np.repeat(
                -mesh.cell_density_kg_m3[idx] * divergence[idx], len(weights)
            )
            _accumulate_static(bulk_acc, position, bulk_density * volume_weight, targets)
        return len(indices) * len(weights)

    def _face_pass(indices, bary, weights, interface_acc) -> int:
        for start in range(0, len(indices), face_batch_size):
            idx = indices[start : start + face_batch_size]
            position = np.einsum("ql,fli->fqi", bary, face_nodes[idx]).reshape(-1, 3)
            u_value = np.einsum("ql,fli->fqi", bary, face_u[idx])
            normal_u = np.einsum("fqi,fi->fq", u_value, normals[idx])
            w = (
                jumps[idx, None]
                * normal_u
                * cross_norm[idx, None]
                * weights[None, :]
            ).reshape(-1)
            _accumulate_static(interface_acc, position, w, targets)
        return len(indices) * len(weights)

    coarse_tetra_bary, coarse_tetra_weights = _duffy_tetra_rule(
        coarse_subcells, coarse_order
    )
    fine_tetra_bary, fine_tetra_weights = _duffy_tetra_rule(fine_subcells, fine_order)
    coarse_tri_bary, coarse_tri_weights = _duffy_triangle_rule(
        coarse_subcells, coarse_order
    )
    fine_tri_bary, fine_tri_weights = _duffy_triangle_rule(fine_subcells, fine_order)

    all_tetra = np.arange(len(mesh.tetra))
    all_faces = np.arange(len(faces))

    def _run(mode: str) -> dict:
        mass_acc = _mutable_observables(len(targets), dtype)
        bulk_acc = _mutable_observables(len(targets), dtype)
        interface_acc = _mutable_observables(len(targets), dtype)
        if mode == "coarse":
            tetra_points = _tetra_pass(
                all_tetra, coarse_tetra_bary, coarse_tetra_weights, mass_acc, bulk_acc
            )
            face_points = _face_pass(
                all_faces, coarse_tri_bary, coarse_tri_weights, interface_acc
            )
        elif mode == "fine":
            tetra_points = _tetra_pass(
                all_tetra, fine_tetra_bary, fine_tetra_weights, mass_acc, bulk_acc
            )
            face_points = _face_pass(
                all_faces, fine_tri_bary, fine_tri_weights, interface_acc
            )
        elif mode == "local":
            tetra_points = _tetra_pass(
                np.flatnonzero(~tetra_near),
                coarse_tetra_bary,
                coarse_tetra_weights,
                mass_acc,
                bulk_acc,
            ) + _tetra_pass(
                np.flatnonzero(tetra_near),
                fine_tetra_bary,
                fine_tetra_weights,
                mass_acc,
                bulk_acc,
            )
            face_points = _face_pass(
                np.flatnonzero(~face_near),
                coarse_tri_bary,
                coarse_tri_weights,
                interface_acc,
            ) + _face_pass(
                np.flatnonzero(face_near), fine_tri_bary, fine_tri_weights, interface_acc
            )
        else:
            raise ValueError(f"unknown mode {mode!r}")
        combined = {name: bulk_acc[name] + interface_acc[name] for name in OBSERVABLES}
        observables = {
            name: {
                "mass_displacement_physical": _array_record(G_SI * mass_acc[name]),
                "bulk_plus_interface_physical": _array_record(
                    G_SI * combined[name]
                ),
            }
            for name in OBSERVABLES
        }
        return {
            "observables": observables,
            "total_tetra_quadrature_points": tetra_points,
            "total_face_quadrature_points": face_points,
        }

    configs = {mode: _run(mode) for mode in ("coarse", "fine", "local")}
    errors: dict[str, dict[str, list[float]]] = {"coarse": {}, "local": {}}
    for name in OBSERVABLES:
        fine_mass = _decode_array_record(
            configs["fine"]["observables"][name]["mass_displacement_physical"]
        )
        for mode in ("coarse", "local"):
            mass = _decode_array_record(
                configs[mode]["observables"][name]["mass_displacement_physical"]
            )
            errors[mode][name] = normalized_error(mass, fine_mass, name).tolist()

    return {
        "classification": "Observed independent quadrature (cavity-local refinement)",
        "field_mesh_frozen": True,
        "near_distance_m": float(near_distance_m),
        "coarse_rule": {"subcells": coarse_subcells, "order": coarse_order},
        "fine_rule": {"subcells": fine_subcells, "order": fine_order},
        "near_tetra_count": int(np.count_nonzero(tetra_near)),
        "near_face_count": int(np.count_nonzero(face_near)),
        "total_tetra_count": len(mesh.tetra),
        "total_face_count": len(faces),
        "configs": configs,
        "mass_vs_uniform_fine_normalized_error": errors,
    }


def run_quadrature_order_study(
    output_dir: Path,
    *,
    orders: tuple[int, ...] = (2, 4, 6),
    near_distance_m: float = 8.0,
) -> Path:
    started = datetime.now(UTC)
    result = {
        "schema_version": 1,
        "classification": {
            "dimensions": "Proposed",
            "computed_curves": "Observed",
            "interpretations": "Inferred",
            "imputed_values": [],
        },
        "model_scope": (
            "stage-4 follow-up to docs/research/wave4-6-F.md: quadrature "
            "order and cavity-local refinement on the same frozen h=4m "
            "prescribed P-wave field used by the stage-3 subcell sweep"
        ),
        "orders": list(orders),
        "near_distance_m": near_distance_m,
        "cases": [],
    }
    for wavelength in WAVELENGTHS_M:
        tick = time.perf_counter()
        mesh = build_cartesian_six_tetra_mesh(
            HOST_BOUNDS_M,
            4.0,
            inner_bounds_m=INNER_BOUNDS_M,
            host_density_kg_m3=HOST_DENSITY_KG_M3,
            inner_density_kg_m3=0.0,
        )
        displacement = prescribed_p_wave_displacement(
            mesh.vertices_m, wavelength_m=wavelength
        )
        order_sweep = order_sweep_quadrature_study(
            mesh, displacement, CAVITY_TARGETS_M, orders=orders, subcells=4
        )
        local_refinement = local_refinement_quadrature_study(
            mesh,
            displacement,
            CAVITY_TARGETS_M,
            near_distance_m=near_distance_m,
            coarse_subcells=4,
            coarse_order=2,
            fine_subcells=4,
            fine_order=max(orders),
        )
        result["cases"].append(
            {
                "wavelength_m": wavelength,
                "order_sweep": order_sweep,
                "local_refinement": local_refinement,
                "elapsed_s": time.perf_counter() - tick,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024.0,
            }
        )
    result["started_utc"] = started.isoformat()
    result["finished_utc"] = datetime.now(UTC).isoformat()
    result["elapsed_s"] = (datetime.now(UTC) - started).total_seconds()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "wave6_quadrature_order_study_results.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--order", type=int, nargs="*", default=[2, 4, 6])
    parser.add_argument("--near-distance-m", type=float, default=8.0)
    args = parser.parse_args()
    print(
        run_quadrature_order_study(
            args.output_dir,
            orders=tuple(args.order),
            near_distance_m=args.near_distance_m,
        )
    )


if __name__ == "__main__":
    main()
