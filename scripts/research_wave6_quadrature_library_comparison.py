#!/usr/bin/env python3
"""Wave 6-B: production selectable higher/local-order quadrature comparison.

Three comparisons, matching the ⑥-B task design (see
``docs/research/wave4-6-F.md``'s "⑥-B" section for the narrative writeup of
the numbers this script produces):

(a) ``compare_a_affine_oracle_case``: a globally-affine density/displacement
    box (no interpolation error on any mesh, see
    ``tests/oracles/generate_finite_body_affine_oracle.py``'s module
    docstring) compared against the independent, from-scratch, in-repo A
    reference committed at ``tests/oracles/data/finite_body_affine_v1.json``
    -- replacing the external ``gragra-research`` oracle module dependency
    for this comparison. Both the fixed default (degree-2) rule and the new
    selectable higher-order rule (``quadrature_order``,
    ``gragra.geometry.tetra_quadrature``) are evaluated against the same A
    reference for both the mass-displacement and bulk+interface
    representations.

(b) ``compare_b_pwave_cavity_case``: returns to the existing host+cavity
    prescribed-P-wave case (``scripts/research_wave6_finite_body_cavity.py``)
    at the fixed ``h=4 m`` mesh and reports ``Q_low[xi_h] - Q_high[xi_h]``
    (same discretized P1 field, quadrature order only) separately from an
    order self-convergence sweep. It does not claim the continuous-P-wave
    vs. A-oracle gap (0.57-0.81, ``docs/research/wave4-6-F.md``'s 第六波
    section) closes from higher-order quadrature alone -- connecting this
    case to an independent A reference remains out of scope (unchanged from
    that document's stage-⑤ decision); comparison (a)'s affine-field result
    is what stands in for that connection here.

(c) ``compare_c_uniform_vs_local_case``: uniform high order vs. local
    (per-cell) high order restricted to cells near the cavity boundary,
    reporting accuracy against the uniform-high configuration, quadrature
    point counts, wall time, and peak RSS.

Peak RSS caveat (Limitation, not silently rounded up): ``peak_rss_mib`` is
``resource.getrusage(...).ru_maxrss`` -- a monotonic, process-lifetime
high-water mark, not a per-configuration isolated measurement. Configs run
in increasing-cost order within one process, so each entry's value is a
cumulative watermark up to that point, not a clean delta. Wall time and
quadrature point counts are the reliable primary comparison in (c); peak
RSS is a rough upper-bound indicator only.
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
from gragra.fields.finite_body_p1 import (
    compare_tetra_p1_density_representations,
    piecewise_tetra_p1_density_sources,
)
from gragra.fields.unstructured import UnstructuredDisplacementField
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
    _decode_array_record,
    prescribed_p_wave_displacement,
)
from scripts.research_wave6_quadrature_order_study import _distance_to_box_m
from tests.oracles.generate_finite_body_affine_oracle import (
    BOX_BOUNDS_M,
    DENSITY_RHO0_KG_M3,
    TARGETS_M,
    density_kg_m3,
    displacement_m,
)

AFFINE_ORACLE_JSON = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "oracles"
    / "data"
    / "finite_body_affine_v1.json"
)


def compare_a_affine_oracle_case(
    *, h_m: float = 1.0, orders: tuple[int, ...] = (2, 4, 6, 8, 10)
) -> dict:
    """Affine box: default degree-2 and selectable-order rules vs. the A oracle."""
    oracle = json.loads(AFFINE_ORACLE_JSON.read_text())
    mesh = build_cartesian_six_tetra_mesh(
        np.array(BOX_BOUNDS_M), h_m, host_density_kg_m3=DENSITY_RHO0_KG_M3
    )
    field = UnstructuredDisplacementField(
        mesh.vertices_m, mesh.tetra, displacement_m(mesh.vertices_m)
    )
    density_nodes = density_kg_m3(mesh.vertices_m[mesh.tetra])
    targets = np.array(TARGETS_M)

    oracle_physical = {
        observable: G_SI
        * np.stack(
            [
                np.asarray(point["observables"][observable]["accepted_value"]["values"])
                for point in oracle["points"]
            ]
        )
        for observable in OBSERVABLES
    }

    configurations: dict[str, int | None] = {"default_degree2": None}
    configurations.update({f"order_{order}": order for order in orders})

    per_config = {}
    for label, quadrature_order in configurations.items():
        sources = piecewise_tetra_p1_density_sources(
            field, density_nodes, quadrature_order=quadrature_order
        )
        per_observable = {}
        for observable in OBSERVABLES:
            report = compare_tetra_p1_density_representations(
                sources, targets, observable=observable, chunk_size=8192
            )
            mass_physical = G_SI * report.mass_displacement_response
            combined_physical = G_SI * report.bulk_interface_response
            per_observable[observable] = {
                "mass_vs_A_normalized_error": normalized_error(
                    mass_physical, oracle_physical[observable], observable
                ).tolist(),
                "bulk_interface_vs_A_normalized_error": normalized_error(
                    combined_physical, oracle_physical[observable], observable
                ).tolist(),
                "representation_normalized_residual": report.rel_residual.tolist(),
            }
        per_config[label] = per_observable

    return {
        "case": "affine_box_vs_independent_oracle",
        "h_m": h_m,
        "tetra_count": len(mesh.tetra),
        "targets_m": targets.tolist(),
        "oracle_golden_artifact_sha256": oracle["provenance"]["golden_artifact_sha256"],
        "configurations": per_config,
    }


def compare_b_pwave_cavity_case(
    *, uniform_high_order: int = 8, sweep_orders: tuple[int, ...] = (2, 4, 6, 8)
) -> dict:
    """Host+cavity prescribed P-wave: Q_low vs Q_high, plus order self-convergence."""
    mesh = build_cartesian_six_tetra_mesh(
        HOST_BOUNDS_M,
        4.0,
        inner_bounds_m=INNER_BOUNDS_M,
        host_density_kg_m3=HOST_DENSITY_KG_M3,
        inner_density_kg_m3=0.0,
    )
    density_nodes = np.repeat(mesh.cell_density_kg_m3[:, None], 4, axis=1)

    cases = []
    for wavelength in WAVELENGTHS_M:
        field = UnstructuredDisplacementField(
            mesh.vertices_m,
            mesh.tetra,
            prescribed_p_wave_displacement(mesh.vertices_m, wavelength_m=wavelength),
        )
        default_sources = piecewise_tetra_p1_density_sources(field, density_nodes)
        high_sources = piecewise_tetra_p1_density_sources(
            field, density_nodes, quadrature_order=uniform_high_order
        )

        observables_out = {}
        for observable in OBSERVABLES:
            default_report = compare_tetra_p1_density_representations(
                default_sources,
                CAVITY_TARGETS_M,
                observable=observable,
                chunk_size=8192,
            )
            high_report = compare_tetra_p1_density_representations(
                high_sources, CAVITY_TARGETS_M, observable=observable, chunk_size=8192
            )
            q_low_mass = G_SI * default_report.mass_displacement_response
            q_high_mass = G_SI * high_report.mass_displacement_response
            q_low_combined = G_SI * default_report.bulk_interface_response
            q_high_combined = G_SI * high_report.bulk_interface_response
            observables_out[observable] = {
                "mass_low_vs_high_normalized_diff": normalized_error(
                    q_low_mass, q_high_mass, observable
                ).tolist(),
                "bulk_interface_low_vs_high_normalized_diff": normalized_error(
                    q_low_combined, q_high_combined, observable
                ).tolist(),
                "default_representation_normalized_residual": (
                    default_report.rel_residual.tolist()
                ),
                "high_representation_normalized_residual": (
                    high_report.rel_residual.tolist()
                ),
            }

        mass_by_order = {observable: {} for observable in OBSERVABLES}
        for order in sweep_orders:
            sources = piecewise_tetra_p1_density_sources(
                field, density_nodes, quadrature_order=order
            )
            for observable in OBSERVABLES:
                report = compare_tetra_p1_density_representations(
                    sources, CAVITY_TARGETS_M, observable=observable, chunk_size=8192
                )
                mass_by_order[observable][order] = (
                    G_SI * report.mass_displacement_response
                )

        sweep_out = {}
        for observable in OBSERVABLES:
            finest = mass_by_order[observable][sweep_orders[-1]]
            sweep_out[observable] = {
                str(order): normalized_error(
                    mass_by_order[observable][order], finest, observable
                ).tolist()
                for order in sweep_orders[:-1]
            }

        cases.append(
            {
                "wavelength_m": wavelength,
                "uniform_high_order": uniform_high_order,
                "observables": observables_out,
                "order_self_convergence_vs_finest_swept_order": {
                    "finest_order": sweep_orders[-1],
                    "orders": sweep_out,
                },
            }
        )

    return {
        "case": "pwave_host_cavity_quadrature_library",
        "h_m": 4.0,
        "tetra_count": len(mesh.tetra),
        "cases": cases,
        "note": (
            "Q_low/Q_high are the SAME discretized P1 field (h=4m mesh); "
            "this isolates quadrature order from mesh/interpolation. No "
            "independent A reference is used here -- see comparison (a) for "
            "the A-reference-backed affine case."
        ),
    }


def compare_c_uniform_vs_local_case(
    *, near_distance_m: float = 8.0, high_order: int = 8, low_order: int = 2
) -> dict:
    """Uniform-high vs. cavity-local-high order: accuracy, points, time, RSS."""
    mesh = build_cartesian_six_tetra_mesh(
        HOST_BOUNDS_M,
        4.0,
        inner_bounds_m=INNER_BOUNDS_M,
        host_density_kg_m3=HOST_DENSITY_KG_M3,
        inner_density_kg_m3=0.0,
    )
    density_nodes = np.repeat(mesh.cell_density_kg_m3[:, None], 4, axis=1)
    wavelength = WAVELENGTHS_M[0]
    field = UnstructuredDisplacementField(
        mesh.vertices_m,
        mesh.tetra,
        prescribed_p_wave_displacement(mesh.vertices_m, wavelength_m=wavelength),
    )

    tetra_centroids = mesh.vertices_m[mesh.tetra].mean(axis=1)
    near = _distance_to_box_m(tetra_centroids, INNER_BOUNDS_M) <= near_distance_m
    local_orders = np.where(near, high_order, low_order).astype(np.int64)

    configurations: dict[str, int | np.ndarray | None] = {
        "uniform_low_default": None,
        "local_high_near_cavity": local_orders,
        "uniform_high": high_order,
    }

    results = {}
    for label, quadrature_order in configurations.items():
        tick = time.perf_counter()
        sources = piecewise_tetra_p1_density_sources(
            field, density_nodes, quadrature_order=quadrature_order
        )
        observables = {}
        for observable in OBSERVABLES:
            report = compare_tetra_p1_density_representations(
                sources, CAVITY_TARGETS_M, observable=observable, chunk_size=8192
            )
            observables[observable] = _array_record(
                G_SI * report.mass_displacement_response
            )
        elapsed = time.perf_counter() - tick
        results[label] = {
            "total_mass_quadrature_points": int(len(sources.mass_kg)),
            "total_face_quadrature_points": int(
                len(sources.interface_source.weights_kg)
            ),
            "elapsed_s": elapsed,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "observables": observables,
        }

    uniform_high_by_observable = {
        observable: _decode_array_record(
            results["uniform_high"]["observables"][observable]
        )
        for observable in OBSERVABLES
    }
    local_by_observable = {
        observable: _decode_array_record(
            results["local_high_near_cavity"]["observables"][observable]
        )
        for observable in OBSERVABLES
    }
    accuracy = {
        observable: normalized_error(
            local_by_observable[observable],
            uniform_high_by_observable[observable],
            observable,
        ).tolist()
        for observable in OBSERVABLES
    }

    return {
        "case": "uniform_vs_local_high_order",
        "wavelength_m": wavelength,
        "near_distance_m": near_distance_m,
        "high_order": high_order,
        "low_order": low_order,
        "near_tetra_count": int(np.count_nonzero(near)),
        "total_tetra_count": len(mesh.tetra),
        "configurations": results,
        "local_vs_uniform_high_normalized_error": accuracy,
        "peak_rss_caveat": (
            "cumulative process high-water mark across configs run in "
            "increasing-cost order, not an isolated per-config measurement"
        ),
    }


def run_wave6_b(output_dir: Path) -> Path:
    started = datetime.now(UTC)
    result = {
        "schema_version": 1,
        "classification": {
            "dimensions": "Proposed",
            "computed_values": "Observed",
            "interpretations": "Inferred",
            "imputed_values": [],
        },
        "model_scope": (
            "production selectable higher/local-order tetra/face quadrature "
            "(gragra.geometry.tetra_quadrature via "
            "piecewise_tetra_p1_density_sources(quadrature_order=...)), "
            "compared against a new in-repo independent affine oracle and "
            "against the existing prescribed P-wave host+cavity case"
        ),
        "comparison_a_affine_oracle": compare_a_affine_oracle_case(),
        "comparison_b_pwave_cavity": compare_b_pwave_cavity_case(),
        "comparison_c_uniform_vs_local": compare_c_uniform_vs_local_case(),
    }
    result["started_utc"] = started.isoformat()
    result["finished_utc"] = datetime.now(UTC).isoformat()
    result["elapsed_s"] = (datetime.now(UTC) - started).total_seconds()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "wave6_b_quadrature_library_comparison_results.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run_wave6_b(args.output_dir))


if __name__ == "__main__":
    main()
