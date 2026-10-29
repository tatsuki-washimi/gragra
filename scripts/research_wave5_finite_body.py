#!/usr/bin/env python3
"""Wave 5 finite-motion, Jacobian-density, and free-interface study."""

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
from gragra.integrators.point_direct import direct_point_displacement_sum
from scripts.research_wave4_finite_body import (
    AFFINE_A,
    AFFINE_B_M,
    AFFINE_CENTER_M,
    CAVITY_TARGETS_M,
    HOST_BOUNDS_M,
    HOST_DENSITY_KG_M3,
    INNER_BOUNDS_M,
    OBSERVABLES,
    ObservableTriple,
    _array_record,
    _observable_record,
    a_oracle_status,
    build_cartesian_six_tetra_mesh,
    degree2_mass_quadrature,
    evaluate_static_observables_g_free,
    normalized_error,
    response_norm,
    to_physical_observables,
)


def affine_displacement(
    points_m, *, b_m=AFFINE_B_M, matrix=AFFINE_A, center_m=AFFINE_CENTER_M
) -> np.ndarray:
    result = (
        np.asarray(b_m, dtype=np.float64)
        + (
            np.asarray(points_m, dtype=np.float64)
            - np.asarray(center_m, dtype=np.float64)
        )
        @ np.asarray(matrix, dtype=np.float64).T
    )
    result = np.asarray(result, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _linear_observables(points, mass, displacement, targets) -> ObservableTriple:
    return ObservableTriple(
        *(
            direct_point_displacement_sum(
                mass,
                displacement,
                points,
                targets,
                observable=name,
                chunk_size=8192,
                backend="numpy",
            )
            for name in OBSERVABLES
        )
    )


def finite_motion_taylor_study(
    mesh,
    targets_m,
    *,
    alphas,
    b_m=AFFINE_B_M,
    matrix=AFFINE_A,
    center_m=AFFINE_CENTER_M,
) -> dict:
    """Compare fixed material mass with its first derivative and Jacobian form."""
    alphas = np.asarray(alphas, dtype=np.float64)
    if alphas.ndim != 1 or np.any(alphas <= 0.0) or np.any(np.diff(alphas) >= 0.0):
        raise ValueError("alphas must be a strictly decreasing positive vector")
    points, mass = degree2_mass_quadrature(mesh)
    displacement = affine_displacement(
        points, b_m=b_m, matrix=matrix, center_m=center_m
    )
    base = evaluate_static_observables_g_free(points, mass, targets_m)
    repeated_base = evaluate_static_observables_g_free(points, mass, targets_m)
    linear = _linear_observables(points, mass, displacement, targets_m)
    matrix = np.asarray(matrix, dtype=np.float64)
    jacobian_differences = []
    records = {}
    for name in OBSERVABLES:
        base_norm = response_norm(getattr(base, name), name)
        linear_norm = response_norm(getattr(linear, name), name)
        physical_scale = np.maximum(
            base_norm,
            np.maximum(alphas[0] * linear_norm, np.finfo(np.float64).tiny),
        )
        reference_noise = response_norm(
            getattr(repeated_base, name) - getattr(base, name), name
        )
        records[name] = {
            "physical_scale_g_free_by_target": physical_scale.tolist(),
            "physical_scale_physical_by_target": (G_SI * physical_scale).tolist(),
            "reference_noise_source": "deterministic_base_reevaluation",
            "reference_noise_floor_g_free_by_target": reference_noise.tolist(),
            "samples": [],
            "order_assessment": {"positive": [], "negative": []},
            "second_order_confirmed": False,
        }
    for alpha in alphas:
        jacobian = float(np.linalg.det(np.eye(3) + alpha * matrix))
        jacobian_mass = (mesh.cell_density_kg_m3 / jacobian) * (
            mesh.tetra_volume_m3 * jacobian
        )
        original_cell_mass = mesh.cell_density_kg_m3 * mesh.tetra_volume_m3
        denominator = max(
            float(np.max(np.abs(original_cell_mass))), np.finfo(float).tiny
        )
        jacobian_differences.append(
            float(np.max(np.abs(jacobian_mass - original_cell_mass)) / denominator)
        )
        plus = evaluate_static_observables_g_free(
            points + alpha * displacement, mass, targets_m
        )
        minus = evaluate_static_observables_g_free(
            points - alpha * displacement, mass, targets_m
        )
        for name in OBSERVABLES:
            q0 = getattr(base, name)
            q1 = getattr(linear, name)
            qp = getattr(plus, name)
            qm = getattr(minus, name)
            positive_difference = qp - q0 - alpha * q1
            negative_difference = qm - q0 + alpha * q1
            positive_abs = response_norm(positive_difference, name)
            negative_abs = response_norm(negative_difference, name)
            scale = np.asarray(records[name]["physical_scale_g_free_by_target"])
            positive_floor = (
                128.0
                * np.finfo(float).eps
                * (
                    response_norm(qp, name)
                    + response_norm(q0, name)
                    + alpha * response_norm(q1, name)
                )
            )
            negative_floor = (
                128.0
                * np.finfo(float).eps
                * (
                    response_norm(qm, name)
                    + response_norm(q0, name)
                    + alpha * response_norm(q1, name)
                )
            )
            reference_floor = np.asarray(
                records[name]["reference_noise_floor_g_free_by_target"]
            )
            records[name]["samples"].append(
                {
                    "alpha": float(alpha),
                    "positive": {
                        "raw_difference_g_free": _array_record(positive_difference),
                        "raw_difference_physical": _array_record(
                            G_SI * positive_difference
                        ),
                        "absolute_residual_by_target": positive_abs.tolist(),
                        "normalized_residual_by_target": (
                            positive_abs / scale
                        ).tolist(),
                        "rounding_floor_g_free_by_target": positive_floor.tolist(),
                        "total_noise_floor_g_free_by_target": np.maximum(
                            positive_floor, reference_floor
                        ).tolist(),
                    },
                    "negative": {
                        "raw_difference_g_free": _array_record(negative_difference),
                        "raw_difference_physical": _array_record(
                            G_SI * negative_difference
                        ),
                        "absolute_residual_by_target": negative_abs.tolist(),
                        "normalized_residual_by_target": (
                            negative_abs / scale
                        ).tolist(),
                        "rounding_floor_g_free_by_target": negative_floor.tolist(),
                        "total_noise_floor_g_free_by_target": np.maximum(
                            negative_floor, reference_floor
                        ).tolist(),
                    },
                }
            )
    for name in OBSERVABLES:
        for sign in ("positive", "negative"):
            for target_index in range(len(np.asarray(targets_m))):
                accepted = []
                candidates = []
                discarded = []
                for index in range(len(alphas) - 1):
                    first = records[name]["samples"][index][sign]
                    second = records[name]["samples"][index + 1][sign]
                    first_abs = first["absolute_residual_by_target"][target_index]
                    second_abs = second["absolute_residual_by_target"][target_index]
                    first_floor = first["total_noise_floor_g_free_by_target"][
                        target_index
                    ]
                    second_floor = second["total_noise_floor_g_free_by_target"][
                        target_index
                    ]
                    pair = [float(alphas[index]), float(alphas[index + 1])]
                    if first_abs <= first_floor or second_abs <= second_floor:
                        discarded.append(
                            {
                                "alpha_pair": pair,
                                "reason": (
                                    "residual_at_or_below_rounding_or_"
                                    "reference_noise_floor"
                                ),
                            }
                        )
                        continue
                    order = float(
                        np.log(first_abs / second_abs)
                        / np.log(alphas[index] / alphas[index + 1])
                    )
                    candidates.append(order)
                    if 1.8 <= order <= 2.2:
                        accepted.append(order)
                    else:
                        discarded.append(
                            {
                                "alpha_pair": pair,
                                "reason": "observed_order_outside_1.8_to_2.2",
                                "observed_order": order,
                            }
                        )
                status = (
                    "CONFIRMED"
                    if accepted and len(accepted) == len(candidates)
                    else "UNRESOLVED"
                )
                records[name]["order_assessment"][sign].append(
                    {
                        "target_index": target_index,
                        "candidate_orders": candidates,
                        "accepted_orders": accepted,
                        "discarded": discarded,
                        "status": status,
                    }
                )
        records[name]["second_order_confirmed"] = all(
            item["status"] == "CONFIRMED"
            for sign in ("positive", "negative")
            for item in records[name]["order_assessment"][sign]
        )
    return {
        "classification": "Observed",
        "mass_policy": "material point masses fixed",
        "jacobian_density_policy": (
            "rho_prime=rho/det(I+alpha*A), transformed volume=det(I+alpha*A)*volume"
        ),
        "jacobian_mass_relative_difference_by_alpha": jacobian_differences,
        "jacobian_mass_relative_difference": max(jacobian_differences),
        "taylor": records,
        "linear_g_free": {
            name: _array_record(getattr(linear, name)) for name in OBSERVABLES
        },
    }


def _literal_surface_points(order: int, alpha: float, density_jump: float):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    wx, wy = np.meshgrid(weights, weights, indexing="ij")
    eta = alpha * np.cos(np.pi * x / 2.0) * np.cos(np.pi * y / 2.0)
    points = np.stack((x, y, np.zeros_like(x)), axis=-1).reshape(-1, 3)
    mass = (density_jump * eta * wx * wy).reshape(-1)
    return points, mass


def _literal_moved_volume_points(order: int, alpha: float, density_jump: float):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    records, masses = [], []
    for ix, x in enumerate(nodes):
        for iy, y in enumerate(nodes):
            eta = alpha * np.cos(np.pi * x / 2.0) * np.cos(np.pi * y / 2.0)
            z = 0.5 * eta * (nodes + 1.0)
            wz = 0.5 * eta * weights
            records.append(np.stack((np.full(order, x), np.full(order, y), z), axis=1))
            masses.append(density_jump * weights[ix] * weights[iy] * wz)
    return np.concatenate(records), np.concatenate(masses)


def free_surface_column_study(*, alphas_m, targets_m, order: int = 16) -> dict:
    """Compare a signed moved layer with one linear surface sheet.

    The two responses represent the same interface and are never summed.
    """
    alphas = np.asarray(alphas_m, dtype=np.float64)
    density_jump = 1000.0 - 1.2
    values = {}
    for alpha in alphas:
        volume = evaluate_static_observables_g_free(
            *_literal_moved_volume_points(order, float(alpha), density_jump), targets_m
        )
        sheet = evaluate_static_observables_g_free(
            *_literal_surface_points(order, float(alpha), density_jump), targets_m
        )
        values[float(alpha)] = (volume, sheet)
    positives = sorted(float(value) for value in alphas if value > 0.0)
    relative = {name: [] for name in OBSERVABLES}
    for alpha in sorted(positives, reverse=True):
        volume, sheet = values[alpha]
        for name in OBSERVABLES:
            scale = max(
                float(np.max(response_norm(getattr(sheet, name), name))),
                np.finfo(float).tiny,
            )
            relative[name].append(
                float(
                    np.max(
                        response_norm(
                            getattr(volume, name) - getattr(sheet, name), name
                        )
                    )
                    / scale
                )
            )
    zero_volume, zero_sheet = values[0.0]
    zero_jump = evaluate_static_observables_g_free(
        *_literal_surface_points(order, 0.02, 0.0), targets_m
    )
    sign_control = True
    for alpha in positives:
        for name in OBSERVABLES:
            sign_control &= np.array_equal(
                getattr(values[-alpha][1], name), -getattr(values[alpha][1], name)
            )
    return {
        "classification": "Observed numerical quadrature",
        "column_bounds_m": [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
        "densities_kg_m3": [1000.0, 1.2],
        "alphas_m": alphas.tolist(),
        "quadrature_order": order,
        "combination_policy": "compare_only_do_not_add",
        "mass_change_zero_assumed": False,
        "signed_mass_change_kg": {
            str(alpha): float(density_jump * alpha * 16.0 / np.pi**2)
            for alpha in alphas
        },
        "relative_error_by_abs_alpha": relative,
        "responses": {
            str(alpha): {
                "signed_volume_physical": _observable_record(
                    to_physical_observables(pair[0])
                ),
                "surface_sheet_physical": _observable_record(
                    to_physical_observables(pair[1])
                ),
            }
            for alpha, pair in values.items()
        },
        "controls": {
            "zero_displacement_exact_zero": all(
                np.all(value == 0.0) for value in (*zero_volume, *zero_sheet)
            ),
            "zero_density_jump_exact_zero": all(
                np.all(value == 0.0) for value in zero_jump
            ),
            "sign_reversal_linear_sheet": bool(sign_control),
        },
    }


def run_wave5(
    output_dir: Path,
    oracle_json: Path,
    *,
    h_m: float = 2.0,
    a_parent_accepted_sha256: str | None = None,
) -> Path:
    started = datetime.now(UTC)
    mesh = build_cartesian_six_tetra_mesh(
        HOST_BOUNDS_M,
        h_m,
        inner_bounds_m=INNER_BOUNDS_M,
        host_density_kg_m3=HOST_DENSITY_KG_M3,
        inner_density_kg_m3=0.0,
    )
    taylor = finite_motion_taylor_study(
        mesh, CAVITY_TARGETS_M, alphas=np.array((1 / 16, 1 / 32, 1 / 64, 1 / 128))
    )
    points, _ = degree2_mass_quadrature(mesh)
    u = affine_displacement(points)
    scale = 1e-9 / float(np.max(np.linalg.norm(u, axis=1)))
    _, mass = degree2_mass_quadrature(mesh)
    physical_1nm = to_physical_observables(
        _linear_observables(points, mass, scale * u, CAVITY_TARGETS_M)
    )
    field = UnstructuredDisplacementField(
        mesh.vertices_m, mesh.tetra, affine_displacement(mesh.vertices_m)
    )
    sources = piecewise_tetra_p1_density_sources(
        field, np.repeat(mesh.cell_density_kg_m3[:, None], 4, axis=1)
    )
    oracle_bytes = oracle_json.read_bytes()
    import hashlib

    oracle_sha256 = hashlib.sha256(oracle_bytes).hexdigest()
    oracle = json.loads(oracle_bytes)["finite_bodies"]["affine_displacement"]
    representation = {}
    for name in OBSERVABLES:
        report = compare_tetra_p1_density_representations(
            sources, CAVITY_TARGETS_M, observable=name, chunk_size=8192, backend="numpy"
        )
        reference = np.asarray(oracle["reference"][name])
        representation[name] = {
            "mass_displacement_physical": _array_record(
                G_SI * report.mass_displacement_response
            ),
            "bulk_physical": _array_record(G_SI * report.bulk_response),
            "interface_physical": _array_record(G_SI * report.interface_response),
            "bulk_plus_interface_physical": _array_record(
                G_SI * report.bulk_interface_response
            ),
            "comparison_policy": (
                "mass_displacement compared with bulk_plus_interface; never added"
            ),
            "representation_relative_residual": report.rel_residual.tolist(),
            "mass_vs_A_normalized_error": normalized_error(
                report.mass_displacement_response, reference, name
            ).tolist(),
            "bulk_interface_vs_A_normalized_error": normalized_error(
                report.bulk_interface_response, reference, name
            ).tolist(),
        }
    result = {
        "schema_version": 1,
        "classification": {
            "dimensions": "Proposed",
            "computed_values": "Observed",
            "interpretations": "Inferred",
            "imputed_values": [],
        },
        "h_m": h_m,
        "finite_motion": taylor,
        "representations": representation,
        "A_affine_oracle_convergence": oracle["convergence"],
        "A_source": {
            "path": str(oracle_json),
            "sha256": oracle_sha256,
            "parent_acceptance": "accepted_exact_bytes"
            if a_parent_accepted_sha256 == oracle_sha256
            else "PROVISIONAL_PARENT_UNACCEPTED",
        },
        "A_numerical_status": a_oracle_status(
            oracle["convergence"],
            parent_accepted=a_parent_accepted_sha256 == oracle_sha256,
        ),
        "physical_1nm": {
            "normalization": "max quadrature-point |u| = 1 nm",
            "observables": _observable_record(physical_1nm),
        },
        "free_surface": free_surface_column_study(
            alphas_m=np.array((-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02)),
            targets_m=np.array(((0.2, -0.1, 2.5),)),
            order=16,
        ),
        "elapsed_s": (datetime.now(UTC) - started).total_seconds(),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "wave5_finite_body_results.json"
    tick = time.perf_counter()
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["output_write_s"] = time.perf_counter() - tick
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oracle-json", type=Path, required=True)
    parser.add_argument("--h", type=float, default=2.0)
    parser.add_argument("--a-parent-accepted-sha256")
    args = parser.parse_args()
    print(
        run_wave5(
            args.output_dir,
            args.oracle_json,
            h_m=args.h,
            a_parent_accepted_sha256=args.a_parent_accepted_sha256,
        )
    )


if __name__ == "__main__":
    main()
