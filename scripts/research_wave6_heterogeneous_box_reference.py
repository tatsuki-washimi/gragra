#!/usr/bin/env python3
"""Independent left/right heterogeneous-box reference for Wave 6-M.

The mesh path uses cell-centroid point masses and the existing rigid-transform
observables. The reference path independently sums two analytic rectangular
volume integrals, one for each material region. The two paths therefore share
the body geometry and motion, but not the spatial discretisation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gragra.adapters.finite_body_mesh import cell_point_mass_cloud_from_group_tags
from gragra.adapters.meshio_adapter import VolumeMesh
from gragra.analytic.wave456_reference import (
    integrate_box_observables_g_free,
    multipole_observables_g_free,
)
from gragra.constants import G_SI
from gragra.gravity_sources import PointMassCloud
from gragra.gravity_sources.trajectory import PrescribedRigidTransform
from gragra.observables.trajectory import (
    rigid_transform_acceleration,
    rigid_transform_gravity_gradient,
    rigid_transform_potential,
)
from scripts.research_wave6_finite_motion_mesh import grid_box_tetra_mesh
from scripts.research_wave456_rotation import PASSING_TARGETS_M

BOX_BOUNDS_M = np.array(((-1.0, 1.0), (-0.5, 0.5), (-0.5, 0.5)))
LEFT_BOUNDS_M = np.array(((-1.0, 0.0), (-0.5, 0.5), (-0.5, 0.5)))
RIGHT_BOUNDS_M = np.array(((0.0, 1.0), (-0.5, 0.5), (-0.5, 0.5)))
ORIGINAL_LEFT_DENSITY_KG_M3 = 1080.0
ORIGINAL_RIGHT_DENSITY_KG_M3 = 4320.0
ADDITIONAL_LEFT_DENSITY_KG_M3 = 1800.0
ADDITIONAL_RIGHT_DENSITY_KG_M3 = 3600.0
UNIFORM_DENSITY_KG_M3 = 2700.0
TIMES_S = np.linspace(-2.0, 2.0, 41, dtype=np.float64)
REFERENCE_CONVERGENCE_TIMES_S = TIMES_S[::10]


def _density_case(name: str) -> tuple[float, float, float]:
    if name == "original_ratio_4":
        return (
            ORIGINAL_LEFT_DENSITY_KG_M3,
            ORIGINAL_RIGHT_DENSITY_KG_M3,
            4.0,
        )
    if name == "additional_ratio_2":
        return (
            ADDITIONAL_LEFT_DENSITY_KG_M3,
            ADDITIONAL_RIGHT_DENSITY_KG_M3,
            2.0,
        )
    raise ValueError(f"unknown density case: {name}")


def _bounds_density(case_name: str) -> tuple[tuple[np.ndarray, float], ...]:
    left_density, right_density, _ = _density_case(case_name)
    return (
        (LEFT_BOUNDS_M, left_density),
        (RIGHT_BOUNDS_M, right_density),
    )


def _translation(time_s: float) -> np.ndarray:
    return np.array((10.0 * time_s, 5.0, 0.0), dtype=np.float64)


def _transform(times_s: np.ndarray) -> PrescribedRigidTransform:
    translations = np.stack([_translation(value) for value in times_s])
    rotations = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], len(times_s), axis=0)
    return PrescribedRigidTransform(times_s, translations, rotations)


def _mesh_cloud(
    cell_counts: tuple[int, int, int], case_name: str = "original_ratio_4"
) -> PointMassCloud:
    def tag_by_x(centroids_m: np.ndarray) -> np.ndarray:
        return np.where(centroids_m[:, 0] < 0.0, 1, 2)

    mesh: VolumeMesh = grid_box_tetra_mesh(
        BOX_BOUNDS_M[:, 1] - BOX_BOUNDS_M[:, 0], cell_counts, group_tag=tag_by_x
    )
    left_density, right_density, _ = _density_case(case_name)
    return cell_point_mass_cloud_from_group_tags(
        mesh, {1: left_density, 2: right_density}
    )


def _mesh_series(
    cell_counts: tuple[int, int, int],
    times_s: np.ndarray,
    targets_m: np.ndarray,
    *,
    case_name: str = "original_ratio_4",
) -> dict[str, np.ndarray]:
    source = _mesh_cloud(cell_counts, case_name)
    transform = _transform(times_s)
    return {
        "potential": rigid_transform_potential(source, transform, targets_m),
        "acceleration": rigid_transform_acceleration(source, transform, targets_m),
        "gradient": rigid_transform_gravity_gradient(source, transform, targets_m),
    }


def _add_results(left: dict[str, np.ndarray], right) -> dict[str, np.ndarray]:
    return {
        "potential": left["potential"] + right.potential,
        "acceleration": left["acceleration"] + right.acceleration,
        "gradient": left["gradient"] + right.gradient,
    }


def _analytic_series(
    bounds_density: tuple[tuple[np.ndarray, float], ...],
    times_s: np.ndarray,
    targets_m: np.ndarray,
    *,
    order: int = 24,
    shift_m: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    shift = np.zeros(3, dtype=np.float64) if shift_m is None else np.asarray(shift_m)
    frames = []
    for time_s in times_s:
        translation = _translation(float(time_s)) + shift
        total = {
            name: np.zeros(len(targets_m))
            for name in ("potential", "acceleration", "gradient")
        }
        total["acceleration"] = np.zeros((len(targets_m), 3))
        total["gradient"] = np.zeros((len(targets_m), 3, 3))
        for bounds, density in bounds_density:
            moved = bounds + translation[:, None]
            value = integrate_box_observables_g_free(
                moved, density, targets_m, order=order
            )
            total = _add_results(total, value)
        frames.append(total)
    return {
        name: G_SI * np.stack([frame[name] for frame in frames])
        for name in ("potential", "acceleration", "gradient")
    }


def _relative_l2_per_target(actual: np.ndarray, reference: np.ndarray) -> list[float]:
    if actual.ndim == 2:
        axes = (0,)
    elif actual.ndim == 3:
        axes = (0, 2)
    else:
        axes = (0, 2, 3)
    numerator = np.sqrt(np.sum(np.abs(actual - reference) ** 2, axis=axes))
    denominator = np.sqrt(np.sum(np.abs(reference) ** 2, axis=axes))
    return (numerator / np.maximum(denominator, np.finfo(float).tiny)).tolist()


def _contrast(
    times_s: np.ndarray,
    targets_m: np.ndarray,
    *,
    order: int,
    case_name: str = "original_ratio_4",
) -> dict[str, dict[str, list[float]]]:
    uniform = _analytic_series(
        ((BOX_BOUNDS_M, UNIFORM_DENSITY_KG_M3),), times_s, targets_m, order=order
    )
    heterogeneous = _analytic_series(
        _bounds_density(case_name), times_s, targets_m, order=order
    )
    com_offset = _multipole_moments(_bounds_density(case_name))["center_m"]
    com_aligned = _analytic_series(
        _bounds_density(case_name),
        times_s,
        targets_m,
        order=order,
        shift_m=-com_offset,
    )
    return {
        name: {
            "fixed_occupancy_vs_uniform": _relative_l2_per_target(
                heterogeneous[name], uniform[name]
            ),
            "com_aligned_vs_uniform": _relative_l2_per_target(
                com_aligned[name], uniform[name]
            ),
        }
        for name in uniform
    }


def run_case(
    cell_counts: tuple[int, int, int],
    times_s: np.ndarray = TIMES_S,
    targets_m: np.ndarray = PASSING_TARGETS_M,
    *,
    order: int = 24,
    case_name: str = "original_ratio_4",
) -> dict:
    cloud = _mesh_cloud(cell_counts, case_name)
    mesh = _mesh_series(cell_counts, times_s, targets_m, case_name=case_name)
    reference = _analytic_series(
        _bounds_density(case_name),
        times_s,
        targets_m,
        order=order,
    )
    return {
        "cell_counts": list(cell_counts),
        "point_mass_count": int(len(cloud.masses_kg)),
        "total_mass_kg": float(cloud.masses_kg.sum()),
        "density_case": case_name,
        "density_ratio": _density_case(case_name)[2],
        "relative_l2_per_target": {
            name: _relative_l2_per_target(mesh[name], reference[name]) for name in mesh
        },
    }


def _multipole_moments(
    bounds_density: tuple[tuple[np.ndarray, float], ...],
) -> dict[str, np.ndarray | float]:
    """Return exact mass, COM, central moments, and traceless quadrupole."""
    mass = 0.0
    first = np.zeros(3)
    second = np.zeros((3, 3))
    for bounds, density in bounds_density:
        lengths = bounds[:, 1] - bounds[:, 0]
        volume = float(np.prod(lengths))
        region_mass = density * volume
        center = np.mean(bounds, axis=1)
        covariance_integral = np.diag(region_mass * lengths**2 / 12.0)
        mass += region_mass
        first += region_mass * center
        second += covariance_integral + region_mass * np.outer(center, center)
    center_of_mass = first / mass
    central_second = second - mass * np.outer(center_of_mass, center_of_mass)
    quadrupole = 3.0 * central_second - np.trace(central_second) * np.eye(3)
    return {
        "mass_kg": mass,
        "center_m": center_of_mass,
        "dipole_about_com": first - mass * center_of_mass,
        "second_central_moment_about_com": central_second,
        "quadrupole_about_com": quadrupole,
    }


def _reference_order_convergence(
    case_name: str = "original_ratio_4",
) -> dict[str, dict[str, list[float]]]:
    low = _analytic_series(
        _bounds_density(case_name),
        REFERENCE_CONVERGENCE_TIMES_S,
        PASSING_TARGETS_M,
        order=12,
    )
    middle = _analytic_series(
        _bounds_density(case_name),
        REFERENCE_CONVERGENCE_TIMES_S,
        PASSING_TARGETS_M,
        order=24,
    )
    high = _analytic_series(
        _bounds_density(case_name),
        REFERENCE_CONVERGENCE_TIMES_S,
        PASSING_TARGETS_M,
        order=36,
    )
    return {
        name: {
            "order12_vs_24": _relative_l2_per_target(low[name], middle[name]),
            "order24_vs_36": _relative_l2_per_target(middle[name], high[name]),
        }
        for name in low
    }


def _far_field_residuals(
    case_name: str = "original_ratio_4",
) -> dict[str, list[float]]:
    moments = _multipole_moments(_bounds_density(case_name))
    shift = -np.asarray(moments["center_m"], dtype=np.float64)
    distances = (50.0, 100.0, 200.0)
    output = {name: [] for name in ("potential", "acceleration", "gradient")}
    for distance in distances:
        targets = np.array([[distance, 0.0, 0.0]])
        heterogeneous = _analytic_series(
            _bounds_density(case_name),
            np.array([0.0]),
            targets,
            order=32,
            shift_m=shift,
        )
        uniform = _analytic_series(
            ((BOX_BOUNDS_M, UNIFORM_DENSITY_KG_M3),),
            np.array([0.0]),
            targets,
            order=32,
        )
        for name in output:
            output[name].append(
                _relative_l2_per_target(heterogeneous[name], uniform[name])[0]
            )
    return output


def _far_field_quadrupole_attribution(
    case_name: str = "original_ratio_4",
    *,
    world_offset_m: np.ndarray | None = None,
) -> dict[str, object]:
    """Compare the COM-aligned box difference with its quadrupole prediction.

    The observed difference is the order-32 independent two-box Gauss result
    for the heterogeneous body minus the order-32 result for the uniform body,
    with both bodies sharing the same world-space center.  The inferred
    prediction uses only the difference of the exact COM-central second
    moments, with zero monopole and zero dipole, evaluated about that same
    world-space center.  It therefore tests whether the leading nonzero mass
    moment accounts for the far-field difference.
    """
    if world_offset_m is None:
        world_offset = np.zeros(3, dtype=np.float64)
    else:
        world_offset = np.asarray(world_offset_m, dtype=np.float64)
        if world_offset.shape != (3,) or not np.isfinite(world_offset).all():
            raise ValueError("world_offset_m must be a finite vector with shape (3,)")

    heterogeneous_moments = _multipole_moments(_bounds_density(case_name))
    uniform_moments = _multipole_moments(((BOX_BOUNDS_M, UNIFORM_DENSITY_KG_M3),))
    delta_second = np.asarray(
        heterogeneous_moments["second_central_moment_about_com"]
    ) - np.asarray(uniform_moments["second_central_moment_about_com"])
    delta_quadrupole = np.asarray(
        heterogeneous_moments["quadrupole_about_com"]
    ) - np.asarray(uniform_moments["quadrupole_about_com"])

    distances = (50.0, 100.0, 200.0)
    observed: dict[str, list[object]] = {
        name: [] for name in ("potential", "acceleration", "gradient")
    }
    predicted: dict[str, list[object]] = {
        name: [] for name in ("potential", "acceleration", "gradient")
    }
    relative_error: dict[str, list[float]] = {
        name: [] for name in ("potential", "acceleration", "gradient")
    }

    common_center = _translation(0.0) + world_offset
    com_shift = (
        -np.asarray(heterogeneous_moments["center_m"], dtype=np.float64) + world_offset
    )
    target_positions: list[list[float]] = []
    for distance in distances:
        targets = np.array([[distance, 0.0, 0.0]], dtype=np.float64) + world_offset
        target_positions.append(targets[0].tolist())
        heterogeneous = _analytic_series(
            _bounds_density(case_name),
            np.array([0.0]),
            targets,
            order=32,
            shift_m=com_shift,
        )
        uniform = _analytic_series(
            ((BOX_BOUNDS_M, UNIFORM_DENSITY_KG_M3),),
            np.array([0.0]),
            targets,
            order=32,
            shift_m=world_offset,
        )
        quadrupole = multipole_observables_g_free(
            common_center,
            0.0,
            delta_second,
            targets,
        )
        for name in observed:
            observed_value = heterogeneous[name][0, 0] - uniform[name][0, 0]
            predicted_value = G_SI * getattr(quadrupole, name)[0]
            observed[name].append(np.asarray(observed_value).tolist())
            predicted[name].append(np.asarray(predicted_value).tolist())
            denominator = max(
                float(np.linalg.norm(observed_value)), np.finfo(float).tiny
            )
            relative_error[name].append(
                float(np.linalg.norm(predicted_value - observed_value) / denominator)
            )

    return {
        "case_name": case_name,
        "reference_order": 32,
        "distances_m": list(distances),
        "target_positions_m": target_positions,
        "common_center_m": common_center.tolist(),
        "expansion_center_m": common_center.tolist(),
        "delta_second_moment_kg_m2": delta_second.tolist(),
        "delta_quadrupole_kg_m2": delta_quadrupole.tolist(),
        "observed_difference_g_si": observed,
        "quadrupole_prediction_g_si": predicted,
        "relative_error": relative_error,
        "classification": {
            "observed_difference": (
                "order-32 independent two-box Gauss result after COM alignment"
            ),
            "quadrupole_prediction": (
                "Inferred leading second-moment field with zero monopole and dipole"
            ),
        },
    }


def _json_moments(case_name: str) -> dict[str, object]:
    moments = _multipole_moments(_bounds_density(case_name))
    return {
        "mass_kg": float(moments["mass_kg"]),
        "center_m": np.asarray(moments["center_m"]).tolist(),
        "dipole_about_com": np.asarray(moments["dipole_about_com"]).tolist(),
        "second_central_moment_about_com": np.asarray(
            moments["second_central_moment_about_com"]
        ).tolist(),
        "quadrupole_about_com": np.asarray(moments["quadrupole_about_com"]).tolist(),
    }


def run(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    resolutions = ((2, 1, 1), (4, 2, 2), (8, 4, 4))
    primary_cases = [run_case(counts) for counts in resolutions]
    additional_cases = [
        run_case(counts, case_name="additional_ratio_2") for counts in resolutions
    ]
    moment_cases = {
        name: _json_moments(name) for name in ("original_ratio_4", "additional_ratio_2")
    }
    result = {
        "classification": "Observed independent heterogeneous-object reference",
        "materials": {
            "original_ratio_4": {
                "left_density_kg_m3": ORIGINAL_LEFT_DENSITY_KG_M3,
                "right_density_kg_m3": ORIGINAL_RIGHT_DENSITY_KG_M3,
            },
            "additional_ratio_2": {
                "left_density_kg_m3": ADDITIONAL_LEFT_DENSITY_KG_M3,
                "right_density_kg_m3": ADDITIONAL_RIGHT_DENSITY_KG_M3,
            },
            "uniform_density_kg_m3": UNIFORM_DENSITY_KG_M3,
        },
        "times_s": TIMES_S.tolist(),
        "targets_m": PASSING_TARGETS_M.tolist(),
        "analytic_order": 24,
        "mesh_vs_two_box_reference": {
            "original_ratio_4": primary_cases,
            "additional_ratio_2": additional_cases,
        },
        "contrast": {
            "original_ratio_4": _contrast(TIMES_S, PASSING_TARGETS_M, order=24),
            "additional_ratio_2": _contrast(
                TIMES_S,
                PASSING_TARGETS_M,
                order=24,
                case_name="additional_ratio_2",
            ),
        },
        "reference_order_convergence": {
            "original_ratio_4": _reference_order_convergence(),
            "additional_ratio_2": _reference_order_convergence("additional_ratio_2"),
        },
        "multipole_moments_about_com": moment_cases,
        "far_field_residuals_com_aligned": {
            "original_ratio_4": _far_field_residuals(),
            "additional_ratio_2": _far_field_residuals("additional_ratio_2"),
        },
        "far_field_quadrupole_attribution": {
            "original_ratio_4": _far_field_quadrupole_attribution(),
            "additional_ratio_2": _far_field_quadrupole_attribution(
                "additional_ratio_2"
            ),
        },
        "note": (
            "The original ratio-4 model is the primary comparison; the ratio-2 "
            "model from the earlier pilot is retained as an additional case. "
            "The two-box reference is independent of the centroid point-mass "
            "mesh path. Fixed-occupancy and COM-aligned contrasts are reported "
            "separately; neither is a continuum geophysical claim."
        ),
    }
    path = output_dir / "wave6_heterogeneous_box_reference.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.output_dir))


if __name__ == "__main__":
    main()
