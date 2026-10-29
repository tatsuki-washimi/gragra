#!/usr/bin/env python3
"""Wave 6-C: connect prescribed rigid motion to a mesh-derived point-mass body.

This is deliberately a bounded research script, not a source-model API.  It
reuses the established mesh adapter (``VolumeMesh`` / region-tag density) and
the existing framewise rigid-transform trajectory observables exactly as they
are; it does not add a new material-region abstraction or a new integrator.

Two contrasts are computed:

1. A tetra-mesh discretisation of the fixed passing 2x1x1 m, 2700 kg/m3
   cuboid (``gragra.analytic.wave456_reference.passing_cuboid_observables_g_free``)
   against that same closed-form Gauss-Legendre reference, at increasing mesh
   resolution, to characterise discretisation error.
2. Two internal density arrangements of the *same* outer box, same total
   mass, same rigid-transform trajectory: a uniform-density baseline and an
   off-centre heavy-core/light-shell body, compared with and without
   re-centring the core/shell body's centre of mass onto the same body-frame
   origin the baseline uses.  This isolates the trivial "centre of mass moved"
   effect from the residual internal-mass-distribution (multipole) effect.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if type(finder).__name__ != "ScikitBuildRedirectingFinder"
]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from gragra.adapters.finite_body_mesh import cell_point_mass_cloud_from_group_tags
from gragra.adapters.meshio_adapter import VolumeMesh
from gragra.gravity_sources import PointMassCloud
from gragra.gravity_sources.trajectory import PrescribedRigidTransform
from gragra.observables.trajectory import (
    rigid_transform_acceleration,
    rigid_transform_gravity_gradient,
    rigid_transform_potential,
)
from gragra.observables.wave456_reference import passing_cuboid_observables
from scripts.research_wave456_rotation import (  # noqa: E402
    PASSING_TARGETS_M,
    RESEARCH_BACKEND,
    _comparison_by_target,
)

BOX_SIDE_LENGTHS_M = (2.0, 1.0, 1.0)
BOX_DENSITY_KG_M3 = 2700.0
CORE_DENSITY_RATIO = 4.0
PASSING_TIMES_S = np.arange(-2.0, 2.0 + 0.05, 0.1, dtype=np.float64)

# Kuhn's fixed-permutation triangulation of a unit cell into 6 tetrahedra
# sharing its (0,0,0)-(1,1,1) main diagonal.  Applying the *same* local corner
# order to every grid cell (never alternating) makes the resulting
# triangulation conforming across shared faces -- verified in
# tests/test_finite_body_mesh_rigid.py by summing tetra volumes back to the
# analytic box volume.
_KUHN_LOCAL_CORNERS = np.array(
    [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ]
)
_KUHN_LOCAL_TETRA = np.array(
    [
        [0, 1, 2, 6],
        [0, 1, 5, 6],
        [0, 3, 2, 6],
        [0, 3, 7, 6],
        [0, 4, 5, 6],
        [0, 4, 7, 6],
    ],
    dtype=np.int64,
)


def grid_box_tetra_mesh(
    side_lengths_m: tuple[float, float, float],
    cell_counts: tuple[int, int, int],
    *,
    group_tag: int | Callable[[np.ndarray], np.ndarray] = 1,
) -> VolumeMesh:
    """Build a body-frame axis-aligned box as a regular Kuhn-tetra grid.

    ``group_tag`` is either a fixed integer tag for every tetra cell, or a
    callable mapping the ``(Nc, 3)`` tetra centroids to an ``(Nc,)`` integer
    tag array, used to carve material regions (e.g. a dense core inside a
    lighter shell) while keeping the same outer geometry and grid.
    """
    nx, ny, nz = cell_counts
    if min(nx, ny, nz) < 1:
        raise ValueError("cell_counts must all be positive")
    lx, ly, lz = side_lengths_m
    if min(lx, ly, lz) <= 0.0:
        raise ValueError("side_lengths_m must all be positive")

    xs = np.linspace(-lx / 2.0, lx / 2.0, nx + 1)
    ys = np.linspace(-ly / 2.0, ly / 2.0, ny + 1)
    zs = np.linspace(-lz / 2.0, lz / 2.0, nz + 1)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    vertices = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel())).astype(np.float64)

    ci, cj, ck = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    ci = ci.ravel()
    cj = cj.ravel()
    ck = ck.ravel()

    def vid(i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    corners = np.stack(
        [vid(ci + di, cj + dj, ck + dk) for di, dj, dk in _KUHN_LOCAL_CORNERS],
        axis=-1,
    )  # (n_cells, 8)
    tetra = corners[:, _KUHN_LOCAL_TETRA].reshape(-1, 4)  # (n_cells*6, 4)

    if callable(group_tag):
        tetra_centroids = vertices[tetra].mean(axis=1)
        tags = np.asarray(group_tag(tetra_centroids), dtype=np.int64)
    else:
        tags = np.full(len(tetra), int(group_tag), dtype=np.int64)

    return VolumeMesh(
        vertices_m=vertices, cell_blocks=(("tetra", tetra),), cell_groups=(tags,)
    )


def uniform_box_cloud(
    cell_counts: tuple[int, int, int],
    *,
    side_lengths_m: tuple[float, float, float] = BOX_SIDE_LENGTHS_M,
    density_kg_m3: float = BOX_DENSITY_KG_M3,
) -> PointMassCloud:
    """Discretise the fixed passing cuboid at a given mesh resolution."""
    mesh = grid_box_tetra_mesh(side_lengths_m, cell_counts, group_tag=1)
    return cell_point_mass_cloud_from_group_tags(mesh, {1: density_kg_m3})


def core_shell_box_clouds(
    cell_counts: tuple[int, int, int],
    *,
    side_lengths_m: tuple[float, float, float] = BOX_SIDE_LENGTHS_M,
    total_mass_kg: float,
    core_density_ratio: float = CORE_DENSITY_RATIO,
) -> dict[str, Any]:
    """Split the box into a heavy core (x>0) and light shell (x<=0) half.

    Densities are chosen so the two halves (each exactly half the box volume,
    by construction: the x=0 plane is always a grid line here) recover
    ``total_mass_kg`` for an idealised 50/50 volume split; the returned
    ``uncorrected`` cloud measures the actual discretised total mass instead
    of assuming it. Returns both the natural (uncorrected) cloud, whose
    centre of mass sits off the body-frame origin, and a ``corrected`` cloud
    translated so its centre of mass sits exactly at the body-frame origin
    (matching a uniform-density body's centre of mass), isolating the
    internal-distribution effect from the trivial centre-of-mass shift.
    """
    lx, ly, lz = side_lengths_m
    nx = cell_counts[0]
    if nx % 2 != 0:
        raise ValueError("cell_counts[0] must be even for an exact x=0 split")
    half_volume = 0.5 * lx * ly * lz
    shell_density = total_mass_kg / (half_volume * (core_density_ratio + 1.0))
    core_density = core_density_ratio * shell_density

    def tag_by_x(centroids: np.ndarray) -> np.ndarray:
        return np.where(centroids[:, 0] > 0.0, 1, 2)

    mesh = grid_box_tetra_mesh(side_lengths_m, cell_counts, group_tag=tag_by_x)
    uncorrected = cell_point_mass_cloud_from_group_tags(
        mesh, {1: core_density, 2: shell_density}
    )
    com_offset = np.average(
        uncorrected.positions_m, axis=0, weights=uncorrected.masses_kg
    )
    corrected = PointMassCloud(
        uncorrected.masses_kg, uncorrected.positions_m - com_offset
    )
    return {
        "uncorrected": uncorrected,
        "corrected": corrected,
        "com_offset_m": com_offset,
        "core_density_kg_m3": core_density,
        "shell_density_kg_m3": shell_density,
    }


def passing_transform(times_s: np.ndarray) -> PrescribedRigidTransform:
    """Same fixed passing translation as the accepted analytic reference."""
    translations = np.column_stack(
        (10.0 * times_s, np.full(times_s.size, 5.0), np.zeros(times_s.size))
    )
    rotations = np.repeat(np.eye(3)[None], times_s.size, axis=0)
    return PrescribedRigidTransform(times_s, translations, rotations)


def _rigid_observable_series(
    source: PointMassCloud, transform: PrescribedRigidTransform, targets: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "potential": rigid_transform_potential(
            source, transform, targets, chunk_size=8192, backend=RESEARCH_BACKEND
        ),
        "acceleration": rigid_transform_acceleration(
            source, transform, targets, chunk_size=8192, backend=RESEARCH_BACKEND
        ),
        "gradient": rigid_transform_gravity_gradient(
            source, transform, targets, chunk_size=8192, backend=RESEARCH_BACKEND
        ),
    }


def mesh_vs_analytic_passing(
    cell_counts: tuple[int, int, int],
    times_s: np.ndarray = PASSING_TIMES_S,
    targets_m: np.ndarray = PASSING_TARGETS_M,
    *,
    order: int = 24,
) -> dict[str, Any]:
    """Contrast experiment 1: mesh point-mass cloud vs. the analytic box."""
    cloud = uniform_box_cloud(cell_counts)
    transform = passing_transform(times_s)
    mesh_series = _rigid_observable_series(cloud, transform, targets_m)
    # G-included, matching rigid_transform_* above (both multiply by G_SI
    # exactly once, in their respective observables layers).
    reference = passing_cuboid_observables(times_s, targets_m, order=order)
    reference_series = {
        "potential": reference.potential,
        "acceleration": reference.acceleration,
        "gradient": reference.gradient,
    }
    comparison = {
        name: _comparison_by_target(mesh_series[name], reference_series[name], name)
        for name in mesh_series
    }
    return {
        "cell_counts": cell_counts,
        "n_point_masses": int(cloud.masses_kg.size),
        "total_mass_kg": float(cloud.masses_kg.sum()),
        "comparison_vs_analytic": comparison,
    }


def material_region_contrast(
    cell_counts: tuple[int, int, int],
    times_s: np.ndarray = PASSING_TIMES_S,
    targets_m: np.ndarray = PASSING_TARGETS_M,
    *,
    core_density_ratio: float = CORE_DENSITY_RATIO,
) -> dict[str, Any]:
    """Contrast experiment 2: internal mass distribution at fixed mass/shape."""
    baseline = uniform_box_cloud(cell_counts)
    total_mass = float(baseline.masses_kg.sum())
    regions = core_shell_box_clouds(
        cell_counts, total_mass_kg=total_mass, core_density_ratio=core_density_ratio
    )
    transform = passing_transform(times_s)
    baseline_series = _rigid_observable_series(baseline, transform, targets_m)
    uncorrected_series = _rigid_observable_series(
        regions["uncorrected"], transform, targets_m
    )
    corrected_series = _rigid_observable_series(
        regions["corrected"], transform, targets_m
    )
    return {
        "cell_counts": cell_counts,
        "total_mass_kg": total_mass,
        "com_offset_m": regions["com_offset_m"].tolist(),
        "core_density_kg_m3": regions["core_density_kg_m3"],
        "shell_density_kg_m3": regions["shell_density_kg_m3"],
        "uncorrected_vs_baseline": {
            name: _comparison_by_target(
                uncorrected_series[name], baseline_series[name], name
            )
            for name in baseline_series
        },
        "corrected_vs_baseline": {
            name: _comparison_by_target(
                corrected_series[name], baseline_series[name], name
            )
            for name in baseline_series
        },
    }


def run(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    resolutions = [(2, 1, 1), (4, 2, 2), (8, 4, 4), (16, 8, 8)]
    discretisation = [mesh_vs_analytic_passing(res) for res in resolutions]
    contrast = material_region_contrast((16, 8, 8))
    result = {
        "discretisation_error_by_resolution": discretisation,
        "material_region_contrast": contrast,
    }
    output_path = output_dir / "wave6_finite_motion_mesh.json"
    output_path.write_text(json.dumps(result, indent=2, default=float))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_path = run(args.output_dir)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
