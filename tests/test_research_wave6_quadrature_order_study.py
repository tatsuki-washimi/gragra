"""Stage-4 follow-up tests: quadrature order sweep and cavity-local refinement."""

import numpy as np

from scripts.research_wave4_finite_body import build_cartesian_six_tetra_mesh
from scripts.research_wave5_finite_body import affine_displacement
from scripts.research_wave6_quadrature_order_study import (
    local_refinement_quadrature_study,
    order_sweep_quadrature_study,
)


def _small_mesh():
    return build_cartesian_six_tetra_mesh(
        np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]),
        1.0,
        host_density_kg_m3=2.0,
    )


def _small_displacement(mesh):
    return affine_displacement(
        mesh.vertices_m,
        b_m=np.array([0.2, -0.1, 0.05]),
        matrix=np.zeros((3, 3)),
        center_m=np.zeros(3),
    )


def test_order_sweep_isolates_gauss_order_from_subcell_refinement():
    mesh = _small_mesh()
    displacement = _small_displacement(mesh)
    study = order_sweep_quadrature_study(
        mesh,
        displacement,
        np.array([[5.0, 3.0, 4.0]]),
        orders=(2, 4, 6),
        subcells=2,
        tetra_batch_size=8,
        face_batch_size=16,
    )
    assert study["field_mesh_frozen"]
    assert study["orders"] == [2, 4, 6]
    assert [
        item["gauss_order_per_subcell_axis"] for item in study["curves"]
    ] == [2, 4, 6]
    assert [item["subcells_per_duffy_axis"] for item in study["curves"]] == [2, 2, 2]
    for observable in ("potential", "acceleration", "gradient"):
        quadrature_error = [
            max(
                item["observables"][observable][
                    "mass_vs_finest_order_normalized_error"
                ]
            )
            for item in study["curves"]
        ]
        assert quadrature_error[-1] == 0.0
        assert quadrature_error[0] >= quadrature_error[1] >= quadrature_error[-1]


def test_local_refinement_reduces_to_coarse_and_fine_limits():
    mesh = _small_mesh()
    displacement = _small_displacement(mesh)
    targets = np.array([[5.0, 3.0, 4.0]])
    inner_bounds = np.array([[-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3]])

    all_far = local_refinement_quadrature_study(
        mesh,
        displacement,
        targets,
        near_distance_m=0.0,
        inner_bounds_m=inner_bounds,
        coarse_subcells=2,
        coarse_order=2,
        fine_subcells=2,
        fine_order=6,
        tetra_batch_size=8,
        face_batch_size=16,
    )
    assert all_far["near_tetra_count"] == 0
    assert (
        all_far["configs"]["local"]["total_tetra_quadrature_points"]
        == all_far["configs"]["coarse"]["total_tetra_quadrature_points"]
    )
    for observable in ("potential", "acceleration", "gradient"):
        np.testing.assert_allclose(
            all_far["mass_vs_uniform_fine_normalized_error"]["local"][observable],
            all_far["mass_vs_uniform_fine_normalized_error"]["coarse"][observable],
        )

    all_near = local_refinement_quadrature_study(
        mesh,
        displacement,
        targets,
        near_distance_m=1.0e9,
        inner_bounds_m=inner_bounds,
        coarse_subcells=2,
        coarse_order=2,
        fine_subcells=2,
        fine_order=6,
        tetra_batch_size=8,
        face_batch_size=16,
    )
    assert all_near["near_tetra_count"] == all_near["total_tetra_count"]
    assert (
        all_near["configs"]["local"]["total_tetra_quadrature_points"]
        == all_near["configs"]["fine"]["total_tetra_quadrature_points"]
    )
    for observable in ("potential", "acceleration", "gradient"):
        np.testing.assert_allclose(
            all_near["mass_vs_uniform_fine_normalized_error"]["local"][observable],
            0.0,
            atol=1e-12,
        )

    partial = local_refinement_quadrature_study(
        mesh,
        displacement,
        targets,
        near_distance_m=0.5,
        inner_bounds_m=inner_bounds,
        coarse_subcells=2,
        coarse_order=2,
        fine_subcells=2,
        fine_order=6,
        tetra_batch_size=8,
        face_batch_size=16,
    )
    assert 0 < partial["near_tetra_count"] < partial["total_tetra_count"]
    assert (
        partial["configs"]["coarse"]["total_tetra_quadrature_points"]
        < partial["configs"]["local"]["total_tetra_quadrature_points"]
        < partial["configs"]["fine"]["total_tetra_quadrature_points"]
    )
