"""Tests connecting mesh-derived point-mass clouds to prescribed rigid motion.

Covers the positive-mass validation of
``cell_point_mass_cloud_from_group_tags`` and contrast experiment 1 (a
tetra-mesh discretisation of the fixed passing cuboid converging to the
existing analytic Gauss-Legendre reference under the same rigid-transform
trajectory), plus a check that the material-region contrast (experiment 2)
correctly isolates a centre-of-mass shift from the residual internal-mass-
distribution effect.
"""

import numpy as np
import pytest

from gragra.adapters.finite_body_mesh import cell_point_mass_cloud_from_group_tags
from gragra.adapters.meshio_adapter import VolumeMesh, cell_centroids_volumes
from gragra.gravity_sources import PointMassCloud
from scripts.research_wave6_finite_motion_mesh import (
    PASSING_TARGETS_M,
    grid_box_tetra_mesh,
    material_region_contrast,
    mesh_vs_analytic_passing,
)


def _two_tetra_mesh() -> VolumeMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    return VolumeMesh(
        vertices,
        (("tetra", np.array([[0, 1, 2, 3]])), ("tetra", np.array([[1, 2, 3, 4]]))),
        cell_groups=(np.array([1]), np.array([2])),
        group_names={"host": 1, "cavity": 2},
    )


def test_cell_point_mass_cloud_excludes_cavity_and_matches_density_times_volume():
    mesh = _two_tetra_mesh()
    _, volumes = cell_centroids_volumes(mesh)

    cloud = cell_point_mass_cloud_from_group_tags(mesh, {1: 2300.0, 2: 0.0})

    assert isinstance(cloud, PointMassCloud)
    assert cloud.masses_kg.shape == (1,)
    assert cloud.positions_m.shape == (1, 3)
    np.testing.assert_allclose(cloud.masses_kg, [2300.0 * volumes[0]])
    assert cloud.masses_kg.flags.writeable is False
    assert cloud.positions_m.flags.writeable is False


def test_cell_point_mass_cloud_raises_when_every_cell_is_a_cavity():
    mesh = _two_tetra_mesh()
    with pytest.raises(ValueError, match="non-positive density"):
        cell_point_mass_cloud_from_group_tags(mesh, {1: 0.0, 2: 0.0})


def test_cell_point_mass_cloud_rejects_negative_density_via_existing_guard():
    mesh = _two_tetra_mesh()
    with pytest.raises(ValueError, match="non-negative"):
        cell_point_mass_cloud_from_group_tags(mesh, {1: -1.0, 2: 0.0})


@pytest.mark.parametrize("cell_counts", [(2, 1, 1), (4, 2, 2), (4, 4, 2)])
def test_grid_box_tetra_mesh_is_a_conforming_kuhn_triangulation(cell_counts):
    """The fixed-permutation Kuhn decomposition must tile the box exactly.

    If adjacent grid cells' triangulations left gaps or overlaps at shared
    faces, the summed tetra volume would drift from the analytic box volume
    (2 x 1 x 1 = 2 m^3) as resolution changes; it does not.
    """
    mesh = grid_box_tetra_mesh((2.0, 1.0, 1.0), cell_counts, group_tag=1)
    _, volumes = cell_centroids_volumes(mesh)
    assert volumes.size == 6 * cell_counts[0] * cell_counts[1] * cell_counts[2]
    np.testing.assert_allclose(volumes.sum(), 2.0, rtol=1e-12)


def test_mesh_point_mass_cloud_converges_to_the_analytic_passing_cuboid():
    """Contrast experiment 1: mesh discretisation error shrinks with resolution."""
    times = np.linspace(-2.0, 2.0, 5)
    coarse = mesh_vs_analytic_passing((2, 1, 1), times_s=times)
    fine = mesh_vs_analytic_passing((8, 4, 4), times_s=times)

    coarse_gradient = coarse["comparison_vs_analytic"]["gradient"][
        "relative_l2_per_target"
    ]
    fine_gradient = fine["comparison_vs_analytic"]["gradient"]["relative_l2_per_target"]

    assert max(fine_gradient) < max(coarse_gradient)
    # Generous bound: measured relative L2 error at (8,4,4) is ~5e-4.
    assert max(fine_gradient) < 2.0e-3
    for observable in ("potential", "acceleration", "gradient"):
        fine_relative = fine["comparison_vs_analytic"][observable][
            "relative_l2_per_target"
        ]
        assert all(np.isfinite(fine_relative))
        assert max(fine_relative) < 2.0e-3


def test_material_region_contrast_isolates_com_shift_from_shape_effect():
    """Contrast experiment 2: COM-shift dominates; the residual is smaller but present.

    Removing the residual is measured relative to a same-mass, same-shape,
    same-trajectory uniform-density baseline.
    """
    times = np.linspace(-2.0, 2.0, 5)
    result = material_region_contrast(
        (8, 4, 4), times_s=times, targets_m=PASSING_TARGETS_M
    )

    np.testing.assert_allclose(result["com_offset_m"], [0.3, 0.0, 0.0], atol=1e-9)

    for observable in ("potential", "acceleration", "gradient"):
        uncorrected = result["uncorrected_vs_baseline"][observable][
            "relative_l2_per_target"
        ]
        corrected = result["corrected_vs_baseline"][observable][
            "relative_l2_per_target"
        ]
        for uncorrected_value, corrected_value in zip(
            uncorrected, corrected, strict=True
        ):
            # Removing the COM offset must shrink the discrepancy...
            assert corrected_value < uncorrected_value
            # ...but not erase it: a heavy-core/light-shell body still differs
            # from a uniform one at fixed mass, shape, and COM trajectory.
            assert corrected_value > 1.0e-5

    # The uncorrected (COM-shift-dominated) discrepancy is at least an order
    # of magnitude larger than the corrected (shape-only) residual.
    uncorrected_gradient = result["uncorrected_vs_baseline"]["gradient"][
        "relative_l2_per_target"
    ]
    corrected_gradient = result["corrected_vs_baseline"]["gradient"][
        "relative_l2_per_target"
    ]
    assert min(uncorrected_gradient) > 5.0 * max(corrected_gradient)
