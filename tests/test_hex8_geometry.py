"""Tests for hex8 geometry foundation and VolumeMesh PointMassCloud discretization."""

import numpy as np
import pytest

from gragra.adapters.finite_body_mesh import cell_point_mass_cloud_from_group_tags
from gragra.adapters.meshio_adapter import VolumeMesh
from gragra.fields.hex8_geometry import (
    check_hex8_jacobians,
    hex8_centroids_and_volumes_gauss,
    hex8_shape_and_derivative,
)


def _make_unit_cube_hex8():
    """Return vertices and connectivity for a [0, 1]^3 unit cube."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    return vertices, conn


def test_hex8_shape_partition_of_unity():
    """Shape functions must sum to 1 at any reference point."""
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [-0.5, 0.3, 0.8],
            [0.7, -0.2, -0.4],
            [1.0, 1.0, 1.0],
        ]
    )
    shape, deriv = hex8_shape_and_derivative(pts)
    assert np.allclose(np.sum(shape, axis=1), 1.0, rtol=1e-14, atol=1e-14)
    # Derivative sum must be 0
    assert np.allclose(np.sum(deriv, axis=1), 0.0, rtol=1e-14, atol=1e-14)


def test_hex8_gauss_rule_volume_and_centroid():
    """Unit cube volume must be 1.0 and centroid (0.5, 0.5, 0.5)."""
    vertices, conn = _make_unit_cube_hex8()
    check_hex8_jacobians(vertices, conn)

    centroids, volumes = hex8_centroids_and_volumes_gauss(vertices, conn, order=2)
    assert np.isclose(volumes[0], 1.0, rtol=1e-14, atol=1e-14)
    assert np.allclose(centroids[0], [0.5, 0.5, 0.5], rtol=1e-14, atol=1e-14)


def test_hex8_inverted_cell_rejection():
    """Inverted or degenerate hex8 cell must raise ValueError."""
    vertices, conn = _make_unit_cube_hex8()
    # Invert connectivity by swapping nodes 0 and 1
    inv_conn = conn.copy()
    inv_conn[0, 0], inv_conn[0, 1] = inv_conn[0, 1], inv_conn[0, 0]

    with pytest.raises(ValueError, match="Jacobian is non-positive"):
        check_hex8_jacobians(vertices, inv_conn)

    # Duplicate node index
    deg_conn = conn.copy()
    deg_conn[0, 1] = deg_conn[0, 0]
    with pytest.raises(ValueError, match="duplicate vertex indices"):
        check_hex8_jacobians(vertices, deg_conn)


def test_volume_mesh_hex8_point_mass_cloud():
    """VolumeMesh with hex8 cells converts cleanly to PointMassCloud."""
    vertices, conn = _make_unit_cube_hex8()
    mesh = VolumeMesh(
        vertices_m=vertices,
        cell_blocks=(("hexahedron", conn),),
        cell_groups=(np.array([10], dtype=np.int64),),
    )
    density_map = {10: 2500.0}

    cloud = cell_point_mass_cloud_from_group_tags(mesh, density_map)
    assert len(cloud.masses_kg) == 1
    assert np.isclose(cloud.masses_kg[0], 2500.0, rtol=1e-14)
    assert np.allclose(cloud.positions_m[0], [0.5, 0.5, 0.5], rtol=1e-14)
