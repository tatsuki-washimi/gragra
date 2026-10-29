"""Tests for meshio adapter."""

import os
import tempfile

import numpy as np
import pytest

from gragra.adapters.meshio_adapter import (
    TriangleSurfaceMesh,
    centroids_normals_areas,
    from_meshio,
    to_surface_mass_sheet,
)
from gragra.gravity_sources.surface import SurfaceMassSheetSource


def test_triangle_surface_mesh_validation():
    # Valid construction
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    triangles = np.array([[0, 1, 2]])
    mesh = TriangleSurfaceMesh(vertices, triangles)
    assert mesh.vertices_m.shape == (3, 3)
    assert mesh.triangles.shape == (1, 3)
    assert not mesh.vertices_m.flags.writeable
    assert not mesh.triangles.flags.writeable

    # Invalid shape
    with pytest.raises(ValueError, match="vertices_m must have shape"):
        TriangleSurfaceMesh(np.ones(3), triangles)
    with pytest.raises(ValueError, match="triangles must have shape"):
        TriangleSurfaceMesh(vertices, np.array([0, 1, 2]))

    # Negative index
    with pytest.raises(ValueError, match="negative indices"):
        TriangleSurfaceMesh(vertices, np.array([[-1, 1, 2]]))

    # Out of bounds index
    with pytest.raises(ValueError, match="index out of bounds"):
        TriangleSurfaceMesh(vertices, np.array([[0, 1, 3]]))

    # Degenerate zero-area triangle
    collinear_v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="degenerate zero-area triangles"):
        TriangleSurfaceMesh(collinear_v, triangles)

    # Float triangle index
    float_triangles = np.array([[0.0, 1.0, 2.0]])
    with pytest.raises(TypeError, match="integer dtype"):
        TriangleSurfaceMesh(vertices, float_triangles)

    # Bool triangle index
    bool_triangles = np.array([[True, False, True]])
    with pytest.raises(TypeError, match="bool values"):
        TriangleSurfaceMesh(vertices, bool_triangles)

    # Empty vertices + nonempty triangles
    empty_v = np.zeros((0, 3))
    with pytest.raises(ValueError, match="index out of bounds"):
        TriangleSurfaceMesh(empty_v, triangles)


def test_mesh_centroids_normals_areas():
    vertices = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    triangles = np.array([[0, 1, 2]])
    mesh = TriangleSurfaceMesh(vertices, triangles)

    centroids, normals, areas = centroids_normals_areas(mesh)
    np.testing.assert_allclose(centroids, [[2.0 / 3.0, 2.0 / 3.0, 0.0]], rtol=1e-12)
    np.testing.assert_allclose(normals, [[0.0, 0.0, 1.0]], rtol=1e-12)
    np.testing.assert_allclose(areas, [2.0], rtol=1e-12)

    # Empty triangles
    empty_mesh = TriangleSurfaceMesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=int))
    c, n, a = centroids_normals_areas(empty_mesh)
    assert c.shape == (0, 3)
    assert n.shape == (0, 3)
    assert a.shape == (0,)


def test_to_surface_mass_sheet():
    vertices = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    triangles = np.array([[0, 1, 2]])
    mesh = TriangleSurfaceMesh(vertices, triangles)

    src = to_surface_mass_sheet(mesh, surface_density_kg_m2=150.0)
    assert isinstance(src, SurfaceMassSheetSource)
    np.testing.assert_allclose(src.positions_m, [[2.0 / 3.0, 2.0 / 3.0, 0.0]])
    np.testing.assert_allclose(src.normals, [[0.0, 0.0, 1.0]])
    np.testing.assert_allclose(src.areas_m2, [2.0])
    assert src.density_kg_m3 == 150.0
    assert src.normal_displacement_m == 1.0


def test_from_meshio_file():
    # Import meshio or skip
    meshio = pytest.importorskip("meshio")

    # Create temporary mesh file
    mesh = meshio.Mesh(
        points=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        cells=[("triangle", np.array([[0, 1, 2]]))],
    )
    with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
        temp_path = f.name
    try:
        meshio.write(temp_path, mesh)

        # Test read
        mesh_read = from_meshio(temp_path, length_scale_m=2.0)
        assert isinstance(mesh_read, TriangleSurfaceMesh)
        np.testing.assert_allclose(
            mesh_read.vertices_m,
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            rtol=1e-12,
        )
        np.testing.assert_allclose(mesh_read.triangles, [[0, 1, 2]])

        # Test invalid length_scale_m
        with pytest.raises(TypeError, match="must not be bool"):
            from_meshio(temp_path, length_scale_m=True)
        with pytest.raises(TypeError, match="must be numeric"):
            from_meshio(temp_path, length_scale_m="invalid")
        with pytest.raises(ValueError, match="must be finite"):
            from_meshio(temp_path, length_scale_m=np.nan)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
