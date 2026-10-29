"""Tests for volume mesh adapters."""

import numpy as np
import pytest

from gragra.adapters.meshio_adapter import (
    VolumeMesh,
    cell_centroids_volumes,
    from_meshio_volume,
    select_group,
    select_group_tag,
    to_volume_element_source,
)
from gragra.gravity_sources.volume import VolumeElementSource


def test_volume_mesh_validation():
    # Valid tetra mesh
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cells = (("tetra", np.array([[0, 1, 2, 3]])),)
    mesh = VolumeMesh(vertices_m=vertices, cell_blocks=cells)

    assert mesh.vertices_m.shape == (4, 3)
    assert len(mesh.cell_blocks) == 1
    assert mesh.cell_blocks[0][0] == "tetra"
    assert not mesh.vertices_m.flags.writeable
    assert not mesh.cell_blocks[0][1].flags.writeable

    # Valid hexahedron mesh
    vertices_hex = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    cells_hex = (("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]])),)
    mesh_hex = VolumeMesh(vertices_m=vertices_hex, cell_blocks=cells_hex)
    assert mesh_hex.cell_blocks[0][0] == "hexahedron"

    # Empty cells allowed
    empty_mesh = VolumeMesh(vertices_m=vertices, cell_blocks=())
    assert len(empty_mesh.cell_blocks) == 0

    # Invalid vertex shape
    with pytest.raises(ValueError, match="vertices_m must have shape"):
        VolumeMesh(np.ones(3), cells)

    # Non-finite vertices
    with pytest.raises(ValueError, match="vertices_m must contain only finite"):
        VolumeMesh(np.array([[np.nan, 0.0, 0.0]]), ())

    # Invalid cell type
    with pytest.raises(ValueError, match="Unsupported cell type"):
        VolumeMesh(vertices, (("triangle", np.array([[0, 1, 2]])),))

    # Float index
    with pytest.raises(TypeError, match="integer dtype"):
        VolumeMesh(vertices, (("tetra", np.array([[0.0, 1.0, 2.0, 3.0]])),))

    # Bool index
    with pytest.raises(TypeError, match="bool values"):
        VolumeMesh(vertices, (("tetra", np.array([[True, False, True, False]])),))

    # Negative index
    with pytest.raises(ValueError, match="negative indices"):
        VolumeMesh(vertices, (("tetra", np.array([[-1, 1, 2, 3]])),))

    # Out of bounds index
    with pytest.raises(ValueError, match="index out of bounds"):
        VolumeMesh(vertices, (("tetra", np.array([[0, 1, 2, 4]])),))

    # Incorrect columns for tetra (should be Nc x 4)
    with pytest.raises(ValueError, match="tetra cell connectivity must have shape"):
        VolumeMesh(vertices, (("tetra", np.array([[0, 1, 2]])),))

    # Incorrect columns for hexahedron (should be Nc x 8)
    msg = "hexahedron cell connectivity must have shape"
    with pytest.raises(ValueError, match=msg):
        VolumeMesh(vertices_hex, (("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6]])),))


def test_cell_centroids_volumes_tetra():
    # Regular tetra: exact volume = 1/6, centroid = (0.25, 0.25, 0.25)
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cells = (("tetra", np.array([[0, 1, 2, 3]])),)
    mesh = VolumeMesh(vertices, cells)

    centroids, volumes = cell_centroids_volumes(mesh)
    np.testing.assert_allclose(centroids, [[0.25, 0.25, 0.25]], rtol=1e-12)
    np.testing.assert_allclose(volumes, [1.0 / 6.0], rtol=1e-12)
    assert not centroids.flags.writeable
    assert not volumes.flags.writeable


def test_cell_centroids_volumes_hex():
    # Unit cube 1 hex: V = 1, centroid = (0.5, 0.5, 0.5)
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
        ]
    )
    cells = (("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]])),)
    mesh = VolumeMesh(vertices, cells)

    centroids, volumes = cell_centroids_volumes(mesh)
    np.testing.assert_allclose(centroids, [[0.5, 0.5, 0.5]], rtol=1e-12)
    np.testing.assert_allclose(volumes, [1.0], rtol=1e-12)


def test_cell_centroids_volumes_6_tetra():
    # Unit cube partitioned into 6 tetras. Volume sum must be 1.0.
    # HEX nodes: 0 to 7
    # HEX_TETS_0_6:
    # 0: [0, 1, 2, 6]
    # 1: [0, 2, 3, 6]
    # 2: [0, 3, 7, 6]
    # 3: [0, 7, 4, 6]
    # 4: [0, 4, 5, 6]
    # 5: [0, 5, 1, 6]
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
        ]
    )
    cells = (
        (
            "tetra",
            np.array(
                [
                    [0, 1, 2, 6],
                    [0, 2, 3, 6],
                    [0, 3, 7, 6],
                    [0, 7, 4, 6],
                    [0, 4, 5, 6],
                    [0, 5, 1, 6],
                ]
            ),
        ),
    )
    mesh = VolumeMesh(vertices, cells)
    centroids, volumes = cell_centroids_volumes(mesh)

    total_volume = np.sum(volumes)
    assert np.isclose(total_volume, 1.0, rtol=1e-12)

    # Combined centroid
    w_sum = np.sum(centroids * volumes[:, np.newaxis], axis=0)
    combined_centroid = w_sum / total_volume
    np.testing.assert_allclose(combined_centroid, [0.5, 0.5, 0.5], rtol=1e-12)


def test_cell_centroids_volumes_degenerate():
    # Degenerate tetra (all coplanar / zero volume)
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    # The first 3 points are collinear (0,0,0), (1,0,0), (2,0,0). So volume is 0.
    cells = (("tetra", np.array([[0, 1, 2, 3]])),)
    mesh = VolumeMesh(vertices, cells)
    with pytest.raises(ValueError, match="degenerate zero-volume cell"):
        cell_centroids_volumes(mesh)


def test_empty_volume_mesh():
    mesh = VolumeMesh(np.zeros((0, 3)), ())
    c, v = cell_centroids_volumes(mesh)
    assert c.shape == (0, 3)
    assert v.shape == (0,)
    assert not c.flags.writeable
    assert not v.flags.writeable


def test_to_volume_element_source():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cells = (("tetra", np.array([[0, 1, 2, 3]])),)
    mesh = VolumeMesh(vertices, cells)

    # Real density
    src = to_volume_element_source(mesh, density_kg_m3=100.0)
    assert isinstance(src, VolumeElementSource)
    np.testing.assert_allclose(src.positions_m, [[0.25, 0.25, 0.25]], rtol=1e-12)
    np.testing.assert_allclose(src.volumes_m3, [1.0 / 6.0], rtol=1e-12)
    assert src.density_kg_m3 == 100.0
    assert not src.positions_m.flags.writeable
    assert not src.volumes_m3.flags.writeable

    # Complex density
    src_complex = to_volume_element_source(mesh, density_kg_m3=100.0 + 50.0j)
    assert src_complex.density_kg_m3 == 100.0 + 50.0j


def test_physics_far_field():
    # Small cube of 1 m^3 with density 1e3 kg/m^3.
    # Far away (e.g. at 100 m on x-axis), potential should match a point mass.
    vertices = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ]
    )
    cells = (("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]])),)
    mesh = VolumeMesh(vertices, cells)

    density = 1000.0  # kg/m^3
    src = to_volume_element_source(mesh, density)

    # Far-field target
    target = np.array([[100.0, 0.0, 0.0]])

    # Point mass equivalent
    # Mass M = rho * V = 1000 * 1 = 1000 kg.
    # Potential = - G * M / r
    # Potential return arrays of observables must be G_SI multiplied!
    # Let's get potential from the source.
    from gragra.observables.potential import point_potential

    pot_mesh = point_potential(src.as_weighted_source(), target)

    # Expected analytical point mass potential
    from gragra.constants import G_SI

    r = 100.0
    pot_expected = -G_SI * 1000.0 / r

    np.testing.assert_allclose(pot_mesh, [pot_expected], rtol=1e-4)


def test_from_meshio_volume(monkeypatch):
    meshio = pytest.importorskip("meshio")

    # Define a custom CellBlock mock
    class MockCellBlock:
        def __init__(self, type_str, data_arr):
            self.type = type_str
            self.data = data_arr

    # Synthetic mesh parameters
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cells = [
        MockCellBlock("tetra", np.array([[0, 1, 2, 3]])),
        MockCellBlock("triangle", np.array([[0, 1, 2]])),  # should be ignored
    ]
    cell_data = {"gmsh:physical": [np.array([42]), np.array([99])]}
    field_data = {
        "domain_one": np.array([42, 3]),  # tag 42, dim 3
        "boundary_one": np.array([99, 2]),  # tag 99, dim 2
    }

    class MockMesh:
        def __init__(self):
            self.points = points
            self.cells = cells
            self.cell_data = cell_data
            self.field_data = field_data

    # Monkeypatch meshio.read
    monkeypatch.setattr(meshio, "read", lambda _: MockMesh())

    # Test read
    mesh = from_meshio_volume("dummy_path.msh", length_scale_m=2.0)
    assert isinstance(mesh, VolumeMesh)
    np.testing.assert_allclose(mesh.vertices_m, points * 2.0)
    assert len(mesh.cell_blocks) == 1
    assert mesh.cell_blocks[0][0] == "tetra"
    np.testing.assert_allclose(mesh.cell_blocks[0][1], [[0, 1, 2, 3]])

    # check group mapping (only tetra's tag 42 resolved to domain_one)
    assert mesh.cell_groups is not None
    np.testing.assert_allclose(mesh.cell_groups[0], [42])
    assert mesh.group_names is not None
    assert mesh.group_names["domain_one"] == 42
    assert "boundary_one" in mesh.group_names  # all field data mapped

    # Test invalid cell_types argument
    with pytest.raises(ValueError, match="Unsupported cell_types parameter"):
        from_meshio_volume("dummy_path.msh", cell_types=("invalid_type",))

    # Test case when no supported volume cells found
    cells_no_vol = [MockCellBlock("triangle", np.array([[0, 1, 2]]))]

    class MockMeshNoVol:
        def __init__(self):
            self.points = points
            self.cells = cells_no_vol
            self.cell_data = {}
            self.field_data = {}

    monkeypatch.setattr(meshio, "read", lambda _: MockMeshNoVol())
    with pytest.raises(ValueError, match="No supported volume cells found"):
        from_meshio_volume("dummy_path.msh")


def test_select_group():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    # Block 1 (tetra, tag 10)
    # Block 2 (tetra, tag 20)
    cells = (("tetra", np.array([[0, 1, 2, 3], [0, 1, 2, 4]])),)
    cell_groups = (np.array([10, 20]),)
    group_names = {"domain_A": 10, "domain_B": 20}

    mesh = VolumeMesh(
        vertices_m=vertices,
        cell_blocks=cells,
        cell_groups=cell_groups,
        group_names=group_names,
    )

    # Select domain_A (tag 10)
    mesh_a = select_group(mesh, "domain_A")
    assert mesh_a.vertices_m.shape == (5, 3)  # Vertices not reindexed
    assert len(mesh_a.cell_blocks) == 1
    assert mesh_a.cell_blocks[0][0] == "tetra"
    np.testing.assert_allclose(mesh_a.cell_blocks[0][1], [[0, 1, 2, 3]])
    np.testing.assert_allclose(mesh_a.cell_groups[0], [10])

    # Select invalid group name
    with pytest.raises(ValueError, match="Group name 'invalid' not found"):
        select_group(mesh, "invalid")

    # Select from mesh with no group info
    mesh_no_groups = VolumeMesh(vertices_m=vertices, cell_blocks=cells)
    with pytest.raises(ValueError, match="No group names defined in mesh"):
        select_group(mesh_no_groups, "domain_A")


def test_select_group_tag():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    cells = (("tetra", np.array([[0, 1, 2, 3], [0, 1, 2, 4]])),)
    cell_groups = (np.array([10, 20]),)

    # select_group_tag works without group_names
    mesh = VolumeMesh(
        vertices_m=vertices,
        cell_blocks=cells,
        cell_groups=cell_groups,
    )

    mesh_10 = select_group_tag(mesh, 10)
    assert len(mesh_10.cell_blocks) == 1
    np.testing.assert_allclose(mesh_10.cell_blocks[0][1], [[0, 1, 2, 3]])
    np.testing.assert_allclose(mesh_10.cell_groups[0], [10])

    # np.integer tags accepted
    mesh_20 = select_group_tag(mesh, np.int64(20))
    np.testing.assert_allclose(mesh_20.cell_blocks[0][1], [[0, 1, 2, 4]])

    # Nonexistent tag -> explicit error
    with pytest.raises(ValueError, match="Group tag 99 not found"):
        select_group_tag(mesh, 99)

    # Non-integer / bool tag rejected
    with pytest.raises(TypeError, match="tag must be an integer"):
        select_group_tag(mesh, 10.0)
    with pytest.raises(TypeError, match="tag must be an integer"):
        select_group_tag(mesh, True)

    # Mesh without cell_groups -> error
    mesh_no_tags = VolumeMesh(vertices_m=vertices, cell_blocks=cells)
    with pytest.raises(ValueError, match="no cell group tags"):
        select_group_tag(mesh_no_tags, 10)


def test_cell_groups_dtype_validation():
    # cell_groups must be validated as strictly as cell connectivity:
    # bool and non-integer dtypes are rejected (no silent int64 cast).
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cells = (("tetra", np.array([[0, 1, 2, 3]])),)

    # Integer cell_groups accepted
    mesh = VolumeMesh(
        vertices_m=vertices,
        cell_blocks=cells,
        cell_groups=(np.array([7]),),
    )
    np.testing.assert_allclose(mesh.cell_groups[0], [7])

    # Bool cell_groups rejected
    with pytest.raises(TypeError, match="bool values"):
        VolumeMesh(
            vertices_m=vertices,
            cell_blocks=cells,
            cell_groups=(np.array([True]),),
        )

    # Float cell_groups rejected (silent cast prevented)
    with pytest.raises(TypeError, match="integer dtype"):
        VolumeMesh(
            vertices_m=vertices,
            cell_blocks=cells,
            cell_groups=(np.array([7.0]),),
        )
