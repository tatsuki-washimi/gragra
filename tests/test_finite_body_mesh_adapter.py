"""Tests for the finite-body boundary around ``VolumeMesh``."""

from pathlib import Path

import numpy as np
import pytest

from gragra.adapters.finite_body_mesh import (
    cell_density_from_group_tags,
    cell_density_nodes_from_group_tags,
    to_unstructured_displacement_field,
)
from gragra.adapters.meshio_adapter import VolumeMesh, from_meshio_volume
from gragra.fields.finite_body_p1 import (
    compare_tetra_p1_density_representations,
    piecewise_tetra_p1_density_sources,
)
from gragra.fields.unstructured import UnstructuredDisplacementField
from scripts.research_wave6_mesh_exchange_e2e import (
    OBSERVABLES,
    build_host_cavity_volume_mesh,
    build_mass_displacement_sources,
    compare_connectivity,
    write_host_cavity_gmsh_file,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_tetra_blocks_and_group_density_preserve_block_order():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    mesh = VolumeMesh(
        vertices,
        (("tetra", np.array([[0, 1, 2, 3]])), ("tetra", np.array([[1, 2, 3, 4]]))),
        cell_groups=(np.array([7]), np.array([3])),
        group_names={"left": 7, "right": 3},
    )
    displacement = 0.01 * vertices

    field = to_unstructured_displacement_field(mesh, displacement)
    density = cell_density_from_group_tags(mesh, {3: 5.0, 7: 2.0})

    np.testing.assert_array_equal(field.tetra, np.array([[0, 1, 2, 3], [1, 2, 3, 4]]))
    np.testing.assert_array_equal(density, np.array([2.0, 5.0]))
    assert density.flags.writeable is False


def test_adapter_rejects_nonempty_hex_and_incomplete_or_unused_group_maps():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    hexahedron = VolumeMesh(vertices, (("hexahedron", np.arange(8).reshape(1, 8)),))
    with pytest.raises(ValueError, match="tetra cells only"):
        to_unstructured_displacement_field(hexahedron, np.zeros_like(vertices))

    tetra = VolumeMesh(
        vertices[:5],
        (("tetra", np.array([[0, 1, 2, 4]])),),
        cell_groups=(np.array([4]),),
    )
    with pytest.raises(ValueError, match="missing"):
        cell_density_from_group_tags(tetra, {})
    with pytest.raises(ValueError, match="unused"):
        cell_density_from_group_tags(tetra, {4: 1.0, 9: 2.0})
    for value, error in ((True, TypeError), (1 + 1j, TypeError), (-1.0, ValueError)):
        with pytest.raises(error):
            cell_density_from_group_tags(tetra, {4: value})


def test_adapter_rejects_missing_groups_and_bad_mapping_keys():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    mesh = VolumeMesh(vertices, (("tetra", np.array([[0, 1, 2, 3]])),))
    with pytest.raises(ValueError, match="cell group tags"):
        cell_density_from_group_tags(mesh, {1: 2.0})

    tagged = VolumeMesh(
        vertices,
        (("tetra", np.array([[0, 1, 2, 3]])),),
        cell_groups=(np.array([1]),),
    )
    with pytest.raises(TypeError, match="tags must be integers"):
        cell_density_from_group_tags(tagged, {True: 2.0})


def test_cell_density_nodes_from_group_tags_broadcasts_cell_constant_density():
    vertices = np.array(
        [[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0, 0, -1.0]]
    )
    tetra = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
    mesh = VolumeMesh(
        vertices,
        (("tetra", tetra),),
        cell_groups=(np.array([100, 200]),),
        group_names={"host": 100, "cavity": 200},
    )

    nodes = cell_density_nodes_from_group_tags(mesh, {100: 2300.0, 200: 0.0})

    np.testing.assert_array_equal(nodes, np.array([[2300.0] * 4, [0.0] * 4]))
    assert nodes.flags.writeable is False


def test_cell_density_nodes_from_group_tags_matches_a_hand_built_nc4_array():
    """5-B completion criterion: VolumeMesh + region tags reproduces a
    hand-constructed (Nc, 4) density array through piecewise_tetra_p1_density_sources.
    """
    vertices = np.array(
        [[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0, 0, -1.0]]
    )
    tetra = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
    displacement = np.tile([0.0, 0.0, 1.0], (5, 1))
    targets = np.array([[3.0, 2.0, 1.5]])

    mesh = VolumeMesh(
        vertices,
        (("tetra", tetra),),
        cell_groups=(np.array([100, 200]),),
        group_names={"host": 100, "cavity": 200},
    )
    nodes_via_adapter = cell_density_nodes_from_group_tags(
        mesh, {100: 2300.0, 200: 0.0}
    )

    field = UnstructuredDisplacementField(vertices, tetra, displacement)
    hand_built_nodes = np.array([[2300.0] * 4, [0.0] * 4])

    for observable in ("potential", "acceleration", "gradient"):
        adapter_sources = piecewise_tetra_p1_density_sources(field, nodes_via_adapter)
        hand_sources = piecewise_tetra_p1_density_sources(field, hand_built_nodes)
        adapter_report = compare_tetra_p1_density_representations(
            adapter_sources, targets, observable=observable
        )
        hand_report = compare_tetra_p1_density_representations(
            hand_sources, targets, observable=observable
        )
        np.testing.assert_array_equal(
            adapter_report.mass_displacement_response,
            hand_report.mass_displacement_response,
        )
        np.testing.assert_array_equal(
            adapter_report.bulk_interface_response, hand_report.bulk_interface_response
        )


def test_cell_density_nodes_from_group_tags_rejects_hexahedron():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    mesh = VolumeMesh(
        vertices,
        (("hexahedron", np.arange(8).reshape(1, 8)),),
        cell_groups=(np.array([1]),),
    )
    with pytest.raises(ValueError, match="tetra cells only"):
        cell_density_nodes_from_group_tags(mesh, {1: 2.0})


def test_cell_density_nodes_from_group_tags_rejects_mixed_tetra_and_hexahedron():
    tetra_vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    hex_vertices = np.array(
        [
            [2, 0, 0],
            [3, 0, 0],
            [3, 1, 0],
            [2, 1, 0],
            [2, 0, 1],
            [3, 0, 1],
            [3, 1, 1],
            [2, 1, 1],
        ],
        dtype=float,
    )
    vertices = np.vstack([tetra_vertices, hex_vertices])
    mesh = VolumeMesh(
        vertices,
        (
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("hexahedron", np.array([[4, 5, 6, 7, 8, 9, 10, 11]])),
        ),
        cell_groups=(np.array([10]), np.array([20])),
        group_names={"tetra_host": 10, "hex_host": 20},
    )
    with pytest.raises(ValueError, match="tetra cells only"):
        cell_density_nodes_from_group_tags(mesh, {10: 2300.0, 20: 2300.0})


def test_from_meshio_volume_round_trips_a_generated_fixture_file_bit_identically():
    """``tests/fixtures/two_tetra_host_cavity.msh`` is a fixture file generated
    for this test (via ``meshio.write(..., file_format="gmsh22")``); it is not
    a real solver output or measured geometry. It encodes the exact same
    vertices/tetra/tags as
    ``test_cell_density_nodes_from_group_tags_matches_a_hand_built_nc4_array``,
    so reading it back through the real ``from_meshio_volume`` (no monkeypatch)
    must reproduce that hand-built ``VolumeMesh`` bit-identically end to end.
    """
    pytest.importorskip("meshio")
    vertices = np.array(
        [[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0, 0, -1.0]]
    )
    tetra = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
    displacement = np.tile([0.0, 0.0, 1.0], (5, 1))
    targets = np.array([[3.0, 2.0, 1.5]])
    hand_built_mesh = VolumeMesh(
        vertices,
        (("tetra", tetra),),
        cell_groups=(np.array([100, 200]),),
        group_names={"host": 100, "cavity": 200},
    )

    file_mesh = from_meshio_volume(FIXTURES_DIR / "two_tetra_host_cavity.msh")

    assert isinstance(file_mesh, VolumeMesh)
    np.testing.assert_array_equal(file_mesh.vertices_m, hand_built_mesh.vertices_m)
    np.testing.assert_array_equal(
        file_mesh.cell_blocks[0][1], hand_built_mesh.cell_blocks[0][1]
    )
    np.testing.assert_array_equal(
        file_mesh.cell_groups[0], hand_built_mesh.cell_groups[0]
    )
    assert file_mesh.group_names == {"host": 100, "cavity": 200}

    density_by_tag = {100: 2300.0, 200: 0.0}
    file_nodes = cell_density_nodes_from_group_tags(file_mesh, density_by_tag)
    hand_nodes = cell_density_nodes_from_group_tags(hand_built_mesh, density_by_tag)
    np.testing.assert_array_equal(file_nodes, hand_nodes)

    file_field = to_unstructured_displacement_field(file_mesh, displacement)
    hand_field = UnstructuredDisplacementField(vertices, tetra, displacement)

    for observable in OBSERVABLES:
        file_sources = piecewise_tetra_p1_density_sources(file_field, file_nodes)
        hand_sources = piecewise_tetra_p1_density_sources(hand_field, hand_nodes)
        file_report = compare_tetra_p1_density_representations(
            file_sources, targets, observable=observable
        )
        hand_report = compare_tetra_p1_density_representations(
            hand_sources, targets, observable=observable
        )
        np.testing.assert_array_equal(
            file_report.mass_displacement_response,
            hand_report.mass_displacement_response,
        )
        np.testing.assert_array_equal(
            file_report.bulk_interface_response, hand_report.bulk_interface_response
        )


def test_wave6_a_host_cavity_connectivity_matches_between_file_and_memory_mesh(
    tmp_path,
):
    """(a) Connectivity comparison for the wave6-A host+cavity representative
    case: a hand-built ``VolumeMesh`` and the same mesh read back from a real
    gmsh file must be bit-identical through the whole P1 source pipeline.
    """
    pytest.importorskip("meshio")
    volume_mesh = build_host_cavity_volume_mesh()
    mesh_path = write_host_cavity_gmsh_file(volume_mesh, tmp_path / "host_cavity.msh")
    file_mesh = from_meshio_volume(mesh_path)

    connectivity = compare_connectivity(volume_mesh, file_mesh, wavelength_m=20.0)

    assert all(connectivity.values()), connectivity


@pytest.mark.parametrize("observable", OBSERVABLES)
def test_wave6_a_host_cavity_representation_comparison_reports_separately(
    tmp_path, observable
):
    """(c) Physics representation comparison for the wave6-A host+cavity
    representative case: mass-displacement and bulk+interface responses are
    reported separately (never summed), reusing the existing
    ``compare_tetra_p1_density_representations`` contract unmodified.
    """
    pytest.importorskip("meshio")
    volume_mesh = build_host_cavity_volume_mesh()
    mesh_path = write_host_cavity_gmsh_file(volume_mesh, tmp_path / "host_cavity.msh")
    file_mesh = from_meshio_volume(mesh_path)
    sources = build_mass_displacement_sources(file_mesh, wavelength_m=20.0)
    targets = np.array(
        ((2.31, -0.27, -11.81), (7.5, 0.37, -12.23), (24.0, 3.0, -14.0)),
    )

    report = compare_tetra_p1_density_representations(
        sources, targets, observable=observable
    )

    assert report.mass_displacement_response.shape[0] == len(targets)
    assert report.bulk_interface_response.shape[0] == len(targets)
    assert np.isfinite(report.mass_displacement_response).all()
    assert np.isfinite(report.bulk_interface_response).all()
    # The two representations are reported side by side, never added.
    assert not np.array_equal(
        report.bulk_interface_response, report.mass_displacement_response
    )
    assert np.isfinite(report.rel_residual).all()
