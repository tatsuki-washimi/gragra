import numpy as np
import pytest

from gragra.fields.unstructured import UnstructuredDisplacementField
from gragra.gravity_sources.volume import VolumeElementSource
from gragra.observables.acceleration import point_acceleration
from gragra.targets.points import TargetPoints


def test_unstructured_field_basic():
    # Define a single tetrahedron
    # Vertex coordinates
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    # Tetra indices
    tetra = np.array([[0, 1, 2, 3]])

    # Let A be a simple diagonal scaling matrix: A = diag(0.1, 0.2, 0.3)
    # trace(A) = 0.6
    # Let b = [1.0, -2.0, 3.0]
    a_mat = np.diag([0.1, 0.2, 0.3])
    b = np.array([1.0, -2.0, 3.0])

    disp = vertices @ a_mat.T + b

    rho0 = 1000.0
    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=disp
    )

    source = field.as_density_perturbation_source(rho0)
    assert isinstance(source, VolumeElementSource)

    # Check shape/values
    assert len(source.positions_m) == 1
    assert np.allclose(source.positions_m[0], [0.25, 0.25, 0.25])
    assert np.isclose(source.volumes_m3[0], 1.0 / 6.0)

    # div(u) = 0.6. delta_rho = -rho0 * div(u) = -600.0
    assert np.allclose(source.density_kg_m3, -600.0)

    # Read-only verification
    with pytest.raises(ValueError):
        source.volumes_m3[0] = 9.9
    with pytest.raises(ValueError):
        source.positions_m[0, 0] = 9.9
    with pytest.raises(ValueError):
        source.density_kg_m3[0] = 9.9

    with pytest.raises(ValueError):
        field.vertices_m[0, 0] = 9.9
    with pytest.raises(ValueError):
        field.tetra[0, 0] = 9
    with pytest.raises(ValueError):
        field.displacement_m[0, 0] = 9.9


def test_unstructured_field_skew_off_diagonal():
    # A skew tetrahedron that is not axis-aligned
    # Vertices:
    vertices = np.array(
        [
            [0.1, 0.2, 0.3],
            [1.2, 0.4, 0.5],
            [0.3, 1.5, 0.6],
            [0.4, 0.5, 1.7],
        ]
    )
    tetra = np.array([[0, 1, 2, 3]])

    # An affine transformation matrix with non-zero off-diagonal elements
    a_mat = np.array([[0.1, 0.2, -0.15], [0.05, -0.3, 0.1], [-0.2, 0.12, 0.4]])
    # trace(A) = 0.1 - 0.3 + 0.4 = 0.2
    b = np.array([1.5, 2.5, -3.5])

    disp = vertices @ a_mat.T + b
    rho0 = 2500.0

    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=disp
    )
    source = field.as_density_perturbation_source(rho0)

    # delta_rho = -rho0 * trace(A) = -2500.0 * 0.2 = -500.0
    assert np.allclose(source.density_kg_m3, -500.0)


def test_unstructured_field_orientation_invariance():
    # Vertices of a reference tet
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    # We will test different node orderings (which change the determinant sign)
    orderings = [
        [0, 1, 2, 3],  # Det > 0 (standard)
        [0, 2, 1, 3],  # Det < 0 (swapped)
        [3, 2, 1, 0],
        [1, 3, 0, 2],
    ]

    a_mat = np.diag([0.1, 0.2, 0.3])
    disp = vertices @ a_mat.T
    rho0 = 1000.0

    for order in orderings:
        field = UnstructuredDisplacementField(
            vertices_m=vertices, tetra=np.array([order]), displacement_m=disp
        )
        source = field.as_density_perturbation_source(rho0)
        # The determinant changes sign, but we take absolute volume, and
        # the gradient calculation in UnstructuredDisplacementField must be
        # invariant to the vertex ordering.
        assert np.isclose(source.volumes_m3[0], 1.0 / 6.0)
        assert np.allclose(source.density_kg_m3, -600.0)


def test_unstructured_field_constant_translation():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    tetra = np.array([[0, 1, 2, 3]])

    # Constant displacement (pure translation)
    disp = np.ones((4, 3)) * 5.0
    rho0 = 1000.0

    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=disp
    )
    source = field.as_density_perturbation_source(rho0)
    assert np.allclose(source.density_kg_m3, 0.0)


def test_unstructured_field_zero_rho0():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    tetra = np.array([[0, 1, 2, 3]])
    a_mat = np.diag([0.1, 0.2, 0.3])
    disp = vertices @ a_mat.T

    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=disp
    )
    source = field.as_density_perturbation_source(0.0)
    assert np.allclose(source.density_kg_m3, 0.0)


def test_unstructured_field_complex_affine():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    tetra = np.array([[0, 1, 2, 3]])

    # Complex affine transformation matrix
    a_mat = np.array(
        [
            [0.1 + 0.2j, 0.0, 0.0],
            [0.0, -0.3 + 0.05j, 0.0],
            [0.0, 0.0, 0.4 - 0.1j],
        ]
    )
    # trace(A) = 0.2 + 0.15j
    disp = vertices @ a_mat.T
    rho0 = 1000.0

    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=disp
    )

    assert np.iscomplexobj(field.displacement_m)

    source = field.as_density_perturbation_source(rho0)
    assert np.iscomplexobj(source.density_kg_m3)
    assert np.allclose(source.density_kg_m3, -1000.0 * (0.2 + 0.15j))


def test_unstructured_field_gravity_physics():
    # E2E gravity physics check: verify point_acceleration from field
    # matches direct VolumeElementSource
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    tetra = np.array([[0, 1, 2, 3]])
    a_mat = np.diag([0.1, -0.1, 0.05])
    disp = vertices @ a_mat.T
    rho0 = 1200.0

    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=disp
    )
    source_field = field.as_density_perturbation_source(rho0)

    # Manually constructed VolumeElementSource
    source_direct = VolumeElementSource(
        positions_m=source_field.positions_m.copy(),
        volumes_m3=source_field.volumes_m3.copy(),
        density_kg_m3=source_field.density_kg_m3.copy(),
    )

    targets = TargetPoints(np.array([[2.0, 3.0, 4.0]]))

    acc_field = point_acceleration(source_field.as_weighted_source(), targets)
    acc_direct = point_acceleration(source_direct.as_weighted_source(), targets)

    assert np.allclose(acc_field, acc_direct)


def test_unstructured_field_validation():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    tetra = np.array([[0, 1, 2, 3]])
    disp = np.zeros((4, 3))

    # Infinite coordinate
    inf_vertices = vertices.copy()
    inf_vertices[0, 0] = np.inf
    with pytest.raises(ValueError):
        UnstructuredDisplacementField(
            vertices_m=inf_vertices, tetra=tetra, displacement_m=disp
        )

    # Shape mismatch for vertices
    with pytest.raises(ValueError):
        UnstructuredDisplacementField(
            vertices_m=np.zeros(4), tetra=tetra, displacement_m=disp
        )

    # Shape mismatch for displacement
    with pytest.raises(ValueError):
        UnstructuredDisplacementField(
            vertices_m=vertices, tetra=tetra, displacement_m=np.zeros(4)
        )

    # Displacement length mismatch
    with pytest.raises(ValueError):
        UnstructuredDisplacementField(
            vertices_m=vertices, tetra=tetra, displacement_m=np.zeros((5, 3))
        )

    # Non-integer tetra
    with pytest.raises(TypeError):
        UnstructuredDisplacementField(
            vertices_m=vertices,
            tetra=np.array([[0.0, 1.0, 2.0, 3.0]]),
            displacement_m=disp,
        )

    # Bool in tetra
    with pytest.raises(TypeError):
        UnstructuredDisplacementField(
            vertices_m=vertices,
            tetra=np.array([[False, True, True, True]]),
            displacement_m=disp,
        )

    # Tetra index out of bounds
    with pytest.raises(ValueError):
        UnstructuredDisplacementField(
            vertices_m=vertices,
            tetra=np.array([[0, 1, 2, 4]]),
            displacement_m=disp,
        )

    # Degenerate tetrahedron (volume <= 1e-15)
    degen_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    with pytest.raises(ValueError, match="degenerate tetrahedron detected"):
        UnstructuredDisplacementField(
            vertices_m=degen_vertices, tetra=tetra, displacement_m=disp
        )

    # Empty tetra (0, 4) should be allowed
    empty_field = UnstructuredDisplacementField(
        vertices_m=vertices,
        tetra=np.zeros((0, 4), dtype=int),
        displacement_m=disp,
    )
    empty_source = empty_field.as_density_perturbation_source(1000.0)
    assert len(empty_source.positions_m) == 0
    assert len(empty_source.volumes_m3) == 0
    assert len(empty_source.density_kg_m3) == 0

    # Empty list [] (shape (0,)) should be rejected
    with pytest.raises(ValueError):
        UnstructuredDisplacementField(
            vertices_m=vertices, tetra=np.array([]), displacement_m=disp
        )

    # Invalid rho0 validations
    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=disp
    )
    # rho0 must be scalar
    with pytest.raises(TypeError):
        field.as_density_perturbation_source([1000.0])
    # rho0 must be real-valued
    with pytest.raises(TypeError):
        field.as_density_perturbation_source(1000.0 + 1j)
    # rho0 must be finite
    with pytest.raises(ValueError):
        field.as_density_perturbation_source(np.inf)
    # rho0 must be non-negative
    with pytest.raises(ValueError):
        field.as_density_perturbation_source(-100.0)
