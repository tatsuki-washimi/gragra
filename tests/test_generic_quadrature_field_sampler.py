"""Tests for generic quadrature plans and field sampling."""

import math

import numpy as np
import pytest

from gragra.fields.finite_body_p1 import piecewise_tetra_p1_density_sources
from gragra.fields.generic_integration import (
    FieldSampler,
    QuadraturePlan,
)
from gragra.fields.surface import SurfaceDisplacementField
from gragra.fields.unstructured import UnstructuredDisplacementField
from gragra.geometry.tetra_quadrature import (
    tetra_gauss_quadrature,
    triangle_gauss_quadrature,
)


def _unit_triangle() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0, 1, 2]], dtype=np.int64),
    )


def _unit_tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.array([[0, 1, 2, 3]], dtype=np.int64),
    )


def test_surface_plan_scales_triangle_rule_and_keeps_winding_normal() -> None:
    vertices, triangles = _unit_triangle()
    original_vertices = vertices.copy()
    plan = QuadraturePlan.from_surface_mesh(vertices, triangles, order=3)
    barycentric, normalized_weights = triangle_gauss_quadrature(3)

    assert plan.measure == "surface"
    assert plan.normals.shape == plan.points_m.shape == (9, 3)
    assert plan.element_ids.dtype == np.int64
    np.testing.assert_allclose(plan.points_m, barycentric @ vertices)
    np.testing.assert_allclose(plan.weights, normalized_weights * 0.5)
    np.testing.assert_allclose(plan.normals, np.tile([0.0, 0.0, 1.0], (9, 1)))
    np.testing.assert_array_equal(plan.element_ids, np.zeros(9, dtype=np.int64))
    np.testing.assert_array_equal(vertices, original_vertices)
    assert not plan.points_m.flags.writeable
    assert not plan.weights.flags.writeable
    assert not plan.normals.flags.writeable
    assert not plan.element_ids.flags.writeable


def test_volume_plan_scales_tetrahedron_rule_and_integrates_affine_geometry() -> None:
    vertices, tetra = _unit_tetrahedron()
    plan = QuadraturePlan.from_tetra_mesh(vertices, tetra, order=2)
    barycentric, normalized_weights = tetra_gauss_quadrature(2)

    assert plan.measure == "volume"
    assert plan.normals is None
    assert plan.points_m.shape == (8, 3)
    np.testing.assert_allclose(plan.points_m, barycentric @ vertices)
    np.testing.assert_allclose(plan.weights, normalized_weights / 6.0)
    assert plan.weights.sum() == pytest.approx(1.0 / 6.0)
    np.testing.assert_allclose(plan.weights @ plan.points_m, np.full(3, 1.0 / 24.0))
    np.testing.assert_array_equal(plan.element_ids, np.zeros(8, dtype=np.int64))
    assert not plan.points_m.flags.writeable
    assert not plan.weights.flags.writeable
    assert not plan.element_ids.flags.writeable


def test_surface_sampler_uses_pointwise_linear_complex_displacement() -> None:
    vertices, triangles = _unit_triangle()
    displacement = np.array(
        [[1.0 + 2.0j, 0.0, 1.0j], [0.0, 2.0j, 0.0], [3.0, 1.0j, 2.0]]
    )
    field = SurfaceDisplacementField(vertices, triangles, displacement)
    plan = QuadraturePlan.from_surface_mesh(vertices, triangles, order=3)
    barycentric, _ = triangle_gauss_quadrature(3)

    sampled = FieldSampler().sample_surface_displacement(field, plan)

    assert sampled.shape == (9, 1, 3)
    assert sampled.dtype == np.complex128
    np.testing.assert_allclose(sampled[:, 0], barycentric @ displacement[triangles[0]])
    assert not sampled.flags.writeable


def test_surface_sampler_preserves_complex_parameter_axis() -> None:
    vertices, triangles = _unit_triangle()
    field = SurfaceDisplacementField(vertices, triangles, np.zeros((3, 3)))
    displacement_parameters = np.array(
        [
            [[1.0 + 2.0j, 0.0, 1.0j], [2.0 - 1.0j, 3.0j, 0.0]],
            [[0.0, 2.0j, 0.0], [1.0j, 1.0, 0.0]],
            [[3.0, 1.0j, 2.0], [4.0j, -1.0j, 2.0]],
        ]
    )
    plan = QuadraturePlan.from_surface_mesh(vertices, triangles, order=3)
    barycentric, _ = triangle_gauss_quadrature(3)

    sampled = FieldSampler().sample_surface_displacement(
        field, plan, displacement_m=displacement_parameters
    )

    assert sampled.shape == (9, 2, 3)
    assert sampled.dtype == np.complex128
    np.testing.assert_allclose(
        sampled, np.einsum("ql,lpi->qpi", barycentric, displacement_parameters)
    )
    assert not sampled.flags.writeable


def test_tetra_sampler_matches_existing_p1_density_quadrature() -> None:
    vertices, tetra = _unit_tetrahedron()
    displacement = np.zeros((4, 3), dtype=np.float64)
    field = UnstructuredDisplacementField(vertices, tetra, displacement)
    density = np.array([[1.0, 2.0, 3.0, 4.0]])
    plan = QuadraturePlan.from_tetra_mesh(vertices, tetra, order=3)
    barycentric, _ = tetra_gauss_quadrature(3)

    sampled = FieldSampler().sample_tetra_p1_density(field, density, plan)
    existing = piecewise_tetra_p1_density_sources(field, density, quadrature_order=3)

    assert sampled.shape == (27, 1)
    assert sampled.dtype == np.float64
    np.testing.assert_allclose(sampled[:, 0], barycentric @ density[0])
    np.testing.assert_allclose(sampled[:, 0] * plan.weights, existing.mass_kg)
    assert not sampled.flags.writeable


def test_tetra_sampler_rejects_incorrect_sub_tolerance_quadrature_weights() -> None:
    side = 5e-5
    vertices = side * np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    tetra = np.array([[0, 1, 2, 3]], dtype=np.int64)
    field = UnstructuredDisplacementField(vertices, tetra, np.zeros((4, 3)))
    expected = QuadraturePlan.from_tetra_mesh(vertices, tetra, order=2)
    invalid = QuadraturePlan(
        points_m=expected.points_m,
        weights=np.zeros_like(expected.weights),
        normals=None,
        measure="volume",
        element_ids=expected.element_ids,
        quadrature_order=expected.quadrature_order,
    )

    assert np.allclose(invalid.weights, expected.weights, rtol=0.0, atol=2e-14)
    with pytest.raises(ValueError, match="plan geometry does not match"):
        FieldSampler().sample_tetra_p1_density(field, np.ones((1, 4)), invalid)


@pytest.mark.parametrize("active_axes", [(0, 1), (0, 2), (1, 2)])
@pytest.mark.parametrize("height_m", [1e-6, 1e-8, 1e-9])
def test_surface_sampler_reconstructs_p1_displacement_on_skinny_triangles(
    height_m: float, active_axes: tuple[int, int]
) -> None:
    planar_vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, height_m]])
    vertices = np.zeros((3, 3))
    vertices[:, active_axes] = planar_vertices
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    displacement = np.column_stack((np.zeros(3), np.zeros(3), [0.0, 1.0, 2.0]))
    field = SurfaceDisplacementField(vertices, triangles, displacement)
    plan = QuadraturePlan.from_surface_mesh(vertices, triangles, order=2)
    barycentric, _ = triangle_gauss_quadrature(2)

    sampled = FieldSampler().sample_surface_displacement(field, plan)

    np.testing.assert_allclose(
        sampled[:, 0], barycentric @ displacement[triangles[0]], rtol=0.0, atol=1e-12
    )


def test_surface_sampler_rejects_points_off_the_indexed_triangle() -> None:
    vertices, triangles = _unit_triangle()
    field = SurfaceDisplacementField(vertices, triangles, np.zeros((3, 3)))
    plan = QuadraturePlan(
        points_m=np.array([[0.2, 0.2, 1.0]]),
        weights=np.array([0.5]),
        normals=np.array([[0.0, 0.0, 1.0]]),
        measure="surface",
        element_ids=np.array([0], dtype=np.int64),
        quadrature_order=2,
    )

    with pytest.raises(ValueError, match="lie on their indexed surface elements"):
        FieldSampler().sample_surface_displacement(field, plan)


@pytest.mark.parametrize("translation_x", [0.0, 1e8])
def test_surface_sampler_plane_validation_is_translation_invariant(
    translation_x: float,
) -> None:
    vertices, triangles = _unit_triangle()
    vertices[:, 0] += translation_x
    field = SurfaceDisplacementField(vertices, triangles, np.zeros((3, 3)))
    valid_plan = QuadraturePlan.from_surface_mesh(vertices, triangles, order=2)

    sampled = FieldSampler().sample_surface_displacement(field, valid_plan)

    assert sampled.shape == (4, 1, 3)
    off_plane_plan = QuadraturePlan(
        points_m=np.array([[translation_x + 0.25, 0.25, 1e-6]]),
        weights=np.array([0.5]),
        normals=np.array([[0.0, 0.0, 1.0]]),
        measure="surface",
        element_ids=np.array([0], dtype=np.int64),
        quadrature_order=2,
    )
    with pytest.raises(ValueError, match="lie on their indexed surface elements"):
        FieldSampler().sample_surface_displacement(field, off_plane_plan)


def test_surface_sampler_accepts_roundoff_for_translated_oblique_face() -> None:
    normal = np.array([1.0, 2.0, 3.0])
    normal /= np.linalg.norm(normal)
    first_edge = np.array([1.0, -1.0, 0.0])
    first_edge /= np.linalg.norm(first_edge)
    second_edge = np.cross(normal, first_edge)
    second_edge /= np.linalg.norm(second_edge)
    vertices = np.array([[0.0, 0.0, 0.0], first_edge, second_edge])
    vertices += np.array([1e8, -1e8, 1e8])
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    field = SurfaceDisplacementField(vertices, triangles, np.zeros((3, 3)))
    plan = QuadraturePlan.from_surface_mesh(vertices, triangles, order=3)

    sampled = FieldSampler().sample_surface_displacement(field, plan)

    assert sampled.shape == (9, 1, 3)


@pytest.mark.parametrize("translation_m", [0.0, 1e8])
@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_surface_sampler_reuses_reference_coordinates_for_translated_skinny_triangle(
    translation_m: float, order: int
) -> None:
    vertices = (
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1e-6, 0.0]]) + translation_m
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    nodal_displacement = np.arange(18, dtype=np.float64).reshape(3, 2, 3)
    nodal_displacement = nodal_displacement * (1.0 + 0.5j)
    field = SurfaceDisplacementField(vertices, triangles, np.zeros((3, 3)))
    plan = QuadraturePlan.from_surface_mesh(vertices, triangles, order=order)
    reference_barycentric, _ = triangle_gauss_quadrature(order)

    sampled = FieldSampler().sample_surface_displacement(
        field, plan, displacement_m=nodal_displacement
    )

    expected = np.einsum("ql,lpi->qpi", reference_barycentric, nodal_displacement)
    assert sampled.shape == (order**2, 2, 3)
    assert sampled.dtype == np.complex128
    np.testing.assert_allclose(sampled, expected, rtol=0.0, atol=1e-12)


def test_surface_sampler_rejects_unresolved_external_plan_precision() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1e-6, 0.0]]) + 1e8
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    field = SurfaceDisplacementField(vertices, triangles, np.zeros((3, 3)))
    generated = QuadraturePlan.from_surface_mesh(vertices, triangles, order=8)
    external = QuadraturePlan(
        points_m=generated.points_m,
        weights=generated.weights,
        normals=generated.normals,
        measure="surface",
        element_ids=generated.element_ids,
        quadrature_order=generated.quadrature_order,
    )

    with pytest.raises(ValueError, match="barycentric coordinates cannot be resolved"):
        FieldSampler().sample_surface_displacement(field, external)


def test_surface_sampler_rejects_reference_rule_for_different_mesh_geometry() -> None:
    vertices, triangles = _unit_triangle()
    plan = QuadraturePlan.from_surface_mesh(vertices, triangles, order=2)
    mismatched_vertices = vertices.copy()
    mismatched_vertices[2, 1] += 0.1
    field = SurfaceDisplacementField(mismatched_vertices, triangles, np.zeros((3, 3)))

    with pytest.raises(ValueError, match="plan geometry does not match"):
        FieldSampler().sample_surface_displacement(field, plan)


def test_surface_sampler_binds_reference_rule_to_exact_source_geometry() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1e-6, 0.0]]) + 1e8
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    plan = QuadraturePlan.from_surface_mesh(vertices, triangles, order=2)
    changed_vertices = vertices.copy()
    changed_vertices[2, 1] = np.nextafter(changed_vertices[2, 1], np.inf)
    changed_plan = QuadraturePlan.from_surface_mesh(
        changed_vertices, triangles, order=2
    )
    field = SurfaceDisplacementField(changed_vertices, triangles, np.zeros((3, 3)))

    np.testing.assert_array_equal(plan.points_m, changed_plan.points_m)
    np.testing.assert_allclose(plan.normals, changed_plan.normals, rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="source surface geometry"):
        FieldSampler().sample_surface_displacement(field, plan)


def test_surface_sampler_accepts_empty_surface_plan() -> None:
    vertices = np.empty((0, 3), dtype=np.float64)
    triangles = np.empty((0, 3), dtype=np.int64)
    field = SurfaceDisplacementField(vertices, triangles, vertices)
    plan = QuadraturePlan.from_surface_mesh(vertices, triangles, order=2)

    sampled = FieldSampler().sample_surface_displacement(field, plan)

    assert plan.points_m.shape == (0, 3)
    assert plan.weights.shape == plan.element_ids.shape == (0,)
    assert sampled.shape == (0, 1, 3)


def test_surface_sampler_rejects_external_point_outside_indexed_triangle() -> None:
    vertices, triangles = _unit_triangle()
    field = SurfaceDisplacementField(vertices, triangles, np.zeros((3, 3)))
    external = QuadraturePlan(
        points_m=np.array([[1.25, 0.25, 0.0]]),
        weights=np.array([0.5]),
        normals=np.array([[0.0, 0.0, 1.0]]),
        measure="surface",
        element_ids=np.array([0], dtype=np.int64),
        quadrature_order=2,
    )

    with pytest.raises(ValueError, match="lie in their indexed surface elements"):
        FieldSampler().sample_surface_displacement(field, external)


def test_surface_quadrature_rejects_triangle_lost_to_coordinate_rounding() -> None:
    base = 1e8
    vertices = np.array(
        [[base, base, base], [base + 1.0, base, base], [base + 1.0, base + 1e-10, base]]
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    np.testing.assert_array_equal(vertices[1], vertices[2])

    with pytest.raises(ValueError, match="degenerate zero-area triangle"):
        QuadraturePlan.from_surface_mesh(vertices, triangles, order=2)


@pytest.mark.parametrize(
    ("measure", "exact_integral"),
    [("surface", math.e - 2.0), ("volume", math.e - 2.5)],
)
def test_exponential_quadrature_refinement_meets_predeclared_error_budget(
    measure: str, exact_integral: float
) -> None:
    """Order 8 must reach 2e-13 relative-to-measure independent-reference error."""
    errors = []
    for order in (2, 4, 6, 8):
        if measure == "surface":
            vertices, elements = _unit_triangle()
            plan = QuadraturePlan.from_surface_mesh(vertices, elements, order=order)
        else:
            vertices, elements = _unit_tetrahedron()
            plan = QuadraturePlan.from_tetra_mesh(vertices, elements, order=order)
        integral = float(plan.weights @ np.exp(plan.points_m[:, 0]))
        errors.append(abs(integral - exact_integral))

    integration_measure = 0.5 if measure == "surface" else 1.0 / 6.0
    assert errors[-1] / integration_measure <= 2e-13
