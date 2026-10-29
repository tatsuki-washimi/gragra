"""Finite tetra-body representation and scientific-regression tests."""

from dataclasses import replace

import numpy as np
import pytest

from gragra.analytic import cuboid_filled_internal_field
from gragra.fields.finite_body import (
    _oriented_face_topology,
    compare_tetra_linearized_representations,
    piecewise_tetra_linear_sources,
)
from gragra.fields.unstructured import UnstructuredDisplacementField
from gragra.gravity_sources import PointMassCloud, PrescribedRigidTransform
from gragra.observables import (
    displacement_transfer_acceleration,
    displacement_transfer_gravity_gradient,
    displacement_transfer_potential,
    rigid_transform_acceleration,
    rigid_transform_gravity_gradient,
    rigid_transform_potential,
)
from tests._finite_body_reference import (
    TARGETS,
    A,
    B,
    continuum_oracle,
    displacement,
    response_norm,
    static_piecewise_acceleration,
    structured_two_density_tetra,
    target_scales,
)


def test_affine_cube_bulk_interface_mass_is_conserved(cube_vertices, cube_tetra):
    displacement_values = B + cube_vertices @ A.T
    field = UnstructuredDisplacementField(
        cube_vertices, cube_tetra, displacement_values
    )

    sources = piecewise_tetra_linear_sources(field, 2.0)

    bulk_mass = sources.bulk_source.as_weighted_source().weights_kg
    interface_mass = sources.interface_source.weights_kg
    scale = np.sum(np.abs(bulk_mass)) + np.sum(np.abs(interface_mass))
    assert (
        abs(np.sum(bulk_mass) + np.sum(interface_mass))
        <= 256 * np.finfo(float).eps * scale
    )
    assert sources.mass_positions_m.shape == (6, 3)
    assert sources.mass_kg.shape == (6,)
    assert sources.mass_displacement_m.shape == (6, 3)
    assert sources.interface_density_jump_kg_m3.shape == (12,)
    assert np.all(sources.positive_side_cell == -1)
    for array in (
        sources.mass_positions_m,
        sources.mass_kg,
        sources.mass_displacement_m,
        sources.interface_density_jump_kg_m3,
        sources.negative_side_cell,
        sources.positive_side_cell,
    ):
        assert array.flags.writeable is False


@pytest.mark.parametrize("motion", ["translation", "expansion"])
def test_translation_and_expansion_obey_discrete_mass_balance(
    cube_vertices, cube_tetra, motion
):
    alpha = 0.02
    displacement_values = (
        np.tile(B, (len(cube_vertices), 1))
        if motion == "translation"
        else alpha * cube_vertices
    )
    sources = piecewise_tetra_linear_sources(
        UnstructuredDisplacementField(cube_vertices, cube_tetra, displacement_values),
        2.0,
    )
    np.testing.assert_allclose(
        sources.bulk_source.volumes_m3,
        np.full(6, 8.0 / 6.0),
        rtol=5e-13,
        atol=0.0,
    )
    assert np.sum(sources.mass_kg) == pytest.approx(16.0, rel=5e-13)

    bulk_mass = sources.bulk_source.as_weighted_source().weights_kg
    interface_mass = sources.interface_source.weights_kg
    scale = np.sum(np.abs(bulk_mass)) + np.sum(np.abs(interface_mass))
    residual = abs(np.sum(bulk_mass) + np.sum(interface_mass))
    if scale == 0.0:
        assert residual == 0.0
    else:
        assert residual <= 256 * np.finfo(float).eps * scale
    if motion == "translation":
        np.testing.assert_array_equal(bulk_mass, 0.0)
    else:
        assert np.sum(bulk_mass) == pytest.approx(-0.96, rel=5e-13)
        assert np.sum(interface_mass) == pytest.approx(0.96, rel=5e-13)


def test_comparison_supports_all_three_g_free_observables(cube_vertices, cube_tetra):
    displacement = np.tile([0.01, -0.02, 0.03], (len(cube_vertices), 1))
    sources = piecewise_tetra_linear_sources(
        UnstructuredDisplacementField(cube_vertices, cube_tetra, displacement),
        2.0,
    )
    targets = np.array([[3.0, 2.0, 4.0], [-4.0, 3.0, 2.0]])

    expected_shapes = {
        "potential": (2,),
        "acceleration": (2, 3),
        "gradient": (2, 3, 3),
    }
    for observable, shape in expected_shapes.items():
        report = compare_tetra_linearized_representations(
            sources, targets, observable=observable, chunk_size=3
        )
        assert report.bulk_response.shape == shape
        assert report.interface_response.shape == shape
        assert report.bulk_interface_response.shape == shape
        assert report.mass_displacement_response.shape == shape
        assert report.abs_residual.shape == (2,)
        assert report.rel_residual.shape == (2,)
        for array in (
            report.bulk_response,
            report.interface_response,
            report.bulk_interface_response,
            report.mass_displacement_response,
            report.abs_residual,
            report.rel_residual,
        ):
            assert array.flags.writeable is False


def test_piecewise_density_internal_faces_are_explicit_and_mass_conserving():
    vertices, tetra, density = structured_two_density_tetra(2)
    sources = piecewise_tetra_linear_sources(
        UnstructuredDisplacementField(vertices, tetra, displacement(vertices)), density
    )

    assert np.sum(sources.mass_kg) == pytest.approx(28.0, rel=5e-13)
    internal = sources.positive_side_cell >= 0
    assert np.any(internal)
    np.testing.assert_allclose(
        np.abs(sources.interface_density_jump_kg_m3[internal]), 3.0
    )
    bulk = sources.bulk_source.as_weighted_source().weights_kg
    interface = sources.interface_source.weights_kg
    scale = np.sum(np.abs(bulk)) + np.sum(np.abs(interface))
    assert abs(np.sum(bulk) + np.sum(interface)) <= 256 * np.finfo(float).eps * scale

    tetra_centroids = np.mean(vertices[tetra], axis=1)
    for face_index in np.flatnonzero(internal):
        triangle = sources.interface_field.triangles[face_index]
        points = vertices[triangle]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        negative = sources.negative_side_cell[face_index]
        positive = sources.positive_side_cell[face_index]
        assert negative < positive
        assert (
            np.dot(normal, tetra_centroids[positive] - tetra_centroids[negative]) > 0.0
        )
        assert sources.interface_density_jump_kg_m3[face_index] == pytest.approx(
            density[negative] - density[positive]
        )


def test_density_independent_topology_retains_legacy_filtered_order():
    vertices, tetra, density = structured_two_density_tetra(2)
    field = UnstructuredDisplacementField(vertices, tetra, displacement(vertices))
    sources = piecewise_tetra_linear_sources(field, density)
    faces, negative, positive = _oriented_face_topology(field)
    positive_density = np.zeros_like(density[negative])
    internal = positive >= 0
    positive_density[internal] = density[positive[internal]]
    jumps = density[negative] - positive_density
    keep = (positive < 0) | (jumps != 0.0)
    np.testing.assert_array_equal(faces[keep], sources.interface_field.triangles)
    np.testing.assert_array_equal(negative[keep], sources.negative_side_cell)
    np.testing.assert_array_equal(positive[keep], sources.positive_side_cell)
    np.testing.assert_array_equal(jumps[keep], sources.interface_density_jump_kg_m3)


@pytest.mark.parametrize(
    "density_zero,displacement_zero", [(True, False), (False, True)]
)
def test_zero_inputs_give_exact_zero_all_paths(density_zero, displacement_zero):
    vertices, tetra, density = structured_two_density_tetra(2)
    if density_zero:
        density = np.zeros_like(density)
    disp = np.zeros_like(vertices) if displacement_zero else displacement(vertices)
    sources = piecewise_tetra_linear_sources(
        UnstructuredDisplacementField(vertices, tetra, disp), density
    )
    for observable in ("potential", "acceleration", "gradient"):
        report = compare_tetra_linearized_representations(
            sources, TARGETS, observable=observable
        )
        np.testing.assert_array_equal(report.bulk_interface_response, 0.0)
        np.testing.assert_array_equal(report.mass_displacement_response, 0.0)
        np.testing.assert_array_equal(report.abs_residual, 0.0)
        np.testing.assert_array_equal(report.rel_residual, 0.0)


def test_complex_displacement_is_linear():
    vertices, tetra, density = structured_two_density_tetra(2)
    real = displacement(vertices)
    complex_sources = piecewise_tetra_linear_sources(
        UnstructuredDisplacementField(vertices, tetra, (1.0 + 2.0j) * real), density
    )
    real_sources = piecewise_tetra_linear_sources(
        UnstructuredDisplacementField(vertices, tetra, real), density
    )
    for observable in ("potential", "acceleration", "gradient"):
        got = compare_tetra_linearized_representations(
            complex_sources, TARGETS, observable=observable
        )
        expected = compare_tetra_linearized_representations(
            real_sources, TARGETS, observable=observable
        )
        np.testing.assert_allclose(
            got.bulk_interface_response,
            (1.0 + 2.0j) * expected.bulk_interface_response,
            rtol=1e-12,
            atol=0.0,
        )
        np.testing.assert_allclose(
            got.mass_displacement_response,
            (1.0 + 2.0j) * expected.mass_displacement_response,
            rtol=1e-12,
            atol=0.0,
        )


def test_invalid_tetra_topology_is_rejected():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.2, 0.2, 1.0],
            [0.2, 0.2, -1.0],
        ]
    )
    with pytest.raises(ValueError, match="duplicate tetrahedra"):
        piecewise_tetra_linear_sources(
            UnstructuredDisplacementField(
                vertices,
                np.array([[0, 1, 2, 3], [2, 1, 0, 3]]),
                np.zeros_like(vertices),
            ),
            1.0,
        )
    with pytest.raises(ValueError, match="same side|ambiguous"):
        piecewise_tetra_linear_sources(
            UnstructuredDisplacementField(
                vertices,
                np.array([[0, 1, 2, 3], [0, 1, 2, 4]]),
                np.zeros_like(vertices),
            ),
            1.0,
        )
    with pytest.raises(ValueError, match="more than two"):
        piecewise_tetra_linear_sources(
            UnstructuredDisplacementField(
                vertices,
                np.array([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]]),
                np.zeros_like(vertices),
            ),
            1.0,
        )


def test_boundary_that_is_not_a_closed_manifold_is_rejected():
    # Two otherwise valid tetrahedra touch along one edge only.  Their union's
    # boundary has a non-manifold edge and is not one finite closed body mesh.
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    field = UnstructuredDisplacementField(
        vertices,
        np.array([[0, 1, 2, 3], [0, 1, 4, 5]]),
        np.zeros_like(vertices),
    )
    with pytest.raises(ValueError, match="closed manifold"):
        piecewise_tetra_linear_sources(field, 1.0)


@pytest.mark.parametrize("density", [np.empty(0), 0.0, 1.0])
def test_empty_tetra_body_is_supported(density):
    field = UnstructuredDisplacementField(
        np.empty((0, 3)), np.empty((0, 4), dtype=np.int64), np.empty((0, 3))
    )
    sources = piecewise_tetra_linear_sources(field, density)
    assert sources.mass_positions_m.shape == (0, 3)
    assert sources.interface_field.triangles.shape == (0, 3)
    for observable, shape in (
        ("potential", (2,)),
        ("acceleration", (2, 3)),
        ("gradient", (2, 3, 3)),
    ):
        report = compare_tetra_linearized_representations(
            sources, TARGETS, observable=observable
        )
        assert report.mass_displacement_response.shape == shape
        np.testing.assert_array_equal(report.mass_displacement_response, 0.0)


@pytest.mark.parametrize("density", [-1.0, np.nan, np.inf])
def test_empty_tetra_body_still_rejects_invalid_scalar_density(density):
    field = UnstructuredDisplacementField(
        np.empty((0, 3)), np.empty((0, 4), dtype=np.int64), np.empty((0, 3))
    )
    with pytest.raises(ValueError):
        piecewise_tetra_linear_sources(field, density)


@pytest.mark.parametrize("field_name", ["negative_side_cell", "positive_side_cell"])
def test_source_metadata_rejects_cell_indices_above_cell_count(field_name):
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    sources = piecewise_tetra_linear_sources(
        UnstructuredDisplacementField(
            vertices, np.array([[0, 1, 2, 3]]), np.zeros_like(vertices)
        ),
        1.0,
    )
    bad_indices = np.full(len(sources.interface_density_jump_kg_m3), 42)
    with pytest.raises(ValueError, match="outside their valid range"):
        replace(sources, **{field_name: bad_indices})


def test_empty_targets_and_backend_errors_follow_existing_dispatch():
    vertices, tetra, density = structured_two_density_tetra(2)
    sources = piecewise_tetra_linear_sources(
        UnstructuredDisplacementField(vertices, tetra, displacement(vertices)), density
    )
    empty = compare_tetra_linearized_representations(
        sources, np.empty((0, 3)), observable="gradient"
    )
    assert empty.mass_displacement_response.shape == (0, 3, 3)
    with pytest.raises(ValueError, match="Unsupported observable"):
        compare_tetra_linearized_representations(sources, TARGETS, observable="unknown")
    with pytest.raises(NotImplementedError, match="does not implement"):
        compare_tetra_linearized_representations(
            sources, TARGETS, observable="potential", backend="cpp"
        )


@pytest.mark.parametrize(
    "density,error",
    [
        (True, TypeError),
        (1.0 + 1.0j, TypeError),
        (-1.0, ValueError),
        (np.nan, ValueError),
        (np.ones(3), ValueError),
    ],
)
def test_density_validation(density, error):
    vertices, tetra, _ = structured_two_density_tetra(2)
    field = UnstructuredDisplacementField(vertices, tetra, displacement(vertices))
    with pytest.raises(error):
        piecewise_tetra_linear_sources(field, density)


@pytest.mark.parametrize("observable", ["potential", "acceleration", "gradient"])
def test_fixed_refinement_converges_to_literal_continuum_oracle(observable):
    oracle10 = continuum_oracle(10, TARGETS, observable)
    oracle12 = continuum_oracle(12, TARGETS, observable)
    scales = target_scales(observable)
    oracle_delta = response_norm(oracle12 - oracle10, observable) / scales
    assert np.max(oracle_delta) <= 1e-10

    representation_errors = []
    dipole_errors = []
    bulk_errors = []
    for n in (2, 4, 8):
        vertices, tetra, density = structured_two_density_tetra(n)
        sources = piecewise_tetra_linear_sources(
            UnstructuredDisplacementField(vertices, tetra, displacement(vertices)),
            density,
        )
        report = compare_tetra_linearized_representations(
            sources, TARGETS, observable=observable, chunk_size=37
        )
        representation_errors.append(np.max(report.abs_residual / scales))
        dipole_errors.append(
            np.max(
                response_norm(report.mass_displacement_response - oracle12, observable)
                / scales
            )
        )
        bulk_errors.append(
            np.max(
                response_norm(report.bulk_interface_response - oracle12, observable)
                / scales
            )
        )

    for errors in (representation_errors, dipole_errors, bulk_errors):
        assert np.isfinite(errors).all()
        assert errors[-1] < errors[0]
        assert errors[-1] / errors[-2] <= 0.60
        assert errors[-1] <= 0.02


def test_literal_static_box_quadrature_matches_existing_analytic_acceleration():
    literal = static_piecewise_acceleration(12, TARGETS)
    analytic = cuboid_filled_internal_field(
        [1.0, 2.0, 2.0], TARGETS - np.array([-0.5, 0.0, 0.0]), 2.0
    ) + cuboid_filled_internal_field(
        [1.0, 2.0, 2.0], TARGETS - np.array([0.5, 0.0, 0.0]), 5.0
    )
    np.testing.assert_allclose(literal, analytic, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "rigid_observable,linear_observable",
    [
        (rigid_transform_potential, displacement_transfer_potential),
        (rigid_transform_acceleration, displacement_transfer_acceleration),
        (
            rigid_transform_gravity_gradient,
            displacement_transfer_gravity_gradient,
        ),
    ],
)
def test_finite_rigid_motion_has_quadratic_taylor_remainder(
    rigid_observable, linear_observable
):
    vertices, tetra, density = structured_two_density_tetra(2)
    sources = piecewise_tetra_linear_sources(
        UnstructuredDisplacementField(vertices, tetra, displacement(vertices)), density
    )
    cloud = PointMassCloud(sources.mass_kg, sources.mass_positions_m)
    r_star = np.min(
        np.linalg.norm(
            sources.mass_positions_m[:, np.newaxis, :] - TARGETS[np.newaxis, :, :],
            axis=-1,
        )
    )
    direction = np.array([2.0, -1.0, 3.0]) / np.sqrt(14.0)
    remainders = []
    for relative in (1e-3, 5e-4, 2.5e-4):
        shift = relative * r_star * direction
        transform = PrescribedRigidTransform(
            np.array([0.0, 1.0]),
            np.array([[0.0, 0.0, 0.0], shift]),
            np.array([np.eye(3), np.eye(3)]),
        )
        finite = rigid_observable(cloud, transform, TARGETS, chunk_size=5)
        linear = linear_observable(
            sources.mass_positions_m,
            sources.mass_kg,
            np.tile(shift, (len(sources.mass_kg), 1)),
            TARGETS,
            chunk_size=5,
        )
        remainders.append(np.linalg.norm(finite[1] - finite[0] - linear))
    ratios = np.asarray(remainders[:-1]) / np.asarray(remainders[1:])
    assert np.all((ratios >= 3.0) & (ratios <= 5.0))


@pytest.mark.parametrize("observable", ["potential", "acceleration", "gradient"])
def test_chunk_variants_stay_within_fixed_physical_scale(observable):
    vertices, tetra, density = structured_two_density_tetra(4)
    sources = piecewise_tetra_linear_sources(
        UnstructuredDisplacementField(vertices, tetra, displacement(vertices)), density
    )
    baseline = compare_tetra_linearized_representations(
        sources, TARGETS, observable=observable
    ).mass_displacement_response
    scale = np.max(target_scales(observable))
    for chunk_size in (1, 37):
        candidate = compare_tetra_linearized_representations(
            sources, TARGETS, observable=observable, chunk_size=chunk_size
        ).mass_displacement_response
        assert np.max(np.abs(candidate - baseline)) <= 5e-13 * scale
