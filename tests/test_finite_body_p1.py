"""Cell-local P1-density finite tetra-body tests."""

from dataclasses import fields

import numpy as np
import pytest

from gragra.fields.finite_body import piecewise_tetra_linear_sources
from gragra.fields.finite_body_p1 import (
    TetraP1DensityLinearizedSources,
    compare_tetra_p1_density_representations,
    piecewise_tetra_p1_density_sources,
)
from gragra.fields.unstructured import UnstructuredDisplacementField
from gragra.gravity_sources import (
    PointMassCloud,
    PrescribedRigidTransform,
    WeightedPointSource,
)
from gragra.observables import (
    displacement_transfer_acceleration,
    displacement_transfer_gravity_gradient,
    displacement_transfer_potential,
    rigid_transform_acceleration,
    rigid_transform_gravity_gradient,
    rigid_transform_potential,
)
from tests._finite_body_p1_reference import (
    TARGETS,
    A,
    B,
    affine_density_mass_and_first_moment,
    continuum_oracle,
    displacement,
    response_norm,
    structured_p1_density_tetra,
    target_scales,
    tetra_polynomial_integral,
    triangle_polynomial_integral,
)
from tests._finite_body_reference import (
    continuum_oracle as p0_continuum_oracle,
)
from tests._finite_body_reference import (
    structured_two_density_tetra,
)


def single_tetra(u=None):
    v = np.array([[0.1, -0.2, 0.3], [1.4, 0, 0.2], [0.2, 1.1, 0.1], [-0.1, 0.3, 1.5]])
    return UnstructuredDisplacementField(
        v, np.array([[0, 1, 2, 3]]), displacement(v) if u is None else u
    )


def test_builder_only_and_degree_two_tetra_rule():
    with pytest.raises(TypeError, match="builder"):
        TetraP1DensityLinearizedSources()
    field = single_tetra()
    rho = np.array([[1.7, 2.1, 2.8, 3.4]])
    s = piecewise_tetra_p1_density_sources(field, rho)
    q = np.array([[0.3, 0.04, -0.02], [0.04, -0.1, 0.03], [-0.02, 0.03, 0.05]])
    sample = (
        0.37
        + s.mass_positions_m @ np.array([0.2, -0.11, 0.08])
        + np.einsum("ni,ij,nj->n", s.mass_positions_m, q, s.mass_positions_m)
    )
    expected = tetra_polynomial_integral(
        field.vertices_m[field.tetra[0]], 0.37, np.array([0.2, -0.11, 0.08]), q
    )
    assert np.sum(sample * s.bulk_source.volumes_m3) == pytest.approx(
        expected, rel=1e-12, abs=1e-13
    )
    mass, moment = affine_density_mass_and_first_moment(
        field.vertices_m[field.tetra[0]], rho[0]
    )
    assert np.sum(s.mass_kg) == pytest.approx(mass, rel=1e-12, abs=1e-13)
    np.testing.assert_allclose(
        np.sum(s.mass_kg[:, None] * s.mass_positions_m, axis=0),
        moment,
        rtol=1e-12,
        atol=1e-13,
    )

    face = s.interface_triangles[0]
    triangle = field.vertices_m[face]
    face_points = s.interface_source.positions_m[:3]
    area = (
        np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
        / 2
    )
    face_sample = (
        0.37
        + face_points @ np.array([0.2, -0.11, 0.08])
        + np.einsum("ni,ij,nj->n", face_points, q, face_points)
    )
    face_expected = triangle_polynomial_integral(
        triangle, 0.37, np.array([0.2, -0.11, 0.08]), q
    )
    assert np.sum(face_sample * area / 3) == pytest.approx(
        face_expected, rel=1e-12, abs=1e-13
    )


def test_shapes_indices_readonly_and_all_faces(cube_vertices, cube_tetra):
    field = UnstructuredDisplacementField(
        cube_vertices, cube_tetra, displacement(cube_vertices)
    )
    s = piecewise_tetra_p1_density_sources(field, np.full((6, 4), 2.0))
    nc = 6
    nf = len(s.interface_triangles)
    assert nf == 18 and np.count_nonzero(s.positive_side_cell >= 0) == 6
    assert s.mass_positions_m.shape == (4 * nc, 3)
    assert s.interface_source.positions_m.shape == (3 * nf, 3)
    assert s.interface_density_jump_kg_m3.shape == (nf, 3)
    np.testing.assert_array_equal(s.mass_cell_index, np.repeat(np.arange(nc), 4))
    np.testing.assert_array_equal(
        s.interface_quadrature_face_index, np.repeat(np.arange(nf), 3)
    )
    np.testing.assert_array_equal(s.bulk_source.positions_m, s.mass_positions_m)
    np.testing.assert_allclose(
        s.interface_density_jump_kg_m3[s.positive_side_cell >= 0], 0, atol=1e-15
    )
    for item in fields(s):
        value = getattr(s, item.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
    for array in (
        s.bulk_source.positions_m,
        s.bulk_source.volumes_m3,
        s.interface_source.positions_m,
        s.interface_source.weights_kg,
    ):
        assert not array.flags.writeable
    density_input = np.full((6, 4), 2.0)
    defensive = piecewise_tetra_p1_density_sources(field, density_input)
    density_input[:] = 9.0
    np.testing.assert_array_equal(defensive.cell_density_nodes_kg_m3, 2.0)
    centroids = np.mean(cube_vertices[cube_tetra], axis=1)
    for face, neg, pos in zip(
        s.interface_triangles, s.negative_side_cell, s.positive_side_cell, strict=True
    ):
        if pos >= 0:
            normal = np.cross(
                cube_vertices[face[1]] - cube_vertices[face[0]],
                cube_vertices[face[2]] - cube_vertices[face[0]],
            )
            assert neg < pos and normal @ (centroids[pos] - centroids[neg]) > 0


def test_constant_cell_density_preserves_w4_mass_and_first_moments():
    vertices, tetra, density = structured_two_density_tetra(2)
    field = UnstructuredDisplacementField(vertices, tetra, displacement(vertices))
    p0 = piecewise_tetra_linear_sources(field, density)
    p1 = piecewise_tetra_p1_density_sources(
        field, np.repeat(density[:, None], 4, axis=1)
    )
    assert np.sum(p1.mass_kg) == pytest.approx(np.sum(p0.mass_kg), rel=5e-15)
    np.testing.assert_allclose(
        np.sum(p1.mass_kg[:, None] * p1.mass_positions_m, axis=0),
        np.sum(p0.mass_kg[:, None] * p0.mass_positions_m, axis=0),
        rtol=5e-15,
        atol=5e-15,
    )
    np.testing.assert_allclose(
        np.sum(p1.mass_kg[:, None] * p1.mass_displacement_m, axis=0),
        np.sum(p0.mass_kg[:, None] * p0.mass_displacement_m, axis=0),
        rtol=5e-15,
        atol=5e-15,
    )


@pytest.mark.parametrize("observable", ["potential", "acceleration", "gradient"])
def test_constant_cell_density_matches_w4_continuum_oracle(observable):
    vertices, tetra, density = structured_two_density_tetra(8)
    field = UnstructuredDisplacementField(vertices, tetra, displacement(vertices))
    sources = piecewise_tetra_p1_density_sources(
        field, np.repeat(density[:, None], 4, axis=1)
    )
    report = compare_tetra_p1_density_representations(
        sources, TARGETS, observable=observable, chunk_size=37
    )
    oracle = p0_continuum_oracle(12, TARGETS, observable)
    p0_mass = 28.0
    scales = target_scales(observable) * (p0_mass / 27.3)
    for response in (
        report.mass_displacement_response,
        report.bulk_interface_response,
    ):
        error = response_norm(response - oracle, observable) / scales
        assert np.max(error) <= 0.02


def test_discontinuous_trace_and_affine_mass_conservation(cube_vertices, cube_tetra):
    v = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0, 0, -1.0]])
    tetra = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
    field = UnstructuredDisplacementField(v, tetra, np.tile([0.0, 0, 1.0], (5, 1)))
    rho = np.array([[2.0, 3, 5, 7], [11.0, 17, 13, 19]])
    s = piecewise_tetra_p1_density_sources(field, rho)
    fi = int(np.flatnonzero(s.positive_side_cell >= 0)[0])
    face = s.interface_triangles[fi]
    rule = np.array(
        [[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]]
    )

    def trace(cell):
        local = [int(np.flatnonzero(tetra[cell] == node)[0]) for node in face]
        return rule @ rho[cell, local]

    np.testing.assert_allclose(s.interface_density_jump_kg_m3[fi], trace(0) - trace(1))
    assert np.ptp(trace(0) - trace(1)) > 0

    field = UnstructuredDisplacementField(
        cube_vertices, cube_tetra, B + cube_vertices @ A.T
    )
    r = (
        3
        + 0.2 * cube_vertices[cube_tetra, 0]
        - 0.1 * cube_vertices[cube_tetra, 1]
        + 0.07 * cube_vertices[cube_tetra, 2]
    )
    s = piecewise_tetra_p1_density_sources(field, r)
    bulk = s.bulk_source.as_weighted_source().weights_kg
    interface = s.interface_source.weights_kg
    scale = np.sum(abs(bulk)) + np.sum(abs(interface))
    assert abs(np.sum(bulk) + np.sum(interface)) <= 512 * np.finfo(float).eps * scale


def test_translation_bulk_is_minus_u_dot_grad_rho():
    u = np.tile([0.4, -0.3, 0.2], (4, 1))
    field = single_tetra(u)
    grad = np.array([0.2, -0.1, 0.05])
    s = piecewise_tetra_p1_density_sources(field, (2 + field.vertices_m @ grad)[None])
    np.testing.assert_allclose(
        s.bulk_source.density_kg_m3, -u[0] @ grad, rtol=1e-13, atol=1e-14
    )


@pytest.mark.parametrize("observable", ["potential", "acceleration", "gradient"])
def test_fixed_refinement_and_independent_halfbox_oracle(observable):
    o10 = continuum_oracle(10, TARGETS, observable)
    o12 = continuum_oracle(12, TARGETS, observable)
    scales = target_scales(observable)
    assert np.max(response_norm(o12 - o10, observable) / scales) <= 1e-10
    errors = [[], [], []]
    for n in (2, 4, 8):
        v, t, r = structured_p1_density_tetra(n)
        report = compare_tetra_p1_density_representations(
            piecewise_tetra_p1_density_sources(
                UnstructuredDisplacementField(v, t, displacement(v)), r
            ),
            TARGETS,
            observable=observable,
            chunk_size=37,
        )
        errors[0].append(np.max(report.abs_residual / scales))
        errors[1].append(
            np.max(
                response_norm(report.mass_displacement_response - o12, observable)
                / scales
            )
        )
        errors[2].append(
            np.max(
                response_norm(report.bulk_interface_response - o12, observable) / scales
            )
        )
    for e in errors:
        assert np.isfinite(e).all() and e[-1] < e[0] and e[-1] <= 0.02
        assert e[-1] / e[-2] <= 0.60 if e[-2] > 1e-10 else e[-1] <= 1e-10


@pytest.mark.parametrize(
    "rigid,linear",
    [
        (rigid_transform_potential, displacement_transfer_potential),
        (rigid_transform_acceleration, displacement_transfer_acceleration),
        (rigid_transform_gravity_gradient, displacement_transfer_gravity_gradient),
    ],
)
def test_positive_quadrature_taylor_remainder(rigid, linear):
    v, t, r = structured_p1_density_tetra(2)
    s = piecewise_tetra_p1_density_sources(
        UnstructuredDisplacementField(v, t, displacement(v)), r
    )
    assert np.all(s.mass_kg > 0)
    cloud = PointMassCloud(s.mass_kg, s.mass_positions_m)
    rstar = np.min(np.linalg.norm(s.mass_positions_m[:, None] - TARGETS[None], axis=-1))
    direction = np.array([2.0, -1, 3]) / np.sqrt(14)
    rem = []
    for relative in (1e-3, 5e-4, 2.5e-4):
        shift = relative * rstar * direction
        transform = PrescribedRigidTransform(
            np.array([0.0, 1.0]),
            np.array([[0.0, 0, 0], shift]),
            np.array([np.eye(3), np.eye(3)]),
        )
        finite = rigid(cloud, transform, TARGETS, chunk_size=5)
        lin = linear(
            s.mass_positions_m,
            s.mass_kg,
            np.tile(shift, (len(s.mass_kg), 1)),
            TARGETS,
            chunk_size=5,
        )
        rem.append(np.linalg.norm(finite[1] - finite[0] - lin))
    ratio = np.array(rem[:-1]) / np.array(rem[1:])
    assert np.all((ratio >= 3) & (ratio <= 5))


def test_complex_linearity_empty_zero_and_errors():
    v, t, r = structured_p1_density_tetra(2)
    real = piecewise_tetra_p1_density_sources(
        UnstructuredDisplacementField(v, t, displacement(v)), r
    )
    comp = piecewise_tetra_p1_density_sources(
        UnstructuredDisplacementField(v, t, (1 + 2j) * displacement(v)), r
    )
    assert comp.mass_displacement_m.dtype == np.complex128
    for observable in ("potential", "acceleration", "gradient"):
        got = compare_tetra_p1_density_representations(
            comp, TARGETS, observable=observable
        )
        expected = compare_tetra_p1_density_representations(
            real, TARGETS, observable=observable
        )
        np.testing.assert_allclose(
            got.bulk_interface_response,
            (1 + 2j) * expected.bulk_interface_response,
            rtol=1e-12,
            atol=1e-14,
        )
    empty = UnstructuredDisplacementField(
        np.empty((0, 3)), np.empty((0, 4), dtype=np.int64), np.empty((0, 3))
    )
    es = piecewise_tetra_p1_density_sources(empty, np.empty((0, 4)))
    assert es.mass_positions_m.shape == (0, 3) and es.interface_triangles.shape == (
        0,
        3,
    )
    for obs, shape in (
        ("potential", (0,)),
        ("acceleration", (0, 3)),
        ("gradient", (0, 3, 3)),
    ):
        report = compare_tetra_p1_density_representations(
            es, np.empty((0, 3)), observable=obs
        )
        assert report.mass_displacement_response.shape == shape
    zero = [
        piecewise_tetra_p1_density_sources(
            single_tetra(np.zeros((4, 3))), np.ones((1, 4))
        ),
        piecewise_tetra_p1_density_sources(single_tetra(), np.zeros((1, 4))),
    ]
    for source in zero:
        for obs in ("potential", "acceleration", "gradient"):
            report = compare_tetra_p1_density_representations(
                source, TARGETS, observable=obs
            )
            np.testing.assert_array_equal(report.bulk_interface_response, 0)
            np.testing.assert_array_equal(report.mass_displacement_response, 0)
    with pytest.raises(ValueError, match="Unsupported observable"):
        compare_tetra_p1_density_representations(es, TARGETS, observable="bad")
    with pytest.raises(ValueError, match="chunk_size"):
        compare_tetra_p1_density_representations(es, TARGETS, chunk_size=0)


@pytest.mark.parametrize(
    "rho,error",
    [
        (1.0, ValueError),
        (np.ones(4), ValueError),
        (np.ones((2, 4)), ValueError),
        (np.array([[True] * 4]), TypeError),
        (np.array([[1 + 0j] * 4]), TypeError),
        (np.array([[-1.0, 1, 1, 1]]), ValueError),
        (np.array([[np.nan, 1, 1, 1]]), ValueError),
    ],
)
def test_density_validation(rho, error):
    with pytest.raises(error):
        piecewise_tetra_p1_density_sources(single_tetra(), rho)


def test_factory_rechecks_cross_field_invariants():
    field = single_tetra()
    s = piecewise_tetra_p1_density_sources(field, np.ones((1, 4)))
    values = {item.name: getattr(s, item.name) for item in fields(s)}
    values["mass_cell_index"] = np.array([0, 0, 0, 1])
    with pytest.raises(ValueError, match="mass_cell_index"):
        TetraP1DensityLinearizedSources._from_builder(field=field, **values)

    for bad_index in (
        np.array([False, False, False, False]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0j, 0j, 0j, 0j]),
    ):
        changed = dict(values, mass_cell_index=bad_index)
        with pytest.raises((TypeError, ValueError)):
            TetraP1DensityLinearizedSources._from_builder(field=field, **changed)

    positions = s.interface_source.positions_m.copy()
    positions[0, 0] += 1e-6
    changed = dict(
        values,
        mass_cell_index=s.mass_cell_index,
        interface_source=WeightedPointSource(positions, s.interface_source.weights_kg),
    )
    with pytest.raises(ValueError, match="interface_source.positions_m"):
        TetraP1DensityLinearizedSources._from_builder(field=field, **changed)

    with pytest.raises(TypeError, match="mass_kg"):
        TetraP1DensityLinearizedSources._from_builder(
            field=field,
            **dict(values, mass_cell_index=s.mass_cell_index, mass_kg=np.ones(4, bool)),
        )
    with pytest.raises(TypeError, match="mass_positions_m"):
        TetraP1DensityLinearizedSources._from_builder(
            field=field,
            **dict(
                values,
                mass_cell_index=s.mass_cell_index,
                mass_positions_m=s.mass_positions_m.astype(complex),
            ),
        )
