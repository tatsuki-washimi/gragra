"""Tests for G-free mass weighting and linearized-displacement coupling."""

import numpy as np
import pytest

from gragra.analytic.cavity_sphere import (
    spherical_cavity_bulk_p,
    spherical_cavity_wall,
)
from gragra.fields.finite_body_p1 import (
    compare_tetra_p1_density_representations,
    piecewise_tetra_p1_density_sources,
)
from gragra.fields.generic_integration import FieldSampler, MassCoupling, QuadraturePlan
from gragra.fields.surface import SurfaceDisplacementField
from gragra.fields.unstructured import UnstructuredDisplacementField
from gragra.integrators.generic import (
    DisplacementSourceBlock,
    ScalarSourceBlock,
)
from gragra.kernels.generic_backend import NumpyContractionBackend


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


def test_volume_density_perturbation_becomes_signed_scalar_mass_block() -> None:
    vertices, tetra = _unit_tetrahedron()
    plan = QuadraturePlan.from_tetra_mesh(vertices, tetra, order=2)
    density_delta = np.column_stack(
        (np.ones(len(plan.weights)), -2.0 * np.ones(len(plan.weights)))
    )
    original = density_delta.copy()

    block = MassCoupling.volume_density(
        plan, density_delta, parameter_slice=slice(4, 6)
    )

    assert isinstance(block, ScalarSourceBlock)
    assert block.weights_kg.shape == (8, 2)
    assert block.parameter_slice == slice(4, 6)
    np.testing.assert_allclose(block.positions_m, plan.points_m)
    np.testing.assert_allclose(block.weights_kg, plan.weights[:, None] * density_delta)
    np.testing.assert_array_equal(density_delta, original)
    assert not block.weights_kg.flags.writeable


def test_precomputed_surface_density_is_integrated_once_as_sheet_mass() -> None:
    vertices, triangles = _unit_triangle()
    plan = QuadraturePlan.from_surface_mesh(vertices, triangles, order=3)
    delta_sigma = np.full((len(plan.weights), 1), 4.0 + 2.0j)

    block = MassCoupling.surface_density(plan, delta_sigma)

    assert isinstance(block, ScalarSourceBlock)
    np.testing.assert_allclose(block.weights_kg, plan.weights[:, None] * delta_sigma)
    assert block.weights_kg.sum() == pytest.approx(2.0 + 1.0j)


def test_interface_jump_uses_raw_complex_dot_and_reverses_with_normal() -> None:
    vertices, triangles = _unit_triangle()
    displacement = np.tile([1.0 + 2.0j, -3.0j, 5.0 + 4.0j], (3, 1))
    field = SurfaceDisplacementField(vertices, triangles, displacement)
    plan = QuadraturePlan.from_surface_mesh(vertices, triangles, order=3)
    sampled = FieldSampler().sample_surface_displacement(field, plan)

    outward = MassCoupling.interface_density_jump(
        plan, sampled, density_jump_kg_m3=-2.0
    )

    reversed_field = SurfaceDisplacementField(
        vertices, triangles[:, ::-1], displacement
    )
    reversed_plan = QuadraturePlan.from_surface_mesh(
        vertices, triangles[:, ::-1], order=3
    )
    reversed_sampled = FieldSampler().sample_surface_displacement(
        reversed_field, reversed_plan
    )
    inward = MassCoupling.interface_density_jump(
        reversed_plan, reversed_sampled, density_jump_kg_m3=2.0
    )

    expected_weight = plan.weights * -2.0 * (5.0 + 4.0j)
    np.testing.assert_allclose(outward.weights_kg[:, 0], expected_weight)
    np.testing.assert_allclose(inward.weights_kg, outward.weights_kg)
    assert np.iscomplexobj(outward.weights_kg)


def test_linearized_displacement_keeps_background_mass_real_and_parameter_axis():
    vertices, tetra = _unit_tetrahedron()
    plan = QuadraturePlan.from_tetra_mesh(vertices, tetra, order=2)
    displacement = np.full((len(plan.weights), 2, 3), 1.0 + 2.0j)
    original = displacement.copy()

    block = MassCoupling.linearized_displacement(
        plan,
        background_density_kg_m3=3000.0,
        displacement_m=displacement,
        parameter_slice=slice(3, 5),
    )

    assert isinstance(block, DisplacementSourceBlock)
    assert block.displacement_m.shape == (8, 2, 3)
    assert block.parameter_slice == slice(3, 5)
    np.testing.assert_allclose(block.mass_kg, 3000.0 * plan.weights)
    assert np.isrealobj(block.mass_kg)
    assert np.all(block.mass_kg >= 0.0)
    np.testing.assert_array_equal(block.displacement_m, displacement)
    np.testing.assert_array_equal(displacement, original)
    assert not block.mass_kg.flags.writeable
    assert not block.displacement_m.flags.writeable


def test_closed_tetra_bulk_surface_and_dipole_representations_agree() -> None:
    vertices, tetra = _unit_tetrahedron()
    displacement = np.tile([0.2, -0.1, 0.3], (4, 1))
    field = UnstructuredDisplacementField(vertices, tetra, displacement)
    density = np.full((1, 4), 3000.0)
    sources = piecewise_tetra_p1_density_sources(field, density, quadrature_order=8)

    report = compare_tetra_p1_density_representations(
        sources, np.array([[20.0, 21.0, 22.0]]), observable="acceleration"
    )

    assert float(np.max(report.rel_residual)) < 2e-10


def test_generic_mass_coupling_end_to_end_bulk_interface_matches_dipole() -> None:
    vertices, tetra = _unit_tetrahedron()
    triangles = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], dtype=np.int64)
    density = 3000.0
    deformation = np.diag([0.1, -0.04, 0.07])
    translation = np.array([0.2, -0.1, 0.3])
    nodal_displacement = vertices @ deformation.T + translation
    field = UnstructuredDisplacementField(vertices, tetra, nodal_displacement)
    density_nodes = np.full((1, len(vertices)), density)
    sampler = FieldSampler()
    order = 8

    volume_plan = QuadraturePlan.from_tetra_mesh(vertices, tetra, order=order)
    sampled_density = sampler.sample_tetra_p1_density(field, density_nodes, volume_plan)
    bulk_source = MassCoupling.volume_density(
        volume_plan, -np.trace(deformation) * sampled_density
    )
    displacement_at_volume_points = (
        volume_plan.points_m @ deformation.T + translation
    )[:, None, :]
    dipole_source = MassCoupling.linearized_displacement(
        volume_plan,
        background_density_kg_m3=density,
        displacement_m=displacement_at_volume_points,
    )

    surface_field = SurfaceDisplacementField(vertices, triangles, nodal_displacement)
    surface_plan = QuadraturePlan.from_surface_mesh(vertices, triangles, order=order)
    sampled_surface_displacement = sampler.sample_surface_displacement(
        surface_field, surface_plan
    )
    interface_source = MassCoupling.interface_density_jump(
        surface_plan,
        sampled_surface_displacement,
        density_jump_kg_m3=density,
    )

    targets = np.array([[20.0, 21.0, 22.0]])
    backend = NumpyContractionBackend()
    bulk = np.zeros((1, 1, 3), dtype=np.float64)
    interface = np.zeros_like(bulk)
    dipole = np.zeros_like(bulk)
    backend.acceleration_contract(
        bulk_source.positions_m,
        targets,
        bulk_source.weights_kg,
        out=bulk,
    )
    backend.acceleration_contract(
        interface_source.positions_m,
        targets,
        interface_source.weights_kg,
        out=interface,
    )
    backend.dipole_contract(
        dipole_source.positions_m,
        targets,
        dipole_source.mass_kg,
        dipole_source.displacement_m,
        out=dipole,
    )

    combined = bulk + interface
    residual = np.linalg.norm(combined - dipole) / np.linalg.norm(dipole)
    assert residual < 2e-10


def test_legacy_include_free_long_wave_factor_is_not_bulk_reference() -> None:
    """Keep the old 8π/3 coefficient separate from full-space bulk 4π."""
    density = 2700.0
    displacement = 1e-6
    radius = 5.0

    bulk = float(spherical_cavity_bulk_p(radius, 0.0, density, displacement))
    wall = float(
        spherical_cavity_wall(radius, 0.0, density, displacement, wave_type="P")
    )

    np.testing.assert_allclose(bulk, 4.0 * np.pi * density * displacement)
    np.testing.assert_allclose(wall, -(4.0 * np.pi / 3.0) * density * displacement)
    np.testing.assert_allclose(
        bulk + wall, (8.0 * np.pi / 3.0) * density * displacement
    )
