"""End-to-end comparison of results from static source producers."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gragra.adapters.meshio_adapter import VolumeMesh, to_volume_element_source
from gragra.analytic.objects import uniform_sphere
from gragra.fields import DensityPerturbationGrid
from gragra.geometry import Frame, Sphere
from gragra.gravity_sources import PointMass, discretize, point_sum
from gragra.results import (
    ChangeDefinition,
    ChangeForm,
    Contribution,
    EvaluationRecord,
    EvaluationResult,
    FieldKind,
    GravityFieldDefinition,
    PotentialReference,
    ReferenceDefinition,
    ReferenceStatus,
    ReferenceTemporalKind,
    compare,
)
from gragra.targets import TargetPoints


def _field_change() -> GravityFieldDefinition:
    return GravityFieldDefinition.field_change(
        "density_perturbation",
        Contribution("total"),
        PotentialReference.ZERO_AT_INFINITY,
        ChangeDefinition(
            ChangeForm.FINITE_STATE_DIFFERENCE,
            ReferenceDefinition(
                ReferenceStatus.DEFINED,
                ReferenceTemporalKind.STATE,
                "reference_density",
            ),
            "density_perturbation",
        ),
    )


def _assert_definition_and_g(
    result: EvaluationResult, definition: GravityFieldDefinition
) -> None:
    assert result.field_definition is definition
    assert all(
        binding.field_definition is definition for binding in result.bindings.values()
    )
    assert result.g_applications == 1
    assert result.context.g_convention == "SI_G_applied_once"


def _density_grid_source() -> DensityPerturbationGrid:
    return DensityPerturbationGrid(
        x_m=np.array([0.0, 1.0]),
        y_m=np.array([0.0]),
        z_m=np.array([0.0]),
        delta_density_kg_m3=np.array([[[2.0]], [[-1.0]]]),
        cell_size_m=(1.0, 1.0, 1.0),
    )


def _unit_tetrahedron() -> VolumeMesh:
    vertices_m = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return VolumeMesh(vertices_m, (("tetra", np.array([[0, 1, 2, 3]])),))


def test_point_sum_compares_with_exterior_analytic_sphere_after_id_alignment() -> None:
    world = Frame("comparison-world")
    sphere = Sphere(np.zeros(3) * u.m, 1.0 * u.m, world)
    density = 1000.0 * u.kg / u.m**3
    continuous_mass_kg = (4.0 * np.pi / 3.0) * 1000.0
    analytic_model = uniform_sphere(sphere, density)
    point_model = point_sum(
        PointMass(continuous_mass_kg, [0.0, 0.0, 0.0]),
        frame=world,
        provenance={"representation": "sphere exterior point-mass oracle"},
    )

    analytic_targets = TargetPoints(
        np.array([[0.0, 0.0, 2.5], [0.0, 0.0, 4.0]]) * u.m,
        ("near", "far"),
        world,
    )
    point_targets = TargetPoints(
        np.array([[0.0, 0.0, 400.0], [0.0, 0.0, 250.0]]) * u.cm,
        ("far", "near"),
        world,
    )
    analytic_result = analytic_model.at(analytic_targets).run()
    point_result = point_model.at(point_targets).run()

    _assert_definition_and_g(analytic_result, analytic_model.spec.field_definition)
    _assert_definition_and_g(point_result, point_model.spec.field_definition)
    assert (
        point_result.record.gravitational_constant_si
        == analytic_result.record.gravitational_constant_si
    )
    assert point_result.field_definition.kind is FieldKind.SOURCE_FIELD
    assert point_result.parameters.meta["source_provenance"] == {
        "representation": "sphere exterior point-mass oracle"
    }
    assert point_result.potential.unit == u.m**2 / u.s**2
    assert point_result.acceleration.unit == u.m / u.s**2
    assert point_result.gradient.unit == u.s**-2

    # Compare an explicitly selected common subset. Candidate rows use a
    # different order and compatible length units; compare aligns both by ID.
    compared = compare(
        reference=analytic_result.select("potential", "acceleration"),
        candidate=point_result.select("potential", "acceleration"),
    )
    assert compared.summary["target_id"].tolist() == ["near", "far"]
    for observable in ("potential", "acceleration"):
        relative = compared.summary[f"{observable}_relative_difference"].value
        assert np.isfinite(relative).all()
        assert np.max(relative) < 1e-12
    assert compared.differences.gradient is None


def test_discrete_sphere_comparison_accepts_different_spacing_and_source_count() -> (
    None
):
    world = Frame("sphere-refinement-world")
    sphere = Sphere(np.zeros(3) * u.m, 1.0 * u.m, world)
    density = 1000.0 * u.kg / u.m**3
    coarse_source = discretize(sphere, density, 0.5 * u.m)
    fine_source = discretize(sphere, density, 0.25 * u.m)
    coarse_model = point_sum(coarse_source)
    fine_model = point_sum(fine_source)

    coarse_targets = TargetPoints(
        np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 4.0]]) * u.m,
        ("three", "four"),
        world,
    )
    fine_targets = TargetPoints(
        np.array([[0.0, 0.0, 400.0], [0.0, 0.0, 300.0]]) * u.cm,
        ("four", "three"),
        world,
    )
    coarse_result = coarse_model.at(coarse_targets).run()
    fine_result = fine_model.at(fine_targets).run()

    assert (
        coarse_result.parameters["source_count"][0]
        != fine_result.parameters["source_count"][0]
    )
    assert coarse_result.parameters["spacing_m"][0].to_value(u.m) == 0.5
    assert fine_result.parameters["spacing_m"][0].to_value(u.m) == 0.25
    _assert_definition_and_g(coarse_result, coarse_model.spec.field_definition)
    _assert_definition_and_g(fine_result, fine_model.spec.field_definition)

    compared = compare(
        reference=coarse_result.select("potential"),
        candidate=fine_result.select("potential"),
    )
    assert compared.summary["target_id"].tolist() == ["three", "four"]
    assert np.isfinite(compared.differences.potential.to_value(u.m**2 / u.s**2)).all()


def test_density_grid_and_mesh_results_compare_with_distinct_provenance() -> None:
    world = Frame("grid-mesh-comparison-world")
    grid_frame = Frame(
        "grid-local", origin=np.array([20.0, 0.0, 0.0]) * u.m, parent=world
    )
    mesh_frame = Frame(
        "mesh-local", origin=np.array([20.0, 0.0, 0.0]) * u.m, parent=world
    )
    grid_source = _density_grid_source().as_volume_element_source()
    mesh_source = to_volume_element_source(_unit_tetrahedron(), density_kg_m3=6.0)
    definition = _field_change()
    grid_provenance = {
        "coordinate_frame": "grid-import-frame",
        "coordinate_origin_m": (20.0, 0.0, 0.0),
        "input": "regular density grid",
    }
    mesh_provenance = {
        "coordinate_frame": "mesh-import-frame",
        "coordinate_origin_m": (20.0, 0.0, 0.0),
        "mesh_kind": "unit tetrahedron",
    }
    grid_model = point_sum(
        grid_source,
        frame=grid_frame,
        field_definition=definition,
        provenance=grid_provenance,
    )
    mesh_model = point_sum(
        mesh_source,
        frame=mesh_frame,
        field_definition=definition,
        provenance=mesh_provenance,
    )
    grid_targets = TargetPoints(
        np.array([[25.0, 1.0, 1.0], [26.0, 0.5, 0.5]]) * u.m,
        ("first", "second"),
        world,
    )
    mesh_targets = TargetPoints(
        np.array([[2600.0, 50.0, 50.0], [2500.0, 100.0, 100.0]]) * u.cm,
        ("second", "first"),
        world,
    )
    grid_result = grid_model.at(grid_targets).run()
    mesh_result = mesh_model.at(mesh_targets).run()

    _assert_definition_and_g(grid_result, definition)
    _assert_definition_and_g(mesh_result, definition)
    assert grid_result.parameters.meta["source_count"] == 2
    assert mesh_result.parameters.meta["source_count"] == 1
    assert grid_result.parameters.meta["source_provenance"] == grid_provenance
    assert mesh_result.parameters.meta["source_provenance"] == mesh_provenance

    # Execution labels are result metadata, not physical compatibility keys.
    # Keep the computed values and producer context intact while changing only
    # the recorded backend/version fields for this compatibility check.
    mesh_with_distinct_record = EvaluationResult(
        mesh_result.values,
        evaluation=mesh_result.evaluation,
        record=EvaluationRecord(
            backend="mesh-adapter-record",
            versions={"input": "mesh-fixture"},
        ),
    )
    compared = compare(
        reference=grid_result.select("potential", "acceleration"),
        candidate=mesh_with_distinct_record.select("potential", "acceleration"),
    )
    assert compared.summary["target_id"].tolist() == ["first", "second"]
    assert np.isfinite(compared.differences.potential.to_value(u.m**2 / u.s**2)).all()
    assert np.isfinite(compared.differences.acceleration.to_value(u.m / u.s**2)).all()


def test_real_static_producers_reject_disconnected_frames_and_field_meaning() -> None:
    left_world = Frame("same-name")
    right_world = Frame("same-name")
    sphere = Sphere(np.zeros(3) * u.m, 1.0 * u.m, left_world)
    left_result = (
        uniform_sphere(sphere, 1.0 * u.kg / u.m**3)
        .at(TargetPoints([[0.0, 0.0, 3.0]] * u.m, ("target",), left_world))
        .run()
    )
    right_sphere = Sphere(np.zeros(3) * u.m, 1.0 * u.m, right_world)
    right_result = (
        uniform_sphere(right_sphere, 1.0 * u.kg / u.m**3)
        .at(TargetPoints([[0.0, 0.0, 3.0]] * u.m, ("target",), right_world))
        .run()
    )
    with pytest.raises(ValueError, match="connected root Frame"):
        compare(reference=left_result, candidate=right_result)

    shared_world = Frame("field-meaning-world")
    targets = TargetPoints([[5.0, 0.0, 0.0]] * u.m, ("target",), shared_world)
    source_field_result = (
        point_sum(PointMass(1.0, [0.0, 0.0, 0.0]), frame=shared_world).at(targets).run()
    )
    change_model = point_sum(
        _density_grid_source().as_volume_element_source(),
        frame=shared_world,
        field_definition=_field_change(),
    )
    change_result = change_model.at(targets).run()
    with pytest.raises(ValueError, match="field definition kind differs"):
        compare(reference=source_field_result, candidate=change_result)
