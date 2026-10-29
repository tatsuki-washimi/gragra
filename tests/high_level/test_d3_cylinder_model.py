"""High-level contracts for the analytic uniform-cylinder model."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gragra.analytic import cylinder_analytic_observables_g_free
from gragra.constants import G_SI
from gragra.geometry import Cylinder, Frame
from gragra.results import (
    Contribution,
    EvaluationContext,
    EvaluationResult,
    FieldKind,
    PotentialReference,
)
from gragra.targets import TargetPoints


def _frames_and_targets():
    root = Frame("root")
    angle = 0.63
    geometry_basis = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    target_basis = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    geometry_frame = Frame(
        "cylinder",
        origin=np.array([3.0, -2.0, 1.0]) * u.m,
        basis=geometry_basis,
        parent=root,
    )
    target_frame = Frame(
        "sensor",
        origin=np.array([-1.0, 0.4, 2.0]) * u.m,
        basis=target_basis,
        parent=root,
    )
    cylinder = Cylinder(
        np.array([0.5, -0.3, 0.2]) * u.m,
        2.0 * u.m,
        3.0 * u.m,
        geometry_frame,
    )
    offsets = np.array(
        [
            [0.0, 0.0, 0.4],
            [0.2, 0.3, 0.1],
            [2.6, -1.05, 2.1],
        ]
    )
    positions_geometry = cylinder.center_m + offsets
    positions_sensor = geometry_frame.transform_points_m(
        positions_geometry, target_frame
    )
    targets = TargetPoints(
        positions_sensor * u.m, ("axis", "inside", "outside"), target_frame
    )
    return root, geometry_frame, cylinder, offsets, positions_geometry, targets


def test_uniform_cylinder_runs_analytic_g_free_with_common_result_and_frame_contract(
    monkeypatch,
):
    import gragra.analytic.cavity_cylinder as numeric_cylinder_module
    import gragra.analytic.objects._cylinder_model as cylinder_model_module
    from gragra.analytic.objects._cylinder_model import uniform_cylinder

    root, geometry_frame, cylinder, offsets, positions_geometry, targets = (
        _frames_and_targets()
    )

    def forbidden_numeric_path(*_args, **_kwargs):
        raise AssertionError("the numerical cylinder diagnostic is not a fallback")

    monkeypatch.setattr(
        "gragra.analytic.cylinder_observables_g_free", forbidden_numeric_path
    )
    monkeypatch.setattr(
        numeric_cylinder_module,
        "cylinder_observables_g_free",
        forbidden_numeric_path,
    )
    # Guard the model's lookup site too, so a future direct import of the
    # numerical diagnostic cannot bypass the public re-export patch above.
    monkeypatch.setattr(
        cylinder_model_module,
        "cylinder_observables_g_free",
        forbidden_numeric_path,
        raising=False,
    )
    result = (
        uniform_cylinder(cylinder, 1800.0 * u.kg / u.m**3)
        .at(targets, output_frame=root)
        .run()
    )

    assert isinstance(result, EvaluationResult)
    assert isinstance(result.context, EvaluationContext)
    assert result.context.frame is root
    assert result.context.targets is targets
    assert result.observables == ("potential", "acceleration", "gradient")
    assert result.potential.shape == (3,)
    assert result.acceleration.shape == (3, 3)
    assert result.gradient.shape == (3, 3, 3)
    assert result.potential.unit.is_equivalent(u.m**2 / u.s**2)
    assert result.acceleration.unit.is_equivalent(u.m / u.s**2)
    assert result.gradient.unit.is_equivalent(u.s**-2)
    assert result.g_applications == 1
    assert result.gravitational_constant_si == G_SI

    definition = result.field_definition
    assert definition.kind is FieldKind.SOURCE_FIELD
    assert definition.source_kind == "uniform_cylinder"
    assert definition.contribution == Contribution("total")
    assert definition.potential_reference is PotentialReference.ZERO_AT_INFINITY

    native = cylinder_analytic_observables_g_free(
        cylinder.center_m,
        cylinder.radius_m,
        cylinder.height_m,
        1800.0,
        positions_geometry,
    )
    expected_acceleration = geometry_frame.transform_vectors_m(
        native.acceleration, root
    )
    expected_gradient = geometry_frame.transform_tensors(native.gradient, root)
    np.testing.assert_allclose(
        result.potential.to_value(u.m**2 / u.s**2), G_SI * native.potential
    )
    np.testing.assert_allclose(
        result.acceleration.to_value(u.m / u.s**2),
        G_SI * expected_acceleration,
        atol=1e-20,
    )
    np.testing.assert_allclose(
        result.gradient.to_value(u.s**-2), G_SI * expected_gradient, atol=1e-20
    )

    parameters = result.parameters
    assert parameters["radius"].unit == u.m
    assert parameters["height"].unit == u.m
    assert parameters["density"].unit == u.kg / u.m**3
    assert parameters["continuous_mass"].unit == u.kg
    assert parameters["target_count"][0] == len(targets)
    assert parameters["geometry_frame"][0] == geometry_frame.name


def test_uniform_cylinder_preserves_native_boundary_target_rejection():
    from gragra.analytic.objects._cylinder_model import uniform_cylinder

    frame = Frame("world")
    cylinder = Cylinder(np.zeros(3) * u.m, 2 * u.m, 3 * u.m, frame)
    targets = TargetPoints(np.array([[2.0, 0.0, 0.4]]) * u.m, ("rim",), frame)

    with pytest.raises(ValueError, match="physical cylinder boundary"):
        uniform_cylinder(cylinder, 1800.0 * u.kg / u.m**3).at(targets).run()


def test_uniform_cylinder_matches_converged_independent_potential_and_gradient():
    pytest.importorskip("scipy")
    from oracles.cylinder_potential_reference import boundary_potential
    from oracles.cylinder_surface_reference import surface_reference

    from gragra.analytic.objects._cylinder_model import uniform_cylinder

    frame = Frame("world")
    cylinder = Cylinder(np.zeros(3) * u.m, 2 * u.m, 3 * u.m, frame)
    point = np.array([0.2, 0.3, 0.1])
    targets = TargetPoints(point.reshape(1, 3) * u.m, ("general",), frame)
    result = uniform_cylinder(cylinder, 1800.0 * u.kg / u.m**3).at(targets).run()

    low_potential, _ = boundary_potential(2.0, 3.0, 1800.0, point, tolerance=1e-8)
    high_potential, _ = boundary_potential(2.0, 3.0, 1800.0, point, tolerance=1e-11)
    potential_budget = 1e-10 * (1800.0 * 2.0**2 + abs(high_potential))
    assert abs(low_potential - high_potential) <= potential_budget / 10
    assert (
        abs(result.potential[0].to_value(u.m**2 / u.s**2) / G_SI - high_potential)
        <= potential_budget
    )

    low_acceleration, low_gradient, _ = surface_reference(
        2.0, 3.0, 1800.0, point, tolerance=1e-8
    )
    high_acceleration, high_gradient, _ = surface_reference(
        2.0, 3.0, 1800.0, point, tolerance=1e-11
    )
    acceleration_budget = 1e-10 * (1800.0 * 2.0 + np.linalg.norm(high_acceleration))
    gradient_budget = 1e-9 * (1800.0 + np.linalg.norm(high_gradient))
    assert (
        np.linalg.norm(low_acceleration - high_acceleration) <= acceleration_budget / 10
    )
    assert np.linalg.norm(low_gradient - high_gradient) <= gradient_budget / 10
    np.testing.assert_allclose(
        result.acceleration[0].to_value(u.m / u.s**2) / G_SI,
        high_acceleration,
        rtol=0,
        atol=acceleration_budget,
    )
    np.testing.assert_allclose(
        result.gradient[0].to_value(u.s**-2) / G_SI,
        high_gradient,
        rtol=0,
        atol=gradient_budget,
    )
