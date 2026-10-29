"""High-level static uniform-cuboid analytic model."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from astropy import units as u


def _factory():
    module = importlib.import_module("gragra.analytic.objects._cuboid_model")
    return module.uniform_cuboid


def _geometry_inputs(frame=None):
    from gragra.geometry._cuboid import Cuboid
    from gragra.geometry._frame import Frame

    frame = Frame("cuboid") if frame is None else frame
    geometry = Cuboid(
        np.array([0.25, -0.5, 0.75]) * u.m,
        np.array([2.0, 3.0, 4.0]) * u.m,
        frame,
    )
    density = 2.4 * u.g / u.cm**3
    return geometry, density


def test_uniform_cuboid_factory_is_resolvable() -> None:
    try:
        module = importlib.import_module("gragra.analytic.objects._cuboid_model")
    except ModuleNotFoundError:
        module = None

    assert module is not None and callable(getattr(module, "uniform_cuboid", None))


def test_factory_uses_common_context_and_canonical_source_definition() -> None:
    from gragra.geometry._frame import Frame
    from gragra.results import (
        Contribution,
        FieldKind,
        PotentialReference,
    )
    from gragra.results._context import EvaluationContext
    from gragra.targets._high_points import TargetPoints

    frame = Frame("cube-frame")
    geometry, density = _geometry_inputs(frame)
    targets = TargetPoints([[3.0, 1.0, 2.0]] * u.m, ("outside",), frame)

    model = _factory()(geometry, density)
    setup = model.at(targets)

    assert model.geometry is geometry
    assert model.density.to_value(u.kg / u.m**3) == pytest.approx(2400.0)
    assert isinstance(setup.context, EvaluationContext)
    assert setup.resolved.model is model
    assert setup.resolved.model_spec is model.spec
    assert setup.resolved.computation_frame is frame
    assert setup.resolved.output_frame is frame
    definition = setup.context.field_definition
    assert definition.kind is FieldKind.SOURCE_FIELD
    assert definition.source_kind == "uniform_cuboid"
    assert definition.contribution == Contribution("total")
    assert definition.potential_reference is PotentialReference.ZERO_AT_INFINITY


def test_run_matches_g_free_kernel_with_one_si_g_and_static_shapes() -> None:
    from gragra.analytic.cavity_cuboid import cuboid_observables_g_free
    from gragra.constants import G_SI
    from gragra.geometry._frame import Frame
    from gragra.results._evaluation import EvaluationResult
    from gragra.targets._high_points import TargetPoints

    frame = Frame("native-comparison")
    geometry, density = _geometry_inputs(frame)
    targets = TargetPoints(
        np.array(
            [
                [0.35, -0.25, 0.5],  # interior
                [2.0, 0.0, 0.0],  # exterior
                [1.2501, -0.2, 0.65],  # just outside the +x face
            ]
        )
        * u.m,
        ("inside", "outside", "near-face"),
        frame,
    )
    setup = _factory()(geometry, density).at(targets)

    result = setup.run()
    native = cuboid_observables_g_free(
        geometry.center_m,
        geometry.lengths_m,
        setup.resolved.model_spec.density_kg_m3,
        setup.resolved.target_positions_m,
    )

    assert result.potential.shape == (3,)
    assert result.acceleration.shape == (3, 3)
    assert result.gradient.shape == (3, 3, 3)
    assert result.potential.unit == u.m**2 / u.s**2
    assert result.acceleration.unit == u.m / u.s**2
    assert result.gradient.unit == u.s**-2
    np.testing.assert_allclose(result.potential.value, G_SI * native.potential)
    np.testing.assert_allclose(result.acceleration.value, G_SI * native.acceleration)
    np.testing.assert_allclose(result.gradient.value, G_SI * native.gradient)
    assert result.record.g_applications == 1
    assert result.gravitational_constant_si == G_SI
    assert isinstance(result, EvaluationResult)
    assert result.context is setup.context
    assert result.field_definition is setup.resolved.model_spec.field_definition
    assert result.parameters["continuous_mass"][0].to_value(u.kg) == pytest.approx(
        np.prod(geometry.lengths_m) * 2400.0
    )
    assert any(
        "cuboid_observables_g_free" in source for source in result.references["source"]
    )
    assert setup.validate().target_regions == {
        "inside": "interior",
        "outside": "exterior",
        "near-face": "exterior",
    }


def test_target_mapping_and_output_frame_rotate_vectors_and_tensors() -> None:
    from gragra.analytic.cavity_cuboid import cuboid_observables_g_free
    from gragra.constants import G_SI
    from gragra.geometry._frame import Frame
    from gragra.targets._high_points import TargetPoints

    root = Frame("root")
    angle = np.pi / 3
    geometry_basis = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target_angle = -np.pi / 6
    target_basis = np.array(
        [
            [np.cos(target_angle), -np.sin(target_angle), 0.0],
            [np.sin(target_angle), np.cos(target_angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    geometry_frame = Frame(
        "geometry-frame",
        origin=[1.0, -2.0, 0.5] * u.m,
        basis=geometry_basis,
        parent=root,
    )
    target_frame = Frame(
        "target-frame",
        origin=[-1.5, 0.25, 2.0] * u.m,
        basis=target_basis,
        parent=root,
    )
    geometry, density = _geometry_inputs(geometry_frame)
    targets = TargetPoints([[2.0, 1.0, 3.0]] * u.m, ("T",), target_frame)
    setup = _factory()(geometry, density).at(targets, output_frame=root)

    result = setup.run()
    target_m = targets.frame.transform_points_m(targets.positions_m, geometry_frame)
    native = cuboid_observables_g_free(
        geometry.center_m,
        geometry.lengths_m,
        setup.resolved.model_spec.density_kg_m3,
        target_m,
    )
    expected_acceleration = geometry_frame.transform_vectors_m(
        G_SI * native.acceleration, root
    )
    expected_gradient = geometry_frame.transform_tensors(G_SI * native.gradient, root)

    np.testing.assert_allclose(setup.resolved.target_positions_m, target_m)
    np.testing.assert_allclose(result.potential.value, G_SI * native.potential)
    np.testing.assert_allclose(result.acceleration.value, expected_acceleration)
    np.testing.assert_allclose(result.gradient.value, expected_gradient)


@pytest.mark.parametrize("point", ([1.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0, 0, -2.0]))
def test_boundary_targets_keep_native_rejection(point) -> None:
    from gragra.analytic.cavity_cuboid import cuboid_observables_g_free
    from gragra.geometry._cuboid import Cuboid
    from gragra.geometry._frame import Frame
    from gragra.targets._high_points import TargetPoints

    frame = Frame("boundary")
    geometry = Cuboid([0, 0, 0] * u.m, [2, 3, 4] * u.m, frame)
    density = 1000 * u.kg / u.m**3
    targets = TargetPoints([point] * u.m, ("face",), frame)
    setup = _factory()(geometry, density).at(targets)
    with pytest.raises(ValueError) as native_error:
        cuboid_observables_g_free(
            geometry.center_m, geometry.lengths_m, 1000.0, targets.positions_m
        )

    report = setup.validate()
    assert report.target_regions == {"face": "boundary"}
    assert report.valid is False
    with pytest.raises(type(native_error.value)) as high_level_error:
        setup.run()
    assert str(high_level_error.value) == str(native_error.value)


def test_validation_report_matches_native_boundary_tolerance() -> None:
    from gragra.geometry._cuboid import Cuboid
    from gragra.geometry._frame import Frame
    from gragra.targets._high_points import TargetPoints

    frame = Frame("boundary-tolerance")
    geometry = Cuboid([0, 0, 0] * u.m, [2, 3, 4] * u.m, frame)
    density = 1000 * u.kg / u.m**3
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0 + 1e-13, 0.0, 0.0],
            [1.0 - 1e-13, 0.0, 0.0],
            [1.0 + 1e-10, 0.0, 0.0],
        ]
    )
    targets = TargetPoints(
        points * u.m,
        ("face", "near-outside", "near-inside", "outside-tolerance"),
        frame,
    )

    report = _factory()(geometry, density).at(targets).validate()

    assert report.valid is False
    assert report.target_regions == {
        "face": "boundary",
        "near-outside": "boundary",
        "near-inside": "boundary",
        "outside-tolerance": "exterior",
    }
    valid_outside = TargetPoints(points[-1:] * u.m, ("outside",), frame)
    assert _factory()(geometry, density).at(valid_outside).validate().valid is True


def test_acceleration_and_gradient_derivatives_near_boundary_inside_and_outside() -> (
    None
):
    from gragra.constants import G_SI
    from gragra.geometry._frame import Frame
    from gragra.targets._high_points import TargetPoints

    frame = Frame("derivative")
    geometry, density = _geometry_inputs(frame)
    model = _factory()(geometry, density)
    half_x = geometry.lengths_m[0] / 2
    face_x = geometry.center_m[0] + half_x
    sample_points = np.array([[face_x - 1e-4, 0.13, 0.21], [face_x + 1e-4, 0.13, 0.21]])
    step = 1e-5

    for index, point in enumerate(sample_points):
        target = TargetPoints([point] * u.m, (f"p{index}",), frame)
        result = model.at(target).run()
        finite_difference = np.empty(3)
        finite_difference_gradient = np.empty((3, 3))
        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = step
            pair = TargetPoints(
                np.array([point + delta, point - delta]) * u.m,
                ("plus", "minus"),
                frame,
            )
            potential = model.at(pair, observables=("potential",)).run().potential
            acceleration_pair = (
                model.at(pair, observables=("acceleration",)).run().acceleration
            )
            finite_difference[axis] = -(
                potential[0].to_value(u.m**2 / u.s**2)
                - potential[1].to_value(u.m**2 / u.s**2)
            ) / (2 * step)
            finite_difference_gradient[:, axis] = (
                acceleration_pair[0].to_value(u.m / u.s**2)
                - acceleration_pair[1].to_value(u.m / u.s**2)
            ) / (2 * step)
        np.testing.assert_allclose(
            result.acceleration[0].to_value(u.m / u.s**2),
            finite_difference,
            rtol=2e-7,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            result.gradient[0].to_value(u.s**-2),
            finite_difference_gradient,
            rtol=2e-7,
            atol=1e-15,
        )
        expected_trace = (
            -4 * np.pi * G_SI * density.to_value(u.kg / u.m**3) if index == 0 else 0.0
        )
        np.testing.assert_allclose(
            np.trace(result.gradient[0].to_value(u.s**-2)),
            expected_trace,
            rtol=2e-12,
            atol=1e-20,
        )
