"""Contract tests for the C5 owned input and resolved-evaluation layer."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gragra._evaluation_setup import (
    CoordinatePrecisionError,
    DiscreteSource,
    EvaluationRecord,
    ResolvedEvaluation,
    UniformSphereModelSpec,
    check_pair_precision,
)
from gragra.evaluation import EvaluationSetup
from gragra.geometry._frame import Frame
from gragra.geometry._sphere import Sphere
from gragra.targets._high_points import TargetPoints


def _rx(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_two_level_parent_transform_matches_independent_composition() -> None:
    world = Frame("world")
    b1, b2 = _rz(np.pi / 3), _rx(np.pi / 4)
    o1, o2 = np.array([7.0, -3.0, 2.0]), np.array([0.5, 4.0, -1.0])
    lab = Frame("lab", origin=o1 * u.m, basis=b1, parent=world)
    sphere = Frame("sphere", origin=o2 * u.m, basis=b2, parent=lab)
    local = np.array([[1.0, 2.0, -0.5], [-2.0, 0.0, 3.0]])
    expected = o1 + (o2 + local @ b2.T) @ b1.T
    np.testing.assert_allclose(
        sphere.transform_points_m(local, world), expected, atol=1e-14
    )
    np.testing.assert_allclose(
        world.transform_points_m(expected, sphere), local, atol=1e-14
    )
    np.testing.assert_allclose(sphere.rotation_to(world), b1 @ b2, atol=1e-14)

    vector = np.array([2.0, -1.0, 3.0])
    tensor = np.array([[1.0, 2.0, 3.0], [7.0, 4.0, 5.0], [8.0, 9.0, 6.0]])
    rotation = b1 @ b2
    np.testing.assert_allclose(
        sphere.transform_vectors_m(vector, world), rotation @ vector
    )
    np.testing.assert_allclose(
        sphere.transform_tensors(tensor, world), rotation @ tensor @ rotation.T
    )


def test_same_named_roots_are_disconnected_and_invalid_bases_rejected() -> None:
    first, second = Frame("world"), Frame("world")
    with pytest.raises(ValueError, match="disconnected"):
        first.transform_points_m(np.zeros(3), second)
    with pytest.raises(ValueError, match="right.handed"):
        Frame("left", parent=first, basis=np.diag([-1, 1, 1]))
    with pytest.raises(ValueError, match="orthogonal"):
        Frame("skew", parent=first, basis=np.diag([1, 1, 1.01]))
    with pytest.raises(ValueError, match="finite"):
        Frame("bad", parent=first, origin=[np.inf, 0, 0] * u.m)


def test_frame_basis_requires_dimensionless_quantity_and_normalizes_percent() -> None:
    world = Frame("world")
    with pytest.raises(u.UnitConversionError, match="dimensionless"):
        Frame("bad", basis=np.eye(3) * u.m, parent=world)
    scaled = Frame("scaled", basis=100 * np.eye(3) * u.percent, parent=world)
    np.testing.assert_array_equal(scaled.basis, np.eye(3))


def test_input_arrays_are_owned_even_if_original_and_returned_views_are_mutated() -> (
    None
):
    world = Frame("world")
    origin = np.array([1.0, 2.0, 3.0]) * u.m
    basis = _rz(np.pi / 6)
    local = Frame("local", origin=origin, basis=basis, parent=world)
    center = np.array([20.0, -30.0, 5.0]) * u.cm
    radius = np.array(50.0) * u.cm
    sphere = Sphere(center, radius, local)
    positions = np.array([[10.0, 0.0, 0.0], [20.0, 1.0, 3.0]]) * u.cm
    targets = TargetPoints(positions, ("A", "B"), local)

    origin[0] = 90 * u.m
    basis[0, 0] = 0
    center[0] = 900 * u.cm
    positions[0, 0] = 900 * u.cm
    np.testing.assert_allclose(local.origin_m, [1, 2, 3])
    np.testing.assert_allclose(sphere.center_m, [0.2, -0.3, 0.05])
    assert sphere.radius_m == pytest.approx(0.5)
    np.testing.assert_allclose(targets.positions_m[0], [0.1, 0, 0])

    local.origin_m[0] = -1
    sphere.center_m[0] = -1
    view = targets.positions_m
    with pytest.raises(ValueError):
        view.setflags(write=True)
    np.testing.assert_allclose(local.origin_m, [1, 2, 3])
    np.testing.assert_allclose(sphere.center_m, [0.2, -0.3, 0.05])


@pytest.mark.parametrize("ids", [("A",), ("A", "A"), ("A", ""), ("A", 3)])
def test_target_ids_are_nonempty_unique_strings_matching_count(ids: tuple) -> None:
    with pytest.raises((TypeError, ValueError), match="ids"):
        TargetPoints(np.zeros((2, 3)) * u.m, ids, Frame("world"))


def test_bare_string_ids_are_rejected() -> None:
    world = Frame("world")
    with pytest.raises((TypeError, ValueError), match="ids"):
        TargetPoints(np.zeros((2, 3)) * u.m, "AB", world)
    sphere = Sphere(np.zeros(3) * u.m, 1 * u.m, world)
    targets = TargetPoints(np.zeros((1, 3)) * u.m, ("A",), world)
    request = ResolvedEvaluation(
        UniformSphereModelSpec(sphere, 1 * u.kg / u.m**3),
        targets,
        world,
        world,
        ("potential",),
        None,
    )
    assert (
        EvaluationSetup(request).context.binding("potential").field_definition
        is request.model_spec.field_definition
    )


def test_quantity_units_and_finite_inputs_are_required() -> None:
    world = Frame("world")
    with pytest.raises((TypeError, u.UnitConversionError)):
        Sphere([0, 0, 0], 1 * u.m, world)
    with pytest.raises((TypeError, u.UnitConversionError)):
        TargetPoints(np.zeros((1, 3)) * u.s, ("A",), world)
    with pytest.raises(ValueError, match="positive"):
        Sphere(np.zeros(3) * u.m, 0 * u.m, world)
    with pytest.raises(ValueError, match="finite"):
        TargetPoints(np.array([[np.nan, 0, 0]]) * u.m, ("A",), world)
    with pytest.raises(ValueError, match="real"):
        Frame("complex", parent=world, basis=np.eye(3, dtype=complex))
    with pytest.raises(ValueError, match="real"):
        Frame("complex", parent=world, origin=np.array([1 + 1j, 0, 0]) * u.m)
    with pytest.raises(ValueError, match="real"):
        Sphere(np.array([1 + 1j, 0, 0]) * u.m, 1 * u.m, world)
    with pytest.raises(ValueError, match="real"):
        TargetPoints(np.array([[1 + 1j, 0, 0]]) * u.m, ("A",), world)


def test_transform_overflow_is_rejected() -> None:
    world = Frame("world")
    child = Frame("child", origin=[1e308, 0, 0] * u.m, parent=world)
    with pytest.raises(ValueError, match="finite"):
        child.trace_point_m(np.array([1e308, 0, 0]), world)


def test_tensor_transform_rejects_complex_input() -> None:
    world = Frame("world")
    child = Frame("child", parent=world)
    with pytest.raises(ValueError, match="real"):
        child.transform_tensors(1j * np.eye(3), world)


def test_vector_transform_rejects_nonfinite_output() -> None:
    world = Frame("world")
    child = Frame("child", basis=_rz(np.pi / 4), parent=world)
    with pytest.raises(ValueError, match="non-finite"):
        child.transform_vectors_m([1.5e308, 1.5e308, 0], world)


def test_tensor_transform_rejects_nonfinite_output() -> None:
    world = Frame("world")
    child = Frame("child", basis=_rz(np.pi / 4), parent=world)
    with pytest.raises(ValueError, match="non-finite"):
        child.transform_tensors(np.full((3, 3), 1e308), world)


def test_frame_transforms_preserve_batched_and_empty_shapes() -> None:
    world = Frame("world")
    child = Frame("child", basis=_rz(np.pi / 4), parent=world)
    assert child.transform_vectors_m(np.empty((0, 3)), world).shape == (0, 3)
    assert child.transform_tensors(np.empty((0, 3, 3)), world).shape == (0, 3, 3)
    assert child.transform_vectors_m(np.ones((2, 3)), world).shape == (2, 3)
    assert child.transform_tensors(np.ones((2, 3, 3)), world).shape == (2, 3, 3)


def test_resolved_evaluation_shares_real_source_and_exposes_bindings() -> None:
    world = Frame("world")
    model_frame = Frame("model", origin=[2, 3, 4] * u.m, parent=world)
    target_frame = Frame("targets", origin=[3, 3, 4] * u.m, parent=world)
    sphere = Sphere(np.zeros(3) * u.m, 2 * u.m, model_frame)
    model = UniformSphereModelSpec(sphere, 1 * u.kg / u.m**3)
    targets = TargetPoints(
        np.array([[0, 0, 0], [1, 0, 0]]) * u.m, ("A", "B"), target_frame
    )
    positions = np.array([[0.0, 0.0, 0.0]])
    weights = np.array([1.0])
    source = DiscreteSource(
        geometry=sphere,
        density=1 * u.kg / u.m**3,
        frame=model_frame,
        positions_m=positions,
        weights_kg=weights,
        discretization={"spacing_m": 0.5, "max_points": 100},
    )
    evaluation = ResolvedEvaluation(
        model=model,
        targets=targets,
        computation_frame=model_frame,
        output_frame=target_frame,
        observables=("potential", "acceleration"),
        source=source,
    )
    positions[0, 0] = 100
    weights[0] = 100
    np.testing.assert_allclose(source.positions_m, [[0, 0, 0]])
    np.testing.assert_allclose(source.weights_kg, [1])
    assert evaluation.source is source
    np.testing.assert_allclose(evaluation.target_positions_m, [[1, 0, 0], [2, 0, 0]])
    context = EvaluationSetup(evaluation).context
    bindings = {name: context.binding(name) for name in evaluation.observables}
    assert all(
        binding.field_definition is model.field_definition
        for binding in bindings.values()
    )
    assert bindings["potential"].frame is None
    assert bindings["acceleration"].frame is target_frame
    assert bindings["acceleration"].target_ids == ("A", "B")
    assert evaluation.parameters["continuous_mass"].unit == u.kg
    assert evaluation.parameters["discrete_mass"][0].to_value(u.kg) == pytest.approx(
        1.0
    )
    assert set(evaluation.formulas["observable"]) == {"potential", "acceleration"}
    assert "sum_j" in evaluation.formulas.loc[0, "expression"]
    assert "gragra.observables.potential.point_potential" in set(
        evaluation.references["source"]
    )


def test_shared_model_and_record_validate_density_and_execution_facts() -> None:
    sphere = Sphere(np.zeros(3) * u.m, 2 * u.m, Frame("world"))
    with pytest.raises(ValueError, match="real"):
        UniformSphereModelSpec(sphere, (1 + 2j) * u.kg / u.m**3)
    with pytest.raises(ValueError, match="finite"):
        UniformSphereModelSpec(sphere, np.inf * u.kg / u.m**3)
    spec = UniformSphereModelSpec(sphere, 3 * u.g / u.cm**3)
    assert spec.density_kg_m3 == pytest.approx(3000)
    assert spec.continuous_mass.to_value(u.kg) == pytest.approx(
        4 * np.pi / 3 * 8 * 3000
    )
    record = EvaluationRecord(backend="numpy", versions={"numpy": "1.0"})
    assert record.backend == "numpy" and record.g_applications == 1
    with pytest.raises(TypeError):
        record.versions["numpy"] = "changed"


def test_equivalent_density_units_share_one_resolved_model() -> None:
    frame = Frame("world")
    sphere = Sphere(np.zeros(3) * u.m, 1 * u.m, frame)
    spec = UniformSphereModelSpec(sphere, np.nextafter(2.7, np.inf) * u.g / u.cm**3)
    source = DiscreteSource(
        sphere,
        2700 * u.kg / u.m**3,
        frame,
        [[0, 0, 0]],
        [1.0],
        {"spacing_m": 1.0, "max_points": 100},
    )
    targets = TargetPoints(np.array([[0, 0, 2]]) * u.m, ("A",), frame)
    resolved = ResolvedEvaluation(spec, targets, frame, frame, ("potential",), source)
    assert resolved.source is source


def test_pair_precision_uses_only_actual_path_and_eps64() -> None:
    world = Frame("world")
    source_frame = Frame("source", origin=[1e12, 0, 0] * u.m, parent=world)
    target_frame = Frame("target", origin=[1e12, 0, 0] * u.m, parent=world)
    trace = target_frame.trace_point_m(np.zeros(3), source_frame)
    assert trace.edge_count == 2
    assert trace.scale_m == pytest.approx(1e12)
    tau = 64 * 2 * np.finfo(np.float64).eps * 1e12

    with pytest.raises(CoordinatePrecisionError) as caught:
        check_pair_precision(
            np.array([tau, 0, 0]),
            trace,
            target_id="A",
            source_index=7,
            observables=("potential", "gradient"),
        )
    error = caught.value
    assert error.target_id == "A" and error.source_index == 7
    assert error.distance_m == pytest.approx(tau)
    assert error.threshold_m == pytest.approx(tau)
    assert error.edge_count == 2 and error.scale_m == pytest.approx(1e12)
    assert error.unit == "m" and error.observables == ("potential", "gradient")
    assert error.reason == "coordinate_precision"

    check_pair_precision(
        np.array([np.nextafter(tau, np.inf), 0, 0]),
        trace,
        target_id="A",
        source_index=7,
        observables=("potential",),
    )
    with pytest.raises(CoordinatePrecisionError, match="collision"):
        check_pair_precision(
            np.zeros(3),
            trace,
            target_id="A",
            source_index=7,
            observables=("potential",),
        )


def test_same_frame_nearpoint_has_no_precision_diagnostic() -> None:
    frame = Frame("world")
    trace = frame.trace_point_m(np.zeros(3), frame)
    check_pair_precision(
        np.array([1e-15, 0, 0]),
        trace,
        target_id="A",
        source_index=0,
        observables=("potential",),
    )
    with pytest.raises(CoordinatePrecisionError, match="collision"):
        check_pair_precision(
            np.zeros(3),
            trace,
            target_id="A",
            source_index=0,
            observables=("potential",),
        )


def test_pair_trace_uses_common_ancestor_and_ignores_unrelated_points() -> None:
    world = Frame("world")
    unrelated = Frame("far", origin=[1e18, 0, 0] * u.m, parent=world)
    parent = Frame("parent", origin=[1e12, 0, 0] * u.m, parent=world)
    source = Frame("source", origin=[1, 2, 3] * u.m, parent=parent)
    target = Frame("target", origin=[1, 2, 3] * u.m, parent=parent)
    trace = target.trace_point_m(np.zeros(3), source)
    assert trace.edge_count == 2
    assert trace.scale_m == pytest.approx(3)
    assert unrelated.root is world
    with pytest.raises(CoordinatePrecisionError) as caught:
        check_pair_precision(
            np.array([1e-14, 0, 0]),
            trace,
            target_id="A",
            source_index=0,
            observables=("gradient",),
        )
    assert caught.value.scale_m == pytest.approx(3)


def test_equivalent_cm_and_m_frame_inputs_have_same_pair_threshold() -> None:
    world = Frame("world")
    source_m = Frame("source", origin=[1, 0, 0] * u.m, parent=world)
    target_m = Frame("target", origin=[1, 0, 0] * u.m, parent=world)
    source_cm = Frame("source", origin=[100, 0, 0] * u.cm, parent=world)
    target_cm = Frame("target", origin=[100, 0, 0] * u.cm, parent=world)
    trace_m = target_m.trace_point_m(np.array([20, 0, 0]), source_m)
    trace_cm = target_cm.trace_point_m(np.array([20, 0, 0]), source_cm)
    assert trace_m.point_m.tolist() == trace_cm.point_m.tolist()
    assert trace_m.scale_m == pytest.approx(trace_cm.scale_m)
    for trace in (trace_m, trace_cm):
        with pytest.raises(CoordinatePrecisionError) as caught:
            check_pair_precision(
                np.array([20 + 1e-14, 0, 0]),
                trace,
                target_id="A",
                source_index=0,
                observables=("potential",),
            )
        assert caught.value.threshold_m == pytest.approx(
            64 * 2 * np.finfo(np.float64).eps * 20
        )
