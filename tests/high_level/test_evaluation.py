"""Tests for analytic and midpoint point-sum evaluation of a uniform sphere."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gragra._evaluation_setup import CoordinatePrecisionError, DiscreteSource
from gragra.analytic.objects import uniform_sphere
from gragra.analytic.uniform_sphere import uniform_sphere_potential
from gragra.constants import G_SI
from gragra.evaluation import EvaluationError, _compute_values, _transform_values
from gragra.geometry import Appearance
from gragra.geometry._frame import Frame
from gragra.geometry._sphere import Sphere
from gragra.geometry.primitives import Sphere as CartesianSphere
from gragra.gravity_sources import WeightedPointSource
from gragra.gravity_sources._discretize import discretize
from gragra.gravity_sources._models import point_sum
from gragra.observables import (
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)
from gragra.observables.analytic import uniform_sphere_observables
from gragra.targets._high_points import TargetPoints


def _case():
    frame = Frame("body")
    geometry = Sphere(np.array([1.0, -2.0, 0.5]) * u.m, 2.0 * u.m, frame)
    density = 2.7 * u.g / u.cm**3
    points = np.array(
        [
            [1.3, -1.6, 0.7],  # interior
            [3.0, -2.0, 0.5],  # boundary
            [4.0, 2.0, 0.5],  # exterior
        ]
    )
    targets = TargetPoints(points * u.m, ("inside", "surface", "outside"), frame)
    return frame, geometry, density, points, targets


def test_analytic_values_match_existing_si_wrapper_and_physics_checks() -> None:
    frame, geometry, density, points, targets = _case()
    setup = uniform_sphere(geometry, density).at(targets)
    values = _compute_values(setup.resolved)
    reference = uniform_sphere_observables(
        geometry.center_m, geometry.radius_m, 2700.0, points
    )

    for name in ("potential", "acceleration", "gradient"):
        scale = float(np.max(np.abs(getattr(reference, name))))
        np.testing.assert_allclose(
            values[name], getattr(reference, name), rtol=1e-12, atol=1e-12 * scale
        )
    assert values["potential"][2] < 0
    relative = points[2] - geometry.center_m
    assert np.dot(values["acceleration"][2], relative) < 0
    np.testing.assert_allclose(
        values["gradient"][2], values["gradient"][2].T, rtol=0, atol=1e-28
    )
    assert np.trace(values["gradient"][2]) == pytest.approx(0, abs=1e-28)
    assert frame is geometry.frame
    assert setup.parameters["continuous_mass"][0].unit == u.kg
    assert set(setup.formulas["observable"]) == {
        "potential",
        "acceleration",
        "gradient",
    }
    assert any(
        "uniform_sphere_observables" in source for source in setup.references["source"]
    )


def test_potential_only_run_skips_unrequested_analytic_gradient() -> None:
    frame = Frame("tiny")
    geometry = Sphere(np.zeros(3) * u.m, 1e-102 * u.m, frame)
    targets = TargetPoints([[1e-102, 0, 0]] * u.m, ("edge",), frame)
    setup = uniform_sphere(geometry, 1e308 * u.kg / u.m**3).at(
        targets, observables=("potential",)
    )

    result = setup.run()

    expected = G_SI * uniform_sphere_potential(
        geometry.center_m,
        geometry.radius_m,
        1e308,
        targets.positions_m,
    )
    assert result.potential.shape == (1,)
    assert result.potential.unit == u.m**2 / u.s**2
    np.testing.assert_allclose(result.potential.value, expected, rtol=1e-14)
    assert result.acceleration is None
    assert result.gradient is None
    assert result.record.g_applications == 1

    references = setup.references
    details = references.set_index("source")["detail"]
    assert (
        details["gragra.analytic.uniform_sphere.uniform_sphere_potential"]
        == "G-free analytical uniform-sphere calculation"
    )
    assert details["gragra.constants.G_SI"] == (
        "G_SI is applied once per requested result"
    )
    assert details["CODATA 2018"] == "gravitational constant G_SI"
    assert not any(
        "uniform_sphere_observables" in source for source in references["source"]
    )


def test_analytic_extreme_finite_distance_fails_with_target_context() -> None:
    frame = Frame("extreme-analytic")
    geometry = Sphere(np.zeros(3) * u.m, 1 * u.m, frame)
    targets = TargetPoints([[1e200, 1e200, 0]] * u.m, ("far",), frame)
    setup = uniform_sphere(geometry, 1e300 * u.kg / u.m**3).at(
        targets, observables=("potential",)
    )

    for operation in (setup.validate, setup.run):
        with pytest.raises(EvaluationError) as caught:
            operation()
        error = caught.value
        assert error.target_id == "far"
        assert error.observable == "potential"
        assert error.observables == ("potential",)


@pytest.mark.parametrize(
    "observables",
    [
        ("potential", "acceleration", "gradient"),
        ("gradient", "potential", "acceleration"),
        ("acceleration", "gradient", "potential"),
    ],
)
def test_batched_analytic_extreme_distance_reports_failing_observable(
    observables: tuple[str, ...],
) -> None:
    frame = Frame("batched-extreme")
    geometry = Sphere(np.zeros(3) * u.m, 1 * u.m, frame)
    targets = TargetPoints([[1e70, 0, 0]] * u.m, ("far",), frame)
    setup = uniform_sphere(geometry, 1000 * u.kg / u.m**3).at(
        targets, observables=observables
    )

    with pytest.raises(EvaluationError) as caught:
        setup.run()

    assert caught.value.target_id == "far"
    assert caught.value.observable == "gradient"
    assert caught.value.observables == observables


@pytest.mark.parametrize("observable", ["potential", "acceleration"])
def test_analytic_extreme_distance_individual_observable_succeeds(
    observable: str,
) -> None:
    frame = Frame("single-extreme")
    geometry = Sphere(np.zeros(3) * u.m, 1 * u.m, frame)
    targets = TargetPoints([[1e70, 0, 0]] * u.m, ("far",), frame)

    result = (
        uniform_sphere(geometry, 1000 * u.kg / u.m**3)
        .at(targets, observables=(observable,))
        .run()
    )

    assert getattr(result, observable) is not None


def test_point_sum_extreme_finite_distance_fails_with_target_context() -> None:
    frame = Frame("extreme-point-sum")
    geometry = Sphere(np.zeros(3) * u.m, 1 * u.m, frame)
    source = DiscreteSource(
        geometry,
        1 * u.kg / u.m**3,
        frame,
        [[0, 0, 0]],
        [1e300],
        {"spacing_m": 1.0, "max_points": 1, "grid_points": 1},
    )
    targets = TargetPoints([[1e200, 1e200, 0]] * u.m, ("far",), frame)
    setup = point_sum(source).at(targets, observables=("potential",))

    for operation in (setup.validate, setup.run):
        with pytest.raises(EvaluationError) as caught:
            operation()
        error = caught.value
        assert error.target_id == "far"
        assert error.source_index == 0
        assert error.observable == "potential"
        assert error.observables == ("potential",)


def test_analytic_quantity_units_signed_density_and_fixed_input() -> None:
    frame = Frame("body")
    center = np.array([2.0, -1.0, 4.0]) * u.m
    radius = 150 * u.cm
    geometry = Sphere(center, radius, frame)
    density = -3.0 * u.g / u.cm**3
    original_positions = np.array([[5.0, -1.0, 4.0]]) * u.m
    targets = TargetPoints(original_positions, ("far",), frame)
    model = uniform_sphere(geometry, density)
    setup = model.at(targets)

    center[0] = 50 * u.m
    original_positions[0, 0] = -20 * u.m
    values = _compute_values(setup.resolved)
    assert setup.resolved.model is model
    assert setup.resolved.model_spec.geometry is geometry
    assert setup.resolved.model_spec.density_kg_m3 == pytest.approx(-3000)
    assert values["potential"][0] > 0
    assert not targets.positions_m.flags.writeable
    np.testing.assert_allclose(geometry.center_m, [2, -1, 4])
    np.testing.assert_allclose(targets.positions_m, [[5, -1, 4]])


def test_interior_and_boundary_use_existing_exterior_gradient_branch() -> None:
    frame = Frame("body")
    geometry = Sphere(np.zeros(3) * u.m, 2 * u.m, frame)
    targets = TargetPoints(
        np.array([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]]) * u.m,
        ("inside", "boundary"),
        frame,
    )
    values = _compute_values(
        uniform_sphere(geometry, 1000 * u.kg / u.m**3).at(targets).resolved
    )
    expected_interior_gradient = -4 * np.pi * G_SI * 1000 / 3 * np.eye(3)
    np.testing.assert_allclose(values["gradient"][0], expected_interior_gradient)
    expected_surface_gradient = (
        G_SI * (4 * np.pi / 3 * 2**3 * 1000) * np.diag([2 / 2**3, -1 / 2**3, -1 / 2**3])
    )
    np.testing.assert_allclose(values["gradient"][1], expected_surface_gradient)


def test_output_frame_rotates_vector_and_both_gradient_axes() -> None:
    world = Frame("world")
    angle = np.pi / 3
    basis = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    local = Frame("local", origin=[1, 2, -1] * u.m, basis=basis, parent=world)
    geometry = Sphere([0, 0, 0] * u.m, 2 * u.m, local)
    points = np.array([[3.0, 2.0, 1.0]])
    targets = TargetPoints(points * u.m, ("T",), local)
    setup = uniform_sphere(geometry, 1300 * u.kg / u.m**3).at(
        targets, output_frame=world
    )
    raw = _compute_values(setup.resolved)
    rotation = local.rotation_to(world)
    # Compare against a separately evaluated source-frame reference before rotation.
    local_values = uniform_sphere_observables(
        geometry.center_m, geometry.radius_m, 1300, points
    )
    np.testing.assert_allclose(
        raw["acceleration"][0], rotation @ local_values.acceleration[0]
    )
    np.testing.assert_allclose(
        raw["gradient"][0], rotation @ local_values.gradient[0] @ rotation.T
    )


def test_discrete_values_match_existing_point_observable_wrappers() -> None:
    frame, geometry, density, points, targets = _case()
    source = discretize(geometry, density, 0.6 * u.m)
    setup = point_sum(source).at(targets)
    values = _compute_values(setup.resolved)
    weighted = WeightedPointSource(source.positions_m, source.weights_kg)

    references = {
        "potential": point_potential(weighted, points),
        "acceleration": point_acceleration(weighted, points),
        "gradient": point_gravity_gradient(weighted, points),
    }
    for name, reference in references.items():
        scale = float(np.max(np.abs(reference)))
        np.testing.assert_allclose(
            values[name], reference, rtol=1e-12, atol=1e-12 * scale
        )
    assert setup.resolved.source is source
    assert setup.parameters["source_count"][0] == len(source)
    assert setup.parameters["cell_volume_m3"][0].to_value(u.m**3) == pytest.approx(
        source.discretization["cell_volume_m3"]
    )
    assert "sum_j" in setup.formulas.loc[0, "expression"]
    assert frame is geometry.frame


def test_point_sum_empty_targets_keep_requested_shapes() -> None:
    frame = Frame("empty")
    geometry = Sphere(np.zeros(3) * u.m, 1 * u.m, frame)
    source = discretize(geometry, 1000 * u.kg / u.m**3, 0.5 * u.m)
    targets = TargetPoints(np.empty((0, 3)) * u.m, (), frame)
    setup = point_sum(source).at(targets)

    values = _compute_values(setup.resolved)
    assert setup.validate().checked_pairs == 0
    assert values["potential"].shape == (0,)
    assert values["acceleration"].shape == (0, 3)
    assert values["gradient"].shape == (0, 3, 3)


def test_empty_point_sum_run_keeps_native_quantity_shapes() -> None:
    frame = Frame("empty")
    geometry = Sphere(np.zeros(3) * u.m, 1 * u.m, frame)
    source = discretize(geometry, 1000 * u.kg / u.m**3, 0.5 * u.m)
    targets = TargetPoints(np.empty((0, 3)) * u.m, (), frame)
    result = point_sum(source).at(targets).run()

    assert result.potential.shape == (0,)
    assert result.acceleration.shape == (0, 3)
    assert result.gradient.shape == (0, 3, 3)
    assert result.potential.unit == u.m**2 / u.s**2
    assert result.acceleration.unit == u.m / u.s**2
    assert result.gradient.unit == u.s**-2


def _one_point_source(frame: Frame, position_m: np.ndarray) -> DiscreteSource:
    geometry = Sphere(np.zeros(3) * u.m, 1 * u.m, frame)
    return DiscreteSource(
        geometry,
        1 * u.kg / u.m**3,
        frame,
        np.asarray(position_m, dtype=float).reshape(1, 3),
        np.ones(1),
        {"spacing_m": 1.0, "max_points": 1, "grid_points": 1, "cell_volume_m3": 1.0},
    )


def test_discrete_validation_rejects_collision_with_target_and_source_indices() -> None:
    frame = Frame("source")
    source = _one_point_source(frame, np.zeros(3))
    targets = TargetPoints([[0, 0, 0]] * u.m, ("touch",), frame)
    setup = point_sum(source).at(targets, observables=("potential",))

    with pytest.raises(CoordinatePrecisionError, match="target ID='touch'") as caught:
        setup.validate()
    assert caught.value.source_index == 0
    assert caught.value.reason == "collision"
    assert caught.value.observables == ("potential",)


@pytest.mark.parametrize("factor", [np.nextafter(1.0, 0.0), 1.0])
def test_cross_frame_precision_threshold_stops_validation(factor: float) -> None:
    root = Frame("root")
    source_frame = Frame("source", origin=[1e12, 0, 0] * u.m, parent=root)
    target_frame = Frame("target", origin=[1e12, 0, 0] * u.m, parent=root)
    target = TargetPoints([[0, 0, 0]] * u.m, ("near",), target_frame)
    trace = target_frame.trace_point_m(target.positions_m[0], source_frame)
    threshold = 64 * trace.edge_count * np.finfo(np.float64).eps * trace.scale_m
    source = _one_point_source(source_frame, np.array([factor * threshold, 0, 0]))
    setup = point_sum(source).at(target, observables=("gradient",))

    with pytest.raises(CoordinatePrecisionError) as caught:
        setup.validate()
    assert caught.value.reason == "coordinate_precision"
    assert caught.value.target_id == "near"
    assert caught.value.source_index == 0
    assert caught.value.threshold_m == pytest.approx(threshold)
    assert caught.value.observables == ("gradient",)
    assert setup.resolved.model_spec.geometry is source.geometry


def test_cross_frame_just_above_threshold_and_same_frame_near_point_validate() -> None:
    root = Frame("root")
    source_frame = Frame("source", origin=[1e12, 0, 0] * u.m, parent=root)
    target_frame = Frame("target", origin=[1e12, 0, 0] * u.m, parent=root)
    targets = TargetPoints([[0, 0, 0]] * u.m, ("near",), target_frame)
    trace = target_frame.trace_point_m(targets.positions_m[0], source_frame)
    threshold = 64 * trace.edge_count * np.finfo(np.float64).eps * trace.scale_m
    above = np.nextafter(threshold, np.inf)
    point_sum(_one_point_source(source_frame, [above, 0, 0])).at(targets).validate()

    same_frame = Frame("same")
    local_targets = TargetPoints([[1e-15, 0, 0]] * u.m, ("close",), same_frame)
    point_sum(_one_point_source(same_frame, [0, 0, 0])).at(local_targets).validate()


def test_pair_precision_is_independent_of_unrelated_distant_source() -> None:
    root = Frame("root")
    source_frame = Frame("source", origin=[1e12, 0, 0] * u.m, parent=root)
    target_frame = Frame("target", origin=[1e12, 0, 0] * u.m, parent=root)
    targets = TargetPoints([[0, 0, 0]] * u.m, ("near",), target_frame)
    trace = target_frame.trace_point_m(targets.positions_m[0], source_frame)
    near = np.array(
        [0.5 * 64 * trace.edge_count * np.finfo(float).eps * trace.scale_m, 0, 0]
    )
    first = _one_point_source(source_frame, near)
    second = DiscreteSource(
        first.geometry,
        first.density,
        source_frame,
        np.vstack((near, [1e15, 0, 0])),
        np.ones(2),
        first.discretization,
    )

    failures = []
    for source in (first, second):
        with pytest.raises(CoordinatePrecisionError) as caught:
            point_sum(source).at(targets).validate()
        failures.append(caught.value)
    assert failures[0].threshold_m == failures[1].threshold_m
    assert failures[0].distance_m == failures[1].distance_m


def test_finite_coordinates_with_overflowed_pair_distance_get_context() -> None:
    frame = Frame("far")
    source = _one_point_source(frame, [1e308, 0, 0])
    targets = TargetPoints([[-1e308, 0, 0]] * u.m, ("opposed",), frame)
    setup = point_sum(source).at(targets, observables=("potential", "gradient"))

    with pytest.raises(EvaluationError) as caught:
        setup.validate()
    error = caught.value
    assert error.target_id == "opposed"
    assert error.source_index == 0
    assert error.observable == "potential"
    assert error.observables == ("potential", "gradient")
    assert "not finite" in str(error.cause)


def test_finite_extreme_sphere_input_wraps_si_wrapper_failure_with_target_id() -> None:
    frame = Frame("extreme")
    geometry = Sphere(np.zeros(3) * u.m, 1e200 * u.m, frame)
    targets = TargetPoints([[0, 0, 0]] * u.m, ("center",), frame)
    setup = uniform_sphere(geometry, 1e100 * u.kg / u.m**3).at(
        targets, observables=("potential",)
    )

    with pytest.raises(EvaluationError) as caught:
        setup.run()
    error = caught.value
    assert error.target_id == "center"
    assert error.observable == "potential"
    assert error.observables == ("potential",)
    assert error.__cause__ is error.cause


@pytest.mark.parametrize(
    ("observable", "shape"),
    [("acceleration", (1, 3)), ("gradient", (1, 3, 3))],
)
def test_finite_extreme_frame_transform_failure_gets_target_context(
    observable: str, shape: tuple[int, ...]
) -> None:
    world = Frame("world")
    angle = np.pi / 4
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    local = Frame("local", basis=rotation, parent=world)
    geometry = Sphere([0, 0, 0] * u.m, 1 * u.m, local)
    targets = TargetPoints([[0, 0, 2]] * u.m, ("extreme",), local)
    setup = uniform_sphere(geometry, 1 * u.kg / u.m**3).at(
        targets, observables=(observable,), output_frame=world
    )
    values = {observable: np.full(shape, np.finfo(np.float64).max)}

    with pytest.raises(EvaluationError) as caught:
        _transform_values(setup.resolved, values)
    error = caught.value
    assert error.target_id == "extreme"
    assert error.observable == observable
    assert error.observables == (observable,)
    assert "non-finite" in str(error.cause)


def test_run_returns_native_quantities_and_preserves_resolved_identity() -> None:
    _, geometry, density, _, targets = _case()
    setup = uniform_sphere(geometry, density).at(
        targets, observables=("potential", "acceleration")
    )
    result = setup.run()

    assert result.evaluation is setup
    assert result.context is setup.context
    assert result.potential is result.values.potential
    assert result.acceleration is result.values.acceleration
    assert result.gradient is None
    assert result.potential.unit == u.m**2 / u.s**2
    assert result.acceleration.unit == u.m / u.s**2
    assert result.potential.shape == (len(targets),)
    assert result.acceleration.shape == (len(targets), 3)
    assert result.record.backend == "numpy"
    assert result.record.g_applications == 1
    assert np.isclose(result.record.gravitational_constant_si, G_SI, rtol=0, atol=0)
    assert result.bindings["acceleration"].frame is targets.frame


def test_run_point_sum_and_plot_use_the_same_eager_source(monkeypatch) -> None:
    _, geometry, density, _, targets = _case()
    original = CartesianSphere.to_weighted_points
    calls = 0

    def counted(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(CartesianSphere, "to_weighted_points", counted)
    source = discretize(geometry, density, 0.5 * u.m)
    setup = point_sum(source).at(targets, observables=("potential",))
    result = setup.run()
    figure = setup.plot_setup(max_sources=20)

    assert result.evaluation.resolved.source is source
    assert result.potential.shape == (len(targets),)
    assert figure is not None
    assert setup.resolved.source is source
    assert calls == 1
    from matplotlib import pyplot as plt

    plt.close(figure)


def test_public_evaluation_plot_setup_accepts_appearance_without_changing_results() -> (
    None
):
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgba

    _, geometry, density, _, targets = _case()
    setup = uniform_sphere(geometry, density).at(targets)
    before = setup.run()

    figure = setup.plot_setup(
        appearance=Appearance(color="#123456", alpha=0.6, label="sphere setup")
    )
    try:
        surface = next(
            artist
            for artist in figure.axes[0].collections
            if artist.get_label() == "sphere setup"
        )
        figure.canvas.draw()
        assert surface.get_facecolor()[0] == pytest.approx(
            to_rgba("#123456", alpha=0.6)
        )
        after = setup.run()
        for observable in ("potential", "acceleration", "gradient"):
            np.testing.assert_array_equal(
                getattr(after, observable).value,
                getattr(before, observable).value,
            )
    finally:
        plt.close(figure)
