"""Contracts for static point sums over caller-provided source representations."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gragra._evaluation_setup import CoordinatePrecisionError, DiscreteSource
from gragra.constants import G_SI
from gragra.geometry import Frame
from gragra.geometry._sphere import Sphere
from gragra.gravity_sources import (
    PointMass,
    PointMassCloud,
    VolumeElementSource,
    WeightedPointSource,
)
from gragra.gravity_sources._models import point_sum
from gragra.observables import (
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)
from gragra.results import (
    ChangeDefinition,
    ChangeForm,
    Contribution,
    FieldKind,
    GravityFieldDefinition,
    PotentialReference,
    ReferenceDefinition,
    ReferenceStatus,
    ReferenceTemporalKind,
)
from gragra.targets import TargetPoints


def _source_field() -> GravityFieldDefinition:
    return GravityFieldDefinition.source_field(
        "point_sources", Contribution("total"), PotentialReference.ZERO_AT_INFINITY
    )


def _field_change(
    reference: PotentialReference = PotentialReference.ZERO_AT_INFINITY,
) -> GravityFieldDefinition:
    return GravityFieldDefinition.field_change(
        "density_change",
        Contribution("total"),
        reference,
        ChangeDefinition(
            ChangeForm.FINITE_STATE_DIFFERENCE,
            ReferenceDefinition(
                ReferenceStatus.DEFINED,
                ReferenceTemporalKind.STATE,
                "reference_density",
            ),
            "density",
        ),
    )


def _target(frame: Frame, position=(3.0, 4.0, 0.0)) -> TargetPoints:
    return TargetPoints(np.asarray([position]) * u.m, ("far",), frame)


@pytest.mark.parametrize(
    ("source_kind", "needs_definition"),
    [
        ("point_mass", False),
        ("point_cloud", False),
        ("weighted", True),
        ("volume", True),
    ],
)
def test_point_sum_binds_supported_bare_sources_with_explicit_frame(
    source_kind: str, needs_definition: bool
) -> None:
    frame = Frame(f"{source_kind}-frame")
    sources = {
        "point_mass": PointMass(2.0, [0.0, 0.0, 0.0]),
        "point_cloud": PointMassCloud([2.0], [[0.0, 0.0, 0.0]]),
        "weighted": WeightedPointSource([[0.0, 0.0, 0.0]], [2.0]),
        "volume": VolumeElementSource([[0.0, 0.0, 0.0]], [0.5], 4.0),
    }
    source = sources[source_kind]
    kwargs = {"frame": frame}
    if needs_definition:
        kwargs["field_definition"] = _source_field()

    model = point_sum(source, **kwargs)
    setup = model.at(_target(frame))

    assert setup.resolved.computation_frame is frame
    assert setup.resolved.output_frame is frame
    assert setup.resolved.source.weight_sum_kg == 2.0
    assert setup.resolved.model_spec.field_definition.kind is FieldKind.SOURCE_FIELD
    assert setup.resolved.model_spec.field_definition.contribution == Contribution(
        "total"
    )
    report = setup.validate()
    assert report.source_count == 1
    assert report.target_regions == {}
    if not needs_definition:
        assert (
            setup.resolved.model_spec.field_definition.potential_reference
            is PotentialReference.ZERO_AT_INFINITY
        )


@pytest.mark.parametrize(
    "source",
    [
        PointMass(1.0, [0.0, 0.0, 0.0]),
        PointMassCloud([1.0], [[0.0, 0.0, 0.0]]),
        WeightedPointSource([[0.0, 0.0, 0.0]], [1.0]),
        VolumeElementSource([[0.0, 0.0, 0.0]], [1.0], 1.0),
    ],
)
def test_bare_point_sum_never_infers_a_frame(source: object) -> None:
    with pytest.raises(TypeError, match="frame must be explicitly provided"):
        point_sum(source)


def test_bare_point_sum_requires_a_frame_object() -> None:
    with pytest.raises(TypeError, match="frame must be a Frame"):
        point_sum(PointMass(1.0, [0.0, 0.0, 0.0]), frame="local")


@pytest.mark.parametrize(
    "source",
    [
        WeightedPointSource([[0.0, 0.0, 0.0]], [1.0]),
        VolumeElementSource([[0.0, 0.0, 0.0]], [1.0], 1.0),
    ],
)
def test_general_weight_sources_require_explicit_field_meaning(source: object) -> None:
    with pytest.raises(TypeError, match="field_definition must be provided"):
        point_sum(source, frame=Frame("meaning-required"))


def test_point_mass_run_matches_independent_point_oracle_and_context() -> None:
    frame = Frame("oracle")
    targets = _target(frame)
    model = point_sum(PointMass(2.0, [0.0, 0.0, 0.0]), frame=frame)

    result = model.at(targets).run()

    distance = 5.0
    delta = np.array([3.0, 4.0, 0.0])
    expected_potential = -G_SI * 2.0 / distance
    expected_acceleration = -G_SI * 2.0 * delta / distance**3
    expected_gradient = (
        G_SI
        * 2.0
        * (3.0 * np.outer(delta, delta) / distance**5 - np.eye(3) / distance**3)
    )
    assert result.potential.unit == u.m**2 / u.s**2
    assert result.acceleration.unit == u.m / u.s**2
    assert result.gradient.unit == u.s**-2
    np.testing.assert_allclose(
        result.potential.value, [expected_potential], rtol=1e-14, atol=0.0
    )
    np.testing.assert_allclose(
        result.acceleration.value, [expected_acceleration], rtol=1e-14, atol=0.0
    )
    np.testing.assert_allclose(
        result.gradient.value, [expected_gradient], rtol=1e-14, atol=0.0
    )
    assert result.field_definition is model.spec.field_definition
    assert result.record.g_applications == 1


def test_signed_field_change_is_valid_but_kernel_reference_must_match() -> None:
    frame = Frame("field-change")
    source = WeightedPointSource([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [2.0, -0.5])
    definition = _field_change()

    model = point_sum(source, frame=frame, field_definition=definition)
    result = model.at(_target(frame, (10.0, 0.0, 0.0))).run()

    assert model.spec.field_definition is definition
    assert definition.kind is FieldKind.FIELD_CHANGE
    assert result.field_definition is definition
    with pytest.raises(ValueError, match="ZERO_AT_INFINITY"):
        point_sum(
            source,
            frame=frame,
            field_definition=_field_change(
                PotentialReference.SPHERICAL_MEAN_ZERO_AT_INFINITY
            ),
        )


def test_static_point_binding_rejects_complex_weights() -> None:
    frame = Frame("real-static-only")
    source = WeightedPointSource([[0.0, 0.0, 0.0]], np.array([1.0 + 0.0j]))

    with pytest.raises(ValueError, match="complex weights"):
        point_sum(source, frame=frame, field_definition=_source_field())


def test_binding_snapshots_weights_and_exposes_immutable_provenance() -> None:
    frame = Frame("metadata")
    source = WeightedPointSource([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [2.0, -0.75])
    model = point_sum(
        source,
        frame=frame,
        field_definition=_field_change(),
        provenance={"coordinate_frame": "external-local", "origin_m": [1, 2, 3]},
    )
    binding = model.binding

    assert binding is not None
    assert binding.weight_sum_kg == 1.25
    assert binding.source_type == "WeightedPointSource"
    assert binding.provenance["coordinate_frame"] == "external-local"
    assert binding.provenance["origin_m"] == (1.0, 2.0, 3.0)
    assert not np.shares_memory(binding.positions_m, source.positions_m)
    assert not np.shares_memory(binding.weights_kg, source.weights_kg)
    with pytest.raises(ValueError):
        binding.positions_m.setflags(write=True)
    with pytest.raises(ValueError):
        binding.weights_kg.setflags(write=True)
    with pytest.raises(TypeError):
        binding.provenance["new"] = "value"

    first = model.at(_target(frame)).parameters
    second = model.at(_target(frame)).parameters
    assert first is not second
    assert first.meta["source_provenance"]["origin_m"] == (1.0, 2.0, 3.0)
    first.meta["source_provenance"]["origin_m"] = (9.0, 9.0, 9.0)
    assert second.meta["source_provenance"]["origin_m"] == (1.0, 2.0, 3.0)


def test_same_frame_name_does_not_connect_disconnected_roots() -> None:
    source_frame = Frame("same-name")
    disconnected_target_frame = Frame("same-name")
    model = point_sum(PointMass(1.0, [0.0, 0.0, 0.0]), frame=source_frame)

    with pytest.raises(ValueError, match="disconnected roots"):
        model.at(_target(disconnected_target_frame))


def test_discrete_source_keeps_sphere_meaning_and_geometry_frame() -> None:
    frame = Frame("discrete-frame")
    geometry = Sphere(np.zeros(3) * u.m, 1 * u.m, frame)
    source = DiscreteSource(
        geometry,
        1 * u.kg / u.m**3,
        frame,
        [[0.0, 0.0, 0.0]],
        [1.0],
        {"source_count": 1},
    )
    original = point_sum(source)

    same_definition = point_sum(
        source, frame=frame, field_definition=original.spec.field_definition
    )

    assert same_definition.source is source
    assert same_definition.spec.field_definition is original.spec.field_definition
    with pytest.raises(ValueError, match="Sphere meaning"):
        point_sum(source, field_definition=_field_change())
    with pytest.raises(ValueError, match="geometry.frame"):
        point_sum(source, frame=Frame("discrete-frame"))


def test_source_and_target_frames_transform_points_and_output_vectors() -> None:
    root = Frame("root")
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    source_frame = Frame(
        "source-local",
        origin=np.array([1.0, 0.0, 0.0]) * u.m,
        basis=rotation,
        parent=root,
    )
    output_frame = Frame("output-local", basis=rotation, parent=root)
    targets = TargetPoints(
        np.array([[1.0, 2.0, 0.0], [1.0, 0.0, 3.0]]) * u.m,
        ("side", "above"),
        root,
    )
    model = point_sum(PointMass(2.0, [0.0, 0.0, 0.0]), frame=source_frame)

    result = model.at(targets, output_frame=output_frame).run()

    assert result.evaluation.resolved.output_frame is output_frame
    assert result.bindings["acceleration"].target_ids == ("side", "above")
    expected_in_root = (
        G_SI * 2.0 * np.array([[0.0, -1.0 / 4.0, 0.0], [0.0, 0.0, -1.0 / 9.0]])
    )
    np.testing.assert_allclose(
        result.acceleration.to_value(u.m / u.s**2),
        expected_in_root @ rotation,
        rtol=1e-14,
        atol=0.0,
    )


def test_signed_point_sum_matches_existing_numpy_point_kernels() -> None:
    frame = Frame("signed-kernel-parity")
    positions = np.array([[0.0, 0.0, 0.0], [1.0, -2.0, 0.5]])
    targets = np.array([[4.0, 2.0, -1.0], [-3.0, 1.0, 0.0]])
    source = WeightedPointSource(positions, [3.0, -1.25])
    model = point_sum(source, frame=frame, field_definition=_field_change())

    result = model.at(TargetPoints(targets * u.m, ("first", "second"), frame)).run()
    expected = (
        point_potential(source, targets, backend="numpy"),
        point_acceleration(source, targets, backend="numpy"),
        point_gravity_gradient(source, targets, backend="numpy"),
    )

    for actual, reference in zip(
        (result.potential.value, result.acceleration.value, result.gradient.value),
        expected,
        strict=True,
    ):
        np.testing.assert_allclose(actual, reference, rtol=1e-14, atol=0.0)


def test_bound_point_source_retains_collision_context() -> None:
    frame = Frame("raw-collision")
    setup = point_sum(PointMass(1.0, [0.0, 0.0, 0.0]), frame=frame).at(
        TargetPoints([[0.0, 0.0, 0.0]] * u.m, ("hit",), frame)
    )

    with pytest.raises(CoordinatePrecisionError) as caught:
        setup.validate()
    assert caught.value.target_id == "hit"
    assert caught.value.source_index == 0


def test_raw_source_cross_frame_precision_error_is_preserved() -> None:
    root = Frame("precision-root")
    source_frame = Frame("precision-local", origin=[1e6, 0.0, 0.0] * u.m, parent=root)
    target = TargetPoints([[1e6 + 1e-8, 0.0, 0.0]] * u.m, ("near",), root)
    setup = point_sum(PointMass(1.0, [0.0, 0.0, 0.0]), frame=source_frame).at(target)

    with pytest.raises(CoordinatePrecisionError) as caught:
        setup.validate()
    assert caught.value.reason == "coordinate_precision"
    assert caught.value.target_id == "near"
    assert caught.value.source_index == 0


def test_generic_static_plot_setup_reports_unsupported_route() -> None:
    frame = Frame("no-generic-plot")
    setup = point_sum(PointMass(1.0, [0.0, 0.0, 0.0]), frame=frame).at(_target(frame))

    with pytest.raises(NotImplementedError, match="Sphere-backed"):
        setup.plot_setup()
