"""D3 high-level contracts for the spherical-cavity phasor model."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeries
from gwpy.time import LIGOTimeGPS

from gragra.geometry._frame import Frame
from gragra.geometry._sphere import Sphere
from gragra.results import (
    ChangeForm,
    Contribution,
    FieldKind,
    PhasorRepresentation,
    PotentialReference,
    ReferenceDefinition,
    ReferenceStatus,
    ReferenceTemporalKind,
)
from gragra.targets._high_points import TargetPoints


def _module():
    return importlib.import_module("gragra.analytic.seismic._spherical_cavity_model")


def _inputs(*, mode: str = "P", part: str = "total", phase_speed=None, epoch=None):
    frame = Frame("cavity")
    frequencies = np.array([2.0, 3.0]) * u.Hz
    if phase_speed is None:
        phase_speed = (2 * np.pi * frequencies[0].to_value(u.Hz) / 0.8) * u.m / u.s
    return {
        "geometry": Sphere(np.zeros(3) * u.m, 1.0 * u.m, frame),
        "density": 2700 * u.kg / u.m**3,
        "phase_speed": phase_speed,
        "displacement": FrequencySeries(
            np.array([1.0 + 0.4j, -0.2 + 0.7j]),
            frequencies=frequencies,
            unit=u.m,
            epoch=epoch,
        ),
        "representation": PhasorRepresentation(),
        "reference_time": (1_000_000_000, 123),
        "wave_type": mode,
        "propagation_direction": np.array([0.0, 0.0, 1.0]),
        "polarization": np.array([0.0, 0.0, 1.0] if mode == "P" else [1.0, 0.0, 0.0]),
        "phase_origin": np.array([0.2, -0.1, 0.3]) * u.m,
        "part": part,
    }


def _targets(frame: Frame) -> TargetPoints:
    return TargetPoints(
        np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.4], [1.4, 0.2, -0.1]]) * u.m,
        ("center", "inside", "outside"),
        frame,
    )


@pytest.mark.parametrize(
    ("mode", "part", "potential_reference"),
    [
        ("P", "bulk", PotentialReference.SPHERICAL_MEAN_ZERO_AT_INFINITY),
        ("P", "wall", PotentialReference.ZERO_AT_INFINITY),
        ("P", "total", PotentialReference.SPHERICAL_MEAN_ZERO_AT_INFINITY),
        ("S", "bulk", PotentialReference.ZERO_AT_INFINITY),
        ("S", "wall", PotentialReference.ZERO_AT_INFINITY),
        ("S", "total", PotentialReference.ZERO_AT_INFINITY),
    ],
)
def test_model_field_definition_records_selected_part_and_reference(
    mode, part, potential_reference
):
    model = _module().spherical_cavity(**_inputs(mode=mode, part=part))

    definition = model.spec.field_definition
    assert definition.kind is FieldKind.FIELD_CHANGE
    assert definition.source_kind == "spherical_cavity_perturbation"
    assert definition.potential_reference is potential_reference
    assert definition.contribution == (
        Contribution("total", ("bulk", "wall"))
        if part == "total"
        else Contribution("selected", (part,))
    )
    assert definition.change is not None
    assert definition.change.form is ChangeForm.LINEARIZED_RESPONSE
    assert definition.change.reference == ReferenceDefinition(
        ReferenceStatus.DEFINED,
        ReferenceTemporalKind.STATE,
        "zero_displacement_spherical_cavity",
    )
    assert definition.change.changed_input == "displacement"
    assert definition.change.order == 1
    assert definition.change.fixed_conditions == tuple(
        sorted(
            (
                "geometry",
                "density",
                "phase_speed",
                "wave_type",
                "propagation_direction",
                "polarization",
                "phase_origin",
            )
        )
    )


@pytest.mark.parametrize("part", ["bulk", "wall", "total"])
def test_run_uses_exact_native_selection_and_applies_g_once(monkeypatch, part):
    from gragra.analytic.values import GFreeGravityValues
    from gragra.constants import G_SI

    module = _module()
    captured = {}
    native_values = {
        name: GFreeGravityValues(
            np.full((2, 3), value, dtype=complex),
            np.full((2, 3, 3), value + 1j, dtype=complex),
            np.full((2, 3, 3, 3), value + 2j, dtype=complex),
        )
        for name, value in (("bulk", 2.0), ("wall", 3.0), ("total", 17.0))
    }

    def native(targets_m, **kwargs):
        captured["targets"] = np.array(targets_m, copy=True)
        captured.update(kwargs)
        return native_values

    monkeypatch.setattr(module, "spherical_cavity_observables_g_free", native)
    model = module.spherical_cavity(**_inputs(part=part))
    result = model.at(_targets(model.geometry.frame)).run()

    assert len(captured["targets"]) == 3
    assert captured["radius_m"] == 1.0
    assert captured["density_kg_m3"] == 2700.0
    assert captured["wave_type"] == "P"
    assert captured["wavenumber_rad_m"].shape == (2,)
    assert captured["displacement_m"].shape == (2,)
    for observable, offset in (
        ("potential", 0.0),
        ("acceleration", 1j),
        ("gradient", 2j),
    ):
        matrix = getattr(result, observable)
        expected = 17.0 if part == "total" else {"bulk": 2.0, "wall": 3.0}[part]
        expected = (expected + offset) * G_SI
        np.testing.assert_array_equal(np.asarray(matrix)[0, 0], [expected, expected])
    assert result.g_applications == 1
    assert result.record.g_applications == 1
    assert (
        result.field_definition.contribution == model.spec.field_definition.contribution
    )


def test_phase_speed_scalar_series_equivalence_and_exact_axis_rejection():
    from gragra.analytic.seismic._phase_speed import (
        _spherical_phase_speed_to_wavenumbers,
    )

    module = _module()
    frequencies = np.array([2.0, 3.0]) * u.Hz
    scalar = (2 * np.pi * frequencies[0].to_value(u.Hz) / 0.8) * u.m / u.s
    series = FrequencySeries(
        [scalar.to_value(u.m / u.s)] * 2, f0=2, df=1, unit=u.m / u.s
    )
    scalar_model = module.spherical_cavity(**_inputs(phase_speed=scalar))
    series_model = module.spherical_cavity(**_inputs(phase_speed=series))
    scalar_result = scalar_model.at(_targets(scalar_model.geometry.frame)).run()
    series_result = series_model.at(_targets(series_model.geometry.frame)).run()
    for name in ("potential", "acceleration", "gradient"):
        np.testing.assert_allclose(
            np.asarray(getattr(scalar_result, name)),
            np.asarray(getattr(series_result, name)),
            rtol=2e-14,
            atol=0,
        )

    bad_axis = FrequencySeries([3000.0, 3000.0], f0=2, df=1.01, unit=u.m / u.s)
    with pytest.raises(ValueError, match="exactly match"):
        module.spherical_cavity(**_inputs(phase_speed=bad_axis))
    with pytest.raises(ValueError, match="exactly match"):
        _spherical_phase_speed_to_wavenumbers(bad_axis, frequencies, mode="P")


def test_reference_time_is_required_and_epoch_is_provenance_only():
    module = _module()
    inputs = _inputs(epoch=LIGOTimeGPS(1_234_567_890, 999))
    model = module.spherical_cavity(**inputs)
    assert model.spec.displacement_epoch_gps is not None
    assert model.spec.displacement_epoch_gps != model.spec.reference_time_gps

    no_epoch_inputs = _inputs()
    no_epoch_model = module.spherical_cavity(**no_epoch_inputs)
    with_epoch_result = model.at(_targets(model.geometry.frame)).run()
    no_epoch_result = no_epoch_model.at(_targets(no_epoch_model.geometry.frame)).run()
    assert with_epoch_result.context.reference_time_gps == (1_000_000_000, 123)
    for name in ("potential", "acceleration", "gradient"):
        np.testing.assert_array_equal(
            np.asarray(getattr(with_epoch_result, name)),
            np.asarray(getattr(no_epoch_result, name)),
        )

    del inputs["reference_time"]
    with pytest.raises(TypeError, match="reference_time"):
        module.spherical_cavity(**inputs)


def test_nonunit_direction_and_polarization_are_normalized_in_spec_and_phase():
    module = _module()
    normalized_inputs = _inputs()
    nonunit_inputs = dict(normalized_inputs)
    nonunit_inputs["propagation_direction"] = np.array([0.0, 0.0, 1e308])
    nonunit_inputs["polarization"] = np.array([0.0, 0.0, 1e307])

    normalized_model = module.spherical_cavity(**normalized_inputs)
    nonunit_model = module.spherical_cavity(**nonunit_inputs)

    np.testing.assert_array_equal(
        nonunit_model.spec.propagation_direction,
        normalized_model.spec.propagation_direction,
    )
    np.testing.assert_array_equal(
        nonunit_model.spec.polarization,
        normalized_model.spec.polarization,
    )
    parameters = nonunit_model.at(_targets(nonunit_model.geometry.frame)).parameters
    assert parameters["direction_z"][0] == 1.0
    assert parameters["polarization_z"][0] == 1.0

    normalized_result = normalized_model.at(
        _targets(normalized_model.geometry.frame)
    ).run()
    nonunit_result = nonunit_model.at(_targets(nonunit_model.geometry.frame)).run()
    for name in ("potential", "acceleration", "gradient"):
        np.testing.assert_array_equal(
            np.asarray(getattr(nonunit_result, name)),
            np.asarray(getattr(normalized_result, name)),
        )


@pytest.mark.parametrize("field", ("propagation_direction", "polarization"))
@pytest.mark.parametrize("boolean", (True, np.bool_(True)))
def test_direction_vectors_reject_mixed_boolean_components(field, boolean):
    inputs = _inputs()
    inputs[field] = [boolean, 0, 0]

    with pytest.raises(TypeError, match="must not contain bool"):
        _module().spherical_cavity(**inputs)


def test_empty_targets_have_an_explicit_validation_error():
    module = _module()
    model = module.spherical_cavity(**_inputs())
    targets = TargetPoints(np.empty((0, 3)) * u.m, (), model.geometry.frame)

    with pytest.raises(ValueError, match="targets must be nonempty"):
        model.at(targets)


def test_any_bin_convergence_failure_fails_the_evaluation():
    module = _module()
    inputs = _inputs()
    inputs["displacement"] = FrequencySeries(
        [1e-6 + 0.2e-6j, 1e-6 + 0.2e-6j],
        frequencies=np.array([1.0, 100.0]) * u.Hz,
        unit=u.m,
    )
    inputs["phase_speed"] = 1.0 * u.m / u.s
    model = module.spherical_cavity(**inputs)
    with pytest.raises(
        ValueError, match="series convergence requires max_degree above ka"
    ):
        model.at(_targets(model.geometry.frame)).run()


def test_spherical_cavity_against_independent_integral_and_s_bulk_zero():
    from gragra.constants import G_SI
    from scripts.research_wave6_spherical_cavity_general_point import (
        _cavity_observables_at_order,
    )

    module = _module()
    for mode in ("P", "S"):
        inputs = _inputs(mode=mode, part="total")
        model = module.spherical_cavity(**inputs)
        result = model.at(_targets(model.geometry.frame)).run()
        spec = model.spec
        k = spec.wavenumber_rad_m
        targets = _targets(model.geometry.frame)
        reference = _cavity_observables_at_order(
            targets.positions_m,
            radius_m=spec.geometry.radius_m,
            density_kg_m3=spec.density_kg_m3,
            wavenumber_rad_m=float(k[0]),
            displacement_m=spec.displacement_m[0],
            propagation_direction=spec.propagation_direction,
            polarization=spec.polarization,
            wave_type=mode,
            order=72,
        )
        phase_origin_factor = np.exp(
            -1j * k[0] * np.dot(spec.propagation_direction, spec.phase_origin_m)
        )
        for name in ("potential", "acceleration", "gradient"):
            observed = np.asarray(getattr(result, name))[:, :, 0]
            if name == "potential":
                observed = observed[:, 0]
            elif name == "gradient":
                observed = observed.reshape(len(targets), 3, 3)
            np.testing.assert_allclose(
                observed,
                np.asarray(reference["total"][name]) * phase_origin_factor * G_SI,
                rtol=1e-10,
                atol=1e-15,
            )
        if mode == "S":
            bulk_inputs = _inputs(mode="S", part="bulk")
            bulk_model = module.spherical_cavity(**bulk_inputs)
            bulk_result = bulk_model.at(_targets(bulk_model.geometry.frame)).run()
            for name in ("potential", "acceleration", "gradient"):
                np.testing.assert_array_equal(np.asarray(getattr(bulk_result, name)), 0)


def test_phase_origin_phasor_rotation_and_geometry_frame_components():
    from gragra.analytic.spherical_cavity_series import (
        spherical_cavity_observables_g_free,
    )
    from gragra.constants import G_SI

    root = Frame("root")
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    geometry_frame = Frame("geometry", basis=rotation, parent=root)
    inputs = _inputs()
    inputs["geometry"] = Sphere(np.zeros(3) * u.m, 1.0 * u.m, geometry_frame)
    inputs["phase_origin"] = np.array([0.2, -0.1, 0.3]) * u.m
    model = _module().spherical_cavity(**inputs)
    targets = _targets(geometry_frame)
    native = spherical_cavity_observables_g_free(
        targets.positions_m,
        radius_m=1.0,
        density_kg_m3=2700.0,
        wavenumber_rad_m=model.spec.wavenumber_rad_m,
        displacement_m=model.spec.displacement_m,
        propagation_direction=model.spec.propagation_direction,
        polarization=model.spec.polarization,
        wave_type="P",
        center_m=model.spec.geometry.center_m,
        phase_origin_m=model.spec.phase_origin_m,
    )["total"]
    result = model.at(targets, output_frame=root).run()
    expected_acceleration = (
        native.acceleration @ geometry_frame.rotation_to(root).T * G_SI
    )
    expected_gradient = (
        np.einsum(
            "ij,ktjl,ml->ktim",
            geometry_frame.rotation_to(root),
            native.gradient,
            geometry_frame.rotation_to(root),
        )
        * G_SI
    )
    np.testing.assert_allclose(
        np.asarray(result.acceleration)[:, :, :],
        expected_acceleration.transpose(1, 2, 0),
    )
    np.testing.assert_allclose(
        np.asarray(result.gradient)[:, :, :],
        expected_gradient.reshape(2, 3, 9).transpose(1, 2, 0),
    )

    shifted_inputs = dict(inputs, phase_origin=np.array([0.2, -0.1, 0.4]) * u.m)
    shifted = _module().spherical_cavity(**shifted_inputs).at(targets).run()
    expected_phase = np.exp(
        -1j * model.spec.wavenumber_rad_m * 0.1 * model.spec.propagation_direction[2]
    )
    # The phase origin shift is a geometry-basis displacement and only affects
    # the native phasor through d·(center-origin).
    if expected_phase.shape == (2,):
        expected_phase = expected_phase[None, None, :]
    np.testing.assert_allclose(
        np.asarray(shifted.potential),
        np.asarray(model.at(targets).run().potential) * expected_phase,
        rtol=1e-12,
        atol=0,
    )
