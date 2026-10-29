"""High-level spherical-shell model contract."""

import numpy as np
import pytest
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeries
from gwpy.time import LIGOTimeGPS

from gragra.geometry._frame import Frame
from gragra.geometry._spherical_shell import SphericalShell
from gragra.results import (
    ChangeForm,
    Contribution,
    FieldKind,
    PhasorRepresentation,
    PotentialReference,
    ReferenceStatus,
    ReferenceTemporalKind,
)


def test_factory_owns_inputs_and_declares_linearized_shell_change():
    from gragra.analytic.seismic._spherical_shell_model import spherical_shell

    frame = Frame("shell")
    geometry = SphericalShell([0, 0, 0] * u.m, 10 * u.m, 40 * u.m, frame)
    displacement = FrequencySeries(
        [2 + 3j, -1 + 4j], frequencies=[1, 2] * u.Hz, unit=u.cm
    )
    origin = np.array([2.0, -3.0, 1.0]) * u.m
    direction = np.array([0.0, 0.0, 2.0])
    polarization = np.array([0.0, 0.0, 1.0])

    model = spherical_shell(
        geometry,
        2.5 * u.g / u.cm**3,
        2.0 * u.km / u.s,
        displacement,
        PhasorRepresentation(),
        (1_000_000_000, 12),
        origin,
        direction,
        polarization,
        "P",
    )
    displacement[0] = 99 * u.cm
    origin[0] = 100 * u.m
    direction[2] = 4
    polarization[2] = 3

    np.testing.assert_allclose(model.spec.density_kg_m3, 2500)
    np.testing.assert_array_equal(
        model.spec.displacement_m, [0.02 + 0.03j, -0.01 + 0.04j]
    )
    np.testing.assert_array_equal(model.spec.phase_origin_m, [2.0, -3.0, 1.0])
    np.testing.assert_array_equal(model.spec.propagation_direction, [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(model.spec.polarization, [0.0, 0.0, 1.0])
    assert model.spec.reference_time_gps == (1_000_000_000, 12)

    definition = model.spec.field_definition
    assert definition.kind is FieldKind.FIELD_CHANGE
    assert definition.source_kind == "spherical_shell_perturbation"
    assert definition.contribution == Contribution(
        "total", ("bulk", "inner_wall", "outer_boundary")
    )
    assert definition.potential_reference is PotentialReference.ZERO_AT_INFINITY
    assert definition.change.form is ChangeForm.LINEARIZED_RESPONSE
    assert definition.change.changed_input == "displacement"
    assert definition.change.order == 1
    assert definition.change.reference.status is ReferenceStatus.DEFINED
    assert definition.change.reference.temporal_kind is ReferenceTemporalKind.STATE
    assert definition.change.reference.state_kind == "zero_displacement_spherical_shell"
    assert set(definition.change.fixed_conditions) == {
        "geometry",
        "density",
        "phase_speed",
        "wave_type",
        "propagation_direction",
        "polarization",
        "phase_origin",
    }


def _inputs(
    *,
    contribution="total",
    displacement=None,
    phase_speed=None,
    geometry=None,
    direction=(0, 0, 1),
    polarization=(0, 0, 1),
    phase_origin=None,
):
    from gragra.analytic.seismic._spherical_shell_model import spherical_shell

    frame = Frame("shell") if geometry is None else geometry.frame
    shell = (
        SphericalShell([0, 0, 0] * u.m, 10 * u.m, 40 * u.m, frame)
        if geometry is None
        else geometry
    )
    if displacement is None:
        displacement = FrequencySeries(
            [1e-5 + 2e-6j, -3e-6 + 4e-6j],
            frequencies=[1.0, 2.0] * u.Hz,
            unit=u.m,
        )
    if phase_speed is None:
        phase_speed = 2.0 * u.km / u.s
    if phase_origin is None:
        phase_origin = [0, 0, 0] * u.m
    return spherical_shell(
        shell,
        2500 * u.kg / u.m**3,
        phase_speed,
        displacement,
        PhasorRepresentation(),
        (1_000_000_000, 12),
        phase_origin,
        direction,
        polarization,
        "P",
        contribution=contribution,
    )


def _one_target(frame, position=(0.0, 0.0, 0.0)):
    from gragra.targets._high_points import TargetPoints

    return TargetPoints(np.asarray([position]) * u.m, ("target",), frame)


def test_nonunit_direction_and_polarization_are_normalized_in_spec_and_phase():
    phase_origin = np.array([2.0, -3.0, 1.0]) * u.m
    unit_model = _inputs(phase_origin=phase_origin)
    nonunit_model = _inputs(
        direction=(0.0, 0.0, 1e308),
        polarization=(0.0, 0.0, 1e307),
        phase_origin=phase_origin,
    )

    np.testing.assert_array_equal(
        nonunit_model.spec.propagation_direction,
        unit_model.spec.propagation_direction,
    )
    np.testing.assert_array_equal(
        nonunit_model.spec.polarization, unit_model.spec.polarization
    )
    unit_result = unit_model.at(_one_target(unit_model.geometry.frame)).run()
    nonunit_result = nonunit_model.at(_one_target(nonunit_model.geometry.frame)).run()
    for name in ("potential", "acceleration", "gradient"):
        np.testing.assert_array_equal(
            np.asarray(getattr(nonunit_result, name)),
            np.asarray(getattr(unit_result, name)),
        )


@pytest.mark.parametrize("field", ("direction", "polarization"))
@pytest.mark.parametrize("boolean", (True, np.bool_(True)))
def test_direction_vectors_reject_mixed_boolean_components(field, boolean):
    with pytest.raises(TypeError, match="must not contain bool"):
        _inputs(**{field: [boolean, 0, 0]})


def test_setup_and_result_expose_parameters_formulas_and_references():
    model = _inputs()
    setup = model.at(_one_target(model.geometry.frame))
    result = setup.run()

    for evaluation_view in (setup, result):
        parameters = evaluation_view.parameters
        assert parameters["density"][0].to_value(u.kg / u.m**3) == 2500
        assert parameters["inner_radius"][0].to_value(u.m) == 10
        assert parameters["outer_radius"][0].to_value(u.m) == 40
        assert parameters["direction_z"][0] == 1.0
        assert parameters["polarization_z"][0] == 1.0
        assert parameters["target_count"][0] == 1
        assert set(evaluation_view.formulas["observable"]) == {
            "potential",
            "acceleration",
            "gradient",
        }
        assert any(
            "spherical_shell_observables_g_free" in source
            for source in evaluation_view.references["source"]
        )


def _fake_native_parts(frequency_count=2):
    from gragra.analytic.values import GFreeGravityValues

    values = {}
    for name, value in zip(
        ("bulk", "inner_wall", "outer_boundary", "total"),
        (2.0, 20.0, 200.0, 999.0),
        strict=True,
    ):
        values[name] = GFreeGravityValues(
            np.full((frequency_count, 1), value + 1j * value),
            np.full((frequency_count, 1, 3), value + 1j * value),
            np.full((frequency_count, 1, 3, 3), value + 1j * value),
        )
    return values


@pytest.mark.parametrize(
    ("selector", "definition"),
    [
        ("bulk", Contribution("selected", ("bulk",))),
        ("inner_wall", Contribution("selected", ("inner_wall",))),
        ("outer_boundary", Contribution("selected", ("outer_boundary",))),
        ("total", Contribution("total", ("bulk", "inner_wall", "outer_boundary"))),
    ],
)
def test_each_selector_uses_its_native_payload_and_records_contract(
    selector, definition, monkeypatch
):
    from gragra.analytic import u10_spherical_shell_general as native_shell
    from gragra.constants import G_SI

    calls = []

    def fake_native(targets_m, **kwargs):
        calls.append((np.array(targets_m, copy=True), kwargs))
        return _fake_native_parts()

    monkeypatch.setattr(native_shell, "spherical_shell_observables_g_free", fake_native)
    model = _inputs(contribution=selector)
    result = model.at(_one_target(model.geometry.frame)).run()

    assert len(calls) == 1
    expected_native = _fake_native_parts()[selector]
    np.testing.assert_array_equal(
        np.asarray(result.values.potential)[0, 0, :],
        expected_native.potential[:, 0] * G_SI,
    )
    np.testing.assert_array_equal(
        np.asarray(result.values.acceleration)[0, :, :],
        expected_native.acceleration[:, 0, :].T * G_SI,
    )
    np.testing.assert_array_equal(
        np.asarray(result.values.gradient)[0, :, :],
        expected_native.gradient[:, 0, :, :].transpose(1, 2, 0).reshape(9, 2) * G_SI,
    )
    assert result.field_definition.contribution == definition
    assert result.field_definition.kind is FieldKind.FIELD_CHANGE
    assert (
        result.field_definition.potential_reference
        is PotentialReference.ZERO_AT_INFINITY
    )
    assert result.record.g_applications == 1
    assert result.record.gravitational_constant_si == G_SI
    assert result.context.g_convention == "SI_G_applied_once"
    assert result.values.potential.meta[0, 0].unit == u.m**2 / u.s**2
    assert result.values.acceleration.meta[0, 0].unit == u.m / u.s**2
    assert result.values.gradient.meta[0, 0].unit == u.s**-2
    if selector == "total":
        # The fake total differs from the sum of its components by design.
        assert np.asarray(result.values.potential)[0, 0, 0] == (999 + 999j) * G_SI


def test_phase_speed_series_matches_exact_normalized_displacement_axis():
    displacement = FrequencySeries(
        [1e-5, 2e-5], frequencies=[1.0, 2.0] * u.Hz, unit=u.m
    )
    speed = FrequencySeries(
        [2.0, 4.0], frequencies=[1000.0, 2000.0] * u.mHz, unit=u.km / u.s
    )
    model = _inputs(displacement=displacement, phase_speed=speed)

    np.testing.assert_allclose(model.spec.wavenumbers_rad_m, np.full(2, np.pi / 1000.0))


@pytest.mark.parametrize(
    "speed_frequencies",
    ([1000.0, 2000.1] * u.mHz, [1000.0] * u.mHz),
)
def test_phase_speed_series_rejects_mismatched_axis_without_resampling(
    speed_frequencies,
):
    displacement = FrequencySeries(
        [1e-5, 2e-5], frequencies=[1.0, 2.0] * u.Hz, unit=u.m
    )
    speed = FrequencySeries(
        [2.0, 4.0][: len(speed_frequencies)],
        frequencies=speed_frequencies,
        unit=u.km / u.s,
    )

    with pytest.raises(ValueError, match="exactly match"):
        _inputs(displacement=displacement, phase_speed=speed)


def test_phase_origin_is_a_position_and_vectors_follow_geometry_basis(monkeypatch):
    from gragra.analytic import u10_spherical_shell_general as native_shell
    from gragra.constants import G_SI

    root = Frame("root")
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    geometry_frame = Frame(
        "geometry", origin=[10, 20, -30] * u.m, basis=rotation, parent=root
    )
    geometry = SphericalShell([1, 2, 3] * u.m, 10 * u.m, 40 * u.m, geometry_frame)
    model = _inputs(geometry=geometry)
    local_target = np.array([[7.0, -4.0, 55.0]])
    root_target = geometry_frame.transform_points_m(local_target, root)
    captured = {}

    def fake_native(targets_m, **kwargs):
        captured["targets_m"] = np.array(targets_m, copy=True)
        captured.update(kwargs)
        return {
            "total": _fake_native_parts()["total"],
            "bulk": _fake_native_parts()["bulk"],
            "inner_wall": _fake_native_parts()["inner_wall"],
            "outer_boundary": _fake_native_parts()["outer_boundary"],
        }

    monkeypatch.setattr(native_shell, "spherical_shell_observables_g_free", fake_native)
    result = model.at(_one_target(root, root_target[0]), output_frame=root).run()

    np.testing.assert_allclose(captured["targets_m"], local_target)
    np.testing.assert_array_equal(captured["center_m"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(captured["phase_origin_m"], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(captured["propagation_direction"], [0, 0, 1])
    np.testing.assert_array_equal(captured["polarization"], [0, 0, 1])
    np.testing.assert_allclose(
        np.asarray(result.values.acceleration)[0, :, 0],
        geometry_frame.rotation_to(root) @ np.array([999 + 999j] * 3) * G_SI,
    )


def test_reference_time_is_required_even_when_displacement_has_an_epoch():
    from gragra.analytic.seismic._spherical_shell_model import spherical_shell

    frame = Frame("shell")
    geometry = SphericalShell([0, 0, 0] * u.m, 10 * u.m, 40 * u.m, frame)
    displacement = FrequencySeries(
        [1.0 + 2.0j],
        frequencies=[1.25] * u.Hz,
        unit=u.cm,
        epoch=LIGOTimeGPS(10_000, 250_000_000),
    )

    with pytest.raises(TypeError, match="reference_time"):
        spherical_shell(
            geometry=geometry,
            density=2500 * u.kg / u.m**3,
            phase_speed=2.0 * u.km / u.s,
            displacement=displacement,
            representation=PhasorRepresentation(),
            phase_origin=[0, 0, 0] * u.m,
            propagation_direction=[0, 0, 1],
            polarization=[0, 0, 1],
            wave_type="P",
        )


def test_explicit_reference_time_and_epoch_provenance_do_not_rephase_displacement(
    monkeypatch,
):
    from gragra.analytic import u10_spherical_shell_general as native_shell

    epoch = LIGOTimeGPS(10_000, 250_000_000)
    displacement = FrequencySeries(
        [1.0 + 2.0j], frequencies=[1.25] * u.Hz, unit=u.cm, epoch=epoch
    )
    model = _inputs(displacement=displacement)
    captured = {}

    def fake_native(_targets_m, **kwargs):
        captured.update(kwargs)
        return _fake_native_parts(frequency_count=1)

    monkeypatch.setattr(native_shell, "spherical_shell_observables_g_free", fake_native)
    setup = model.at(_one_target(model.geometry.frame))
    result = setup.run()

    np.testing.assert_array_equal(model.spec.displacement_m, [0.01 + 0.02j])
    np.testing.assert_array_equal(captured["displacement_m"], [0.01 + 0.02j])
    assert model.spec.displacement_epoch_gps == (10_000, 250_000_000)
    assert model.spec.reference_time_gps == (1_000_000_000, 12)
    assert setup.context.reference_time_gps == (1_000_000_000, 12)
    assert result.context.reference_time_gps == (1_000_000_000, 12)


@pytest.mark.parametrize("radius", (10.0, 40.0))
def test_targets_on_either_shell_boundary_are_rejected(radius):
    model = _inputs(
        displacement=FrequencySeries([1e-6], frequencies=[1.0] * u.Hz, unit=u.m)
    )

    with pytest.raises(ValueError, match="shell boundary"):
        model.at(_one_target(model.geometry.frame, (radius, 0.0, 0.0))).run()


def test_s_wave_wall_acceleration_matches_independent_surface_quadrature():
    from gragra.analytic.seismic._spherical_shell_model import spherical_shell
    from gragra.constants import G_SI

    frequency_hz = 1.0
    wavenumber = 0.02
    rho = 2000.0
    displacement_m = 1e-5
    inner_radius, outer_radius = 10.0, 40.0
    model = spherical_shell(
        SphericalShell(
            [0, 0, 0] * u.m,
            inner_radius * u.m,
            outer_radius * u.m,
            Frame("shell"),
        ),
        rho * u.kg / u.m**3,
        (2.0 * np.pi * frequency_hz / wavenumber) * u.m / u.s,
        FrequencySeries([displacement_m], frequencies=[frequency_hz] * u.Hz, unit=u.m),
        PhasorRepresentation(),
        (1_000_000_000, 12),
        [0, 0, 0] * u.m,
        [0, 0, 1],
        [1, 0, 0],
        "S",
        contribution="total",
    )
    result = model.at(_one_target(model.geometry.frame)).run()

    # Direct Eq. (22)-style surface quadrature: analytically integrate phi and
    # use Gauss-Legendre quadrature only for theta. This is independent of the
    # production series and shell decomposition implementation.
    nodes, weights = np.polynomial.legendre.leggauss(256)
    theta = 0.5 * np.pi * (nodes + 1.0)
    theta_weights = 0.5 * np.pi * weights
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    inner_integral = -np.sum(
        theta_weights * sin_theta**3 * np.cos(wavenumber * inner_radius * cos_theta)
    )
    outer_integral = np.sum(
        theta_weights * sin_theta**3 * np.cos(wavenumber * outer_radius * cos_theta)
    )
    expected = G_SI * rho * displacement_m * np.pi * (inner_integral + outer_integral)
    np.testing.assert_allclose(
        np.asarray(result.values.acceleration)[0, 0, 0], expected, rtol=2e-10
    )


def test_later_frequency_convergence_failure_returns_no_partial_evaluation():
    from gragra.analytic.u10_spherical_shell_general import ShellConvergenceError

    frequencies = np.array([1.0, 3.0]) * u.Hz
    target_wavenumbers = np.array([0.04, 0.08])
    phase_speeds = FrequencySeries(
        2.0 * np.pi * frequencies.to_value(u.Hz) / target_wavenumbers,
        frequencies=frequencies,
        unit=u.m / u.s,
    )
    displacement = FrequencySeries(
        [1e-6 * (1.0 + 0.5j)] * 2, frequencies=frequencies, unit=u.m
    )
    model = _inputs(
        displacement=displacement,
        phase_speed=phase_speeds,
        geometry=SphericalShell(
            [0, 0, 0] * u.m, 10 * u.m, 400 * u.m, Frame("wide-shell")
        ),
    )

    result = None
    with pytest.raises(ShellConvergenceError) as caught:
        result = model.at(_one_target(model.geometry.frame)).run()

    assert result is None
    assert caught.value.diagnostics["status"] == "NOT_CONVERGED"
    assert caught.value.diagnostics["failed_index"] == [1, 0]
    assert [query["status"] for query in caught.value.diagnostics["queries"]] == [
        "CONVERGED",
        "NOT_CONVERGED",
    ]
