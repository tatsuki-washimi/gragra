"""Coefficient generation tests for the amplitude-free Rayleigh response."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
from astropy import units as u

from gragra._rayleigh_contract import RayleighSurfacePrecisionError
from gragra.constants import G_SI
from gragra.geometry._frame import Frame
from gragra.geometry._halfspace import HalfSpace
from gragra.targets._high_points import TargetPoints


def _rotation() -> np.ndarray:
    angle_z = 0.37
    angle_x = -0.29
    rz = np.array(
        [
            [np.cos(angle_z), -np.sin(angle_z), 0.0],
            [np.sin(angle_z), np.cos(angle_z), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle_x), -np.sin(angle_x)],
            [0.0, np.sin(angle_x), np.cos(angle_x)],
        ]
    )
    return rz @ rx


def _fixture():
    from gragra.analytic.seismic._rayleigh_response import rayleigh_response

    root = Frame("root")
    surface = Frame(
        "surface",
        origin=np.array([7.0, -4.0, 2.5]) * u.m,
        basis=_rotation(),
        parent=root,
    )
    local_targets = np.array([[3.2, -1.4, 2.1], [-2.6, 4.3, -1.7]])
    targets = TargetPoints(
        surface.transform_points_m(local_targets, root) * u.m,
        ("above", "below"),
        root,
    )
    frequencies = np.array([2.5, 3.25, 5.5]) * u.Hz
    model = rayleigh_response(
        HalfSpace(surface),
        density=2300 * u.kg / u.m**3,
        vp=np.sqrt(3.0) * 1500 * u.m / u.s,
        vs=1500 * u.m / u.s,
        frequencies=frequencies,
        direction=(np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0),
        phase_origin=np.array([1.1, -0.8, 0.0]) * u.m,
    )
    setup = model.at(targets, output_frame=root)
    return model, setup, surface, root


def _direct_values(setup):
    from gragra.observables.rayleigh_literature import (
        rayleigh_halfspace_allspace_observables,
    )

    spec = setup.resolved.model_spec
    return rayleigh_halfspace_allspace_observables(
        setup.resolved.target_positions_m,
        density_kg_m3=spec.density_kg_m3,
        v_p_m_s=spec.vp_m_s,
        v_s_m_s=spec.vs_m_s,
        frequency_hz=np.asarray(spec.frequencies_hz),
        vertical_surface_amplitude_m=np.ones(
            len(spec.frequencies_hz), dtype=np.complex128
        ),
        direction=spec.direction,
        phase_origin_m=spec.phase_origin_m,
    )


def _coefficients_from_values(
    values, *, basis_amplitude_m: float, output_rotation: np.ndarray
):
    potential = np.asarray(values.potential, dtype=np.complex128)
    acceleration = np.asarray(values.acceleration, dtype=np.complex128)
    gradient = np.asarray(values.gradient, dtype=np.complex128)
    acceleration = np.einsum("ij,...j->...i", output_rotation, acceleration)
    gradient = np.einsum(
        "ij,...jk,lk->...il", output_rotation, gradient, output_rotation
    )
    target_count = potential.shape[1]
    frequency_count = potential.shape[0]
    return {
        "potential": potential.transpose(1, 0)[:, None, None, :] / basis_amplitude_m,
        "acceleration": acceleration.transpose(1, 2, 0)[:, :, None, :]
        / basis_amplitude_m,
        "gradient": gradient.reshape(frequency_count, target_count, 9).transpose(
            1, 2, 0
        )[:, :, None, :]
        / basis_amplitude_m,
    }


def _scales(setup):
    spec = setup.resolved.model_spec
    frequencies = np.asarray(spec.frequencies_hz)
    wavenumbers = 2 * np.pi * frequencies / spec.rayleigh_speed_m_s
    common = G_SI * spec.density_kg_m3
    return {
        "potential": common / wavenumbers,
        "acceleration": np.full_like(wavenumbers, common),
        "gradient": common * wavenumbers,
    }


def _assert_componentwise_close(actual, expected, scale):
    scale = np.asarray(scale).reshape((1,) * (actual.ndim - 1) + (-1,))
    bound = 1e-12 * scale + 1e-12 * np.abs(expected)
    assert np.all(np.abs(actual - expected) <= bound)


def test_run_builds_one_meter_coefficients_with_one_kernel_call(monkeypatch):
    from gragra.observables import rayleigh_literature

    model, setup, surface, root = _fixture()
    direct = _direct_values(setup)
    original = rayleigh_literature.rayleigh_halfspace_allspace_observables
    spy = Mock(wraps=original)
    monkeypatch.setattr(
        rayleigh_literature, "rayleigh_halfspace_allspace_observables", spy
    )

    response = setup.run()

    assert spy.call_count == 1
    call = spy.call_args
    assert len(call.args) == 1
    np.testing.assert_array_equal(call.args[0], setup.resolved.target_positions_m)
    kwargs = call.kwargs
    spec = model.spec
    assert kwargs["density_kg_m3"] == spec.density_kg_m3
    assert kwargs["v_p_m_s"] == spec.vp_m_s
    assert kwargs["v_s_m_s"] == spec.vs_m_s
    np.testing.assert_array_equal(kwargs["frequency_hz"], spec.frequencies_hz)
    np.testing.assert_array_equal(
        kwargs["vertical_surface_amplitude_m"],
        np.ones(len(spec.frequencies_hz), dtype=np.complex128),
    )
    np.testing.assert_array_equal(kwargs["direction"], spec.direction)
    np.testing.assert_array_equal(kwargs["phase_origin_m"], spec.phase_origin_m)

    expected = _coefficients_from_values(
        direct,
        basis_amplitude_m=float(response.contract.basis_amplitude.to_value(u.m)),
        output_rotation=setup.resolved.computation_frame.rotation_to(
            setup.resolved.output_frame
        ),
    )
    scales = _scales(setup)
    for observable in ("potential", "acceleration", "gradient"):
        actual = response._coefficient_array(observable)
        assert actual.dtype == np.dtype(np.complex128)
        assert actual.shape == expected[observable].shape
        _assert_componentwise_close(actual, expected[observable], scales[observable])

    displayed = setup.parameters
    assert displayed["density"][0].to_value(u.kg / u.m**3) == kwargs["density_kg_m3"]
    assert displayed["vp"][0].to_value(u.m / u.s) == kwargs["v_p_m_s"]
    assert displayed["vs"][0].to_value(u.m / u.s) == kwargs["v_s_m_s"]
    assert displayed["frequency_min"][0].to_value(u.Hz) == np.min(
        kwargs["frequency_hz"]
    )
    assert displayed["frequency_max"][0].to_value(u.Hz) == np.max(
        kwargs["frequency_hz"]
    )
    np.testing.assert_array_equal(
        [displayed[f"direction_{axis}"][0] for axis in "xyz"], kwargs["direction"]
    )
    np.testing.assert_array_equal(
        [
            displayed["phase_origin_x"][0].to_value(u.m),
            displayed["phase_origin_y"][0].to_value(u.m),
            0.0,
        ],
        kwargs["phase_origin_m"],
    )

    assert response.record.g_applications == 1
    assert response.contract.input_location_m.unit == u.m
    assert response.evaluation is setup.resolved

    rotation = surface.rotation_to(root)
    direct_gradient = np.asarray(direct.gradient, dtype=np.complex128)
    for frequency_index in range(len(spec.frequencies_hz)):
        for target_index in range(len(setup.resolved.targets)):
            rotated = (
                rotation @ direct_gradient[frequency_index, target_index] @ rotation.T
            )
            actual = response._coefficient_array("gradient")[
                target_index, :, 0, frequency_index
            ].reshape(3, 3)
            _assert_componentwise_close(
                actual.reshape(1, 9, 1, 1),
                (rotated / response.contract.basis_amplitude.to_value(u.m)).reshape(
                    1, 9, 1, 1
                ),
                np.asarray([scales["gradient"][frequency_index]]),
            )
            conjugated = (
                rotation
                @ direct_gradient[frequency_index, target_index].conj()
                @ rotation.T
            ) / response.contract.basis_amplitude.to_value(u.m)
            assert np.max(np.abs(actual.reshape(3, 3) - conjugated)) > (
                1e-12 * scales["gradient"][frequency_index]
            )

    gamma = response._coefficient_array("gradient")[:, :, 0, :].reshape(
        len(setup.resolved.targets), 3, 3, len(spec.frequencies_hz)
    )
    symmetry_error = np.abs(gamma - gamma.swapaxes(1, 2))
    symmetry_bound = 1e-12 * scales["gradient"][None, None, None, :]
    assert np.all(symmetry_error <= symmetry_bound)

    trace = np.trace(gamma, axis1=1, axis2=2)
    direct_gamma = np.asarray(direct.gradient, dtype=np.complex128)
    direct_trace = np.trace(direct_gamma, axis1=2, axis2=3).T
    trace_bound = 1e-12 * scales["gradient"][None, :]
    assert np.all(np.abs(trace - direct_trace) <= trace_bound)
    assert np.all(np.abs(trace[0]) <= 1e-12 * scales["gradient"])


def test_setup_displays_response_inputs_without_fake_amplitude_or_time():
    model, setup, _, _ = _fixture()

    report = setup.validate()

    assert dict(report.target_regions) == {"above": "above", "below": "below"}
    assert setup.resolved.model is model
    assert "frequency_min" in setup.parameters.colnames
    assert "law_name" in setup.parameters.colnames
    assert not {"displacement_max", "reference_gps_seconds"} & set(
        setup.parameters.colnames
    )
    assert not hasattr(model.spec, "vertical_displacement_m")
    assert not hasattr(model.spec, "reference_time_gps")
    assert set(setup.formulas["observable"]) == {
        "potential",
        "acceleration",
        "gradient",
    }
    assert len(setup.references) == 1


@pytest.mark.parametrize("case", ("surface", "ambiguous"))
def test_surface_rejections_keep_diagnostic_metadata_while_plotting_is_allowed(
    case,
):
    import matplotlib.pyplot as plt

    from gragra.analytic.seismic._rayleigh_response import rayleigh_response

    root = Frame("root")
    if case == "surface":
        surface = root
        position = np.array([[0.0, 0.0, 0.0]])
    else:
        basis = np.eye(3)
        basis[2, :2] = 1e-13
        surface = Frame("surface", basis=basis, parent=root)
        position = np.array([[3.0, -3.0, np.ldexp(1.0, -95)]])
    model = rayleigh_response(
        HalfSpace(surface),
        density=2300 * u.kg / u.m**3,
        vp=np.sqrt(3.0) * 1500 * u.m / u.s,
        vs=1500 * u.m / u.s,
        frequencies=np.array([2.5]) * u.Hz,
        direction=(1.0, 0.0, 0.0),
        phase_origin=np.zeros(3) * u.m,
    )
    targets = TargetPoints(position * u.m, (case,), root)
    setup = model.at(targets, output_frame=root)

    figure = setup.plot_setup(view="xz")
    try:
        assert figure.axes
    finally:
        plt.close(figure)

    errors = []
    for operation in (setup.validate, setup.run):
        with pytest.raises(RayleighSurfacePrecisionError) as caught:
            operation()
        error = caught.value
        assert error.target_id == case
        if case == "surface":
            assert error.reason == "surface"
        else:
            assert error.reason in ("coordinate_precision", "surface")
        assert error.unit == "m"
        assert error.observables == ("potential", "acceleration", "gradient")
        if case == "surface":
            assert error.signed_distance_m == 0.0
        else:
            assert abs(error.signed_distance_m) <= error.threshold_m
            assert error.edge_count == 1
        errors.append(
            (
                error.target_id,
                error.signed_distance_m,
                error.threshold_m,
                error.edge_count,
                error.scale_m,
                error.reason,
                error.unit,
                error.observables,
                str(error),
            )
        )
    assert errors[0] == errors[1]
