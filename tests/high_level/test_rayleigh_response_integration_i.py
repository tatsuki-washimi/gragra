"""Public integration checks for reusable Rayleigh gravity responses."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeries

from gragra.constants import G_SI
from gragra.geometry import Frame, HalfSpace
from gragra.results import PhasorRepresentation, compare
from gragra.targets import TargetPoints
from gragra.waves import rayleigh_speed

_FREQUENCIES_HZ = np.array([0.75, 1.375, 5.25])
_AMPLITUDES_M = np.array([0.7 - 0.2j, -0.35 + 0.9j, 0.45 + 0.65j]) * 1e-9
_DIRECTION = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
_PHASE_ORIGIN_M = np.array([1.1, -0.8, 0.0])
_REFERENCE_TIME = (1_600_000_000, 250_000_000)
_DENSITY_KG_M3 = 2300.0
_VP_M_S = np.sqrt(3.0) * 1500.0
_VS_M_S = 1500.0


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


def _frames_and_targets(local_positions_m: object, target_ids: tuple[str, ...]):
    root = Frame("root")
    surface = Frame(
        "surface",
        origin=np.array([7.0, -4.0, 2.5]) * u.m,
        basis=_rotation(),
        parent=root,
    )
    local_positions = np.asarray(local_positions_m, dtype=np.float64)
    targets = TargetPoints(
        surface.transform_points_m(local_positions, root) * u.m,
        target_ids,
        root,
    )
    return surface, root, targets


def _default_targets():
    return _frames_and_targets(
        [
            [3.2, -1.4, 2.1],
            [-2.6, 4.3, 1.7],
            [1.8, 2.9, -1.7],
            [-3.7, -1.1, -2.4],
        ],
        ("above-east", "above-west", "below-east", "below-west"),
    )


def _response_setup(
    *,
    surface: Frame,
    targets: TargetPoints,
    root: Frame,
    frequencies_hz: object = _FREQUENCIES_HZ,
    density_kg_m3: float = _DENSITY_KG_M3,
    vp_m_s: float = _VP_M_S,
    vs_m_s: float = _VS_M_S,
    direction: object = _DIRECTION,
    phase_origin_m: object = _PHASE_ORIGIN_M,
):
    from gragra.analytic.seismic import rayleigh_response

    return rayleigh_response(
        HalfSpace(surface),
        density=density_kg_m3 * u.kg / u.m**3,
        vp=vp_m_s * u.m / u.s,
        vs=vs_m_s * u.m / u.s,
        frequencies=np.asarray(frequencies_hz) * u.Hz,
        direction=direction,
        phase_origin=np.asarray(phase_origin_m) * u.m,
    ).at(targets, output_frame=root)


def _direct_result(
    *,
    surface: Frame,
    targets: TargetPoints,
    output_frame: Frame,
    amplitudes_m: object,
    frequencies_hz: object = _FREQUENCIES_HZ,
    density_kg_m3: float = _DENSITY_KG_M3,
    vp_m_s: float = _VP_M_S,
    vs_m_s: float = _VS_M_S,
    direction: object = _DIRECTION,
    phase_origin_m: object = _PHASE_ORIGIN_M,
    reference_time: object = _REFERENCE_TIME,
):
    from gragra.analytic.seismic import rayleigh_halfspace

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    amplitude = np.asarray(amplitudes_m, dtype=np.complex128)
    model = rayleigh_halfspace(
        geometry=HalfSpace(surface),
        density=density_kg_m3 * u.kg / u.m**3,
        vp=vp_m_s * u.m / u.s,
        vs=vs_m_s * u.m / u.s,
        vertical_displacement=FrequencySeries(
            amplitude, frequencies=frequencies * u.Hz, unit=u.m
        ),
        representation=PhasorRepresentation(),
        reference_time=reference_time,
        phase_origin=np.asarray(phase_origin_m) * u.m,
        direction=direction,
    )
    return model.at(targets, output_frame=output_frame).run()


def _scales(
    amplitudes_m: object,
    frequencies_hz: object,
    density_kg_m3: float,
    vp_m_s: float = _VP_M_S,
    vs_m_s: float = _VS_M_S,
):
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    amplitude = np.asarray(amplitudes_m, dtype=np.complex128)
    wavenumber = 2 * np.pi * frequencies / rayleigh_speed(vp_m_s, vs_m_s)
    common = G_SI * density_kg_m3 * np.abs(amplitude)
    return {
        "potential": common / wavenumber,
        "acceleration": common,
        "gradient": common * wavenumber,
    }


def _coefficient_scales(
    frequencies_hz: object,
    density_kg_m3: float,
    vp_m_s: float = _VP_M_S,
    vs_m_s: float = _VS_M_S,
):
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    wavenumber = 2 * np.pi * frequencies / rayleigh_speed(vp_m_s, vs_m_s)
    common = G_SI * density_kg_m3
    return {
        "potential": common / wavenumber,
        "acceleration": np.full_like(wavenumber, common),
        "gradient": common * wavenumber,
    }


def _assert_componentwise_close(actual, expected, scale):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    shape = (1,) * (actual.ndim - 1) + (len(scale),)
    bound = 1e-12 * np.asarray(scale).reshape(shape) + 1e-12 * np.abs(expected)
    assert np.all(np.abs(actual - expected) <= bound)


def _assert_parameters_match(response_parameters, actual_parameters):
    for name in response_parameters.colnames:
        if name == "law_name":
            continue
        assert name in actual_parameters.colnames
        response_column = response_parameters[name]
        actual_column = actual_parameters[name]
        response_unit = getattr(response_column, "unit", None)
        actual_unit = getattr(actual_column, "unit", None)
        if response_unit is not None:
            assert actual_unit is not None
            np.testing.assert_array_equal(
                response_column.to_value(response_unit),
                actual_column.to_value(response_unit),
            )
        else:
            np.testing.assert_array_equal(
                np.asarray(response_column), np.asarray(actual_column)
            )


def _assert_parameter_tables_equal(left, right):
    assert left.colnames == right.colnames
    for name in left.colnames:
        left_column = left[name]
        right_column = right[name]
        left_unit = getattr(left_column, "unit", None)
        right_unit = getattr(right_column, "unit", None)
        assert left_unit == right_unit
        if left_unit is not None:
            np.testing.assert_array_equal(
                left_column.to_value(left_unit), right_column.to_value(left_unit)
            )
        else:
            np.testing.assert_array_equal(
                np.asarray(left_column), np.asarray(right_column)
            )


def _assert_formula_and_reference_tables_match(left, right):
    if isinstance(left, pd.DataFrame):
        assert isinstance(right, pd.DataFrame)
        assert left.equals(right)
    else:
        assert left == right


def _field_arrays(result):
    return {
        name: np.asarray(getattr(result, name))
        for name in ("potential", "acceleration", "gradient")
    }


def test_applied_result_evaluation_does_not_run_direct_rayleigh():
    surface, root, targets = _default_targets()
    response = _response_setup(surface=surface, targets=targets, root=root).run()
    applied = response.apply(
        vertical_displacement=FrequencySeries(
            _AMPLITUDES_M, frequencies=_FREQUENCIES_HZ * u.Hz, unit=u.m
        ),
        representation=PhasorRepresentation(),
        reference_time=_REFERENCE_TIME,
    )

    assert applied.record.backend == "gravity_response:rayleigh_literature"
    with pytest.raises(RuntimeError, match=r"response\.apply\(\)"):
        applied.evaluation.run()


def test_direct_rayleigh_result_operations_do_not_run_kernel(monkeypatch):
    from gragra.results import EvaluationResult

    surface, root, targets = _default_targets()
    result = _direct_result(
        surface=surface,
        targets=targets,
        output_frame=root,
        amplitudes_m=_AMPLITUDES_M,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("direct Rayleigh kernel was called")

    monkeypatch.setattr(
        "gragra.observables.rayleigh_literature.rayleigh_halfspace_allspace_observables",
        forbidden,
    )
    with pytest.raises(AssertionError, match="kernel was called"):
        result.evaluation.run()

    rebuilt = EvaluationResult(
        result.values, evaluation=result.evaluation, record=result.record
    )
    assert rebuilt.parameters.colnames
    assert not rebuilt.formulas.empty
    assert not rebuilt.references.empty
    selected = rebuilt.select("potential")
    comparison = compare(reference=selected, candidate=result.select("potential"))
    np.testing.assert_array_equal(np.asarray(comparison.differences.potential), 0)


def test_public_response_apply_matches_direct_rayleigh_and_compare():
    surface, root, targets = _default_targets()
    setup = _response_setup(surface=surface, targets=targets, root=root)
    response = setup.run()
    applied = response.apply(
        vertical_displacement=FrequencySeries(
            _AMPLITUDES_M, frequencies=_FREQUENCIES_HZ * u.Hz, unit=u.m
        ),
        representation=PhasorRepresentation(),
        reference_time=_REFERENCE_TIME,
    )
    direct = _direct_result(
        surface=surface,
        targets=targets,
        output_frame=root,
        amplitudes_m=_AMPLITUDES_M,
    )

    assert applied.context.targets is targets
    assert direct.context.targets is targets
    assert applied.context.frame is root
    assert direct.context.frame is root
    assert applied.context.reference_time_gps == _REFERENCE_TIME
    np.testing.assert_array_equal(
        applied.context.model_spec.vertical_displacement_m, _AMPLITUDES_M
    )
    assert response.contract.law_name == "newton"
    assert response.parameters["law_name"][0] == response.contract.law_name
    assert direct.context.interaction.name == response.contract.law_name
    assert applied.context.interaction.name == response.contract.law_name
    assert not response.contract.law_parameters
    assert direct.context.interaction.parameters == response.contract.law_parameters
    assert applied.context.interaction.parameters == response.contract.law_parameters
    assert response.output_definition is applied.context.field_definition
    assert direct.field_definition == response.output_definition
    _assert_parameters_match(response.parameters, applied.parameters)
    _assert_parameters_match(response.parameters, direct.parameters)
    _assert_parameter_tables_equal(applied.parameters, direct.parameters)
    _assert_formula_and_reference_tables_match(response.formulas, applied.formulas)
    _assert_formula_and_reference_tables_match(response.references, applied.references)
    _assert_formula_and_reference_tables_match(response.formulas, direct.formulas)
    _assert_formula_and_reference_tables_match(response.references, direct.references)

    direct_arrays = _field_arrays(direct)
    applied_arrays = _field_arrays(applied)
    scales = _scales(_AMPLITUDES_M, _FREQUENCIES_HZ, _DENSITY_KG_M3)
    for observable in scales:
        _assert_componentwise_close(
            applied_arrays[observable], direct_arrays[observable], scales[observable]
        )

    difference = compare(reference=direct, candidate=applied)
    for observable in scales:
        _assert_componentwise_close(
            np.asarray(getattr(difference.differences, observable)),
            np.zeros_like(direct_arrays[observable]),
            scales[observable],
        )


def test_public_response_same_fixture_as_independent_rayleigh_oracle(
    rayleigh_green_oracle_case,
):
    case = rayleigh_green_oracle_case
    root = Frame("oracle-world")
    surface_origin = np.asarray(case["surface_origin_world_m"])
    surface_basis = np.asarray(case["surface_basis_world"])
    local_targets = np.asarray(case["local_targets_m"])
    surface = Frame(
        "oracle-surface",
        surface_origin * u.m,
        surface_basis,
        root,
    )
    targets = TargetPoints(
        surface.transform_points_m(local_targets, root) * u.m,
        case["target_ids"],
        root,
    )
    frequencies = np.array([case["frequency_hz"]])
    amplitudes = np.array([case["amplitude_m"]])
    phase_origin = case["phase_origin_local_m"]
    direction = case["direction_local"]
    density = case["density_kg_m3"]
    vp_m_s = case["vp_m_s"]
    vs_m_s = case["vs_m_s"]
    reference_time = case["reference_time_gps"]
    setup = _response_setup(
        surface=surface,
        targets=targets,
        root=root,
        frequencies_hz=frequencies,
        density_kg_m3=density,
        vp_m_s=vp_m_s,
        vs_m_s=vs_m_s,
        direction=direction,
        phase_origin_m=phase_origin,
    )
    response = setup.run()
    applied = response.apply(
        vertical_displacement=FrequencySeries(
            amplitudes, frequencies=frequencies * u.Hz, unit=u.m
        ),
        representation=PhasorRepresentation(),
        reference_time=reference_time,
    )
    direct = _direct_result(
        surface=surface,
        targets=targets,
        output_frame=root,
        amplitudes_m=amplitudes,
        frequencies_hz=frequencies,
        density_kg_m3=density,
        vp_m_s=vp_m_s,
        vs_m_s=vs_m_s,
        direction=direction,
        phase_origin_m=phase_origin,
        reference_time=reference_time,
    )

    scales = _scales(amplitudes, frequencies, density, vp_m_s, vs_m_s)
    for observable, scale in scales.items():
        _assert_componentwise_close(
            np.asarray(getattr(applied, observable)),
            np.asarray(getattr(direct, observable)),
            scale,
        )


def test_public_coefficients_have_transfer_units_rotations_and_trace_contract():
    surface, root, targets = _default_targets()
    response = _response_setup(surface=surface, targets=targets, root=root).run()
    one_metre = np.ones(len(_FREQUENCIES_HZ), dtype=np.complex128)
    direct_root = _direct_result(
        surface=surface,
        targets=targets,
        output_frame=root,
        amplitudes_m=one_metre,
    )
    direct_surface = _direct_result(
        surface=surface,
        targets=targets,
        output_frame=surface,
        amplitudes_m=one_metre,
    )
    output_units = {
        "potential": u.m**2 / u.s**2,
        "acceleration": u.m / u.s**2,
        "gradient": u.s**-2,
    }
    coefficient_scales = _coefficient_scales(_FREQUENCIES_HZ, _DENSITY_KG_M3)
    coefficient_matrices = {}

    for observable, output_unit in output_units.items():
        coefficient = response.coefficients(observable)
        field = getattr(direct_root, observable)
        expected_unit = output_unit / u.m
        assert coefficient.meta[0, 0].unit == expected_unit
        assert field.meta[0, 0].unit == output_unit
        expected = (np.asarray(field) * field.meta[0, 0].unit / u.m).to_value(
            expected_unit
        )
        actual = np.asarray(coefficient)
        coefficient_matrices[observable] = actual
        _assert_componentwise_close(
            actual,
            expected,
            coefficient_scales[observable],
        )

    acceleration_root = coefficient_matrices["acceleration"]
    acceleration_surface = np.asarray(direct_surface.acceleration)
    rotation = surface.rotation_to(root)
    expected_acceleration = np.einsum("ij,mjf->mif", rotation, acceleration_surface)
    _assert_componentwise_close(
        acceleration_root,
        expected_acceleration,
        coefficient_scales["acceleration"],
    )

    gradient_root = coefficient_matrices["gradient"].reshape(
        len(targets), 3, 3, len(_FREQUENCIES_HZ)
    )
    gradient_surface = np.asarray(direct_surface.gradient).reshape(
        len(targets), 3, 3, len(_FREQUENCIES_HZ)
    )
    rotated_gradient = np.einsum(
        "ia,mabf,jb->mijf", rotation, gradient_surface, rotation
    )
    _assert_componentwise_close(
        gradient_root,
        rotated_gradient,
        coefficient_scales["gradient"],
    )
    conjugated_rotation = np.einsum(
        "ia,mabf,jb->mijf", rotation, gradient_surface.conj(), rotation
    )
    assert np.any(
        np.abs(gradient_root - conjugated_rotation)
        > 1e3 * 1e-12 * coefficient_scales["gradient"].reshape(1, 1, 1, -1)
    )

    trace = np.trace(gradient_root, axis1=1, axis2=2)
    trace_scale = coefficient_scales["gradient"]
    assert np.all(np.abs(trace[:2]) <= 1e-12 * trace_scale)
    expected_gradient = (
        np.asarray(direct_root.gradient).reshape(
            len(targets), 3, 3, len(_FREQUENCIES_HZ)
        )
        * direct_root.gradient.meta[0, 0].unit
        / u.m
    ).to_value((u.s**-2) / u.m)
    expected_below_trace = np.trace(expected_gradient[2:], axis1=1, axis2=2)
    assert np.all(np.abs(expected_below_trace) > 1e3 * 1e-12 * trace_scale)
    assert np.all(np.abs(trace[2:] - expected_below_trace) <= 1e-12 * trace_scale)
    symmetry = np.abs(gradient_root - gradient_root.swapaxes(1, 2))
    assert np.all(symmetry <= 1e-12 * trace_scale.reshape(1, 1, 1, -1))


def test_public_apply_is_complex_linear_and_exact_for_zero_bins_and_zero_density():
    surface, root, targets = _default_targets()
    response = _response_setup(surface=surface, targets=targets, root=root).run()
    x = np.array([0.45 - 0.2j, 0.0j, -0.3 + 0.75j]) * 1e-9
    y = np.array([-0.15 + 0.9j, 0.0j, 0.65 - 0.4j]) * 1e-9
    alpha = 0.8 - 1.1j
    beta = -0.35 + 0.6j
    combined = alpha * x + beta * y

    def apply(amplitudes):
        return response.apply(
            vertical_displacement=FrequencySeries(
                amplitudes, frequencies=_FREQUENCIES_HZ * u.Hz, unit=u.m
            ),
            representation=PhasorRepresentation(),
            reference_time=_REFERENCE_TIME,
        )

    result_x = apply(x)
    result_y = apply(y)
    result_combined = apply(combined)
    scales = _scales(
        np.abs(alpha) * np.abs(x) + np.abs(beta) * np.abs(y),
        _FREQUENCIES_HZ,
        _DENSITY_KG_M3,
    )
    for observable in scales:
        expected = alpha * np.asarray(
            getattr(result_x, observable)
        ) + beta * np.asarray(getattr(result_y, observable))
        actual = np.asarray(getattr(result_combined, observable))
        _assert_componentwise_close(actual, expected, scales[observable])
        assert np.all(actual[:, :, 1] == 0)
        assert np.isfinite(actual).all()

    before = {
        name: np.asarray(response.coefficients(name)).copy()
        for name in response.observables
    }
    zero = apply(np.zeros_like(x))
    for observable in response.observables:
        assert np.all(np.asarray(getattr(zero, observable)) == 0)
        np.testing.assert_array_equal(
            np.asarray(response.coefficients(observable)), before[observable]
        )

    zero_density = _response_setup(
        surface=surface,
        targets=targets,
        root=root,
        density_kg_m3=0.0,
    ).run()
    for observable in response.observables:
        assert np.all(np.asarray(zero_density.coefficients(observable)) == 0)
    zero_density_result = zero_density.apply(
        vertical_displacement=FrequencySeries(
            x, frequencies=_FREQUENCIES_HZ * u.Hz, unit=u.m
        ),
        representation=PhasorRepresentation(),
        reference_time=_REFERENCE_TIME,
    )
    for observable in response.observables:
        actual = np.asarray(getattr(zero_density_result, observable))
        assert np.all(actual == 0)
        assert np.isfinite(actual).all()


def test_public_response_reuse_rephases_without_rebuilding_coefficients(monkeypatch):
    from gragra.analytic import rayleigh_literature as analytic_rayleigh
    from gragra.observables import rayleigh_literature as si_rayleigh

    surface, root, targets = _default_targets()
    setup = _response_setup(surface=surface, targets=targets, root=root)
    original = si_rayleigh.rayleigh_halfspace_allspace_observables
    spy = Mock(wraps=original)
    monkeypatch.setattr(si_rayleigh, "rayleigh_halfspace_allspace_observables", spy)
    response = setup.run()
    assert spy.call_count == 1
    coefficient_bits = {
        name: np.asarray(response.coefficients(name)).copy()
        for name in response.observables
    }
    editable_coefficients = response.coefficients("acceleration")
    editable_coefficients[0, 0, 0] = 123 + 456j
    editable_coefficients.meta[0, 0].unit = u.kg
    displacement = np.array([0.7 - 0.2j, -0.35 + 0.9j, 0.45 + 0.65j]) * 1e-9
    dt = 0.03125
    rephase = np.exp(-2j * np.pi * _FREQUENCIES_HZ * dt)
    later_displacement = displacement * rephase
    assert not np.array_equal(later_displacement, displacement)

    def fail_if_recomputed(*_args, **_kwargs):
        raise AssertionError("apply must reuse the stored response coefficients")

    with monkeypatch.context() as patcher:
        patcher.setattr(
            si_rayleigh,
            "rayleigh_halfspace_allspace_observables",
            fail_if_recomputed,
        )
        patcher.setattr(
            si_rayleigh,
            "rayleigh_halfspace_allspace_observables_g_free",
            fail_if_recomputed,
        )
        patcher.setattr(
            analytic_rayleigh,
            "rayleigh_halfspace_allspace_observables_g_free",
            fail_if_recomputed,
        )
        first = response.apply(
            vertical_displacement=FrequencySeries(
                displacement, frequencies=_FREQUENCIES_HZ * u.Hz, unit=u.m
            ),
            representation=PhasorRepresentation(),
            reference_time=_REFERENCE_TIME,
        )
        later_gps = (_REFERENCE_TIME[0], _REFERENCE_TIME[1] + int(dt * 1e9))
        later = response.apply(
            vertical_displacement=FrequencySeries(
                later_displacement,
                frequencies=_FREQUENCIES_HZ * u.Hz,
                unit=u.m,
            ),
            representation=PhasorRepresentation(),
            reference_time=later_gps,
        )

    assert spy.call_count == 1
    assert first.context.reference_time_gps == _REFERENCE_TIME
    assert later.context.reference_time_gps == later_gps
    for observable in response.observables:
        np.testing.assert_array_equal(
            np.asarray(response.coefficients(observable)), coefficient_bits[observable]
        )
    assert response.coefficients("acceleration").meta[0, 0].unit == u.s**-2
    for observable in response.observables:
        coefficient = coefficient_bits[observable]
        expected_input = later_displacement
        expected = coefficient * expected_input[None, None, :]
        np.testing.assert_array_equal(np.asarray(getattr(later, observable)), expected)


def test_public_response_spatial_phase_origin_compensation():
    surface, root, targets = _default_targets()
    shift = np.array([0.8, 0.6, 0.0])
    initial = _response_setup(surface=surface, targets=targets, root=root).run()
    shifted = _response_setup(
        surface=surface,
        targets=targets,
        root=root,
        phase_origin_m=_PHASE_ORIGIN_M + shift,
    ).run()
    input_series = FrequencySeries(
        _AMPLITUDES_M, frequencies=_FREQUENCIES_HZ * u.Hz, unit=u.m
    )
    corrected_input = _AMPLITUDES_M * np.exp(
        1j
        * (2 * np.pi * _FREQUENCIES_HZ / rayleigh_speed(_VP_M_S, _VS_M_S))
        * np.dot(_DIRECTION, shift)
    )
    corrected = shifted.apply(
        vertical_displacement=FrequencySeries(
            corrected_input, frequencies=_FREQUENCIES_HZ * u.Hz, unit=u.m
        ),
        representation=PhasorRepresentation(),
        reference_time=_REFERENCE_TIME,
    )
    uncorrected = shifted.apply(
        vertical_displacement=input_series,
        representation=PhasorRepresentation(),
        reference_time=_REFERENCE_TIME,
    )
    reference = initial.apply(
        vertical_displacement=input_series,
        representation=PhasorRepresentation(),
        reference_time=_REFERENCE_TIME,
    )

    scales = _scales(_AMPLITUDES_M, _FREQUENCIES_HZ, _DENSITY_KG_M3)
    for observable, scale in scales.items():
        expected = np.asarray(getattr(reference, observable))
        corrected_values = np.asarray(getattr(corrected, observable))
        _assert_componentwise_close(corrected_values, expected, scale)
        uncorrected_values = np.asarray(getattr(uncorrected, observable))
        allowed = 1e-12 * scale.reshape(1, 1, -1) + 1e-12 * np.abs(expected)
        assert np.any(np.abs(uncorrected_values - expected) > 1e3 * allowed)


def test_public_select_preserves_payload_context_and_compare_rejects_response_types():
    surface, root, targets = _default_targets()
    response = _response_setup(surface=surface, targets=targets, root=root).run()
    result = response.apply(
        vertical_displacement=FrequencySeries(
            _AMPLITUDES_M, frequencies=_FREQUENCIES_HZ * u.Hz, unit=u.m
        ),
        representation=PhasorRepresentation(),
        reference_time=_REFERENCE_TIME,
    )

    selected = result.select("gradient", "potential")
    assert selected.values.gradient is result.values.gradient
    assert selected.values.potential is result.values.potential
    assert selected.evaluation is result.evaluation
    assert selected.bindings["gradient"] is result.bindings["gradient"]
    assert selected.bindings["potential"] is result.bindings["potential"]
    with pytest.raises(TypeError, match="reference must be an EvaluationResult"):
        compare(reference=response, candidate=result)
    with pytest.raises(TypeError, match="candidate must be an EvaluationResult"):
        compare(reference=result, candidate=response.coefficients("acceleration"))
