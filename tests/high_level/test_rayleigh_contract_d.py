"""Contracts shared by the Rayleigh model, result, and plotting lanes."""

import numpy as np
import pytest
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeries

from gragra.geometry._frame import Frame
from gragra.targets._high_points import TargetPoints


def inputs(frame=None, *, frequencies=(1.0, 2.0), amplitude=(1 + 2j, 3 + 4j)):
    from gragra.geometry._halfspace import HalfSpace

    frame = Frame("root") if frame is None else frame
    from gragra.results import PhasorRepresentation

    return {
        "geometry": HalfSpace(frame),
        "density": 2500 * u.kg / u.m**3,
        "vp": 4000 * u.m / u.s,
        "vs": 2300 * u.m / u.s,
        "vertical_displacement": FrequencySeries(
            np.asarray(amplitude), frequencies=np.asarray(frequencies) * u.Hz, unit=u.m
        ),
        "representation": PhasorRepresentation(),
        "reference_time": (1000000000, 1_000_000_002),
        "phase_origin": np.array([0.0, 0.0, 0.0]) * u.m,
        "direction": (3.0, 4.0, 0.0),
    }


def test_spec_owns_input_and_normalizes_gps_direction() -> None:
    from gragra._rayleigh_contract import RayleighModelSpec

    kw = inputs()
    spec = RayleighModelSpec(**kw)
    assert spec.reference_time_gps == (1000000001, 2)
    assert spec.frequencies_hz == (1.0, 2.0)
    np.testing.assert_allclose(spec.direction, [0.6, 0.8, 0.0])
    kw["vertical_displacement"][0] = (99 + 88j) * u.m
    np.testing.assert_array_equal(spec.vertical_displacement_m, [1 + 2j, 3 + 4j])
    assert not spec.vertical_displacement_m.flags.writeable


def test_spec_retains_owned_series_labels_without_using_epoch_as_phase_reference() -> (
    None
):
    from gwpy.time import LIGOTimeGPS

    from gragra._rayleigh_contract import RayleighModelSpec
    from gragra.results import PotentialReference

    kw = inputs(frequencies=(1.0, 1.3, 2.7), amplitude=(1j, 2j, 3j))
    kw["vertical_displacement"].name = "vertical displacement"
    kw["vertical_displacement"].channel = "SENSOR:Z"
    kw["vertical_displacement"].epoch = LIGOTimeGPS(123, 4)
    spec = RayleighModelSpec(**kw)
    assert spec.displacement_name == "vertical displacement"
    assert spec.displacement_channel == "SENSOR:Z"
    assert spec.displacement_epoch_gps == (123, 4)
    assert spec.reference_time_gps == (1000000001, 2)
    assert spec.displacement_unit == u.m
    assert spec.displacement_input_unit == u.m
    assert (
        spec.field_definition.potential_reference
        is PotentialReference.VERTICAL_FAR_FIELD_DECAY_POSITIVE_KH
    )
    kw["vertical_displacement"].name = "changed"
    assert spec.vertical_displacement.name == "vertical displacement"
    np.testing.assert_array_equal(
        spec.vertical_displacement.frequencies.to_value(u.Hz), [1.0, 1.3, 2.7]
    )


def test_spec_retains_original_displacement_unit_as_provenance() -> None:
    from gragra._rayleigh_contract import RayleighModelSpec

    kw = inputs(frequencies=(1.0,), amplitude=(1 + 2j,))
    kw["vertical_displacement"] = FrequencySeries(
        [100 + 200j], frequencies=[1] * u.Hz, unit=u.cm
    )
    spec = RayleighModelSpec(**kw)
    assert spec.displacement_input_unit == u.cm
    assert spec.displacement_unit == u.m
    np.testing.assert_array_equal(spec.vertical_displacement_m, [1 + 2j])


@pytest.mark.parametrize("axis", [(), (0.0,), (1.0, 1.0), (2.0, 1.0), (1.0, np.inf)])
def test_spec_rejects_invalid_frequency_axis(axis) -> None:
    from gragra._rayleigh_contract import RayleighModelSpec

    kw = inputs(frequencies=axis, amplitude=np.ones(len(axis), dtype=complex))
    with pytest.raises(ValueError, match="frequenc"):
        RayleighModelSpec(**kw)


def test_resolved_requires_exact_rayleigh_spec_and_nonempty_targets() -> None:
    from gragra._rayleigh_contract import RayleighModelSpec, RayleighResolvedEvaluation

    spec = RayleighModelSpec(**inputs())
    target = TargetPoints(
        np.array([[0.0, 0.0, -1.0]]) * u.m, ("a",), spec.geometry.frame
    )
    with pytest.raises(TypeError, match="RayleighModelSpec"):
        RayleighResolvedEvaluation(object(), target, spec.geometry.frame)

    class Model:
        def __init__(self, spec):
            self.spec = spec

    resolved = RayleighResolvedEvaluation(Model(spec), target, spec.geometry.frame)
    assert resolved.observables == ("potential", "acceleration", "gradient")
    np.testing.assert_array_equal(resolved.target_positions_m, [[0, 0, -1]])
    with pytest.raises(ValueError, match="nonempty"):
        RayleighResolvedEvaluation(
            Model(spec),
            TargetPoints(np.empty((0, 3)) * u.m, (), spec.geometry.frame),
            spec.geometry.frame,
        )


def test_surface_diagnosis_distinguishes_surface_and_ambiguous_side() -> None:
    from gragra._rayleigh_contract import (
        RayleighModelSpec,
        RayleighResolvedEvaluation,
        RayleighSurfacePrecisionError,
    )
    from gragra._rayleigh_evaluation_setup import RayleighEvaluationSetup

    root = Frame("root")
    surface = Frame("surface", origin=[0, 0, 1e9] * u.m, parent=root)
    spec = RayleighModelSpec(**inputs(surface))

    class Model:
        spec = None

    model = Model()
    model.spec = spec
    on_setup = RayleighEvaluationSetup(
        RayleighResolvedEvaluation(
            model, TargetPoints([[0, 0, 0]] * u.m, ("on",), surface), surface
        )
    )
    assert on_setup.resolved.target_positions_m[0, 2] == 0
    with pytest.raises(RayleighSurfacePrecisionError) as on:
        on_setup.validate()
    assert on.value.reason == "surface"
    assert on.value.target_id == "on"
    near_setup = RayleighEvaluationSetup(
        RayleighResolvedEvaluation(
            model, TargetPoints([[0, 0, 1e9 - 1e-7]] * u.m, ("near",), root), surface
        )
    )
    with pytest.raises(RayleighSurfacePrecisionError) as near:
        near_setup.validate()
    assert near.value.reason == "coordinate_precision"
    assert near.value.threshold_m > 0
    assert near.value.edge_count == 1


def test_complex_rotation_uses_transpose_without_conjugation() -> None:
    from gragra._rayleigh_contract import rotate_rayleigh_tensor, rotate_rayleigh_vector

    root = Frame("root")
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    turned = Frame("turned", basis=rotation, parent=root)
    vector = np.array([1 + 2j, 3 + 4j, 5 + 6j])
    tensor = np.arange(9).reshape(3, 3) + 1j * np.arange(9, 18).reshape(3, 3)
    np.testing.assert_array_equal(
        rotate_rayleigh_vector(vector, turned, root), rotation @ vector
    )
    np.testing.assert_array_equal(
        rotate_rayleigh_tensor(tensor, turned, root), rotation @ tensor @ rotation.T
    )
    with pytest.raises(ValueError, match="real"):
        turned.transform_tensors(tensor, root)


def test_setup_reports_regions_and_exposes_pre_run_metadata() -> None:
    from gragra._rayleigh_contract import (
        RayleighModelSpec,
        RayleighResolvedEvaluation,
    )
    from gragra._rayleigh_evaluation_setup import RayleighEvaluationSetup

    spec = RayleighModelSpec(**inputs())

    class Model:
        pass

    model = Model()
    model.spec = spec
    targets = TargetPoints(
        [[0, 0, -1], [0, 0, 2]] * u.m, ("under", "over"), spec.geometry.frame
    )
    resolved = RayleighResolvedEvaluation(model, targets, spec.geometry.frame)
    setup = RayleighEvaluationSetup(resolved)
    assert setup.resolved is resolved and setup.evaluation is resolved
    assert setup.validate().target_regions == {"under": "below", "over": "above"}
    assert setup.parameters["density"].unit == u.kg / u.m**3
    assert setup.parameters["frequency_min"].unit == u.Hz
    assert setup.parameters["frequency_max"].unit == u.Hz
    assert setup.parameters["displacement_max"].unit == u.m
    assert "direction_x" in setup.parameters.colnames
    assert set(setup.formulas["observable"]) == set(resolved.observables)
    assert len(setup.references)
    bindings = {name: setup.context.binding(name) for name in resolved.observables}
    assert bindings["gradient"].component_axes == (("x", "y", "z"), ("x", "y", "z"))
    assert bindings["potential"].frame is None
    assert bindings["gradient"].reference_time_gps == (1000000001, 2)


def test_native_matrix_packing_preserves_all_asymmetric_complex_tensor_components() -> (
    None
):
    from gragra._rayleigh_evaluation_setup import _native_matrices

    tensor = np.arange(9).reshape(1, 1, 3, 3) + 1j * np.arange(9, 18).reshape(
        1, 1, 3, 3
    )
    matrices = _native_matrices(
        {
            "potential": np.ones((1, 1), complex),
            "acceleration": np.ones((1, 1, 3), complex),
            "gradient": tensor,
        },
        ("t",),
        (1.25,),
    )
    gradient = matrices["gradient"]
    assert gradient.shape == (1, 9, 1)
    assert list(gradient.cols.keys()) == [
        "xx",
        "xy",
        "xz",
        "yx",
        "yy",
        "yz",
        "zx",
        "zy",
        "zz",
    ]
    np.testing.assert_array_equal(
        np.asarray(gradient)[0, :, 0], tensor[0, 0].reshape(9)
    )
    assert all(gradient.meta[0, j].unit == u.s**-2 for j in range(9))


def test_surface_diagnosis_parallel_translation_and_local_boundary() -> None:
    from gragra._rayleigh_contract import (
        RayleighModelSpec,
        RayleighResolvedEvaluation,
        RayleighSurfacePrecisionError,
    )
    from gragra._rayleigh_evaluation_setup import RayleighEvaluationSetup

    root = Frame("root")
    surface = Frame("surface", origin=[1e12, -1e12, 10] * u.m, parent=root)
    spec = RayleighModelSpec(**inputs(surface))

    class Model:
        pass

    model = Model()
    model.spec = spec
    safe = TargetPoints(
        [[1e12, -1e12, 9], [9e15, -9e15, 11]] * u.m,
        ("below", "far-above"),
        root,
    )
    resolved = RayleighResolvedEvaluation(model, safe, surface)
    np.testing.assert_array_equal(resolved.target_positions_m[:, 2], [-1, 1])
    on = RayleighEvaluationSetup(
        RayleighResolvedEvaluation(
            model, TargetPoints([[0, 0, 0]] * u.m, ("on",), surface), surface
        )
    )
    with pytest.raises(RayleighSurfacePrecisionError) as error:
        on.validate()
    assert error.value.reason == "surface"
    for side in (-1.0, 1.0):
        local = TargetPoints([[0, 0, side * 1e-15]] * u.m, ("near",), surface)
        local_resolved = RayleighResolvedEvaluation(model, local, surface)
        np.testing.assert_array_equal(
            local_resolved.target_positions_m[0, 2],
            side * 1e-15,
        )
        assert RayleighEvaluationSetup(local_resolved).validate().valid


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_surface_precision_rejects_roundoff_from_nearly_orthogonal_frame(
    side: float,
) -> None:
    from gragra._rayleigh_contract import RayleighSurfacePrecisionError
    from gragra.analytic.seismic import rayleigh_halfspace

    root = Frame("root")
    basis = np.eye(3)
    basis[2, :2] = 1e-13
    surface = Frame("surface", basis=basis, parent=root)
    model = rayleigh_halfspace(**inputs(surface))
    targets = TargetPoints(
        [[3.0, -3.0, side * np.ldexp(1.0, -95)]] * u.m,
        ("near-surface",),
        root,
    )
    setup = model.at(targets)

    errors = []
    for operation in (setup.validate, setup.run):
        with pytest.raises(RayleighSurfacePrecisionError) as caught:
            operation()
        assert caught.value.target_id == "near-surface"
        assert caught.value.reason == "coordinate_precision"
        assert abs(caught.value.signed_distance_m) <= caught.value.threshold_m
        errors.append(
            (
                caught.value.target_id,
                caught.value.reason,
                caught.value.signed_distance_m,
                caught.value.threshold_m,
                caught.value.edge_count,
                caught.value.scale_m,
                caught.value.unit,
                caught.value.observables,
            )
        )
    assert errors[0] == errors[1]


def test_two_edge_surface_diagnosis_is_target_specific() -> None:
    from gragra._rayleigh_contract import (
        RayleighModelSpec,
        RayleighResolvedEvaluation,
        RayleighSurfacePrecisionError,
    )
    from gragra._rayleigh_evaluation_setup import RayleighEvaluationSetup

    root = Frame("root")
    source = Frame("source", origin=[0, 0, 1e9] * u.m, parent=root)
    surface = Frame("surface", origin=[0, 0, 1e9] * u.m, parent=root)
    spec = RayleighModelSpec(**inputs(surface))

    class Model:
        pass

    model = Model()
    model.spec = spec
    setup = RayleighEvaluationSetup(
        RayleighResolvedEvaluation(
            model, TargetPoints([[0, 0, 1e-7]] * u.m, ("near",), source), surface
        )
    )
    with pytest.raises(RayleighSurfacePrecisionError) as error:
        setup.validate()
    assert error.value.reason == "coordinate_precision"
    assert error.value.edge_count == 2
    assert error.value.target_id == "near"


def test_spatial_origin_and_gps_reparameterization() -> None:
    from gragra._rayleigh_contract import RayleighModelSpec, gps_offset_seconds
    from gragra.observables.rayleigh_literature import (
        rayleigh_halfspace_allspace_observables,
    )

    spec = RayleighModelSpec(**inputs(frequencies=(1.25,), amplitude=(1 + 2j,)))
    frequency = spec.frequencies_hz[0]
    k = 2 * np.pi * frequency / spec.rayleigh_speed_m_s
    shift = np.array([17.0, -9.0, 0.0])
    assert gps_offset_seconds((1000000001, 2), (1000000000, 1_000_000_002)) == 0
    assert gps_offset_seconds((1000000000, 2), (1000000000, 1)) == 1e-9
    dt = gps_offset_seconds((1000000003, 2), spec.reference_time_gps)
    assert dt == 2
    args = {
        "targets_m": [[12, 5, -3]],
        "density_kg_m3": spec.density_kg_m3,
        "v_p_m_s": spec.vp_m_s,
        "v_s_m_s": spec.vs_m_s,
        "frequency_hz": frequency,
        "direction": spec.direction,
    }
    original = rayleigh_halfspace_allspace_observables(
        **args,
        vertical_surface_amplitude_m=spec.vertical_displacement_m[0],
        phase_origin_m=spec.phase_origin_m,
    )
    shifted = rayleigh_halfspace_allspace_observables(
        **args,
        vertical_surface_amplitude_m=spec.vertical_displacement_m[0]
        * np.exp(1j * k * np.dot(spec.direction, shift)),
        phase_origin_m=shift,
    )
    for name in ("potential", "acceleration", "gradient"):
        old = np.asarray(getattr(original, name))
        np.testing.assert_allclose(getattr(shifted, name), old, rtol=1e-14)
        new = old * np.exp(-2j * np.pi * frequency * dt)
        at_old_reference = np.real(old * np.exp(-2j * np.pi * frequency * 4.125))
        at_new_reference = np.real(new * np.exp(-2j * np.pi * frequency * (4.125 - dt)))
        np.testing.assert_allclose(
            at_new_reference, at_old_reference, rtol=1e-12, atol=1e-20
        )


def test_native_matrices_keep_four_target_irregular_three_bin_axis() -> None:
    from gragra._rayleigh_evaluation_setup import _native_matrices

    frequencies = (0.5, 1.2, 3.7)
    ids = ("a", "b", "c", "d")
    data = {
        "potential": np.ones((3, 4), complex),
        "acceleration": np.ones((3, 4, 3), complex),
        "gradient": np.ones((3, 4, 3, 3), complex),
    }
    matrices = _native_matrices(data, ids, frequencies)
    assert [
        matrices[name].shape for name in ("potential", "acceleration", "gradient")
    ] == [(4, 1, 3), (4, 3, 3), (4, 9, 3)]
    for matrix in matrices.values():
        assert list(matrix.rows.keys()) == list(ids)
        np.testing.assert_array_equal(matrix.frequencies.to_value(u.Hz), frequencies)
        assert np.iscomplexobj(np.asarray(matrix))
