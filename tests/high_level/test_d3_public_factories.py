"""Shared connection contract for the five D3 public factories."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesMatrix
from gwpy.time import LIGOTimeGPS

from gragra.analytic.objects import uniform_cuboid, uniform_cylinder
from gragra.analytic.seismic import halfspace_psv, spherical_cavity, spherical_shell
from gragra.analytic.values import GFreeGravityValues
from gragra.constants import G_SI
from gragra.geometry import Cuboid, Cylinder, Frame, HalfSpace, Sphere, SphericalShell
from gragra.results import PhasorRepresentation, compare
from gragra.targets import TargetPoints

FACTORIES = (
    "uniform_cuboid",
    "uniform_cylinder",
    "spherical_cavity",
    "spherical_shell",
    "halfspace_psv",
)
PHASOR_FACTORIES = ("spherical_cavity", "spherical_shell", "halfspace_psv")
FREQUENCIES_HZ = np.array([1.25, 2.75])
REFERENCE_TIME = (1_000_000_000, 125_000_000)
EPOCH_GPS = (1_234_567_890, 500_000_000)
EPOCH = LIGOTimeGPS(*EPOCH_GPS)
PHASE_ORIGIN_M = np.array([0.25, -0.4, 0.7])
DISPLACEMENT_M = np.array([1.0e-5 + 2.0e-6j, -3.0e-6 + 4.0e-6j])
TARGET_IDS = ("t0", "t1", "t2")
LOCAL_TARGETS_M = np.array(
    [[0.4, 0.2, 1.0], [2.3, -1.5, -1.8], [-2.7, 1.1, 0.8]], dtype=np.float64
)


def _frames_and_targets():
    root = Frame("root")
    yaw, pitch = 0.41, -0.37
    rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ry = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )
    geometry_frame = Frame(
        "geometry",
        origin=np.array([3.0, -2.0, 1.0]) * u.m,
        basis=rz @ ry,
        parent=root,
    )
    positions_root = geometry_frame.transform_points_m(LOCAL_TARGETS_M, root)
    targets = TargetPoints(positions_root * u.m, TARGET_IDS, root)
    return root, geometry_frame, targets


def _phasor_inputs(*, include_epoch: bool = True):
    displacement = FrequencySeries(
        DISPLACEMENT_M.copy(),
        frequencies=FREQUENCIES_HZ * u.Hz,
        unit=u.m,
        epoch=EPOCH if include_epoch else None,
    )
    phase_origin = PHASE_ORIGIN_M.copy() * u.m
    return displacement, phase_origin


def _make_model(factory_name: str, geometry_frame: Frame, *, include_epoch=True):
    if factory_name == "uniform_cuboid":
        return (
            uniform_cuboid(
                Cuboid(
                    np.zeros(3) * u.m, np.array([4.0, 4.0, 4.0]) * u.m, geometry_frame
                ),
                2500 * u.kg / u.m**3,
            ),
            None,
            None,
        )
    if factory_name == "uniform_cylinder":
        return (
            uniform_cylinder(
                Cylinder(np.zeros(3) * u.m, 2.0 * u.m, 4.0 * u.m, geometry_frame),
                2500 * u.kg / u.m**3,
            ),
            None,
            None,
        )

    displacement, phase_origin = _phasor_inputs(include_epoch=include_epoch)
    representation = PhasorRepresentation()
    if factory_name == "spherical_cavity":
        model = spherical_cavity(
            Sphere(np.zeros(3) * u.m, 1.0 * u.m, geometry_frame),
            2500 * u.kg / u.m**3,
            3000.0 * u.m / u.s,
            displacement,
            representation,
            REFERENCE_TIME,
            "P",
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
            phase_origin,
        )
    elif factory_name == "spherical_shell":
        model = spherical_shell(
            SphericalShell(np.zeros(3) * u.m, 0.7 * u.m, 5.0 * u.m, geometry_frame),
            2500 * u.kg / u.m**3,
            3000.0 * u.m / u.s,
            displacement,
            representation,
            REFERENCE_TIME,
            phase_origin,
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
            "P",
        )
    elif factory_name == "halfspace_psv":
        model = halfspace_psv(
            HalfSpace(geometry_frame),
            2500 * u.kg / u.m**3,
            4000.0 * u.m / u.s,
            2300.0 * u.m / u.s,
            displacement,
            representation,
            REFERENCE_TIME,
            "SV",
            0.4,
            0.65,
            phase_origin,
        )
    else:
        raise AssertionError(f"unknown D3 factory: {factory_name}")
    return model, displacement, phase_origin


def _assert_result_axes(result, factory_name: str) -> None:
    count = len(TARGET_IDS)
    assert result.context.targets.ids == TARGET_IDS
    if factory_name in PHASOR_FACTORIES:
        assert isinstance(result.potential, FrequencySeriesMatrix)
        assert isinstance(result.acceleration, FrequencySeriesMatrix)
        assert isinstance(result.gradient, FrequencySeriesMatrix)
        assert result.potential.shape == (count, 1, len(FREQUENCIES_HZ))
        assert result.acceleration.shape == (count, 3, len(FREQUENCIES_HZ))
        assert result.gradient.shape == (count, 9, len(FREQUENCIES_HZ))
        assert tuple(result.potential.rows.keys()) == TARGET_IDS
        assert tuple(result.acceleration.cols.keys()) == ("x", "y", "z")
        assert tuple(result.gradient.cols.keys()) == (
            "xx",
            "xy",
            "xz",
            "yx",
            "yy",
            "yz",
            "zx",
            "zy",
            "zz",
        )
        np.testing.assert_array_equal(
            result.gradient.frequencies.to_value(u.Hz), FREQUENCIES_HZ
        )
    else:
        assert isinstance(result.potential, u.Quantity)
        assert result.potential.shape == (count,)
        assert result.acceleration.shape == (count, 3)
        assert result.gradient.shape == (count, 3, 3)


def _assert_frame_rotation(local_result, root_result, rotation, factory_name: str):
    if factory_name in PHASOR_FACTORIES:
        np.testing.assert_allclose(
            np.asarray(root_result.potential, dtype=np.complex128),
            np.asarray(local_result.potential, dtype=np.complex128),
        )
        local_acceleration = np.asarray(local_result.acceleration)
        local_gradient = np.asarray(local_result.gradient).reshape(
            len(TARGET_IDS), 3, 3, len(FREQUENCIES_HZ)
        )
        expected_acceleration = np.einsum("ij,kjf->kif", rotation, local_acceleration)
        expected_gradient = np.einsum(
            "ia,kabf,jb->kijf", rotation, local_gradient, rotation
        )
        np.testing.assert_allclose(
            np.asarray(root_result.acceleration), expected_acceleration, rtol=2e-12
        )
        np.testing.assert_allclose(
            np.asarray(root_result.gradient),
            expected_gradient.reshape(len(TARGET_IDS), 9, len(FREQUENCIES_HZ)),
            rtol=2e-12,
        )
    else:
        np.testing.assert_allclose(
            root_result.potential.to_value(u.m**2 / u.s**2),
            local_result.potential.to_value(u.m**2 / u.s**2),
        )
        unit_a = u.m / u.s**2
        unit_g = u.s**-2
        local_acceleration = local_result.acceleration.to_value(unit_a)
        local_gradient = local_result.gradient.to_value(unit_g)
        expected_acceleration = np.einsum("ij,kj->ki", rotation, local_acceleration)
        expected_gradient = np.einsum(
            "ia,kab,jb->kij", rotation, local_gradient, rotation
        )
        np.testing.assert_allclose(
            root_result.acceleration.to_value(unit_a),
            expected_acceleration,
            rtol=2e-12,
        )
        np.testing.assert_allclose(
            root_result.gradient.to_value(unit_g), expected_gradient, rtol=2e-12
        )


@pytest.mark.parametrize("factory_name", FACTORIES)
def test_d3_public_factories_share_result_axes_select_compare_and_frame_contract(
    factory_name: str,
) -> None:
    root, geometry_frame, targets = _frames_and_targets()
    model, displacement, phase_origin = _make_model(factory_name, geometry_frame)
    expected_displacement = DISPLACEMENT_M.copy()
    if displacement is not None:
        displacement[0] = 99.0 * u.m
        phase_origin[0] = 99.0 * u.m
        np.testing.assert_array_equal(model.spec.displacement_m, expected_displacement)
        np.testing.assert_array_equal(model.spec.phase_origin_m, PHASE_ORIGIN_M)

    result = model.at(targets, output_frame=root).run()
    local_result = model.at(targets, output_frame=geometry_frame).run()

    assert result.context.frame is root
    assert result.context.targets is targets
    assert result.g_applications == 1
    assert result.record.gravitational_constant_si == G_SI
    assert result.parameters["target_count"][0] == len(TARGET_IDS)
    assert tuple(result.formulas["observable"]) == (
        "potential",
        "acceleration",
        "gradient",
    )
    assert not result.references.empty
    _assert_result_axes(result, factory_name)
    _assert_result_axes(local_result, factory_name)

    selected = result.select("potential", "acceleration")
    assert selected.observables == ("potential", "acceleration")
    assert selected.context is result.context
    assert selected.evaluation is result.evaluation
    assert selected.record is result.record
    assert selected.potential is result.potential
    comparison = compare(reference=result, candidate=result)
    assert comparison.reference is result
    assert comparison.candidate is result

    rotation = geometry_frame.rotation_to(root)
    assert not np.allclose(rotation, np.eye(3))
    _assert_frame_rotation(local_result, result, rotation, factory_name)

    if factory_name in PHASOR_FACTORIES:
        assert result.context.reference_time_gps == REFERENCE_TIME
        assert model.spec.reference_time_gps == REFERENCE_TIME
        assert model.spec.displacement_epoch_gps == EPOCH_GPS
        parameters = result.parameters
        assert parameters["phase_origin_x"][0].to_value(u.m) == PHASE_ORIGIN_M[0]
        assert parameters["phase_origin_y"][0].to_value(u.m) == PHASE_ORIGIN_M[1]
        assert parameters["phase_origin_z"][0].to_value(u.m) == PHASE_ORIGIN_M[2]


@pytest.mark.parametrize("factory_name", FACTORIES)
def test_d3_public_factories_apply_g_once_to_native_g_free_values(
    factory_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, geometry_frame, targets = _frames_and_targets()
    model, _, _ = _make_model(factory_name, geometry_frame)
    is_phasor = factory_name in PHASOR_FACTORIES
    prefix = (len(FREQUENCIES_HZ), len(TARGET_IDS)) if is_phasor else (len(TARGET_IDS),)
    dtype = np.complex128 if is_phasor else np.float64
    values = GFreeGravityValues(
        np.full(prefix, 1.0 + 2.0j if is_phasor else 1.0, dtype=dtype),
        np.full((*prefix, 3), 2.0 - 1.0j if is_phasor else 2.0, dtype=dtype),
        np.full((*prefix, 3, 3), 3.0 + 0.5j if is_phasor else 3.0, dtype=dtype),
    )
    calls = []

    def fake_native(*_args, **_kwargs):
        calls.append(None)
        if factory_name == "spherical_cavity":
            return dict.fromkeys(("bulk", "wall", "total"), values)
        if factory_name == "spherical_shell":
            return dict.fromkeys(
                ("bulk", "inner_wall", "outer_boundary", "total"), values
            )
        return values

    lookup = {
        "uniform_cuboid": (
            "gragra.analytic.cavity_cuboid",
            "cuboid_observables_g_free",
        ),
        "uniform_cylinder": (
            "gragra.analytic.objects._cylinder_model",
            "cylinder_analytic_observables_g_free",
        ),
        "spherical_cavity": (
            "gragra.analytic.seismic._spherical_cavity_model",
            "spherical_cavity_observables_g_free",
        ),
        "spherical_shell": (
            "gragra.analytic.u10_spherical_shell_general",
            "spherical_shell_observables_g_free",
        ),
        "halfspace_psv": (
            "gragra.analytic.seismic._halfspace_psv_model",
            "halfspace_psv_observables_g_free",
        ),
    }
    module_name, function_name = lookup[factory_name]
    monkeypatch.setattr(
        importlib.import_module(module_name), function_name, fake_native
    )
    result = model.at(targets, output_frame=geometry_frame).run()

    assert len(calls) == 1
    assert result.g_applications == 1
    assert result.context.g_convention == "SI_G_applied_once"
    if is_phasor:
        np.testing.assert_allclose(np.asarray(result.potential), G_SI * (1.0 + 2.0j))
        np.testing.assert_allclose(np.asarray(result.acceleration), G_SI * (2.0 - 1.0j))
        np.testing.assert_allclose(np.asarray(result.gradient), G_SI * (3.0 + 0.5j))
    else:
        np.testing.assert_allclose(result.potential.to_value(u.m**2 / u.s**2), G_SI)
        np.testing.assert_allclose(
            result.acceleration.to_value(u.m / u.s**2), G_SI * 2.0
        )
        np.testing.assert_allclose(result.gradient.to_value(u.s**-2), G_SI * 3.0)


@pytest.mark.parametrize("factory_name", PHASOR_FACTORIES)
def test_d3_epoch_is_provenance_and_does_not_rephase(factory_name: str) -> None:
    root, geometry_frame, targets = _frames_and_targets()
    with_epoch, _, _ = _make_model(factory_name, geometry_frame, include_epoch=True)
    no_epoch, _, _ = _make_model(factory_name, geometry_frame, include_epoch=False)
    result_with_epoch = with_epoch.at(targets, output_frame=root).run()
    result_without_epoch = no_epoch.at(targets, output_frame=root).run()

    assert result_with_epoch.context.reference_time_gps == REFERENCE_TIME
    assert with_epoch.spec.displacement_epoch_gps == EPOCH_GPS
    assert no_epoch.spec.displacement_epoch_gps is None
    for name in ("potential", "acceleration", "gradient"):
        np.testing.assert_array_equal(
            np.asarray(getattr(result_with_epoch, name)),
            np.asarray(getattr(result_without_epoch, name)),
        )


def test_compare_rejects_cavity_results_with_different_potential_references() -> None:
    root, geometry_frame, targets = _frames_and_targets()
    displacement, phase_origin = _phasor_inputs()
    geometry = Sphere(np.zeros(3) * u.m, 1.0 * u.m, geometry_frame)
    common = {
        "geometry": geometry,
        "density": 2500 * u.kg / u.m**3,
        "phase_speed": 3000.0 * u.m / u.s,
        "displacement": displacement,
        "representation": PhasorRepresentation(),
        "reference_time": REFERENCE_TIME,
        "phase_origin": phase_origin,
        "part": "total",
    }
    p_result = (
        spherical_cavity(
            **common,
            wave_type="P",
            propagation_direction=[0, 0, 1],
            polarization=[0, 0, 1],
        )
        .at(targets, output_frame=root)
        .run()
    )
    s_result = (
        spherical_cavity(
            **common,
            wave_type="S",
            propagation_direction=[0, 0, 1],
            polarization=[1, 0, 0],
        )
        .at(targets, output_frame=root)
        .run()
    )

    with pytest.raises(ValueError, match="potential_reference"):
        compare(reference=p_result, candidate=s_result)
