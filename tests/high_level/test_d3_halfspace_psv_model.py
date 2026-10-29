"""D3 high-level contracts for traction-free P/SV half-space phasors."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesMatrix
from gwpy.time import LIGOTimeGPS

from gragra.constants import G_SI
from gragra.geometry._frame import Frame
from gragra.geometry._halfspace import HalfSpace
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


def _inputs(
    *,
    geometry: HalfSpace | None = None,
    wave_type: str = "SV",
    incidence_angle_rad: float = 0.63,
    azimuth_rad: float = 0.47,
    displacement: FrequencySeries | None = None,
    reference_time: object = (1_000_000_000, 500_000_000),
) -> dict[str, object]:
    frame = Frame("surface") if geometry is None else geometry.frame
    if displacement is None:
        displacement = FrequencySeries(
            np.array([1.2 - 0.4j, -0.7 + 0.9j]) * u.nm,
            frequencies=np.array([1.3, 2.7]) * u.Hz,
            unit=u.m,
            epoch=LIGOTimeGPS(1_250_000_000, 125_000_000),
        )
    return {
        "geometry": HalfSpace(frame) if geometry is None else geometry,
        "density": 2500 * u.kg / u.m**3,
        "vp": 4000 * u.m / u.s,
        "vs": 2300 * u.m / u.s,
        "displacement": displacement,
        "representation": PhasorRepresentation(),
        "reference_time": reference_time,
        "wave_type": wave_type,
        "incidence_angle_rad": incidence_angle_rad,
        "azimuth_rad": azimuth_rad,
        "phase_origin": np.array([0.3, -0.7, 1.2]) * u.m,
    }


def test_factory_describes_total_linearized_psv_field_and_q_reference() -> None:
    from gragra.analytic.seismic._halfspace_psv_model import halfspace_psv

    model = halfspace_psv(**_inputs())
    definition = model.spec.field_definition

    assert definition.kind is FieldKind.FIELD_CHANGE
    assert definition.source_kind == "halfspace_psv_perturbation"
    assert definition.contribution == Contribution("total")
    assert definition.potential_reference is (
        PotentialReference.HALFSPACE_FOURIER_GREEN_POSITIVE_Q
    )
    assert definition.change is not None
    assert definition.change.form is ChangeForm.LINEARIZED_RESPONSE
    assert definition.change.changed_input == "displacement"
    assert definition.change.order == 1
    assert definition.change.reference == ReferenceDefinition(
        ReferenceStatus.DEFINED,
        ReferenceTemporalKind.STATE,
        "zero_displacement_halfspace_psv",
    )
    assert definition.change.fixed_conditions == (
        "azimuth",
        "density",
        "frequency",
        "geometry",
        "incidence_angle",
        "phase_origin",
        "vp",
        "vs",
        "wave_type",
    )


@pytest.mark.parametrize("wave_type", ("P", "SV"))
def test_normal_incidence_uses_only_the_documented_continuous_q_family(
    wave_type: str,
) -> None:
    from gragra.analytic.seismic._halfspace_psv_model import halfspace_psv

    model = halfspace_psv(**_inputs(wave_type=wave_type, incidence_angle_rad=0.0))
    assert model.spec.field_definition.potential_reference is (
        PotentialReference.SPHERICAL_MEAN_ZERO_AT_INFINITY
    )
    targets = TargetPoints(
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]) * u.m,
        ("above", "below"),
        model.geometry.frame,
    )
    result = model.at(targets).run()
    assert result.g_applications == 1
    assert result.field_definition.potential_reference is (
        PotentialReference.SPHERICAL_MEAN_ZERO_AT_INFINITY
    )


def test_rejects_q_zero_inputs_outside_positive_frequency_phasor_domain() -> None:
    from gragra.analytic.seismic._halfspace_psv_model import halfspace_psv

    zero_frequency = FrequencySeries([1 + 0j] * u.m, frequencies=[0.0] * u.Hz)
    with pytest.raises(ValueError, match="positive"):
        halfspace_psv(
            **_inputs(
                wave_type="P", incidence_angle_rad=0.0, displacement=zero_frequency
            )
        )

    with pytest.raises(TypeError, match="reference_time"):
        halfspace_psv(
            **_inputs(wave_type="P", incidence_angle_rad=0.0, reference_time=None)
        )


def test_run_uses_only_native_g_free_values_and_rotates_both_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gragra.analytic.halfspace_psv import halfspace_psv_observables_g_free
    from gragra.analytic.seismic._halfspace_psv_model import halfspace_psv

    world = Frame("world")
    basis = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    surface = Frame(
        "surface", origin=np.array([6.0, -3.0, 2.0]) * u.m, basis=basis, parent=world
    )
    local_targets_m = np.array([[2.2, -1.3, 2.4], [-3.1, 4.2, -1.8]])
    targets = TargetPoints(
        (surface.origin_m + local_targets_m @ basis.T) * u.m,
        ("above", "below"),
        world,
    )
    kw = _inputs(geometry=HalfSpace(surface))
    model = halfspace_psv(**kw)
    setup = model.at(targets, output_frame=world)

    assert setup.resolved.target_positions_m.shape == (2, 3)
    np.testing.assert_allclose(setup.resolved.target_positions_m, local_targets_m)
    assert setup.context.reference_time_gps == (1_000_000_000, 500_000_000)
    assert model.spec.displacement_epoch_gps == (1_250_000_000, 125_000_000)
    assert model.spec.phase_origin_m.tolist() == [0.3, -0.7, 1.2]

    import gragra.observables.halfspace_psv as si_wrapper

    def unexpected_si_path(*_args: object, **_kwargs: object) -> object:
        pytest.fail("the high-level model must call the G-free kernel directly")

    monkeypatch.setattr(si_wrapper, "halfspace_psv_observables", unexpected_si_path)
    result = setup.run()

    spec = model.spec
    local = halfspace_psv_observables_g_free(
        local_targets_m,
        wave_type=spec.wave_type,
        density_kg_m3=spec.density_kg_m3,
        v_p_m_s=spec.vp_m_s,
        v_s_m_s=spec.vs_m_s,
        frequency_hz=np.asarray(spec.frequencies_hz),
        incidence_angle_rad=spec.incidence_angle_rad,
        azimuth_rad=spec.azimuth_rad,
        incident_displacement_m=spec.displacement_m,
        phase_origin_m=spec.phase_origin_m,
    )
    rotation = surface.rotation_to(world)
    expected_acceleration = G_SI * np.einsum(
        "ij,kmj->kmi", rotation, local.acceleration
    )
    expected_gradient = G_SI * np.einsum(
        "ia,kmab,jb->kmij", rotation, local.gradient, rotation
    )

    assert result.g_applications == 1
    assert isinstance(result.potential, FrequencySeriesMatrix)
    assert isinstance(result.acceleration, FrequencySeriesMatrix)
    assert isinstance(result.gradient, FrequencySeriesMatrix)
    assert result.potential.shape == (2, 1, 2)
    assert result.acceleration.shape == (2, 3, 2)
    assert result.gradient.shape == (2, 9, 2)
    assert tuple(result.acceleration.rows.keys()) == ("above", "below")
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
        result.potential.frequencies.to_value(u.Hz), [1.3, 2.7]
    )
    assert result.potential.meta[0, 0].unit == u.m**2 / u.s**2
    assert result.acceleration.meta[0, 0].unit == u.m / u.s**2
    assert result.gradient.meta[0, 0].unit == u.s**-2
    np.testing.assert_allclose(
        np.asarray(result.potential)[:, 0, :],
        (G_SI * local.potential).T,
        rtol=2e-14,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(result.acceleration),
        expected_acceleration.transpose(1, 2, 0),
        rtol=2e-14,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(result.gradient),
        expected_gradient.reshape(2, 2, 9).transpose(1, 2, 0),
        rtol=2e-14,
        atol=0,
    )


def test_empty_targets_are_rejected_before_result_construction() -> None:
    from gragra.analytic.seismic._halfspace_psv_model import halfspace_psv

    model = halfspace_psv(**_inputs())
    targets = TargetPoints(np.empty((0, 3)) * u.m, (), model.geometry.frame)

    with pytest.raises(ValueError, match="targets must be nonempty"):
        model.at(targets)
