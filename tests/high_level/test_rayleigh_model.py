"""Public high-level Rayleigh model and evaluation handoff."""

import numpy as np
import pytest
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeries

from gragra._rayleigh_contract import RayleighSurfacePrecisionError
from gragra.geometry._frame import Frame
from gragra.geometry._halfspace import HalfSpace
from gragra.results import PhasorRepresentation
from gragra.targets._high_points import TargetPoints


def _inputs(frame: Frame | None = None) -> dict[str, object]:
    frame = Frame("surface") if frame is None else frame
    return {
        "geometry": HalfSpace(frame),
        "density": 2500 * u.kg / u.m**3,
        "vp": 4000 * u.m / u.s,
        "vs": 2300 * u.m / u.s,
        "vertical_displacement": FrequencySeries(
            np.array([1 + 2j, 3 - 4j]), frequencies=[1.0, 2.0] * u.Hz, unit=u.m
        ),
        "representation": PhasorRepresentation(),
        "reference_time": (1_000_000_000, 12),
        "phase_origin": np.array([1.0, -2.0, 0.0]) * u.m,
        "direction": np.array([3.0, 4.0, 0.0]),
    }


def test_seismic_package_exports_the_public_model_factories() -> None:
    import gragra.analytic.seismic as seismic

    assert seismic.__all__ == [
        "halfspace_psv",
        "rayleigh_halfspace",
        "rayleigh_response",
        "spherical_cavity",
        "spherical_shell",
        "small_cavity_wall",
        "point_force",
    ]
    assert not hasattr(seismic, "RayleighModel")


def test_factory_owns_displacement_phase_origin_and_direction_inputs() -> None:
    from gragra.analytic.seismic import rayleigh_halfspace

    kw = _inputs()
    model = rayleigh_halfspace(**kw)
    kw["vertical_displacement"][0] = (99 + 88j) * u.m
    kw["phase_origin"][0] = 17 * u.m
    kw["direction"][:] = (0.0, 1.0, 0.0)

    assert model.spec.vertical_displacement_m.tolist() == [1 + 2j, 3 - 4j]
    np.testing.assert_array_equal(model.spec.phase_origin_m, [1.0, -2.0, 0.0])
    np.testing.assert_allclose(model.spec.direction, [0.6, 0.8, 0.0])


def test_at_hands_owned_spec_to_resolved_setup_for_mixed_surface_sides() -> None:
    from gragra.analytic.seismic import rayleigh_halfspace

    surface = Frame("surface")
    target_frame = Frame(
        "targets",
        origin=np.array([10.0, 20.0, 0.0]) * u.m,
        basis=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        parent=surface,
    )
    targets = TargetPoints(
        np.array([[1.0, 2.0, 3.0], [1.0, 2.0, -4.0]]) * u.m,
        ("above", "below"),
        target_frame,
    )
    model = rayleigh_halfspace(**_inputs(surface))
    setup = model.at(targets)

    assert setup.resolved.model is model
    assert setup.resolved.model_spec is model.spec
    assert setup.resolved.targets is targets
    assert setup.resolved.output_frame is target_frame
    assert setup.resolved.target_positions_m.shape == (2, 3)
    np.testing.assert_allclose(
        setup.resolved.target_positions_m, [[8.0, 21.0, 3.0], [8.0, 21.0, -4.0]]
    )
    assert dict(setup.validate().target_regions) == {
        "above": "above",
        "below": "below",
    }

    in_surface_frame = model.at(targets, output_frame=surface)
    assert in_surface_frame.resolved.output_frame is surface


@pytest.mark.parametrize("case", ("surface", "ambiguous"))
def test_at_can_plot_invalid_side_then_validate_and_run_reject_same_diagnostic(
    case,
) -> None:
    import matplotlib.pyplot as plt

    from gragra.analytic.seismic import rayleigh_halfspace

    root = Frame("root")
    surface = (
        root
        if case == "surface"
        else Frame("surface", origin=[0, 0, 1e9] * u.m, parent=root)
    )
    kw = _inputs(surface)
    model = rayleigh_halfspace(**kw)
    targets = TargetPoints(
        [[0.0, 0.0, 0.0 if case == "surface" else 1e9 - 1e-7]] * u.m,
        (case,),
        root,
    )

    with pytest.raises(TypeError):
        model.at(targets, observables=("potential",))
    setup = model.at(targets)
    original_z = setup.resolved.target_positions_m[0, 2]
    figure = setup.plot_setup(view="xz")
    try:
        assert figure.axes
        assert setup.resolved.target_positions_m[0, 2] == original_z
    finally:
        plt.close(figure)
    with pytest.raises(RayleighSurfacePrecisionError) as validated:
        setup.validate()
    with pytest.raises(RayleighSurfacePrecisionError) as executed:
        setup.run()
    for attribute in (
        "target_id",
        "signed_distance_m",
        "threshold_m",
        "edge_count",
        "scale_m",
        "reason",
        "unit",
        "observables",
    ):
        assert getattr(executed.value, attribute) == getattr(validated.value, attribute)
    assert validated.value.reason == (
        "surface" if case == "surface" else "coordinate_precision"
    )


def test_at_rejects_output_frame_with_disconnected_root() -> None:
    from gragra.analytic.seismic import rayleigh_halfspace

    kw = _inputs()
    model = rayleigh_halfspace(**kw)
    targets = TargetPoints([[0.0, 0.0, 1.0]] * u.m, ("above",), kw["geometry"].frame)

    with pytest.raises(ValueError, match="disconnected roots"):
        model.at(targets, output_frame=Frame("other-root"))


def test_setup_run_matrices_match_si_wrapper_before_result_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check SI wrapper consistency while the phasor result layer is pending."""
    import gragra.results._evaluation as evaluation_module
    from gragra.analytic.seismic import rayleigh_halfspace
    from gragra.observables.rayleigh_literature import (
        rayleigh_halfspace_allspace_observables,
    )

    captured: dict[str, object] = {}
    result_marker = object()

    def capture_result(**kwargs: object) -> object:
        captured.update(kwargs)
        return result_marker

    monkeypatch.setattr(evaluation_module, "EvaluationResult", capture_result)

    kw = _inputs()
    frame = kw["geometry"].frame
    targets = TargetPoints(
        [[0.0, 0.0, 2.0], [1.0, -2.0, -3.0]] * u.m,
        ("above", "below"),
        frame,
    )
    setup = rayleigh_halfspace(**kw).at(targets)
    captured_result = setup.run()

    assert captured_result is result_marker
    assert captured["evaluation"] is setup
    assert captured["record"].g_applications == 1
    matrices = captured["values"]
    assert matrices.potential.shape == (2, 1, 2)
    assert matrices.acceleration.shape == (2, 3, 2)
    assert matrices.gradient.shape == (2, 9, 2)

    spec = setup.resolved.model_spec
    expected = rayleigh_halfspace_allspace_observables(
        setup.resolved.target_positions_m,
        density_kg_m3=spec.density_kg_m3,
        v_p_m_s=spec.vp_m_s,
        v_s_m_s=spec.vs_m_s,
        frequency_hz=np.asarray(spec.frequencies_hz),
        vertical_surface_amplitude_m=spec.vertical_displacement_m,
        direction=spec.direction,
        phase_origin_m=spec.phase_origin_m,
    )
    np.testing.assert_array_equal(
        matrices.potential.value[:, 0, :], expected.potential.T
    )
    np.testing.assert_array_equal(
        matrices.acceleration.value, expected.acceleration.transpose(1, 2, 0)
    )
    np.testing.assert_array_equal(
        matrices.gradient.value,
        expected.gradient.reshape(2, 2, 9).transpose(1, 2, 0),
    )
