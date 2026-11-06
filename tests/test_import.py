"""Test that the package imports correctly and exposes the research API."""

import builtins
import importlib

import gragra
import gragra.coherent as coherent
import gragra.waves as waves

_EXPECTED_PUBLIC_SYMBOLS = [
    "G_SI",
    "WeightedPointSource",
    "PointMass",
    "PointMassCloud",
    "TargetPoints",
    "point_potential",
    "point_acceleration",
    "point_gravity_gradient",
    "displacement_field_acceleration",
    "coherent_point_acceleration",
    "coherent_point_gravity_gradient",
    "coherent_gravitoelastic_tensor",
    "coherent_acceleration_csd",
    "displacement_transfer_acceleration",
]

_EXPECTED_PLOTTING_SYMBOLS = {
    "formula_to_latex",
    "hist",
    "plot_contribution_hist",
    "plot_cumulative_contribution",
    "plot_field",
    "plot_hist",
    "plot_map",
    "plot_result",
    "plot_shape",
    "radial_histogram",
    "gravity_gradient_radial_histogram",
    "dynamic_radial_histogram",
    "RadialHistogramResult",
    "GravityGradientRadialHistogramResult",
    "DynamicRadialHistogramResult",
    "source_field",
    "source_field_result",
    "SourceFieldResult",
    "source_contributions",
    "SourceContributionResult",
    "contribution_statistics",
    "directional_contribution_statistics",
    "ContributionValidityMask",
    "ContributionFrequencyStatistic",
    "ContributionRMSStatistics",
    "ContributionStatistics",
    "DirectionalContributionStatistics",
}


def test_import_g_si():
    """Verify that G_SI is exported and has the correct value."""
    assert hasattr(gragra, "G_SI")
    assert gragra.G_SI == 6.67430e-11


def test_public_api_symbols_exist():
    """The top-level API exposes the implemented research entry points."""
    for name in _EXPECTED_PUBLIC_SYMBOLS:
        assert hasattr(gragra, name), f"gragra.{name} is missing from the API"


def test_all_contains_public_symbols():
    """__all__ declares the implemented top-level contract."""
    assert set(gragra.__all__) == set(_EXPECTED_PUBLIC_SYMBOLS)


def test_water_module_import_does_not_require_scipy_special(monkeypatch):
    import gragra.analytic.water as water

    original_import = builtins.__import__

    def import_without_scipy_special(name, *args, **kwargs):
        if name == "scipy.special":
            raise ModuleNotFoundError("No module named 'scipy.special'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_scipy_special)
    importlib.reload(water)


def test_c5_names_are_subpackage_exports_and_target_points_is_lazy():
    """C5 types and factories stay at their subpackage boundaries."""
    from gragra.analytic.objects import uniform_sphere
    from gragra.geometry import Frame, Sphere
    from gragra.gravity_sources import discretize, point_sum
    from gragra.results import EvaluationResult, compare
    from gragra.targets import TargetPoints

    assert gragra.TargetPoints is TargetPoints
    assert Frame.__module__ == "gragra.geometry._frame"
    assert Sphere.__module__ == "gragra.geometry._sphere"
    assert TargetPoints.__module__ == "gragra.targets._high_points"
    assert callable(uniform_sphere)
    assert callable(discretize)
    assert callable(point_sum)
    assert EvaluationResult.__module__ == "gragra.results._evaluation"
    assert callable(compare)

    assert not hasattr(gragra, "Frame")
    assert not hasattr(gragra, "compare")
    assert not hasattr(gragra, "GravityResult")
    for name in (
        "point_projected_acceleration",
        "project_vectors",
        "coherent_projected_acceleration",
    ):
        assert not hasattr(gragra, name)


def test_rayleigh_factory_and_halfspace_stay_at_their_subpackage_boundaries():
    from gragra.analytic import seismic
    from gragra.geometry import HalfSpace, Sphere

    assert HalfSpace.__module__ == "gragra.geometry._halfspace"
    assert Sphere.__module__ == "gragra.geometry._sphere"
    assert seismic.__all__ == [
        "halfspace_psv",
        "rayleigh_halfspace",
        "rayleigh_response",
        "spherical_cavity",
        "spherical_shell",
        "small_cavity_wall",
        "point_force",
    ]
    assert callable(seismic.halfspace_psv)
    assert callable(seismic.spherical_cavity)
    assert callable(seismic.spherical_shell)
    assert callable(seismic.rayleigh_halfspace)
    assert callable(seismic.rayleigh_response)
    assert callable(seismic.small_cavity_wall)
    assert callable(seismic.point_force)
    assert not hasattr(gragra, "halfspace_psv")
    assert not hasattr(gragra, "spherical_cavity")
    assert not hasattr(gragra, "spherical_shell")
    assert not hasattr(gragra, "rayleigh_halfspace")
    assert not hasattr(gragra, "rayleigh_response")
    assert not hasattr(gragra.analytic, "rayleigh_halfspace")
    assert not hasattr(seismic, "sca_wall")
    assert not hasattr(gragra, "small_cavity_wall")
    assert not hasattr(gragra, "point_force")


def test_gen1_atmospheric_factories_stay_at_their_subpackage_boundary():
    from gragra.analytic import atmospheric

    assert set(atmospheric.__all__) == {
        "acoustic_halfspace",
        "isobaric_temperature",
        "acoustic_cavity_mode",
        "SphereAcousticMode",
        "CuboidAcousticMode",
        "CylinderAcousticMode",
        "AcousticCavityNumerics",
    }
    assert callable(atmospheric.acoustic_halfspace)
    assert callable(atmospheric.isobaric_temperature)
    assert callable(atmospheric.acoustic_cavity_mode)
    assert not hasattr(gragra, "acoustic")
    assert not hasattr(gragra, "temperature")


def test_gen1_motion_factories_stay_at_their_subpackage_boundary():
    from gragra.analytic import objects

    assert {
        "constant_velocity",
        "translational_oscillation",
        "fixed_axis_rotation",
    } <= set(objects.__all__)
    assert callable(objects.constant_velocity)
    assert callable(objects.translational_oscillation)
    assert callable(objects.fixed_axis_rotation)
    assert not hasattr(gragra, "constant_speed")
    assert not hasattr(gragra, "oscillation")
    assert not hasattr(gragra, "rotation")


def test_gravity_response_types_stay_in_results_subpackage():
    from gragra.results import GravityResponse, ResponseContract

    assert GravityResponse.__module__ == "gragra.results._response"
    assert ResponseContract.__module__ == "gragra.results._response"
    assert not hasattr(gragra, "GravityResponse")
    assert not hasattr(gragra, "ResponseContract")


def test_gen1_gravity_source_helpers_are_subpackage_exports():
    import gragra.gravity_sources as gravity_sources
    from gragra.gravity_sources import ForceHistory, smooth_pulse, smooth_ramp_to_hold

    assert ForceHistory.__module__ == "gragra.gravity_sources._force_history"
    assert callable(smooth_pulse)
    assert callable(smooth_ramp_to_hold)
    assert {
        "ForceHistory",
        "smooth_pulse",
        "smooth_ramp_to_hold",
    } <= set(gravity_sources.__all__)
    assert not hasattr(gragra, "ForceHistory")
    assert not hasattr(gragra, "smooth_pulse")
    assert not hasattr(gragra, "smooth_ramp_to_hold")


def test_gen1_result_time_and_diagnostic_types_are_subpackage_exports():
    from gragra.results import (
        NumericalDiagnostic,
        TimeSampling,
        TimeSeriesRepresentation,
    )

    assert TimeSampling.__module__ == "gragra.results._time"
    assert TimeSeriesRepresentation.__module__ == "gragra.results._representation"
    assert NumericalDiagnostic.__module__ == "gragra.exceptions"
    assert not hasattr(gragra, "TimeSampling")
    assert not hasattr(gragra, "TimeSeriesRepresentation")
    assert not hasattr(gragra, "NumericalDiagnostic")


def test_gen1_temperature_profiles_are_fields_subpackage_exports():
    from gragra.fields import (
        TemperatureFourierMode,
        TemperatureGaussian,
        TemperaturePerturbationGrid,
    )

    assert TemperatureFourierMode.__module__ == "gragra.fields._temperature"
    assert TemperatureFourierMode.__name__ == "TemperatureFourierMode"
    assert TemperatureGaussian.__module__ == "gragra.fields._temperature"
    assert TemperaturePerturbationGrid.__module__ == "gragra.fields._temperature"
    from astropy import units as u

    from gragra.geometry import Frame

    profile = TemperatureFourierMode(
        wavevector=[1, 0, 0] * u.rad / u.m,
        phase_origin=[0, 0, 0] * u.m,
        frame=Frame("temperature profile"),
    )
    assert repr(profile).startswith("TemperatureFourierMode(")
    assert not hasattr(gragra, "TemperaturePlaneWave")
    assert not hasattr(gragra, "TemperatureGaussian")
    assert not hasattr(gragra, "DeltaTemperatureGrid")


def test_gen1_acoustic_cavity_api_is_atmospheric_subpackage_only():
    import gragra.analytic.atmospheric as atmospheric
    from gragra.analytic.atmospheric import (
        AcousticCavityNumerics,
        CuboidAcousticMode,
        CylinderAcousticMode,
        SphereAcousticMode,
        acoustic_cavity_mode,
    )

    cavity_names = {
        "acoustic_cavity_mode",
        "SphereAcousticMode",
        "CuboidAcousticMode",
        "CylinderAcousticMode",
        "AcousticCavityNumerics",
    }
    assert cavity_names <= set(atmospheric.__all__)
    assert callable(acoustic_cavity_mode)
    assert all(
        value.__module__ == "gragra.analytic.atmospheric._cavity._model"
        for value in (
            SphereAcousticMode,
            CuboidAcousticMode,
            CylinderAcousticMode,
            AcousticCavityNumerics,
        )
    )
    for name in cavity_names:
        assert not hasattr(gragra, name)
        assert name not in gragra.__all__


def test_gen1_public_api_uses_physical_scope_names():
    import gragra.fields as fields
    from gragra.analytic import atmospheric, objects, seismic

    assert {"small_cavity_wall", "point_force"} <= set(seismic.__all__)
    assert {
        "acoustic_halfspace",
        "isobaric_temperature",
        "acoustic_cavity_mode",
        "SphereAcousticMode",
        "CuboidAcousticMode",
        "CylinderAcousticMode",
        "AcousticCavityNumerics",
    } == set(atmospheric.__all__)
    assert {
        "constant_velocity",
        "translational_oscillation",
        "fixed_axis_rotation",
    } <= set(objects.__all__)
    assert {
        "TemperatureFourierMode",
        "TemperatureGaussian",
        "TemperaturePerturbationGrid",
    } <= set(fields.__all__)

    assert not hasattr(seismic, "sca_wall")
    for name in ("acoustic", "temperature", "acoustic_cavity", "CavityNumerics"):
        assert not hasattr(atmospheric, name)
    for name in ("constant_speed", "oscillation", "rotation"):
        assert not hasattr(objects, name)
    assert not hasattr(fields, "TemperaturePlaneWave")
    assert not hasattr(fields, "DeltaTemperatureGrid")


def test_phase15b_symbols_are_subpackage_exports():
    wave_symbols = {"StochasticPSParameters", "stochastic_ps_realization"}
    coherent_symbols = {"ensemble_outer_cross_average"}

    assert wave_symbols <= set(waves.__all__)
    assert coherent_symbols <= set(coherent.__all__)
    for name in wave_symbols:
        assert hasattr(waves, name)
    for name in coherent_symbols:
        assert hasattr(coherent, name)


def test_phase15b_symbols_are_not_promoted_to_top_level():
    for name in (
        "StochasticPSParameters",
        "stochastic_ps_realization",
        "ensemble_outer_cross_average",
    ):
        assert not hasattr(gragra, name)
        assert name not in gragra.__all__


def test_plotting_symbols_are_subpackage_exports():
    import gragra.plotting as plotting

    expected = _EXPECTED_PLOTTING_SYMBOLS
    assert set(plotting.__all__) == expected
    for name in expected:
        assert hasattr(plotting, name)
        assert not hasattr(gragra, name)
        assert name not in gragra.__all__
