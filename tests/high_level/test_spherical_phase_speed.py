"""Contracts for phase-speed inputs shared by spherical cavity and shell models."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeries

from gragra.analytic.seismic._phase_speed import (
    _spherical_phase_speed_to_wavenumbers,
)


@pytest.mark.parametrize("mode", ["P", "S"])
def test_scalar_speed_is_constant_over_positive_frequency_axis(mode):
    frequencies = np.array([1000.0, 1500.0, 2000.0]) * u.mHz

    wavenumbers = _spherical_phase_speed_to_wavenumbers(
        2.0 * u.km / u.s, frequencies, mode=mode
    )

    expected = 2.0 * np.pi * frequencies.to_value(u.Hz) / 2000.0
    assert wavenumbers.unit == u.rad / u.m
    np.testing.assert_allclose(wavenumbers.value, expected)
    assert wavenumbers.shape == (3,)
    assert wavenumbers.base.flags.owndata
    assert not wavenumbers.base.flags.writeable
    assert not wavenumbers.flags.writeable
    assert not wavenumbers.value.flags.writeable


def test_frequency_series_uses_exact_normalized_coordinates_and_speed_units():
    phase_speed = FrequencySeries(
        [1.0, 2.0], f0=1.0, df=1.0, unit=u.km / u.s, name="phase_speed"
    )
    frequencies = np.array([1000.0, 2000.0]) * u.mHz

    wavenumbers = _spherical_phase_speed_to_wavenumbers(
        phase_speed, frequencies, mode="P"
    )

    np.testing.assert_allclose(
        wavenumbers.to_value(u.rad / u.m), np.full(2, 2 * np.pi / 1000)
    )
    assert wavenumbers.base.flags.owndata
    assert not wavenumbers.base.flags.writeable
    assert not wavenumbers.flags.writeable
    assert not wavenumbers.value.flags.writeable


def test_constant_frequency_series_is_equivalent_to_scalar_speed():
    frequencies = np.array([1000.0, 1500.0, 2000.0]) * u.mHz
    phase_speed = FrequencySeries(
        [2.0, 2.0, 2.0], f0=1.0, df=0.5, unit=u.km / u.s, name="phase_speed"
    )

    from_series = _spherical_phase_speed_to_wavenumbers(
        phase_speed, frequencies, mode="S"
    )
    from_scalar = _spherical_phase_speed_to_wavenumbers(
        2.0 * u.km / u.s, frequencies, mode="S"
    )

    np.testing.assert_array_equal(from_series.value, from_scalar.value)


@pytest.mark.parametrize(
    "frequencies",
    [
        np.array([], dtype=float) * u.Hz,
        np.array([0.0, 1.0]) * u.Hz,
        np.array([-1.0, 1.0]) * u.Hz,
        np.array([1.0, np.nan]) * u.Hz,
        np.array([2.0, 1.0]) * u.Hz,
        np.array([1.0, 1.0]) * u.Hz,
    ],
)
def test_frequency_axis_must_be_finite_positive_and_strictly_increasing(frequencies):
    with pytest.raises(ValueError):
        _spherical_phase_speed_to_wavenumbers(1000 * u.m / u.s, frequencies, mode="P")


@pytest.mark.parametrize(
    "phase_speed",
    [
        0 * u.m / u.s,
        -1 * u.m / u.s,
        np.nan * u.m / u.s,
        np.inf * u.m / u.s,
        1000 * u.Hz,
        np.array([1000.0, 1200.0]) * u.m / u.s,
        (1000 + 0j) * u.m / u.s,
    ],
)
def test_scalar_phase_speed_must_be_real_finite_positive_velocity_scalar(phase_speed):
    with pytest.raises((TypeError, ValueError, u.UnitConversionError)):
        _spherical_phase_speed_to_wavenumbers(
            phase_speed, np.array([1.0, 2.0]) * u.Hz, mode="P"
        )


@pytest.mark.parametrize(
    "values",
    [
        [1000.0, 0.0],
        [1000.0, -1.0],
        [1000.0, np.inf],
        [1000.0, np.nan],
        [1000.0 + 0.0j, 1200.0 + 0.0j],
    ],
)
def test_frequency_series_speed_values_must_be_finite_and_positive(values):
    phase_speed = FrequencySeries(
        values, f0=1.0, df=1.0, unit=u.m / u.s, name="phase_speed"
    )

    with pytest.raises((TypeError, ValueError)):
        _spherical_phase_speed_to_wavenumbers(
            phase_speed, np.array([1.0, 2.0]) * u.Hz, mode="S"
        )


def test_frequency_series_requires_exact_coordinate_match_without_resampling():
    phase_speed = FrequencySeries(
        [1000.0, 2000.0], f0=1.0, df=1.0, unit=u.m / u.s, name="phase_speed"
    )

    with pytest.raises(ValueError, match="exactly match"):
        _spherical_phase_speed_to_wavenumbers(
            phase_speed, np.array([1.0, 2.0001]) * u.Hz, mode="P"
        )


@pytest.mark.parametrize("mode", ["p", "SV", "Rayleigh", None])
def test_mode_must_be_explicitly_p_or_s(mode):
    with pytest.raises(ValueError, match="mode"):
        _spherical_phase_speed_to_wavenumbers(
            1000 * u.m / u.s, np.array([1.0]) * u.Hz, mode=mode
        )


def test_derived_wavenumber_must_remain_finite_and_positive():
    with pytest.raises(ValueError, match="derived"):
        _spherical_phase_speed_to_wavenumbers(
            1e-308 * u.m / u.s, np.array([1e308]) * u.Hz, mode="P"
        )


def test_derived_wavenumber_avoids_overflow_in_frequency_product():
    wavenumber = _spherical_phase_speed_to_wavenumbers(
        1e308 * u.m / u.s, np.array([1e308]) * u.Hz, mode="P"
    )

    np.testing.assert_allclose(wavenumber.to_value(u.rad / u.m), [2 * np.pi])


def test_derived_wavenumber_preserves_representable_result_across_ratio_underflow():
    smallest_positive_frequency = np.array([np.nextafter(0.0, 1.0)]) * u.Hz

    wavenumber = _spherical_phase_speed_to_wavenumbers(
        2.0 * u.m / u.s, smallest_positive_frequency, mode="P"
    )

    assert np.isfinite(wavenumber.value[0])
    assert wavenumber.value[0] > 0


@pytest.mark.parametrize("mode", ["P", "S"])
def test_nonconstant_frequency_series_matches_native_cavity_and_shell_per_bin(mode):
    from gragra.analytic.spherical_cavity_series import (
        spherical_cavity_observables_g_free,
    )
    from gragra.analytic.u10_spherical_shell_general import (
        spherical_shell_observables_g_free,
    )

    frequencies = np.array([2.0, 3.0]) * u.Hz
    phase_speed = FrequencySeries(
        [2.0, 2.5], f0=2.0, df=1.0, unit=u.km / u.s, name="phase_speed"
    )
    wavenumbers = _spherical_phase_speed_to_wavenumbers(
        phase_speed, frequencies, mode=mode
    ).to_value(u.rad / u.m)
    direction = (0.0, 0.0, 1.0)
    polarization = direction if mode == "P" else (1.0, 0.0, 0.0)
    displacement = 1e-6 * (1.0 + 0.4j)

    cavity_targets = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.3], [1.4, 0.2, -0.1]])
    cavity_arguments = {
        "radius_m": 1.0,
        "density_kg_m3": 2700.0,
        "displacement_m": displacement,
        "propagation_direction": direction,
        "polarization": polarization,
        "wave_type": mode,
    }
    cavity_series = spherical_cavity_observables_g_free(
        cavity_targets,
        **(cavity_arguments | {"wavenumber_rad_m": wavenumbers}),
    )
    cavity_per_bin = [
        spherical_cavity_observables_g_free(
            cavity_targets,
            **(cavity_arguments | {"wavenumber_rad_m": float(k)}),
        )
        for k in wavenumbers
    ]
    for part in cavity_series:
        for observable in ("potential", "acceleration", "gradient"):
            np.testing.assert_allclose(
                getattr(cavity_series[part], observable),
                np.stack(
                    [getattr(result[part], observable) for result in cavity_per_bin]
                ),
                rtol=1e-13,
                atol=0.0,
            )

    shell_targets = np.array([[0.0, 0.0, 0.0]])
    shell_arguments = {
        "inner_radius_m": 2.0,
        "outer_radius_m": 10.0,
        "density_kg_m3": 2300.0,
        "displacement_m": displacement,
        "propagation_direction": direction,
        "polarization": polarization,
        "wave_type": mode,
        "return_diagnostics": True,
    }
    shell_series, diagnostics = spherical_shell_observables_g_free(
        shell_targets,
        **(shell_arguments | {"wavenumber_rad_m": wavenumbers}),
    )
    shell_per_bin = [
        spherical_shell_observables_g_free(
            shell_targets,
            **(shell_arguments | {"wavenumber_rad_m": float(k)}),
        )
        for k in wavenumbers
    ]
    assert diagnostics["status"] == "CONVERGED"
    assert all(query["status"] == "CONVERGED" for query in diagnostics["queries"])
    for part in shell_series:
        for observable in ("potential", "acceleration", "gradient"):
            np.testing.assert_allclose(
                getattr(shell_series[part], observable),
                np.stack(
                    [getattr(result[0][part], observable) for result in shell_per_bin]
                ),
                rtol=1e-13,
                atol=0.0,
            )


def test_shell_later_frequency_failure_raises_without_returning_partial_result():
    from gragra.analytic.u10_spherical_shell_general import (
        ShellConvergenceError,
        spherical_shell_observables_g_free,
    )

    frequencies = np.array([1.0, 3.0]) * u.Hz
    target_wavenumbers = np.array([0.04, 0.08])
    phase_speeds = FrequencySeries(
        2.0 * np.pi * frequencies.to_value(u.Hz) / target_wavenumbers,
        f0=1.0,
        df=2.0,
        unit=u.m / u.s,
        name="phase_speed",
    )
    wavenumbers = _spherical_phase_speed_to_wavenumbers(
        phase_speeds, frequencies, mode="P"
    ).to_value(u.rad / u.m)
    np.testing.assert_allclose(wavenumbers, target_wavenumbers, rtol=1e-15)

    result = None
    with pytest.raises(ShellConvergenceError) as caught:
        result = spherical_shell_observables_g_free(
            [[0.0, 0.0, 0.0]],
            inner_radius_m=10.0,
            outer_radius_m=400.0,
            density_kg_m3=2000.0,
            wavenumber_rad_m=wavenumbers,
            displacement_m=1e-6 * (1.0 + 0.5j),
            propagation_direction=(0.0, 0.0, 1.0),
            polarization=(0.0, 0.0, 1.0),
            wave_type="P",
            max_degree=128,
            return_diagnostics=True,
        )

    assert result is None
    diagnostics = caught.value.diagnostics
    assert diagnostics["failed_index"] == [1, 0]
    assert [query["status"] for query in diagnostics["queries"]] == [
        "CONVERGED",
        "NOT_CONVERGED",
    ]
