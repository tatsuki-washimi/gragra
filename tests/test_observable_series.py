"""Stage-5 (5-A) tests: series-type observable dispatch matches direct calls."""

from __future__ import annotations

import numpy as np
import pytest

from gragra.fields.density_series import DensityPhasors, DensityTimeSeries
from gragra.fields.displacement_series import FixedMeshDisplacementSeries
from gragra.fields.surface_density_series import FixedMeshSurfaceDensitySeries
from gragra.gravity_sources import WeightedPointSource
from gragra.observables.acceleration import point_acceleration
from gragra.observables.dipole import (
    displacement_transfer_acceleration,
    displacement_transfer_gravity_gradient,
    displacement_transfer_potential,
)
from gragra.observables.gradient import point_gravity_gradient
from gragra.observables.potential import point_potential
from gragra.observables.series import (
    density_phasors_observable,
    density_time_series_observable,
    fixed_mesh_displacement_series_observable,
    surface_density_series_observable,
)

_POSITIONS = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
_TARGETS = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
_DIRECT_FN = {
    "potential": point_potential,
    "acceleration": point_acceleration,
    "gradient": point_gravity_gradient,
}
_TRANSFER_FN = {
    "potential": displacement_transfer_potential,
    "acceleration": displacement_transfer_acceleration,
    "gradient": displacement_transfer_gravity_gradient,
}


def _density_time_series(density_kg_m3, *, time_s):
    return DensityTimeSeries(
        _POSITIONS,
        np.array([1.5, 2.5]),
        density_kg_m3,
        time_s,
        "rho",
        "geocentric",
        np.zeros(3),
        ("a", "b"),
    )


@pytest.mark.parametrize("observable", ["potential", "acceleration", "gradient"])
def test_density_time_series_observable_matches_direct_weighted_point_source(
    observable,
):
    density_kg_m3 = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    series = _density_time_series(density_kg_m3, time_s=np.array([0.0, 1.0, 2.0]))

    result = density_time_series_observable(series, observable, _TARGETS)

    expected = np.stack(
        [
            _DIRECT_FN[observable](
                WeightedPointSource(_POSITIONS, density_kg_m3[s] * series.volumes_m3),
                _TARGETS,
            )
            for s in range(3)
        ],
        axis=0,
    )
    np.testing.assert_array_equal(result, expected)
    assert result.flags.writeable is False


def test_density_time_series_observable_rejects_unknown_observable():
    series = _density_time_series(np.array([[1.0, 1.0]]), time_s=np.array([0.0]))
    with pytest.raises(ValueError, match="observable"):
        density_time_series_observable(series, "bogus", _TARGETS)


@pytest.mark.parametrize("observable", ["potential", "acceleration", "gradient"])
def test_density_phasors_observable_matches_direct_weighted_point_source(observable):
    density_kg_m3 = np.array(
        [[2.0 + 1.0j, 3.0 - 0.5j], [4.0j, 5.0 + 2.0j]], dtype=np.complex128
    )
    series = DensityPhasors(
        _POSITIONS,
        np.array([1.5, 2.5]),
        density_kg_m3,
        np.array([1.0, 2.0]),
        0.0,
        np.zeros(3),
        "geocentric",
        np.zeros(3),
        ("a", "b"),
    )

    result = density_phasors_observable(series, observable, _TARGETS)

    expected = np.stack(
        [
            _DIRECT_FN[observable](
                WeightedPointSource(_POSITIONS, density_kg_m3[s] * series.volumes_m3),
                _TARGETS,
            )
            for s in range(2)
        ],
        axis=0,
    )
    np.testing.assert_array_equal(result, expected)
    assert np.iscomplexobj(result)


@pytest.mark.parametrize("observable", ["potential", "acceleration", "gradient"])
def test_surface_density_series_observable_matches_direct_weighted_point_source(
    observable,
):
    surface_density_kg_m2 = np.array([[10.0, 20.0], [30.0, 40.0]])
    series = FixedMeshSurfaceDensitySeries(
        _POSITIONS,
        np.array([0.5, 0.75]),
        np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        surface_density_kg_m2,
        np.array([0.0, 1.0]),
        "time",
        {},
        "geocentric",
        np.zeros(3),
        ("a", "b"),
    )

    result = surface_density_series_observable(series, observable, _TARGETS)

    expected = np.stack(
        [
            _DIRECT_FN[observable](
                WeightedPointSource(
                    _POSITIONS, series.areas_m2 * surface_density_kg_m2[s]
                ),
                _TARGETS,
            )
            for s in range(2)
        ],
        axis=0,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("observable", ["potential", "acceleration", "gradient"])
def test_fixed_mesh_displacement_series_observable_matches_direct_transfer(
    observable,
):
    displacement_m = np.array(
        [[[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]], [[0.0, 0.0, 0.3], [0.1, 0.1, 0.0]]]
    )
    series = FixedMeshDisplacementSeries(
        _POSITIONS,
        np.array([1000.0, 2000.0]),
        displacement_m,
        np.array([0.0, 1.0]),
        "time",
        "geocentric",
        np.zeros(3),
        ("a", "b"),
    )

    result = fixed_mesh_displacement_series_observable(series, observable, _TARGETS)

    expected = np.stack(
        [
            _TRANSFER_FN[observable](
                series.positions_m, series.mass_kg, displacement_m[s], _TARGETS
            )
            for s in range(2)
        ],
        axis=0,
    )
    np.testing.assert_array_equal(result, expected)


def test_fixed_mesh_displacement_series_observable_rejects_unknown_observable():
    series = FixedMeshDisplacementSeries(
        _POSITIONS,
        np.array([1000.0, 2000.0]),
        np.zeros((1, 2, 3)),
        np.array([0.0]),
        "time",
        "geocentric",
        np.zeros(3),
        ("a", "b"),
    )
    with pytest.raises(ValueError, match="observable"):
        fixed_mesh_displacement_series_observable(series, "bogus", _TARGETS)
