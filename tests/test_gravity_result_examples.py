"""Integration tests for the executable GravityResult examples."""

from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeriesMatrix
from gwexpy.timeseries import TimeSeriesMatrix


@pytest.fixture
def examples():
    spec = importlib.util.find_spec("examples.gravity_result")
    if spec is None:
        return None
    return importlib.import_module("examples.gravity_result")


def test_uniform_sphere_example_wraps_existing_si_observables(examples):
    assert examples is not None, "the GravityResult example module is missing"
    from gragra.observables.analytic import uniform_sphere_observables

    result = examples.uniform_sphere_result_example()
    expected = uniform_sphere_observables(**examples.uniform_sphere_case())

    assert result.name == "uniform sphere SI observables"
    assert result.potential.unit == u.m**2 / u.s**2
    assert result.acceleration.unit == u.m / u.s**2
    assert result.gradient.unit == u.s**-2
    np.testing.assert_array_equal(result.potential.value, expected.potential)
    np.testing.assert_array_equal(result.acceleration.value, expected.acceleration)
    np.testing.assert_array_equal(result.gradient.value, expected.gradient)
    assert (
        len({id(result.potential), id(result.acceleration), id(result.gradient)}) == 3
    )


def test_u11_example_wraps_the_three_existing_si_parts(examples):
    assert examples is not None, "the GravityResult example module is missing"
    from gragra.observables.u11_rayleigh_small_cavity import (
        rayleigh_small_cavity_acceleration,
    )

    targets, kwargs = examples.u11_case()
    expected = rayleigh_small_cavity_acceleration(targets, **kwargs)
    results = examples.u11_result_examples()

    assert set(results) == {
        "background_halfspace",
        "cavity_wall_sca",
        "total_leading_sca",
    }
    assert all(
        result.potential is None and result.gradient is None
        for result in results.values()
    )
    for name, result in results.items():
        assert result.name == name
        assert result.acceleration is not None
        np.testing.assert_array_equal(result.acceleration, expected[name])


@pytest.mark.parametrize(
    ("factory", "expected_class", "axis_name", "expected_axis"),
    [
        (
            "time_series_matrix_payload_example",
            TimeSeriesMatrix,
            "times",
            [0.0, 0.25, 0.5],
        ),
        (
            "frequency_series_matrix_payload_example",
            FrequencySeriesMatrix,
            "frequencies",
            [1.0, 2.0, 3.0],
        ),
    ],
)
def test_native_matrix_metadata_is_checked_before_and_retained_after_wrapping(
    examples, factory, expected_class, axis_name, expected_axis
):
    assert examples is not None, "the GravityResult example module is missing"
    payload = getattr(examples, factory)()
    assert isinstance(payload, expected_class)
    assert payload.shape == (2, 3, 3)
    assert list(payload.rows.keys()) == ["target_a", "target_b"]
    assert list(payload.cols.keys()) == ["x", "y", "z"]
    np.testing.assert_array_equal(
        getattr(payload, axis_name).to_value(u.s if axis_name == "times" else u.Hz),
        expected_axis,
    )

    metadata_snapshot = []
    cell_names = set()
    cell_channels = set()
    for row in range(2):
        for col in range(3):
            metadata = payload.meta[row, col]
            assert metadata.unit == u.m / u.s**2
            assert metadata.name
            assert metadata.channel
            cell_names.add(metadata.name)
            cell_channels.add(str(metadata.channel))
            metadata_snapshot.append(
                (metadata.unit, metadata.name, str(metadata.channel))
            )
    assert len(cell_names) == 6
    assert len(cell_channels) == 6

    container_name = f"GravityResult container for {axis_name}"
    result = examples.wrap_native_matrix_result(payload, name=container_name)

    assert result.acceleration is payload
    assert result.name == container_name
    assert result.name not in cell_names
    result_metadata_snapshot = []
    for row in range(2):
        for col in range(3):
            result_metadata = result.acceleration.meta[row, col]
            result_metadata_snapshot.append(
                (
                    result_metadata.unit,
                    result_metadata.name,
                    str(result_metadata.channel),
                )
            )
    assert result_metadata_snapshot == metadata_snapshot
    np.testing.assert_array_equal(
        getattr(result.acceleration, axis_name).to_value(
            u.s if axis_name == "times" else u.Hz
        ),
        expected_axis,
    )
