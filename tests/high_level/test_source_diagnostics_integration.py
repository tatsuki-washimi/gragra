"""Public source-diagnostic APIs connected across dynamic and static inputs."""

from __future__ import annotations

import numpy as np
from astropy import units as u
from gwexpy import fields
from gwexpy.frequencyseries import FrequencySeriesMatrix

from gragra.fields.density_series import DensityPhasors
from gragra.geometry import Frame
from gragra.gravity_sources import WeightedPointSource
from gragra.gravity_sources._models import point_sum
from gragra.plotting import (
    ContributionStatistics,
    RadialHistogramResult,
    SourceContributionResult,
    SourceFieldResult,
    contribution_statistics,
    plot_hist,
    radial_histogram,
    source_contributions,
    source_field,
    source_field_result,
)
from gragra.results import Contribution, GravityFieldDefinition, PotentialReference
from gragra.targets import TargetPoints


def test_public_dynamic_tensor_contributions_flow_into_source_field() -> None:
    positions = np.array([[-0.6, -0.3, 0.2], [0.5, -0.6, 0.1]], dtype=np.float64)
    metadata = {
        "flags": np.array([True, False], dtype=np.bool_),
        "labels": np.array(["west", "east"], dtype="U5"),
        "large_ids": np.array([2**53 + 1, 2**53 + 3], dtype=np.int64),
    }
    phasors = DensityPhasors(
        positions,
        np.array([1.5, 2.0]),
        np.array(
            [
                [1.0 + 0.5j, 0.25 - 0.75j],
                [-0.5 + 1.25j, 1.5 + 0.25j],
                [0.75 - 0.25j, -0.5 + 0.3j],
            ],
            dtype=np.complex128,
        ),
        np.array([10.0, 20.0, 40.0]),
        0.25,
        np.array([0.25, -0.5, 1.0]),
        "integration-lab",
        np.array([3.0, -2.0, 1.0]),
        ("west", "east"),
        metadata,
    )
    targets = np.array([[0.0, 0.0, 2.5], [0.2, 0.4, 3.0]])

    result = source_contributions(
        phasors,
        "GG",
        targets,
        softening_m=0.05,
        chunk_size=1,
        group_labels=np.array([7, 3]),
    )

    assert isinstance(result, SourceContributionResult)
    assert result.observable == "gravity_gradient"
    assert result.component_labels == (
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
    assert result.group_ids == (3, 7)
    assert result.target_ids == ("target-0", "target-1")
    assert result.source_ids == ("west", "east")

    source_values = result.to_array(kind="source", layout="sample_first")
    group_values = result.to_array(kind="group", layout="sample_first")
    total_values = result.to_array(kind="total", layout="sample_first")
    assert source_values.shape == (3, 2, 2, 3, 3)
    np.testing.assert_allclose(source_values.sum(axis=2), total_values)
    np.testing.assert_allclose(group_values.sum(axis=2), total_values)

    matrix = result.total
    assert isinstance(matrix, FrequencySeriesMatrix)
    assert tuple(matrix.rows.keys()) == result.target_ids
    assert tuple(matrix.cols.keys()) == result.component_labels
    np.testing.assert_array_equal(matrix.frequencies.to_value(u.Hz), [10.0, 20.0, 40.0])
    np.testing.assert_allclose(
        np.asarray(matrix), np.moveaxis(total_values.reshape(3, 2, 9), 0, -1)
    )
    assert result.source("west").shape == (2, 9, 3)

    statistics = contribution_statistics(result, frequency_weights=[1.0, 2.0, 3.0])
    assert isinstance(statistics, ContributionStatistics)
    assert statistics.rms is not None
    assert statistics.rms.group_ids == result.group_ids
    phase_valid = statistics.phase_rad.validity(3).to_array(layout="sample_first")
    assert phase_valid.shape == (3, 2, 9)
    interference_valid = statistics.interference_share.validity(7).to_array(
        layout="sample_first"
    )
    assert interference_valid.shape == (3, 2, 9)

    edges = (
        np.array([-1.0, 0.0, 1.0]) * u.m,
        np.array([-1.0, 0.0, 1.0]) * u.m,
        np.array([0.0, 0.5]) * u.m,
    )
    field_result = source_field_result(
        result,
        "complex_contribution",
        target_id="target-1",
        grid=edges,
        aggregation="sum",
        frequency_hz=40.0 * u.Hz,
    )
    assert isinstance(field_result, SourceFieldResult)
    assert isinstance(field_result.field, fields.TensorField)
    field_values = field_result.field.to_array()
    assert field_values.shape == (1, 2, 2, 1, 3, 3)
    assert field_result.valid.shape == field_values.shape
    assert field_result.valid.all()
    np.testing.assert_allclose(field_values[0].sum(axis=(0, 1, 2)), total_values[2, 1])
    assert np.any(np.abs(total_values[2, 1][np.triu_indices(3, 1)]) > 0.0)
    frequency_axis = field_result.field[(0, 0)].axis("frequency")
    np.testing.assert_array_equal(frequency_axis.index.to_value(u.Hz), [40.0])
    assert field_result.metadata["frequency_hz"] == 40.0
    assert field_result.metadata["target_id"] == "target-1"
    assert field_result.metadata["t_ref_s"] == 0.25
    assert field_result.metadata["phase_origin_m"] == (0.25, -0.5, 1.0)

    raw_field = source_field(
        result,
        "complex_contribution",
        target_id="target-1",
        grid=edges,
        aggregation="sum",
        frequency_hz=40.0 * u.Hz,
    )
    np.testing.assert_array_equal(raw_field.to_array(), field_values)

    assert result.global_metadata["flags"].dtype == np.dtype(np.bool_)
    np.testing.assert_array_equal(result.global_metadata["flags"], [True, False])
    assert result.global_metadata["labels"].dtype.kind == "U"
    np.testing.assert_array_equal(result.global_metadata["labels"], ["west", "east"])
    assert result.global_metadata["large_ids"].dtype == np.dtype(np.int64)
    assert result.global_metadata["large_ids"][0] == 2**53 + 1


def test_public_radial_result_and_source_field_share_static_mass_domain() -> None:
    frame = Frame("radial-integration")
    definition = GravityFieldDefinition.source_field(
        "radial integration",
        Contribution("total"),
        PotentialReference.ZERO_AT_INFINITY,
    )
    model = point_sum(
        WeightedPointSource([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [2.0, 8.0]),
        frame=frame,
        field_definition=definition,
    )
    setup = model.at(TargetPoints(np.zeros((1, 3)) * u.m, ("origin",), frame))
    grid = (
        np.array([0.5, 1.5, 2.5]) * u.m,
        np.array([-0.5, 0.5]) * u.m,
        np.array([-0.5, 0.5]) * u.m,
    )

    field_result = source_field_result(setup, "mass", grid=grid, aggregation="sum")
    assert isinstance(field_result, SourceFieldResult)
    assert isinstance(field_result.field, fields.ScalarField)
    field_values = np.asarray(field_result.field)[0, :, 0, 0]

    radial = radial_histogram(
        setup,
        "mass",
        target_id="origin",
        bins=[0.5, 1.5, 2.5],
        normalization="fraction_per_length",
    )
    assert isinstance(radial, RadialHistogramResult)
    assert radial.histogram.values.unit == u.kg
    np.testing.assert_array_equal(radial.histogram.values.value, field_values)
    np.testing.assert_allclose(radial.values.to_value(1 / u.m), [0.2, 0.8])
    assert radial.unit == 1 / u.m

    figure, axes, returned = plot_hist(radial)
    stairs = axes.patches[0].get_data()
    assert returned is radial
    np.testing.assert_allclose(stairs.values, radial.values.value)
    np.testing.assert_array_equal(stairs.edges, [0.5, 1.5, 2.5])
    assert "fraction per length" in axes.get_ylabel()
    assert str(radial.unit) in axes.get_ylabel()
    import matplotlib.pyplot as plt

    plt.close(figure)
