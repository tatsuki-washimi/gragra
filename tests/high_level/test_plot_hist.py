"""Artist-level contract tests for ``gragra.plotting.plot_hist``."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u
from matplotlib.figure import Figure

from gragra.geometry import Frame
from gragra.gravity_sources import VolumeElementSource, WeightedPointSource
from gragra.gravity_sources._models import point_sum
from gragra.results import Contribution, GravityFieldDefinition, PotentialReference
from gragra.targets import TargetPoints

MAX_PLOT_BINS = 100_000

ROOT = Frame("root")
SOURCES_3 = [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 12.0]]


def _api_plot_hist(setup, quantity, **kwargs):
    from gragra.plotting import plot_hist

    return plot_hist(setup, quantity, **kwargs)


def _api_hist(setup, quantity, **kwargs):
    from gragra.plotting import hist

    return hist(setup, quantity, **kwargs)


def _field(name: str) -> GravityFieldDefinition:
    return GravityFieldDefinition.source_field(
        name, Contribution("total"), PotentialReference.ZERO_AT_INFINITY
    )


def _targets() -> TargetPoints:
    # Target A is at (-3, -4, 0). Distances to SOURCES_3 are 5, 10 and 13.
    return TargetPoints(np.array([[-3.0, -4.0, 0.0]]) * u.m, ("A",), ROOT)


def _weighted_setup():
    return point_sum(
        WeightedPointSource(SOURCES_3, [1.0, 1.0, 1.0]),
        frame=ROOT,
        field_definition=_field("weighted"),
    ).at(_targets())


def _volume_setup():
    # Densities 0.5, 1.5, 1.5, 2.5 kg/m^3 with volumes 1, 2, 3, 4 m^3.
    positions = [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 12.0], [1.0, 0.0, 0.0]]
    return point_sum(
        VolumeElementSource(
            positions, [1.0, 2.0, 3.0, 4.0], np.array([0.5, 1.5, 1.5, 2.5])
        ),
        frame=ROOT,
        field_definition=_field("volume"),
    ).at(_targets())


def _norm(text: str) -> str:
    return text.replace(" ", "")


def test_stairs_values_and_edges_match_hist() -> None:
    setup = _volume_setup()
    bins = [0.0, 1.0, 2.0, 3.0] * u.kg / u.m**3

    fig = _api_plot_hist(setup, "density", bins=bins)
    stairs = fig.axes[0].patches[0].get_data()
    hist = _api_hist(setup, "density", bins=bins)

    np.testing.assert_array_equal(stairs.values, hist.values.value)
    np.testing.assert_allclose(stairs.edges, hist.edges.value, rtol=0, atol=0)
    np.testing.assert_array_equal(stairs.values, [1.0, 2.0, 1.0])


def test_stairs_edges_use_requested_xunit() -> None:
    setup = _weighted_setup()

    fig = _api_plot_hist(
        setup,
        "distance",
        bins=[0.0, 0.006, 0.012, 0.014],
        xunit="km",
    )
    stairs = fig.axes[0].patches[0].get_data()

    np.testing.assert_array_equal(stairs.values, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(stairs.edges, [0.0, 0.006, 0.012, 0.014], rtol=1e-12)


@pytest.mark.parametrize(
    ("quantity", "setup_builder", "bins", "xunit", "expected_unit"),
    [
        ("density", _volume_setup, [0.0, 1.0, 2.0, 3.0], None, "kg/m3"),
        ("distance", _weighted_setup, [0.0, 6.0, 12.0, 14.0], "km", "km"),
        ("inverse_square", _weighted_setup, [0.0, 0.02, 0.05], None, "1/m2"),
    ],
)
def test_axis_label_contains_unit(
    quantity: str, setup_builder, bins, xunit, expected_unit: str
) -> None:
    kwargs = {"bins": bins}
    if xunit is not None:
        kwargs["xunit"] = xunit

    fig = _api_plot_hist(setup_builder(), quantity, **kwargs)
    label = fig.axes[0].get_xlabel()

    assert label
    assert _norm(u.Unit(expected_unit).to_string()) in _norm(label)


def test_count_and_volume_y_labels_differ() -> None:
    count_fig = _api_plot_hist(_volume_setup(), "density", bins=[0.0, 1.0, 2.0, 3.0])
    volume_fig = _api_plot_hist(
        _volume_setup(), "density", bins=[0.0, 1.0, 2.0, 3.0], weight="volume"
    )

    count_label = count_fig.axes[0].get_ylabel()
    volume_label = volume_fig.axes[0].get_ylabel()
    assert count_label and volume_label
    assert count_label != volume_label
    assert _norm(u.Unit("m3").to_string()) in _norm(volume_label)


def test_returns_figure() -> None:
    fig = _api_plot_hist(_weighted_setup(), "distance", bins=[0.0, 6.0, 12.0, 14.0])

    assert isinstance(fig, Figure)


def test_radial_result_plot_uses_normalized_values_and_returns_result() -> None:
    from gragra._plotting._hist import plot_hist
    from gragra._plotting._radial_histogram import radial_histogram

    setup = point_sum(
        WeightedPointSource([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [2.0, 8.0]),
        frame=ROOT,
        field_definition=_field("radial"),
    ).at(TargetPoints(np.zeros((1, 3)) * u.m, ("origin",), ROOT))
    result = radial_histogram(
        setup,
        "mass_over_r2",
        bins=[0.5, 1.5, 2.5],
        normalization="fraction",
    )

    fig, ax, returned = plot_hist(result)
    stairs = ax.patches[0].get_data()
    assert returned is result
    assert isinstance(fig, Figure)
    np.testing.assert_array_equal(stairs.values, [0.5, 0.5])
    np.testing.assert_array_equal(stairs.edges, [0.5, 1.5, 2.5])
    assert u.Unit(result.unit) == u.dimensionless_unscaled
    assert ax.get_ylabel()
    assert _norm(u.dimensionless_unscaled.to_string()) in _norm(ax.get_ylabel())


def test_radial_result_plot_does_not_apply_irrelevant_histogram_options() -> None:
    from gragra._plotting._hist import plot_hist
    from gragra._plotting._radial_histogram import radial_histogram

    setup = point_sum(
        WeightedPointSource([[1.0, 0.0, 0.0]], [2.0]),
        frame=ROOT,
        field_definition=_field("radial"),
    ).at(TargetPoints(np.zeros((1, 3)) * u.m, ("origin",), ROOT))
    result = radial_histogram(setup, "mass", bins=[0.0, 2.0])

    with pytest.raises(ValueError, match="cannot be changed"):
        plot_hist(result, bins=10)


def test_radial_plot_preserves_small_bin_after_large_neighbor() -> None:
    from gragra._plotting._hist import plot_hist
    from gragra._plotting._radial_histogram import radial_histogram

    setup = point_sum(
        WeightedPointSource([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [1.0e16, 1.0]),
        frame=ROOT,
        field_definition=_field("radial"),
    ).at(TargetPoints(np.zeros((1, 3)) * u.m, ("origin",), ROOT))
    result = radial_histogram(
        setup, "mass", bins=[0.0, 1.5, 2.5], normalization="fraction"
    )

    _, ax, returned = plot_hist(result)
    assert returned is result
    np.testing.assert_array_equal(ax.patches[0].get_data().values, [1.0, 1.0e-16])


def test_reuses_given_empty_axes() -> None:
    fig_in, ax = plt.subplots()

    out = _api_plot_hist(
        _weighted_setup(), "distance", bins=[0.0, 6.0, 12.0, 14.0], ax=ax
    )

    assert out is fig_in
    assert len(ax.patches) == 1


def test_nonempty_axes_are_rejected() -> None:
    _, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])

    with pytest.raises(ValueError, match="requires empty axes"):
        _api_plot_hist(_weighted_setup(), "distance", bins=[0.0, 6.0], ax=ax)


def test_plot_bin_limit_rejects_before_drawing() -> None:
    _, ax = plt.subplots()

    with pytest.raises(ValueError, match="draws at most"):
        _api_plot_hist(_weighted_setup(), "distance", bins=MAX_PLOT_BINS + 1, ax=ax)

    assert len(ax.patches) == 0


def test_hist_itself_is_not_limited_by_plot_bin_limit() -> None:
    hist = _api_hist(_weighted_setup(), "distance", bins=MAX_PLOT_BINS + 1)

    assert len(hist.values) == MAX_PLOT_BINS + 1


def test_auto_bin_limit_is_checked_before_drawing(monkeypatch) -> None:
    from gragra._plotting import _hist

    setup = _weighted_setup()
    # Distances 5, 10 and 13 give three automatic bins, above the limit set below.
    assert len(_api_hist(setup, "distance", bins="auto").edges) - 1 > 2
    monkeypatch.setattr(_hist, "_MAX_PLOT_BINS", 2)
    _, ax = plt.subplots()
    figures_before = plt.get_fignums()

    with pytest.raises(ValueError, match="draws at most"):
        _api_plot_hist(setup, "distance", bins="auto")
    with pytest.raises(ValueError, match="draws at most"):
        _api_plot_hist(setup, "distance", bins="auto", ax=ax)

    assert plt.get_fignums() == figures_before
    assert len(ax.patches) == 0
