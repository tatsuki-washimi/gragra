"""Contract tests for signed source contribution histograms by radius."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from gwexpy.histogram import Histogram

from gragra.fields.density_series import DensityPhasors
from gragra.geometry import Frame
from gragra.gravity_sources import VolumeElementSource, WeightedPointSource
from gragra.gravity_sources._models import point_sum
from gragra.results import Contribution, GravityFieldDefinition, PotentialReference
from gragra.targets import TargetPoints

ROOT = Frame("root")


def _setup(
    radii,
    masses,
    *,
    volumes=None,
    density=None,
    target_ids=("A",),
    targets=((0.0, 0.0, 0.0),),
):
    coordinates = np.asarray(radii, dtype=np.float64)
    positions = np.column_stack(
        (coordinates, np.zeros(len(radii)), np.zeros(len(radii)))
    )
    if volumes is None:
        source = WeightedPointSource(positions, np.asarray(masses, dtype=np.float64))
    else:
        source = VolumeElementSource(positions, volumes, density)
    field = GravityFieldDefinition.source_field(
        "radial", Contribution("total"), PotentialReference.ZERO_AT_INFINITY
    )
    model = point_sum(source, frame=ROOT, field_definition=field)
    return model.at(
        TargetPoints(np.asarray(targets, dtype=np.float64) * u.m, target_ids, ROOT)
    )


def _api_radial(setup, quantity="mass", **kwargs):
    from gragra._plotting._radial_histogram import radial_histogram

    return radial_histogram(setup, quantity, **kwargs)


def _api_gg_radial(setup, **kwargs):
    from gragra.plotting import gravity_gradient_radial_histogram

    return gravity_gradient_radial_histogram(setup, **kwargs)


def _setup_positions(
    positions, masses, *, frame=ROOT, target=(0.0, 0.0, 0.0), target_frame=None
):
    source = WeightedPointSource(
        np.asarray(positions, dtype=np.float64), np.asarray(masses, dtype=np.float64)
    )
    field = GravityFieldDefinition.source_field(
        "radial-cartesian", Contribution("total"), PotentialReference.ZERO_AT_INFINITY
    )
    model = point_sum(source, frame=frame, field_definition=field)
    selected_target_frame = frame if target_frame is None else target_frame
    return model.at(
        TargetPoints(
            np.asarray(target, dtype=np.float64).reshape(1, 3) * u.m,
            ("A",),
            selected_target_frame,
        )
    )


def _dynamic_result(
    positions,
    density,
    *,
    edges,
    frequencies=(10.0, 20.0),
    observable="potential",
    targets=((0.0, 0.0, 0.0),),
):
    from gragra._plotting._dynamic_contribution import source_contributions

    positions = np.asarray(positions, dtype=np.float64)
    phasors = DensityPhasors(
        positions,
        np.ones(len(positions)),
        np.asarray(density, dtype=np.complex128),
        np.asarray(frequencies, dtype=np.float64),
        0.0,
        np.zeros(3),
        "root",
        np.zeros(3),
        tuple(f"source-{index}" for index in range(len(positions))),
    )
    return source_contributions(
        phasors,
        observable,
        targets,
        radial_edges_m=np.asarray(edges, dtype=np.float64),
    )


def _api_dynamic(result, quantity, **kwargs):
    from gragra._plotting._radial_histogram import dynamic_radial_histogram

    return dynamic_radial_histogram(result, quantity, **kwargs)


def test_raw_bins_keep_flows_last_edge_and_signed_unit() -> None:
    setup = _setup([0.5, 1.0, 2.0, 4.0, 5.0], [1.0, 2.0, 8.0, 32.0, 16.0])
    result = _api_radial(setup, bins=[1.0, 2.0, 4.0])

    assert isinstance(result.histogram, Histogram)
    assert result.quantity == "mass"
    assert result.normalization == "none"
    assert result.histogram.unit == u.kg
    assert result.histogram.sumw2 is None
    assert result.histogram.cov is None
    assert result.unit == u.kg
    np.testing.assert_array_equal(result.histogram.values.value, [2.0, 40.0])
    assert result.histogram.underflow == 1.0 * u.kg
    assert result.histogram.overflow == 16.0 * u.kg
    np.testing.assert_array_equal(result.histogram.edges.to_value(u.m), [1.0, 2.0, 4.0])
    # The source at the final edge is included in the last finite bin.
    np.testing.assert_array_equal(result.values.to_value(u.kg), [2.0, 40.0])
    assert result.normalization_total == 59.0 * u.kg
    assert result.target_id == "A"
    np.testing.assert_array_equal(result.source_index, np.arange(5))
    assert result.domain == (0.5, 5.0)


@pytest.mark.parametrize(
    ("normalization", "expected", "expected_unit"),
    [
        ("none", [2.0, 40.0], u.kg),
        ("per_length", [2.0, 20.0], u.kg / u.m),
        ("fraction", [2.0 / 59.0, 40.0 / 59.0], u.dimensionless_unscaled),
        ("fraction_per_length", [2.0 / 59.0, 20.0 / 59.0], u.m**-1),
    ],
)
def test_normalization_modes_and_raw_cumulative(
    normalization: str, expected, expected_unit: u.UnitBase
) -> None:
    setup = _setup([0.5, 1.0, 2.0, 4.0, 5.0], [1.0, 2.0, 8.0, 32.0, 16.0])
    result = _api_radial(setup, bins=[1.0, 2.0, 4.0], normalization=normalization)

    np.testing.assert_allclose(result.values.value, expected, rtol=1e-12, atol=0.0)
    assert result.unit == expected_unit
    np.testing.assert_array_equal(result.cumulative().value, [3.0, 43.0])
    assert result.cumulative().unit == u.kg
    np.testing.assert_allclose(
        result.cumulative(fraction=True).value, [3.0 / 59.0, 43.0 / 59.0]
    )
    # A density is integrated with shell width; summing its displayed values is wrong.
    if normalization == "per_length":
        widths = np.diff(result.histogram.edges.to_value(u.m))
        assert np.dot(result.values.value, widths) == 42.0
    if normalization == "fraction_per_length":
        widths = np.diff(result.histogram.edges.to_value(u.m))
        assert np.dot(result.values.value, widths) == pytest.approx(42.0 / 59.0)


def test_viewport_does_not_change_signed_global_fraction_denominator() -> None:
    setup = _setup([1.0, 2.0, 3.0], [2.0, 4.0, -1.0])
    result = _api_radial(setup, bins=[0.0, 2.0], normalization="fraction")

    assert result.normalization_total == 5.0 * u.kg
    assert result.values == [1.2] * u.dimensionless_unscaled
    assert result.histogram.overflow == -1.0 * u.kg
    assert result.cumulative(fraction=True)[-1] == 1.2


def test_binned_sums_do_not_subtract_adjacent_large_contributions() -> None:
    setup = _setup([1.0, 2.0], [1.0e16, 1.0])
    raw = _api_radial(setup, bins=[0.0, 1.5, 2.5])
    fraction = _api_radial(setup, bins=[0.0, 1.5, 2.5], normalization="fraction")

    np.testing.assert_array_equal(raw.histogram.values.value, [1.0e16, 1.0])
    np.testing.assert_array_equal(fraction.values.value, [1.0, 1.0e-16])


def test_fraction_rejects_zero_signed_denominator() -> None:
    setup = _setup([1.0, 2.0], [3.0, -3.0])
    with pytest.raises(ValueError, match="normalization total"):
        _api_radial(setup, bins=[0.0, 3.0], normalization="fraction")


@pytest.mark.parametrize("quantity", ["distance", "density", "inverse_r"])
def test_radial_histogram_requires_verified_volume_for_intensive_quantity(
    quantity: str,
) -> None:
    setup = _setup([1.0, 2.0], [2.0, 8.0])
    with pytest.raises(ValueError, match="density and volume"):
        _api_radial(setup, quantity, bins=[0.0, 3.0])


@pytest.mark.parametrize("normalization", ["raw", "density", "absolute_fraction"])
def test_invalid_normalization_is_rejected(normalization: str) -> None:
    setup = _setup([1.0, 2.0], [2.0, 8.0])
    with pytest.raises(ValueError, match="normalization"):
        _api_radial(setup, bins=[0.0, 3.0], normalization=normalization)


@pytest.mark.parametrize("bins", [True, 0, -1, [], [2.0, 1.0], [0.0, np.inf]])
def test_invalid_bin_specs_are_rejected(bins) -> None:
    setup = _setup([1.0, 2.0], [2.0, 8.0])
    with pytest.raises(ValueError):
        _api_radial(setup, bins=bins)


def test_mass_over_r2_aliases_have_kg_per_square_meter_values() -> None:
    setup = _setup([1.0, 2.0], [2.0, 8.0])
    result = _api_radial(
        setup,
        "mass/r^2",
        bins=[0.5, 1.5, 2.5],
        normalization="fraction",
    )

    assert result.quantity == "mass_over_r2"
    assert result.histogram.unit == u.kg / u.m**2
    np.testing.assert_array_equal(result.histogram.values.value, [2.0, 2.0])
    np.testing.assert_array_equal(result.values.value, [0.5, 0.5])
    assert result.unit == u.dimensionless_unscaled


def test_mass_over_r_has_integrated_kilogram_per_meter_values() -> None:
    setup = _setup([1.0, 2.0], [2.0, 8.0])
    result = _api_radial(setup, "mass/r", bins=[0.5, 1.5, 2.5])

    assert result.quantity == "mass_over_r"
    assert result.histogram.unit == u.kg / u.m
    np.testing.assert_array_equal(result.histogram.values.value, [2.0, 4.0])
    np.testing.assert_array_equal(result.values.value, [2.0, 4.0])


def test_density_over_r2_integrates_verified_cell_volume_by_radius() -> None:
    setup = _setup(
        [1.0, 2.0],
        [8.0, 15.0],
        volumes=[2.0, 3.0],
        density=[4.0, 5.0],
    )
    result = _api_radial(
        setup,
        "density_over_r2",
        bins=[0.5, 1.5, 2.5],
    )

    # rho * dV / r^2 gives 8 and 15/4 kg/m^2 in the two shells.
    assert result.histogram.unit == u.kg / u.m**2
    np.testing.assert_allclose(result.histogram.values.value, [8.0, 15.0 / 4.0])
    np.testing.assert_allclose(
        result.normalization_total.to_value(u.kg / u.m**2),
        8.0 + 15.0 / 4.0,
    )
    assert result.values.unit == u.kg / u.m**2


def test_radial_histogram_integrates_intensive_quantities_and_keeps_aliases() -> None:
    setup = _setup(
        [1.0, 2.0],
        [8.0, 15.0],
        volumes=[2.0, 3.0],
        density=[4.0, 5.0],
    )
    density = _api_radial(setup, "density", bins=[0.5, 1.5, 2.5])
    inverse_cube = _api_radial(setup, "1/r^3", bins=[0.5, 1.5, 2.5])
    mass_r2 = _api_radial(setup, "dm/r^2", bins=[0.5, 1.5, 2.5])
    mass_r3 = _api_radial(setup, "dm/r^3", bins=[0.5, 1.5, 2.5])

    assert density.histogram.unit == u.kg
    np.testing.assert_array_equal(density.histogram.values.value, [8.0, 15.0])
    assert inverse_cube.histogram.unit == u.dimensionless_unscaled
    np.testing.assert_allclose(inverse_cube.histogram.values.value, [2.0, 3.0 / 8.0])
    assert mass_r2.quantity == "mass_over_r2"
    np.testing.assert_array_equal(mass_r2.histogram.values.value, [8.0, 15.0 / 4.0])
    assert mass_r3.quantity == "mass_over_r3"
    np.testing.assert_array_equal(mass_r3.histogram.values.value, [8.0, 15.0 / 8.0])


@pytest.mark.parametrize(
    ("norm", "expected"),
    [("frobenius", np.sqrt(6.0) / 4.0), ("spectral", 0.5)],
)
def test_gg_kernel_norm_radial_histogram_sums_per_source_norms(norm, expected):
    positions = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    source = WeightedPointSource(positions, np.array([3.0, 5.0]))
    definition = GravityFieldDefinition.source_field(
        "radial-gg-kernel-norm",
        Contribution("total"),
        PotentialReference.ZERO_AT_INFINITY,
    )
    setup = point_sum(source, frame=ROOT, field_definition=definition).at(
        TargetPoints([[0.0, 0.0, 0.0]] * u.m, ("target-0",), ROOT)
    )

    result = _api_radial(
        setup,
        "GG_kernel_norm",
        target_id="target-0",
        norm=norm,
        bins=[0.0, 3.0],
    )

    np.testing.assert_allclose(
        result.histogram.values.value, [expected], rtol=0, atol=1e-15
    )
    assert result.quantity == "gravity_gradient_kernel_norm"
    assert result.norm == norm
    assert result.histogram.unit == u.m**-3


def test_gg_kernel_norm_radial_histogram_requires_explicit_norm():
    setup = _setup([2.0], [3.0])

    with pytest.raises(ValueError, match="norm must be 'frobenius' or 'spectral'"):
        _api_radial(
            setup,
            "gravity_gradient_kernel_norm",
            target_id="A",
            bins=[0.0, 3.0],
        )


def test_static_gg_radial_histogram_matches_one_source_analytic_tensor():
    from gragra.constants import G_SI

    result = _api_gg_radial(_setup([2.0], [3.0]), target_id="A", bins=[0.0, 1.0, 2.0])
    expected = G_SI * 3.0 / 2.0**3 * np.diag([2.0, -1.0, -1.0])

    assert result.tensor_values.shape == (2, 3, 3)
    np.testing.assert_array_equal(result.tensor_values.value[0], np.zeros((3, 3)))
    np.testing.assert_allclose(result.tensor_values.value[1], expected)
    np.testing.assert_allclose(result.total_tensor.value, expected)
    assert result.unit == u.s**-2
    assert result.target_id == "A"
    assert result.frame is ROOT
    assert result.metadata["softening_m"] == 0.0
    assert result.metadata["radial_distance_definition"] == (
        "unsoftened Euclidean source-target distance"
    )

    histogram = result.histogram("xx")
    assert isinstance(histogram, Histogram)
    assert histogram.unit == u.s**-2
    np.testing.assert_allclose(histogram.values.value, [0.0, expected[0, 0]])
    assert histogram.edges.to_value(u.m).tolist() == [0.0, 1.0, 2.0]
    with pytest.raises(ValueError, match="component must be one of"):
        result.histogram("trace")


def test_static_gg_radial_result_rejects_invalid_index_and_complex_tensor():
    from dataclasses import replace

    result = _api_gg_radial(_setup([2.0], [3.0]), target_id="A", bins=[0.0, 3.0])

    with pytest.raises(TypeError, match="source_index must contain integers"):
        replace(result, source_index=[0.5])
    with pytest.raises(TypeError, match="tensors must contain real values"):
        replace(
            result,
            tensor_values=u.Quantity(
                result.tensor_values.value.astype(np.complex128) + 1j,
                unit=result.unit,
            ),
        )


def test_static_gg_radial_histogram_preserves_signed_flows_and_total_tensor():
    from gragra.constants import G_SI
    from gragra.plotting import source_field

    setup = _setup([0.5, 1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0, 5.0])
    result = _api_gg_radial(setup, target_id="A", bins=[1.0, 2.0, 3.0])
    scale = G_SI * np.diag([2.0, -1.0, -1.0])

    np.testing.assert_allclose(result.underflow_tensor.value, scale * (1.0 / 0.5**3))
    np.testing.assert_allclose(result.tensor_values.value[0], scale * (2.0 / 1.0**3))
    np.testing.assert_allclose(
        result.tensor_values.value[1],
        scale * (3.0 / 2.0**3 + 4.0 / 3.0**3),
    )
    np.testing.assert_allclose(result.overflow_tensor.value, scale * (5.0 / 4.0**3))

    grouped_total = (
        result.underflow_tensor.value
        + result.tensor_values.value.sum(axis=0)
        + result.overflow_tensor.value
    )
    np.testing.assert_allclose(grouped_total, result.total_tensor.value)
    grid_total = source_field(
        setup,
        "GG",
        target_id="A",
        grid=([-1.0, 5.0], [-1.0, 1.0], [-1.0, 1.0]),
        aggregation="sum",
    ).to_array()[0, 0, 0, 0]
    np.testing.assert_allclose(result.total_tensor.value, grid_total)

    xx = result.histogram("xx")
    assert xx.underflow.value > 0.0
    assert xx.overflow.value > 0.0
    assert xx.values.value[0] > 0.0
    assert xx.values.value[1] > 0.0
    yy = result.histogram("yy")
    assert yy.underflow.value < 0.0
    assert yy.overflow.value < 0.0
    assert np.all(yy.values.value < 0.0)


def test_static_gg_radial_tensor_keeps_component_cancellation():
    from gragra.constants import G_SI

    setup = _setup_positions([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], [1.0, 2.0])
    result = _api_gg_radial(setup, target_id="A", bins=[0.0, 3.0])
    scale = G_SI / 2.0**3
    expected = scale * np.diag([0.0, 3.0, -3.0])

    np.testing.assert_allclose(result.tensor_values.value[0], expected)
    assert result.tensor_values.value[0, 0, 0] == pytest.approx(0.0, abs=1e-30)
    assert result.tensor_values.value[0, 1, 1] > 0.0
    assert result.tensor_values.value[0, 2, 2] < 0.0


def test_static_gg_radial_histogram_matches_softened_source_field_at_zero_radius():
    from gragra._evaluation_setup import CoordinatePrecisionError
    from gragra.constants import G_SI
    from gragra.plotting import source_field

    setup = _setup([0.0], [3.0])
    result = _api_gg_radial(setup, target_id="A", bins=[0.0, 1.0], softening_m=0.5)
    expected = -G_SI * 3.0 / 0.5**3 * np.eye(3)
    np.testing.assert_allclose(result.tensor_values.value[0], expected)
    assert result.softening_m == 0.5
    field_total = source_field(
        setup,
        "GG",
        target_id="A",
        grid=([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]),
        aggregation="sum",
        softening_m=0.5,
    ).to_array()[0, 0, 0, 0]
    np.testing.assert_allclose(result.total_tensor.value, field_total)
    with pytest.raises(CoordinatePrecisionError) as caught:
        _api_gg_radial(setup, target_id="A", bins=[0.0, 1.0])
    assert caught.value.reason == "collision"


def test_static_gg_radial_membership_uses_geometric_radius_with_softening():
    from gragra.plotting import source_field

    setup = _setup([0.25], [3.0])
    result = _api_gg_radial(
        setup,
        target_id="A",
        bins=[0.0, 0.4, 1.0],
        softening_m=0.5,
    )
    softened_total = source_field(
        setup,
        "GG",
        target_id="A",
        grid=([-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]),
        aggregation="sum",
        softening_m=0.5,
    ).to_array()[0, 0, 0, 0]

    np.testing.assert_allclose(result.tensor_values.value[0], softened_total)
    np.testing.assert_array_equal(result.tensor_values.value[1], np.zeros((3, 3)))
    np.testing.assert_array_equal(result.underflow_tensor.value, np.zeros((3, 3)))
    np.testing.assert_array_equal(result.overflow_tensor.value, np.zeros((3, 3)))
    np.testing.assert_allclose(result.total_tensor.value, softened_total)


def test_static_gg_radial_softening_keeps_subnormal_squared_radius():
    from gragra.plotting import source_field

    setup = _setup_positions([[1e-160, 0.0, 0.0]], [1.0])
    result = _api_gg_radial(
        setup,
        target_id="A",
        bins=[0.0, 2e-160],
        softening_m=0.5,
    )
    field_total = source_field(
        setup,
        "GG",
        target_id="A",
        grid=([-1e-159, 1e-159], [-1.0, 1.0], [-1.0, 1.0]),
        aggregation="sum",
        softening_m=0.5,
    ).to_array()[0, 0, 0, 0]

    np.testing.assert_allclose(result.total_tensor.value, field_total)
    np.testing.assert_allclose(result.tensor_values.value.sum(axis=0), field_total)
    assert result.softening_m == 0.5


def test_static_gg_radial_histogram_uses_source_computation_frame():
    from gragra.constants import G_SI

    root = ROOT
    local = Frame("gg-radial-local", origin=[1.0, 0.0, 0.0] * u.m, parent=root)
    setup = _setup_positions(
        [[3.0, 0.0, 0.0]],
        [2.0],
        frame=local,
        target=[0.0, 0.0, 0.0],
        target_frame=root,
    )
    result = _api_gg_radial(setup, target_id="A", bins=[4.0, 5.0])
    expected = G_SI * 2.0 / 4.0**3 * np.diag([2.0, -1.0, -1.0])

    assert result.frame is local
    np.testing.assert_allclose(result.target_xyz_m, [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(result.tensor_values.value[0], expected)
    assert result.metadata["frame"] == local.name


def test_static_gg_radial_histogram_preserves_coordinate_precision_gate():
    from gragra._evaluation_setup import CoordinatePrecisionError

    root = Frame("gg-radial-precision-root")
    source_frame = Frame(
        "gg-radial-precision-local", origin=[1e6, 0.0, 0.0] * u.m, parent=root
    )
    setup = _setup_positions(
        [[0.0, 0.0, 0.0]],
        [1.0],
        frame=source_frame,
        target=[1e6 + 1e-8, 0.0, 0.0],
        target_frame=root,
    )

    with pytest.raises(CoordinatePrecisionError) as caught:
        _api_gg_radial(setup, target_id="A", bins=[0.0, 1.0], softening_m=0.5)
    assert caught.value.reason == "coordinate_precision"
    assert caught.value.target_id == "A"
    assert caught.value.source_index == 0


@pytest.mark.parametrize(
    ("quantity", "kwargs", "expected_title"),
    [
        ("mass", {}, "mass; target=A"),
        (
            "GG_kernel_norm",
            {"norm": "spectral"},
            "gravity_gradient_kernel_norm; target=A; norm=spectral",
        ),
    ],
)
def test_plot_radial_histogram_title_shows_selected_norm(
    quantity, kwargs, expected_title
):
    from matplotlib import pyplot as plt

    from gragra.plotting import plot_hist

    result = _api_radial(
        _setup([2.0], [3.0]), quantity, target_id="A", bins=[0.0, 3.0], **kwargs
    )

    fig, ax, returned = plot_hist(result)
    try:
        assert returned is result
        assert ax.get_title() == expected_title
    finally:
        plt.close(fig)


def test_softened_gg_kernel_norm_radial_histogram_allows_zero_radius():
    setup = _setup([0.0], [3.0])

    result = _api_radial(
        setup,
        "GG_kernel_norm",
        target_id="A",
        norm="spectral",
        softening_m=0.5,
        bins=[0.0, 1.0],
    )

    np.testing.assert_allclose(result.histogram.values.value, [8.0], rtol=0, atol=0)
    assert result.domain == (0.0, 0.0)
    assert result.softening_m == 0.5


def test_radial_intensive_quantity_requires_verified_source_volume() -> None:
    setup = _setup([1.0, 2.0], [2.0, 8.0])
    with pytest.raises(ValueError, match="density and volume"):
        _api_radial(setup, "inverse_cube", bins=[0.5, 2.5])


def test_mass_can_omit_target_id_when_only_one_target_exists() -> None:
    setup = _setup([1.0, 2.0], [2.0, 8.0])
    result = _api_radial(setup, "mass", bins=[0.0, 1.5, 2.5])
    assert result.target_id == "A"


def test_target_id_is_required_for_multiple_targets() -> None:
    setup = _setup(
        [1.0, 2.0],
        [2.0, 8.0],
        target_ids=("A", "B"),
        targets=((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)),
    )
    with pytest.raises(ValueError, match="target_id is required"):
        _api_radial(setup, bins=[0.0, 5.0])


def test_result_snapshots_are_read_only_and_input_is_unchanged() -> None:
    positions = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    masses = np.array([2.0, 8.0])
    setup = _setup([1.0, 2.0], masses)
    source = setup.resolved.source
    source_positions_before = np.array(source.positions_m, copy=True)
    source_masses_before = np.array(source.weights_kg, copy=True)
    result = _api_radial(setup, bins=[0.0, 3.0])

    np.testing.assert_array_equal(source.positions_m, source_positions_before)
    np.testing.assert_array_equal(source.weights_kg, source_masses_before)
    np.testing.assert_array_equal(positions, [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    np.testing.assert_array_equal(masses, [2.0, 8.0])
    assert not result.values.value.flags.writeable
    assert not result.normalization_total.value.flags.writeable
    assert not result.source_index.flags.writeable
    assert not result.target_xyz_m.flags.writeable
    with pytest.raises(ValueError):
        result.values.value[0] = 0.0


def test_volume_weighted_uniform_sphere_center_matches_exact_integral() -> None:
    # The source grid is shifted from the singular target by half a cell.
    from gragra.geometry import Sphere
    from gragra.gravity_sources import discretize

    radius = 1.0
    density = 1000.0
    sphere = Sphere(np.zeros(3) * u.m, radius * u.m, ROOT)
    source = discretize(sphere, density * u.kg / u.m**3, 0.025 * u.m)
    model = point_sum(
        source,
        frame=ROOT,
    )
    center_setup = model.at(TargetPoints(np.zeros((1, 3)) * u.m, ("center",), ROOT))
    center = _api_radial(center_setup, "mass_over_r2", bins=[0.0, 2.0])
    center_expected = 4.0 * np.pi * density * radius
    np.testing.assert_allclose(
        center.histogram.values.to_value(u.kg / u.m**2),
        [center_expected],
        rtol=0.02,
        atol=0.0,
    )

    distance = 3.0 * radius
    external_setup = model.at(
        TargetPoints(np.array([[0.0, 0.0, distance]]) * u.m, ("external",), ROOT)
    )
    external = _api_radial(external_setup, "mass_over_r2", bins=[0.0, 5.0])
    analytic_external = (
        2.0
        * np.pi
        * density
        * (
            radius
            + (radius**2 - distance**2) / distance * np.arctanh(radius / distance)
        )
    )
    np.testing.assert_allclose(
        external.histogram.values.to_value(u.kg / u.m**2),
        [analytic_external],
        rtol=0.01,
        atol=0.0,
    )


def test_dynamic_radial_histograms_select_frequency_and_keep_complex_bins():
    result = _dynamic_result(
        [[0.5, 0, 0], [1.5, 0, 0], [2.5, 0, 0]],
        [[1 + 2j, 0, -1 + 0.5j], [2 - 1j, 1 + 1j, 3 + 0.5j]],
        edges=[1.0, 2.0],
    )
    groups = result.to_array(kind="group", layout="sample_first")[:, 0, :]

    real = _api_dynamic(result, "complex_real", frequency_hz=10.0)
    imaginary = _api_dynamic(result, "complex_imag", frequency_hz=10.0)
    amplitude = _api_dynamic(result, "amplitude", frequency_hz=10.0)

    assert isinstance(real.histogram, Histogram)
    assert real.histogram.unit == u.Unit(result.unit)
    assert real.frequency_hz == 10.0
    assert real.component == "potential"
    np.testing.assert_allclose(real.values.value, groups[0, 1:-1].real)
    np.testing.assert_allclose(imaginary.values.value, groups[0, 1:-1].imag)
    np.testing.assert_allclose(amplitude.values.value, np.abs(groups[0, 1:-1]))
    assert real.histogram.underflow.value == pytest.approx(groups[0, 0].real)
    assert real.histogram.overflow.value == pytest.approx(groups[0, -1].real)
    assert real.histogram.edges.to_value(u.m).tolist() == [1.0, 2.0]
    reconstructed = (
        real.histogram.underflow.value
        + real.values.value.sum()
        + real.histogram.overflow.value
    ) + 1j * (
        imaginary.histogram.underflow.value
        + imaginary.values.value.sum()
        + imaginary.histogram.overflow.value
    )
    expected_total = result.to_array(kind="total", layout="sample_first")[0, 0]
    assert reconstructed == pytest.approx(expected_total)
    with pytest.raises(ValueError, match="frequency_hz"):
        _api_dynamic(result, "amplitude")


def test_direct_complex_histograms_do_not_derive_unselected_frequencies():
    from gragra._plotting._source_contracts import SourceContributionResult
    from gragra._plotting._source_data import SourceContributionData

    sources = np.array(
        [[[1 + 2j, 3 - 1j]], [[1e308 + 0j, -1e308 + 0j]]],
        dtype=np.complex128,
    )
    groups = np.zeros((2, 1, 4), dtype=np.complex128)
    groups[0, 0, 1:3] = sources[0, 0]
    groups[1, 0, 1:3] = sources[1, 0]
    record = SourceContributionResult(
        SourceContributionData(
            source_values=sources,
            group_values=groups,
            total_values=sources.sum(axis=2),
            positions_m=[[1.5, 0, 0], [2.5, 0, 0]],
            targets_m=[[0, 0, 0]],
            target_ids=("target-0",),
            frequencies_hz=[10.0, 20.0],
            unit="m2/s2",
            observable="potential",
            frame="root",
            membership=[[1, 2]],
            group_ids=("underflow", "bin_0", "bin_1", "overflow"),
            radial_edges_m=[1.0, 2.0, 3.0],
        )
    )

    real = _api_dynamic(record, "complex_real", frequency_hz=10.0)
    imaginary = _api_dynamic(record, "complex_imag", frequency_hz=10.0)
    amplitude = _api_dynamic(record, "amplitude", frequency_hz=10.0)

    np.testing.assert_allclose(real.values.value, [1.0, 3.0])
    np.testing.assert_allclose(imaginary.values.value, [2.0, -1.0])
    np.testing.assert_allclose(amplitude.values.value, [np.sqrt(5.0), np.sqrt(10.0)])


def test_dynamic_radial_histogram_selects_target_and_vector_component():
    result = _dynamic_result(
        [[1.0, 0, 0], [2.0, 0, 0]],
        [[1 + 1j, 2 - 1j], [2 + 0j, -1 + 2j]],
        edges=[0.5, 1.5, 2.5],
        observable="acceleration",
        targets=[[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
    )
    groups = result.to_array(kind="group", layout="sample_first")

    selected = _api_dynamic(
        result,
        "complex_real",
        target_id="target-1",
        frequency_hz=20 * u.Hz,
        component="x",
    )
    np.testing.assert_allclose(selected.values.value, groups[1, 1, 1:-1, 0].real)
    assert selected.target_id == "target-1"
    assert selected.component == "x"
    assert selected.frequency_hz == 20.0
    assert not selected.values.value.flags.writeable
    assert not selected.valid.flags.writeable
    with pytest.raises(ValueError, match="target_id is required"):
        _api_dynamic(result, "complex_real", frequency_hz=20.0, component="x")
    with pytest.raises(ValueError, match="component is required"):
        _api_dynamic(result, "complex_real", target_id="target-1", frequency_hz=20.0)


def test_dynamic_phase_and_fractions_keep_validity_and_sum_rules():
    result = _dynamic_result(
        [[0.5, 0, 0], [1.5, 0, 0], [2.5, 0, 0]],
        [[1 + 0j, 0j, -0.25 + 0.5j], [1 + 1j, 2 - 1j, -0.5 + 0j]],
        edges=[1.0, 2.0],
    )

    phase = _api_dynamic(result, "phase", frequency_hz=10.0)
    amplitude_fraction = _api_dynamic(result, "amplitude_fraction", frequency_hz=10.0)
    interference = _api_dynamic(result, "interference_share", frequency_hz=10.0)

    assert phase.values[0].value == 0.0
    assert not phase.valid[0]
    assert phase.histogram.unit == u.rad
    assert amplitude_fraction.valid.all()
    amplitude_total = (
        amplitude_fraction.histogram.underflow.value
        + amplitude_fraction.values.value.sum()
        + amplitude_fraction.histogram.overflow.value
    )
    assert amplitude_total == pytest.approx(1.0)
    assert interference.valid.all()
    interference_total = (
        interference.histogram.underflow.value
        + interference.values.value.sum()
        + interference.histogram.overflow.value
    )
    assert interference_total == pytest.approx(1.0)

    zero_result = _dynamic_result(
        [[0.5, 0, 0], [1.5, 0, 0], [2.5, 0, 0]],
        [[0j, 0j, 0j], [0j, 0j, 0j]],
        edges=[1.0, 2.0],
    )
    zero_amplitude_fraction = _api_dynamic(
        zero_result, "amplitude_fraction", frequency_hz=10.0
    )
    zero_rms_fraction = _api_dynamic(
        zero_result,
        "rms_amplitude_fraction",
        frequency_weights=[1.0, 1.0],
    )
    assert not zero_amplitude_fraction.valid.any()
    assert not zero_rms_fraction.valid.any()


def test_dynamic_cumulative_complex_histogram_starts_at_near_radius():
    result = _dynamic_result(
        [[0.5, 0, 0], [1.5, 0, 0], [2.5, 0, 0]],
        [[1 + 2j, 2 - 1j, -0.5 + 0.25j]],
        frequencies=(10.0,),
        edges=[1.0, 2.0],
    )
    groups = result.to_array(kind="group", layout="sample_first")[0, 0, :]
    cumulative = np.cumsum(groups)

    real = _api_dynamic(result, "cumulative_real", frequency_hz=10.0)
    imaginary = _api_dynamic(result, "cumulative_imag", frequency_hz=10.0)

    np.testing.assert_allclose(real.values.value, cumulative[1:-1].real)
    np.testing.assert_allclose(imaginary.values.value, cumulative[1:-1].imag)
    assert real.histogram.underflow.value == pytest.approx(cumulative[0].real)
    assert real.histogram.overflow.value == pytest.approx(cumulative[-1].real)
    assert real.valid.all()


def test_dynamic_rms_fraction_uses_frequency_and_direction_weights():
    from gragra._plotting._dynamic_contribution import (
        directional_contribution_statistics,
    )

    positions = [[1.0, 0, 0], [2.0, 0, 0]]
    edges = [0.5, 1.5, 2.5]
    first = _dynamic_result(
        positions,
        [[1 + 1j, 2 - 1j], [2 + 0j, -1 + 2j]],
        edges=edges,
    )
    second = _dynamic_result(
        positions,
        [[2 - 1j, 1 + 1j], [1 + 2j, 3 - 1j]],
        edges=edges,
    )

    rms = _api_dynamic(
        (first, second),
        "rms_amplitude_fraction",
        frequency_weights=[0.25, 0.75],
        direction_weights=[0.4, 0.6],
    )

    assert rms.frequency_hz is None
    assert rms.valid.all()
    assert rms.unit == u.dimensionless_unscaled
    total_share = (
        rms.histogram.underflow.value
        + rms.values.value.sum()
        + rms.histogram.overflow.value
    )
    assert total_share == pytest.approx(1.0)
    assert rms.metadata["direction_weights"] == (0.4, 0.6)
    assert rms.metadata["frequency_weights"] == (0.25, 0.75)
    expected_statistics = directional_contribution_statistics(
        (first, second),
        direction_weights=[0.4, 0.6],
        frequency_weights=[0.25, 0.75],
    )
    expected_values = [
        expected_statistics.rms.rms_share[group_id].value[0, 0]
        for group_id in ("bin_0", "bin_1")
    ]
    np.testing.assert_allclose(rms.values.value, expected_values)


def test_interference_histogram_reproduces_constructive_and_cancelling_pairs():
    positions = [[1.0, 0, 0], [2.0, 0, 0]]
    edges = [0.5, 1.5, 2.5]
    constructive = _dynamic_result(
        positions, [[1.0 + 0j, 2.0 + 0j]], frequencies=(10.0,), edges=edges
    )
    cancelling = _dynamic_result(
        positions, [[1.0 + 0j, -2.0 + 0j]], frequencies=(10.0,), edges=edges
    )

    constructive_share = _api_dynamic(
        constructive, "interference_share", frequency_hz=10.0
    )
    cancelling_share = _api_dynamic(cancelling, "interference_share", frequency_hz=10.0)

    np.testing.assert_allclose(constructive_share.values.value, [0.5, 0.5])
    assert constructive_share.valid.all()
    np.testing.assert_array_equal(cancelling_share.values.value, [0.0, 0.0])
    assert not cancelling_share.valid.any()


def test_direction_averaged_interference_share_averages_cross_terms_before_ratio():
    positions = [[1.0, 0, 0], [2.0, 0, 0]]
    edges = [0.5, 1.5, 2.5]
    first = _dynamic_result(positions, [[1.0, 0.0]], frequencies=(10.0,), edges=edges)
    second = _dynamic_result(positions, [[0.0, 4.0]], frequencies=(10.0,), edges=edges)

    averaged = _api_dynamic(
        (first, second),
        "interference_share",
        frequency_hz=10.0,
        direction_weights=[1.0, 1.0],
    )
    first_share = _api_dynamic(first, "interference_share", frequency_hz=10.0)
    second_share = _api_dynamic(second, "interference_share", frequency_hz=10.0)

    np.testing.assert_allclose(averaged.values.value, [0.2, 0.8])
    np.testing.assert_allclose(
        0.5 * (first_share.values.value + second_share.values.value), [0.5, 0.5]
    )
    assert averaged.valid.all()
    assert averaged.metadata["direction_weights"] == (1.0, 1.0)
    assert averaged.metadata["normalized_direction_weights"] == (0.5, 0.5)
    assert "direction-weighted mean" in averaged.metadata["metric_definition"]
    total_share = (
        averaged.histogram.underflow.value
        + averaged.values.value.sum()
        + averaged.histogram.overflow.value
    )
    assert total_share == pytest.approx(1.0)

    signed = _dynamic_result(positions, [[-2.0, 1.0]], frequencies=(10.0,), edges=edges)
    signed_share = _api_dynamic(signed, "interference_share", frequency_hz=10.0)
    np.testing.assert_allclose(signed_share.values.value, [4.0 / 3.0, -1.0 / 3.0])
    assert signed_share.valid.all()


def test_direction_averaged_interference_share_masks_zero_total_power():
    positions = [[1.0, 0, 0], [2.0, 0, 0]]
    edges = [0.5, 1.5, 2.5]
    first = _dynamic_result(positions, [[1.0, -2.0]], frequencies=(10.0,), edges=edges)
    second = _dynamic_result(positions, [[2.0, -4.0]], frequencies=(10.0,), edges=edges)

    result = _api_dynamic(
        (first, second),
        "interference_share",
        frequency_hz=10.0,
        direction_weights=[0.25, 0.75],
    )

    np.testing.assert_array_equal(result.values.value, [0.0, 0.0])
    assert not result.valid.any()
    assert not result.underflow_valid
    assert not result.overflow_valid


@pytest.mark.parametrize("mismatch", ["frequency", "edges", "target", "observable"])
def test_direction_averaged_interference_rejects_incompatible_results(mismatch):
    positions = [[1.0, 0, 0], [2.0, 0, 0]]
    edges = [0.5, 1.5, 2.5]
    reference = _dynamic_result(
        positions, [[1.0, 2.0]], frequencies=(10.0,), edges=edges
    )
    if mismatch == "frequency":
        candidate = _dynamic_result(
            positions, [[1.0, 2.0]], frequencies=(11.0,), edges=edges
        )
    elif mismatch == "edges":
        candidate = _dynamic_result(
            positions,
            [[1.0, 2.0]],
            frequencies=(10.0,),
            edges=[0.25, 1.5, 2.75],
        )
    elif mismatch == "target":
        candidate = _dynamic_result(
            positions,
            [[1.0, 2.0]],
            frequencies=(10.0,),
            edges=edges,
            targets=[[0.0, 0.25, 0.0]],
        )
    else:
        candidate = _dynamic_result(
            positions,
            [[1.0, 2.0]],
            frequencies=(10.0,),
            edges=edges,
            observable="acceleration",
        )

    with pytest.raises(ValueError, match="not compatible for directional statistics"):
        _api_dynamic(
            (reference, candidate),
            "interference_share",
            frequency_hz=10.0,
            direction_weights=[1.0, 1.0],
        )


def test_direction_averaged_interference_validates_direction_weights():
    positions = [[1.0, 0, 0], [2.0, 0, 0]]
    first = _dynamic_result(
        positions, [[1.0, 2.0]], frequencies=(10.0,), edges=[0.5, 1.5, 2.5]
    )
    second = _dynamic_result(
        positions, [[2.0, 1.0]], frequencies=(10.0,), edges=[0.5, 1.5, 2.5]
    )

    with pytest.raises(ValueError, match=r"direction_weights must have shape \(2,\)"):
        _api_dynamic(
            (first, second),
            "interference_share",
            frequency_hz=10.0,
            direction_weights=[1.0],
        )
