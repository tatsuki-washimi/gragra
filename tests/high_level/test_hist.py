"""Contract tests for ``gragra.plotting.hist`` (source diagnostic histograms).

Expected values are derived by hand from the small inputs in this file. The
public API and the private helpers are imported inside functions so that each
test reports its own missing-implementation failure instead of a collection
error.
"""

from __future__ import annotations

import tracemalloc
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from astropy import units as u
from gwexpy.histogram import Histogram

from gragra._evaluation_setup import (
    OBSERVABLES,
    CoordinatePrecisionError,
    DiscreteSource,
    ResolvedEvaluation,
    _StaticPointSourceBinding,
)
from gragra.analytic.objects import uniform_sphere
from gragra.evaluation import EvaluationSetup
from gragra.geometry import Frame
from gragra.geometry._sphere import Sphere
from gragra.gravity_sources import (
    PointMass,
    PointMassCloud,
    VolumeElementSource,
    WeightedPointSource,
)
from gragra.gravity_sources._models import point_sum
from gragra.results import Contribution, GravityFieldDefinition, PotentialReference
from gragra.targets import TargetPoints

# Limits from the design plan. Re-declared here so parametrize lists do not
# need the implementation module at collection time; a separate test checks
# that the module constants agree.
MAX_BINS = 1_000_000

ROOT = Frame("root")
ROOT_TARGET = (-3.0, -4.0, 0.0)
# Distances from ROOT_TARGET are exactly 5, 10 and 13 (3-4-5 and 5-12-13 triangles).
SOURCES_3 = [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 12.0]]
# Four volume elements: densities 0.5, 1.5, 1.5, 2.5 kg/m^3 with volumes 1, 2, 3, 4 m^3.
POSITIONS_4 = [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 12.0], [1.0, 0.0, 0.0]]
VOLUMES_4 = [1.0, 2.0, 3.0, 4.0]
DENSITIES_4 = [0.5, 1.5, 1.5, 2.5]

# 90 degrees about z. Columns are the rotated frame's axes in the root frame,
# so global = origin + B @ local and local = B.T @ (global - origin).
ROT_Z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
FRAME_ORIGIN = (1.0, 2.0, 3.0)
# The SOURCES_3 points expressed in the rotated frame S defined by
# origin=(1, 2, 3) and basis ROT_Z.
LOCAL_SOURCES_S = [[-2.0, 1.0, -3.0], [2.0, -2.0, -3.0], [-2.0, 1.0, 9.0]]

DISCRETE_POSITIONS = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]


def _api_hist(setup, quantity, **kwargs):
    from gragra.plotting import hist

    return hist(setup, quantity, **kwargs)


def _hist_module():
    from gragra._plotting import _hist

    return _hist


def _field(name: str = "source") -> GravityFieldDefinition:
    return GravityFieldDefinition.source_field(
        name, Contribution("total"), PotentialReference.ZERO_AT_INFINITY
    )


def _targets(points=(ROOT_TARGET,), *, ids=("A",), frame=ROOT) -> TargetPoints:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return TargetPoints(values * u.m, tuple(ids), frame)


def _weighted_setup(positions, weights=None, *, frame=ROOT, targets=None):
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    weights = np.ones(len(positions)) if weights is None else weights
    model = point_sum(
        WeightedPointSource(positions, weights),
        frame=frame,
        field_definition=_field("weighted"),
    )
    return model.at(_targets() if targets is None else targets)


def _volume_setup(positions, volumes, density, *, frame=ROOT, targets=None):
    model = point_sum(
        VolumeElementSource(positions, volumes, density),
        frame=frame,
        field_definition=_field("volume"),
    )
    return model.at(_targets() if targets is None else targets)


def _volume_density_setup():
    return _volume_setup(POSITIONS_4, VOLUMES_4, np.array(DENSITIES_4))


def _height_setup(z_values):
    z = np.asarray(z_values, dtype=np.float64)
    positions = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    return _weighted_setup(
        positions, np.ones(len(z)), targets=_targets(((100.0, 0.0, 0.0),))
    )


def _values(hist: Histogram) -> np.ndarray:
    return np.asarray(hist.values.value, dtype=np.float64)


def _underflow(hist: Histogram) -> float:
    return float(hist.underflow.value)


def _overflow(hist: Histogram) -> float:
    return float(hist.overflow.value)


def _snapshot(setup) -> tuple[bytes, ...]:
    resolved = setup.resolved
    model_source = resolved.model.source
    parts = [
        np.asarray(model_source.positions_m).tobytes(),
        np.asarray(model_source.volumes_m3).tobytes()
        if hasattr(model_source, "volumes_m3")
        else b"",
        np.asarray(model_source.density_kg_m3).tobytes()
        if hasattr(model_source, "density_kg_m3")
        else b"",
        np.asarray(resolved.targets.positions_m).tobytes(),
        np.asarray(resolved.target_positions_m).tobytes(),
    ]
    if resolved.source is not None:
        parts.append(np.asarray(resolved.source.positions_m).tobytes())
        parts.append(np.asarray(resolved.source.weights_kg).tobytes())
    return tuple(parts)


def _numpy_auto_rule() -> str:
    """Identify the running NumPy 'auto' bin rule from public behaviour."""
    calibration = np.zeros(10_000)
    calibration[0] = 1.0  # IQR = 0 with a small spread: legacy and relaxed differ.
    expected = len(np.histogram_bin_edges(calibration, "auto")) - 1
    estimate = _hist_module().estimate_auto_bin_count
    matches = [
        rule
        for rule in ("legacy", "relaxed")
        if estimate(calibration, rule=rule) == expected
    ]
    assert len(matches) == 1, f"cannot identify NumPy auto rule: {matches}"
    return matches[0]


_AUTO_DATASETS = {
    "normal": lambda: np.random.default_rng(0).normal(size=1000),
    "constant": lambda: np.full(50, 2.0),
    "small_n": lambda: np.array([0.0, 1.0, 3.0]),
    "one_point": lambda: np.array([5.0]),
    "iqr_zero_small_outlier": lambda: np.concatenate([np.zeros(1000), [1e6]]),
}


# ---------------------------------------------------------------------------
# Hand-calculated values, units and bin conventions
# ---------------------------------------------------------------------------


def test_density_count_hand_values() -> None:
    hist = _api_hist(
        _volume_density_setup(),
        "density",
        bins=np.array([0.0, 1.0, 2.0, 3.0]) * u.kg / u.m**3,
    )

    assert isinstance(hist, Histogram)
    # Densities 0.5 | 1.5, 1.5 | 2.5 fall into bins [0,1), [1,2), [2,3].
    np.testing.assert_array_equal(_values(hist), [1.0, 2.0, 1.0])
    np.testing.assert_allclose(
        hist.edges.to_value(u.kg / u.m**3), [0.0, 1.0, 2.0, 3.0], rtol=0, atol=0
    )
    assert _underflow(hist) == 0.0
    assert _overflow(hist) == 0.0
    assert hist.unit == u.dimensionless_unscaled
    assert u.Unit(hist.xunit) == u.kg / u.m**3


def test_density_volume_weight_sums_element_volumes() -> None:
    hist = _api_hist(
        _volume_density_setup(),
        "density",
        bins=[0.0, 1.0, 2.0, 3.0] * u.kg / u.m**3,
        weight="volume",
    )

    # Volumes 1 | 2 + 3 | 4 m^3.
    np.testing.assert_array_equal(_values(hist), [1.0, 5.0, 4.0])
    assert hist.unit == u.m**3


@pytest.mark.parametrize("weight", ["count", "volume"])
def test_plain_bins_are_interpreted_in_requested_xunit(weight: str) -> None:
    hist = _api_hist(
        _volume_density_setup(),
        "density",
        bins=[0.0, 0.001, 0.002, 0.003],
        xunit="g/cm3",
        weight=weight,
    )

    expected = [1.0, 2.0, 1.0] if weight == "count" else [1.0, 5.0, 4.0]
    np.testing.assert_array_equal(_values(hist), expected)
    assert u.Unit(hist.xunit) == u.g / u.cm**3
    np.testing.assert_allclose(
        hist.edges.to_value(u.g / u.cm**3), [0.0, 0.001, 0.002, 0.003], rtol=1e-12
    )


def test_quantity_bins_are_converted_to_xunit() -> None:
    hist = _api_hist(
        _volume_density_setup(),
        "density",
        bins=[0.0, 0.001, 0.002, 0.003] * u.g / u.cm**3,
    )

    # Default xunit for density is kg/m^3, and 1 g/cm^3 = 1000 kg/m^3.
    assert u.Unit(hist.xunit) == u.kg / u.m**3
    np.testing.assert_allclose(
        hist.edges.to_value(u.kg / u.m**3), [0.0, 1.0, 2.0, 3.0], rtol=1e-12
    )
    np.testing.assert_array_equal(_values(hist), [1.0, 2.0, 1.0])


def test_distance_hand_values() -> None:
    hist = _api_hist(
        _weighted_setup(SOURCES_3), "distance", bins=[0.0, 6.0, 12.0, 14.0]
    )

    np.testing.assert_array_equal(_values(hist), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(hist.edges.to_value(u.m), [0.0, 6.0, 12.0, 14.0])
    assert u.Unit(hist.xunit) == u.m


def test_height_hand_values() -> None:
    # z values are 0, 0 and 12.
    hist = _api_hist(_weighted_setup(SOURCES_3), "height", bins=[-1.0, 1.0, 13.0])

    np.testing.assert_array_equal(_values(hist), [2.0, 1.0])
    assert u.Unit(hist.xunit) == u.m


def test_inverse_square_hand_values() -> None:
    # 1/r^2 = 0.04, 0.01 and 1/169 (about 0.0059) for r = 5, 10 and 13.
    hist = _api_hist(
        _weighted_setup(SOURCES_3), "inverse_square", bins=[0.0, 0.02, 0.05]
    )

    np.testing.assert_array_equal(_values(hist), [2.0, 1.0])
    assert u.Unit(hist.xunit) == u.Unit("1/m2")


def test_inverse_square_xunit_per_km2() -> None:
    # 0.02 m^-2 = 2e4 km^-2 and 0.05 m^-2 = 5e4 km^-2.
    hist = _api_hist(
        _weighted_setup(SOURCES_3),
        "inverse_square",
        bins=[0.0, 2.0e4, 5.0e4],
        xunit="1/km2",
    )

    np.testing.assert_array_equal(_values(hist), [2.0, 1.0])
    assert u.Unit(hist.xunit) == u.Unit("1/km2")


def test_inverse_cube_histogram_remains_a_value_distribution() -> None:
    # The input values are 1/r^3 at r=5, 10 and 13 m; the bins classify
    # those values rather than integrating them over radial distance.
    hist = _api_hist(
        _weighted_setup(SOURCES_3),
        "1/r^3",
        bins=[0.0, 0.0008, 0.003, 0.01],
    )

    np.testing.assert_array_equal(_values(hist), [1.0, 1.0, 1.0])
    assert u.Unit(hist.xunit) == u.m**-3


def test_mass_and_mass_ratio_histograms_classify_values_not_radii() -> None:
    setup = _weighted_setup(SOURCES_3, [2.0, 8.0, 16.0])
    mass_hist = _api_hist(setup, "mass", bins=[0.0, 5.0, 10.0])
    ratio_hist = _api_hist(
        setup,
        "mass_over_r2",
        bins=[0.0, 0.085, 0.1],
        target_id="A",
    )

    np.testing.assert_array_equal(_values(mass_hist), [1.0, 1.0])
    assert u.Unit(mass_hist.xunit) == u.kg
    # Values are 2/25, 8/100 and 16/169 kg/m^2; they occupy two bins.
    np.testing.assert_array_equal(_values(ratio_hist), [2.0, 1.0])
    assert u.Unit(ratio_hist.xunit) == u.kg / u.m**2


@pytest.mark.parametrize(
    ("quantity", "bins", "expected", "unit"),
    [
        ("inverse_r", [0.0, 0.09, 0.15, 0.21], [1.0, 1.0, 2.0], "1/m"),
        ("1/r", [0.0, 0.09, 0.15, 0.21], [1.0, 1.0, 2.0], "1/m"),
        (
            "density_over_r",
            [0.0, 0.11, 0.2, 0.6],
            [1.0, 2.0, 1.0],
            "kg/m4",
        ),
        (
            "density_over_r2",
            [0.0, 0.01, 0.025, 0.1, 1.0],
            [1.0, 2.0, 1.0, 0.0],
            "kg/m5",
        ),
        ("rho/r", [0.0, 0.11, 0.2, 0.6], [1.0, 2.0, 1.0], "kg/m4"),
        ("rho/r^2", [0.0, 0.01, 0.025, 0.1, 1.0], [1.0, 2.0, 1.0, 0.0], "kg/m5"),
    ],
)
def test_added_geometric_histogram_quantities_and_aliases(
    quantity: str, bins: list[float], expected: list[float], unit: str
) -> None:
    # Distances are 5, 10, 13 and 5 m for the volume elements.
    result = _api_hist(
        _volume_density_setup(),
        quantity,
        bins=np.array(bins) * u.Unit(unit),
        target_id="A",
    )

    np.testing.assert_array_equal(_values(result), expected)
    assert u.Unit(result.xunit) == u.Unit(unit)


def test_existing_inverse_square_alias_is_exact_and_uses_same_values() -> None:
    result = _api_hist(
        _weighted_setup(SOURCES_3), "inverse_square_distance", bins=[0.0, 0.02, 0.05]
    )
    np.testing.assert_array_equal(_values(result), [2.0, 1.0])
    assert u.Unit(result.xunit) == u.m**-2


@pytest.mark.parametrize(
    "bins",
    [
        [0.0, 0.006, 0.012, 0.014],
        np.array([0.0, 6.0, 12.0, 14.0]) * u.m,
    ],
)
def test_distance_km_xunit_with_plain_or_quantity_bins(bins) -> None:
    hist = _api_hist(_weighted_setup(SOURCES_3), "distance", bins=bins, xunit="km")

    np.testing.assert_array_equal(_values(hist), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(
        hist.edges.to_value(u.km), [0.0, 0.006, 0.012, 0.014], rtol=1e-12, atol=0
    )
    assert u.Unit(hist.xunit) == u.km


@pytest.mark.parametrize(
    ("quantity", "expected_unit"),
    [
        ("density", u.kg / u.m**3),
        ("height", u.m),
        ("distance", u.m),
        ("inverse_square", u.Unit("1/m2")),
    ],
)
def test_default_xunit_per_quantity(quantity: str, expected_unit) -> None:
    setup = (
        _volume_density_setup() if quantity == "density" else _weighted_setup(SOURCES_3)
    )

    hist = _api_hist(setup, quantity, bins=2)

    assert u.Unit(hist.xunit) == expected_unit


@pytest.mark.parametrize(
    ("norm", "expected_counts"),
    [("frobenius", [0.0, 2.0]), ("spectral", [2.0, 0.0])],
)
def test_gg_kernel_norm_histogram_uses_explicit_per_source_norm(norm, expected_counts):
    setup = _weighted_setup(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        [3.0, 5.0],
        targets=_targets(((0.0, 0.0, 0.0),), ids=("A",)),
    )

    histogram = _api_hist(
        setup,
        "GG_kernel_norm",
        target_id="A",
        bins=[0.0, 0.275, 0.5],
        norm=norm,
    )

    assert isinstance(histogram, Histogram)
    np.testing.assert_array_equal(_values(histogram), expected_counts)
    assert u.Unit(histogram.xunit) == u.m**-3
    assert f"norm={norm}" in histogram.name


def test_gg_kernel_norm_histogram_requires_explicit_norm():
    setup = _weighted_setup(
        [[2.0, 0.0, 0.0]],
        [3.0],
        targets=_targets(((0.0, 0.0, 0.0),), ids=("A",)),
    )

    with pytest.raises(ValueError, match="norm must be 'frobenius' or 'spectral'"):
        _api_hist(setup, "gravity_gradient_kernel_norm", target_id="A")


@pytest.mark.parametrize(
    ("quantity", "kwargs"),
    [
        ("distance", {"xunit": "kg"}),
        ("density", {"xunit": "s"}),
        ("inverse_square", {"xunit": "m"}),
        ("height", {"bins": np.array([0.0, 1.0, 2.0]) * u.s}),
    ],
)
def test_incompatible_xunit_or_bins_units_raise(quantity: str, kwargs: dict) -> None:
    setup = (
        _volume_density_setup() if quantity == "density" else _weighted_setup(SOURCES_3)
    )
    arguments = {"bins": [0.0, 1.0, 2.0], **kwargs}

    with pytest.raises(ValueError, match=r"not (equivalent|compatible)"):
        _api_hist(setup, quantity, **arguments)


def test_distance_volume_weight_uses_element_volumes() -> None:
    # Distances 5, 10, 13 with volumes 1, 2, 4.
    setup = _volume_setup(SOURCES_3, [1.0, 2.0, 4.0], 1.0)

    hist = _api_hist(setup, "distance", bins=[0.0, 6.0, 12.0, 14.0], weight="volume")

    np.testing.assert_array_equal(_values(hist), [1.0, 2.0, 4.0])
    assert hist.unit == u.m**3


def test_underflow_overflow_and_last_edge_inclusion() -> None:
    setup = _weighted_setup(SOURCES_3)

    hist = _api_hist(setup, "distance", bins=[6.0, 12.0])
    # 5 is underflow, 10 is in the bin, 13 is overflow.
    np.testing.assert_array_equal(_values(hist), [1.0])
    assert _underflow(hist) == 1.0
    assert _overflow(hist) == 1.0

    # 13 equals the final edge; numpy places it in the last bin, not overflow.
    hist_edge = _api_hist(setup, "distance", bins=[0.0, 6.0, 12.0, 13.0])
    np.testing.assert_array_equal(_values(hist_edge), [1.0, 1.0, 1.0])
    assert _underflow(hist_edge) == 0.0
    assert _overflow(hist_edge) == 0.0


def test_single_point_histogram() -> None:
    hist = _api_hist(_weighted_setup([[0.0, 0.0, 0.0]]), "distance", bins=[0.0, 6.0])

    np.testing.assert_array_equal(_values(hist), [1.0])


def test_constant_data_uses_numpy_range_extension() -> None:
    # All three distances are exactly 5.
    setup = _weighted_setup([[0.0, 0.0, 0.0], [-6.0, -8.0, 0.0], [-3.0, -4.0, 5.0]])

    hist = _api_hist(setup, "distance", bins=3)

    # numpy widens a constant range to [5 - 0.5, 5 + 0.5] before splitting it.
    np.testing.assert_allclose(
        hist.edges.to_value(u.m), np.linspace(4.5, 5.5, 4), rtol=1e-14, atol=0
    )
    np.testing.assert_array_equal(_values(hist), [0.0, 3.0, 0.0])


def test_integer_bins_across_zero() -> None:
    setup = _weighted_setup(
        [[0.0, 0.0, -1.5], [0.0, 0.0, -0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 1.5]]
    )

    hist = _api_hist(setup, "height", bins=4)

    # Integer bins split the data range [-1.5, 1.5] into four equal bins.
    np.testing.assert_allclose(
        hist.edges.to_value(u.m), [-1.5, -0.75, 0.0, 0.75, 1.5], rtol=0, atol=1e-15
    )
    np.testing.assert_array_equal(_values(hist), [1.0, 1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# Frame handling
# ---------------------------------------------------------------------------


def _rotated_frame(parent: Frame = ROOT) -> Frame:
    return Frame(
        "S",
        origin=np.asarray(FRAME_ORIGIN, dtype=np.float64) * u.m,
        basis=ROT_Z,
        parent=parent,
    )


def test_distance_and_inverse_square_invariant_under_source_frame_change() -> None:
    frame_s = _rotated_frame()
    root_setup = _weighted_setup(SOURCES_3)
    s_setup = _weighted_setup(LOCAL_SOURCES_S, frame=frame_s)

    for setup in (root_setup, s_setup):
        np.testing.assert_array_equal(
            _values(_api_hist(setup, "distance", bins=[0.0, 6.0, 12.0, 14.0])),
            [1.0, 1.0, 1.0],
        )
        np.testing.assert_array_equal(
            _values(_api_hist(setup, "inverse_square", bins=[0.0, 0.02, 0.05])),
            [2.0, 1.0],
        )

    # An explicit frame changes only the validated connection, not the distances.
    explicit = _api_hist(s_setup, "distance", bins=[0.0, 6.0, 12.0, 14.0], frame=ROOT)
    np.testing.assert_array_equal(_values(explicit), [1.0, 1.0, 1.0])


def test_height_is_z_component_of_requested_frame() -> None:
    frame_s = _rotated_frame()
    setup = _weighted_setup(LOCAL_SOURCES_S, frame=frame_s)

    # Root z values are 0, 0 and 12.
    in_root = _api_hist(setup, "height", bins=[-1.0, 1.0, 13.0], frame=ROOT)
    np.testing.assert_array_equal(_values(in_root), [2.0, 1.0])
    assert _underflow(in_root) == 0.0

    # Local z values are -3, -3 and 9, so the root bins underflow two values.
    in_s = _api_hist(setup, "height", bins=[-4.0, 0.0, 10.0], frame=frame_s)
    np.testing.assert_array_equal(_values(in_s), [2.0, 1.0])


def test_frame_none_is_computation_frame_not_root() -> None:
    frame_s = _rotated_frame()
    setup = _weighted_setup(LOCAL_SOURCES_S, frame=frame_s)

    default = _api_hist(setup, "height", bins=[-1.0, 1.0, 13.0])
    explicit = _api_hist(setup, "height", bins=[-1.0, 1.0, 13.0], frame=frame_s)

    # Default frame gives local z = -3, -3, 9: two underflows and one value in bin 1.
    np.testing.assert_array_equal(_values(default), [0.0, 1.0])
    assert _underflow(default) == 2.0
    np.testing.assert_array_equal(_values(default), _values(explicit))
    assert _underflow(default) == _underflow(explicit)


def test_target_in_rotated_frame_gives_root_distances() -> None:
    # Local target (-4, 3, 0) in the rotated frame is (-3, -4, 0) in the root frame.
    target_frame = Frame("T", basis=ROT_Z, parent=ROOT)
    targets = TargetPoints(np.array([[-4.0, 3.0, 0.0]]) * u.m, ("A",), target_frame)
    setup = _weighted_setup(SOURCES_3, targets=targets)

    hist = _api_hist(setup, "distance", bins=[0.0, 6.0, 12.0, 14.0])

    np.testing.assert_array_equal(_values(hist), [1.0, 1.0, 1.0])


@pytest.mark.parametrize(
    ("quantity", "kwargs"),
    [
        ("distance", {}),
        ("inverse_square", {}),
        ("height", {"bins": [0.0, 1.0]}),
    ],
)
def test_frame_from_other_root_raises(quantity: str, kwargs: dict) -> None:
    setup = _weighted_setup(SOURCES_3)
    arguments = {"bins": [0.0, 20.0], **kwargs, "frame": Frame("other")}

    with pytest.raises(ValueError, match="disconnected roots"):
        _api_hist(setup, quantity, **arguments)


# ---------------------------------------------------------------------------
# Provenance: density and volume are verified against the Source model
# ---------------------------------------------------------------------------


def test_volume_scalar_density_expands_per_element() -> None:
    setup = _volume_setup(POSITIONS_4, VOLUMES_4, 2.0)

    density = _api_hist(setup, "density", bins=[0.0, 1.0, 3.0])
    volume = _api_hist(setup, "density", bins=[0.0, 1.0, 3.0], weight="volume")

    # All four elements have density 2 kg/m^3; their volumes sum to 10 m^3.
    np.testing.assert_array_equal(_values(density), [0.0, 4.0])
    np.testing.assert_array_equal(_values(volume), [0.0, 10.0])


def _mismatched_setup(bound_positions, bound_weights):
    """Build a ResolvedEvaluation whose bound weights disagree with the model."""
    model = point_sum(
        VolumeElementSource(POSITIONS_4, VOLUMES_4, 2.0),
        frame=ROOT,
        field_definition=_field("volume"),
    )
    binding = _StaticPointSourceBinding(
        bound_positions,
        bound_weights,
        ROOT,
        model.spec.field_definition,
        "VolumeElementSource",
        {},
    )
    resolved = ResolvedEvaluation(model, _targets(), ROOT, ROOT, OBSERVABLES, binding)
    return EvaluationSetup(resolved)


def test_direct_resolved_mismatch_rejects_density_and_volume() -> None:
    # Model density is 2 kg/m^3; the bound weights correspond to density 3.
    setup = _mismatched_setup(POSITIONS_4, 3.0 * np.array(VOLUMES_4))

    with pytest.raises(ValueError, match="does not reproduce"):
        _api_hist(setup, "density", bins=2)
    with pytest.raises(ValueError, match="does not reproduce"):
        _api_hist(setup, "density", bins=2, weight="volume")


def test_permuted_bound_elements_are_rejected() -> None:
    # Same multiset of positions and weights, but in reversed element order.
    setup = _mismatched_setup(
        np.array(POSITIONS_4)[::-1], (2.0 * np.array(VOLUMES_4))[::-1]
    )

    with pytest.raises(ValueError, match="element order"):
        _api_hist(setup, "density", bins=2)


def test_position_quantities_do_not_check_provenance() -> None:
    setup = _mismatched_setup(POSITIONS_4, 3.0 * np.array(VOLUMES_4))

    # z values are 0, 0, 12 and 0.
    height = _api_hist(setup, "height", bins=[-1.0, 1.0, 13.0])
    np.testing.assert_array_equal(_values(height), [3.0, 1.0])
    count = _api_hist(setup, "inverse_square", bins=[0.0, 1.0])
    np.testing.assert_array_equal(_values(count), [4.0])
    distance = _api_hist(setup, "distance", bins=[0.0, 100.0])
    np.testing.assert_array_equal(_values(distance), [4.0])


@pytest.mark.parametrize(
    ("builder", "quantity", "kwargs", "type_name"),
    [
        (
            lambda: _weighted_setup(SOURCES_3),
            "density",
            {},
            "WeightedPointSource",
        ),
        (
            lambda: _weighted_setup(SOURCES_3),
            "height",
            {"weight": "volume"},
            "WeightedPointSource",
        ),
        (
            lambda: point_sum(PointMass(2.0, [0.0, 0.0, 0.0]), frame=ROOT).at(
                _targets()
            ),
            "density",
            {},
            "PointMass",
        ),
        (
            lambda: point_sum(
                PointMassCloud([1.0, 1.0, 1.0], SOURCES_3), frame=ROOT
            ).at(_targets()),
            "density",
            {},
            "PointMassCloud",
        ),
    ],
)
def test_sources_without_density_or_volume_reject_those_weights(
    builder, quantity: str, kwargs: dict, type_name: str
) -> None:
    setup = builder()

    with pytest.raises(ValueError, match=type_name):
        _api_hist(setup, quantity, bins=2, **kwargs)


@pytest.mark.parametrize(
    "quantity", ["density", "height", "distance", "inverse_square"]
)
def test_analytic_model_is_unsupported_for_every_quantity(quantity: str) -> None:
    sphere = Sphere(np.zeros(3) * u.m, 1.0 * u.m, ROOT)
    setup = uniform_sphere(sphere, 1000.0 * u.kg / u.m**3).at(_targets())

    with pytest.raises(ValueError, match="discrete point source"):
        _api_hist(setup, quantity, bins=2)


@pytest.mark.parametrize("density", [1.0 + 0.5j, np.array([1.0, 2.0, 3.0]) + 0.5j])
def test_complex_density_fails_at_point_sum_construction(density) -> None:
    with pytest.raises(ValueError, match="complex"):
        point_sum(
            VolumeElementSource(SOURCES_3, [1.0, 2.0, 3.0], density),
            frame=ROOT,
            field_definition=_field("volume"),
        )


def _discrete_setup(weight_scale: float = 1.0, *, metadata=None):
    frame = Frame("disc")
    sphere = Sphere(np.zeros(3) * u.m, 1.0 * u.m, frame)
    meta = {"cell_volume_m3": 0.5, "source_count": 3} if metadata is None else metadata
    # Density 4 kg/m^3 times cell volume 0.5 m^3 gives weight 2 kg per element.
    source = DiscreteSource(
        sphere,
        4.0 * u.kg / u.m**3,
        frame,
        DISCRETE_POSITIONS,
        np.full(3, 2.0) * weight_scale,
        meta,
    )
    targets = _targets(((10.0, 0.0, 0.0),), frame=frame)
    return point_sum(source).at(targets)


def test_discrete_density_accepts_weights_within_relative_1e12() -> None:
    setup = _discrete_setup(1.0 + 5e-13)

    hist = _api_hist(setup, "density", bins=[3.9, 4.1])

    np.testing.assert_array_equal(_values(hist), [3.0])


def test_discrete_density_rejects_weights_beyond_relative_1e12() -> None:
    setup = _discrete_setup(1.0 + 1e-10)

    with pytest.raises(ValueError, match="disagree"):
        _api_hist(setup, "density", bins=[3.9, 4.1])


def test_discrete_volume_total_is_cell_volume_times_count() -> None:
    setup = _discrete_setup()

    hist = _api_hist(setup, "density", bins=[3.9, 4.1], weight="volume")

    # Three cells of 0.5 m^3 each.
    assert float(np.sum(_values(hist))) == pytest.approx(1.5, rel=1e-12)


@pytest.mark.parametrize("weight", ["count", "volume"])
def test_discrete_without_cell_volume_rejects_density_and_volume(weight: str) -> None:
    setup = _discrete_setup(metadata={"source_count": 3})

    with pytest.raises(ValueError, match="cell_volume_m3"):
        _api_hist(setup, "density", bins=[3.9, 4.1], weight=weight)


def test_discrete_without_cell_volume_keeps_position_quantities() -> None:
    setup = _discrete_setup(metadata={"source_count": 3})

    distance = _api_hist(setup, "distance", bins=[0.0, 20.0])
    height = _api_hist(setup, "height", bins=[-1.0, 1.0])
    inverse = _api_hist(setup, "inverse_square", bins=[0.0, 1.0])
    count = _api_hist(setup, "distance", bins=[0.0, 20.0], weight="count")

    np.testing.assert_array_equal(_values(distance), [3.0])
    np.testing.assert_array_equal(_values(height), [3.0])
    np.testing.assert_array_equal(_values(inverse), [3.0])
    np.testing.assert_array_equal(_values(count), [3.0])


@pytest.mark.parametrize(
    "builder",
    [
        lambda: _weighted_setup(SOURCES_3),
        lambda: point_sum(PointMassCloud([1.0, 1.0, 1.0], SOURCES_3), frame=ROOT).at(
            _targets()
        ),
    ],
)
def test_partial_sources_support_position_quantities(builder) -> None:
    setup = builder()

    np.testing.assert_array_equal(
        _values(_api_hist(setup, "distance", bins=[0.0, 6.0, 12.0, 14.0])),
        [1.0, 1.0, 1.0],
    )
    np.testing.assert_array_equal(
        _values(_api_hist(setup, "height", bins=[-1.0, 1.0, 13.0])), [2.0, 1.0]
    )
    np.testing.assert_array_equal(
        _values(_api_hist(setup, "inverse_square", bins=[0.0, 0.02, 0.05])),
        [2.0, 1.0],
    )


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------


def test_single_target_is_selected_without_target_id() -> None:
    hist = _api_hist(
        _weighted_setup(SOURCES_3), "distance", bins=[0.0, 6.0, 12.0, 14.0]
    )

    np.testing.assert_array_equal(_values(hist), [1.0, 1.0, 1.0])


def test_multiple_targets_require_target_id_and_select_by_id() -> None:
    # Target A (-3,-4,0) has distances 5 and 10.
    # Target B (-6,-8,0) has distances 10 and 15.
    setup = _weighted_setup(
        [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]],
        targets=_targets([ROOT_TARGET, (-6.0, -8.0, 0.0)], ids=("A", "B")),
    )

    with pytest.raises(ValueError, match="target_id is required"):
        _api_hist(setup, "distance", bins=[0.0, 7.0, 12.0, 16.0])

    hist_a = _api_hist(setup, "distance", bins=[0.0, 7.0, 12.0, 16.0], target_id="A")
    hist_b = _api_hist(setup, "distance", bins=[0.0, 7.0, 12.0, 16.0], target_id="B")
    np.testing.assert_array_equal(_values(hist_a), [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(_values(hist_b), [0.0, 1.0, 1.0])


def test_unknown_target_id_raises() -> None:
    with pytest.raises(ValueError, match="unknown target_id"):
        _api_hist(_weighted_setup(SOURCES_3), "distance", bins=2, target_id="Z")


@pytest.mark.parametrize("quantity", ["density", "height"])
def test_target_id_rejected_for_target_independent_quantities(quantity: str) -> None:
    setup = (
        _volume_density_setup() if quantity == "density" else _weighted_setup(SOURCES_3)
    )

    with pytest.raises(ValueError, match="applies only"):
        _api_hist(setup, quantity, bins=2, target_id="A")


def test_zero_targets_density_and_height_still_work() -> None:
    no_targets = _targets(np.empty((0, 3)), ids=())
    volume_setup = _volume_setup(
        POSITIONS_4, VOLUMES_4, np.array(DENSITIES_4), targets=no_targets
    )
    weighted_setup = _weighted_setup(SOURCES_3, targets=no_targets)

    density = _api_hist(volume_setup, "density", bins=[0.0, 1.0, 2.0, 3.0])
    height = _api_hist(weighted_setup, "height", bins=[-1.0, 1.0, 13.0])

    np.testing.assert_array_equal(_values(density), [1.0, 2.0, 1.0])
    np.testing.assert_array_equal(_values(height), [2.0, 1.0])


# ---------------------------------------------------------------------------
# Numerical safety for distances and inverse squares
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quantity", ["distance", "inverse_square"])
def test_collision_raises_structured_error_with_indices(quantity: str) -> None:
    # The colliding source is at index 1; the target is at the origin.
    setup = _weighted_setup(
        [[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        targets=_targets(((0.0, 0.0, 0.0),), ids=("hit",)),
    )

    with pytest.raises(CoordinatePrecisionError) as caught:
        _api_hist(setup, quantity, bins=2)

    assert caught.value.target_id == "hit"
    assert caught.value.source_index == 1
    assert caught.value.reason == "collision"


def test_cross_frame_near_coincidence_raises_precision_error() -> None:
    source_frame = Frame("near-local", origin=[1e6, 0.0, 0.0] * u.m, parent=ROOT)
    setup = _weighted_setup(
        [[0.0, 0.0, 0.0]],
        frame=source_frame,
        targets=_targets(((1e6 + 1e-8, 0.0, 0.0),), ids=("near",)),
    )

    with pytest.raises(CoordinatePrecisionError) as caught:
        _api_hist(setup, "distance", bins=2)

    assert caught.value.reason == "coordinate_precision"
    assert caught.value.target_id == "near"
    assert caught.value.source_index == 0


def test_nonfinite_coordinate_difference_is_rejected() -> None:
    # The difference 2e308 overflows float64 to infinity.
    setup = _weighted_setup(
        [[1e308, 0.0, 0.0]],
        targets=_targets(((-1e308, 0.0, 0.0),), ids=("far",)),
    )

    with pytest.raises(ValueError, match="pair distance is not finite"):
        _api_hist(setup, "distance", bins=2)


@pytest.mark.parametrize("quantity", ["distance", "inverse_square"])
def test_squared_sum_overflow_is_rejected(quantity: str) -> None:
    # The coordinate difference 1e200 is finite, but its square overflows.
    setup = _weighted_setup(
        [[1e200, 0.0, 0.0]],
        targets=_targets(((0.0, 0.0, 0.0),), ids=("origin",)),
    )

    with pytest.raises(ValueError, match="squared separation overflows"):
        _api_hist(setup, quantity, bins=2)


@pytest.mark.parametrize("separation", [1e-170, 1e-160])
@pytest.mark.parametrize("quantity", ["distance", "inverse_square"])
def test_squared_underflow_of_nonzero_separation_is_rejected(
    quantity: str, separation: float
) -> None:
    # 1e-170 squares to zero. 1e-160 squares to a subnormal value.
    setup = _weighted_setup(
        [[separation, 0.0, 0.0]],
        targets=_targets(((0.0, 0.0, 0.0),), ids=("origin",)),
    )

    with pytest.raises(ValueError, match="subnormal"):
        _api_hist(setup, quantity, bins=2)


def test_tiny_but_representable_separation_is_accepted() -> None:
    # Squared separation 1e-200 is normal, and 1/r^2 = 1e200 is finite.
    setup = _weighted_setup(
        [[1e-100, 0.0, 0.0]],
        targets=_targets(((0.0, 0.0, 0.0),), ids=("origin",)),
    )

    distance = _api_hist(setup, "distance", bins=[0.0, 1e-99])
    inverse = _api_hist(setup, "inverse_square", bins=[0.0, 1e199, 1e201])

    np.testing.assert_array_equal(_values(distance), [1.0])
    np.testing.assert_array_equal(_values(inverse), [0.0, 1.0])


def _collision_setup():
    return _weighted_setup(
        [[0.0, 0.0, 0.0]], targets=_targets(((0.0, 0.0, 0.0),), ids=("hit",))
    )


def _near_setup():
    source_frame = Frame("near-local", origin=[1e6, 0.0, 0.0] * u.m, parent=ROOT)
    return _weighted_setup(
        [[0.0, 0.0, 0.0]],
        frame=source_frame,
        targets=_targets(((1e6 + 1e-8, 0.0, 0.0),), ids=("near",)),
    )


def _nonfinite_setup():
    return _weighted_setup(
        [[1e308, 0.0, 0.0]], targets=_targets(((-1e308, 0.0, 0.0),), ids=("far",))
    )


def _overflow_setup():
    return _weighted_setup(
        [[1e200, 0.0, 0.0]], targets=_targets(((0.0, 0.0, 0.0),), ids=("origin",))
    )


# Each builder fails in both validate() and hist(). The expected messages differ
# by path: validate() wraps the failure in an EvaluationError for the non-precision
# cases, and hist() raises its own ValueError.
@pytest.mark.parametrize(
    ("builder", "validate_re", "hist_re"),
    [
        (_collision_setup, r"^collision:", r"^collision:"),
        (_near_setup, r"^coordinate_precision:", r"^coordinate_precision:"),
        (
            _nonfinite_setup,
            r"pair distance is not finite",
            r"^pair distance is not finite",
        ),
        (_overflow_setup, r"overflow", r"^squared separation overflows float64"),
    ],
)
def test_hist_rejects_the_same_pairs_as_validate(
    builder, validate_re: str, hist_re: str
) -> None:
    setup = builder()

    with pytest.raises(ValueError, match=validate_re) as validate_caught:
        setup.validate()
    with pytest.raises(ValueError, match=hist_re) as hist_caught:
        _api_hist(setup, "distance", bins=2)

    assert isinstance(hist_caught.value, CoordinatePrecisionError) is isinstance(
        validate_caught.value, CoordinatePrecisionError
    )


@pytest.mark.parametrize("builder", [_collision_setup, _near_setup])
def test_hist_precision_error_fields_match_validate(builder) -> None:
    setup = builder()

    with pytest.raises(CoordinatePrecisionError) as validate_caught:
        setup.validate()
    with pytest.raises(CoordinatePrecisionError) as hist_caught:
        _api_hist(setup, "distance", bins=2)

    assert hist_caught.value.reason == validate_caught.value.reason
    assert hist_caught.value.target_id == validate_caught.value.target_id
    assert hist_caught.value.source_index == validate_caught.value.source_index


@pytest.mark.parametrize("builder", [_nonfinite_setup, _overflow_setup])
def test_hist_non_precision_pair_errors_are_plain_value_errors(builder) -> None:
    with pytest.raises(ValueError) as caught:
        _api_hist(builder(), "distance", bins=2)

    assert not isinstance(caught.value, CoordinatePrecisionError)


def test_other_target_problem_does_not_block_selected_target() -> None:
    # Target B collides with source 0, so validate() fails, but target A is clean.
    setup = _weighted_setup(
        SOURCES_3,
        targets=_targets([ROOT_TARGET, (0.0, 0.0, 0.0)], ids=("A", "B")),
    )

    hist = _api_hist(setup, "distance", bins=[0.0, 6.0, 12.0, 14.0], target_id="A")

    np.testing.assert_array_equal(_values(hist), [1.0, 1.0, 1.0])
    with pytest.raises(CoordinatePrecisionError) as caught:
        setup.validate()
    assert caught.value.target_id == "B"
    assert caught.value.source_index == 0
    assert caught.value.reason == "collision"


# ---------------------------------------------------------------------------
# Bins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("edges", [[6.0, 0.0, 12.0], [0.0, 6.0, 3.0, 12.0]])
def test_unsorted_or_nonmonotonic_edges_are_rejected(edges) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _api_hist(_weighted_setup(SOURCES_3), "distance", bins=edges)


@pytest.mark.parametrize(
    ("bins", "match"),
    [
        (float("nan"), r"must be an int"),
        (float("inf"), r"must be an int"),
        ([0.0, np.inf, 12.0], r"must be finite"),
    ],
)
def test_nonfinite_bins_are_rejected(bins, match: str) -> None:
    # Scalar nan/inf fail the edge-shape check; a nonfinite edge fails the finite check.
    with pytest.raises(ValueError, match=match):
        _api_hist(_weighted_setup(SOURCES_3), "distance", bins=bins)


@pytest.mark.parametrize("bins", [True, False])
def test_bool_bins_are_rejected(bins: bool) -> None:
    with pytest.raises(ValueError, match="boolean"):
        _api_hist(_weighted_setup(SOURCES_3), "distance", bins=bins)


@pytest.mark.parametrize(
    ("bins", "match"),
    [
        (0, r"between 1 and"),
        (-3, r"between 1 and"),
        (MAX_BINS + 1, r"between 1 and"),
        (np.linspace(0.0, 1.0, MAX_BINS + 2), r"exceed the limit"),
    ],
)
def test_bin_count_above_limit_or_nonpositive_is_rejected(bins, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _api_hist(_weighted_setup(SOURCES_3), "distance", bins=bins)


def test_limit_constants_match_the_plan() -> None:
    module = _hist_module()

    assert module._MAX_BINS == MAX_BINS
    assert module._MAX_PLOT_BINS == 100_000


@pytest.mark.parametrize("name", sorted(_AUTO_DATASETS))
def test_auto_edges_match_numpy_and_estimate(name: str) -> None:
    values = _AUTO_DATASETS[name]()

    hist = _api_hist(_height_setup(values), "height", bins="auto")

    expected = np.histogram_bin_edges(values, "auto")
    np.testing.assert_allclose(hist.edges.to_value(u.m), expected, rtol=1e-12, atol=0)
    assert _hist_module().estimate_auto_bin_count(values) == len(expected) - 1


def test_auto_iqr_zero_extreme_outlier_completes() -> None:
    # 999,999 zeros and one value of 1e12. NumPy gives 21 bins under the legacy
    # rule and about 2000 under the relaxed rule.
    values = np.zeros(1_000_000)
    values[-1] = 1e12

    hist = _api_hist(_height_setup(values), "height", bins="auto")

    expected = np.histogram_bin_edges(values, "auto")
    np.testing.assert_allclose(hist.edges.to_value(u.m), expected, rtol=1e-12, atol=0)
    assert len(expected) - 1 in {21, 2000, 2001}
    assert _values(hist).sum() == 1_000_000
    assert _hist_module().estimate_auto_bin_count(values) == len(expected) - 1


def test_auto_positive_iqr_extreme_outlier_follows_numpy_rule() -> None:
    values = np.concatenate([np.random.default_rng(3).normal(size=1_000_000), [1e12]])
    setup = _height_setup(values)
    rule = _numpy_auto_rule()

    if rule == "legacy":
        # Estimated count is about 4e13, which is far above MAX_BINS. The
        # estimate must reject before any large array is allocated.
        tracemalloc.start()
        try:
            with pytest.raises(ValueError, match="would produce"):
                _api_hist(setup, "height", bins="auto")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 64 * 1024 * 1024
    else:
        hist = _api_hist(setup, "height", bins="auto")
        expected = np.histogram_bin_edges(values, "auto")
        np.testing.assert_allclose(
            hist.edges.to_value(u.m), expected, rtol=1e-12, atol=0
        )
        assert len(expected) - 1 <= 2 * np.sqrt(values.size) + 1


def test_estimator_legacy_rule_rejects_positive_iqr_outlier() -> None:
    values = np.concatenate([np.random.default_rng(4).normal(size=1000), [1e12]])

    with pytest.raises(ValueError, match="would produce"):
        _hist_module().estimate_auto_bin_count(values, rule="legacy")


def test_estimator_relaxed_rule_is_bounded_by_two_sqrt_n() -> None:
    values = np.concatenate([np.random.default_rng(4).normal(size=1000), [1e12]])

    count = _hist_module().estimate_auto_bin_count(values, rule="relaxed")

    assert 1 <= count <= MAX_BINS
    assert count <= 2 * np.sqrt(values.size) + 1


# ---------------------------------------------------------------------------
# Side effects and returned object
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("weight", ["count", "volume"])
@pytest.mark.parametrize(
    ("quantity", "bins"),
    [
        ("density", [0.0, 1.0, 2.0, 3.0]),
        ("height", [-1.0, 1.0, 13.0]),
        ("distance", [0.0, 6.0, 12.0, 14.0]),
        ("inverse_square", [0.0, 0.02, 0.05]),
        ("distance", "auto"),
    ],
)
def test_inputs_are_not_mutated(quantity: str, bins, weight: str) -> None:
    setup = _volume_density_setup()
    before = _snapshot(setup)
    # Target-dependent quantities select the single target explicitly.
    target_kwargs = (
        {"target_id": "A"} if quantity in {"distance", "inverse_square"} else {}
    )

    _api_hist(setup, quantity, bins=bins, weight=weight, **target_kwargs)

    assert _snapshot(setup) == before


def test_returned_histogram_is_independent_of_later_calls() -> None:
    setup = _volume_density_setup()
    bins = [0.0, 1.0, 2.0, 3.0] * u.kg / u.m**3
    first = _api_hist(setup, "density", bins=bins)
    expected_values = _values(first).copy()
    expected_edges = first.edges.to_value(u.kg / u.m**3).copy()

    first.values.value[:] = -1.0
    first.edges.value[:] = -1.0
    second = _api_hist(setup, "density", bins=bins)

    np.testing.assert_array_equal(_values(second), expected_values)
    np.testing.assert_array_equal(second.edges.to_value(u.kg / u.m**3), expected_edges)


@pytest.mark.parametrize(
    ("quantity", "target_id", "expected_tokens"),
    [
        ("distance", "A", ("distance", "A")),
        ("height", None, ("height",)),
    ],
)
def test_histogram_metadata_has_no_sumw2_and_names_quantity(
    quantity: str, target_id, expected_tokens
) -> None:
    kwargs = {"bins": 2}
    if target_id is not None:
        kwargs["target_id"] = target_id

    hist = _api_hist(_weighted_setup(SOURCES_3), quantity, **kwargs)

    assert hist.sumw2 is None
    for token in expected_tokens:
        assert token in hist.name


@pytest.mark.parametrize(
    ("quantity", "kwargs"),
    [("energy", {}), ("density", {"weight": "mass"})],
)
def test_unknown_quantity_or_weight_is_rejected(quantity: str, kwargs: dict) -> None:
    with pytest.raises(ValueError, match="must be one of"):
        _api_hist(_volume_density_setup(), quantity, bins=2, **kwargs)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def _peak_bytes(call) -> int:
    tracemalloc.start()
    try:
        call()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


@pytest.mark.parametrize("count", [100_000, 1_000_000])
def test_distance_peak_memory_is_a_few_source_sized_arrays(count: int) -> None:
    rng = np.random.default_rng(1)
    positions = rng.normal(size=(count, 3))
    setup = _weighted_setup(
        positions,
        np.ones(count),
        targets=_targets(((100.0, 0.0, 0.0),), ids=("t0",)),
    )
    bins = np.linspace(0.0, 200.0, 21)

    peak = _peak_bytes(lambda: _api_hist(setup, "distance", bins=bins))

    # Budget: at most 12 float64 arrays of length N.
    assert peak <= 12 * 8 * count


def test_peak_memory_does_not_grow_with_target_count() -> None:
    rng = np.random.default_rng(2)
    count = 100_000
    positions = rng.normal(size=(count, 3))
    weights = np.ones(count)
    bins = np.linspace(0.0, 400.0, 41)

    one = _weighted_setup(
        positions, weights, targets=_targets(((100.0, 0.0, 0.0),), ids=("t0",))
    )
    many_points = [(100.0, 0.0, 0.0)] + [(200.0 + i, 0.0, 0.0) for i in range(49)]
    many = _weighted_setup(
        positions,
        weights,
        targets=_targets(many_points, ids=tuple(f"t{i}" for i in range(50))),
    )

    peak_one = _peak_bytes(
        lambda: _api_hist(one, "distance", bins=bins, target_id="t0")
    )
    peak_many = _peak_bytes(
        lambda: _api_hist(many, "distance", bins=bins, target_id="t0")
    )

    # Only the selected target may be processed, so 50 targets must not
    # multiply the peak. Allow 2 MiB of slack for interpreter overhead.
    assert peak_many <= peak_one + 2 * 1024 * 1024


# ---------------------------------------------------------------------------
# Input, unit and provenance validation
# ---------------------------------------------------------------------------


def _tiny_inverse_square_setup(*positions):
    # A source 1.5e-154 m from the origin gives 1/r^2 = 4.44e307 m^-2. That is
    # finite in SI units, but about 1e6 times larger in 1/km^2, which overflows.
    points = positions or ([1.5e-154, 0.0, 0.0],)
    return _weighted_setup(
        points, targets=_targets(((0.0, 0.0, 0.0),), ids=("origin",))
    )


@pytest.mark.parametrize("bins", [[0.0, 1e308], "auto", 2])
def test_xunit_overflow_is_rejected_on_every_bin_path(bins) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="overflows float64"):
            _api_hist(
                _tiny_inverse_square_setup(), "inverse_square", bins=bins, xunit="1/km2"
            )


def test_xunit_overflow_names_the_first_offending_element() -> None:
    setup = _tiny_inverse_square_setup([10.0, 0.0, 0.0], [1.5e-154, 0.0, 0.0])

    with pytest.raises(ValueError, match="element index 1"):
        _api_hist(setup, "inverse_square", bins=[0.0, 1e308], xunit="1/km2")


def test_height_xunit_overflow_is_rejected() -> None:
    with pytest.raises(ValueError, match="overflows float64"):
        _api_hist(_height_setup([1e305]), "height", bins=[0.0, 1e308], xunit="nm")


# ---------------------------------------------------------------------------
# xunit conversion: exact zero, underflow and conversion-factor checks
# ---------------------------------------------------------------------------


def _density_setup_from(densities):
    count = len(densities)
    positions = [[float(i), 0.0, 0.0] for i in range(count)]
    return _volume_setup(
        positions, np.ones(count), np.asarray(densities, dtype=np.float64)
    )


def test_exact_zero_density_is_accepted_and_stays_zero() -> None:
    converted = _hist_module()._to_xunit(np.array([0.0]), u.kg / u.m**3, u.g / u.cm**3)
    np.testing.assert_array_equal(converted, [0.0])

    hist = _api_hist(
        _density_setup_from([0.0]), "density", bins=[0.0, 1.0], xunit="g/cm3"
    )
    np.testing.assert_array_equal(_values(hist), [1.0])
    assert _underflow(hist) == 0.0


@pytest.mark.parametrize("density", [5e-324, 1e-305])
def test_tiny_density_that_underflows_after_conversion_is_rejected(density) -> None:
    with pytest.raises(ValueError, match="underflows float64 at element index 0"):
        _api_hist(
            _density_setup_from([density]), "density", bins=[0.0, 1.0], xunit="g/cm3"
        )


def test_first_underflowing_element_is_reported() -> None:
    setup = _density_setup_from([0.0, 1e-305, 5e-324])

    with pytest.raises(ValueError, match="underflows float64 at element index 1"):
        _api_hist(setup, "density", bins=[0.0, 1.0], xunit="g/cm3")


def test_quantity_bin_that_underflows_names_the_edge_index() -> None:
    with pytest.raises(ValueError, match="underflows float64 at bin edge index 1"):
        _api_hist(
            _volume_density_setup(),
            "density",
            bins=[0.0, 1e-323] * u.kg / u.m**3,
            xunit="g/cm3",
        )


def test_quantity_bins_that_convert_without_underflow_are_accepted() -> None:
    hist = _api_hist(
        _volume_density_setup(),
        "density",
        bins=[0.0, 1e-3] * u.kg / u.m**3,
        xunit="g/cm3",
    )

    assert u.Unit(hist.xunit) == u.g / u.cm**3
    np.testing.assert_allclose(
        hist.edges.to_value(u.g / u.cm**3), [0.0, 1e-6], rtol=1e-12
    )


def test_nonfinite_quantity_edge_is_reported_as_nonfinite_not_overflow() -> None:
    with pytest.raises(
        ValueError, match="bin edges must be finite; edge index 1 is not"
    ):
        _api_hist(
            _volume_density_setup(),
            "density",
            bins=[0.0, np.inf] * u.kg / u.m**3,
            xunit="g/cm3",
        )


def test_subnormal_conversion_factor_is_rejected() -> None:
    # 1e308 kg/m^3 as a CompositeUnit: powers are numeric, so m^-3 is [1, -3].
    # The factor kg/m^3 -> 1e308 kg/m^3 is 1e-308, which is subnormal.
    xunit = u.CompositeUnit(1e308, [u.kg, u.m], [1, -3])

    with pytest.raises(ValueError, match="not representable in float64"):
        _api_hist(_volume_density_setup(), "density", bins=[0.0, 1.0], xunit=xunit)


def test_overflowing_conversion_factor_is_rejected() -> None:
    # 1e-310 kg/m^3 as a CompositeUnit. The factor from kg/m^3 to this unit
    # is 1e310, which overflows float64.
    xunit = u.CompositeUnit(1e-310, [u.kg, u.m], [1, -3])

    with pytest.raises(ValueError, match="not representable in float64"):
        _api_hist(_volume_density_setup(), "density", bins=[0.0, 1.0], xunit=xunit)


def test_subnormal_quantity_bins_factor_is_rejected() -> None:
    # The factor kg/km^3 -> 1e300 kg/m^3 is 1e-309 (subnormal). The density
    # values themselves convert to normal float64 values with the factor 1e-300.
    xunit = u.CompositeUnit(1e300, [u.kg, u.m], [1, -3])

    with pytest.raises(ValueError, match="not representable in float64"):
        _api_hist(
            _volume_density_setup(),
            "density",
            bins=[0.0, 1.0] * u.kg / u.km**3,
            xunit=xunit,
        )


def test_overflowing_quantity_bins_factor_is_rejected() -> None:
    # The factor from the bin unit (1e9 kg/m^3) to xunit (1e-300 kg/m^3) is
    # 1e309, which overflows float64. The density values convert with the
    # finite factor 1e300 and stay normal.
    bins_unit = u.CompositeUnit(1e9, [u.kg, u.m], [1, -3])
    xunit = u.CompositeUnit(1e-300, [u.kg, u.m], [1, -3])

    with pytest.raises(ValueError, match="not representable in float64"):
        _api_hist(
            _volume_density_setup(),
            "density",
            bins=[0.0, 1.0] * bins_unit,
            xunit=xunit,
        )


@pytest.mark.parametrize(
    "bins",
    [np.array([0.0, 1.0, 2.0 + 1.0j]), np.array([0.0, 1.0, 2.0 + 1.0j]) * u.m],
)
def test_complex_bins_are_rejected(bins) -> None:
    with pytest.raises(ValueError, match="complex"):
        _api_hist(_height_setup([0.5]), "height", bins=bins)


@pytest.mark.parametrize("xunit", [u.mag(u.m), "nonsense", 5])
def test_unparseable_or_incompatible_xunit_is_a_value_error(xunit) -> None:
    with pytest.raises(ValueError, match="xunit"):
        _api_hist(_height_setup([0.5]), "height", bins=2, xunit=xunit)


@pytest.mark.parametrize("value", [True, "0.5"])
def test_cell_volume_m3_must_be_a_real_number(value) -> None:
    setup = _discrete_setup(metadata={"cell_volume_m3": value, "source_count": 3})

    with pytest.raises(ValueError, match="must be a real number"):
        _api_hist(setup, "density", bins=[3.9, 4.1])


def test_auto_range_overflow_has_its_own_message() -> None:
    values = np.array([-1e308, 1e308])

    with pytest.raises(ValueError, match="range is not finite or overflows float64"):
        _hist_module().estimate_auto_bin_count(values)
    with pytest.raises(ValueError, match="range is not finite or overflows float64"):
        _api_hist(_height_setup(values), "height", bins="auto")


def test_density_volume_provenance_is_verified_once(monkeypatch) -> None:
    from gragra._plotting import _diagnostics

    original = _diagnostics._verify_volume_elements
    calls = []

    def counting(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(_diagnostics, "_verify_volume_elements", counting)

    hist = _api_hist(
        _volume_density_setup(), "density", bins=[0.0, 1.0, 2.0, 3.0], weight="volume"
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(_values(hist), [1.0, 5.0, 4.0])


def test_legacy_rule_rejects_outlier_before_allocating_edges(monkeypatch) -> None:
    from gragra._plotting import _hist

    monkeypatch.setattr(_hist, "_numpy_auto_rule", lambda: "legacy")
    values = np.concatenate([np.random.default_rng(5).normal(size=100_000), [1e12]])
    setup = _height_setup(values)

    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="would produce"):
            _api_hist(setup, "height", bins="auto")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # The legacy estimate is about 1e13 bins. Only a few copies of the
    # 100,000 values may be allocated before the rejection.
    assert peak < 16 * 1024 * 1024
