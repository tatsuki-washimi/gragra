"""Static source quantity registry and geometry contracts."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from astropy import units as u

from gragra._evaluation_setup import CoordinatePrecisionError
from gragra._plotting._source_quantities import (
    QUANTITY_SPECS,
    StaticSourceQuantityData,
    normalize_targets_m,
    resolve_quantity,
    source_quantity_data,
    source_radial_membership,
)
from gragra.geometry import Frame
from gragra.gravity_sources import VolumeElementSource, WeightedPointSource
from gragra.gravity_sources._models import point_sum
from gragra.results import Contribution, GravityFieldDefinition, PotentialReference
from gragra.targets import TargetPoints as EvaluationTargetPoints
from gragra.targets.points import TargetPoints as NumericTargetPoints

ROOT = Frame("source-quantity-root")


def _setup(positions, weights, *, volumes=None, density=None):
    definition = GravityFieldDefinition.source_field(
        "test", Contribution("total"), PotentialReference.ZERO_AT_INFINITY
    )
    if volumes is None:
        source = WeightedPointSource(positions, weights)
        model = point_sum(source, frame=ROOT, field_definition=definition)
    else:
        source = VolumeElementSource(positions, volumes, density)
        model = point_sum(source, frame=ROOT, field_definition=definition)
    targets = EvaluationTargetPoints([[0.0, 0.0, 0.0]] * u.m, ("target-0",), ROOT)
    return model.at(targets)


def test_registry_has_exact_aliases_units_and_extensive_flags():
    assert resolve_quantity("r").canonical == "distance"
    assert resolve_quantity("inverse_square_distance").canonical == "inverse_square"
    assert resolve_quantity("r^-2").canonical == "inverse_square"
    assert resolve_quantity("rho_over_r2").canonical == "density_over_r2"
    assert resolve_quantity("1/r^3").canonical == "inverse_cube"
    assert resolve_quantity("rho/r^3").canonical == "density_over_r3"
    assert resolve_quantity("ρ/r³").canonical == "density_over_r3"
    assert resolve_quantity("mass/r").canonical == "mass_over_r"
    assert resolve_quantity("dm/r^2").canonical == "mass_over_r2"
    assert resolve_quantity("dm/r^3").canonical == "mass_over_r3"
    assert resolve_quantity("GG_kernel_norm").canonical == (
        "gravity_gradient_kernel_norm"
    )
    assert resolve_quantity("mass_over_r2").extensive is True
    assert resolve_quantity("density_over_r").extensive is False
    assert QUANTITY_SPECS["inverse_square"].unit == u.m**-2
    assert QUANTITY_SPECS["inverse_cube"].unit == u.m**-3
    assert resolve_quantity("density_over_r3").unit == u.kg / u.m**6
    assert resolve_quantity("mass_over_r2").unit == u.kg / u.m**2
    assert resolve_quantity("mass_over_r3").unit == u.kg / u.m**3
    assert resolve_quantity("GG_kernel_norm").unit == u.m**-3
    assert resolve_quantity("GG_kernel_norm").extensive is True
    with pytest.raises(ValueError, match="unknown"):
        resolve_quantity("1 / r ** 2")


def test_static_source_values_use_verified_density_and_unsoftened_geometry():
    setup = _setup(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        [2.0, 8.0],
        volumes=[1.0, 2.0],
        density=[2.0, 4.0],
    )
    expected = {
        "distance": [1.0, 2.0],
        "inverse_r": [1.0, 0.5],
        "inverse_square": [1.0, 0.25],
        "inverse_cube": [1.0, 0.125],
        "density": [2.0, 4.0],
        "density_over_r": [2.0, 2.0],
        "density_over_r2": [2.0, 1.0],
        "density_over_r3": [2.0, 0.5],
        "mass": [2.0, 8.0],
        "mass_over_r": [2.0, 4.0],
        "mass_over_r2": [2.0, 2.0],
        "mass_over_r3": [2.0, 1.0],
    }
    for quantity, values in expected.items():
        kwargs = (
            {"target_id": "target-0"}
            if quantity
            in {
                "distance",
                "inverse_r",
                "inverse_square",
                "inverse_cube",
                "density_over_r",
                "density_over_r2",
                "density_over_r3",
                "mass_over_r",
                "mass_over_r2",
                "mass_over_r3",
            }
            else {}
        )
        record = source_quantity_data(setup, quantity, **kwargs)
        np.testing.assert_allclose(record.values, values, rtol=0, atol=0)
        assert record.source_index.tolist() == [0, 1]
        assert record.values.flags.writeable is False


def test_new_gg_weight_quantities_match_single_source_values_and_units():
    setup = _setup(
        [[2.0, 0.0, 0.0]],
        [12.0],
        volumes=[3.0],
        density=[4.0],
    )
    expected = {
        "inverse_cube": (0.125, u.m**-3),
        "density_over_r3": (0.5, u.kg / u.m**6),
        "mass_over_r3": (1.5, u.kg / u.m**3),
    }

    for quantity, (value, unit) in expected.items():
        record = source_quantity_data(
            setup,
            quantity,
            **({"target_id": "target-0"} if quantity != "mass_over_r3" else {}),
        )
        np.testing.assert_allclose(record.values, [value], rtol=0, atol=0)
        assert record.unit == unit


@pytest.mark.parametrize(
    ("norm", "expected"),
    [("frobenius", np.sqrt(6.0) / 8.0), ("spectral", 0.25)],
)
def test_gg_kernel_norm_is_computed_per_source(norm, expected):
    setup = _setup([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], [3.0, 5.0])

    record = source_quantity_data(
        setup,
        "GG_kernel_norm",
        target_id="target-0",
        norm=norm,
    )

    np.testing.assert_allclose(record.values, [expected, expected], rtol=0, atol=1e-15)
    assert record.quantity == "gravity_gradient_kernel_norm"
    assert record.kernel_norm == norm
    assert record.unit == u.m**-3
    assert record.extensive is True


def test_gg_kernel_norm_requires_an_explicit_supported_norm():
    setup = _setup([[2.0, 0.0, 0.0]], [3.0])

    with pytest.raises(ValueError, match="norm must be 'frobenius' or 'spectral'"):
        source_quantity_data(
            setup,
            "gravity_gradient_kernel_norm",
            target_id="target-0",
        )
    with pytest.raises(ValueError, match="norm must be 'frobenius' or 'spectral'"):
        source_quantity_data(
            setup,
            "gravity_gradient_kernel_norm",
            target_id="target-0",
            norm="euclidean",
        )


def test_gg_kernel_norm_uses_the_softened_point_kernel():
    setup = _setup([[2.0, 0.0, 0.0]], [3.0])
    softening = 0.5
    q = 4.0 + softening**2
    kxx = 12.0 / q**2.5 - 1.0 / q**1.5
    kyy = -1.0 / q**1.5
    expected = np.sqrt(kxx**2 + 2.0 * kyy**2)

    record = source_quantity_data(
        setup,
        "gravity_gradient_kernel_norm",
        target_id="target-0",
        norm="frobenius",
        softening_m=softening,
    )

    np.testing.assert_allclose(record.values, [expected], rtol=1e-14, atol=0)
    assert record.softening_m == softening


def test_gg_kernel_norm_is_invariant_to_large_output_frame_translation():
    setup = _setup([[2.0, 0.0, 0.0]], [3.0])
    output = Frame(
        "source-quantity-distant-origin",
        origin=[1.0e17, 0.0, 0.0] * u.m,
        parent=ROOT,
    )

    record = source_quantity_data(
        setup,
        "GG_kernel_norm",
        target_id="target-0",
        frame=output,
        norm="frobenius",
    )

    np.testing.assert_allclose(record.values, [np.sqrt(6.0) / 8.0], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(record.positions_m, [[-1.0e17, 0.0, 0.0]])
    np.testing.assert_array_equal(record.target_xyz_m, [-1.0e17, 0.0, 0.0])
    np.testing.assert_array_equal(record.radial_distance_m, [2.0])


def test_static_quantities_reject_missing_verified_density_and_softened_geometry():
    setup = _setup([[1.0, 0.0, 0.0]], [2.0])
    with pytest.raises(ValueError, match="density and volume are not defined"):
        source_quantity_data(setup, "density")
    with pytest.raises(ValueError, match="unsoftened"):
        source_quantity_data(setup, "mass_over_r", softening_m=0.1)


def test_static_quantity_record_rejects_complex_values_without_cast_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match="values must contain only real values"):
            StaticSourceQuantityData(
                values=np.array([1.0 + 2.0j]),
                positions_m=[[1.0, 0.0, 0.0]],
                source_index=[0],
                quantity="mass",
                unit=u.kg,
                extensive=True,
                frame=ROOT,
                coordinate_origin_m=[0.0, 0.0, 0.0],
            )
    assert caught == []


def test_static_source_geometry_reuses_frame_transform_and_collision_checks():
    setup = _setup([[1.0, 0.0, 0.0]], [2.0])
    output = Frame("source-quantity-output", origin=[10.0, 0.0, 0.0] * u.m, parent=ROOT)
    record = source_quantity_data(setup, "distance", target_id="target-0", frame=output)
    np.testing.assert_array_equal(record.positions_m, [[-9.0, 0.0, 0.0]])
    np.testing.assert_array_equal(record.target_xyz_m, [-10.0, 0.0, 0.0])
    np.testing.assert_array_equal(record.values, [1.0])
    assert record.frame is output
    assert record.target_trace is setup.resolved.target_traces[0]

    colliding = _setup([[0.0, 0.0, 0.0]], [2.0])
    with pytest.raises(CoordinatePrecisionError, match="collision"):
        source_quantity_data(colliding, "distance", target_id="target-0")


def test_target_normalizer_accepts_numeric_target_points_only_and_snapshots():
    points = np.array([1.0, 2.0, 3.0])
    normalized = normalize_targets_m(points)
    points[0] = 10.0
    np.testing.assert_array_equal(normalized, [[1.0, 2.0, 3.0]])
    assert not normalized.flags.writeable
    np.testing.assert_array_equal(
        normalize_targets_m(NumericTargetPoints([[4.0, 5.0, 6.0]])),
        [[4.0, 5.0, 6.0]],
    )
    with pytest.raises(TypeError, match="gragra.targets.points.TargetPoints"):
        normalize_targets_m(object())


def test_radial_membership_has_exclusive_flows_and_closed_final_edge():
    distances = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])
    membership, group_ids = source_radial_membership(distances, [1.0, 2.0, 3.0])
    assert group_ids == ("underflow", "bin_0", "bin_1", "overflow")
    np.testing.assert_array_equal(membership, [[0, 1, 2, 2, 3]])
    assert np.all(np.bincount(membership[0], minlength=4) == [1, 1, 2, 1])


@pytest.mark.parametrize("edges", [[1.0], [1.0, 1.0], [2.0, 1.0], [0.0, np.inf]])
def test_radial_membership_rejects_nonincreasing_or_nonfinite_edges(edges):
    with pytest.raises(ValueError, match="edges"):
        source_radial_membership(np.array([[1.0]]), edges)


def test_radial_membership_rejects_boolean_distances():
    with pytest.raises(ValueError, match="distances_m"):
        source_radial_membership(np.array([[True]]), [0.0, 1.0])
