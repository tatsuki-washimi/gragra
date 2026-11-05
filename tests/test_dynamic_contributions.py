"""Dynamic source-resolved contribution and statistics contracts."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

import gragra._plotting._dynamic_contribution as dynamic_contribution
from gragra._plotting._dynamic_contribution import (
    contribution_statistics,
    directional_contribution_statistics,
    source_contributions,
)
from gragra._plotting._source_contracts import SourceContributionResult
from gragra._plotting._source_data import SourceContributionData
from gragra.fields.density_series import DensityPhasors
from gragra.geometry import Frame
from gragra.observables.series import density_phasors_observable
from gragra.targets.points import TargetPoints as NumericTargetPoints


def _phasors(
    density: np.ndarray | None = None,
    *,
    t_ref_s: float = 0.25,
    phase_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    source_positions: np.ndarray | None = None,
    source_metadata: dict | None = None,
) -> DensityPhasors:
    positions = (
        np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        if source_positions is None
        else np.array(source_positions, dtype=np.float64, copy=True)
    )
    if density is None:
        density = np.array(
            [
                [1.0 + 0.5j, 0.25 - 0.75j, -0.5 + 0.25j],
                [-0.25 + 0.5j, 1.0 + 0.25j, 0.5 - 1.0j],
            ],
            dtype=np.complex128,
        )
    return DensityPhasors(
        positions,
        np.array([0.5, 1.25, 2.0]),
        np.array(density, dtype=np.complex128, copy=True),
        np.array([1.0, 3.0]),
        t_ref_s,
        np.array(phase_origin_m, dtype=np.float64),
        "local",
        np.array([3.0, -2.0, 1.0]),
        np.array(["s0", "s1", "s2"]),
        {"mesh": "fixed", "revision": 3}
        if source_metadata is None
        else source_metadata,
    )


TARGETS = np.array([[0.0, 0.0, 4.0], [3.0, 1.0, -2.0]])


@pytest.mark.parametrize(
    ("observable", "expected_shape"),
    [
        ("potential", (2, 2, 3)),
        ("acceleration", (2, 2, 3, 3)),
        ("gravity_gradient", (2, 2, 3, 3, 3)),
    ],
)
@pytest.mark.parametrize("softening_m", [0.0, 0.3])
def test_source_and_group_sums_match_existing_phasor_observable(
    observable, expected_shape, softening_m
):
    phasors = _phasors()
    result = source_contributions(
        phasors,
        observable,
        TARGETS,
        softening_m=softening_m,
        chunk_size=2,
        group_labels=np.array([5, 1, 5]),
    )
    source = result.to_array(kind="source", layout="sample_first")
    groups = result.to_array(kind="group", layout="sample_first")
    total = result.to_array(kind="total", layout="sample_first")
    expected = density_phasors_observable(
        phasors,
        "gradient" if observable == "gravity_gradient" else observable,
        TARGETS,
        softening_m=softening_m,
        chunk_size=2,
    )

    assert source.shape == expected_shape
    np.testing.assert_allclose(source.sum(axis=2), total, rtol=3e-14, atol=0.0)
    np.testing.assert_allclose(groups.sum(axis=2), total, rtol=3e-14, atol=0.0)
    np.testing.assert_allclose(total, expected, rtol=3e-14, atol=0.0)
    assert result.observable == observable
    assert result.group_ids == (1, 5)
    assert result._data.group_label_mapping == ((1, 0), (5, 1))
    assert result.source_ids == ("s0", "s1", "s2")
    assert result.target_ids == ("target-0", "target-1")
    assert result._data.source_index.tolist() == [0, 1, 2]
    assert result._data.source_metadata is None
    assert result._data.global_metadata["mesh"] == "fixed"
    assert result._data.global_metadata["revision"] == 3
    assert result.global_metadata["mesh"] == "fixed"
    assert result.metadata["global_metadata"]["revision"] == 3
    assert result.metadata["t_ref_s"] == phasors.t_ref_s
    assert result.metadata["fourier_convention"] == phasors.fourier_convention
    assert result.metadata["peak_phasor"] is True


@pytest.mark.parametrize("alias", ["GG", "gradient"])
def test_gravity_gradient_aliases_are_canonicalized(alias):
    result = source_contributions(_phasors(), alias, TARGETS)
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
    np.testing.assert_allclose(
        result.to_array(kind="total", layout="sample_first"),
        source_contributions(_phasors(), "gravity_gradient", TARGETS).to_array(
            kind="total", layout="sample_first"
        ),
    )


def test_default_group_and_source_accessor_keep_source_values_deferred():
    result = source_contributions(_phasors(), "potential", TARGETS, chunk_size=1)
    assert result.group_ids == ("all",)
    assert result._data.source_values is None
    assert result._data.source_block_provider is not None
    block = result._data.source_block(1, 2)
    assert block.shape == (2, 2, 1)
    matrix = result.source("s1")
    assert matrix.shape == (2, 1, 2)
    assert result._data.source_values is None


def test_provider_closes_over_phasor_snapshot_not_caller_arrays():
    positions = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    volumes = np.array([2.0, 3.0])
    density = np.array([[1.0 + 1.0j, -0.25 + 0.5j]])
    frequencies = np.array([2.0])
    origin = np.zeros(3)
    phasors = DensityPhasors(
        positions,
        volumes,
        density,
        frequencies,
        0.0,
        origin,
        "local",
        origin,
        np.array(["left", "right"]),
    )
    result = source_contributions(phasors, "potential", [0.0, 0.0, 3.0])
    expected = result._data.source_block(0, 2)
    positions[:] = 99.0
    volumes[:] = 99.0
    density[:] = 99.0
    np.testing.assert_array_equal(result._data.source_block(0, 2), expected)


def test_global_metadata_is_stored_once_and_preserves_numpy_dtypes():
    metadata = {
        "flags": np.array([True, False], dtype=np.bool_),
        "labels": np.array(["north", "south"], dtype=np.str_),
        "large_integer": np.array([2**53 + 1], dtype=np.int64),
    }
    phasors = _phasors(source_metadata=metadata)
    metadata["flags"][:] = False
    metadata["labels"][:] = "changed"
    metadata["large_integer"][:] = 0

    result = source_contributions(phasors, "potential", TARGETS)
    stored = result.global_metadata
    assert result._data.source_metadata is None
    assert tuple(stored) == ("flags", "labels", "large_integer")
    assert stored["flags"].dtype == np.dtype(np.bool_)
    np.testing.assert_array_equal(stored["flags"], [True, False])
    assert stored["labels"].dtype.kind == "U"
    np.testing.assert_array_equal(stored["labels"], ["north", "south"])
    assert stored["large_integer"].dtype == np.dtype(np.int64)
    assert stored["large_integer"][0] == 2**53 + 1
    assert result.metadata["global_metadata"]["large_integer"][0] == 2**53 + 1


def test_targets_accept_numeric_points_snapshot_and_reject_high_level_points():
    coordinates = np.array([[1.0, 2.0, 4.0]])
    result = source_contributions(
        _phasors(), "potential", NumericTargetPoints(coordinates)
    )
    coordinates[:] = 100.0
    np.testing.assert_array_equal(result.metadata["targets_m"], ((1.0, 2.0, 4.0),))

    from gragra.targets import TargetPoints as HighLevelTargetPoints

    high_level = HighLevelTargetPoints(
        np.array([[1.0, 2.0, 4.0]]) * u.m,
        ("custom",),
        Frame("targets"),
    )
    with pytest.raises(TypeError, match="numeric coordinates"):
        source_contributions(_phasors(), "potential", high_level)
    bare = source_contributions(_phasors(), "potential", [1.0, 2.0, 4.0])
    assert bare.target_ids == ("target-0",)


def test_radial_groups_include_flows_and_the_last_edge_exactly_once():
    phasors = _phasors(
        source_positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    )
    targets = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, -2.0]])
    result = source_contributions(
        phasors, "potential", targets, radial_edges_m=np.array([1.0, 2.0])
    )
    assert result.group_ids == ("underflow", "bin_0", "overflow")
    # Target 0 has distances .5, sqrt(1.25), sqrt(4.25). Target 1's first
    # source lies exactly on the inclusive final edge.
    np.testing.assert_array_equal(result._data.membership, [[0, 1, 2], [1, 2, 2]])
    assert result._data.radial_edges_m.tolist() == [1.0, 2.0]
    np.testing.assert_allclose(
        result.to_array(kind="group", layout="sample_first").sum(axis=2),
        result.to_array(kind="total", layout="sample_first"),
    )


def test_group_labels_are_validated_and_mutually_exclusive_with_radial_edges():
    phasors = _phasors()
    with pytest.raises(ValueError, match="exclusive"):
        source_contributions(
            phasors,
            "potential",
            TARGETS,
            group_labels=[0, 0, 1],
            radial_edges_m=[0.0, 1.0],
        )
    with pytest.raises(ValueError, match="non-negative"):
        source_contributions(phasors, "potential", TARGETS, group_labels=[0, -1, 0])
    with pytest.raises(TypeError, match="integer"):
        source_contributions(phasors, "potential", TARGETS, group_labels=[0, 1.5, 0])


def test_many_unique_groups_scatter_each_source_block_once_per_target(monkeypatch):
    n_sources = 101
    n_frequencies = 2
    n_targets = 2
    chunk_size = 17
    phasors = DensityPhasors(
        np.column_stack(
            (np.arange(n_sources, dtype=np.float64), np.zeros((n_sources, 2)))
        ),
        np.ones(n_sources),
        np.ones((n_frequencies, n_sources), dtype=np.complex128),
        np.array([1.0, 2.0]),
        0.0,
        np.zeros(3),
        "local",
        np.zeros(3),
        tuple(f"source-{index}" for index in range(n_sources)),
    )
    targets = np.array([[0.0, 0.0, 1000.0], [0.0, 0.0, -1000.0]])
    block_visits = []
    original = dynamic_contribution._accumulate_group_block

    def track_block(grouped_values, group_indices, source_terms):
        block_visits.append((len(group_indices), len(grouped_values)))
        original(grouped_values, group_indices, source_terms)

    monkeypatch.setattr(dynamic_contribution, "_accumulate_group_block", track_block)
    result = source_contributions(
        phasors,
        "potential",
        targets,
        chunk_size=chunk_size,
        group_labels=np.arange(n_sources),
    )

    blocks_per_frequency = (n_sources + chunk_size - 1) // chunk_size
    assert len(block_visits) == n_frequencies * n_targets * blocks_per_frequency
    assert sum(block_size for block_size, _ in block_visits) == (
        n_frequencies * n_targets * n_sources
    )
    assert all(group_count == n_sources for _, group_count in block_visits)
    np.testing.assert_allclose(
        result.to_array(kind="group", layout="sample_first").sum(axis=2),
        result.to_array(kind="total", layout="sample_first"),
        rtol=3e-14,
    )


def test_source_contributions_reject_bad_softening_and_chunk_size():
    for softening in (-1.0, np.nan, np.inf, True):
        with pytest.raises(ValueError, match="softening_m"):
            source_contributions(
                _phasors(), "potential", TARGETS, softening_m=softening
            )
    for chunk_size in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="chunk_size"):
            source_contributions(
                _phasors(), "potential", TARGETS, chunk_size=chunk_size
            )


def test_zero_softening_collision_preserves_global_source_index():
    positions = np.array([[5.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    phasors = _phasors(source_positions=positions)
    with pytest.raises(ValueError, match="source point 1"):
        source_contributions(
            phasors,
            "potential",
            [[1.0, 0.0, 0.0]],
            chunk_size=1,
            softening_m=0.0,
        )


def test_statistics_produce_componentwise_shares_phase_interference_and_cumulative():
    result = source_contributions(
        _phasors(), "acceleration", TARGETS, group_labels=[0, 1, 1]
    )
    stats = contribution_statistics(result)
    groups = result.to_array(kind="group", layout="sample_first")
    total = result.to_array(kind="total", layout="sample_first")
    denominator = np.sum(np.abs(groups), axis=2)
    expected_share = np.divide(
        np.abs(groups),
        denominator[:, :, np.newaxis, :],
        out=np.zeros_like(np.abs(groups)),
        where=denominator[:, :, np.newaxis, :] != 0,
    )
    for index, group_id in enumerate(result.group_ids):
        np.testing.assert_allclose(
            stats.amplitude_share.to_array(group_id=group_id, layout="sample_first"),
            expected_share[:, :, index, :],
        )
        np.testing.assert_allclose(
            stats.phase_rad.to_array(group_id=group_id, layout="sample_first"),
            np.angle(groups[:, :, index, :]),
        )
        np.testing.assert_allclose(
            stats.interference_share.to_array(group_id=group_id, layout="sample_first"),
            np.real(groups[:, :, index, :] * np.conj(total)) / np.abs(total) ** 2,
        )
    valid_total = np.abs(total) > 0.0
    all_interference = np.stack(
        [
            stats.interference_share.to_array(group_id=group_id, layout="sample_first")
            for group_id in result.group_ids
        ],
        axis=2,
    )
    np.testing.assert_allclose(np.sum(all_interference, axis=2)[valid_total], 1.0)
    last_group = result.group_ids[-1]
    np.testing.assert_allclose(
        stats.cumulative_complex.to_array(group_id=last_group, layout="sample_first"),
        total,
    )
    assert stats.rms is None
    with pytest.raises(TypeError, match="layout"):
        stats.phase_rad.to_array(group_id=0)
    mask = stats.phase_rad.validity(0)
    assert mask.target_ids == result.target_ids
    assert mask.component_labels == result.component_labels
    assert mask.frequency_hz.tolist() == [1.0, 3.0]
    assert mask.to_array(layout="sample_first").shape == (2, 2, 3)
    matrix = stats.phase_rad.series(0)
    assert tuple(matrix.rows.keys()) == result.target_ids
    assert tuple(matrix.cols.keys()) == result.component_labels
    assert matrix.frequencies.to_value("Hz").tolist() == [1.0, 3.0]


@pytest.mark.parametrize(
    ("observable", "density"),
    [
        ("potential", np.array([[2.0, -0.8, 0.0], [2.0, -0.8, 0.0]])),
        ("acceleration", np.array([[2.0, 0.8, 0.0], [2.0, 0.8, 0.0]])),
    ],
)
def test_exactly_zero_total_has_false_interference_mask_and_zero_placeholder(
    observable, density
):
    phasors = _phasors(
        density,
        source_positions=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
    )
    result = source_contributions(
        phasors, observable, [[0.0, 0.0, 0.0]], group_labels=[0, 1, 2]
    )
    stats = contribution_statistics(result)
    np.testing.assert_array_equal(result._data.total_values[:, 0, ...], 0.0)
    for group_id in result.group_ids:
        assert (
            not stats.interference_share.validity(group_id)
            .to_array(layout="sample_first")
            .any()
        )
        np.testing.assert_array_equal(
            stats.interference_share.to_array(group_id=group_id, layout="sample_first"),
            0.0,
        )


def test_amplitude_share_and_phase_have_independent_exact_zero_masks():
    phasors = _phasors(
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    result = source_contributions(phasors, "potential", TARGETS, group_labels=[0, 1, 2])
    stats = contribution_statistics(result)
    zero_group = result.group_ids[1]
    assert (
        not stats.phase_rad.validity(zero_group).to_array(layout="sample_first").any()
    )
    assert (
        stats.amplitude_share.validity(zero_group).to_array(layout="sample_first").all()
    )
    np.testing.assert_array_equal(
        stats.phase_rad.to_array(group_id=zero_group, layout="sample_first"), 0.0
    )


def test_same_opposite_and_orthogonal_source_phasors_keep_componentwise_meaning():
    symmetric_positions = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    def two_source_phasors(density, positions=symmetric_positions):
        return DensityPhasors(
            np.array(positions, dtype=np.float64),
            np.ones(2),
            np.array([density], dtype=np.complex128),
            np.array([2.0]),
            0.0,
            np.zeros(3),
            "local",
            np.zeros(3),
            ("a", "b"),
        )

    same = source_contributions(
        two_source_phasors([1.0, 1.0]),
        "potential",
        [[0.0, 0.0, 0.0]],
        group_labels=[0, 1],
    )
    same_stats = contribution_statistics(same)
    for group_id in same.group_ids:
        np.testing.assert_allclose(
            same_stats.amplitude_share.to_array(
                group_id=group_id, layout="sample_first"
            ),
            [[[0.5]]],
        )
        np.testing.assert_allclose(
            same_stats.interference_share.to_array(
                group_id=group_id, layout="sample_first"
            ),
            [[[0.5]]],
        )

    opposite = source_contributions(
        two_source_phasors([1.0, -1.0]),
        "potential",
        [[0.0, 0.0, 0.0]],
        group_labels=[0, 1],
    )
    opposite_stats = contribution_statistics(opposite)
    assert (
        not opposite_stats.interference_share.validity(0)
        .to_array(layout="sample_first")
        .any()
    )

    orthogonal = source_contributions(
        two_source_phasors([1.0, 1.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "acceleration",
        [[0.0, 0.0, 0.0]],
        group_labels=[0, 1],
    )
    orthogonal_stats = contribution_statistics(orthogonal)
    np.testing.assert_allclose(
        orthogonal_stats.interference_share.to_array(group_id=0, layout="sample_first"),
        [[[1.0, 0.0, 0.0]]],
    )
    np.testing.assert_allclose(
        orthogonal_stats.interference_share.to_array(group_id=1, layout="sample_first"),
        [[[0.0, 1.0, 0.0]]],
    )


def test_interference_share_allows_negative_and_above_one_values():
    phasors = _phasors(
        np.array([[1.0, -2.0, 0.0], [1.0, -2.0, 0.0]]),
    )
    result = source_contributions(
        phasors, "potential", TARGETS[:1], group_labels=[0, 1, 2]
    )
    values = result.to_array(kind="group", layout="sample_first")
    total = result.to_array(kind="total", layout="sample_first")
    ratios = (
        np.real(values * np.conj(total[:, :, np.newaxis]))
        / np.abs(total[:, :, np.newaxis]) ** 2
    )
    assert (ratios < 0.0).any()
    assert (ratios > 1.0).any()
    stats = contribution_statistics(result)
    for index, group_id in enumerate(result.group_ids):
        np.testing.assert_allclose(
            stats.interference_share.to_array(group_id=group_id, layout="sample_first"),
            ratios[:, :, index, np.newaxis],
        )


def test_weighted_rms_quantities_and_rms_share_use_unormalized_frequency_weights():
    result = source_contributions(
        _phasors(), "potential", TARGETS, group_labels=[0, 1, 1]
    )
    weights = np.array([2.0, 0.5])
    stats = contribution_statistics(result, frequency_weights=weights)
    groups = result.to_array(kind="group", layout="sample_first")
    total = result.to_array(kind="total", layout="sample_first")
    expected_group = np.sqrt(
        0.5 * np.sum(weights[:, None, None] * abs(groups) ** 2, axis=0)
    )
    expected_total = np.sqrt(0.5 * np.sum(weights[:, None] * abs(total) ** 2, axis=0))
    np.testing.assert_allclose(
        stats.rms.group_rms[result.group_ids[0]].value, expected_group[:, 0, None]
    )
    np.testing.assert_allclose(stats.rms.total_rms.value, expected_total[:, None])
    np.testing.assert_allclose(
        stats.rms.rms_share[result.group_ids[0]].value,
        stats.rms.group_rms[result.group_ids[0]].value
        / np.sum(
            np.stack([value.value for value in stats.rms.group_rms.values()]), axis=0
        ),
    )
    all_shares = np.stack(
        [value.value for value in stats.rms.rms_share.values()], axis=1
    )
    np.testing.assert_allclose(np.sum(all_shares, axis=1), 1.0)
    np.testing.assert_allclose(
        stats.rms.total_ratio[result.group_ids[0]].value,
        stats.rms.group_rms[result.group_ids[0]].value / stats.rms.total_rms.value,
    )
    expected_cross = 0.5 * np.sum(
        weights[:, None, None] * np.real(groups * np.conj(total[:, :, np.newaxis])),
        axis=0,
    )
    np.testing.assert_allclose(
        stats.rms.group_total_cross[result.group_ids[0]].value[:, 0],
        expected_cross[:, 0],
    )
    expected_power = 0.5 * np.sum(weights[:, None] * abs(total) ** 2, axis=0)
    np.testing.assert_allclose(stats.rms.total_power.value[:, 0], expected_power)
    total_cross = np.sum(
        np.stack(
            [value.value[:, 0] for value in stats.rms.group_total_cross.values()],
            axis=1,
        ),
        axis=1,
    )
    np.testing.assert_allclose(total_cross, stats.rms.total_power.value[:, 0])
    assert stats.rms.total_rms.unit == u.Unit(result.unit)
    assert (
        stats.rms.group_total_cross[result.group_ids[0]].unit
        == u.Unit(result.unit) ** 2
    )
    assert stats.rms.total_power.unit == u.Unit(result.unit) ** 2
    assert stats.rms.rms_share_valid[result.group_ids[0]].to_array().shape == (2, 1)


@pytest.mark.parametrize("density", [1e-160, 1e160])
def test_statistics_preserve_finite_interference_and_weighted_rms(density):
    phasors = DensityPhasors(
        np.array([[1.0, 0.0, 0.0]]),
        np.array([1.0]),
        np.array([[density + 0.0j]]),
        np.array([10.0]),
        0.0,
        np.zeros(3),
        "local",
        np.zeros(3),
        ("tiny",),
    )
    result = source_contributions(
        phasors, "potential", [[0.0, 0.0, 0.0]], group_labels=[0]
    )
    response = result.to_array(kind="total", layout="sample_first")
    assert response.shape == (1, 1)
    assert np.isfinite(response).all()
    assert response.item() != 0.0

    stats = contribution_statistics(result, frequency_weights=[1.0])
    interference = stats.interference_share.to_array(group_id=0, layout="sample_first")
    assert stats.interference_share.validity(0).to_array(layout="sample_first").all()
    np.testing.assert_allclose(interference, 1.0)
    np.testing.assert_allclose(
        stats.rms.total_rms.value, np.abs(response) / np.sqrt(2.0)
    )
    np.testing.assert_allclose(
        stats.rms.group_rms[0].value, np.abs(response) / np.sqrt(2.0)
    )
    np.testing.assert_allclose(stats.rms.total_ratio[0].value, 1.0)
    assert stats.rms.total_ratio_valid[0].to_array().all()
    assert stats.rms.rms_share_valid[0].to_array().all()
    if density == 1e-160:
        assert stats.rms.total_power.value.item() == 0.0


@pytest.mark.parametrize("density", [1e-160, 1e160])
def test_directional_statistics_preserve_finite_power_ratios_and_rms(density):
    phasors = DensityPhasors(
        np.array([[1.0, 0.0, 0.0]]),
        np.array([1.0]),
        np.array([[density + 0.0j]]),
        np.array([10.0]),
        0.0,
        np.zeros(3),
        "local",
        np.zeros(3),
        ("tiny",),
    )
    result = source_contributions(
        phasors, "potential", [[0.0, 0.0, 0.0]], group_labels=[0]
    )
    response = result.to_array(kind="total", layout="sample_first")
    stats = directional_contribution_statistics(
        [result, result], direction_weights=[1.0, 1.0], frequency_weights=[1.0]
    )

    interference = stats.interference_share.to_array(group_id=0, layout="sample_first")
    assert stats.interference_share.validity(0).to_array(layout="sample_first").all()
    np.testing.assert_allclose(interference, 1.0)
    np.testing.assert_allclose(
        stats.rms.total_rms.value, np.abs(response) / np.sqrt(2.0)
    )
    np.testing.assert_allclose(stats.rms.total_ratio[0].value, 1.0)
    assert stats.rms.total_ratio_valid[0].to_array().all()
    if density == 1e-160:
        assert stats.rms.total_power.value.item() == 0.0


def test_statistics_preserve_subnormal_complex_response_ratio_and_rms():
    phasors = DensityPhasors(
        np.array([[1.0, 0.0, 0.0]]),
        np.array([1.0]),
        np.array([[1e-300 + 1e-300j]]),
        np.array([10.0]),
        0.0,
        np.zeros(3),
        "local",
        np.zeros(3),
        ("tiny-complex",),
    )
    result = source_contributions(
        phasors, "potential", [[0.0, 0.0, 0.0]], group_labels=[0]
    )
    response = result.to_array(kind="total", layout="sample_first")
    assert np.isfinite(response).all()
    assert response.item() != 0.0
    assert response.real.item() != 0.0
    assert response.imag.item() != 0.0

    stats = contribution_statistics(result, frequency_weights=[1.0])
    assert stats.interference_share.validity(0).to_array(layout="sample_first").all()
    np.testing.assert_allclose(
        stats.interference_share.to_array(group_id=0, layout="sample_first"), 1.0
    )
    np.testing.assert_allclose(
        stats.rms.total_rms.value, np.abs(response) / np.sqrt(2.0)
    )


def _single_potential_result(value):
    source_values = np.asarray([[[value]]], dtype=np.complex128)
    data = SourceContributionData(
        source_values=source_values,
        group_values=source_values.copy(),
        total_values=source_values[:, :, 0].copy(),
        positions_m=np.array([[1.0, 0.0, 0.0]]),
        targets_m=np.array([[0.0, 0.0, 0.0]]),
        target_ids=("target-0",),
        frequencies_hz=np.array([1.0]),
        unit="m**2 / s**2",
        observable="potential",
        frame="local",
        membership=np.zeros((1, 1), dtype=np.int64),
        group_ids=("all",),
        source_ids=("source-0",),
    )
    return SourceContributionResult(data)


def _grouped_potential_result(values):
    values = np.asarray(values, dtype=np.complex128)
    source_values = values.reshape(1, 1, len(values))
    group_ids = tuple(range(len(values)))
    data = SourceContributionData(
        source_values=source_values,
        group_values=source_values.copy(),
        total_values=np.sum(source_values, axis=2),
        positions_m=np.column_stack(
            (
                np.arange(1, len(values) + 1, dtype=np.float64),
                np.zeros((len(values), 2)),
            )
        ),
        targets_m=np.array([[0.0, 0.0, 0.0]]),
        target_ids=("target-0",),
        frequencies_hz=np.array([1.0]),
        unit="m**2 / s**2",
        observable="potential",
        frame="local",
        membership=np.arange(len(values), dtype=np.int64).reshape(1, -1),
        group_ids=group_ids,
        source_ids=tuple(f"source-{index}" for index in range(len(values))),
    )
    return SourceContributionResult(data)


def test_directional_subnormal_complex_ratio_with_fractional_weights():
    result = _single_potential_result(1e-310 + 1e-310j)
    stats = directional_contribution_statistics(
        [result, result], direction_weights=[0.25, 0.75]
    )

    assert (
        stats.interference_share.validity("all").to_array(layout="sample_first").all()
    )
    np.testing.assert_allclose(
        stats.interference_share.to_array(group_id="all", layout="sample_first"),
        1.0,
    )


def test_orthogonal_large_groups_keep_tiny_aligned_interference_in_both_paths():
    result = _grouped_potential_result([1e200j, 1e-200 - 1e200j])
    assert result._data.total_values.item() == 1e-200

    ordinary = contribution_statistics(result)
    directional = directional_contribution_statistics(
        [result, result], direction_weights=[0.25, 0.75]
    )
    for stats in (ordinary, directional):
        np.testing.assert_allclose(
            stats.interference_share.to_array(group_id=0, layout="sample_first"),
            0.0,
        )
        np.testing.assert_allclose(
            stats.interference_share.to_array(group_id=1, layout="sample_first"),
            1.0,
        )
        assert (
            stats.interference_share.validity(0).to_array(layout="sample_first").all()
        )
        assert (
            stats.interference_share.validity(1).to_array(layout="sample_first").all()
        )


def test_cancelling_large_cross_terms_keep_small_aligned_interference_in_both_paths():
    result = _grouped_potential_result(
        [1e200 - 1e200j, -1e200 + 1e200j, 1e-200 + 1e-200j]
    )
    np.testing.assert_array_equal(
        result._data.total_values, np.array([[1e-200 + 1e-200j]])
    )

    ordinary = contribution_statistics(result)
    directional = directional_contribution_statistics(
        [result, result], direction_weights=[0.25, 0.75]
    )
    for stats in (ordinary, directional):
        for group_id, expected in enumerate((0.0, 0.0, 1.0)):
            np.testing.assert_allclose(
                stats.interference_share.to_array(
                    group_id=group_id, layout="sample_first"
                ),
                expected,
            )
            assert (
                stats.interference_share.validity(group_id)
                .to_array(layout="sample_first")
                .all()
            )


def test_directional_response_scaling_precedes_fractional_weights():
    smallest = np.nextafter(0.0, 1.0)
    nonzero = _single_potential_result(smallest)
    zero = _single_potential_result(0.0)
    stats = directional_contribution_statistics(
        [nonzero, zero], direction_weights=[0.25, 0.75]
    )

    assert (
        stats.interference_share.validity("all").to_array(layout="sample_first").all()
    )
    np.testing.assert_allclose(
        stats.interference_share.to_array(group_id="all", layout="sample_first"),
        1.0,
    )


def test_rms_ratios_survive_unrepresentable_subnormal_rms():
    smallest = np.nextafter(0.0, 1.0)
    result = _single_potential_result(smallest)
    stats = contribution_statistics(result, frequency_weights=[0.25])

    assert stats.rms.total_rms.value.item() == 0.0
    assert stats.rms.group_rms["all"].value.item() == 0.0
    assert stats.rms.rms_share_valid["all"].to_array().all()
    assert stats.rms.total_ratio_valid["all"].to_array().all()
    np.testing.assert_allclose(stats.rms.rms_share["all"].value, 1.0)
    np.testing.assert_allclose(stats.rms.total_ratio["all"].value, 1.0)


def test_directional_rms_ratios_survive_unrepresentable_subnormal_rms():
    smallest = np.nextafter(0.0, 1.0)
    result = _single_potential_result(smallest)
    stats = directional_contribution_statistics(
        [result, result],
        direction_weights=[0.25, 0.75],
        frequency_weights=[0.25],
    )

    assert stats.rms.total_rms.value.item() == 0.0
    assert stats.rms.group_rms["all"].value.item() == 0.0
    assert stats.rms.rms_share_valid["all"].to_array().all()
    assert stats.rms.total_ratio_valid["all"].to_array().all()
    np.testing.assert_allclose(stats.rms.rms_share["all"].value, 1.0)
    np.testing.assert_allclose(stats.rms.total_ratio["all"].value, 1.0)


@pytest.mark.parametrize(
    "weights",
    [
        [0.0, 0.0],
        [-1.0, 1.0],
        [1.0],
        [1.0, np.inf],
        [1.0 + 1.0j, 2.0],
        [True, False],
    ],
)
def test_frequency_weights_must_be_valid_finite_nonnegative_and_nonempty(weights):
    result = source_contributions(_phasors(), "potential", TARGETS)
    with pytest.raises((TypeError, ValueError), match="frequency_weights"):
        contribution_statistics(result, frequency_weights=weights)


def test_frequency_weights_accept_only_dimensionless_quantity():
    result = source_contributions(_phasors(), "potential", TARGETS)
    stats = contribution_statistics(
        result, frequency_weights=[1.0, 2.0] * u.dimensionless_unscaled
    )
    assert stats.rms is not None
    with pytest.raises(ValueError, match="dimensionless"):
        contribution_statistics(result, frequency_weights=[1.0, 2.0] * u.Hz)


def test_directional_statistics_use_mean_cross_over_mean_power_not_mean_ratios():
    # Equal geometry but opposite relative source phases make direction-wise
    # interference ratios differ. Their ensemble ratio is computed only after
    # the weighted cross and power are averaged.
    density_a = np.array([[1.0, 0.4, 0.0], [1.0, 0.4, 0.0]], dtype=complex)
    density_b = np.array([[0.2, -1.5, 0.0], [0.2, -1.5, 0.0]], dtype=complex)
    first = source_contributions(
        _phasors(density_a), "potential", TARGETS, group_labels=[0, 1, 2]
    )
    second = source_contributions(
        _phasors(density_b), "potential", TARGETS, group_labels=[0, 1, 2]
    )
    direction_weights = np.array([1.0, 3.0])
    stats = directional_contribution_statistics(
        [first, second], direction_weights=direction_weights
    )
    q = direction_weights / direction_weights.sum()
    g0 = np.stack(
        [first._data.group_values, second._data.group_values], axis=2
    )  # F,M,D,B
    total = np.stack([first._data.total_values, second._data.total_values], axis=2)
    cross = np.real(
        np.sum(q[None, None, :, None] * g0 * np.conj(total[:, :, :, None]), axis=2)
    )
    power = np.sum(q[None, None, :] * abs(total) ** 2, axis=2)
    expected = cross / power[:, :, None]
    for index, group_id in enumerate(first.group_ids):
        np.testing.assert_allclose(
            stats.interference_share.to_array(group_id=group_id, layout="sample_first"),
            expected[:, :, index, np.newaxis],
        )
        np.testing.assert_allclose(
            stats.directional_cross.to_array(group_id=group_id, layout="sample_first")[
                :, :, 0
            ],
            cross[:, :, index],
        )
    per_direction = (
        np.real(g0 * np.conj(total[:, :, :, None])) / abs(total[:, :, :, None]) ** 2
    )
    mean_ratios = np.sum(q[None, None, :, None] * per_direction, axis=2)
    assert not np.allclose(expected, mean_ratios)
    np.testing.assert_allclose(expected.sum(axis=2), 1.0)


def test_directional_statistics_preserve_distinct_global_provenance():
    first = source_contributions(
        _phasors(source_metadata={"direction": "north"}), "potential", TARGETS
    )
    second = source_contributions(
        _phasors(source_metadata={"direction": "south"}), "potential", TARGETS
    )

    stats = directional_contribution_statistics(
        [first, second], direction_weights=[1.0, 1.0]
    )

    assert tuple(metadata["direction"] for metadata in stats.direction_metadata) == (
        "north",
        "south",
    )
    assert tuple(
        metadata["direction"] for metadata in stats.metadata["direction_metadata"]
    ) == ("north", "south")


def test_directional_metadata_also_preserves_per_source_provenance():
    source_values = np.array([[[-1.0 - 0.5j, -2.0 + 0.25j]]])
    group_values = source_values.sum(axis=2)[:, :, np.newaxis]
    total_values = group_values[:, :, 0]

    def make_result(direction, prefix):
        data = SourceContributionData(
            source_values=source_values,
            group_values=group_values,
            total_values=total_values,
            positions_m=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            targets_m=np.array([[0.0, 0.0, 3.0]]),
            target_ids=("target-0",),
            frequencies_hz=np.array([1.0]),
            unit="m**2 / s**2",
            observable="potential",
            frame="local",
            membership=np.zeros((1, 2), dtype=np.int64),
            group_ids=("all",),
            source_ids=("source-0", "source-1"),
            source_metadata=(
                {"particle": f"{prefix}-0"},
                {"particle": f"{prefix}-1"},
            ),
            global_metadata={"direction": direction},
        )
        return SourceContributionResult(data)

    stats = directional_contribution_statistics(
        [make_result("north", "n"), make_result("south", "s")],
        direction_weights=[1.0, 1.0],
    )

    assert tuple(item["direction"] for item in stats.direction_metadata) == (
        "north",
        "south",
    )
    assert tuple(
        tuple(item["particle"] for item in direction)
        for direction in stats.direction_source_metadata
    ) == (("n-0", "n-1"), ("s-0", "s-1"))
    assert (
        stats.metadata["direction_source_metadata"] == stats.direction_source_metadata
    )


def test_directional_rms_uses_weighted_power_not_complex_direction_sum():
    density_a = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    density_b = -density_a
    first = source_contributions(_phasors(density_a), "potential", TARGETS)
    second = source_contributions(_phasors(density_b), "potential", TARGETS)
    stats = directional_contribution_statistics(
        [first, second], direction_weights=[0.5, 0.5], frequency_weights=[1.0, 1.0]
    )
    assert np.all(stats.rms.total_rms.value > 0.0)


@pytest.mark.parametrize("field", ["t_ref_s", "phase_origin_m", "frame", "frequency"])
def test_directional_statistics_reject_incompatible_phase_geometry_and_frequency(field):
    first_phasors = _phasors()
    kwargs = {}
    if field == "t_ref_s":
        kwargs["t_ref_s"] = first_phasors.t_ref_s + 1.0
    elif field == "phase_origin_m":
        kwargs["phase_origin_m"] = (0.0, 0.0, 1.0)
    second = _phasors(**kwargs)
    if field == "frame":
        second = DensityPhasors(
            second.positions_m,
            second.volumes_m3,
            second.density_kg_m3,
            second.frequency_hz,
            second.t_ref_s,
            second.phase_origin_m,
            "other-frame",
            second.coordinate_origin_m,
            second.source_ids,
            second.source_metadata,
        )
    elif field == "frequency":
        second = DensityPhasors(
            second.positions_m,
            second.volumes_m3,
            second.density_kg_m3,
            np.array([1.0, 4.0]),
            second.t_ref_s,
            second.phase_origin_m,
            second.coordinate_frame,
            second.coordinate_origin_m,
            second.source_ids,
            second.source_metadata,
        )
    first_result = source_contributions(first_phasors, "potential", TARGETS)
    second_result = source_contributions(second, "potential", TARGETS)
    with pytest.raises(ValueError, match="compatible"):
        directional_contribution_statistics(
            [first_result, second_result], direction_weights=[1.0, 1.0]
        )


def test_directional_weights_are_nonnegative_and_have_positive_sum():
    result = source_contributions(_phasors(), "potential", TARGETS)
    for weights in ([-1.0], [0.0], [np.inf], [1.0, 2.0]):
        with pytest.raises((TypeError, ValueError), match="direction_weights"):
            directional_contribution_statistics([result], direction_weights=weights)
    with pytest.raises(ValueError, match="dimensionless"):
        directional_contribution_statistics([result], direction_weights=[1.0] * u.Hz)
