"""Contracts for bounded, G-free per-source point kernel terms."""

from __future__ import annotations

import numpy as np
import pytest

from gragra.constants import G_SI
from gragra.gravity_sources import WeightedPointSource
from gragra.kernels._source_terms import (
    iter_source_kernel_terms,
    source_kernel_terms,
)
from gragra.kernels.inverse_square import (
    acceleration_contract,
    gradient_contract,
    potential_contract,
)
from gragra.observables import (
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)
from gragra.observables._source_terms import point_observable_source_terms


@pytest.mark.parametrize(
    ("observable", "contract", "component_shape"),
    [
        ("potential", potential_contract, ()),
        ("acceleration", acceleration_contract, (3,)),
        ("gradient", gradient_contract, (3, 3)),
    ],
)
@pytest.mark.parametrize("softening_m", [0.0, 0.25])
@pytest.mark.parametrize("complex_weights", [False, True])
def test_source_terms_sum_to_existing_g_free_contract(
    observable, contract, component_shape, softening_m, complex_weights
):
    sources = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    targets = np.array([[4.0, 0.5, 0.0], [0.0, 0.0, 3.0]])
    weights = np.array([2.0, -0.5, 1.25])
    if complex_weights:
        weights = weights * np.array([1.0 + 0.5j, -0.25j, 0.75 - 1.0j])

    terms = source_kernel_terms(
        sources,
        weights,
        targets,
        observable=observable,
        softening_m=softening_m,
        chunk_size=2,
    )
    expected = contract(
        sources,
        weights,
        targets,
        softening_m=softening_m,
        chunk_size=2,
    )
    assert terms.shape == (len(targets), len(sources), *component_shape)
    np.testing.assert_allclose(terms.sum(axis=1), expected, rtol=1e-14, atol=0.0)
    assert not terms.flags.writeable


def test_source_kernel_iterator_is_ordered_bounded_and_reconstructs_full_array():
    sources = np.arange(18, dtype=np.float64).reshape(6, 3)
    targets = np.array([[100.0, 0.0, 0.0]])
    weights = np.arange(1.0, 7.0)
    pieces = list(
        iter_source_kernel_terms(
            sources,
            weights,
            targets,
            observable="acceleration",
            chunk_size=2,
        )
    )
    assert [sl.start for sl, _ in pieces] == [0, 2, 4]
    assert [sl.stop for sl, _ in pieces] == [2, 4, 6]
    assert all(values.shape == (1, 2, 3) for _, values in pieces)
    expected = np.concatenate([values for _, values in pieces], axis=1)
    actual = source_kernel_terms(
        sources, weights, targets, observable="acceleration", chunk_size=2
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("softening_m", [-0.1, np.nan, np.inf])
def test_source_kernel_terms_reject_invalid_softening(softening_m):
    with pytest.raises(ValueError, match="softening_m"):
        source_kernel_terms(
            [[0.0, 0.0, 0.0]],
            [1.0],
            [[2.0, 0.0, 0.0]],
            observable="potential",
            softening_m=softening_m,
        )


def test_source_kernel_terms_keep_collision_error_from_existing_kernel_contract():
    with pytest.raises(ValueError, match="Collision detected"):
        source_kernel_terms(
            [[0.0, 0.0, 0.0]],
            [1.0],
            [[0.0, 0.0, 0.0]],
            observable="potential",
        )


@pytest.mark.parametrize(
    ("observable", "point_observable"),
    [
        ("potential", point_potential),
        ("acceleration", point_acceleration),
        ("gravity_gradient", point_gravity_gradient),
    ],
)
@pytest.mark.parametrize("softening_m", [0.0, 0.4])
@pytest.mark.parametrize("complex_weights", [False, True])
def test_g_once_source_observables_sum_to_existing_point_observables(
    observable, point_observable, softening_m, complex_weights
):
    sources = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, -1.0, 0.5]])
    targets = np.array([[0.0, 0.0, 2.0], [4.0, 1.0, -1.0]])
    weights = np.array([2.0, -0.75, 1.25])
    if complex_weights:
        weights = weights * np.array([1.0 + 2.0j, -0.5j, -0.75 + 0.25j])
    source = WeightedPointSource(sources, weights)

    terms = point_observable_source_terms(
        sources,
        weights,
        targets,
        observable=observable,
        softening_m=softening_m,
        chunk_size=2,
    )
    expected = point_observable(source, targets, softening_m=softening_m, chunk_size=2)
    np.testing.assert_allclose(terms.sum(axis=1), expected, rtol=2e-14, atol=0.0)


def test_observable_wrapper_keeps_formal_gravity_gradient_aliases():
    sources = np.array([[0.0, 0.0, 0.0]])
    targets = np.array([[2.0, 0.0, 0.0]])
    terms = point_observable_source_terms(sources, [1.0], targets, observable="GG")
    np.testing.assert_allclose(
        terms[0, 0],
        np.diag([2.0, -1.0, -1.0]) * (G_SI / 8.0),
        rtol=1e-14,
        atol=0.0,
    )
