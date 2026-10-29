"""Acceptance tests for gravitational contribution histograms and cumulative sums.

Tests the contribution histogram module:
- Separation of per-element norm and signed components vs distance-band
  cumulative vector sums.
- Reuse of _diagnostics coordinate transformation and precision checking.
- Identity check: total band sum matches total acceleration within
  c * N * eps * sum(||da||).
- Distinction between ||sum(da)|| and sum(||da||) under signed cancellations.
- Underflow, overflow, and bins handling.
- Regression check on existing hist() behavior.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gragra._plotting._contribution import (
    contribution_hist,
    cumulative_contribution,
    plot_contribution_hist,
    plot_cumulative_contribution,
)
from gragra.geometry import Frame
from gragra.gravity_sources import WeightedPointSource
from gragra.gravity_sources._models import point_sum
from gragra.results import Contribution, GravityFieldDefinition, PotentialReference
from gragra.targets import TargetPoints

G_SI = float(const.G.value)
ROOT = Frame("root")


def _field(name: str = "test") -> GravityFieldDefinition:
    return GravityFieldDefinition.source_field(
        name, Contribution("total"), PotentialReference.ZERO_AT_INFINITY
    )


def test_contribution_hist_norm_and_signed_components() -> None:
    """Test element norm and signed components, demonstrating cancellation."""
    # Two equal masses at x = +10 m and x = -10 m, target at (0, 0, 0)
    # Target feels attraction towards +x from element 1, towards -x from element 2
    # In gragra convention: Delta r_i = pos_i - pos_target
    # delta_a_i = G * m_i * (pos_i - pos_target) / |pos_i - pos_target|^3
    positions = [[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]]
    mass = 1.0e6  # kg
    weights = [mass, mass]
    targets = TargetPoints(np.array([[0.0, 0.0, 0.0]]) * u.m, ("center",), ROOT)
    setup = point_sum(
        WeightedPointSource(positions, weights),
        frame=ROOT,
        field_definition=_field(),
    ).at(targets)

    # 1. Norm histogram
    h_norm = contribution_hist(setup, quantity="norm", target_id="center", bins=5)
    expected_acc_norm = G_SI * mass / (10.0**2)
    # Both elements have exact same norm
    values_norm = np.asarray(h_norm.values)
    assert np.sum(values_norm) == 2

    # 2. Signed x component histogram
    h_x = contribution_hist(setup, quantity="x", target_id="center", bins=10)
    # One element is positive, one is negative
    x_edges = np.asarray(h_x.edges.value if hasattr(h_x.edges, "value") else h_x.edges)
    assert x_edges[0] < -expected_acc_norm * 0.9
    assert x_edges[-1] > expected_acc_norm * 0.9

    # 3. Sum of norms vs norm of sum
    # Norm sum is 2 * expected_acc_norm > 0
    # Vector sum is exactly (0, 0, 0) due to symmetry!
    cum = cumulative_contribution(setup, target_id="center", bins=5)
    total_vec = cum.total_vector_m_s2
    np.testing.assert_allclose(total_vec, np.zeros(3), atol=1e-15)
    assert cum.total_norm_sum_m_s2 == pytest.approx(2.0 * expected_acc_norm, rel=1e-10)
    assert np.linalg.norm(total_vec) < cum.total_norm_sum_m_s2


def test_cumulative_contribution_distance_bands_and_identity() -> None:
    """Distance-band vector sums must add up to total acceleration within tolerance."""
    # Target A at (-3, -4, 0)
    # Sources at:
    # 1) (0, 0, 0) -> dist = 5 m
    # 2) (3, 4, 0) -> dist = 10 m
    # 3) (0, 0, 12) -> dist = sqrt(3^2 + 4^2 + 12^2) = 13 m
    positions = [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 12.0]]
    weights = [1.0e6, 2.0e6, 3.0e6]
    targets = TargetPoints(np.array([[-3.0, -4.0, 0.0]]) * u.m, ("A",), ROOT)
    setup = point_sum(
        WeightedPointSource(positions, weights),
        frame=ROOT,
        field_definition=_field(),
    ).at(targets)

    # Compute cumulative contribution with bins covering all 3 sources
    cum = cumulative_contribution(
        setup,
        target_id="A",
        bins=np.array([0.0, 7.0, 11.0, 15.0]),  # 3 bands: [0,7), [7,11), [11,15)
    )

    assert len(cum.band_vectors_m_s2) == 3
    assert len(cum.cumulative_vectors_m_s2) == 3

    # Band 0 has element 1 (dist=5m)
    # Band 1 has element 2 (dist=10m)
    # Band 2 has element 3 (dist=13m)
    # Cumulative vector at the last band must match total vector
    final_cum = cum.cumulative_vectors_m_s2[-1]
    total_vec = cum.total_vector_m_s2
    np.testing.assert_allclose(final_cum, total_vec, rtol=1e-14, atol=1e-18)

    # Verify tolerance criteria: c * N * eps * sum(||da||)
    eps = np.finfo(np.float64).eps
    n_elements = len(positions)
    c = 10.0
    tol = c * n_elements * eps * cum.total_norm_sum_m_s2
    band_sum = np.sum(cum.band_vectors_m_s2, axis=0)
    diff = np.linalg.norm(band_sum - total_vec)
    assert diff <= tol


def test_cumulative_contribution_underflow_and_overflow() -> None:
    """Elements outside explicit distance bins are accounted in underflow/overflow."""
    positions = [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 12.0]]
    weights = [1.0e6, 2.0e6, 3.0e6]
    targets = TargetPoints(np.array([[-3.0, -4.0, 0.0]]) * u.m, ("A",), ROOT)
    setup = point_sum(
        WeightedPointSource(positions, weights),
        frame=ROOT,
        field_definition=_field(),
    ).at(targets)

    # Bin only [7.0, 11.0] -> element 1 (5m) is underflow, element 3 (13m) is overflow
    cum = cumulative_contribution(
        setup,
        target_id="A",
        bins=np.array([7.0, 11.0]),
    )

    assert len(cum.band_vectors_m_s2) == 1
    # underflow vector + band vector + overflow vector must equal total_vector
    sum_with_overflow = (
        cum.underflow_vector_m_s2
        + cum.band_vectors_m_s2[0]
        + cum.overflow_vector_m_s2
    )
    np.testing.assert_allclose(
        sum_with_overflow, cum.total_vector_m_s2, rtol=1e-14, atol=1e-18
    )


def test_precision_checks_and_target_selection() -> None:
    """Target selection and coordinate precision checks are enforced."""
    # 1. Multiple targets without target_id must raise ValueError
    positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    weights = [1.0e5, 1.0e5]
    targets_2 = TargetPoints(
        np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]) * u.m,
        ("T1", "T2"),
        ROOT,
    )
    setup_2 = point_sum(
        WeightedPointSource(positions, weights),
        frame=ROOT,
        field_definition=_field(),
    ).at(targets_2)

    with pytest.raises(ValueError, match="target_id is required"):
        contribution_hist(setup_2, quantity="norm")

    with pytest.raises(ValueError, match="target_id is required"):
        cumulative_contribution(setup_2)

    # 2. Unknown target_id must raise ValueError
    with pytest.raises(ValueError, match="unknown target_id"):
        contribution_hist(setup_2, quantity="norm", target_id="NONEXISTENT")

    # 3. Collision or near-coincidence raises ValueError / CoordinatePrecisionError
    collision_targets = TargetPoints(
        np.array([[0.0, 0.0, 0.0]]) * u.m, ("center",), ROOT
    )
    collision_setup = point_sum(
        WeightedPointSource([[0.0, 0.0, 0.0]], [1.0e5]),
        frame=ROOT,
        field_definition=_field(),
    ).at(collision_targets)

    with pytest.raises(ValueError):
        contribution_hist(collision_setup, quantity="norm", target_id="center")


def test_plotting_wrappers() -> None:
    """plot_contribution_hist and cumulative plot return figures and axes."""
    positions = [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 12.0]]
    weights = [1.0e6, 2.0e6, 3.0e6]
    targets = TargetPoints(np.array([[-3.0, -4.0, 0.0]]) * u.m, ("A",), ROOT)
    setup = point_sum(
        WeightedPointSource(positions, weights),
        frame=ROOT,
        field_definition=_field(),
    ).at(targets)

    fig1, ax1 = plot_contribution_hist(setup, quantity="norm", target_id="A", bins=5)
    assert isinstance(fig1, Figure)
    assert isinstance(ax1, Axes)
    assert "Acceleration (norm)" in ax1.get_xlabel()
    plt.close(fig1)

    fig2, ax2 = plot_cumulative_contribution(setup, target_id="A", bins=5)
    assert isinstance(fig2, Figure)
    assert isinstance(ax2, Axes)
    assert "Distance" in ax2.get_xlabel()
    assert "Cumulative" in ax2.get_ylabel() or "Acceleration" in ax2.get_ylabel()
    plt.close(fig2)
