"""Gravitational contribution histograms and cumulative distance sums.

Provides:
- contribution_hist: Histograms of per-element contribution norms and signed
  vector components.
- cumulative_contribution: Distance-band vector sums and cumulative acceleration
  vector a(<R).
- plot_contribution_hist: Plotting wrapper for contribution histograms.
- plot_cumulative_contribution: Plotting wrapper for cumulative distance sums.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy import units as u

from gragra._plotting._diagnostics import (
    _check_frame,
    _extract_map_data,
    _extract_values,
    _resolve_source,
    _resolved_from,
    _select_target,
)
from gragra._plotting._hist import (
    _aggregate,
    _check_plot_bin_request,
    _resolve_edges,
    _to_histogram,
    _validate_plot_axes,
)

if TYPE_CHECKING:
    from gwexpy.histogram import Histogram
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from gragra.geometry import Frame


@dataclass(frozen=True)
class CumulativeContribution:
    """Distance-band vector sums and cumulative acceleration."""

    distance_edges_m: np.ndarray
    band_vectors_m_s2: np.ndarray
    cumulative_vectors_m_s2: np.ndarray
    underflow_vector_m_s2: np.ndarray
    overflow_vector_m_s2: np.ndarray
    total_vector_m_s2: np.ndarray
    total_norm_sum_m_s2: float
    target_id: str
    frame: Frame


_QUANTITY_MAP = {
    "norm": "delta_a_norm",
    "delta_a_norm": "delta_a_norm",
    "x": "delta_a_x",
    "component_x": "delta_a_x",
    "delta_a_x": "delta_a_x",
    "y": "delta_a_y",
    "component_y": "delta_a_y",
    "delta_a_y": "delta_a_y",
    "z": "delta_a_z",
    "component_z": "delta_a_z",
    "delta_a_z": "delta_a_z",
}


def contribution_hist(
    setup: Any,
    quantity: str = "norm",
    *,
    target_id: str | None = None,
    bins: int | str | np.ndarray | u.Quantity = "auto",
    frame: Frame | None = None,
) -> Histogram:
    """Histogram per-element gravitational acceleration contribution.

    Parameters
    ----------
    setup : EvaluationSetup or ResolvedEvaluation
        A setup built from a discrete point source.
    quantity : {"norm", "x", "y", "z"}, default="norm"
        Contribution component to histogram.
    target_id : str | None, default=None
        Target identifier. Required if multiple targets are present.
    bins : int, "auto", 1-D array, or astropy Quantity, default="auto"
        Binning specification.
    frame : Frame | None, default=None
        Coordinate frame for vectors and components.

    Returns
    -------
    Histogram
        A GWexpy Histogram object of acceleration contributions in m/s^2.
    """
    if quantity not in _QUANTITY_MAP:
        raise ValueError(
            f"unknown contribution quantity {quantity!r}; "
            f"expected one of {sorted(_QUANTITY_MAP.keys())}"
        )
    diag_quantity = _QUANTITY_MAP[quantity]

    resolved = _resolved_from(setup)
    _resolve_source(resolved)  # Validates discrete source presence
    target_index = _select_target(resolved, target_id, required=True)
    target_id_str = resolved.targets.ids[target_index]
    _check_frame(resolved, frame)

    map_data = _extract_map_data(
        resolved, target_id=target_id_str, quantity=diag_quantity, frame=frame
    )
    values = map_data.values

    acc_unit = u.m / (u.s**2)
    edges = _resolve_edges(bins, values, acc_unit)
    counts, underflow, overflow = _aggregate(values, weights=None, edges=edges)

    name = f"acceleration {quantity} target={target_id_str}"
    return _to_histogram(
        counts,
        edges,
        xunit=acc_unit,
        weight="count",
        name=name,
        underflow=underflow,
        overflow=overflow,
    )


def cumulative_contribution(
    setup: Any,
    *,
    target_id: str | None = None,
    bins: int | str | np.ndarray | u.Quantity = "auto",
    frame: Frame | None = None,
) -> CumulativeContribution:
    """Compute distance-band vector sums and cumulative acceleration a(<R).

    Parameters
    ----------
    setup : EvaluationSetup or ResolvedEvaluation
        A setup built from a discrete point source.
    target_id : str | None, default=None
        Target identifier. Required if multiple targets are present.
    bins : int, "auto", 1-D array, or astropy Quantity, default="auto"
        Distance bin edges or count in meters.
    frame : Frame | None, default=None
        Coordinate frame for vectors. Defaults to computation frame.

    Returns
    -------
    CumulativeContribution
        A structured result containing band vectors, cumulative vectors,
        underflow, overflow, and total sums.
    """
    resolved = _resolved_from(setup)
    view = _resolve_source(resolved)
    target_index = _select_target(resolved, target_id, required=True)
    target_id_str = resolved.targets.ids[target_index]
    _check_frame(resolved, frame)
    target_frame = resolved.computation_frame if frame is None else frame

    distances_m = _extract_values(
        view, "distance", resolved=resolved, target_index=target_index, frame=frame
    )
    map_data = _extract_map_data(
        resolved, target_id=target_id_str, quantity="delta_a_norm", frame=frame
    )
    delta_a_vecs = map_data.delta_a_m_s2
    da_norms = map_data.values

    edges = _resolve_edges(bins, distances_m, u.m)
    n_bands = len(edges) - 1

    below = distances_m < edges[0]
    above = distances_m > edges[-1]
    underflow_vec = (
        np.sum(delta_a_vecs[below], axis=0) if np.any(below) else np.zeros(3)
    )
    overflow_vec = (
        np.sum(delta_a_vecs[above], axis=0) if np.any(above) else np.zeros(3)
    )

    band_vecs = np.zeros((n_bands, 3), dtype=np.float64)
    for j in range(n_bands):
        if j == n_bands - 1:
            mask = (distances_m >= edges[j]) & (distances_m <= edges[j + 1])
        else:
            mask = (distances_m >= edges[j]) & (distances_m < edges[j + 1])
        if np.any(mask):
            band_vecs[j] = np.sum(delta_a_vecs[mask], axis=0)

    cum_vecs = underflow_vec + np.cumsum(band_vecs, axis=0)
    total_vec = np.sum(delta_a_vecs, axis=0)
    total_norm_sum = float(np.sum(da_norms))

    return CumulativeContribution(
        distance_edges_m=edges,
        band_vectors_m_s2=band_vecs,
        cumulative_vectors_m_s2=cum_vecs,
        underflow_vector_m_s2=underflow_vec,
        overflow_vector_m_s2=overflow_vec,
        total_vector_m_s2=total_vec,
        total_norm_sum_m_s2=total_norm_sum,
        target_id=target_id_str,
        frame=target_frame,
    )


def plot_contribution_hist(
    setup: Any,
    quantity: str = "norm",
    *,
    target_id: str | None = None,
    bins: int | str | np.ndarray | u.Quantity = "auto",
    frame: Frame | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot per-element gravitational acceleration contribution histogram.

    Parameters
    ----------
    setup : EvaluationSetup or ResolvedEvaluation
        A setup built from a discrete point source.
    quantity : {"norm", "x", "y", "z"}, default="norm"
        Contribution component to histogram.
    target_id : str | None, default=None
        Target identifier.
    bins : int, "auto", 1-D array, or astropy Quantity, default="auto"
        Binning specification.
    frame : Frame | None, default=None
        Coordinate frame for vectors.
    ax : Axes | None, default=None
        Axes to draw on.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes containing the histogram.
    """
    from matplotlib import pyplot as plt

    if ax is not None:
        _validate_plot_axes(ax)
    _check_plot_bin_request(bins)

    h = contribution_hist(
        setup, quantity=quantity, target_id=target_id, bins=bins, frame=frame
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    values = np.asarray(h.values)
    edges = np.asarray(h.edges)
    ax.stairs(values, edges, fill=True, alpha=0.5, color="#316aa5")
    ax.set_xlabel(f"Acceleration ({quantity}) [{h.xunit}]")
    ax.set_ylabel("Count")
    ax.set_title(h.name)

    return fig, ax


def plot_cumulative_contribution(
    setup: Any,
    *,
    quantity: str = "norm",
    target_id: str | None = None,
    bins: int | str | np.ndarray | u.Quantity = "auto",
    frame: Frame | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot cumulative gravitational acceleration a(<R) versus distance.

    Parameters
    ----------
    setup : EvaluationSetup or ResolvedEvaluation
        A setup built from a discrete point source.
    quantity : {"norm", "delta_a_norm", "x", "y", "z"}, default="norm"
        Component of cumulative acceleration vector to plot.
    target_id : str | None, default=None
        Target identifier.
    bins : int, "auto", 1-D array, or astropy Quantity, default="auto"
        Distance binning specification.
    frame : Frame | None, default=None
        Coordinate frame for vectors.
    ax : Axes | None, default=None
        Axes to draw on.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes containing the cumulative curve.
    """
    from matplotlib import pyplot as plt

    if ax is not None:
        _validate_plot_axes(ax)
    _check_plot_bin_request(bins)

    valid_quantities = {"norm", "delta_a_norm", "x", "y", "z"}
    if quantity not in valid_quantities:
        raise ValueError(
            f"unknown cumulative quantity {quantity!r}; "
            f"expected one of {sorted(valid_quantities)}"
        )

    cum = cumulative_contribution(
        setup, target_id=target_id, bins=bins, frame=frame
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    edges = cum.distance_edges_m
    # Use right bin edges so that a(<R) is evaluated at the upper boundary R
    r_eval = edges[1:]

    if quantity in ("norm", "delta_a_norm"):
        cum_vals = np.linalg.norm(cum.cumulative_vectors_m_s2, axis=1)
        y_label = "Cumulative Acceleration [m/s^2]"
        curve_label = r"$\|\mathbf{a}(<R)\|$"
    elif quantity == "x":
        cum_vals = cum.cumulative_vectors_m_s2[:, 0]
        y_label = "Cumulative a_x [m/s^2]"
        curve_label = r"$a_x(<R)$"
    elif quantity == "y":
        cum_vals = cum.cumulative_vectors_m_s2[:, 1]
        y_label = "Cumulative a_y [m/s^2]"
        curve_label = r"$a_y(<R)$"
    elif quantity == "z":
        cum_vals = cum.cumulative_vectors_m_s2[:, 2]
        y_label = "Cumulative a_z [m/s^2]"
        curve_label = r"$a_z(<R)$"
    else:
        cum_vals = np.linalg.norm(cum.cumulative_vectors_m_s2, axis=1)
        y_label = "Cumulative Acceleration [m/s^2]"
        curve_label = r"$\|\mathbf{a}(<R)\|$"

    ax.plot(
        r_eval,
        cum_vals,
        marker="o",
        color="#27864b",
        label=curve_label,
    )
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel(y_label)
    ax.set_title(f"Cumulative contribution target={cum.target_id}")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")

    return fig, ax
