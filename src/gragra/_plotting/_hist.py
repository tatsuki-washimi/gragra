"""``hist``: histograms of source-element diagnostics as GWexpy ``Histogram``.

Matplotlib and GWexpy are imported inside functions only, so importing this
module does not load either package.
"""

from __future__ import annotations

import math
import numbers
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

from gragra._plotting._diagnostics import (
    QUANTITIES,
    STANDARD_XUNIT,
    TARGET_INDEPENDENT_QUANTITIES,
    _check_frame,
    _extract_values,
    _require_finite,
    _resolve_source,
    _resolved_from,
    _select_target,
)

if TYPE_CHECKING:
    from gwexpy.histogram import Histogram
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from gragra._evaluation_setup import ResolvedEvaluation
    from gragra.evaluation import EvaluationSetup
    from gragra.geometry import Frame

_MAX_BINS = 1_000_000
_MAX_PLOT_BINS = 100_000
_WEIGHTS = ("count", "volume")
_AUTO_RULES = ("legacy", "relaxed")
_COMPLEX_BINS_MESSAGE = "bins must be real; complex values are not accepted"
_RANGE_OVERFLOW_MESSAGE = (
    "the data range is not finite or overflows float64, so bin edges cannot be derived"
)
# Smallest normal float64. A nonzero magnitude below it is subnormal.
_TINY = float(np.finfo(np.float64).tiny)


def hist(
    setup: EvaluationSetup | ResolvedEvaluation,
    quantity: str,
    *,
    target_id: str | None = None,
    bins: int | str | np.ndarray | u.Quantity = "auto",
    weight: str = "count",
    xunit: str | u.UnitBase | None = None,
    frame: Frame | None = None,
) -> Histogram:
    """Histogram one source-element diagnostic as a GWexpy ``Histogram``.

    Calling this function loads GWexpy, and with it Matplotlib's ``pyplot``.
    Importing ``gragra.plotting`` and obtaining ``hist`` does not load them.

    Parameters
    ----------
    setup : EvaluationSetup or ResolvedEvaluation
        Fixed request built from a point-sum source, for example
        ``point_sum(source, ...).at(targets)``. Analytic models have no
        source elements and raise ``ValueError``.
    quantity : {"density", "height", "distance", "inverse_square"}
        Diagnostic to histogram. ``density`` is the per-element density in
        kg/m^3. ``height`` is the z coordinate of each element in ``frame``;
        it is not an elevation above a geoid or ground surface. ``distance``
        is the Euclidean distance from the selected target. ``inverse_square``
        is ``1/r**2`` for that distance. It is the plain inverse square, not
        a softened kernel, and softening is not supported.
    target_id : str, optional, default None
        Target used by ``distance`` and ``inverse_square``. It is required
        when the setup has more than one target. It must be omitted for
        ``density`` and ``height``, which do not depend on targets, so those
        quantities also work for a setup with no targets.
    bins : int, "auto", 1-D array, or astropy Quantity, default "auto"
        An integer gives that many equal bins over the data range. ``"auto"``
        uses NumPy's automatic rule. A 1-D array gives the bin edges. Plain
        arrays are read in ``xunit``. A Quantity is converted to ``xunit``
        and must have a compatible unit. Edges must be real, finite and
        strictly increasing.
    weight : {"count", "volume"}, default "count"
        ``"count"`` gives each element weight 1, so the values are
        dimensionless. ``"volume"`` weights each element by its volume in
        m^3. Density and volume are available only for sources whose mass
        weights can be checked against them: a ``VolumeElementSource``, or a
        source from ``gragra.gravity_sources.discretize`` whose discretization
        records ``cell_volume_m3``. Other sources raise ``ValueError``.
        ``density`` with ``weight="volume"`` is accepted only under the same
        condition. For a source from ``gragra.gravity_sources.discretize``, the
        volume is the discretized cell volume without boundary-cell correction,
        so it can differ from the geometric volume of the sphere.
    xunit : str or astropy unit, optional, default None
        Unit of the horizontal axis (the bin edges) only. The default is
        ``kg/m3`` for density, ``m`` for height and distance, and ``1/m2``
        for inverse_square. The unit must be equivalent to the default.
        Converting ``xunit`` never changes the volume weights.
    frame : Frame, optional, default None
        Frame used for ``height``. ``None`` means the computation frame of
        the setup, which is the frame of the source. It is not the root
        frame. This differs from ``plot_field``, where ``frame`` is used only
        for the figure labels. Here, ``height`` is transformed into ``frame``.
        Distances are invariant under frame changes, so ``distance`` and
        ``inverse_square`` are always computed in the computation frame. For
        those quantities ``frame`` only has to be connected to it. For
        ``density``, ``frame`` is checked for connection and is otherwise
        ignored.

    Returns
    -------
    gwexpy.histogram.Histogram
        Counts or volumes per bin. The vertical unit is dimensionless for
        ``weight="count"`` and m^3 for ``weight="volume"``. Elements outside
        the edges are added to ``underflow`` and ``overflow``; they are not
        dropped. ``sumw2`` is not set because the sums are deterministic.

    Raises
    ------
    TypeError
        When ``setup`` is not an EvaluationSetup or ResolvedEvaluation, when
        ``frame`` is not a Frame or None, or when the source binding is not a
        supported point source.
    ValueError
        For an unsupported quantity or weight; for an analytic model; for a
        missing, unknown or unexpected ``target_id``, including a ``target_id``
        given with ``density`` or ``height``; for an ``xunit`` that is not a
        valid unit or not equivalent to the default; for a bin unit that is not
        compatible with ``xunit``; for an integer bin count that is a bool,
        below 1, or above 1,000,000; for edges that are not a 1-D sequence of at
        least two real, finite, strictly increasing values; for complex bins;
        for an ``xunit`` conversion that overflows or underflows float64, or
        whose conversion factor is not representable in float64; for a data
        range that overflows float64 when edges are derived; for density or volume
        that cannot be verified, including a source without density or volume
        and a source from ``gragra.gravity_sources.discretize`` whose
        ``cell_volume_m3`` is missing or invalid;
        for a frame that is not connected to the computation frame; for a
        non-finite value; for squared separations that overflow, or that are
        zero or subnormal for a nonzero separation; and for an ``"auto"`` bin
        count above 1,000,000. For ``distance`` and ``inverse_square``, a source
        element that coincides with the selected target, or lies so close to it
        across frames that the coordinate difference is not reliable, raises a
        ``ValueError`` subclass carrying ``source_index`` and ``target_id``
        attributes.

    Notes
    -----
    Density, height, distance and inverse_square are geometric or mass
    diagnostics. They do not multiply by G, and they do not apply a mass
    weight. ``inverse_square`` is not the magnitude of a gravitational
    acceleration. This function does not establish the scientific acceptance
    of any physical model.

    Density is rejected for ``WeightedPointSource``, ``PointMass`` and
    ``PointMassCloud``, and for any source without the volume information
    that would let its mass weights be matched to a density. The weights are
    not inverted to recover a density.

    When all values are equal, NumPy widens the data range to
    ``value +/- 0.5`` before it derives the edges. This happens only for integer
    bins and ``"auto"``. Explicit edges are not widened. The last edge belongs
    to the last bin, so a value equal to the last edge is not counted as
    overflow.

    The ``"auto"`` bin count is estimated before any edges are allocated.
    The estimate follows the NumPy version that is installed, because NumPy
    changed its ``auto`` rule. If the estimate exceeds 1,000,000 bins, the
    call raises ``ValueError``; pass explicit edges in that case. Values are
    float64, so NumPy's integer minimum-width rule does not apply.

    For ``distance`` and ``inverse_square``, a nonzero separation whose square
    is zero or subnormal is rejected. This check is stricter than the
    ``validate()`` method of the setup. Where the squared separation is near
    the float64 minimum, ``1/r**2`` loses precision. Only the selected target
    is checked. ``validate()`` checks every target, so this function can
    succeed while ``validate()`` fails for another target.

    When ``frame`` differs from the computation frame, each point is
    transformed separately in Python. The call is therefore slow for large
    sources.
    """
    resolved = _resolved_from(setup)
    _check_quantity(quantity)
    _check_weight(weight)
    xunit_unit = _xunit_for(quantity, xunit)
    _check_frame(resolved, frame)
    view = _resolve_source(resolved)
    target_index = _select_target(
        resolved,
        target_id,
        required=quantity not in TARGET_INDEPENDENT_QUANTITIES,
    )
    values = _require_finite(
        _extract_values(
            view,
            quantity,
            resolved=resolved,
            target_index=target_index,
            frame=frame,
        ),
        quantity,
    )
    weights = None if weight == "count" else _require_finite(view.volumes(), "volume")
    values_x = _to_xunit(values, STANDARD_XUNIT[quantity], xunit_unit)
    edges = _resolve_edges(bins, values_x, xunit_unit)
    counts, underflow, overflow = _aggregate(values_x, weights, edges)
    target_name = None if target_index is None else resolved.targets.ids[target_index]
    return _to_histogram(
        counts,
        edges,
        xunit=xunit_unit,
        weight=weight,
        name=_histogram_name(quantity, weight, target_name),
        underflow=underflow,
        overflow=overflow,
    )


def estimate_auto_bin_count(values: object, *, rule: str | None = None) -> int:
    """Estimate the bin count that NumPy's ``bins="auto"`` would produce.

    Parameters
    ----------
    values : array-like
        Finite data values. They are flattened to one dimension.
    rule : {"legacy", "relaxed"}, optional, default None
        Formula to use. ``"legacy"`` is the rule of older NumPy releases;
        ``"relaxed"`` is the rule of NumPy 2.3 and later. When None, the
        formula is identified once from the installed NumPy through its public
        ``histogram_bin_edges`` output. If the installed rule cannot be
        identified, the larger of the two estimates is used.

    Returns
    -------
    int
        The estimated bin count. It is 1 for empty or zero-width data.

    Raises
    ------
    ValueError
        When ``rule`` is unknown, when the data range is not finite or
        overflows float64, or when the estimate exceeds 1,000,000 bins.

    Notes
    -----
    The estimate uses only the range and percentiles, so it does not allocate
    bin edges.
    """
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size == 0:
        return 1
    selected = _numpy_auto_rule() if rule is None else _checked_rule(rule)
    if selected == "conservative":
        ratio = max(_auto_ratio(data, "legacy"), _auto_ratio(data, "relaxed"))
    else:
        ratio = _auto_ratio(data, selected)
    if not math.isfinite(ratio):
        raise ValueError(_RANGE_OVERFLOW_MESSAGE)
    if math.ceil(ratio) > _MAX_BINS:
        raise ValueError(
            f"bins='auto' would produce about {ratio:.3g} bins, above the limit "
            f"of {_MAX_BINS:,}; pass explicit bin edges or an integer bin count"
        )
    return int(math.ceil(ratio))


def _checked_rule(rule: str) -> str:
    if rule not in _AUTO_RULES:
        raise ValueError(f"rule must be one of {_AUTO_RULES}, got {rule!r}")
    return rule


def _auto_ratio(data: np.ndarray, rule: str) -> float:
    """Unrounded ``range / width`` for one NumPy auto rule; 1.0 for zero width."""
    size = data.size
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        q75, q25 = np.percentile(data, [75, 25])
        spread = float(np.max(data) - np.min(data))
        h_fd = 2.0 * (q75 - q25) * size ** (-1.0 / 3.0)
        h_sturges = spread / (np.log2(size) + 1.0)
        h_sqrt = spread / np.sqrt(size)
        if rule == "legacy":
            width = min(h_fd, h_sturges) if h_fd else h_sturges
        else:
            width = min(max(h_fd, h_sqrt / 2.0), h_sturges)
        if width == 0.0:
            return 1.0
        return float(spread / width)


def _rule_count(data: np.ndarray, rule: str) -> int:
    ratio = _auto_ratio(data, rule)
    return math.ceil(ratio) if math.isfinite(ratio) else -1


@lru_cache(maxsize=1)
def _numpy_auto_rule() -> str:
    """Identify the installed NumPy ``auto`` rule from public behaviour."""
    calibration = np.zeros(10_000)
    calibration[0] = 1.0  # IQR is 0 and the spread is small: the rules differ.
    expected = len(np.histogram_bin_edges(calibration, "auto")) - 1
    matches = [
        rule for rule in _AUTO_RULES if _rule_count(calibration, rule) == expected
    ]
    return matches[0] if len(matches) == 1 else "conservative"


def _check_quantity(quantity: object) -> None:
    if quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {QUANTITIES}, got {quantity!r}")


def _check_weight(weight: object) -> None:
    if weight not in _WEIGHTS:
        raise ValueError(f"weight must be one of {_WEIGHTS}, got {weight!r}")


def _xunit_for(quantity: str, xunit: object) -> u.UnitBase:
    default = STANDARD_XUNIT[quantity]
    if xunit is None:
        return default
    try:
        unit = u.Unit(xunit)
        equivalent = unit.is_equivalent(default)
    except (TypeError, ValueError, u.UnitsError) as exc:
        raise ValueError(f"xunit {xunit!r} is not a valid unit: {exc}") from exc
    if isinstance(unit, u.FunctionUnitBase):
        raise ValueError(
            f"xunit {xunit!r} must be a plain unit, not a logarithmic or function unit"
        )
    if not equivalent:
        raise ValueError(
            f"xunit {xunit!r} is not equivalent to {default} for quantity {quantity!r}"
        )
    return unit


def _to_xunit(values: np.ndarray, si_unit: u.UnitBase, xunit: u.UnitBase) -> np.ndarray:
    """Convert SI values to ``xunit``; reject a conversion that leaves float64."""
    scale = _conversion_scale(si_unit, xunit)
    if scale == 1.0:
        return values
    return _scale_checked(values, scale, xunit, "element")


def _conversion_scale(from_unit: u.UnitBase, xunit: u.UnitBase) -> float:
    """Factor from ``from_unit`` to ``xunit``; it must be a usable float64."""
    scale = float(from_unit.to(xunit))
    if not math.isfinite(scale) or abs(scale) < _TINY:
        raise ValueError(
            f"conversion factor from {from_unit} to xunit {xunit} is not "
            f"representable in float64 (factor {scale!r})"
        )
    return scale


def _scale_checked(
    values: np.ndarray, scale: float, xunit: u.UnitBase, where: str
) -> np.ndarray:
    """Multiply by ``scale``; reject a finite value that overflows or underflows.

    A non-finite input is not reported here, so the caller's finiteness check
    reports it. Exact zeros stay zero. ``where`` names the element kind in the
    message: ``"element"`` or ``"bin edge"``.
    """
    with np.errstate(over="ignore", under="ignore"):
        converted = np.asarray(values * scale, dtype=np.float64)
    finite_in = np.isfinite(values)
    overflow = finite_in & ~np.isfinite(converted)
    underflow = finite_in & (values != 0.0) & (np.abs(converted) < _TINY)
    offending = overflow | underflow
    if offending.any():
        # flatnonzero flattens, so a 0-d input reports index 0 safely.
        index = int(np.flatnonzero(offending)[0])
        kind = "overflows" if np.ravel(overflow)[index] else "underflows"
        raise ValueError(
            f"conversion to xunit {xunit} {kind} float64 at {where} index {index}"
        )
    return converted


def _resolve_edges(bins: object, values: np.ndarray, xunit: u.UnitBase) -> np.ndarray:
    """Bin edges in ``xunit``. Automatic counts are checked before allocation."""
    if isinstance(bins, (bool, np.bool_)):
        raise ValueError("bins must not be a boolean")
    if isinstance(bins, str):
        if bins != "auto":
            raise ValueError(f"bins must be 'auto' when a string, got {bins!r}")
        estimate_auto_bin_count(values)
        edges = np.histogram_bin_edges(values, "auto")
        _check_bin_count(len(edges) - 1)
        return edges
    if isinstance(bins, numbers.Integral):
        count = int(bins)
        if not 1 <= count <= _MAX_BINS:
            raise ValueError(
                f"integer bins must be between 1 and {_MAX_BINS:,}, got {count}"
            )
        _require_finite_range(values)
        return np.histogram_bin_edges(values, bins=count).astype(np.float64)
    if isinstance(bins, u.Quantity):
        if not bins.unit.is_equivalent(xunit):
            raise ValueError(f"bins unit {bins.unit} is not compatible with {xunit}")
        if np.iscomplexobj(bins.value):
            raise ValueError(_COMPLEX_BINS_MESSAGE)
        scale = _conversion_scale(bins.unit, xunit)
        edges = _scale_checked(
            np.asarray(bins.value, dtype=np.float64), scale, xunit, "bin edge"
        )
    else:
        if np.iscomplexobj(bins):
            raise ValueError(_COMPLEX_BINS_MESSAGE)
        edges = np.array(bins, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError(
            "bins must be an int, 'auto', or a 1-D sequence with at least two edges"
        )
    if not np.isfinite(edges).all():
        index = int(np.flatnonzero(~np.isfinite(edges))[0])
        raise ValueError(f"bin edges must be finite; edge index {index} is not")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("bin edges must be strictly increasing")
    _check_bin_count(len(edges) - 1)
    return edges


def _require_finite_range(values: np.ndarray) -> None:
    """Reject a data range that overflows float64 before NumPy derives edges."""
    if values.size == 0:
        return
    with np.errstate(over="ignore", invalid="ignore"):
        spread = np.max(values) - np.min(values)
    if not np.isfinite(spread):
        raise ValueError(_RANGE_OVERFLOW_MESSAGE)


def _check_bin_count(count: int) -> None:
    if count > _MAX_BINS:
        raise ValueError(
            f"{count:,} bins exceed the limit of {_MAX_BINS:,}; pass explicit edges"
        )


def _aggregate(
    values: np.ndarray, weights: np.ndarray | None, edges: np.ndarray
) -> tuple[np.ndarray, float, float]:
    counts, _ = np.histogram(values, bins=edges, weights=weights)
    below = values < edges[0]
    above = values > edges[-1]
    if weights is None:
        underflow = float(np.count_nonzero(below))
        overflow = float(np.count_nonzero(above))
    else:
        underflow = float(np.sum(weights, where=below))
        overflow = float(np.sum(weights, where=above))
    return counts.astype(np.float64), underflow, overflow


def _histogram_name(quantity: str, weight: str, target_name: str | None) -> str:
    parts = [quantity, weight]
    if target_name is not None:
        parts.append(f"target={target_name}")
    return " ".join(parts)


def _to_histogram(
    counts: np.ndarray,
    edges: np.ndarray,
    *,
    xunit: u.UnitBase,
    weight: str,
    name: str,
    underflow: float,
    overflow: float,
) -> Histogram:
    from gwexpy.histogram import Histogram

    unit = None if weight == "count" else u.m**3
    return Histogram(
        values=counts,
        edges=edges,
        unit=unit,
        xunit=xunit,
        underflow=underflow,
        overflow=overflow,
        name=name,
    )


def plot_hist(
    setup: EvaluationSetup | ResolvedEvaluation,
    quantity: str,
    *,
    target_id: str | None = None,
    bins: int | str | np.ndarray | u.Quantity = "auto",
    weight: str = "count",
    xunit: str | u.UnitBase | None = None,
    frame: Frame | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Draw the histogram that ``hist()`` returns as a step plot.

    The arguments have the same meaning as in ``hist()``. The drawn values
    and edges are those of the ``Histogram`` that ``hist()`` returns for the
    same arguments. Nothing is re-binned here. Calling this function loads
    GWexpy and Matplotlib's ``pyplot`` through ``hist()``.

    Parameters
    ----------
    setup : EvaluationSetup or ResolvedEvaluation
        Fixed request built from a point-sum source, for example
        ``point_sum(source, ...).at(targets)``. Analytic models raise
        ``ValueError``.
    quantity : {"density", "height", "distance", "inverse_square"}
        Diagnostic to draw. See ``hist()``.
    target_id : str, optional, default None
        Target for ``distance`` and ``inverse_square``. It is required when
        the setup has more than one target. It must be omitted for ``density``
        and ``height``.
    bins : int, "auto", 1-D array, or astropy Quantity, default "auto"
        Bin count or bin edges, read as in ``hist()``. An integer above
        100,000 is rejected before anything is computed. ``"auto"``, explicit
        edges and Quantity edges are checked after ``hist()`` has built the
        histogram.
    weight : {"count", "volume"}, default "count"
        ``"count"`` counts elements. ``"volume"`` sums their volumes in m^3.
        The sources that allow each weight are those of ``hist()``.
    xunit : str or astropy unit, optional, default None
        Unit of the horizontal axis (the bin edges). The default depends on
        the quantity; see ``hist()``.
    frame : Frame, optional, default None
        Frame used for ``height``. See ``hist()``.
    ax : matplotlib.axes.Axes, optional, default None
        Empty, 2-D Cartesian axes with linear scales to draw on. If omitted, a
        new figure is created. The axes is not closed.

    Returns
    -------
    matplotlib.figure.Figure
        The figure that contains the drawn histogram. A figure created by this
        function stays open; close it with ``matplotlib.pyplot.close`` when it
        is no longer needed.

    Raises
    ------
    TypeError
        For the same setup, frame and binding errors as ``hist()``.
    ValueError
        For an ``ax`` that is not an empty, 2-D, linear Matplotlib Axes; for
        more than 100,000 bins, which is stricter than the 1,000,000 limit of
        ``hist()``; and for every other error that ``hist()`` raises. The axes
        check and the integer bin-count check run before anything is drawn.

    Notes
    -----
    With ``bins="auto"`` the bin count is known only after ``hist()`` has
    estimated it, so the 100,000-bin limit is checked after the histogram is
    built. Integer bin counts are checked before the call.

    When the underflow or overflow is nonzero, the axes shows a text note with
    both values. Those elements are not part of the drawn bins.
    """
    from matplotlib import pyplot as plt

    if ax is not None:
        _validate_plot_axes(ax)
    _check_plot_bin_request(bins)
    histogram = hist(
        setup,
        quantity,
        target_id=target_id,
        bins=bins,
        weight=weight,
        xunit=xunit,
        frame=frame,
    )
    values = histogram.values.value
    if len(values) > _MAX_PLOT_BINS:
        raise ValueError(
            f"plot_hist draws at most {_MAX_PLOT_BINS:,} bins, got {len(values):,}; "
            "use hist() for the full histogram or fewer bins"
        )
    element_count = _source_element_count(setup)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
    else:
        fig = ax.figure
    ax.stairs(values, histogram.edges.value)
    ax.set_xlabel(f"{quantity} [{histogram.xunit.to_string()}]")
    ax.set_ylabel(_y_label(weight))
    ax.set_title(f"{histogram.name}; {element_count} source elements")
    underflow = float(histogram.underflow.value)
    overflow = float(histogram.overflow.value)
    if underflow or overflow:
        ax.text(
            0.98,
            0.98,
            f"underflow: {underflow:.6g}\noverflow: {overflow:.6g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
    return fig


def _validate_plot_axes(ax: object) -> None:
    from matplotlib.axes import Axes as _Axes

    if not isinstance(ax, _Axes):
        raise ValueError("plot_hist requires ax to be a Matplotlib Axes instance")
    if ax.name != "rectilinear":
        raise ValueError("plot_hist requires 2-D Cartesian (rectilinear) axes")
    if ax.get_xscale() != "linear" or ax.get_yscale() != "linear":
        raise ValueError("plot_hist requires linear x and y scales")
    if ax.has_data():
        raise ValueError("plot_hist requires empty axes; pass a new Axes or omit ax")


def _check_plot_bin_request(bins: object) -> None:
    """Reject an integer bin count above the plotting limit before any work."""
    if isinstance(bins, numbers.Integral) and int(bins) > _MAX_PLOT_BINS:
        raise ValueError(
            f"plot_hist draws at most {_MAX_PLOT_BINS:,} bins, got {int(bins):,}; "
            "use hist() for the full histogram or fewer bins"
        )


def _source_element_count(setup: object) -> int:
    return int(_resolve_source(_resolved_from(setup)).positions_m.shape[0])


def _y_label(weight: str) -> str:
    if weight == "count":
        return "count"
    return f"volume [{(u.m**3).to_string()}]"
