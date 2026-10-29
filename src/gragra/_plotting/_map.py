"""Spatial map visualization of gravity source diagnostics (plot_map).

Visualizes discrete source diagnostics (Newtonian kernel, gravitational
acceleration contributions) mapped across spatial positions in 2D or 3D.
Matplotlib is imported inside functions only, preserving zero-overhead lazy import.
"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Literal

import numpy as np

from gragra._plotting._diagnostics import _extract_map_data
from gragra._plotting._setup import _display_indices, _length_unit

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from gragra._evaluation_setup import ResolvedEvaluation
    from gragra.evaluation import EvaluationSetup
    from gragra.geometry import Frame

_VIEWS_2D = {
    "xy": (0, 1, "x", "y"),
    "xz": (0, 2, "x", "z"),
    "yz": (1, 2, "y", "z"),
}
_VALID_VIEWS = ("3d", "xy", "xz", "yz")
_VALID_SCALES = ("linear", "log")
_SIGNED_QUANTITIES = frozenset(
    {"delta_a_x", "delta_a_y", "delta_a_z", "delta_a_projection"}
)


def plot_map(
    setup: EvaluationSetup | ResolvedEvaluation,
    quantity: str,
    *,
    target_id: str | None = None,
    view: Literal["3d", "xy", "xz", "yz"] = "3d",
    frame: Frame | None = None,
    direction: tuple[float, float, float] | np.ndarray | None = None,
    coord_unit: str = "m",
    max_points: int | None = None,
    color_scale: Literal["linear", "log"] = "linear",
    ax: Axes | None = None,
) -> Figure:
    """Plot discrete source diagnostics over space in 2D or 3D.

    Calling this function loads Matplotlib's ``pyplot``. Importing
    ``gragra.plotting`` and obtaining ``plot_map`` does not load it.

    Parameters
    ----------
    setup : EvaluationSetup or ResolvedEvaluation
        Fixed request built from a point-sum source. Analytic models raise ValueError.
    quantity : {"kernel_norm", "delta_a_norm", "delta_a_x", "delta_a_y",
        "delta_a_z", "delta_a_projection"}
        Diagnostic to plot.
    target_id : str, optional, default None
        Target used for kernel and acceleration vectors. Required when the setup
        has more than one target.
    view : {"3d", "xy", "xz", "yz"}, default "3d"
        3D scatter plot or 2D projection plane.
    frame : Frame, optional, default None
        Coordinate frame in which positions, vectors, and components are expressed.
        If None, the setup's computation_frame is used.
    direction : 3-element sequence or array, optional, default None
        Direction vector for "delta_a_projection". Required if and only if
        quantity="delta_a_projection".
    coord_unit : str, default "m"
        Length unit for spatial coordinates. Must be compatible with meters.
    max_points : int or None, default None
        Maximum number of source points displayed. If specified, points are
        thinned with deterministic linear spacing. Must be a positive integer
        or None (no thinning; all points displayed).
        Total physical sum is always computed across all points.
    color_scale : {"linear", "log"}, default "linear"
        Colormap normalization scaling.
    ax : matplotlib.axes.Axes, optional, default None
        Empty Axes to draw on. Must match the view dimension (3D Axes for "3d",
        2D Cartesian for projections) and use linear scales.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the drawn scatter plot and colorbar.
    """
    from matplotlib import pyplot as plt

    # 1. Validate view and scale
    if view not in _VALID_VIEWS:
        raise ValueError(f"view must be one of {_VALID_VIEWS!r}, got {view!r}")
    if color_scale not in _VALID_SCALES:
        raise ValueError(
            f"color_scale must be one of {_VALID_SCALES!r}, got {color_scale!r}"
        )

    # 2. Validate max_points
    if max_points is not None:
        if (
            isinstance(max_points, (bool, np.bool_))
            or not isinstance(max_points, numbers.Integral)
            or int(max_points) <= 0
        ):
            raise ValueError(
                f"max_points must be a positive integer or None, got {max_points!r}"
            )
        max_points = int(max_points)

    # 3. Validate coord_unit
    _, length_scale, unit_str = _length_unit(coord_unit)

    # 4. Validate ax
    is_3d = view == "3d"
    if ax is not None:
        _validate_plot_map_axes(ax, is_3d)

    # 5. Extract diagnostic data (this handles all physics & numerical safety)
    map_data = _extract_map_data(
        setup,
        quantity,
        target_id=target_id,
        frame=frame,
        direction=direction,
    )

    # 6. Check color_scale compatibility
    norm, chosen_cmap = _setup_color_norm_and_cmap(
        map_data.values,
        quantity,
        color_scale,
    )

    # 7. Thinning for display
    n_total = len(map_data.values)
    indices = _display_indices(n_total, max_points)
    n_shown = len(indices)

    # Coordinates scaled to requested length unit
    coords = map_data.positions_m[indices] * length_scale
    vals_shown = map_data.values[indices]

    # 8. Create or use Figure and Axes
    if ax is None:
        if is_3d:
            fig = plt.figure(figsize=(7.5, 6.0))
            ax = fig.add_subplot(projection="3d")
        else:
            fig, ax = plt.subplots(figsize=(7.0, 5.5))
    else:
        fig = ax.figure

    # 9. Draw scatter
    if is_3d:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=vals_shown,
            cmap=chosen_cmap,
            norm=norm,
            s=12,
            edgecolors="none",
        )
        ax.set_xlabel(f"x [{unit_str}]")
        ax.set_ylabel(f"y [{unit_str}]")
        ax.set_zlabel(f"z [{unit_str}]")
    else:
        h_idx, v_idx, h_name, v_name = _VIEWS_2D[view]
        scatter = ax.scatter(
            coords[:, h_idx],
            coords[:, v_idx],
            c=vals_shown,
            cmap=chosen_cmap,
            norm=norm,
            s=12,
            edgecolors="none",
        )
        ax.set_xlabel(f"{h_name} [{unit_str}]")
        ax.set_ylabel(f"{v_name} [{unit_str}]")
        ax.set_aspect("equal", adjustable="datalim")

    # 10. Title
    if n_shown < n_total:
        points_info = f"{n_shown:,} / {n_total:,} points"
    else:
        points_info = f"{n_total:,} points"
    ax.set_title(f"{quantity} (Target: {map_data.target_id}; {points_info})")

    # 11. Colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.08, shrink=0.85)
    cbar.set_label(f"{quantity} [{map_data.unit_str}]")

    return fig


def _validate_plot_map_axes(ax: Axes, is_3d: bool) -> None:
    from matplotlib.axes import Axes as _Axes

    if not isinstance(ax, _Axes):
        raise ValueError("plot_map requires ax to be a Matplotlib Axes instance")

    axes_is_3d = getattr(ax, "name", None) == "3d"
    if is_3d != axes_is_3d:
        expected = "3D" if is_3d else "2D"
        raise ValueError(f"plot_map view requires a matching {expected} axes")

    if not is_3d and getattr(ax, "name", None) != "rectilinear":
        raise ValueError("plot_map 2D view requires Cartesian rectilinear axes")

    scales = (ax.get_xscale(), ax.get_yscale())
    if is_3d:
        scales = (*scales, ax.get_zscale())
    if any(scale != "linear" for scale in scales):
        raise ValueError("plot_map requires linear axes scales")

    # Require empty axes
    has_user_artists = any(
        (
            ax.has_data(),
            len(ax.lines),
            len(ax.collections),
            len(ax.images),
            len(ax.patches),
            len(ax.texts),
            len(ax.tables),
            len(ax.artists),
            bool(ax.get_legend()),
            bool(ax.get_title()),
            bool(ax.get_title(loc="left")),
            bool(ax.get_title(loc="right")),
            bool(ax.get_xlabel()),
            bool(ax.get_ylabel()),
            bool(getattr(ax, "get_zlabel", lambda: "")()),
        )
    )
    if has_user_artists:
        raise ValueError("plot_map requires empty axes; pass a new Axes or omit ax")


def _setup_color_norm_and_cmap(
    values: np.ndarray,
    quantity: str,
    color_scale: str,
) -> tuple[object, str]:
    """Determine Normalize instance and colormap name for values."""
    from matplotlib import colors

    is_signed = quantity in _SIGNED_QUANTITIES
    val_min = float(np.min(values))
    val_max = float(np.max(values))

    if color_scale == "log":
        if is_signed:
            raise ValueError(
                f"color_scale='log' cannot be used with signed quantity {quantity!r}; "
                "use 'linear'"
            )
        if val_min <= 0.0:
            raise ValueError(
                f"color_scale='log' requires strictly positive values, "
                f"but minimum value is {val_min:.3e}"
            )
        chosen_cmap = "viridis"
        norm = colors.LogNorm(vmin=val_min, vmax=val_max)
        return norm, chosen_cmap

    # color_scale == "linear"
    if is_signed:
        chosen_cmap = "coolwarm"
        abs_max = max(abs(val_min), abs(val_max))
        if abs_max == 0.0:
            norm = colors.Normalize(vmin=-1.0, vmax=1.0)
        else:
            norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-abs_max, vmax=abs_max)
        return norm, chosen_cmap

    # Non-negative quantity with linear scale
    chosen_cmap = "viridis"
    if val_min == val_max:
        if val_min == 0.0:
            norm = colors.Normalize(vmin=0.0, vmax=1.0)
        else:
            with np.errstate(over="ignore"):
                offset = val_min * 0.1
                vmin = max(0.0, val_min - offset)
                vmax = val_min + offset

            # Check if offset underflowed to 0 (vmin == vmax)
            if vmin == vmax:
                vmin = max(0.0, float(np.nextafter(val_min, -np.inf)))
                vmax = float(np.nextafter(val_min, np.inf))
                if not np.isfinite(vmax):
                    vmax = val_min
            elif not np.isfinite(vmax):
                vmax = float(np.finfo(np.float64).max)
                if vmin == vmax:
                    vmin = float(np.nextafter(val_min, -np.inf))
                else:
                    vmin = min(vmin, val_min * 0.9)

            # Guarantee vmin < vmax and 0 <= vmin <= val <= vmax < inf
            if vmin == vmax:
                if vmin > 0.0:
                    vmin = float(np.nextafter(vmin, -np.inf))
                else:
                    vmax = float(np.nextafter(vmax, np.inf))

            norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=val_min, vmax=val_max)
    return norm, chosen_cmap
