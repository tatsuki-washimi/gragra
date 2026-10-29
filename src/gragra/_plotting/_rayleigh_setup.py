"""Plot a resolved Rayleigh half-space setup without evaluating its fields."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gragra._plotting._appearance import appearance_fields

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from gragra._rayleigh_contract import (
        RayleighResolvedEvaluation,
        RayleighResponseEvaluation,
    )
    from gragra._rayleigh_evaluation_setup import RayleighEvaluationSetup
    from gragra._rayleigh_response_setup import RayleighResponseSetup
    from gragra.geometry import Appearance


_PROJECTIONS = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}


def _frame_label(frame: object) -> str:
    names = []
    current = frame
    while current is not None:
        names.append(current.name)
        current = current.parent
    return " → ".join(reversed(names))


def _length_unit(unit: str) -> tuple[float, str]:
    from astropy import units as u

    try:
        parsed = u.Unit(unit)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unit must be a valid length unit, got {unit!r}") from error
    if not parsed.is_equivalent(u.m):
        raise ValueError(f"unit must be compatible with a length unit, got {unit!r}")
    factor = float((1.0 * u.m).to_value(parsed))
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(f"unit must have a finite positive length scale, got {unit!r}")
    return factor, parsed.to_string()


def _require_empty_axes(ax: Axes) -> None:
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
            len(ax.containers),
            ax.get_legend() is not None,
            bool(ax.get_title()),
            bool(ax.get_title(loc="left")),
            bool(ax.get_title(loc="right")),
            bool(ax.get_xlabel()),
            bool(ax.get_ylabel()),
            bool(getattr(ax, "get_zlabel", lambda: "")()),
        )
    )
    if has_user_artists:
        raise ValueError("provided ax must be empty before plotting the setup")


def _window(
    target_surface_m: np.ndarray, phase_origin_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    support = np.vstack((target_surface_m[:, :2], phase_origin_m[None, :2]))
    low = np.min(support, axis=0)
    high = np.max(support, axis=0)
    span = float(np.max(high - low))
    padding = max(0.2 * span, 1.0)
    low -= padding
    high += padding
    corners = np.array(
        [
            [low[0], low[1], 0.0],
            [high[0], low[1], 0.0],
            [high[0], high[1], 0.0],
            [low[0], high[1], 0.0],
        ],
        dtype=np.float64,
    )
    depth = max(
        1.0,
        0.35 * float(np.max(high - low)),
        1.15 * max(0.0, -float(np.min(target_surface_m[:, 2]))),
    )
    return corners, np.column_stack((corners[:, :2], np.full(4, -depth))), depth


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Return the 2D convex hull in counterclockwise order."""
    ordered = sorted({(float(x), float(y)) for x, y in points})
    if len(ordered) <= 2:
        return np.asarray(ordered, dtype=np.float64)

    def cross(
        origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
    ) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (
            b[0] - origin[0]
        )

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _set_limits_3d(ax: Axes, points: np.ndarray) -> None:
    low, high = np.min(points, axis=0), np.max(points, axis=0)
    center = (low + high) / 2
    half_span = max(float(np.max(high - low)) / 2, 0.5)
    radius = half_span * 1.06
    ax.set_xlim3d(center[0] - radius, center[0] + radius)
    ax.set_ylim3d(center[1] - radius, center[1] + radius)
    ax.set_zlim3d(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _set_limits_2d(ax: Axes, points: np.ndarray, indices: Sequence[int]) -> None:
    selected = points[:, indices]
    low, high = np.min(selected, axis=0), np.max(selected, axis=0)
    spans = high - low
    padding = np.maximum(spans * 0.06, 0.5)
    ax.set_xlim(low[0] - padding[0], high[0] + padding[0])
    ax.set_ylim(low[1] - padding[1], high[1] + padding[1])
    ax.set_aspect("equal", adjustable="box")


def _draw_arrow_3d(
    ax: Axes,
    start: np.ndarray,
    vector: np.ndarray,
    length: float,
    *,
    color: str,
    label: str,
    gid: str,
) -> object:
    arrow = ax.quiver(
        *start,
        *vector,
        length=length,
        normalize=True,
        arrow_length_ratio=0.14,
        color=color,
        linewidth=1.8,
        label=label,
    )
    arrow.set_gid(gid)
    return arrow


def _draw_arrow_2d(
    ax: Axes,
    start: np.ndarray,
    vector: np.ndarray,
    indices: Sequence[int],
    length: float,
    *,
    color: str,
    label: str,
    gid: str,
) -> object | None:
    projected = vector[list(indices)]
    magnitude = float(np.linalg.norm(projected))
    if magnitude <= 1e-12:
        return None
    delta = projected * length
    arrow = ax.quiver(
        start[indices[0]],
        start[indices[1]],
        delta[0],
        delta[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        pivot="tail",
        color=color,
        width=0.006,
        label=label,
        zorder=5,
    )
    arrow.set_gid(gid)
    return arrow


def plot_rayleigh_setup(
    resolved: (
        RayleighResolvedEvaluation
        | RayleighResponseEvaluation
        | RayleighEvaluationSetup
        | RayleighResponseSetup
    ),
    *,
    view: str = "3d",
    unit: str = "m",
    ax: Axes | None = None,
    appearance: Appearance | None = None,
) -> Figure:
    """Draw fixed half-space, targets, propagation direction, and phase origin.

    The surface and medium are shown in a finite window for display. The
    physical geometry remains an infinite half-space with its medium at local
    ``z < 0``. No field is evaluated and no resolved coordinate is changed.
    ``appearance`` overrides half-space metadata field by field on the
    medium and free-surface artists. Its label applies to the medium; the
    free-surface artist keeps its descriptive label.
    """
    from gragra._rayleigh_contract import (
        RayleighResolvedEvaluation as _Resolved,
    )
    from gragra._rayleigh_contract import (
        RayleighResponseEvaluation as _Response,
    )
    from gragra._rayleigh_evaluation_setup import RayleighEvaluationSetup
    from gragra._rayleigh_response_setup import RayleighResponseSetup

    if isinstance(resolved, (RayleighEvaluationSetup, RayleighResponseSetup)):
        resolved = resolved.resolved

    if not isinstance(resolved, (_Resolved, _Response)):
        raise TypeError(
            "resolved must be a RayleighResolvedEvaluation or "
            "RayleighResponseEvaluation"
        )
    if view not in {"3d", *_PROJECTIONS}:
        raise ValueError("view must be '3d', 'xy', 'xz', or 'yz'")
    unit_factor, unit_label = _length_unit(unit)

    is_3d = view == "3d"
    if ax is not None:
        from matplotlib.axes import Axes as _Axes

        if not isinstance(ax, _Axes):
            raise ValueError("ax must be a Matplotlib Axes instance")
        axes_is_3d = getattr(ax, "name", None) == "3d"
        if axes_is_3d != is_3d:
            expected = "3D" if is_3d else "2D"
            raise ValueError(f"view={view!r} requires a matching {expected} axes")
        if not is_3d and getattr(ax, "name", None) != "rectilinear":
            raise ValueError("2D projections require Cartesian rectilinear axes")
        scales = (ax.get_xscale(), ax.get_yscale())
        if is_3d:
            scales = (*scales, ax.get_zscale())
        if any(scale != "linear" for scale in scales):
            raise ValueError("plot_rayleigh_setup requires linear axes scales")
        _require_empty_axes(ax)

    spec = resolved.model_spec
    medium_label_default = "Elastic medium (local z < 0)"
    medium_color, medium_alpha, medium_label, _ = appearance_fields(
        appearance,
        spec.geometry,
        color="#d6e3ec",
        alpha=0.34 if is_3d else 0.4,
        label=medium_label_default,
    )
    surface_color, surface_alpha, _, _ = appearance_fields(
        appearance,
        spec.geometry,
        color="#7fb5d7",
        alpha=0.48 if is_3d else 0.35,
        label="Free surface (local z = 0)",
    )
    surface_frame = resolved.computation_frame
    phase_origin_m = spec.phase_origin_m
    target_surface_m = resolved.target_positions_m
    top_local, bottom_local, _ = _window(target_surface_m, phase_origin_m)
    top_display = surface_frame.transform_points_m(top_local, resolved.output_frame)
    bottom_display = surface_frame.transform_points_m(
        bottom_local, resolved.output_frame
    )
    target_display = surface_frame.transform_points_m(
        target_surface_m, resolved.output_frame
    )
    phase_display = surface_frame.transform_points_m(
        phase_origin_m, resolved.output_frame
    )
    direction_display = surface_frame.transform_vectors_m(
        spec.direction, resolved.output_frame
    )
    normal_display = surface_frame.transform_vectors_m(
        np.array([0.0, 0.0, 1.0]), resolved.output_frame
    )
    direction_length = 0.18 * float(np.max(np.ptp(top_local[:, :2], axis=0)))
    normal_length = 0.22 * float(np.max(np.ptp(top_local[:, :2], axis=0)))

    top_display *= unit_factor
    bottom_display *= unit_factor
    target_display *= unit_factor
    phase_display *= unit_factor
    direction_length *= unit_factor
    normal_length *= unit_factor
    direction_end = phase_display + direction_display * direction_length
    normal_end = phase_display + normal_display * normal_length
    limits_points = np.vstack(
        (
            top_display,
            bottom_display,
            target_display,
            phase_display,
            direction_end,
            normal_end,
        )
    )
    window_label = "Window of infinite half-space (finite display extent)"

    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        figure = plt.figure(figsize=(12.0, 7.5))
        axes = figure.add_subplot(projection="3d" if is_3d else None)
        figure.subplots_adjust(left=0.08, right=0.64, bottom=0.11, top=0.81)
    else:
        axes = ax
        figure = axes.figure

    surface_label = "Free surface (local z = 0)"
    direction_label = (
        "Horizontal propagation direction\n(arrow length is not amplitude)"
    )
    if is_3d:
        faces = [
            top_display,
            bottom_display,
            np.array(
                [top_display[0], top_display[1], bottom_display[1], bottom_display[0]]
            ),
            np.array(
                [top_display[1], top_display[2], bottom_display[2], bottom_display[1]]
            ),
            np.array(
                [top_display[2], top_display[3], bottom_display[3], bottom_display[2]]
            ),
            np.array(
                [top_display[3], top_display[0], bottom_display[0], bottom_display[3]]
            ),
        ]
        medium = Poly3DCollection(
            faces[1:],
            facecolor=medium_color,
            edgecolor="#8095a5",
            alpha=medium_alpha,
        )
        medium.set_gid("elastic-medium")
        medium.set_label(medium_label)
        axes.add_collection3d(medium)
        surface = Poly3DCollection(
            [faces[0]],
            facecolor=surface_color,
            edgecolor="#255e7c",
            alpha=surface_alpha,
        )
        surface.set_gid("free-surface")
        surface.set_label(surface_label)
        axes.add_collection3d(surface)
        for index in range(4):
            (line,) = axes.plot(
                *np.vstack((top_display[index], top_display[(index + 1) % 4])).T,
                color="#244f68",
                linewidth=1.2,
                label="_nolegend_",
            )
            line.set_gid(f"surface-window-edge-{index}")
            (vertical,) = axes.plot(
                *np.vstack((top_display[index], bottom_display[index])).T,
                color="#8095a5",
                linewidth=0.8,
                alpha=0.75,
                label="_nolegend_",
            )
            vertical.set_gid(f"medium-window-edge-{index}")
        axes.scatter(
            *target_display.T,
            s=48,
            color="#d36d31",
            edgecolors="#292929",
            linewidths=0.7,
            depthshade=False,
            label=f"Targets ({len(target_display)})",
            gid="target-points",
        )
        axes.scatter(
            *phase_display,
            s=58,
            marker="x",
            color="#7a3e91",
            linewidths=2,
            label="Phase origin",
            gid="phase-origin",
        )
        _draw_arrow_3d(
            axes,
            phase_display,
            direction_display,
            direction_length,
            color="#d24c35",
            label=direction_label,
            gid="propagation-direction",
        )
        _draw_arrow_3d(
            axes,
            phase_display,
            normal_display,
            normal_length,
            color="#276f9e",
            label="+z surface normal",
            gid="surface-normal",
        )
        for position, target_id in zip(
            target_display, resolved.targets.ids, strict=True
        ):
            axes.text(
                *position,
                target_id,
                fontsize=8.5,
                fontweight="semibold",
                color="#5a3212",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.7,
                    "pad": 1.2,
                },
                zorder=10,
            )
        _set_limits_3d(axes, limits_points)
        axes.set_proj_type("ortho")
        axes.set_xlabel(f"x [{unit_label}]")
        axes.set_ylabel(f"y [{unit_label}]")
        axes.set_zlabel(f"z [{unit_label}]")
        view_label = "3D orthographic view"
    else:
        indices = _PROJECTIONS[view]
        projected_volume = np.vstack((top_display, bottom_display))[:, indices]
        medium_polygon = Polygon(
            _convex_hull(projected_volume),
            closed=True,
            facecolor=medium_color,
            edgecolor="#8095a5",
            linewidth=0.8,
            alpha=medium_alpha,
            label=medium_label,
            zorder=1,
        )
        medium_polygon.set_gid("elastic-medium")
        axes.add_patch(medium_polygon)
        surface_polygon = Polygon(
            top_display[:, indices],
            closed=True,
            facecolor=surface_color,
            edgecolor="#255e7c",
            linewidth=1.4,
            alpha=surface_alpha,
            label=surface_label,
            zorder=2,
        )
        surface_polygon.set_gid("free-surface")
        axes.add_patch(surface_polygon)
        for index in range(4):
            (line,) = axes.plot(
                *top_display[[index, (index + 1) % 4]][:, indices].T,
                color="#244f68",
                linewidth=1.2,
                label="_nolegend_",
                zorder=3,
            )
            line.set_gid(f"surface-window-edge-{index}")
        axes.scatter(
            *target_display[:, indices].T,
            s=48,
            color="#d36d31",
            edgecolors="#292929",
            linewidths=0.7,
            label=f"Targets ({len(target_display)})",
            gid="target-points",
            zorder=4,
        )
        axes.scatter(
            *phase_display[list(indices)],
            s=58,
            marker="x",
            color="#7a3e91",
            linewidths=2,
            label="Phase origin",
            gid="phase-origin",
            zorder=5,
        )
        _draw_arrow_2d(
            axes,
            phase_display,
            direction_display,
            indices,
            direction_length,
            color="#d24c35",
            label=direction_label,
            gid="propagation-direction",
        )
        _draw_arrow_2d(
            axes,
            phase_display,
            normal_display,
            indices,
            normal_length,
            color="#276f9e",
            label="+z surface normal",
            gid="surface-normal",
        )
        for position, target_id in zip(
            target_display, resolved.targets.ids, strict=True
        ):
            axes.annotate(
                target_id,
                xy=position[list(indices)],
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=9,
                color="#5a3212",
                zorder=6,
            )
        _set_limits_2d(axes, limits_points, indices)
        axes.set_xlabel(f"{view[0]} [{unit_label}]")
        axes.set_ylabel(f"{view[1]} [{unit_label}]")
        view_label = f"{view.upper()} projection"

    axes.set_title(
        "Rayleigh half-space setup — "
        f"{view_label}\n"
        f"Frame: {_frame_label(resolved.output_frame)} · Unit: {unit_label}\n"
        f"{window_label}"
    )
    if ax is None:
        axes.legend(
            loc="center left",
            bbox_to_anchor=(1.1, 0.5),
            fontsize=8,
            framealpha=0.9,
            borderaxespad=0,
        )
    else:
        axes.legend(loc="upper left", fontsize=8, framealpha=0.9)
    return figure
