"""Plot the resolved geometry, targets, and actual discrete source positions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gragra._plotting._appearance import appearance_fields

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from gragra._evaluation_setup import ResolvedEvaluation
    from gragra.evaluation import EvaluationSetup
    from gragra.geometry import Appearance


_PROJECTIONS = {
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
}
_AXIS_COLORS = ("#c43c39", "#27864b", "#316aa5")


def _frame_label(frame: object) -> str:
    frames = []
    current = frame
    while current is not None:
        frames.append(current.name)
        current = current.parent
    return " → ".join(reversed(frames))


def _length_unit(unit: str) -> tuple[object, float, str]:
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
    return parsed, factor, parsed.to_string()


def _display_indices(count: int, maximum: int | None) -> np.ndarray:
    if maximum is None or count <= maximum:
        return np.arange(count, dtype=np.intp)
    return np.linspace(0, count - 1, num=maximum, dtype=np.intp)


def _source_label(has_source: bool, shown: int, total: int) -> str:
    if has_source:
        return f"Actual sources ({shown} / {total} shown)"
    return "Analytic model (no discrete sources)"


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


def _bounds_points(
    sphere_center: np.ndarray,
    radius: float,
    targets: np.ndarray,
    sources: np.ndarray,
) -> np.ndarray:
    support = [sphere_center - radius, sphere_center + radius]
    if len(targets):
        support.append(targets)
    if len(sources):
        support.append(sources)
    points = np.concatenate(
        [np.asarray(item, dtype=np.float64).reshape(-1, 3) for item in support],
        axis=0,
    )
    if not np.isfinite(points).all():
        raise ValueError("coordinates must remain finite in the output frame")
    return points


def _set_limits_3d(ax: Axes, points: np.ndarray) -> None:
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    center = (low + high) / 2
    half_span = float(np.max(high - low)) / 2
    padding = max(half_span * 0.06, np.finfo(np.float64).eps)
    radius = half_span + padding
    ax.set_xlim3d(center[0] - radius, center[0] + radius)
    ax.set_ylim3d(center[1] - radius, center[1] + radius)
    ax.set_zlim3d(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _set_limits_2d(ax: Axes, points: np.ndarray, indices: Sequence[int]) -> None:
    low = np.min(points[:, indices], axis=0)
    high = np.max(points[:, indices], axis=0)
    spans = high - low
    padding = np.maximum(spans * 0.06, np.finfo(np.float64).eps)
    ax.set_xlim(low[0] - padding[0], high[0] + padding[0])
    ax.set_ylim(low[1] - padding[1], high[1] + padding[1])
    ax.set_aspect("equal", adjustable="box")


def _draw_orientation_3d(
    ax: Axes,
    center: np.ndarray,
    basis_in_output: np.ndarray,
    radius: float,
) -> None:
    length = radius * 0.55
    for direction, color in zip(basis_in_output, _AXIS_COLORS, strict=True):
        ax.quiver(
            *center,
            *direction,
            length=length,
            normalize=True,
            arrow_length_ratio=0.18,
            color=color,
            linewidth=1.5,
        )


def _draw_orientation_2d(
    ax: Axes,
    center: np.ndarray,
    basis_in_output: np.ndarray,
    radius: float,
    indices: Sequence[int],
) -> None:
    length = radius * 0.55
    for index, (direction, color) in enumerate(
        zip(basis_in_output, _AXIS_COLORS, strict=True)
    ):
        projected = direction[list(indices)]
        magnitude = float(np.linalg.norm(projected))
        if magnitude <= 1e-12:
            continue
        start = center[list(indices)]
        delta = projected * length
        arrow = ax.quiver(
            start[0],
            start[1],
            delta[0],
            delta[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            pivot="tail",
            color=color,
            width=0.006,
            zorder=5,
        )
        arrow.set_gid(f"model-frame-axis-{('x', 'y', 'z')[index]}")
        end = start + delta
        ax.annotate(
            f"model +{('x', 'y', 'z')[index]}",
            xy=end,
            xytext=(5, 5),
            textcoords="offset points",
            color=color,
            fontsize=8,
            zorder=5,
        )


def plot_setup(
    evaluation: ResolvedEvaluation | EvaluationSetup,
    *,
    view: str = "3d",
    unit: str = "m",
    ax: Axes | None = None,
    max_sources: int | None = 5000,
    appearance: Appearance | None = None,
) -> Figure:
    """Draw a resolved uniform-sphere setup without running or changing it.

    Coordinates are shown in ``evaluation.output_frame``. ``view`` selects an
    orthographic 3D view or a labeled Cartesian projection. A discrete source
    is displayed from its already-resolved positions. A limited stable,
    evenly spaced selection includes both endpoints when at least two points
    are shown; a one-point selection uses the first source. Axis extents still
    account for every source, target, and the full ideal sphere. A supplied
    axes must be empty, Cartesian, and use linear scales. ``appearance``
    overrides geometry metadata field by field for the sphere artist only.
    """
    from gragra._evaluation_setup import ResolvedEvaluation as _ResolvedEvaluation
    from gragra.evaluation import EvaluationSetup

    if isinstance(evaluation, EvaluationSetup):
        evaluation = evaluation.resolved

    if not isinstance(evaluation, _ResolvedEvaluation):
        raise TypeError("evaluation must be a ResolvedEvaluation")
    if view not in {"3d", *_PROJECTIONS}:
        raise ValueError("view must be '3d', 'xy', 'xz', or 'yz'")
    if max_sources is not None and (
        isinstance(max_sources, (bool, np.bool_))
        or not isinstance(max_sources, (int, np.integer))
        or max_sources <= 0
    ):
        raise ValueError("max_sources must be a positive integer or None")

    parsed_unit, unit_factor, unit_label = _length_unit(unit)
    del parsed_unit  # The scale and canonical label are the only needed values.

    is_3d = view == "3d"
    if ax is not None:
        from matplotlib.axes import Axes as _Axes

        if not isinstance(ax, _Axes):
            raise ValueError("ax must be a Matplotlib Axes instance")
        axes_is_3d = getattr(ax, "name", None) == "3d"
        if is_3d != axes_is_3d:
            expected = "3D" if is_3d else "2D"
            raise ValueError(f"view={view!r} requires a matching {expected} axes")
        if not is_3d and getattr(ax, "name", None) != "rectilinear":
            raise ValueError("2D projections require Cartesian rectilinear axes")
        scales = (ax.get_xscale(), ax.get_yscale())
        if is_3d:
            scales = (*scales, ax.get_zscale())
        if any(scale != "linear" for scale in scales):
            raise ValueError("plot_setup requires linear axes scales")
        _require_empty_axes(ax)

    try:
        geometry = evaluation.model_spec.geometry
        output_frame = evaluation.output_frame
        sphere_center_m = evaluation.computation_frame.transform_points_m(
            geometry.center_m, output_frame
        )
        targets_m = evaluation.targets.frame.transform_points_m(
            evaluation.targets.positions_m, output_frame
        )
        source = evaluation.source
        if source is None:
            sources_m = np.empty((0, 3), dtype=np.float64)
        else:
            sources_m = source.frame.transform_points_m(
                source.positions_m, output_frame
            )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "plot_setup could not interpret all setup coordinates in "
            f"output frame {_frame_label(evaluation.output_frame)!r}: {error}"
        ) from error

    sphere_center = sphere_center_m * unit_factor
    target_positions = targets_m * unit_factor
    source_positions = sources_m * unit_factor
    radius = geometry.radius_m * unit_factor
    points_for_limits = _bounds_points(
        sphere_center, radius, target_positions, source_positions
    )
    target_indices = _PROJECTIONS.get(view, (0, 1, 2))
    sphere_color, sphere_alpha, sphere_label, sphere_color_set = appearance_fields(
        appearance,
        geometry,
        color="#70a9d2",
        alpha=0.28 if is_3d else 0.3,
        label="Ideal sphere" if is_3d else "Ideal sphere projection",
    )
    basis_in_output = evaluation.computation_frame.transform_vectors_m(
        np.eye(3), output_frame
    )
    selected_indices = _display_indices(len(source_positions), max_sources)
    selected_sources = source_positions[selected_indices]

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    if ax is None:
        figure = plt.figure()
        axes = figure.add_subplot(projection="3d" if is_3d else None)
    else:
        axes = ax
        figure = axes.figure

    if is_3d:
        longitude = np.linspace(0.0, 2.0 * np.pi, 48)
        latitude = np.linspace(0.0, np.pi, 25)
        xx = sphere_center[0] + radius * np.outer(np.cos(longitude), np.sin(latitude))
        yy = sphere_center[1] + radius * np.outer(np.sin(longitude), np.sin(latitude))
        zz = sphere_center[2] + radius * np.outer(
            np.ones_like(longitude), np.cos(latitude)
        )
        surface = axes.plot_surface(
            xx,
            yy,
            zz,
            color=sphere_color,
            alpha=sphere_alpha,
            linewidth=0,
            antialiased=True,
            shade=not sphere_color_set,
        )
        surface.set_label(sphere_label)
        axes.scatter(
            *selected_sources.T if len(selected_sources) else ([], [], []),
            s=9,
            color="#414b55",
            alpha=0.7,
            depthshade=False,
            label=_source_label(
                source is not None, len(selected_sources), len(source_positions)
            ),
            gid="source-points",
        )
        axes.scatter(
            *target_positions.T if len(target_positions) else ([], [], []),
            s=44,
            color="#e28a37",
            edgecolors="#292929",
            linewidths=0.7,
            depthshade=False,
            label=f"Targets ({len(target_positions)})",
            gid="target-points",
        )
        label_offset = radius * 0.025
        for position, target_id in zip(
            target_positions, evaluation.targets.ids, strict=True
        ):
            axes.text(
                position[0] + label_offset,
                position[1] + label_offset,
                position[2],
                target_id,
                fontsize=9,
                color="#5a3212",
            )
        _draw_orientation_3d(axes, sphere_center, basis_in_output, radius)
        _set_limits_3d(axes, points_for_limits)
        axes.set_proj_type("ortho")
        axes.set_xlabel(f"x [{unit_label}]")
        axes.set_ylabel(f"y [{unit_label}]")
        axes.set_zlabel(f"z [{unit_label}]")
        view_label = "3D orthographic view"
    else:
        first, second = target_indices
        axes.add_patch(
            Circle(
                sphere_center[[first, second]],
                radius,
                facecolor=sphere_color,
                edgecolor="#316b96",
                alpha=sphere_alpha,
                linewidth=1.2,
                label=sphere_label,
                zorder=1,
            )
        )
        axes.scatter(
            *selected_sources[:, [first, second]].T
            if len(selected_sources)
            else ([], []),
            s=9,
            color="#414b55",
            alpha=0.7,
            label=_source_label(
                source is not None, len(selected_sources), len(source_positions)
            ),
            gid="source-points",
            zorder=2,
        )
        axes.scatter(
            *target_positions[:, [first, second]].T
            if len(target_positions)
            else ([], []),
            s=44,
            color="#e28a37",
            edgecolors="#292929",
            linewidths=0.7,
            label=f"Targets ({len(target_positions)})",
            gid="target-points",
            zorder=3,
        )
        for position, target_id in zip(
            target_positions, evaluation.targets.ids, strict=True
        ):
            axes.annotate(
                target_id,
                xy=position[[first, second]],
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=9,
                color="#5a3212",
            )
        _draw_orientation_2d(
            axes, sphere_center, basis_in_output, radius, target_indices
        )
        _set_limits_2d(axes, points_for_limits, target_indices)
        axes.set_xlabel(f"{view[0]} [{unit_label}]")
        axes.set_ylabel(f"{view[1]} [{unit_label}]")
        view_label = f"{view.upper()} projection"

    axes.set_title(
        f"Uniform sphere setup — {view_label}\n"
        f"Frame: {_frame_label(output_frame)} · R = {radius:g} {unit_label}"
    )
    if is_3d:
        handles, _ = axes.get_legend_handles_labels()
        handles.append(
            Line2D(
                [],
                [],
                color="#555555",
                marker=">",
                linestyle="-",
                label="Model-frame axes: x red · y green · z blue",
            )
        )
        axes.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
    else:
        axes.legend(loc="upper left", framealpha=0.9)
    if ax is None:
        figure.tight_layout()
    return figure
