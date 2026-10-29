"""Internal field plotting implementation for DensityPerturbationGrid and
DisplacementGrid."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from gragra._plotting._setup import _length_unit, _require_empty_axes
from gragra.fields.regular_grid import DensityPerturbationGrid, DisplacementGrid
from gragra.geometry._frame import Frame

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


_PROJECTIONS = {
    "xy": (0, 1, 2),  # (horiz_axis, vert_axis, normal_axis)
    "xz": (0, 2, 1),
    "yz": (1, 2, 0),
}
_AXIS_NAMES = ("x", "y", "z")


def _extract_part(data: np.ndarray, part: str | None) -> tuple[np.ndarray, str]:
    """Extract real component or scalar representation from real or complex data."""
    is_complex = np.iscomplexobj(data)
    if not is_complex:
        if part is not None:
            raise ValueError(
                f"part={part!r} was specified, but input field data is real. "
                "part is only valid and required for complex field data."
            )
        return np.asarray(data, dtype=np.float64), ""

    # Complex data requires part
    if part is None:
        raise ValueError(
            "Input field data is complex, so part must be explicitly specified "
            "as one of {'real', 'imag', 'abs', 'phase'}."
        )

    valid_parts = {"real", "imag", "abs", "phase"}
    if part not in valid_parts:
        raise ValueError(f"part must be one of {valid_parts}, got {part!r}")

    if part == "real":
        return np.real(data).astype(np.float64), " (Re)"
    if part == "imag":
        return np.imag(data).astype(np.float64), " (Im)"
    if part == "abs":
        return np.abs(data).astype(np.float64), " (|z|)"
    if part == "phase":
        # Arg(z) in (-pi, pi], normalized: zero amplitude is masked
        amp = np.abs(data)
        zero_mask = amp == 0.0
        # np.angle returns in [-pi, pi]. Under IEEE 754 branch cut,
        # -1.0 - 0.0j gives -pi.
        # Strictly normalize exact -pi to +pi without disturbing nearby
        # non-boundary values:
        angle = np.angle(data)
        angle = np.where(angle == -np.pi, np.pi, angle)
        phase_masked = np.ma.masked_where(zero_mask, angle)
        return phase_masked, " (Phase)"

    raise ValueError(f"Unknown part: {part!r}")


def _validate_axes(ax: Axes, is_3d: bool) -> None:
    """Validate that user-supplied Axes is empty, Cartesian rectilinear, and
    uses linear scale."""
    from matplotlib.axes import Axes as _Axes

    if not isinstance(ax, _Axes):
        raise ValueError("ax must be a Matplotlib Axes instance")

    axes_is_3d = getattr(ax, "name", None) == "3d"
    if is_3d != axes_is_3d:
        expected = "3D" if is_3d else "2D"
        raise ValueError(f"plot_field requires a matching {expected} axes")

    if not is_3d and getattr(ax, "name", None) != "rectilinear":
        raise ValueError("2D slices require Cartesian rectilinear axes")

    scales = (ax.get_xscale(), ax.get_yscale())
    if is_3d:
        scales = (*scales, ax.get_zscale())
    if any(scale != "linear" for scale in scales):
        raise ValueError("plot_field requires linear axes scales")

    _require_empty_axes(ax)


def _set_limits_2d(
    ax: Axes, x_min: float, x_max: float, y_min: float, y_max: float
) -> None:
    span_x = x_max - x_min
    span_y = y_max - y_min
    pad_x = max(span_x * 0.06, 1e-9)
    pad_y = max(span_y * 0.06, 1e-9)
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_aspect("equal", adjustable="box")


def _set_limits_3d(ax: Axes, points: np.ndarray) -> None:
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    center = (low + high) / 2.0
    half_span = float(np.max(high - low)) / 2.0
    padding = max(half_span * 0.06, 1e-9)
    radius = half_span + padding
    ax.set_xlim3d(center[0] - radius, center[0] + radius)
    ax.set_ylim3d(center[1] - radius, center[1] + radius)
    ax.set_zlim3d(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def plot_field(
    field: DensityPerturbationGrid | DisplacementGrid,
    *,
    frame: Frame,
    plane: str | None = None,
    index: int | None = None,
    part: str | None = None,
    component: str | None = None,
    unit: str = "m",
    max_points: int | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot density perturbation or displacement field data faithful to native
    grid coordinates.

    Parameters
    ----------
    field : DensityPerturbationGrid or DisplacementGrid
        Input field helper object.
    frame : Frame
        Frame in which the field coordinates are defined.
    plane : {'xy', 'xz', 'yz'}, optional
        2D plane slice along native grid axes. Must be specified together with
        ``index``.
        If omitted (None), creates a 3D plot.
    index : int, optional
        Integer grid index along the axis orthogonal to ``plane``.
        Must be specified together with ``plane``.
    part : {'real', 'imag', 'abs', 'phase'}, optional
        Complex decomposition. Required if field values are complex.
        Disallowed if field values are real.
    component : {'x', 'y', 'z', 'norm'}, optional
        Vector component for DisplacementGrid. If specified, plots a scalar field
        of that component. If None for DisplacementGrid, plots displacement
        vectors (quiver).
        Disallowed for DensityPerturbationGrid.
    unit : str, default "m"
        Coordinate axis display unit (must be a valid length unit).
    max_points : int, optional
        Deterministic downsampling limit for displayed points/arrows.
        If None, all points are displayed.
    ax : matplotlib.axes.Axes, optional
        Existing Axes to draw into. Must be empty, Cartesian rectilinear, with
        linear scales.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    if not isinstance(field, (DensityPerturbationGrid, DisplacementGrid)):
        raise TypeError(
            "field must be DensityPerturbationGrid or DisplacementGrid, got "
            f"{type(field).__name__}"
        )
    if not isinstance(frame, Frame):
        raise TypeError(
            f"frame must be an instance of Frame, got {type(frame).__name__}"
        )

    # Validate plane and index pairing
    if (plane is None and index is not None) or (plane is not None and index is None):
        raise ValueError("plane and index must be specified together as a pair")

    if plane is not None and plane not in _PROJECTIONS:
        raise ValueError(f"plane must be one of {set(_PROJECTIONS)}, got {plane!r}")

    if index is not None and (
        isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
    ):
        raise TypeError(f"index must be an integer, got {type(index).__name__}")

    if max_points is not None and (
        isinstance(max_points, (bool, np.bool_))
        or not isinstance(max_points, (int, np.integer))
        or max_points <= 0
    ):
        raise ValueError(f"max_points must be a positive integer, got {max_points!r}")

    # Validate unit
    _, unit_factor, unit_str = _length_unit(unit)

    is_3d = plane is None
    # Validate ax if provided
    if ax is not None:
        _validate_axes(ax, is_3d=is_3d)

    # Dispatch scalar vs vector
    if isinstance(field, DensityPerturbationGrid):
        if component is not None:
            raise ValueError(
                f"component={component!r} was specified, but "
                "DensityPerturbationGrid is a scalar field"
            )
        raw_data = field.delta_density_kg_m3
        values, part_label = _extract_part(raw_data, part)
        is_phase = part == "phase"
        unit_val = "rad" if is_phase else "kg/m³"
        colorbar_label = f"Density Perturbation{part_label} [{unit_val}]"
        field_title = "Density Perturbation"

        return _render_scalar_grid(
            x_m=field.x_m,
            y_m=field.y_m,
            z_m=field.z_m,
            values=values,
            frame=frame,
            plane=plane,
            index=index,
            is_phase=is_phase,
            colorbar_label=colorbar_label,
            field_title=field_title,
            unit_factor=unit_factor,
            unit_str=unit_str,
            max_points=max_points,
            ax=ax,
        )

    # DisplacementGrid
    return _plot_displacement_grid(
        field=field,
        frame=frame,
        plane=plane,
        index=index,
        part=part,
        component=component,
        unit_factor=unit_factor,
        unit_str=unit_str,
        max_points=max_points,
        ax=ax,
    )


def _render_scalar_grid(
    *,
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    values: np.ndarray,
    frame: Frame,
    plane: str | None,
    index: int | None,
    is_phase: bool,
    colorbar_label: str,
    field_title: str,
    unit_factor: float,
    unit_str: str,
    max_points: int | None,
    ax: Axes | None,
) -> Figure:
    """Unified internal scalar field rendering for density and displacement
    components."""
    x_coords = x_m * unit_factor
    y_coords = y_m * unit_factor
    z_coords = z_m * unit_factor
    cmap = "twilight" if is_phase else "viridis"

    if plane is not None:
        assert index is not None
        h_idx, v_idx, n_idx = _PROJECTIONS[plane]
        coords_list = [x_coords, y_coords, z_coords]
        n_len = len(coords_list[n_idx])
        if not (-n_len <= index < n_len):
            raise IndexError(
                f"index={index} out of bounds for axis {_AXIS_NAMES[n_idx]} "
                f"with size {n_len}"
            )
        idx = int(index if index >= 0 else index + n_len)

        # Slice 2D plane
        slicing: list[Any] = [slice(None), slice(None), slice(None)]
        slicing[n_idx] = idx
        val_slice = values[tuple(slicing)]

        h_axis = coords_list[h_idx]
        v_axis = coords_list[v_idx]

        h_grid, v_grid = np.meshgrid(h_axis, v_axis, indexing="ij")
        total_pts = h_grid.size
        h_flat = h_grid.ravel()
        v_flat = v_grid.ravel()
        val_flat = val_slice.ravel()

        if max_points is not None and total_pts > max_points:
            sel = np.linspace(0, total_pts - 1, num=max_points, dtype=np.intp)
            shown_pts = len(sel)
            h_flat = h_flat[sel]
            v_flat = v_flat[sel]
            val_flat = val_flat[sel]
        else:
            shown_pts = total_pts

        if ax is None:
            fig, target_ax = plt.subplots(figsize=(6.5, 5.0))
        else:
            fig = ax.figure
            target_ax = ax

        sc = target_ax.scatter(
            h_flat,
            v_flat,
            c=val_flat,
            cmap=cmap,
            s=25,
            edgecolors="none",
        )
        if is_phase:
            sc.set_clim(-np.pi, np.pi)

        cbar = fig.colorbar(sc, ax=target_ax)
        cbar.set_label(colorbar_label)

        h_name, v_name, n_name = (
            _AXIS_NAMES[h_idx],
            _AXIS_NAMES[v_idx],
            _AXIS_NAMES[n_idx],
        )
        target_ax.set_xlabel(f"{h_name} [{unit_str}]")
        target_ax.set_ylabel(f"{v_name} [{unit_str}]")
        n_coord_val = coords_list[n_idx][idx]

        downsample_txt = (
            f" ({shown_pts}/{total_pts} points)" if shown_pts < total_pts else ""
        )
        target_ax.set_title(
            f"{field_title} ({plane}-slice at {n_name}={n_coord_val:.3g} {unit_str})\n"
            f"Frame: {frame.name}{downsample_txt}"
        )
        # Limits always use full grid plane bounds regardless of max_points downsampling
        _set_limits_2d(
            target_ax,
            float(np.min(h_axis)),
            float(np.max(h_axis)),
            float(np.min(v_axis)),
            float(np.max(v_axis)),
        )
        return fig

    # 3D representation: avoid building full (Nx, Ny, Nz) meshgrid arrays if
    # downsampling
    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
    total_pts = nx * ny * nz

    if max_points is not None and total_pts > max_points:
        sel = np.linspace(0, total_pts - 1, num=max_points, dtype=np.intp)
        shown_pts = len(sel)
        ix, iy, iz = np.unravel_index(sel, (nx, ny, nz))
        x_flat = x_coords[ix]
        y_flat = y_coords[iy]
        z_flat = z_coords[iz]
        val_flat = values[ix, iy, iz]
    else:
        shown_pts = total_pts
        x_grid, y_grid, z_grid = np.meshgrid(
            x_coords, y_coords, z_coords, indexing="ij"
        )
        x_flat = x_grid.ravel()
        y_flat = y_grid.ravel()
        z_flat = z_grid.ravel()
        val_flat = values.ravel()

    if ax is None:
        fig = plt.figure(figsize=(7.5, 6.0))
        target_ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
        target_ax = ax

    sc = target_ax.scatter(
        x_flat,
        y_flat,
        z_flat,
        c=val_flat,
        cmap=cmap,
        s=15,
        alpha=0.85,
    )
    if is_phase:
        sc.set_clim(-np.pi, np.pi)

    cbar = fig.colorbar(sc, ax=target_ax, pad=0.1)
    cbar.set_label(colorbar_label)

    target_ax.set_xlabel(f"x [{unit_str}]")
    target_ax.set_ylabel(f"y [{unit_str}]")
    target_ax.set_zlabel(f"z [{unit_str}]")

    downsample_txt = (
        f" ({shown_pts}/{total_pts} points)" if shown_pts < total_pts else ""
    )
    target_ax.set_title(f"{field_title} (3D)\nFrame: {frame.name}{downsample_txt}")

    # Full grid 8 bounding corner coordinates to preserve original spatial domain
    corners_x = [float(np.min(x_coords)), float(np.max(x_coords))]
    corners_y = [float(np.min(y_coords)), float(np.max(y_coords))]
    corners_z = [float(np.min(z_coords)), float(np.max(z_coords))]
    cx, cy, cz = np.meshgrid(corners_x, corners_y, corners_z, indexing="ij")
    full_domain_pts = np.stack([cx.ravel(), cy.ravel(), cz.ravel()], axis=-1)
    _set_limits_3d(target_ax, full_domain_pts)
    return fig


def _plot_displacement_grid(
    *,
    field: DisplacementGrid,
    frame: Frame,
    plane: str | None,
    index: int | None,
    part: str | None,
    component: str | None,
    unit_factor: float,
    unit_str: str,
    max_points: int | None,
    ax: Axes | None,
) -> Figure:
    """Plot displacement grid either as scalar component or true-scale vector quiver."""
    raw_disp = field.displacement_m
    is_complex = np.iscomplexobj(raw_disp)

    valid_components = {"x", "y", "z", "norm"}
    if component is not None and component not in valid_components:
        raise ValueError(
            f"component must be one of {valid_components}, got {component!r}"
        )

    # Case A: Component specified -> Scalar plot of that component
    if component is not None:
        comp_idx_map = {"x": 0, "y": 1, "z": 2}

        if component == "norm":
            if is_complex:
                if part is None:
                    raise ValueError(
                        "Input displacement data is complex, so part must be "
                        "specified for component='norm'."
                    )
                if part not in {"real", "imag", "abs"}:
                    raise ValueError(f"part={part!r} is not supported for vector norm")
                if part == "real":
                    vec_real = np.real(raw_disp)
                    comp_values = np.linalg.norm(vec_real, axis=-1)
                    part_label = " (Re norm)"
                elif part == "imag":
                    vec_imag = np.imag(raw_disp)
                    comp_values = np.linalg.norm(vec_imag, axis=-1)
                    part_label = " (Im norm)"
                elif part == "abs":
                    # Absolute magnitude of complex vector components:
                    # sqrt(sum |xi_i|^2)
                    comp_values = np.sqrt(np.sum(np.abs(raw_disp) ** 2, axis=-1))
                    part_label = " (|v|)"
            else:
                if part is not None:
                    raise ValueError("part is only valid for complex field data")
                comp_values = np.linalg.norm(raw_disp, axis=-1)
                part_label = " (norm)"
            is_phase = False
            unit_val = "m"
        else:
            c_idx = comp_idx_map[component]
            raw_comp = raw_disp[..., c_idx]
            comp_values, part_label = _extract_part(raw_comp, part)
            is_phase = part == "phase"
            unit_val = "rad" if is_phase else "m"

        colorbar_label = f"Displacement {component}{part_label} [{unit_val}]"
        field_title = f"Displacement {component}"

        return _render_scalar_grid(
            x_m=field.x_m,
            y_m=field.y_m,
            z_m=field.z_m,
            values=comp_values,
            frame=frame,
            plane=plane,
            index=index,
            is_phase=is_phase,
            colorbar_label=colorbar_label,
            field_title=field_title,
            unit_factor=unit_factor,
            unit_str=unit_str,
            max_points=max_points,
            ax=ax,
        )

    # Case B: component is None -> Quiver vector plot
    if is_complex:
        if part is None:
            raise ValueError(
                "Input displacement data is complex, so part must be specified "
                "as 'real' or 'imag' for vector quiver."
            )
        if part not in {"real", "imag"}:
            raise ValueError(
                f"part={part!r} is not allowed for vector quiver plot. "
                "Vector arrows represent physical displacement directions and "
                "must be real vectors; use part='real' or part='imag', or "
                "specify component to plot scalar 'abs'/'phase'."
            )
        disp_real = np.real(raw_disp) if part == "real" else np.imag(raw_disp)
        part_txt = " (Re)" if part == "real" else " (Im)"
    else:
        if part is not None:
            raise ValueError(
                f"part={part!r} was specified, but displacement data is real. "
                "part is only valid for complex data."
            )
        disp_real = np.asarray(raw_disp, dtype=np.float64)
        part_txt = ""

    x_coords = field.x_m * unit_factor
    y_coords = field.y_m * unit_factor
    z_coords = field.z_m * unit_factor

    # Scale displacement vector components by unit_factor so x_end = x_start
    # + xi is consistent in display unit
    disp_scaled = disp_real * unit_factor

    if plane is not None:
        assert index is not None
        h_idx, v_idx, n_idx = _PROJECTIONS[plane]
        coords_list = [x_coords, y_coords, z_coords]
        n_len = len(coords_list[n_idx])
        if not (-n_len <= index < n_len):
            raise IndexError(
                f"index={index} out of bounds for axis {_AXIS_NAMES[n_idx]} "
                f"with size {n_len}"
            )
        idx = int(index if index >= 0 else index + n_len)

        slicing: list[Any] = [slice(None), slice(None), slice(None), slice(None)]
        slicing[n_idx] = idx
        disp_slice = disp_scaled[tuple(slicing)]

        h_axis = coords_list[h_idx]
        v_axis = coords_list[v_idx]

        h_grid, v_grid = np.meshgrid(h_axis, v_axis, indexing="ij")
        total_pts = h_grid.size

        # In 2D plane quiver, we display only the two in-plane components
        u_slice = disp_slice[..., h_idx]
        v_slice = disp_slice[..., v_idx]

        h_flat = h_grid.ravel()
        v_flat = v_grid.ravel()
        u_disp_flat = u_slice.ravel()
        v_disp_flat = v_slice.ravel()

        # Compute full spatial extent including vector endpoints before downsampling
        full_x_min = float(min(np.min(h_flat), np.min(h_flat + u_disp_flat)))
        full_x_max = float(max(np.max(h_flat), np.max(h_flat + u_disp_flat)))
        full_y_min = float(min(np.min(v_flat), np.min(v_flat + v_disp_flat)))
        full_y_max = float(max(np.max(v_flat), np.max(v_flat + v_disp_flat)))

        if max_points is not None and total_pts > max_points:
            sel = np.linspace(0, total_pts - 1, num=max_points, dtype=np.intp)
            shown_pts = len(sel)
            h_flat = h_flat[sel]
            v_flat = v_flat[sel]
            u_disp_flat = u_disp_flat[sel]
            v_disp_flat = v_disp_flat[sel]
        else:
            shown_pts = total_pts

        if ax is None:
            fig, target_ax = plt.subplots(figsize=(6.5, 5.0))
        else:
            fig = ax.figure
            target_ax = ax

        # True-scale quiver: angles='xy', scale_units='xy', scale=1
        target_ax.quiver(
            h_flat,
            v_flat,
            u_disp_flat,
            v_disp_flat,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="#1f77b4",
            width=0.004,
        )

        h_name, v_name, n_name = (
            _AXIS_NAMES[h_idx],
            _AXIS_NAMES[v_idx],
            _AXIS_NAMES[n_idx],
        )
        target_ax.set_xlabel(f"{h_name} [{unit_str}]")
        target_ax.set_ylabel(f"{v_name} [{unit_str}]")
        n_coord_val = coords_list[n_idx][idx]

        downsample_txt = (
            f" ({shown_pts}/{total_pts} vectors)" if shown_pts < total_pts else ""
        )
        target_ax.set_title(
            f"Displacement Vectors{part_txt} (in-plane {plane}-slice at "
            f"{n_name}={n_coord_val:.3g} {unit_str})\n"
            f"Frame: {frame.name}{downsample_txt}"
        )

        # Limits encompass full domain and all vector ends, invariant to downsampling
        _set_limits_2d(target_ax, full_x_min, full_x_max, full_y_min, full_y_max)
        return fig

    # 3D Quiver: avoid building full meshgrids when downsampling
    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
    total_pts = nx * ny * nz

    # Pre-calculate full bounds including endpoints without allocating 3D meshgrids
    x_ends = x_coords[:, None, None] + disp_scaled[..., 0]
    y_ends = y_coords[None, :, None] + disp_scaled[..., 1]
    z_ends = z_coords[None, None, :] + disp_scaled[..., 2]
    full_x_min = float(min(np.min(x_coords), np.min(x_ends)))
    full_x_max = float(max(np.max(x_coords), np.max(x_ends)))
    full_y_min = float(min(np.min(y_coords), np.min(y_ends)))
    full_y_max = float(max(np.max(y_coords), np.max(y_ends)))
    full_z_min = float(min(np.min(z_coords), np.min(z_ends)))
    full_z_max = float(max(np.max(z_coords), np.max(z_ends)))
    cx, cy, cz = np.meshgrid(
        [full_x_min, full_x_max],
        [full_y_min, full_y_max],
        [full_z_min, full_z_max],
        indexing="ij",
    )
    full_bounds_pts = np.stack([cx.ravel(), cy.ravel(), cz.ravel()], axis=-1)

    if max_points is not None and total_pts > max_points:
        sel = np.linspace(0, total_pts - 1, num=max_points, dtype=np.intp)
        shown_pts = len(sel)
        ix, iy, iz = np.unravel_index(sel, (nx, ny, nz))
        x_flat = x_coords[ix]
        y_flat = y_coords[iy]
        z_flat = z_coords[iz]
        u_flat = disp_scaled[ix, iy, iz, 0]
        v_flat = disp_scaled[ix, iy, iz, 1]
        w_flat = disp_scaled[ix, iy, iz, 2]
    else:
        shown_pts = total_pts
        x_grid, y_grid, z_grid = np.meshgrid(
            x_coords, y_coords, z_coords, indexing="ij"
        )
        x_flat = x_grid.ravel()
        y_flat = y_grid.ravel()
        z_flat = z_grid.ravel()
        u_flat = disp_scaled[..., 0].ravel()
        v_flat = disp_scaled[..., 1].ravel()
        w_flat = disp_scaled[..., 2].ravel()

    if ax is None:
        fig = plt.figure(figsize=(7.5, 6.0))
        target_ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
        target_ax = ax

    target_ax.quiver(
        x_flat,
        y_flat,
        z_flat,
        u_flat,
        v_flat,
        w_flat,
        length=1.0,
        normalize=False,
        color="#1f77b4",
    )

    target_ax.set_xlabel(f"x [{unit_str}]")
    target_ax.set_ylabel(f"y [{unit_str}]")
    target_ax.set_zlabel(f"z [{unit_str}]")

    downsample_txt = (
        f" ({shown_pts}/{total_pts} vectors)" if shown_pts < total_pts else ""
    )
    target_ax.set_title(
        f"Displacement Vectors{part_txt} (3D)\nFrame: {frame.name}{downsample_txt}"
    )

    # Set limits based on full field origins and ends
    _set_limits_3d(target_ax, full_bounds_pts)
    return fig
