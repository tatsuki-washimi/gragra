"""Shape plotting for Cuboid, Cylinder, and SphericalShell models and setups.

Provides plot_shape to visualize analytical source geometries in 3D or 2D projections
without executing or modifying the underlying models.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np

from gragra.geometry import Cuboid, Cylinder, Frame, SphericalShell

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _is_unsupported_name(name: str) -> bool:
    unsupported_markers = (
        "Cavity",
        "cavity",
        "Tetra",
        "tetra",
        "Hex8",
        "hex8",
        "RigidMotion",
        "MovingBody",
        "constant_velocity",
        "translational_oscillation",
        "fixed_axis_rotation",
    )
    return any(marker in name for marker in unsupported_markers)


def _resolve_geometry(target: Any) -> tuple[Cuboid | Cylinder | SphericalShell, str]:
    """Resolve geometry from a geometry object, model, or setup.

    Returns
    -------
    tuple[Cuboid | Cylinder | SphericalShell, str]
        The resolved geometry and the original type name.

    Raises
    ------
    NotImplementedError
        If target represents an unsupported shape (Cavity, Mesh, Moving body).
    TypeError
        If target cannot be resolved to a supported geometry.
    """
    target_type = target if isinstance(target, type) else type(target)
    type_name = target_type.__name__

    # Explicitly reject unsupported shape categories (P2 scope)
    if _is_unsupported_name(type_name) or (
        hasattr(target, "__module__") and _is_unsupported_name(str(target.__module__))
    ):
        raise NotImplementedError(
            f"Shape plotting for {type_name!r} is not implemented (deferred to P2)."
        )

    # Check setup.model or target.model
    if hasattr(target, "model"):
        model_name = type(target.model).__name__
        if _is_unsupported_name(model_name):
            raise NotImplementedError(
                f"Shape plotting for {model_name!r} is not implemented "
                "(deferred to P2)."
            )

    # Check target.resolved
    if hasattr(target, "resolved"):
        resolved_model = getattr(target.resolved, "model_spec", None)
        if resolved_model is not None:
            res_name = type(resolved_model).__name__
            if _is_unsupported_name(res_name):
                raise NotImplementedError(
                    f"Shape plotting for {res_name!r} is not implemented "
                    "(deferred to P2)."
                )

    # 1. Direct geometry object
    if isinstance(target, (Cuboid, Cylinder, SphericalShell)):
        return target, type_name

    # 2. Model object with .geometry attribute
    if hasattr(target, "geometry") and isinstance(
        target.geometry, (Cuboid, Cylinder, SphericalShell)
    ):
        return target.geometry, type_name

    # 3. Setup with resolved.model_spec.geometry attribute
    if (
        hasattr(target, "resolved")
        and hasattr(target.resolved, "model_spec")
        and hasattr(target.resolved.model_spec, "geometry")
        and isinstance(
            target.resolved.model_spec.geometry, (Cuboid, Cylinder, SphericalShell)
        )
    ):
        return target.resolved.model_spec.geometry, type_name

    # 4. Model or setup with .spec.geometry attribute
    if (
        hasattr(target, "spec")
        and hasattr(target.spec, "geometry")
        and isinstance(target.spec.geometry, (Cuboid, Cylinder, SphericalShell))
    ):
        return target.spec.geometry, type_name

    raise TypeError(
        "Cannot resolve supported geometry (Cuboid, Cylinder, SphericalShell) "
        f"from {type_name!r}."
    )


def plot_shape(
    target: Any,
    *,
    view: str = "3d",
    ax: Axes | None = None,
    display_frame: Frame | None = None,
    color: str | None = None,
    alpha: float | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot an analytical source shape (Cuboid, Cylinder, SphericalShell).

    Parameters
    ----------
    target : Any
        A geometry object, analytic model, or evaluation setup.
    view : str, default="3d"
        The projection view. Currently "3d" is supported.
    ax : Axes | None, default=None
        An optional Matplotlib Axes instance. If None, a new Figure and Axes
        are created.
    display_frame : Frame | None, default=None
        The destination coordinate frame for rendering. Defaults to the
        geometry frame's root.
    color : str | None, default=None
        Surface / face color for the shape.
    alpha : float | None, default=None
        Transparency alpha (0.0 to 1.0).
    **kwargs : Any
        Additional keyword arguments passed to Matplotlib surface / collection artists.

    Returns
    -------
    tuple[Figure, Axes]
        The created or updated figure and axes.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    geom, _ = _resolve_geometry(target)

    if display_frame is None:
        display_frame = geom.frame.root if geom.frame.root is not None else geom.frame

    if view != "3d":
        raise ValueError(f"view={view!r} is not supported yet; only '3d' is supported.")

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    all_points: list[np.ndarray] = []

    if isinstance(geom, Cuboid):
        c = geom.center_m
        lx, ly, lz = geom.lengths_m
        local_corners = np.array(
            list(
                itertools.product(
                    [c[0] - lx / 2, c[0] + lx / 2],
                    [c[1] - ly / 2, c[1] + ly / 2],
                    [c[2] - lz / 2, c[2] + lz / 2],
                )
            ),
            dtype=np.float64,
        )
        world_corners = geom.frame.transform_points_m(local_corners, display_frame)
        all_points.append(world_corners)

        # Six faces of the cuboid defined by vertex indices
        faces = [
            [world_corners[0], world_corners[1], world_corners[3], world_corners[2]],
            [world_corners[4], world_corners[5], world_corners[7], world_corners[6]],
            [world_corners[0], world_corners[1], world_corners[5], world_corners[4]],
            [world_corners[2], world_corners[3], world_corners[7], world_corners[6]],
            [world_corners[0], world_corners[2], world_corners[6], world_corners[4]],
            [world_corners[1], world_corners[3], world_corners[7], world_corners[5]],
        ]
        face_color = color if color is not None else "#316aa5"
        face_alpha = alpha if alpha is not None else 0.35
        edge_color = kwargs.pop("edgecolor", "black")
        poly = Poly3DCollection(
            faces,
            alpha=face_alpha,
            edgecolor=edge_color,
            facecolor=face_color,
            **kwargs,
        )
        ax.add_collection3d(poly)

    elif isinstance(geom, Cylinder):
        c = geom.center_m
        r = geom.radius_m
        h = geom.height_m

        theta = np.linspace(0, 2 * np.pi, 33)
        z_local = np.linspace(c[2] - h / 2, c[2] + h / 2, 10)
        theta_grid, z_grid = np.meshgrid(theta, z_local)
        x_grid = c[0] + r * np.cos(theta_grid)
        y_grid = c[1] + r * np.sin(theta_grid)

        grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])
        world_grid = geom.frame.transform_points_m(grid_points, display_frame)
        all_points.append(world_grid)

        x_w = world_grid[:, 0].reshape(x_grid.shape)
        y_w = world_grid[:, 1].reshape(y_grid.shape)
        z_w = world_grid[:, 2].reshape(z_grid.shape)

        surf_color = color if color is not None else "#27864b"
        surf_alpha = alpha if alpha is not None else 0.4
        ax.plot_surface(x_w, y_w, z_w, alpha=surf_alpha, color=surf_color, **kwargs)

        # Top and bottom caps
        for z_cap in [c[2] - h / 2, c[2] + h / 2]:
            cap_points = np.column_stack(
                [
                    c[0] + r * np.cos(theta),
                    c[1] + r * np.sin(theta),
                    np.full_like(theta, z_cap),
                ]
            )
            world_cap = geom.frame.transform_points_m(cap_points, display_frame)
            all_points.append(world_cap)
            poly_cap = Poly3DCollection(
                [world_cap], alpha=surf_alpha, facecolor=surf_color
            )
            ax.add_collection3d(poly_cap)

    elif isinstance(geom, SphericalShell):
        c = geom.center_m
        r_in = geom.inner_radius_m
        r_out = geom.outer_radius_m

        u_theta = np.linspace(0, 2 * np.pi, 33)
        v_phi = np.linspace(0, np.pi, 17)
        u_grid, v_grid = np.meshgrid(u_theta, v_phi)

        surf_color = color if color is not None else "#c43c39"
        surf_alpha = alpha if alpha is not None else 0.3

        # Outer shell
        x_out = c[0] + r_out * np.sin(v_grid) * np.cos(u_grid)
        y_out = c[1] + r_out * np.sin(v_grid) * np.sin(u_grid)
        z_out = c[2] + r_out * np.cos(v_grid)
        pts_out = np.column_stack([x_out.ravel(), y_out.ravel(), z_out.ravel()])
        world_out = geom.frame.transform_points_m(pts_out, display_frame)
        all_points.append(world_out)
        ax.plot_surface(
            world_out[:, 0].reshape(x_out.shape),
            world_out[:, 1].reshape(y_out.shape),
            world_out[:, 2].reshape(z_out.shape),
            alpha=surf_alpha,
            color=surf_color,
            **kwargs,
        )

        # Inner shell
        x_in = c[0] + r_in * np.sin(v_grid) * np.cos(u_grid)
        y_in = c[1] + r_in * np.sin(v_grid) * np.sin(u_grid)
        z_in = c[2] + r_in * np.cos(v_grid)
        pts_in = np.column_stack([x_in.ravel(), y_in.ravel(), z_in.ravel()])
        world_in = geom.frame.transform_points_m(pts_in, display_frame)
        all_points.append(world_in)
        ax.plot_surface(
            world_in[:, 0].reshape(x_in.shape),
            world_in[:, 1].reshape(y_in.shape),
            world_in[:, 2].reshape(z_in.shape),
            alpha=max(0.1, surf_alpha * 0.7),
            color=color if color is not None else "#e28c8a",
            **kwargs,
        )

    all_pts_arr = np.concatenate(all_points, axis=0)
    min_xyz = np.min(all_pts_arr, axis=0)
    max_xyz = np.max(all_pts_arr, axis=0)

    # Set explicit axis limits so that min/max bounds are correctly spanned
    ax.set_xlim(min_xyz[0], max_xyz[0])
    ax.set_ylim(min_xyz[1], max_xyz[1])
    ax.set_zlim(min_xyz[2], max_xyz[2])

    ax.set_xlabel(f"x [{display_frame.name}]")
    ax.set_ylabel(f"y [{display_frame.name}]")
    ax.set_zlabel(f"z [{display_frame.name}]")

    return fig, ax
