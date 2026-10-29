"""Plot evaluation results across static fields, time series, and spectra.

Connects EvaluationResult objects to Matplotlib visualizations:
- Static fields: 2D spatial cross sections (imshow/pcolormesh) or scatter.
- Time series: Physical time waveforms vs time (s).
- Amplitude spectra: Discrete Fourier series amplitudes vs frequency (Hz).
- Fourier transforms: Continuous Fourier transform vs frequency (Hz).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from astropy import units as u

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_OBSERVABLE_UNITS = {
    "potential": "m^2/s^2",
    "acceleration": "m/s^2",
    "gradient": "1/s^2",
    "gravity_gradient": "1/s^2",
}

_EXPECTED_SI_UNITS = {
    "potential": u.m**2 / u.s**2,
    "acceleration": u.m / u.s**2,
    "gradient": 1 / u.s**2,
    "gravity_gradient": 1 / u.s**2,
}

_EXPECTED_FT_UNITS = {
    "potential": (u.m**2 / u.s**2) * u.s,
    "acceleration": (u.m / u.s**2) * u.s,
    "gradient": (1 / u.s**2) * u.s,
    "gravity_gradient": (1 / u.s**2) * u.s,
}

_COMPONENT_INDEX_3D = {
    "x": 0,
    "y": 1,
    "z": 2,
}

_GRADIENT_TENSOR_INDEX_3X3 = {
    "xx": (0, 0),
    "xy": (0, 1),
    "xz": (0, 2),
    "yx": (1, 0),
    "yy": (1, 1),
    "yz": (1, 2),
    "zx": (2, 0),
    "zy": (2, 1),
    "zz": (2, 2),
}

_GRADIENT_MATRIX_INDEX = {
    "xx": 0,
    "xy": 1,
    "xz": 2,
    "yx": 3,
    "yy": 4,
    "yz": 5,
    "zx": 6,
    "zy": 7,
    "zz": 8,
}


def _scale_matrix_to_si(matrix: Any, expected_unit: u.Unit) -> np.ndarray:
    """Scale a 3D matrix (targets, components, samples) to expected SI units.

    Retrieves unit metadata per cell (using matrix.meta[r, c].unit or
    matrix.units[r, c]) and converts each cell to expected_unit.
    Raises ValueError if unit metadata is missing or cannot be converted.
    """
    raw = np.asarray(matrix)
    if raw.ndim != 3:
        raise ValueError(
            f"Expected 3D matrix with shape (targets, components, samples), "
            f"got ndim={raw.ndim}"
        )
    n_targets, n_comps = raw.shape[0], raw.shape[1]
    scaled = np.empty_like(raw, dtype=raw.dtype)

    for r in range(n_targets):
        for c in range(n_comps):
            cell_unit = None
            if hasattr(matrix, "meta"):
                try:
                    meta_cell = matrix.meta[r, c]
                    cell_unit = getattr(meta_cell, "unit", None)
                except Exception:
                    cell_unit = None
            if cell_unit is None and hasattr(matrix, "units"):
                try:
                    cell_unit = matrix.units[r, c]
                except Exception:
                    cell_unit = None
            if cell_unit is None and hasattr(matrix, "unit"):
                cell_unit = matrix.unit

            if cell_unit is None:
                raise ValueError(
                    f"Missing unit metadata for cell ({r}, {c}) in matrix; "
                    f"cannot scale to expected SI unit {expected_unit}"
                )

            try:
                factor = u.Unit(cell_unit).to(expected_unit)
            except (TypeError, ValueError, u.UnitConversionError) as exc:
                raise ValueError(
                    f"Cell ({r}, {c}) unit {cell_unit!r} cannot be converted "
                    f"to expected SI unit {expected_unit}"
                ) from exc

            scaled[r, c] = raw[r, c] * factor

    return scaled


def _cluster_1d(coords: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Cluster 1D coordinates within tolerance tol.

    Returns (unique_centers, inverse_indices) where:
    - unique_centers: sorted 1D array of cluster centroids
    - inverse_indices: array of integer cluster IDs matching coords order
    """
    if len(coords) == 0:
        return np.empty(0, dtype=coords.dtype), np.empty(0, dtype=int)
    sort_idx = np.argsort(coords)
    sorted_c = coords[sort_idx]
    diffs = np.diff(sorted_c)
    split_points = np.where(diffs > tol)[0] + 1
    clusters = np.split(sorted_c, split_points)
    unique_centers = np.array(
        [float(np.mean(cl)) for cl in clusters], dtype=np.float64
    )
    sorted_inv = np.repeat(np.arange(len(clusters)), [len(cl) for cl in clusters])
    inverse = np.empty_like(sorted_inv)
    inverse[sort_idx] = sorted_inv
    return unique_centers, inverse


def plot_result(
    result: Any,
    *,
    observable: str = "acceleration",
    component: str | None = None,
    part: str = "magnitude",
    target_id: str | None = None,
    view: str = "auto",
    plane: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot physical fields, time series, or spectra from an EvaluationResult.

    Parameters
    ----------
    result : EvaluationResult
        The evaluation result object to plot.
    observable : {"potential", "acceleration", "gradient"}, default="acceleration"
        The physical observable to plot.
    component : str | None, default=None
        For vector/tensor observables: "norm", "x", "y", "z", "xx", "xy", etc.
        Defaults to "norm" for vector/tensor observables, or None for scalars.
    part : {"magnitude", "abs", "real", "imag", "phase", "angle"}, default="magnitude"
        For complex spectral or phasor quantities, specifies which representation
        to plot. Defaults to "magnitude".
    target_id : str | None, default=None
        Target identifier. Required for series/spectra if multiple targets exist.
    view : str, default="auto"
        Projection view or plot type: "xy", "xz", "yz", "slice", "scatter",
        "spectrum", "line", or "auto".
    plane : str | None, default=None
        Optional alias for view plane (e.g. "xy", "xz", "yz").
    ax : Axes | None, default=None
        Optional existing Matplotlib Axes.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes containing the plot.
    """
    from matplotlib import pyplot as plt

    from gragra.results._representation import (
        StaticRepresentation,
        TimeSeriesRepresentation,
    )
    from gragra.results._spectral import (
        AmplitudeSpectrumRepresentation,
        FourierTransformRepresentation,
        PhasorRepresentation,
    )

    if not hasattr(result, "context") or not hasattr(
        result.context, "representation"
    ):
        raise TypeError("result must be an EvaluationResult instance")

    rep = result.context.representation
    target_points = result.context.targets
    target_ids = tuple(target_points.ids)

    # Normalize plane / view aliases
    is_scatter_view = view == "scatter"
    if is_scatter_view:
        view = plane if plane is not None else "xy"
    elif plane is not None:
        view = plane
    elif view == "slice":
        view = "xy"

    # Normalize observable name
    canonical_obs = "gradient" if observable == "gravity_gradient" else observable
    payload = getattr(result, canonical_obs, None)
    if payload is None:
        payload = getattr(result, observable, None)
    if payload is None:
        raise ValueError(f"Observable {observable!r} not found in result")

    unit_base = _OBSERVABLE_UNITS.get(canonical_obs, "SI")

    # Determine component default
    if component is None:
        component = "norm" if canonical_obs in ("acceleration", "gradient") else None

    valid_parts = {"magnitude", "abs", "real", "imag", "phase", "angle"}
    if part not in valid_parts:
        raise ValueError(
            f"unknown part {part!r}; expected one of {sorted(valid_parts)}"
        )

    # -------------------------------------------------------------------------
    # Handle 1: Static Representation (2D spatial field or scatter)
    # -------------------------------------------------------------------------
    if isinstance(rep, StaticRepresentation):
        context_frame = getattr(result.context, "frame", None)
        if (
            context_frame is not None
            and hasattr(target_points, "frame")
            and target_points.frame is not context_frame
        ):
            coords_m = target_points.frame.transform_points_m(
                target_points.positions_m,
                context_frame,
            )
        else:
            coords_m = target_points.positions_m

        # Ensure raw_vals is converted to expected SI units
        expected_si = _EXPECTED_SI_UNITS.get(canonical_obs)
        if expected_si is not None:
            if hasattr(payload, "to_value"):
                try:
                    raw_vals = np.asarray(payload.to_value(expected_si))
                except Exception as exc:
                    raise ValueError(
                        f"Cannot convert {canonical_obs} payload to expected "
                        f"SI unit {expected_si}"
                    ) from exc
            elif isinstance(payload, u.Quantity):
                try:
                    raw_vals = np.asarray(payload.to(expected_si).value)
                except Exception as exc:
                    raise ValueError(
                        f"Cannot convert {canonical_obs} payload to expected "
                        f"SI unit {expected_si}"
                    ) from exc
            else:
                raise ValueError(
                    f"{canonical_obs} payload has no unit information; "
                    f"expected {expected_si}"
                )
        else:
            raw_vals = np.asarray(payload)

        # Coordinate axes according to projection plane
        if view in ("auto", "xy"):
            h_idx, v_idx, w_idx, h_name, v_name, w_name = 0, 1, 2, "x", "y", "z"
        elif view == "xz":
            h_idx, v_idx, w_idx, h_name, v_name, w_name = 0, 2, 1, "x", "z", "y"
        elif view == "yz":
            h_idx, v_idx, w_idx, h_name, v_name, w_name = 1, 2, 0, "y", "z", "x"
        else:
            h_idx, v_idx, w_idx, h_name, v_name, w_name = 0, 1, 2, "x", "y", "z"

        total_targets = len(coords_m)
        coord_scale = (
            float(np.max(np.abs(coords_m))) if total_targets > 0 else 1.0
        )
        coord_scale = max(coord_scale, 1.0)
        # Numerical tolerance for coordinates after 3D rotations & translations
        coord_tol = max(1e-10, 1e-12 * coord_scale)
        plane_tol = max(1e-9, 1e-12 * coord_scale)

        # Validate that the orthogonal axis is constant within numerical tolerance
        w_coords = coords_m[:, w_idx]
        w_span = (
            float(np.max(w_coords) - np.min(w_coords))
            if total_targets > 0
            else 0.0
        )
        if w_span > plane_tol:
            w_clusters, _ = _cluster_1d(w_coords, plane_tol)
            raise ValueError(
                f"Target points span multiple coordinates along the orthogonal "
                f"{w_name} axis (span={w_span:.3e} m > tol={plane_tol:.3e} m; "
                f"{len(w_clusters)} distinct positions found). A 2D slice "
                f"field requires targets in a single plane; filter targets to a "
                f"constant {w_name}."
            )

        # Extract scalar field according to observable and component
        if canonical_obs == "potential":
            field_vals = np.squeeze(raw_vals)
            comp_str = ""
        elif canonical_obs == "acceleration":
            if component == "norm":
                field_vals = np.linalg.norm(raw_vals, axis=1)
                comp_str = " (norm)"
            elif component in _COMPONENT_INDEX_3D:
                field_vals = raw_vals[:, _COMPONENT_INDEX_3D[component]]
                comp_str = f" ({component})"
            else:
                raise ValueError(
                    f"Invalid component {component!r} for acceleration; "
                    f"expected 'norm', 'x', 'y', or 'z'"
                )
        elif canonical_obs == "gradient":
            if component == "norm":
                field_vals = np.linalg.norm(raw_vals, axis=(1, 2))
                comp_str = " (norm)"
            elif component in _GRADIENT_TENSOR_INDEX_3X3:
                r_i, c_i = _GRADIENT_TENSOR_INDEX_3X3[component]
                field_vals = raw_vals[:, r_i, c_i]
                comp_str = f" ({component})"
            else:
                expected_keys = sorted(_GRADIENT_TENSOR_INDEX_3X3.keys())
                raise ValueError(
                    f"Invalid component {component!r} for gradient; "
                    f"expected 'norm' or one of {expected_keys}"
                )
        else:
            field_vals = np.squeeze(raw_vals)
            comp_str = f" ({component})" if component else ""

        if ax is None:
            fig, ax = plt.subplots(figsize=(7.5, 6.0))
        else:
            fig = ax.get_figure()

        u_coords = coords_m[:, h_idx]
        v_coords = coords_m[:, v_idx]
        total_targets = len(coords_m)

        if is_scatter_view:
            sc = ax.scatter(
                u_coords,
                v_coords,
                c=field_vals,
                cmap="viridis",
            )
            fig.colorbar(sc, ax=ax, label=f"{canonical_obs}{comp_str} [{unit_base}]")
            ax.set_xlabel(f"{h_name} [m]")
            ax.set_ylabel(f"{v_name} [m]")
            ax.set_title(
                f"{canonical_obs}{comp_str} (scatter, {h_name}{v_name}-plane)"
            )
            return fig, ax

        # Cluster coordinate axes within numerical tolerance
        u_unique, u_inv = _cluster_1d(u_coords, coord_tol)
        v_unique, v_inv = _cluster_1d(v_coords, coord_tol)
        nu, nv = len(u_unique), len(v_unique)

        # Detect duplicate projected points using integer cluster bins
        grid_counts = np.zeros((nv, nu), dtype=int)
        np.add.at(grid_counts, (v_inv, u_inv), 1)
        has_duplicate_coords = np.any(grid_counts > 1)

        if has_duplicate_coords:
            n_unq = int(np.count_nonzero(grid_counts > 0))
            raise ValueError(
                f"Target points contain duplicate coordinates in the "
                f"{h_name}-{v_name} projection plane ({n_unq} unique out of "
                f"{total_targets}). Filter targets to a single slice or specify "
                f"a non-overlapping projection plane."
            )

        is_complete_grid = (
            (nu * nv == total_targets)
            and np.all(grid_counts == 1)
            and nu >= 2
            and nv >= 2
        )

        if is_complete_grid:
            # Coordinate-based placement: immune to input target ordering
            grid_2d = np.empty((nv, nu), dtype=field_vals.dtype)
            grid_2d[v_inv, u_inv] = field_vals

            du = np.diff(u_unique)
            dv = np.diff(v_unique)
            is_uniform_u = np.allclose(du, du[0], rtol=1e-7, atol=1e-9)
            is_uniform_v = np.allclose(dv, dv[0], rtol=1e-7, atol=1e-9)

            if is_uniform_u and is_uniform_v:
                step_u = float(du[0])
                step_v = float(dv[0])
                # Pixel edges enclosing sample centers
                extent = [
                    float(u_unique[0] - 0.5 * step_u),
                    float(u_unique[-1] + 0.5 * step_u),
                    float(v_unique[0] - 0.5 * step_v),
                    float(v_unique[-1] + 0.5 * step_v),
                ]
                im = ax.imshow(
                    grid_2d,
                    origin="lower",
                    extent=extent,
                    cmap="viridis",
                    aspect="auto",
                )
            else:
                # Non-uniform Cartesian grid: use pcolormesh with exact cell boundaries
                u_edges = np.empty(nu + 1, dtype=np.float64)
                u_edges[0] = u_unique[0] - 0.5 * (u_unique[1] - u_unique[0])
                u_edges[1:-1] = 0.5 * (u_unique[:-1] + u_unique[1:])
                u_edges[-1] = u_unique[-1] + 0.5 * (u_unique[-1] - u_unique[-2])

                v_edges = np.empty(nv + 1, dtype=np.float64)
                v_edges[0] = v_unique[0] - 0.5 * (v_unique[1] - v_unique[0])
                v_edges[1:-1] = 0.5 * (v_unique[:-1] + v_unique[1:])
                v_edges[-1] = v_unique[-1] + 0.5 * (v_unique[-1] - v_unique[-2])

                im = ax.pcolormesh(u_edges, v_edges, grid_2d, cmap="viridis")
                ax.set_xlim(u_edges[0], u_edges[-1])
                ax.set_ylim(v_edges[0], v_edges[-1])

            fig.colorbar(im, ax=ax, label=f"{canonical_obs}{comp_str} [{unit_base}]")
            ax.set_xlabel(f"{h_name} [m]")
            ax.set_ylabel(f"{v_name} [m]")
            ax.set_title(f"{canonical_obs}{comp_str} field")
        else:
            # Missing points or irregular targets: fallback gracefully to scatter
            sc = ax.scatter(
                u_coords,
                v_coords,
                c=field_vals,
                cmap="viridis",
            )
            fig.colorbar(sc, ax=ax, label=f"{canonical_obs}{comp_str} [{unit_base}]")
            ax.set_xlabel(f"{h_name} [m]")
            ax.set_ylabel(f"{v_name} [m]")
            ax.set_title(f"{canonical_obs}{comp_str} scatter")

        return fig, ax

    # -------------------------------------------------------------------------
    # Target selection for series / spectra
    # -------------------------------------------------------------------------
    if target_id is None:
        if len(target_ids) == 1:
            target_idx = 0
            target_id_str = target_ids[0]
        else:
            raise ValueError(
                f"target_id is required when result has {len(target_ids)} targets"
            )
    else:
        if target_id not in target_ids:
            raise ValueError(f"unknown target_id {target_id!r}")
        target_idx = target_ids.index(target_id)
        target_id_str = target_id

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
    else:
        fig = ax.get_figure()

    # -------------------------------------------------------------------------
    # Handle 2: TimeSeriesRepresentation
    # -------------------------------------------------------------------------
    if isinstance(rep, TimeSeriesRepresentation):
        matrix = payload
        t_arr = np.asarray(
            matrix.times.value if hasattr(matrix.times, "value") else matrix.times
        )
        raw_mat = _scale_matrix_to_si(
            matrix, _EXPECTED_SI_UNITS[canonical_obs]
        )[target_idx]

        if canonical_obs == "potential":
            y_arr = np.asarray(raw_mat[0])
            comp_str = ""
        elif canonical_obs == "acceleration":
            if component == "norm":
                y_arr = np.linalg.norm(raw_mat, axis=0)
                comp_str = " (norm)"
            elif component in _COMPONENT_INDEX_3D:
                c_idx = _COMPONENT_INDEX_3D[component]
                y_arr = np.asarray(raw_mat[c_idx])
                comp_str = f" ({component})"
            else:
                raise ValueError(
                    f"Invalid component {component!r} for acceleration"
                )
        elif canonical_obs == "gradient":
            if component == "norm":
                y_arr = np.linalg.norm(raw_mat, axis=0)
                comp_str = " (norm)"
            elif component in _GRADIENT_MATRIX_INDEX:
                c_idx = _GRADIENT_MATRIX_INDEX[component]
                y_arr = np.asarray(raw_mat[c_idx])
                comp_str = f" ({component})"
            else:
                raise ValueError(
                    f"Invalid component {component!r} for gradient"
                )
        else:
            y_arr = np.asarray(raw_mat)
            comp_str = f" ({component})" if component else ""

        ax.plot(t_arr, y_arr, color="#316aa5")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{canonical_obs.capitalize()}{comp_str} [{unit_base}]")
        ax.set_title(f"{canonical_obs}{comp_str} target={target_id_str}")
        ax.grid(True, linestyle="--", alpha=0.6)
        return fig, ax

    # -------------------------------------------------------------------------
    # Helper to extract spectral/phasor arrays with complex part handling
    # -------------------------------------------------------------------------
    def _extract_complex_spectral(matrix: Any) -> tuple[np.ndarray, str]:
        exp_u = (
            _EXPECTED_FT_UNITS[canonical_obs]
            if isinstance(rep, FourierTransformRepresentation)
            else _EXPECTED_SI_UNITS[canonical_obs]
        )
        scaled_mat = _scale_matrix_to_si(matrix, exp_u)[target_idx]

        if canonical_obs == "potential":
            complex_vals = np.asarray(scaled_mat[0])
            comp_name = ""
        elif canonical_obs == "acceleration":
            if component == "norm":
                if part in ("magnitude", "abs"):
                    return np.linalg.norm(scaled_mat, axis=0), " (norm)"
                raise ValueError(
                    f"part={part!r} is ambiguous with component='norm'; "
                    f"specify a single component ('x', 'y', 'z')"
                )
            if component in _COMPONENT_INDEX_3D:
                c_idx = _COMPONENT_INDEX_3D[component]
                complex_vals = np.asarray(scaled_mat[c_idx])
                comp_name = f" ({component})"
            else:
                raise ValueError(f"Invalid component {component!r} for acceleration")
        elif canonical_obs == "gradient":
            if component == "norm":
                if part in ("magnitude", "abs"):
                    return np.linalg.norm(scaled_mat, axis=0), " (norm)"
                raise ValueError(
                    f"part={part!r} is ambiguous with component='norm'; "
                    f"specify a single component (e.g. 'xx', 'xy')"
                )
            if component in _GRADIENT_MATRIX_INDEX:
                c_idx = _GRADIENT_MATRIX_INDEX[component]
                complex_vals = np.asarray(scaled_mat[c_idx])
                comp_name = f" ({component})"
            else:
                raise ValueError(f"Invalid component {component!r} for gradient")
        else:
            complex_vals = np.asarray(scaled_mat)
            comp_name = f" ({component})" if component else ""

        if part in ("magnitude", "abs"):
            vals = np.abs(complex_vals)
        elif part == "real":
            vals = np.real(complex_vals)
        elif part == "imag":
            vals = np.imag(complex_vals)
        elif part in ("phase", "angle"):
            vals = np.angle(complex_vals)
        else:
            vals = np.abs(complex_vals)

        return vals, comp_name

    # Check reference time for phase and spectral titles
    tref = getattr(result.context, "reference_time", None)
    tref_info = f" [tref={tref}]" if tref is not None else ""

    # -------------------------------------------------------------------------
    # Handle 3: AmplitudeSpectrumRepresentation or PhasorRepresentation
    # -------------------------------------------------------------------------
    if isinstance(rep, (AmplitudeSpectrumRepresentation, PhasorRepresentation)):
        matrix = payload
        f_arr = np.asarray(
            matrix.frequencies.to_value("Hz")
            if hasattr(matrix.frequencies, "to_value")
            else matrix.frequencies
        )

        y_arr, comp_name = _extract_complex_spectral(matrix)

        if part in ("magnitude", "abs"):
            y_label = f"Amplitude [{unit_base}]"
        elif part == "real":
            y_label = f"Re({canonical_obs}) [{unit_base}]"
        elif part == "imag":
            y_label = f"Im({canonical_obs}) [{unit_base}]"
        elif part in ("phase", "angle"):
            y_label = "Phase [rad]"
        else:
            y_label = f"Amplitude [{unit_base}]"

        ax.plot(f_arr, y_arr, marker="o", color="#27864b")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(y_label)
        ax.set_title(
            f"{canonical_obs}{comp_name} ({part}) spectrum "
            f"target={target_id_str}{tref_info}"
        )
        ax.grid(True, linestyle="--", alpha=0.6)
        return fig, ax

    # -------------------------------------------------------------------------
    # Handle 4: FourierTransformRepresentation
    # -------------------------------------------------------------------------
    if isinstance(rep, FourierTransformRepresentation):
        matrix = payload
        f_arr = np.asarray(
            matrix.frequencies.to_value("Hz")
            if hasattr(matrix.frequencies, "to_value")
            else matrix.frequencies
        )

        y_arr, comp_name = _extract_complex_spectral(matrix)

        ft_unit = f"({unit_base}) s"
        if part in ("magnitude", "abs"):
            y_label = f"Fourier Transform [{ft_unit}]"
        elif part == "real":
            y_label = f"Re(FT) [{ft_unit}]"
        elif part == "imag":
            y_label = f"Im(FT) [{ft_unit}]"
        elif part in ("phase", "angle"):
            y_label = "Phase [rad]"
        else:
            y_label = f"Fourier Transform [{ft_unit}]"

        ax.plot(f_arr, y_arr, marker="s", color="#c43c39")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(y_label)
        ax.set_title(
            f"{canonical_obs}{comp_name} ({part}) FT "
            f"target={target_id_str}{tref_info}"
        )
        ax.grid(True, linestyle="--", alpha=0.6)
        return fig, ax

    raise NotImplementedError(
        f"Plotting for representation {type(rep).__name__!r} is not implemented"
    )
