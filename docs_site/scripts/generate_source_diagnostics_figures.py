#!/usr/bin/env python3
"""Regenerate figures for docs_site/docs/guide/source-diagnostics.md.

Run from the repository root:

    python docs_site/scripts/generate_source_diagnostics_figures.py

Requires ``pip install -e ".[dev]"`` (gragra, GWexpy, Matplotlib). Uses Agg backend.
Figure models reuse common targets where possible; each function documents its setup.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if str(Path(__file__).resolve().parents[2] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs_site" / "docs" / "assets" / "guide" / "source-diagnostics"

# Shared radial bins for static diagnostics (m).
RADIAL_BINS_M = [0.0, 6.0, 12.0, 20.0, 35.0]

if TYPE_CHECKING:
    from gragra.evaluation import EvaluationSetup
    from gragra.geometry import Frame
    from gragra.gravity_sources import VolumeElementSource
    from gragra.results import GravityFieldDefinition
    from gragra.targets import TargetPoints


def _setup_mpl() -> None:
    import matplotlib

    matplotlib.use("Agg")


def _tutorial_volume_setup() -> tuple[
    EvaluationSetup, VolumeElementSource, Frame, GravityFieldDefinition, TargetPoints
]:
    """Four volume elements for short guide snippets (minimal element count)."""
    from astropy import units as u

    from gragra.geometry import Frame
    from gragra.gravity_sources import VolumeElementSource, point_sum
    from gragra.results import Contribution, GravityFieldDefinition, PotentialReference
    from gragra.targets import TargetPoints

    root = Frame("root")
    field = GravityFieldDefinition.source_field(
        "source", Contribution("total"), PotentialReference.ZERO_AT_INFINITY
    )
    source = VolumeElementSource(
        [
            [2.0, 0.0, -3.0],
            [8.0, 1.5, 0.0],
            [15.0, -1.0, 4.0],
            [0.0, 5.0, 2.0],
        ],
        [1.0, 2.0, 3.0, 4.0],
        np.array([500.0, 1500.0, 1500.0, 2500.0]),
    )
    targets = TargetPoints(
        [[-3.0, -4.0, 0.0], [10.0, 0.0, 0.0]] * u.m, ("A", "B"), root
    )
    setup = point_sum(source, frame=root, field_definition=field).at(targets)
    return setup, source, root, field, targets


def _rich_volume_setup() -> tuple[
    EvaluationSetup, VolumeElementSource, Frame, GravityFieldDefinition, TargetPoints
]:
    """~240 volume elements over a 3D grid (15x2x8 figure-quality spatial spread)."""
    from astropy import units as u

    from gragra.geometry import Frame
    from gragra.gravity_sources import VolumeElementSource, point_sum
    from gragra.results import Contribution, GravityFieldDefinition, PotentialReference
    from gragra.targets import TargetPoints

    rng = np.random.default_rng(17)
    nx, ny, nz = 15, 2, 8
    xs = np.linspace(-18.0, 18.0, nx)
    ys = np.linspace(-10.0, 10.0, ny)
    zs = np.linspace(-8.0, 8.0, nz)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    positions = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    positions += rng.normal(0.0, 0.25, positions.shape)
    n = len(positions)
    cell = 0.6
    volumes = np.full(n, cell**3, dtype=np.float64)
    r = np.linalg.norm(positions[:, :2], axis=1)
    density = 600.0 + 350.0 * np.sin(positions[:, 2] / 4.0) + 180.0 * (r / 20.0)
    density = np.clip(density, 200.0, 2800.0)
    root = Frame("viz_root")
    field = GravityFieldDefinition.source_field(
        "source", Contribution("total"), PotentialReference.ZERO_AT_INFINITY
    )
    source = VolumeElementSource(positions, volumes, density)
    targets = TargetPoints(
        [[-5.0, -3.0, 1.5], [12.0, 2.0, -1.0]] * u.m, ("A", "B"), root
    )
    setup = point_sum(source, frame=root, field_definition=field).at(targets)
    return setup, source, root, field, targets


def fig_map_xz_kernel() -> None:
    from gragra.plotting import plot_map

    setup, _, _, _, _ = _rich_volume_setup()
    fig = plot_map(setup, "kernel_norm", target_id="A", view="xz")
    fig.savefig(OUT_DIR / "map_xz_kernel_norm.svg", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def fig_map_xz_delta_a_x() -> None:
    from gragra.plotting import plot_map

    setup, _, _, _, _ = _rich_volume_setup()
    fig = plot_map(setup, "delta_a_x", target_id="A", view="xz", color_scale="linear")
    fig.savefig(OUT_DIR / "map_xz_delta_a_x.svg", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def fig_radial_mass_over_r2() -> None:
    """Radial :math:`\\Delta m/r^2` with ``normalization=\"fraction\"``
    (bin fraction).
    """
    from gragra.plotting import plot_hist, radial_histogram

    setup, _, _, _, _ = _rich_volume_setup()
    radial = radial_histogram(
        setup,
        "mass_over_r2",
        target_id="A",
        bins=RADIAL_BINS_M,
        normalization="fraction",
    )
    fig, _, _ = plot_hist(radial)
    fig.savefig(OUT_DIR / "radial_mass_over_r2_fraction.svg", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def fig_gg_components() -> None:
    import matplotlib.pyplot as plt

    from gragra.plotting import gravity_gradient_radial_histogram

    setup, _, _, _, _ = _rich_volume_setup()
    gg = gravity_gradient_radial_histogram(
        setup,
        target_id="A",
        bins=RADIAL_BINS_M,
        softening_m=0.0,
    )
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.5), sharex=True)
    total = gg.total_tensor.value
    trace = float(total[0, 0] + total[1, 1] + total[2, 2])
    for ax, comp in zip(axes, ("xx", "yy", "zz"), strict=True):
        h = gg.histogram(comp)
        ax.stairs(h.values.value, h.edges.value, color="#1f77b4", linewidth=1.5)
        ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle=":")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_ylabel(f"GG_{comp} [{h.unit.to_string()}]", fontsize=10)
        ax.set_title(f"$\\Gamma_{{{comp}}}$ by distance (signed)", fontsize=11)
        ax.tick_params(labelsize=9)
    axes[-1].set_xlabel(f"distance [{h.xunit.to_string()}]", fontsize=10)
    fig.suptitle(
        f"Static GG components (target A, softening=0); "
        f"tr($\\Gamma$)={trace:.3g} s$^{{-2}}$",
        y=0.99,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "radial_gg_components_signed.svg", bbox_inches="tight")
    plt.close(fig)


def fig_contribution_and_cumulative() -> None:
    from gragra.plotting import plot_contribution_hist, plot_cumulative_contribution

    setup, _, _, _, _ = _rich_volume_setup()
    fig1, _ = plot_contribution_hist(setup, "x", target_id="A", bins=30)
    fig1.savefig(OUT_DIR / "contribution_hist_delta_a_x.svg", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig1)
    fig2, _ = plot_cumulative_contribution(setup, target_id="A", bins=RADIAL_BINS_M)
    fig2.savefig(OUT_DIR / "cumulative_norm.svg", bbox_inches="tight")
    plt.close(fig2)


def fig_source_field_mass_over_r2_xz_sum() -> None:
    import matplotlib.pyplot as plt
    from astropy import units as u
    from matplotlib.colors import LogNorm

    from gragra.geometry import Frame
    from gragra.gravity_sources import PointMassCloud, point_sum
    from gragra.plotting import source_field
    from gragra.results import Contribution, GravityFieldDefinition, PotentialReference
    from gragra.targets import TargetPoints

    rng = np.random.default_rng(9)
    n = 420
    positions = rng.uniform([-14.0, -9.0, -5.0], [14.0, 9.0, 6.0], size=(n, 3))
    masses = 1.5 + 6.0 * (0.5 + 0.5 * np.sin(positions[:, 0] / 5.0)) * (
        0.5 + 0.5 * np.cos(positions[:, 2] / 3.5)
    )
    root = Frame("sf_root")
    field = GravityFieldDefinition.source_field(
        "source", Contribution("total"), PotentialReference.ZERO_AT_INFINITY
    )
    source = PointMassCloud(masses.astype(np.float64), positions.astype(np.float64))
    targets = TargetPoints([[0.0, 0.0, 0.0]] * u.m, ("T",), root)
    setup = point_sum(source, frame=root, field_definition=field).at(targets)
    x_edges = np.linspace(-16.0, 16.0, 33)
    y_edges = np.linspace(-10.0, 10.0, 21)
    z_edges = np.linspace(-6.0, 7.0, 27)
    scalar_field = source_field(
        setup,
        "mass_over_r2",
        target_id="T",
        grid=(x_edges, y_edges, z_edges),
        aggregation="sum",
        spatial_projection="xz",
    )
    arr = np.asarray(scalar_field)
    xz = np.asarray(arr[0, :, 0, :], dtype=np.float64)
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    xz_masked = np.ma.masked_equal(xz, 0.0)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f0f0f0")
    norm = LogNorm(vmin=1e-2, vmax=10.0)
    im = ax.pcolormesh(xc, zc, xz_masked.T, shading="auto", cmap=cmap, norm=norm)
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$z$ [m]")
    ax.set_title(
        rf"Y-integrated $\sum \Delta m/r^2$ ({n} point masses)"
        + "\n(log scale; zero cells masked)",
        fontsize=11,
    )
    ax.set_aspect("equal")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\sum \Delta m/r^2$ [kg/m$^2$]")
    fig.savefig(OUT_DIR / "source_field_mass_over_r2_xz_sum.svg", bbox_inches="tight")
    plt.close(fig)


def _weighted_mean_direction_interference_ratios(
    results: tuple,
    direction_weights: list[float],
    *,
    target_id: str,
    frequency_hz: float,
    component: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Per-group weighted mean of single-direction interference_share ratios."""
    from gragra._plotting._dynamic_contribution import (
        _direction_weights,
        contribution_statistics,
    )

    weights = np.asarray(
        _direction_weights(direction_weights, len(results)), dtype=np.float64
    )
    reference = results[0]
    data = reference._data
    frequency_index = int(np.flatnonzero(data.frequencies_hz == frequency_hz)[0])
    target_index = data.target_ids.index(target_id)
    component_index = reference.component_labels.index(component)
    group_ids = tuple(data.group_ids)
    values = np.zeros(len(group_ids), dtype=np.float64)
    valid = np.zeros(len(group_ids), dtype=bool)
    for group_index, group_id in enumerate(group_ids):
        numerator = 0.0
        weight_sum = 0.0
        for weight, result in zip(weights, results, strict=True):
            stats = contribution_statistics(result)
            sample = stats.interference_share.to_array(
                group_id=group_id, layout="sample_first"
            )
            value = float(
                np.real(sample[frequency_index, target_index, component_index])
            )
            is_valid = bool(
                stats.interference_share.validity(group_id).to_array(
                    layout="sample_first"
                )[frequency_index, target_index, component_index]
            )
            if is_valid:
                numerator += float(weight) * value
                weight_sum += float(weight)
        if weight_sum > 0.0:
            values[group_index] = numerator / weight_sum
            valid[group_index] = True
    return values, valid, group_ids


def fig_dynamic_interference() -> None:
    """Compare direction-averaged I_j with weighted mean of per-direction ratios."""
    import matplotlib.pyplot as plt

    from gragra.fields.density_series import DensityPhasors
    from gragra.plotting import dynamic_radial_histogram, source_contributions

    positions = np.array(
        [[-3.0, 0.0, 0.0], [-7.0, 0.0, 0.0], [-14.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    volumes = np.ones(3)
    frequencies = np.array([12.0, 24.0])
    density_a = np.array(
        [
            [1.0 + 0.0j, -1.0 + 0.0j, 0.25 + 0.0j],
            [0.5 + 0.5j, 0.5 + 0.5j, 0.25 + 0.25j],
        ],
        dtype=np.complex128,
    )
    density_b = np.array(
        [
            [0.25 + 0.0j, 1.0 + 0.0j, 0.5 + 0.5j],
            [-0.5 + 0.5j, 0.25 + 0.0j, 1.0 + 0.0j],
        ],
        dtype=np.complex128,
    )
    phas_a = DensityPhasors(
        positions,
        volumes,
        density_a,
        frequencies,
        0.0,
        np.zeros(3),
        "root",
        np.zeros(3),
        ("s0", "s1", "s2"),
    )
    phas_b = DensityPhasors(
        positions,
        volumes,
        density_b,
        frequencies,
        0.0,
        np.zeros(3),
        "root",
        np.zeros(3),
        ("s0", "s1", "s2"),
    )
    target = np.array([[0.0, 0.0, 0.0]])
    edges = [0.0, 5.0, 10.0, 20.0]
    r_a = source_contributions(
        phas_a, "acceleration", target, radial_edges_m=edges, softening_m=0.0
    )
    r_b = source_contributions(
        phas_b, "acceleration", target, radial_edges_m=edges, softening_m=0.0
    )
    direction_weights = [1.0, 2.0]
    results = (r_a, r_b)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), sharey=True)
    for ax, f_hz in zip(axes, (12.0, 24.0), strict=True):
        dyn = dynamic_radial_histogram(
            results,
            "interference_share",
            target_id="target-0",
            frequency_hz=f_hz,
            component="x",
            direction_weights=direction_weights,
        )
        h = dyn.histogram
        i_dir = np.where(dyn.valid, h.values.value, np.nan)

        wmean_all, wmean_valid, group_ids = (
            _weighted_mean_direction_interference_ratios(
                results,
                direction_weights,
                target_id="target-0",
                frequency_hz=f_hz,
                component="x",
            )
        )
        finite_valid = np.asarray(
            [wmean_valid[group_ids.index(gid)] for gid in group_ids[1:-1]], dtype=bool
        )
        wmean = np.where(finite_valid, wmean_all[1:-1], np.nan)

        both = np.isfinite(i_dir) & finite_valid & np.isfinite(wmean)
        if both.any():
            max_abs_diff = float(np.max(np.abs(i_dir[both] - wmean[both])))
        else:
            max_abs_diff = float("nan")
        if not (np.isfinite(max_abs_diff) and max_abs_diff > 1.0e-12):
            raise RuntimeError(
                f"expected direction-averaged I_j and weighted per-direction mean "
                f"to differ at {f_hz:g} Hz; max |Δ|={max_abs_diff}"
            )

        ax.stairs(
            i_dir,
            h.edges.value,
            label=r"Direction-averaged $I_j(f)$",
            linewidth=1.8,
            color="#1f77b4",
        )
        ax.stairs(
            wmean,
            h.edges.value,
            linestyle="--",
            label=r"Weighted mean of per-direction ratios",
            linewidth=1.8,
            color="#ff7f0e",
        )
        ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle=":")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_xlabel(f"distance [{h.xunit.to_string()}]", fontsize=10.5)
        ax.set_title(
            f"{f_hz:g} Hz (max $|I_j-\\bar r|$ = {max_abs_diff:.3g})",
            fontsize=11,
        )
        ax.legend(loc="upper right", fontsize=9.5)
        ax.tick_params(labelsize=9.5)
    axes[0].set_ylabel("interference share (x) [dimensionless]", fontsize=10.5)
    fig.suptitle(
        "Synthetic peak complex densities; invalid bins masked (zero total power)",
        y=1.01,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dynamic_interference_share_x.svg", bbox_inches="tight")
    plt.close(fig)


def fig_plot_result_static_and_series() -> None:
    import astropy.units as u

    from gragra.geometry import Frame
    from gragra.gravity_sources import PointMassCloud, point_sum, rigid_motion
    from gragra.gravity_sources.trajectory import PrescribedRigidTransform
    from gragra.plotting import plot_result
    from gragra.results import (
        Contribution,
        GravityFieldDefinition,
        PotentialReference,
        TimeSampling,
    )
    from gragra.targets import TargetPoints

    root = Frame("root")
    field = GravityFieldDefinition.source_field(
        "source", Contribution("total"), PotentialReference.ZERO_AT_INFINITY
    )
    source = PointMassCloud(np.array([800.0]), np.array([[1.5, -1.0, 3.0]]))
    xs = np.linspace(-20.0, 20.0, 41)
    ys = np.linspace(-20.0, 20.0, 41)
    gx, gy = np.meshgrid(xs, ys)
    grid_coords = np.column_stack(
        [gx.ravel(), gy.ravel(), np.zeros(gx.size, dtype=np.float64)]
    )
    targets = TargetPoints(
        grid_coords * u.m,
        tuple(f"g{i}" for i in range(len(grid_coords))),
        root,
    )
    eval_result = (
        point_sum(source, frame=root, field_definition=field).at(targets).run()
    )
    fig_field, _ = plot_result(
        eval_result, observable="acceleration", component="norm", plane="xy"
    )

    frames = 200
    epoch_gps = (1_000_000_000, 0)
    sampling = TimeSampling(epoch=epoch_gps, dt=0.05 * u.s, count=frames)
    epoch = sampling.epoch
    t0_s = float(epoch.gpsSeconds) + float(epoch.gpsNanoSeconds) * 1e-9
    time_s = t0_s + sampling.dt_seconds * np.arange(frames, dtype=np.float64)
    pos = np.column_stack(
        [
            0.4 * np.sin(2.0 * np.pi * 0.8 * (time_s - time_s[0])),
            0.25 * (time_s - time_s[0]),
            np.zeros(frames),
        ]
    )
    transform = PrescribedRigidTransform(
        time_s, pos, np.repeat(np.eye(3)[np.newaxis], frames, axis=0)
    )
    ts_targets = TargetPoints([[-3.0, -4.0, 0.0]] * u.m, ("A",), root)
    ts_result = (
        rigid_motion(
            PointMassCloud(np.array([5.0]), np.array([[0.0, 0.0, 0.0]])),
            transform,
            frame=root,
            sampling=sampling,
        )
        .at(ts_targets, observables=("acceleration",))
        .run()
    )
    fig_ts, _ = plot_result(ts_result, target_id="A", view="line", component="x")

    fig_field.savefig(OUT_DIR / "plot_result_static_field_xy.svg", bbox_inches="tight")
    fig_ts.savefig(OUT_DIR / "plot_result_time_series.svg", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig_field)
    plt.close(fig_ts)


def fig_plot_result_spectrum_and_phase() -> None:
    import astropy.units as u
    import matplotlib.pyplot as plt
    from gwpy.time import LIGOTimeGPS
    from scipy.spatial.transform import Rotation

    from gragra._continuous_contract import normalize_relative_times
    from gragra.geometry import Frame
    from gragra.gravity_sources import PointMassCloud, rigid_motion
    from gragra.plotting import plot_result
    from gragra.targets import TargetPoints

    world = Frame("world")
    epoch = LIGOTimeGPS(1_000_000_000, 0)
    cloud = PointMassCloud(np.array([3.0]), np.array([[0.0, 0.0, 0.0]]))

    def circular_pose(times: u.Quantity):
        t = normalize_relative_times(times).to_value(u.s)
        angle = 2.0 * np.pi * t
        translation = np.stack(
            [np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=1
        )
        return Rotation.identity(len(t)), translation * u.m

    targets = TargetPoints(np.array([[0.0, 0.0, 0.0]]) * u.m, ("origin",), world)
    atol_qty = 1e-9 * 6.67430e-11 * (u.m / u.s**2)
    result = (
        rigid_motion(cloud, circular_pose, frame=world, epoch=epoch)
        .at(targets, observables=("acceleration",))
        .fourier_series(
            period=1.0 * u.s,
            orders=[0, 1, 2, 3],
            tref=epoch,
            tau_min=0.125 * u.s,
            atol=atol_qty,
            rtol=1e-10,
        )
    )
    fig_mag, _ = plot_result(
        result,
        observable="acceleration",
        component="x",
        view="spectrum",
        part="magnitude",
    )
    fig_phase, ax_phase = plot_result(
        result, observable="acceleration", component="x", view="spectrum", part="phase"
    )

    # R02: Mask phase values where amplitude is zero / below numerical solver tolerance.
    # The pure harmonic circular motion has physical acceleration amplitude only at 1 Hz
    # (~2.00e-10 m/s^2); other orders (0, 2, 3 Hz) are numerical noise (~1e-26 m/s^2),
    # well below the solver atol (~6.67e-20 m/s^2). A mathematical phase is undefined
    # for zero amplitude, so we mask these sub-tolerance residuals with NaN.
    matrix = np.asarray(result.acceleration.value[0, 0, :])
    magnitudes = np.abs(matrix)
    atol_threshold = 1e-15  # well above solver residual (~1e-26) and atol (~6.7e-20)
    line = ax_phase.lines[0]
    y_data = np.where(magnitudes >= atol_threshold, line.get_ydata(), np.nan)
    line.set_ydata(y_data)
    ax_phase.set_title(
        ax_phase.get_title() + "\n(zero amplitude masked)",
        fontsize=9.5,
    )
    ax_phase.set_ylim(-3.5, 3.5)

    fig_mag.savefig(OUT_DIR / "plot_result_spectrum_magnitude.svg", bbox_inches="tight")
    fig_phase.savefig(OUT_DIR / "plot_result_spectrum_phase.svg", bbox_inches="tight")
    plt.close(fig_mag)
    plt.close(fig_phase)


def fig_plot_shape_cuboid() -> None:
    import astropy.units as u

    from gragra.geometry import Cuboid, Frame
    from gragra.plotting import plot_shape

    world = Frame("world")
    cuboid = Cuboid(
        center=[0.0, 0.0, 0.0] * u.m,
        lengths=[6.0, 4.0, 2.0] * u.m,
        frame=world,
    )
    fig, _ = plot_shape(cuboid, color="skyblue", alpha=0.55)
    fig.savefig(OUT_DIR / "plot_shape_cuboid.svg", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def fig_hist_density_volume() -> None:
    from gragra.plotting import plot_hist

    setup, *_ = _tutorial_volume_setup()
    fig = plot_hist(
        setup,
        "density",
        weight="volume",
        bins=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        xunit="g/cm3",
    )
    fig.savefig(OUT_DIR / "hist_density_volume.svg", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _setup_mpl()
    fig_map_xz_kernel()
    fig_map_xz_delta_a_x()
    fig_radial_mass_over_r2()
    fig_gg_components()
    fig_contribution_and_cumulative()
    fig_source_field_mass_over_r2_xz_sum()
    fig_dynamic_interference()
    fig_plot_result_static_and_series()
    fig_plot_result_spectrum_and_phase()
    fig_plot_shape_cuboid()
    fig_hist_density_volume()
    print(f"Wrote figures under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
