#!/usr/bin/env python3
"""Regenerate figures for docs_site/docs/guide/source-diagnostics.md.

Run from the repository root:

    python docs_site/scripts/generate_source_diagnostics_figures.py

Requires a dev install (``pip install -e ".[dev]"``) so that ``gragra``, GWexpy,
and Matplotlib are available. Uses a non-interactive Matplotlib backend.
"""

from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[2] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs_site" / "docs" / "assets" / "guide" / "source-diagnostics"


def _setup_mpl() -> None:
    import matplotlib

    matplotlib.use("Agg")


def _demo_setup():
    import numpy as np
    from astropy import units as u

    from gragra.geometry import Frame
    from gragra.gravity_sources import VolumeElementSource, point_sum
    from gragra.results import Contribution, GravityFieldDefinition, PotentialReference
    from gragra.targets import TargetPoints

    root = Frame("root")
    field = GravityFieldDefinition.source_field(
        "source", Contribution("total"), PotentialReference.ZERO_AT_INFINITY
    )
    # Four cells at distinct source–target distances to target A so that
    # static radial diagnostics (mass/r², GG, cumulative norm) spread across bins.
    source = VolumeElementSource(
        [[2.0, 0.0, 0.0], [8.0, 0.0, 0.0], [15.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
        [1.0, 2.0, 3.0, 4.0],
        # kg/m³ → ~0.5–3.0 g/cm³ after hist(..., xunit="g/cm3") for the guide snippet.
        np.array([500.0, 1500.0, 1500.0, 2500.0]),
    )
    targets = TargetPoints(
        [[-3.0, -4.0, 0.0], [10.0, 0.0, 0.0]] * u.m, ("A", "B"), root
    )
    setup = point_sum(source, frame=root, field_definition=field).at(targets)
    return setup, source, root, field, targets


def fig_map_xz_kernel() -> None:
    """XZ scatter: per-element inverse-square kernel :math:`1/r^2` (no mass, no G).

    For the mass-weighted field :math:`\\Delta m/r^2` (kg/m²) on a grid, see
    ``source_field(..., "mass_over_r2", ...)`` / ``fig_source_field_mass_over_r2_xz_sum``.
    """
    from gragra.plotting import plot_map

    setup, _, _, _, _ = _demo_setup()
    fig = plot_map(setup, "kernel_norm", target_id="A", view="xz")
    fig.savefig(OUT_DIR / "map_xz_kernel_norm.svg", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def fig_radial_mass_over_r2() -> None:
    """Radial :math:`\\Delta m/r^2` fraction-per-length and step plot."""
    from gragra.plotting import plot_hist, radial_histogram

    setup, _, _, _, _ = _demo_setup()
    radial = radial_histogram(
        setup,
        "mass_over_r2",
        target_id="A",
        bins=[0.0, 6.0, 12.0, 25.0],
        normalization="fraction",
    )
    fig, ax, _ = plot_hist(radial)
    fig.savefig(OUT_DIR / "radial_mass_over_r2_fraction.svg", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def fig_gg_component() -> None:
    """Signed GG tensor component in radial bins."""
    import matplotlib.pyplot as plt

    from gragra.plotting import gravity_gradient_radial_histogram

    setup, _, _, _, _ = _demo_setup()
    gg = gravity_gradient_radial_histogram(
        setup,
        target_id="A",
        bins=[0.0, 6.0, 12.0, 25.0],
        softening_m=0.0,
    )
    h_xx = gg.histogram("xx")
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.stairs(h_xx.values.value, h_xx.edges.value)
    ax.set_xlabel(f"distance [{h_xx.xunit.to_string()}]")
    ax.set_ylabel(f"GG_xx [{h_xx.unit.to_string()}] (signed)")
    ax.set_title("Static gravity-gradient tensor, xx component by distance (target A)")
    uf = float(h_xx.underflow.value)
    of = float(h_xx.overflow.value)
    if uf or of:
        ax.text(
            0.98,
            0.98,
            f"underflow: {uf:.6g} {h_xx.unit}\noverflow: {of:.6g} {h_xx.unit}",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
    fig.savefig(OUT_DIR / "radial_gg_xx_signed.svg", bbox_inches="tight")
    plt.close(fig)


def fig_contribution_and_cumulative() -> None:
    """Per-element norm histogram and cumulative acceleration norm vs distance."""
    from gragra.plotting import plot_contribution_hist, plot_cumulative_contribution

    setup, _, _, _, _ = _demo_setup()
    fig1, _ = plot_contribution_hist(
        setup, "norm", target_id="A", bins=[0.0, 6.0, 12.0, 25.0]
    )
    fig1.savefig(OUT_DIR / "contribution_hist_norm.svg", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig1)

    fig2, _ = plot_cumulative_contribution(
        setup, target_id="A", bins=[0.0, 6.0, 12.0, 25.0]
    )
    fig2.savefig(OUT_DIR / "cumulative_norm.svg", bbox_inches="tight")
    plt.close(fig2)


def fig_source_field_mass_over_r2_xz_sum() -> None:
    """Cell-sum field of :math:`\\sum \\Delta m/r^2` with Y projection (sum over y)."""
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy import units as u

    from gragra.geometry import Frame
    from gragra.gravity_sources import PointMassCloud, point_sum
    from gragra.plotting import source_field
    from gragra.results import Contribution, GravityFieldDefinition, PotentialReference
    from gragra.targets import TargetPoints

    root = Frame("root")
    field = GravityFieldDefinition.source_field(
        "source", Contribution("total"), PotentialReference.ZERO_AT_INFINITY
    )
    masses = np.array([50.0, 120.0, 80.0])
    positions = np.array(
        [
            [-2.0, 0.5, -1.0],
            [3.0, -0.25, 1.0],
            [0.5, 1.0, 4.0],
        ]
    )
    source = PointMassCloud(masses, positions)
    targets = TargetPoints([[0.0, 0.0, 0.0]] * u.m, ("T",), root)
    setup = point_sum(source, frame=root, field_definition=field).at(targets)

    x_edges = np.linspace(-5.0, 5.0, 21)
    y_edges = np.linspace(-3.0, 3.0, 13)
    z_edges = np.linspace(-3.0, 6.0, 19)
    grid = (x_edges, y_edges, z_edges)
    scalar_field = source_field(
        setup,
        "mass_over_r2",
        target_id="T",
        grid=grid,
        aggregation="sum",
        spatial_projection="xz",
    )
    arr = np.asarray(scalar_field)
    # Shape (1, Nx, 1, Nz): collapse over y while keeping full y-range as one cell.
    xz = np.asarray(arr[0, :, 0, :], dtype=np.float64)
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    im = ax.pcolormesh(
        xc,
        zc,
        xz.T,
        shading="auto",
        cmap="viridis",
    )
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$z$ [m]")
    ax.set_title(r"$\sum \Delta m/r^2$ in each XZ cell (summed over $y$)")
    ax.set_aspect("equal")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\sum \Delta m/r^2$ [kg/m$^2$]")
    fig.savefig(OUT_DIR / "source_field_mass_over_r2_xz_sum.svg", bbox_inches="tight")
    plt.close(fig)


def fig_dynamic_interference() -> None:
    """Direction-averaged interference share vs radius (two synthetic directions).

    Sources at 3 m, 7 m, and 14 m fall into distinct finite radial groups for
    edges [0, 5, 10, 20] m. The two ``DensityPhasors`` ensembles use peak complex
    densities (kg/m³) chosen so that the direction-weighted interference ratio
    ``I_j`` (cross terms and total power averaged over directions *before* the
    ratio) differs from a naive average of per-direction interference ratios.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from gragra.fields.density_series import DensityPhasors
    from gragra.plotting import dynamic_radial_histogram, source_contributions

    positions = np.array(
        [
            [-3.0, 0.0, 0.0],
            [-7.0, 0.0, 0.0],
            [-14.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    volumes = np.array([1.0, 1.0, 1.0])
    # Two frequencies; complex peak phasors (kg/m³) differ between two ensembles.
    density_a = np.array(
        [
            [1.0 + 0.0j, 0.5 + 0.5j, 0.25 + 0.0j],
            [0.5 + 0.0j, 0.5 + 0.5j, 0.25 + 0.25j],
        ],
        dtype=np.complex128,
    )
    density_b = np.array(
        [
            [0.25 + 0.0j, 1.0 + 0.0j, 0.5 + 0.5j],
            [0.5 + 0.5j, 0.25 + 0.0j, 1.0 + 0.0j],
        ],
        dtype=np.complex128,
    )
    frequencies = np.array([12.0, 24.0])
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
    dyn = dynamic_radial_histogram(
        (r_a, r_b),
        "interference_share",
        target_id="target-0",
        frequency_hz=12.0,
        component="x",
        direction_weights=[1.0, 2.0],
    )
    h = dyn.histogram
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    mask = dyn.valid
    masked = np.where(mask, h.values.value, np.nan)
    ax.stairs(masked, h.edges.value)
    ax.set_xlabel(f"distance [{h.xunit.to_string()}]")
    ax.set_ylabel(f"{dyn.quantity} (x), direction-averaged [dimensionless]")
    ax.set_title("Interference share by radial group at 12 Hz (two directions)")
    fig.savefig(OUT_DIR / "dynamic_interference_share_x.svg", bbox_inches="tight")
    plt.close(fig)


def fig_plot_result_static_and_series() -> None:
    import astropy.units as u
    import numpy as np

    from gragra.geometry import Frame
    from gragra.gravity_sources import PointMassCloud, rigid_motion
    from gragra.gravity_sources.trajectory import PrescribedRigidTransform
    from gragra.plotting import plot_result
    from gragra.results import TimeSampling
    from gragra.targets import TargetPoints

    root = Frame("root")
    from gragra.gravity_sources import point_sum
    from gragra.results import Contribution, GravityFieldDefinition, PotentialReference

    field = GravityFieldDefinition.source_field(
        "source", Contribution("total"), PotentialReference.ZERO_AT_INFINITY
    )
    source = PointMassCloud(np.array([1000.0]), np.array([[0.0, 0.0, 0.0]]))
    targets = TargetPoints(
        [[-2.0, -2.0, 0.0], [-2.0, 2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0]] * u.m,
        ("1", "2", "3", "4"),
        root,
    )
    eval_result = (
        point_sum(source, frame=root, field_definition=field).at(targets).run()
    )
    fig_field, _ = plot_result(eval_result, plane="xy")

    epoch_gps = (1_000_000_000, 0)
    sampling = TimeSampling(epoch=epoch_gps, dt=0.1 * u.s, count=10)
    epoch = sampling.epoch
    t0_s = float(epoch.gpsSeconds) + float(epoch.gpsNanoSeconds) * 1e-9
    time_s = t0_s + sampling.dt_seconds * np.arange(sampling.count, dtype=np.float64)
    pos = np.column_stack([0.5 * (time_s - time_s[0]), np.zeros(10), np.zeros(10)])
    transform = PrescribedRigidTransform(
        time_s, pos, np.repeat(np.eye(3)[np.newaxis], 10, axis=0)
    )
    ts_targets = TargetPoints(
        [[-3.0, -4.0, 0.0], [10.0, 0.0, 0.0]] * u.m, ("A", "B"), root
    )
    ts_result = (
        rigid_motion(
            PointMassCloud(np.array([1.0]), np.array([[0.0, 0.0, 0.0]])),
            transform,
            frame=root,
            sampling=sampling,
        )
        .at(ts_targets, observables=("acceleration",))
        .run()
    )
    fig_ts, _ = plot_result(ts_result, target_id="A", view="line")

    fig_field.savefig(OUT_DIR / "plot_result_static_field_xy.svg", bbox_inches="tight")
    fig_ts.savefig(OUT_DIR / "plot_result_time_series.svg", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig_field)
    plt.close(fig_ts)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _setup_mpl()

    fig_map_xz_kernel()
    fig_radial_mass_over_r2()
    fig_gg_component()
    fig_contribution_and_cumulative()
    fig_source_field_mass_over_r2_xz_sum()
    fig_dynamic_interference()
    fig_plot_result_static_and_series()
    print(f"Wrote figures under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
