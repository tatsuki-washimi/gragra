#!/usr/bin/env python3
"""Regenerate figures for docs_site/docs/guide/fields.md."""

from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[2] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs_site" / "docs" / "assets" / "guide" / "fields"


def _setup_mpl() -> None:
    import matplotlib

    matplotlib.use("Agg")


def fig_density_slice() -> None:
    import matplotlib.pyplot as plt
    from astropy import units as u

    from gragra.fields.regular_grid import DensityPerturbationGrid
    from gragra.geometry import Frame
    from gragra.plotting import plot_field

    root = Frame("root")
    ground = Frame("ground", np.zeros(3) * u.m, np.eye(3), root)
    x = np.linspace(-40.0, 40.0, 41)
    y = np.linspace(-40.0, 40.0, 41)
    z = np.linspace(-20.0, 5.0, 11)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    # kg/m³ bulk perturbation: Gaussian blob plus mild gradient (non-uniform field).
    density = (
        80.0 * np.exp(-((xx / 18.0) ** 2 + (yy / 22.0) ** 2 + (zz / 8.0) ** 2))
        + 12.0 * (xx / 40.0)
        + 6.0 * np.sin(yy / 7.0)
    )
    grid = DensityPerturbationGrid(x, y, z, density.astype(np.float64))
    fig = plot_field(grid, frame=ground, plane="xy", index=5, unit="m")
    fig.axes[0].set_title(
        r"$\delta\rho$ slice at $z \approx -2.5\,\mathrm{m}$ [kg/m$^3$]"
    )
    fig.savefig(OUT_DIR / "density_perturbation_xy_slice.svg", bbox_inches="tight")
    plt.close(fig)


def fig_displacement_amp_phase() -> None:
    import matplotlib.pyplot as plt
    from astropy import units as u

    from gragra.fields.regular_grid import DisplacementGrid
    from gragra.geometry import Frame
    from gragra.plotting import plot_field

    root = Frame("root")
    ground = Frame("ground", np.zeros(3) * u.m, np.eye(3), root)
    x = np.linspace(-30.0, 30.0, 31)
    y = np.linspace(-30.0, 30.0, 31)
    z = np.linspace(-15.0, 0.0, 6)
    xx, yy, _ = np.meshgrid(x, y, z, indexing="ij")
    phase_rad = np.arctan2(yy, xx + 1e-12)
    amplitude_m = (
        2.0e-6 * (1.0 + 0.6 * np.cos(xx / 12.0)) * np.exp(-(xx**2 + yy**2) / (25.0**2))
    )
    disp = (
        amplitude_m[..., np.newaxis]
        * np.exp(1j * phase_rad[..., np.newaxis])
        * np.array([1.0, 0.35, 0.15])
    )
    disp = np.broadcast_to(disp, (len(x), len(y), len(z), 3)).copy()
    grid = DisplacementGrid(x, y, z, disp.astype(np.complex128))
    fig_amp = plot_field(
        grid, frame=ground, plane="xy", index=2, component="norm", part="abs", unit="m"
    )
    fig_amp.axes[0].set_title(
        r"$|\boldsymbol{\xi}|$ at $z \approx -7.5\,\mathrm{m}$ [m]"
    )
    fig_amp.savefig(OUT_DIR / "displacement_amplitude_xy.svg", bbox_inches="tight")
    plt.close(fig_amp)
    fig_phase = plot_field(
        grid, frame=ground, plane="xy", index=2, component="x", part="phase", unit="m"
    )
    fig_phase.axes[0].set_title(
        r"$\operatorname{Arg}(\xi_x)$ [rad]; zero amplitude masked"
    )
    fig_phase.savefig(OUT_DIR / "displacement_phase_xy.svg", bbox_inches="tight")
    plt.close(fig_phase)


def fig_displacement_quiver() -> None:
    import matplotlib.pyplot as plt
    from astropy import units as u

    from gragra.fields.regular_grid import DisplacementGrid
    from gragra.geometry import Frame
    from gragra.plotting import plot_field

    root = Frame("root")
    ground = Frame("ground", np.zeros(3) * u.m, np.eye(3), root)
    x = np.linspace(-20.0, 20.0, 15)
    y = np.linspace(-20.0, 20.0, 15)
    z = np.linspace(-10.0, 0.0, 5)
    xx, yy, _ = np.meshgrid(x, y, z, indexing="ij")
    disp = np.zeros((len(x), len(y), len(z), 3), dtype=np.float64)
    disp[..., 0] = 3.0e-7 * xx
    disp[..., 1] = 2.0e-7 * np.sin(yy / 5.0)
    disp[..., 2] = 5.0e-8 * np.cos(xx / 8.0)
    grid = DisplacementGrid(x, y, z, disp)
    fig = plot_field(grid, frame=ground, plane="xy", index=2, unit="m")
    fig.axes[0].set_title(
        r"True-scale $\operatorname{Re}(\boldsymbol{\xi})$ quiver "
        r"at $z \approx -5\,\mathrm{m}$"
    )
    fig.savefig(OUT_DIR / "displacement_quiver_xy.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _setup_mpl()
    fig_density_slice()
    fig_displacement_amp_phase()
    fig_displacement_quiver()
    print(f"Wrote figures under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
