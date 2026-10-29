"""Acceptance tests for shape visualization (plot_shape).

Tests plot_shape for Cuboid, Cylinder, and SphericalShell geometry, models,
and setups, ensuring coordinates, frames, and bounds are accurately represented,
and unsupported shapes (Cavity, Mesh, Moving body) are explicitly rejected.
"""

from __future__ import annotations

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from gwexpy.frequencyseries import FrequencySeries
from mpl_toolkits.mplot3d import Axes3D

from gragra._plotting._setup_shapes import plot_shape
from gragra.analytic.objects import uniform_cuboid
from gragra.analytic.seismic import spherical_cavity
from gragra.fields.finite_body import TetraLinearizedSources
from gragra.fields.material_hex8 import Hex8DisplacementField
from gragra.geometry import Cuboid, Cylinder, Frame, Sphere, SphericalShell
from gragra.results import PhasorRepresentation
from gragra.targets import TargetPoints

matplotlib.use("Agg")


@pytest.fixture
def world_frame() -> Frame:
    return Frame("world")


@pytest.fixture
def shifted_rotated_frame(world_frame: Frame) -> Frame:
    # 90 degree rotation around z-axis: x' = y, y' = -x, z' = z
    basis = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    origin = np.array([10.0, 20.0, 30.0]) * u.m
    return Frame("shifted_rotated", origin=origin, basis=basis, parent=world_frame)


@pytest.fixture
def sample_targets(world_frame: Frame) -> TargetPoints:
    pts = np.array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0]]) * u.m
    return TargetPoints(pts, ids=("t0", "t1"), frame=world_frame)


def test_plot_shape_cuboid_geometry_bounds_and_frame(
    shifted_rotated_frame: Frame,
) -> None:
    """Cuboid bounds in world coordinates must match rotated and translated vertices."""
    # Cuboid with lengths (2, 4, 6) centered at (0, 0, 0) in shifted_rotated_frame.
    # In shifted frame: x in [-1, 1], y in [-2, 2], z in [-3, 3]
    # In world frame:
    #   world_pos = origin + basis @ local_pos
    #   basis = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    #   world_x = 10 + local_y -> 10 + [-2, 2] = [8, 12]
    #   world_y = 20 - local_x -> 20 - [-1, 1] = [19, 21]
    #   world_z = 30 + local_z -> 30 + [-3, 3] = [27, 33]
    geom = Cuboid(
        center=np.zeros(3) * u.m,
        lengths=np.array([2.0, 4.0, 6.0]) * u.m,
        frame=shifted_rotated_frame,
    )

    fig, ax = plot_shape(geom)
    assert isinstance(ax, Axes3D)

    # Check bounds on artist collections
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    assert xlim[0] <= 8.0 and xlim[1] >= 12.0
    assert ylim[0] <= 19.0 and ylim[1] >= 21.0
    assert zlim[0] <= 27.0 and zlim[1] >= 33.0
    plt.close(fig)


def test_plot_shape_cylinder_bounds(world_frame: Frame) -> None:
    """Cylinder bounds must match center, radius, and height."""
    # Center (1, 2, 3), radius 2, height 10 (along z) ->
    # x in [-1, 3], y in [0, 4], z in [-2, 8]
    geom = Cylinder(
        center=np.array([1.0, 2.0, 3.0]) * u.m,
        radius=2.0 * u.m,
        height=10.0 * u.m,
        frame=world_frame,
    )

    fig, ax = plot_shape(geom)
    assert isinstance(ax, Axes3D)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    assert xlim[0] <= -1.0 and xlim[1] >= 3.0
    assert ylim[0] <= 0.0 and ylim[1] >= 4.0
    assert zlim[0] <= -2.0 and zlim[1] >= 8.0
    plt.close(fig)


def test_plot_shape_spherical_shell_distinguishes_radii(world_frame: Frame) -> None:
    """SphericalShell must represent both inner and outer radii distinctly."""
    geom = SphericalShell(
        center=np.array([5.0, 5.0, 5.0]) * u.m,
        inner_radius=2.0 * u.m,
        outer_radius=4.0 * u.m,
        frame=world_frame,
    )

    fig, ax = plot_shape(geom)
    assert isinstance(ax, Axes3D)

    # Outer radius spans [5-4, 5+4] = [1, 9]
    xlim = ax.get_xlim()
    assert xlim[0] <= 1.0 and xlim[1] >= 9.0

    # Verify that multiple surfaces (inner and outer shells) are drawn
    collections = ax.collections
    assert len(collections) >= 2, "Expected both inner and outer shell surfaces"
    plt.close(fig)


def test_plot_shape_accepts_model_and_setup(
    world_frame: Frame, sample_targets: TargetPoints
) -> None:
    """plot_shape accepts models and setups via geometry attributes."""
    geom = Cuboid(
        center=np.zeros(3) * u.m,
        lengths=np.array([2.0, 2.0, 2.0]) * u.m,
        frame=world_frame,
    )
    model = uniform_cuboid(geometry=geom, density=2500 * u.kg / u.m**3)
    setup = model.at(sample_targets)

    # 1. Test model
    fig1, ax1 = plot_shape(model)
    assert isinstance(ax1, Axes3D)
    plt.close(fig1)

    # 2. Test setup
    fig2, ax2 = plot_shape(setup)
    assert isinstance(ax2, Axes3D)
    plt.close(fig2)


def test_unsupported_shapes_raise_not_implemented_error(
    world_frame: Frame, sample_targets: TargetPoints
) -> None:
    """Cavity, mesh, and moving body must raise NotImplementedError."""
    # 1. Cavity
    disp = FrequencySeries(
        np.array([1e-5 + 2e-6j, -3e-6 + 4e-6j]),
        frequencies=np.array([10.0, 20.0]) * u.Hz,
        unit=u.m,
    )
    cavity_model = spherical_cavity(
        Sphere(np.zeros(3) * u.m, 5.0 * u.m, world_frame),
        2500 * u.kg / u.m**3,
        3000.0 * u.m / u.s,
        disp,
        PhasorRepresentation(),
        (1_000_000_000, 0),
        "P",
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
        np.zeros(3) * u.m,
    )
    cavity_setup = cavity_model.at(sample_targets)
    with pytest.raises(NotImplementedError) as exc_info:
        plot_shape(cavity_setup)
    assert "SphericalCavity" in str(exc_info.value) or "Cavity" in str(exc_info.value)

    # 2. Tetra mesh dummy/mock or type check
    class DummyTetra(TetraLinearizedSources):
        pass

    with pytest.raises(NotImplementedError) as exc_info:
        plot_shape(DummyTetra)
    assert "Tetra" in str(exc_info.value)

    # 3. Hex8 mesh
    class DummyHex8(Hex8DisplacementField):
        pass

    with pytest.raises(NotImplementedError) as exc_info:
        plot_shape(DummyHex8)
    assert "Hex8" in str(exc_info.value)
