"""Acceptance tests for result field and series visualization (plot_result).

Verifies plot_result across:
1. Static field evaluations on 2D grids (uniform_sphere, uniform_cuboid).
2. Time series representations (sampled rigid_motion).
3. Periodic Fourier series (circular motion closed form comparison with tolerance).
4. Finite-window Fourier transforms (constant pose vs sinc closed form).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeriesMatrix
from gwexpy.timeseries import TimeSeriesMatrix
from gwpy.frequencyseries import FrequencySeries
from gwpy.time import LIGOTimeGPS
from gwpy.timeseries import TimeSeries
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.spatial.transform import Rotation

from gragra._continuous_contract import normalize_relative_times
from gragra._plotting._result_field import plot_result
from gragra.analytic.objects import uniform_sphere
from gragra.geometry import Frame, Sphere
from gragra.gravity_sources import PointMassCloud, rigid_motion
from gragra.gravity_sources.trajectory import PrescribedRigidTransform
from gragra.results import FiniteRecordInterpretation, TimeSampling
from gragra.results._representation import TimeSeriesRepresentation
from gragra.results._spectral import AmplitudeSpectrumRepresentation
from gragra.results._time_grid import build_uniform_time_s
from gragra.targets import TargetPoints
from tests.high_level._plot_extract import extract_image_data, extract_line_data

G_SI = float(const.G.value)
WORLD = Frame("world")
EPOCH = LIGOTimeGPS(1_000_000_000, 0)
CLOUD = PointMassCloud(np.array([1.0]), np.array([[0.0, 0.0, 0.0]]))


def circular_pose():
    """Circular translation with radius 1 m and period 1 s, identity rotation."""

    def pose(times: u.Quantity) -> tuple[Rotation, u.Quantity]:
        t = normalize_relative_times(times).to_value(u.s)
        angle = 2.0 * np.pi * t
        translation = np.stack(
            [np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=1
        )
        return Rotation.identity(len(t)), translation * u.m

    return pose


def _constant_pose(pos=(0.0, 2.0, 0.0)):
    # Constant pose at pos
    def pose(times: u.Quantity) -> tuple[Rotation, u.Quantity]:
        t = normalize_relative_times(times).to_value(u.s)
        translation = np.repeat(np.array([pos], dtype=np.float64), len(t), axis=0)
        return Rotation.identity(len(t)), translation * u.m

    return pose


def test_plot_result_static_grid_field() -> None:
    """plot_result for a 2D grid static result matches direct evaluation."""
    geom = Sphere(center=np.zeros(3) * u.m, radius=1.0 * u.m, frame=WORLD)
    density = 2500.0 * u.kg / u.m**3
    model = uniform_sphere(geom, density)

    # 5x5 grid in xy plane at z = 2m
    xs = np.linspace(-2.0, 2.0, 5)
    ys = np.linspace(-2.0, 2.0, 5)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_coords = np.column_stack(
        [grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, 2.0)]
    )

    targets = TargetPoints(
        grid_coords * u.m,
        ids=tuple(f"pt_{i}" for i in range(len(grid_coords))),
        frame=WORLD,
    )
    result = model.at(targets).run()

    # Plot acceleration norm
    fig, ax = plot_result(
        result, observable="acceleration", component="norm", view="xy"
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    # Verify that plotted array matches direct evaluation
    acc_vals = np.asarray(result.acceleration)
    expected_norm = np.linalg.norm(acc_vals, axis=1).reshape((5, 5))
    plotted_data = extract_image_data(ax)
    np.testing.assert_allclose(plotted_data, expected_norm, rtol=1e-12)
    plt.close(fig)


def test_plot_result_time_series() -> None:
    """plot_result for a time-series result matches direct evaluation."""
    frames = 10
    time_s = build_uniform_time_s((1_000_000_000, 0), 0.1, frames)
    # Moving along x axis: x(t) = 0.5 * t
    pos = np.column_stack(
        [0.5 * (time_s - 1_000_000_000), np.zeros(frames), np.zeros(frames)]
    )
    transform = PrescribedRigidTransform(
        time_s,
        pos,
        np.repeat(np.eye(3)[np.newaxis], frames, axis=0),
    )
    sampling = TimeSampling(epoch=EPOCH, dt=0.1 * u.s, count=frames)
    model = rigid_motion(CLOUD, transform, frame=WORLD, sampling=sampling)

    target = TargetPoints(
        np.array([[0.0, 3.0, 0.0]]) * u.m, ids=("T0",), frame=WORLD
    )
    result = model.at(target, observables=("acceleration",)).run()

    fig, ax = plot_result(result, observable="acceleration", component="y")
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    t_data, a_data = extract_line_data(ax)
    expected_acc = np.asarray(result.acceleration)[0, 1]  # target 0, y component
    np.testing.assert_allclose(a_data, expected_acc, rtol=1e-12)
    assert "Time" in ax.get_xlabel()
    assert "m/s^2" in ax.get_ylabel()
    plt.close(fig)


def test_plot_result_periodic_fourier_series() -> None:
    """plot_result for Fourier series matches circular motion closed form within tol."""
    targets = TargetPoints(
        np.array([[0.0, 0.0, 0.0]]) * u.m, ids=("origin",), frame=WORLD
    )
    result = (
        rigid_motion(CLOUD, circular_pose(), frame=WORLD, epoch=EPOCH)
        .at(targets, observables=("acceleration",))
        .fourier_series(
            period=1.0 * u.s,
            orders=[0, 1, 2, 3],
            tref=EPOCH,
            tau_min=0.125 * u.s,
            atol=1e-9 * G_SI * (u.m / u.s**2),
            rtol=1e-10,
        )
    )

    fig, ax = plot_result(
        result, observable="acceleration", component="x", view="spectrum"
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    freqs, amps = extract_line_data(ax)
    tol = 1e-6 * G_SI
    # Order 1 (f = 1 Hz) amplitude must match G_SI within numerical tolerance
    idx_1 = int(np.where(np.isclose(freqs, 1.0))[0][0])
    assert abs(amps[idx_1] - G_SI) <= tol

    # Orders 0, 2, 3 must vanish within tolerance
    for order in (0, 2, 3):
        idx_order = int(np.where(np.isclose(freqs, float(order)))[0][0])
        assert abs(amps[idx_order]) <= tol

    plt.close(fig)


def test_plot_result_fourier_transform_sinc_closed_form() -> None:
    """plot_result for Fourier transform of constant pose matches sinc closed form."""
    targets = TargetPoints(
        np.array([[0.0, 0.0, 0.0]]) * u.m, ids=("origin",), frame=WORLD
    )
    # Source at (0, 2, 0) -> distance r = 2m, direction +y
    # a_y = G * m / r^2 = G / 4
    # Symmetric window [-1, 1] s, T = 2s, tref = 0
    # Continuous Fourier transform: a_y * T * sinc(f * T)
    a0 = G_SI / 4.0
    t_span = 2.0
    t_start, t_stop = -1.0 * u.s, 1.0 * u.s
    freqs_test = np.array([0.0, 0.25, 0.5, 1.0, 2.0]) * u.Hz

    result = (
        rigid_motion(CLOUD, _constant_pose((0.0, 2.0, 0.0)), frame=WORLD, epoch=EPOCH)
        .at(targets, observables=("acceleration",))
        .fourier_transform(
            t_start=t_start,
            t_stop=t_stop,
            frequencies=freqs_test,
            tref=EPOCH,
            tau_min=0.25 * u.s,
            atol=1e-9 * G_SI * (u.m / u.s),
            rtol=1e-10,
        )
    )

    fig, ax = plot_result(
        result, observable="acceleration", component="y", view="spectrum"
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    f_plotted, ft_plotted = extract_line_data(ax)
    # Expected sinc: a0 * T * sinc(pi * f * T) = a0 * T * np.sinc(f * T)
    expected_sinc = a0 * t_span * np.sinc(freqs_test.value * t_span)
    tol = 1e-6 * G_SI
    np.testing.assert_allclose(ft_plotted, expected_sinc, atol=tol)

    assert "Frequency" in ax.get_xlabel()
    assert "(m / s2) s" in ax.get_ylabel() or "(m/s^2) s" in ax.get_ylabel()
    plt.close(fig)


def test_plot_result_fourier_series_complex_parts() -> None:
    """Fourier series correctly plots magnitude, real, imag, and phase components."""
    targets = TargetPoints(
        np.array([[0.0, 0.0, 0.0]]) * u.m, ids=("origin",), frame=WORLD
    )
    result = (
        rigid_motion(CLOUD, circular_pose(), frame=WORLD, epoch=EPOCH)
        .at(targets, observables=("acceleration",))
        .fourier_series(
            period=1.0 * u.s,
            orders=[0, 1],
            tref=EPOCH,
            tau_min=0.125 * u.s,
            atol=1e-9 * G_SI * (u.m / u.s**2),
            rtol=1e-10,
        )
    )

    # 1. Component y with magnitude: |-i G| = G (must be G, not 0!)
    fig_mag, ax_mag = plot_result(
        result, observable="acceleration", component="y", part="magnitude"
    )
    freqs, amp_mag = extract_line_data(ax_mag)
    np.testing.assert_allclose(amp_mag[1], G_SI, rtol=1e-5)
    plt.close(fig_mag)

    # 2. Component y with imag: Im(-i G) = -G
    fig_im, ax_im = plot_result(
        result, observable="acceleration", component="y", part="imag"
    )
    _, amp_im = extract_line_data(ax_im)
    np.testing.assert_allclose(amp_im[1], -G_SI, rtol=1e-5)
    assert "Im(acceleration)" in ax_im.get_ylabel()
    plt.close(fig_im)

    # 3. Component y with real: Re(-i G) = 0
    fig_re, ax_re = plot_result(
        result, observable="acceleration", component="y", part="real"
    )
    _, amp_re = extract_line_data(ax_re)
    assert abs(amp_re[1]) <= 1e-6 * G_SI
    plt.close(fig_re)

    # 4. Component y with phase: angle(-i G) = -pi / 2
    fig_ph, ax_ph = plot_result(
        result, observable="acceleration", component="y", part="phase"
    )
    _, amp_ph = extract_line_data(ax_ph)
    np.testing.assert_allclose(amp_ph[1], -np.pi / 2.0, atol=1e-4)
    assert "Phase" in ax_ph.get_ylabel()
    plt.close(fig_ph)

    # 5. Component norm with magnitude:
    # sqrt(Ax^2 + Ay^2) = sqrt(G^2 + G^2) = sqrt(2) * G
    fig_norm, ax_norm = plot_result(
        result, observable="acceleration", component="norm", part="magnitude"
    )
    _, amp_norm = extract_line_data(ax_norm)
    np.testing.assert_allclose(amp_norm[1], np.sqrt(2.0) * G_SI, rtol=1e-5)
    plt.close(fig_norm)


def test_plot_result_static_grid_order_invariance_and_duplicates() -> None:
    """Coordinate mapping is invariant to target input order and rejects duplicates."""
    sphere = Sphere(radius=1.0 * u.m, center=[0, 0, 0] * u.m, frame=WORLD)
    model = uniform_sphere(sphere, density=1000.0 * u.kg / u.m**3)

    # 1. Regular 3x3 grid in xy
    x = np.array([-2.0, 0.0, 2.0])
    y = np.array([-2.0, 0.0, 2.0])
    xx, yy = np.meshgrid(x, y)
    coords_ordered = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(9)]) * u.m
    targets_ordered = TargetPoints(
        coords_ordered, ids=tuple(f"T{i}" for i in range(9)), frame=WORLD
    )
    res_ordered = model.at(targets_ordered).run()

    fig_ord, ax_ord = plot_result(res_ordered, plane="xy")
    im_ordered = extract_image_data(ax_ord)

    # Pixel edges: [-3, 3, -3, 3] so pixel centers are (-2, 0, 2)
    np.testing.assert_allclose(
        ax_ord.images[0].get_extent(), [-3.0, 3.0, -3.0, 3.0]
    )
    plt.close(fig_ord)

    # 2. Shuffle target points arbitrarily
    perm = np.array([7, 2, 0, 5, 1, 8, 3, 6, 4])
    coords_shuffled = coords_ordered[perm]
    targets_shuffled = TargetPoints(
        coords_shuffled, ids=tuple(f"T{perm[i]}" for i in range(9)), frame=WORLD
    )
    res_shuffled = model.at(targets_shuffled).run()

    fig_shuf, ax_shuf = plot_result(res_shuffled, plane="xy")
    im_shuffled = extract_image_data(ax_shuf)
    np.testing.assert_allclose(
        ax_shuf.images[0].get_extent(), [-3.0, 3.0, -3.0, 3.0]
    )
    plt.close(fig_shuf)

    # Coordinate-based placement ensures pixel-for-pixel identity
    np.testing.assert_allclose(im_ordered, im_shuffled, rtol=1e-12)

    # 3. Duplicate coordinates should raise ValueError
    coords_dup = np.vstack([coords_ordered.value, coords_ordered.value[:1]]) * u.m
    targets_dup = TargetPoints(
        coords_dup, ids=tuple(f"T{i}" for i in range(10)), frame=WORLD
    )
    res_dup = model.at(targets_dup).run()
    import pytest

    with pytest.raises(ValueError, match="duplicate coordinates"):
        plot_result(res_dup, plane="xy")

    # 4. Orthogonal axis variation should raise ValueError
    coords_ortho = coords_ordered.value.copy()
    coords_ortho[0, 2] = 5.0  # z = 5 for one point, 0 for others
    targets_ortho = TargetPoints(
        coords_ortho * u.m, ids=tuple(f"T{i}" for i in range(9)), frame=WORLD
    )
    res_ortho = model.at(targets_ortho).run()
    with pytest.raises(ValueError, match="orthogonal z axis"):
        plot_result(res_ortho, plane="xy")

    # 5. Non-uniform grid uses pcolormesh
    x_nu = np.array([-4.0, 0.0, 1.0])
    y_nu = np.array([-2.0, 2.0])
    xx_nu, yy_nu = np.meshgrid(x_nu, y_nu)
    coords_nu = np.column_stack([xx_nu.ravel(), yy_nu.ravel(), np.zeros(6)]) * u.m
    targets_nu = TargetPoints(
        coords_nu, ids=tuple(f"N{i}" for i in range(6)), frame=WORLD
    )
    res_nu = model.at(targets_nu).run()
    fig_nu, ax_nu = plot_result(res_nu, plane="xy")
    assert len(ax_nu.collections) > 0  # QuadMesh from pcolormesh
    plt.close(fig_nu)

    # 6. Missing grid points falls back to scatter (M-4)
    # L-shape: 3 points from a 2x2 grid
    coords_l = np.array([[-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0]]) * u.m
    targets_l = TargetPoints(coords_l, ids=("L1", "L2", "L3"), frame=WORLD)
    res_l = model.at(targets_l).run()
    fig_l, ax_l = plot_result(res_l, plane="xy")
    assert "scatter" in ax_l.get_title()
    plt.close(fig_l)


def test_plot_result_potential_and_gradient_observables() -> None:
    """plot_result handles potential and tensor gradients with correct SI units."""
    sphere = Sphere(radius=1.0 * u.m, center=[0, 0, 0] * u.m, frame=WORLD)
    model = uniform_sphere(sphere, density=1000.0 * u.kg / u.m**3)

    x = np.array([-3.0, 3.0])
    y = np.array([-3.0, 3.0])
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(4)]) * u.m
    targets = TargetPoints(coords, ids=("A", "B", "C", "D"), frame=WORLD)

    res = model.at(
        targets, observables=("potential", "acceleration", "gradient")
    ).run()

    # Potential: unit m^2/s^2
    fig_phi, ax_phi = plot_result(res, observable="potential", plane="xy")
    # Verify colorbar label contains m^2/s^2
    cb_phi = fig_phi.axes[-1]
    assert "m^2/s^2" in cb_phi.get_ylabel()
    plt.close(fig_phi)

    # Gradient: xx component, unit 1/s^2
    fig_gxx, ax_gxx = plot_result(
        res, observable="gradient", component="xx", plane="xy"
    )
    cb_gxx = fig_gxx.axes[-1]
    assert "1/s^2" in cb_gxx.get_ylabel()
    plt.close(fig_gxx)

    # Gradient: norm, unit 1/s^2
    fig_gnorm, ax_gnorm = plot_result(
        res, observable="gradient", component="norm", plane="xy"
    )
    cb_gnorm = fig_gnorm.axes[-1]
    assert "1/s^2" in cb_gnorm.get_ylabel()
    plt.close(fig_gnorm)


def test_plot_result_non_si_unit_conversion() -> None:
    """plot_result scales non-SI quantities to SI units matching axis label."""
    sphere = Sphere(radius=1.0 * u.m, center=[0, 0, 0] * u.m, frame=WORLD)
    model = uniform_sphere(sphere, density=1000.0 * u.kg / u.m**3)

    coords = (
        np.array(
            [
                [-3.0, -3.0, 0.0],
                [-3.0, 3.0, 0.0],
                [3.0, -3.0, 0.0],
                [3.0, 3.0, 0.0],
            ]
        )
        * u.m
    )
    targets = TargetPoints(coords, ids=("A", "B", "C", "D"), frame=WORLD)
    res = model.at(targets).run()

    # Convert acceleration to km / s^2
    acc_si = res.acceleration
    acc_km = acc_si.to(u.km / u.s**2)

    class NonSIResult:
        context = res.context
        acceleration = acc_km

    fig, ax = plot_result(NonSIResult(), plane="xy")
    im_data = extract_image_data(ax)
    expected_si = np.linalg.norm(acc_si.value, axis=1)
    # Plotted data must be in SI (m/s^2), not raw km/s^2 numerical values
    np.testing.assert_allclose(sorted(im_data.ravel()), sorted(expected_si), rtol=1e-10)
    plt.close(fig)


def test_plot_result_transformed_frame_alignment() -> None:
    """plot_result aligns spatial coordinates with result.context.frame."""
    rot_90z = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    frame_trans = Frame(
        "transformed",
        origin=[10.0, 0.0, 0.0] * u.m,
        basis=rot_90z,
        parent=WORLD,
    )

    sphere = Sphere(radius=1.0 * u.m, center=[0, 0, 0] * u.m, frame=WORLD)
    model = uniform_sphere(sphere, density=1000.0 * u.kg / u.m**3)

    coords_world = (
        np.array(
            [
                [-2.0, -2.0, 0.0],
                [-2.0, 2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
            ]
        )
        * u.m
    )
    targets_world = TargetPoints(
        coords_world, ids=("A", "B", "C", "D"), frame=WORLD
    )

    res_trans = model.at(targets_world, output_frame=frame_trans).run()
    assert res_trans.context.frame is frame_trans

    expected_coords = WORLD.transform_points_m(coords_world.value, frame_trans)
    assert np.allclose(expected_coords[:, 2], 0.0)

    fig, ax = plot_result(res_trans, plane="xy")
    extent = ax.images[0].get_extent()
    u_min = float(np.min(expected_coords[:, 0]))
    u_max = float(np.max(expected_coords[:, 0]))
    v_min = float(np.min(expected_coords[:, 1]))
    v_max = float(np.max(expected_coords[:, 1]))
    du = u_max - u_min
    dv = v_max - v_min
    expected_extent = [
        u_min - 0.5 * du,
        u_max + 0.5 * du,
        v_min - 0.5 * dv,
        v_max + 0.5 * dv,
    ]
    np.testing.assert_allclose(extent, expected_extent, rtol=1e-10)
    plt.close(fig)


def test_plot_result_time_series_non_si_and_incompatible_units() -> None:
    """plot_result scales non-SI TimeSeriesMatrix to SI and raises on bad units."""
    dt = 0.1
    ts_x = TimeSeries([1.0, 2.0, 3.0], dt=dt, unit="km / s2")
    ts_y = TimeSeries([0.0, 0.0, 0.0], dt=dt, unit="km / s2")
    ts_z = TimeSeries([0.0, 0.0, 0.0], dt=dt, unit="km / s2")

    mat_km = TimeSeriesMatrix(
        [[ts_x, ts_y, ts_z]],
        rows={"T0": 0},
        cols={"x": 0, "y": 1, "z": 2},
    )

    class MockContext:
        representation = TimeSeriesRepresentation()
        targets = TargetPoints(
            np.array([[0.0, 0.0, 0.0]]) * u.m, ids=("T0",), frame=WORLD
        )

    class MockTSResult:
        context = MockContext()
        acceleration = mat_km

    # 1. Valid non-SI conversion: 1.0, 2.0, 3.0 km/s^2 -> 1000, 2000, 3000 m/s^2
    fig, ax = plot_result(MockTSResult(), observable="acceleration", component="x")
    _, line_data = extract_line_data(ax)
    np.testing.assert_allclose(line_data, [1000.0, 2000.0, 3000.0], rtol=1e-10)
    assert "m/s^2" in ax.get_ylabel()
    plt.close(fig)

    # 2. Incompatible unit raises ValueError
    ts_bad = TimeSeries([1.0, 2.0, 3.0], dt=dt, unit="km")
    mat_bad = TimeSeriesMatrix(
        [[ts_bad, ts_y, ts_z]],
        rows={"T0": 0},
        cols={"x": 0, "y": 1, "z": 2},
    )

    class MockBadResult:
        context = MockContext()
        acceleration = mat_bad

    with pytest.raises(ValueError, match="cannot be converted"):
        plot_result(MockBadResult(), observable="acceleration", component="x")


def test_plot_result_spectral_non_si_and_incompatible_units() -> None:
    """plot_result scales non-SI FrequencySeriesMatrix to SI and raises on bad units."""
    df = 1.0
    fs_x = FrequencySeries([2.0 + 0.0j, 4.0 + 0.0j], df=df, unit="km / s2")
    fs_y = FrequencySeries([0.0 + 0.0j, 0.0 + 0.0j], df=df, unit="km / s2")
    fs_z = FrequencySeries([0.0 + 0.0j, 0.0 + 0.0j], df=df, unit="km / s2")

    mat_km = FrequencySeriesMatrix(
        [[fs_x, fs_y, fs_z]],
        rows={"T0": 0},
        cols={"x": 0, "y": 1, "z": 2},
    )

    class MockContext:
        representation = AmplitudeSpectrumRepresentation(
            record_interpretation=FiniteRecordInterpretation.PERIODIC
        )
        targets = TargetPoints(
            np.array([[0.0, 0.0, 0.0]]) * u.m, ids=("T0",), frame=WORLD
        )
        reference_time = None

    class MockSpectralResult:
        context = MockContext()
        acceleration = mat_km

    # 1. Valid non-SI conversion: 2.0, 4.0 km/s^2 -> 2000, 4000 m/s^2
    fig, ax = plot_result(
        MockSpectralResult(), observable="acceleration", component="x"
    )
    _, line_data = extract_line_data(ax)
    np.testing.assert_allclose(line_data, [2000.0, 4000.0], rtol=1e-10)
    assert "m/s^2" in ax.get_ylabel()
    plt.close(fig)

    # 2. Incompatible unit raises ValueError
    fs_bad = FrequencySeries([2.0 + 0.0j, 4.0 + 0.0j], df=df, unit="s")
    mat_bad = FrequencySeriesMatrix(
        [[fs_bad, fs_y, fs_z]],
        rows={"T0": 0},
        cols={"x": 0, "y": 1, "z": 2},
    )

    class MockBadSpectralResult:
        context = MockContext()
        acceleration = mat_bad

    with pytest.raises(ValueError, match="cannot be converted"):
        plot_result(MockBadSpectralResult(), observable="acceleration", component="x")


def test_plot_result_explicit_scatter_view() -> None:
    """plot_result supports explicit view='scatter'."""
    sphere = Sphere(radius=1.0 * u.m, center=[0, 0, 0] * u.m, frame=WORLD)
    model = uniform_sphere(sphere, density=1000.0 * u.kg / u.m**3)

    coords = (
        np.array(
            [
                [-2.0, -1.0, 0.0],
                [0.0, 1.5, 0.0],
                [2.0, -0.5, 0.0],
            ]
        )
        * u.m
    )
    targets = TargetPoints(coords, ids=("P1", "P2", "P3"), frame=WORLD)
    res = model.at(targets).run()

    fig, ax = plot_result(res, view="scatter", plane="xy")
    assert len(ax.collections) > 0  # PathCollection for scatter
    assert "(scatter, xy-plane)" in ax.get_title()
    assert "x [m]" in ax.get_xlabel()
    assert "y [m]" in ax.get_ylabel()
    plt.close(fig)


def test_plot_result_arbitrary_3d_rotation_and_translation() -> None:
    """plot_result clusters coordinates robustly under arbitrary 3D rotation."""
    # Arbitrary 3D Euler rotation angles and translation
    rot = Rotation.from_euler("xyz", [25.0, -35.0, 45.0], degrees=True).as_matrix()
    origin = np.array([12.3, -4.5, 6.7])
    frame_tilted = Frame(
        "tilted", origin=origin * u.m, basis=rot, parent=WORLD
    )

    sphere = Sphere(radius=1.0 * u.m, center=[0, 0, 0] * u.m, frame=WORLD)
    model = uniform_sphere(sphere, density=1000.0 * u.kg / u.m**3)

    # In frame_tilted, define a 5x7 uniform grid in xy-plane (z = 2.0 m)
    u_vals = np.linspace(-4.0, 4.0, 5)
    v_vals = np.linspace(-3.0, 3.0, 7)
    uu, vv = np.meshgrid(u_vals, v_vals)
    pts_tilted = np.column_stack(
        [uu.ravel(), vv.ravel(), np.full(len(uu.ravel()), 2.0)]
    )

    # Express targets in WORLD coordinates so transform_points_m roundtrips
    pts_world = frame_tilted.transform_points_m(pts_tilted, WORLD)
    targets_world = TargetPoints(
        pts_world * u.m,
        ids=tuple(f"pt_{i}" for i in range(len(pts_world))),
        frame=WORLD,
    )

    # Evaluate with output_frame=frame_tilted
    res = model.at(targets_world, output_frame=frame_tilted).run()

    fig, ax = plot_result(res, plane="xy")
    # Must form a 2D complete grid rendered via imshow despite floating-point roundoff
    assert len(ax.images) > 0
    extent = ax.images[0].get_extent()
    # Expected extent: u in [-4, 4] with du=2, v in [-3, 3] with dv=1
    expected_extent = [-5.0, 5.0, -3.5, 3.5]
    np.testing.assert_allclose(extent, expected_extent, rtol=1e-10)
    plt.close(fig)


def test_plot_result_almost_uniform_grid_branches_to_pcolormesh() -> None:
    """plot_result branches to pcolormesh when grid spacing is almost uniform."""
    sphere = Sphere(radius=1.0 * u.m, center=[0, 0, 0] * u.m, frame=WORLD)
    model = uniform_sphere(sphere, density=1000.0 * u.kg / u.m**3)

    # x spacing: delta1 = 1.0, delta2 = 1.00005 (0.005% non-uniformity)
    x_nu = np.array([0.0, 1.0, 2.00005])
    y_nu = np.array([-1.0, 1.0])
    xx_nu, yy_nu = np.meshgrid(x_nu, y_nu)
    coords_nu = (
        np.column_stack([xx_nu.ravel(), yy_nu.ravel(), np.zeros(6)]) * u.m
    )
    targets_nu = TargetPoints(
        coords_nu, ids=tuple(f"M{i}" for i in range(6)), frame=WORLD
    )
    res_nu = model.at(targets_nu).run()

    fig, ax = plot_result(res_nu, plane="xy")
    # Must use pcolormesh (QuadMesh collection), NOT imshow
    assert len(ax.collections) > 0
    assert len(ax.images) == 0
    plt.close(fig)
