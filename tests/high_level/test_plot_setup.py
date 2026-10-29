"""Artist-level contract tests for the high-level sphere setup plot."""

from __future__ import annotations

import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u
from mpl_toolkits.mplot3d import Axes3D

from gragra._evaluation_setup import (
    DiscreteSource,
    ResolvedEvaluation,
    UniformSphereModelSpec,
)
from gragra.geometry import Appearance
from gragra.geometry._frame import Frame
from gragra.geometry._sphere import Sphere
from gragra.targets._high_points import TargetPoints


def _rx(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _frame_case(
    appearance: Appearance | None = None,
) -> tuple[ResolvedEvaluation, dict[str, np.ndarray]]:
    root = Frame("root")
    basis_lab, basis_model = _rz(0.43), _rx(-0.31)
    basis_targets, basis_output = _rz(-0.27), _rx(0.52)
    origin_lab = np.array([3.0, -2.0, 1.5])
    origin_model = np.array([-0.8, 1.2, 0.4])
    origin_targets = np.array([-4.0, 0.7, 2.1])
    origin_output = np.array([1.1, -3.3, 0.8])
    lab = Frame("lab", origin_lab * u.m, basis_lab, root)
    model_frame = Frame("model", origin_model * u.m, basis_model, lab)
    target_frame = Frame("target frame", origin_targets * u.m, basis_targets, root)
    output_frame = Frame("display frame", origin_output * u.m, basis_output, root)

    sphere_center_local = np.array([0.4, -0.2, 0.3])
    sphere = Sphere(
        sphere_center_local * u.m,
        0.7 * u.m,
        model_frame,
        appearance=appearance,
    )
    spec = UniformSphereModelSpec(sphere, 2.7 * u.g / u.cm**3)
    target_local = np.array([[0.2, 1.0, 4.0], [-1.2, 0.1, 2.8]])
    targets = TargetPoints(target_local * u.m, ("A", "B"), target_frame)

    source_local = np.array(
        [
            [0.1, 0.2, 0.3],
            [100.0, -20.0, 5.0],  # Deliberately omitted from the display subset.
            [-0.3, 0.2, 0.4],
            [0.0, -0.5, 0.1],
            [0.2, -0.1, -0.3],
        ]
    )
    source = DiscreteSource(
        sphere,
        spec.density,
        model_frame,
        source_local,
        np.ones(len(source_local)),
        {"spacing_m": 0.25, "max_points": 1000},
    )
    evaluation = ResolvedEvaluation(
        spec, targets, model_frame, output_frame, source=source
    )

    model_to_root_origin = origin_lab + basis_lab @ origin_model
    model_to_root_basis = basis_lab @ basis_model
    expected = {
        "sphere_center": basis_output.T
        @ (
            model_to_root_origin
            + model_to_root_basis @ sphere_center_local
            - origin_output
        ),
        "source_all": (
            model_to_root_origin + source_local @ model_to_root_basis.T - origin_output
        )
        @ basis_output,
        "targets": (origin_targets + target_local @ basis_targets.T - origin_output)
        @ basis_output,
        "model_basis": basis_output.T @ model_to_root_basis,
        "source_local": source_local,
    }
    return evaluation, expected


def _scatter_values(artist: object, *, three_dimensional: bool) -> np.ndarray:
    if three_dimensional:
        x, y, z = artist._offsets3d  # type: ignore[attr-defined]
        return np.column_stack((x, y, z))
    return np.asarray(artist.get_offsets())  # type: ignore[attr-defined]


def test_plot_uses_resolved_frame_actual_sources_ids_and_requested_units() -> None:
    evaluation, expected = _frame_case()
    source_before = evaluation.source.positions_m.copy()
    targets_before = evaluation.targets.positions_m.copy()
    backend_before = matplotlib.get_backend()

    from gragra._plotting._setup import plot_setup

    figure = plot_setup(evaluation, unit="cm", max_sources=3)
    try:
        axes = figure.axes[0]
        sphere_artist = next(
            artist
            for artist in axes.collections
            if artist.get_label() == "Ideal sphere"
        )
        assert sphere_artist.get_alpha() == pytest.approx(0.28)
        assert isinstance(axes, Axes3D)
        assert axes._focal_length == np.inf
        box_aspect = axes.get_box_aspect()
        axis_spans = np.array(
            [
                np.diff(axes.get_xlim3d())[0],
                np.diff(axes.get_ylim3d())[0],
                np.diff(axes.get_zlim3d())[0],
            ]
        )
        np.testing.assert_allclose(axis_spans, axis_spans[0])
        np.testing.assert_allclose(box_aspect / box_aspect[0], np.ones(3))
        assert "display frame" in axes.get_title()
        assert "cm" in axes.get_title()
        assert any(
            "Model-frame axes" in text.get_text()
            for text in axes.get_legend().get_texts()
        )
        assert axes.get_xlabel().endswith("[cm]")
        assert axes.get_ylabel().endswith("[cm]")
        assert axes.get_zlabel().endswith("[cm]")

        target_artist = next(
            artist for artist in axes.collections if artist.get_gid() == "target-points"
        )
        np.testing.assert_allclose(
            _scatter_values(target_artist, three_dimensional=True),
            expected["targets"] * 100,
        )
        assert {text.get_text() for text in axes.texts} >= {"A", "B"}

        source_artist = next(
            artist for artist in axes.collections if artist.get_gid() == "source-points"
        )
        selected = expected["source_all"][[0, 2, 4]] * 100
        np.testing.assert_allclose(
            _scatter_values(source_artist, three_dimensional=True), selected
        )
        assert "3 / 5" in source_artist.get_label()

        # Limits account for every actual source, including omitted index 1.
        radius_cm = evaluation.model_spec.geometry.radius_m * 100
        sphere_center_cm = expected["sphere_center"] * 100
        all_points = np.vstack(
            (
                expected["source_all"] * 100,
                expected["targets"] * 100,
                sphere_center_cm - radius_cm,
                sphere_center_cm + radius_cm,
            )
        )
        for values, limits in zip(
            all_points.T,
            (axes.get_xlim3d(), axes.get_ylim3d(), axes.get_zlim3d()),
            strict=True,
        ):
            assert limits[0] <= values.min()
            assert limits[1] >= values.max()

        np.testing.assert_array_equal(evaluation.source.positions_m, source_before)
        np.testing.assert_array_equal(evaluation.targets.positions_m, targets_before)
        assert matplotlib.get_backend() == backend_before
    finally:
        plt.close(figure)


def test_plot_appearance_uses_geometry_metadata_and_call_overrides_per_field() -> None:
    evaluation, _ = _frame_case(
        Appearance(color="#123456", alpha=0.4, label="geometry sphere")
    )
    from gragra._plotting._setup import plot_setup

    figure = plot_setup(
        evaluation,
        appearance=Appearance(alpha=0.8, label="call sphere"),
    )
    try:
        axes = figure.axes[0]
        figure.canvas.draw()
        sphere = next(
            artist for artist in axes.collections if artist.get_label() == "call sphere"
        )
        target = next(
            artist for artist in axes.collections if artist.get_gid() == "target-points"
        )
        source = next(
            artist for artist in axes.collections if artist.get_gid() == "source-points"
        )
        from matplotlib.colors import to_rgba

        assert np.asarray(sphere.get_facecolor())[0] == pytest.approx(
            to_rgba("#123456", alpha=0.8)
        )
        assert target.get_label() == "Targets (2)"
        assert np.asarray(target.get_facecolor())[0] == pytest.approx(
            to_rgba("#e28a37", alpha=1.0)
        )
        assert source.get_label().startswith("Actual sources")
        assert np.asarray(source.get_facecolor())[0] == pytest.approx(
            to_rgba("#414b55", alpha=0.7)
        )
    finally:
        plt.close(figure)


def test_plot_appearance_keeps_projection_geometry_default_fields() -> None:
    evaluation, _ = _frame_case(
        Appearance(color="#123456", alpha=0.4, label="geometry sphere")
    )
    from gragra._plotting._setup import plot_setup

    figure = plot_setup(evaluation, view="xy")
    try:
        patch = next(
            patch
            for patch in figure.axes[0].patches
            if patch.get_label() == "geometry sphere"
        )
        from matplotlib.colors import to_rgba

        assert patch.get_facecolor() == pytest.approx(to_rgba("#123456", alpha=0.4))
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("view", "coordinate_indices"),
    [("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))],
)
def test_projection_views_are_labeled_and_keep_target_coordinates(
    view: str, coordinate_indices: tuple[int, int]
) -> None:
    evaluation, expected = _frame_case()

    from gragra._plotting._setup import plot_setup

    figure = plot_setup(evaluation, view=view, unit="m", max_sources=None)
    try:
        axes = figure.axes[0]
        assert axes.name == "rectilinear"
        assert view in axes.get_title().lower()
        assert axes.get_aspect() == pytest.approx(1.0)
        assert axes.get_xlabel().startswith(view[0])
        assert axes.get_ylabel().startswith(view[1])
        target_artist = next(
            artist for artist in axes.collections if artist.get_gid() == "target-points"
        )
        np.testing.assert_allclose(
            _scatter_values(target_artist, three_dimensional=False),
            expected["targets"][:, coordinate_indices],
        )
        source_artist = next(
            artist for artist in axes.collections if artist.get_gid() == "source-points"
        )
        np.testing.assert_allclose(
            _scatter_values(source_artist, three_dimensional=False),
            expected["source_all"][:, coordinate_indices],
        )
        assert "5 / 5" in source_artist.get_label()

        for axis_index, axis_name in enumerate("xyz"):
            axis_artist = next(
                artist
                for artist in axes.collections
                if artist.get_gid() == f"model-frame-axis-{axis_name}"
            )
            projected = (
                expected["model_basis"][list(coordinate_indices), axis_index]
                * evaluation.model_spec.geometry.radius_m
                * 0.55
            )
            np.testing.assert_allclose([axis_artist.U[0], axis_artist.V[0]], projected)
    finally:
        plt.close(figure)


def test_analytic_setup_without_discrete_source_and_existing_axes() -> None:
    evaluation, _ = _frame_case()
    analytic = ResolvedEvaluation(
        evaluation.model,
        evaluation.targets,
        evaluation.computation_frame,
        evaluation.output_frame,
        source=None,
    )
    figure = plt.figure()
    axes = figure.add_subplot(1, 2, 1)
    other_axes = figure.add_subplot(1, 2, 2)
    (other_line,) = other_axes.plot([0, 1], [1, 0])
    other_axes_position = other_axes.get_position().frozen()

    from gragra._plotting._setup import plot_setup

    try:
        returned = plot_setup(analytic, view="xy", ax=axes)
        assert returned is figure
        assert figure.axes == [axes, other_axes]
        assert list(other_axes.lines) == [other_line]
        np.testing.assert_allclose(
            other_axes.get_position().bounds, other_axes_position.bounds
        )
        assert any(
            "analytic" in artist.get_label().lower() for artist in axes.collections
        )
        assert {text.get_text() for text in axes.texts} >= {"A", "B"}
    finally:
        plt.close(figure)


def test_empty_target_domain_keeps_the_analytic_sphere_plotable() -> None:
    evaluation, _ = _frame_case()
    empty_targets = TargetPoints(np.empty((0, 3)) * u.m, (), evaluation.targets.frame)
    empty_evaluation = ResolvedEvaluation(
        evaluation.model,
        empty_targets,
        evaluation.computation_frame,
        evaluation.output_frame,
        source=None,
    )

    from gragra._plotting._setup import plot_setup

    figure = plot_setup(empty_evaluation, view="xz")
    try:
        axes = figure.axes[0]
        target_artist = next(
            artist for artist in axes.collections if artist.get_gid() == "target-points"
        )
        assert len(target_artist.get_offsets()) == 0
        assert "Targets (0)" in target_artist.get_label()
    finally:
        plt.close(figure)


def test_plot_does_not_run_or_recreate_the_resolved_source() -> None:
    evaluation, _ = _frame_case()

    class ModelWithExecutionGuards:
        spec = evaluation.model_spec

        def run(self):
            pytest.fail("plot_setup must not run the model")

        def discretize(self):
            pytest.fail("plot_setup must not discretize a source")

    guarded_evaluation = ResolvedEvaluation(
        ModelWithExecutionGuards(),
        evaluation.targets,
        evaluation.computation_frame,
        evaluation.output_frame,
        source=evaluation.source,
    )

    from gragra._plotting._setup import plot_setup

    figure = plot_setup(guarded_evaluation, max_sources=2)
    try:
        source_artist = next(
            artist
            for artist in figure.axes[0].collections
            if artist.get_gid() == "source-points"
        )
        assert "2 / 5 shown" in source_artist.get_label()
    finally:
        plt.close(figure)


def test_plot_rejects_invalid_view_units_limits_and_mismatched_axes() -> None:
    evaluation, _ = _frame_case()

    from gragra._plotting._setup import plot_setup

    figure, two_dimensional_axes = plt.subplots()
    three_dimensional_figure = plt.figure()
    three_dimensional_axes = three_dimensional_figure.add_subplot(projection="3d")
    polar_figure = plt.figure()
    polar_axes = polar_figure.add_subplot(projection="polar")
    logarithmic_figure, logarithmic_axes = plt.subplots()
    logarithmic_axes.set_xscale("log")
    logarithmic_3d_figure = plt.figure()
    logarithmic_3d_axes = logarithmic_3d_figure.add_subplot(projection="3d")
    logarithmic_3d_axes.set_zscale("log")
    occupied_figure, occupied_axes = plt.subplots()
    (occupied_line,) = occupied_axes.plot([0, 1], [0, 1])
    titled_figure, titled_axes = plt.subplots()
    titled_axes.set_title("Existing work")
    try:
        with pytest.raises(ValueError, match="view"):
            plot_setup(evaluation, view="side")
        with pytest.raises(ValueError, match="length unit"):
            plot_setup(evaluation, unit="s")
        with pytest.raises(ValueError, match="max_sources"):
            plot_setup(evaluation, max_sources=0)
        with pytest.raises(ValueError, match="max_sources"):
            plot_setup(evaluation, max_sources=True)
        with pytest.raises(ValueError, match="3D"):
            plot_setup(evaluation, view="3d", ax=two_dimensional_axes)
        with pytest.raises(ValueError, match="2D"):
            plot_setup(evaluation, view="xy", ax=three_dimensional_axes)
        with pytest.raises(ValueError, match="Cartesian"):
            plot_setup(evaluation, view="xy", ax=polar_axes)
        assert len(polar_axes.collections) == 0
        assert len(polar_axes.patches) == 0
        with pytest.raises(ValueError, match="linear"):
            plot_setup(evaluation, view="xy", ax=logarithmic_axes)
        assert len(logarithmic_axes.lines) == 0
        with pytest.raises(ValueError, match="linear"):
            plot_setup(evaluation, view="3d", ax=logarithmic_3d_axes)
        assert len(logarithmic_3d_axes.collections) == 0
        with pytest.raises(ValueError, match="empty"):
            plot_setup(evaluation, view="xy", ax=occupied_axes)
        assert list(occupied_axes.lines) == [occupied_line]
        with pytest.raises(ValueError, match="empty"):
            plot_setup(evaluation, view="xy", ax=titled_axes)
        assert titled_axes.get_title() == "Existing work"
    finally:
        plt.close(figure)
        plt.close(three_dimensional_figure)
        plt.close(polar_figure)
        plt.close(logarithmic_figure)
        plt.close(logarithmic_3d_figure)
        plt.close(occupied_figure)
        plt.close(titled_figure)


def test_setup_import_does_not_load_matplotlib_or_change_backend() -> None:
    code = """
import sys
import gragra._plotting._setup
assert 'matplotlib' not in sys.modules
"""
    environment = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr
