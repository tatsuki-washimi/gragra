"""Artist-level contract tests for the Rayleigh half-space setup plot."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeries
from mpl_toolkits.mplot3d import Axes3D

from gragra._rayleigh_contract import RayleighModelSpec, RayleighResolvedEvaluation
from gragra._rayleigh_evaluation_setup import RayleighEvaluationSetup
from gragra.geometry import Appearance
from gragra.geometry._frame import Frame
from gragra.geometry._halfspace import HalfSpace
from gragra.results._spectral import PhasorRepresentation
from gragra.targets._high_points import TargetPoints


def _rx(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _resolved_case(
    appearance: Appearance | None = None,
) -> tuple[RayleighResolvedEvaluation, dict[str, np.ndarray]]:
    root = Frame("root")
    surface_origin = np.array([3.0, -2.0, 1.5])
    surface_basis = _rz(0.43) @ _rx(-0.31)
    surface = Frame("surface", surface_origin * u.m, surface_basis, root)
    target_origin = np.array([-4.0, 0.7, 2.1])
    target_basis = _rz(-0.27) @ _rx(0.52)
    target_frame = Frame("targets", target_origin * u.m, target_basis, root)
    output_origin = np.array([1.1, -3.3, 0.8])
    output_basis = _rz(0.61) @ _rx(-0.22)
    output_frame = Frame("display", output_origin * u.m, output_basis, root)

    target_surface = np.array([[-2.0, 1.0, -1.0], [3.0, -2.0, 0.8], [0.5, 4.0, -3.0]])
    target_root = surface_origin + target_surface @ surface_basis.T
    target_coordinates = (target_root - target_origin) @ target_basis
    targets = TargetPoints(
        target_coordinates * u.m,
        ("below-west", "above-east", "below-north"),
        target_frame,
    )
    spec = RayleighModelSpec(
        HalfSpace(surface, appearance=appearance),
        2700 * u.kg / u.m**3,
        6000 * u.m / u.s,
        3500 * u.m / u.s,
        FrequencySeries(
            np.array([1.0 + 0.5j, 0.2 - 0.7j]),
            frequencies=[1.0, 2.5] * u.Hz,
            unit=u.m,
        ),
        PhasorRepresentation(),
        (1_000_000_000, 123_000_000),
        np.array([0.8, -0.4, 0.0]) * u.m,
        (0.6, 0.8, 0.0),
    )

    class Model:
        def __init__(self, model_spec: RayleighModelSpec) -> None:
            self.spec = model_spec

        def run(self):
            pytest.fail("plotting must not evaluate the model")

    resolved = RayleighResolvedEvaluation(Model(spec), targets, output_frame)
    target_output = (target_root - output_origin) @ output_basis
    expected = {
        "target_surface": target_surface,
        "target_output": target_output,
        "surface_origin": surface_origin,
        "surface_basis": surface_basis,
        "output_origin": output_origin,
        "output_basis": output_basis,
        "phase_origin": np.array([0.8, -0.4, 0.0]),
        "direction": np.array([0.6, 0.8, 0.0]),
    }
    return resolved, expected


def _offsets(artist: object, *, three_dimensional: bool) -> np.ndarray:
    if three_dimensional:
        x, y, z = artist._offsets3d  # type: ignore[attr-defined]
        return np.column_stack((x, y, z))
    return np.asarray(artist.get_offsets())  # type: ignore[attr-defined]


def _plot_function():
    from gragra._plotting._rayleigh_setup import plot_rayleigh_setup

    return plot_rayleigh_setup


def test_3d_artists_use_independent_rotated_frame_coordinates_and_show_halfspace() -> (
    None
):
    from matplotlib.colors import to_rgba

    resolved, expected = _resolved_case()
    positions_before = resolved.target_positions_m.copy()
    figure = _plot_function()(resolved, unit="cm")
    try:
        axes = figure.axes[0]
        assert isinstance(axes, Axes3D)
        assert axes._focal_length == np.inf
        spans = np.array(
            [
                np.diff(axes.get_xlim3d())[0],
                np.diff(axes.get_ylim3d())[0],
                np.diff(axes.get_zlim3d())[0],
            ]
        )
        np.testing.assert_allclose(spans, spans[0])
        np.testing.assert_allclose(
            axes.get_box_aspect(), np.ones(3) * axes.get_box_aspect()[0]
        )
        assert "display" in axes.get_title()
        assert "cm" in axes.get_title()
        assert "window of infinite half-space" in axes.get_title().lower()
        assert axes.get_xlabel().endswith("[cm]")
        assert axes.get_ylabel().endswith("[cm]")
        assert axes.get_zlabel().endswith("[cm]")
        figure.canvas.draw()
        medium = next(c for c in axes.collections if c.get_gid() == "elastic-medium")
        surface = next(c for c in axes.collections if c.get_gid() == "free-surface")
        assert medium.get_facecolor()[0] == pytest.approx(
            to_rgba("#d6e3ec", alpha=0.34)
        )
        assert surface.get_facecolor()[0] == pytest.approx(
            to_rgba("#7fb5d7", alpha=0.48)
        )
        assert medium.get_label() == "Elastic medium (local z < 0)"
        assert surface.get_label() == "Free surface (local z = 0)"

        targets = next(c for c in axes.collections if c.get_gid() == "target-points")
        np.testing.assert_allclose(
            _offsets(targets, three_dimensional=True), expected["target_output"] * 100
        )
        assert {text.get_text() for text in axes.texts} >= {
            "below-west",
            "above-east",
            "below-north",
        }

        phase = next(c for c in axes.collections if c.get_gid() == "phase-origin")
        phase_root = (
            expected["surface_origin"]
            + expected["surface_basis"] @ expected["phase_origin"]
        )
        phase_display = expected["output_basis"].T @ (
            phase_root - expected["output_origin"]
        )
        np.testing.assert_allclose(
            _offsets(phase, three_dimensional=True)[0], phase_display * 100
        )

        direction = next(
            c for c in axes.collections if c.get_gid() == "propagation-direction"
        )
        direction_root = expected["surface_basis"] @ expected["direction"]
        direction_display = expected["output_basis"].T @ direction_root
        direction_segment = max(
            direction._segments3d,
            key=lambda segment: np.linalg.norm(segment[1] - segment[0]),
        )
        direction_endpoints = np.asarray(direction_segment)
        direction_tail = int(
            np.argmin(np.linalg.norm(direction_endpoints - phase_display * 100, axis=1))
        )
        arrow_delta = (
            direction_endpoints[1 - direction_tail]
            - direction_endpoints[direction_tail]
        )
        np.testing.assert_allclose(
            arrow_delta / np.linalg.norm(arrow_delta), direction_display
        )
        assert "not amplitude" in direction.get_label().lower()

        normal = next(c for c in axes.collections if c.get_gid() == "surface-normal")
        normal_display = expected["output_basis"].T @ expected["surface_basis"][:, 2]
        normal_segment = max(
            normal._segments3d,
            key=lambda segment: np.linalg.norm(segment[1] - segment[0]),
        )
        normal_endpoints = np.asarray(normal_segment)
        normal_tail = int(
            np.argmin(np.linalg.norm(normal_endpoints - phase_display * 100, axis=1))
        )
        normal_delta = normal_endpoints[1 - normal_tail] - normal_endpoints[normal_tail]
        np.testing.assert_allclose(
            normal_delta / np.linalg.norm(normal_delta), normal_display
        )

        edges = [
            line
            for line in axes.lines
            if (line.get_gid() or "").startswith("surface-window-edge-")
        ]
        assert len(edges) == 4
        displayed_corners = np.array(
            [np.asarray(line.get_data_3d()).T[0] / 100 for line in edges]
        )
        root_corners = (
            displayed_corners @ expected["output_basis"].T + expected["output_origin"]
        )
        local_corners = (root_corners - expected["surface_origin"]) @ expected[
            "surface_basis"
        ]
        np.testing.assert_allclose(local_corners[:, 2], 0.0, atol=1e-12)
        np.testing.assert_array_equal(resolved.target_positions_m, positions_before)
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("view", "indices"), [("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))]
)
def test_projection_artists_keep_targets_direction_origin_and_surface(
    view: str, indices: tuple[int, int]
) -> None:
    from matplotlib.colors import to_rgba

    resolved, expected = _resolved_case()
    figure = _plot_function()(resolved, view=view, unit="km")
    try:
        axes = figure.axes[0]
        assert axes.name == "rectilinear"
        assert axes.get_aspect() == pytest.approx(1.0)
        assert "display" in axes.get_title()
        assert "km" in axes.get_title()
        assert "window of infinite half-space" in axes.get_title().lower()
        assert axes.get_xlabel().startswith(view[0])
        assert axes.get_ylabel().startswith(view[1])

        target_artist = next(
            c for c in axes.collections if c.get_gid() == "target-points"
        )
        np.testing.assert_allclose(
            _offsets(target_artist, three_dimensional=False),
            expected["target_output"][:, indices] / 1000,
        )
        assert {text.get_text() for text in axes.texts} >= {
            "below-west",
            "above-east",
            "below-north",
        }
        phase_artist = next(
            c for c in axes.collections if c.get_gid() == "phase-origin"
        )
        phase_root = (
            expected["surface_origin"]
            + expected["surface_basis"] @ expected["phase_origin"]
        )
        phase_display = expected["output_basis"].T @ (
            phase_root - expected["output_origin"]
        )
        np.testing.assert_allclose(
            _offsets(phase_artist, three_dimensional=False)[0],
            phase_display[list(indices)] / 1000,
        )

        direction_artist = next(
            c for c in axes.collections if c.get_gid() == "propagation-direction"
        )
        direction_root = expected["surface_basis"] @ expected["direction"]
        direction_display = expected["output_basis"].T @ direction_root
        projected = direction_display[list(indices)]
        np.testing.assert_allclose(
            direction_artist.get_offsets()[0], phase_display[list(indices)] / 1000
        )
        assert (
            np.dot(np.array([direction_artist.U[0], direction_artist.V[0]]), projected)
            > 0
        )
        np.testing.assert_allclose(
            np.array([direction_artist.U[0], direction_artist.V[0]])
            / np.linalg.norm([direction_artist.U[0], direction_artist.V[0]]),
            projected / np.linalg.norm(projected),
        )
        assert "not amplitude" in direction_artist.get_label().lower()

        normal_artist = next(
            c for c in axes.collections if c.get_gid() == "surface-normal"
        )
        normal_display = expected["output_basis"].T @ expected["surface_basis"][:, 2]
        projected_normal = normal_display[list(indices)]
        np.testing.assert_allclose(
            np.array([normal_artist.U[0], normal_artist.V[0]])
            / np.linalg.norm([normal_artist.U[0], normal_artist.V[0]]),
            projected_normal / np.linalg.norm(projected_normal),
        )

        surface_edges = [
            line
            for line in axes.lines
            if (line.get_gid() or "").startswith("surface-window-edge-")
        ]
        assert len(surface_edges) == 4
        assert {patch.get_gid() for patch in axes.patches} >= {
            "free-surface",
            "elastic-medium",
        }
        medium = next(p for p in axes.patches if p.get_gid() == "elastic-medium")
        surface = next(p for p in axes.patches if p.get_gid() == "free-surface")
        assert medium.get_facecolor() == pytest.approx(to_rgba("#d6e3ec", alpha=0.4))
        assert surface.get_facecolor() == pytest.approx(to_rgba("#7fb5d7", alpha=0.35))
        assert medium.get_label() == "Elastic medium (local z < 0)"
        assert surface.get_label() == "Free surface (local z = 0)"
    finally:
        plt.close(figure)


def test_halfspace_appearance_styles_geometry_artists_and_keeps_surface_label() -> None:
    from matplotlib.colors import to_rgba

    resolved, _ = _resolved_case(
        Appearance(color="#123456", alpha=0.4, label="geometry medium")
    )
    figure = _plot_function()(
        resolved,
        appearance=Appearance(alpha=0.8, label="call medium"),
    )
    try:
        axes = figure.axes[0]
        figure.canvas.draw()
        medium = next(
            artist
            for artist in axes.collections
            if artist.get_gid() == "elastic-medium"
        )
        surface = next(
            artist for artist in axes.collections if artist.get_gid() == "free-surface"
        )
        targets = next(
            artist for artist in axes.collections if artist.get_gid() == "target-points"
        )
        assert medium.get_facecolor()[0] == pytest.approx(to_rgba("#123456", alpha=0.8))
        assert surface.get_facecolor()[0] == pytest.approx(
            to_rgba("#123456", alpha=0.8)
        )
        assert medium.get_label() == "call medium"
        assert surface.get_label() == "Free surface (local z = 0)"
        assert targets.get_label() == "Targets (3)"
        assert targets.get_facecolor()[0] == pytest.approx(
            to_rgba("#d36d31", alpha=1.0)
        )
    finally:
        plt.close(figure)


def test_halfspace_appearance_styles_both_projected_geometry_patches() -> None:
    from matplotlib.colors import to_rgba

    resolved, _ = _resolved_case(
        Appearance(color="#654321", alpha=0.6, label="half-space")
    )
    figure = _plot_function()(resolved, view="xz")
    try:
        patches = {patch.get_gid(): patch for patch in figure.axes[0].patches}
        assert patches["elastic-medium"].get_facecolor() == pytest.approx(
            to_rgba("#654321", alpha=0.6)
        )
        assert patches["free-surface"].get_facecolor() == pytest.approx(
            to_rgba("#654321", alpha=0.6)
        )
        assert patches["elastic-medium"].get_label() == "half-space"
        assert patches["free-surface"].get_label() == "Free surface (local z = 0)"
    finally:
        plt.close(figure)


def test_public_rayleigh_setup_wrappers_accept_appearance() -> None:
    from matplotlib.colors import to_rgba

    from gragra._rayleigh_contract import (
        RayleighResponseEvaluation,
        RayleighResponseSpec,
    )
    from gragra._rayleigh_response_setup import RayleighResponseSetup

    resolved, _ = _resolved_case(
        Appearance(color="#abcdef", alpha=0.5, label="geometry medium")
    )
    call_appearance = Appearance(color="#fedcba")
    evaluation_setup = RayleighEvaluationSetup(resolved)
    evaluation_figure = evaluation_setup.plot_setup(appearance=call_appearance)

    geometry = resolved.model_spec.geometry
    spec = RayleighResponseSpec(
        geometry,
        2700 * u.kg / u.m**3,
        6000 * u.m / u.s,
        3500 * u.m / u.s,
        [1.0, 2.5] * u.Hz,
        [0.8, -0.4, 0.0] * u.m,
        (0.6, 0.8, 0.0),
    )

    class Model:
        pass

    model = Model()
    model.spec = spec
    response_resolved = RayleighResponseEvaluation(
        model, resolved.targets, resolved.output_frame
    )
    response_setup = RayleighResponseSetup(response_resolved)
    response_figure = response_setup.plot_setup(appearance=call_appearance)
    try:
        for figure in (evaluation_figure, response_figure):
            medium = next(
                artist
                for artist in figure.axes[0].collections
                if artist.get_gid() == "elastic-medium"
            )
            assert medium.get_facecolor()[0] == pytest.approx(
                to_rgba("#fedcba", alpha=0.5)
            )
            assert medium.get_label() == "geometry medium"
    finally:
        plt.close(evaluation_figure)
        plt.close(response_figure)


def test_plot_uses_a_provided_empty_axes_and_rejects_mismatches() -> None:
    resolved, _ = _resolved_case()
    figure = plt.figure()
    axes = figure.add_subplot(1, 2, 1)
    other_axes = figure.add_subplot(1, 2, 2)
    (other_line,) = other_axes.plot([0, 1], [0, 1])
    try:
        assert _plot_function()(resolved, view="xz", ax=axes) is figure
        assert figure.axes == [axes, other_axes]
        assert len(axes.collections) > 0
        assert list(other_axes.lines) == [other_line]

        with pytest.raises(ValueError, match="3D"):
            _plot_function()(resolved, view="3d", ax=axes)
        with pytest.raises(ValueError, match="empty"):
            _plot_function()(resolved, view="xz", ax=axes)
    finally:
        plt.close(figure)
