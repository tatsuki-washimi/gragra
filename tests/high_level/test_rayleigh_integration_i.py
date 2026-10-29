"""I-lane integration tests for the public Rayleigh phasor path."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesMatrix

from gragra.constants import G_SI
from gragra.geometry import Frame, HalfSpace, Sphere
from gragra.results import PhasorRepresentation, compare
from gragra.targets import TargetPoints
from gragra.waves import rayleigh_speed


def _rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _phasor_case(
    phase_origin_m: np.ndarray,
    amplitudes_m: np.ndarray,
    *,
    surface: Frame | None = None,
    targets: TargetPoints | None = None,
):
    from gragra.analytic.seismic import rayleigh_halfspace

    if surface is None:
        world = Frame("world")
        origin_m = np.array([6.0, -3.0, 2.0])
        basis = _rz(0.41) @ _rx(-0.28)
        surface = Frame("surface", origin_m * u.m, basis, world)
        local_targets_m = np.array([[2.2, -1.3, 2.4], [-3.1, 4.2, -1.8]])
        world_targets_m = origin_m + local_targets_m @ basis.T
        targets = TargetPoints(world_targets_m * u.m, ("above", "below"), world)
    else:
        world = surface.root
        if targets is None:
            raise ValueError("targets must be supplied with a reused surface frame")
        basis = surface.basis
    density = 2500.0 * u.kg / u.m**3
    vp, vs = 4000.0, 2300.0
    frequencies_hz = np.array([1.7, 4.3])
    model = rayleigh_halfspace(
        geometry=HalfSpace(surface),
        density=density,
        vp=vp * u.m / u.s,
        vs=vs * u.m / u.s,
        vertical_displacement=FrequencySeries(
            amplitudes_m,
            frequencies=frequencies_hz * u.Hz,
            unit=u.m,
        ),
        representation=PhasorRepresentation(),
        reference_time=(1_000_000_000, 250_000_000),
        phase_origin=phase_origin_m * u.m,
        direction=(0.6, 0.8, 0.0),
    )
    return model, targets, world, frequencies_hz, vp, vs, density, basis, surface


def test_public_rayleigh_route_runs_projects_and_compares_compensated_origins() -> None:
    from gragra.analytic.seismic import rayleigh_halfspace
    from gragra.observables.rayleigh_literature import (
        rayleigh_halfspace_allspace_observables,
    )

    assert callable(rayleigh_halfspace)
    assert HalfSpace.__module__ == "gragra.geometry._halfspace"
    assert Sphere.__module__ == "gragra.geometry._sphere"

    vp, vs = 4000.0, 2300.0
    frequencies_hz = np.array([1.7, 4.3])
    amplitudes_m = np.array([1.0 + 0.4j, -0.3 + 0.8j]) * 1e-9
    phase_origin_m = np.array([0.7, -1.2, 0.0])
    direction = np.array([0.6, 0.8, 0.0])
    shifted_by_m = np.array([-2.1, 1.6, 0.0])
    k_r = 2.0 * np.pi * frequencies_hz / rayleigh_speed(vp, vs)
    shifted_amplitudes_m = amplitudes_m * np.exp(
        1j * k_r * np.dot(direction, shifted_by_m)
    )

    reference_model, targets, world, _, vp, vs, density, basis, surface = _phasor_case(
        phase_origin_m, amplitudes_m
    )
    candidate_model, _, _, _, _, _, _, _, _ = _phasor_case(
        phase_origin_m + shifted_by_m,
        shifted_amplitudes_m,
        surface=surface,
        targets=targets,
    )
    setup = reference_model.at(targets, output_frame=world)
    report = setup.validate()
    assert report.valid
    assert report.observables == ("potential", "acceleration", "gradient")

    figure = setup.plot_setup(view="xz", unit="m")
    try:
        assert figure.axes[0].name == "rectilinear"
        assert figure.axes[0].get_xlabel().startswith("x")
        assert figure.axes[0].get_ylabel().startswith("z")
    finally:
        plt.close(figure)

    result = setup.run()
    assert result.g_applications == 1
    assert isinstance(result.potential, FrequencySeriesMatrix)
    assert isinstance(result.acceleration, FrequencySeriesMatrix)
    assert isinstance(result.gradient, FrequencySeriesMatrix)
    assert result.potential.shape == (2, 1, 2)
    assert result.acceleration.shape == (2, 3, 2)
    assert result.gradient.shape == (2, 9, 2)
    assert tuple(result.gradient.rows.keys()) == ("above", "below")
    assert tuple(result.gradient.cols.keys()) == (
        "xx",
        "xy",
        "xz",
        "yx",
        "yy",
        "yz",
        "zx",
        "zy",
        "zz",
    )
    assert result.potential.meta[0, 0].unit == u.m**2 / u.s**2
    assert result.acceleration.meta[0, 0].unit == u.m / u.s**2
    assert result.gradient.meta[0, 0].unit == u.s**-2
    np.testing.assert_array_equal(
        result.gradient.frequencies.to_value(u.Hz), frequencies_hz
    )
    assert np.iscomplexobj(np.asarray(result.gradient))

    # Independently exercise the high-level payload adapter against the SI
    # observable. This uses the same local numerical model, so its separate
    # acceptance tolerance checks only the connection, labels, and rotation.
    spec = setup.resolved.model_spec
    low_level = rayleigh_halfspace_allspace_observables(
        setup.resolved.target_positions_m,
        density_kg_m3=spec.density_kg_m3,
        v_p_m_s=vp,
        v_s_m_s=vs,
        frequency_hz=frequencies_hz,
        vertical_surface_amplitude_m=amplitudes_m,
        direction=direction,
        phase_origin_m=phase_origin_m,
    )
    expected_matrices = {
        "potential": low_level.potential.T[:, np.newaxis, :],
        "acceleration": np.einsum(
            "ij,ktj->kti", basis, low_level.acceleration
        ).transpose(1, 2, 0),
        "gradient": np.einsum("ia,ktab,jb->ktij", basis, low_level.gradient, basis)
        .reshape(len(frequencies_hz), len(targets), 9)
        .transpose(1, 2, 0),
    }
    coupling_scales = {
        "potential": G_SI
        * density.to_value(u.kg / u.m**3)
        * np.abs(amplitudes_m)
        / k_r,
        "acceleration": G_SI * density.to_value(u.kg / u.m**3) * np.abs(amplitudes_m),
        "gradient": G_SI * density.to_value(u.kg / u.m**3) * np.abs(amplitudes_m) * k_r,
    }
    for observable, expected in expected_matrices.items():
        actual = np.asarray(getattr(result, observable))
        scale = coupling_scales[observable].reshape(1, 1, len(frequencies_hz))
        assert np.all(
            np.abs(actual - expected) <= 1e-12 * scale + 1e-12 * np.abs(expected)
        ), observable

    selected = result.select("gradient", "potential")
    assert selected.observables == ("gradient", "potential")
    assert selected.values.gradient is result.values.gradient
    assert selected.values.potential is result.values.potential
    assert selected.acceleration is None

    shifted_result = candidate_model.at(targets, output_frame=world).run()
    comparison = compare(reference=result, candidate=shifted_result)
    scales = {
        "potential": G_SI * 2500.0 * np.abs(amplitudes_m) / k_r,
        "acceleration": G_SI * 2500.0 * np.abs(amplitudes_m),
        "gradient": G_SI * 2500.0 * np.abs(amplitudes_m) * k_r,
    }
    for observable, scale in scales.items():
        reference_values = np.asarray(getattr(result, observable))
        difference = np.asarray(getattr(comparison.differences, observable))
        scale_shape = (1, 1, len(scale))
        assert np.all(
            np.abs(difference)
            <= 1e-12 * scale.reshape(scale_shape) + 1e-12 * np.abs(reference_values)
        ), observable


def test_public_sphere_path_remains_a_static_real_evaluation() -> None:
    from gragra.analytic.objects import uniform_sphere

    root = Frame("sphere-world")
    geometry = Sphere(np.zeros(3) * u.m, 1.0 * u.m, root)
    targets = TargetPoints([[2.0, 0.0, 0.0]] * u.m, ("outside",), root)
    density = 2500.0 * u.kg / u.m**3

    result = uniform_sphere(geometry, density).at(targets).run()

    mass_kg = (4.0 * np.pi / 3.0) * 2500.0
    assert isinstance(result.potential, u.Quantity)
    assert not np.iscomplexobj(result.potential.value)
    np.testing.assert_allclose(
        result.potential.to_value(u.m**2 / u.s**2),
        [-G_SI * mass_kg / 2.0],
        rtol=1e-14,
    )
    assert result.g_applications == 1
