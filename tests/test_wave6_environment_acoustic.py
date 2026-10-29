from __future__ import annotations

from pathlib import Path

import numpy as np

import gragra
from scripts.research_wave6_environment_acoustic import (
    FREQUENCY_HZ,
    SOUND_SPEED_M_S,
    TARGETS_M,
    _partition_bounds,
    acoustic_case_matrix,
    acoustic_density_phasor,
    acoustic_real_density_slab,
    case_representation,
    fit_phasor,
    midpoint_box_sources,
    study_status,
    synthesize_real_series,
)


def test_imports_this_worktree_source():
    expected = Path(__file__).resolve().parents[1] / "src/gragra/__init__.py"
    assert Path(gragra.__file__).resolve() == expected


def test_fixed_acoustic_geometry_and_sweep_matrix():
    wavelength = SOUND_SPEED_M_S / FREQUENCY_HZ
    np.testing.assert_allclose(
        TARGETS_M,
        [
            [0.0, 0.0, -0.25 * wavelength],
            [0.125 * wavelength, 0.0, -0.5 * wavelength],
            [0.0, 0.25 * wavelength, -wavelength],
        ],
    )
    matrix = acoustic_case_matrix()
    assert {item["family"] for item in matrix} == {
        "mesh_width",
        "horizontal_half_width",
        "height",
    }
    assert len(matrix) == 9
    assert [item["cells_per_wavelength"] for item in matrix[:3]] == [8, 12, 16]
    assert [item["horizontal_half_width_wavelengths"] for item in matrix[3:6]] == [
        2,
        3,
        4,
    ]
    assert [item["height_wavelengths"] for item in matrix[6:]] == [1, 2, 3]


def test_acoustic_density_uses_physical_pressure_and_complex128():
    wavelength = SOUND_SPEED_M_S / FREQUENCY_HZ
    points = np.array([[0.0, 0.0, 0.0], [wavelength / 4.0, 0.0, 0.0]])

    density = acoustic_density_phasor(points)

    np.testing.assert_allclose(
        density,
        np.array([1.0, 1.0j], dtype=np.complex128) / SOUND_SPEED_M_S**2,
        rtol=0.0,
        atol=1e-20,
    )
    assert density.dtype == np.dtype(np.complex128)
    assert not density.flags.writeable


def test_midpoint_sources_cover_finite_air_box_without_ground_sheet():
    positions, volumes = midpoint_box_sources(
        horizontal_half_width_wavelengths=2,
        height_wavelengths=1,
        cells_per_wavelength=8,
    )

    assert positions.shape == (32 * 32 * 8, 3)
    assert volumes.shape == (len(positions),)
    assert np.all(positions[:, 2] > 0.0)
    assert np.all(positions[:, 2] < SOUND_SPEED_M_S / FREQUENCY_HZ)
    np.testing.assert_allclose(np.sum(volumes), (4 * 34.3) ** 2 * 34.3)
    assert not positions.flags.writeable
    assert not volumes.flags.writeable


def test_independent_reference_partition_covers_box_once_at_wavelength_scale():
    bounds = np.array([[-2.0, 2.0], [-1.0, 1.0], [0.0, 1.0]]) * 34.3
    panels = _partition_bounds(bounds, maximum_width_m=34.3)

    assert len(panels) == 4 * 2 * 1
    volumes = [np.prod(panel[:, 1] - panel[:, 0]) for panel in panels]
    np.testing.assert_allclose(sum(volumes), np.prod(bounds[:, 1] - bounds[:, 0]))
    assert all(np.max(panel[:, 1] - panel[:, 0]) <= 34.3 for panel in panels)


def test_one_and_two_cycle_fits_recover_all_trailing_observable_components():
    rng = np.random.default_rng(1451)
    phasors = {
        "potential": rng.normal(size=3) + 1j * rng.normal(size=3),
        "acceleration": rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)),
        "gradient": rng.normal(size=(3, 3, 3)) + 1j * rng.normal(size=(3, 3, 3)),
    }
    times = np.arange(128, dtype=np.float64) / (64.0 * FREQUENCY_HZ)

    series = synthesize_real_series(phasors, times)
    one_cycle = fit_phasor(series, times, sample_slice=slice(0, 64))
    two_cycles = fit_phasor(series, times, sample_slice=slice(0, 128))

    for name, expected in phasors.items():
        assert series[name].shape == (128, *expected.shape)
        np.testing.assert_allclose(one_cycle[name], expected, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(two_cycles[name], expected, rtol=1e-14, atol=1e-14)
        assert one_cycle[name].dtype == np.dtype(np.complex128)
        assert not one_cycle[name].flags.writeable


def test_real_density_slab_materializes_only_requested_samples_and_sources():
    phasor = np.array([1.0 + 2.0j, -3.0j, 4.0 - 1.0j])
    times = np.arange(8, dtype=np.float64) / (64.0 * FREQUENCY_HZ)

    slab = acoustic_real_density_slab(
        phasor, times, sample_slice=slice(2, 5), source_slice=slice(1, 3)
    )

    expected = np.real(
        np.exp(-2j * np.pi * FREQUENCY_HZ * times[2:5, None]) * phasor[None, 1:3]
    )
    np.testing.assert_allclose(slab, expected)
    assert slab.shape == (3, 2)
    assert slab.dtype == np.float64
    assert not slab.flags.writeable


def test_repeated_base_cases_are_explicitly_represented_once():
    representation = case_representation(acoustic_case_matrix())

    assert len(representation["unique_case_indices"]) == 7
    assert representation["representative_by_case_index"]["1"] == 1
    assert representation["representative_by_case_index"]["4"] == 1
    assert representation["representative_by_case_index"]["7"] == 1


def test_study_status_propagates_unresolved_oracle_cases():
    assert study_status(["CONVERGED", "ORACLE_UNRESOLVED"]) == "partial"
    assert study_status(["CONVERGED", "CONVERGED"]) == "complete"
