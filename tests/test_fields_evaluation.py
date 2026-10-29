"""Contract tests for the thin fixed-mesh field/source builders."""

import numpy as np
import pytest

from gragra.fields import (
    build_fixed_mesh_displacement_sources,
    build_fixed_mesh_surface_sources,
    build_plane_wave_fixed_mesh_sources,
)
from gragra.fields.displacement_series import FixedMeshDisplacementSeries
from gragra.fields.surface import SurfaceDisplacementField
from gragra.waves import plane_wave_displacement

POSITIONS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=float)
VOLUMES = np.array([2.0, 3.0, 4.0])
DENSITY = 2300.0
DISPLACEMENT = np.array([[1.0 + 0.2j, 0.0, 0.0], [0.0, 2.0j, 0.0], [0.0, 0.0, -1.0j]])
IDS = ("cell-0", "cell-1", "cell-2")


def test_fixed_mesh_builder_preserves_mass_displacement_and_metadata():
    bundle = build_fixed_mesh_displacement_sources(
        POSITIONS,
        VOLUMES,
        DENSITY,
        DISPLACEMENT,
        source_ids=IDS,
        coordinate_frame="world",
        coordinate_origin_m=[0.5, -0.25, 0.75],
        phase_origin_m=[1.0, 2.0, 3.0],
    )

    np.testing.assert_array_equal(bundle.positions_m, POSITIONS)
    np.testing.assert_array_equal(bundle.mass_kg, DENSITY * VOLUMES)
    np.testing.assert_array_equal(bundle.displacement_m, DISPLACEMENT)
    np.testing.assert_array_equal(
        bundle.volume_source.as_weighted_source().weights_kg, DENSITY * VOLUMES
    )
    assert bundle.source_ids == IDS
    assert bundle.coordinate_frame == "world"
    assert bundle.unit_system == "SI"
    np.testing.assert_array_equal(bundle.phase_origin_m, [1.0, 2.0, 3.0])
    assert not bundle.mass_kg.flags.writeable
    assert not bundle.displacement_m.flags.writeable


def test_fixed_mesh_builder_can_create_the_existing_frequency_series():
    bundle = build_fixed_mesh_displacement_sources(
        POSITIONS, VOLUMES, DENSITY, DISPLACEMENT, source_ids=IDS
    )
    series = bundle.as_frequency_series(5.0, t_ref_s=0.25)
    assert isinstance(series, FixedMeshDisplacementSeries)
    np.testing.assert_array_equal(series.mass_kg, DENSITY * VOLUMES)
    np.testing.assert_array_equal(series.displacement_m[0], DISPLACEMENT)
    assert series.source_ids == IDS
    assert series.axis_kind == "frequency"


@pytest.mark.parametrize("wave_type", ["P", "SV"])
def test_plane_wave_builder_matches_existing_wave_constructor(wave_type):
    k_hat = np.array([0.6, 0.2, np.sqrt(1.0 - 0.6**2 - 0.2**2)])
    expected = plane_wave_displacement(
        POSITIONS - np.array([1.0, 2.0, 3.0]),
        k_hat,
        0.4,
        wave_type,
        amplitude_m=2.0e-9,
        phase0=0.3,
    )[:, 0, 0, :]
    bundle = build_plane_wave_fixed_mesh_sources(
        POSITIONS,
        VOLUMES,
        DENSITY,
        k_hat=k_hat,
        wavenumber_rad_m=0.4,
        wave_type=wave_type,
        amplitude_m=2.0e-9,
        phase0=0.3,
        phase_origin_m=[1.0, 2.0, 3.0],
        source_ids=IDS,
    )
    np.testing.assert_allclose(bundle.displacement_m, expected, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(bundle.mass_kg, DENSITY * VOLUMES)


def test_surface_builder_matches_existing_surface_field_without_double_counting():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
    )
    triangles = np.array([[0, 1, 2], [0, 1, 3]])
    displacement = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0], [0.0, 0.0, 4.0]]
    )
    expected = SurfaceDisplacementField(
        vertices, triangles, displacement
    ).as_surface_mass_source(100.0)
    bundle = build_fixed_mesh_surface_sources(
        vertices,
        triangles,
        displacement,
        100.0,
        source_ids=("surf-0", "surf-1"),
        coordinate_frame="world",
        coordinate_origin_m=[0.5, 0.5, 0.5],
        phase_origin_m=[1.0, 1.0, 1.0],
        phase0_rad=0.2,
    )
    np.testing.assert_array_equal(bundle.source.positions_m, expected.positions_m)
    np.testing.assert_array_equal(
        bundle.source.as_weighted_source().weights_kg,
        expected.as_weighted_source().weights_kg,
    )
    assert bundle.source_ids == ("surf-0", "surf-1")
    assert bundle.phase0_rad == 0.2
