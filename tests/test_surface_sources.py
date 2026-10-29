import numpy as np
import pytest

from gragra import point_acceleration
from gragra.gravity_sources.surface import SurfaceMassSheetSource
from gragra.gravity_sources.weighted_points import WeightedPointSource


def test_surface_source_scalar_displacement():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # Normals will be unit-normalized internally
    normals = np.array([[0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    areas = np.array([0.1, 0.2])
    displacement = 0.01
    density = 1000.0

    surf_src = SurfaceMassSheetSource(
        positions_m=positions,
        normals=normals,
        areas_m2=areas,
        normal_displacement_m=displacement,
        density_kg_m3=density,
    )

    # Check properties
    np.testing.assert_allclose(surf_src.positions_m, positions)
    # Verify normalization
    expected_normals = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(surf_src.normals, expected_normals)
    np.testing.assert_allclose(surf_src.areas_m2, areas)
    assert surf_src.normal_displacement_m == 0.01
    assert surf_src.density_kg_m3 == 1000.0

    # Check conversion to WeightedPointSource
    weighted = surf_src.as_weighted_source()
    assert isinstance(weighted, WeightedPointSource)
    np.testing.assert_allclose(weighted.positions_m, positions)
    # weights = density * displacement * area = 1000.0 * 0.01 * [0.1, 0.2] = [1.0, 2.0]
    np.testing.assert_allclose(weighted.weights_kg, [1.0, 2.0])
    assert weighted.weights_kg.dtype == np.float64


def test_surface_source_array_displacement():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    normals = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    areas = np.array([0.1, 0.2])
    displacement = np.array([0.01, 0.02])
    density = 1000.0

    surf_src = SurfaceMassSheetSource(
        positions_m=positions,
        normals=normals,
        areas_m2=areas,
        normal_displacement_m=displacement,
        density_kg_m3=density,
    )

    # Check properties
    np.testing.assert_allclose(surf_src.normal_displacement_m, displacement)

    # Check conversion
    weighted = surf_src.as_weighted_source()
    # weights = 1000.0 * [0.01, 0.02] * [0.1, 0.2] = [1.0, 4.0]
    np.testing.assert_allclose(weighted.weights_kg, [1.0, 4.0])
    assert weighted.weights_kg.dtype == np.float64


def test_surface_source_complex():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    normals = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    areas = np.array([0.1, 0.2])

    # Complex displacement
    surf_src_complex_disp = SurfaceMassSheetSource(
        positions_m=positions,
        normals=normals,
        areas_m2=areas,
        normal_displacement_m=0.01 + 0.005j,
        density_kg_m3=1000.0,
    )
    weighted_complex_disp = surf_src_complex_disp.as_weighted_source()
    assert weighted_complex_disp.weights_kg.dtype == np.complex128
    np.testing.assert_allclose(
        weighted_complex_disp.weights_kg, [1.0 + 0.5j, 2.0 + 1.0j]
    )

    # Complex density
    surf_src_complex_dens = SurfaceMassSheetSource(
        positions_m=positions,
        normals=normals,
        areas_m2=areas,
        normal_displacement_m=0.01,
        density_kg_m3=1000.0 + 500j,
    )
    weighted_complex_dens = surf_src_complex_dens.as_weighted_source()
    assert weighted_complex_dens.weights_kg.dtype == np.complex128
    np.testing.assert_allclose(
        weighted_complex_dens.weights_kg, [1.0 + 0.5j, 2.0 + 1.0j]
    )


def test_surface_source_invalid_inputs():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    normals = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    areas = np.array([0.1, 0.2])
    displacement = 0.01
    density = 1000.0

    # Zero vector normal
    with pytest.raises(ValueError, match="normals must not contain zero vectors"):
        bad_norms = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        SurfaceMassSheetSource(positions, bad_norms, areas, displacement, density)

    # Non-finite normal
    msg_fin_norm = "normals must contain only finite numbers"
    with pytest.raises(ValueError, match=msg_fin_norm):
        bad_norms = np.array([[0.0, np.nan, 0.0], [0.0, 0.0, 1.0]])
        SurfaceMassSheetSource(positions, bad_norms, areas, displacement, density)

    # Negative/zero area
    msg_pos_area = "areas_m2 must contain only positive values"
    with pytest.raises(ValueError, match=msg_pos_area):
        SurfaceMassSheetSource(
            positions, normals, np.array([0.1, 0.0]), displacement, density
        )
    with pytest.raises(ValueError, match=msg_pos_area):
        SurfaceMassSheetSource(
            positions, normals, np.array([0.1, -0.2]), displacement, density
        )

    # Non-finite area/displacement/density
    msg_fin_area = "areas_m2 must contain only finite numbers"
    with pytest.raises(ValueError, match=msg_fin_area):
        SurfaceMassSheetSource(
            positions, normals, np.array([0.1, np.inf]), displacement, density
        )
    with pytest.raises(ValueError, match="normal_displacement_m must be finite"):
        SurfaceMassSheetSource(positions, normals, areas, np.nan, density)
    with pytest.raises(ValueError, match="density_kg_m3 must be finite"):
        SurfaceMassSheetSource(positions, normals, areas, displacement, np.nan)

    # Non-scalar density
    with pytest.raises(ValueError, match="density_kg_m3 must be a scalar"):
        bad_dens = np.array([1000.0, 2000.0])
        SurfaceMassSheetSource(positions, normals, areas, displacement, bad_dens)

    # Booleans in inputs
    with pytest.raises(TypeError, match="normals must not contain bool"):
        SurfaceMassSheetSource(positions, True, areas, displacement, density)
    with pytest.raises(TypeError, match="normals must not contain bool"):
        bad_norms = [[0.0, True, 0.0], [0.0, 0.0, 1.0]]
        SurfaceMassSheetSource(positions, bad_norms, areas, displacement, density)
    with pytest.raises(TypeError, match="areas_m2 must not contain bool"):
        SurfaceMassSheetSource(positions, normals, [0.1, False], displacement, density)
    with pytest.raises(TypeError, match="normal_displacement_m must not contain bool"):
        SurfaceMassSheetSource(positions, normals, areas, True, density)
    with pytest.raises(TypeError, match="density_kg_m3 must not contain bool"):
        SurfaceMassSheetSource(positions, normals, areas, displacement, True)


def test_surface_source_read_only():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    normals = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    areas = np.array([0.1, 0.2])
    displacement = np.array([0.01, 0.02])
    density = 1000.0

    surf_src = SurfaceMassSheetSource(
        positions_m=positions,
        normals=normals,
        areas_m2=areas,
        normal_displacement_m=displacement,
        density_kg_m3=density,
    )

    # Verify attributes cannot be modified directly
    with pytest.raises(AttributeError):
        surf_src.positions_m = positions
    with pytest.raises(AttributeError):
        surf_src.normals = normals
    with pytest.raises(AttributeError):
        surf_src.areas_m2 = areas
    with pytest.raises(AttributeError):
        surf_src.normal_displacement_m = displacement
    with pytest.raises(AttributeError):
        surf_src.density_kg_m3 = density

    # Verify underlying arrays are read-only
    assert not surf_src.positions_m.flags.writeable
    assert not surf_src.normals.flags.writeable
    assert not surf_src.areas_m2.flags.writeable
    assert not surf_src.normal_displacement_m.flags.writeable


def test_surface_source_observable_usage():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    normals = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    areas = np.array([0.1, 0.2])
    displacement = 0.01
    density = 1000.0

    surf_src = SurfaceMassSheetSource(
        positions_m=positions,
        normals=normals,
        areas_m2=areas,
        normal_displacement_m=displacement,
        density_kg_m3=density,
    )

    # SurfaceMassSheetSource is not directly passable to observables
    with pytest.raises(TypeError):
        point_acceleration(surf_src, [[0.5, 1.0, 0.0]])

    # Pass via explicit as_weighted_source()
    acc = point_acceleration(surf_src.as_weighted_source(), [[0.5, 1.0, 0.0]])
    assert acc.shape == (1, 3)
