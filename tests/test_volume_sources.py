import numpy as np
import pytest

from gragra import point_acceleration
from gragra.gravity_sources.volume import VolumeElementSource
from gragra.gravity_sources.weighted_points import WeightedPointSource


def test_volume_source_scalar_density():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    volumes = np.array([0.1, 0.2])
    density = 1000.0

    vol_src = VolumeElementSource(
        positions_m=positions, volumes_m3=volumes, density_kg_m3=density
    )

    # Check properties
    np.testing.assert_allclose(vol_src.positions_m, positions)
    np.testing.assert_allclose(vol_src.volumes_m3, volumes)
    assert vol_src.density_kg_m3 == 1000.0

    # Check conversion to WeightedPointSource
    weighted = vol_src.as_weighted_source()
    assert isinstance(weighted, WeightedPointSource)
    np.testing.assert_allclose(weighted.positions_m, positions)
    np.testing.assert_allclose(weighted.weights_kg, [100.0, 200.0])
    assert weighted.weights_kg.dtype == np.float64


def test_volume_source_array_density():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    volumes = np.array([0.1, 0.2])
    density = np.array([1000.0, 2000.0])

    vol_src = VolumeElementSource(
        positions_m=positions, volumes_m3=volumes, density_kg_m3=density
    )

    # Check properties
    np.testing.assert_allclose(vol_src.density_kg_m3, density)

    # Check conversion
    weighted = vol_src.as_weighted_source()
    np.testing.assert_allclose(weighted.weights_kg, [100.0, 400.0])
    assert weighted.weights_kg.dtype == np.float64


def test_volume_source_complex_density():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    volumes = np.array([0.1, 0.2])

    # Scalar complex density
    vol_src_scalar = VolumeElementSource(
        positions_m=positions, volumes_m3=volumes, density_kg_m3=1000.0 + 500j
    )
    weighted_scalar = vol_src_scalar.as_weighted_source()
    assert weighted_scalar.weights_kg.dtype == np.complex128
    np.testing.assert_allclose(weighted_scalar.weights_kg, [100.0 + 50j, 200.0 + 100j])

    # Array complex density
    density_arr = np.array([1000.0 + 500j, 2000j])
    vol_src_arr = VolumeElementSource(
        positions_m=positions, volumes_m3=volumes, density_kg_m3=density_arr
    )
    weighted_arr = vol_src_arr.as_weighted_source()
    assert weighted_arr.weights_kg.dtype == np.complex128
    np.testing.assert_allclose(weighted_arr.weights_kg, [100.0 + 50j, 400j])


def test_volume_source_invalid_inputs():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    volumes = np.array([0.1, 0.2])
    density = 1000.0

    # Negative/zero volume
    msg_pos = "volumes_m3 must contain only positive values"
    with pytest.raises(ValueError, match=msg_pos):
        VolumeElementSource(positions, np.array([0.1, 0.0]), density)
    with pytest.raises(ValueError, match=msg_pos):
        VolumeElementSource(positions, np.array([0.1, -0.2]), density)

    # Non-finite volume/density
    msg_fin = "volumes_m3 must contain only finite numbers"
    with pytest.raises(ValueError, match=msg_fin):
        VolumeElementSource(positions, np.array([0.1, np.inf]), density)
    with pytest.raises(ValueError, match="density_kg_m3 must be finite"):
        VolumeElementSource(positions, volumes, np.nan)
    msg_fin_dens = "density_kg_m3 must contain only finite numbers"
    with pytest.raises(ValueError, match=msg_fin_dens):
        VolumeElementSource(positions, volumes, np.array([1000.0, np.nan]))

    # Booleans
    with pytest.raises(TypeError, match="volumes_m3 must not contain bool"):
        VolumeElementSource(positions, np.array([True, True]), density)
    with pytest.raises(TypeError, match="volumes_m3 must not contain bool"):
        VolumeElementSource(positions, True, density)
    with pytest.raises(TypeError, match="volumes_m3 must not contain bool"):
        VolumeElementSource(positions, [0.1, False], density)
    with pytest.raises(TypeError, match="density_kg_m3 must not contain bool"):
        VolumeElementSource(positions, volumes, True)
    with pytest.raises(TypeError, match="density_kg_m3 must not contain bool"):
        VolumeElementSource(positions, volumes, [1000.0, False])

    # Shape mismatch
    msg_match_vol = "Length of positions_m and volumes_m3 must match"
    with pytest.raises(ValueError, match=msg_match_vol):
        VolumeElementSource(positions, np.array([0.1]), density)
    msg_match_dens = "Length of density_kg_m3 must match positions_m"
    with pytest.raises(ValueError, match=msg_match_dens):
        VolumeElementSource(positions, volumes, np.array([1000.0]))


def test_volume_source_read_only():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    volumes = np.array([0.1, 0.2])
    density = np.array([1000.0, 2000.0])

    vol_src = VolumeElementSource(
        positions_m=positions, volumes_m3=volumes, density_kg_m3=density
    )

    # Verify attributes cannot be modified directly
    with pytest.raises(AttributeError):
        vol_src.positions_m = positions
    with pytest.raises(AttributeError):
        vol_src.volumes_m3 = volumes
    with pytest.raises(AttributeError):
        vol_src.density_kg_m3 = density

    # Verify underlying arrays are read-only
    assert not vol_src.positions_m.flags.writeable
    assert not vol_src.volumes_m3.flags.writeable
    assert not vol_src.density_kg_m3.flags.writeable


def test_volume_source_observable_usage():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    volumes = np.array([0.1, 0.2])
    density = 1000.0

    vol_src = VolumeElementSource(
        positions_m=positions, volumes_m3=volumes, density_kg_m3=density
    )

    # VolumeElementSource is not directly passable to observables
    # Verify that point_acceleration raises TypeError when passed directly
    with pytest.raises(TypeError):
        point_acceleration(vol_src, [[0.5, 1.0, 0.0]])

    # Pass via explicit as_weighted_source()
    acc = point_acceleration(vol_src.as_weighted_source(), [[0.5, 1.0, 0.0]])
    assert acc.shape == (1, 3)
