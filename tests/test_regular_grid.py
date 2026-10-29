import numpy as np
import pytest

from gragra.constants import G_SI
from gragra.fields.regular_grid import DensityPerturbationGrid, DisplacementGrid
from gragra.gravity_sources.volume import VolumeElementSource
from gragra.observables.acceleration import point_acceleration
from gragra.targets.points import TargetPoints


def test_density_perturbation_grid_basic():
    # 3D grid setup
    x = np.array([10.0, 11.0, 12.0])
    y = np.array([20.0, 22.0, 24.0])
    z = np.array([30.0, 33.0, 36.0])
    delta_rho = np.ones((3, 3, 3)) * 1.5

    # 1. cell_size_m is None (implicit grid widths from gradient)
    # dx = [1, 1, 1], dy = [2, 2, 2], dz = [3, 3, 3] -> volume = 1 * 2 * 3 = 6
    grid = DensityPerturbationGrid(x_m=x, y_m=y, z_m=z, delta_density_kg_m3=delta_rho)
    source = grid.as_volume_element_source()
    assert isinstance(source, VolumeElementSource)

    # Check cell volumes
    assert np.allclose(source.volumes_m3, 6.0)

    # Total mass check: 1.5 * 6.0 * 27 cells = 243.0 kg
    w_source = source.as_weighted_source()
    assert np.isclose(w_source.weights_kg.sum(), 243.0)

    # Read-only verification
    with pytest.raises(ValueError):
        source.volumes_m3[0] = 9.9
    with pytest.raises(ValueError):
        source.positions_m[0, 0] = 9.9
    with pytest.raises(ValueError):
        source.density_kg_m3[0] = 9.9

    # 2. cell_size_m is specified (explicit cell size)
    # cell_size_m = 0.5 -> uniform volume = 0.5^3 = 0.125
    grid_explicit = DensityPerturbationGrid(
        x_m=x, y_m=y, z_m=z, delta_density_kg_m3=delta_rho, cell_size_m=0.5
    )
    source_explicit = grid_explicit.as_volume_element_source()
    assert np.allclose(source_explicit.volumes_m3, 0.125)

    # With explicit size, 3-element tuple is allowed
    grid_tuple = DensityPerturbationGrid(
        x_m=x,
        y_m=y,
        z_m=z,
        delta_density_kg_m3=delta_rho,
        cell_size_m=(0.5, 1.0, 2.0),
    )
    source_tuple = grid_tuple.as_volume_element_source()
    # volume = 0.5 * 1.0 * 2.0 = 1.0
    assert np.allclose(source_tuple.volumes_m3, 1.0)


def test_density_perturbation_grid_validation():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    z = np.array([0.0, 1.0])
    delta_rho = np.ones((2, 2, 2))

    # Axis length mismatch
    with pytest.raises(ValueError, match="Length of x_m must match"):
        DensityPerturbationGrid(
            x_m=np.array([0.0]), y_m=y, z_m=z, delta_density_kg_m3=delta_rho
        )

    # Non-strictly increasing axis
    with pytest.raises(ValueError, match="must be strictly increasing"):
        DensityPerturbationGrid(
            x_m=np.array([1.0, 0.0]),
            y_m=y,
            z_m=z,
            delta_density_kg_m3=delta_rho,
        )

    # Complex-valued coordinate axis
    with pytest.raises(TypeError, match="must be real-valued"):
        DensityPerturbationGrid(
            x_m=np.array([0.0, 1.0]) + 0j,
            y_m=y,
            z_m=z,
            delta_density_kg_m3=delta_rho,
        )

    # cell_size_m is None and length < 2
    with pytest.raises(ValueError, match="Axis length must be >= 2"):
        DensityPerturbationGrid(
            x_m=np.array([0.0]),
            y_m=np.array([0.0]),
            z_m=np.array([0.0]),
            delta_density_kg_m3=np.ones((1, 1, 1)),
            cell_size_m=None,
        )

    # cell_size_m is explicit -> (1, 1, 1) grid is allowed!
    grid_single = DensityPerturbationGrid(
        x_m=np.array([0.0]),
        y_m=np.array([0.0]),
        z_m=np.array([0.0]),
        delta_density_kg_m3=np.ones((1, 1, 1)) * 3.0,
        cell_size_m=2.0,
    )
    source = grid_single.as_volume_element_source()
    # volume = 2.0^3 = 8.0, mass = 3.0 * 8.0 = 24.0
    assert np.allclose(source.volumes_m3, 8.0)
    assert np.allclose(source.density_kg_m3, 3.0)
    assert np.isclose(source.as_weighted_source().weights_kg.sum(), 24.0)

    # Invalid cell_size_m formats
    with pytest.raises(ValueError, match="positive and finite"):
        DensityPerturbationGrid(
            x_m=x, y_m=y, z_m=z, delta_density_kg_m3=delta_rho, cell_size_m=-0.5
        )
    with pytest.raises(ValueError, match="positive and finite"):
        DensityPerturbationGrid(
            x_m=x,
            y_m=y,
            z_m=z,
            delta_density_kg_m3=delta_rho,
            cell_size_m=(0.5, 0.0, 1.0),
        )


def test_density_perturbation_grid_complex():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    z = np.array([0.0, 1.0])
    # Complex-valued density perturbation representing a harmonic amplitude
    delta_rho = np.ones((2, 2, 2), dtype=np.complex128) * (1.5 + 2.5j)

    grid = DensityPerturbationGrid(
        x_m=x, y_m=y, z_m=z, delta_density_kg_m3=delta_rho, cell_size_m=1.0
    )
    source = grid.as_volume_element_source()
    assert np.iscomplexobj(source.density_kg_m3)
    assert np.allclose(source.density_kg_m3, 1.5 + 2.5j)

    # Weighted point source weights must also be complex128
    w_source = source.as_weighted_source()
    assert w_source.weights_kg.dtype == np.complex128
    # Total complex mass: (1.5 + 2.5j) * 1.0 * 8 cells = 12.0 + 20.0j
    assert np.isclose(w_source.weights_kg.sum(), 12.0 + 20.0j)


def test_displacement_grid_basic():
    # Regular 3D grid setup (Nx, Ny, Nz >= 3 required)
    x = np.linspace(0.0, 2.0, 3)
    y = np.linspace(0.0, 2.0, 3)
    z = np.linspace(0.0, 2.0, 3)

    # Set up linear displacement field u = (a*x, 0, 0)
    # div(u) = a. Let a = 0.5.
    disp = np.zeros((3, 3, 3, 3))
    for i, xi in enumerate(x):
        disp[i, :, :, 0] = 0.5 * xi

    disp_grid = DisplacementGrid(x_m=x, y_m=y, z_m=z, displacement_m=disp)

    # delta_rho = -rho0 * div(u) = -2000.0 * 0.5 = -1000.0
    rho0 = 2000.0
    density_grid = disp_grid.as_density_perturbation_grid(rho0)
    assert isinstance(density_grid, DensityPerturbationGrid)
    assert np.allclose(density_grid.delta_density_kg_m3, -1000.0)


def test_displacement_grid_complex():
    x = np.linspace(0.0, 2.0, 3)
    y = np.linspace(0.0, 2.0, 3)
    z = np.linspace(0.0, 2.0, 3)

    # Complex displacement representing harmonic amplitude: u = ( (0.5+0.2j)*x, 0, 0 )
    # div(u) = 0.5 + 0.2j.
    disp = np.zeros((3, 3, 3, 3), dtype=np.complex128)
    for i, xi in enumerate(x):
        disp[i, :, :, 0] = (0.5 + 0.2j) * xi

    disp_grid = DisplacementGrid(x_m=x, y_m=y, z_m=z, displacement_m=disp)

    rho0 = 2000.0
    # delta_rho = -2000.0 * (0.5 + 0.2j) = -1000.0 - 400.0j
    density_grid = disp_grid.as_density_perturbation_grid(rho0)
    assert np.iscomplexobj(density_grid.delta_density_kg_m3)
    assert np.allclose(density_grid.delta_density_kg_m3, -1000.0 - 400.0j)


def test_displacement_grid_validation():
    x = np.linspace(0.0, 2.0, 3)
    y = np.linspace(0.0, 2.0, 3)
    z = np.linspace(0.0, 2.0, 3)
    disp = np.zeros((3, 3, 3, 3))

    # Axis length < 3
    with pytest.raises(ValueError, match="length must be >= 3"):
        DisplacementGrid(
            x_m=np.array([0.0, 1.0]),
            y_m=y,
            z_m=z,
            displacement_m=np.zeros((2, 3, 3, 3)),
        )

    # Invalid rho0 validation
    disp_grid = DisplacementGrid(x_m=x, y_m=y, z_m=z, displacement_m=disp)
    with pytest.raises(ValueError, match="must be 3-D"):
        disp_grid.as_density_perturbation_grid(np.array([1000.0]))
    with pytest.raises(TypeError, match="must be real-valued"):
        disp_grid.as_density_perturbation_grid(1000.0 + 1j)
    with pytest.raises(ValueError, match="must be non-negative"):
        disp_grid.as_density_perturbation_grid(-1000.0)
    with pytest.raises(ValueError, match="must be finite"):
        disp_grid.as_density_perturbation_grid(float("inf"))


def test_external_field_gravity_acceleration():
    # Physical consistency check: gravity acceleration of a dense cube
    # Cell centers: [-1, 1]^3 with spacing 0.5 -> Volume range [-1.25, 1.25]^3
    # Total volume = 2.5^3 = 15.625 m^3, Total Mass M = 15625 kg
    # We evaluate acceleration at a distant target point on the x-axis: (20.0, 0.0, 0.0)
    # Analytic point mass acceleration:
    # G_SI * M / r^2 = G_SI * 15625 / 20.0^2 = 39.0625 * G_SI
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(-1.0, 1.0, 5)
    z = np.linspace(-1.0, 1.0, 5)
    delta_rho = np.ones((5, 5, 5)) * 1000.0

    # Grid cell size = 0.5 (implicitly estimated, since coordinates spacing is 0.5)
    # np.gradient of linspace(-1, 1, 5) is [0.5, 0.5, 0.5, 0.5, 0.5]
    grid = DensityPerturbationGrid(x_m=x, y_m=y, z_m=z, delta_density_kg_m3=delta_rho)
    source = grid.as_volume_element_source()
    weighted_source = source.as_weighted_source()

    # Total mass matches: 15625 kg
    assert np.isclose(weighted_source.weights_kg.sum(), 15625.0)

    # Compute acceleration
    target = TargetPoints(positions_m=np.array([[20.0, 0.0, 0.0]]))
    acc = point_acceleration(source=weighted_source, targets=target)

    # Direction should be in -x direction (attracted to the cube center)
    assert acc[0, 0] < 0
    assert np.isclose(acc[0, 1], 0.0, atol=1e-12)
    assert np.isclose(acc[0, 2], 0.0, atol=1e-12)

    # Point-like approximation comparison
    # acceleration magnitude should be close to G_SI * 15625 / 20^2 = 39.0625 * G_SI
    expected_acc_magnitude = 39.0625 * G_SI
    acc_magnitude = np.linalg.norm(acc[0])
    # The grid representation should agree closely with the point-mass limit
    # (difference is expected due to finite spatial extent, but should be small: <0.5%)
    assert np.isclose(acc_magnitude, expected_acc_magnitude, rtol=5e-3)


def test_density_perturbation_grid_non_uniform_rectilinear():
    # Non-uniform rectilinear grid axes
    # dx = gradient([0.0, 1.0, 3.0]) = [1.0, 1.5, 2.0]
    # dy = gradient([0.0, 2.0, 5.0]) = [2.0, 2.5, 3.0]
    # dz = gradient([0.0, 1.0, 4.0]) = [1.0, 2.0, 3.0]
    x = np.array([0.0, 1.0, 3.0])
    y = np.array([0.0, 2.0, 5.0])
    z = np.array([0.0, 1.0, 4.0])

    delta_rho = np.ones((3, 3, 3)) * 1.5

    grid = DensityPerturbationGrid(
        x_m=x, y_m=y, z_m=z, delta_density_kg_m3=delta_rho, cell_size_m=None
    )
    source = grid.as_volume_element_source()

    # Expected volume components
    dx_expected = np.array([1.0, 1.5, 2.0])
    dy_expected = np.array([2.0, 2.5, 3.0])
    dz_expected = np.array([1.0, 2.0, 3.0])

    expected_v_3d = (
        dx_expected[:, None, None]
        * dy_expected[None, :, None]
        * dz_expected[None, None, :]
    )
    expected_v = expected_v_3d.ravel()

    assert np.allclose(source.volumes_m3, expected_v)

    # Total expected mass
    expected_total_mass = (delta_rho.ravel() * expected_v).sum()
    weighted_source = source.as_weighted_source()
    assert np.isclose(weighted_source.weights_kg.sum(), expected_total_mass)


def test_displacement_grid_non_uniform_rectilinear():
    # Non-uniform axes (Nx, Ny, Nz >= 3 required)
    x = np.array([0.0, 1.0, 3.0, 6.0])
    y = np.array([0.0, 2.0, 5.0, 9.0])
    z = np.array([0.0, 1.5, 4.0, 7.5])

    # Linear displacement field: u = (a*x, b*y, c*z)
    # div(u) = a + b + c. Let a = 0.1, b = 0.2, c = 0.3 -> div(u) = 0.6.
    a, b, c = 0.1, 0.2, 0.3

    # Generate 4D displacement array of shape (Nx, Ny, Nz, 3)
    nx, ny, nz = len(x), len(y), len(z)
    disp = np.zeros((nx, ny, nz, 3))

    for i, xi in enumerate(x):
        disp[i, :, :, 0] = a * xi
    for j, yj in enumerate(y):
        disp[:, j, :, 1] = b * yj
    for k, zk in enumerate(z):
        disp[:, :, k, 2] = c * zk

    disp_grid = DisplacementGrid(x_m=x, y_m=y, z_m=z, displacement_m=disp)

    rho0 = 1500.0
    # Expected delta_rho = -1500.0 * (0.1 + 0.2 + 0.3) = -900.0 kg/m^3
    density_grid = disp_grid.as_density_perturbation_grid(rho0)

    # Central difference (np.gradient with edge_order=2) recovers exact derivatives
    # for a linear field, even on a non-uniform grid.
    assert np.allclose(density_grid.delta_density_kg_m3, -900.0)
