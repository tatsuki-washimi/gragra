"""Example 09: External field grid conversion.

Demonstrates how to import bulk displacement fields on a structured regular grid,
compute density perturbations via linearized mass conservation, convert the grid
into a gragra source representation, and evaluate gravitational acceleration.
"""

import numpy as np

from gragra import point_acceleration
from gragra.fields import DisplacementGrid


def main() -> None:
    # 1. Define coordinate axes for a structured regular grid (Nx, Ny, Nz >= 3)
    x = np.linspace(-5.0, 5.0, 5)
    y = np.linspace(-5.0, 5.0, 5)
    z = np.linspace(-5.0, 5.0, 5)

    # 2. Define displacement field u(x, y, z) = (0.01 * x, 0.0, 0.0)
    # div(u) = 0.01 (constant dilatation).
    disp = np.zeros((5, 5, 5, 3))
    for i, xi in enumerate(x):
        disp[i, :, :, 0] = 0.01 * xi

    print("=== 1. Initialize DisplacementGrid ===")
    disp_grid = DisplacementGrid(x_m=x, y_m=y, z_m=z, displacement_m=disp)
    print(f"Grid size: {len(x)} x {len(y)} x {len(z)}")
    print(f"Displacement shape: {disp_grid.displacement_m.shape}")

    # 3. Compute density perturbation delta_rho = -rho0 * div(u)
    # Background density rho0 = 2000.0 kg/m^3.
    # Expected delta_rho = -2000.0 * 0.01 = -20.0 kg/m^3
    # (constant negative perturbation).
    print("\n=== 2. Compute Density Perturbations ===")
    rho0 = 2000.0
    density_grid = disp_grid.as_density_perturbation_grid(rho0)
    print(f"Density perturbation shape: {density_grid.delta_density_kg_m3.shape}")
    first_val = density_grid.delta_density_kg_m3[0, 0, 0]
    print(f"Computed delta_rho (first cell): {first_val:.1f} kg/m^3")

    # 4. Convert grid to VolumeElementSource
    # With cell_size_m=None, grid spacing is implicitly estimated from coordinates.
    # Spacing is 2.5 m in all directions -> cell volume = 2.5^3 = 15.625 m^3.
    print("\n=== 3. Convert to gragra Source ===")
    source = density_grid.as_volume_element_source()
    weighted_source = source.as_weighted_source()
    print(f"Number of volume elements: {len(source.positions_m)}")
    print(f"Total mass perturbation: {weighted_source.weights_kg.sum():.1f} kg")

    # 5. Evaluate gravitational acceleration at target point (0, 0, 100) m
    print("\n=== 4. Evaluate Gravity Acceleration ===")
    target = [0.0, 0.0, 100.0]
    acc = point_acceleration(weighted_source, target)
    print(f"Acceleration vector: {acc[0]}")
    print("Example complete.")


if __name__ == "__main__":
    main()
