"""Example 06: Continuous source types — volume elements and surface mass sheets.

Demonstrates how to construct ``VolumeElementSource`` and
``SurfaceMassSheetSource``, convert them to the canonical
``WeightedPointSource`` via ``.as_weighted_source()``, and compute
gravitational observables with ``point_acceleration``.

Note on complex outputs
-----------------------
When ``density_kg_m3`` or ``normal_displacement_m`` is complex, the returned
acceleration is complex128.  Complex outputs represent linear complex
amplitudes or transfer-function-like responses, not directly real-valued
time-domain gravitational fields.
"""

import numpy as np

from gragra import G_SI, point_acceleration
from gragra.gravity_sources import SurfaceMassSheetSource, VolumeElementSource


def _volume_element_demo() -> None:
    """Volume element source: uniform-density sphere discretised on a grid."""
    print("=== VolumeElementSource ===")

    # Discretise a 1 m radius sphere into a coarse 5x5x5 Cartesian grid
    n = 5
    coords_1d = np.linspace(-0.9, 0.9, n)
    xx, yy, zz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    positions_all = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    # Keep only points inside the unit sphere
    r2 = np.sum(positions_all**2, axis=-1)
    inside = r2 < 1.0
    positions = positions_all[inside]

    # Uniform voxel volume (dx)^3
    dx = coords_1d[1] - coords_1d[0]
    volumes = np.full(len(positions), dx**3)

    # Density: 1000 kg/m^3 (water-like)
    density_kg_m3 = 1000.0

    source = VolumeElementSource(
        positions_m=positions,
        volumes_m3=volumes,
        density_kg_m3=density_kg_m3,
    )

    print(f"  Number of volume elements : {len(source.positions_m)}")
    total_mass = np.sum(source.volumes_m3) * density_kg_m3
    print(f"  Total mass (approx)       : {total_mass:.3e} kg")
    print(f"  positions_m.shape         : {source.positions_m.shape}")
    print(f"  volumes_m3.shape          : {source.volumes_m3.shape}")
    print(f"  positions_m writeable     : {source.positions_m.flags.writeable}")

    # Convert to WeightedPointSource and compute acceleration at a remote target
    weighted = source.as_weighted_source()
    print(f"  weights_kg.shape          : {weighted.weights_kg.shape}")
    print(f"  weights_kg dtype          : {weighted.weights_kg.dtype}")

    target = [[5.0, 0.0, 0.0]]
    acc = point_acceleration(weighted, target)

    print(f"  Acceleration shape        : {acc.shape}")
    print(f"  Acceleration at (5,0,0)   : {acc[0]}")
    print(f"  |a| at (5,0,0)            : {np.linalg.norm(acc[0]):.5e} m s^-2")
    print(f"  acc writeable             : {acc.flags.writeable}")

    assert acc.shape == (1, 3), f"Unexpected shape: {acc.shape}"
    assert not acc.flags.writeable, "Observable return must be read-only"
    print()


def _surface_mass_sheet_real_demo() -> None:
    """Surface mass sheet source: flat disk in the x-y plane."""
    print("=== SurfaceMassSheetSource (real displacement) ===")

    # Small disk: 8 triangular-centroid-like patches arranged in a ring
    n_patches = 8
    angles = np.linspace(0, 2 * np.pi, n_patches, endpoint=False)
    r_disk = 0.5  # m
    positions = np.column_stack(
        [
            r_disk * np.cos(angles),
            r_disk * np.sin(angles),
            np.zeros(n_patches),
        ]
    )

    # All normals point in +z direction
    normals = np.zeros((n_patches, 3))
    normals[:, 2] = 1.0

    # Uniform patch area
    patch_area = np.pi * r_disk**2 / n_patches
    areas = np.full(n_patches, patch_area)

    # 1 mm normal displacement, granite-like density
    normal_displacement_m = 1e-3  # m
    density_kg_m3 = 2700.0  # kg/m^3

    source = SurfaceMassSheetSource(
        positions_m=positions,
        normals=normals,
        areas_m2=areas,
        normal_displacement_m=normal_displacement_m,
        density_kg_m3=density_kg_m3,
    )

    print(f"  Number of patches         : {len(source.positions_m)}")
    print(f"  positions_m.shape         : {source.positions_m.shape}")
    print(f"  areas_m2.shape            : {source.areas_m2.shape}")
    print(f"  normals.shape             : {source.normals.shape}")
    print(f"  positions_m writeable     : {source.positions_m.flags.writeable}")

    weighted = source.as_weighted_source()
    print(f"  weights_kg.shape          : {weighted.weights_kg.shape}")
    print(f"  weights_kg dtype          : {weighted.weights_kg.dtype}")

    target = [[0.0, 0.0, 3.0]]
    acc = point_acceleration(weighted, target)

    print(f"  Acceleration shape        : {acc.shape}")
    print(f"  Acceleration at (0,0,3)   : {acc[0]}")
    print(f"  acc writeable             : {acc.flags.writeable}")

    assert acc.shape == (1, 3), f"Unexpected shape: {acc.shape}"
    assert not acc.flags.writeable, "Observable return must be read-only"
    # The symmetry of the ring patch (8 evenly-spaced patches, on-axis target)
    # makes the x/y components cancel; the z-component dominates.
    tol = abs(acc[0, 2]) * 1e-6
    assert abs(acc[0, 0]) < tol, "x-component should cancel by symmetry"
    assert abs(acc[0, 1]) < tol, "y-component should cancel by symmetry"
    print()


def _surface_mass_sheet_complex_demo() -> None:
    """Surface mass sheet source: complex displacement (coherent amplitude).

    Complex outputs represent linear complex amplitudes or transfer-function-like
    responses, not directly real-valued time-domain gravitational fields.
    """
    print("=== SurfaceMassSheetSource (complex displacement) ===")

    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    areas = np.array([1.0, 1.0])  # m^2

    # Complex displacement: amplitude + phase
    normal_displacement_m = 1e-3 * np.exp(1j * np.pi / 4)  # complex

    source = SurfaceMassSheetSource(
        positions_m=positions,
        normals=normals,
        areas_m2=areas,
        normal_displacement_m=normal_displacement_m,
        density_kg_m3=2700.0,
    )

    weighted = source.as_weighted_source()
    print(f"  weights_kg dtype          : {weighted.weights_kg.dtype}")  # complex128

    target = [[0.5, 0.0, 5.0]]
    acc = point_acceleration(weighted, target)

    print(f"  Acceleration shape        : {acc.shape}")
    print(f"  Acceleration dtype        : {acc.dtype}")  # complex128
    print(f"  Acceleration at (0.5,0,5) : {acc[0]}")
    print(f"  acc writeable             : {acc.flags.writeable}")

    assert acc.dtype == np.complex128, f"Expected complex128, got {acc.dtype}"
    assert not acc.flags.writeable, "Observable return must be read-only"
    print()


def _check_g_si() -> None:
    """Quick sanity: G_SI value from CODATA."""
    assert np.isclose(G_SI, 6.67430e-11, rtol=1e-4), f"Unexpected G_SI: {G_SI}"


def main() -> None:
    _check_g_si()
    _volume_element_demo()
    _surface_mass_sheet_real_demo()
    _surface_mass_sheet_complex_demo()
    print("Example 06 completed successfully.")


if __name__ == "__main__":
    main()
