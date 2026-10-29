"""Example 08: Surface mesh adapter via meshio (optional dependency).

Demonstrates how to:

1. Generate a small triangle mesh in memory using meshio.
2. Write it to a temporary file and load it with ``from_meshio``.
3. Convert to ``SurfaceMassSheetSource`` via ``to_surface_mass_sheet``.
4. Compute gravitational acceleration with ``point_acceleration``.

This example requires the ``mesh`` extra::

    pip install "gragra[mesh]"   # installs meshio >= 5.0.0

If meshio is not installed, the example prints an informative message and
exits gracefully without raising an exception.
"""

import importlib.util
import tempfile
from pathlib import Path

import numpy as np

from gragra import G_SI, point_acceleration
from gragra.adapters import (
    TriangleSurfaceMesh,
    centroids_normals_areas,
    from_meshio,
    to_surface_mass_sheet,
)


def _build_triangle_mesh_meshio() -> None:
    """Build a small flat triangular mesh and exercise the full adapter pipeline."""
    import meshio  # type: ignore[import-untyped]

    # Vertices of two triangles forming a 1 m x 1 m square in the x-y plane
    # (z = 0)
    #
    #  3-----2
    #  |   / |
    #  |  /  |
    #  | /   |
    #  0-----1
    #
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
        ]
    )
    triangles = np.array(
        [
            [0, 1, 2],  # lower-right triangle
            [0, 2, 3],  # upper-left triangle
        ]
    )

    mesh = meshio.Mesh(
        points=vertices,
        cells=[("triangle", triangles)],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        mesh_path = Path(tmpdir) / "square.vtu"
        meshio.write(str(mesh_path), mesh)
        print(f"  Wrote mesh to temporary file: {mesh_path.name}")

        # 1. Load via from_meshio -----------------------------------------
        tri_mesh: TriangleSurfaceMesh = from_meshio(mesh_path)

        print(f"  TriangleSurfaceMesh.vertices_m shape : {tri_mesh.vertices_m.shape}")
        print(f"  TriangleSurfaceMesh.triangles shape  : {tri_mesh.triangles.shape}")
        writeable = tri_mesh.vertices_m.flags.writeable
        print(f"  vertices_m writeable                 : {writeable}")

        assert tri_mesh.vertices_m.shape == (4, 3)
        assert tri_mesh.triangles.shape == (2, 3)
        assert not tri_mesh.vertices_m.flags.writeable
        assert not tri_mesh.triangles.flags.writeable

        # 2. Inspect centroids, normals, areas ----------------------------
        centroids, normals, areas = centroids_normals_areas(tri_mesh)
        print(f"\n  Centroids shape : {centroids.shape}")  # (2, 3)
        print(f"  Normals shape   : {normals.shape}")  # (2, 3)
        print(f"  Areas shape     : {areas.shape}")  # (2,)
        print(f"  Total area      : {areas.sum():.3f} m^2 (expected 1.0)")

        assert np.isclose(areas.sum(), 1.0, rtol=1e-6), (
            f"Total area mismatch: {areas.sum()}"
        )

        # 3. Convert to SurfaceMassSheetSource ----------------------------
        # surface_density_kg_m2: concrete-like surface density
        surface_density = 2400.0  # kg/m^2

        sheet = to_surface_mass_sheet(tri_mesh, surface_density)
        pos_shape = sheet.positions_m.shape
        areas_shape = sheet.areas_m2.shape
        print(f"\n  SurfaceMassSheetSource.positions_m.shape : {pos_shape}")
        print(f"  SurfaceMassSheetSource.areas_m2.shape    : {areas_shape}")
        total_mass_kg = surface_density * areas.sum()
        print(f"  Total mass (density * area)              : {total_mass_kg:.1f} kg")

        weighted = sheet.as_weighted_source()
        print(f"  weights_kg.shape : {weighted.weights_kg.shape}")
        print(f"  weights_kg dtype : {weighted.weights_kg.dtype}")
        print(f"  weights_kg total : {weighted.weights_kg.sum():.3f} kg")

        # 4. Compute gravitational acceleration ---------------------------
        # Low-level observables consume source-frame SI coordinates directly.
        target_xyz_m = [[0.5, 0.5, 5.0]]
        acc = point_acceleration(weighted, target_xyz_m)

        print(f"\n  Acceleration shape at (0.5, 0.5, 5) : {acc.shape}")
        print(f"  Acceleration vector                 : {acc[0]}")
        acc_norm = np.linalg.norm(acc[0])
        print(f"  |a|                                 : {acc_norm:.5e} m s^-2")
        print(f"  acc writeable                       : {acc.flags.writeable}")
        print(f"  G_SI                                : {G_SI:.5e}  m^3 kg^-1 s^-2")

        assert acc.shape == (1, 3), f"Unexpected shape: {acc.shape}"
        assert not acc.flags.writeable, "Observable return must be read-only"
        # For a source directly below, z-component should dominate
        assert abs(acc[0, 2]) > 0.0, "Expected non-zero z-acceleration"

    # TemporaryDirectory is cleaned up here; no files remain in cwd
    print("\n  Temporary directory cleaned up. No files written to cwd.")


def main() -> None:
    if importlib.util.find_spec("meshio") is None:
        print(
            "meshio is not installed. Skipping example 08.\n"
            "Install the 'mesh' extra to run this example:\n"
            '    pip install "gragra[mesh]"'
        )
        return

    print(f"G_SI = {G_SI:.5e}  m^3 kg^-1 s^-2\n")
    print("=== meshio surface adapter ===")
    _build_triangle_mesh_meshio()
    print("\nExample 08 completed successfully.")


if __name__ == "__main__":
    main()
