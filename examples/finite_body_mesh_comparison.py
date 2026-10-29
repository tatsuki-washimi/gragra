"""Compare two linearized gravity representations on a finite tetra body.

The example uses a two-material box split into tetrahedra.  The reported
numbers are G-free geometric responses.  The two centroid quadratures
converge under refinement but need not agree exactly at a finite mesh size.
"""

import numpy as np

from gragra.adapters.finite_body_mesh import (
    cell_density_from_group_tags,
    to_unstructured_displacement_field,
)
from gragra.adapters.meshio_adapter import VolumeMesh
from gragra.fields.finite_body import (
    compare_tetra_linearized_representations,
    piecewise_tetra_linear_sources,
)

_VTK6 = np.array(
    [
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
        [0, 5, 1, 6],
    ],
    dtype=np.int64,
)


def _two_material_box(n: int = 4) -> VolumeMesh:
    axis = np.linspace(-1.0, 1.0, n + 1)
    vertices = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )

    def vertex_index(i: int, j: int, k: int) -> int:
        return (i * (n + 1) + j) * (n + 1) + k

    tetra = []
    tags = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                corners = np.array(
                    [
                        vertex_index(i, j, k),
                        vertex_index(i + 1, j, k),
                        vertex_index(i + 1, j + 1, k),
                        vertex_index(i, j + 1, k),
                        vertex_index(i, j, k + 1),
                        vertex_index(i + 1, j, k + 1),
                        vertex_index(i + 1, j + 1, k + 1),
                        vertex_index(i, j + 1, k + 1),
                    ],
                    dtype=np.int64,
                )
                tetra.extend(corners[_VTK6])
                tags.extend([1 if i < n // 2 else 2] * 6)
    return VolumeMesh(
        vertices,
        (("tetra", np.asarray(tetra)),),
        cell_groups=(np.asarray(tags),),
        group_names={"left": 1, "right": 2},
    )


def main() -> None:
    mesh = _two_material_box()
    affine = np.array(
        [
            [0.03, 0.01, -0.02],
            [-0.015, -0.02, 0.005],
            [0.012, 0.018, 0.025],
        ]
    )
    displacement = np.array([0.07, -0.04, 0.05]) + mesh.vertices_m @ affine.T
    field = to_unstructured_displacement_field(mesh, displacement)
    density = cell_density_from_group_tags(mesh, {1: 2.0, 2: 5.0})
    sources = piecewise_tetra_linear_sources(field, density)
    targets = np.array([[3.0, -2.5, 2.25], [-2.75, 2.8, 2.4]])

    bulk_mass = sources.bulk_source.as_weighted_source().weights_kg
    interface_mass = sources.interface_source.weights_kg
    mass_balance = float(np.sum(bulk_mass) + np.sum(interface_mass))
    print(f"cells={len(sources.mass_kg)} interfaces={len(interface_mass)}")
    print(f"mass_kg={np.sum(sources.mass_kg):.16g}")
    print(f"linearized_mass_balance_kg={mass_balance:.3e}")
    for observable in ("potential", "acceleration", "gradient"):
        report = compare_tetra_linearized_representations(
            sources,
            targets,
            observable=observable,
            chunk_size=37,
        )
        print(
            f"{observable}: "
            f"max_abs_residual={np.max(report.abs_residual):.6e}, "
            f"max_relative_residual={np.max(report.rel_residual):.6e}"
        )


if __name__ == "__main__":
    main()
