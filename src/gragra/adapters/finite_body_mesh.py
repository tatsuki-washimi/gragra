"""Thin ``VolumeMesh`` boundary for finite tetra-body fields."""

from collections.abc import Mapping

import numpy as np

from gragra.adapters.meshio_adapter import VolumeMesh, cell_centroids_volumes
from gragra.fields.unstructured import UnstructuredDisplacementField
from gragra.gravity_sources import PointMassCloud


def _tetra_blocks(mesh: VolumeMesh) -> list[np.ndarray]:
    if not isinstance(mesh, VolumeMesh):
        raise TypeError("mesh must be a VolumeMesh")
    blocks: list[np.ndarray] = []
    for cell_type, connectivity in mesh.cell_blocks:
        if cell_type == "hexahedron" and len(connectivity):
            raise ValueError("finite-body displacement supports tetra cells only")
        if cell_type == "tetra":
            blocks.append(connectivity)
    return blocks


def to_unstructured_displacement_field(
    mesh: VolumeMesh, displacement_m
) -> UnstructuredDisplacementField:
    """Concatenate tetra blocks in input order and attach nodal displacement."""
    blocks = _tetra_blocks(mesh)
    tetra = (
        np.concatenate(blocks, axis=0) if blocks else np.empty((0, 4), dtype=np.int64)
    )
    return UnstructuredDisplacementField(mesh.vertices_m, tetra, displacement_m)


def cell_density_from_group_tags(
    mesh: VolumeMesh,
    density_by_tag_kg_m3: Mapping[int, float],
) -> np.ndarray:
    """Map every volume cell group tag to density without changing cell order.

    Supports both tetra and hexahedron cell blocks in the order defined by mesh.
    """
    if not isinstance(mesh, VolumeMesh):
        raise TypeError("mesh must be a VolumeMesh")
    if not isinstance(density_by_tag_kg_m3, Mapping):
        raise TypeError("density_by_tag_kg_m3 must be a mapping")
    if mesh.cell_groups is None:
        raise ValueError("mesh must provide cell group tags")

    normalized: dict[int, float] = {}
    for key, value in density_by_tag_kg_m3.items():
        if isinstance(key, (bool, np.bool_)) or not isinstance(key, (int, np.integer)):
            raise TypeError("density group tags must be integers")
        raw = np.asarray(value)
        if (
            isinstance(value, (bool, np.bool_))
            or raw.ndim != 0
            or raw.dtype.kind not in "iuf"
        ):
            raise TypeError("group density values must be real numeric scalars")
        density = float(raw)
        if not np.isfinite(density):
            raise ValueError("group density values must be finite")
        if density < 0.0:
            raise ValueError("group density values must be non-negative")
        normalized[int(key)] = density

    # Collect tags from all supported volume blocks (tetra and hexahedron)
    group_blocks = [
        groups
        for (cell_type, conn), groups in zip(
            mesh.cell_blocks, mesh.cell_groups, strict=True
        )
        if cell_type in {"tetra", "hexahedron"} and len(conn) > 0
    ]
    tags = np.concatenate(group_blocks) if group_blocks else np.empty(0, dtype=np.int64)
    used = {int(tag) for tag in tags}
    supplied = set(normalized)
    if used != supplied:
        missing = sorted(used - supplied)
        unused = sorted(supplied - used)
        raise ValueError(
            f"density tag mapping mismatch: missing={missing}, unused={unused}"
        )
    density = np.asarray([normalized[int(tag)] for tag in tags], dtype=np.float64)
    density.setflags(write=False)
    return density


def cell_density_nodes_from_group_tags(
    mesh: VolumeMesh,
    density_by_tag_kg_m3: Mapping[int, float],
) -> np.ndarray:
    """Map every tetra cell group tag to a per-node ``(Nc, 4)`` density array.

    Broadcasts the per-cell density from :func:`cell_density_from_group_tags`
    across each tetra's four local nodes, producing the
    ``cell_density_nodes_kg_m3`` shape that
    :func:`gragra.fields.finite_body_p1.piecewise_tetra_p1_density_sources`
    requires.

    This is the cell-constant-density case of that function's general
    per-node contract (which allows the trace to differ across a cell's four
    local nodes, and to be discontinuous cell-to-cell). It does not support
    node-level density variation within a single tagged cell; general
    node-level region mapping is future work
    (``docs/roadmap.md`` material-region contract).
    """
    density = cell_density_from_group_tags(mesh, density_by_tag_kg_m3)
    if any(cell_type != "tetra" and len(conn) for cell_type, conn in mesh.cell_blocks):
        raise ValueError("cell_density_nodes_from_group_tags: tetra cells only")
    nodes = np.repeat(density[:, None], 4, axis=1)
    nodes.setflags(write=False)
    return nodes


def cell_point_mass_cloud_from_group_tags(
    mesh: VolumeMesh,
    density_by_tag_kg_m3: Mapping[int, float],
) -> PointMassCloud:
    """Convert positive-density volume cells (tetra or hex8) to one point mass each.

    This is a positive-real-mass convenience-wrapper conversion (see the
    ``PointMass``/``PointMassCloud`` vs. ``WeightedPointSource`` source-type
    discipline), not a general signed/complex source: every tagged cell whose
    mapped density is positive contributes ``mass_kg = density * volume`` at
    the cell centroid via :func:`cell_density_from_group_tags` and
    :func:`gragra.adapters.meshio_adapter.cell_centroids_volumes`.
    """
    for cell_type, conn in mesh.cell_blocks:
        if cell_type == "hexahedron" and len(conn) > 0:
            from gragra.fields.hex8_geometry import check_hex8_jacobians

            check_hex8_jacobians(mesh.vertices_m, conn)

    density = cell_density_from_group_tags(mesh, density_by_tag_kg_m3)
    centroids, volumes = cell_centroids_volumes(mesh)
    masses = density * volumes
    positive = masses > 0.0
    if not positive.any():
        raise ValueError(
            "cell_point_mass_cloud_from_group_tags: every cell maps to a "
            "non-positive density; no point mass would remain"
        )
    return PointMassCloud(masses[positive], centroids[positive])


__all__ = [
    "to_unstructured_displacement_field",
    "cell_density_from_group_tags",
    "cell_density_nodes_from_group_tags",
    "cell_point_mass_cloud_from_group_tags",
]
