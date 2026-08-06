"""Meshio adapter for surface and volume meshes."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from gragra.fields.surface_orientation import (
    check_triangle_orientation,
    reorient_triangles,
)
from gragra.sources.surface import SurfaceMassSheetSource
from gragra.sources.volume import VolumeElementSource

if TYPE_CHECKING:
    from gragra.fields.surface_orientation import OrientationReport


@dataclass(frozen=True)
class TriangleSurfaceMesh:
    """Triangle surface mesh representation.

    Parameters
    ----------
    vertices_m : np.ndarray
        Array of vertex coordinates of shape (Nv, 3) in meters.
    triangles : np.ndarray
        Array of triangle vertex indices of shape (Nt, 3).
    """

    vertices_m: np.ndarray
    triangles: np.ndarray

    def __post_init__(self):
        # Validate vertices
        try:
            v = np.asarray(self.vertices_m, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"vertices_m must be numeric convertible to float64: {e}"
            ) from e

        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"vertices_m must have shape (Nv, 3), got {v.shape}")

        if not np.isfinite(v).all():
            raise ValueError("vertices_m must contain only finite numbers")

        # Validate triangles
        raw_t = np.asarray(self.triangles)
        if raw_t.size > 0:
            if isinstance(self.triangles, bool) or raw_t.dtype.kind == "b":
                raise TypeError("triangles must not contain bool values")
            if raw_t.dtype.kind not in "iu":
                raise TypeError("triangles must be of integer dtype")

        try:
            t = np.asarray(self.triangles, dtype=np.int64)
        except (ValueError, TypeError) as e:
            raise TypeError(f"triangles must be integer convertible: {e}") from e

        if t.ndim != 2 or t.shape[1] != 3:
            raise ValueError(f"triangles must have shape (Nt, 3), got {t.shape}")

        if (t < 0).any():
            raise ValueError("triangles must not contain negative indices")

        nv = len(v)
        if (t >= nv).any():
            raise ValueError(f"triangle index out of bounds for {nv} vertices")

        # Check for degenerate zero-area triangles
        if len(t) > 0:
            a_pts = v[t[:, 0]]
            b_pts = v[t[:, 1]]
            c_pts = v[t[:, 2]]
            cross = np.cross(b_pts - a_pts, c_pts - a_pts)
            norms = np.linalg.norm(cross, axis=-1)
            if (norms <= 1e-15).any():
                raise ValueError("Mesh contains degenerate zero-area triangles")

        # Make defensive copies and read-only
        v = v.copy()
        t = t.copy()
        v.setflags(write=False)
        t.setflags(write=False)

        object.__setattr__(self, "vertices_m", v)
        object.__setattr__(self, "triangles", t)

    def check_orientation(self) -> "OrientationReport":
        """Diagnose the winding consistency and closedness of this mesh.

        Delegates to the numpy-only core diagnostic
        :func:`gragra.fields.surface_orientation.check_triangle_orientation`.
        A consistent winding is the caller's responsibility; this method only
        reports the diagnosis and does not modify the mesh.

        Returns
        -------
        OrientationReport
            Diagnostic flags and counts for the mesh winding/topology.
        """
        return check_triangle_orientation(self.vertices_m, self.triangles)

    def with_consistent_orientation(self) -> "TriangleSurfaceMesh":
        """Return a new mesh with consistently reoriented triangles.

        This method attempts to resolve winding inconsistencies in the mesh
        topologically and returns a new mesh instance.

        .. note::
           **Asymmetry with Diagnostics**:
           Unlike :meth:`check_orientation`, which acts as a non-blocking diagnostic
           by returning an ``OrientationReport`` without raising errors even if
           the mesh is topologically invalid, this method acts as a strict guard.
           If winding consistency cannot be resolved (due to non-manifold edges or
           non-orientable topology like a Mobius strip), it raises a ``ValueError``.

        Returns
        -------
        TriangleSurfaceMesh
            A new mesh instance with reoriented triangles.
        """
        return TriangleSurfaceMesh(
            self.vertices_m,
            reorient_triangles(self.triangles),
        )


def from_meshio(
    path: str | Path, *, length_scale_m: float = 1.0
) -> TriangleSurfaceMesh:
    """Read a mesh via meshio and extract triangle cells.

    Meshio library is lazily imported.
    """
    if isinstance(length_scale_m, bool):
        raise TypeError("length_scale_m must not be bool")
    if not isinstance(length_scale_m, (int, float, np.integer, np.floating)):
        raise TypeError("length_scale_m must be numeric")
    if not np.isfinite(length_scale_m):
        raise ValueError("length_scale_m must be finite")

    import meshio

    mesh = meshio.read(str(path))

    triangle_cells = []
    for cell in mesh.cells:
        if cell.type == "triangle":
            triangle_cells.append(cell.data)

    if not triangle_cells:
        raise ValueError("No triangle cells found in the mesh file")

    triangles = np.concatenate(triangle_cells, axis=0)
    vertices = mesh.points * length_scale_m

    if vertices.shape[1] != 3:
        if vertices.shape[1] == 2:
            vertices = np.pad(vertices, ((0, 0), (0, 1)), mode="constant")
        else:
            raise ValueError(f"Expected 3D vertices, got shape {vertices.shape}")

    return TriangleSurfaceMesh(vertices_m=vertices, triangles=triangles)


def centroids_normals_areas(
    mesh: TriangleSurfaceMesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate centroids, unit normals, and areas of triangles.

    Returns tuple (centroids, unit_normals, areas_m2) of read-only arrays.
    """
    v = mesh.vertices_m
    t = mesh.triangles

    if len(t) == 0:
        empty3 = np.zeros((0, 3))
        empty1 = np.zeros(0)
        empty3.setflags(write=False)
        empty1.setflags(write=False)
        return empty3, empty3, empty1

    a_pts = v[t[:, 0]]
    b_pts = v[t[:, 1]]
    c_pts = v[t[:, 2]]

    centroids = (a_pts + b_pts + c_pts) / 3.0

    cross = np.cross(b_pts - a_pts, c_pts - a_pts)
    lengths = np.linalg.norm(cross, axis=-1, keepdims=True)

    unit_normals = cross / lengths
    areas = lengths.squeeze(axis=-1) / 2.0

    centroids = centroids.copy()
    unit_normals = unit_normals.copy()
    areas = areas.copy()

    centroids.setflags(write=False)
    unit_normals.setflags(write=False)
    areas.setflags(write=False)

    return centroids, unit_normals, areas


def to_surface_mass_sheet(
    mesh: TriangleSurfaceMesh,
    surface_density_kg_m2: float | complex,
) -> SurfaceMassSheetSource:
    """Convert TriangleSurfaceMesh to SurfaceMassSheetSource.

    Note that the surface mass density (kg/m^2) is mapped to density_kg_m3 (kg/m^3)
    in SurfaceMassSheetSource with normal_displacement_m = 1.0 m internally.
    This preserves the total physical mass (weights_kg = density * displacement * area =
    surface_density * area) while matching the volumetric mass density API.
    """
    centroids, normals, areas = centroids_normals_areas(mesh)

    return SurfaceMassSheetSource(
        positions_m=centroids,
        normals=normals,
        areas_m2=areas,
        normal_displacement_m=1.0,
        density_kg_m3=surface_density_kg_m2,
    )


# hexahedron decomposition
HEX_TETS_0_6 = np.array(
    [
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
        [0, 5, 1, 6],
    ]
)


@dataclass(frozen=True)
class VolumeMesh:
    """Volume mesh representation.

    Unlike TriangleSurfaceMesh, degenerate zero-volume cell validation is
    performed during volume calculation in cell_centroids_volumes(), rather
    than at class construction time, to avoid premature rejection of empty
    or complex intermediate meshes.

    Parameters
    ----------
    vertices_m : np.ndarray
        Array of vertex coordinates of shape (Nv, 3) in meters.
    cell_blocks : tuple[tuple[str, np.ndarray], ...]
        Tuple of (cell_type, connectivity) pairs.
    cell_groups : tuple[np.ndarray, ...] | None, optional
        Tuple of integer group tags for each cell in each block.
    group_names : Mapping[str, int] | None, optional
        Mapping from group name to integer tag.
    """

    vertices_m: np.ndarray
    cell_blocks: tuple[tuple[str, np.ndarray], ...]
    cell_groups: tuple[np.ndarray, ...] | None = None
    group_names: Mapping[str, int] | None = None

    def __post_init__(self):
        # Validate vertices
        try:
            v = np.asarray(self.vertices_m, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"vertices_m must be numeric convertible to float64: {e}"
            ) from e

        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"vertices_m must have shape (Nv, 3), got {v.shape}")

        if not np.isfinite(v).all():
            raise ValueError("vertices_m must contain only finite numbers")

        nv = len(v)

        # Validate cell_blocks
        validated_blocks = []
        allowed_types = {"tetra", "hexahedron"}
        for block in self.cell_blocks:
            if not isinstance(block, (tuple, list)) or len(block) != 2:
                raise TypeError(
                    "Each cell block must be a (cell_type, connectivity) pair, "
                    f"got {block}"
                )

            cell_type, conn = block
            if cell_type not in allowed_types:
                raise ValueError(
                    f"Unsupported cell type: '{cell_type}'. "
                    f"Allowed types are {allowed_types}"
                )

            raw_c = np.asarray(conn)
            if raw_c.size > 0:
                if isinstance(conn, bool) or raw_c.dtype.kind == "b":
                    raise TypeError("cell connectivity must not contain bool values")
                if raw_c.dtype.kind not in "iu":
                    raise TypeError("cell connectivity must be of integer dtype")

            try:
                c = np.asarray(conn, dtype=np.int64)
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"cell connectivity must be integer convertible: {e}"
                ) from e

            if cell_type == "tetra" and (c.ndim != 2 or c.shape[1] != 4):
                raise ValueError(
                    f"tetra cell connectivity must have shape (Nc, 4), got {c.shape}"
                )
            if cell_type == "hexahedron" and (c.ndim != 2 or c.shape[1] != 8):
                raise ValueError(
                    "hexahedron cell connectivity must have shape (Nc, 8), "
                    f"got {c.shape}"
                )

            if c.size > 0:
                if (c < 0).any():
                    raise ValueError(
                        "cell connectivity must not contain negative indices"
                    )
                if (c >= nv).any():
                    raise ValueError(f"cell index out of bounds for {nv} vertices")

            c = c.copy()
            c.setflags(write=False)
            validated_blocks.append((cell_type, c))

        # Validate cell_groups
        validated_groups = None
        if self.cell_groups is not None:
            if not isinstance(self.cell_groups, (tuple, list)):
                raise TypeError("cell_groups must be a tuple or list of arrays")
            if len(self.cell_groups) != len(self.cell_blocks):
                raise ValueError(
                    "cell_groups length must match cell_blocks length: "
                    f"{len(self.cell_groups)} != {len(self.cell_blocks)}"
                )

            val_groups_list = []
            for i, group in enumerate(self.cell_groups):
                raw_g = np.asarray(group)
                if raw_g.size > 0:
                    if isinstance(group, bool) or raw_g.dtype.kind == "b":
                        raise TypeError(
                            "cell_groups elements must not contain bool values"
                        )
                    if raw_g.dtype.kind not in "iu":
                        raise TypeError("cell_groups elements must be of integer dtype")
                try:
                    grp = np.asarray(group, dtype=np.int64)
                except (ValueError, TypeError) as e:
                    raise TypeError(
                        f"cell_groups elements must be integer convertible: {e}"
                    ) from e

                expected_len = len(validated_blocks[i][1])
                if grp.shape != (expected_len,):
                    raise ValueError(
                        f"cell_groups[{i}] must have shape ({expected_len},), "
                        f"got {grp.shape}"
                    )

                grp = grp.copy()
                grp.setflags(write=False)
                val_groups_list.append(grp)

            validated_groups = tuple(val_groups_list)

        # Validate group_names
        val_group_names = None
        if self.group_names is not None:
            val_group_names = MappingProxyType(dict(self.group_names))

        # Defensive copy of vertices
        v = v.copy()
        v.setflags(write=False)

        object.__setattr__(self, "vertices_m", v)
        object.__setattr__(self, "cell_blocks", tuple(validated_blocks))
        object.__setattr__(self, "cell_groups", validated_groups)
        object.__setattr__(self, "group_names", val_group_names)


def cell_centroids_volumes(mesh: VolumeMesh) -> tuple[np.ndarray, np.ndarray]:
    """Calculate centroids and volumes of volume cells.

    Returns tuple (centroids, volumes) of read-only arrays.

    Raises
    ------
    ValueError
        If the mesh contains degenerate zero-volume cells (volume <= 1e-15).
        Note that this degenerate cell check is intentionally performed here
        rather than during VolumeMesh construction.
    """
    v = mesh.vertices_m

    # Calculate total number of cells to preallocate
    total_cells = sum(len(block[1]) for block in mesh.cell_blocks)
    if total_cells == 0:
        empty_c = np.zeros((0, 3))
        empty_v = np.zeros(0)
        empty_c.setflags(write=False)
        empty_v.setflags(write=False)
        return empty_c, empty_v

    all_centroids = []
    all_volumes = []

    for cell_type, conn in mesh.cell_blocks:
        if len(conn) == 0:
            continue

        if cell_type == "tetra":
            a = v[conn[:, 0]]
            b = v[conn[:, 1]]
            c = v[conn[:, 2]]
            d = v[conn[:, 3]]

            # Volume = 1/6 * |det([b-a, c-a, d-a])|
            # det([u, v, w]) = u . (v x w)
            u = b - a
            v_vec = c - a
            w = d - a
            cross = np.cross(v_vec, w)
            det = np.sum(u * cross, axis=-1)
            vol = np.abs(det) / 6.0

            # Centroid = (a+b+c+d)/4
            cent = (a + b + c + d) / 4.0

            all_centroids.append(cent)
            all_volumes.append(vol)

        elif cell_type == "hexahedron":
            # hex nodes: (Nc, 8, 3)
            # We partition each hex into 6 tetras using HEX_TETS_0_6
            hex_pts = v[conn]  # (Nc, 8, 3)

            # For each of the 6 tetras, shape will be (Nc, 6, 4, 3)
            # HEX_TETS_0_6 is shape (6, 4)
            tet_nodes = hex_pts[:, HEX_TETS_0_6]  # (Nc, 6, 4, 3)

            a = tet_nodes[:, :, 0]  # (Nc, 6, 3)
            b = tet_nodes[:, :, 1]
            c = tet_nodes[:, :, 2]
            d = tet_nodes[:, :, 3]

            u = b - a
            v_vec = c - a
            w = d - a
            cross = np.cross(v_vec, w)
            det = np.sum(u * cross, axis=-1)  # (Nc, 6)
            vol_t = np.abs(det) / 6.0  # (Nc, 6)

            # Centroid of each tetra: (Nc, 6, 3)
            cent_t = (a + b + c + d) / 4.0

            # Total volume for each hex
            vol_hex = np.sum(vol_t, axis=1)  # (Nc,)

            # Avoid division by zero for degenerate hexes
            # (will raise ValueError later anyway)
            vol_hex_safe = np.where(vol_hex == 0.0, 1.0, vol_hex)

            # Centroid = sum(vol_t * cent_t) / vol_hex
            # vol_t[:, :, np.newaxis] is (Nc, 6, 1)
            # cent_t is (Nc, 6, 3)
            sum_vol_cent = np.sum(vol_t[:, :, np.newaxis] * cent_t, axis=1)
            cent_hex = sum_vol_cent / vol_hex_safe[:, np.newaxis]  # (Nc, 3)

            all_centroids.append(cent_hex)
            all_volumes.append(vol_hex)

    centroids = np.concatenate(all_centroids, axis=0)
    volumes = np.concatenate(all_volumes, axis=0)

    # Check for degenerate zero-volume cells
    if (volumes <= 1e-15).any():
        raise ValueError("Mesh contains degenerate zero-volume cells")

    centroids = centroids.copy()
    volumes = volumes.copy()
    centroids.setflags(write=False)
    volumes.setflags(write=False)

    return centroids, volumes


def to_volume_element_source(
    mesh: VolumeMesh,
    density_kg_m3: float | complex | np.ndarray,
) -> VolumeElementSource:
    """Convert VolumeMesh to VolumeElementSource."""
    centroids, volumes = cell_centroids_volumes(mesh)
    return VolumeElementSource(
        positions_m=centroids,
        volumes_m3=volumes,
        density_kg_m3=density_kg_m3,
    )


def from_meshio_volume(
    path: str | Path,
    *,
    length_scale_m: float = 1.0,
    cell_types: tuple[str, ...] = ("tetra", "hexahedron"),
) -> VolumeMesh:
    """Read a volume mesh via meshio and extract cells.

    Meshio library is lazily imported.
    """
    if isinstance(length_scale_m, bool):
        raise TypeError("length_scale_m must not be bool")
    if not isinstance(length_scale_m, (int, float, np.integer, np.floating)):
        raise TypeError("length_scale_m must be numeric")
    if not np.isfinite(length_scale_m):
        raise ValueError("length_scale_m must be finite")

    allowed_types = {"tetra", "hexahedron"}
    for ct in cell_types:
        if ct not in allowed_types:
            raise ValueError(
                f"Unsupported cell_types parameter: '{ct}'. "
                f"Allowed types are {allowed_types}"
            )

    import meshio

    mesh = meshio.read(str(path))

    cell_blocks = []
    cell_groups_list = []
    has_groups = False

    physical_data_list = mesh.cell_data.get("gmsh:physical")

    for idx, cell in enumerate(mesh.cells):
        if cell.type in cell_types:
            cell_blocks.append((cell.type, cell.data))

            # Extract gmsh physical tags for this cell block if present
            if physical_data_list is not None and idx < len(physical_data_list):
                cell_groups_list.append(physical_data_list[idx])
                has_groups = True
            else:
                cell_groups_list.append(np.zeros(len(cell.data), dtype=np.int64))

    if not cell_blocks:
        raise ValueError("No supported volume cells found in the mesh file")

    vertices = mesh.points * length_scale_m

    if vertices.shape[1] != 3:
        if vertices.shape[1] == 2:
            vertices = np.pad(vertices, ((0, 0), (0, 1)), mode="constant")
        else:
            raise ValueError(f"Expected 3D vertices, got shape {vertices.shape}")

    group_names = None
    if mesh.field_data:
        group_names = {}
        for name, val in mesh.field_data.items():
            group_names[name] = int(val[0])

    cell_groups = tuple(cell_groups_list) if has_groups else None

    return VolumeMesh(
        vertices_m=vertices,
        cell_blocks=tuple(cell_blocks),
        cell_groups=cell_groups,
        group_names=group_names,
    )


def _filter_cells_by_tag(mesh: VolumeMesh, tag: int) -> tuple[VolumeMesh, bool]:
    """Filter cells matching an integer tag.

    Returns the filtered mesh and whether any cell carried the tag. Shared by
    select_group (name-based) and select_group_tag (tag-based).
    """
    new_blocks = []
    new_groups = []
    matched = False

    for i, (cell_type, conn) in enumerate(mesh.cell_blocks):
        tags = mesh.cell_groups[i]
        mask = tags == tag
        if mask.any():
            matched = True

        filtered_conn = conn[mask]
        filtered_tags = tags[mask]

        if len(filtered_conn) > 0:
            new_blocks.append((cell_type, filtered_conn))
            new_groups.append(filtered_tags)

    new_mesh = VolumeMesh(
        vertices_m=mesh.vertices_m,
        cell_blocks=tuple(new_blocks),
        cell_groups=tuple(new_groups) if new_groups else None,
        group_names=mesh.group_names,
    )
    return new_mesh, matched


def select_group(mesh: VolumeMesh, name: str) -> VolumeMesh:
    """Filter mesh cells belonging to a specific group by name."""
    if mesh.group_names is None or len(mesh.group_names) == 0:
        raise ValueError("No group names defined in mesh")

    if name not in mesh.group_names:
        raise ValueError(f"Group name '{name}' not found")

    target_tag = mesh.group_names[name]

    if mesh.cell_groups is None:
        raise ValueError("Mesh contains group names but no cell groups tags")

    new_mesh, _ = _filter_cells_by_tag(mesh, target_tag)
    return new_mesh


def select_group_tag(mesh: VolumeMesh, tag: int) -> VolumeMesh:
    """Filter mesh cells belonging to a specific group by integer tag.

    Unlike :func:`select_group`, this does not require ``group_names``; it
    matches the integer ``cell_groups`` tags directly. Raises ``ValueError`` if
    no cell carries ``tag``.
    """
    if isinstance(tag, bool) or not isinstance(tag, (int, np.integer)):
        raise TypeError("tag must be an integer")
    tag = int(tag)

    if mesh.cell_groups is None:
        raise ValueError("Mesh contains no cell group tags")

    new_mesh, matched = _filter_cells_by_tag(mesh, tag)
    if not matched:
        raise ValueError(f"Group tag {tag} not found in cell groups")

    return new_mesh
