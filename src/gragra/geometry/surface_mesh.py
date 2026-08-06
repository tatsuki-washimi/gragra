"""Surface mesh representation and generation helper utilities."""

import math
from dataclasses import dataclass

import numpy as np

from gragra.array_types import FloatArray


@dataclass(frozen=True)
class MeshResolution:
    """Resolution parameters for surface mesh generation.

    .. warning::
       This class is experimental/provisional.
       API stability is not guaranteed.

    Parameters
    ----------
    n_u : int
        Number of divisions in the azimuthal / primary direction.
    n_v : int
        Number of divisions in the meridional / secondary direction.
    n_cap : int
        Number of radial divisions for disc / end-cap meshes.
    """

    n_u: int = 50
    n_v: int = 50
    n_cap: int = 25

    def __post_init__(self) -> None:
        for attr in ("n_u", "n_v", "n_cap"):
            val = getattr(self, attr)
            if not isinstance(val, int) or val < 3:
                raise ValueError(
                    f"MeshResolution.{attr} must be an integer >= 3, got {val!r}"
                )


@dataclass(frozen=True)
class PrimitiveSurfaceMesh:
    """Container for discretized primitive cavity surface mesh cells.

    .. warning::
       This class is experimental/provisional.
       API stability is not guaranteed. Triangle ordering is not guaranteed
       by the API (only centroids, normals, areas shape, units, outward
       normals direction, and read-only flags are stable).

    Attributes
    ----------
    vertices_m : FloatArray
        Shape (V, 3) coordinates of the mesh vertices in meters (read-only).
    triangles : np.ndarray
        Shape (T, 3) vertex indices for triangular cells (int64, read-only).
    centroids_m : FloatArray
        Shape (T, 3) cell centroids in meters (read-only).
    normals : FloatArray
        Shape (T, 3) unit-length outward-facing normal vectors (read-only).
    areas_m2 : FloatArray
        Shape (T,) cell surface areas in square meters (read-only).
    """

    vertices_m: FloatArray
    triangles: np.ndarray
    centroids_m: FloatArray
    normals: FloatArray
    areas_m2: FloatArray

    def __post_init__(self) -> None:
        from gragra._arrays import _check_no_bool, as_index_table

        # 1. Validate vertices_m
        _check_no_bool(self.vertices_m, "vertices_m")
        try:
            v = np.asarray(self.vertices_m, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise TypeError(f"vertices_m must be convertible to float64: {e}") from e

        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"vertices_m must have shape (V, 3), got {v.shape}")
        if not np.isfinite(v).all():
            raise ValueError("vertices_m must contain only finite numbers")

        # 2. Validate triangles using as_index_table
        t = as_index_table("triangles", self.triangles, n_max=v.shape[0])

        # 3. Validate centroids_m
        _check_no_bool(self.centroids_m, "centroids_m")
        try:
            c = np.asarray(self.centroids_m, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise TypeError(f"centroids_m must be convertible to float64: {e}") from e

        if c.ndim != 2 or c.shape[1] != 3 or c.shape[0] != t.shape[0]:
            raise ValueError(
                "centroids_m must have shape (T, 3) matching triangles length, "
                f"got {c.shape} for T = {t.shape[0]}"
            )
        if not np.isfinite(c).all():
            raise ValueError("centroids_m must contain only finite numbers")

        # 4. Validate normals
        _check_no_bool(self.normals, "normals")
        try:
            n = np.asarray(self.normals, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise TypeError(f"normals must be convertible to float64: {e}") from e

        if n.ndim != 2 or n.shape[1] != 3 or n.shape[0] != t.shape[0]:
            raise ValueError(
                "normals must have shape (T, 3) matching triangles length, "
                f"got {n.shape} for T = {t.shape[0]}"
            )
        if not np.isfinite(n).all():
            raise ValueError("normals must contain only finite numbers")

        # 5. Validate areas_m2
        _check_no_bool(self.areas_m2, "areas_m2")
        try:
            a = np.asarray(self.areas_m2, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise TypeError(f"areas_m2 must be convertible to float64: {e}") from e

        if a.ndim != 1 or a.shape[0] != t.shape[0]:
            raise ValueError(
                "areas_m2 must have shape (T,) matching triangles length, "
                f"got {a.shape} for T = {t.shape[0]}"
            )
        if not np.isfinite(a).all():
            raise ValueError("areas_m2 must contain only finite numbers")

        # Make sure they are read-only copies
        v = v.copy()
        v.setflags(write=False)
        c = c.copy()
        c.setflags(write=False)
        n = n.copy()
        n.setflags(write=False)
        a = a.copy()
        a.setflags(write=False)

        object.__setattr__(self, "vertices_m", v)
        object.__setattr__(self, "triangles", t)
        object.__setattr__(self, "centroids_m", c)
        object.__setattr__(self, "normals", n)
        object.__setattr__(self, "areas_m2", a)

    @classmethod
    def concatenate(cls, *meshes: "PrimitiveSurfaceMesh") -> "PrimitiveSurfaceMesh":
        """Concatenate multiple PrimitiveSurfaceMesh objects into a single mesh."""
        if not meshes:
            raise ValueError(
                "At least one PrimitiveSurfaceMesh must be provided to concatenate."
            )
        for m in meshes:
            if not isinstance(m, cls):
                raise TypeError(
                    "All arguments to concatenate must be "
                    "PrimitiveSurfaceMesh instances."
                )

        # Concatenate vertices
        vertices_list = [m.vertices_m for m in meshes]
        new_vertices = np.vstack(vertices_list)

        # Concatenate triangles, adjusting indices
        triangles_list = []
        offset = 0
        for m in meshes:
            triangles_list.append(m.triangles + offset)
            offset += len(m.vertices_m)
        new_triangles = np.vstack(triangles_list)

        # Concatenate centroids, normals, areas
        new_centroids = np.vstack([m.centroids_m for m in meshes])
        new_normals = np.vstack([m.normals for m in meshes])
        new_areas = np.concatenate([m.areas_m2 for m in meshes])

        return cls(
            vertices_m=new_vertices,
            triangles=new_triangles,
            centroids_m=new_centroids,
            normals=new_normals,
            areas_m2=new_areas,
        )


def _compute_triangle_mesh_properties(
    vertices: np.ndarray, triangles: np.ndarray, interior_point: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized calculation of triangle centroids, normals, and areas.

    .. note::
       This method assumes a convex basic shape where the interior point is
       strictly located inside the shape and strictly away from all facet planes.

    Filters out degenerate cells (areas <= threshold or non-finite values).
    """
    if len(triangles) == 0:
        return (
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0, 3)),
            np.zeros((0, 3)),
            np.zeros(0),
        )

    # 1. Compute centroids
    centroids = np.mean(vertices[triangles], axis=1)

    # 2. Compute edge vectors
    p0 = vertices[triangles[:, 0]]
    p1 = vertices[triangles[:, 1]]
    p2 = vertices[triangles[:, 2]]
    e1 = p1 - p0
    e2 = p2 - p0

    # 3. Compute cross product (normal vector)
    cross = np.cross(e1, e2)
    norm = np.linalg.norm(cross, axis=1)

    # 4. Compute areas and normals
    areas = 0.5 * norm
    norm_epsilon = norm + 1e-300
    normals = cross / norm_epsilon[:, np.newaxis]

    # 5. Outward orientation check
    to_centroid = centroids - interior_point
    dot = np.sum(normals * to_centroid, axis=1)
    flip = np.where(dot < 0.0, -1.0, 1.0)
    normals = normals * flip[:, np.newaxis]

    # 6. Filter degenerate cells
    # We use a combined absolute and relative threshold for degeneracy check.
    # The absolute threshold of 1e-12 m^2 assumes a meter-scale shape representation
    # (as per numerical.md guidelines).
    total_area = np.sum(areas)
    threshold = max(1e-12, 1e-12 * total_area)
    mask = (areas > threshold) & np.isfinite(areas)
    mask &= np.isfinite(centroids).all(axis=1)
    mask &= np.isfinite(normals).all(axis=1)

    return triangles[mask], centroids[mask], normals[mask], areas[mask]


def _structured_grid_to_mesh(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    periodic_u: bool,
    periodic_v: bool,
    interior_point: tuple[float, float, float],
) -> PrimitiveSurfaceMesh:
    """Generate PrimitiveSurfaceMesh from a structured grid of coordinates."""
    nu, nv = x.shape
    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    triangles = []
    nu_limit = nu if periodic_u else nu - 1
    nv_limit = nv if periodic_v else nv - 1

    for i in range(nu_limit):
        iu = (i + 1) if (i + 1 < nu) else 0
        for j in range(nv_limit):
            jv = (j + 1) if (j + 1 < nv) else 0
            v00 = i * nv + j
            v10 = iu * nv + j
            v01 = i * nv + jv
            v11 = iu * nv + jv

            # Split quad into two triangles
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])

    triangles_arr = np.array(triangles, dtype=np.int64)
    t_filt, c, n, a = _compute_triangle_mesh_properties(
        vertices, triangles_arr, np.array(interior_point, dtype=np.float64)
    )

    return PrimitiveSurfaceMesh(
        vertices_m=vertices,
        triangles=t_filt,
        centroids_m=c,
        normals=n,
        areas_m2=a,
    )


def _uv_sphere_mesh(
    radius: float,
    n_phi: int,
    n_theta: int,
    *,
    theta_start: float = 0.0,
    theta_end: float = math.pi,
    cx: float = 0.0,
    cy: float = 0.0,
    cz: float = 0.0,
    interior_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> PrimitiveSurfaceMesh:
    """Generate UV sphere mesh with pole-fan triangles to avoid degenerate quads."""
    is_top_pole = theta_start < 1e-10
    is_bot_pole = abs(theta_end - math.pi) < 1e-10

    phi = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)

    theta_rings = np.linspace(theta_start, theta_end, n_theta + 1)
    if is_top_pole:
        theta_rings = theta_rings[1:]
    if is_bot_pole:
        theta_rings = theta_rings[:-1]

    n_rings = len(theta_rings)
    n_ring_verts = n_rings * n_phi
    n_verts = n_ring_verts + (1 if is_top_pole else 0) + (1 if is_bot_pole else 0)

    pts = np.zeros((n_verts, 3), dtype=np.float64)

    ring_base = 1 if is_top_pole else 0
    if is_top_pole:
        pts[0] = [cx, cy, cz + radius]
    if is_bot_pole:
        pts[-1] = [cx, cy, cz - radius]

    for i, th in enumerate(theta_rings):
        base = ring_base + i * n_phi
        pts[base : base + n_phi, 0] = cx + radius * math.sin(th) * np.cos(phi)
        pts[base : base + n_phi, 1] = cy + radius * math.sin(th) * np.sin(phi)
        pts[base : base + n_phi, 2] = cz + radius * math.cos(th)

    tris = []

    # Top fan
    if is_top_pole:
        for j in range(n_phi):
            tris.append([0, ring_base + j, ring_base + (j + 1) % n_phi])

    # Middle bands (quad-split)
    for i in range(n_rings - 1):
        b = ring_base + i * n_phi
        bn = ring_base + (i + 1) * n_phi
        for j in range(n_phi):
            j_next = (j + 1) % n_phi
            tris.append([b + j, b + j_next, bn + j_next])
            tris.append([b + j, bn + j_next, bn + j])

    # Bottom fan
    if is_bot_pole:
        bot_idx = n_verts - 1
        lb = ring_base + (n_rings - 1) * n_phi
        for j in range(n_phi):
            tris.append([lb + j, bot_idx, lb + (j + 1) % n_phi])

    triangles = np.array(tris, dtype=np.int64)
    t_filt, c, n, a = _compute_triangle_mesh_properties(
        pts, triangles, np.array(interior_point, dtype=np.float64)
    )

    return PrimitiveSurfaceMesh(
        vertices_m=pts,
        triangles=t_filt,
        centroids_m=c,
        normals=n,
        areas_m2=a,
    )


def _disc_mesh(
    radius: float,
    n_rings: int,
    n_sectors: int,
    *,
    z: float = 0.0,
    cx: float = 0.0,
    cy: float = 0.0,
    cz: float = 0.0,
    interior_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> PrimitiveSurfaceMesh:
    """Generate flat disc mesh with center-fan + concentric rings."""
    r_vals = np.linspace(0.0, radius, n_rings + 1)
    phi = np.linspace(0.0, 2.0 * math.pi, n_sectors, endpoint=False)

    n_verts = 1 + n_rings * n_sectors
    pts = np.zeros((n_verts, 3), dtype=np.float64)
    pts[0] = [cx, cy, cz + z]

    for ring in range(1, n_rings + 1):
        r = r_vals[ring]
        base = 1 + (ring - 1) * n_sectors
        pts[base : base + n_sectors, 0] = cx + r * np.cos(phi)
        pts[base : base + n_sectors, 1] = cy + r * np.sin(phi)
        pts[base : base + n_sectors, 2] = cz + z

    tris = []
    for s in range(n_sectors):
        tris.append([0, 1 + s, 1 + (s + 1) % n_sectors])

    for ring in range(1, n_rings):
        bi = 1 + (ring - 1) * n_sectors
        bo = 1 + ring * n_sectors
        for s in range(n_sectors):
            s_next = (s + 1) % n_sectors
            tris.append([bi + s, bi + s_next, bo + s_next])
            tris.append([bi + s, bo + s_next, bo + s])

    triangles = np.array(tris, dtype=np.int64)
    t_filt, c, n, a = _compute_triangle_mesh_properties(
        pts, triangles, np.array(interior_point, dtype=np.float64)
    )

    return PrimitiveSurfaceMesh(
        vertices_m=pts,
        triangles=t_filt,
        centroids_m=c,
        normals=n,
        areas_m2=a,
    )


def _half_disc_mesh(
    radius: float,
    n_rings: int,
    n_sectors: int,
    *,
    x_offset: float = 0.0,
    interior_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> PrimitiveSurfaceMesh:
    """Generate half-disc mesh in yz-plane at x = x_offset, z >= 0."""
    r_vals = np.linspace(0.0, radius, n_rings + 1)
    phi = np.linspace(0.0, math.pi, n_sectors + 1)

    n_verts = 1 + n_rings * (n_sectors + 1)
    pts = np.zeros((n_verts, 3), dtype=np.float64)
    pts[0] = [x_offset, 0.0, 0.0]

    for ring in range(1, n_rings + 1):
        r = r_vals[ring]
        base = 1 + (ring - 1) * (n_sectors + 1)
        pts[base : base + n_sectors + 1, 0] = x_offset
        pts[base : base + n_sectors + 1, 1] = r * np.cos(phi)
        pts[base : base + n_sectors + 1, 2] = r * np.sin(phi)

    tris = []
    for s in range(n_sectors):
        tris.append([0, 1 + s, 1 + s + 1])

    for ring in range(1, n_rings):
        bi = 1 + (ring - 1) * (n_sectors + 1)
        bo = 1 + ring * (n_sectors + 1)
        for s in range(n_sectors):
            tris.append([bi + s, bi + s + 1, bo + s + 1])
            tris.append([bi + s, bo + s + 1, bo + s])

    triangles = np.array(tris, dtype=np.int64)
    t_filt, c, n, a = _compute_triangle_mesh_properties(
        pts, triangles, np.array(interior_point, dtype=np.float64)
    )

    return PrimitiveSurfaceMesh(
        vertices_m=pts,
        triangles=t_filt,
        centroids_m=c,
        normals=n,
        areas_m2=a,
    )
