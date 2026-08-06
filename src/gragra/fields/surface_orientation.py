"""Triangle surface mesh orientation diagnostics.

This module provides a *diagnostic-only* check of the winding consistency and
closedness of a triangle surface mesh. It does not modify the mesh: keeping a
consistent winding remains the caller's responsibility (see
:class:`gragra.fields.surface.SurfaceDisplacementField`). The diagnostic exists
because an inconsistent winding randomizes the per-face normal, and hence the
sign of the surface mass perturbation ``delta_sigma = delta_rho * (u . n_hat)``,
breaking the surface integral and the bulk/surface mass-conservation identity.

The module is numpy-only (core layer); it does not import ``gragra.adapters``,
``gwexpy`` or ``h5py``.
"""

from dataclasses import dataclass

import numpy as np

from gragra._arrays import _check_no_bool, as_index_table, as_xyz_table


@dataclass(frozen=True)
class OrientationReport:
    """Diagnostic result for triangle surface mesh orientation.

    All fields are read-only; ``inconsistent_triangle_indices`` is stored as a
    defensive read-only copy. Inspect the boolean flags explicitly (no truthiness
    is defined on the report itself).

    Attributes
    ----------
    is_consistent : bool
        True when every edge shared by exactly two triangles is traversed in
        opposite directions (a consistent manifold winding). Note that winding
        consistency and manifold-ness are independent: a non-manifold mesh
        (an edge shared by three or more triangles) can still report
        ``is_consistent=True`` because such edges are excluded from the
        opposite-direction check.
    is_closed : bool
        True when the mesh has no boundary edges and no non-manifold edges
        (every edge is shared by exactly two triangles).
    is_outward : bool | None
        Only meaningful when ``is_closed and is_consistent``; otherwise ``None``.
        True when the signed volume is strictly positive, i.e. the per-triangle
        normals ``(v1-v0) x (v2-v0)`` point outward. A degenerate closed mesh
        with ``signed_volume_m3 == 0.0`` reports ``False`` (``0 > 0`` is False).

        Limitation: a single boolean cannot certify outward orientation for a
        *multi-component* closed mesh. Two disjoint closed surfaces, one wound
        outward and one inward, satisfy ``is_consistent`` and ``is_closed`` yet
        the signed volume is ``V_a - V_b``, whose sign does not guarantee that
        every component is outward. Connected-component decomposition is out of
        scope for this diagnostic.
    signed_volume_m3 : float
        Signed volume ``(1/6) * sum_t a_t . (b_t x c_t)`` over triangles in
        connectivity order. Physically meaningful (the enclosed volume) only for
        a closed *and consistently wound* surface; for an open surface it is
        origin-dependent, and for a closed but inconsistently wound surface it is
        neither the enclosed volume nor origin-independent (hence ``is_outward``
        is ``None`` in that case).
    inconsistent_edge_count : int
        Number of edges shared by exactly two co-directed triangles.
    boundary_edge_count : int
        Number of edges belonging to exactly one triangle (open boundary).
    nonmanifold_edge_count : int
        Number of edges shared by three or more triangles.
    inconsistent_triangle_indices : np.ndarray
        Sorted, de-duplicated ``np.intp`` indices of triangles participating in
        an inconsistent edge. Both triangles sharing an inconsistent directed
        edge are included (the diagnostic does not decide which one is "wrong").
    """

    is_consistent: bool
    is_closed: bool
    is_outward: bool | None
    signed_volume_m3: float
    inconsistent_edge_count: int
    boundary_edge_count: int
    nonmanifold_edge_count: int
    inconsistent_triangle_indices: np.ndarray

    def __post_init__(self):
        arr = np.asarray(self.inconsistent_triangle_indices, dtype=np.intp).copy()
        arr.setflags(write=False)
        object.__setattr__(self, "inconsistent_triangle_indices", arr)


def check_triangle_orientation(
    vertices_m: list | tuple | np.ndarray,
    triangles: list | tuple | np.ndarray,
) -> OrientationReport:
    """Diagnose the winding consistency and closedness of a triangle mesh.

    This is a pure function operating on raw arrays. It validates dtype/shape
    only; it does not enforce non-degenerate (non-zero-area) triangles, since a
    zero-area triangle is topologically harmless and contributes ~0 to the
    signed volume. Repeated-index triangles (e.g. ``[0, 1, 1]``) are rejected
    with ``ValueError`` because the self-loop edge would pollute the topology
    counts.

    Parameters
    ----------
    vertices_m : array-like
        Vertex coordinate table of shape (Nv, 3) in meters (float64).
    triangles : array-like
        Triangle connectivity table of shape (Nt, 3) of vertex indices. Integer
        indices in ``[0, Nv)``, normalized to ``np.intp`` via
        :func:`gragra._arrays.as_index_table` (input integer dtype independent).

    Returns
    -------
    OrientationReport
        Diagnostic flags and counts. See :class:`OrientationReport`.
    """
    _check_no_bool(vertices_m, "vertices_m")
    vertices = as_xyz_table("vertices_m", vertices_m)
    nv = len(vertices)

    tri = as_index_table("triangles", triangles, nv)
    nt = tri.shape[0]

    if nt == 0:
        return OrientationReport(
            is_consistent=True,
            is_closed=False,
            is_outward=None,
            signed_volume_m3=0.0,
            inconsistent_edge_count=0,
            boundary_edge_count=0,
            nonmanifold_edge_count=0,
            inconsistent_triangle_indices=np.empty(0, dtype=np.intp),
        )

    i0, i1, i2 = tri[:, 0], tri[:, 1], tri[:, 2]

    # Reject self-loop (repeated-index) triangles which would create i->i edges.
    if np.any(i0 == i1) or np.any(i1 == i2) or np.any(i0 == i2):
        raise ValueError("triangles must not contain repeated vertex indices")

    # 1. Directed edges (i->j, j->k, k->i) with their owning triangle index.
    starts = np.concatenate([i0, i1, i2])
    ends = np.concatenate([i1, i2, i0])
    owners = np.tile(np.arange(nt, dtype=np.intp), 3)

    # 2. Undirected key (min, max) and orientation sign (+1 if start<end else -1).
    #    Key and sign are kept in sync (double management) per directed edge.
    key_lo = np.minimum(starts, ends)
    key_hi = np.maximum(starts, ends)
    signs = np.where(starts < ends, 1, -1)
    keys = np.stack([key_lo, key_hi], axis=1)

    uniq, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    inverse = np.asarray(inverse).reshape(-1)

    # 3. Per-edge classification by multiplicity, and winding sign aggregation.
    boundary_edge_count = int(np.count_nonzero(counts == 1))
    nonmanifold_edge_count = int(np.count_nonzero(counts >= 3))

    sign_sum = np.zeros(len(uniq), dtype=np.int64)
    np.add.at(sign_sum, inverse, signs)
    # A shared edge (count == 2) is inconsistent if its two triangles traverse
    # it in the same direction (signs do not cancel to zero).
    inconsistent_edge_mask = (counts == 2) & (sign_sum != 0)
    inconsistent_edge_count = int(np.count_nonzero(inconsistent_edge_mask))

    entry_is_inconsistent = inconsistent_edge_mask[inverse]
    inconsistent_triangle_indices = np.unique(owners[entry_is_inconsistent]).astype(
        np.intp
    )

    # 4. Closed / consistent verdicts.
    is_consistent = inconsistent_edge_count == 0
    is_closed = boundary_edge_count == 0 and nonmanifold_edge_count == 0

    # 5. Signed volume in CONNECTIVITY order (not the sorted edge keys).
    a = vertices[i0]
    b = vertices[i1]
    c = vertices[i2]
    signed_volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    # Inputs are finite (as_xyz_table guards), so a non-finite volume is a bug.
    assert np.isfinite(signed_volume)

    is_outward: bool | None = (
        bool(signed_volume > 0.0) if is_closed and is_consistent else None
    )

    return OrientationReport(
        is_consistent=is_consistent,
        is_closed=is_closed,
        is_outward=is_outward,
        signed_volume_m3=signed_volume,
        inconsistent_edge_count=inconsistent_edge_count,
        boundary_edge_count=boundary_edge_count,
        nonmanifold_edge_count=nonmanifold_edge_count,
        inconsistent_triangle_indices=inconsistent_triangle_indices,
    )


def reorient_triangles(
    triangles: list | tuple | np.ndarray,
) -> np.ndarray:
    """Consistently reorient triangle winding.

    This function attempts to resolve winding inconsistencies (i.e. ensure that
    for every pair of triangles sharing an edge, the edge is traversed in
    opposite directions).

    Topology-Only & Coordinate-Free:
    -------------------------------
    The algorithm is purely topological and **does not utilize vertex coordinates**
    (it is coordinate-free). Consequently, it does **not** force an outward
    orientation (which would require coordinates and closedness).

    Seed Winding Rule:
    ------------------
    For each connected component, the algorithm selects a starting triangle as
    a "seed" and preserves its original winding direction. The winding of all
    other triangles in the same component is flipped or kept as necessary to
    be consistent with the seed triangle.

    Asymmetry with Diagnostics:
    ---------------------------
    - **Diagnostics (:func:`check_triangle_orientation`)**: Reports topological
      defects (e.g., non-manifold edges, open boundary edges, or winding
      inconsistencies) by setting properties in the returned ``OrientationReport``
      without raising any exceptions.
    - **Correction (:func:`reorient_triangles`)**: Behaves as a strict guard.
      If it is impossible to achieve winding consistency (due to non-manifold
      edges shared by 3 or more triangles, or non-orientable topologies
      such as a Mobius strip), it raises a ``ValueError``.

    This is a pure function. It validates and returns a new read-only ``np.intp``
    array of shape (Nt, 3) without modifying the input array.

    Parameters
    ----------
    triangles : array-like
        Triangle connectivity table of shape (Nt, 3) of vertex indices.
        Integer indices in ``[0, N_max)``, independent of input integer dtype.

    Returns
    -------
    np.ndarray
        A read-only ``np.intp`` array of shape (Nt, 3) representing the
        reoriented connectivity table.
    """
    _check_no_bool(triangles, "triangles")

    raw = np.asarray(triangles)
    if raw.dtype.kind not in "iu":
        raise TypeError(f"triangles must be of integer type, got {raw.dtype}")

    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f"triangles must have shape (Nt, 3), got {raw.shape}")

    tri = raw.astype(np.intp, copy=True)
    nt = tri.shape[0]

    if nt == 0:
        tri.setflags(write=False)
        return tri

    i0, i1, i2 = tri[:, 0], tri[:, 1], tri[:, 2]

    # Reject self-loop (repeated-index) triangles which would create i->i edges.
    if np.any(i0 == i1) or np.any(i1 == i2) or np.any(i0 == i2):
        raise ValueError("triangles must not contain repeated vertex indices")

    if (tri < 0).any():
        raise ValueError("triangles indices must be non-negative")

    # Build adjacency list
    edges = {}
    for t in range(nt):
        t_verts = tri[t]
        for local_idx in range(3):
            u = t_verts[local_idx]
            v = t_verts[(local_idx + 1) % 3]
            edge_key = (min(u, v), max(u, v))
            if edge_key not in edges:
                edges[edge_key] = []
            edges[edge_key].append((t, local_idx))

    # Check for non-manifold edges
    for edge_key, owners in edges.items():
        if len(owners) >= 3:
            raise ValueError(
                f"mesh is non-manifold: edge {edge_key} "
                f"shared by {len(owners)} triangles"
            )

    visited = np.zeros(nt, dtype=bool)
    flip = np.zeros(nt, dtype=bool)

    for start_t in range(nt):
        if visited[start_t]:
            continue

        queue = [start_t]
        visited[start_t] = True
        flip[start_t] = False

        head = 0
        while head < len(queue):
            curr_t = queue[head]
            head += 1

            curr_verts = tri[curr_t]
            curr_flip = flip[curr_t]

            for local_idx in range(3):
                u = curr_verts[local_idx]
                v = curr_verts[(local_idx + 1) % 3]

                curr_dir = (v, u) if curr_flip else (u, v)

                edge_key = (min(u, v), max(u, v))
                owners = edges[edge_key]

                for neigh_t, neigh_local_idx in owners:
                    if neigh_t == curr_t:
                        continue

                    nu = tri[neigh_t, neigh_local_idx]
                    nv = tri[neigh_t, (neigh_local_idx + 1) % 3]

                    if visited[neigh_t]:
                        neigh_dir = (nv, nu) if flip[neigh_t] else (nu, nv)

                        if curr_dir == neigh_dir:
                            raise ValueError(
                                "mesh is non-orientable (winding "
                                "inconsistency cannot be resolved)"
                            )
                    else:
                        u_dir, v_dir = curr_dir
                        if nu == u_dir and nv == v_dir:
                            flip[neigh_t] = True
                        elif nu == v_dir and nv == u_dir:
                            flip[neigh_t] = False
                        else:
                            raise ValueError("topological error in edge sharing")

                        visited[neigh_t] = True
                        queue.append(neigh_t)

    new_tri = tri.copy()
    if np.any(flip):
        new_tri[flip] = new_tri[flip][:, [0, 2, 1]]

    new_tri.setflags(write=False)
    return new_tri
