"""Mesh quality checks for externally supplied surface meshes.

These helpers assess whether a tessellated surface mesh (e.g. STL/VTU imported
via the meshio adapter) is geometrically sound before it is used as a
``SurfaceMassSheetSource`` discretization. They are backed by **trimesh**
(an optional dependency, ``pip install "gragra[mesh-quality]"``) because robust
watertight / winding / volume computation on arbitrary meshes is a solved
problem there and re-implementing it in NumPy would be both redundant and less
reliable. trimesh is imported lazily inside the functions, so the gragra core
stays numpy-only.

Vertex coordinates are interpreted as **metres (SI)** — gragra's metre-scale
convention. STL and similar formats carry no unit metadata, so meshes exported
in millimetres must be rescaled before the volume/edge-length gates are
meaningful.

This is a *geometric topology* screen. It is distinct from the physics
diagnostic in ``gragra.fields.dipole_diagnostics`` ("bulk + surface =
dipole" consistency), which cross-checks mass conservation rather than mesh
topology.

Known blind spots (documented deliberately so callers do not over-trust the
gate):

* ``is_watertight`` checks that every edge is shared by exactly two faces
  (manifold + closed). It does **not** detect self-intersection: two
  interpenetrating closed lobes can still report ``True``.
* ``is_winding_consistent`` checks that neighbouring faces agree locally. It
  does **not** detect a *global* flip: a consistently inward-facing closed
  shell reports ``True``. Directional power averages ``|H|^2`` are invariant
  under a global normal flip, so pair this screen with an orientation check
  against an interior point (see ``gragra.fields.check_triangle_orientation``)
  when the sign of the response matters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only, trimesh stays lazy
    import trimesh

# Volume gate folded into the ``pass`` flag when an expected volume is given.
_VOLUME_REL_ERROR_GATE = 0.05


def _require_trimesh():
    """Import trimesh or raise a clear, actionable error."""
    try:
        import trimesh
    except ImportError as e:
        raise ImportError(
            "trimesh is required for mesh quality checks. "
            'Install it with: pip install "gragra[mesh-quality]"'
        ) from e
    return trimesh


def _validate_positive_finite(name: str, value) -> float:
    """Validate a strictly positive finite scalar (TypeError/ValueError split)."""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a positive finite number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def load_trimesh(mesh_or_path, *, process: bool = False) -> trimesh.Trimesh:
    """Load a ``trimesh.Trimesh`` from a file path or a meshio-like Mesh.

    trimesh is imported lazily (optional ``mesh-quality`` extra).

    Parameters
    ----------
    mesh_or_path : str, os.PathLike, or meshio-like Mesh
        Source geometry. A path is read directly by trimesh. A Mesh object is
        accepted by duck typing (``.points`` and ``.cells`` with per-block
        ``.type``/``.data``, as in ``meshio.Mesh``) so meshio itself is never
        imported here; quad cells are split into two triangles.
    process : bool, optional
        Forwarded to trimesh. Defaults to **False** so that duplicate vertices
        and degenerate faces are *not* silently merged away — otherwise quality
        metrics such as ``n_duplicate_vertices`` would always read zero.

    Returns
    -------
    trimesh.Trimesh
    """
    trimesh = _require_trimesh()

    if hasattr(mesh_or_path, "points") and hasattr(mesh_or_path, "cells"):
        points = np.asarray(mesh_or_path.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"mesh points must have shape (N, 3), got {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError("mesh points must contain only finite values")
        faces_list = []
        for cell in mesh_or_path.cells:
            if cell.type == "triangle":
                faces_list.append(np.asarray(cell.data, dtype=np.int64))
            elif cell.type == "quad":
                quad = np.asarray(cell.data, dtype=np.int64)
                faces_list.append(quad[:, [0, 1, 2]])
                faces_list.append(quad[:, [0, 2, 3]])
        if not faces_list:
            raise ValueError("mesh object has no triangle or quad cells")
        faces = np.vstack(faces_list)
        # NumPy fancy indexing would silently wrap negative indices, so reject
        # any out-of-range cell index explicitly (mirrors TriangleSurfaceMesh).
        if faces.size and ((faces < 0).any() or (faces >= len(points)).any()):
            raise ValueError("mesh cell indices are out of range for points")
        return trimesh.Trimesh(vertices=points, faces=faces, process=process)

    try:
        loaded = trimesh.load(mesh_or_path, process=process, force="mesh")
    except NotImplementedError as e:
        # meshio-backed formats (vtu/msh/...) register with trimesh only when
        # meshio was importable at trimesh import time.
        raise ValueError(
            f"trimesh cannot read {mesh_or_path!r} ({e}). meshio-backed "
            'formats require the "gragra[mesh-quality]" extra (meshio) to be '
            "installed before trimesh is first imported."
        ) from e
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(
            f"loaded object from {mesh_or_path!r} is not a single triangle mesh"
        )
    if not np.isfinite(np.asarray(loaded.vertices, dtype=np.float64)).all():
        raise ValueError("mesh points must contain only finite values")
    return loaded


def _count_duplicate_vertices(mesh, rounding: float) -> int:
    """Number of vertices coinciding with another vertex (within rounding)."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.shape[0] == 0:
        return 0
    # Round to `rounding` of the bounding box so near-coincident vertices
    # (e.g. STL per-face duplicates) collapse onto the same integer key.
    scale = max(np.ptp(vertices, axis=0).max(), 1.0)
    keyed = np.round(vertices / (scale * rounding)).astype(np.int64)
    unique = np.unique(keyed, axis=0)
    return int(vertices.shape[0] - unique.shape[0])


def check_mesh_quality(
    mesh_or_path,
    *,
    expected_volume_m3: float | None = None,
    duplicate_vertex_rounding: float = 1e-8,
) -> dict[str, bool | float | int | None]:
    """Compute geometric quality metrics for a surface mesh.

    Returns a dict with both the raw metric values and an overall ``pass``
    flag, so that callers can record the numbers (not just pass/fail).

    Parameters
    ----------
    mesh_or_path : str, os.PathLike, or meshio-like Mesh
        Forwarded to :func:`load_trimesh`.
    expected_volume_m3 : float, optional
        Analytic reference volume in m^3 (must be positive and finite). When
        given, ``volume_rel_error`` is reported and folded into ``pass``
        (relative error < 5%).
    duplicate_vertex_rounding : float, optional
        Rounding used to treat near-coincident vertices as duplicates,
        relative to the bounding-box extent floored at 1 m (i.e. sub-metre
        meshes use an absolute ``rounding`` metres key). Default ``1e-8``;
        must be positive and finite.

    Returns
    -------
    dict
        Metric values keyed as follows.
    is_watertight, is_winding_consistent : bool
    volume_m3 : float
        ``abs`` of the signed divergence-theorem volume ``|(1/3) oint r.n dS|``.
    area_m2 : float
    max_edge_length_m, edge_length_p99_m, min_edge_length_m : float
        Edge-length statistics in metres (drive phase-sampling gates
        downstream).
    n_faces, n_vertices, n_duplicate_vertices : int
        Counted on the *raw* (unmerged) mesh — STL-style per-face storage
        reports ``n_vertices = 3 * n_faces``. Topology keys below are
        evaluated after merging coincident vertices.
    euler_number : int
    volume_rel_error : float or None
        ``|volume - expected| / expected`` when ``expected_volume_m3`` is
        given.
    pass : bool
        ``is_watertight and is_winding_consistent`` (and volume within 5% when
        an expected volume is provided). This is a screen, not a proof — see
        the module docstring for known blind spots (self-intersection, global
        normal flip).
    """
    rounding = _validate_positive_finite(
        "duplicate_vertex_rounding", duplicate_vertex_rounding
    )
    if expected_volume_m3 is not None:
        expected_volume_m3 = _validate_positive_finite(
            "expected_volume_m3", expected_volume_m3
        )

    # Load WITHOUT processing so duplicate vertices are preserved for reporting.
    raw = load_trimesh(mesh_or_path, process=False)
    n_duplicates = _count_duplicate_vertices(raw, rounding)

    # STL (and many CAD exports) store per-face triangles with unshared
    # vertices, so topology (watertight/winding/euler) can only be assessed
    # AFTER merging coincident vertices. Evaluate on a merged copy while
    # keeping ``raw`` for the duplicate counts.
    merged = raw.copy()
    merged.merge_vertices()

    edge_lengths = np.asarray(merged.edges_unique_length, dtype=np.float64)
    if edge_lengths.size == 0:
        raise ValueError("mesh has no edges; cannot assess quality")

    volume_m3 = float(abs(merged.volume))
    volume_rel_error = None
    if expected_volume_m3 is not None:
        volume_rel_error = abs(volume_m3 - float(expected_volume_m3)) / float(
            expected_volume_m3
        )

    is_watertight = bool(merged.is_watertight)
    is_winding_consistent = bool(merged.is_winding_consistent)

    overall = is_watertight and is_winding_consistent
    if volume_rel_error is not None:
        overall = overall and (volume_rel_error < _VOLUME_REL_ERROR_GATE)

    return {
        "is_watertight": is_watertight,
        "is_winding_consistent": is_winding_consistent,
        "volume_m3": volume_m3,
        "area_m2": float(merged.area),
        "max_edge_length_m": float(edge_lengths.max()),
        "edge_length_p99_m": float(np.percentile(edge_lengths, 99)),
        "min_edge_length_m": float(edge_lengths.min()),
        "n_faces": int(len(raw.faces)),
        "n_vertices": int(len(raw.vertices)),
        "n_duplicate_vertices": n_duplicates,
        "euler_number": int(merged.euler_number),
        "volume_rel_error": volume_rel_error,
        "pass": bool(overall),
    }
