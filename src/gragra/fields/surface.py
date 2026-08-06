"""Surface/interface displacement field helpers."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gragra._arrays import _check_no_bool, as_xyz_table
from gragra.array_types import ComplexArray, FloatArray
from gragra.sources.surface import SurfaceMassSheetSource

if TYPE_CHECKING:
    from gragra.fields.surface_orientation import OrientationReport

# Degenerate-triangle threshold on the cross-product norm (= 2 * area). This is
# an absolute threshold in SI (meter) scale, matching the bulk tetra helper's
# ``vol <= 1e-15`` convention (see gragra/fields/unstructured.py); gragra works
# in meter-scale geometry (see numerical.md).
_DEGENERATE_AREA_NORM = 1e-15


@dataclass(frozen=True)
class SurfaceDisplacementField:
    """Helper for displacement fields on unstructured triangular surface meshes.

    Computes an effective surface mass-density perturbation from displacement
    vectors defined at the nodes of a triangle surface mesh. A surface
    displacement ``u`` across an interface with unit normal ``n_hat`` and density
    jump ``delta_rho`` generates ``delta_sigma = delta_rho * (u . n_hat)``.

    Sign / orientation convention
    -----------------------------
    The unit normal of each triangle follows the triangle winding (right-hand
    rule on ``v0 -> v1 -> v2``); it is **not** guaranteed to point outward, so a
    consistent winding is the caller's responsibility (use
    :meth:`check_orientation` to diagnose it). The density jump is
    defined relative to that normal as ``delta_rho = rho(-n_hat side) -
    rho(+n_hat side)`` (for an outward normal this is ``rho_inside -
    rho_outside``). With that convention ``delta_sigma = +delta_rho * (u . n_hat)``
    (no extra sign). ``delta_rho`` is a signed real density difference and may be
    negative (e.g. a lighter medium over a heavier one); it is not constrained to
    be non-negative, unlike the absolute background density ``rho0`` of the bulk
    helpers.

    The per-triangle normal displacement uses the face-mean displacement
    ``u_face = (u0 + u1 + u2) / 3`` (which equals the area average for a linear
    P1 element), dotted with the unit normal via a raw (non-conjugate) inner
    product so that complex harmonic amplitudes are preserved linearly.

    Supported:
    - Linear triangle nodal displacement.
    - Uniform scalar density jump (delta_rho), surface/interface term only.
    - Real or complex harmonic amplitude representation.

    Not supported:
    - Quadrilateral or higher-order surface elements.
    - Position-dependent density jump models.
    - Solver native importer (HDF5, SPECFEM, etc.).

    Complex outputs represent linear complex amplitudes or transfer-function-like
    responses, not directly real-valued time-domain gravitational fields.

    The resulting ``SurfaceMassSheetSource.as_weighted_source()`` weights are mass
    elements only (``delta_m = delta_sigma * dS`` in kg); transfer factors or
    coherent phase terms belong to a separate (P4) abstraction.

    Parameters
    ----------
    vertices_m : array-like
        Coordinate table of mesh nodes (vertices) in meters, shape (Nv, 3).
    triangles : array-like
        Triangle connectivity table of shape (Nt, 3) of vertex indices.
    displacement_m : array-like
        Displacement vectors defined at the mesh nodes (vertices), shape (Nv, 3).
        Can be real-valued (float64) or complex-valued (complex128 for harmonic
        amplitude). Component axis is trailing, per the array conventions.
    """

    vertices_m: FloatArray
    triangles: np.ndarray
    displacement_m: FloatArray | ComplexArray

    def __post_init__(self):
        # Validate vertices_m
        _check_no_bool(self.vertices_m, "vertices_m")
        vertices_validated = as_xyz_table("vertices_m", self.vertices_m)
        nv = len(vertices_validated)

        # Validate displacement_m
        _check_no_bool(self.displacement_m, "displacement_m")
        is_complex = np.iscomplexobj(self.displacement_m)
        dtype = np.complex128 if is_complex else np.float64
        try:
            disp = np.asarray(self.displacement_m, dtype=dtype)
        except (ValueError, TypeError) as e:
            raise TypeError(f"displacement_m must be numeric: {e}") from e

        if disp.ndim != 2 or disp.shape[1] != 3:
            raise ValueError(f"displacement_m must have shape (N, 3), got {disp.shape}")
        if len(disp) != nv:
            raise ValueError(
                f"displacement_m length must match vertices_m: {len(disp)} != {nv}"
            )
        if not np.isfinite(disp).all():
            raise ValueError("displacement_m must contain only finite numbers")

        disp_copy = disp.copy()
        disp_copy.setflags(write=False)

        # Validate triangles
        _check_no_bool(self.triangles, "triangles")
        raw_tri = np.asarray(self.triangles)
        if raw_tri.ndim != 2 or raw_tri.shape[1] != 3:
            raise ValueError(f"triangles must have shape (Nt, 3), got {raw_tri.shape}")
        if not np.issubdtype(raw_tri.dtype, np.integer):
            raise TypeError(f"triangles must be of integer type, got {raw_tri.dtype}")

        if raw_tri.size > 0:
            if np.any(raw_tri < 0) or np.any(raw_tri >= nv):
                raise ValueError("triangle indices must be within the vertex range")

            # Reject degenerate (zero-area) triangles.
            a = vertices_validated[raw_tri[:, 0]]
            b = vertices_validated[raw_tri[:, 1]]
            c = vertices_validated[raw_tri[:, 2]]
            cross = np.cross(b - a, c - a)
            norms = np.linalg.norm(cross, axis=-1)
            if np.any(norms <= _DEGENERATE_AREA_NORM):
                raise ValueError("degenerate zero-area triangle detected")

        tri_copy = raw_tri.copy()
        tri_copy.setflags(write=False)

        object.__setattr__(self, "vertices_m", vertices_validated)
        object.__setattr__(self, "displacement_m", disp_copy)
        object.__setattr__(self, "triangles", tri_copy)

    def as_surface_mass_source(self, delta_rho_kg_m3: float) -> SurfaceMassSheetSource:
        """Compute the surface mass perturbation and return a sheet source.

        ``delta_sigma = delta_rho * (u_face . n_hat)`` is folded per triangle into
        a :class:`~gragra.sources.surface.SurfaceMassSheetSource`, whose
        ``as_weighted_source()`` yields the mass elements ``delta_m =
        delta_sigma * area``.

        Parameters
        ----------
        delta_rho_kg_m3 : float
            Density jump across the interface in kg/m^3 (real, finite, signed;
            ``int``/``numpy`` real scalars are accepted). See the class docstring
            for the sign convention relative to the triangle normal.

        Returns
        -------
        SurfaceMassSheetSource
            Centroids, unit normals, areas, and normal displacements per triangle.
        """
        # Validate delta_rho_kg_m3 (real, finite scalar; signed allowed).
        # Accept Python/NumPy real scalars and 0-d real arrays, matching the
        # scalar validator in SurfaceMassSheetSource; reject complex/bool/array.
        _check_no_bool(delta_rho_kg_m3, "delta_rho_kg_m3")
        raw_delta_rho = np.asarray(delta_rho_kg_m3)
        if raw_delta_rho.ndim != 0 or raw_delta_rho.dtype.kind not in "iuf":
            raise TypeError("delta_rho_kg_m3 must be a real-valued scalar")
        delta_rho_val = float(raw_delta_rho)
        if not np.isfinite(delta_rho_val):
            raise ValueError("delta_rho_kg_m3 must be finite")

        is_complex = np.iscomplexobj(self.displacement_m)
        out_dtype = np.complex128 if is_complex else np.float64

        nt = self.triangles.shape[0]
        if nt == 0:
            centroids = np.zeros((0, 3), dtype=np.float64)
            unit_normals = np.zeros((0, 3), dtype=np.float64)
            areas = np.zeros((0,), dtype=np.float64)
            u_n = np.zeros((0,), dtype=out_dtype)
        else:
            a = self.vertices_m[self.triangles[:, 0]]
            b = self.vertices_m[self.triangles[:, 1]]
            c = self.vertices_m[self.triangles[:, 2]]

            centroids = (a + b + c) / 3.0
            cross = np.cross(b - a, c - a)
            lengths = np.linalg.norm(cross, axis=-1)
            unit_normals = cross / lengths[:, np.newaxis]
            areas = lengths / 2.0

            # Face-mean displacement (area average for a linear P1 element).
            ua = self.displacement_m[self.triangles[:, 0]]
            ub = self.displacement_m[self.triangles[:, 1]]
            uc = self.displacement_m[self.triangles[:, 2]]
            u_face = (ua + ub + uc) / 3.0
            # Raw (non-conjugate) inner product with the real unit normal.
            u_n = np.einsum("ij,ij->i", u_face, unit_normals).astype(out_dtype)

        return SurfaceMassSheetSource(
            positions_m=centroids,
            normals=unit_normals,
            areas_m2=areas,
            normal_displacement_m=u_n,
            density_kg_m3=delta_rho_val,
        )

    def check_orientation(self) -> "OrientationReport":
        """Diagnose the winding consistency and closedness of this mesh.

        A consistent winding is the caller's responsibility; this method only
        reports the diagnosis (it does not modify the mesh). See
        :func:`gragra.fields.surface_orientation.check_triangle_orientation`.

        Returns
        -------
        OrientationReport
            Diagnostic flags and counts for the mesh winding/topology.
        """
        from gragra.fields.surface_orientation import check_triangle_orientation

        return check_triangle_orientation(self.vertices_m, self.triangles)

    def with_consistent_orientation(self) -> "SurfaceDisplacementField":
        """Return a new field with consistently reoriented triangles.

        This method attempts to resolve winding inconsistencies in the mesh
        topologically and returns a new field instance. Since displacement
        vectors are node-based, flipping triangle winding does not affect them.

        .. note::
           **Asymmetry with Diagnostics**:
           Unlike :meth:`check_orientation`, which acts as a non-blocking diagnostic
           by returning an ``OrientationReport`` without raising errors even if
           the mesh is topologically invalid, this method acts as a strict guard.
           If winding consistency cannot be resolved (due to non-manifold edges or
           non-orientable topology like a Mobius strip), it raises a ``ValueError``.

        Returns
        -------
        SurfaceDisplacementField
            A new field instance with reoriented triangles.
        """
        from gragra.fields.surface_orientation import reorient_triangles

        return SurfaceDisplacementField(
            self.vertices_m,
            reorient_triangles(self.triangles),
            self.displacement_m,
        )
