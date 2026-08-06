"""Unstructured grid field helpers."""

from dataclasses import dataclass

import numpy as np

from gragra._arrays import _check_no_bool, as_xyz_table
from gragra.array_types import ComplexArray, FloatArray
from gragra.fields.background import BackgroundDensityModel
from gragra.sources.volume import VolumeElementSource


def _tetra_geometry(
    vertices_m: FloatArray, tetra: np.ndarray
) -> tuple[FloatArray, FloatArray]:
    """Return (centroids (nc,3), vols (nc,)) for a linear tetrahedron mesh.

    Shared geometry helper for ``as_density_perturbation_source`` and
    ``as_dipole_source``. Inputs are assumed already validated (shape,
    dtype, non-degeneracy) by the caller's ``__post_init__`` -- this
    function does not re-validate.

    nc == 0 (empty mesh) returns freshly-allocated read-only zero-length
    arrays without calling ``np.linalg.det`` on an empty stack (matching
    the pre-existing ``as_density_perturbation_source`` early-return
    behavior). For nc > 0 the returned arrays are NOT set read-only --
    callers must apply their own defensive copy + ``setflags(write=False)``
    before exposing them externally (this is a private, module-internal
    helper, not a public contract).
    """
    nc = tetra.shape[0]
    if nc == 0:
        centroids = np.zeros((0, 3), dtype=np.float64)
        vols = np.zeros((0,), dtype=np.float64)
        centroids.setflags(write=False)
        vols.setflags(write=False)
        return centroids, vols

    p0 = vertices_m[tetra[:, 0]]
    p1 = vertices_m[tetra[:, 1]]
    p2 = vertices_m[tetra[:, 2]]
    p3 = vertices_m[tetra[:, 3]]
    jac = np.stack([p1 - p0, p2 - p0, p3 - p0], axis=-1)
    vols = np.abs(np.linalg.det(jac)) / 6.0
    centroids = (p0 + p1 + p2 + p3) / 4.0
    return centroids, vols


@dataclass(frozen=True)
class UnstructuredDisplacementField:
    """Helper for bulk displacement fields on unstructured linear tetra meshes.

    This helper calculates density perturbations from displacement vectors defined
    at the nodes of an unstructured linear tetrahedron mesh.

    Supported:
    - Linear tetrahedron nodal displacement.
    - Uniform scalar background density (rho0) or background density model
      rho0(centroids) (supporting optional spatial gradient).
    - Bulk term (-rho0 * div(u)) and advection term (-u . grad(rho0)) for
      spatially varying density (when include_rho0_gradient_term=True).
    - Real or complex harmonic amplitude representation.

    Not supported:
    - Hexahedron or other non-tetrahedral elements.
    - Surface/interface perturbation terms or density jumps at interfaces
      (these are handled by SurfaceDisplacementField).
    - Solver native importer (HDF5, SPECFEM, etc.).

    Parameters
    ----------
    vertices_m : array-like
        Coordinate table of mesh nodes (vertices) in meters, shape (Nv, 3).
    tetra : array-like
        Element connectivity table of shape (Nc, 4), where each row contains
        the indices of the four vertices forming a linear tetrahedron.
    displacement_m : array-like
        Displacement vectors defined at the mesh nodes (vertices), shape (Nv, 3).
        Can be real-valued (float64) or complex-valued (complex128 for
        harmonic amplitude).
    """

    vertices_m: FloatArray
    tetra: np.ndarray
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

        # Validate tetra
        _check_no_bool(self.tetra, "tetra")
        raw_tetra = np.asarray(self.tetra)
        if raw_tetra.ndim != 2 or raw_tetra.shape[1] != 4:
            raise ValueError(f"tetra must have shape (Nc, 4), got {raw_tetra.shape}")

        if not np.issubdtype(raw_tetra.dtype, np.integer):
            raise TypeError(f"tetra must be of integer type, got {raw_tetra.dtype}")

        if raw_tetra.size > 0:
            if np.any(raw_tetra < 0) or np.any(raw_tetra >= nv):
                raise ValueError("tetra indices must be within the range of vertices")

            # Check degenerate tetrahedra (vol <= 1e-15)
            p0 = vertices_validated[raw_tetra[:, 0]]
            p1 = vertices_validated[raw_tetra[:, 1]]
            p2 = vertices_validated[raw_tetra[:, 2]]
            p3 = vertices_validated[raw_tetra[:, 3]]

            jac = np.stack([p1 - p0, p2 - p0, p3 - p0], axis=-1)
            # Use abs(det)/6 to check volume
            dets = np.linalg.det(jac)
            vols = np.abs(dets) / 6.0
            if np.any(vols <= 1e-15):
                raise ValueError("degenerate tetrahedron detected (volume <= 1e-15)")

        tetra_copy = raw_tetra.copy()
        tetra_copy.setflags(write=False)

        # Set frozen fields
        object.__setattr__(self, "vertices_m", vertices_validated)
        object.__setattr__(self, "displacement_m", disp_copy)
        object.__setattr__(self, "tetra", tetra_copy)

    def as_density_perturbation_source(
        self,
        rho0_kg_m3: (float | BackgroundDensityModel),
        include_rho0_gradient_term: bool = False,
    ) -> VolumeElementSource:
        """Compute the density perturbation and return a VolumeElementSource.

        Calculated via delta_density = -rho0 * div(u) - u_centroid . grad(rho0).

        Parameters
        ----------
        rho0_kg_m3 : float or BackgroundDensityModel
            float: uniform background density.
            BackgroundDensityModel:
                background density evaluated at centroids.
        include_rho0_gradient_term : bool, default False
            If True, include the advection term -u . grad(rho0).
            Requires rho0_kg_m3 to support the gradient() method.

        Returns
        -------
        VolumeElementSource
            The volume element source representing the perturbed density.
        """
        if not isinstance(include_rho0_gradient_term, bool):
            raise TypeError("include_rho0_gradient_term must be a boolean")

        # Early validation gate: check gradient method when
        # include_rho0_gradient_term is True
        grad_func = None
        if include_rho0_gradient_term:
            grad_func = getattr(rho0_kg_m3, "gradient", None)
            if not callable(grad_func):
                raise TypeError(
                    "rho0_kg_m3 must support a callable gradient method "
                    "when include_rho0_gradient_term=True"
                )

        is_callable = callable(rho0_kg_m3)
        if not is_callable:
            _check_no_bool(rho0_kg_m3, "rho0_kg_m3")
            if not isinstance(rho0_kg_m3, (int, float, np.integer, np.floating)):
                raise TypeError(f"rho0_kg_m3 must be a scalar, got {type(rho0_kg_m3)}")
            if np.iscomplexobj(rho0_kg_m3):
                raise TypeError("rho0_kg_m3 must be real-valued")
            rho0_val = float(rho0_kg_m3)
            if not np.isfinite(rho0_val):
                raise ValueError("rho0_kg_m3 must be finite")
            if rho0_val < 0.0:
                raise ValueError("rho0_kg_m3 must be non-negative")
            rho0_arr = rho0_val

        nc = self.tetra.shape[0]
        if nc == 0:
            is_complex = np.iscomplexobj(self.displacement_m)
            out_dtype = np.complex128 if is_complex else np.float64

            positions = np.zeros((0, 3), dtype=np.float64)
            volumes = np.zeros((0,), dtype=np.float64)
            densities = np.zeros((0,), dtype=out_dtype)

            positions.setflags(write=False)
            volumes.setflags(write=False)
            densities.setflags(write=False)

            return VolumeElementSource(
                positions_m=positions,
                volumes_m3=volumes,
                density_kg_m3=densities,
            )

        centroids, vols = _tetra_geometry(self.vertices_m, self.tetra)

        # jac (affine map Jacobian) is needed again below for the shape
        # function gradients (jac_inv); _tetra_geometry's 2-value contract
        # (centroids, vols) does not expose it, so it is recomputed here
        # (cheap indexing + subtraction -- det()/inv() were already two
        # separate LAPACK calls before this refactor, no new cost).
        p0 = self.vertices_m[self.tetra[:, 0]]
        p1 = self.vertices_m[self.tetra[:, 1]]
        p2 = self.vertices_m[self.tetra[:, 2]]
        p3 = self.vertices_m[self.tetra[:, 3]]
        jac = np.stack([p1 - p0, p2 - p0, p3 - p0], axis=-1)

        # Validate and evaluate background density rho0
        if is_callable:
            raw_out = rho0_kg_m3(centroids)
            _check_no_bool(raw_out, "rho0_kg_m3 output")
            raw_arr = np.asarray(raw_out)
            if np.iscomplexobj(raw_arr):
                raise TypeError("rho0_kg_m3 output must be real-valued")
            if raw_arr.dtype.kind not in "iuf":
                raise TypeError(
                    f"rho0_kg_m3 output must be numeric, got {raw_arr.dtype}"
                )
            rho0_arr = raw_arr.astype(np.float64)
            if rho0_arr.shape != (nc,):
                raise ValueError(
                    f"callable rho0 must return shape ({nc},), got {rho0_arr.shape}"
                )
            if not np.isfinite(rho0_arr).all():
                raise ValueError("rho0_kg_m3 output must be finite")
            if (rho0_arr < 0.0).any():
                raise ValueError("rho0_kg_m3 output must be non-negative")

        # Compute shape function gradients (constant within each linear tetrahedron)
        jac_inv = np.linalg.inv(jac)

        grad_n = np.zeros((nc, 4, 3), dtype=np.float64)
        grad_n[:, 1, :] = jac_inv[:, 0, :]
        grad_n[:, 2, :] = jac_inv[:, 1, :]
        grad_n[:, 3, :] = jac_inv[:, 2, :]
        grad_n[:, 0, :] = -(grad_n[:, 1, :] + grad_n[:, 2, :] + grad_n[:, 3, :])

        # Displacement at nodes of each tetrahedron: (Nc, 4, 3)
        u0 = self.displacement_m[self.tetra[:, 0]]
        u1 = self.displacement_m[self.tetra[:, 1]]
        u2 = self.displacement_m[self.tetra[:, 2]]
        u3 = self.displacement_m[self.tetra[:, 3]]

        u_nodes = np.stack([u0, u1, u2, u3], axis=1)  # (Nc, 4, 3)

        # Compute divergence: div(u)_c = sum_{i=0..3} u_i . gradN_i
        div_u = np.einsum("nki,nki->n", u_nodes, grad_n)

        # delta_density = -rho0 * div(u)
        delta_density = -rho0_arr * div_u

        if include_rho0_gradient_term:
            assert grad_func is not None

            # Evaluate gradient at centroids
            grad_rho0_raw = grad_func(centroids)
            _check_no_bool(grad_rho0_raw, "rho0_kg_m3 gradient output")
            grad_rho0_arr = np.asarray(grad_rho0_raw)

            if np.iscomplexobj(grad_rho0_arr):
                raise TypeError("rho0_kg_m3 gradient output must be real-valued")
            if grad_rho0_arr.dtype.kind not in "iuf":
                raise TypeError(
                    "rho0_kg_m3 gradient output must be numeric, "
                    f"got {grad_rho0_arr.dtype}"
                )
            grad_rho0 = grad_rho0_arr.astype(np.float64)

            if grad_rho0.shape != (nc, 3):
                raise ValueError(
                    "rho0_kg_m3 gradient output must have shape "
                    f"({nc}, 3), got {grad_rho0.shape}"
                )
            if not np.isfinite(grad_rho0).all():
                raise ValueError("rho0_kg_m3 gradient output must be finite")

            # u_centroid = (u0 + u1 + u2 + u3) / 4.0
            u_centroid = (u0 + u1 + u2 + u3) / 4.0
            # adv_term = u . grad(rho0)
            adv_term = np.einsum("ni,ni->n", u_centroid, grad_rho0)
            delta_density = delta_density - adv_term

        # Create defensive copies
        centroids_copy = centroids.copy()
        centroids_copy.setflags(write=False)
        vols_copy = vols.copy()
        vols_copy.setflags(write=False)
        delta_density_copy = delta_density.copy()
        delta_density_copy.setflags(write=False)

        return VolumeElementSource(
            positions_m=centroids_copy,
            volumes_m3=vols_copy,
            density_kg_m3=delta_density_copy,
        )

    def as_dipole_source(
        self, rho0_kg_m3: float
    ) -> tuple[FloatArray, FloatArray, FloatArray | ComplexArray]:
        """Return (positions_m, mass_kg, displacement_m) at tetra centroids
        for the dipole/total (vector-weight) path
        (kernels.inverse_square.dipole_contract, via
        integrators.direct_point_dipole_sum).

        mass_kg = rho0 * vols at each tetra centroid -- real, finite,
        non-negative (matches dipole_contract's mass_kg contract,
        kernel_contract.md Section 2.1.1; non-negativity follows
        automatically here since rho0_kg_m3 is validated non-negative and
        vols are always strictly positive per __post_init__'s degenerate-
        tetra rejection). displacement_m = centroid-averaged nodal
        displacement (u0+u1+u2+u3)/4, real or complex (follows
        self.displacement_m's dtype). Complex outputs represent linear
        complex amplitudes or transfer-function-like responses, not
        directly real-valued time-domain gravitational fields.

        Unlike ``as_density_perturbation_source``, this method does not
        support a callable / BackgroundDensityModel ``rho0_kg_m3`` -- only a
        uniform scalar background density.

        Parameters
        ----------
        rho0_kg_m3 : float
            Uniform background density in kg/m^3. Real, finite,
            non-negative (bool rejected).

        Returns
        -------
        tuple of (FloatArray, FloatArray, FloatArray or ComplexArray)
            ``positions_m`` (nc, 3), ``mass_kg`` (nc,), ``displacement_m``
            (nc, 3). All read-only.
        """
        # Scalar-only validation, mirrors the non-callable branch of
        # as_density_perturbation_source (real/finite/non-negative, bool
        # rejected) -- kept duplicated rather than extracted into a shared
        # helper to avoid touching the working callable/BackgroundDensityModel
        # validation path of the existing method.
        _check_no_bool(rho0_kg_m3, "rho0_kg_m3")
        if not isinstance(rho0_kg_m3, (int, float, np.integer, np.floating)):
            raise TypeError(f"rho0_kg_m3 must be a scalar, got {type(rho0_kg_m3)}")
        if np.iscomplexobj(rho0_kg_m3):
            raise TypeError("rho0_kg_m3 must be real-valued")
        rho0_val = float(rho0_kg_m3)
        if not np.isfinite(rho0_val):
            raise ValueError("rho0_kg_m3 must be finite")
        if rho0_val < 0.0:
            raise ValueError("rho0_kg_m3 must be non-negative")

        centroids, vols = _tetra_geometry(self.vertices_m, self.tetra)

        nc = self.tetra.shape[0]
        is_complex = np.iscomplexobj(self.displacement_m)
        out_dtype = np.complex128 if is_complex else np.float64

        if nc == 0:
            mass = np.zeros((0,), dtype=np.float64)
            disp_out = np.zeros((0, 3), dtype=out_dtype)
        else:
            mass = rho0_val * vols
            u0 = self.displacement_m[self.tetra[:, 0]]
            u1 = self.displacement_m[self.tetra[:, 1]]
            u2 = self.displacement_m[self.tetra[:, 2]]
            u3 = self.displacement_m[self.tetra[:, 3]]
            disp_out = (u0 + u1 + u2 + u3) / 4.0

        centroids = centroids.copy()
        centroids.setflags(write=False)
        mass = mass.copy()
        mass.setflags(write=False)
        disp_out = disp_out.copy()
        disp_out.setflags(write=False)

        return centroids, mass, disp_out
