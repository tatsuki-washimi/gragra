"""Geometry primitives for gragra source discretization.

We support three simple uniform density shape primitives: Sphere, Cuboid, and Cylinder.
These shapes are discretized using a deterministic axis-aligned Cartesian midpoint
grid into a WeightedPointSource representation (mass elements only).
"""

from dataclasses import dataclass

import numpy as np

from gragra._arrays import as_xyz_table, as_xyz_vector
from gragra.array_types import FloatArray
from gragra.geometry._grid import (
    _generate_grid,
    _validate_density,
    _validate_lengths,
    _validate_max_points,
    _validate_positive_scalar,
)
from gragra.geometry.surface_mesh import MeshResolution, PrimitiveSurfaceMesh
from gragra.sources import WeightedPointSource


@dataclass(frozen=True)
class Cuboid:
    """Cuboid geometry primitive with uniform density.

    Discretized using a deterministic axis-aligned Cartesian midpoint grid.
    Out of scope for this primitive: FEM/SEM mesh, Gaussian quadrature, surface/cavern
    wall integration, arbitrary rotation, adaptive refinement, stochastic sampling,
    Boolean geometry, CAD/STL/meshio formats. (Compute backends — numpy/numba/
    C++/CuPy — are selected at the observables layer, not by geometry.)

    Parameters
    ----------
    center_m : array-like
        Shape (3,) coordinates of the cuboid center in meters (float64).
    lengths_m : array-like
        Shape (3,) lengths of the cuboid sides in meters (float64).
    density_kg_m3 : float or complex
        Mass density in kg/m^3. Real densities yield float64 weights; complex
        densities yield complex128 weights.
    spacing_m : float
        Grid spacing (maximum cell width) in meters.
    max_points : int or None, optional
        Maximum allowed grid cells in the bounding box (default: 1,000,000).
    """

    center_m: FloatArray
    lengths_m: FloatArray
    density_kg_m3: float | complex
    spacing_m: float
    max_points: int | None = 1_000_000

    def __post_init__(self):
        # Validate and set attributes
        center = as_xyz_vector("center_m", self.center_m)
        lengths = _validate_lengths("lengths_m", self.lengths_m)
        density = _validate_density(self.density_kg_m3)
        spacing = _validate_positive_scalar("spacing_m", self.spacing_m)
        max_pts = _validate_max_points(self.max_points)

        # Set frozen dataclass attributes
        object.__setattr__(self, "center_m", center)
        object.__setattr__(self, "lengths_m", lengths)
        object.__setattr__(self, "density_kg_m3", density)
        object.__setattr__(self, "spacing_m", spacing)
        object.__setattr__(self, "max_points", max_pts)

    def to_weighted_points(self) -> WeightedPointSource:
        """Discretize the cuboid into a WeightedPointSource of mass elements.

        This geometry discretization is G-free (does not import or multiply by
        G_SI). Weights are mass elements (δm = δρ * dV) in kg only (not P4
        coherent transfer weights). Complex density produces complex weights
        representing linear complex amplitude.

        Returns
        -------
        WeightedPointSource
            The discretized representation with coordinates in float64 and
            weights in float64 or complex128.
        """
        grid_points, dv = _generate_grid(
            center_m=self.center_m,
            bbox_lengths=self.lengths_m,
            spacing_m=self.spacing_m,
            max_points=self.max_points,
        )
        dtype = np.complex128 if isinstance(self.density_kg_m3, complex) else np.float64
        weights_kg = np.full(len(grid_points), self.density_kg_m3 * dv, dtype=dtype)
        return WeightedPointSource(positions_m=grid_points, weights_kg=weights_kg)

    def contains(
        self, points_m: list | tuple | np.ndarray, *, tol_m: float = 0.0
    ) -> np.ndarray:
        """Check if target points lie inside or on the boundary of the cuboid.

        .. note::
           This method assumes a convex basic shape.

        Parameters
        ----------
        points_m : list | tuple | np.ndarray
            Shape (N, 3) coordinates of target points in meters.
        tol_m : float, optional
            Boundary expansion tolerance in meters. Must be non-negative and finite.

        Returns
        -------
        np.ndarray
            Boolean array of shape (N,) indicating whether each point is inside.
        """
        if not np.isfinite(tol_m) or tol_m < 0.0:
            raise ValueError(f"tol_m must be a finite non-negative number, got {tol_m}")
        pts = as_xyz_table("points_m", points_m)
        dp = np.abs(pts - self.center_m)
        return np.all(dp <= self.lengths_m / 2.0 + tol_m, axis=1)

    @property
    def analytic_area_m2(self) -> float:
        """Exact surface area of the cuboid in square meters."""
        lx, ly, lz = self.lengths_m
        return float(2.0 * (lx * ly + ly * lz + lz * lx))

    @property
    def analytic_volume_m3(self) -> float:
        """Exact enclosed volume of the cuboid in cubic meters."""
        lx, ly, lz = self.lengths_m
        return float(lx * ly * lz)

    def surface_mesh(
        self, resolution: MeshResolution | None = None
    ) -> PrimitiveSurfaceMesh:
        """Generate a PrimitiveSurfaceMesh representation of the cuboid surface.

        .. warning::
           This method is experimental/provisional.
           API stability is not guaranteed. Triangle ordering is not guaranteed.

        .. note::
           This method uses `resolution.n_u` for divisions on all faces of the
           cuboid, and ignores `resolution.n_v`.
        """

        from gragra.geometry.surface_mesh import _structured_grid_to_mesh

        if resolution is None:
            resolution = MeshResolution()

        n = resolution.n_u + 1
        hx, hy, hz = self.lengths_m / 2.0
        cx, cy, cz = self.center_m

        a_x = np.linspace(cx - hx, cx + hx, n)
        a_y = np.linspace(cy - hy, cy + hy, n)
        a_z = np.linspace(cz - hz, cz + hz, n)

        def make_face(a1_vals, a2_vals, axis_u, axis_v, axis_const, const_val):
            u_grid, v_grid = np.meshgrid(a1_vals, a2_vals)
            coords = [None, None, None]
            coords[axis_u] = u_grid
            coords[axis_v] = v_grid
            coords[axis_const] = np.full_like(u_grid, const_val)
            return _structured_grid_to_mesh(
                coords[0],
                coords[1],
                coords[2],
                periodic_u=False,
                periodic_v=False,
                interior_point=(cx, cy, cz),
            )

        faces = [
            make_face(a_y, a_z, 1, 2, 0, cx + hx),  # +x face
            make_face(a_y, a_z, 1, 2, 0, cx - hx),  # -x face
            make_face(a_x, a_z, 0, 2, 1, cy + hy),  # +y face
            make_face(a_x, a_z, 0, 2, 1, cy - hy),  # -y face
            make_face(a_x, a_y, 0, 1, 2, cz + hz),  # +z face
            make_face(a_x, a_y, 0, 1, 2, cz - hz),  # -z face
        ]
        return PrimitiveSurfaceMesh.concatenate(*faces)


@dataclass(frozen=True)
class Sphere:
    """Sphere geometry primitive with uniform density.

    Discretized using a deterministic Cartesian midpoint grid. Points whose
    midpoint falls inside the sphere are included.
    Out of scope for this primitive: FEM/SEM mesh, Gaussian quadrature, surface/cavern
    wall integration, arbitrary rotation, adaptive refinement, stochastic sampling,
    Boolean geometry, CAD/STL/meshio formats. (Compute backends — numpy/numba/
    C++/CuPy — are selected at the observables layer, not by geometry.)

    Parameters
    ----------
    center_m : array-like
        Shape (3,) coordinates of the sphere center in meters (float64).
    radius_m : float
        Radius of the sphere in meters.
    density_kg_m3 : float or complex
        Mass density in kg/m^3. Real densities yield float64 weights; complex
        densities yield complex128 weights.
    spacing_m : float
        Grid spacing (maximum cell width) in meters.
    max_points : int or None, optional
        Maximum allowed grid cells in the bounding box (default: 1,000,000).
    """

    center_m: FloatArray
    radius_m: float
    density_kg_m3: float | complex
    spacing_m: float
    max_points: int | None = 1_000_000

    def __post_init__(self):
        center = as_xyz_vector("center_m", self.center_m)
        radius = _validate_positive_scalar("radius_m", self.radius_m)
        density = _validate_density(self.density_kg_m3)
        spacing = _validate_positive_scalar("spacing_m", self.spacing_m)
        max_pts = _validate_max_points(self.max_points)

        object.__setattr__(self, "center_m", center)
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "density_kg_m3", density)
        object.__setattr__(self, "spacing_m", spacing)
        object.__setattr__(self, "max_points", max_pts)

    def to_weighted_points(self) -> WeightedPointSource:
        """Discretize the sphere into a WeightedPointSource of mass elements.

        Uses cell-center inclusion, representing a mass approximation (does
        not account for partial cell volumes at boundaries).
        This geometry discretization is G-free (does not import or multiply by
        G_SI). Weights are mass elements (δm = δρ * dV) in kg only (not P4
        coherent transfer weights). Complex density produces complex weights
        representing linear complex amplitude.

        Returns
        -------
        WeightedPointSource
            The discretized representation with coordinates in float64 and
            weights in float64 or complex128.
        """
        bbox_lengths = np.array([2.0 * self.radius_m] * 3, dtype=np.float64)
        grid_points, dv = _generate_grid(
            center_m=self.center_m,
            bbox_lengths=bbox_lengths,
            spacing_m=self.spacing_m,
            max_points=self.max_points,
        )

        dists = np.linalg.norm(grid_points - self.center_m, axis=1)
        mask = dists <= self.radius_m
        if not mask.any():
            # Fallback to the closest point to center to ensure at least 1 point
            idx = np.argmin(dists)
            mask[idx] = True

        filtered_points = grid_points[mask]
        dtype = np.complex128 if isinstance(self.density_kg_m3, complex) else np.float64
        weights_kg = np.full(len(filtered_points), self.density_kg_m3 * dv, dtype=dtype)
        return WeightedPointSource(positions_m=filtered_points, weights_kg=weights_kg)

    def contains(
        self, points_m: list | tuple | np.ndarray, *, tol_m: float = 0.0
    ) -> np.ndarray:
        """Check if target points lie inside or on the boundary of the sphere.

        .. note::
           This method assumes a convex basic shape.

        Parameters
        ----------
        points_m : list | tuple | np.ndarray
            Shape (N, 3) coordinates of target points in meters.
        tol_m : float, optional
            Boundary expansion tolerance in meters. Must be non-negative and finite.

        Returns
        -------
        np.ndarray
            Boolean array of shape (N,) indicating whether each point is inside.
        """
        if not np.isfinite(tol_m) or tol_m < 0.0:
            raise ValueError(f"tol_m must be a finite non-negative number, got {tol_m}")
        pts = as_xyz_table("points_m", points_m)
        dp = pts - self.center_m
        dists_sq = np.sum(dp**2, axis=1)
        return dists_sq <= (self.radius_m + tol_m) ** 2

    @property
    def analytic_area_m2(self) -> float:
        """Exact surface area of the sphere in square meters."""
        return float(4.0 * np.pi * self.radius_m**2)

    @property
    def analytic_volume_m3(self) -> float:
        """Exact enclosed volume of the sphere in cubic meters."""
        return float((4.0 / 3.0) * np.pi * self.radius_m**3)

    def surface_mesh(
        self, resolution: MeshResolution | None = None
    ) -> PrimitiveSurfaceMesh:
        """Generate a PrimitiveSurfaceMesh representation of the sphere surface.

        .. warning::
           This method is experimental/provisional.
           API stability is not guaranteed. Triangle ordering is not guaranteed.
        """

        from gragra.geometry.surface_mesh import _uv_sphere_mesh

        if resolution is None:
            resolution = MeshResolution()

        return _uv_sphere_mesh(
            self.radius_m,
            resolution.n_u,
            resolution.n_v,
            theta_start=0.0,
            theta_end=np.pi,
            cx=self.center_m[0],
            cy=self.center_m[1],
            cz=self.center_m[2],
            interior_point=(self.center_m[0], self.center_m[1], self.center_m[2]),
        )


@dataclass(frozen=True)
class Cylinder:
    """Cylinder geometry primitive with uniform density.

    Discretized using a deterministic Cartesian midpoint grid. Points whose
    midpoint falls inside the cylinder volume are included.
    Out of scope for this primitive: FEM/SEM mesh, Gaussian quadrature, surface/cavern
    wall integration, arbitrary rotation, adaptive refinement, stochastic sampling,
    Boolean geometry, CAD/STL/meshio formats. (Compute backends — numpy/numba/
    C++/CuPy — are selected at the observables layer, not by geometry.)

    Parameters
    ----------
    center_m : array-like
        Shape (3,) coordinates of the cylinder center in meters (float64).
    radius_m : float
        Radius of the cylinder cross-section in meters.
    height_m : float
        Height/length of the cylinder in meters.
    axis : {"x", "y", "z"}
        Orientation axis of the cylinder.
    density_kg_m3 : float or complex
        Mass density in kg/m^3. Real densities yield float64 weights; complex
        densities yield complex128 weights.
    spacing_m : float
        Grid spacing (maximum cell width) in meters.
    max_points : int or None, optional
        Maximum allowed grid cells in the bounding box (default: 1,000,000).
    """

    center_m: FloatArray
    radius_m: float
    height_m: float
    axis: str
    density_kg_m3: float | complex
    spacing_m: float
    max_points: int | None = 1_000_000

    def __post_init__(self):
        center = as_xyz_vector("center_m", self.center_m)
        radius = _validate_positive_scalar("radius_m", self.radius_m)
        height = _validate_positive_scalar("height_m", self.height_m)
        density = _validate_density(self.density_kg_m3)
        spacing = _validate_positive_scalar("spacing_m", self.spacing_m)
        max_pts = _validate_max_points(self.max_points)

        if self.axis not in {"x", "y", "z"}:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {self.axis!r}")

        object.__setattr__(self, "center_m", center)
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "height_m", height)
        object.__setattr__(self, "axis", self.axis)
        object.__setattr__(self, "density_kg_m3", density)
        object.__setattr__(self, "spacing_m", spacing)
        object.__setattr__(self, "max_points", max_pts)

    def to_weighted_points(self) -> WeightedPointSource:
        """Discretize the cylinder into a WeightedPointSource of mass elements.

        Uses cell-center inclusion, representing a mass approximation (does
        not account for partial cell volumes at boundaries).
        This geometry discretization is G-free (does not import or multiply by
        G_SI). Weights are mass elements (δm = δρ * dV) in kg only (not P4
        coherent transfer weights). Complex density produces complex weights
        representing linear complex amplitude.

        Returns
        -------
        WeightedPointSource
            The discretized representation with coordinates in float64 and
            weights in float64 or complex128.
        """
        if self.axis == "x":
            bbox_lengths = np.array(
                [self.height_m, 2.0 * self.radius_m, 2.0 * self.radius_m],
                dtype=np.float64,
            )
        elif self.axis == "y":
            bbox_lengths = np.array(
                [2.0 * self.radius_m, self.height_m, 2.0 * self.radius_m],
                dtype=np.float64,
            )
        else:  # "z"
            bbox_lengths = np.array(
                [2.0 * self.radius_m, 2.0 * self.radius_m, self.height_m],
                dtype=np.float64,
            )

        grid_points, dv = _generate_grid(
            center_m=self.center_m,
            bbox_lengths=bbox_lengths,
            spacing_m=self.spacing_m,
            max_points=self.max_points,
        )

        dp = grid_points - self.center_m
        if self.axis == "x":
            axis_dist = np.abs(dp[:, 0])
            radial_dist = np.sqrt(dp[:, 1] ** 2 + dp[:, 2] ** 2)
        elif self.axis == "y":
            axis_dist = np.abs(dp[:, 1])
            radial_dist = np.sqrt(dp[:, 0] ** 2 + dp[:, 2] ** 2)
        else:  # "z"
            axis_dist = np.abs(dp[:, 2])
            radial_dist = np.sqrt(dp[:, 0] ** 2 + dp[:, 1] ** 2)

        mask = (axis_dist <= self.height_m / 2.0) & (radial_dist <= self.radius_m)
        if not mask.any():
            # Fallback to the closest point to center to ensure at least 1 point
            dists = np.linalg.norm(dp, axis=1)
            idx = np.argmin(dists)
            mask[idx] = True

        filtered_points = grid_points[mask]
        dtype = np.complex128 if isinstance(self.density_kg_m3, complex) else np.float64
        weights_kg = np.full(len(filtered_points), self.density_kg_m3 * dv, dtype=dtype)
        return WeightedPointSource(positions_m=filtered_points, weights_kg=weights_kg)

    def contains(
        self, points_m: list | tuple | np.ndarray, *, tol_m: float = 0.0
    ) -> np.ndarray:
        """Check if target points lie inside or on the boundary of the cylinder.

        .. note::
           This method assumes a convex basic shape.

        Parameters
        ----------
        points_m : list | tuple | np.ndarray
            Shape (N, 3) coordinates of target points in meters.
        tol_m : float, optional
            Boundary expansion tolerance in meters. Must be non-negative and finite.

        Returns
        -------
        np.ndarray
            Boolean array of shape (N,) indicating whether each point is inside.
        """
        if not np.isfinite(tol_m) or tol_m < 0.0:
            raise ValueError(f"tol_m must be a finite non-negative number, got {tol_m}")

        pts = as_xyz_table("points_m", points_m)
        dp = pts - self.center_m
        if self.axis == "x":
            axis_dist = np.abs(dp[:, 0])
            radial_dist_sq = dp[:, 1] ** 2 + dp[:, 2] ** 2
        elif self.axis == "y":
            axis_dist = np.abs(dp[:, 1])
            radial_dist_sq = dp[:, 0] ** 2 + dp[:, 2] ** 2
        else:  # "z"
            axis_dist = np.abs(dp[:, 2])
            radial_dist_sq = dp[:, 0] ** 2 + dp[:, 1] ** 2

        in_radius = radial_dist_sq <= (self.radius_m + tol_m) ** 2
        in_height = axis_dist <= (self.height_m / 2.0) + tol_m
        return in_radius & in_height

    @property
    def analytic_area_m2(self) -> float:
        """Exact surface area of the cylinder in square meters."""
        return float(
            2.0 * np.pi * self.radius_m * self.height_m + 2.0 * np.pi * self.radius_m**2
        )

    @property
    def analytic_volume_m3(self) -> float:
        """Exact enclosed volume of the cylinder in cubic meters."""
        return float(np.pi * self.radius_m**2 * self.height_m)

    def surface_mesh(
        self, resolution: MeshResolution | None = None
    ) -> PrimitiveSurfaceMesh:
        """Generate a PrimitiveSurfaceMesh representation of the cylinder surface.

        .. warning::
           This method is experimental/provisional.
           API stability is not guaranteed. Triangle ordering is not guaranteed.
        """

        from gragra.geometry.surface_mesh import _disc_mesh, _structured_grid_to_mesh

        if resolution is None:
            resolution = MeshResolution()

        r = self.radius_m
        h = self.height_m
        nu, nv = resolution.n_u, resolution.n_v

        # 1. Generate cylinder side surface assuming z-axis at origin
        phi = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
        z_side = np.linspace(-h / 2.0, h / 2.0, nv + 1)
        zg, ph = np.meshgrid(z_side, phi)
        xg = r * np.cos(ph)
        yg = r * np.sin(ph)

        side = _structured_grid_to_mesh(
            xg,
            yg,
            zg,
            periodic_u=True,
            periodic_v=False,
            interior_point=(0.0, 0.0, 0.0),
        )

        # 2. Generate bottom and top caps assuming z-axis at origin
        bot = _disc_mesh(
            r, resolution.n_cap, nu, z=-h / 2.0, interior_point=(0.0, 0.0, 0.0)
        )
        top = _disc_mesh(
            r, resolution.n_cap, nu, z=h / 2.0, interior_point=(0.0, 0.0, 0.0)
        )

        raw_mesh = PrimitiveSurfaceMesh.concatenate(side, bot, top)

        # 3. Rotate and translate based on self.axis and self.center_m
        cx, cy, cz = self.center_m
        if self.axis == "x":

            def rotate(arr):
                out = np.empty_like(arr)
                out[..., 0] = arr[..., 2]
                out[..., 1] = arr[..., 0]
                out[..., 2] = arr[..., 1]
                return out
        elif self.axis == "y":

            def rotate(arr):
                out = np.empty_like(arr)
                out[..., 0] = arr[..., 1]
                out[..., 1] = arr[..., 2]
                out[..., 2] = arr[..., 0]
                return out
        else:  # "z"

            def rotate(arr):
                return arr.copy()

        new_vertices = rotate(raw_mesh.vertices_m)
        new_vertices[:, 0] += cx
        new_vertices[:, 1] += cy
        new_vertices[:, 2] += cz

        new_centroids = rotate(raw_mesh.centroids_m)
        new_centroids[:, 0] += cx
        new_centroids[:, 1] += cy
        new_centroids[:, 2] += cz

        new_normals = rotate(raw_mesh.normals)

        new_vertices.setflags(write=False)
        new_centroids.setflags(write=False)
        new_normals.setflags(write=False)

        return PrimitiveSurfaceMesh(
            vertices_m=new_vertices,
            triangles=raw_mesh.triangles,
            centroids_m=new_centroids,
            normals=new_normals,
            areas_m2=raw_mesh.areas_m2,
        )


@dataclass(frozen=True)
class HalfSphere:
    """Hemispherical dome geometry primitive with uniform density.

    Floor at z = 0 relative to center, dome occupying z >= center_z.

    Parameters
    ----------
    center_m : array-like
        Shape (3,) coordinates of the sphere center in meters (float64).
    radius_m : float
        Radius of the hemisphere in meters.
    density_kg_m3 : float or complex
        Mass density in kg/m^3. Real densities yield float64 weights; complex
        densities yield complex128 weights.
    spacing_m : float
        Grid spacing (maximum cell width) in meters.
    max_points : int or None, optional
        Maximum allowed grid cells in the bounding box (default: 1,000,000).
    """

    center_m: FloatArray
    radius_m: float
    density_kg_m3: float | complex
    spacing_m: float
    max_points: int | None = 1_000_000

    def __post_init__(self):
        center = as_xyz_vector("center_m", self.center_m)
        radius = _validate_positive_scalar("radius_m", self.radius_m)
        density = _validate_density(self.density_kg_m3)
        spacing = _validate_positive_scalar("spacing_m", self.spacing_m)
        max_pts = _validate_max_points(self.max_points)

        object.__setattr__(self, "center_m", center)
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "density_kg_m3", density)
        object.__setattr__(self, "spacing_m", spacing)
        object.__setattr__(self, "max_points", max_pts)

    def to_weighted_points(self) -> WeightedPointSource:
        """Discretize the half-sphere into a WeightedPointSource of mass elements.

        Uses cell-center inclusion, representing a mass approximation (does
        not account for partial cell volumes at boundaries).
        This geometry discretization is G-free (does not import or multiply by
        G_SI). Weights are mass elements (δm = δρ * dV) in kg only (not P4
        coherent transfer weights). Complex density produces complex weights
        representing linear complex amplitude.

        Returns
        -------
        WeightedPointSource
            The discretized representation with coordinates in float64 and
            weights in float64 or complex128.
        """
        cx, cy, cz = self.center_m
        r = self.radius_m
        grid_center = np.array([cx, cy, cz + r / 2.0], dtype=np.float64)
        bbox_lengths = np.array([2.0 * r, 2.0 * r, r], dtype=np.float64)

        grid_points, dv = _generate_grid(
            center_m=grid_center,
            bbox_lengths=bbox_lengths,
            spacing_m=self.spacing_m,
            max_points=self.max_points,
        )

        dp = grid_points - self.center_m
        dists = np.linalg.norm(dp, axis=1)
        mask = (dists <= self.radius_m) & (dp[:, 2] >= 0.0)
        if not mask.any():
            idx = np.argmin(dists)
            mask[idx] = True

        filtered_points = grid_points[mask]
        dtype = np.complex128 if isinstance(self.density_kg_m3, complex) else np.float64
        weights_kg = np.full(len(filtered_points), self.density_kg_m3 * dv, dtype=dtype)
        return WeightedPointSource(positions_m=filtered_points, weights_kg=weights_kg)

    def contains(
        self, points_m: list | tuple | np.ndarray, *, tol_m: float = 0.0
    ) -> np.ndarray:
        """Check if target points lie inside or on the boundary of the half-sphere.

        .. note::
           This method assumes a convex basic shape.

        Parameters
        ----------
        points_m : list | tuple | np.ndarray
            Shape (N, 3) coordinates of target points in meters.
        tol_m : float, optional
            Boundary expansion tolerance in meters. Must be non-negative and finite.

        Returns
        -------
        np.ndarray
            Boolean array of shape (N,) indicating whether each point is inside.
        """
        if not np.isfinite(tol_m) or tol_m < 0.0:
            raise ValueError(f"tol_m must be a finite non-negative number, got {tol_m}")
        pts = as_xyz_table("points_m", points_m)
        dp = pts - self.center_m
        dists_sq = np.sum(dp**2, axis=1)
        in_sphere = dists_sq <= (self.radius_m + tol_m) ** 2
        above_floor = dp[:, 2] >= -tol_m
        return in_sphere & above_floor

    @property
    def analytic_area_m2(self) -> float:
        """Exact surface area of the half-sphere in square meters."""
        return float(3.0 * np.pi * self.radius_m**2)

    @property
    def analytic_volume_m3(self) -> float:
        """Exact enclosed volume of the half-sphere in cubic meters."""
        return float((2.0 / 3.0) * np.pi * self.radius_m**3)

    def surface_mesh(
        self, resolution: MeshResolution | None = None
    ) -> PrimitiveSurfaceMesh:
        """Generate a PrimitiveSurfaceMesh representation of the half-sphere surface.

        .. warning::
           This method is experimental/provisional.
           API stability is not guaranteed. Triangle ordering is not guaranteed.
        """

        from gragra.geometry.surface_mesh import _disc_mesh, _uv_sphere_mesh

        if resolution is None:
            resolution = MeshResolution()

        r = self.radius_m
        cx, cy, cz = self.center_m
        interior = (cx, cy, cz + r / 2.0)

        dome = _uv_sphere_mesh(
            r,
            resolution.n_u,
            resolution.n_v,
            theta_start=0.0,
            theta_end=np.pi / 2.0,
            cx=cx,
            cy=cy,
            cz=cz,
            interior_point=interior,
        )
        floor = _disc_mesh(
            r,
            resolution.n_cap,
            resolution.n_u,
            z=0.0,
            cx=cx,
            cy=cy,
            cz=cz,
            interior_point=interior,
        )
        return PrimitiveSurfaceMesh.concatenate(dome, floor)


@dataclass(frozen=True)
class HalfCylinder:
    """Kamaboko half-cylinder geometry primitive with uniform density.

    Oriented with axis along x-axis and flat floor at z = 0 relative to center.

    Parameters
    ----------
    center_m : array-like
        Shape (3,) coordinates of the half-cylinder center in meters (float64).
    radius_m : float
        Radius of the semi-cylindrical arch in meters.
    length_m : float
        Length of the half-cylinder along x-axis in meters.
    density_kg_m3 : float or complex
        Mass density in kg/m^3. Real densities yield float64 weights; complex
        densities yield complex128 weights.
    spacing_m : float
        Grid spacing (maximum cell width) in meters.
    max_points : int or None, optional
        Maximum allowed grid cells in the bounding box (default: 1,000,000).
    """

    center_m: FloatArray
    radius_m: float
    length_m: float
    density_kg_m3: float | complex
    spacing_m: float
    max_points: int | None = 1_000_000

    def __post_init__(self):
        center = as_xyz_vector("center_m", self.center_m)
        radius = _validate_positive_scalar("radius_m", self.radius_m)
        length = _validate_positive_scalar("length_m", self.length_m)
        density = _validate_density(self.density_kg_m3)
        spacing = _validate_positive_scalar("spacing_m", self.spacing_m)
        max_pts = _validate_max_points(self.max_points)

        object.__setattr__(self, "center_m", center)
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "length_m", length)
        object.__setattr__(self, "density_kg_m3", density)
        object.__setattr__(self, "spacing_m", spacing)
        object.__setattr__(self, "max_points", max_pts)

    def to_weighted_points(self) -> WeightedPointSource:
        """Discretize the half-cylinder into a WeightedPointSource of mass elements.

        Uses cell-center inclusion, representing a mass approximation (does
        not account for partial cell volumes at boundaries).
        This geometry discretization is G-free (does not import or multiply by
        G_SI). Weights are mass elements (δm = δρ * dV) in kg only (not P4
        coherent transfer weights). Complex density produces complex weights
        representing linear complex amplitude.

        Returns
        -------
        WeightedPointSource
            The discretized representation with coordinates in float64 and
            weights in float64 or complex128.
        """
        cx, cy, cz = self.center_m
        r = self.radius_m
        l_val = self.length_m
        grid_center = np.array([cx, cy, cz + r / 2.0], dtype=np.float64)
        bbox_lengths = np.array([l_val, 2.0 * r, r], dtype=np.float64)

        grid_points, dv = _generate_grid(
            center_m=grid_center,
            bbox_lengths=bbox_lengths,
            spacing_m=self.spacing_m,
            max_points=self.max_points,
        )

        dp = grid_points - self.center_m
        radial_dist = np.sqrt(dp[:, 1] ** 2 + dp[:, 2] ** 2)
        mask = (radial_dist <= self.radius_m) & (dp[:, 2] >= 0.0)
        if not mask.any():
            dists = np.linalg.norm(dp, axis=1)
            idx = np.argmin(dists)
            mask[idx] = True

        filtered_points = grid_points[mask]
        dtype = np.complex128 if isinstance(self.density_kg_m3, complex) else np.float64
        weights_kg = np.full(len(filtered_points), self.density_kg_m3 * dv, dtype=dtype)
        return WeightedPointSource(positions_m=filtered_points, weights_kg=weights_kg)

    def contains(
        self, points_m: list | tuple | np.ndarray, *, tol_m: float = 0.0
    ) -> np.ndarray:
        """Check if target points lie inside or on the boundary of the half-cylinder.

        .. note::
           This method assumes a convex basic shape.

        Parameters
        ----------
        points_m : list | tuple | np.ndarray
            Shape (N, 3) coordinates of target points in meters.
        tol_m : float, optional
            Boundary expansion tolerance in meters. Must be non-negative and finite.

        Returns
        -------
        np.ndarray
            Boolean array of shape (N,) indicating whether each point is inside.
        """
        if not np.isfinite(tol_m) or tol_m < 0.0:
            raise ValueError(f"tol_m must be a finite non-negative number, got {tol_m}")
        pts = as_xyz_table("points_m", points_m)
        dp = pts - self.center_m
        r2 = dp[:, 1] ** 2 + dp[:, 2] ** 2
        in_arch = r2 <= (self.radius_m + tol_m) ** 2
        above_floor = dp[:, 2] >= -tol_m
        in_length = np.abs(dp[:, 0]) <= (self.length_m / 2.0) + tol_m
        return in_arch & above_floor & in_length

    @property
    def analytic_area_m2(self) -> float:
        """Exact surface area of the half-cylinder in square meters."""
        r = self.radius_m
        l_val = self.length_m
        return float(np.pi * r * l_val + 2.0 * r * l_val + np.pi * r**2)

    @property
    def analytic_volume_m3(self) -> float:
        """Exact enclosed volume of the half-cylinder in cubic meters."""
        return float(0.5 * np.pi * self.radius_m**2 * self.length_m)

    def surface_mesh(
        self, resolution: MeshResolution | None = None
    ) -> PrimitiveSurfaceMesh:
        """Generate a PrimitiveSurfaceMesh representation of the half-cylinder.

        .. warning::
           This method is experimental/provisional.
           API stability is not guaranteed. Triangle ordering is not guaranteed.
        """

        from gragra.geometry.surface_mesh import (
            _half_disc_mesh,
            _structured_grid_to_mesh,
        )

        if resolution is None:
            resolution = MeshResolution()

        r = self.radius_m
        l_val = self.length_m
        nu, nv = resolution.n_u, resolution.n_v
        cx, cy, cz = self.center_m
        interior = (cx, cy, cz + r / 4.0)

        # --- Arch surface ---
        x_vals = np.linspace(cx - l_val / 2.0, cx + l_val / 2.0, nv + 1)
        phi = np.linspace(0.0, np.pi, nu + 1)
        xv, ph = np.meshgrid(x_vals, phi)
        x_arch = xv
        y_arch = cy + r * np.cos(ph)
        z_arch = cz + r * np.sin(ph)
        arch = _structured_grid_to_mesh(
            x_arch,
            y_arch,
            z_arch,
            periodic_u=False,
            periodic_v=False,
            interior_point=interior,
        )

        # --- Rectangular floor ---
        x_floor = np.linspace(cx - l_val / 2.0, cx + l_val / 2.0, nv + 1)
        y_floor = np.linspace(cy - r, cy + r, nu + 1)
        xf, yf = np.meshgrid(x_floor, y_floor)
        zf = np.full_like(xf, cz)
        floor = _structured_grid_to_mesh(
            xf, yf, zf, periodic_u=False, periodic_v=False, interior_point=interior
        )

        # --- Half-disc end-caps ---
        # Note on mixed coordinate system of interior_point=(cx, 0.0, r / 4.0):
        # x is absolute (cx), while y and z are local (relative to cap origin).
        # This works because the cap's normal points strictly along the x-axis
        # (±x̂), so the outward orientation check (dot product of normal and
        # centroid-to-interior vector) only depends on the x-components.
        # The local y/z values do not affect the orientation sign.
        cap_neg = _half_disc_mesh(
            r,
            resolution.n_cap,
            nu,
            x_offset=cx - l_val / 2.0,
            interior_point=(cx, 0.0, r / 4.0),
        )
        cap_pos = _half_disc_mesh(
            r,
            resolution.n_cap,
            nu,
            x_offset=cx + l_val / 2.0,
            interior_point=(cx, 0.0, r / 4.0),
        )

        def translate_cap(cap):
            new_verts = cap.vertices_m.copy()
            new_verts[:, 1] += cy
            new_verts[:, 2] += cz
            new_cents = cap.centroids_m.copy()
            new_cents[:, 1] += cy
            new_cents[:, 2] += cz

            new_verts.setflags(write=False)
            new_cents.setflags(write=False)

            return PrimitiveSurfaceMesh(
                vertices_m=new_verts,
                triangles=cap.triangles,
                centroids_m=new_cents,
                normals=cap.normals,
                areas_m2=cap.areas_m2,
            )

        cap_neg_translated = translate_cap(cap_neg)
        cap_pos_translated = translate_cap(cap_pos)

        return PrimitiveSurfaceMesh.concatenate(
            arch, floor, cap_neg_translated, cap_pos_translated
        )
