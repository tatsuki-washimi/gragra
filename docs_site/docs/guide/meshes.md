
# Using external meshes

## Supported mesh input

`gragra.adapters` reads surface and volume mesh files via `trimesh`/`meshio`
(the `mesh` extra), converting them into `TriangleSurfaceMesh` or
`VolumeMesh`. This is a mesh **importer**, not a solver — it reads a file that
already exists; it does not create or refine geometry. An in-memory
`VolumeMesh` can use the static `VolumeMesh → VolumeElementSource → point_sum`
route without the file-reader extra; file-based reads still need the relevant
optional dependency.

## Surface meshes

`TriangleSurfaceMesh` holds triangle vertices, connectivity, and per-face
centroids/normals/areas (`centroids_normals_areas`), converting to a
`SurfaceMassSheetSource` via `to_surface_mass_sheet` (see
[Volume and surface sources](continuous-sources.md)).

## Volume meshes

`VolumeMesh` holds a tetrahedral (or other supported cell-type) volume
mesh, with per-cell centroid and volume (`cell_centroids_volumes`),
converting to a `VolumeElementSource` via `to_volume_element_source`.
`select_group`/`select_group_tag` let you restrict to a named region or
tag before conversion, rather than importing the whole file as one source.

For a finite piecewise-tetra displacement body,
`to_unstructured_displacement_field` concatenates tetra blocks in their input
order, and `cell_density_from_group_tags` supplies one density for each tagged
tetra cell. Nonempty hexahedron blocks are rejected by this boundary because
the finite-body field representation is limited to linear tetrahedra.

For the cell-local P1-density finite-body builder, the four density entries of
each cell follow that tetra's local node order. A shared geometric vertex does
not require equal density values in neighbouring cells: a differing trace is a
material interface term, while the displacement mesh itself remains conforming.

## Mesh quality checks

`check_mesh_quality` (the `mesh-quality` extra, via `load_trimesh`) reports
degenerate elements, non-manifold edges, and other quality issues in an
imported mesh **before** you use it as a source — running it first is
cheaper than debugging a suspicious gravity field that turns out to trace
back to a broken input mesh.

## Orientation and winding diagnostics

`check_triangle_orientation` and `reorient_triangles` (`gragra.fields`)
detect and correct inconsistent triangle winding on a surface mesh — see
the equivalence conditions in
[Displacement and density-perturbation fields](fields.md#equivalence-conditions-for-the-bulk-and-surface-forms),
several of which depend on winding and outward-normal consistency being
verified rather than assumed.

## What is not supported

This is an **importer for meshes gragra's own source/geometry layer can
consume**, not a general-purpose FEM/SEM solver interface. It does not run
a solver, does not adapt an arbitrary solver's native mesh format beyond
what `trimesh`/`meshio` already parse, and does not validate the physics
that produced the mesh — only its geometric quality. See
[Using external meshes vs. native solver import](../external/hdf5-fields.md)
for the one solver (SPECFEM3D Cartesian) with dedicated, narrower native
import support.

## Status and verification

Mesh import has no top-level public Verification Block of its own. The
experimental static volume-mesh conversion and evaluation route has its own
[Verification Block](static-sources.md#status-volume-mesh-static-point-sum);
it covers the converted discrete point field, not continuous-mesh accuracy or
all file readers. Mesh quality and orientation diagnostics are checked by
`tests/test_meshio_adapter.py`, `tests/test_mesh_quality.py`, and
`tests/test_surface_orientation.py`.
