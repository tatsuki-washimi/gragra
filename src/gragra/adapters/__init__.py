"""Adapters package for external ecosystem integrations."""

from gragra.adapters.field_series_hdf5 import (
    read_fixed_mesh_series_hdf5,
    write_fixed_mesh_series_hdf5,
    write_observable_series_hdf5,
)
from gragra.adapters.finite_body_mesh import (
    cell_density_from_group_tags,
    to_unstructured_displacement_field,
)
from gragra.adapters.gwexpy_adapter import (
    to_gwexpy_scalar_field,
    to_gwexpy_tensor_field,
    to_gwexpy_vector_field,
)
from gragra.adapters.gwexpy_density_series import (
    from_gwexpy_density_phasors,
    from_gwexpy_density_series,
    from_gwexpy_density_time_series,
    to_gwexpy_density_phasors,
    to_gwexpy_density_series,
    to_gwexpy_density_time_series,
)
from gragra.adapters.gwexpy_displacement_series import (
    from_gwexpy_displacement_series,
    to_gwexpy_displacement_series,
)
from gragra.adapters.gwexpy_observable import (
    GwexpyObservableBundle,
    ObservableArrayBundle,
    from_gwexpy_observable,
    to_gwexpy_observable,
)
from gragra.adapters.gwexpy_source import (
    DensitySnapshotArrayBundle,
    GwexpyDensitySnapshotBundle,
    from_gwexpy_density_snapshot,
    to_gwexpy_density_snapshot,
)
from gragra.adapters.gwexpy_surface_density_series import (
    from_gwexpy_surface_density_series,
    to_gwexpy_surface_density_series,
)
from gragra.adapters.hdf5_field import (
    DifferentiableDiscreteBackgroundDensity,
    DiscreteBackgroundDensity,
    from_hdf5_batched_weights,
    from_hdf5_displacement_grid,
    from_hdf5_field,
    from_hdf5_unstructured_field,
    iter_hdf5_field_snapshots,
    read_hdf5_background_density,
)
from gragra.adapters.lebedev import lebedev_grid
from gragra.adapters.mesh_quality import check_mesh_quality, load_trimesh
from gragra.adapters.meshio_adapter import (
    TriangleSurfaceMesh,
    VolumeMesh,
    cell_centroids_volumes,
    centroids_normals_areas,
    from_meshio,
    from_meshio_volume,
    select_group,
    select_group_tag,
    to_surface_mass_sheet,
    to_volume_element_source,
)
from gragra.adapters.openfoam_density import (
    OpenFOAMDensitySnapshot,
    load_openfoam_density,
)
from gragra.adapters.specfem import (
    ExportConfig,
    SpecfemGLLDataset,
    export_specfem_gll_hdf5,
    read_specfem_gll_hdf5,
)
from gragra.adapters.specfem.streaming import SpecfemGLLStreamReader

__all__ = [
    "TriangleSurfaceMesh",
    "VolumeMesh",
    "from_meshio",
    "from_meshio_volume",
    "centroids_normals_areas",
    "cell_centroids_volumes",
    "select_group",
    "select_group_tag",
    "to_surface_mass_sheet",
    "to_volume_element_source",
    "to_unstructured_displacement_field",
    "cell_density_from_group_tags",
    "OpenFOAMDensitySnapshot",
    "load_openfoam_density",
    "to_gwexpy_scalar_field",
    "to_gwexpy_vector_field",
    "to_gwexpy_tensor_field",
    "GwexpyObservableBundle",
    "ObservableArrayBundle",
    "to_gwexpy_observable",
    "from_gwexpy_observable",
    "GwexpyDensitySnapshotBundle",
    "DensitySnapshotArrayBundle",
    "to_gwexpy_density_snapshot",
    "from_gwexpy_density_snapshot",
    "to_gwexpy_density_series",
    "from_gwexpy_density_series",
    "to_gwexpy_density_time_series",
    "from_gwexpy_density_time_series",
    "to_gwexpy_density_phasors",
    "from_gwexpy_density_phasors",
    "to_gwexpy_displacement_series",
    "from_gwexpy_displacement_series",
    "to_gwexpy_surface_density_series",
    "from_gwexpy_surface_density_series",
    "lebedev_grid",
    "check_mesh_quality",
    "load_trimesh",
    "DiscreteBackgroundDensity",
    "DifferentiableDiscreteBackgroundDensity",
    "from_hdf5_displacement_grid",
    "from_hdf5_field",
    "from_hdf5_unstructured_field",
    "read_hdf5_background_density",
    "read_fixed_mesh_series_hdf5",
    "write_fixed_mesh_series_hdf5",
    "write_observable_series_hdf5",
    "SpecfemGLLStreamReader",
    "iter_hdf5_field_snapshots",
    "from_hdf5_batched_weights",
    "ExportConfig",
    "SpecfemGLLDataset",
    "export_specfem_gll_hdf5",
    "read_specfem_gll_hdf5",
]
