"""Adapters package for external ecosystem integrations."""

from gragra.adapters.coherent_combination import (
    CoherentTotal,
    DirectionalComplexKernel,
    coherent_total,
    volume_transfer_kernel,
)
from gragra.adapters.gwexpy_adapter import (
    to_gwexpy_scalar_field,
    to_gwexpy_tensor_field,
    to_gwexpy_vector_field,
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
from gragra.adapters.specfem import (
    ExportConfig,
    SpecfemGLLDataset,
    export_specfem_gll_hdf5,
    read_specfem_gll_hdf5,
)

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
    "to_gwexpy_scalar_field",
    "to_gwexpy_vector_field",
    "to_gwexpy_tensor_field",
    "DirectionalComplexKernel",
    "CoherentTotal",
    "volume_transfer_kernel",
    "coherent_total",
    "lebedev_grid",
    "check_mesh_quality",
    "load_trimesh",
    "DiscreteBackgroundDensity",
    "DifferentiableDiscreteBackgroundDensity",
    "from_hdf5_displacement_grid",
    "from_hdf5_field",
    "from_hdf5_unstructured_field",
    "read_hdf5_background_density",
    "iter_hdf5_field_snapshots",
    "from_hdf5_batched_weights",
    "ExportConfig",
    "SpecfemGLLDataset",
    "export_specfem_gll_hdf5",
    "read_specfem_gll_hdf5",
]
