"""External field helper package."""

from gragra.fields.background import (
    BackgroundDensityModel,
    DifferentiableBackgroundDensityModel,
)
from gragra.fields.batch import (
    batched_displacement_grid_weights,
    batched_surface_weights,
    batched_unstructured_weights,
)
from gragra.fields.dipole_diagnostics import (
    DipoleConsistencyReport,
    dipole_consistency_check,
)
from gragra.fields.regular_grid import (
    DensityPerturbationGrid,
    DisplacementGrid,
)
from gragra.fields.surface import SurfaceDisplacementField
from gragra.fields.surface_orientation import (
    OrientationReport,
    check_triangle_orientation,
    reorient_triangles,
)
from gragra.fields.unstructured import UnstructuredDisplacementField

__all__ = [
    "BackgroundDensityModel",
    "DifferentiableBackgroundDensityModel",
    "DensityPerturbationGrid",
    "DipoleConsistencyReport",
    "DisplacementGrid",
    "OrientationReport",
    "SurfaceDisplacementField",
    "UnstructuredDisplacementField",
    "check_triangle_orientation",
    "reorient_triangles",
    "dipole_consistency_check",
    "batched_displacement_grid_weights",
    "batched_unstructured_weights",
    "batched_surface_weights",
]
