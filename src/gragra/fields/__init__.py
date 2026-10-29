"""External field helper package."""

from typing import TYPE_CHECKING

from gragra.fields.background import (
    BackgroundDensityModel,
    DifferentiableBackgroundDensityModel,
)
from gragra.fields.batch import (
    batched_displacement_grid_weights,
    batched_surface_weights,
    batched_unstructured_weights,
)
from gragra.fields.density_series import DensityPhasors, DensityTimeSeries
from gragra.fields.dipole_diagnostics import (
    DipoleConsistencyReport,
    dipole_consistency_check,
)
from gragra.fields.displacement_series import FixedMeshDisplacementSeries
from gragra.fields.evaluation import (
    FixedMeshDisplacementSources,
    FixedMeshSurfaceSources,
    build_fixed_mesh_displacement_sources,
    build_fixed_mesh_surface_sources,
    build_plane_wave_fixed_mesh_sources,
)
from gragra.fields.finite_body import (
    TetraLinearizedSources,
    TetraRepresentationReport,
    compare_tetra_linearized_representations,
    piecewise_tetra_linear_sources,
)
from gragra.fields.finite_body_p1 import (
    TetraP1DensityLinearizedSources,
    compare_tetra_p1_density_representations,
    piecewise_tetra_p1_density_sources,
)
from gragra.fields.material_hex8 import (
    Hex8DisplacementField,
    Hex8Sources,
    MaterialRegions,
)
from gragra.fields.regular_grid import (
    DensityPerturbationGrid,
    DisplacementGrid,
)
from gragra.fields.surface import SurfaceDisplacementField
from gragra.fields.surface_density_series import FixedMeshSurfaceDensitySeries
from gragra.fields.surface_orientation import (
    OrientationReport,
    check_triangle_orientation,
    reorient_triangles,
)
from gragra.fields.thermodynamic import (
    adiabatic_pressure_density_perturbation,
    isobaric_temperature_density_perturbation,
)
from gragra.fields.unstructured import UnstructuredDisplacementField

__all__ = [
    "BackgroundDensityModel",
    "DifferentiableBackgroundDensityModel",
    "DensityPerturbationGrid",
    "DensityPhasors",
    "DensityTimeSeries",
    "DipoleConsistencyReport",
    "DisplacementGrid",
    "FixedMeshDisplacementSeries",
    "FixedMeshSurfaceDensitySeries",
    "OrientationReport",
    "SurfaceDisplacementField",
    "TetraLinearizedSources",
    "TetraRepresentationReport",
    "UnstructuredDisplacementField",
    "check_triangle_orientation",
    "reorient_triangles",
    "dipole_consistency_check",
    "piecewise_tetra_linear_sources",
    "compare_tetra_linearized_representations",
    "TetraP1DensityLinearizedSources",
    "piecewise_tetra_p1_density_sources",
    "compare_tetra_p1_density_representations",
    "batched_displacement_grid_weights",
    "batched_unstructured_weights",
    "batched_surface_weights",
    "adiabatic_pressure_density_perturbation",
    "isobaric_temperature_density_perturbation",
    "MaterialRegions",
    "Hex8DisplacementField",
    "Hex8Sources",
    "FixedMeshDisplacementSources",
    "FixedMeshSurfaceSources",
    "build_fixed_mesh_displacement_sources",
    "build_fixed_mesh_surface_sources",
    "build_plane_wave_fixed_mesh_sources",
    "TemperatureFourierMode",
    "TemperatureGaussian",
    "TemperaturePerturbationGrid",
]

if TYPE_CHECKING:
    from gragra.fields._temperature import (
        TemperatureFourierMode,
        TemperatureGaussian,
        TemperaturePerturbationGrid,
    )


def __getattr__(name: str):
    """Load Gen1 Temperature profiles only when explicitly requested."""
    if name in {
        "MaterialRegions",
        "Hex8DisplacementField",
        "Hex8Sources",
        "FixedMeshDisplacementSources",
        "FixedMeshSurfaceSources",
        "build_fixed_mesh_displacement_sources",
        "build_fixed_mesh_surface_sources",
        "build_plane_wave_fixed_mesh_sources",
        "TemperatureFourierMode",
        "TemperatureGaussian",
        "TemperaturePerturbationGrid",
    }:
        from gragra.fields._temperature import (
            TemperatureFourierMode,
            TemperatureGaussian,
            TemperaturePerturbationGrid,
        )

        return {
            "TemperatureFourierMode": TemperatureFourierMode,
            "TemperatureGaussian": TemperatureGaussian,
            "TemperaturePerturbationGrid": TemperaturePerturbationGrid,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
