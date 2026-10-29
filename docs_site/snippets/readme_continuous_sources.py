from gragra.gravity_sources import SurfaceMassSheetSource, VolumeElementSource

# Volume element source with a specific density distribution
vol_source = VolumeElementSource(
    positions_m=[[0.0, 0.0, 0.0]], volumes_m3=[1.0], density_kg_m3=1000.0
)

# Surface mass sheet (phase 4 feature)
surf_source = SurfaceMassSheetSource(
    positions_m=[[0.0, 0.0, 0.0]],
    normals=[[0.0, 0.0, 1.0]],
    areas_m2=[1.0],
    normal_displacement_m=1e-9,
    density_kg_m3=2700.0,
)
