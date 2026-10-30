<!-- GENERATED CATALOGUE: model ledger and generator live in gragra-dev. -->
# Models, methods and API catalogue

**24 model groups** · **40 catalogue API paths** · **48 inventoried API/calculation-path variants**

The 48 variants are inventory subdivisions, not 48 independent numerical methods.

This catalogue distinguishes **source-declared observable availability**, **representative output-contract probes**, and **scientific accuracy evidence**. Source declarations alone do not establish accuracy or API stability.

**Observable legend:** Φ = gravitational potential (m²/s²); a = gravitational acceleration (m/s²); Γ = gravity-gradient tensor (s⁻²). Displacement inputs are different from acceleration output a.

**Representation legend:** Static, Phasor, Time series, Fourier series, Fourier transform, and Spectral density (ASD / PSD / CSD). An entry may have several source-declared paths; each observable set is paired with the representation on the same line.

**Level:** — (definition, number of levels, assessment unit and assignments pending).

**Not assessed** refers only to a missing catalogue output probe; it is not a statement about implementation or scientific validation.

<div id="gragra-catalogue-controls"></div>

## Model catalogue

### Static gravity and finite masses

<table class="gragra-catalogue-table">
<thead><tr><th>Model</th><th>Geometry</th><th>Φ/a/Γ</th><th>Representation</th><th>Level</th></tr></thead>
<tbody>
<tr data-gragra-category="static_gravity" data-gragra-formats="static" data-gragra-search="uniform-density sphere finite sphere; static potential, acceleration and gravity gradient, inside or outside static_gravity sphere gragra.analytic.objects.uniform_sphere gragra.observables.uniform_sphere_observables static"><td><a href="../models/uniform_density_sphere/">Uniform-density sphere</a></td><td>sphere</td><td class="gragra-observables">Φ a Γ</td><td>Static</td><td class="gragra-level">—</td></tr>
<tr data-gragra-category="static_gravity" data-gragra-formats="static" data-gragra-search="uniform-density cuboid finite rectangular prism; static newton field static_gravity cuboid gragra.analytic.objects.uniform_cuboid gragra.observables.cuboid_observables static"><td><a href="../models/uniform_density_cuboid/">Uniform-density cuboid</a></td><td>cuboid</td><td class="gragra-observables">Φ a Γ</td><td>Static</td><td class="gragra-level">—</td></tr>
<tr data-gragra-category="static_gravity" data-gragra-formats="static" data-gragra-search="oriented cuboid rotated finite prism and target-frame response static_gravity cuboid gragra.observables.oriented_cuboid_observables static"><td><a href="../models/oriented_cuboid/">Oriented cuboid</a></td><td>cuboid</td><td class="gragra-observables">Φ a Γ</td><td>Static</td><td class="gragra-level">—</td></tr>
<tr data-gragra-category="static_gravity" data-gragra-formats="static" data-gragra-search="uniform-density cylinder finite cylinder; check the selected function&#x27;s axial/off-axis domain static_gravity cylinder gragra.analytic.objects.uniform_cylinder gragra.observables.cylinder_axis_observables gragra.observables.cylinder_observables static"><td><a href="../models/uniform_density_cylinder/">Uniform-density cylinder</a></td><td>cylinder</td><td class="gragra-observables">Φ a Γ</td><td>Static</td><td class="gragra-level">—</td></tr>
</tbody></table>

### Seismic and elastic waves

<table class="gragra-catalogue-table">
<thead><tr><th>Model</th><th>Geometry</th><th>Φ/a/Γ</th><th>Representation</th><th>Level</th></tr></thead>
<tbody>
<tr data-gragra-category="seismic_elastic" data-gragra-formats="phasor" data-gragra-search="spherical cavity and finite spherical shell elastic displacement response, cavity wall and finite-shell boundary contributions seismic_elastic cavity, shell gragra.analytic.seismic.spherical_cavity gragra.analytic.seismic.spherical_shell gragra.observables.spherical_shell_p_acceleration_components phasor"><td><a href="../models/spherical_cavity_and_finite_spherical_shell/">Spherical cavity and finite spherical shell</a></td><td>cavity, shell</td><td class="gragra-observables">Φ a Γ</td><td>Phasor</td><td class="gragra-level">—</td></tr>
<tr data-gragra-category="seismic_elastic" data-gragra-formats="phasor static" data-gragra-search="small-cavity wall approximation sphere, cuboid or cylinder; selected linearized wall contribution only seismic_elastic cavity gragra.analytic.seismic.small_cavity_wall gragra.analytic.sca_wall_coupling_factor static phasor"><td><a href="../models/small_cavity_wall_approximation/">Small-cavity wall approximation</a></td><td>cavity</td><td class="gragra-observables">a<br>a</td><td>Static<br>Phasor</td><td class="gragra-level">—</td></tr>
<tr data-gragra-category="seismic_elastic" data-gragra-formats="phasor" data-gragra-search="rayleigh wave at an elastic halfspace surface-wave gravity, depth response and coherent phasors seismic_elastic halfspace gragra.analytic.seismic.rayleigh_halfspace gragra.analytic.seismic.rayleigh_response gragra.observables.rayleigh_halfspace_allspace_observables phasor"><td><a href="../models/rayleigh_wave_at_an_elastic_halfspace/">Rayleigh wave at an elastic halfspace</a></td><td>halfspace</td><td class="gragra-observables">Φ a Γ</td><td>Phasor</td><td class="gragra-level">—</td></tr>
<tr data-gragra-category="seismic_elastic" data-gragra-formats="phasor" data-gragra-search="p/sv waves at a free surface homogeneous elastic halfspace; incidence, reflection and mode conversion seismic_elastic halfspace gragra.analytic.seismic.halfspace_psv gragra.observables.halfspace_psv_observables phasor"><td><a href="../models/p_sv_waves_at_a_free_surface/">P/SV waves at a free surface</a></td><td>halfspace</td><td class="gragra-observables">Φ a Γ</td><td>Phasor</td><td class="gragra-level">—</td></tr>
<tr data-gragra-category="seismic_elastic" data-gragra-formats="phasor" data-gragra-search="individual p/sv and reflected-p benchmarks single incident mode versus a traction-free reflected composite seismic_elastic halfspace gragra.observables.halfspace_single_bodywave_observables gragra.observables.halfspace_reflected_p_above_observables phasor"><td><a href="../models/individual_p_sv_and_reflected_p_benchmarks/">Individual P/SV and reflected-P benchmarks</a></td><td>halfspace</td><td class="gragra-observables">Φ a Γ</td><td>Phasor</td><td class="gragra-level">—</td></tr>
<tr data-gragra-category="seismic_elastic" data-gragra-formats="time_series" data-gragra-search="causal point force time-dependent force in an infinite homogeneous elastic medium seismic_elastic point gragra.analytic.seismic.point_force time series"><td><a href="../models/causal_point_force/">Causal point force</a></td><td>point</td><td class="gragra-observables">Φ a Γ</td><td>Time series</td><td class="gragra-level">—</td></tr>
</tbody></table>

### Moving masses and rigid motion

<table class="gragra-catalogue-table">
<thead><tr><th>Model</th><th>Geometry</th><th>Φ/a/Γ</th><th>Representation</th><th>Level</th></tr></thead>
<tbody>
<tr data-gragra-category="rigid_motion" data-gragra-formats="time_series" data-gragra-search="prescribed point-mass translation constant-velocity mass; time-sampled field rigid_motion point gragra.analytic.objects.constant_velocity time series"><td><a href="../models/prescribed_point_mass_translation/">Prescribed point-mass translation</a></td><td>point</td><td class="gragra-observables">Φ a Γ</td><td>Time series</td><td class="gragra-level">—</td></tr>
<tr data-gragra-category="rigid_motion" data-gragra-formats="phasor" data-gragra-search="small translational oscillation linearized point-mass or cloud phasor rigid_motion point gragra.analytic.objects.translational_oscillation phasor"><td><a href="../models/small_translational_oscillation/">Small translational oscillation</a></td><td>point</td><td class="gragra-observables">Φ a Γ</td><td>Phasor</td><td class="gragra-level">—</td></tr>
<tr data-gragra-category="rigid_motion" data-gragra-formats="time_series" data-gragra-search="fixed-axis rotation rotating discrete mass cloud rigid_motion point_cloud gragra.analytic.objects.fixed_axis_rotation time series"><td><a href="../models/fixed_axis_rotation/">Fixed-axis rotation</a></td><td>point_cloud</td><td class="gragra-observables">Φ a Γ</td><td>Time series</td><td class="gragra-level">—</td></tr>
<tr data-gragra-category="rigid_motion" data-gragra-formats="fourier_series fourier_transform time_series" data-gragra-search="rigid-body time series and fourier response prescribed positions/rotations; periodic or finite-window fourier evaluation rigid_motion point_cloud gragra.gravity_sources.rigid_motion gragra.observables.rigid_motion_observable time series fourier series fourier transform"><td><a href="../models/rigid_body_time_series_and_fourier_response/">Rigid-body time series and Fourier response</a></td><td>point_cloud</td><td class="gragra-observables">Φ a Γ<br>Φ a Γ<br>Φ a Γ</td><td>Time series<br>Fourier series<br>Fourier transform</td><td class="gragra-level">—</td></tr>
</tbody></table>

### Atmospheric acoustics

<table class="gragra-catalogue-table">
<thead><tr><th>Model</th><th>Geometry</th><th>Φ/a/Γ</th><th>Representation</th><th>Level</th></tr></thead>
<tbody>
<tr data-gragra-category="atmospheric_acoustic" data-gragra-formats="phasor" data-gragra-search="atmospheric acoustic halfspace plane acoustic wave and rigid-ground reflection atmospheric_acoustic halfspace gragra.analytic.atmospheric.acoustic_halfspace phasor"><td><a href="../models/atmospheric_acoustic_halfspace/">Atmospheric acoustic halfspace</a></td><td>halfspace</td><td class="gragra-observables">Φ a Γ</td><td>Phasor</td><td class="gragra-level">—</td></tr>
<tr data-gragra-category="atmospheric_acoustic" data-gragra-formats="phasor" data-gragra-search="rigid acoustic cavity modes neumann modes in a sphere, cuboid or cylinder; geometry-dependent normalization and numerical limits atmospheric_acoustic cavity gragra.analytic.atmospheric.acoustic_cavity_mode phasor"><td><a href="../models/rigid_acoustic_cavity_modes/">Rigid acoustic cavity modes</a></td><td>cavity</td><td class="gragra-observables">Φ a Γ</td><td>Phasor</td><td class="gragra-level">—</td></tr>
</tbody></table>

### Atmospheric temperature

<table class="gragra-catalogue-table">
<thead><tr><th>Model</th><th>Geometry</th><th>Φ/a/Γ</th><th>Representation</th><th>Level</th></tr></thead>
<tbody>
<tr data-gragra-category="atmospheric_temperature" data-gragra-formats="phasor static" data-gragra-search="isobaric temperature perturbation fourier, gaussian and bounded temperature anomalies; density-to-gravity coupling atmospheric_temperature sphere, cuboid, cylinder, grid gragra.analytic.atmospheric.isobaric_temperature static phasor"><td><a href="../models/isobaric_temperature_perturbation/">Isobaric temperature perturbation</a></td><td>sphere, cuboid, cylinder, grid</td><td class="gragra-observables">Φ a Γ<br>Φ a Γ</td><td>Static<br>Phasor</td><td class="gragra-level">—</td></tr>
</tbody></table>

### Water Newtonian noise

<table class="gragra-catalogue-table">
<thead><tr><th>Model</th><th>Geometry</th><th>Φ/a/Γ</th><th>Representation</th><th>Level</th></tr></thead>
<tbody>
<tr data-gragra-category="water" data-gragra-formats="phasor spectral_density:asd" data-gragra-search="frozen-advection channel water noise narrow water channel; transfer functions, acceleration/strain asd water channel gragra.observables.water_channel_advection_observables gragra.observables.water_channel_advection_asd phasor spectral density (asd)"><td><a href="../models/frozen_advection_channel_water_noise/">Frozen-advection channel water noise</a></td><td>channel</td><td class="gragra-observables">Φ a Γ<br>a</td><td>Phasor<br>Spectral density (ASD)</td><td class="gragra-level">—</td></tr>
<tr data-gragra-category="water" data-gragra-formats="spectral_density:asd" data-gragra-search="waterfall and spatially uncorrelated water surface idealized falling stream or incoherent channel cells water channel, waterfall gragra.observables.water_waterfall_asd gragra.observables.water_channel_uncorrelated_asd spectral density (asd)"><td><a href="../models/waterfall_and_spatially_uncorrelated_water_surface/">Waterfall and spatially uncorrelated water surface</a></td><td>channel, waterfall</td><td class="gragra-observables">a</td><td>Spectral density (ASD)</td><td class="gragra-level">—</td></tr>
<tr data-gragra-category="water" data-gragra-formats="other" data-gragra-search="water-channel inflow step deterministic transient fourier amplitude, **not** a stationary asd water channel gragra.observables.water_edge_step_fourier_amplitude other (fourier amplitude; see api)"><td><a href="../models/water_channel_inflow_step/">Water-channel inflow step</a></td><td>channel</td><td class="gragra-observables">a</td><td>Other (Fourier amplitude; see API)</td><td class="gragra-level">—</td></tr>
</tbody></table>

### Stochastic seismic fields

<table class="gragra-catalogue-table">
<thead><tr><th>Model</th><th>Geometry</th><th>Φ/a/Γ</th><th>Representation</th><th>Level</th></tr></thead>
<tbody>
<tr data-gragra-category="stochastic_seismic" data-gragra-formats="spectral_density:csd" data-gragra-search="isotropic stochastic rayleigh field multiple test masses at depth; acceleration csd stochastic_seismic halfspace gragra.observables.stochastic_rayleigh.rayleigh_isotropic_acceleration_csd spectral density (csd)"><td><a href="../models/isotropic_stochastic_rayleigh_field/">Isotropic stochastic Rayleigh field</a></td><td>halfspace</td><td class="gragra-observables">a</td><td>Spectral density (CSD)</td><td class="gragra-level">—</td></tr>
<tr data-gragra-category="stochastic_seismic" data-gragra-formats="spectral_density:csd" data-gragra-search="isotropic stochastic elastic body waves infinite-medium p field and idealized p/s cavity-center csd stochastic_seismic fullspace, cavity gragra.observables.stochastic_bodywave.fullspace_p_isotropic_acceleration_csd gragra.observables.stochastic_bodywave.spherical_cavity_ps_isotropic_acceleration_csd spectral density (csd)"><td><a href="../models/isotropic_stochastic_elastic_body_waves/">Isotropic stochastic elastic body waves</a></td><td>fullspace, cavity</td><td class="gragra-observables">a</td><td>Spectral density (CSD)</td><td class="gragra-level">—</td></tr>
</tbody></table>

### Stochastic atmospheric fields

<table class="gragra-catalogue-table">
<thead><tr><th>Model</th><th>Geometry</th><th>Φ/a/Γ</th><th>Representation</th><th>Level</th></tr></thead>
<tbody>
<tr data-gragra-category="stochastic_atmospheric" data-gragra-formats="spectral_density:csd" data-gragra-search="stochastic atmospheric infrasound surface, underground and centered-cavern idealizations; acceleration csd stochastic_atmospheric halfspace, cavity gragra.observables.stochastic_acoustic.surface_isotropic_acoustic_acceleration_csd gragra.observables.stochastic_acoustic.underground_isotropic_acoustic_acceleration_csd spectral density (csd)"><td><a href="../models/stochastic_atmospheric_infrasound/">Stochastic atmospheric infrasound</a></td><td>halfspace, cavity</td><td class="gragra-observables">a</td><td>Spectral density (CSD)</td><td class="gragra-level">—</td></tr>
<tr data-gragra-category="stochastic_atmospheric" data-gragra-formats="spectral_density:psd" data-gragra-search="advected atmospheric temperature turbulence frozen-flow hypothesis; acceleration psd stochastic_atmospheric atmosphere gragra.observables.stochastic_temperature.uniform_advection_temperature_acceleration_psd spectral density (psd)"><td><a href="../models/advected_atmospheric_temperature_turbulence/">Advected atmospheric temperature turbulence</a></td><td>atmosphere</td><td class="gragra-observables">a</td><td>Spectral density (PSD)</td><td class="gragra-level">—</td></tr>
</tbody></table>

## Numerical methods and data sources

These are numerical/data pathways, not independent claims about which physical observables each method returns.

| Method / source | Method tags | Geometry tags | Representative APIs | Documentation |
|---|---|---|---|---|
| **Weighted point-mass sum** | numerical | point | `gragra.gravity_sources.point_sum`, `gragra.integrators.direct_point_sum` | [Point sources](point-sources.md), [static sources](static-sources.md), [core API](../api/core.md) |
| **Point-dipole sum** | numerical | point | `gragra.integrators.direct_point_dipole_sum`, `gragra.observables.displacement_transfer_acceleration` | [Fields](fields.md), [core API](../api/core.md) |
| **Shape discretization** | discretization | sphere | `gragra.gravity_sources.discretize`, `gragra.geometry.Sphere` | [Geometry](geometry.md), [static sources](static-sources.md) |
| **Grid-to-source conversion** | discretization | grid | `gragra.fields.DensityPerturbationGrid`, `gragra.fields.DisplacementGrid` | [Static sources](static-sources.md), [fields](fields.md) |
| **Volume and surface sources** | discretization | mesh | `gragra.gravity_sources.VolumeElementSource`, `gragra.gravity_sources.SurfaceMassSheetSource` | [Continuous sources](continuous-sources.md), [core API](../api/core.md) |
| **Unstructured tetrahedral fields** | discretization | mesh | `gragra.fields.piecewise_tetra_linear_sources`, `gragra.fields.piecewise_tetra_p1_density_sources` | [Fields](fields.md), [meshes](meshes.md), [physics API](../api/physics.md) |
| **Hex8 and material regions** | discretization | mesh | `gragra.fields.Hex8Sources`, `gragra.fields.MaterialRegions` | [Fields API](../api/physics.md), [meshes](meshes.md) |
| **Uniform box integration** | numerical | cuboid | `gragra.observables.integrate_box_observables`, `gragra.observables.integrate_displacement_box_observables` | [Numerical reference](../tutorials/numerical-reference.md), [reference API](../api/wave456.md) |
| **Adaptive box integration** | numerical | cuboid | `gragra.observables.adaptive_box_observables`, `gragra.observables.adaptive_displacement_box_observables` | [Numerical reference](../tutorials/numerical-reference.md), [reference API](../api/wave456.md) |
| **Directional averaging** | numerical | none | `gragra.coherent.directional_cross_average`, `gragra.adapters.lebedev_grid` | [Coherent responses](coherent.md), [numerical reference](../tutorials/numerical-reference.md) |
| **Analytical plane wave on a mesh** | numerical, discretization | mesh | `gragra.waves.PlaneWaveSources`, `gragra.fields.build_plane_wave_fixed_mesh_sources` | [Waves](waves.md), [fields API](../api/physics.md) |
| **External mesh input** | external_data | mesh | `gragra.adapters.from_meshio`, `gragra.adapters.from_meshio_volume` | [Meshing guide](meshes.md), [additional APIs](../api/experimental.md) |
| **Fixed-mesh HDF5 series** | external_data | mesh | `gragra.adapters.read_fixed_mesh_series_hdf5`, `gragra.observables.stream_observable` | [External HDF5](../external/hdf5-fields.md), [series API](../api/wave456.md) |
| **SPECFEM3D Cartesian** | external_data | mesh | `gragra.adapters.SpecfemGLLStreamReader` | [SPECFEM3D](../external/specfem3d.md) |
| **Independent high-precision reference** | numerical | cavity | `gragra.reference.centered_cavern_volume_kernel` | [Numerical reference](../tutorials/numerical-reference.md), [reference API](../api/reference.md) |

## Visualization and formula display

These functions are **packaged**; inclusion here does not establish a separate stable plotting API contract.

| Function | Description | Scope / limitations | Documentation |
|---|---|---|---|
| `plot_field` | Density/displacement field visualization | Gridded field types, not a universal unstructured-mesh viewer | [Source diagnostics](source-diagnostics.md) |
| `hist` | Weighted histogram data | Per-element density, height, distance or inverse-square | [Source diagnostics](source-diagnostics.md) |
| `plot_hist` | Histogram figure | Same source diagnostics as `hist` | [Source diagnostics](source-diagnostics.md) |
| `plot_map` | Source-by-source gravity contribution map | Discrete source / selected target; display thinning differs from physical sum | [Source diagnostics](source-diagnostics.md) |
| `plot_shape` | Ideal analytical geometry | Documented sphere/cuboid/cylinder/shell shape support varies by entry point | [Source diagnostics](source-diagnostics.md) |
| `plot_contribution_hist` | Gravity-acceleration contribution distribution | Per-element magnitude/components/projection | [Source diagnostics](source-diagnostics.md) |
| `plot_cumulative_contribution` | Cumulative contribution versus source-target distance | Signed components are not the sum of absolute magnitudes | [Source diagnostics](source-diagnostics.md) |
| `plot_result` | Static field, time series, phasor and Fourier views | Not a generic PSD/ASD plotting interface | [Source diagnostics](source-diagnostics.md) |
| `formula_to_latex` | Convert **registered** formula expressions to LaTeX | Does not parse arbitrary model equations | [Source diagnostics](source-diagnostics.md) |

## Reading the entries

- **Model groups:** one group may contain multiple API and calculation paths.
- **Observable/representation lines:** source-declared path-level unions; use each detail page to determine exact API applicability.
- **Output probes:** representative output contract only, not scientific accuracy.
- **Level:** not yet defined or assigned.

See [API reference](../api/index.md), [verification matrix](../verification/matrix.md) and [limitations](../verification/limitations.md).
