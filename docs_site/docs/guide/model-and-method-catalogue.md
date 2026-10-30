<!-- GENERATED CATALOGUE: source ledger and generator are maintained separately. -->
<!-- Do not edit manually; publish a regenerated snapshot through the catalogue workflow. -->
# Models, methods and API catalogue

This catalogue is a generated publication snapshot, validated against the corresponding public `gragra` source tree.
It is organized by **physical phenomenon**. Methods, geometries and
physical observables are attributes, not competing category trees.

**Important:** presence in the catalogue does not establish scientific
acceptance or an API stability promise. See [verification](../verification/index.md),
[limitations](../verification/limitations.md) and [API policy](../api/index.md).

## Model catalogue

**Observable legend:** Φ = gravitational potential; a = gravitational
acceleration; Γ = gravity-gradient tensor. These denote available
**physical observables**, not time/frequency representations.
**Representation** and **spectral statistic** are separate axes.
A field marked **Not assessed** does **not** mean unsupported;
it means no runnable catalogue output probe has been recorded.
Only the declared API and inputs in an output probe are verified.

### Static gravity and finite masses

| Model | Method tags | Geometry tags | Φ/a/Γ | Representation | Spectral statistic | APIs / references |
|---|---|---|---|---|---|---|
| **Uniform-density sphere** | analytical | sphere | Φ a Γ | static | — | `gragra.analytic.objects.uniform_sphere`, `gragra.observables.uniform_sphere_observables`; [Analytic benchmarks](analytic-benchmarks.md), [sphere example](../examples/uniform_sphere_workflow.md) |
| **Uniform-density cuboid** | analytical | cuboid | Φ a Γ | static | — | `gragra.analytic.objects.uniform_cuboid`, `gragra.observables.cuboid_observables`; [Analytic benchmarks](analytic-benchmarks.md), [analytic models](gen1-analytic-models.md) |
| **Oriented cuboid** | analytical | cuboid | Not assessed | Not assessed | — | `gragra.observables.oriented_cuboid_observables`; [Physics API](../api/physics.md), [coordinate conventions](../concepts/conventions.md) |
| **Uniform-density cylinder** | analytical | cylinder | Φ a Γ | static | — | `gragra.analytic.objects.uniform_cylinder`, `gragra.observables.cylinder_axis_observables`, `gragra.observables.cylinder_observables`; [Analytic benchmarks](analytic-benchmarks.md), [physics API](../api/physics.md) |

### Seismic and elastic waves

| Model | Method tags | Geometry tags | Φ/a/Γ | Representation | Spectral statistic | APIs / references |
|---|---|---|---|---|---|---|
| **Spherical cavity and finite spherical shell** | analytical | cavity, shell | Not assessed | Not assessed | — | `gragra.analytic.seismic.spherical_cavity`, `gragra.analytic.seismic.spherical_shell`, `gragra.observables.spherical_shell_p_acceleration_components`; [Analytic benchmarks](analytic-benchmarks.md), [analytic models](gen1-analytic-models.md) |
| **Small-cavity wall approximation** | analytical | cavity | Not assessed | Not assessed | — | `gragra.analytic.seismic.small_cavity_wall`, `gragra.analytic.sca_wall_coupling_factor`; [Analytic models](gen1-analytic-models.md), [physics API](../api/physics.md) |
| **Rayleigh wave at an elastic halfspace** | analytical | halfspace | Not assessed | Not assessed | — | `gragra.analytic.seismic.rayleigh_halfspace`, `gragra.analytic.seismic.rayleigh_response`, `gragra.observables.rayleigh_halfspace_allspace_observables`; [Waves](waves.md), [analytic benchmarks](analytic-benchmarks.md), [Rayleigh workflow](../examples/rayleigh_phasor_workflow.md) |
| **P/SV waves at a free surface** | analytical | halfspace | Not assessed | Not assessed | — | `gragra.analytic.seismic.halfspace_psv`, `gragra.observables.halfspace_psv_observables`; [Analytic models](gen1-analytic-models.md), [waves](waves.md) |
| **Individual P/SV and reflected-P benchmarks** | analytical | halfspace | Not assessed | Not assessed | — | `gragra.observables.halfspace_single_bodywave_observables`, `gragra.observables.halfspace_reflected_p_above_observables`; [Analytic benchmarks](analytic-benchmarks.md) |
| **Causal point force** | analytical | point | Not assessed | Not assessed | — | `gragra.analytic.seismic.point_force`; [Analytic models](gen1-analytic-models.md) |

### Moving masses and rigid motion

| Model | Method tags | Geometry tags | Φ/a/Γ | Representation | Spectral statistic | APIs / references |
|---|---|---|---|---|---|---|
| **Prescribed point-mass translation** | analytical, time_domain | point | Not assessed | Not assessed | — | `gragra.analytic.objects.constant_velocity`; [Analytic models](gen1-analytic-models.md) |
| **Small translational oscillation** | analytical, time_domain | point | Not assessed | Not assessed | — | `gragra.analytic.objects.translational_oscillation`; [Analytic models](gen1-analytic-models.md) |
| **Fixed-axis rotation** | analytical, time_domain | point_cloud | Not assessed | Not assessed | — | `gragra.analytic.objects.fixed_axis_rotation`; [Analytic models](gen1-analytic-models.md) |
| **Rigid-body time series and Fourier response** | numerical, fourier | point_cloud | Not assessed | Not assessed | — | `gragra.gravity_sources.rigid_motion`, `gragra.observables.rigid_motion_observable`; [Rigid-motion Fourier guide](rigid-motion-fourier.md), [results API](../api/results.md) |

### Atmospheric acoustics

| Model | Method tags | Geometry tags | Φ/a/Γ | Representation | Spectral statistic | APIs / references |
|---|---|---|---|---|---|---|
| **Atmospheric acoustic halfspace** | analytical | halfspace | Not assessed | Not assessed | — | `gragra.analytic.atmospheric.acoustic_halfspace`; [Analytic models](gen1-analytic-models.md) |
| **Rigid acoustic cavity modes** | analytical | cavity | Not assessed | Not assessed | — | `gragra.analytic.atmospheric.acoustic_cavity_mode`; [Analytic models](gen1-analytic-models.md) |

### Atmospheric temperature

| Model | Method tags | Geometry tags | Φ/a/Γ | Representation | Spectral statistic | APIs / references |
|---|---|---|---|---|---|---|
| **Isobaric temperature perturbation** | analytical | sphere, cuboid, cylinder, grid | Not assessed | Not assessed | — | `gragra.analytic.atmospheric.isobaric_temperature`; [Analytic models](gen1-analytic-models.md), [fields API](../api/physics.md) |

### Water Newtonian noise

| Model | Method tags | Geometry tags | Φ/a/Γ | Representation | Spectral statistic | APIs / references |
|---|---|---|---|---|---|---|
| **Frozen-advection channel water noise** | analytical, spectral | channel | Φ a Γ | phasor, spectral density | ASD | `gragra.observables.water_channel_advection_observables`, `gragra.observables.water_channel_advection_asd`; [Water NN](water-newtonian-noise.md), [water API](../api/water.md) |
| **Waterfall and spatially uncorrelated water surface** | analytical, spectral | channel, waterfall | Not assessed | Not assessed | — | `gragra.observables.water_waterfall_asd`, `gragra.observables.water_channel_uncorrelated_asd`; [Water NN](water-newtonian-noise.md), [water API](../api/water.md) |
| **Water-channel inflow step** | analytical, spectral | channel | Not assessed | Not assessed | — | `gragra.observables.water_edge_step_fourier_amplitude`; [Water NN](water-newtonian-noise.md), [water API](../api/water.md) |

### Stochastic seismic fields

| Model | Method tags | Geometry tags | Φ/a/Γ | Representation | Spectral statistic | APIs / references |
|---|---|---|---|---|---|---|
| **Isotropic stochastic Rayleigh field** | stochastic, spectral | halfspace | a | spectral density | CSD | `gragra.observables.stochastic_rayleigh.rayleigh_isotropic_acceleration_csd`; [Stochastic NN](stochastic-nn.md) |
| **Isotropic stochastic elastic body waves** | stochastic, spectral | fullspace, cavity | Not assessed | Not assessed | — | `gragra.observables.stochastic_bodywave.fullspace_p_isotropic_acceleration_csd`, `gragra.observables.stochastic_bodywave.spherical_cavity_ps_isotropic_acceleration_csd`; [Stochastic NN](stochastic-nn.md) |

### Stochastic atmospheric fields

| Model | Method tags | Geometry tags | Φ/a/Γ | Representation | Spectral statistic | APIs / references |
|---|---|---|---|---|---|---|
| **Stochastic atmospheric infrasound** | stochastic, spectral | halfspace, cavity | Not assessed | Not assessed | — | `gragra.observables.stochastic_acoustic.surface_isotropic_acoustic_acceleration_csd`, `gragra.observables.stochastic_acoustic.underground_isotropic_acoustic_acceleration_csd`; [Stochastic NN](stochastic-nn.md), [limitations](../verification/limitations.md) |
| **Advected atmospheric temperature turbulence** | stochastic, spectral | atmosphere | Not assessed | Not assessed | — | `gragra.observables.stochastic_temperature.uniform_advection_temperature_acceleration_psd`; [Stochastic NN](stochastic-nn.md) |

## Numerical methods and data sources

These are numerical/data pathways, not independent claims about
which physical observables each method returns.

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

These functions are **packaged**; inclusion here does not establish
a separate stable plotting API contract.

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

- **Models:** physical domains and representative entry points; one row
  may cover multiple implementations and should not be read as a unique API.
- **Method / geometry tags:** categorical search attributes, not a
  guarantee that every combination of tags is supported.
- **Physical observables, representations and statistics:** only
  **runnable-probed** combinations are listed; all other entries remain
  explicitly **Not assessed** until a corresponding test exists.
- **Science:** the runtime probes check selected output contracts,
  not source attribution, accuracy across all parameter space, or
  independent physics validation.

See [the complete API reference](../api/index.md),
[numerical reference](../tutorials/numerical-reference.md),
[verification matrix](../verification/matrix.md) and
[limitations](../verification/limitations.md).
