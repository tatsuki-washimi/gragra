# gragra Examples

Runnable examples covering the full public API of **gragra**.
Each script prints results to stdout and exits cleanly with no files written to disk.

Run any example with:

```bash
conda activate gragra
python examples/NN_name.py
```

Or run all of them in sequence:

```bash
for f in examples/*.py; do echo "--- $f ---"; python "$f"; done
```

---

## Example Index

| File | Description | Required extras |
|------|-------------|-----------------|
| [`01_point_mass.py`](01_point_mass.py) | Single `PointMass` at origin → acceleration at `(1,0,0)` m; magnitude equals `G_SI` | none |
| [`02_weighted_point_source.py`](02_weighted_point_source.py) | `WeightedPointSource` with signed (real) weights and complex weights; shows output dtype follows weight dtype | none |
| [`03_geometry_primitives.py`](03_geometry_primitives.py) | Build a pure sphere, discretize once, then validate, plot and run a point-sum evaluation | Astropy |
| [`04_analytic_benchmark.py`](04_analytic_benchmark.py) | Compare analytic and midpoint evaluations through target IDs and unit-bearing results | Astropy |
| [`05_gravity_gradient_tensor.py`](05_gravity_gradient_tensor.py) | Gradient tensor sign convention `diag(2,-1,-1)` and `tr(T)≈0` (softening=0 only); shows non-zero trace with Plummer softening | none |
| [`06_continuous_sources.py`](06_continuous_sources.py) | Build `VolumeElementSource` and `SurfaceMassSheetSource`; use source-frame SI arrays with low-level observables | none |
| [`07_coherent_response.py`](07_coherent_response.py) | Plane-wave displacement → `plane_wave_mass_weights` → `coherent_point_acceleration`; direction-averaged power via `directional_power_average` | none |
| [`08_meshio_surface_adapter.py`](08_meshio_surface_adapter.py) | Triangle mesh via meshio → `from_meshio` → `to_surface_mass_sheet` → observable (graceful no-op if meshio is not installed) | `pip install -e ".[mesh]"` |
| [`09_external_field_density.py`](09_external_field_density.py) | Structured regular-grid displacement field → `DisplacementGrid` → `as_density_perturbation_grid` → `as_volume_element_source` → gravity acceleration | none |
| [`10_halfspace_rayleigh.py`](10_halfspace_rayleigh.py) | Rayleigh wave complex displacement field and evanescent decay factor at several depths in a homogeneous elastic half-space (smoke example; plotting and file output disabled) | none |
| [`rayleigh_phasor_workflow.ipynb`](rayleigh_phasor_workflow.ipynb) | Public Rayleigh half-space phasor workflow: validate target regions, inspect setup plots, evaluate native potential/acceleration/gradient matrices, and check phase-origin compensation | Astropy, GWexpy, Jupyter |
| [`rigid_motion_fourier_workflow.ipynb`](rigid_motion_fourier_workflow.ipynb) | Continuous `rigid_motion` with a callable pose: `fourier_series` for a circular motion checked against its closed form, and `fourier_transform` for a passage over a finite window | Astropy, SciPy, Jupyter |
| [`rayleigh_response_workflow.ipynb`](rayleigh_response_workflow.ipynb) | Build one Rayleigh gravity response, apply it to two displacement phasor spectra, compare one application with direct evaluation, and plot a selected acceleration component | Astropy, GWexpy, Jupyter, Matplotlib |
| [`12_cross_statistics_witness.py`](12_cross_statistics_witness.py) | Bulk mass-element source modulated by a P-wave plane wave (`plane_wave_bulk_mass_weights`) combined with a witness displacement field built from the same wave parameters; computes `coherent_gravitoelastic_tensor` / `coherent_acceleration_csd` | none |
| [`13_sca_lower_limit_gate.py`](13_sca_lower_limit_gate.py) | Phase 12-b Slice 4: compare the sphere representative-center and cube/3D-center Table 1 rows with static-field checks; the floor-label rows remain source-value gates | `scipy` optional (one sub-check skipped gracefully without it) |
| [`gravity_result.py`](gravity_result.py) | Group existing SI observables and native GWexpy time/frequency matrices in Experimental `GravityResult` envelopes; the U11 lower-level SI path is shown only as a wrapping example | none |
| [`15_analytic_stage1_benchmarks.py`](15_analytic_stage1_benchmarks.py) | Stage 1 analytic benchmarks: uniform sphere, cuboid closed-form 3x3 gradient, cylinder on-axis, and spherical shell P-wave 4-component decomposition with G-free and SI observables | none |
| [`literature_c.py`](literature_c.py) | LIT-C: analytic rotated finite-cylinder acceleration/gradient and the separate numerical potential diagnostic | none |
| [`cylinder_potential.py`](cylinder_potential.py) | V: analytic infinity-zero cylinder potential, acceleration, and gradient in G-free/SI units, with proper rotation and an independent numerical API comparison | none |
| [`oriented_analytic_fields.py`](oriented_analytic_fields.py) | O: translated and rotated uniform-cuboid potential, acceleration, and gradient with G-free and SI outputs | none |
| [`trajectory_passage_and_rotation.py`](trajectory_passage_and_rotation.py) | Evaluate a prescribed point-mass passage and a rotating `PointMassCloud` with the trajectory observables | none |
| [`finite_body_mesh_comparison.py`](finite_body_mesh_comparison.py) | Build a tagged finite P1 tetra body, sum bulk and interface responses, and compare the G-free result with its mass-displacement representation | none |
| [`openfoam_density_pipeline.py`](openfoam_density_pipeline.py) | Load two bounded, manufactured utility-exported OpenFOAM density snapshots and compare their native-cell gravity inputs | `pip install -e ".[mesh]"` |
| [`static_sources_workflow.py`](static_sources_workflow.py) | Convert a density grid and in-memory `VolumeMesh` to static point sources, then run them through `point_sum` and `EvaluationResult` | none |
| [`literature_r.py`](literature_r.py) | LIT-R: literature-sourced Rayleigh-wave gravity above and below a homogeneous half-space (H19 Eq. 96-98); prints SI potential and vertical acceleration across the free surface | none |
| [`literature_b.py`](literature_b.py) | LIT-B: literature-sourced single P/SV body-wave gravity (H19 Eq. 86-88, no reflection) and the reflected-P + converted-SV composite above ground (H19 Eq. 89/90); prints SI potential and the theta=0 zero of the reflected composite | none |

---

## Notes

- **G-free helpers**: `gragra.analytic.*` functions return geometric/mass contractions
  without the gravitational constant.  Multiply by `gragra.G_SI` before comparing
  to public observables.  See [`04_analytic_benchmark.py`](04_analytic_benchmark.py).
- **Complex outputs**: Complex-weight examples produce complex128 outputs that
  represent *linear complex amplitudes or transfer-function-like responses*, not
  real-valued time-domain gravitational fields.
- **Optional adapters**: Example 08 requires `meshio` (install with `pip install -e ".[mesh]"`).
  Without it the script prints a message and exits without error.
- **GWexpy adapters**: GWexpy is a base dependency. The experimental bridge
  is not demonstrated here.
