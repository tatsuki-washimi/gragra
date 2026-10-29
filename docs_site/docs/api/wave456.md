# Wave 4--6 independent references

These NumPy-only references are independent of production Newton kernels. The
physical wrappers retain the structured `GFreeObservables` container but scale
every value to SI exactly once with `G_SI`.

These experimental symbols describe implemented calculation paths rather than
a general accuracy guarantee. Read [Limitations and non-guarantees](../verification/limitations.md)
and the narrow [SPECFEM3D Cartesian profile](../external/specfem3d.md) before
using the external-field paths for quantitative work.

## Streaming fixed-mesh exchange

The following experimental API preserves source order and trailing observable
components, uses bounded blocks, and applies `G_SI` only at the observable
layer.

::: gragra.integrators.SeriesMetadata

Stores immutable geometry, IDs, axis, and frequency-phase metadata shared by
each replayable fixed-mesh source-series provider.

::: gragra.integrators.SeriesProvider

Represents a replayable fixed-mesh source block factory with immutable global
metadata and full block-coverage validation.

::: gragra.integrators.ScalarSourceSeriesBlock

Carries a bounded scalar mass-weight slab with its source offset and sample
slice for G-free integration.

::: gragra.integrators.DisplacementSourceSeriesBlock

Carries a bounded background-mass and vector-displacement slab without
collapsing those physically distinct quantities.

::: gragra.integrators.as_series_provider

Builds a replayable fixed-mesh provider from a supported scalar, displacement,
or surface-density series.

::: gragra.integrators.iter_g_free_blocks

Yields validated bounded G-free observable blocks over sample and target
tiles, without applying the gravitational constant.

::: gragra.observables.ObservableSeriesBlock

Stores a defensive read-only observable slab with explicit sample and target
coverage slices.

::: gragra.observables.stream_observable

Converts each G-free streaming slab to its SI observable exactly once while
preserving the block coverage and trailing component axes.

::: gragra.observables.collect_observable_blocks

Collects a complete validated set of SI observable blocks into public time or
frequency axis order.

::: gragra.observables.RigidMotionBlock

Stores a defensive read-only rigid-motion observable block with explicit sample and target coverage.

::: gragra.observables.rigid_motion_observable

Evaluates the SI gravitational observable for a prescribed rigid-body motion across target points.

::: gragra.observables.stream_rigid_motion_observable

Yields streaming SI observable blocks for prescribed rigid-body motions.

::: gragra.adapters.read_fixed_mesh_series_hdf5

Reads the bounded `gragra.fixed-mesh-series` HDF5 profile after validating the
version, completion state, geometry, IDs, units, and phase metadata.

::: gragra.adapters.write_fixed_mesh_series_hdf5

Writes a fixed-mesh input series through a readback-validated partial file and
only then publishes the complete HDF5 profile.

::: gragra.adapters.write_observable_series_hdf5

Writes validated SI observable blocks with bounded readback and explicit
time/frequency metadata.

::: gragra.adapters.SpecfemGLLStreamReader

Streams canonical SPECFEM GLL slabs through shared eager-reader validation,
preserving all element-local 125-point IDs and weights.

::: gragra.analytic.GFreeObservables

The read-only potential, acceleration, and gravity-gradient container keeps
static or leading-sample axes without flattening either representation.

::: gragra.analytic.QuadratureConvergence

The immutable quadrature evidence records convergence status, error, orders,
subdivision depth, and the independent reference evaluation budget.

::: gragra.observables.point_mass_observables

Computes SI point-mass reference observables from literal independent Newton
expressions while returning the shared structured container.

::: gragra.observables.multipole_observables

Computes SI monopole-plus-second-moment reference observables for far-field
comparisons without reusing production multipole code.

::: gragra.observables.solid_cylinder_observables

Computes SI reference observables for a finite cylinder using independent
radial, angular, and axial quadrature.

::: gragra.observables.opposed_rotor_observables

Computes SI time-series observables for the fixed opposed-cylinder rotor
geometry while retaining its input sample order.

::: gragra.observables.opposed_rotor_leading_phasor

Computes the SI leading two-omega phasor from the rotor's analytic second
moment rather than from a sampled discrete Fourier transform.

::: gragra.observables.passing_cuboid_observables

Computes SI time-series observables for the prescribed passing cuboid and its
fixed target coordinates.

::: gragra.observables.integrate_box_observables

Computes SI exterior-domain box observables with independent tensor-product
quadrature and explicit material-target exclusion.

::: gragra.observables.integrate_displacement_box_observables

Computes SI linearized box-displacement observables while retaining density
and displacement as separate inputs.

::: gragra.observables.adaptive_box_observables

Computes SI box observables with adaptive independent quadrature and returns
the value together with its convergence evidence.

::: gragra.observables.adaptive_displacement_box_observables

Computes SI displacement-box observables with adaptive independent quadrature
and returns the value together with its convergence evidence.

::: gragra.observables.host_inclusion_observables

Computes SI host-and-inclusion observables after partitioning material
interfaces so the contrast is counted exactly once.

::: gragra.observables.acoustic_finite_box_observables

Computes SI finite-air-box acoustic-density observables for the documented
fixed-ground geometry and exterior targets.

::: gragra.observables.acoustic_infinite_halfspace_observables

Computes SI infinite horizontal air-halfspace observables from the independent
horizontal-Fourier reference formula.

::: gragra.observables.rayleigh_halfspace_observables

Computes SI Rayleigh halfspace observables from the independently bracketed
secular root and a single bulk-plus-surface-sheet contribution.
