
# Fields, waves and coherent

This page covers everything downstream of a raw displacement or a plane
wave: turning a field into mass elements, building kinematic wave
solutions, and combining per-direction responses into coherent statistics.
None of it multiplies by [`G_SI`](top-level.md) except the `Observables`
re-exports at the end, which are documented in full on
[Sources, targets and geometry](core.md) and [Top-level API](top-level.md)
and are only cross-referenced here.

## Fields

`gragra.fields` turns a displacement or density-perturbation field into the
mass elements an observable can integrate: a bulk term from
δρ = −∇·(ρ₀ξ), a surface term from the displacement of an interface, or
both. See [Displacement and density-perturbation fields](../guide/fields.md)
for the physics; this section is the reference for the container types and
the diagnostic helpers.

::: gragra.fields.DensityTimeSeries

An immutable real density series on fixed volume elements, retaining the
strictly increasing time axis, source geometry, IDs, and density convention.

::: gragra.fields.DensityPhasors

An immutable complex density-amplitude series on fixed volume elements, with
positive frequencies and explicit time and spatial phase references.

::: gragra.fields.FixedMeshDisplacementSeries

An immutable fixed-mesh displacement series that keeps background mass and
vector displacement separate for time or frequency-domain observations.

::: gragra.fields.FixedMeshSurfaceDensitySeries

An immutable fixed-quadrature surface-density series whose signed density and
area are separate from the unit normals and immutable face metadata.

::: gragra.fields.DisplacementGrid

A bulk displacement field on a regular or rectilinear grid, paired with a
[`BackgroundDensityModel`](#gragra.fields.BackgroundDensityModel) to derive
δρ = −∇·(ρ₀ξ) via finite differences.

::: gragra.fields.DensityPerturbationGrid

A bulk density-perturbation field already expressed as δρ on a
regular/rectilinear grid, for callers who have δρ directly rather than a
displacement to differentiate.

::: gragra.plotting.plot_field

Plot a density perturbation or displacement field faithful to native grid
coordinates without modifying the input field object.

::: gragra.plotting.hist

Histogram one per-element diagnostic (density, height, distance, or inverse
square) of a discrete point-sum setup as a GWexpy `Histogram`, with counts or
volumes per bin. See [Source diagnostics](../guide/source-diagnostics.md) for
a short example and the units and frame rules.

::: gragra.plotting.plot_hist

Draw the histogram that `hist` returns as a step plot on a new or supplied
empty Matplotlib axes and return the figure. It draws the same values and
edges as `hist` with the same arguments, and it does not re-bin them.

::: gragra.plotting.plot_map

Plot Newtonian kernel or gravitational acceleration contributions of discrete
source elements across space in 3D or 2D projections. Total physical sum is
preserved and agrees with PointSumModel evaluation within floating-point
tolerance (rtol=1e-12, atol=1e-20); it is not bitwise identical. See
[Source diagnostics](../guide/source-diagnostics.md) for details.

::: gragra.plotting.plot_shape

Render 3D geometric shapes of analytical sources as surfaces or wireframes on Matplotlib 3D axes.

::: gragra.plotting.formula_to_latex

Format verified analytical source equations and potential formulas into LaTeX math expressions.

::: gragra.plotting.plot_contribution_hist

Display histogram distributions of directional or magnitude gravitational acceleration contributions from discrete elements.

::: gragra.plotting.plot_cumulative_contribution

Graph the radial cumulative profile of gravitational acceleration as a function of source-to-target distance.

::: gragra.plotting.plot_result

Visualize physical fields, time series, or spectra directly from an EvaluationResult container.

### Gen1 temperature fields

These immutable inputs describe one prescribed temperature perturbation. The
analytic model applies the isobaric ideal-gas relation
`δρ = -rho0 * δT / T0`; finite shapes describe bounded anomalies in a
homogeneous background, while `TemperaturePerturbationGrid` is converted through the
existing volume-element and `point_sum` route. It does not replace a continuum
analytic field with an implicit grid approximation.

::: gragra.fields.TemperatureFourierMode

Without `frequency`, a real plane-wave amplitude gives a Static cosine.
Supplying `frequency` selects Phasor output for either real or complex
amplitude, with a positive spatial exponential and the shared negative-time
reconstruction convention.

::: gragra.fields.TemperatureGaussian

An all-space isotropic Gaussian temperature anomaly with finite center limits
and a zero-at-infinity potential reference.

::: gragra.fields.TemperaturePerturbationGrid

A real Kelvin array on explicit increasing length edges. Cells use their
centers and product edge widths, with C-order flattening and z varying fastest.
Its public `field_definition` and `point_sum_provenance()` provide the
metadata for the existing `point_sum` route.

::: gragra.fields.UnstructuredDisplacementField

The unstructured-mesh analogue of `DisplacementGrid`: a bulk displacement
field on a linear tetrahedral mesh, from which the bulk mass-element weights
are computed per element rather than per grid cell.

::: gragra.fields.SurfaceDisplacementField

A displacement field sampled on an unstructured triangular surface mesh,
used to build the surface term — the mass redistributed as an interface
moves — independently of any bulk contribution.

::: gragra.fields.TetraLinearizedSources

Stores read-only mass-displacement, bulk, and interface source terms for a
finite, piecewise-constant-density P1 tetra body. The bulk and interface terms
form one representation; the mass-displacement term is an alternative for comparison.

::: gragra.fields.TetraRepresentationReport

Records G-free responses for the separate bulk and interface terms, their sum,
and the alternative mass-displacement response with residuals per target.

::: gragra.fields.piecewise_tetra_linear_sources

Builds the two finite-body representations from a conforming tetra field and
cell densities. It matches faces by vertex ID, treats the exterior as vacuum,
and rejects duplicate or non-manifold face ownership.

::: gragra.fields.compare_tetra_linearized_representations

Computes G-free potential, acceleration, or gradient responses. It sums bulk
and interface responses, then compares that result with the mass-displacement
response; it never adds the two representations together.

::: gragra.fields.TetraP1DensityLinearizedSources

Stores read-only degree-2 tetra and face-quadrature source terms for a finite
body with cell-local P1 density. It is separate from the existing P0 result so
the old face-metadata contract remains unchanged.

::: gragra.fields.piecewise_tetra_p1_density_sources

Builds the P1-density representation from each tetra's local-node density
values. Adjacent cells may have discontinuous traces; the exterior remains
vacuum and the displacement field remains conforming P1.

::: gragra.fields.compare_tetra_p1_density_representations

Compares, without adding, the G-free bulk-plus-interface response and the
mass-displacement response for the same cell-local P1-density finite body.

::: gragra.fields.BackgroundDensityModel

A `Protocol` describing anything that can supply a background density ρ₀ at
arbitrary points: the minimal interface `DisplacementGrid` needs to compute
∇·(ρ₀ξ).

::: gragra.fields.DifferentiableBackgroundDensityModel

The stricter `Protocol` variant that can also supply ∇ρ₀ analytically,
avoiding the finite-difference approximation `BackgroundDensityModel` alone
would require for the density-gradient term.

::: gragra.fields.OrientationReport

The diagnostic result of `check_triangle_orientation`: whether a mesh's
winding is internally consistent and whether it forms a closed manifold,
plus which facets disagree if not.

::: gragra.fields.check_triangle_orientation

Diagnoses the winding consistency and closedness of a triangle mesh before
it is used as a surface source — a mesh with inconsistent winding produces
a surface term whose contributions partially cancel instead of summing.

::: gragra.fields.reorient_triangles

Consistently reorients a triangle mesh's winding, typically applied after
`check_triangle_orientation` reports an inconsistency.

::: gragra.fields.DipoleConsistencyReport

The diagnostic result of `dipole_consistency_check`: a G-free comparison
between the divergence-based (bulk + surface) path and the direct
dipole-contraction path for the same displacement field, which should agree
under the six equivalence conditions in
[kernel_contract.md §2.2](https://github.com/tatsuki-washimi/gragra/blob/main/docs/architecture/kernel_contract.md).

::: gragra.fields.dipole_consistency_check

Cross-checks the divergence path (bulk mass elements + surface mass sheet)
against the dipole/total path (`displacement_transfer_acceleration`) for the
same field; a non-trivial residual signals a broken equivalence condition,
not necessarily a bug — see [guide/fields.md](../guide/fields.md).

::: gragra.fields.batched_displacement_grid_weights

Computes batched bulk mass-element weights from a batch of structured-grid
displacement snapshots in one call, for time- or parameter-batched fields.

::: gragra.fields.batched_unstructured_weights

The unstructured-mesh analogue of `batched_displacement_grid_weights`: batched
bulk weights from a batch of tetrahedral-mesh displacement snapshots.

::: gragra.fields.batched_surface_weights

Batched surface-term weights from a batch of surface-mesh displacement
snapshots, the surface-term counterpart to the two bulk batching helpers
above.

::: gragra.fields.adiabatic_pressure_density_perturbation

Converts a physical pressure perturbation to its adiabatic density
perturbation using δρ = δp/c_s², with explicit SI pressure and sound speed.

::: gragra.fields.isobaric_temperature_density_perturbation

Converts a temperature perturbation to its ideal-gas isobaric density
perturbation using δρ = −ρ₀δT/T₀.

::: gragra.fields.FixedMeshDisplacementSources

Stores fixed-mesh displacement field source terms for time or frequency evaluations.

::: gragra.fields.FixedMeshSurfaceSources

Stores fixed-mesh surface density and normal source terms for boundary interfaces.

::: gragra.fields.Hex8DisplacementField

Represents an 8-node hexahedral (Hex8) finite-element displacement field.

::: gragra.fields.Hex8Sources

Stores mass-element sources derived from a Hex8 displacement field.

::: gragra.fields.MaterialRegions

Defines piecewise-continuous material density and elastic parameter regions.

::: gragra.fields.build_fixed_mesh_displacement_sources

Builds mass displacement sources from a fixed mesh and nodal displacement vectors.

::: gragra.fields.build_fixed_mesh_surface_sources

Builds interface mass sheet sources from a fixed surface mesh.

::: gragra.fields.build_plane_wave_fixed_mesh_sources

Builds fixed-mesh mass sources excited by a propagating plane wave.

## Waves

`gragra.waves` supplies closed-form *kinematic* field constructors — plane
waves, Rayleigh eigenfunctions, stochastic P/S realizations — that you can
feed into `gragra.coherent` or `gragra.fields` as input. It does not solve a
dynamical PDE for you beyond the Rayleigh secular equation itself. See
[guide/waves.md](../guide/waves.md) for the physics and the confirmed-zero
cases (pure S wave, rigid translation).

::: gragra.waves.plane_wave_phase

The scalar plane-wave phase k·(k̂·r) + phase0, the argument that every other
plane-wave helper on this page builds a complex factor or displacement from.

::: gragra.waves.plane_wave_complex_factor

The complex factor exp(i·sign·phase) for a plane wave, with `sign` fixing
the time-convention (e^{+iωt} vs e^{-iωt}) a caller commits to.

::: gragra.waves.plane_wave_displacement

The complex displacement field of a plane wave: amplitude times polarization
times `plane_wave_complex_factor`, for the wave type (P/S1/S2/SH/SV)
requested.

::: gragra.waves.evanescent_complex_factor

The complex factor for an evanescent (exponentially decaying with depth)
wave component, used where a wave is inhomogeneous rather than uniformly
propagating.

::: gragra.waves.normalize_directions

Normalizes an array of direction vectors to unit length, guarding against
zero/non-finite input rather than dividing silently.

::: gragra.waves.orthonormal_basis_from_direction

Builds an orthonormal basis (e1, e2) orthogonal to a given propagation
direction k̂, the basis the S-wave polarizations are expressed in.

::: gragra.waves.p_wave_polarization

The P-wave polarization direction, which is simply the normalized
propagation direction k̂ (longitudinal motion).

::: gragra.waves.s_wave_basis

The S-wave polarization basis (e1, e2) orthogonal to k̂, from which S1, S2,
SH and SV polarizations are constructed.

::: gragra.waves.displacement_direction

The unit displacement direction for a requested wave type (P, S1, S2, SH,
SV), dispatching to `p_wave_polarization` or `s_wave_basis` as appropriate.

::: gragra.waves.rayleigh_secular

The Rayleigh secular function R(c) in the standard Aki & Richards form,
whose root in phase speed c is the Rayleigh wave speed for a homogeneous
elastic half-space.

::: gragra.waves.rayleigh_speed

The Rayleigh phase speed c_R, found as the root of `rayleigh_secular` via
NumPy-based bisection.

::: gragra.waves.rayleigh_displacement

The complex Rayleigh-wave displacement field u(r) in the half-space,
combining the horizontal and vertical eigenfunctions with their fixed
relative phase (see [guide/waves.md](../guide/waves.md) for the phase
convention).

::: gragra.waves.rayleigh_ellipticity

The depth-dependent Rayleigh wave H/V ellipticity |u_x(h)| / |u_z(h)|, a
diagnostic quantity independent of absolute amplitude.

::: gragra.waves.free_surface_reflection_vertical

The free-surface displacement reflection coefficients for vertical
incidence, needed to combine an incident and reflected body wave into the
correct free-surface displacement.

::: gragra.waves.surface_displacement_factor_vertical

The free-surface displacement-doubling factor at vertical incidence — the
scalar consequence of `free_surface_reflection_vertical` for the common
vertically-incident case.

::: gragra.waves.StochasticPSParameters

The parameter container for `stochastic_ps_realization`: propagation
direction, vertical unit, wave types, amplitudes, wavenumbers and phase
offsets for a correlated P/S wave pair.

::: gragra.waves.stochastic_ps_realization

Generates a stochastic P/S correlation-preserving plane-wave ensemble
realization from a `StochasticPSParameters` set — a randomized field for
Monte Carlo style studies rather than a single deterministic wave.

::: gragra.waves.PlaneWave

Kinematic plane wave container with propagation direction, frequency, phase velocity, and amplitude.

::: gragra.waves.PlanePWave

Compressional plane P-wave with longitudinal polarization along the propagation vector.

::: gragra.waves.PlaneWaveSources

Mass-element source representation excited by a kinematic plane wave.

## Coherent

`gragra.coherent` turns a kinematic wave field into mass-element weights and
computes cross-statistics between per-direction or per-realization
responses. See
[Coherent responses and cross statistics](../guide/coherent.md) for the G⁰
witness vs. G¹ vs. G² distinction and the element-wise, conjugate-second-
argument convention that the cross-average functions below follow.

::: gragra.coherent.plane_wave_mass_weights

The plane-wave coherent mass weights: base (real) mass weights modulated by
`plane_wave_complex_factor`, giving each source element a complex,
direction- and phase-dependent weight.

::: gragra.coherent.plane_wave_bulk_mass_weights

The polarization-projected bulk mass weights δm = −iρ₀k(k̂·ê)ξV·e^{ik·r} for
a plane wave — identically zero for a pure S wave (k̂·ê = 0), which is a
physically correct null result, not a bug.

::: gragra.coherent.directional_power_average

The directional power average of a response: ⟨|response|²⟩ averaged over a
set of directions (e.g. a Lebedev grid), under the "directions are
uncorrelated" ensemble model.

::: gragra.coherent.directional_cross_average

The directional cross average between two responses, element-wise with the
second argument conjugated (not a dot product) — the primitive the
gravitoelastic and CSD observables are built from.

::: gragra.coherent.directional_outer_cross_average

The direction-averaged component-outer cross tensor of two vector
responses, the tensor-valued generalization of `directional_cross_average`.

::: gragra.coherent.ensemble_outer_cross_average

The ensemble outer cross average over a set of realizations (rather than
directions), used with `stochastic_ps_realization`-style Monte Carlo
ensembles.

::: gragra.observables.coherent_point_potential

Computes the coherent gravitational potential from a point mass excitation.

## Analytic

`gragra.analytic` supplies the closed-form benchmarks this project treats
as its primary verification strategy (analytic tests over finite-difference
checks). Every function here is G-free — multiply by
[`G_SI`](top-level.md) before comparing to a public observable. See
[Analytic benchmarks](../guide/analytic-benchmarks.md).

### General free-surface P/SV waves

The uniform, lossless half-space at z<0 supports incident P or SV and both
reflected modes, including converted evanescent P. Potential, acceleration
and gradient are analytic on either side of z=0. Exact surface and grazing
points are excluded, as is the nonunique zero-Lame-lambda SV critical point.

::: gragra.analytic.HalfspacePSVReflection

Stores the wave-number ratios and potential/displacement reflection ratios
with explicit polarization conventions.

::: gragra.analytic.halfspace_psv_reflection

Evaluates traction-free reflection coefficients without a matrix solve.
The evanescent branch decays into the medium.

::: gragra.analytic.halfspace_psv_observables_g_free

Returns G-free ground/underground potential, acceleration and gradient for
the same physical displacement field, with frequency-first arrays.

::: gragra.observables.halfspace_psv_observables

Converts the general P/SV result to SI at the observable boundary.
Its complex incident amplitude is referenced at the fixed phase origin.

### Static and other wave benchmarks

The high-level object factory binds a pure sphere geometry and density to an
analytic model whose `.at()` request returns validated, unit-bearing results.

::: gragra.analytic.objects.uniform_sphere

::: gragra.analytic.uniform_sphere_potential

The G-free gravitational potential contraction of a uniform sphere, the
standard closed-form target for the `Sphere` discretization's convergence
test.

::: gragra.analytic.uniform_sphere_acceleration

The G-free gravitational acceleration contraction of a uniform sphere,
exact both inside and outside the sphere.

::: gragra.analytic.uniform_sphere_gradient

The G-free gravity-gradient tensor contraction of a uniform sphere,
completing the potential/acceleration/gradient triple for the sphere
benchmark.

::: gragra.analytic.spherical_cavity_bulk_p

The G-free Newtonian acceleration amplitude of the P-wave bulk term for a
spherical cavity, one term of the small-cavern approximation.

::: gragra.analytic.spherical_cavity_total_p

The fixed spherical cavity series evaluates bulk, wall, and total potential,
acceleration, and gradient at general interior and material-side targets.
It uses analytic solid harmonics and preserves the center-only functions above.

::: gragra.analytic.spherical_cavity_observables_g_free

The corresponding SI entry point converts each of the three component fields
through the existing gravitational-observable conversion boundary.

::: gragra.observables.spherical_cavity_observables

The G-free Newtonian acceleration amplitude of the P-wave total term
(bulk + wall) for a spherical cavity.

::: gragra.analytic.spherical_cavity_wall

The G-free Newtonian acceleration amplitude of the cavity-wall term alone
for a spherical cavity.

::: gragra.analytic.spherical_shell_dipole_factor

The dimensionless spherical-shell dipole factor −j₁(x)/x (a spherical
Bessel function ratio), a building block of the cavity closed forms above.

::: gragra.analytic.halfspace_transfer_bulk

The dimensionless bulk transfer function T_bulk(kH) = 1 − 2cos(kH) for a
half-space at depth H, one term of the full vertical-incidence transfer.

::: gragra.analytic.halfspace_transfer_full

The dimensionless full transfer function T_full(kH) = −2cos(kH) =
T_bulk + T_sheet for a half-space at depth H.

::: gragra.analytic.rayleigh_depth_reduction

The kinematic Rayleigh displacement-magnitude depth-reduction factor R(h),
used to scale a surface-referenced Rayleigh amplitude to depth h.

::: gragra.analytic.sca_wall_coupling_factor

Looks up the dimensionless small-cavern-approximation (SCA) cavern-wall
coupling factor from Harms et al. 2022 Table 1 — a plain Python `float`,
not an array, because the underlying formula has no wavenumber or
component axis (see the array-conventions exception this function is
documented under).

::: gragra.analytic.sca_wall_acceleration

The G-free small-cavern-limit cavern-wall acceleration amplitude, built
from `sca_wall_coupling_factor`; also returns a plain `float`.

::: gragra.analytic.sphere_sca_wall_coupling_tensor_g_free

Evaluates the dimensionless sphere wall-coupling tensor from an independent
surface quadrature for general interior or exterior targets; physical wall
points are rejected.

::: gragra.analytic.cuboid_sca_wall_coupling_tensor_g_free

Evaluates the dimensionless cuboid wall-coupling tensor, including an
explicit proper local-to-world rotation when requested.

::: gragra.analytic.sca_wall_acceleration_from_tensor_g_free

Applies the SCA wall convention `delta_a/G = -rho * C @ displacement` to a
tensor and returns the G-free acceleration amplitude.

::: gragra.analytic.cuboid_filled_internal_field

The G-free static gravitational field of a filled uniform rectangular
prism, evaluated at points that may lie inside the prism (unlike a
point-mass sum, which is undefined at r=0).

::: gragra.analytic.cylinder_filled_axis_field

The G-free filled cylinder's scalar on-axis gravitational field.

::: gragra.analytic.cylinder_sca_wall_coupling_tensor_g_free

The dimensionless cylinder SCA wall tensor is minus the unit-density filled
cylinder gradient. It supports general interior/exterior points and proper
rotation, and excludes the physical wall. Density and displacement are
applied by the separate SCA contraction; this function does not multiply by G.

The G-free static on-axis gravitational field of a filled uniform cylinder;
restricted to the axis because no general off-axis closed form is
implemented.

::: gragra.analytic.GFreeGravityValues

Frozen container holding G-free potential, acceleration, and gradient arrays evaluated from an analytic source.

::: gragra.analytic.GFreeSphericalShellPComponents

Frozen container holding G-free acceleration components (bulk, inner_wall, outer_boundary, total) for a spherical shell cavity under P-wave excitation.

::: gragra.analytic.GFreeSphericalShellSComponents

Frozen container holding G-free acceleration components (bulk, inner_wall, outer_boundary, total) for a spherical shell cavity under S-wave excitation. `bulk` is always identically zero because S waves are divergence-free.

::: gragra.analytic.uniform_sphere_observables_g_free

Bundles the G-free potential, acceleration, and gradient tensor of a uniform sphere into a GFreeGravityValues container.

::: gragra.analytic.cuboid_potential_g_free

Evaluates the G-free gravitational potential of a uniform rectangular prism using 3D closed-form vertex antiderivatives.

::: gragra.analytic.cuboid_acceleration_g_free

Evaluates the G-free gravitational acceleration vector of a uniform rectangular prism using 3D closed-form vertex antiderivatives.

::: gragra.analytic.cuboid_gradient_g_free

Evaluates the G-free symmetric 3x3 gravity-gradient tensor of a uniform rectangular prism using 3D closed-form vertex antiderivatives.

::: gragra.analytic.cuboid_observables_g_free

Bundles the G-free potential, acceleration, and 3x3 gradient tensor of a uniform rectangular prism into a GFreeGravityValues container.

::: gragra.analytic.oriented_cuboid_observables_g_free

Evaluates the same cuboid closed form after an explicit proper
local-to-world rotation. The potential is invariant, while acceleration and
the 3x3 gradient are transported with `Q` and `Q @ gradient @ Q.T`; invalid
rotation matrices are rejected rather than projected onto SO(3).

::: gragra.analytic.cylinder_potential_axis_g_free

Evaluates the G-free gravitational potential on the symmetry axis of a uniform cylinder using closed-form antiderivatives.

::: gragra.analytic.cylinder_gradient_axis_g_free

Evaluates the G-free 3x3 gravity-gradient tensor on the symmetry axis of a uniform cylinder using longitudinal derivatives and Poisson constraints.

::: gragra.analytic.cylinder_axis_observables_g_free

Bundles the G-free on-axis potential, acceleration, and gradient tensor of a uniform cylinder into a GFreeGravityValues container.

::: gragra.analytic.cylinder_observables_g_free

Evaluates a z-aligned uniform cylinder at general target points with an
opt-in cylindrical Gauss-quadrature path. This is a numerical semi-analytic
reference/exploratory API, not the Conway closed elliptic-integral formula;
`quadrature_order` controls the radial and axial orders. The exact on-axis
closed form is reused, and physical side/end-boundary targets are rejected.

::: gragra.analytic.cylinder_acceleration_gradient_g_free

Returns analytic acceleration and the full gradient as a pair of read-only
arrays. Real complete elliptic integrals are supplemented by analytic
near-axis and far-field series. `rotation_matrix` maps local to world
coordinates and must be a proper rotation; the physical boundary is rejected.
The separate bundled API above retains the numerical potential diagnostic.

::: gragra.analytic.cylinder_analytic_observables_g_free

Returns the analytic finite-cylinder potential, acceleration, and gradient
in one read-only container. The infinity-zero potential uses convergent
analytic elliptic/logarithmic series, not spatial quadrature or a claimed
finite closed form. Proper rotation and material-boundary exclusions match
the analytic acceleration/gradient pair; the numerical API stays separate.

::: gragra.analytic.rayleigh_halfspace_allspace_observables_g_free

G-free Rayleigh-wave gravity above (H19 Eq. 96) and below (H19 Eq. 97) a
homogeneous elastic half-space, with acceleration and gradient from this
module's own analytic derivative of those closed forms. A separate entry
point from the legacy ground-only `rayleigh_halfspace_observables_g_free`
(see the Wave 4-6 reference page); see `docs/design/literature-r.md` for
the full literature mapping.

::: gragra.analytic.halfspace_single_bodywave_observables_g_free

G-free gravity of a single upward-propagating plane P or SV wave, without
compositing a free-surface reflection (H19 Eq. 86-88), with acceleration
and gradient from this module's own analytic derivative of those closed
forms. `wave_type="P"` targets may mix `z>0` and `z<0`; see
`docs/design/literature-b.md` for the full literature mapping.

::: gragra.analytic.halfspace_reflected_p_above_observables_g_free

G-free gravity above ground of an incident P + reflected P + converted SV
composite. The production observable uses the coefficient reconstructed from
the same traction-free displacement field and its actual surface mass sheet;
the literal H19 Eq. (90) coefficient is retained separately in the private
`_harms2019_reflected_p_delta` helper. Acceleration and gradient come from
this module's own analytic derivative. Ground only (`z>0`); see
`docs/design/literature-b.md`.

::: gragra.analytic.spherical_shell_p_acceleration_components_g_free

Calculates the G-free 4-component acceleration decomposition (bulk, inner wall, outer boundary, total) at the center of a spherical shell cavity under a P-wave.

::: gragra.analytic.spherical_shell_p_infinite_difference_g_free

Calculates the G-free acceleration difference between an infinite medium and a finite spherical shell cavity under P-wave excitation.

::: gragra.analytic.spherical_shell_s_acceleration_components_g_free

Calculates the G-free 4-component acceleration decomposition (bulk, inner wall, outer boundary, total) at the center of a spherical shell cavity under an S-wave. Requires separate propagation and polarization directions, validated to be orthogonal (S-wave transversality); the closed form itself does not depend on the propagation direction.

::: gragra.analytic.spherical_shell_s_infinite_difference_g_free

Calculates the G-free acceleration difference between an infinite medium and a finite spherical shell cavity under S-wave excitation.

## Observables

The seventeen `gragra.observables` symbols — `point_potential`,
`point_acceleration`, `point_gravity_gradient`, `coherent_point_acceleration`,
`coherent_point_gravity_gradient`,
`coherent_gravitoelastic_tensor`, `coherent_acceleration_csd`,
`displacement_transfer_acceleration`, `displacement_transfer_potential`,
`displacement_transfer_gravity_gradient`, `trajectory_potential`,
`trajectory_acceleration`, `trajectory_gravity_gradient`,
`rigid_transform_potential`, `rigid_transform_acceleration`,
`rigid_transform_gravity_gradient` and `displacement_field_acceleration`
— are the layer that multiplies by
[`G_SI`](top-level.md) and are documented in full on
[Sources, targets and geometry](core.md#gragra.integrators.direct_point_sum)'s
sibling page, [Top-level API](top-level.md), and their dedicated guide pages
([Point sources](../guide/point-sources.md), [Observables](../guide/observables.md),
[Coherent responses and cross statistics](../guide/coherent.md)). They are
not repeated here to avoid two `:::` directives rendering the same symbol,
except for the two subpackage-only displacement observables below.

::: gragra.observables.displacement_transfer_potential

Returns the first-order potential perturbation of discrete, non-negative mass
elements under real or complex displacement vectors. Its NumPy implementation
uses the G-free displacement kernel and returns a read-only `(M,)` array.

::: gragra.observables.displacement_transfer_gravity_gradient

Returns the first-order gravity-gradient perturbation for the same discrete
mass-element model as `displacement_transfer_potential`. Its read-only
`(M, 3, 3)` result is symmetric, and Plummer softening can give it nonzero trace.

::: gragra.observables.trajectory_potential

Returns read-only potential samples shaped `(P, M)` for a prescribed positive
point trajectory.

::: gragra.observables.trajectory_acceleration

Returns read-only acceleration samples shaped `(P, M, 3)` for a prescribed
positive point trajectory.

::: gragra.observables.trajectory_gravity_gradient

Returns read-only gravity-gradient samples shaped `(P, M, 3, 3)` for a
prescribed positive point trajectory.

::: gragra.observables.rigid_transform_potential

Returns read-only potential samples shaped `(P, M)` for a `PointMassCloud`
under a prescribed rigid transform.

::: gragra.observables.rigid_transform_acceleration

Returns read-only acceleration samples shaped `(P, M, 3)` for a `PointMassCloud`
under a prescribed rigid transform.

::: gragra.observables.rigid_transform_gravity_gradient

Returns read-only gravity-gradient samples shaped `(P, M, 3, 3)` for a
`PointMassCloud` under a prescribed rigid transform.

### Analytic SI Observables

::: gragra.observables.GravityObservables

Frozen container holding potential, acceleration, and gradient arrays in standard physical SI units (m^2/s^2, m/s^2, s^-2).

::: gragra.observables.to_gravity_observables

Converts a GFreeGravityValues bundle to physical SI GravityObservables by multiplying every component by the gravitational constant G_SI exactly once.

::: gragra.observables.SphericalShellPComponents

Frozen container holding physical SI acceleration components (bulk, inner_wall, outer_boundary, total) in m/s^2 for a spherical shell cavity under P-wave excitation.

::: gragra.observables.to_spherical_shell_p_components

Converts a GFreeSphericalShellPComponents bundle to physical SI SphericalShellPComponents by multiplying every component by G_SI exactly once.

::: gragra.observables.SphericalShellSComponents

Frozen container holding physical SI acceleration components (bulk, inner_wall, outer_boundary, total) in m/s^2 for a spherical shell cavity under S-wave excitation. `bulk` is always identically zero.

::: gragra.observables.to_spherical_shell_s_components

Converts a GFreeSphericalShellSComponents bundle to physical SI SphericalShellSComponents by multiplying every component by G_SI exactly once.

::: gragra.observables.uniform_sphere_observables

Evaluates the gravitational potential, acceleration, and gradient tensor of a uniform sphere in physical SI units.

::: gragra.observables.cuboid_observables

Evaluates the closed-form potential, acceleration, and 3x3 gravity-gradient tensor of a uniform cuboid in physical SI units.

::: gragra.observables.oriented_cuboid_observables

Evaluates an arbitrarily translated and properly rotated uniform cuboid in
physical SI units by scaling the G-free oriented wrapper exactly once.

::: gragra.observables.sca_wall_acceleration_from_tensor

Applies an SCA wall-coupling tensor and multiplies the resulting G-free
acceleration by `G_SI` exactly once.

::: gragra.observables.cylinder_axis_observables

Evaluates the closed-form on-axis potential, acceleration, and 3x3 gradient tensor of a uniform cylinder in physical SI units.

::: gragra.observables.cylinder_observables

Evaluates the general-point z-aligned uniform-cylinder observables in SI units
by scaling `gragra.analytic.cylinder_observables_g_free` by `G_SI`. The
general-point implementation is an opt-in numerical semi-analytic path; it is
not a claim of Conway's closed elliptic-integral solution.

::: gragra.observables.cylinder_acceleration_gradient

SI acceleration and gradient from the analytic cylinder route, multiplied by
`G_SI` once. Returns `(acceleration, gradient)` with shapes `(M,3)`, `(M,3,3)`.

::: gragra.observables.cylinder_analytic_observables

SI potential, acceleration, and gradient from the analytic cylinder triple,
scaled by `G_SI` once at the observable boundary. Supports proper rotations
and returns read-only arrays with shapes `(M,)`, `(M,3)`, and `(M,3,3)`.

::: gragra.observables.rayleigh_halfspace_allspace_observables

SI potential/acceleration/gradient of a Rayleigh wave above and below a
homogeneous elastic half-space (H19 Eq. 96-98), multiplying the G-free
`gragra.analytic.rayleigh_halfspace_allspace_observables_g_free` by `G_SI`
exactly once. A separate entry point from the legacy ground-only
`rayleigh_halfspace_observables`.

The high-level seismic namespace accepts a `HalfSpace`, material quantities,
and an explicit complex phasor contract, then evaluates potential,
acceleration, and gradient together as frequency matrices.

::: gragra.analytic.seismic.rayleigh_halfspace

`rayleigh_response` builds a reusable Newtonian transfer from vertical surface
displacement to potential, acceleration, and gradient. The setup computes its
coefficients from a one-metre phasor and can apply later phasor inputs without
running the Rayleigh observable again.

::: gragra.analytic.seismic.rayleigh_response

::: gragra.observables.halfspace_single_bodywave_observables

SI potential/acceleration/gradient of a single upward-propagating P or SV
wave, without a free-surface reflection composite (H19 Eq. 86-88),
multiplying the G-free `gragra.analytic.halfspace_single_bodywave_observables_g_free`
by `G_SI` exactly once.

::: gragra.observables.halfspace_reflected_p_above_observables

SI potential/acceleration/gradient above ground of an incident P + reflected
P + converted SV composite (H19 Eq. 90), multiplying the G-free
`gragra.analytic.halfspace_reflected_p_above_observables_g_free` by `G_SI`
exactly once.

::: gragra.observables.spherical_shell_p_acceleration_components

Calculates the 4-component P-wave cavity acceleration decomposition (bulk, inner wall, outer boundary, total) in physical SI units.

::: gragra.observables.spherical_shell_p_infinite_difference

Calculates the physical SI acceleration difference between an infinite medium and a finite spherical shell cavity under P-wave excitation.

::: gragra.observables.spherical_shell_s_acceleration_components

Calculates the 4-component S-wave cavity acceleration decomposition (bulk, inner wall, outer boundary, total) in physical SI units.

::: gragra.observables.spherical_shell_s_infinite_difference

Calculates the physical SI acceleration difference between an infinite medium and a finite spherical shell cavity under S-wave excitation.
