
# Sources, targets and geometry

This page covers the input side of a calculation and the two layers beneath
the observables. It is organised the way a calculation flows: you describe
mass (`gragra.gravity_sources`), you say where you want the answer
(`gragra.targets`), you turn a shape into mass elements
(`gragra.geometry`), and — if you are building something new — you reach the
integrator and kernel layers that the observables are made of.

The layering is strict and worth stating once: an observable calls an
integrator, the integrator dispatches to a kernel, and the kernel returns a
purely geometric factor. Only the observable multiplies by
[`G_SI`](top-level.md). Anything you build on top of `gragra.kernels` is
G-free and stays that way until you multiply it yourself.

## Sources

`WeightedPointSource`, `PointMass`, and `PointMassCloud` are re-exported at
the top level and documented on the [Top-level API](top-level.md) page. The
remaining subpackage source types are documented below. The volume and surface
types carry a discretization and density; the prescribed-motion types carry
externally supplied time samples.

::: gragra.gravity_sources.VolumeElementSource

A set of volume elements — centroid positions, per-element volumes and a
density — which `as_weighted_source()` collapses into mass elements
δm = ρ·dV. This is what a discretized solid, a voxelised density field or a
tetrahedral mesh becomes on its way into an observable, and keeping the
volume separate from the density until that point is what lets you change one
without re-deriving the other.

For a uniform sphere, `discretize` accepts pure geometry, a density Quantity
and a spacing Quantity, then returns the fixed point source used by evaluation.
The same source model also accepts caller-provided real static point sources
and volume elements. Density grids and volume meshes can be converted to
`VolumeElementSource` values and evaluated with an explicit Frame and field
definition; the [static-source guide](../guide/static-sources.md) documents
those input routes and their evidence.

::: gragra.gravity_sources.discretize

`point_sum` binds a static source to the numerical point-sum model. A
`DiscreteSource` from `discretize` carries its Sphere Frame and field meaning;
bare point and volume sources require an explicit Frame, and general weighted
or volume sources also require a `GravityFieldDefinition`. Calling `.at()`
fixes the targets and observables before validation or calculation begins.

::: gragra.gravity_sources.point_sum

`rigid_motion` builds a rigid-motion model of a point-mass cloud. A
`PrescribedRigidTransform` with `sampling=` gives a time-series evaluation. A
callable pose with `epoch=` gives continuous Fourier-series and
Fourier-transform evaluations. The [rigid-motion guide](../guide/rigid-motion-fourier.md)
covers units, phase conventions, applicability and the memory budget.

::: gragra.gravity_sources.rigid_motion

::: gragra.gravity_sources.SurfaceMassSheetSource

A set of triangle facets — centroids, unit normals, areas, a normal
displacement and a density jump — representing the mass redistributed when an
interface moves. The sign follows the normal direction, so a mesh with
inconsistent winding produces a sheet whose contributions partially cancel;
[Volume and surface sources](../guide/continuous-sources.md) explains how to
check that before it silently reduces your answer.

::: gragra.gravity_sources.PrescribedPointTrajectory

A positive point mass at prescribed, strictly increasing time samples. The
stored positions are read-only float64 coordinates, ready for the trajectory
observables without interpolation or equation-of-motion solving.

::: gragra.gravity_sources.PrescribedRigidTransform

A prescribed sequence of translations and proper rotation matrices for a
rigid point-mass cloud. It validates each `R Rᵀ = I` and `det(R) = 1` frame
without correcting supplied rotations.

::: gragra.gravity_sources.read_prescribed_point_trajectory_csv

Read a prescribed single-particle trajectory from a CSV file into a PrescribedPointTrajectory.

::: gragra.gravity_sources.read_prescribed_rigid_transform_csv

Read a prescribed rigid-body translation and rotation history from a CSV file into a PrescribedRigidTransform.

The trajectory observables evaluate only the supplied frames. They support
the NumPy backend and have a direct NumPy/Numba comparison; interpolation and
motion generation are outside this interface.

### Causal force histories

`ForceHistory` describes a signed real force after a GPS onset by polynomial
pieces in elapsed time. It is intended for the causal analytic
`analytic.seismic.point_force` model; the class itself does not calculate a
field. Knot values are elapsed seconds, each interval uses a normalized local
coordinate, adjacent pieces must join in force and first derivative, and the
final continuation is either zero or a held force.

::: gragra.gravity_sources.ForceHistory

`smooth_pulse` builds a C1 force pulse that returns to zero at the final knot.

::: gragra.gravity_sources.smooth_pulse

`smooth_ramp_to_hold` builds a C1 ramp whose final force continues unchanged.

::: gragra.gravity_sources.smooth_ramp_to_hold

### Prescribed motion verification { #status-prescribed-motion }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Experimental |
    | **Verified against** | closed-form analytic solution; invariance property; backend equivalence |
    | **Agreement** | A point flyby matches its closed form, and rigid rotations preserve the documented covariance. |
    | **Test evidence** | `tests/test_trajectory.py::test_trajectory_point_flyby_closed_form_and_readonly`, `tests/test_trajectory.py::test_rigid_rotation_covariance_for_all_observables`, `tests/test_trajectory.py::test_trajectory_superposition_and_chunk_size_invariance` |
    | **Conditions** | Prescribed finite time samples, positive point masses, and proper rotation matrices |
    | **Not verified** | Interpolation or generated trajectories |
    | **Known limits** | The tested non-NumPy backend is Numba; cpp and cupy are not verified for this API |
    | **Scope** | Potential, acceleration, and gravity-gradient samples for point trajectories and rigid point-mass clouds |

## Targets

There is a single high-level target container, `TargetPoints`, and it is
re-exported lazily at the top level, so its reference entry lives on the
[Top-level API](top-level.md) page. It stores length quantities, unique IDs
and a coordinate frame. Low-level numerical observables still accept bare
coordinate arrays without this context.

The target object requires an explicit `(M, 3)` Quantity and one unique ID
per row. Low-level functions such as `point_acceleration` continue to promote
a bare `(3,)` array to one row and return shape `(1, 3)`.

## Geometry and numerical primitives

`Frame` defines a static Cartesian coordinate system. The public `Sphere`,
`Cuboid`, `Cylinder`, and `SphericalShell` are pure, Quantity-backed shapes:
their centers are expressed in their Frame, and they carry no density or
discretization settings. Existing numerical `Cuboid` and `Cylinder` classes
remain at `gragra.geometry.primitives` for volume-grid workflows. The shared
short names identify different classes in those two namespaces; no compatibility
wrapper is provided.

`HalfSpace` defines the flat surface geometry used by high-level Rayleigh,
P/SV, and rigid-reflection acoustic models. Its frame's local `z=0` plane is
the interface. The seismic medium lies at local `z<0`; the acoustic model
uses local `z>0` for air and `z<0` for ground. Density and wave properties
remain model inputs.

::: gragra.geometry.HalfSpace

`Frame` stores a Cartesian origin and orientation, letting geometry and
targets use named local coordinates within one connected coordinate system.

::: gragra.geometry.Frame

::: gragra.geometry.Sphere

A solid sphere given by centre, radius and frame. It contains no density or
grid settings; pass it to `uniform_sphere` or `discretize` to choose those
independently. The midpoint point-sum model is the standard convergence test.

::: gragra.geometry.Cuboid

A pure box given by centre, three positive edge lengths, and Frame. Its edges
follow the Frame's local x, y, and z axes; orient the Frame to place the box.
This high-level shape does not carry a discretization grid.

::: gragra.geometry.Cylinder

A pure solid cylinder given by centre, positive radius and height, and Frame.
Its axis follows the Frame's local z direction. The older numerical cylinder
retains its axis-label interface under `gragra.geometry.primitives`.

::: gragra.geometry.SphericalShell

A pure spherical shell given by centre, positive inner radius, greater outer
radius, and Frame. It describes geometry only; pass it with density and
phasor inputs to `spherical_shell` for the finite-shell wave evaluation.

::: gragra.geometry.Appearance

Optional color, opacity, and display label metadata for a high-level shape.
It does not affect geometry checks or physics. For each field, an explicit
drawing-call value wins, followed by shape metadata and the plotter default.

::: gragra.geometry.HalfSphere

A hemispherical dome: the half of a sphere on one side of the plane through
its centre. It exists because cavern end-caps are modelled this way, and it
shares the sphere's spacing and mass-guard behaviour.

::: gragra.geometry.HalfCylinder

A half-cylinder with a flat floor — the "kamaboko" cross-section used for
tunnel and cavern models. It is the primitive most often combined with
`HalfSphere` end-caps to approximate an underground detector hall.

## High-level analytic factories

These factories bind pure geometry and physical inputs to the shared
`EvaluationResult` contract. Static cuboid and cylinder models use their
analytic G-free native functions and apply `G_SI` once. The cylinder factory
does not fall back to the numerical quadrature diagnostic. Seismic factories
require explicit phasor reference time; a `FrequencySeries` epoch is retained
as provenance and does not rephase the displacement. The approved Gen1
additions live in their `analytic.objects`, `analytic.seismic`, and
`analytic.atmospheric` namespaces; they are not re-exported at the package
root. Their assumptions and runnable examples are collected in the
[Gen1 analytic models guide](../guide/gen1-analytic-models.md).

::: gragra.analytic.objects.uniform_cuboid

Use this factory for a uniformly dense cuboid evaluated through its analytic
G-free field calculation.

::: gragra.analytic.objects.uniform_cylinder

Use this factory for a uniformly dense cylinder; evaluation stays on the
analytic native path and does not invoke the quadrature diagnostic.

::: gragra.analytic.objects.constant_velocity

Moves one positive point mass at constant velocity, sampled on an explicit
`TimeSampling` axis. The source position is defined at `motion_epoch`.

::: gragra.analytic.objects.translational_oscillation

Evaluates a first-order uniform-translation peak phasor for a point mass or
cloud, with one positive frequency and explicit `reference_time`.

::: gragra.analytic.objects.fixed_axis_rotation

Rotates a positive point-mass cloud exactly at constant angular velocity
about a fixed axis and origin. The input cloud is the zero-angle geometry.

::: gragra.analytic.seismic.small_cavity_wall

Evaluates only the selected first-order SCA cavern-wall contribution for a
Sphere, Cuboid, or Cylinder and one displacement input.

::: gragra.analytic.seismic.point_force

Evaluates a causal point force in an infinite homogeneous elastic medium from
a piecewise C1 force history and a positive P-wave speed.

::: gragra.analytic.atmospheric.acoustic_halfspace

Evaluates the rigid-reflection acoustic halfspace for a total surface-pressure
phasor, oblique incidence, and targets away from the interface.

::: gragra.analytic.atmospheric.isobaric_temperature

Evaluates a prescribed isobaric temperature perturbation for a plane wave,
all-space Gaussian, or bounded uniform Sphere, Cuboid, or Cylinder anomaly.

::: gragra.analytic.atmospheric.acoustic_cavity_mode

Builds a single-frequency rigid-cavity pressure-mode gravity response for a
Sphere, Cuboid, or Cylinder and the matching geometry-specific mode type.

::: gragra.analytic.atmospheric.SphereAcousticMode

Real spherical Neumann mode indices `(l, m, n)`; `n` selects a strictly
positive radial derivative root.

::: gragra.analytic.atmospheric.CuboidAcousticMode

Nonnegative cosine-mode indices `(nx, ny, nz)` for a cuboid Neumann mode; the
all-zero constant mode is excluded.

::: gragra.analytic.atmospheric.CylinderAcousticMode

Real cylindrical Neumann mode indices, including a cosine or sine azimuthal
choice when the order is nonzero.

::: gragra.analytic.atmospheric.AcousticCavityNumerics

Positive resource ceilings for evaluation count and refinement count. They do
not set a physical error tolerance or replace the fixed convergence policy.

::: gragra.analytic.seismic.spherical_cavity

Use this factory for the selected P or S displacement contribution of a
spherical cavity, with the documented potential reference attached per part.

::: gragra.analytic.seismic.spherical_shell

Use this factory for P or S displacement contributions from the inner and
outer boundaries of a spherical shell.

::: gragra.analytic.seismic.halfspace_psv

Use this factory for supported P/SV plane-wave displacement families in a
half-space, including only the documented limited zero-horizontal-wavenumber
cases.

## Surface meshes

The surface of a primitive is a separate object from its volume filling,
because the two are used for different terms: the volume gives the bulk
density perturbation, the surface gives the interface term. These two types
describe the surface side.

::: gragra.geometry.MeshResolution

How finely `surface_mesh()` subdivides a primitive, as counts along the two
surface parameters plus a cap count. It is a small explicit container rather
than three loose integers so that the same resolution can be handed to
several primitives and be recognisably the same choice.

::: gragra.geometry.PrimitiveSurfaceMesh

The result: vertices, triangles, and the per-facet centroids, unit normals
and areas that a surface source needs. `concatenate()` joins the meshes of
several primitives into one, which is how a compound cavern — a hall with
domed ends — is assembled while keeping every facet's winding as generated.

## Integrators

The integrator layer normalises inputs, picks the kernel that matches the
requested observable, and applies the chunking strategy. It is public
because it is the right place to stand if you are adding an observable, and
because it is G-free: everything it returns still needs the gravitational
constant applied.

::: gragra.integrators.direct_point_sum

The direct O(N·M) sum over source elements for a scalar-weighted source,
dispatching by observable name to the potential, acceleration or gradient
kernel. It never materialises the full `(M, N, ...)` array — the sum is
accumulated in chunks along the source axis — which is what makes a
million-element source a memory question rather than an impossibility.

::: gragra.integrators.direct_point_dipole_sum

The same direct sum for a *vector*-weighted source: mass elements plus a
displacement each, contracted against the gravity-gradient kernel. This is
the machinery under `displacement_transfer_acceleration`, and it is separate
from the scalar path because the weight is a vector, not because the physics
differs.

## Kernels

A kernel returns the geometric factor K only, summed against the weights as
Σⱼ wⱼ·K. Kernels do not import the gravitational constant and do not multiply
by it; that boundary exists so that an alternative force law — Yukawa,
modified gravity — can be introduced by substituting a kernel rather than by
auditing every observable for a stray factor.

All seven take source positions, weights, target positions, a softening
length and a chunk size. The four non-parametric forms take an `(N,)` weight
column; the three `_parametric` forms take `(N, ...)` and carry the extra
axes through to the result.

::: gragra.kernels.potential_contract

The potential kernel, K_Φ = −1/r_eff, contracted against scalar weights.
The minus sign is part of the kernel and not applied later, which is why a
positive mass gives a negative potential all the way through the stack.

::: gragra.kernels.acceleration_contract

The acceleration kernel, K_a = Δ/r_eff³ with Δ = r_source − r_target,
contracted against scalar weights. The direction convention is fixed by that
subtraction order, and it is the single most useful thing to check first when
a result comes out with the wrong sign.

::: gragra.kernels.gradient_contract

The gravity-gradient kernel, K_T = 3ΔΔ/r_eff⁵ − I/r_eff³, differentiated with
respect to the target coordinate. Its trace is identically zero only in the
unsoftened case; with `softening_m > 0` the kernel describes a smeared mass
and its trace is a real, non-zero number rather than round-off.

::: gragra.kernels.dipole_contract

The vector-weight contraction δaᵢ = −Σₙ mₙ K_T,ᵢⱼ dₙʲ, which turns a
displacement field into an acceleration perturbation in one step. It uses the
same K_T as `gradient_contract`; what differs is that the tensor is
contracted against a per-element displacement instead of being returned.

::: gragra.kernels.potential_contract_parametric

The potential contraction with parametric weights, for weights that carry
extra axes — direction, wavenumber, snapshot — beyond the source index. The
parameter axes are broadcast through the sum and appear in the result before
the (absent, for a scalar) component axis.

::: gragra.kernels.acceleration_contract_parametric

The acceleration contraction with parametric weights, returning
`(M, ..., 3)`. This is the kernel behind `coherent_point_acceleration`, and
the reason coherent responses cost no more per parameter value than a plain
sum does per source.

::: gragra.kernels.gradient_contract_parametric

The gravity-gradient contraction with parametric weights, returning
`(M, ..., 3, 3)`. The same softening caveat as the non-parametric form
applies to every slice along the parameter axes.
