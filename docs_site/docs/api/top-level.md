# Top-level API

The implemented source/target containers and point, coherent, and displacement
observables are importable directly from `gragra`. This page documents the
current numerical entry points; subpackage pages provide the lower-level and
adapter interfaces.

These stable names are re-exports. `gragra.point_acceleration` and
`gragra.observables.point_acceleration` are the same function object, and
`gragra.PointMass` is `gragra.gravity_sources.PointMass`; the short path is the one
to use and the one documented here.

## Constants

gragra separates *geometry* from *coupling*: the kernels compute a purely
geometric factor and know nothing about gravity's strength, and exactly one
layer — the observables — multiplies by the gravitational constant. That
makes the constant below the single point where the physical scale enters,
and it is why the helpers in `gragra.analytic` are described as "G-free".

::: gragra.G_SI

The value is the CODATA 2018 recommendation, 6.674 30 × 10⁻¹¹ m³ kg⁻¹ s⁻².
It carries the largest relative uncertainty of any constant gragra uses — of
order 2 × 10⁻⁵ — which is far above the numerical agreement the test suite
asserts, so it, and not floating-point error, sets the accuracy floor of any
absolute result.

## Sources and targets

A calculation needs two things: a description of the mass that perturbs the
field, and the points at which you want to know the answer. All four
containers below are frozen dataclasses whose stored arrays are made
read-only on construction, so a source cannot be mutated behind the back of
a result computed from it.

The distinction between the first container and the next two is a
deliberate discipline, not redundancy. `WeightedPointSource` is the general
representation and accepts signed and complex weights; `PointMass` and
`PointMassCloud` are positive-real conveniences that convert into one.

::: gragra.WeightedPointSource

This is the generic source type: positions as an `(N, 3)` float64 table and
weights as an `(N,)` array whose kind is preserved — real input stays
`float64`, complex input stays `complex128`, and the result's dtype follows.
Weights are mass elements in kilograms, and they may be negative or complex,
which is what makes it the right container for a density perturbation
δm = δρ·dV rather than for a physical object.

::: gragra.PointMass

A single positive real mass at a single position, for when the source really
is an object — a vehicle, a piece of equipment, a test mass. A mass that is
zero, negative or non-finite raises `ValueError` at construction rather than
producing a silently signed result; call `as_weighted_source()` to hand it to
an observable.

::: gragra.PointMassCloud

The same positive-real convenience for many masses at once, with `masses_kg`
of shape `(N,)` and `positions_m` of shape `(N, 3)`. Summing the cloud gives
exactly the same field as summing the individual masses, which is the
cheapest available check that a discretization has been assembled correctly.

::: gragra.TargetPoints

The observation points as a length `Quantity` of shape `(M, 3)`, stable
string IDs, and their coordinate `Frame`. Low-level numerical observables
continue to accept bare array-like target positions directly.

## Observables

These three field quantities take a source and a set of targets, return an
array whose leading axis is the target index and whose component axes are
last, and accept the same `softening_m`, `chunk_size` and `backend` controls.
All three return read-only arrays.

::: gragra.point_potential

The gravitational potential at each target, shape `(M,)`. Note the sign
convention: the potential of a positive mass is negative, so a naive
"magnitude" comparison against another code is a comparison that cannot fail
and cannot tell you anything.

::: gragra.point_acceleration

The acceleration vector at each target, shape `(M, 3)`, pointing from the
target toward the source for a positive mass. For a single 1 kg source at
the origin observed at (1, 0, 0) m, the result is exactly `-G_SI` in x and
zero in the other two components — a fixed point worth reproducing once when
you set up.

::: gragra.point_gravity_gradient

The gravity-gradient tensor ∂aᵢ/∂xⱼ at each target, shape `(M, 3, 3)`,
differentiated with respect to the *target* coordinate. For the same 1 kg
fixed point it is `G_SI · diag(2, -1, -1)`. Its trace vanishes only when
`softening_m` is exactly zero; with softening the field is that of a smeared
mass distribution, which is not a vacuum solution.

## Additional observables

These observables are part of the current research API and retain the
subpackage implementations described in the source. They use the same
array, unit, sign, and backend conventions as the direct point path.

::: gragra.displacement_field_acceleration

Calculates a time-indexed acceleration array from a closed
`SpecfemGLLDataset`. Each GLL quadrature row is retained as a separate
background-mass contribution, and the output has shape `(Nt, M, 3)`.
The dataset retains every snapshot in memory; the observable computes one
time index at a time and does not stream the HDF5 file.

## Coherent observables

The coherent entry points take *parametric* weights: an `(N, ...)` array
carrying one weight per source element per parameter — per propagation
direction, per wavenumber, per snapshot — and return the response with those
parameter axes preserved in front of the component axes. They accept raw
position and weight arrays rather than a source container, because the weight
column is no longer a plain `(N,)` mass vector.

Complex results here are linear complex amplitudes or transfer-function-like
responses. They are not real-valued time-domain fields, and taking the real
part of one is not the same thing as computing the field at t = 0 unless you
have separately fixed the phase convention.

::: gragra.coherent_point_acceleration

The acceleration for parametric weights, with the parameter axes inserted
before the trailing component axis. It is the coherent counterpart of
`point_acceleration` and shares its kernel; the difference is entirely in the
shape of the weights.

::: gragra.coherent_point_gravity_gradient

The gravity-gradient tensor for parametric weights, ending in `(..., 3, 3)`.
The same trace caveat applies as for the non-coherent form: harmonicity holds
only at zero softening.

::: gragra.coherent_gravitoelastic_tensor

The cross tensor cᵢⱼ = ⟨δaᵢ ξⱼ*⟩ between the gravity perturbation and a
witness displacement — the quantity that says how much of a seismometer's
reading predicts the gravity signal. It is a second-order statistic: the
second argument is conjugated, the product is element-wise, and the result is
always complex even for real inputs.

::: gragra.coherent_acceleration_csd

The cross-spectral density of the acceleration between two sets of target
points, for the mirror-pair geometry a detector actually measures. It is
second order in the source weights, so it carries G², where the single-point
observables carry G¹ and a bare witness displacement carries none.

## Field transfer

The last top-level entry belongs to neither group: it takes a displacement
field on a set of mass elements rather than a set of masses or weights.

::: gragra.displacement_transfer_acceleration

The acceleration produced by displacing masses by a small vector field —
the dipole, or "total", path. Rather than differencing two configurations,
it contracts the gravity-gradient kernel against the displacement directly,
which avoids the catastrophic cancellation that differencing a
sub-micrometre motion between two large fields would produce. This is the
top-level route for taking a solver's displacement output into a gravity
signal; see [Displacement and density-perturbation fields](../guide/fields.md)
for how to build the inputs, including the case where the bulk term is
identically zero.
