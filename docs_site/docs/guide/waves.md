
# Waves

## Kinematic wave fields

`gragra.waves` generates closed-form *kinematic* displacement fields —
plane waves, Rayleigh eigenfunctions, stochastic P/S realizations — that
you can feed into the source/geometry layer (see
[What gragra computes](../concepts/model-and-scope.md)). Generating one of
these is not the same as solving the wave equation for an arbitrary medium
or boundary condition; each field is a specific, named closed form.

## Plane waves

### Phase and complex factor

`plane_wave_phase` and `plane_wave_complex_factor` compute
`k·r − ωt`-style phase and the corresponding `exp(i·phase)` complex factor
for a plane wave with a given wavenumber and propagation direction. All
angular quantities are radians (see
[Coordinates, signs and units](../concepts/conventions.md)).

### Displacement

`plane_wave_displacement` returns the vector displacement field of a
P, SV, or SH plane wave at given points, as a linear complex amplitude
(disclaimer A, see [Complex outputs](../concepts/complex-outputs.md)).

### Bulk mass weights

`plane_wave_mass_weights` (`gragra.coherent`) and `plane_wave_bulk_mass_weights`
convert a plane-wave displacement field into the mass-element weights that
feed a coherent-path source. `plane_wave_bulk_mass_weights` implements
`δm = −iρ₀k(k̂·ê)ξV·e^{ik·r}` — a linear complex amplitude (disclaimer A)
despite the resemblance of its name to the second-order coherent functions
on [Coherent responses and cross statistics](coherent.md); see
[Complex outputs](../concepts/complex-outputs.md#telling-the-two-apart).

### A pure S wave produces no bulk density perturbation

**A pure S wave produces no bulk density perturbation.**
`plane_wave_bulk_mass_weights` implements
`δm = −iρ₀k(k̂·ê)ξV e^{ik·r}`, which is identically zero when `k̂·ê = 0` —
that is, for any pure S wave in an unbounded homogeneous medium. An
all-zero result here is physically correct, not a bug. Shear motion
perturbs gravity only through boundaries and interfaces — see
[The surface term](fields.md#the-surface-term).

## Evanescent fields

`evanescent_complex_factor` gives the exponentially decaying complex
amplitude factor for an evanescent (non-propagating) plane-wave component,
used when the wave's horizontal wavenumber exceeds `ω/c` for the medium in
question.

## Rayleigh waves

`rayleigh_secular` and `rayleigh_speed` solve the Rayleigh secular equation
for a homogeneous elastic half-space, giving the Rayleigh wave speed as a
function of P- and S-wave speed. `rayleigh_displacement` gives the
resulting depth-dependent particle-motion field — this is a genuine
solution of a boundary-value problem (traction-free surface), not merely a
kinematic ansatz, and is checked against a residual of the boundary
condition itself; see [How gragra is verified](../verification/index.md)
for why a quadrature error here (dropping the 90° phase relationship
between the horizontal and vertical particle-motion components) would
break the retrograde-elliptical motion while an `|u|`-magnitude-only test
would not catch it.

## Stochastic realizations

`stochastic_ps_realization` generates a single realization of a stochastic
P/S wavefield with a specified power spectrum — useful for Monte Carlo
studies of Newtonian-noise contributions from an ensemble of
directions/phases rather than a single deterministic plane wave. Each call
returns one realization; averaging over many calls is the caller's
responsibility (see [Directional averages](coherent.md#directional-averages)
for the analogous ensemble-model caveat on the coherent path).

## Status and verification

None of the functions on this page owns a top-level 8-field Verification
Block; their verification claims live in their own docstrings, rendered on
the [API reference](../api/physics.md). The Rayleigh secular-equation residual
and traction-free boundary-condition residual are checked at `< 1e-12` and
`< 1e-10` respectively (normalized, per
[Chunking, softening and tolerances](../concepts/numerics.md)) — see
`tests/test_wave_rayleigh.py` and `tests/test_wave_halfspace.py`.
