
# Complex outputs

Complex dtype alone does not identify a value's meaning. Deterministic
amplitudes and Fourier-domain coefficients follow different conventions from
second-order statistics and spectral densities. The
[`gragra.results` reference](../api/results.md) documents descriptors for
Fourier transforms, amplitude spectra, phasors, PSD, ASD and CSD.

## Two different meanings of a complex array

A complex-valued gragra result is not a single concept. Which of the two
meanings below applies depends on the *order*, in input amplitude, of the
quantity being returned — not on which subpackage the function lives in.
`plane_wave_mass_weights` (order A) lives in `coherent/`; `coherent_point_acceleration`
(order A) and `coherent_gravitoelastic_tensor` (order B) both live in
`observables/`. Check the disclaimer text in the function's own docstring,
not its import path.

## A. Linear complex amplitudes

### Which functions return one

`point_potential`, `point_acceleration`, `point_gravity_gradient`,
`displacement_transfer_acceleration`, `coherent_point_acceleration`,
`coherent_point_gravity_gradient`, `plane_wave_phase`,
`plane_wave_complex_factor`, `plane_wave_displacement`,
`plane_wave_mass_weights`, `evanescent_complex_factor`,
`rayleigh_displacement`, `stochastic_ps_realization` — any observable given
a complex-weighted source becomes complex the moment the source's
`weights_kg` is complex (see [Point sources](../guide/point-sources.md)).

### How to interpret it

> Complex outputs represent linear complex amplitudes or transfer-function-like
> responses, not directly real-valued time-domain gravitational fields.

To recover a physical, real-valued signal you must attach your own time
factor and take the real part — gragra never does this for you and cannot
know which sign convention (`exp(-iωt)` vs `exp(+iωt)`) your upstream data
used.

```python
--8<-- "snippets/quickstart_complex_amplitude.py"
```

## B. Second-order cross statistics

### Which functions return one

`directional_cross_average`, `directional_outer_cross_average`,
`ensemble_outer_cross_average`, `coherent_gravitoelastic_tensor`,
`coherent_acceleration_csd` — these consume amplitudes and produce a
**second-order** (cross-spectral) quantity, not an amplitude itself.
`coherent_acceleration_csd` is a weighted raw second-order sum and does not
automatically have CSD density semantics. A CSD descriptor denotes a one-sided
density under its stated units and conversion conditions.

### How to interpret it

> Complex outputs are second-order cross statistics (cross-spectral quantities
> such as ⟨A·B*⟩), not time-domain fields; the phase encodes the relative
> propagation delay or coherence between the two inputs, not an absolute
> field phase.

See [Coherent responses and cross statistics](../guide/coherent.md) for the
conjugation convention (element-wise, second argument conjugated), the
G-order distinction (`coherent_gravitoelastic_tensor` is G¹,
`coherent_acceleration_csd` is G²), and why mixed-sign weights make a
negative `cross(A, A, w)` a legitimate, non-buggy result.

## Telling the two apart

If a function's docstring carries the order-A disclaimer text verbatim, its
output is a linear amplitude — attach a time factor and take the real part
to get something physical. If it carries the order-B text, its output is
already a cross-spectral quantity — the phase itself is the answer (a delay
or coherence), and there is no time factor to attach. `plane_wave_bulk_mass_weights`
is order A (it returns a source weight, one input amplitude in) even though
its name resembles the order-B functions; do not infer order from the name
alone.
