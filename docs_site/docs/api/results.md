# Results and spectral representations

`GravityResult` is an Experimental envelope for grouping existing observable
payloads under the `gragra.results` namespace. It does not alter the return
types of numerical APIs, and it is not re-exported from the `gragra` root
package.

`EvaluationResult` groups native payloads, their shared `EvaluationContext`,
an evaluation object, and an execution record. The context
owns the normalized targets, resolved frame, requested observables, field
definition, interaction, and exactly one of a static axis, phasor axes, or a
time-sampling axis. It remains available only from `gragra.results`.

`EvaluationContext` provides model-independent interpretation shared by all
observable bindings in one result. `result.context` returns the same object as
`result.evaluation.context`.

::: gragra.results.EvaluationContext

`FieldEvaluation` is the structural contract for the object returned by a
model's `.at()` method. It exposes one context, model parameters, formulas,
references, validation, and a `run()` method.

::: gragra.results.FieldEvaluation

`ModelSpec` supplies the field definition and interaction that identify a
model's physical interpretation. The context reads both from this spec.

::: gragra.results.ModelSpec

`StaticRepresentation` marks ordinary real `Quantity` payloads and keeps
static interpretation distinct from spectral descriptors.

::: gragra.results.StaticRepresentation

`EvaluationRecord` stores execution facts such as backend, dependency versions,
and the single application of the gravitational constant.

::: gragra.results.EvaluationRecord

`TimeSampling` is the immutable canonical axis for a uniform time series. It
stores an integer GPS seconds/nanoseconds epoch, a positive `dt` in seconds,
and a positive sample count; construction uses keyword-only arguments such as
`TimeSampling(epoch=..., dt=1e-6 * u.s, count=1)`. At large GPS epochs,
floating-point display coordinates may not distinguish adjacent samples.
Validation and comparison use the canonical epoch, `dt`, and count rather than
subtracting rounded absolute time values.

::: gragra.results.TimeSampling

`TimeSeriesRepresentation` marks real time-sampled payloads and requires one
`TimeSampling` in the shared context. A context cannot combine time sampling
with frequencies or a phasor reference time.

::: gragra.results.TimeSeriesRepresentation

`NumericalDiagnostic` is immutable per-target, per-observable numerical
evidence in SI units. A quadrature/numerical error estimate and a rigorous
integral-tail bound remain separate; runtime cavity convergence checks their
sum against its fixed tolerance policy. The estimate is not a rigorous bound,
and a successful diagnostic does not certify values edited after evaluation.

::: gragra.results.NumericalDiagnostic

`ConvergenceError` reports a numerical evaluation failure and carries the
immutable diagnostics for at least one failed target/observable. Inspect its
`diagnostics` and the original exception cause to identify the failing term;
changing a tolerance does not establish accuracy.

`EvaluationResult` validates high-level payloads against this shared context
and exposes the model evaluation and execution record that produced them.

::: gragra.results.EvaluationResult

Direct construction of `EvaluationResult` no longer accepts one separately
created binding per observable. Pass the values, a `FieldEvaluation` (the
object returned by `.at()`), and one execution record; the result reads its
shared context from that evaluation and derives bindings from it.

```python
# Previous construction
EvaluationResult(
    values=values,
    bindings=bindings,
    evaluation=evaluation,
    record=record,
)

# Current construction
result = EvaluationResult(
    values,
    evaluation=evaluation,
    record=record,
)
context = result.context
binding = context.binding("potential")
```

For `.at().run()` results, `result.evaluation` is the `FieldEvaluation`
returned by `.at()`. For `GravityResponse.apply()` results, it is a private
adapter for the applied input and context; its `run()` raises `RuntimeError`.
Call `response.apply()` to reuse the stored coefficients. Read shared metadata
such as targets, frame, representation, frequencies, and interaction from
`result.context`; read model parameters, formulas, and references through
`result.evaluation` or the corresponding `result` properties. `result.record`
contains execution facts.

High-level results retain the original potential, acceleration, and gradient
payload objects. Access revalidates shape, observable labels, unit
compatibility, finite values, and exact canonical frequency or time axes.
Finite payload edits remain visible when those declared contracts still
hold; result validation does not recompute fields or certify that edited
values satisfy the source physics. The result's context owns target and source
interpretation, while each vector or tensor binding records the component
Frame chosen with `output_frame`.

`compare` aligns compatible results by target ID and SI position, then returns
candidate-minus-reference differences in SI units and a summary table. Static
summaries have one row per target; phasor summaries add one row per target and
frequency. TimeSeries summaries have one row per target and sample and include
`sample_index` and `time_offset`; their `epoch_gps` is table metadata. Time
axes must match exactly by epoch, interval, and count. Comparison never
resamples, interpolates, rotates components, or rephases. Compatibility uses
the result context and observable descriptors, so independent models can be
compared through the same result contract.

::: gragra.results.compare

`GravityResponse` owns precomputed single-input transfer coefficients and the
contract that explains their input, output, and gravity law. Its `apply()`
method accepts a phasor input with an explicit GPS reference time and returns
the usual `EvaluationResult` payload; `coefficients()` returns an independent
frequency matrix with field-unit-per-input-unit metadata.

::: gragra.results.GravityResponse

`ResponseContract` records the physical input, basis amplitude, mapping,
coefficient storage, varied source, fixed target position, and gravity law.
The Rayleigh response reports its linearization applicability as
`not_assessed`. Gravity response contracts are documented in the package API
reference and concept pages for this release.

::: gragra.results.ResponseContract

`GravityResult` accepts potential, acceleration, and gradient payloads by
reference. At least one observable must be present; `None` marks an observable
that was not supplied. Validation checks `name` first: it must be a string or
`None`, and the empty string is accepted. Another name type raises `TypeError`;
if the name is valid but all observables are `None`, construction raises
`ValueError`, even when a name is given. The wrapper does not inspect payload
types, units, shapes, axes, compatibility, or physical validity, and it does
not copy or coerce values.

`name` is an optional human-readable label. It is not a physical identity and
is not propagated into a payload's name or channel metadata. The caller is
responsible for grouping payloads that belong to the same evaluation; multiple
results can be kept in `dict[str, GravityResult]`. `GravityResult` has no
`parts` or `term` field, and does not implement arithmetic because target and
frame alignment and the connection of spectral descriptors to results remain
unspecified; this avoids
implicitly combining potentially incompatible data. It also has no time or
FFT operations, I/O, or display. Coupling and other model metadata remain
separate.

The dataclass freezes its field bindings and uses identity equality and
hashing. Reassigning or deleting a field raises `FrozenInstanceError`.
Payloads can remain mutable, and changes through either reference are visible
through the other. This does not provide a fully immutable scientific
snapshot.

For high-level time and frequency representations, GWexpy Matrix payloads use
rows for targets and columns for components; the matrix type and labels carry
axis information. TimeSeries payloads have shape `(Ntarget, Ncomponent, Ntime)`
and use `TimeSampling` as their canonical expected axis. The example also uses
sample-last internal storage as provided by these Matrix types. A
frequency-indexed example makes no claim
about any particular spectral meaning; the descriptors below state those
meanings independently of the payload. This layout rule does not apply to CSD
matrices, whose row and column semantics remain unspecified.

The example wraps an existing uniform-sphere SI result and native GWexpy
matrices. Its U11 case calls a lower-level SI implementation to demonstrate
wrapping. That import path is not in `gragra.observables.__all__` and is not a
new recommended public entry point. The complete cases are in
`examples/gravity_result.py`.

`GravityResult` and the spectral names below are public from this subpackage.

::: gragra.results.GravityResult

## Gravity field definition

`GravityFieldDefinition` is an immutable typed contract that connects physical
meaning to an existing gravity result. During one evaluation, the model spec,
binding, and result share the same definition object. The field-definition
contract for this release is described in the results API reference and concept
pages.

`FieldKind` distinguishes a source field from a field change. A source field
has no change definition, while a field change requires one.

::: gragra.results.FieldKind

`ReferenceStatus` states whether the reference is defined, unknown, or
explicitly zero. Defined and explicit-zero references require both a temporal
kind and a nonempty state kind.

::: gragra.results.ReferenceStatus

`ReferenceTemporalKind` distinguishes a reference state from a reference
history.

::: gragra.results.ReferenceTemporalKind

`ChangeForm` separates a finite state difference from a first-order
linearized response. The latter is not a numerical finite-difference method.

::: gragra.results.ChangeForm

`PotentialReference` identifies the convention used to define the potential
reference level.

::: gragra.results.PotentialReference

`ReferenceDefinition` records reference status and the temporal and state
categories needed to compare field meanings. A defined reference does not
store the concrete reference state.

::: gragra.results.ReferenceDefinition

`ChangeDefinition` records the changed input, reference, mathematical form,
order, and fixed conditions for a field change.

::: gragra.results.ChangeDefinition

`Contribution` declares either all contributions or a selected, nonempty set
of contribution components. Components are normalized to a sorted tuple.

::: gragra.results.Contribution

`GravityFieldDefinition` combines field kind, source kind, contribution
coverage, potential convention, and an optional change definition.

::: gragra.results.GravityFieldDefinition

## Spectral representation descriptors

The descriptors are Experimental APIs and immutable tags for spectral meaning.
They do not store payload values or estimator settings and do not calculate
spectra. This interface provides spectral distinctions independently of
`GravityResult`. The entries below summarize each representation and the
conditions attached to it.

There are nine public spectral names: two enums, six descriptor classes, and
the `SpectralRepresentation` union.

### Frequency sidedness

Use `Sidedness` to distinguish storage of positive frequencies alone from
storage of both frequency signs.
Because `Sidedness` is a `StrEnum`, `Sidedness.ONE_SIDED == "one_sided"` is
true. Descriptor validation still requires the enum member: passing the bare
string as `sidedness="one_sided"` raises `TypeError`.

::: gragra.results.Sidedness

### Finite-record interpretation

`FiniteRecordInterpretation` states whether an amplitude spectrum interprets
the finite record as periodic or as a zero-extended transient.

::: gragra.results.FiniteRecordInterpretation

### Fourier transform

`FourierTransformRepresentation` describes the transform sign convention,
physical unit, and whether negative-frequency coefficients are stored.

::: gragra.results.FourierTransformRepresentation

### Amplitude spectrum

`AmplitudeSpectrumRepresentation` identifies the one-sided coefficient
normalization and its finite-record interpretation.

::: gragra.results.AmplitudeSpectrumRepresentation

### Phasor

`PhasorRepresentation` describes a complex peak amplitude under the stated
negative-exponential time convention.

::: gragra.results.PhasorRepresentation

### Power spectral density

`PSDRepresentation` denotes a one-sided power density for a real signal, with
endpoint and conversion conditions stated in its docstring.

::: gragra.results.PSDRepresentation

### Amplitude spectral density

`ASDRepresentation` denotes the square root of a one-sided power density for
a real signal.

::: gragra.results.ASDRepresentation

### Cross spectral density

`CSDRepresentation` states the cross-spectrum conjugation direction and the
conditions for interpreting coefficients as a one-sided density.

::: gragra.results.CSDRepresentation

### Descriptor union

`SpectralRepresentation` is the type alias union of the six representation
descriptor classes above. It is not a runtime wrapper or an additional
descriptor kind.

::: gragra.results.SpectralRepresentation
