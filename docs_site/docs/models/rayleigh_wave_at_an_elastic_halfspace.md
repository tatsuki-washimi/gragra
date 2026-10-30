<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Rayleigh wave at an elastic halfspace

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** seismic elastic  
**Geometry:** halfspace

Surface-wave gravity, depth response and coherent phasors

**Source-declared observable / representation pairs:**

- Φ a Γ — Phasor

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

- High-level HalfSpace phasor APIs and separate all-space literature wrapper; coefficient response differs from amplitude-bearing field.

### Equations

**Existing source formula notes:**

- Phi Eq.96/97 above/below ground; Eq.98 amplitude; a=-grad(Phi), Gamma=grad(a).

## Parameters and calculation paths

### Parameters and units

- density kg/m³, vp/vs m/s, f Hz, vertical displacement m, direction, phase origin m, reference_time; response API has no amplitude and uses one-metre basis.

### Calculation paths

#### Explicit displacement-bearing Rayleigh field

- `gragra.analytic.seismic.rayleigh_halfspace`

**Path-level source-declared outputs:** Φ a Γ — Phasor

#### Amplitude-free Rayleigh response coefficients

- `gragra.analytic.seismic.rayleigh_response`

**Path-level source-declared outputs:** Φ a Γ — Phasor

#### Literature all-space Rayleigh observables

- `gragra.observables.rayleigh_halfspace_allspace_observables`

**Path-level source-declared outputs:** Φ a Γ — Phasor

Factory APIs construct model/evaluation objects; observables may be obtained after the documented .at(...).run() sequence rather than directly from the factory call. Follow the linked API examples.

## Derivation and sources

### Derivation

Committed design and validation records describe the local independent derivation/verification approach; they remain documentary evidence and were not rerun here.

- (internal evidence record) maps H19 Eq.96/97 and labels acceleration/gradient derivative class B; records Eq.92 hand-algebra cross-check.
- (internal evidence record) describes independent Poisson/Green-function numerical re-derivation and surface checks.

### References

Existing public documentation and original formulation context:

- [Waves](../guide/waves.md)
- [analytic benchmarks](../guide/analytic-benchmarks.md)
- [Rayleigh workflow](../examples/rayleigh_phasor_workflow.md)

**Bibliographic inventory:** partial.

- Harms, Living Reviews Relativity 22, 6 (2019), DOI 10.1007/s41114-019-0022-2, Eq.96-98.
- [DOI 10.1007/s41114-019-0022-2](https://doi.org/10.1007/s41114-019-0022-2)

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- This documentation generation does not perform scientific re-execution or assign an accuracy Level.

**Existing verification inventory:**

Verification method, tolerances, and artifact status recorded; tests were not run in this task.

- (internal evidence record) uses independent finite integral/extrapolation, 12 point vP/vS,kH grid; saved result (internal evidence record); source reports worst extrapolation error ~9.3e-5 absolute/~0.72% relative. Test code and artifact, not run now.

### Applicability and limitations

- Literature API uses z-up and free surface z=0; z==0 targets rejected. Underground formula z&lt;0; ground formula z&gt;0. Potential continuous at interface, normal acceleration jumps, gradient generally discontinuous.
- Model APIs require HalfSpace geometry and explicit frame/phase origin; response coefficients omit amplitude/reference time.
- Cavity-wall corrections are outside the cited literature API scope.

**Scope note:** Applicability limits recorded.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- Catalogue output_contracts are empty; this is a missing representative probe, not evidence of absent science.
- No additional committed plot asset identified in these assigned API sources.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
