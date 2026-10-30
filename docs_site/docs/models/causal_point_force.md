<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Causal point force

**Level:** — (definition and assignment pending).

**Physical domain:** seismic elastic  
**Geometry:** point

Time-dependent force in an infinite homogeneous elastic medium

**Source-declared observable / representation pairs:**

- Φ a Γ — Time series

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry

- Causal C1 signed history, retarded P delay, source collision rejected; uniform supplied sampling.

## Equations

**Existing source formula notes:**

- Phi = -G mu J / r^2; a = G[(J/r^3)e + (f/(vp^2 r)-3J/r^3)mu n]; Gamma formula C,D coefficients; all three outputs.

## Parameters and units

- position m, unit direction, vp m/s, ForceHistory knots seconds/coefficients; GPS onset, sampling epoch, dt seconds and count.

## APIs and calculation paths

### implemented model/API path

- `gragra.analytic.seismic.point_force`

**Path-level source-declared outputs:** Φ a Γ — Time series

Factory APIs construct model/evaluation objects; observables may be obtained after the documented .at(...).run() sequence rather than directly from the factory call. Follow the linked API examples.

## Derivation and provenance

**Inventory status:** partial.

Formula expressions are declared in source; a separate local derivation proof was not located in the focused scan.

The page cites public sources below. A fully typeset public derivation is a separate editorial item wherever the existing source material has not been reproduced here.

## References

Existing public documentation and original formulation context:

- [Analytic models](../guide/gen1-analytic-models.md)

**Bibliographic inventory:** partial.

- Reference table records local formula-ledger Green function and causal delay moment.

## Verification evidence

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- This documentation generation does not perform scientific re-execution or assign an accuracy Level.

**Existing verification inventory:**

Verification method, tolerances, and artifact status recorded; tests were not run in this task.

- CAS script (internal evidence record) documents r&gt;0,c&gt;0,C1 history assumptions and J derivatives; test code only. Hand-sample (internal evidence record) uses rtol 1e-14; no saved output found.

## Applicability and limitations

- Causal C1 signed history, retarded P delay, source collision rejected; uniform supplied sampling.

**Scope note:** Applicability limits recorded.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)

## Information still to locate or publish

- Catalogue output_contracts are empty; this is a missing representative probe, not evidence of absent science.
- No additional committed plot asset identified in these assigned API sources.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
