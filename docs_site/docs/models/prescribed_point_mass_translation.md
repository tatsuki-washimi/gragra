<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Prescribed point-mass translation

**Level:** — (definition and assignment pending).

**Physical domain:** rigid motion  
**Geometry:** point

Constant-velocity mass; time-sampled field

**Source-declared observable / representation pairs:**

- Φ a Γ — Time series

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry

- Exact prescribed trajectory, not a dynamical motion solver; distinct from linearized phasor model.

## Equations

**Existing source formula notes:**

- Phi=-G sum(m/r); a=-G sum(m r/r³); Gamma=G sum(m(3rr/r⁵-I/r³)); time-series outputs.

## Parameters and units

- mass kg, position m, velocity m/s, frame, motion_epoch GPS, TimeSampling.

## APIs and calculation paths

### implemented model/API path

- `gragra.analytic.objects.constant_velocity`

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

- Shared motion implementation reference metadata; local formulas in source.

## Verification evidence

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- This documentation generation does not perform scientific re-execution or assign an accuracy Level.

**Existing verification inventory:**

Verification method, tolerances, and artifact status recorded; tests were not run in this task.

- (internal evidence record) uses independent mpmath transform oracle/frozen fixtures; key comparison 3e-14 tolerances; test code only, no saved result.

## Applicability and limitations

- Exact prescribed trajectory, not a dynamical motion solver; distinct from linearized phasor model.

**Scope note:** Applicability limits recorded.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)

## Information still to locate or publish

- Catalogue output_contracts are empty; this is a missing representative probe, not evidence of absent science.
- No additional committed plot asset identified in these assigned API sources.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
