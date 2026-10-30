<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Water-channel inflow step

**Level:** — (definition and assignment pending).

**Physical domain:** water  
**Geometry:** channel

Deterministic transient Fourier amplitude, **not** a stationary ASD

**Source-declared observable / representation pairs:**

- a — Other (Fourier amplitude; see API)

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry


## Equations

**Existing source formula notes:**

- Strain Fourier amplitude = |Δa_x|/((2πf)² L_arm), L_arm default 3000 m.

## Parameters and units


## APIs and calculation paths

### Single edge-step acceleration Fourier amplitude

- `gragra.observables.water_edge_step_fourier_amplitude`

**Path-level source-declared outputs:** a — Other (Fourier amplitude; see API)

Factory APIs construct model/evaluation objects; observables may be obtained after the documented .at(...).run() sequence rather than directly from the factory call. Follow the linked API examples.

## Derivation and provenance

**Inventory status:** partial.

The formula is documented in the API docstring. No separate derivation note was identified among the assigned files.

The page cites public sources below. A fully typeset public derivation is a separate editorial item wherever the existing source material has not been reproduced here.

## References

Existing public documentation and original formulation context:

- [Water NN](../guide/water-newtonian-noise.md)
- [water API](../api/water.md)

**Bibliographic inventory:** located.


## Verification evidence

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- This documentation generation does not perform scientific re-execution or assign an accuracy Level.

**Existing verification inventory:**

test_analytic_water.py has equation comparison rtol=1e-15; test code only, not run here; no committed executed result located.


## Applicability and limitations


**Scope note:** f&gt;0, D&gt;0, ΔV&gt;0, finite x0 and positive rho; output has acceleration×time unit m/s, and MUST NOT be treated as stationary ASD.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)

## Information still to locate or publish

- The API does not provide the event-rate or amplitude statistics needed to model a stationary process.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
