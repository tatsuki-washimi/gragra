<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Isotropic stochastic elastic body waves

**Level:** — (definition and assignment pending).

**Physical domain:** stochastic seismic  
**Geometry:** fullspace, cavity

Infinite-medium P field and idealized P/S cavity-center CSD

**Source-declared observable / representation pairs:**

- a — Spectral density (CSD)

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry


## Equations

**Existing source formula notes:**

- A complete public LaTeX formulation has not yet been assembled.

## Parameters and units


## APIs and calculation paths

### Fullspace isotropic P

- `gragra.observables.stochastic_bodywave.fullspace_p_isotropic_acceleration_csd`

**Path-level source-declared outputs:** a — Spectral density (CSD)

### Centered spherical cavity isotropic P/S

- `gragra.observables.stochastic_bodywave.spherical_cavity_ps_isotropic_acceleration_csd`

**Path-level source-declared outputs:** a — Spectral density (CSD)

Factory APIs construct model/evaluation objects; observables may be obtained after the documented .at(...).run() sequence rather than directly from the factory call. Follow the linked API examples.

## Derivation and provenance

**Inventory status:** partial.

Full-space uses local bulk gravity transfer. Cavity implementation uses stable j1(x)/x series for |x|&lt;1e-3; local computational transcription, not independent proof.

The page cites public sources below. A fully typeset public derivation is a separate editorial item wherever the existing source material has not been reproduced here.

## References

Existing public documentation and original formulation context:

- [Stochastic NN](../guide/stochastic-nn.md)

**Bibliographic inventory:** located.

- local formula specification は implementation mapping であり、それ自体は文献レコードではない。

## Verification evidence

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- This documentation generation does not perform scientific re-execution or assign an accuracy Level.

**Existing verification inventory:**

Test implementation records concrete checks; no tests were run in this inventory. No committed executed-result artifact was identified for the assigned stochastic bodywave tests.

- (internal evidence record): fullspace formula uses rho=2500 kg/m³, f=2 Hz, S=3e-18 m²/Hz and exact allclose rtol=0, atol=0.
- Cavity direct angular/vector-sum comparison checks real P and S CSD separately at rtol=2e-11, atol=0; imaginary residual atol=1e-13*max(expected). Code is a test oracle route, not saved run output.

## Applicability and limitations


**Scope note:** f&gt;0, one-sided per-component PSD nonnegative; B2 radius, density, vp, vs conditions; no scattering. No general displacement/potential/gradient outputs.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)

## Information still to locate or publish

- Boundary scattering, non-spherical cavity and potential/gradient are absent from these APIs.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
