<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Stochastic atmospheric infrasound

**Validation level:** Unassigned (the scale and assignment criteria have not yet been defined).

**Physical domain:** stochastic atmospheric  
**Geometry:** halfspace, cavity

Surface, underground and centered-cavern idealizations; acceleration CSD

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

### A1 isotropic surface acoustic

- `gragra.observables.stochastic_acoustic.surface_isotropic_acoustic_acceleration_csd`

**Path-level source-declared outputs:** a — Spectral density (CSD)

### A2 isotropic underground acoustic

- `gragra.observables.stochastic_acoustic.underground_isotropic_acoustic_acceleration_csd`

**Path-level source-declared outputs:** a — Spectral density (CSD)

### A3 centered cavern acoustic

- `gragra.observables.stochastic_acoustic.centered_cavern_isotropic_acoustic_acceleration_csd`

**Path-level source-declared outputs:** a — Spectral density (CSD)

Factory APIs construct model/evaluation objects; observables may be obtained after the documented .at(...).run() sequence rather than directly from the factory call. Follow the linked API examples.

## Derivation and provenance

**Inventory status:** partial.

A2 implementation records integral definition and three numerical domains; A3 specification distinguishes published hemispherical-cavern Eq.13 from separately developed response. Text inventory only, no algebraic certification.

The page cites public sources below. A fully typeset public derivation is a separate editorial item wherever the existing source material has not been reproduced here.

## References

Existing public documentation and original formulation context:

- [Stochastic NN](../guide/stochastic-nn.md)
- [limitations](../verification/limitations.md)

**Bibliographic inventory:** located.

- The local model specification distinguishes the provenance of Eq. 13 from the separate A3 response. It is not an independent bibliographic record.
- The A3 centered-cavern API is a related API identified in the source. The ledger group lists only the surface and underground APIs; A3 is not treated as an original catalogue symbol.

## Verification evidence

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- This documentation generation does not perform scientific re-execution or assign an accuracy Level.

**Existing verification inventory:**

Test implementation records explicit A1/A2/A3 checks; no tests were run in this inventory. No committed executed-result artifact for these exact test functions was identified.

- (internal evidence record) A1: c_s=343 m/s, pressure PSD=4e-4 Pa²/Hz, f=2 Hz; closed tensor equality rtol=0, atol=0; doubled-frequency 1/f² scaling rtol=2e-15.
- A2 depth factor independent angular quadrature uses x={0,0.1,0.7,1,5,10,20,50}, scipy quad epsabs=epsrel=2e-14, limit=200; comparison rtol=2e-4, atol=2e-14. Test comment says implementation-conditioning, not final science acceptance.

## Applicability and limitations


**Scope note:** frequency strictly positive; c_s positive; depth/radius nonnegative in kernels; PSD nonnegative. A3 center and idealized geometry; acoustic formulations noted as under refinement in guide.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)

## Information still to locate or publish

- Modal/resolved cavern, arbitrary cavity geometry, acoustic frequency-dependent boundary conditions not represented.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
