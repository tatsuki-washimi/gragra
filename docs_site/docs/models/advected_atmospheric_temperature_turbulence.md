<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Advected atmospheric temperature turbulence

**Validation level:** Unassigned (the scale and assignment criteria have not yet been defined).

**Physical domain:** stochastic atmospheric  
**Geometry:** atmosphere

Frozen-flow hypothesis; acceleration PSD

**Source-declared observable / representation pairs:**

- a — Spectral density (PSD)

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry


## Equations

**Existing source formula notes:**

- A complete public LaTeX formulation has not yet been assembled.

## Parameters and units


## APIs and calculation paths

### T1 uniform advection projected acceleration PSD

- `gragra.observables.stochastic_temperature.uniform_advection_temperature_acceleration_psd`

**Path-level source-declared outputs:** a — Spectral density (PSD)

Factory APIs construct model/evaluation objects; observables may be obtained after the documented .at(...).run() sequence rather than directly from the factory call. Follow the linked API examples.

## Derivation and provenance

**Inventory status:** partial.

Local module implements scaled Bessel geometry and log-tail branch for float64 range; equations are local specification/transcription, not independently proven here.

The page cites public sources below. A fully typeset public derivation is a separate editorial item wherever the existing source material has not been reproduced here.

## References

Existing public documentation and original formulation context:

- [Stochastic NN](../guide/stochastic-nn.md)

**Bibliographic inventory:** partial.

- Actual repository literature record located for Harms: (internal evidence record)
- For Creighton, the targeted source search located local metadata in the T1 specification but no separate title/author/year/arXiv bibliography record; source identity beyond that recorded metadata is not independently established.

## Verification evidence

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- This documentation generation does not perform scientific re-execution or assign an accuracy Level.

**Existing verification inventory:**

Local unit/oracle test code exists; no tests were run in this inventory. T1 formal family record is still HOLD, so test code and candidate protocol records do not establish formal science acceptance.

- (internal evidence record) BASE inputs rho=1.2 kg/m³, T=293 K, c_T=0.1 K m^(-p/2), p=2/3, wind=10 m/s, r_min=5 m, theta=0; geometry factor comparison at x={0.1,0.5,1,3,10} uses scipy quad epsabs=1e-13, epsrel=2e-12, limit=300, comparison rtol=3e-11, atol=1e-14. Large-x x=50 comparison uses rtol 0.02 parallel and 0.04 transverse.
- Separate independent oracle test code uses mpmath 35/45 decimal precision at p={0.4, 2/3, 1.3}, x={0.07,1.7,13}; checks integral identity residual &lt;1e-28 and positive-tail bound. This is test code, not execution evidence.

## Applicability and limitations


**Scope note:** Strict 0&lt;p&lt;2, positive T, wind, minimum approach, positive frequencies; density and c_T nonnegative. Taylor frozen flow, uniform straight advection, homogeneous isotropic structure-function assumptions.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)

## Information still to locate or publish

- No arbitrary wind field, finite-outer-scale cutoff, or spatially varying temperature-statistics API is present in this function.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
