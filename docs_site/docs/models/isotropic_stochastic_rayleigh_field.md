<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Isotropic stochastic Rayleigh field

**Level:** — (definition and assignment pending).

**Physical domain:** stochastic seismic  
**Geometry:** halfspace

Multiple test masses at depth; acceleration CSD

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

### 等方 Rayleigh acceleration CSD

- `gragra.observables.stochastic_rayleigh.rayleigh_isotropic_acceleration_csd`

**Path-level source-declared outputs:** a — Spectral density (CSD)

Factory APIs construct model/evaluation objects; observables may be obtained after the documented .at(...).run() sequence rather than directly from the factory call. Follow the linked API examples.

## Derivation and provenance

**Inventory status:** partial.

Analytic azimuth integration is stated; local implementation gives transfer, gamma, q/s and tensor factors. No external derivation independently checked.

The page cites public sources below. A fully typeset public derivation is a separate editorial item wherever the existing source material has not been reproduced here.

## References

Existing public documentation and original formulation context:

- [Stochastic NN](../guide/stochastic-nn.md)

**Bibliographic inventory:** located.

- レコードは (internal evidence record) に収録。文献内容をここで再検証したという意味ではない。

## Verification evidence

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 1.
- Catalogue probe state: recorded (a representative API output contract, not an accuracy test).
- This documentation generation does not perform scientific re-execution or assign an accuracy Level.

**Existing verification inventory:**

Tests include Harms Eq.112 and shape/PSD/SI checks; report conditions only as coded. Test code, no rerun. No assigned committed result artifact found.


### Registered representative output probes

| API | Observable | Representation | Probe |
|---|---|---|---|
| `gragra.observables.stochastic_rayleigh.rayleigh_isotropic_acceleration_csd` | a | Spectral density (CSD) | `rayleigh_csd_si` |

## Applicability and limitations


**Scope note:** f&gt;0; targets strictly z&gt;0; rho&gt;=0; stable isotropic halfspace requires vp²&gt;(4/3)vs²; speeds positive. Homogeneous isotropic field and one-sided input PSD assumptions.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)

## Information still to locate or publish

- High-level EvaluationResult/setup integration deferred; no PSD/CSD integration in result workflows.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
