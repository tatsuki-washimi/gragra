<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Uniform-density cuboid

**Level:** — (definition and assignment pending).

**Physical domain:** static gravity  
**Geometry:** cuboid

Finite rectangular prism; static Newton field

**Source-declared observable / representation pairs:**

- Φ a Γ — Static

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry

- high-level Cuboid holds frame/lengths. Vertex antiderivative domain interior/exterior away from material boundary. No arbitrary mesh reuse implied.

## Equations

**Existing source formula notes:**

- F1 = uv ln(w+r)+vw ln(u+r)+wu ln(v+r) - 1/2 u² atan(vw/(ur)) -1/2 v² atan(wu/(vr)) -1/2 w² atan(uv/(wr)); source report gives explicit adopted sign convention. Potential G-free kg/m; acceleration kg/m²; gradient kg/m³; SI conversion once.

## Parameters and units

- center_m (3,) m; lengths_m=(Lx,Ly,Lz) m all positive; density_kg_m3 kg/m³; targets_m (M,3) m; cuboid material-boundary targets rejected. High-level Cuboid + density Quantity.

## APIs and calculation paths

### axis-aligned cuboid closed form

- `gragra.analytic.objects.uniform_cuboid`
- `gragra.observables.cuboid_observables`

**Path-level source-declared outputs:** Φ a Γ — Static

Factory APIs construct model/evaluation objects; observables may be obtained after the documented .at(...).run() sequence rather than directly from the factory call. Follow the linked API examples.

## Derivation and provenance

**Inventory status:** located.

Committed independent derivation document exists; its text is inventoried as a derivation record, not re-proven here.

The page cites public sources below. A fully typeset public derivation is a separate editorial item wherever the existing source material has not been reproduced here.

## References

Existing public documentation and original formulation context:

- [Analytic benchmarks](../guide/analytic-benchmarks.md)
- [analytic models](../guide/gen1-analytic-models.md)

**Bibliographic inventory:** located.

- Nagy, D. (1966), ‘The gravitational attraction of a right rectangular prism.’ Nagy, D., Papp, G., & Benedek, J. (2000), ‘The gravitational potential and its ...’; 2002 erratum. Petrović (1996) is named in analytic benchmark/report, but complete record not transcribed here.

## Verification evidence

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 3.
- Catalogue probe state: recorded (a representative API output contract, not an accuracy test).
- This documentation generation does not perform scientific re-execution or assign an accuracy Level.

**Existing verification inventory:**

Test implementation and documentary claims found; no checks were run for this inventory.

- (internal evidence record) includes derivative consistency, symmetry, Poisson trace, boundary rejection, independent interior/exterior quadrature, SI wrapper. Report states boundary tolerance atol=1e-12*max(half_lengths), interior trace -4πGρ, exterior trace 0. formula ledger records unrelated historical run metadata; revision-matched raw output path for assigned cuboid probe not established. cuboid_si contract is representative probe only.

## Applicability and limitations

- interior tr Γ=-4πGρ; exterior tr Γ=0. Numerical tolerance does not define boundary values.

**Scope note:** Uniform-density, rectangular prism only; points on faces/edges/vertices rejected due discontinuity/log singularity.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)

## Information still to locate or publish

- Catalog probe not executed; no numeric output recorded here.
- Petrović full bibliographic record not transcribed.
- Source derivation/tests do not establish a rerun at this task revision.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
