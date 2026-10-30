<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Atmospheric acoustic halfspace

**Level:** — (definition and assignment pending).

**Physical domain:** atmospheric acoustic  
**Geometry:** halfspace

Plane acoustic wave and rigid-ground reflection

**Source-declared observable / representation pairs:**

- Φ a Γ — Phasor

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry

- z=0 interface; target points on interface rejected.

## Equations

**Existing source formula notes:**

- air: potential_factor=exp(−qz)/2−cos(k_z z); ground: −exp(qz)/2. phase=exp(i horizontal·(x−phase_origin)).
- 0&lt;θ≤π/2; grazing valid, normal incidence excluded.

## Parameters and units

- surface_pressure Pa (total surface coefficient); sound_speed m/s; frequency Hz; incidence_angle, azimuth rad; phase_origin m; reference_time GPS; k,q,k_z m⁻¹.

## APIs and calculation paths

### plane-wave rigid reflection

- `gragra.analytic.atmospheric.acoustic_halfspace`

**Path-level source-declared outputs:** Φ a Γ — Phasor

Factory APIs construct model/evaluation objects; observables may be obtained after the documented .at(...).run() sequence rather than directly from the factory call. Follow the linked API examples.

## Derivation and provenance

**Inventory status:** partial.

Closed form present; no independent derivation text located in inspected guide.

The page cites public sources below. A fully typeset public derivation is a separate editorial item wherever the existing source material has not been reproduced here.

## References

Existing public documentation and original formulation context:

- [Analytic models](../guide/gen1-analytic-models.md)

**Bibliographic inventory:** located.

- source=(internal evidence record)#acoustic-halfspace; detail=approved G-free air and ground solutions and interface conditions.
- source=gragra.constants.G_SI; detail=applied exactly once by the shared analytic bridge.

## Verification evidence

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- This documentation generation does not perform scientific re-execution or assign an accuracy Level.

**Existing verification inventory:**

Committed oracle fixture and (internal evidence record) exist; not run now.

- (internal evidence record) is stored reference data; test code is not a newly reproduced result. Exact per-test tolerances and lineage need follow-up.

## Applicability and limitations

- θ in (0,π/2]; grazing has k_z=0 and q≠0.

**Scope note:** Single plane wave in homogeneous sound speed halfspace; no normal incidence or interface targets.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)

## Information still to locate or publish

- dedicated geometry illustration and exact reference record.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
