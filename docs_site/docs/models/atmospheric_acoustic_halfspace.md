<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Atmospheric acoustic halfspace

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** atmospheric acoustic  
**Geometry:** halfspace

Plane acoustic wave and rigid-ground reflection

**Source-declared observable / representation pairs:**

- Φ a Γ — Phasor

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

- z=0 interface; target points on interface rejected.

### Equations

**Existing source formula notes:**

- air: potential_factor=exp(−qz)/2−cos(k_z z); ground: −exp(qz)/2. phase=exp(i horizontal·(x−phase_origin)).
- 0&lt;θ≤π/2; grazing valid, normal incidence excluded.

## Parameters and calculation paths

### Parameters and units

- surface_pressure Pa (total surface coefficient); sound_speed m/s; frequency Hz; incidence_angle, azimuth rad; phase_origin m; reference_time GPS; k,q,k_z m⁻¹.

### Calculation paths

#### plane-wave rigid reflection

- `gragra.analytic.atmospheric.acoustic_halfspace`

**Path-level source-declared output union:** Φ a Γ — Phasor

## Derivation and sources

### Derivation

Closed form present; no independent derivation text located in inspected guide.

- implementation formulas do not establish derivation correctness.

### References

Existing public documentation and original formulation context:

- [Analytic models](../guide/gen1-analytic-models.md)

**Bibliographic inventory:** located.

- The G-free air and ground solutions and interface conditions are specified in a non-public design note, which is not reproduced here.
- source=gragra.constants.G_SI; detail=applied exactly once by the shared analytic bridge.

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

Committed oracle fixture and tests/generator exist; not run now.

- [`tests/oracles/data/halfspace_v1.json`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/oracles/data/halfspace_v1.json) is stored reference data; test code is not a newly reproduced result. Exact per-test tolerances and lineage need follow-up.

### Applicability and limitations

- θ in (0,π/2]; grazing has k_z=0 and q≠0.

**Scope note:** Single plane wave in homogeneous sound speed halfspace; no normal incidence or interface targets.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- dedicated geometry illustration and exact reference record.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
