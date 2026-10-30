<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Oriented cuboid

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** static gravity  
**Geometry:** cuboid

Rotated finite prism and target-frame response

**Source-declared observable / representation pairs:**

- Φ a Γ — Static

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

- API receives center_m, lengths_m, targets_m and rotation_matrix directly; it is not a Cuboid geometry object factory. The separate plotting helper takes a direct geometry/model/setup and resolves only Cuboid, Cylinder, SphericalShell; it can transform geometry coordinates through its Frame to display_frame.
- Oriented API proper Q and inherited base cuboid boundary rejection.

### Equations

**Existing source formula notes:**

- Only coordinate transform differs: acceleration Q a_local; Gamma Q Gamma_local Qᵀ; potential unchanged.

## Parameters and calculation paths

### Parameters and units

- center_m, lengths_m, density_kg_m3, targets_m; rotation_matrix explicit proper local-to-world rotation.

### Calculation paths

#### rotated cuboid G-free/SI transport

- `gragra.observables.oriented_cuboid_observables`

**Path-level source-declared outputs:** Φ a Γ — Static

Factory APIs construct model/evaluation objects; observables may be obtained after the documented .at(...).run() sequence rather than directly from the factory call. Follow the linked API examples.

## Derivation and sources

### Derivation

Inherits the aligned-cuboid vertex-antiderivative derivation; the oriented wrapper applies a local-to-world coordinate transformation.

- cuboid derivation document applies to same closed form. oriented wrapper computes local_targets=(targets-center)@Q, evaluates cuboid_observables_g_free, then covariant vector/tensor transport; this is coordinate transformation, not a new physical derivation.

### References

Existing public documentation and original formulation context:

- [Physics API](../api/physics.md)
- [coordinate conventions](../concepts/conventions.md)

**Bibliographic inventory:** located.

- The foundational closed form is documented by Nagy (1966), Nagy, Papp & Benedek (2000) and its 2002 erratum, and Blakely (1995). The derivation document does not list DOIs.

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- This documentation generation does not perform scientific re-execution or assign an accuracy Level.

**Existing verification inventory:**

Tests and acceptance claims are recorded separately; no rerun in this task.

- (internal evidence record) covers identity, translation, rotation, SI conversion. Ledger states world-coordinate Gauss reference tolerances potential/acceleration 1e-10, gradient 1e-9, atol 1e-10; integration-int63/ot-focused.meta is recorded result path in external evidence tree but not checked/reproduced here.

### Applicability and limitations

- No general non-cuboid geometry or arbitrary affine scaling is claimed.

**Scope note:** Rotation must be proper; local cuboid boundary restrictions continue to apply.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- Catalogue has no output_contracts; no representative catalogue probe is defined.
- Sphere geometry is not among plot_shape-supported types; direct oriented_cuboid_observables arrays are not themselves a geometry/model/setup argument. Reuse requires a separate Cuboid geometry object, with its own Frame transform.
- formula_to_latex applicability is limited to its registered expressions; no claim for arbitrary oriented equations.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
