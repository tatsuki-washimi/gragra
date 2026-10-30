<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Rigid-body time series and Fourier response

**Level:** — (definition and assignment pending).

**Physical domain:** rigid motion  
**Geometry:** point_cloud

Prescribed positions/rotations; periodic or finite-window Fourier evaluation

**Source-declared observable / representation pairs:**

- Φ a Γ — Time series
- Φ a Γ — Fourier series
- Φ a Γ — Fourier transform

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry

- plot_shape is not a Fourier trajectory renderer.

## Equations

**Existing source formula notes:**

- x=R x_body+c; exp[−2πif(t−tref)]。周期: A0=C0, An=2Cn。有限窓係数は倍化しない。

## Parameters and units

- source PointMassCloud; positions m, times s, acceleration m/s²; observable collector default memory_budget_bytes=2 GiB; f_max*|epoch−tref|&lt;=1e6 cycles。

## APIs and calculation paths

### sampled rigid-body time series

- `gragra.gravity_sources.rigid_motion`
- `gragra.observables.rigid_motion_observable`

**Path-level source-declared outputs:** Φ a Γ — Time series

### callable pose periodic Fourier series

- `gragra.gravity_sources.rigid_motion`
- `gragra.observables.rigid_motion_observable`

**Path-level source-declared outputs:** Φ a Γ — Fourier series

### callable pose finite-window Fourier transform

- `gragra.gravity_sources.rigid_motion`
- `gragra.observables.rigid_motion_observable`

**Path-level source-declared outputs:** Φ a Γ — Fourier transform

Factory APIs construct model/evaluation objects; observables may be obtained after the documented .at(...).run() sequence rather than directly from the factory call. Follow the linked API examples.

## Derivation and provenance

**Inventory status:** partial.

Newton oracle is code; Fourier numerical integration prose is not a local derivation.

The page cites public sources below. A fully typeset public derivation is a separate editorial item wherever the existing source material has not been reproduced here.

## References

Existing public documentation and original formulation context:

- [Rigid-motion Fourier guide](../guide/rigid-motion-fourier.md)
- [results API](../api/results.md)

**Bibliographic inventory:** located.

- RigidMotionModelSpec.references returns an empty DataFrame. Continuous Fourier model and its series/transform specs likewise return empty DataFrames.

## Verification evidence

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- This documentation generation does not perform scientific re-execution or assign an accuracy Level.

**Existing verification inventory:**

Tests cover observables/Fourier/oracle comparisons; not run here.

- test-code-only; rigid_motion_science oracle comparisons rtol=3e-14; no committed executed-result path located.

## Applicability and limitations

- return type provisional; G applied once at observable layer; memory budget per-lane.

**Scope note:** sampled/continuous argument combinations exclusive; finite window has aliasing/phase caveats.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)

## Information still to locate or publish

- saved scientific runs not located.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
