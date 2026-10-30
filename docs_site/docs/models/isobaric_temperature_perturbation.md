<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Isobaric temperature perturbation

**Validation level:** Unassigned (the scale and assignment criteria have not yet been defined).

**Physical domain:** atmospheric temperature  
**Geometry:** sphere, cuboid, cylinder

Fourier, Gaussian and bounded temperature anomalies; density-to-gravity coupling

**Source-declared observable / representation pairs:**

- Φ a Γ — Static
- Φ a Γ — Phasor

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry

- The API supports these profile objects; geometry dimensions are profile inputs. No dedicated temperature anomaly map asset was located.

## Equations

**Existing source formula notes:**

- δρ=−ρ0 δT/T0.
- Gaussian center: Φ(0)=−4πGρcσ², a(0)=0, Γ(0)=−4πGρc I/3; far field M=(2π)^(3/2)ρcσ³.
- Fourier acceleration and gradient use wavevector and phasor potential.

## Parameters and units

- rho0 kg/m³; T0 K; delta_temperature K; wavevector rad/m; phase_origin/center m; Gaussian sigma; Sphere radius; Cuboid lengths; Cylinder radius/height; frequency Hz and reference_time GPS for phasor.

## APIs and calculation paths

### Fourier temperature mode (static and single-frequency phasor)

- `gragra.analytic.atmospheric.isobaric_temperature`

**Path-level source-declared outputs:** Φ a Γ — Static; Φ a Γ — Phasor

### Gaussian temperature anomaly (all-space)

- `gragra.analytic.atmospheric.isobaric_temperature`

**Path-level source-declared outputs:** Φ a Γ — Static; Φ a Γ — Phasor

### Finite Sphere temperature body

- `gragra.analytic.atmospheric.isobaric_temperature`

**Path-level source-declared outputs:** Φ a Γ — Static; Φ a Γ — Phasor

### Finite Cuboid temperature body

- `gragra.analytic.atmospheric.isobaric_temperature`

**Path-level source-declared outputs:** Φ a Γ — Static; Φ a Γ — Phasor

### Finite Cylinder temperature body

- `gragra.analytic.atmospheric.isobaric_temperature`

**Path-level source-declared outputs:** Φ a Γ — Static; Φ a Γ — Phasor

Factory APIs construct model/evaluation objects; observables may be obtained after the documented .at(...).run() sequence rather than directly from the factory call. Follow the linked API examples.

## Derivation and provenance

**Inventory status:** partial.

equation prose plus test oracle/derivation scripts; local text is not proof of correctness.

The page cites public sources below. A fully typeset public derivation is a separate editorial item wherever the existing source material has not been reproduced here.

## References

Existing public documentation and original formulation context:

- [Analytic models](../guide/gen1-analytic-models.md)
- [fields API](../api/physics.md)

**Bibliographic inventory:** located.

- (internal evidence record)#temperature: approved isobaric, plane-wave, Gaussian and finite-body formulas.
- gragra.fields.thermodynamic.isobaric_temperature_density_perturbation: signed δρ=−ρ0δT/T0 linearization.

## Verification evidence

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- This documentation generation does not perform scientific re-execution or assign an accuracy Level.

**Existing verification inventory:**

High-level test code covers plane wave phasor/static and Gaussian limits; not run here.

- Gaussian center and far field equations checked; tests use assertions including exact center limits and Poisson trace. Other cases use rtol=2e-13 scale invariance. Test-code-only.

## Applicability and limitations

- Gaussian all-space support; finite geometry material boundary rejected for potential/acceleration/gradient; phasors are peak coefficients with no 1/2.

**Scope note:** linearized isobaric model; bounded geometry not sealed heating; Fourier wavevector nonzero and zero mode excluded.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)

## Information still to locate or publish

- dedicated profile visualization and exact bibliography/result records.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
