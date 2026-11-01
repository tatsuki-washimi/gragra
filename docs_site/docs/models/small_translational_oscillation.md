<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Small translational oscillation

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** rigid motion  
**Geometry:** point

Linearized point-mass or cloud phasor

**Source-declared observable / representation pairs:**

- Φ a Γ — Phasor

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

- Small first-order translation approximation, phasor not exact finite motion.

### Equations

**Existing source formula notes:**

- For point masses \(m_i\) at equilibrium positions \(\mathbf x_{0,i}\) with offset \(\mathbf r_i = \mathbf x - \mathbf x_{0,i}\) (\(r_i = \|\mathbf r_i\|\)), subject to harmonic displacement \(\mathbf u_i(t) = \operatorname{Re}[\mathbf u_i e^{-2\pi i f (t - t_{\mathrm{ref}})}]\) (\(\|\mathbf u_i\| \ll r_i\)).
- Potential phasor: \(\delta\Phi = -G \sum_i \frac{m_i (\mathbf r_i \cdot \mathbf u_i)}{r_i^3}\); acceleration phasor: \(\delta\mathbf a = G \sum_i m_i \left[ \frac{\mathbf u_i}{r_i^3} - \frac{3 (\mathbf r_i \cdot \mathbf u_i) \mathbf r_i}{r_i^5} \right]\).
- Gravity-gradient phasor: \(\delta\mathbf\Gamma = G \sum_i m_i \left[ -\frac{3 (\mathbf u_i \otimes \mathbf r_i + \mathbf r_i \otimes \mathbf u_i + (\mathbf r_i \cdot \mathbf u_i)\mathbf I)}{r_i^5} + \frac{15 (\mathbf r_i \cdot \mathbf u_i) \mathbf r_i \otimes \mathbf r_i}{r_i^7} \right]\). Uses peak phasor convention (no factor of 1/2).

## Parameters and calculation paths

### Parameters and units

- mass: array-like (N,), positive masses in kg; position: array-like (N, 3), equilibrium positions in meters.
- displacement: array-like (3,) or (N, 3), complex peak displacement vector in meters; frequency: float or Quantity, oscillation frequency in Hz.
- reference_time: GPS epoch for phasor phase convention; targets: array-like (M, 3), evaluation points in meters.

### Calculation paths

#### implemented model/API path

- `gragra.analytic.objects.translational_oscillation`

**Observables after model evaluation:** Φ a Γ — Phasor

High-level model factories create evaluation objects; call the documented `.at(...).run()` workflow to evaluate them. Observable functions listed above are direct calls with their own return values.

## Derivation and sources

### Derivation

Formula expressions are declared in source; a separate local derivation proof was not located in the focused scan.

- No separate derivation is asserted from the formula declaration alone.

### References

Existing public documentation and original formulation context:

- [Analytic models](../guide/gen1-analytic-models.md)

**Bibliographic inventory:** partial.

- Shared motion reference metadata; formulas recorded in source.

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

Verification method, tolerances, and artifact status recorded; tests were not run in this task.

- Rigid-motion oracle tests exist, but oscillation-specific saved result not located; no current execution.

### Applicability and limitations

- Small first-order translation approximation, phasor not exact finite motion.

**Scope note:** Applicability limits recorded.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- Catalogue output_contracts are empty; this is a missing representative probe, not evidence of absent science.
- No additional committed plot asset identified in these assigned API sources.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
