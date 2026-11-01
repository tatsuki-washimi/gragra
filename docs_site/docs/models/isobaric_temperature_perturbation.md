<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Isobaric temperature perturbation

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** atmospheric temperature  
**Geometry:** sphere, cuboid, cylinder

Fourier, Gaussian and bounded temperature anomalies; density-to-gravity coupling

**Source-declared observable / representation pairs:**

- ① Φ a Γ — Static
- ② Φ a Γ — Phasor

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

![Isobaric temperature perturbation schematic](../assets/models/isobaric_temperature_perturbation.svg)

*Thermal parcel with temperature fluctuation δT inducing isobaric density fluctuation δρ/ρ_0 = - δT/T_0.*

- The API supports these profile objects; geometry dimensions are profile inputs. No dedicated temperature anomaly map asset was located.

### Equations

**Analytical formulation and field equations:**

- Under isobaric thermodynamic conditions, small temperature variations \(\delta T\) couple linearly to density as \(\delta\rho = -\rho_0 \frac{\delta T}{T_0}\), where \(\rho_0 &gt; 0\) is ambient background density and \(T_0 &gt; 0\) is background temperature in Kelvin.
- Fourier mode: \(\delta T = \Delta T \cos(\mathbf k \cdot (\mathbf x - \mathbf x_0))\) (static) or \(\widetilde{\delta T} = \widetilde{\Delta T} e^{i \mathbf k \cdot (\mathbf x - \mathbf x_0)}\) (phasor). The G-free potential is \(\Phi^{(G\text{-free})} = -\frac{4\pi \rho_c}{k^2} e^{i \mathbf k \cdot (\mathbf x - \mathbf x_0)}\) with zero-mean gauge (\(k \ne 0\)), and \(\mathbf a^{(G\text{-free})} = -i \mathbf k \Phi^{(G\text{-free})}\).
- Gaussian profile: \(\delta T(\mathbf x) = \Delta T e^{-\|\mathbf x - \mathbf c\|^2 / (2\sigma^2)}\). At center \(\mathbf x = \mathbf c\), \(\Phi^{(G\text{-free})}(0) = -4\pi \rho_c \sigma^2\), \(\mathbf a(0) = \mathbf 0\), and \(\mathbf\Gamma^{(G\text{-free})}(0) = -\frac{4\pi \rho_c}{3} \mathbf I_3\) where \(\rho_c = -\rho_0 \Delta T / T_0\). Far from center, fields approach a point monopole of mass \(M = (2\pi)^{3/2} \rho_c \sigma^3\).
- Targets on material boundaries of finite geometric bodies (Sphere, Cuboid, Cylinder) are rejected for potential, acceleration, and gradient.

## Parameters and calculation paths

### Parameters and units

| Symbol | Parameter | Meaning | SI unit |
|---|---|---|---|
| \(\rho_0\) | `ambient_density_kg_m3` | Ambient unperturbed air density | kg/m³ |
| \(T_0\) | `ambient_temperature_k` | Ambient absolute temperature | K |
| \(\delta T(\mathbf x, t)\) | `temperature_perturbation_k` | Temperature fluctuation field (delta rho = -rho0 delta T / T0) | K |
| \(\mathbf x\) | `targets_m` | Sensor evaluation coordinates | m |

**Additional notes on parameters and models:**

- rho0_kg_m3: float &gt; 0, ambient atmospheric density; T0_k: float &gt; 0, ambient temperature in Kelvin; delta_temperature: float, complex, or profile-specific amplitude in Kelvin; targets_m: array-like (M, 3), evaluation points in meters.
- Profile parameters: TemperatureFourierMode(wavevector_rad_m, phase_origin_m, frequency_hz); TemperatureGaussian(center_m, sigma_m); TemperaturePerturbationGrid(edges_m, temperatures_k).

### Calculation paths

#### Fourier temperature mode (static and single-frequency phasor)

- `gragra.analytic.atmospheric.isobaric_temperature`

**Path-level source-declared output union:** Φ a Γ — Static; Φ a Γ — Phasor

#### Gaussian temperature anomaly (all-space)

- `gragra.analytic.atmospheric.isobaric_temperature`

**Path-level source-declared output union:** Φ a Γ — Static; Φ a Γ — Phasor

#### Finite Sphere temperature body

- `gragra.analytic.atmospheric.isobaric_temperature`

**Path-level source-declared output union:** Φ a Γ — Static; Φ a Γ — Phasor

#### Finite Cuboid temperature body

- `gragra.analytic.atmospheric.isobaric_temperature`

**Path-level source-declared output union:** Φ a Γ — Static; Φ a Γ — Phasor

#### Finite Cylinder temperature body

- `gragra.analytic.atmospheric.isobaric_temperature`

**Path-level source-declared output union:** Φ a Γ — Static; Φ a Γ — Phasor

## Derivation and sources

### Derivation

equation prose plus test oracle/derivation scripts; local text is not proof of correctness.

- test suites exercise center, far-field, trace and scale invariance; no saved executed artifact identified.

### References

Existing public documentation and original formulation context:

- [Analytic models](../guide/gen1-analytic-models.md)
- [fields API](../api/physics.md)

**Bibliographic inventory:** located.

- an unpublished design note: approved isobaric, plane-wave, Gaussian and finite-body formulas.
- gragra.fields.thermodynamic.isobaric_temperature_density_perturbation: signed δρ=−ρ0δT/T0 linearization.
- gragra.constants.G_SI: applied exactly once by shared analytic bridge.

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

High-level test code covers plane wave phasor/static and Gaussian limits; not run here.

- Gaussian center and far field equations checked; tests use assertions including exact center limits and Poisson trace. Other cases use rtol=2e-13 scale invariance. Test-code-only.

### Applicability and limitations

- Gaussian all-space support; finite geometry material boundary rejected for potential/acceleration/gradient; phasors are peak coefficients with no 1/2.

**Scope note:** linearized isobaric model; bounded geometry not sealed heating; Fourier wavevector nonzero and zero mode excluded.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- dedicated profile visualization and exact bibliography/result records.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
