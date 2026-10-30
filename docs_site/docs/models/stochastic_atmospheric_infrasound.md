<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Stochastic atmospheric infrasound

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** stochastic atmospheric  
**Geometry:** halfspace, cavity

Surface, underground and centered-cavern idealizations; acceleration CSD

**Source-declared observable / representation pairs:**

- a — Spectral density (CSD)

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

A1 rigid surface; A2 depth halfspace; A3 centered cavern idealization. Basic SphericalShell plotting is static only and does not encode modal acoustic fields.

### Equations

**Existing source formula notes:**

A1 G-free diagonal [1/3,1/3,2/3] S_p/(c_s f)². A2 x=4πfh/c_s, I_iso(x)=1/2∫ sin³θ exp(-x sinθ)dθ; diagonal [1,1,2] S_p I_iso/(c_s f)². A3 component=(2/(c_s f))²S_p(1-sinc(kr))²/3, k r=2πfr/c_s. SI CSD ×G².

## Parameters and calculation paths

### Parameters and units

A1 sound_speed_m_s, frequency_hz, surface_pressure_psd_pa2_hz. A2 adds depth_m. A3 radius_m, sound speed, cavern_pressure_psd_pa2_hz. Pressure PSD Pa²/Hz; f&gt;0; geometry nonnegative; c_s&gt;0.

### Calculation paths

#### A1 isotropic surface acoustic

- `gragra.observables.stochastic_acoustic.surface_isotropic_acoustic_acceleration_csd`

**Path-level source-declared output union:** a — Spectral density (CSD)

#### A2 isotropic underground acoustic

- `gragra.observables.stochastic_acoustic.underground_isotropic_acoustic_acceleration_csd`

**Path-level source-declared output union:** a — Spectral density (CSD)

#### A3 centered cavern acoustic

- `gragra.observables.stochastic_acoustic.centered_cavern_isotropic_acoustic_acceleration_csd`

**Path-level source-declared output union:** a — Spectral density (CSD)

## Derivation and sources

### Derivation

A2 implementation records integral definition and three numerical domains; A3 specification distinguishes published hemispherical-cavern Eq.13 from separately developed response. Text inventory only, no algebraic certification.

### References

Existing public documentation and original formulation context:

- [Stochastic NN](../guide/stochastic-nn.md)
- [limitations](../verification/limitations.md)

**Bibliographic inventory:** located.

- The local model specification distinguishes the provenance of Eq. 13 from the separate A3 response. It is not an independent bibliographic record.
- The A3 centered-cavern API is a related API identified in the source. The ledger group lists only the surface and underground APIs; A3 is not treated as an original catalogue symbol.

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

Test implementation records explicit A1/A2/A3 checks; no tests were run in this inventory. No committed executed-result artifact for these exact test functions was identified.

- [`tests/test_stochastic_acoustic.py`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/test_stochastic_acoustic.py) A1: c_s=343 m/s, pressure PSD=4e-4 Pa²/Hz, f=2 Hz; closed tensor equality rtol=0, atol=0; doubled-frequency 1/f² scaling rtol=2e-15.
- A2 depth factor independent angular quadrature uses x={0,0.1,0.7,1,5,10,20,50}, scipy quad epsabs=epsrel=2e-14, limit=200; comparison rtol=2e-4, atol=2e-14. Test comment says implementation-conditioning, not final science acceptance.

### Applicability and limitations


**Scope note:** frequency strictly positive; c_s positive; depth/radius nonnegative in kernels; PSD nonnegative. A3 center and idealized geometry; acoustic formulations noted as under refinement in guide.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- Modal/resolved cavern, arbitrary cavity geometry, acoustic frequency-dependent boundary conditions not represented.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
