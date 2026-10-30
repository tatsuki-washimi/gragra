<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Isotropic stochastic elastic body waves

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** stochastic seismic  
**Geometry:** fullspace, cavity

Infinite-medium P field and idealized P/S cavity-center CSD

**Source-declared observable / representation pairs:**

- a — Spectral density (CSD)

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

B2 centered spherical cavity only, scattering neglected; arbitrary cavity geometry unsupported by API. basic plotting helper supports SphericalShell static shape but no source coupled setup stated.

### Equations

**Existing source formula notes:**

Fullspace: S_a=(4πrho)² S_xi I (G-free), SI ×G². Cavity P/S model identified as Harms 2019 Eq.65; full expression includes spherical Bessel response.

## Parameters and calculation paths

### Parameters and units

B1: density_kg_m3; frequency_hz; p_displacement_psd_m2_hz; B2 additionally radius_m, v_p_m_s, v_s_m_s, s_displacement_psd_m2_hz; PSD per Cartesian component m²/Hz; rho≥0, speeds/radius positive and vs&lt;vp.

### Calculation paths

#### Fullspace isotropic P

- `gragra.observables.stochastic_bodywave.fullspace_p_isotropic_acceleration_csd`

**Path-level source-declared output union:** a — Spectral density (CSD)

#### Centered spherical cavity isotropic P/S

- `gragra.observables.stochastic_bodywave.spherical_cavity_ps_isotropic_acceleration_csd`

**Path-level source-declared output union:** a — Spectral density (CSD)

## Derivation and sources

### Derivation

Full-space uses local bulk gravity transfer. Cavity implementation uses stable j1(x)/x series for |x|&lt;1e-3; local computational transcription, not independent proof.

### References

Existing public documentation and original formulation context:

- [Stochastic NN](../guide/stochastic-nn.md)

**Bibliographic inventory:** located.

- The local formula specification maps formulas to the implementation; it is not itself a bibliographic record.

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

Test implementation records concrete checks; no tests were run in this inventory. No committed executed-result artifact was identified for the assigned stochastic bodywave tests.

- [`tests/test_stochastic_bodywave.py`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/test_stochastic_bodywave.py): fullspace formula uses rho=2500 kg/m³, f=2 Hz, S=3e-18 m²/Hz and exact allclose rtol=0, atol=0.
- Cavity direct angular/vector-sum comparison checks real P and S CSD separately at rtol=2e-11, atol=0; imaginary residual atol=1e-13*max(expected). Code is a test oracle route, not saved run output.

### Applicability and limitations


**Scope note:** f&gt;0, one-sided per-component PSD nonnegative; B2 radius, density, vp, vs conditions; no scattering. No general displacement/potential/gradient outputs.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- Boundary scattering, non-spherical cavity and potential/gradient are absent from these APIs.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
