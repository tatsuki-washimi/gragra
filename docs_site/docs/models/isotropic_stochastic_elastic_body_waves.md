<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
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

- Full-space isotropic P-waves (B1): \(\mathbf S_a^{(G\text{-free})}(f) = (4\pi \rho)^2 S_{\xi,P}(f) \mathbf I_3\), where \(S_{\xi,P}(f)\) is the one-sided per-component P-displacement PSD (Harms 2019 Eqs. 55, 60).
- Spherical cavity P/S-waves (B2): Evaluated at center of spherical cavity of radius \(R &gt; 0\) with wavenumbers \(k_P = 2\pi f / v_P\) and \(k_S = 2\pi f / v_S\). With spherical Bessel factors \(f_P = \frac{j_1(k_P R)}{k_P R}\) and \(f_S = \frac{j_1(k_S R)}{k_S R}\), the isotropic CSD is \(\mathbf S_a^{(G\text{-free})}(f) = (4\pi \rho)^2 [4 f_P^2 S_{\xi,P}(f) + f_S^2 S_{\xi,S}(f)] \mathbf I_3\) (Harms 2019 Eq. 65).
- Physical SI CSD: Multiplies G-free CSD by \(G^2\): \(\mathbf S_a^{(\mathrm{SI})}(f) = G^2 \mathbf S_a^{(G\text{-free})}(f)\) in \(\mathrm{(m/s^2)^2/Hz}\).

## Parameters and calculation paths

### Parameters and units

- Full-space B1 (fullspace_p_isotropic_acceleration_csd_g_free): density_kg_m3: float rho &gt;= 0 in kg/m^3; frequency_hz: float or array-like &gt; 0 in Hz; p_displacement_psd_m2_hz: float or array-like &gt;= 0, one-sided per-component P-displacement PSD in m^2/Hz.
- Spherical cavity B2 (spherical_cavity_ps_isotropic_acceleration_csd_g_free): radius_m: float R &gt; 0 in meters; density_kg_m3: float rho &gt;= 0 in kg/m^3; v_p_m_s: float v_P &gt; 0 in m/s; v_s_m_s: float v_S &gt; 0 in m/s (satisfying 0 &lt; v_S &lt; v_P); frequency_hz: float or array-like &gt; 0 in Hz; p_displacement_psd_m2_hz: float or array-like &gt;= 0 in m^2/Hz; s_displacement_psd_m2_hz: float or array-like &gt;= 0 in m^2/Hz.

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

Test implementation records concrete checks; tests were not run in this editorial review. No committed executed-result artifact was identified for the assigned stochastic bodywave tests.

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
