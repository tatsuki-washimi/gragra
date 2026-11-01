<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
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

**Analytical formulation and field equations:**

- A1 (Surface isotropic infrasound): Acceleration CSD tensor in G-free units is \(\mathbf S_a^{(G\text{-free})}(f) = \frac{S_p(f)}{(c_s f)^2} \operatorname{diag}\left(\frac{1}{3}, \frac{1}{3}, \frac{2}{3}\right)\), where \(S_p(f)\) is surface pressure PSD in \(\mathrm{Pa^2/Hz}\), \(c_s\) is sound speed, and \(f\) is frequency.
- A2 (Underground depth \(h \ge 0\)): \(\mathbf S_a^{(G\text{-free})}(f) = \frac{S_p(f) I_{\mathrm{iso}}(x)}{(c_s f)^2} \operatorname{diag}(1, 1, 2)\), where \(x = 4\pi f h / c_s\) and \(I_{\mathrm{iso}}(x) = \frac{1}{2}\int_0^{\\pi/2} \sin^3\theta e^{-x \sin\theta} d\theta\).
- A3 (Centered in spherical cavern of radius \(R \ge 0\)): Isotropic diagonal tensor \(\mathbf S_a^{(G\text{-free})}(f) = S_{a,ii} \mathbf I_3\) with component \(S_{a,ii}^{(G\text{-free})}(f) = \frac{4}{3(c_s f)^2} S_{p,\mathrm{cav}}(f) [1 - \operatorname{sinc}(2\pi f R / c_s)]^2\), vanishing smoothly as \(R \to 0\).

## Parameters and calculation paths

### Parameters and units

- sound_speed_m_s: float &gt; 0, speed of sound in air in m/s; frequency_hz: float or array-like &gt; 0, evaluation frequency in Hz; surface_pressure_psd_pa2_hz: float or array-like &gt;= 0, one-sided pressure PSD at surface in Pa^2/Hz (for A1, A2); depth_m: float &gt;= 0, detector depth below surface in meters (for A2); radius_m: float &gt;= 0, cavern inner radius in meters (for A3); cavern_pressure_psd_pa2_hz: float or array-like &gt;= 0, pressure PSD inside cavern in Pa^2/Hz (for A3).

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

Test implementation records explicit A1/A2/A3 checks; tests were not run in this editorial review. No committed executed-result artifact for these exact test functions was identified.

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
