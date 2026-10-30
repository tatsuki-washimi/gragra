<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Advected atmospheric temperature turbulence

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** stochastic atmospheric  
**Geometry:** atmosphere

Frozen-flow hypothesis; acceleration PSD

**Source-declared observable / representation pairs:**

- a — Spectral density (PSD)

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

Line-of-sight/minimum-approach geometry and wind angle are scalar model parameters; no scene/mesh/cavern geometry. Plot shape reuse not material.

### Equations

**Existing source formula notes:**

D_T(r)=c_T² r^p. G-free PSD coefficient 8π² Γ(p+2) sin(πp/2)(rho r_min c_T/T)² v^p Ω^-(p+1) B(x,theta), x=Ωr_min/v; B=e^-2x[cos²θ(k1e²-k0e²)+0.5sin²θ(k0e k2e-k1e²)]. SI ×G².

## Parameters and calculation paths

### Parameters and units

density_kg_m3 rho≥0; temperature_k T&gt;0; structure_coefficient_k_m_neg_p_over_2 c_T≥0; structure_exponent 0&lt;p&lt;2; wind_speed_m_s v&gt;0; minimum_approach_m r_min&gt;0; wind_axis_angle_rad θ scalar; frequency_hz f&gt;0.

### Calculation paths

#### T1 uniform advection projected acceleration PSD

- `gragra.observables.stochastic_temperature.uniform_advection_temperature_acceleration_psd`

**Path-level source-declared output union:** a — Spectral density (PSD)

## Derivation and sources

### Derivation

Local module implements scaled Bessel geometry and log-tail branch for float64 range; equations are local specification/transcription, not independently proven here.

### References

Existing public documentation and original formulation context:

- [Stochastic NN](../guide/stochastic-nn.md)

**Bibliographic inventory:** partial.

- The bibliography identifies Jan Harms, ‘Terrestrial gravity fluctuations,’ Living Reviews in Relativity 22 (2019), DOI 10.1007/s41114-019-0022-2.
- For Creighton, the targeted source search located local metadata in the T1 specification but no separate title/author/year/arXiv bibliography record; source identity beyond that recorded metadata is not independently established.
- [DOI 10.1007/s41114-019-0022-2](https://doi.org/10.1007/s41114-019-0022-2)

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

Local unit/oracle test code exists; no tests were run in this inventory. T1 formal family record is still HOLD, so test code and candidate protocol records do not establish formal science acceptance.

- [`tests/test_stochastic_temperature.py`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/test_stochastic_temperature.py) BASE inputs rho=1.2 kg/m³, T=293 K, c_T=0.1 K m^(-p/2), p=2/3, wind=10 m/s, r_min=5 m, theta=0; geometry factor comparison at x={0.1,0.5,1,3,10} uses scipy quad epsabs=1e-13, epsrel=2e-12, limit=300, comparison rtol=3e-11, atol=1e-14. Large-x x=50 comparison uses rtol 0.02 parallel and 0.04 transverse.
- Separate independent oracle test code uses mpmath 35/45 decimal precision at p={0.4, 2/3, 1.3}, x={0.07,1.7,13}; checks integral identity residual &lt;1e-28 and positive-tail bound. This is test code, not execution evidence.

### Applicability and limitations


**Scope note:** Strict 0&lt;p&lt;2, positive T, wind, minimum approach, positive frequencies; density and c_T nonnegative. Taylor frozen flow, uniform straight advection, homogeneous isotropic structure-function assumptions.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- No arbitrary wind field, finite-outer-scale cutoff, or spatially varying temperature-statistics API is present in this function.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
