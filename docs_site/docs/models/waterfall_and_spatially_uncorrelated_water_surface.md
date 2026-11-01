<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Waterfall and spatially uncorrelated water surface

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** water  
**Geometry:** channel, waterfall

Idealized falling stream or incoherent channel cells

**Source-declared observable / representation pairs:**

- a — Spectral density (ASD)

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

- plot_shape targets static Cuboid, Cylinder, and SphericalShell shapes. No water-specific plotting adapter was identified in this review.

### Equations

**Existing source formula notes:**

- Waterfall model: Vertical water stream with velocity \(v\) at horizontal distance \(D &gt; 0\) with cross-sectional area perturbation \(A(z,t) = A_0 e^{-i \Omega (t - z/v)}\). G-free transfer functions are \(T_{a_x, A}^{(G\text{-free})} = -2 \rho_w \frac{|\Omega|}{|v|} K_1(|\Omega| D / |v|)\) and \(T_{a_z, A}^{(G\text{-free})} = 2i \rho_w \frac{\Omega}{v} K_0(|\Omega| D / |v|)\).
- Uncorrelated channel surface model: Narrow water channel with independent surface displacement fluctuations of correlation cell area \(A_{\mathrm{cell}}\). The integrated geometric coupling evaluates as \(\int_{-\infty}^\infty \frac{x^2}{(H^2 + x^2)^3} dx = \frac{\pi}{8 H^3}\).
- Acceleration ASD: \(\mathrm{ASD}_a(f) = G \rho_w \sqrt{A_{\mathrm{cell}} w} \sqrt{\frac{\pi}{8 H^3}} \mathrm{ASD}_\eta(f)\) in \(\mathrm{m/(s^2\cdot\sqrt{Hz})}\).

## Parameters and calculation paths

### Parameters and units

- Waterfall API (water_waterfall_asd): frequency_hz: float &gt; 0 in Hz; velocity_mps: float nonzero in m/s; distance_m: float D &gt; 0 in meters; area_asd_m2_per_rt_hz: float &gt;= 0, stream cross-sectional area ASD in m^2/sqrt(Hz); density_kg_m3: float rho &gt; 0 in kg/m^3 (default 1000.0).
- Uncorrelated channel API (water_channel_uncorrelated_asd): distance_m: float H &gt; 0 in meters; width_m: float w &gt; 0 in meters; correlation_area_m2: float A_cell &gt; 0 in m^2; surface_asd_m_per_rt_hz: float &gt;= 0 in m/sqrt(Hz); density_kg_m3: float rho &gt; 0 in kg/m^3 (default 1000.0).

### Calculation paths

#### Vertical waterfall with cross-sectional area fluctuations (ASD)

- `gragra.observables.water_waterfall_asd`

**Path-level source-declared output union:** a — Spectral density (ASD)

#### Incoherent surface cells across the channel width

- `gragra.observables.water_channel_uncorrelated_asd`

**Path-level source-declared output union:** a — Spectral density (ASD)

## Derivation and sources

### Derivation

water module documents independent Model 1 integral: int x²/(H²+x²)³ dx=pi/(8H³).

### References

Existing public documentation and original formulation context:

- [Water NN](../guide/water-newtonian-noise.md)
- [water API](../api/water.md)

**Bibliographic inventory:** located.

- The formula/source attribution text is repository evidence of attribution, not confirmation that the referenced external source was inspected or that source text proves this implementation.
- No full bibliographic citation/DOI for the Nishizawa 2021 water source was located in the scoped water docs/source search.

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

test_analytic_water.py contains assertions against closed forms, quadrature, shape/units and SI conversion; these are test code, not newly executed. Example rtol 1e-14 closed form, 1e-9 quadrature, 1e-4 selected product quadrature.

- No committed executed-result artifact located for these assigned models.

### Applicability and limitations

- uncorrelated formula uses narrow-channel approximation and independent cells; source attribution remains unverified.

**Scope note:** Water positive-frequency Bessel domain requires representable 0&lt;q&lt;=700; DC not allowed by ASD; velocity nonzero, distance/width/density positive.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- No specific water-surface or waterfall geometry data, or model-specific geometry renderer for a shared model view, was identified.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
