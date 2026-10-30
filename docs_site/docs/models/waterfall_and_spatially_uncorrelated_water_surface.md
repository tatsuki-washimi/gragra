<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Waterfall and spatially uncorrelated water surface

**Level:** — (definition and assignment pending).

**Physical domain:** water  
**Geometry:** channel, waterfall

Idealized falling stream or incoherent channel cells

**Source-declared observable / representation pairs:**

- a — Spectral density (ASD)

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry

- plot_shape は選択済 Cuboid/Cylinder/SphericalShell の静的形状向け。水モデル専用 plot adapter はこの調査範囲で確認できず。

## Equations

**Existing source formula notes:**

- 非相関セル ASD_a=G rho sqrt(A_cell w) sqrt(pi/(8H^3)) ASD_eta.

## Parameters and units

- water_channel_uncorrelated_asd: H m&gt;0, width_m w m&gt;0, correlation_area_m2 m²&gt;0, surface_asd_m_per_rt_hz m/√Hz &gt;=0, rho kg/m³ default 1000.

## APIs and calculation paths

### 垂直落水・断面積揺らぎ ASD

- `gragra.observables.water_waterfall_asd`

**Path-level source-declared outputs:** a — Spectral density (ASD)

### 水路幅方向の非相関表面セル

- `gragra.observables.water_channel_uncorrelated_asd`

**Path-level source-declared outputs:** a — Spectral density (ASD)

Factory APIs construct model/evaluation objects; observables may be obtained after the documented .at(...).run() sequence rather than directly from the factory call. Follow the linked API examples.

## Derivation and provenance

**Inventory status:** partial.

water module documents independent Model 1 integral: int x²/(H²+x²)³ dx=pi/(8H³).

The page cites public sources below. A fully typeset public derivation is a separate editorial item wherever the existing source material has not been reproduced here.

## References

Existing public documentation and original formulation context:

- [Water NN](../guide/water-newtonian-noise.md)
- [water API](../api/water.md)

**Bibliographic inventory:** located.

- The formula/source attribution text is repository evidence of attribution, not confirmation that the referenced external source was inspected or that source text proves this implementation.
- No full bibliographic citation/DOI for the Nishizawa 2021 water source was located in the scoped water docs/source search.

## Verification evidence

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- This documentation generation does not perform scientific re-execution or assign an accuracy Level.

**Existing verification inventory:**

test_analytic_water.py contains assertions against closed forms, quadrature, shape/units and SI conversion; these are test code, not newly executed. Example rtol 1e-14 closed form, 1e-9 quadrature, 1e-4 selected product quadrature.

- No committed executed-result artifact located for these assigned models.

## Applicability and limitations

- uncorrelated formula uses narrow-channel approximation and independent cells; source attribution remains unverified.

**Scope note:** Water positive-frequency Bessel domain requires representable 0&lt;q&lt;=700; DC not allowed by ASD; velocity nonzero, distance/width/density positive.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)

## Information still to locate or publish

- 共通モデル表示に使える水面/落水の具体的な形状データやモデル専用 geometry renderer は確認されない。

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
