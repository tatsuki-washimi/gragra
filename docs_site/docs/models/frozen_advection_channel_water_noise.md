<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Frozen-advection channel water noise

**Validation level:** Unassigned (the scale and assignment criteria have not yet been defined).

**Physical domain:** water  
**Geometry:** channel

Narrow water channel; transfer functions, acceleration/strain ASD

**Source-declared observable / representation pairs:**

- Φ a Γ — Phasor
- a — Spectral density (ASD)

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry

- Guide gives channel sketch by prose; no reusable channel geometry plotting adapter identified.

## Equations

**Existing source formula notes:**

- Tφη/G=−2ρw K₀(2πfD/v).
- Taxη/G=2iρw(2πf/v)K₀(2πfD/v); Tazη/G=−2ρw(2πf/v)K₁(2πfD/v).
- Trace Γ=0 in vacuum for full tensor.

## Parameters and units

- frequency_hz Hz &gt;0; velocity_mps m/s nonzero; distance_m D&gt;0; width_m&gt;0; density_kg_m3 default 1000 kg/m³; eta_m m complex/real; surface_asd_m_per_rt_hz m/√Hz; component x/z/all.

## APIs and calculation paths

### G-free complex transfer and SI phasor observables

- `gragra.observables.water_channel_advection_observables`

**Path-level source-declared outputs:** Φ a Γ — Phasor

### acceleration ASD

- `gragra.observables.water_channel_advection_asd`

**Path-level source-declared outputs:** a — Spectral density (ASD)

Factory APIs construct model/evaluation objects; observables may be obtained after the documented .at(...).run() sequence rather than directly from the factory call. Follow the linked API examples.

## Derivation and provenance

**Inventory status:** partial.

water oracle contains integral/Bessel reference functions and validation code; not an admitted derivation here.

The page cites public sources below. A fully typeset public derivation is a separate editorial item wherever the existing source material has not been reproduced here.

## References

Existing public documentation and original formulation context:

- [Water NN](../guide/water-newtonian-noise.md)
- [water API](../api/water.md)

**Bibliographic inventory:** partial.

- Atsushi Nishizawa (2021-03-31), 'Water Newtonian noise for KAGRA' (JGW-T1706440-v2).
- Atsushi Nishizawa (2021-09-03), 'Edge contribution of a water channel to a Newtonian noise' (JGW-T2214308-v1).

## Verification evidence

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 4.
- Catalogue probe state: recorded (a representative API output contract, not an accuracy test).
- This documentation generation does not perform scientific re-execution or assign an accuracy Level.

**Existing verification inventory:**

Analytic-reference and finite surface reference code exists; not run in this inventory.

- (internal evidence record) includes spatial quadrature vs Bessel checks, tensor trace and limits; (internal evidence record) includes identities and stress limits. (internal evidence record) reports 1e-10 then 1e-12 adaptive quadrature tolerances; code/result distinction: no committed executed science result found in this pass.

### Registered representative output probes

| API | Observable | Representation | Probe |
|---|---|---|---|
| `gragra.observables.water_channel_advection_observables` | Φ | Phasor | `water_transfer_si` |
| `gragra.observables.water_channel_advection_observables` | a | Phasor | `water_transfer_si` |
| `gragra.observables.water_channel_advection_observables` | Γ | Phasor | `water_transfer_si` |
| `gragra.observables.water_channel_advection_asd` | a | Spectral density (ASD) | `water_asd_si` |

## Applicability and limitations

- transfer SI layer returns phasor values; ASD uses magnitude and is not a deterministic phasor. channel model's DC limit is separate API; high/low numerical limits are guarded.

**Scope note:** Frozen advection infinite/narrow channel assumptions; f&gt;0 for ASD; geometry and input constraints validated.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)

## Information still to locate or publish

- verified primary source and saved result artifact; channel illustration.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
