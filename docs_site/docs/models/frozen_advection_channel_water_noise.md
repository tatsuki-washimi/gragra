<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Frozen-advection channel water noise

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** water  
**Geometry:** channel

Narrow water channel; transfer functions, acceleration/strain ASD

**Source-declared observable / representation pairs:**

- ① Φ a Γ — Phasor
- ② a — Spectral density (ASD)

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

- Guide gives channel sketch by prose; no reusable channel geometry plotting adapter identified.

### Equations

**Analytical formulation and field equations:**

- The channel flows along the \(x\)-axis with velocity \(v\) and width \(w\) at perpendicular horizontal distance \(D &gt; 0\) from the mirror. Surface elevation wave is \(\eta(x,t) = \eta_0 e^{-i \Omega (t - x/v)}\) with angular frequency \(\Omega = 2\pi f\).
- Dimensionless wavenumber parameter is \(q = |\Omega| D / |v|\) (evaluated for \(0 &lt; q \le 700\); exact DC \(f=0\) gives \(q=0\)).
- Analytic G-free transfer functions use modified Bessel functions of the second kind \(K_0(q)\) and \(K_1(q)\):

## Parameters and calculation paths

### Parameters and units

- frequency_hz: float or array-like &gt; 0, wave frequency in Hz; velocity_mps: float nonzero, water flow velocity v in m/s; distance_m: float D &gt; 0, perpendicular distance to detector in meters; width_m: float w &gt; 0, channel width in meters; density_kg_m3: float rho &gt; 0, fluid density in kg/m^3 (default 1000.0); eta_m: complex or float, surface displacement phasor amplitude in meters (for observables API); surface_asd_m_per_rt_hz: float or array-like &gt;= 0, surface displacement ASD in m/sqrt(Hz) (for ASD API).

### Calculation paths

#### G-free complex transfer and SI phasor observables

- `gragra.observables.water_channel_advection_observables`

**Path-level source-declared output union:** Φ a Γ — Phasor

#### acceleration ASD

- `gragra.observables.water_channel_advection_asd`

**Path-level source-declared output union:** a — Spectral density (ASD)

## Derivation and sources

### Derivation

water oracle contains integral/Bessel reference functions and validation code; not an admitted derivation here.

- [`tests/oracles/water_analytic_reference.py`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/oracles/water_analytic_reference.py) documents line integral and K0/K1 forms. Test/reference code is not a proof nor reproduced now.

### References

Existing public documentation and original formulation context:

- [Water NN](../guide/water-newtonian-noise.md)
- [water API](../api/water.md)

**Bibliographic inventory:** partial.

- Atsushi Nishizawa (2021-03-31), 'Water Newtonian noise for KAGRA' (JGW-T1706440-v2).
- Atsushi Nishizawa (2021-09-03), 'Edge contribution of a water channel to a Newtonian noise' (JGW-T2214308-v1).

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 4.
- Catalogue probe state: recorded (a representative API output contract, not an accuracy test).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

Analytic-reference and finite surface reference code exists; tests were not run in this editorial review.

- [`tests/oracles/water_analytic_reference.py`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/oracles/water_analytic_reference.py) includes spatial quadrature vs Bessel checks, tensor trace and limits; [`tests/test_analytic_water.py`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/test_analytic_water.py) includes identities and stress limits. [`tests/oracles/water_surface_reference.py`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/oracles/water_surface_reference.py) reports 1e-10 then 1e-12 adaptive quadrature tolerances; code/result distinction: no committed executed science result found in this pass.

#### Registered representative output probes

| API | Observable | Representation | Probe |
|---|---|---|---|
| `gragra.observables.water_channel_advection_observables` | Φ | Phasor | `water_transfer_si` |
| `gragra.observables.water_channel_advection_observables` | a | Phasor | `water_transfer_si` |
| `gragra.observables.water_channel_advection_observables` | Γ | Phasor | `water_transfer_si` |
| `gragra.observables.water_channel_advection_asd` | a | Spectral density (ASD) | `water_asd_si` |

### Applicability and limitations

- transfer SI layer returns phasor values; ASD uses magnitude and is not a deterministic phasor. channel model's DC limit is separate API; high/low numerical limits are guarded.

**Scope note:** Frozen advection infinite/narrow channel assumptions; f&gt;0 for ASD; geometry and input constraints validated.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- verified primary source and saved result artifact; channel illustration.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
