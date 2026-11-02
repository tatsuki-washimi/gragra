<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Individual P/SV and reflected-P benchmarks

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** seismic elastic  
**Geometry:** halfspace

Single incident mode versus a traction-free reflected composite

**Source-declared observable / representation pairs:**

- Φ a Γ — Phasor

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

![Individual P/SV and reflected-P benchmarks schematic](../assets/models/individual_p_sv_reflection.svg)

*Decomposition into incident P, reflected P, and mode-converted SV wave potentials with horizontal wavenumber matching.*

- Bare wave excludes free-surface composite and rejects z=0; reflected composite satisfies traction-free boundary and is ground-only z&gt;0; literal Eq90 helper distinct from production coefficient.

### Equations

**Analytical formulation and field equations:**

- Single upward plane P-wave (Harms 2019 Eqs. 86/87): with prefactor \(C_1 = -\frac{\rho \xi_0^P}{i k_P}\), G-free potential is \(\Phi^{(G\text{-free})} = 2\pi C_1 e^{-k_\rho z} e^{i \mathbf k_\rho \cdot \mathbf x_\perp}\) above ground (\(z &gt; 0\)), and \(\Phi^{(G\text{-free})} = 2\pi C_1 [2 e^{i k_z z} - e^{k_\rho z}] e^{i \mathbf k_\rho \cdot \mathbf x_\perp}\) underground (\(z &lt; 0\)).
- Single upward plane SV-wave (Harms 2019 Eq. 88): with prefactor \(C_2 = \frac{\rho \xi_0^{SV}}{k_S}\), G-free potential is \(\Phi^{(G\text{-free})} = 2\pi C_2 e^{-k_\rho |z|} e^{i \mathbf k_\rho \cdot \mathbf x_\perp}\), valid on both sides of the surface.
- Reflected-P composite above ground (\(z &gt; 0\)): incident P + reflected P + converted SV satisfying traction-free surface boundary condition. Potential is \(\Phi^{(G\text{-free})} = 2\pi C_1 \delta(\theta, k_\rho) e^{-k_\rho z} e^{i \mathbf k_\rho \cdot \mathbf x_\perp}\) with dimensionless wavenumber ratios \(r_\rho = k_\rho/k_S\), \(r_{zP} = k_{zP}/k_S\), \(r_{zS} = k_{zS}/k_S\) and transfer coefficient \(\delta(\theta, k_\rho) = \frac{8 r_\rho^2 r_{zP} r_{zS} + 4i r_\rho r_{zP} (1 - 2r_\rho^2)}{(1 - 2r_\rho^2)^2 + 4 r_\rho^2 r_{zP} r_{zS}}\).
- Implementation detail (A-U08): Production API uses physical \(+i\) converted-SV coefficient from the solved surface displacement field. The literal Harms (2019) Eq. (90) expression with \(-i\) is retained in private test helper _harms2019_reflected_p_delta for verification.

## Parameters and calculation paths

### Parameters and units

#### API parameters

| Function / Path | Parameter | Type / Format | Default | Description |
|---|---|---|---|---|
| `halfspace_single_bodywave_observables` | `targets_m` | array-like (M, 3) | required | Target positions in elastic halfspace (meters) |
| `halfspace_single_bodywave_observables` | `wave_type` | Literal['P', 'SV'] | required (kw-only) | Wave mode to evaluate ('P' or 'SV') |
| `halfspace_single_bodywave_observables` | `density_kg_m3` | float | required (kw-only) | Medium mass density in kg/m³ |
| `halfspace_single_bodywave_observables` | `v_p_m_s` | float | required (kw-only) | P-wave speed in m/s |
| `halfspace_single_bodywave_observables` | `v_s_m_s` | float | required (kw-only) | S-wave speed in m/s |
| `halfspace_single_bodywave_observables` | `frequency_hz` | float \| array-like | required (kw-only) | Wave frequency in Hz |
| `halfspace_single_bodywave_observables` | `incidence_angle_rad` | float | required (kw-only) | Incidence angle in radians |
| `halfspace_single_bodywave_observables` | `displacement_m` | float \| complex \| array-like | required (kw-only) | Wave displacement amplitude in meters |
| `halfspace_reflected_p_above_observables` | `incident_displacement_m` | float \| complex \| array-like | required (kw-only) | Incident P-wave displacement amplitude in meters |

#### API return values

| Function / Path | Return type | Representation | Quantity & Unit | Shape / Structure | Description |
|---|---|---|---|---|---|
| `halfspace_single_bodywave_observables` | GravityObservables | Phasor | Φ: m²/s², a: m/s², Γ: s⁻² | potential: (M,), acceleration: (M, 3), gradient: (M, 3, 3) | Complex SI observables of a single upward-propagating plane P or SV wave |
| `halfspace_reflected_p_above_observables` | GravityObservables | Phasor | Φ: m²/s², a: m/s², Γ: s⁻² | potential: (M,), acceleration: (M, 3), gradient: (M, 3, 3) | Complex SI observables above ground for reflected-P composite (incident P + reflected P + converted SV) |

#### Mathematical symbols and notation

| Symbol | Meaning | SI unit | Definition / Relation |
|---|---|---|---|
| \(\rho\) | Medium density | kg/m³ | Elastic background mass density |
| \(v_P, v_S\) | Elastic wave speeds | m/s | Compressional and shear wave speeds |
| \(f\) | Wave frequency | Hz | Linear frequency |
| \(\xi\) | Displacement phasor | m | Wave component phasor amplitude |
| \(\theta\) | Incidence angle | rad | Angle from downward vertical |
| \(\phi\) | Azimuth angle | rad | Horizontal propagation direction |

#### Usage notes

- Benchmark APIs isolate individual wave contributions (e.g. single upward incident P or SV, or incident P plus reflected P and converted SV) for analytical validation.

### Calculation paths

#### Single upward incident P or SV

- `gragra.observables.halfspace_single_bodywave_observables`

**Path-level source-declared output union:** Φ a Γ — Phasor

#### Incident P + reflected P + converted SV

- `gragra.observables.halfspace_reflected_p_above_observables`

**Path-level source-declared output union:** Φ a Γ — Phasor

### Calculation example and output

The following Python snippet evaluates this model with representative parameters and prints the resulting physical quantities:

```python
import numpy as np
from gragra.observables import halfspace_single_bodywave_observables

res = halfspace_single_bodywave_observables(
    targets_m=[[0.0, 0.0, -50.0]], wave_type='P', density_kg_m3=2500.0,
    v_p_m_s=4000.0, v_s_m_s=2000.0, frequency_hz=2.0,
    incidence_angle_rad=np.pi/6, azimuth_rad=0.0, displacement_m=1e-6
)
print('potential:', res.potential)
print('acceleration:', res.acceleration)
print('gradient:', res.gradient)
```

Representative output:

```text
potential: [9.05140363e-11+3.52756031e-10j]
acceleration: [[5.54107878e-13-1.42179116e-13j 0.00000000e+00+0.00000000e+00j
  1.79910020e-12+2.38341657e-13j]]
gradient: [[[ 2.23334433e-16+8.70390619e-16j  0.00000000e+00+0.00000000e+00j
   -3.74386199e-16+2.82601999e-15j]
  [ 0.00000000e+00+0.00000000e+00j  0.00000000e+00+0.00000000e+00j
   -0.00000000e+00+0.00000000e+00j]
  [-3.74386199e-16+2.82601999e-15j -0.00000000e+00+0.00000000e+00j
    6.70003298e-16+5.65602299e-15j]]]
```

## Derivation and sources

### Derivation

Local design and validation records contain a boundary-value re-derivation for the reflected-P coefficient and independent Green reconstruction; this inventory did not execute or re-prove them.

- Inventory fact: Model implementations provide benchmark fields for individual P, SV, and reflected P waves.
- Literature assertion: Field expressions correspond to Harms (2019) Eqs. 86–90 and Aki & Richards (2002) Section 5.2.
- Proof assertion: Cartesian displacement/stress derivation and direct Poisson boundary-value checks are verified in test oracles.

### References

Existing public documentation and original formulation context:

- [Analytic benchmarks](../guide/analytic-benchmarks.md)

**Bibliographic inventory:** partial.

- Harms 2019 DOI 10.1007/s41114-019-0022-2, Sec.3.4.1 pp.42-43 Eq.86-90.
- [DOI 10.1007/s41114-019-0022-2](https://doi.org/10.1007/s41114-019-0022-2)

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

Verification method, tolerances, and artifact status recorded; tests were not run in this task.

- [`tests/test_bodywave_literature_oracle.py`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/test_bodywave_literature_oracle.py) independently checks formulas and boundary-value solve; representative rtol 1e-10 Phi/a, 1e-9 Gamma. Test code only, no saved results found.

### Applicability and limitations

- Both paths reject z==0. Single-mode API allows mixed ground/underground targets; reflected-P composite is ground-only z&gt;0.
- Incidence angle is from surface normal and exact grazing is rejected; reflected path only accepts incident P.

**Scope note:** Applicability limits recorded.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- Catalogue output_contracts are empty; this is a missing representative probe, not evidence of absent science.
- No additional committed plot asset identified in these assigned API sources.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
