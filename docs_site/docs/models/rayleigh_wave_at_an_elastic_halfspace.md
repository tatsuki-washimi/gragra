<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Rayleigh wave at an elastic halfspace

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** seismic elastic  
**Geometry:** `halfspace`

Surface-wave gravity, depth response and coherent phasors

**Source-declared observable / representation pairs:**

- Φ a Γ — Phasor

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

![Rayleigh wave at an elastic halfspace schematic](../assets/models/rayleigh_wave_halfspace.svg)

*Free surface \(z = 0\) and subsurface \(z < 0\) with horizontal wavenumber \(k_R\), elliptical retrograde orbit, and exponential depth decay.*

- High-level HalfSpace phasor APIs and separate all-space literature wrapper; coefficient response differs from amplitude-bearing field.

### Equations

For a harmonic Rayleigh wave propagating along the planar traction-free surface (\(z = 0\)) of a homogeneous isotropic elastic halfspace (\(z \le 0\)) with Rayleigh speed \(c_R\), angular frequency \(\omega = 2\pi f\), and horizontal wavenumber \(k_R = \omega / c_R\):

With medium P-wave speed \(v_P\) and S-wave speed \(v_S\), the dimensionless vertical wave decay parameters are:

$$
q = \sqrt{1 - \left(\frac{c_R}{v_P}\right)^2}, \qquad
s = \sqrt{1 - \left(\frac{c_R}{v_S}\right)^2}, \qquad
\zeta = \sqrt{\frac{q}{s}} = \sqrt{\frac{q_P}{q_S}}
$$

where \(q_P = k_R q = \sqrt{k_R^2 - \omega^2 / v_P^2}\) and \(q_S = k_R s = \sqrt{k_R^2 - \omega^2 / v_S^2}\).

For surface vertical displacement amplitude \(\xi_z(0)\) at reference origin \(\mathbf x_0\), the modal amplitude constant is (Harms 2019 Eq. 98):

$$
A = \frac{\xi_z(0)}{q_P - k_R \zeta}.
$$

Above ground (\(z > 0\)), the combined surface-mass and bulk-dilatation gravitational potential evaluates as (Harms 2019 Eq. 96):

$$
\Phi_+(\mathbf x, t) = -2\pi G \rho A (1 - \zeta) e^{-k_R z} e^{i(\mathbf k_R \cdot (\mathbf x_\perp - \mathbf x_{0,\perp}) - \omega t)}.
$$

Below ground (\(z < 0\)), the gravitational potential combines the bulk compressional dilatation perturbation with surface mass displacement (Harms 2019 Eq. 97):

$$
\Phi_-(\mathbf x, t) = 2\pi G \rho A \left[ -2 e^{q_P z} + (1 + \zeta) e^{k_R z} \right] e^{i(\mathbf k_R \cdot (\mathbf x_\perp - \mathbf x_{0,\perp}) - \omega t)}.
$$

At the free surface (\(z = 0\)), \(\Phi_+(0) = \Phi_-(0) = -2\pi G \rho A (1 - \zeta) e^{i(\mathbf k_R \cdot (\mathbf x_\perp - \mathbf x_{0,\perp}) - \omega t)}\), strictly satisfying potential continuity across the boundary. Across \(z = 0\), the normal derivative jump satisfies the surface mass-sheet condition \(\left. (\partial_z \Phi_+ - \partial_z \Phi_-) \right|_{z=0} = 4\pi G \rho \xi_z(0) e^{i(\mathbf k_R \cdot (\mathbf x_\perp - \mathbf x_{0,\perp}) - \omega t)}\). Evaluation targets on the surface boundary (\(z = 0\)) are rejected with a `ValueError`.

## Parameters and calculation paths

### Parameters and units

#### API parameters

| Function / Path | Parameter | Type / Format | Default | Description |
|---|---|---|---|---|
| `rayleigh_halfspace` | `geometry` | HalfSpace | required | Elastic halfspace geometry instance (z <= 0) |
| `rayleigh_halfspace` | `density` | u.Quantity (density) | required | Medium background mass density |
| `rayleigh_halfspace` | `vp` | u.Quantity (speed) | required | Medium compressional P-wave speed |
| `rayleigh_halfspace` | `vs` | u.Quantity (speed) | required | Medium shear S-wave speed (\(0 < v_S < v_P\)) |
| `rayleigh_halfspace` | `vertical_displacement` | FrequencySeries | required | Vertical surface displacement amplitude series |
| `rayleigh_halfspace_allspace_observables` | `targets_m` | array-like (M, 3) | required | Target evaluation coordinates in meters |
| `rayleigh_halfspace_allspace_observables` | `density_kg_m3` | float | required (kw-only) | Medium density in kg/m³ |
| `rayleigh_halfspace_allspace_observables` | `v_p_m_s` | float | required (kw-only) | P-wave speed in m/s |
| `rayleigh_halfspace_allspace_observables` | `v_s_m_s` | float | required (kw-only) | S-wave speed in m/s |
| `rayleigh_halfspace_allspace_observables` | `frequency_hz` | float \| array-like | required (kw-only) | Wave frequency in Hz |
| `rayleigh_halfspace_allspace_observables` | `vertical_surface_amplitude_m` | float \| complex \| array-like | required (kw-only) | Vertical surface displacement amplitude in meters |
| `rayleigh_halfspace_allspace_observables` | `direction` | array-like (3,) | (1.0, 0.0, 0.0) | Horizontal propagation unit direction vector |
| `rayleigh_halfspace_allspace_observables` | `phase_origin_m` | array-like (3,) | (0.0, 0.0, 0.0) | Phase reference origin coordinate |
| `rayleigh_response` | `geometry` | HalfSpace | required | Elastic halfspace geometry instance (z <= 0) and surface frame |
| `rayleigh_response` | `density` | u.Quantity (density) | required (kw-only) | Medium background mass density convertible to kg/m³ |
| `rayleigh_response` | `vp` | u.Quantity (speed) | required (kw-only) | Medium compressional P-wave speed convertible to m/s |
| `rayleigh_response` | `vs` | u.Quantity (speed) | required (kw-only) | Medium shear S-wave speed convertible to m/s |
| `rayleigh_response` | `frequencies` | u.Quantity (frequency) | required (kw-only) | Positive frequency bins convertible to Hz |
| `rayleigh_response` | `direction` | object | required (kw-only) | Nonzero horizontal propagation direction in surface frame |
| `rayleigh_response` | `phase_origin` | u.Quantity (length) | required (kw-only) | Surface reference location defining the complex displacement phasor |

#### API return values

| Function / Path | Return type | Representation | Quantity & Unit | Shape / Structure | Description |
|---|---|---|---|---|---|
| `rayleigh_response` | _RayleighResponseModel | Phasor | Φ: (m²/s²)/m, a: (m/s²)/m, Γ: s⁻²/m | EvaluationModel; evaluate via .at(targets).run() to yield GravityResponse | Linear transfer response model mapping unit vertical surface displacement to gravity perturbations |
| `rayleigh_halfspace` | RayleighHalfspaceModel | Phasor | Φ: m²/s², a: m/s², Γ: s⁻² | EvaluationResult with complex Astropy Quantities | High-level Rayleigh halfspace model evaluated via .at(targets).run() |
| `rayleigh_halfspace_allspace_observables` | GravityObservables | Phasor | Φ: m²/s², a: m/s², Γ: s⁻² | potential: (M,), acceleration: (M, 3), gradient: (M, 3, 3) | Complex SI observables for Rayleigh waves evaluated both above (z > 0) and below (z < 0) ground |

#### Mathematical symbols and notation

| Symbol | Meaning | SI unit | Definition / Relation |
|---|---|---|---|
| \(\rho\) | Medium density | kg/m³ | Background elastic medium density (\(\rho > 0\)) |
| \(v_P\) | Compressional wave speed | m/s | P-wave propagation speed |
| \(v_S\) | Shear wave speed | m/s | S-wave propagation speed (\(0 < v_S < v_P\)) |
| \(f\) | Wave frequency | Hz | Linear oscillation frequency |
| \(k_R\) | Rayleigh wavenumber | rad/m | Secular equation root \(k_R = 2\pi f / v_R\) |
| \(\lambda_R\) | Rayleigh wavelength | m | \(\lambda_R = 2\pi / k_R\) |
| \(q_P, q_S\) | Vertical decay rates | rad/m | \(q_P = \sqrt{k_R^2 - \omega^2/v_P^2}\), \(q_S = \sqrt{k_R^2 - \omega^2/v_S^2}\) |
| \(\xi_z(0)\) | Surface vertical displacement | m | Vertical displacement phasor at \(z = 0\) |
| \(\hat{\mathbf k}\) | Propagation vector | dimensionless | Surface propagation unit direction |

#### Usage notes

- Wave speeds must satisfy Poisson stability conditions: \(v_P > v_S > 0\) and \(v_P^2 > \frac{4}{3}v_S^2\).
- Coordinate convention: \(z = 0\) is the planar traction-free boundary; ground occupies \(z \ge 0\) in observable function targets.

### Calculation paths

#### Explicit displacement-bearing Rayleigh field

- `gragra.analytic.seismic.rayleigh_halfspace`

**Path-level source-declared output union:** Φ a Γ — Phasor

#### Amplitude-free Rayleigh response coefficients

- `gragra.analytic.seismic.rayleigh_response`

**Path-level source-declared output union:** Φ a Γ — Phasor

#### Literature all-space Rayleigh observables

- `gragra.observables.rayleigh_halfspace_allspace_observables`

**Path-level source-declared output union:** Φ a Γ — Phasor

### Calculation example and output

The following Python snippet evaluates this model with representative parameters and prints the resulting physical quantities:

```python
from gragra.observables import rayleigh_halfspace_allspace_observables

res = rayleigh_halfspace_allspace_observables(
    targets_m=[[0.0, 0.0, -10.0]], density_kg_m3=2500.0, v_p_m_s=4000.0, v_s_m_s=2000.0,
    frequency_hz=1.0, vertical_surface_amplitude_m=1e-6
)
print('potential:', res.potential)
print('acceleration:', res.acceleration)
print('gradient:', res.gradient)
```

Representative output:

```text
potential: [-2.46444768e-10+0.j]
acceleration: [[ 0.00000000e+00+8.302494e-13j -0.00000000e+00+0.000000e+00j
   1.17522099e-12+0.000000e+00j]]
gradient: [[[-2.79703266e-15+0.00000000e+00j -0.00000000e+00+0.00000000e+00j
    0.00000000e+00+3.95920972e-15j]
  [-0.00000000e+00+0.00000000e+00j -0.00000000e+00+0.00000000e+00j
   -0.00000000e+00+0.00000000e+00j]
  [ 0.00000000e+00+3.95920972e-15j -0.00000000e+00+0.00000000e+00j
    4.98732521e-15+0.00000000e+00j]]]
```

## Derivation and sources

### Derivation

Committed design and validation records describe the local independent derivation/verification approach; they remain documentary evidence and were not rerun here.

- Inventory fact: Internal design records map Harms (2019) Eqs. 96/97 to potential and acceleration observables and verify Eq. 92 algebraic cross-checks.
- Literature assertion: Surface and underground Rayleigh wave gravitational fields correspond to Harms (2019) Eqs. 96–99.
- Proof assertion: Independent Poisson and Green-function numerical checks confirm potential continuity and normal acceleration jump across the free surface.

### References

Existing public documentation and original formulation context:

- [Waves](../guide/waves.md)
- [analytic benchmarks](../guide/analytic-benchmarks.md)
- [Rayleigh workflow](../examples/rayleigh_phasor_workflow.md)

**Bibliographic inventory:** partial.

- Harms, Living Reviews Relativity 22, 6 (2019), DOI 10.1007/s41114-019-0022-2, Eq.96-98.
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

- [`tests/test_rayleigh_gravity_oracle.py`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/test_rayleigh_gravity_oracle.py) uses independent finite integral/extrapolation, 12 point vP/vS,kH grid; saved result [`tests/oracles/data/rayleigh_v1.json`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/oracles/data/rayleigh_v1.json); source reports worst extrapolation error ~9.3e-5 absolute/~0.72% relative. Test code and artifact, not run now.

### Applicability and limitations

- Literature API uses z-up and free surface z=0; z==0 targets rejected. Underground formula z&lt;0; ground formula z&gt;0. Potential continuous at interface, normal acceleration jumps, gradient generally discontinuous.
- Model APIs require HalfSpace geometry and explicit frame/phase origin; response coefficients omit amplitude/reference time.
- Cavity-wall corrections are outside the cited literature API scope.

**Scope note:** Applicability limits recorded.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- Catalogue `output_contracts` are empty; this is a missing representative probe, not evidence of absent science.
- No additional committed plot asset identified in these assigned API sources.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
