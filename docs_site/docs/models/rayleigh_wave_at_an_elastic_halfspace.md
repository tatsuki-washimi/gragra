<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Rayleigh wave at an elastic halfspace

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** seismic elastic  
**Geometry:** halfspace

Surface-wave gravity, depth response and coherent phasors

**Source-declared observable / representation pairs:**

- Φ a Γ — Phasor

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

![Rayleigh wave at an elastic halfspace schematic](../assets/models/rayleigh_wave_halfspace.svg)

*Free surface z = 0 and subsurface z < 0 with horizontal wavenumber k_R, elliptical retrograde orbit, and exponential depth decay.*

- High-level HalfSpace phasor APIs and separate all-space literature wrapper; coefficient response differs from amplitude-bearing field.

### Equations

**Analytical formulation and field equations:**

- For a Rayleigh wave with horizontal wavenumber \(k_R = \omega / c_R\), vertical decay parameters \(q_P = \sqrt{k_R^2 - \omega^2/v_P^2}\) and \(q_S = \sqrt{k_R^2 - \omega^2/v_S^2}\), and surface vertical displacement amplitude \(\xi_z(0)\).
- Above ground (\(z &gt; 0\)), the potential is \(\Phi(x, z, t) = 2\pi G \rho \xi_z(0) \frac{\gamma_R}{k_R} e^{-k_R z} e^{i(k_R x - \omega t)}\) (Harms 2019 Eq. 96), where \(\gamma_R = \frac{2 k_R (q_P - q_S)}{k_R^2 + q_S^2}\).
- Below ground (\(z \le 0\)), the field combines the volume dilatation perturbation and the free-surface vertical mass-sheet displacement (Eq. 97). Acceleration and gradient are analytic derivatives \(\mathbf a = -\nabla\Phi\), \(\mathbf\Gamma = \nabla\mathbf a\).

## Parameters and calculation paths

### Parameters and units

#### API parameters

| Function / Path | Parameter | Type / Format | Default | Description |
|---|---|---|---|---|
| `rayleigh_halfspace` | `geometry` | HalfSpace | required | Elastic halfspace geometry instance (z <= 0) |
| `rayleigh_halfspace` | `density` | u.Quantity (density) | required | Medium background mass density |
| `rayleigh_halfspace` | `vp` | u.Quantity (speed) | required | Medium compressional P-wave speed |
| `rayleigh_halfspace` | `vs` | u.Quantity (speed) | required | Medium shear S-wave speed (0 < v_S < v_P) |
| `rayleigh_halfspace` | `vertical_displacement` | FrequencySeries | required | Vertical surface displacement amplitude series |
| `rayleigh_halfspace_allspace_observables` | `targets_m` | array-like (M, 3) | required | Target evaluation coordinates in meters |
| `rayleigh_halfspace_allspace_observables` | `density_kg_m3` | float | required (kw-only) | Medium density in kg/m³ |
| `rayleigh_halfspace_allspace_observables` | `v_p_m_s` | float | required (kw-only) | P-wave speed in m/s |
| `rayleigh_halfspace_allspace_observables` | `v_s_m_s` | float | required (kw-only) | S-wave speed in m/s |
| `rayleigh_halfspace_allspace_observables` | `frequency_hz` | float | array-like | required (kw-only) | Wave frequency in Hz |
| `rayleigh_halfspace_allspace_observables` | `vertical_surface_amplitude_m` | float | complex | array-like | required (kw-only) | Vertical surface displacement amplitude in meters |
| `rayleigh_halfspace_allspace_observables` | `direction` | array-like (3,) | (1.0, 0.0, 0.0) | Horizontal propagation unit direction vector |
| `rayleigh_halfspace_allspace_observables` | `phase_origin_m` | array-like (3,) | (0.0, 0.0, 0.0) | Phase reference origin coordinate |

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

## Derivation and sources

### Derivation

Committed design and validation records describe the local independent derivation/verification approach; they remain documentary evidence and were not rerun here.

- an unpublished design note maps H19 Eq.96/97 and labels acceleration/gradient derivative class B; records Eq.92 hand-algebra cross-check.
- An unpublished validation note describes an independent Poisson/Green-function numerical re-derivation and surface checks; the note is not reproduced here.

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

- Catalogue output_contracts are empty; this is a missing representative probe, not evidence of absent science.
- No additional committed plot asset identified in these assigned API sources.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
