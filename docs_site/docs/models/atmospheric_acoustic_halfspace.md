<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Atmospheric acoustic halfspace

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** atmospheric acoustic  
**Geometry:** halfspace

Plane acoustic wave and rigid-ground reflection

**Source-declared observable / representation pairs:**

- Φ a Γ — Phasor

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

![Atmospheric acoustic halfspace schematic](../assets/models/atmospheric_acoustic_halfspace.svg)

*Acoustic waves in atmospheric halfspace z > 0 with density perturbation δρ = δp / c_s² acting on ground sensor at z = 0.*

- z=0 interface; target points on interface rejected.

### Equations

**Analytical formulation and field equations:**

- The acoustic plane wave has frequency \(f\), sound speed \(c_s\), horizontal wavenumber \(q = k \sin\theta\), and vertical wavenumber \(k_z = k \cos\theta\), where \(k = 2\pi f / c_s\) and \(\theta \in (0, \pi/2]\) is the angle of incidence from zenith (normal incidence \(\theta=0\) is excluded; grazing incidence \(\theta=\pi/2\) is valid).
- With horizontal wavevector \(\mathbf q = (q\cos\phi_0, q\sin\phi_0)\) and phase factor \(e^{i \mathbf q \cdot (\mathbf x_\perp - \mathbf x_{0,\perp})}\), the surface pressure amplitude is \(P_0\). The G-free potential coefficient is \(C = \frac{4\pi P_0}{c_s^2 k^2}\).
- In the atmospheric layer (\(z &gt; 0\)), \(\Phi^{(G\text{-free})} = C e^{i \mathbf q \cdot (\mathbf x_\perp - \mathbf x_{0,\perp})} \left[\frac{1}{2} e^{-q z} - \cos(k_z z)\right]\), satisfying \(\nabla^2 \Phi^{(G\text{-free})} = 4\pi \delta\rho = -4\pi (P_0/c_s^2) e^{i \mathbf q \cdot \mathbf x_\perp} \cos(k_z z)\).
- In the underground halfspace (\(z &lt; 0\)), density perturbation is zero and \(\Phi^{(G\text{-free})} = -\frac{1}{2} C e^{i \mathbf q \cdot (\mathbf x_\perp - \mathbf x_{0,\perp})} e^{q z}\), satisfying Laplace\'s equation \(\nabla^2 \Phi = 0\).
- Acceleration and gradient are \(\mathbf a^{(G\text{-free})} = -\nabla \Phi^{(G\text{-free})}\) and \(\mathbf\Gamma^{(G\text{-free})} = -\nabla\nabla \Phi^{(G\text{-free})}\). SI observables multiply by \(G\).

## Parameters and calculation paths

### Parameters and units

| Symbol | Parameter | Meaning | SI unit |
|---|---|---|---|
| \(\rho_{\mathrm{air}}\) | `air_density_kg_m3` | Atmospheric air mass density | kg/m³ |
| \(\rho_{\mathrm{ground}}\) | `ground_density_kg_m3` | Subsurface ground mass density | kg/m³ |
| \(c_s\) | `sound_speed_m_s` | Speed of sound in air | m/s |
| \(f\) | `frequency_hz` | Acoustic wave frequency | Hz |
| \(P_0\) | `pressure_amplitude_pa` | Acoustic pressure wave amplitude | Pa |
| \(\mathbf k_H\) | `horizontal_wavenumber_rad_m` | Horizontal wavevector (k_x, k_y) | rad/m |
| \(\mathbf x\) | `targets_m` | Sensor evaluation coordinates | m |

**Additional notes on parameters and models:**

- surface_pressure_pa: float, acoustic pressure amplitude P0 at z=0 in Pa; sound_speed_m_s: float, positive speed of sound in m/s; frequency_hz: float, positive wave frequency in Hz; incidence_angle_rad: float in (0, pi/2], zenith angle theta (theta=0 normal incidence excluded, theta=pi/2 grazing allowed); azimuth_rad: float, horizontal propagation azimuth phi0 in radians; phase_origin_m: array-like (3,), reference point for phase zero in meters; targets_m: array-like (M, 3), target coordinates in meters.
- Observables return complex phasors (peak amplitude, time reconstruction Re[X exp(-2 pi i f (t - t_ref))]). SI outputs multiply G_free by G.

### Calculation paths

#### plane-wave rigid reflection

- `gragra.analytic.atmospheric.acoustic_halfspace`

**Path-level source-declared output union:** Φ a Γ — Phasor

## Derivation and sources

### Derivation

Closed form present; no independent derivation text located in inspected guide.

- implementation formulas do not establish derivation correctness.

### References

Existing public documentation and original formulation context:

- [Analytic models](../guide/gen1-analytic-models.md)

**Bibliographic inventory:** located.

- The G-free air and ground solutions and interface conditions are specified in a non-public design note, which is not reproduced here.
- source=gragra.constants.G_SI; detail=applied exactly once by the shared analytic bridge.

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

Noncavity production comparison script and module test suites exist; not run in this editorial review.

- [`tests/oracles/gen1/compare_noncavity_production.py`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/oracles/gen1/compare_noncavity_production.py) checks non-cavity atmospheric and seismic production paths. An unpublished test suite verifies acoustic halfspace transfer functions against analytical limits; no committed golden json artifact is used for this model.

### Applicability and limitations

- θ in (0,π/2]; grazing has k_z=0 and q≠0.

**Scope note:** Single plane wave in homogeneous sound speed halfspace; no normal incidence or interface targets.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- dedicated geometry illustration and exact reference record.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
