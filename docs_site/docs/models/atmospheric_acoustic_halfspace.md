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

#### API parameters

| Function / Path | Parameter | Type / Format | Default | Description |
|---|---|---|---|---|
| `acoustic_halfspace` | `halfspace` | HalfSpace | required | HalfSpace geometry representing atmospheric interface (z >= 0) |
| `acoustic_halfspace` | `surface_pressure` | u.Quantity (pressure) | required | Acoustic surface pressure perturbation amplitude |
| `acoustic_halfspace` | `sound_speed` | u.Quantity (speed) | required (kw-only) | Speed of sound in air c_s |
| `acoustic_halfspace` | `frequency` | u.Quantity (frequency) | required (kw-only) | Acoustic wave frequency f in Hz |
| `acoustic_halfspace` | `incidence_angle` | u.Quantity (angle) | required (kw-only) | Incidence angle theta from normal in (0, pi/2] (0 < theta <= pi/2) |
| `acoustic_halfspace` | `azimuth` | u.Quantity (angle) | required (kw-only) | Horizontal propagation azimuth angle phi |
| `acoustic_halfspace` | `phase_origin` | u.Quantity (length 3-vector) | required (kw-only) | Coordinate phase reference origin |
| `acoustic_halfspace` | `reference_time` | object | required (kw-only) | Reference time epoch for phasor evaluation |

#### API return values

| Function / Path | Return type | Representation | Quantity & Unit | Shape / Structure | Description |
|---|---|---|---|---|---|
| `acoustic_halfspace` | AcousticHalfspaceModel | Phasor | Φ: m²/s², a: m/s², Γ: s⁻² | EvaluationResult with complex Astropy Quantities | High-level atmospheric acoustic wavefield model evaluated via .at(targets).run() |

#### Mathematical symbols and notation

| Symbol | Meaning | SI unit | Definition / Relation |
|---|---|---|---|
| \(P_0\) | Surface pressure amplitude | Pa | Peak acoustic pressure wave amplitude at ground |
| \(c_s\) | Speed of sound | m/s | Speed of sound in atmospheric air |
| \(f\) | Acoustic frequency | Hz | Acoustic oscillation frequency |
| \(\theta\) | Incidence angle | rad | Angle from downward normal (\(0 < \theta \le \pi/2\)) |
| \(\phi\) | Azimuth angle | rad | Horizontal propagation azimuth |
| \(\mathbf k_H\) | Horizontal wavevector | rad/m | \((k_x, k_y) = \frac{2\pi f}{c_s} \sin\theta (\cos\phi, \sin\phi)\) |
| \(\rho_{\mathrm{air}}\) | Air density | kg/m³ | Ambient atmospheric air mass density |
| \(\rho_{\mathrm{ground}}\) | Ground density | kg/m³ | Subsurface medium density |

#### Usage notes

- High-level factory returns an `AcousticHalfspaceModel` evaluated via `.at(targets).run()`.
- Models acoustic reflection from a rigid ground boundary with air occupying \(z \ge 0\).

### Calculation paths

#### plane-wave rigid reflection

- `gragra.analytic.atmospheric.acoustic_halfspace`

**Path-level source-declared output union:** Φ a Γ — Phasor

### Calculation example and output

The following Python snippet evaluates this model with representative parameters and prints the resulting physical quantities:

```python
import numpy as np
import astropy.units as u
from gragra.geometry import Frame, HalfSpace
from gragra.targets import TargetPoints
from gragra.analytic.atmospheric import acoustic_halfspace

frame = Frame('world')
model = acoustic_halfspace(
    HalfSpace(frame),
    surface_pressure=100.0*u.Pa,
    sound_speed=340.0*u.m/u.s,
    frequency=1.0*u.Hz,
    incidence_angle=(np.pi / 4.0)*u.rad,
    azimuth=0.0*u.rad,
    phase_origin=[0.0, 0.0, 0.0]*u.m,
    reference_time=(100, 0),
)
res = model.at(TargetPoints([[0.0, 0.0, 50.0]]*u.m, ('t1',), frame)).run()
print('potential:', res.potential.value, res.potential.units)
print('acceleration:', res.acceleration.value, res.acceleration.units)
print('gradient:', res.gradient.value, res.gradient.units)
```

Representative output:

```text
potential: [[[-1.13426138e-09+0.j]]] [[Unit("m2 / s2")]]
acceleration: [[[ 0.00000000e+00+1.48217369e-11j]
  [ 0.00000000e+00+0.00000000e+00j]
  [-9.65309052e-12+0.00000000e+00j]]] [[Unit("m / s2") Unit("m / s2") Unit("m / s2")]]
gradient: [[[-1.93680124e-13+0.00000000e+00j]
  [ 0.00000000e+00+0.00000000e+00j]
  [ 0.00000000e+00-1.26139856e-13j]
  [ 0.00000000e+00+0.00000000e+00j]
  [ 0.00000000e+00+0.00000000e+00j]
  [ 0.00000000e+00+0.00000000e+00j]
  [ 0.00000000e+00-1.26139856e-13j]
  [ 0.00000000e+00+0.00000000e+00j]
  [-3.82424875e-13+0.00000000e+00j]]] [[Unit("1 / s2") Unit("1 / s2") Unit("1 / s2") Unit("1 / s2")
  Unit("1 / s2") Unit("1 / s2") Unit("1 / s2") Unit("1 / s2")
  Unit("1 / s2")]]
```

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

- \(\theta \in (0, \pi/2]\); grazing has \(k_z = 0\) and \(q \ne 0\).

**Scope note:** Single plane wave in homogeneous sound speed halfspace; no normal incidence or interface targets.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- dedicated geometry illustration and exact reference record.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
