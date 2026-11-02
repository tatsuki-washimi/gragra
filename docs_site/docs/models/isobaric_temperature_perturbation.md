<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Isobaric temperature perturbation

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** atmospheric temperature  
**Geometry:** `sphere`, `cuboid`, `cylinder`

Fourier, Gaussian and bounded temperature anomalies; density-to-gravity coupling

**Source-declared observable / representation pairs:**

- ① Φ a Γ — Static
- ② Φ a Γ — Phasor

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

![Isobaric temperature perturbation schematic](../assets/models/isobaric_temperature_perturbation.svg)

*Thermal parcel with temperature fluctuation \(\delta T\) inducing isobaric density fluctuation \(\delta\rho/\rho_0 = -\delta T / T_0\).*

- The API supports these profile objects; geometry dimensions are profile inputs. No dedicated temperature anomaly map asset was located.

### Equations

Under isobaric thermodynamic conditions, small temperature variations \(\delta T\) couple linearly to mass density perturbations:

$$
\delta\rho(\mathbf x) = -\rho_0 \frac{\delta T(\mathbf x)}{T_0}
$$

where \(\rho_0 > 0\) is ambient background density and \(T_0 > 0\) is background temperature in Kelvin.

**1. Harmonic Fourier temperature mode**:

For temperature perturbation \(\delta T(\mathbf x) = \Delta T \cos(\mathbf k \cdot (\mathbf x - \mathbf x_0))\) with central density amplitude \(\rho_c = -\rho_0 \Delta T / T_0\), the G-free gravitational potential is:

$$
\Phi^{(G\text{-free})}(\mathbf x) = -\frac{4\pi \rho_c}{k^2} e^{i \mathbf k \cdot (\mathbf x - \mathbf x_0)} \qquad (k = \|\mathbf k\| > 0)
$$

with gravitational acceleration vector:

$$
\mathbf a^{(G\text{-free})}(\mathbf x) = -i \mathbf k \Phi^{(G\text{-free})}(\mathbf x).
$$

**2. Gaussian temperature plume**:

For Gaussian temperature distribution \(\delta T(\mathbf x) = \Delta T e^{-\|\mathbf x - \mathbf c\|^2 / (2\sigma^2)}\):

At the plume center (\(\mathbf x = \mathbf c\)):

$$
\Phi^{(G\text{-free})}(\mathbf c) = -4\pi \rho_c \sigma^2, \qquad
\mathbf a(\mathbf c) = \mathbf 0, \qquad
\mathbf\Gamma^{(G\text{-free})}(\mathbf c) = -\frac{4\pi \rho_c}{3} \mathbf I_3.
$$

In the far field (\(\|\mathbf x - \mathbf c\| \gg \sigma\)), the gravitational field approaches a point mass with effective monopole mass:

$$
M_{\mathrm{eff}} = (2\pi)^{3/2} \rho_c \sigma^3.
$$

## Parameters and calculation paths

### Parameters and units

#### API parameters

| Function / Path | Parameter | Type / Format | Default | Description |
|---|---|---|---|---|
| `isobaric_temperature` | `profile` | _Profile | required | Temperature spatial profile instance (Fourier, Gaussian, or finite body) |
| `isobaric_temperature` | `delta_temperature` | u.Quantity (temperature) | required | Temperature fluctuation amplitude delta T |
| `isobaric_temperature` | `rho0` | u.Quantity (density) | required (kw-only) | Ambient unperturbed air mass density |
| `isobaric_temperature` | `T0` | u.Quantity (temperature) | required (kw-only) | Ambient absolute temperature in Kelvin |
| `isobaric_temperature` | `frequency` | u.Quantity \| None | None | Optional oscillation frequency for phasor evaluation |
| `isobaric_temperature` | `reference_time` | object \| None | None | Optional reference epoch for phasor representation |

#### API return values

| Function / Path | Return type | Representation | Quantity & Unit | Shape / Structure | Description |
|---|---|---|---|---|---|
| `isobaric_temperature` | TemperatureModel | Static / Phasor | Φ: m²/s², a: m/s², Γ: s⁻² | EvaluationResult with Astropy Quantities | Isobaric temperature perturbation model evaluated via .at(targets).run() |

#### Mathematical symbols and notation

| Symbol | Meaning | SI unit | Definition / Relation |
|---|---|---|---|
| \(\rho_0\) | Ambient air density | kg/m³ | Unperturbed background air mass density |
| \(T_0\) | Ambient temperature | K | Background absolute temperature (\(T_0 > 0\)) |
| \(\delta T\) | Temperature perturbation | K | Local temperature fluctuation |
| \(\delta\rho\) | Density perturbation | kg/m³ | Isobaric ideal gas relation \(\delta\rho = -\rho_0 \frac{\delta T}{T_0}\) |

#### Usage notes

- Evaluates via `.at(targets).run()` based on the linearized isobaric coupling \(\delta\rho / \rho_0 = -\delta T / T_0\).

### Calculation paths

#### Fourier temperature mode (static and single-frequency phasor)

- `gragra.analytic.atmospheric.isobaric_temperature`

**Path-level source-declared output union:** Φ a Γ — Static; Φ a Γ — Phasor

#### Gaussian temperature anomaly (all-space)

- `gragra.analytic.atmospheric.isobaric_temperature`

**Path-level source-declared output union:** Φ a Γ — Static; Φ a Γ — Phasor

#### Finite Sphere temperature body

- `gragra.analytic.atmospheric.isobaric_temperature`

**Path-level source-declared output union:** Φ a Γ — Static; Φ a Γ — Phasor

#### Finite Cuboid temperature body

- `gragra.analytic.atmospheric.isobaric_temperature`

**Path-level source-declared output union:** Φ a Γ — Static; Φ a Γ — Phasor

#### Finite Cylinder temperature body

- `gragra.analytic.atmospheric.isobaric_temperature`

**Path-level source-declared output union:** Φ a Γ — Static; Φ a Γ — Phasor

### Calculation example and output

The following Python snippet evaluates this model with representative parameters and prints the resulting physical quantities:

```python
import numpy as np
import astropy.units as u
from gragra.geometry import Frame
from gragra.targets import TargetPoints
from gragra.analytic.atmospheric import isobaric_temperature
from gragra.fields import TemperatureFourierMode

frame = Frame('world')
profile = TemperatureFourierMode(
    wavevector=np.array([0.1, 0.0, 0.0]) / u.m,
    phase_origin=np.zeros(3) * u.m,
    frame=frame,
)
model = isobaric_temperature(
    profile, 2.0 * u.K, rho0=1.2 * u.kg / u.m**3, T0=290.0 * u.K
)
res = model.at(TargetPoints([[0.0, 0.0, 10.0]]*u.m, ('t1',), frame)).run()
print('potential:', res.potential)
print('acceleration:', res.acceleration)
print('gradient:', res.gradient)
```

Representative output:

```text
potential: [6.94110847e-10] m2 / s2
acceleration: [[0. 0. 0.]] m / s2
gradient: [[[6.94110847e-12 0.00000000e+00 0.00000000e+00]
  [0.00000000e+00 0.00000000e+00 0.00000000e+00]
  [0.00000000e+00 0.00000000e+00 0.00000000e+00]]] 1 / s2
```

## Derivation and sources

### Derivation

equation prose plus test oracle/derivation scripts; local text is not proof of correctness.

- test suites exercise center, far-field, trace and scale invariance; no saved executed artifact identified.

### References

Existing public documentation and original formulation context:

- [Analytic models](../guide/gen1-analytic-models.md)
- [fields API](../api/physics.md)

**Bibliographic inventory:** located.

- an unpublished design note: approved isobaric, plane-wave, Gaussian and finite-body formulas.
- `gragra.fields.thermodynamic.isobaric_temperature_density_perturbation`: signed \(\delta\rho=-\rho_0\delta T/T_0\) linearization.
- `gragra.constants.G_SI`: applied exactly once by shared analytic bridge.

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

High-level test code covers plane wave phasor/static and Gaussian limits; not run here.

- Gaussian center and far field equations checked; tests use assertions including exact center limits and Poisson trace. Other cases use rtol=2e-13 scale invariance. Test-code-only.

### Applicability and limitations

- Gaussian all-space support; finite geometry material boundary rejected for potential/acceleration/gradient; phasors are peak coefficients with no 1/2.

**Scope note:** linearized isobaric model; bounded geometry not sealed heating; Fourier wavevector nonzero and zero mode excluded.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- dedicated profile visualization and exact bibliography/result records.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
