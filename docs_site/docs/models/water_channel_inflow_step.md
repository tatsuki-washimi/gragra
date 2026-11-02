<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Water-channel inflow step

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** water  
**Geometry:** `channel`

Deterministic transient Fourier amplitude, **not** a stationary ASD

**Source-declared observable / representation pairs:**

- a — Other (Fourier amplitude; see API)

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

![Water-channel edge step volume perturbation schematic](../assets/models/water_channel_inflow_step.svg)

*Single Heaviside step displacement of water volume \(\Delta V\) at coordinate \(x_0\) inducing transient acceleration at perpendicular distance \(D\).*

An endpoint step in a linear channel. No dedicated plotting or reuse mapping was identified; the general shape helper targets basic static shapes.

### Equations

For a transient deterministic volume inflow step event \(\Delta V\) occurring at inflow coordinate \(x_0\) in an open water channel at perpendicular horizontal distance \(D > 0\):

The single-event acceleration Fourier amplitude in physical units \(\mathrm{m/(s^2\cdot Hz) = m/s}\) is:

$$
|\Delta a_x(f)| = G \rho_w \Delta V \frac{|x_0|}{(D^2 + x_0^2)^{3/2}} \frac{1}{2\pi f}.
$$

The single-event differential gravitational strain Fourier amplitude is:

$$
|h(f)| = \frac{|\Delta a_x(f)|}{(2\pi f)^2 L_{\mathrm{arm}}}
$$

where \(L_{\mathrm{arm}}\) is the interferometer arm length (default \(3000\text{ m}\)).

*Note: This formulation represents the Fourier transform of a single deterministic transient Heaviside event, not a stationary stochastic ASD.*

## Parameters and calculation paths

### Parameters and units

#### API parameters

| Function / Path | Parameter | Type / Format | Default | Description |
|---|---|---|---|---|
| `water_edge_step_fourier_amplitude` | `x0_m` | float | required | Inflow edge coordinate along channel axis in meters |
| `water_edge_step_fourier_amplitude` | `distance_m` | float | required | Perpendicular distance from channel to detector D in meters (D > 0) |
| `water_edge_step_fourier_amplitude` | `volume_m3` | float | required | Displaced water volume step Delta V in m³ |
| `water_edge_step_fourier_amplitude` | `frequency_hz` | float \| array-like | required | Fourier transform evaluation frequency in Hz (f > 0) |
| `water_edge_step_fourier_amplitude` | `density_kg_m3` | float | 1000.0 | Fluid mass density in kg/m³ |

#### API return values

| Function / Path | Return type | Representation | Quantity & Unit | Shape / Structure | Description |
|---|---|---|---|---|---|
| `water_edge_step_fourier_amplitude` | ndarray | Fourier | a: (m/s²)·s | \((N_{\mathrm{freq}},)\) complex128 array | Fourier spectral amplitude of single transient acceleration step during water channel inflow |

#### Mathematical symbols and notation

| Symbol | Meaning | SI unit | Definition / Relation |
|---|---|---|---|
| \(x_0\) | Inflow edge coordinate | m | Position of step volume event along channel |
| \(D\) | Perpendicular distance | m | Distance from channel axis to test mass (\(D > 0\)) |
| \(\Delta V\) | Volume step | m³ | Displaced water volume magnitude (\(\Delta V > 0\)) |
| \(f\) | Fourier frequency | Hz | Evaluation frequency in Hz |
| \(\rho_w\) | Water density | kg/m³ | Fluid mass density (default 1000.0 kg/m³) |
| \(\Delta a_x(t)\) | Step acceleration response | m/s² | \(\Delta a_x(t) = G \rho_w \Delta V \frac{x_0}{(D^2 + x_0^2)^{3/2}} \Theta(t)\) |

#### Usage notes

- Evaluates single-event volume step Fourier amplitude \(\tilde{a}_x(f) = \frac{G \rho_w \Delta V x_0}{2\pi f (D^2 + x_0^2)^{3/2}}\).

### Calculation paths

#### Single edge-step acceleration Fourier amplitude

- `gragra.observables.water_edge_step_fourier_amplitude`

**Path-level source-declared output union:** a — Other (Fourier amplitude; see API)

### Calculation example and output

The following Python snippet evaluates this model with representative parameters and prints the resulting physical quantities:

```python
import numpy as np
from gragra.observables import water_edge_step_fourier_amplitude

f_amp = water_edge_step_fourier_amplitude(
    x0_m=1.0, distance_m=3.0, volume_m3=0.5, frequency_hz=np.array([0.5, 1.0])
)
print('Fourier amplitude:', f_amp)
```

Representative output:

```text
Fourier amplitude: [3.3591226e-10 1.6795613e-10]
```

## Derivation and sources

### Derivation

The formula is documented in the API docstring. No separate derivation note was identified among the assigned files.

### References

Existing public documentation and original formulation context:

- [Water NN](../guide/water-newtonian-noise.md)
- [water API](../api/water.md)

**Bibliographic inventory:** located.


## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

`test_analytic_water.py` has equation comparison rtol=1e-15; test code only, not run here; no committed executed result located.


### Applicability and limitations


**Scope note:** f&gt;0, D&gt;0, ΔV&gt;0, finite x0 and positive rho; output has acceleration×time unit m/s, and MUST NOT be treated as stationary ASD.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- The API does not provide the event-rate or amplitude statistics needed to model a stationary process.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
