<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Water-channel inflow step

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** water  
**Geometry:** channel

Deterministic transient Fourier amplitude, **not** a stationary ASD

**Source-declared observable / representation pairs:**

- a — Other (Fourier amplitude; see API)

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

![Water-channel inflow step schematic](../assets/models/water_channel_inflow_step.svg)

*Step change in channel water level Δh propagating at speed U and its transient acceleration Fourier response.*

An endpoint step in a linear channel. No dedicated plotting or reuse mapping was identified; the general shape helper targets basic static shapes.

### Equations

**Analytical formulation and field equations:**

- Models the transient gravitational perturbation produced by a single deterministic Heaviside volume step event \(\Delta V\) at channel inflow coordinate \(x_0\) at perpendicular distance \(D &gt; 0\).
- Single-event acceleration Fourier amplitude: \(|\Delta a_x(f)| = G \rho_w \Delta V \frac{|x_0|}{(D^2 + x_0^2)^{3/2}} \frac{1}{2\pi f}\) with physical units \(\mathrm{m/(s^2\cdot Hz) = m/s}\).
- Single-event strain Fourier spectral amplitude: \(|h(f)| = \frac{|\Delta a_x(f)|}{(2\pi f)^2 L_{\mathrm{arm}}}\), where \(L_{\mathrm{arm}}\) is the interferometer arm length (default \(3000\text{ m}\)).

## Parameters and calculation paths

### Parameters and units

- x0_m: float finite, coordinate of inflow edge along channel axis in meters; distance_m: float D &gt; 0, perpendicular distance to test mass in meters; volume_m3: float Delta V &gt; 0, displaced volume step in m^3; frequency_hz: float or array-like &gt; 0, Fourier frequency in Hz; arm_length_m: float L &gt; 0, interferometer arm length in meters (default 3000.0 m); density_kg_m3: float rho &gt; 0, fluid density in kg/m^3 (default 1000.0).

### Calculation paths

#### Single edge-step acceleration Fourier amplitude

- `gragra.observables.water_edge_step_fourier_amplitude`

**Path-level source-declared output union:** a — Other (Fourier amplitude; see API)

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

test_analytic_water.py has equation comparison rtol=1e-15; test code only, not run here; no committed executed result located.


### Applicability and limitations


**Scope note:** f&gt;0, D&gt;0, ΔV&gt;0, finite x0 and positive rho; output has acceleration×time unit m/s, and MUST NOT be treated as stationary ASD.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- The API does not provide the event-rate or amplitude statistics needed to model a stationary process.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
