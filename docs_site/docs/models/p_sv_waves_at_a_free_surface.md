<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# P/SV waves at a free surface

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** seismic elastic  
**Geometry:** halfspace

Homogeneous elastic halfspace; incidence, reflection and mode conversion

**Source-declared observable / representation pairs:**

- Φ a Γ — Phasor

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

- High-level phasor model vs array-based SI wrapper; flat free surface and local frame conventions.

### Equations

**Existing source formula notes:**

- Evaluates the exact gravity response of an incident plane P or SV wave reflecting at a planar traction-free boundary (\(z = 0\)), including converted SV-from-P or P-from-SV modes.
- The total gravitational field combines the interior halfspace dilatational perturbation and the surface vertical displacement mass sheet \(u_z(x, y, 0, t)\).
- Returns potential \(\Phi\), acceleration \(\mathbf a\), and gradient tensor \(\mathbf\Gamma\) phasors in physical SI units (\(\mathrm{m^2/s^2}\), \(\mathrm{m/s^2}\), \(\mathrm{s^{-2}}\)). Non-grazing incident angles (\(0 \le \theta &lt; \pi/2\)) are supported.

## Parameters and calculation paths

### Parameters and units

- density_kg_m3: float, medium mass density \(\rho\) in \(\mathrm{kg/m^3}\); vp_m_s, vs_m_s: float, elastic wave speeds in m/s.
- displacement: FrequencySeries or Quantity, incident displacement amplitude in meters; wave_type: 'P' or 'SV'.
- incidence_angle_rad: float, incidence angle \(\theta \in [0, \pi/2)\) from surface normal (exact grazing \(\pi/2\) rejected); azimuth_rad: float, azimuth angle.

### Calculation paths

#### implemented model/API path

- `gragra.analytic.seismic.halfspace_psv`
- `gragra.observables.halfspace_psv_observables`

**Path-level source-declared output union:** Φ a Γ — Phasor

## Derivation and sources

### Derivation

Formula expressions are declared in source; a separate local derivation proof was not located in the focused scan.

- No separate derivation is asserted from the formula declaration alone.

### References

Existing public documentation and original formulation context:

- [Analytic models](../guide/gen1-analytic-models.md)
- [waves](../guide/waves.md)

**Bibliographic inventory:** partial.

- Local model metadata calls D1 traction-free P/SV halfspace; Fourier matching q&gt;0 and documented continuous normal-incidence q=0 family.

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

Verification method, tolerances, and artifact status recorded; tests were not run in this task.

- [`tests/test_halfspace_psv.py`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/test_halfspace_psv.py) uses numerical z-source convolution (scipy quad epsrel 2e-12 default, epsabs tolerance/10), plus SI/phase shifts/gradient trace checks; test code only, no result artifact found.

### Applicability and limitations

- High-level phasor model vs array-based SI wrapper; flat free surface and local frame conventions.

**Scope note:** Applicability limits recorded.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- Catalogue output_contracts are empty; this is a missing representative probe, not evidence of absent science.
- No additional committed plot asset identified in these assigned API sources.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
