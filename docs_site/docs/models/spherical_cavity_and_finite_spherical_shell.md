<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Spherical cavity and finite spherical shell

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** seismic elastic  
**Geometry:** cavity, shell

Elastic displacement response, cavity wall and finite-shell boundary contributions

**Source-declared observable / representation pairs:**

- Φ a Γ — Phasor

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

![Spherical cavity and spherical shell schematic](../assets/models/spherical_cavity_shell.svg)

*Concentric spherical cavity of radius a and outer shell boundary b with host medium density ρ.*

- center-only helper targets cavity center. high-level models accept geometry frame/phase origin and target fields; physical boundaries have exclusions per model.

### Equations

**Analytical formulation and field equations:**

- For a finite spherical shell (inner radius \(a\), outer radius \(R\), density \(\rho\)) under a plane harmonic P-wave displacement \(\mathbf\xi e^{i(\mathbf k\cdot\mathbf x - \omega t)}\), center acceleration decomposes as: \(\mathbf a_{\mathrm{bulk}} = 4\pi G \rho [j_0(ka) - j_0(kR)] \mathbf\xi\); \(\mathbf a_{\mathrm{inner\_wall}} = 4\pi G \rho \left[\frac{2 j_1(ka)}{ka} - j_0(ka)\right] \mathbf\xi\); \(\mathbf a_{\mathrm{outer\_boundary}} = 4\pi G \rho \left[j_0(kR) - \frac{2 j_1(kR)}{kR}\right] \mathbf\xi\); total \(\mathbf a_{\mathrm{total}} = 8\pi G \rho \left[\frac{j_1(ka)}{ka} - \frac{j_1(kR)}{kR}\right] \mathbf\xi\) (ANNA v4, arXiv:2603.15157v4, Section 4.1).
- Difference with infinite medium: \(\Delta\mathbf a = 8\pi G \rho \left[\frac{j_1(kR)}{kR}\right] \mathbf\xi\) for \(k &gt; 0\).
- Broader high-level cavity and shell models evaluate potential, acceleration, and gradient (\(\Phi, \mathbf a, \mathbf\Gamma\)) at general observation targets using spherical Bessel and Legendre series.

## Parameters and calculation paths

### Parameters and units

#### API parameters

| Function / Path | Parameter | Type / Format | Default | Description |
|---|---|---|---|---|
| `spherical_cavity` | `geometry` | Sphere | required | Sphere geometry defining cavity radius and center |
| `spherical_cavity` | `density` | u.Quantity (density) | required | Surrounding medium mass density |
| `spherical_cavity` | `phase_speed` | u.Quantity | FrequencySeries | required | Compressional or shear phase speed |
| `spherical_cavity` | `displacement` | FrequencySeries | required | Incident displacement amplitude FrequencySeries |
| `spherical_cavity` | `wave_type` | str | required | Incident wave type ('P' or 'S') |
| `spherical_cavity` | `propagation_direction` | array-like (3,) | required | Incident wave propagation direction vector |
| `spherical_shell` | `geometry` | SphericalShell | required | SphericalShell geometry (inner radius a, outer radius R) |
| `spherical_shell_p_acceleration_components` | `inner_radius_m` | float | required | Inner cavity radius a in meters |
| `spherical_shell_p_acceleration_components` | `outer_radius_m` | float | required | Outer shell radius R in meters (R > a) |
| `spherical_shell_p_acceleration_components` | `wavenumber_rad_m` | float | array-like | required | Incident compressional wavenumber k in rad/m |
| `spherical_shell_p_acceleration_components` | `density_kg_m3` | float | required | Shell medium density in kg/m³ |
| `spherical_shell_p_acceleration_components` | `displacement_m` | float | complex | required | Incident displacement phasor amplitude in meters |
| `spherical_shell_p_acceleration_components` | `direction` | array-like (3,) | required | Wave propagation unit direction vector |

#### Mathematical symbols and notation

| Symbol | Meaning | SI unit | Definition / Relation |
|---|---|---|---|
| \(a\) | Inner cavity radius | m | Inner boundary radius of cavity |
| \(R\) | Outer shell radius | m | Outer boundary radius of shell (\(R > a\)) |
| \(k\) | Wavenumber | rad/m | Compressional or shear wavenumber \(k = 2\pi f / v\) |
| \(\rho\) | Medium density | kg/m³ | Mass density of shell medium |
| \(\xi\) | Displacement phasor | m | Incident wave displacement phasor amplitude |
| \(\hat{\mathbf n}\) | Propagation vector | dimensionless | Incident wave direction unit vector |
| \(v_P, v_S\) | Elastic wave speeds | m/s | Compressional and shear wave speeds |

#### Usage notes

- High-level models evaluate to `SphericalCavityModel` or `_SphericalShellModel` via `.at(targets).run()`.
- Center helper `spherical_shell_p_acceleration_components` calculates the 4 individual contributions at the cavity center: bulk compression, inner wall, outer wall, and total acceleration.

### Calculation paths

#### high-level spherical cavity phasor model

- `gragra.analytic.seismic.spherical_cavity`

**Path-level source-declared output union:** Φ a Γ — Phasor

#### high-level finite spherical shell P/S phasor model

- `gragra.analytic.seismic.spherical_shell`

**Path-level source-declared output union:** Φ a Γ — Phasor

#### center-only finite-shell P-wave four-component acceleration

- `gragra.observables.spherical_shell_p_acceleration_components`

**Path-level source-declared output union:** a — Phasor

## Derivation and sources

### Derivation

Finite shell general-point harmonic coefficient derivations are documented; exact CAS residuals are claimed in saved validation record. Cavity-series local derivation/implementation record exists.

- U10 document says 35 exact CAS residuals and general-point finite-interval harmonic coefficients/Cartesian jets are additional derivations. Formula ledger documents adopted fixed spherical cavity series and exact CAS/axis oracle scripts.

### References

Existing public documentation and original formulation context:

- [Analytic benchmarks](../guide/analytic-benchmarks.md)
- [analytic models](../guide/gen1-analytic-models.md)

**Bibliographic inventory:** located.

- ANNA v4 Section 4.1 Eqs.47–53 for finite-shell center P formulas. U10 validation says Harms 2019 continuity/bulk/surface definitions and Reumers v4 P/S center formulas were checked against actual PDFs in that recorded work.
- [arXiv:2603.15157v4](https://arxiv.org/abs/2603.15157v4)

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

Existing tests/docs/results are evidence records only; none rerun. Shell validation records a fixed 24-case matrix and tolerances, but those runs were before candidate and raw artifacts external to repo.

- U10 cases: b=10m, a/b=.2,.8,.99,.9999; kb=.01,.8,5; P/S; rho=2300 kg/m³; ξ=1e-9(1+.4i)m; center/offaxis/inner/material/external targets. Acceptance=1e-12 Q_scale+r_Q norm(reference), r_Q=(1e-10,1e-10,1e-9); ratios recorded, reference &lt;=0.1. No elastic scattering comparison. P center formula test implementation [`tests/test_u10_spherical_shell_general.py`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/test_u10_spherical_shell_general.py). Separate legacy P helper documented ANNA formula. Historical numerical artifacts are stored outside the public repository and were not reproduced or inspected for this catalogue update.

### Applicability and limitations

- U10 validation explicitly claims no comparison with an elastically scattered field. General spherical cavity series is no-scattering; physical wall excluded. Do not infer static response or PSD/time-series outputs.

**Scope note:** Prescribed plane-wave displacement and finite first-order perturbation; finite shell response does not satisfy elastic scattering conditions. Center-only helper is P-wave acceleration only; high-level shell supports distinct P/S and contribution path.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- catalogue output_contracts empty; no runnable representative probe specified in ledger.
- Saved artifact root external and historical; not re-run or inspected here.
- No ANNA execution or elastically scattered field comparison claimed.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
