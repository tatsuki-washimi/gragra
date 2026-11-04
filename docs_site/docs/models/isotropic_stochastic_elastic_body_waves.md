<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Isotropic stochastic elastic body waves

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** stochastic seismic  
**Geometry:** `fullspace`, `cavity`

Infinite-medium P field and idealized P/S cavity-center CSD

**Source-declared observable / representation pairs:**

- a — Spectral density (CSD)

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry

![Isotropic stochastic elastic body waves centered cavity schematic](../assets/models/isotropic_stochastic_body_waves.svg)

*3D isotropic diffuse field of P and S body waves surrounding a centered spherical cavity with sensor at center c.*

B2 centered spherical cavity only, scattering neglected; arbitrary cavity geometry unsupported by API. basic plotting helper supports SphericalShell static shape but no source coupled setup stated.

## Analytic formulation, sources, and scope

### Equations

**1. Full-space isotropic P-waves (B1)** (Harms 2019 Eqs. 55, 60):

$$
\mathbf S_a^{(G\text{-free})}(f) = (4\pi \rho)^2 S_{\xi,P}(f) \, \mathbf I_3
$$

where \(S_{\xi,P}(f)\) is the one-sided per-component P-wave displacement PSD.

**2. Spherical cavity isotropic P/S-waves (B2)** (Harms 2019 Eq. 65):

Evaluated at the center of a spherical cavity of radius \(R > 0\) in an elastic medium with wavenumbers \(k_P = 2\pi f / v_P\) and \(k_S = 2\pi f / v_S\):

Define the spherical Bessel suppression factors:

$$
f_P = \frac{j_1(k_P R)}{k_P R}, \qquad
f_S = \frac{j_1(k_S R)}{k_S R}.
$$

The isotropic G-free acceleration CSD tensor is:

$$
\mathbf S_a^{(G\text{-free})}(f) = (4\pi \rho)^2 \left[ 4 f_P^2 S_{\xi,P}(f) + f_S^2 S_{\xi,S}(f) \right] \mathbf I_3
$$

where \(j_1(x) = \sin(x)/x^2 - \cos(x)/x\).

Physical SI acceleration CSD tensors are obtained by multiplying by \(G^2\):

$$
\mathbf S_a^{(\mathrm{SI})}(f) = G^2 \mathbf S_a^{(G\text{-free})}(f).
$$

### Derivation

Full-space uses local bulk gravity transfer. Cavity implementation uses stable j1(x)/x series for |x|&lt;1e-3; local computational transcription, not independent proof.

### References

Existing public documentation and original formulation context:

- [Stochastic NN](../guide/stochastic-nn.md)

**Bibliographic inventory:** located.

### Applicability and scope

f&gt;0, one-sided per-component PSD nonnegative; B2 radius, density, vp, vs conditions; no scattering. No general displacement/potential/gradient outputs.

## STEP 1 — Setup

Primary path: `fullspace_p_isotropic_acceleration_csd`

**A+ (`tutorial-display` axes):** wrap outputs with GWexpy only in the generator; do not invent frequency grids on the output side.

Build inputs consistent with **Detailed reference**:

```python
density = 2500.0
frequency = 2.0
displacement_psd = 1e-14
```

## STEP 2 — Validate / Inspect

STEP 2 items are tagged by information source:

- **`api`**: read from a gragra evaluation object: `validate()` is a **method**; `parameters`, `formulas`, and `references` are **`@property`** objects (not callable).
- **`tutorial-display`**: assembled for this page from documented inputs or catalogue-verified display equations (not a separate gragra registry lookup).
- **`api-output`**: taken from executed result payloads (units, shapes, GWexpy axes).

#### Display equations (`tutorial-display`)

Catalogue-registered analytic formulation (same as **Analytic formulation, sources, and scope** above):

**1. Full-space isotropic P-waves (B1)** (Harms 2019 Eqs. 55, 60):

$$
\mathbf S_a^{(G\text{-free})}(f) = (4\pi \rho)^2 S_{\xi,P}(f) \, \mathbf I_3
$$

where \(S_{\xi,P}(f)\) is the one-sided per-component P-wave displacement PSD.

**2. Spherical cavity isotropic P/S-waves (B2)** (Harms 2019 Eq. 65):

Evaluated at the center of a spherical cavity of radius \(R > 0\) in an elastic medium with wavenumbers \(k_P = 2\pi f / v_P\) and \(k_S = 2\pi f / v_S\):

Define the spherical Bessel suppression factors:

$$
f_P = \frac{j_1(k_P R)}{k_P R}, \qquad
f_S = \frac{j_1(k_S R)}{k_S R}.
$$

The isotropic G-free acceleration CSD tensor is:

$$
\mathbf S_a^{(G\text{-free})}(f) = (4\pi \rho)^2 \left[ 4 f_P^2 S_{\xi,P}(f) + f_S^2 S_{\xi,S}(f) \right] \mathbf I_3
$$

where \(j_1(x) = \sin(x)/x^2 - \cos(x)/x\).

Physical SI acceleration CSD tensors are obtained by multiplying by \(G^2\):

$$
\mathbf S_a^{(\mathrm{SI})}(f) = G^2 \mathbf S_a^{(G\text{-free})}(f).
$$

`request.formulas` lists **implementation routing** (`api`), which may differ from these display equations.

For direct observable evaluation, parameters and display equations are verified above (`tutorial-display`).

```python
try:
    from IPython.display import display
except ImportError:
    display = print
# Direct observable evaluation: parameters verified against catalogue specification
print('Input configuration verified.')
```

## STEP 3 — Calculate

Evaluate the model (`api-output`):

```python
import numpy as np
from gragra.observables.stochastic_bodywave import fullspace_p_isotropic_acceleration_csd

csd = fullspace_p_isotropic_acceleration_csd(
    density_kg_m3=density, frequency_hz=frequency, p_displacement_psd_m2_hz=displacement_psd,
)
print('CSD matrix shape:', csd.shape)
print('trace(CSD):', np.trace(np.real(csd)))
```


## STEP 4 — Results / Comparison

The code below computes the result for each case and plots it. Each case is named on the x-axis or in the legend (`api-output`).

```python
import matplotlib.pyplot as plt
import numpy as np
from gragra.observables.stochastic_bodywave import fullspace_p_isotropic_acceleration_csd

psds = np.array([1e-15, 5e-15, 1e-14, 5e-14, 1e-13])
traces = [
    float(np.trace(np.real(fullspace_p_isotropic_acceleration_csd(
        density_kg_m3=2500.0, frequency_hz=2.0, p_displacement_psd_m2_hz=float(p),
    ))))
    for p in psds
]

fig, ax = plt.subplots(figsize=(8.0, 4.8))
ax.plot(psds, traces, "o-", color="#316aa5", label=r"$\mathrm{Tr}(\mathrm{CSD})$")
ax.set_xlabel(r"Displacement PSD [m$^2$/Hz]")
ax.set_ylabel(r"Trace of CSD [m$^2$/s$^4$/Hz]")
ax.set_title(r"Stochastic isotropic P-body waves ($f=2\,\mathrm{Hz}$)")
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend()
fig.tight_layout()
plt.show()
```


![STEP 4 comparison (_isotropic_stochastic_elastic_body_waves_step4_comparison.svg_)](../assets/models/tutorials/isotropic_stochastic_elastic_body_waves_step4_comparison.svg)

*Figure generated by executing the STEP 4 comparison code above.*

Saved with `fig.savefig(path, bbox_inches='tight', metadata={'Date': None})` and `svg.hashsalt='gragra'` under Matplotlib 3.11.2. SVG byte reproducibility was verified with Matplotlib 3.11.2 under these save conditions. Other versions may produce different SVG bytes or layouts.

## Detailed reference and verification

### Parameters and units

#### API parameters

| Function / Path | Parameter | Type / Format | Default | Description |
|---|---|---|---|---|
| `fullspace_p_isotropic_acceleration_csd` | `density_kg_m3` | float | required (kw-only) | Elastic medium mass density in kg/m³ |
| `fullspace_p_isotropic_acceleration_csd` | `frequency_hz` | float \| ArrayLike | required (kw-only) | Evaluation frequency in Hz (f > 0) |
| `fullspace_p_isotropic_acceleration_csd` | `p_displacement_psd_m2_hz` | float \| ArrayLike | required (kw-only) | One-sided per-component P-wave displacement PSD in m²/Hz |
| `spherical_cavity_ps_isotropic_acceleration_csd` | `radius_m` | float | required (kw-only) | Cavity radius R in meters (R > 0) |
| `spherical_cavity_ps_isotropic_acceleration_csd` | `v_p_m_s` | float | required (kw-only) | Compressional P-wave speed in m/s |
| `spherical_cavity_ps_isotropic_acceleration_csd` | `v_s_m_s` | float | required (kw-only) | Shear S-wave speed in m/s (\(0 < v_S < v_P\)) |
| `spherical_cavity_ps_isotropic_acceleration_csd` | `s_displacement_psd_m2_hz` | float \| ArrayLike | required (kw-only) | One-sided per-component S-wave displacement PSD in m²/Hz |

#### API return values

| Function / Path | Return type | Representation | Quantity & Unit | Shape / Structure | Description |
|---|---|---|---|---|---|
| `fullspace_p_isotropic_acceleration_csd` | ndarray | CSD | a: (m/s²)²/Hz | (3, 3) complex128 array | One-sided acceleration CSD tensor for isotropic full-space P waves |
| `spherical_cavity_ps_isotropic_acceleration_csd` | ndarray | CSD | a: (m/s²)²/Hz | (3, 3) complex128 array | One-sided acceleration CSD tensor at center of spherical cavity for isotropic P/S waves |

#### Mathematical symbols and notation

| Symbol | Meaning | SI unit | Definition / Relation |
|---|---|---|---|
| \(\rho\) | Medium density | kg/m³ | Elastic background mass density |
| \(v_P, v_S\) | Elastic wave speeds | m/s | Compressional and shear wave speeds |
| \(R\) | Cavity radius | m | Spherical cavity radius (\(R > 0\), cavity B2 model) |
| \(f\) | Frequency | Hz | Evaluation frequency in Hz |
| \(S_{\xi,P}(f)\) | P displacement PSD | m²/Hz | Per-component one-sided P-wave displacement PSD |
| \(S_{\xi,S}(f)\) | S displacement PSD | m²/Hz | Per-component one-sided S-wave displacement PSD |
| \(\mathbf S_a(f)\) | Acceleration CSD tensor | (m/s²)²/Hz | 3x3 diagonal isotropic tensor \(\frac{4\pi}{3} G^2 \rho^2 S_{\xi,P}(f) \mathbf I_3\) (fullspace) |

#### Usage notes

- Fullspace model B1 evaluates the homogeneous unbounded P-wave field.
- Cavity model B2 evaluates centered test mass inside spherical cavity with P and S boundary couplings.

### Calculation paths

#### Fullspace isotropic P

- `gragra.observables.stochastic_bodywave.fullspace_p_isotropic_acceleration_csd`

**Path-level source-declared output union:** a — Spectral density (CSD)

#### Centered spherical cavity isotropic P/S

- `gragra.observables.stochastic_bodywave.spherical_cavity_ps_isotropic_acceleration_csd`

**Path-level source-declared output union:** a — Spectral density (CSD)

### Derivation and source notes

- The local formula specification maps formulas to the implementation; it is not itself a bibliographic record.
- [DOI 10.1007/s41114-019-0022-2](https://doi.org/10.1007/s41114-019-0022-2)

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

Test implementation records concrete checks; tests were not run in this editorial review. No committed executed-result artifact was identified for the assigned stochastic bodywave tests.

- [`tests/test_stochastic_bodywave.py`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/test_stochastic_bodywave.py): fullspace formula uses rho=2500 kg/m³, f=2 Hz, S=3e-18 m²/Hz and exact allclose rtol=0, atol=0.
- Cavity direct angular/vector-sum comparison checks real P and S CSD separately at rtol=2e-11, atol=0; imaginary residual atol=1e-13*max(expected). Code is a test oracle route, not saved run output.

### Applicability and limitations


**Scope note:** f&gt;0, one-sided per-component PSD nonnegative; B2 radius, density, vp, vs conditions; no scattering. No general displacement/potential/gradient outputs.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- Boundary scattering, non-spherical cavity and potential/gradient are absent from these APIs.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
