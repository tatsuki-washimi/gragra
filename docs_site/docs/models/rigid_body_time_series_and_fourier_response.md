<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Rigid-body time series and Fourier response

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** rigid motion  
**Geometry:** point_cloud

Prescribed positions/rotations; periodic or finite-window Fourier evaluation

**Source-declared observable / representation pairs:**

- ① Φ a Γ — Time series
- ② Φ a Γ — Fourier series
- ③ Φ a Γ — Fourier transform

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

![Rigid-body kinematics and Fourier response schematic](../assets/models/rigid_body_fourier_response.svg)

*Rigid-body time-domain motion (translation c(t) and rotation R(t)) and corresponding Fourier series harmonic response.*

- plot_shape is not a Fourier trajectory renderer.

### Equations

**Analytical formulation and field equations:**

- Body kinematics follow general pose transformation \(\mathbf x(t) = \mathbf R(t) \mathbf x_{\mathrm{body}} + \mathbf c(t)\) with proper rotation \(\mathbf R(t) \in \mathrm{SO}(3)\) and translation \(\mathbf c(t)\).
- Sampled time-series path evaluates instantaneous Newtonian field (\(\Phi, \mathbf a, \mathbf\Gamma\)) at discrete time sampling steps via rigid_motion_observable.
- Continuous Fourier path integrates callable pose over time using Fourier convention \(\exp[-2\pi i f (t - t_{\mathrm{ref}})]\). For periodic motion, one-sided Fourier series amplitude coefficients satisfy \(A_0 = C_0, A_n = 2 C_n\) (\(n \ge 1\)). For finite-window Fourier transforms, windowed integral coefficients are retained directly without doubling.

## Parameters and calculation paths

### Parameters and units

#### API parameters

| Function / Path | Parameter | Type / Format | Default | Description |
|---|---|---|---|---|
| `rigid_motion` | `source` | PointMassCloud | required | Cloud of discrete point masses representing rigid body |
| `rigid_motion` | `transform` | PrescribedRigidTransform \| PoseFunction | required | Rigid pose function or time-series transformation |
| `rigid_motion` | `frame` | Frame | required (kw-only) | Reference coordinate frame instance |
| `rigid_motion` | `sampling` | TimeSampling \| None | None | Discrete time sampling (required for sampled time-series path) |
| `rigid_motion_observable` | `observable` | str | required (kw-only) | 'potential', 'acceleration', or 'gradient' |
| `rigid_motion_observable` | `softening_m` | float | 0.0 | Plummer softening parameter in meters |

#### API return values

| Function / Path | Return type | Representation | Quantity & Unit | Shape / Structure | Description |
|---|---|---|---|---|---|
| `rigid_motion` | _ContinuousRigidMotionModel | TimeSeries / Fourier | Φ: m²/s², a: m/s², Γ: s⁻² | EvaluationResult with continuous/discrete time or Fourier payloads | Continuous or sampled rigid-body motion model evaluated via .at(targets).run() |
| `rigid_motion_observable` | ndarray | TimeSeries / Fourier | Φ: m²/s², a: m/s², Γ: s⁻² | (N_frames, N_targets, ...) float64 array | Assembled dense SI observables across time frames and targets for requested observable |

#### Mathematical symbols and notation

| Symbol | Meaning | SI unit | Definition / Relation |
|---|---|---|---|
| \(M\) | Total body mass | kg | Sum of constituent point masses \(M = \sum_i m_i\) |
| \(\mathbf x_{\mathrm{cm}}(t)\) | Center of mass | m | Rigid body center-of-mass trajectory |
| \(\mathbf R(t)\) | Rotation matrix | dimensionless | Instantaneous orientation matrix in \(\mathrm{SO}(3)\) |
| \(\mathbf I\) | Inertia tensor | kg·m² | Moment of inertia tensor around center of mass |
| \(\mathbf x_i(t)\) | Point mass position | m | \(\mathbf x_i(t) = \mathbf x_{\mathrm{cm}}(t) + \mathbf R(t) \mathbf x_{i,\mathrm{body}}\) |

#### Usage notes

- Supports sampled time-series evaluation (`rigid_motion_observable`) and continuous Fourier series / Fourier transform evaluation.

### Calculation paths

#### sampled rigid-body time series

- `gragra.gravity_sources.rigid_motion`
- `gragra.observables.rigid_motion_observable`

**Path-level source-declared output union:** Φ a Γ — Time series

#### callable pose periodic Fourier series

- `gragra.gravity_sources.rigid_motion`

**Path-level source-declared output union:** Φ a Γ — Fourier series

#### callable pose finite-window Fourier transform

- `gragra.gravity_sources.rigid_motion`

**Path-level source-declared output union:** Φ a Γ — Fourier transform

### Calculation example and output

The following Python snippet evaluates this model with representative parameters and prints the resulting physical quantities:

```python
import numpy as np
from gragra.gravity_sources import PointMassCloud, PrescribedRigidTransform
from gragra.observables import rigid_motion_observable

cloud = PointMassCloud([1000.0], [[0.0, 0.0, 0.0]])
transform = PrescribedRigidTransform(
    time_s=[0.0, 1.0],
    translations_m=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
    rotation_matrices=[np.eye(3), np.eye(3)]
)
acc = rigid_motion_observable(cloud, transform, [[0.0, 0.0, 5.0]], observable='acceleration')
print('acceleration:', acc)
```

Representative output:

```text
acceleration: [[[ 0.00000000e+00  0.00000000e+00 -2.66972000e-09]]

 [[ 5.33623794e-11  0.00000000e+00 -2.66811897e-09]]]
```

## Derivation and sources

### Derivation

Newton oracle is code; Fourier numerical integration prose is not a local derivation.

- 80-digit independent mpmath oracle; no proof of periodic quadrature found.

### References

Existing public documentation and original formulation context:

- [Rigid-motion Fourier guide](../guide/rigid-motion-fourier.md)
- [results API](../api/results.md)

**Bibliographic inventory:** located.

- RigidMotionModelSpec.references returns an empty DataFrame. Continuous Fourier model and its series/transform specs likewise return empty DataFrames.

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

Tests cover observables/Fourier/oracle comparisons; not run here.

- test-code-only; rigid_motion_science oracle comparisons rtol=3e-14; no committed executed-result path located.

### Applicability and limitations

- return type provisional; G applied once at observable layer; memory budget per-lane.

**Scope note:** sampled/continuous argument combinations exclusive; finite window has aliasing/phase caveats.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- saved scientific runs not located.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
