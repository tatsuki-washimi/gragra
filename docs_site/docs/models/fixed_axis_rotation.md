<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Fixed-axis rotation

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** rigid motion  
**Geometry:** `point_cloud`

Rotating discrete mass cloud

**Source-declared observable / representation pairs:**

- Φ a Γ — Time series

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

![Fixed-axis rotation schematic](../assets/models/fixed_axis_rotation.svg)

*Point-mass cloud rotating at constant angular velocity Ω about fixed rotation axis n̂.*

- `plot_shape` supports a limited set of geometry primitives; arbitrary point-cloud trajectories were not confirmed.

### Equations

For point masses \(m_i\) revolving at constant angular velocity \(\omega\) about an axis through origin with unit direction vector \(\hat{\mathbf n}_{\mathrm{axis}}\), the position of mass \(i\) at time \(t\) with initial angle \(\theta_{0,i}\) is:

$$
\mathbf x_i(t) = \mathbf R(\hat{\mathbf n}_{\mathrm{axis}}, \omega t + \theta_{0,i}) \, \mathbf x_{i,0}.
$$

Let \(\mathbf r_i(t) = \mathbf x - \mathbf x_i(t)\) with \(r_i(t) = \|\mathbf r_i(t)\| > 0\). The instantaneous gravitational potential and acceleration are:

$$
\Phi(\mathbf x, t) = -G \sum_{i=1}^N \frac{m_i}{r_i(t)}, \qquad
\mathbf a(\mathbf x, t) = -G \sum_{i=1}^N \frac{m_i \mathbf r_i(t)}{r_i(t)^3}.
$$

The instantaneous gravity-gradient tensor is:

$$
\mathbf\Gamma(\mathbf x, t) = G \sum_{i=1}^N m_i \left[ \frac{3 \mathbf r_i(t) \otimes \mathbf r_i(t)}{r_i(t)^5} - \frac{\mathbf I_3}{r_i(t)^3} \right].
$$

Fourier-series decomposition yields harmonic components at integer multiples \(k\omega\) of the rotational frequency.

## Parameters and calculation paths

### Parameters and units

#### API parameters

| Function / Path | Parameter | Type / Format | Default | Description |
|---|---|---|---|---|
| `fixed_axis_rotation` | `point_mass` | PointMass | required | Point mass object defining mass m |
| `fixed_axis_rotation` | `radius` | u.Quantity (length) | required | Perpendicular orbital radius r > 0 |
| `fixed_axis_rotation` | `angular_velocity` | u.Quantity (angular speed) | required | Constant rotation angular velocity omega |
| `fixed_axis_rotation` | `axis` | array-like (3,) | required (kw-only) | Rotation axis direction vector |
| `fixed_axis_rotation` | `initial_phase` | u.Quantity (angle) | required (kw-only) | Initial azimuthal phase angle at t = 0 |
| `fixed_axis_rotation` | `frame` | Frame | required (kw-only) | Reference coordinate frame instance |
| `fixed_axis_rotation` | `sampling` | TimeSampling | required (kw-only) | Output time axis discretization |

#### API return values

| Function / Path | Return type | Representation | Quantity & Unit | Shape / Structure | Description |
|---|---|---|---|---|---|
| `fixed_axis_rotation` | _MotionModel | TimeSeries | Φ: m²/s², a: m/s², Γ: s⁻² | EvaluationResult with TimeSeriesMatrix payloads | Point-mass circular trajectory motion model evaluated via .at(targets).run() |

#### Mathematical symbols and notation

| Symbol | Meaning | SI unit | Definition / Relation |
|---|---|---|---|
| \(m\) | Rotating mass | kg | Mass of revolving point body (\(m > 0\)) |
| \(\mathbf x_c\) | Axis center | m | Origin point on rotation axis |
| \(\hat{\mathbf n}\) | Axis unit vector | dimensionless | Unit vector defining orientation of rotation axis |
| \(r\) | Orbital radius | m | Perpendicular radius from axis (\(r > 0\)) |
| \(\omega\) | Angular velocity | rad/s | Constant rotation speed |
| \(\phi_0\) | Initial phase | rad | Azimuthal phase angle at \(t = 0\) |

#### Usage notes

- Axis vector must be nonzero. Target points lying on the circular orbit path raise collision `ValueError`.

### Calculation paths

#### Discrete point-mass cloud rotating about a fixed axis at constant angular velocity

- `gragra.analytic.objects.fixed_axis_rotation`

**Observables after model evaluation:** Φ a Γ — Time series

High-level model factories create evaluation objects; call the documented `.at(...).run()` workflow to evaluate them. Observable functions listed above are direct calls with their own return values.

### Calculation example and output

The following Python snippet evaluates this model with representative parameters and prints the resulting physical quantities:

```python
import astropy.units as u
from gragra.geometry import Frame
from gragra.gravity_sources import PointMassCloud
from gragra.results import TimeSampling
from gragra.targets import TargetPoints
from gragra.analytic.objects import fixed_axis_rotation

frame = Frame('world')
cloud = PointMassCloud([1000.0], [[1.0, 0.0, 0.0]])
sampling = TimeSampling(epoch=(100, 0), dt=0.05*u.s, count=5)
model = fixed_axis_rotation(
    cloud, axis=[0.0, 0.0, 1.0], angular_velocity=10.0*u.rad/u.s,
    frame=frame, origin=[0.0, 0.0, 0.0]*u.m, angle0=0.0*u.rad, angle_epoch=(100, 0),
    sampling=sampling
)
res = model.at(TargetPoints([[0.0, 0.0, 10.0]]*u.m, ('t1',), frame)).run()
print('potential:', res.potential.value, res.potential.units)
print('acceleration:', res.acceleration.value, res.acceleration.units)
```

Representative output:

```text
potential: [[[-6.64117672e-09 -6.64117672e-09 -6.64117672e-09
   -6.64117672e-09 -6.64117672e-09]]] [[Unit("m2 / s2")]]
acceleration: [[[ 6.57542249e-11  5.77047612e-11  3.55271594e-11
    4.65126987e-12 -2.73634127e-11]
  [ 0.00000000e+00  3.15242547e-11  5.53302724e-11
    6.55895097e-11  5.97901475e-11]
  [-6.57542249e-10 -6.57542249e-10 -6.57542249e-10
   -6.57542249e-10 -6.57542249e-10]]] [[Unit("m / s2") Unit("m / s2") Unit("m / s2")]]
```

## Derivation and sources

### Derivation

The oracle includes formulas, but they are not a derivation proof.

- The oracle converts binary64 inputs to exact rational values and computes the rotation and field with 80-digit mpmath precision.

### References

Existing public documentation and original formulation context:

- [Analytic models](../guide/gen1-analytic-models.md)

**Bibliographic inventory:** located.

- A non-public design note specifies the exact constant-speed or fixed-axis rotating point-mass field. This is local provenance, not an external bibliography entry.

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

An independent oracle-comparison test exists. It was not run for this update.

- A selected case in [`tests/test_rigid_motion_science.py`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/test_rigid_motion_science.py) uses rtol=3e-14. This is test-code evidence only; no result artifact was identified in this review.

### Applicability and limitations

- Requires `PointMassCloud` and `TimeSampling`. `angle_epoch` is independent of the sampling epoch.

**Scope note:** Point-mass clouds only; collisions are rejected during evaluation.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- A dedicated arbitrary point-cloud trajectory illustration and equation renderer.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
