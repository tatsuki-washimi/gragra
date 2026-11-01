<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Fixed-axis rotation

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** rigid motion  
**Geometry:** point_cloud

Rotating discrete mass cloud

**Source-declared observable / representation pairs:**

- Φ a Γ — Time series

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

![Fixed-axis rotation schematic](../assets/models/fixed_axis_rotation.svg)

*Point-mass cloud rotating at constant angular velocity Ω about fixed rotation axis n̂.*

- plot_shape supports a limited set of geometry primitives; arbitrary point-cloud trajectories were not confirmed.

### Equations

**Analytical formulation and field equations:**

- Source mass positions rotate rigidly around a fixed axis defined by origin \(\mathbf x_{\mathrm{orig}}\) and unit axis \(\hat{\mathbf n}\): \(\mathbf x_i(t) = \mathbf x_{\mathrm{orig}} + \mathbf R(\hat{\mathbf n}, \theta(t)) (\mathbf x_{\mathrm{body},i} - \mathbf x_{\mathrm{orig}})\), where \(\theta(t) = \theta_0 + \omega (t - t_{\mathrm{epoch}})\).
- At target \(\mathbf x\) with offset \(\mathbf r_i(t) = \mathbf x - \mathbf x_i(t)\) and distance \(r_i(t) = \|\mathbf r_i(t)\|\), each point mass contributes instantaneous Newtonian field: \(\Phi = -G \sum_i \frac{m_i}{r_i}\), \(\mathbf a = -G \sum_i \frac{m_i \mathbf r_i}{r_i^3}\), \(\mathbf\Gamma = G \sum_i m_i \left[ \frac{3 \mathbf r_i \otimes \mathbf r_i}{r_i^5} - \frac{\mathbf I}{r_i^3} \right]\).
- Evaluated at discrete time series sampling points. Single-source target collisions (\(r_i = 0\)) are strictly rejected.

## Parameters and calculation paths

### Parameters and units

#### API parameters

| Function / Path | Parameter | Type / Format | Default | Description |
|---|---|---|---|---|
| `rotating_point_mass` | `point_mass` | PointMass | required | Point mass object defining mass m |
| `rotating_point_mass` | `radius` | u.Quantity (length) | required | Perpendicular orbital radius r > 0 |
| `rotating_point_mass` | `angular_velocity` | u.Quantity (angular speed) | required | Constant rotation angular velocity omega |
| `rotating_point_mass` | `axis` | array-like (3,) | required (kw-only) | Rotation axis direction vector |
| `rotating_point_mass` | `initial_phase` | u.Quantity (angle) | required (kw-only) | Initial azimuthal phase angle at t = 0 |
| `rotating_point_mass` | `frame` | Frame | required (kw-only) | Reference coordinate frame instance |
| `rotating_point_mass` | `sampling` | TimeSampling | required (kw-only) | Output time axis discretization |

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

- Requires PointMassCloud and TimeSampling. angle_epoch is independent of the sampling epoch.

**Scope note:** Point-mass clouds only; collisions are rejected during evaluation.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- A dedicated arbitrary point-cloud trajectory illustration and equation renderer.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
