<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Prescribed point-mass translation

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** rigid motion  
**Geometry:** point

Constant-velocity mass; time-sampled field

**Source-declared observable / representation pairs:**

- Φ a Γ — Time series

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

![Prescribed point-mass kinematics schematic](../assets/models/prescribed_point_mass_translation.svg)

*Discrete point masses m_i moving with individual velocities v_i from initial positions x_{0,i} and their Newtonian field at target x.*

- Exact prescribed trajectory, not a dynamical motion solver; distinct from linearized phasor model.

### Equations

**Analytical formulation and field equations:**

- For point masses \(m_i\) with positions \(\mathbf x_i(t) = \mathbf x_{i,0} + \mathbf v_i (t - t_0)\), the instantaneous Newtonian field at target \(\mathbf x\) has offset vector \(\mathbf r_i(t) = \mathbf x - \mathbf x_i(t)\) and distance \(r_i(t) = \|\mathbf r_i(t)\|\).
- Potential: \(\Phi(\mathbf x, t) = -G \sum_i \frac{m_i}{r_i(t)}\); acceleration: \(\mathbf a(\mathbf x, t) = -G \sum_i \frac{m_i \mathbf r_i(t)}{r_i(t)^3}\); gravity-gradient tensor: \(\mathbf\Gamma(\mathbf x, t) = G \sum_i m_i \left[ \frac{3 \mathbf r_i(t) \otimes \mathbf r_i(t)}{r_i(t)^5} - \frac{\mathbf I}{r_i(t)^3} \right]\).
- Outputs are evaluated as discrete sampled time series. Source collision (\(r_i(t) = 0\)) is rejected.

## Parameters and calculation paths

### Parameters and units

#### API parameters

| Function / Path | Parameter | Type / Format | Default | Description |
|---|---|---|---|---|
| `constant_velocity` | `point_mass` | PointMass | required | Point mass object containing mass and initial position |
| `constant_velocity` | `velocity` | u.Quantity (speed 3-vector) | required | Constant velocity vector v in frame |
| `constant_velocity` | `frame` | Frame | required (kw-only) | Reference coordinate frame instance |
| `constant_velocity` | `motion_epoch` | object | required (kw-only) | Reference epoch t0 at which position equals initial position |
| `constant_velocity` | `sampling` | TimeSampling | required (kw-only) | Evaluation time axis discretization |

#### API return values

| Function / Path | Return type | Representation | Quantity & Unit | Shape / Structure | Description |
|---|---|---|---|---|---|
| `constant_velocity` | _MotionModel | TimeSeries | Φ: m²/s², a: m/s², Γ: s⁻² | EvaluationResult with TimeSeriesMatrix payloads | Constant-velocity point-mass motion model evaluated via .at(targets).run() |

#### Mathematical symbols and notation

| Symbol | Meaning | SI unit | Definition / Relation |
|---|---|---|---|
| \(m\) | Point mass | kg | Discrete mass magnitude (\(m > 0\)) |
| \(\mathbf x_0\) | Initial position | m | Body location at reference epoch \(t = t_0\) |
| \(\mathbf v\) | Velocity vector | m/s | Constant translation velocity |
| \(t_0\) | Motion epoch | s | Reference time for initial coordinates |
| \(t\) | Evaluation time | s | Evaluation time sample points |
| \(\mathbf x(t)\) | Trajectory position | m | \(\mathbf x(t) = \mathbf x_0 + \mathbf v (t - t_0)\) |
| \(\mathbf x_{\mathrm{target}}\) | Target position | m | Sensor evaluation coordinates |
| \(r(t)\) | Target distance | m | \(r(t) = \|\mathbf x_{\mathrm{target}} - \mathbf x(t)\|\) |

#### Usage notes

- Evaluates via `.at(targets).run()` to yield discrete time-series for potential, acceleration, and gradient.
- Collision check: if the trajectory intersects any target point (\(r(t) = 0\)), evaluation raises `ValueError`.

### Calculation paths

#### implemented model/API path

- `gragra.analytic.objects.constant_velocity`

**Observables after model evaluation:** Φ a Γ — Time series

High-level model factories create evaluation objects; call the documented `.at(...).run()` workflow to evaluate them. Observable functions listed above are direct calls with their own return values.

### Calculation example and output

The following Python snippet evaluates this model with representative parameters and prints the resulting physical quantities:

```python
import astropy.units as u
from gragra.geometry import Frame
from gragra.gravity_sources import PointMass
from gragra.results import TimeSampling
from gragra.targets import TargetPoints
from gragra.analytic.objects import constant_velocity

frame = Frame('world')
pm = PointMass(1000.0, [0.0, 0.0, 0.0])
sampling = TimeSampling(epoch=(100, 0), dt=0.1*u.s, count=5)
model = constant_velocity(
    pm, [10.0, 0.0, 0.0]*u.m/u.s, frame=frame, motion_epoch=(100, 0), sampling=sampling
)
res = model.at(TargetPoints([[0.0, 50.0, 0.0]]*u.m, ('t1',), frame)).run()
print('potential:', res.potential.value, res.potential.units)
print('acceleration:', res.acceleration.value, res.acceleration.units)
```

Representative output:

```text
potential: [[[-1.33486000e-09 -1.33459311e-09 -1.33379339e-09
   -1.33246372e-09 -1.33060884e-09]]] [[Unit("m2 / s2")]]
acceleration: [[[ 0.00000000e+00  5.33623794e-13  1.06533019e-12
    1.59322087e-12  2.11543536e-12]
  [-2.66972000e-11 -2.66811897e-11 -2.66332546e-11
   -2.65536811e-11 -2.64429420e-11]
  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00
    0.00000000e+00  0.00000000e+00]]] [[Unit("m / s2") Unit("m / s2") Unit("m / s2")]]
```

## Derivation and sources

### Derivation

Formula expressions are declared in source; a separate local derivation proof was not located in the focused scan.

- No separate derivation is asserted from the formula declaration alone.

### References

Existing public documentation and original formulation context:

- [Analytic models](../guide/gen1-analytic-models.md)

**Bibliographic inventory:** partial.

- Shared motion implementation reference metadata; local formulas in source.

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

Verification method, tolerances, and artifact status recorded; tests were not run in this task.

- [`tests/test_rigid_motion_science.py`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/test_rigid_motion_science.py) uses independent mpmath transform oracle/frozen fixtures; key comparison 3e-14 tolerances; test code only, no saved result.

### Applicability and limitations

- Exact prescribed trajectory, not a dynamical motion solver; distinct from linearized phasor model.

**Scope note:** Applicability limits recorded.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- Catalogue output_contracts are empty; this is a missing representative probe, not evidence of absent science.
- No additional committed plot asset identified in these assigned API sources.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
