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

| Symbol | Parameter | Meaning | SI unit |
|---|---|---|---|
| \(m_i\) | `masses_kg` | Discrete point masses (all positive) | kg |
| \(\mathbf x_{0,i}\) | `initial_positions_m` | Initial positions at t = t_0 | m |
| \(\mathbf v_i\) | `velocities_m_s` | Constant velocity vectors | m/s |
| \(t_0\) | `t0_s` | Initial reference time | s |
| \(t\) | `times_s` | Evaluation time points | s |
| \(\mathbf x\) | `targets_m` | Sensor evaluation coordinates | m |

**Additional notes on parameters and models:**

- mass: array-like (N,), positive point masses in kg; position: array-like (N, 3), initial coordinates in meters; velocity: array-like (N, 3), velocities in m/s.
- motion_epoch: GPS time of initial position definition; targets: array-like (M, 3), evaluation coordinates in meters; time_sampling: TimeSampling object defining timestamps.

### Calculation paths

#### implemented model/API path

- `gragra.analytic.objects.constant_velocity`

**Observables after model evaluation:** Φ a Γ — Time series

High-level model factories create evaluation objects; call the documented `.at(...).run()` workflow to evaluate them. Observable functions listed above are direct calls with their own return values.

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
