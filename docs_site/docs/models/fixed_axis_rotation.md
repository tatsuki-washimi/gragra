<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
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

- plot_shape supports a limited set of geometry primitives; arbitrary point-cloud trajectories were not confirmed.

### Equations

**Existing source formula notes:**

- x(t)=R(axis, angle0+ω(t−angle_epoch))(x_body−origin)+origin。
- For each point mass: Φ=−Gm/r; a=−GmΔ/r³; Γ=Gm(3ΔΔᵀ/r⁵−I/r³). These formulas are transcribed from the oracle implementation and are not a derivation proof.

## Parameters and calculation paths

### Parameters and units

- angular_velocity: rad/s; origin: m vector; angle0: rad; angle_epoch: GPS time; sampling: TimeSampling; cloud masses kg, positions m; frame: Frame.

### Calculation paths

#### Discrete point-mass cloud rotating about a fixed axis at constant angular velocity

- `gragra.analytic.objects.fixed_axis_rotation`

**Path-level source-declared outputs:** Φ a Γ — Time series

Factory APIs construct model/evaluation objects; observables may be obtained after the documented .at(...).run() sequence rather than directly from the factory call. Follow the linked API examples.

## Derivation and sources

### Derivation

The oracle includes formulas, but they are not a derivation proof.

- The oracle converts binary64 inputs to exact rational values and computes the rotation and field with 80-digit mpmath precision.

### References

Existing public documentation and original formulation context:

- [Analytic models](../guide/gen1-analytic-models.md)

**Bibliographic inventory:** located.

- source=(internal evidence record); detail=Gen1 exact constant-speed or fixed-axis rotating point-mass field. This is local provenance, not an external bibliography entry.

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- This documentation generation does not perform scientific re-execution or assign an accuracy Level.

**Existing verification inventory:**

An independent oracle-comparison test exists. It was not run for this update.

- A selected case in the internal evidence record uses rtol=3e-14. This is test-code evidence only; no result artifact was identified in this review.

### Applicability and limitations

- Requires PointMassCloud and TimeSampling. angle_epoch is independent of the sampling epoch.

**Scope note:** Point-mass clouds only; collisions are rejected during evaluation.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- A dedicated arbitrary point-cloud trajectory illustration and equation renderer.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
