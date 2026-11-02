<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Oriented cuboid

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** static gravity  
**Geometry:** `cuboid`

Rotated finite prism and target-frame response

**Source-declared observable / representation pairs:**

- Φ a Γ — Static

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

![Oriented cuboid, coordinate transformation schematic](../assets/models/oriented_cuboid.svg)

*World frame (X, Y, Z) and rotated cuboid frame (x', y', z') related by rotation matrix R.*

- API receives `center_m`, `lengths_m`, `targets_m` and `rotation_matrix` directly; it is not a Cuboid geometry object factory. The separate plotting helper takes a direct geometry/model/setup and resolves only Cuboid, Cylinder, SphericalShell; it can transform geometry coordinates through its Frame to `display_frame`.
- Oriented API proper Q and inherited base cuboid boundary rejection.

### Equations

For an arbitrarily oriented cuboid centered at \(\mathbf c\) with rotation matrix \(\mathbf Q \in \mathrm{SO}(3)\) mapping coordinates from the cuboid body frame to the world frame:

The target coordinates in the body frame are:

$$
\mathbf x_{\mathrm{body}} = \mathbf Q^T (\mathbf x - \mathbf c)
$$

The gravitational observables transform covariantly from the body-frame field:

$$
\begin{aligned}
\Phi(\mathbf x) &= \Phi_{\mathrm{body}}(\mathbf x_{\mathrm{body}}), \\
\mathbf a(\mathbf x) &= \mathbf Q \mathbf a_{\mathrm{body}}(\mathbf x_{\mathrm{body}}), \\
\mathbf\Gamma(\mathbf x) &= \mathbf Q \mathbf\Gamma_{\mathrm{body}}(\mathbf x_{\mathrm{body}}) \mathbf Q^T.
\end{aligned}
$$

Exterior targets satisfy the Laplace trace condition \(\mathrm{tr}(\mathbf\Gamma) = 0\), while interior targets satisfy the Poisson equation \(\mathrm{tr}(\mathbf\Gamma) = -4\pi G\rho\). Boundary points on cuboid faces, edges, or vertices are strictly rejected.

## Parameters and calculation paths

### Parameters and units

#### API parameters

| Function / Path | Parameter | Type / Format | Default | Description |
|---|---|---|---|---|
| `oriented_cuboid_observables` | `center_m` | array-like (3,) | required | Cuboid center coordinates in world frame (meters) |
| `oriented_cuboid_observables` | `lengths_m` | array-like (3,) | required | Full side lengths in local body frame (meters) |
| `oriented_cuboid_observables` | `density_kg_m3` | float | required | Uniform mass density in kg/m³ |
| `oriented_cuboid_observables` | `targets_m` | array-like (M, 3) | required | Evaluation target coordinates in world frame (meters) |
| `oriented_cuboid_observables` | `rotation_matrix` | array-like (3, 3) | required (kw-only) | Proper rotation matrix Q in SO(3) mapping local body to world frame |

#### API return values

| Function / Path | Return type | Representation | Quantity & Unit | Shape / Structure | Description |
|---|---|---|---|---|---|
| `oriented_cuboid_observables` | GravityObservables | Static | Φ: m²/s², a: m/s², Γ: s⁻² | potential: (M,), acceleration: (M, 3), gradient: (M, 3, 3) | Direct SI observables transformed to world coordinates via proper rotation matrix Q |

#### Mathematical symbols and notation

| Symbol | Meaning | SI unit | Definition / Relation |
|---|---|---|---|
| \(\mathbf c\) | Cuboid center | m | Center in world coordinate frame |
| \(\mathbf L\) | Body side lengths | m | Edge lengths in local principal axis frame |
| \(\rho\) | Mass density | kg/m³ | Uniform interior mass density |
| \(\mathbf Q\) | Rotation matrix | dimensionless | Proper orthogonal matrix \(\mathbf Q \in \mathrm{SO}(3)\) (\(\det \mathbf Q = +1, \mathbf Q^T \mathbf Q = \mathbf I\)) |
| \(\mathbf x\) | World target position | m | Transformed to body frame via \(\mathbf x_{\mathrm{local}} = \mathbf Q^T (\mathbf x - \mathbf c)\) |

#### Usage notes

- The `rotation_matrix` argument must satisfy proper rotation conditions (\(\det \mathbf Q = +1\) and \(\mathbf Q^T \mathbf Q = \mathbf I\)). Improper rotations or reflections raise `ValueError`.
- Observables are transformed back to world frame: \(\Phi = \Phi_{\mathrm{local}}\), \(\mathbf a = \mathbf Q \mathbf a_{\mathrm{local}}\), \(\mathbf \Gamma = \mathbf Q \mathbf \Gamma_{\mathrm{local}} \mathbf Q^T\).

### Calculation paths

#### rotated cuboid G-free/SI transport

- `gragra.observables.oriented_cuboid_observables`

**Path-level source-declared output union:** Φ a Γ — Static

### Calculation example and output

The following Python snippet evaluates this model with representative parameters and prints the resulting physical quantities:

```python
import numpy as np
from gragra.observables import oriented_cuboid_observables

# 90-degree rotation around z-axis
Q = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
res = oriented_cuboid_observables(
    [0.0, 0.0, 0.0], [2.0, 3.0, 4.0], 2500.0, [[0.0, 0.0, 5.0]],
    rotation_matrix=Q
)
print('potential:', res.potential)
print('acceleration:', res.acceleration)
print('gradient:', res.gradient)
```

Representative output:

```text
potential: [-8.25392095e-07]
acceleration: [[ 0.00000000e+00  0.00000000e+00 -1.74413976e-07]]
gradient: [[[-3.61278900e-08  0.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00 -3.87983548e-08  0.00000000e+00]
  [ 0.00000000e+00  0.00000000e+00  7.49262448e-08]]]
```

## Derivation and sources

### Derivation

Inherits the aligned-cuboid vertex-antiderivative derivation; the oriented wrapper applies a local-to-world coordinate transformation.

- cuboid derivation document applies to same closed form. oriented wrapper computes `local_targets = (targets - center) @ Q`, evaluates `cuboid_observables_g_free`, then covariant vector/tensor transport; this is coordinate transformation, not a new physical derivation.

### References

Existing public documentation and original formulation context:

- [Physics API](../api/physics.md)
- [coordinate conventions](../concepts/conventions.md)

**Bibliographic inventory:** located.

- The foundational closed form is documented by Nagy (1966), Nagy, Papp & Benedek (2000) and its 2002 erratum, and Blakely (1995). The derivation document does not list DOIs.

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

Tests and acceptance claims are recorded separately; no rerun in this task.

- [`tests/test_analytic_oriented.py`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/test_analytic_oriented.py) covers identity, translation, rotation, SI conversion. Ledger states world-coordinate Gauss reference tolerances potential/acceleration 1e-10, gradient 1e-9, atol 1e-10; integration-int63/ot-focused.meta is recorded result path in external evidence tree but not checked/reproduced here.

### Applicability and limitations

- No general non-cuboid geometry or arbitrary affine scaling is claimed.

**Scope note:** Rotation must be proper; local cuboid boundary restrictions continue to apply.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- Catalogue has no `output_contracts`; no representative catalogue probe is defined.
- Sphere geometry is not among `plot_shape`-supported types; direct `oriented_cuboid_observables` arrays are not themselves a geometry/model/setup argument. Reuse requires a separate Cuboid geometry object, with its own Frame transform.
- `formula_to_latex` applicability is limited to its registered expressions; no claim for arbitrary oriented equations.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
