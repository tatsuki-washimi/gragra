<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Uniform-density cuboid

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** static gravity  
**Geometry:** cuboid

Finite rectangular prism; static Newton field

**Source-declared observable / representation pairs:**

- Φ a Γ — Static

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

![Uniform-density cuboid, 3D schematic](../assets/models/uniform_density_cuboid.svg)

*Axis-aligned cuboid with center c, dimensions (Lx, Ly, Lz), uniform density ρ, and evaluation target x.*

- high-level Cuboid holds frame/lengths. Vertex antiderivative domain interior/exterior away from material boundary. No arbitrary mesh reuse implied.

### Equations

**Analytical formulation and field equations:**

- For an axis-aligned cuboid centered at \(\mathbf c\) with side lengths \((L_x, L_y, L_z)\) and uniform density \(\rho\), relative corner coordinates are \((u_i, v_j, w_k) = (x_t - c_x \mp L_x/2, y_t - c_y \mp L_y/2, z_t - c_z \mp L_z/2)\) with corner distance \(r_{ijk} = \sqrt{u_i^2 + v_j^2 + w_k^2}\).
- Potential: \(\Phi(\mathbf x) = G\rho \sum_{i,j,k=0}^1 (-1)^{i+j+k} F_1(u_i, v_j, w_k)\), where \(F_1(u, v, w) = u v \ln(w+r) + v w \ln(u+r) + w u \ln(v+r) - \frac{u^2}{2}\operatorname{atan}\left(\frac{vw}{ur}\right) - \frac{v^2}{2}\operatorname{atan}\left(\frac{wu}{vr}\right) - \frac{w^2}{2}\operatorname{atan}\left(\frac{uv}{wr}\right)\) (Nagy et al. 2000, Petrovic 1996).
- Acceleration: \(a_z(\mathbf x) = G\rho \sum_{i,j,k=0}^1 (-1)^{i+j+k} F_2(u_i, v_j, w_k)\) where \(F_2(u, v, w) = u \ln(v+r) + v \ln(u+r) - w \operatorname{atan}\left(\frac{uv}{wr}\right)\). The \(a_x\) and \(a_y\) components follow by cyclic permutation (\(F_2(v, w, u)\) and \(F_2(w, u, v)\)).
- Gravity gradient tensor: diagonal components are \(\Gamma_{zz}(\mathbf x) = G\rho \sum_{i,j,k=0}^1 (-1)^{i+j+k} \operatorname{atan}\left(\frac{u_i v_j}{w_k r_{ijk}}\right)\) with \(\Gamma_{xx}, \Gamma_{yy}\) by cyclic permutation. Off-diagonal components are pairwise sums \(\Gamma_{xy}(\mathbf x) = -G\rho \sum_{i,j=0}^1 (-1)^{i+j} [\ln(w_1 + r_{ij1}) - \ln(w_0 + r_{ij0})]\), \(\Gamma_{xz}(\mathbf x) = -G\rho \sum_{i,k=0}^1 (-1)^{i+k} [\ln(v_1 + r_{i1k}) - \ln(v_0 + r_{i0k})]\), and \(\Gamma_{yz}(\mathbf x) = -G\rho \sum_{j,k=0}^1 (-1)^{j+k} [\ln(u_1 + r_{1jk}) - \ln(u_0 + r_{0jk})]\).
- Interior points strictly satisfy \(\mathrm{tr}(\mathbf\Gamma) = -4\pi G\rho\), and exterior points satisfy \(\mathrm{tr}(\mathbf\Gamma) = 0\). Targets on material boundaries (faces, edges, vertices) have logarithmic singularities or jump discontinuities and raise ValueError.

## Parameters and calculation paths

### Parameters and units

#### API parameters

| Function / Path | Parameter | Type / Format | Default | Description |
|---|---|---|---|---|
| `cuboid_observables` | `center_m` | array-like (3,) | required | Cuboid center coordinates in meters |
| `cuboid_observables` | `lengths_m` | array-like (3,) | required | Full side lengths (Lx, Ly, Lz) in meters (all positive) |
| `cuboid_observables` | `density_kg_m3` | float | required | Uniform mass density in kg/m³ (positive) |
| `cuboid_observables` | `targets_m` | array-like (M, 3) | required | Evaluation target coordinates in meters |
| `uniform_cuboid` | `geometry` | Cuboid | required | Astropy Cuboid geometry (center and full lengths) |
| `uniform_cuboid` | `density` | u.Quantity (density) | required | Astropy Quantity uniform mass density |

#### Mathematical symbols and notation

| Symbol | Meaning | SI unit | Definition / Relation |
|---|---|---|---|
| \(\mathbf c\) | Cuboid center | m | Center of rectangular prism |
| \(\mathbf L\) | Side lengths | m | Full prism edge lengths \((L_x, L_y, L_z)\), all positive |
| \(\rho\) | Mass density | kg/m³ | Uniform interior mass density |
| \(\mathbf x\) | Evaluation target | m | Observation target coordinates |
| \(u_i, v_j, w_k\) | Corner relative coordinates | m | Offsets \(x_m - c_m \mp L_m/2\) for \(i,j,k \in \{0, 1\}\) |
| \(r_{ijk}\) | Corner distance | m | \(r_{ijk} = \sqrt{u_i^2 + v_j^2 + w_k^2}\) |
| \(F_1(u, v, w)\) | Potential antiderivative | m² | Nagy et al. (2000) closed-form kernel for \(1/R\) |
| \(F_2(u, v, w)\) | Acceleration antiderivative | m | Nagy (1966) closed-form kernel for \(-\nabla(1/R)\) |

#### Usage notes

- High-level factory `uniform_cuboid(geometry, density)` constructs a `UniformCuboidModel` evaluated via `.at(targets).run()`.
- Boundary handling: points on cuboid faces, edges, or vertices raise `ValueError` because the gravity gradient tensor is discontinuous or singular there.

### Calculation paths

#### axis-aligned cuboid closed form

- `gragra.analytic.objects.uniform_cuboid`
- `gragra.observables.cuboid_observables`

**Observables after model evaluation:** Φ a Γ — Static

High-level model factories create evaluation objects; call the documented `.at(...).run()` workflow to evaluate them. Observable functions listed above are direct calls with their own return values.

## Derivation and sources

### Derivation

Committed independent derivation document exists; its text is inventoried as a derivation record, not re-proven here.

- A non-public validation note includes the source-formula discussion and implementation conclusions; its contents are not reproduced in this catalogue.

### References

Existing public documentation and original formulation context:

- [Analytic benchmarks](../guide/analytic-benchmarks.md)
- [analytic models](../guide/gen1-analytic-models.md)

**Bibliographic inventory:** located.

- Nagy, D. (1966), ‘The gravitational attraction of a right rectangular prism.’ Nagy, D., Papp, G., & Benedek, J. (2000), ‘The gravitational potential and its ...’; 2002 erratum. Petrović (1996) is named in analytic benchmark/report, but complete record not transcribed here.

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 3.
- Catalogue probe state: recorded (a representative API output contract, not an accuracy test).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

Test implementation and documentary claims found; no checks were run for this inventory.

- [`tests/test_analytic_cuboid.py`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/test_analytic_cuboid.py) includes derivative consistency, symmetry, Poisson trace, boundary rejection, independent interior/exterior quadrature, SI wrapper. Report states boundary tolerance atol=1e-12*max(half_lengths), interior trace -4πGρ, exterior trace 0. formula ledger records unrelated historical run metadata; revision-matched raw output path for assigned cuboid probe not established. cuboid_si contract is representative probe only.

#### Registered representative output probes

| API | Observable | Representation | Probe |
|---|---|---|---|
| `gragra.observables.cuboid_observables` | Φ | Static | `cuboid_si` |
| `gragra.observables.cuboid_observables` | a | Static | `cuboid_si` |
| `gragra.observables.cuboid_observables` | Γ | Static | `cuboid_si` |

### Applicability and limitations

- interior tr Γ=-4πGρ; exterior tr Γ=0. Numerical tolerance does not define boundary values.

**Scope note:** Uniform-density, rectangular prism only; points on faces/edges/vertices rejected due discontinuity/log singularity.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- Catalog probe not executed; no numeric output recorded here.
- Petrović full bibliographic record not transcribed.
- Source derivation/tests do not establish a rerun at this task revision.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
