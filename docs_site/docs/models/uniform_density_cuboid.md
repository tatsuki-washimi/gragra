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

For an axis-aligned cuboid centered at \(\mathbf c\) with side lengths \((L_x, L_y, L_z)\) and uniform density \(\rho\), let the relative coordinates from observation point \(\mathbf x_t\) to the cuboid bounding faces along each Cartesian axis for \(i, j, k \in \{0, 1\}\) be

$$
\begin{aligned}
u_0 &= c_x - \frac{L_x}{2} - x_t, &\quad u_1 &= c_x + \frac{L_x}{2} - x_t, \\
v_0 &= c_y - \frac{L_y}{2} - y_t, &\quad v_1 &= c_y + \frac{L_y}{2} - y_t, \\
w_0 &= c_z - \frac{L_z}{2} - z_t, &\quad w_1 &= c_z + \frac{L_z}{2} - z_t,
\end{aligned}
$$

with corner distance \(r_{ijk} = \sqrt{u_i^2 + v_j^2 + w_k^2}\).

The gravitational potential in SI units is

$$
\Phi(\mathbf x) = G\rho \sum_{i=0}^1 \sum_{j=0}^1 \sum_{k=0}^1 (-1)^{i+j+k} F_1(u_i, v_j, w_k),
$$

where

$$
F_1(u, v, w) = u v \ln(w+r) + v w \ln(u+r) + w u \ln(v+r) - \frac{u^2}{2}\operatorname{atan}\left(\frac{vw}{ur}\right) - \frac{v^2}{2}\operatorname{atan}\left(\frac{wu}{vr}\right) - \frac{w^2}{2}\operatorname{atan}\left(\frac{uv}{wr}\right).
$$

The gravitational acceleration components are

$$
a_z(\mathbf x) = G\rho \sum_{i,j,k=0}^1 (-1)^{i+j+k} F_2(u_i, v_j, w_k),
\qquad
F_2(u, v, w) = u \ln(v+r) + v \ln(u+r) - w \operatorname{atan}\left(\frac{uv}{wr}\right),
$$

with \(a_x\) and \(a_y\) obtained by cyclic permutation (\(F_2(v, w, u)\) and \(F_2(w, u, v)\)).

The gravity-gradient tensor diagonal components are

$$
\Gamma_{zz}(\mathbf x) = G\rho \sum_{i,j,k=0}^1 (-1)^{i+j+k} \operatorname{atan}\left(\frac{u_i v_j}{w_k r_{ijk}}\right),
$$

with \(\Gamma_{xx}, \Gamma_{yy}\) by cyclic permutation. The off-diagonal components are pairwise face sums:

$$
\begin{aligned}
\Gamma_{xy}(\mathbf x) &= G\rho \sum_{i,j=0}^1 (-1)^{i+j} [\ln(w_1 + r_{ij1}) - \ln(w_0 + r_{ij0})], \\
\Gamma_{xz}(\mathbf x) &= G\rho \sum_{i,k=0}^1 (-1)^{i+k} [\ln(v_1 + r_{i1k}) - \ln(v_0 + r_{i0k})], \\
\Gamma_{yz}(\mathbf x) &= G\rho \sum_{j,k=0}^1 (-1)^{j+k} [\ln(u_1 + r_{1jk}) - \ln(u_0 + r_{0jk})].
\end{aligned}
$$

Interior targets strictly satisfy \(\mathrm{tr}(\mathbf\Gamma) = -4\pi G\rho\), while exterior targets satisfy \(\mathrm{tr}(\mathbf\Gamma) = 0\). Targets lying on cuboid boundary surfaces, edges, or vertices have logarithmic singularities or jump discontinuities and raise a `ValueError`.

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

#### API return values

| Function / Path | Return type | Representation | Quantity & Unit | Shape / Structure | Description |
|---|---|---|---|---|---|
| `cuboid_observables` | GravityObservables | Static | Φ: m²/s², a: m/s², Γ: s⁻² | potential: (M,), acceleration: (M, 3), gradient: (M, 3, 3) | Direct SI observables container with closed-form Nagy 8-vertex antiderivatives |
| `uniform_cuboid` | UniformCuboidModel | Static | Φ: m²/s², a: m/s², Γ: s⁻² | EvaluationResult with Astropy Quantities | High-level model factory; evaluate via .at(targets).run() to yield physical field quantities |

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

### Calculation example and output

The following Python snippet evaluates this model with representative parameters and prints the resulting physical quantities:

```python
from gragra.observables import cuboid_observables

res = cuboid_observables(
    center_m=[0.0, 0.0, 0.0], lengths_m=[2.0, 3.0, 4.0], density_kg_m3=2500.0,
    targets_m=[[0.0, 0.0, 5.0]]
)
print('potential:', res.potential)
print('acceleration:', res.acceleration)
print('gradient:', res.gradient)
```

Representative output:

```text
potential: [-8.25392095e-07]
acceleration: [[ 0.00000000e+00  0.00000000e+00 -1.74413976e-07]]
gradient: [[[-3.87983548e-08  0.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00 -3.61278900e-08  0.00000000e+00]
  [ 0.00000000e+00  0.00000000e+00  7.49262448e-08]]]
```

## Derivation and sources

### Derivation

Closed-form antiderivative derivation documented in reference literature; internal derivation records catalogued as inventory facts, not re-proven here.

- Inventory fact: An archived derivation record exists in internal records documenting the 8-vertex antiderivative summation.
- Literature assertion: Field expressions match the Nagy (1966, 2000) and Petrović (1996) analytical antiderivatives.
- Proof assertion: Boundary jump conditions and Poisson traces (interior -4πGρ, exterior 0) are verified in [`tests/test_analytic_cuboid.py`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/test_analytic_cuboid.py), but are not re-derived here.

### References

Existing public documentation and original formulation context:

- [Analytic benchmarks](../guide/analytic-benchmarks.md)
- [analytic models](../guide/gen1-analytic-models.md)

**Bibliographic inventory:** located.

- Nagy, D. (1966), ‘The gravitational attraction of a right rectangular prism’, Geophysics, 31(2), 362–371, DOI 10.1190/1.1439779.
- Nagy, D., Papp, G., & Benedek, J. (2000), ‘The gravitational potential and its derivatives for the prism’, Journal of Geodesy, 74(7-8), 552–560, DOI 10.1007/s001900000116 (Erratum 2002, 76(8), 475).
- Petrović, S. (1996), ‘Determination of the gravity potential and its first and second derivatives for a right rectangular prism’, Journal of Geodesy, 70(9), 562–571, DOI 10.1007/BF00868228.
- [DOI 10.1190/1.1439779](https://doi.org/10.1190/1.1439779)
- [DOI 10.1007/s001900000116](https://doi.org/10.1007/s001900000116)
- [DOI 10.1007/BF00868228](https://doi.org/10.1007/BF00868228)

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 3.
- Catalogue probe state: recorded (a representative API output contract, not an accuracy test).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

Test implementation and documentary claims found; no checks were run for this inventory.

- [`tests/test_analytic_cuboid.py`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/test_analytic_cuboid.py) includes derivative consistency, symmetry, Poisson trace, boundary rejection, independent interior/exterior quadrature, SI wrapper. Report states boundary tolerance atol=1e-12*max(`half_lengths`), interior trace -4πGρ, exterior trace 0. formula ledger records unrelated historical run metadata; revision-matched raw output path for assigned cuboid probe not established. `cuboid_si` contract is representative probe only.

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
