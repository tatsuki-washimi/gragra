<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Uniform-density sphere

**Level:** — (definition and assignment pending).

**Physical domain:** static gravity  
**Geometry:** sphere

Finite sphere; static potential, acceleration and gravity gradient, inside or outside

**Source-declared observable / representation pairs:**

- Φ a Γ — Static

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry

The finite uniform-density sphere has center \(\mathbf c\), radius \(R>0\), density \(\rho\), and target position \(\mathbf x\).

![Uniform-density sphere, 3D schematic](../assets/models/sphere-3d.svg)

![Uniform-density sphere, radial cross-section schematic](../assets/models/sphere-section.svg)

*The figures are schematic, not sampled or measured scientific data.*

- Sphere shapeは幾何データを供給し、解析モデルが密度を結合。任意のmesh/shapeへの一般化はこのAPIからは示されない。

## Equations

Let \(\mathbf r=\mathbf x-\mathbf c\), \(r=|\mathbf r|\),
\(M=4\pi\rho R^3/3\). The physical SI results are

$$
\Phi(r)=
\begin{cases}
-\dfrac{GM}{2R^3}(3R^2-r^2), & r<R,\\
-\dfrac{GM}{r}, & r\ge R,
\end{cases}
$$

$$
\mathbf a(\mathbf r)=
\begin{cases}
-\dfrac{GM}{R^3}\mathbf r, & r<R,\\
-\dfrac{GM}{r^3}\mathbf r, & r\ge R,
\end{cases}
$$

$$
\Gamma_{ij}=
\begin{cases}
-\dfrac{GM}{R^3}\delta_{ij}, & r<R,\\
GM\left(\dfrac{3r_ir_j}{r^5}-\dfrac{\delta_{ij}}{r^3}\right), & r\ge R.
\end{cases}
$$

The implementation chooses the **exterior gradient expression at \(r=R\)**.
Potential and acceleration are continuous there; the tensor requires
one-sided limits because the density is discontinuous.

## Parameters and units

| Parameter | Meaning | SI unit |
|---|---|---|
| center_m / \(\mathbf c\) | Sphere center | m |
| radius_m / \(R\) | Positive radius | m |
| density_kg_m3 / \(\rho\) | Uniform density | kg/m³ |
| targets_m / \(\mathbf x\) | Evaluation targets, shape (N, 3) | m |

The high-level factory instead accepts a Sphere geometry and an Astropy Quantity density. G-free helpers and SI wrappers have different return types.

## APIs and calculation paths

### G-free closed form / SI static wrapper

- `gragra.observables.uniform_sphere_observables`

**Path-level source-declared outputs:** Φ a Γ — Static

### Quantity model EvaluationSetup

- `gragra.analytic.objects.uniform_sphere`

**Path-level source-declared outputs:** Φ a Γ — Static

Factory APIs construct model/evaluation objects; observables may be obtained after the documented .at(...).run() sequence rather than directly from the factory call. Follow the linked API examples.

## Derivation and provenance

The spherical symmetry solution follows from \(\nabla^2\Phi=4\pi G\rho\) inside the source and \(\nabla^2\Phi=0\) outside. Finite potential at the center, \(\Phi\to0\) at infinity, and matching \(\Phi\) and radial acceleration at \(r=R\) determine the constants. Differentiating gives \(\mathbf a=-\nabla\Phi\) and \(\Gamma_{ij}=\partial a_i/\partial x_j\).

This is a standard mathematical outline; a separately archived independent sphere derivation was not located in the inventory.

## References

Existing public documentation and original formulation context:

- [Analytic benchmarks](../guide/analytic-benchmarks.md)
- [sphere example](../examples/uniform_sphere_workflow.md)

**Bibliographic inventory:** partial.

- この inventory範囲の該当箇所で球場固有のbibliographic recordは見つけていない。

## Verification evidence

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 3.
- Catalogue probe state: recorded (a representative API output contract, not an accuracy test).
- This documentation generation does not perform scientific re-execution or assign an accuracy Level.

**Existing verification inventory:**

test実装、文書上の検証主張、保存済みrun記録が別々に存在。今回実行なし。

- (internal evidence record)は球の内外potential/acceleration比較とcomplex densityを実装。(internal evidence record)は外部点でpotential/acceleration/gradientの一部を実装。docsは既存sphere convergenceを一bounded caseとし、最細分割の相対誤差と粗→細potential改善のみを要求、fitted slopeや各段階単調性は主張しない。保存結果はformula-ledgerが記す旧SHA向け検証一般記録で、当該球の個別結果を本revisionで再現した記録として扱わない。catalogue sphere_siは代表output probeである。

## Applicability and limitations

- 球の対象領域は内外。discretization convergence evidenceは限定された単一条件。

**Scope note:** 均一密度・有限球の理想化であり、解析式の前提外に精度保証なし。

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)

## Information still to locate or publish

- カタログの代表probe値は出力契約であり、数値を実行していない。
- sphere固有のbibliographic recordおよび独立導出保存先は未特定。
- 既存(internal evidence record) claimと本revisionでの実行済み結果は異なる証拠区分。

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
