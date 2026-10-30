<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Fixed-axis rotation

**Level:** — (definition and assignment pending).

**Physical domain:** rigid motion  
**Geometry:** point_cloud

Rotating discrete mass cloud

**Source-declared observable / representation pairs:**

- Φ a Γ — Time series

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry

- plot_shape は限られた geometry primitive を扱い、任意点群軌跡表示は確認できず。

## Equations

**Existing source formula notes:**

- x(t)=R(axis, angle0+ω(t−angle_epoch))(x_body−origin)+origin。
- 各質点: Φ=−Gm/r; a=−GmΔ/r³; Γ=Gm(3ΔΔᵀ/r⁵−I/r³)。これは oracle 実装式の転記で導出の証明ではない。

## Parameters and units

- angular_velocity: rad/s; origin: m vector; angle0: rad; angle_epoch: GPS time; sampling: TimeSampling; cloud masses kg, positions m; frame: Frame.

## APIs and calculation paths

### 固定軸・一定角速度の離散質量クラウド

- `gragra.analytic.objects.fixed_axis_rotation`

**Path-level source-declared outputs:** Φ a Γ — Time series

Factory APIs construct model/evaluation objects; observables may be obtained after the documented .at(...).run() sequence rather than directly from the factory call. Follow the linked API examples.

## Derivation and provenance

**Inventory status:** partial.

oracle に式はあるが導出証明ではない。

The page cites public sources below. A fully typeset public derivation is a separate editorial item wherever the existing source material has not been reproduced here.

## References

Existing public documentation and original formulation context:

- [Analytic models](../guide/gen1-analytic-models.md)

**Bibliographic inventory:** located.

- source=(internal evidence record); detail=Gen1 exact constant-speed or fixed-axis rotating point-mass field. This is local provenance, not an external bibliography entry.

## Verification evidence

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- This documentation generation does not perform scientific re-execution or assign an accuracy Level.

**Existing verification inventory:**

独立 oracle 比較テストがある。今回未実行。

- (internal evidence record) の選択例は rtol=3e-14。test-code-only; この調査では実行結果 artifact を特定できず。

## Applicability and limitations

- PointMassCloud と TimeSampling が必要。angle_epoch は sampling epoch と独立。

**Scope note:** cloud only、評価時衝突拒否。

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)

## Information still to locate or publish

- 任意点群軌跡の専用図、式 renderer。

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
