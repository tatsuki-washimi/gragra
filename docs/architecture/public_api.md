# gragra API Inventory

This document lists and classifies exposed symbols in the `gragra` library. The
categories describe how users normally reach an interface; they are not a
publication or approval process:
- **Top-level API** (`gragra.*`): Core physics and research-calculation APIs.
- **Subpackage API** (`gragra.subpackage.*`): Component-level APIs for users needing finer control.
- **Additional API**: Developing features and solver adapters that may change.
- **Internal**: Undocumented helper modules/functions meant only for internal library use.

The current package-export snapshot and the Generation 1 static-source
old/new usage map are maintained in the
[public export inventory](../design/public_export_inventory_2026-10-01.md).
The counts there include only non-private names listed in package
`__init__.py` `__all__` values; module-only names are outside that count.

---

## 1. Top-Level API

The following symbols are re-exported by `gragra` for the current research
interface:

| Symbol | Source Module | Category | Description |
|---|---|---|---|
| `G_SI` | `gragra.constants` | Physics Constants | CODATA Newton's gravitational constant ($6.67430 \times 10^{-11} \, \text{m}^3 \, \text{kg}^{-1} \, \text{s}^{-2}$). |
| `WeightedPointSource` | `gragra.gravity_sources.weighted_points` | Source Type | Canonical point source container supporting signed/complex weights. |
| `PointMass` | `gragra.gravity_sources.weighted_points` | Source Type | Convenience wrapper for a single point mass (real, positive mass). |
| `PointMassCloud` | `gragra.gravity_sources.weighted_points` | Source Type | Convenience wrapper for multiple point masses (real, positive masses). |
| `TargetPoints` | `gragra.targets` | Target Type | Quantity-aware target coordinates, identifiers, and `Frame`; numeric calls receive source-frame SI coordinates through `.at(...)`. |
| `point_potential` | `gragra.observables.potential` | Observable | Computes Newtonian potential $V$ at target points. |
| `point_acceleration` | `gragra.observables.acceleration` | Observable | Computes vector gravitational acceleration $\vec{a}$ at targets. |
| `point_gravity_gradient` | `gragra.observables.gradient` | Observable | Computes the Gravity Gradient Tensor $T_{ij}$. |
| `coherent_point_acceleration` | `gragra.observables.coherent` | Observable | Computes coherent point-source acceleration. |
| `coherent_point_gravity_gradient` | `gragra.observables.coherent` | Observable | Computes the coherent gravity-gradient response. |
| `coherent_gravitoelastic_tensor` | `gragra.observables.coherent` | Observable | Computes a coherent gravitoelastic response tensor. |
| `coherent_acceleration_csd` | `gragra.observables.coherent` | Observable | Computes the coherent acceleration cross-spectral density. |
| `displacement_transfer_acceleration` | `gragra.observables.dipole` | Observable | Computes displacement-transfer acceleration. |
| `displacement_field_acceleration` | `gragra.observables.displacement` | Observable | Computes acceleration from a displacement field. |

The interface is developed together with the numerical implementations and
their analytic and regression tests. Import paths shown here are the current
ones; scientific behavior and constraints are documented beside each module.
The cross-cutting Generation 1 rules and route-by-route evidence are indexed
in the [common contract](common_contract.md) and its
[conformance matrix](../validation/common_contract_matrix.md). The matrix
separates contract conformance, connection to `EvaluationResult`, and tests
run for its recorded revision.

`gragra.results` exports the high-level `EvaluationResult`, `compare`,
`GravityResponse`, and `ResponseContract` alongside the `GravityResult`
envelope, C3-v1 descriptors, and five C5-A context names. These names remain
within the subpackage. C2 adds nine field-definition names, C4 adds two
response names, and C5-A adds five context names, giving `gragra.results` 28
public exports. `EvaluationContext` owns normalized targets, the resolved
frame, requested observables, representation axes, field definition, and
interaction for one evaluation. `EvaluationResult` exposes that shared context
through `result.context`; `result.evaluation` is the `FieldEvaluation` returned
by `.at()`. Comparison operates on this common result contract and does not
need model-specific branches. A `GravityFieldDefinition` connects typed field
meaning to result payloads, and one evaluation shares the same definition
object across its model spec and context-derived bindings. A `GravityResponse`
owns precomputed per-frequency coefficients for one input and applies them to
an explicit phasor without re-running the Rayleigh observable. The C2 contract and comparison projection are documented in
[`c2_field_definition_contract.md`](../design/c2_field_definition_contract.md).
The spectral descriptors record representation conventions without holding
payloads or performing spectral calculations. `GravityResult` retains its
existing behavior: its payload bindings are frozen, payload objects may
remain mutable, and existing observable return types are unchanged.

---

## 2. Subpackage APIs

These symbols are exposed within their respective subpackages for detailed configuration or customization.

### `gragra.gravity_sources`

| Symbol | Category | Description |
|---|---|---|
| `WeightedPointSource` | Source Type | Canonical source representation. |
| `PointMass` | Source Type | Wrapper for a single positive point mass. |
| `PointMassCloud` | Source Type | Wrapper for multiple positive point masses. |
| `VolumeElementSource` | Source Type | Represents continuous mass densities discretized as volume elements. |
| `SurfaceMassSheetSource` | Source Type | Represents thin mass sheets discretized as surface elements. |
| `PrescribedPointTrajectory` | Additional API | Positive point mass at prescribed time samples and positions. |
| `PrescribedRigidTransform` | Additional API | Prescribed translations and proper rotations for a point-mass cloud. |
| `discretize` | High-level factory | Builds a fixed weighted point source from pure sphere geometry, density and Quantity spacing. |
| `point_sum` | High-level model | Creates a static point-source evaluation setup. |
| `rigid_motion` | High-level factory | Rigid-motion model of a point-mass cloud. A `PrescribedRigidTransform` with `sampling=` gives time-series evaluation; a callable pose with `epoch=` gives `.at(targets).fourier_series(...)` and `.fourier_transform(...)`, which return `EvaluationResult`. Only the computation frame is supported in the continuous form. |

### `gragra.observables`

`trajectory_potential`, `trajectory_acceleration`, and
`trajectory_gravity_gradient` evaluate a prescribed positive point-mass
trajectory. `rigid_transform_potential`, `rigid_transform_acceleration`, and
`rigid_transform_gravity_gradient` evaluate a `PointMassCloud` under a
prescribed rigid transform. These additional APIs are subpackage exports and
are not re-exported by `gragra`.

`displacement_transfer_potential` and `displacement_transfer_gravity_gradient`
are additional subpackage observables for the first-order displacement of
discrete mass elements. They are not re-exported by `gragra`.

`ObservableSeriesBlock`, `stream_observable`, and
`collect_observable_blocks` provide the bounded SI-observable layer for
fixed-mesh time and frequency series.
The streaming integrator below stays G-free; `stream_observable` is the layer
that applies `G_SI` exactly once.

`stream_rigid_motion_observable` and `rigid_motion_observable` evaluate a
static `PointMassCloud` or replayable static source provider under a
`PrescribedRigidTransform`, at world-frame target positions. Select
`observable="potential"`, `"acceleration"`, or `"gradient"`. The stream
returns `RigidMotionBlock` values with SI units and axes `(frame, target)`,
`(frame, target, 3)`, or `(frame, target, 3, 3)`. Frames follow the transform's
time order; vector and tensor components are in world coordinates. The
executor is G-free and the public layer applies `G_SI` once. Each yielded
float64 values array owns independent storage and is read-only.
Provider rows with zero mass remain part of canonical digest and replay checks,
but are omitted from coordinate transforms, Newton evaluation, and exact
collision preflight. They cannot trigger a collision; an all-zero source
therefore returns exact-zero values.

The public functions accept `softening_m=0`, `memory_budget_bytes=2*1024**3`,
`motion_block_size=1`, `target_block_size=128`, `source_block_size=8192`, and
`kernel_chunk_size=1024`. Types, shapes, options, and input metadata are
validated when the function is called. NumPy arrays, `TargetPoints`, and
rectangular lists/tuples are supported directly. Other array-like targets must
expose non-owning shape and fixed-width dtype metadata; the accepted dtype kinds
are signed/unsigned integer, real float, bytes, and Unicode (`i`, `u`, `f`, `S`,
`U`), with nonzero itemsize for bytes and Unicode dtypes. Bool, complex, and
object dtypes are rejected. The advertised dtype must
match the dtype returned by materialization: the planner uses it to reserve any
temporary array bytes before conversion, then checks the materialized dtype.
Custom array-like targets therefore need truthful shape and dtype metadata for
the memory budget to hold. This lets dense output and snapshot bytes be
reserved before target materialization. Snapshot creation and the final memory
plan also occur at the call. Provider baseline scans and—when softening is
zero—the exact active-source collision preflight run on the first iteration,
before the first block is yielded. The preflight scans all frames in bounded
source/target tiles and has worst-case `O(F*T*S)` comparisons; its time and
first-yield latency are measured separately. A failure before that point
publishes no block. A later failure ends the stream;
already yielded blocks remain unchanged, and dense evaluation returns only
after complete successful collection.

An empty source or an all-zero-mass source yields exact-zero blocks for
nonempty targets. With zero targets, streaming yields no blocks and dense
evaluation returns the correctly shaped empty array `(frame, 0, ...)`; source
provider validation and replay checks still occur. Memory budget accounting
checks snapshot creation peaks, provider resident bytes, minimum working set,
and dense result/coverage reservations before making executor-owned snapshots.
The final tile plan is then checked against the snapshot sizes. Accounting
covers library-owned snapshots (including transient copies), provider resident
bytes as reported by the provider, received blocks, hashing/working buffers,
transform and kernel tiles, and dense output/coverage. It excludes caller-owned
input storage, consumer-retained blocks, Python/runtime overhead, and
unreported provider memory, so the limit is a library allocation budget rather
than an RSS ceiling.

The Wave 4–6 independent reference wrappers for point mass, multipole, finite
cylinder, rotor, finite/displaced box, host/inclusion, acoustic finite box,
acoustic infinite halfspace, and Rayleigh halfspace are subpackage APIs.
They are verification references with case-specific assumptions, not general
production geometry replacements.

`rayleigh_halfspace_allspace_observables` (SI entry point for
`gragra.analytic.rayleigh_halfspace_allspace_observables_g_free`) covers both
ground (z > 0) and underground (z < 0) Rayleigh-wave gravity per H19
Eq.(96)–(98), returning `GravityObservables`. It is a separate entry point
from the legacy ground-only `rayleigh_halfspace_observables`; neither
replaces the other. See `docs/design/literature-r.md`.

`cylinder_analytic_observables` and its G-free counterpart
`gragra.analytic.cylinder_analytic_observables_g_free` provide an analytic
finite-cylinder triple with infinity-zero potential and proper rotation.
The potential uses convergent analytic series; it is not a finite elliptic
closed form. The existing numerical `cylinder_observables` and analytic
`cylinder_acceleration_gradient` remain separate APIs.
See `docs/design/cylinder-potential.md`.

### `gragra.results`

`EvaluationResult` holds high-level values and target metadata; `compare`
aligns results by target IDs and SI positions, converts units, and subtracts
compatible values. It does not rotate components: differing output component
bases are rejected, so vector and tensor results must be re-evaluated with a
shared `output_frame` before comparison. The context records the field
definition and interaction used by the evaluation; `compare` checks these
interpretation contracts along with the payload axes. `result.evaluation`
retains the model's `FieldEvaluation`, while `result.context` provides stable
model-independent access to normalized coordinates and representation
metadata. The namespace also retains `GravityResult` and C3-v1 representation
descriptors. None are root-package exports. See the [results reference](../../docs_site/docs/api/results.md), the [interaction reference](../../docs_site/docs/api/interactions.md), and the [spectral representation conventions](spectral_representation_conventions.md).

### `gragra.interactions`

`Interaction` is the structural contract for physical identity, represented
by a name and immutable scalar parameters. `Newtonian` is the built-in
parameter-free interaction. The model spec owns this value and the evaluation
context retains it for comparison. Neither name is re-exported from the root
package. See the [interaction reference](../../docs_site/docs/api/interactions.md).

### `gragra.geometry`

| Symbol | Category | Description |
|---|---|---|
| `Frame` | Coordinate Frame | Origin and orientation used by high-level geometry and target coordinates. |
| `Sphere` | Pure Geometry | Sphere defined by Quantity center and radius in a Frame; density and discretization are separate inputs. |
| `HalfSpace` | Surface Geometry | Flat free surface at local `z=0`, with the half-space medium at local `z<0`. |
| `Cuboid` | Pure Geometry | Center and three positive side lengths in a Frame; edges follow its local axes. |
| `Cylinder` | Pure Geometry | Center, radius, and height in a Frame; axis follows local z. |
| `SphericalShell` | Pure Geometry | Center and ordered inner/outer radii in a Frame. |
| `Appearance` | Display Metadata | Optional color, alpha, and label; physics and geometry validation do not depend on it. |
| `HalfSphere` | Numerical Primitive | Dome + floor (z-up, flat side down) — SCA hemisphere cavity. |
| `HalfCylinder` | Numerical Primitive | Kamaboko (axis = x, floor z = 0; arbitrary rotation out of scope). |
| `MeshResolution` | **Experimental** | Surface-mesh resolution (n_u/n_v/n_cap, defaults 50/50/25). |
| `PrimitiveSurfaceMesh` | **Experimental** | Triangle surface mesh container (vertices/triangles/centroids/normals/areas, read-only). Triangle ordering is NOT part of the API. |

`Frame`, `Sphere`, `Cuboid`, `Cylinder`, and `SphericalShell` form the
quantity-aware pure Geometry API. `gragra.geometry.Cuboid` and
`gragra.geometry.Cylinder` differ from the older numerical classes with the
same short names at `gragra.geometry.primitives`; no compatibility wrapper is
provided. The current C5 `discretize` path supports `Sphere`: pass explicit
density and Quantity spacing to create a `DiscreteSource`, then call a
high-level model's `.at(targets)` to select evaluation targets. D3 adds five
production factories that use the common Geometry and Result contracts. The
current sphere factory uses an internal `WeightedPointSource` to generate its
point arrays; that is not its public return type.

### `gragra.gravity_sources`

`discretize(geometry, density, spacing)` eagerly converts a supported sphere
to an immutable `DiscreteSource` with explicit physical inputs. Internally it
uses a `WeightedPointSource` to generate positions and mass weights.
`point_sum(source)` creates a static point-source evaluation model. These
high-level exports are lazy so numeric imports remain cold.

### `gragra.analytic.objects`

`uniform_sphere` provides the analytic uniform-sphere model with the same
high-level evaluation workflow as `point_sum`.

`uniform_cuboid` and `uniform_cylinder` provide static analytic fields from
their pure Geometry plus density. Both call existing G-free native functions;
`uniform_cylinder` uses `cylinder_analytic_observables_g_free` and has no
numerical quadrature fallback.

### `gragra.analytic.seismic`

`rayleigh_halfspace` creates an Experimental high-level phasor model for a
homogeneous elastic half-space. It accepts an explicit `PhasorRepresentation`,
GPS reference time, local surface phase origin, and horizontal direction; its
`.at(targets)` setup validates and plots the resolved request before returning
native frequency matrices for potential, acceleration, and gradient. The
model uses the all-space Rayleigh solution above and below the surface and
applies `G_SI` once.

`rayleigh_response` creates a reusable per-frequency Newton response from
vertical surface displacement to the same three observables. Its `.run()`
stores coefficients generated from a one-metre basis phasor, and the returned
`GravityResponse` applies input phasors through `gragra.results`.

`spherical_cavity` and `spherical_shell` create P/S displacement phasor
evaluations for their respective pure geometries. Both require an explicit
`reference_time`; a supplied `FrequencySeries.epoch` is provenance only. Their
phase origin, propagation direction, and polarization use Geometry-frame
coordinates. Cavity contributions carry the approved reference per mode and
contribution; shell contributions use `ZERO_AT_INFINITY`.

`halfspace_psv` evaluates incident P/SV displacement phasors for a `HalfSpace`.
Its phase origin and incidence/azimuth use the surface frame. Positive
horizontal wavenumber uses the half-space Fourier Green reference; the
spherical-mean reference is accepted only for the documented normal-incidence
traction-free q=0 family.

### `gragra.analytic`

`spherical_cavity_observables_g_free` evaluates analytic solid-harmonic series
for prescribed P/S plane waves outside a fixed spherical cavity, at cavity-side
and material-side targets. It returns `bulk`, `wall`, and `total` mappings of
`GFreeGravityValues`, each containing potential, acceleration, and gradient.
`gragra.observables.spherical_cavity_observables` is the SI counterpart.
Wavenumber may be scalar or `(K,)`; component axes are trailing and outputs
are read-only. See [the derivation and convergence contract](../design/phase12a_analytic_benchmarks.md#9-general-point-fixed-cavity-analytic-series-g-2026-09-20).

```python
from gragra.observables import spherical_cavity_observables

fields = spherical_cavity_observables(
    [[0, 0, 0], [.2, -.1, .1], [1.01, 0, 0]],
    radius_m=1, density_kg_m3=2700,
    wavenumber_rad_m=[.01, .8, 5], displacement_m=1e-6*(1+.4j),
    propagation_direction=[0, 0, 1], polarization=[0, 0, 1], wave_type="P",
    center_m=[0, 0, 0], phase_origin_m=[0, 0, 0],
)
assert fields["total"].gradient.shape == (3, 3, 3, 3)  # K, M, i, j
```

This does not replace the existing center-only spherical-cavity or finite-shell
functions. Wall targets and elastic scattering remain outside its contract.

| Symbol | Category | Description |
|---|---|---|
| `uniform_sphere_potential` | Analytic | Exact potential of a uniform solid sphere. |
| `uniform_sphere_acceleration` | Analytic | Exact acceleration of a uniform solid sphere. |
| `uniform_sphere_gradient` | Analytic | Exact gravity gradient tensor of a uniform solid sphere. |
| `spherical_cavity_bulk_p` | Analytic (G-free) | Bulk term 4πρξ·j₀(k_P R) for a spherical cavity in a plane P wave. |
| `spherical_cavity_wall` | Analytic (G-free) | Wall term −4πρξ(j₀−2j₁/x) (P) / −4πρξ·j₁/x (S). |
| `spherical_cavity_total_p` | Analytic (G-free) | Total = bulk + wall = 8πρξ·j₁(x)/x (dipole/total form). |
| `spherical_shell_dipole_factor` | Analytic (dimensionless) | Shell dipole factor −j₁(x)/x（dipole/total 定式化 — 純 bulk と直接比較不可）. |
| `halfspace_transfer_bulk` | Analytic (dimensionless, G-free) | 鉛直入射 P の T_bulk(kH)=1−2cos(kH)（δa_z=−4πGρ₀ξ_inc·T 規約）. |
| `cylinder_sca_wall_coupling_tensor_g_free` | Analytic (dimensionless, G-free) | Cylinder SCA wall tensor from the existing unit-density gradient, general targets and proper rotation. See [design/usage](../design/cylinder-sca-tensor.md) and [validation](../validation/cylinder-sca-tensor.md); the generic SI contraction has the explicit #115 integration dependency. |
| `halfspace_psv_reflection`, `HalfspacePSVReflection` | Analytic (dimensionless) | General lossless P/SV free-surface potential and displacement ratios, including converted evanescent P. |
| `halfspace_psv_observables_g_free` | Analytic (G-free) | Ground/underground potential, acceleration and gradient for the same traction-free field. SI: `gragra.observables.halfspace_psv_observables`. See [design and usage](../design/halfspace-general-psv.md) and [validation](../validation/halfspace-general-psv.md). |
| `halfspace_transfer_full` | Analytic (dimensionless, G-free) | T_full(kH)=−2cos(kH)=T_bulk−1（表面質量シート込み・媒質内側極限）. |
| `rayleigh_depth_reduction` | Analytic (dimensionless, G-free) | 厳密 2 指数固有関数の変位振幅深度低減 \|u(h)\|/\|u(0)\|（加速度 transfer ではない）. |
| `sca_wall_coupling_factor` | Analytic (dimensionless) | Harms et al. 2022 Table 1 の small-cavern approximation 結合係数（5行の限定lookup。sphere/arbitrary と cube/3D center は同じ 4π/3、残る floor 3行は別の原典条件を保つ）. |
| `sca_wall_acceleration` | Analytic (G-free) | 上記係数を用いた small-cavern 限界の cavern-wall 加速度振幅 `-factor*rho*xi`（wall 項のみ、bulk は含まない — total ではない）. |
| `sphere_sca_wall_coupling_tensor_g_free` / `cuboid_sca_wall_coupling_tensor_g_free` | Analytic (dimensionless) | 有限球・直方体の一般 target に対する SCA wall tensor `C_ij`; 既定は充填体gradientの厳密inversion `C=-Gamma_filled/rho`、明示した `quadrature_order` はtarget別収束確認付きの独立表面求積、物理境界targetは拒否し、直方体は proper rotation を受け付ける. |
| `sca_wall_acceleration_from_tensor_g_free` | Analytic (G-free) | `-rho*C@u` による SCA wall 加速度振幅. |
| `cuboid_filled_internal_field` | Analytic (G-free) | 満充填一様密度直方体の内部重力場（Nagy 型 8頂点閉形式、`filled` term — bulk/wall いずれでもない inversion principle の入力）. |
| `cylinder_filled_axis_field` | Analytic (G-free) | 満充填一様密度円柱の軸上重力場（`filled` term、軸上限定）. |
| `rayleigh_halfspace_allspace_observables_g_free` | Analytic (G-free) | H19 Eq.(96)–(98) の Rayleigh 波地上・地下重力（potential/a/Γ、`GFreeGravityValues`）。既存の地上専用 `rayleigh_halfspace_observables_g_free`（`wave456_reference`, legacy `GFreeObservables`）とは別入口で、両立して保持される。詳細は `docs/design/literature-r.md`。 |

### `gragra.waves`

| Symbol | Category | Description |
|---|---|---|
| `normalize_directions` | Kinematics | Normalizes direction vectors. |
| `orthonormal_basis_from_direction` | Kinematics | Computes transversal basis vectors perpendicular to propagation direction. |
| `p_wave_polarization` | Kinematics | Computes displacement direction for P-waves (compressional). |
| `s_wave_basis` | Kinematics | Computes S-wave displacement basis (e1, e2). Returns a generic basis if `vertical_unit=None` (S1/S2); returns physical vertical-anchored basis (e_SH, e_SV) matching seismic conventions if `vertical_unit` is specified. |
| `displacement_direction` | Kinematics | General wave displacement mapper supporting P, S1, S2, and vertical-anchored SH/SV displacement directions. |
| `plane_wave_displacement` | Kinematics | G-free witness displacement field: `amplitude_m * displacement_direction * plane_wave_complex_factor`; shape (N, Ndir, Nf, 3) complex128. No `sign` argument (+1 fixed, Phase 15-a-2). |
| `StochasticPSParameters` | Kinematics | P/S realizationの再構成に必要な方向、位相、波種、振幅、波数を保持するfrozen dataclass。配列フィールドはfloat64、read-only（Phase 15-b）。 |
| `stochastic_ps_realization` | Kinematics | 同一方向に付随するP/S波の確率的realizationを生成するG-free helper。出力は`(N, R, 3)` complex128、read-onlyで、相関位相と独立位相を明示的に切り替える（Phase 15-b）。 |
| `plane_wave_phase` | Kinematics | Calculates phase shifts $\vec{k} \cdot \vec{r}$. |
| `plane_wave_complex_factor` | Kinematics | Returns $\exp(i \cdot \text{sign} \cdot \vec{k} \cdot \vec{r})$ (default `sign=+1`; pass `sign=-1` to flip). |
| `rayleigh_secular` | Kinematics (half-space) | Rayleigh secular function R(c)（Aki & Richards 標準形、定義域 (0, v_S)）. |
| `rayleigh_speed` | Kinematics (half-space) | Rayleigh 位相速度 c_R（numpy 二分法 + 走査フォールバック、fail-closed 診断）. |
| `rayleigh_displacement` | Kinematics (half-space) | 厳密 2 指数固有関数の複素変位場 (N,3)（z>0 は ValueError — overflow ガード）. |
| `rayleigh_ellipticity` | Kinematics (half-space) | 深度依存 H/V（全深度有限、深部極限 H/V→s）. |
| `evanescent_complex_factor` | Kinematics (half-space) | 水平位相 × 鉛直減衰 e^{−κh}（z>0・κ<0 は ValueError — overflow ガード。κ=0 で plane_wave_complex_factor と一致）. |
| `free_surface_reflection_vertical` | Kinematics (half-space) | 鉛直入射の変位反射係数（恒等式 1+R_PP=2 で規約固定。一般入射角はスコープ外）. |
| `surface_displacement_factor_vertical` | Kinematics (half-space) | 自由表面の変位倍率 2.0（変位規約）. |

### `gragra.coherent`

| Symbol | Category | Description |
|---|---|---|
| `plane_wave_mass_weights` | Math / Physics | Multiplies density/mass weights by complex wave propagation factors. Accepts provisional `backend` parameter. |
| `plane_wave_bulk_mass_weights` | Math / Physics | Polarization-projected bulk mass weights δm = −iρk(k̂·ê_pol)ξV·e^{ik·r}（S 波は構造的厳密 0）. |
| `directional_power_average` | Math / Physics | Averages coherent auto-power ($\sum_i w_i \lvert r_i \rvert^2$) over directions; requires non-negative weights, so Lebedev orders 13/25/27 cannot be passed. `directional_cross_average` and `directional_outer_cross_average` accept finite real weights including negative values under their separate cross-response contracts. |
| `directional_cross_average` | Math / Physics | Element-wise cross average $\sum_d w_d A_d \cdot \bar{B}_d$ over a direction axis (conjugate second argument, real weights, negative values allowed). Complex128 output; direction axis removed (Phase 15-a-1). |
| `directional_outer_cross_average` | Math / Physics | G-free component-outer cross tensor $\sum_d w_d A_{i,d} \cdot \bar{B}_{j,d}$ from two `(M, Ndir, Nf, 3)` responses to `(Ma, Mb, Nf, 3, 3)`; building block for `coherent_gravitoelastic_tensor` / `coherent_acceleration_csd` (Phase 15-a-3). |
| `ensemble_outer_cross_average` | Math / Physics | realization軸`R`だけを均等重みで縮約するcomponent-outer cross平均 $(1/R)\sum_r A_i\bar{B}_j$。サンプル軸`N`を保持し、`(N, R, 3)`から`(N, 3, 3)`のcomplex128 read-only配列を返す（Phase 15-b）。 |

### `gragra.integrators`

| Symbol | Category | Description |
|---|---|---|
| `direct_point_sum` | Integrator | Normalizes inputs and dispatches to G-free kernels; the primary integration entry point for point sources. Low-level integrator; users should normally prefer top-level observables such as `point_acceleration`. |
| `direct_point_dipole_sum` | Integrator | Dispatches the G-free vector-weight (dipole/total) contraction across the `numpy`/`numba`/`cpp`/`cupy` backends; takes raw `mass_kg`/`displacement_m`/`sources_xyz` arrays rather than a `WeightedPointSource`. Low-level integrator; users should normally prefer `displacement_transfer_acceleration`. |
| `SeriesMetadata` | Integrator | Immutable fixed-mesh geometry, ordered source IDs, sample axis, and frequency-phase metadata shared by provider replays. |
| `SeriesProvider` | Integrator | Replayable fixed-mesh source-block provider with validated sample and source coverage. |
| `ScalarSourceSeriesBlock` | Integrator | Bounded scalar mass-weight slab for density or surface-density series. |
| `DisplacementSourceSeriesBlock` | Integrator | Bounded slab keeping background mass and vector displacement separate. |
| `as_series_provider` | Integrator | Adapts a supported fixed-mesh series or explicit metadata/factory to a replayable provider. |
| `iter_g_free_blocks` | Integrator | Computes validated bounded potential, acceleration, or gradient blocks without applying `G_SI`. |

### `gragra.kernels`

Pure geometric contraction kernels (completely **G-free**):

| Symbol | Category | Description |
|---|---|---|
| `potential_contract` | Kernel | Inverse-distance $1/r$ contraction. |
| `acceleration_contract` | Kernel | Inverse-square $\vec{r}/r^3$ contraction. |
| `gradient_contract` | Kernel | Gravity gradient tensor kernel. |
| `potential_contract_parametric` | Kernel | Parametric (wave/frequency-dependent) potential kernel. |
| `acceleration_contract_parametric` | Kernel | Parametric acceleration kernel. |
| `gradient_contract_parametric` | Kernel | Parametric gravity gradient kernel. |
| `dipole_contract` | Kernel | G-free vector-weight (dipole/total) contraction $\delta a_i = -\sum_n m_n K_{T,ij}(r_n) d_n^j$; $K_T$ is the same gravity gradient kernel as `gradient_contract` (kernel_contract.md §2.2/§3). |

### `gragra.plotting`

High-level plotting and diagnostic visualization routines (lazy-loaded via PEP 562):

| Symbol | Category | Description |
|---|---|---|
| `formula_to_latex` | Plotting | Converts verified analytical formulas to LaTeX math syntax. |
| `hist` | Plotting | Histogram of per-element diagnostics for discrete point sources. |
| `plot_contribution_hist` | Plotting | Histogram of directional or magnitude gravitational acceleration contributions. |
| `plot_cumulative_contribution` | Plotting | Cumulative gravitational acceleration profile $a(<R)$ versus distance. |
| `plot_field` | Plotting | Native grid 2D slice visualization of scalar or vector perturbation fields. |
| `plot_hist` | Plotting | Step plot rendering of diagnostic histograms. |
| `plot_map` | Plotting | Spatial 3D scatter or 2D projection map of source element contributions. |
| `plot_result` | Plotting | Visualizes physical fields, time series, or spectra from an `EvaluationResult`. |
| `plot_shape` | Plotting | 3D surface mesh or wireframe visualization of analytical geometric shapes. |

---

## 3. Additional APIs

These modules and adapters are integrations or developing features and may
change as the research code evolves.

### `gragra.reference`

| Symbol | Dependency | Description |
|---|---|---|
| `centered_cavern_volume_kernel` | `mpmath` (`reference` extra) | Optional high-precision scalar volume kernel for a centered spherical cavern in an isotropic, incoherent acoustic field. Its immutable result and exception types are also public in this namespace. |

Install the optional dependency with `pip install gragra[reference]`. Core,
namespace, and exception imports work without mpmath; a numerical call checks
runtime availability. See the [reference API](../../docs_site/docs/api/reference.md)
and [numerical reference tutorial](../../docs_site/docs/tutorials/numerical-reference.md)
for the bounded model, error estimate, and installed-use example.

### `gragra.adapters`

| Symbol | Dependency | Description |
|---|---|---|
| `TriangleSurfaceMesh` | Built-in / `meshio` | Surface mesh representation. |
| `VolumeMesh` | Built-in / `meshio` | Volume mesh representation (tetra/hexahedron). |
| `from_meshio` | `meshio` (optional) | Reads surface mesh files using the `meshio` package. |
| `from_meshio_volume` | `meshio` (optional) | Reads volume mesh files using the `meshio` package. |
| `centroids_normals_areas` | Built-in | Computes geometric properties of triangular mesh elements. |
| `cell_centroids_volumes` | Built-in | Computes centroids and volumes of volume cells. |
| `select_group` | Built-in | Filters mesh cells belonging to a specific group by name. |
| `select_group_tag` | Built-in | Filters mesh cells by integer group tag directly (no `group_names` required). |
| `to_surface_mass_sheet` | Built-in | Converts surface mesh to `SurfaceMassSheetSource`. |
| `to_volume_element_source` | Built-in | Converts volume mesh to `VolumeElementSource`. |
| `to_unstructured_displacement_field` | Built-in | Converts ordered tetra blocks of a `VolumeMesh` into an `UnstructuredDisplacementField`; nonempty hex blocks are rejected. |
| `cell_density_from_group_tags` | Built-in | Maps exactly the used tetra cell tags to finite, non-negative densities without changing cell order. |
| `OpenFOAMDensitySnapshot` | `meshio` (lazy, `mesh` extra) | Immutable loader-created Foundation OpenFOAM 14 density snapshot with native C/Vc/cellID metadata. |
| `load_openfoam_density` | `meshio` (lazy, `mesh` extra) | Loads the bounded legacy-ASCII VTK profile for exactly `rho` or `delta_rho`. |
| `to_gwexpy_scalar_field` | `gwexpy` (required; lazy adapter import) | Exports observables to `gwexpy` scalar fields. |
| `to_gwexpy_vector_field` | `gwexpy` (required; lazy adapter import) | Exports observables to `gwexpy` vector fields. |
| `to_gwexpy_tensor_field` | `gwexpy` (required; lazy adapter import) | Exports observables to `gwexpy` tensor fields. |
| `GwexpyObservableBundle` | `gwexpy` (required; lazy adapter import) | Frozen native Field and companion metadata bundle. |
| `ObservableArrayBundle` | `gwexpy` (required; lazy adapter import) | Frozen reconstructed NumPy array and companion metadata bundle. |
| `to_gwexpy_observable` | `gwexpy` (required; lazy adapter import) | Converts static, time-first, or frequency-target-first observables to a Field bundle. |
| `from_gwexpy_observable` | `gwexpy` (required; lazy adapter import) | Validates a Field bundle and reconstructs its current observable array. |
| `GwexpyDensitySnapshotBundle` | `gwexpy` (required; lazy adapter import) | Frozen native scalar density snapshot and its validated sidecar. |
| `DensitySnapshotArrayBundle` | `gwexpy` (required; lazy adapter import) | Frozen reconstructed `VolumeElementSource` and sidecar; `time_s` comes from the sidecar. |
| `to_gwexpy_density_snapshot` | `gwexpy` (required; lazy adapter import) | Converts one real `rho` or `delta_rho` volume-element snapshot into a native scalar field plus sidecar. |
| `from_gwexpy_density_snapshot` | `gwexpy` (required; lazy adapter import) | Revalidates a density snapshot bundle and reconstructs its current density values without file I/O. |
| `iter_hdf5_field_snapshots` | `h5py` (optional) | Streams a batched HDF5 field dataset (leading `(P, ...)` on-disk axis) snapshot-by-snapshot without materializing the full batch (Phase 10-d). |
| `export_specfem_gll_hdf5` | `h5py` (optional, `field-io` extra) | SPECFEM3D Cartesian GLL exporter: reads a `decompose_mesh` output tree and writes one canonical HDF5 file (`mesh_type="gll"`, `schema_version=1`). See `docs/design/solver_import_contract.md` and `docs/architecture/external_field_schema.md` §GLL Element Schema (mesh_type="gll"). |
| `ExportConfig` | Built-in | Frozen dataclass configuring one `export_specfem_gll_hdf5` call (run tree paths, `output_sampling_kind` static/time_sampled, documented profile input parameters); most fields are auto-derived or must be supplied explicitly rather than guessed. |
| `read_specfem_gll_hdf5` | `h5py` (optional, `field-io` extra) | Loads a canonical schema-v1 time-sampled SPECFEM GLL HDF5 file into a closed immutable `SpecfemGLLDataset`. The documented profile accepts a single elastic material with no PML. |
| `SpecfemGLLDataset` | Built-in | Immutable element-ordered GLL coordinates, density, quadrature volume, displacement snapshots, time, and source-file identity. |
| `SpecfemGLLStreamReader` | `h5py` (optional, `field-io` extra) | Context-managed bounded reader for the same time-sampled, single-elastic-material, non-PML 5×5×5 GLL profile. It validates every requested sample and accounts for decoded chunks, validation buffers, provider destinations, and immutable copies. |
| `read_fixed_mesh_series_hdf5` | `h5py` (optional, `field-io` extra) | Reads the versioned fixed-mesh density, displacement, or surface-density profile with ordered IDs and explicit axis/phase metadata. |
| `write_fixed_mesh_series_hdf5` | `h5py` (optional, `field-io` extra) | Writes through a create-new partial file and publishes complete output only after bounded exact value and metadata readback. |
| `write_observable_series_hdf5` | `h5py` (optional, `field-io` extra) | Writes streamed SI observables and publishes complete output only after bounded exact value, axis, target, unit, frame, and phase readback. |
| `to_gwexpy_density_series` / `from_gwexpy_density_series` | `gwexpy` (required; lazy adapter import) | Converts and reconstructs fixed-volume real density series or complex density phasors while retaining geometry and phase metadata. |
| `to_gwexpy_density_time_series` / `from_gwexpy_density_time_series` | `gwexpy` (required; lazy adapter import) | Typed entry points that accept and restore only real `DensityTimeSeries` values. |
| `to_gwexpy_density_phasors` / `from_gwexpy_density_phasors` | `gwexpy` (required; lazy adapter import) | Typed entry points that accept and restore only complex `DensityPhasors` with frequency-phase metadata. |
| `to_gwexpy_displacement_series` / `from_gwexpy_displacement_series` | `gwexpy` (required; lazy adapter import) | Converts and reconstructs current native vector-field values while background mass remains separate. |
| `to_gwexpy_surface_density_series` / `from_gwexpy_surface_density_series` | `gwexpy` (required; lazy adapter import) | Converts and reconstructs fixed-surface scalar values with areas, normals, IDs, units, and phase metadata. |
| `from_hdf5_batched_weights` | `h5py` (optional) | Reads a batched HDF5 field file into the raw `(positions_m, weights (N,P))` tuple contract, streaming via `iter_hdf5_field_snapshots` (Phase 10-d). |
| `lebedev_grid` | `scipy` (optional >=1.15) | Generates normalised Lebedev directions and weights (khat, wnorm) without importing nnsus. Some orders have negative weights; `directional_power_average` rejects them, while the cross averages have separate signed-weight contracts. |
| `load_trimesh` | `trimesh` (optional, `mesh-quality` extra) | Loads a `trimesh.Trimesh` from a path (trimesh-native) or a duck-typed meshio-like Mesh (meshio not imported). |
| `check_mesh_quality` | `trimesh` (optional, `mesh-quality` extra) | Geometric topology screen: watertight / winding / divergence-theorem volume / edge stats / duplicate vertices / Euler number / 5% volume gate. 自己交差・全体法線反転は検出不能（docstring 参照）。 |
| `DiscreteBackgroundDensity` | Built-in | Discrete background density model for unstructured fields. |
| `DifferentiableDiscreteBackgroundDensity` | Built-in | Differentiable discrete background density model for unstructured fields. |
| `from_hdf5_unstructured_field` | `h5py` (optional) | Imports an `UnstructuredDisplacementField` from HDF5. |
| `from_hdf5_displacement_grid` | `h5py` (optional) | Imports a `DisplacementGrid` from HDF5. |
| `from_hdf5_field` | `h5py` (optional) | Imports a displacement field (unstructured or structured) from HDF5. |
| `read_hdf5_background_density` | `h5py` (optional) | Reads background density from HDF5. |



> [!NOTE]
> **Legacy GWexpy converter scope restrictions:**
> - **Static fields only**: Exports static fields only ($n_t = 1$, single time point `t_s`).
> - **Flat point-cloud mode (`grid_shape=None`)**: $M$ target points are mapped to `axis1` as point index; `positions_m` is accepted for validation but not encoded as full 3D coordinate metadata.
> - **Grid mode (`grid_shape=(nx, ny, nz)`)**: Reshapes $M = nx \times ny \times nz$ points to 3D.
> - **Metadata preservation**: `VectorField` and `TensorField` are built via component `ScalarField` dict to preserve unit and axis metadata.
> - **Out of scope**: Time/frequency series, k-space, and structured `TargetGrid` export are outside these three legacy converters' current scope. The observable bundle bridge separately supports the accepted static, time-first, and frequency-target-first layouts.

### `gragra.fields`

| Symbol | Dependency | Description |
|---|---|---|
| `BackgroundDensityModel` | Built-in | Structural protocol (typing aid) for background density models. Inheritance is not required; any callable with the matching signature satisfies it. |
| `DifferentiableBackgroundDensityModel` | Built-in | Structural protocol (typing aid) extending `BackgroundDensityModel` with a `gradient()` method. Inheritance is not required. |
| `DensityPerturbationGrid` | Built-in | Regular/rectilinear grid representation of density perturbations. |
| `DensityTimeSeries` | Built-in | Immutable real fixed-volume density series with a strictly increasing time axis and explicit `rho` or `delta_rho` convention. |
| `DensityPhasors` | Built-in | Immutable complex fixed-volume density amplitudes with positive frequencies, epoch, and spatial phase origin. |
| `FixedMeshDisplacementSeries` | Built-in | Immutable time or frequency displacement series that keeps background mass and vector displacement separate. |
| `FixedMeshSurfaceDensitySeries` | Built-in | Immutable fixed-surface density series retaining signed density, areas, normals, IDs, and face metadata. |
| `DisplacementGrid` | Built-in | Regular/rectilinear grid representation of displacement fields. |
| `UnstructuredDisplacementField` | Built-in | Unstructured linear tetrahedron mesh representation of displacement fields; also provides `as_dipole_source(rho0_kg_m3)` returning centroid-based `(positions_m, mass_kg, displacement_m)` for the dipole/total path (Phase 14-b). |
| `SurfaceDisplacementField` | Built-in | Triangular surface mesh displacement field; computes surface mass-density perturbations from interface density jumps. |
| `TetraLinearizedSources` | Built-in | Read-only finite P1-tetra source terms: mass displacement, bulk volume, and interface surface terms. |
| `TetraRepresentationReport` | Built-in | Read-only G-free comparison of the bulk-plus-interface representation with the mass-displacement representation. |
| `piecewise_tetra_linear_sources` | Built-in | Builds bulk and interface sources for a conforming finite P1 tetra body with vacuum exterior. |
| `compare_tetra_linearized_representations` | Built-in | Sums bulk plus interface and compares that G-free response with, rather than adding, the mass-displacement response. |
| `TetraP1DensityLinearizedSources` | Built-in | Read-only quadrature source terms for a finite tetra body with cell-local P1 density. |
| `piecewise_tetra_p1_density_sources` | Built-in | Builds degree-2 bulk and face-quadrature interface sources from a conforming P1 tetra displacement field and cell-local P1 density. |
| `compare_tetra_p1_density_representations` | Built-in | G-free comparison of bulk-plus-interface and mass-displacement representations for the cell-local P1 model. |
| `check_triangle_orientation` | Built-in | Diagnoses winding consistency and closedness of a triangle surface mesh. |
| `OrientationReport` | Built-in | Read-only dataclass holding diagnostic results for mesh orientation. |
| `reorient_triangles` | Built-in | Corrects triangle winding inconsistencies topologically. Raises `ValueError` if correction is impossible. |
| `dipole_consistency_check` | Built-in | Cross-checks the bulk+surface divergence path against the dipole/total path as a G-free mesh-quality diagnostic (Phase 14-b); nonzero residuals for inputs violating kernel_contract.md §2.2's six preconditions are expected, not a bug. |
| `DipoleConsistencyReport` | Built-in | Read-only dataclass holding the `dipole_consistency_check` result (`acceleration_bulk_surface`, `acceleration_dipole`, `abs_residual`, `rel_residual`). |
| `batched_displacement_grid_weights` | Built-in | Multi-snapshot structured-grid displacement -> raw `(positions_m (N,3), weights (N,P))` tuple; delegates per-snapshot to `DisplacementGrid` (Phase 10-d). |
| `batched_unstructured_weights` | Built-in | Multi-snapshot unstructured tetra displacement -> raw `(positions_m, weights (N,P))` tuple; delegates per-snapshot to `UnstructuredDisplacementField` (Phase 10-d). |
| `batched_surface_weights` | Built-in | Multi-snapshot triangle-surface displacement -> raw `(positions_m, weights (N,P))` tuple; delegates per-snapshot to `SurfaceDisplacementField` (Phase 10-d). |
| `adiabatic_pressure_density_perturbation` | Built-in | Converts pressure perturbation with `delta_rho = delta_p / c_s**2`. |
| `isobaric_temperature_density_perturbation` | Built-in | Converts temperature perturbation with `delta_rho = -rho_0 * delta_T / T_0`. |

> [!NOTE]
> **Batch Axis helpers (Phase 10-d):** these three functions return a raw `(positions_m, weights)` tuple -- they never construct a `WeightedPointSource` (`weights_kg` there is strict `(N,)`; a batch `(N,P)` weight column is a different, non-kg-only abstraction). The batch axis `P` carries no time/frequency semantics; it is a plain parametric axis consumed by `coherent_point_acceleration` / `coherent_projected_acceleration` / `coherent_point_gravity_gradient`. See `docs/architecture/external_field_schema.md` Batch Axis section and `docs/design/phase10d_batch_import.md`.

> [!NOTE]
> **Field Helpers & Mesh Orientation (Phase 10-c):**
> - **Experimental status**: These helper classes are placed under `gragra.fields` and are experimental. They are not re-exported from the top-level package.
> - **Coordination conventions**: Grids represent cell-center coordinates. The coordinate axes must be finite, real, 1D, and strictly increasing. Non-uniform spacings along axes (rectilinear grids) are fully supported.
> - **Displacement divergence**: `DisplacementGrid` uses central finite differences to calculate the divergence of displacement fields, requiring at least 3 nodes along each axis. `UnstructuredDisplacementField` uses constant-strain shape functions to compute cell-wise divergence on linear tetrahedron meshes.
> - **Surface term**: `SurfaceDisplacementField` projects nodal displacement onto triangle unit normals to compute surface mass-density perturbations $\delta\sigma = \Delta\rho\,(u\cdot\hat{n})$. It takes raw arrays and does not depend on the `adapters` layer; the density jump $\Delta\rho$ is a signed real scalar and the normal orientation follows the triangle winding (caller's responsibility).
> - **Mesh orientation tools**: `check_triangle_orientation` (diagnostics) and `reorient_triangles` (topology-only correction) operate on raw arrays to verify and correct triangle winding consistency. Consistent winding is critical to prevent randomized face normals from corrupting surface mass perturbations.
>   - **`is_outward` is a single-component orientation indicator only**: for a *multi-component* closed mesh, `is_outward` reflects the sign of the **total** signed volume ($V_a - V_b$ for two oppositely-wound disjoint surfaces) and is **not** a component-wise outward indicator. `reorient_triangles` preserves each component's seed winding and does **not** force an outward orientation; determining outward orientation per component requires connected-component decomposition, which is out of scope for these tools.


---

## 4. Wave 4–6 study status and limits

The fixed-mesh containers, bounded HDF5 exchange, streaming path, GWexpy
bridges, independent NumPy references, and thermodynamic conversions are
implemented as subpackage or additional APIs.
They do not change the top-level `gragra.__all__` contract.

The F finite-body study does not establish converged near-interface gradients:
the 1 m cavity gradient error is `0.57–0.81`, and the P1
mass-displacement versus bulk-plus-interface gradient residual is `1.45`.
The E finite-air-box result and the independent infinite-halfspace reference
solve different spatial problems, so their difference is not an accuracy
estimate for either formula.
The observed CFD path is the documented Foundation OpenFOAM upwind scalar
transport case, not a general CFD solver guarantee.
The Wave 5 moving-density study compares direct, HDF5, and GWexpy paths for the
same discrete sources and reports transport discretization separately; that
study evidence does not add another public API level.

The saved older 6 km Rayleigh case completed all 6,912 elements, both time
windows, and all three observables.
Its front/full sensitivity can contain transients or other time variation and
must not be attributed to least-squares ill-conditioning because both fit
matrices have condition number about 1.
The input artifact can be reanalysed exactly, but raw-to-export regeneration
provenance remains incomplete because exporter code identity was not recorded
and cadence provenance is a placeholder.
That study is separate from the new matched 12/18 km S acceleration study.
Both solver runs, canonical exports, full-case A/B/C arrays, and pair-vector
arithmetic completed. The two full analyses stayed below the 2 GiB budget. The required nested
element-key gate rejected 7,491 missing and 7,484 extra keys before the
corresponding static fields could be compared bytewise; B element
contributions were not evaluated. The saved cross-domain vector differences
therefore cannot be interpreted as a finite-domain effect or convergence
result. The S scientific verdict is partial. These research drivers do not
add top-level API symbols.

---

## 5. Internal Components

The following modules/functions are internal to `gragra` and must not be imported by downstream code:
- `gragra._arrays`: Array validation functions (`as_xyz_vector`, `as_xyz_table`, `as_weights`).
- `gragra.array_types`: Internal type aliases (`FloatArray`, `ComplexArray`).
- `gragra.integrators.point_direct`: Internal low-level module implementing `direct_point_sum` and `direct_point_dipole_sum`; import via `gragra.integrators` instead.

---

## 6. Performance Parameters

While top-level functions (e.g., `point_acceleration`) are stable physics APIs, they may accept provisional performance-related keyword arguments:

- **`backend: str = "numpy"`**:
  - `backend` is an experimental performance selector.
  - The default `"numpy"` backend is fully stable and recommended for general use.
  - Non-default backends (e.g., `"numba"`, `"cpp"`, `"cupy"`) are optional research backends; their behavior, requirements, or availability may change as the implementation evolves.
  - Note: The `"numba"`, `"cpp"`, and `"cupy"` backends accelerate the point direct observables (`point_potential`, `point_acceleration`, `point_projected_acceleration`, `point_gravity_gradient`), `displacement_transfer_acceleration` (displacement-transfer/dipole contraction, Phase 14-c/d), the parametric coherent observables `coherent_point_acceleration` / `coherent_point_gravity_gradient` / `coherent_projected_acceleration` (via the parametric kernel contraction), and plane-wave mass-weight construction (`plane_wave_mass_weights`). The displacement-transfer potential and gravity gradient support only NumPy and Numba. Backend comparisons use the applicable local tests (`tests/test_planewave_backends.py`, `tests/test_parametric_numba.py`, `tests/test_parametric_cpp.py`, and `tests/test_cupy_planewave.py`) and their stated tolerances. Requesting an unknown backend raises a `ValueError`; requesting a backend when the optional dependency is not installed/built raises an `ImportError` with an install hint. `lebedev_grid` supplies directions and weights; wall integration and combined bulk/wall transfer are outside this API.
  - The `"cupy"` backend (GPU) covers point direct, displacement transfer, parametric coherent, and plane-wave mass-weight construction. Its error contract adds three failure modes: cupy not installed raises `ImportError` (with a wheel-selection hint), cupy installed but no usable CUDA device/driver raises `RuntimeError` at call time, and requesting execution exceeding 50% of the free VRAM on the device raises a `MemoryError` before any allocation (preflight check). Numerical equivalence to `"numpy"` (`rtol=1e-12, atol=1e-15`) is checked **locally only** on a real GPU (`GRAGRA_REQUIRE_CUPY=1`). On this backend `chunk_size` acts as an effective VRAM tiling parameter for point-direct and parametric paths; `plane_wave_mass_weights` accepts but ignores it.
