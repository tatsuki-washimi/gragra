# Source diagnostics and visualization

`gragra.plotting` provides high-level diagnostic and visualization routines for
discrete source setups, geometric shapes, and calculation results:

- `hist` and `plot_hist`: Histogram per-element density, height, distance,
  inverse-distance, inverse-square, inverse-cube, and density-over-distance
  diagnostics as a GWexpy `Histogram` or step plot.
- `radial_histogram`: Integrate registered scalar source quantities in distance
  bins and retain raw flows alongside finite-bin normalized values.
- `gravity_gradient_radial_histogram`: Sum all signed components of the static
  gravity-gradient tensor by distance, retaining underflow and overflow.
- `source_field` and `source_field_result`: Map source-position values or
  responses onto explicit three-dimensional cells while retaining every axis
  and vector/tensor component.
- `source_contributions` and contribution-statistics functions: Inspect
  complex, source-resolved gravity responses and their group or direction
  statistics.
- `plot_contribution_hist`: Histogram directional or magnitude gravitational
  acceleration contributions from discrete elements.
- `plot_cumulative_contribution`: Plot the radial cumulative acceleration profile
  $a(<R)$ versus distance from a target.
- `plot_map`: Spatial 3D scatter or 2D projection map of element contributions.
- `plot_shape`: 3D surface mesh or wireframe rendering of analytical shapes.
- `plot_result`: Visualizes fields, time series, or spectra from an `EvaluationResult`.

## Histograms (`hist` and `plot_hist`)

The quantities are `density` (kg/m^3), `height` (the z coordinate in `frame`),
`distance` (r from one target), `inverse_r` (1/r), `inverse_square` (1/r^2,
alias `r^-2`, unit m^-2), `gravity_gradient_kernel_norm` (per-source G-free GG
tensor norm, unit m^-3; select `norm="frobenius"` or `norm="spectral"`),
`inverse_cube` (1/r^3), `density_over_r` through `density_over_r3` (ρ/r through
ρ/r^3), and `mass_over_r3`. Distance-based quantities require one target.
`weight="count"` counts elements, and `weight="volume"` sums their volumes in
m^3. This remains a value distribution; use `radial_histogram` to integrate
quantities by source-target distance.

### Example

```python
import numpy as np
from astropy import units as u

from gragra.gravity_sources import VolumeElementSource, point_sum
from gragra.geometry import Frame
from gragra.plotting import hist, plot_hist, plot_map
from gragra.results import Contribution, GravityFieldDefinition, PotentialReference
from gragra.targets import TargetPoints

root = Frame("root")
field = GravityFieldDefinition.source_field(
    "source", Contribution("total"), PotentialReference.ZERO_AT_INFINITY
)
source = VolumeElementSource(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
    [1.0, 2.0, 3.0, 4.0],
    np.array([0.5, 1.5, 1.5, 2.5]),
)
targets = TargetPoints([[-3.0, -4.0, 0.0], [10.0, 0.0, 0.0]] * u.m, ("A", "B"), root)
setup = point_sum(source, frame=root, field_definition=field).at(targets)

# Distance from target A; the counts are [0, 4, 0] over the edges below.
distances = hist(setup, "distance", target_id="A", bins=[0.0, 5.0, 10.0, 20.0])

# Volume-weighted density with the horizontal axis in g/cm^3.
volumes = hist(setup, "density", weight="volume", bins=[0.0, 1.0, 2.0, 3.0], xunit="g/cm3")

# Height needs no target, so target_id must be omitted.
heights = hist(setup, "height", bins=3)

# Inverse square from target B, drawn as a step plot.
fig_hist = plot_hist(setup, "inverse_square", target_id="B", bins=[0.0, 0.01, 0.1, 1.0])

# Spatial 3D scatter map of gravitational acceleration contributions towards target A
fig_map = plot_map(setup, "delta_a_norm", target_id="A", view="3d")
```

## Radial source histograms

`radial_histogram` integrates registered scalar source quantities over
Euclidean source-to-target distance bins. Extensive quantities such as `mass`,
`mass_over_r2`, and `mass_over_r3` are summed directly. Intensive quantities
such as `density_over_r2` are multiplied by each source's verified cell volume
before summation; for example, the result has units kg/m². A volume is required
for intensive quantities and is never inferred from point-source mass weights.
The fixed alias table accepts documented spellings such as `dm/r^2` and
`dm/r^3`; `dm/r^2` has unit kg m^-2. The table does not evaluate arbitrary
expressions. Distances are geometric and remain unsoftened, even if an
evaluation used physical softening.

The result separates two views of the same bins. `result.histogram` is an
un-normalized GWexpy `Histogram` with finite-bin values, underflow, and
overflow in the integrated quantity's unit. `result.values` is a Quantity of
finite-bin values after the selected normalization, and `result.unit` is its
unit. The normalization options are:

| `normalization` | Finite-bin value |
|---|---|
| `"none"` | Integrated contribution in each bin. |
| `"per_length"` | Integrated contribution divided by that bin's width in m. |
| `"fraction"` | Integrated contribution divided by the signed total over all sources. |
| `"fraction_per_length"` | The signed fraction divided by the bin width in m. |

The fraction denominator includes sources outside the displayed edge range,
so changing the shown range does not change it. A zero denominator raises
`ValueError`; negative or greater-than-one fractions are valid for signed
contributions and are not probabilities. These normalized bins are not
additive in the same way as raw integrated contributions.

```python
from gragra.plotting import plot_hist, radial_histogram

radial = radial_histogram(
    setup,
    "mass_over_r2",
    target_id="A",
    bins=[0.0, 5.0, 10.0, 20.0],
    normalization="fraction_per_length",
)
fig, ax, same_result = plot_hist(radial)
```

The result overload of `plot_hist` draws `result.values` against the raw bin
edges without re-binning, then returns `(figure, axes, result)`. The raw
histogram remains available for integrated finite bins and flows. The
`cumulative()` method starts from raw underflow and accumulates through each
finite bin's right edge; overflow is excluded. `cumulative(fraction=True)`
divides those raw cumulative values by the signed all-source denominator.
Direct histogram construction does not define `sumw2`, covariance, or
statistical uncertainty.

## Static gravity-gradient tensor by distance

`gravity_gradient_radial_histogram` sums the physical static GG tensor in each
source-target distance group. For source position \(\mathbf r_i\), target
position \(\mathbf r_T\), mass \(\Delta m_i\), and unsoftened radius \(r_i\),
the zero-softening tensor is

\[
\boldsymbol{\Gamma}_j = G\sum_{i\in B_j}\Delta m_i
\frac{3\hat{\mathbf n}_i\hat{\mathbf n}_i^{\mathsf T}-\mathbf I}{r_i^3},
\qquad
\hat{\mathbf n}_i = \frac{\mathbf r_i-\mathbf r_T}{r_i}.
\]

The result retains every signed component in the computation frame and uses
units of s\(^{-2}\). `tensor_values` contains finite distance bins;
`underflow_tensor`, `overflow_tensor`, and `total_tensor` contain the flow groups
and the all-source sum. All are 3×3 tensors in s⁻², and the total satisfies
\(\boldsymbol{\Gamma}_{\mathrm{total}} =
\boldsymbol{\Gamma}_{\mathrm{underflow}} +
\sum_j\boldsymbol{\Gamma}_j +
\boldsymbol{\Gamma}_{\mathrm{overflow}}\). `histogram(component)` returns one
component, such as `"xx"` or `"yz"`, as a GWexpy `Histogram` with the same
flows.

```python
from gragra.plotting import gravity_gradient_radial_histogram

gg = gravity_gradient_radial_histogram(
    setup,
    target_id="A",
    bins=[0.0, 5.0, 10.0, 20.0],
    softening_m=0.25,
)
xx_histogram = gg.histogram("xx")
total_tensor = gg.total_tensor
```

Bin membership uses the unsoftened Euclidean radius. `softening_m` changes the
existing Plummer gravity-gradient kernel only. A softened source at radius zero
is assigned to the bin containing zero; without softening, the existing
point-kernel collision check raises an error. Internal edges belong to the bin
on their right, and the final right edge belongs to the last finite bin.

This is a signed tensor sum. It preserves component cancellation before any
norm is taken, so the norm of `total_tensor` differs from the sum of per-source
tensor norms. `radial_histogram(..., "GG_kernel_norm", norm=...)` keeps its
existing G-free per-source norm definition.

## Source-position fields

`source_field` and `source_field_result` place source values at physical
coordinates on a caller-supplied Cartesian grid. Pass three finite, strictly
increasing edge vectors for x, y, and z, either as a 3-tuple or a mapping with
exactly those keys. Bare edges are metres; Quantity edges are converted to m.
Every source must lie within the grid. A source on an internal edge belongs to
the cell on its right, and the final outer edge is included in the last cell.
For a static setup, the grid is in its computation frame; dynamic positions use
the phasors' frame and origin. Cells are assigned directly: the functions do
not interpolate or build a histogram of values. They keep all three spatial
axes unless `spatial_projection` requests an XY, XZ, or YZ projection.

Static source quantities include `distance`/`r`, `inverse_r`, `inverse_square`,
`inverse_cube`/`1/r^3`, `density`, `density_over_r` through `density_over_r3`,
and the additive `mass`, `mass_over_r`, `mass_over_r2`, and `mass_over_r3`
quantities. `gravity_gradient_kernel_norm`/`GG_kernel_norm` selects the
per-source norm of the G-free gravity-gradient tensor kernel and has unit
m⁻³. It requires `norm="frobenius"` or `norm="spectral"`. The inverse-distance
and density-over-distance quantities use the unsoftened geometric radius.
`density` is the source density ρ only; it does not name a Newtonian kernel.
Density selectors require an actual verified density and volume
representation; they are not inferred from mass weights.

Physical source terms are separate selectors. `acceleration_kernel` and
`gravity_gradient_kernel` are G-free geometric kernels. Their physical
`acceleration_contribution` and `gravity_gradient_contribution` counterparts
include source mass and apply G once. `GG`/`gg` and `gradient` are accepted as
aliases for the gravity-gradient contribution. With positive `softening_m`,
the physical terms use the existing Plummer kernel. The acceleration and
gravity-gradient kernels are not replaced by a shortcut based on
`1 / (r**2 + eps**2)`.

`GG_kernel` selects the G-free gravity-gradient tensor kernel. `GG_kernel_norm`
selects each source kernel norm before radial or spatial aggregation. For
example, `source_field(..., "GG_kernel_norm", norm="frobenius")` sums the
Frobenius norms of the source tensors in each cell. By contrast,
`source_field(..., "GG_kernel", norm="frobenius")` norms the cell's summed
tensor, so cancellation between source tensors is retained. `GG_xx`, `GG_xy`,
and the other component aliases select a signed component of the physical
gravity-gradient contribution and return a scalar field. The full `GG`
selector continues to return all nine tensor components.

Extensive values and physical contributions require `aggregation="sum"`.
Intensive values require `aggregation="mean"` and use source volumes as
weights. Empty sum cells contain zero. Empty mean cells contain NaN and are
marked invalid by `source_field_result`; the direct field overload retains the
native field value but has no mask return. `norm` or `projection` is optional
and explicit, and the two cannot be combined. Vector norms are Euclidean;
tensor norms are `frobenius` or `spectral`. Vector projections use a real,
finite, non-zero direction; tensor projections use two such directions. These
operations apply after source components have been summed into cells.

For a `SourceContributionResult`, `source_field` maps the dynamic
`complex_contribution` at every stored frequency when `frequency_hz` is
omitted. Supplying `frequency_hz` selects one exact stored frequency; values
between stored frequencies are not interpolated. The result itself retains
every frequency. `phase` and `interference_fraction` are available through
`source_field_result`, which retains a mask on the same frequency axis: phase
is undefined for zero cell contributions, while interference is undefined
when the corresponding global component total is exactly zero. Those derived
undefined values are stored as zero with `valid=False`. The complex cell
components are summed before phase, interference, norm, or projection is
evaluated.

Use `spatial_projection="xy"`, `"xz"`, or `"yz"` to sum along the omitted
z, y, or x axis while retaining that axis's full edge range as one cell.
Intensive static quantities keep their volume-weighted mean by summing the
quantity-volume products and volumes across the projected axis. Vector and
tensor components are summed before optional norm or component projection.
The projection plane, collapsed axis, full projected range, and original grid
edges are recorded in result metadata.

Static scalar components have shape `(1, Nx, Ny, Nz)`; dynamic components have
shape `(Nf, Nx, Ny, Nz)`. Vectors add a final component axis of length 3, and
tensors add final axes `(3, 3)`. A vector contains the x, y, and z scalar
fields on aligned axes; a tensor contains all nine `(i, j)` scalar fields on
aligned axes. The first axis is a structural singleton at 0 s for a static
field, not a time series; for dynamic fields it contains the selected stored
frequencies in hertz. Spatial axes are the physical x/y/z cell centres in
metres. The result helper also returns `.valid` with the same component layout
and immutable `.metadata` containing exact edges, frame, coordinate origin,
target, quantity, units, selected frequencies and phase convention, softening,
and available source provenance. The native GWexpy field object itself has no
new read-only guarantee.

## Complex source contributions

`source_contributions` starts from `DensityPhasors`, whose positive-frequency
values are peak complex amplitudes for δρ. It forms source mass phasors from
`density_kg_m3 * volumes_m3` and computes `potential`, `acceleration`, or
`gravity_gradient`. `GG`, `gg`, and `gradient` are fixed aliases for the
canonical `gravity_gradient` observable. Display it as “gravity gradient” or
“GG”; the word `gradient` by itself can also refer to the older internal
kernel naming.

Pass targets as bare SI coordinates shaped `(3,)` or `(M, 3)`, or as the
numeric `gragra.targets.points.TargetPoints` type. This low-level type is
distinct from the high-level `gragra.targets.TargetPoints` container. The
coordinates must already use the phasors' frame and origin; this call does not
transform frames or adjust phase origins. `softening_m` must be finite and
non-negative.

By default all sources form one group named `"all"`. Supply one
non-negative integer `group_labels` value per source to form groups ordered by
their sorted unique labels, or pass `radial_edges_m` to group by target-specific
source distance; the two options are exclusive. Radial groups order underflow,
finite bins, then overflow. Bins are half-open `[left, right)` except that the
final right edge belongs to the last bin. Each source belongs to exactly one
group for each target, and the original-label mapping or radial edges are
retained. Source indices are stable ordered positions. Original source IDs are
preserved when supplied, and no fabricated ID is assigned otherwise.

The result retains the original frequency axis, source and target positions,
IDs, group mapping, frame and origin, phase reference, Fourier convention,
and softening. `source(id)`, `group(id)`, and `.total` produce native
`FrequencySeriesMatrix` values with rows for targets, columns for observable
components, and frequencies in Hz. Potential has one component, acceleration
has three, and gravity gradient has all nine tensor components. For NumPy
extraction, `to_array(kind=..., layout=...)` requires `kind="source"`,
`"group"`, or `"total"` and `layout="sample_first"` or `"target_first"`;
component axes stay last. Source values are generated through bounded source
blocks and are not eagerly materialized as an `F × M × N` array. Calling
`to_array(kind="source", ...)` explicitly materializes that array. The numeric
snapshots and extracted arrays are read-only; native matrices have no new
read-only guarantee.

`source_index` is the stable source position and remains separate from any
input `source_id`. Global input metadata is available once through
`global_metadata`; source-specific metadata remains associated with its
source. Neither field assigns an ID that was absent from the input.

## Contribution statistics

`contribution_statistics(result)` computes componentwise group amplitude
share `|a_j| / Σ_k |a_k|`, phase `arg(a_j)`, cumulative complex sum, and
interference share

\[
I_j = \frac{\operatorname{Re}(a_j\,\overline{a_{total}})}
           {|a_{total}|^2}.
\]

Interference shares are signed and dimensionless, and may be negative or
greater than one. Exact zero denominators produce a zero placeholder and a
false validity mask; no epsilon is added. Group statistics are available as
native matrices and as NumPy arrays only when the caller selects an explicit
layout. The result fields are
`amplitude_share`, `phase_rad`, `cumulative_complex`, `interference_share`,
and optional `rms`. For each frequency statistic, `series(group_id)` returns
the group's native matrix, `validity(group_id)` returns its mask, and
`to_array(group_id=..., layout=...)` extracts values. The mask's array
extraction also requires `layout`.

Weighted RMS values are optional. Pass a finite, non-negative,
dimensionless `frequency_weights` vector with at least one positive entry.
Weights are used as supplied: there is no automatic normalization and no
frequency-spacing factor. For each group,

\[
R_j = \sqrt{\tfrac{1}{2}\sum_f w_f |a_j(f)|^2}, \qquad
R_{total} = \sqrt{\tfrac{1}{2}\sum_f w_f |\sum_j a_j(f)|^2}.
\]

`rms_share` is `R_j / Σ_k R_k`, while `total_ratio` is `R_j / R_total`; these
are different ratios. This statistic is a weighted spectral quadratic norm
of peak phasors. Its long-time RMS interpretation assumes distinct positive
frequencies are orthogonal; it is not a general finite-window RMS or a PSD
integral.

`directional_contribution_statistics` accepts results with matching physical
metadata and finite, non-negative direction weights with a positive sum. It
normalizes the weights internally, so they need not already sum to one. It
averages each group/total cross term and total power across directions, then
forms the interference ratio. This describes an independent-direction
ensemble; it is not the coherent field from summing responses across
directions. Per-input direction metadata and source provenance remain in input
order.

## Dynamic radial histograms

`dynamic_radial_histogram` converts distance groups from
`source_contributions(..., radial_edges_m=...)` into one real-valued GWexpy
`Histogram`. It uses the exact stored radial edges and the result's
underflow/overflow groups; it does not regroup source values. Select an exact
stored `frequency_hz`, `target_id`, and observable `component`. A scalar
potential has one component and can omit `component`; vector and tensor
responses require labels such as `"x"` or `"xy"`.

The `quantity` selectors are `complex_real`, `complex_imag`, `amplitude`,
`phase`, `amplitude_fraction`, `interference_share`, `cumulative_real`, and
`cumulative_imag`. Real and imaginary values come from the complex sum within
each radial group. `amplitude` is the magnitude of that sum, and `phase` is its
argument. The fractions use `ContributionStatistics` for one result and
`DirectionalContributionStatistics` for an ensemble. Amplitude fraction
normalizes each group amplitude by the sum of amplitudes over all groups;
interference share uses the complex correlation with the total response. It is
signed and dimensionless, so it can be negative or greater than one. The valid
shares sum to one when their denominator is nonzero. Cumulative real and
imaginary selectors separately
show the near-to-far cumulative complex response. Their histogram finite bins
are cumulative values at each outer bin edge; underflow and overflow store the
cumulative values at the corresponding radial boundaries.

Phase is undefined for a zero group response, and interference share is
undefined when the selected total response is exactly zero. Undefined bins
store zero and are marked false in `.valid`; `.underflow_valid` and
`.overflow_valid` mark the flow groups. These masks keep a valid zero distinct
from an undefined ratio. Amplitude and RMS fractions are invalid when all
groups have zero amplitude under their respective denominator.

`rms_amplitude_fraction` uses the weighted spectral RMS share from
`contribution_statistics`. Pass `frequency_weights`; frequency weights are
used as supplied. For an ensemble of directions, pass a sequence of compatible
radial results and `direction_weights`, which are normalized internally.
RMS histograms have no single `frequency_hz`; their metadata retains the
frequency and weight vectors used.

The same compatible-result form is available for `interference_share`, with
one exact `frequency_hz`. For each group it averages the group-total cross term
and total response power over directions, then divides:

\[
S_j(f)=\frac{\sum_d p_d\,\operatorname{Re}[a_j(f,d)a_{\mathrm{tot}}^*(f,d)]}
{\sum_d p_d\,|a_{\mathrm{tot}}(f,d)|^2}.
\]

The direction weights are normalized to sum to one. This is a ratio of
direction-averaged terms, not the average of per-direction ratios. The
`normalized_direction_weights` metadata records the weights used by the
statistic.

```python
import numpy as np
from gragra.fields import DensityPhasors
from gragra.plotting import dynamic_radial_histogram, source_contributions

positions_m = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
volumes_m3 = np.ones(2)
density_kg_m3 = np.array(
    [[1.0 + 0.0j, 0.0 + 1.0j], [0.5 + 0.0j, 1.0 + 0.0j]]
)
frequencies_hz = np.array([10.0, 20.0])
phasors = DensityPhasors(
    positions_m,
    volumes_m3,
    density_kg_m3,
    frequencies_hz,
    0.0,
    np.zeros(3),
    "root",
    np.zeros(3),
    ("source-0", "source-1"),
)
contributions = source_contributions(
    phasors,
    "acceleration",
    [[0.0, 0.0, 0.0]],
    radial_edges_m=[0.0, 5.0, 10.0],
)
direction_0 = contributions

# Represent a second direction with the second source shifted by pi radians.
second_phasors = DensityPhasors(
    positions_m,
    volumes_m3,
    density_kg_m3 * np.array([1.0, -1.0]),
    frequencies_hz,
    0.0,
    np.zeros(3),
    "root",
    np.zeros(3),
    ("source-0", "source-1"),
)
direction_1 = source_contributions(
    second_phasors,
    "acceleration",
    [[0.0, 0.0, 0.0]],
    radial_edges_m=[0.0, 5.0, 10.0],
)
radial = dynamic_radial_histogram(
    contributions,
    "interference_share",
    target_id="target-0",
    frequency_hz=10.0,
    component="x",
)

directional_interference = dynamic_radial_histogram(
    (direction_0, direction_1),
    "interference_share",
    target_id="target-0",
    frequency_hz=10.0,
    component="x",
    direction_weights=[1.0, 1.0],
)
```

## Comparing and interpreting diagnostics

Conservation comparisons apply to additive quantities over the same source
domain and in the same units: source contributions, group sums, totals, and
raw integrated histogram or field values. Means use the source-volume weights
in their definition. Fractions use the whole-source signed denominator, and
per-length histograms divide by their bin widths. Do not expect normalized
values to add like raw contributions. The spatial extent matters: for a
uniform sphere of density ρ viewed from its centre, the ideal `mass_over_r2`
contribution per radial length is `4πρ` away from the origin. A plot or a
matching finite-bin sum is a diagnostic; neither demonstrates numerical
convergence.

## Spatial diagnostic maps (`plot_map`)

`plot_map` draws source diagnostics across space on their actual discrete
coordinates:

- Quantities: `kernel_norm` (1/r^2), `delta_a_norm` (||Gm(r_S - r_T)/r^3||),
  Cartesian vector components `delta_a_x`, `delta_a_y`, `delta_a_z`, and
  directional projection `delta_a_projection` along a user-specified `direction`.
- Conservation and physical oracle: The sum of discrete contributions
  $\sum_i \delta\mathbf{a}_i$ agrees with the forward `PointSumModel`
  evaluated acceleration in the chosen `frame` within floating-point tolerance
  (`rtol=1e-12`, `atol=1e-20`). The two are not bitwise identical. Thinning
  (`max_points`) applies to rendering only and does not alter the underlying physics.
- Views: `"3d"` (3D scatter), or 2D Cartesian projections `"xy"`, `"xz"`, `"yz"`.
- Coordinate units: `coord_unit` sets the spatial coordinates unit (default `"m"`).
- Color scaling: `"linear"` (default), `"log"`. Signed components use
  symmetric TwoSlopeNorm centered at zero under linear scaling.
- Point limit: `max_points` (default `None`, all points displayed). Specifying an
  integer thins display points while preserving the total physical sum.
- Physical scaling: `delta_a_norm`, `delta_a_x`, `delta_a_y`, `delta_a_z`, and
  `delta_a_projection` are accelerations and multiply by `G_SI`. `kernel_norm`
  is the geometric 1/r^2 and involves neither G nor the element mass.

### Numerical limits

- The acceleration quantities (`delta_a_*`) compute r^3 internally. For
  r of about 1.7e-108 to 6.1e-103 m, r^3 is subnormal; for r above about
  5.6e102 m, r^3 overflows float64. In these ranges the result may be
  representable, but `plot_map` rejects it with `ValueError` instead of
  returning a wrong finite value.
- `kernel_norm` (1/r^2) does not use r^3, so it remains available in these ranges.
- For the `delta_a_*` quantities, an element with zero mass gives a zero value,
  and `color_scale="log"` then raises `ValueError`. Use `color_scale="linear"`
  for such data.

## Contribution histograms and cumulative profiles

In addition to scalar geometric diagnostics, `gragra.plotting` provides two
functions focused specifically on physical gravitational acceleration contributions:

### 1. Contribution histograms (`plot_contribution_hist`)

`plot_contribution_hist` bins the gravitational acceleration contribution
$\Delta\mathbf{a}_i = G m_i (\mathbf{r}_i - \mathbf{r}_T) / r_i^3$ towards a selected
target:

- Supported quantities: `delta_a_norm` (magnitude), vector components `delta_a_x`,
  `delta_a_y`, `delta_a_z`, or projected along a specified `direction`.
- Bins can be an integer count, explicit boundary sequence, or `"auto"`.
- Total vector conservation: The sum of all element contributions agrees with
  the forward evaluated model acceleration within floating-point tolerance.

```python
from gragra.plotting import plot_contribution_hist

# Histogram of x-component gravitational acceleration contributions towards target A
fig_chist = plot_contribution_hist(setup, "delta_a_x", target_id="A", bins=30)
```

### 2. Cumulative distance profiles (`plot_cumulative_contribution`)

`plot_cumulative_contribution` computes the radial cumulative sum:

$$
\mathbf{a}(<R) = \sum_{r_i \le R} \Delta\mathbf{a}_i
$$

as a function of the distance $R$ from the target point. This diagnostic allows
researchers to identify the distance band that dominates Newtonian noise.
The final value $\mathbf{a}(<R_{\max})$ converges to the total forward acceleration.

```python
from gragra.plotting import plot_cumulative_contribution

# Plot cumulative acceleration profile towards target A
fig_cum = plot_cumulative_contribution(setup, target_id="A")
```

## Analytical shape rendering (`plot_shape`)

`plot_shape` visualizes the 3D analytical geometry of source bodies on Matplotlib 3D axes:

- `Cuboid`: Renders the 6 rectangular boundary faces as 3D surface polygons.
- `Cylinder`: Renders the cylindrical side surface and circular end caps.
- `SphericalShell`: Renders concentric spherical boundary surfaces.
- Cavity models, moving bodies, and unstructured meshes currently raise `NotImplementedError`.

```python
import matplotlib.pyplot as plt
import astropy.units as u
from gragra.geometry import Cuboid, Frame
from gragra.plotting import plot_shape

world = Frame("world")
cuboid = Cuboid(
    center=[0.0, 0.0, 0.0] * u.m,
    lengths=[2.0, 3.0, 4.0] * u.m,
    frame=world,
)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
plot_shape(cuboid, ax=ax, color="skyblue", alpha=0.6)
```

## Result visualization (`plot_result`)

`plot_result` provides visualization for calculation results stored in `EvaluationResult`:

- **Static spatial fields**: Renders 2D slice heatmaps of scalar potential or
  acceleration norm on a Cartesian grid (`plane="xy"`, `"xz"`, or `"yz"`).
- **Time series**: Draws Cartesian observable components versus time.
- **Spectra / Fourier transforms**: Graphs amplitude spectra, Fourier series, or phasors versus frequency.

```python
import numpy as np
import astropy.units as u
from gragra.gravity_sources import PointMassCloud, rigid_motion
from gragra.gravity_sources.trajectory import PrescribedRigidTransform
from gragra.plotting import plot_result
from gragra.results import TimeSampling
from gragra.results._time_grid import build_uniform_time_s
from gragra.targets import TargetPoints

# 1. Plot 2D slice of static field evaluation on a Cartesian grid
grid_coords = (
    np.array(
        [[-2.0, -2.0, 0.0], [-2.0, 2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0]]
    )
    * u.m
)
grid_targets = TargetPoints(grid_coords, ("1", "2", "3", "4"), root)
eval_result = (
    point_sum(source, frame=root, field_definition=field)
    .at(grid_targets)
    .run()
)
fig_field, ax_field = plot_result(eval_result, plane="xy")

# 2. Plot time series response from rigid motion
epoch_gps = (1_000_000_000, 0)
time_s = build_uniform_time_s(epoch_gps, 0.1, 10)
pos = np.column_stack(
    [0.5 * (time_s - 1_000_000_000), np.zeros(10), np.zeros(10)]
)
transform = PrescribedRigidTransform(
    time_s, pos, np.repeat(np.eye(3)[np.newaxis], 10, axis=0)
)
ts_result = (
    rigid_motion(
        PointMassCloud(np.array([1.0]), np.array([[0.0, 0.0, 0.0]])),
        transform,
        frame=root,
        sampling=TimeSampling(epoch=epoch_gps, dt=0.1 * u.s, count=10),
    )
    .at(targets, observables=("acceleration",))
    .run()
)
fig_ts, ax_ts = plot_result(ts_result, target_id="A", view="line")
```

## Notes

- `xunit` sets the unit of the horizontal axis only. It must be equivalent to
  the default unit of the quantity (`kg/m3`, `m`, or `1/m2`). Converting it
  does not change volume weights. A conversion whose result overflows float64,
  or turns a nonzero value into zero or a subnormal number (underflow), raises
  `ValueError`. An exact zero stays zero.
- `frame=None` means the computation frame of the setup. For `height`, `frame`
  is the frame in which z is read. `distance` and `inverse_square` are
  frame-invariant and are always computed in the computation frame. For
  `density`, `frame` is only checked for connection and is otherwise ignored.
  In `plot_field`, `frame` is used only for the figure labels.
- `density` and `weight="volume"` need a `VolumeElementSource`, or a source
  from `gragra.gravity_sources.discretize` whose discretization records a real
  `cell_volume_m3`. The mass weights must agree with the density and volume.
  Point-mass and weighted sources, and analytic models, raise `ValueError` for
  these. Densities are never inferred from the weights.
- The volume of a discretized source is its cell volume, without boundary-cell
  correction. Its total volume can therefore differ from the geometric volume
  of the sphere. Its mass is the sum of its weights.
- `inverse_square` is the plain 1/r^2. It does not support softening. A nonzero
  separation whose square is zero or subnormal is rejected.
- Analytic models have no source elements and are rejected.
- Integer bins and `bins="auto"` widen a constant data set to `value +/- 0.5`,
  as NumPy does. Explicit edges are not widened. Complex bins are rejected.
- `bins="auto"` is limited to 1,000,000 bins, and `plot_hist` draws at most
  100,000 bins. Pass explicit edges for larger histograms.
- Values that fall outside the edges are kept in `underflow` and `overflow`.
- Calling `hist()` loads GWexpy and `matplotlib.pyplot`. Importing
  `gragra.plotting` and obtaining `hist` does not.
- `hist` and `plot_hist` show geometric and mass diagnostics only. They do not
  multiply by G, and `inverse_square` is not a gravitational acceleration.
- These diagnostics do not establish the scientific acceptance of any physical model.

