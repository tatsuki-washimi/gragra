# Source diagnostics and visualization

`gragra.plotting` provides high-level diagnostic and visualization routines for
discrete source setups, geometric shapes, and calculation results:

- `hist` and `plot_hist`: Histogram one per-element diagnostic (density, height,
  distance, or inverse square) as a GWexpy `Histogram` or step plot.
- `plot_contribution_hist`: Histogram directional or magnitude gravitational
  acceleration contributions from discrete elements.
- `plot_cumulative_contribution`: Plot the radial cumulative acceleration profile
  $a(<R)$ versus distance from a target.
- `plot_map`: Spatial 3D scatter or 2D projection map of element contributions.
- `plot_shape`: 3D surface mesh or wireframe rendering of analytical shapes.
- `plot_result`: Visualizes fields, time series, or spectra from an `EvaluationResult`.

## Histograms (`hist` and `plot_hist`)

Four quantities are available: `density` (kg/m^3), `height` (the z coordinate
in `frame`), `distance` (from one target), and `inverse_square` (1/r^2 from one
target). `weight="count"` counts elements, and `weight="volume"` sums their
volumes in m^3.

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

