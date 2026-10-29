# Gen1 analytic gravity models

This guide covers the approved high-level models for moving point masses,
small-cavern wall motion, acoustic pressure in a halfspace, and prescribed
temperature perturbations. Each factory returns a model; `.at(targets)` fixes
target IDs and coordinate frames, and `.run()` returns SI fields with model
metadata and one application of `G_SI`. The API entries are in
[Sources, targets and geometry](../api/core.md) and
[Fields, waves and coherent](../api/physics.md).

## Frames and outputs

Geometry and source coordinates belong to the Frame supplied with the source.
`TargetPoints` retains the frame in which its positions were entered. Pass
`output_frame` to `.at()` when acceleration and gradient components should be
expressed in another connected frame. Potential is unchanged by rotation;
vectors and both gradient indices are rotated. The result context records the
output frame, while the source model keeps its own frame and geometry.

Each model rejects targets on physical boundaries where its derivatives have
one-sided limits or its native formulas are singular. Move the target to a
physical side of the boundary; increasing a numerical tolerance does not make
the boundary field defined.

## Small-cavern approximation wall contribution

`analytic.seismic.small_cavity_wall` accepts a Sphere, Cuboid, or Cylinder, one uniform
displacement vector in the geometry frame, density, and optionally a single
frequency with a reference GPS time. It evaluates only the selected wall
contribution in the small-cavern approximation. It does not add the bulk
response or a separate scattering term. Treat displacement as a first-order
linearized input whose magnitude must remain small for the approximation to
apply. The physical wall is excluded for every observable.

```python
--8<-- "snippets/analytic_gen1_seismic_models.py"
```

The snippet also exercises the causal point-force model described below.

## Causal point force

`ForceHistory` stores a real scalar force after a GPS onset. Its knot values are
elapsed seconds from that onset. Each coefficient row gives ascending powers
of the interval-local coordinate

\[
u = (s-t_i)/(t_{i+1}-t_i), \qquad s=t-t_{\rm onset},\quad 0 \leq u \leq 1,
\]

with coefficients in newtons. Intervals are left-closed and right-open; the
last knot belongs to the `after` region. Adjacent intervals must join with
continuous force and first time derivative, including the zero-force onset and
the chosen final continuation. Before onset the force is zero. The `after`
option chooses zero force or a held final force after the last knot.

`analytic.seismic.point_force` evaluates this causal source in an infinite,
homogeneous elastic medium using a positive P-wave speed `vp`. Its time axis
comes from a `TimeSampling` object. The force onset and sampling epoch are
separate coordinates; retarded time includes source-target travel time.
The returned matrix stores target rows, component columns, and time samples.
Construct the immutable axis with keyword arguments, for example
`TimeSampling(epoch=..., dt=0.1 * u.s, count=3)`.
At large GPS epochs a displayed `times` coordinate may round in floating
point; the canonical axis remains the integer GPS epoch pair, `dt`, and sample
count.

## Moving point masses

The `analytic.objects` motion factories use positive `PointMass` or
`PointMassCloud` inputs and Newtonian gravity:

* `constant_velocity` moves one point mass at constant velocity from its position
  at `motion_epoch`. It samples the exact translated source at each requested
  time.
* `fixed_axis_rotation` rotates a cloud at constant angular speed about a fixed axis and
  origin. The supplied cloud is its zero-angle configuration; `angle0` is the
  angle at `angle_epoch`. That epoch is independent of the sampling epoch.
* `translational_oscillation` is a first-order, uniform-translation response for a fixed
  point mass or cloud. Its displacement is a complex peak phasor at one
  positive frequency, referenced to `reference_time`.

The time-domain models use the exact `TimeSampling` coordinates and reject
target-source collisions at any evaluated sample. `translational_oscillation` is a
linearized response: it assumes displacement is small relative to each
source-target distance. The phasor convention reconstructs as
\(\mathrm{Re}[P e^{-2\pi i f(t-t_\mathrm{ref})}]\), with no additional
one-half factor.

```python
--8<-- "snippets/analytic_gen1_motion.py"
```

The examples include a time-series result and a request with rotated output
components. `motion_epoch`, `angle_epoch`, and `TimeSampling.epoch` describe
different things and must be supplied independently.

## Acoustic halfspace

`analytic.atmospheric.acoustic_halfspace` accepts a `HalfSpace`, total surface pressure
phasor, sound speed, frequency, incidence angle, azimuth, phase origin, and
reference time. The local halfspace plane is `z=0`; air is `z>0` and ground is
`z<0`. The incidence angle is measured from the surface normal and must obey
\(0 < \theta \leq \pi/2\). Its wavenumbers are
\(k=2\pi f/c_s\), \(q=k\sin\theta\), and
\(k_z=k\cos\theta\). Exact grazing incidence is valid, with \(k_z=0\) and
nonzero horizontal wavenumber. Normal incidence (`q=0`) is excluded by this
model.

`surface_pressure` is the **total** surface-pressure coefficient under the
rigid-reflection solution, including the incident and reflected air-side
waves. `phase_origin` sets the phase in horizontal coordinates. The model
returns the air and ground fields and enforces continuity at the interface;
targets on the interface itself are rejected.

```python
--8<-- "snippets/analytic_gen1_acoustic_halfspace.py"
```

## Rigid acoustic cavity modes

`analytic.atmospheric.acoustic_cavity_mode` evaluates one real positive-frequency
Neumann mode for a rigid cavity. The constant zero mode is excluded. In a
Sphere, `SphereAcousticMode(l, m, n)` uses a real spherical harmonic and the
`n`th strictly positive root of the derivative of the spherical Bessel
function of degree `l`. The positive normalization makes the mode's spatial
RMS one. Thus `pressure_amplitude` is a spatial-RMS peak phasor; a sinusoid's
time RMS is its amplitude divided by \(\sqrt{2}\). The frequency is derived
from sound speed and the mode wavenumber; it is not an independent input.

```python
--8<-- "snippets/analytic_gen1_cavity_sphere.py"
```

The sphere calculation uses a closed-form radial solution and rejects points
on its physical wall.

### Cuboid modes

`CuboidAcousticMode(nx, ny, nz)` uses nonnegative cosine indices. Let `Lx`,
`Ly`, and `Lz` be the full edge lengths and let coordinates be measured from
the cuboid center in its local frame. The real mode is

\[
\Psi=N\prod_{j\in\{x,y,z\}}\cos\!\left(\frac{n_j\pi(x_j+L_j/2)}{L_j}\right),
\qquad
k^2=\sum_j\left(\frac{n_j\pi}{L_j}\right)^2,
\qquad N=2^{r/2},
\]

where `r` counts nonzero indices. This normalization gives spatial RMS one.
The all-zero constant mode is unsupported. The derived frequency is
`sound_speed * k / (2*pi)`. The cuboid's faces are excluded for potential,
acceleration, and gradient.

```python
--8<-- "snippets/analytic_gen1_cavity_cuboid.py"
```

### Cylinder modes

`CylinderAcousticMode(m, n, q, azimuth="cos")` describes a real cylindrical
mode in local coordinates centered on the cylinder. `m` is the nonnegative
azimuthal order, `n` selects the positive root \(\alpha_{mn}\) of
\(J'_m(\alpha)=0\), and `q` is the nonnegative axial cosine index. For `m>0`,
`azimuth` selects `cos(m*phi)` or `sin(m*phi)`; the sine choice is undefined
for `m=0`. The mode shape is proportional to

\[
J_m(\alpha_{mn}r/R)\,A_m(\phi)\,
\cos\!\left(q\pi(z+H/2)/H\right),
\qquad
k^2=(\alpha_{mn}/R)^2+(q\pi/H)^2,
\]

where \(R\) and \(H\) are the radius and height. Its normalization is
geometry-specific and gives spatial RMS one. For `n>=1`, it is
\[
N=[\eta_q\eta_m J_m(\alpha_{mn})^2(1-m^2/\alpha_{mn}^2)]^{-1/2},
\]
where \(\eta_q=1\) at `q=0` and \(1/2\) otherwise, while \(\eta_m=1\) at
`m=0` and \(1/2\) otherwise. The radial-constant case `n=0` is allowed only
for `m=0, q>0`; it uses \(\sqrt{2}\) normalization. The constant `m=n=q=0`
mode is unsupported. The derived frequency is again `sound_speed * k / (2*pi)`.
The side wall, caps, and rims are excluded for all observables.

```python
--8<-- "snippets/analytic_gen1_cavity_cylinder.py"
```

The complex pressure coefficient multiplies the real, spatially normalized
mode. Time dependence reconstructs as
\(\mathrm{Re}[P e^{-2\pi i f(t-t_\mathrm{ref})}]\), without a one-half
factor. Each geometry has its own real-mode normalization; do not reuse a
complex-mode norm or infer frequency from a caller-supplied value. The Cuboid
example requests potential only to keep transform work bounded; the Sphere
and Cylinder examples request all three observables. Every request uses the
same fixed accuracy policy, regardless of the number of observables.

`AcousticCavityNumerics` contains resource ceilings only (`max_evaluations` and
`max_refinements`). They do not change the fixed Gen1 accuracy policy. For a
cavity route that needs numerical integration, `NumericalDiagnostic` separates
the quadrature/numerical error estimate from the rigorous omitted-integral
tail bound; runtime convergence uses their sum per target and observable. A
tail bound covers only the omitted integral and does not bound the total
evaluation error. High-level `run()` wraps producer failures in
`EvaluationError`; for a convergence failure, inspect `exc.cause`
(`gragra.exceptions.ConvergenceError`) and read `exc.cause.diagnostics`. A
resource-limit failure can justify a larger resource ceiling, but relaxing a
tolerance is not a recovery method.

## Isobaric temperature perturbations

`analytic.atmospheric.isobaric_temperature` applies the linearization
\[
\delta\rho=-\rho_0\,\delta T/T_0
\]
to a positive homogeneous background density `rho0` and temperature `T0`.
The finite Sphere, Cuboid, and Cylinder inputs describe a prescribed bounded
temperature anomaly in that background; they do not model sealed uniform
heating or solve a thermodynamic boundary-value problem. Targets on the
material boundary are rejected for potential, acceleration, and gradient.

`TemperatureFourierMode` uses
\(\delta T=\Delta T\cos[k\cdot(x-x_0)]\) for Static output and
\(\widetilde{\delta T}=\widetilde{\Delta T}e^{+ik\cdot(x-x_0)}\) for a
Phasor output. The wavevector must be nonzero. The periodic potential uses a
zero-mean gauge with its zero Fourier mode excluded. Phasors are peak
coefficients and need no factor of one-half.

`TemperatureGaussian` has all-space support and a zero-at-infinity potential
gauge. Its center limits are finite:
\(\Phi(0)=-4\pi G\rho_c\sigma^2\), \(a(0)=0\), and
\(\Gamma(0)=-4\pi G\rho_c I/3\), where
\(\rho_c=-\rho_0\Delta T/T_0\). Far from the Gaussian, the potential and
acceleration approach the monopole field of
\(M=(2\pi)^{3/2}\rho_c\sigma^3\).

```python
--8<-- "snippets/analytic_gen1_temperature.py"
```

### Real temperature grids

`TemperaturePerturbationGrid` stores real Kelvin values with shape `(Nx, Ny, Nz)` and
explicit strictly increasing length-Quantity edges. It places one signed
mass element at every cell center, using the product of the three edge widths
as volume. Flattening is C order, so z is the fastest-changing index.

Use the grid's public `field_definition` and
`point_sum_provenance(rho0=..., T0=...)` to connect it to the existing
`point_sum` model. Keep `frame=grid.frame`; this frame locates source cells.
`output_frame` on `.at()` controls acceleration and gradient components and
does not change source coordinates. This is a cell-center approximation for
the supplied grid, not an analytic replacement for a plane wave, Gaussian, or
finite-body model.

```python
--8<-- "snippets/analytic_gen1_temperature_grid.py"
```

## Numerical diagnostics and edited results

For numerical cavity models, `NumericalDiagnostic` reports one target and
observable in SI units. A numerical or quadrature error estimate (excluding
the integral tail) and a rigorous integral-tail bound are distinct fields.
The runtime stopping rule uses their sum for each requested target and
observable. The estimate is not itself a proved bound; the tail bound covers
only the omitted integral. Neither field alone proves agreement with the exact
physical field. A successful diagnostic describes the native evaluation that
produced the result. It does not certify values a caller edits afterward.

Result payloads can remain mutable where the native matrix API permits it.
The high-level result checks that each payload still matches its declared
observable, units, IDs, component labels, and canonical time or frequency
axis when accessed. Finite edits that preserve those contracts remain visible
and are not recomputed or physically revalidated. `compare` requires exact
canonical axes and does not resample or interpolate results.
