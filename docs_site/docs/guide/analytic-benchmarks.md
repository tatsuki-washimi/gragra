
# Analytic benchmarks

## Why closed-form solutions come first

Analytic cross-validation against a closed-form solution is gragra's
primary test strategy — finite-difference checks are auxiliary, used only
where no closed form exists (see [How gragra is verified](../verification/index.md)).
`gragra.analytic` exists to give both the test suite and you, the user, a
known-correct reference to check a discretized or numerical setup against
before trusting it on a problem with no closed form.

## G-free vs SI Observable API Architecture

`gragra` provides the legacy numerical analytic tiers and a high-level C5
uniform-sphere model:
1. **G-free geometric tier (`gragra.analytic.*_g_free`)**:
   Computes purely geometric contractions ($K = \int \dots$) where gravitational constant $G$ is excluded.
   Returns immutable `GFreeGravityValues` bundles (`potential` in $\mathrm{kg/m}$, `acceleration` in $\mathrm{kg/m^2}$, `gradient` in $\mathrm{kg/m^3}$) or `SphericalShellPComponents`.
2. **Physical SI observable tier (`gragra.observables.*`)**:
   Public entry points that multiply by $G_{\mathrm{SI}}$ exactly once.
   Returns immutable `GravityObservables` (`potential` in $\mathrm{m^2/s^2}$, `acceleration` in $\mathrm{m/s^2}$, `gradient` in $\mathrm{s^{-2}}$).
   Conversions from G-free to SI can also be explicitly performed via `to_gravity_observables(gfree_values)`.
3. **High-level model (`gragra.analytic.objects.uniform_sphere`)**:
   Accepts the pure `geometry.Sphere` and a density Quantity, then uses the
   shared `.at(targets)`, `validate()`, and `run()` workflow. It returns SI
   Quantities and applies `G_SI` once.

## Available closed forms

### Uniform sphere

- `uniform_sphere_potential`, `uniform_sphere_acceleration`, `uniform_sphere_gradient`: Textbook interior and exterior field of a uniform-density sphere (G-free).
- `uniform_sphere_observables_g_free`: Bundled G-free return (`GFreeGravityValues`).
- `uniform_sphere_observables`: Public SI observable entry point (`GravityObservables`).
- `gragra.analytic.objects.uniform_sphere`: High-level analytic model with
  Quantity inputs and `EvaluationResult` output.

### Uniform cuboid (rectangular prism)

- `cuboid_potential_g_free`, `cuboid_acceleration_g_free`, `cuboid_gradient_g_free`: Full 3D closed-form solutions for potential, acceleration, and the symmetric $3 \times 3$ gravity gradient tensor.
- `cuboid_observables_g_free`: Bundled G-free return.
- `cuboid_observables`: Public SI observable entry point.
- **Formulation**: Antiderivatives of $1/R$, $-\nabla(1/R)$, and $\nabla\nabla(1/R)$ evaluated across the 8 prism vertices (Nagy 1966, Nagy et al. 2000, Petrović 1996). Interior points strictly satisfy $\mathrm{tr}(\mathbf{\Gamma}) = -4\pi G\rho$; exterior points satisfy Laplace's equation $\mathrm{tr}(\mathbf{\Gamma}) = 0$. Material boundary points (surface faces, edges, vertices) are strictly rejected with a clear `ValueError` because derivatives of the potential have jump discontinuities or logarithmic singularities across boundaries.

### Uniform cylinder on-axis

- `cylinder_potential_axis_g_free`, `cylinder_acceleration_axis_g_free`, `cylinder_gradient_axis_g_free`: Closed-form potential, acceleration, and gradient tensor along the symmetry axis ($z$-axis).
- `cylinder_axis_observables_g_free`: Bundled G-free return.
- `cylinder_axis_observables`: Public SI observable entry point.
- **Formulation**: On-axis longitudinal acceleration $a_z = -2\pi \rho [(u_2 - \sqrt{u_2^2 + R^2}) - (u_1 - \sqrt{u_1^2 + R^2})]$, gradient $\Gamma_{zz} = \partial a_z / \partial z$, and transverse gradients constrained by axial symmetry and Poisson's equation: $\Gamma_{xx} = \Gamma_{yy} = (-4\pi \rho_{\mathrm{local}} - \Gamma_{zz}) / 2$. End faces $z = \pm H/2$ and off-axis points ($x^2 + y^2 > 0$) are strictly rejected.

### Spherical shell cavity P-wave response

- `spherical_shell_p_acceleration_components_g_free` / `spherical_shell_p_acceleration_components`:
  Evaluates the 4-component acceleration decomposition at the center of a spherical cavity ($r < a$) inside a finite spherical shell ($a < r < R$) under a plane harmonic P-wave displacement $\mathbf{\xi} e^{i(\mathbf{k}\cdot\mathbf{x} - \omega t)}$:
  $$\mathbf{a}_{\mathrm{bulk}} = 4\pi G \rho [j_0(ka) - j_0(kR)] \xi \hat{\mathbf{e}}$$
  $$\mathbf{a}_{\mathrm{inner\_wall}} = 4\pi G \rho \left[ \frac{2 j_1(ka)}{ka} - j_0(ka) \right] \xi \hat{\mathbf{e}}$$
  $$\mathbf{a}_{\mathrm{outer\_boundary}} = 4\pi G \rho \left[ j_0(kR) - \frac{2 j_1(kR)}{kR} \right] \xi \hat{\mathbf{e}}$$
  $$\mathbf{a}_{\mathrm{total}} = 8\pi G \rho \left[ \frac{j_1(ka)}{ka} - \frac{j_1(kR)}{kR} \right] \xi \hat{\mathbf{e}}$$
  Based on ANNA v4 (arXiv:2603.15157v4, Section 4.1, Eqs. 47–53).
- `spherical_shell_p_infinite_difference_g_free` / `spherical_shell_p_infinite_difference`:
  Computes the difference between infinite medium total acceleration and finite shell total acceleration:
  $$\Delta \mathbf{a} = 8\pi G \rho \frac{j_1(kR)}{kR} \xi \hat{\mathbf{e}}$$
  for $k > 0$.

### Literature-sourced Rayleigh-wave half-space gravity (LIT-R)

- `rayleigh_halfspace_allspace_observables_g_free` / `rayleigh_halfspace_allspace_observables`:
  Rayleigh-wave gravity above and below a homogeneous elastic half-space
  (H19 Eq. 96-98). See `docs/design/literature-r.md` and
  the literature comparison examples shipped with this repository under `examples/`.

### Literature-sourced single-mode and reflected-P body-wave gravity (LIT-B)

Two distinct models -- do not confuse them:

- `halfspace_single_bodywave_observables_g_free` / `halfspace_single_bodywave_observables`
  (`wave_type="P"` or `"SV"`): a **single** upward-propagating plane P or SV
  wave, **without** compositing a free-surface reflection (H19 Eq. 86-88).
  This is a building block, not by itself a solution of the traction-free
  boundary condition. `z>0` and `z<0` targets may be mixed; `z==0` is
  rejected.
- `halfspace_reflected_p_above_observables_g_free` / `halfspace_reflected_p_above_observables`:
  the **full** incident-P + reflected-P + converted-SV composite above
  ground, which does satisfy the traction-free boundary condition (H19 Eq.
  89/90; independently verified against a from-scratch boundary-value solve
  in `tests/test_bodywave_literature_oracle.py`). Ground only (`z>0`).

Both take a single scalar `incidence_angle_rad` (`0 <= theta < pi/2`, exact
grazing rejected) and `azimuth_rad` per call. See
`docs/design/literature-b.md` for the full literature mapping, sign/gauge
conventions, and
`examples/literature_b.py` in this repository
for a runnable demonstration of both models.

### Legacy and auxiliary cavity / half-space solutions

- `cuboid_filled_internal_field`, `cylinder_filled_axis_field`: Legacy helpers preserved for backward compatibility.
- `sca_wall_coupling_factor` and `sca_wall_acceleration`: Small-cavern approximation formulas (Harms et al. 2022).
- `halfspace_transfer_bulk`, `halfspace_transfer_full`, `rayleigh_depth_reduction`: Elastic half-space transfer functions.

## Runnable benchmark example

See `examples/15_analytic_stage1_benchmarks.py` for a self-contained demonstration and assertion of all 4 scenarios.

## Using a benchmark to check your own setup

For the existing Sphere route, build a pure shape, call
`gravity_sources.discretize` with explicit density and Quantity spacing (see
[Geometry and discretization](geometry.md)), compute the point-source field,
and compare it with `analytic.objects.uniform_sphere` as spacing is refined.
The current convergence test is one bounded Sphere case: it checks relative
error against the closed form at the finest spacing and requires potential to
improve from coarsest to finest. It does not assert a fitted slope, require
step-by-step monotonicity, or define acceptance rules for another shape. See
[the tested Sphere conditions](geometry.md#discretization-error).

## Runnable object-API comparison

This complete example compares the high-level analytic sphere with a
discretized point-source evaluation at the same Quantity target:

```python
--8<-- "snippets/readme_analytic_benchmark.py"
```

## Assumptions behind each formula

Every closed form on this page assumes a specific idealization — uniform
density, a specific shape, a homogeneous elastic half-space, a small-cavern
approximation valid only when the cavity is much smaller than the distance
to the observation point, and so on. None of them are general-purpose
solvers; each is valid only within the assumptions stated in its own
docstring on the [API reference](../api/physics.md). Using one outside its
stated assumptions (a non-uniform density, a cavity comparable in size to
the source-to-target distance) produces a result with no guaranteed
relationship to the true field.

## Status and verification

None of the functions on this page owns a top-level 8-field Verification
Block (the current set is listed in the
[verification matrix](../verification/matrix.md));
their verification claims live in their own docstrings, rendered on the
[API reference](../api/physics.md). The independent re-derivations for the
cuboid and cylinder closed forms are recorded in
`docs/validation/cuboid_prism_closed_form_derivation.md` and
`docs/validation/cylinder_axis_closed_form_derivation.md`.
