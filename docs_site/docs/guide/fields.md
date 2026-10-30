
# Displacement and density-perturbation fields

## From a displacement field to a density perturbation

A displacement field `ξ(r)` perturbs the background mass density as
`δρ = −∇·(ρ₀ξ)`. `gragra.fields` converts a displacement field
(`DisplacementGrid`, `UnstructuredDisplacementField`, or a surface
counterpart) into the mass elements that feed `WeightedPointSource` (see
[Point sources](point-sources.md)). The divergence decomposes into two
physically distinct contributions, described next.

This page covers density changes derived from a displacement field. For the
separate prescribed temperature-anomaly models and the real Kelvin grid route,
see [Gen1 analytic models](gen1-analytic-models.md#isobaric-temperature-perturbations).

## The bulk term

For a uniform background density `ρ₀`, `δρ = −ρ₀∇·ξ` is the *bulk* term:
proportional to the trace of the strain tensor, evaluated at every interior
point of the volume.

### A rigid translation has no bulk term

**A rigid translation has no bulk term at all.** The bulk density
perturbation is δρ = −∇·(ρ₀ξ), so for a rigidly translated body — ξ
constant — the divergence is identically zero and the bulk path returns
exactly `0`. The entire gravitational signal lives in the surface term.
Computing only the bulk contribution for a moving rigid body gives you a
silent, physically wrong zero.

## The surface term

At a boundary or interface where displacement or density is discontinuous,
the mass perturbation is a surface term proportional to the displacement
(or density jump) times the local surface normal and area — not to the
divergence, which is not even defined across a discontinuity. A closed
surface mesh, correctly oriented, converts to a `SurfaceMassSheetSource`
(see [Volume and surface sources](continuous-sources.md)).

## Finite piecewise-tetra bodies

`piecewise_tetra_linear_sources` accepts a conforming P1 tetra displacement
field and one non-negative density per cell. It computes a cell-volume bulk
term and a face-interface term, including the outer boundary against vacuum.
The two terms must be summed: `bulk_interface_response = bulk_response +
interface_response`.

`compare_tetra_linearized_representations` compares that summed response with
the centroid mass-displacement response. They are alternative first-order
representations, so neither response is added to the other. The comparison is
G-free; multiply only an externally selected physical observable by `G_SI`.

Faces are shared by vertex ID rather than geometric vertex merging. Duplicate
tetrahedra, faces with more than two owners, same-side face owners, and an open
or inconsistent outer boundary are rejected. Global self-intersection
detection is outside this narrow mesh contract.

## Equivalence conditions for the bulk and surface forms

A bulk-only or surface-only calculation agrees with the exact field of a
displaced or perturbed body only when **all six** of the following hold:

1. The surface is a closed manifold.
2. Triangle winding is consistent across the surface.
3. Outward normals have been verified, not merely assumed.
4. The density-perturbation sign convention matches the bulk convention.
5. The bulk volume and the surface mesh describe the same geometry.
6. The background density is truncated outside the region of interest.

A broken equivalence condition on real input produces a suspicious
near-zero or exactly-zero residual when compared to a reference — that is
the diagnostic working as intended, not a false alarm. See
[Troubleshooting](../help/troubleshooting.md#an-all-zero-result) for the
exact same statement phrased for a reader debugging an unexpected zero.

## Strain is not displacement

**Strain alone is not enough.** The bulk term needs only `∇·ξ` — the trace
of the strain tensor — but the surface term needs `ξ·n̂` itself. If your
solver hands you strain and not displacement, you can compute the bulk
contribution and nothing else. gragra has no `strain` API and cannot
reconstruct displacement from strain.

## Non-uniform background density

When `ρ₀` varies in space, an additional advection term involving `∇ρ₀`
enters the bulk expression — `DifferentiableBackgroundDensityModel` and
`DifferentiableDiscreteBackgroundDensity` carry the gradient needed for
this term. Passing a uniform (scalar) `ρ₀` skips it entirely, which is
correct only when the background truly is uniform over the region being
integrated.

## Status and verification

### `displacement_transfer_acceleration` { #status-displacement-transfer-acceleration }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Experimental |
    | **Verified against** | closed-form analytic solution; invariance property (bulk+surface vs. dipole-form equivalence, convergence with mesh distance) |
    | **Agreement** | Exact sign fixed point; equivalence between the closed dipole sum and the bulk+surface mesh path converges with distance from the source region, and is expected to diverge for an open (non-manifold) sub-volume embedded in surrounding material — see [Equivalence conditions](#equivalence-conditions-for-the-bulk-and-surface-forms) |
    | **Test evidence** | `tests/test_observables_dipole.py::test_matches_g_si_times_dipole_sum`, `tests/test_observables_dipole.py::test_sign_fixed_golden`, `tests/test_field_dipole_diagnostics.py::test_equivalence_cube_closed_surface_converges_with_distance`, `tests/test_field_dipole_diagnostics.py::test_nonequivalence_open_subvolume_is_expected` |
    | **Conditions** | The six equivalence conditions above must all hold for the bulk+surface decomposition to agree with this function's direct dipole-sum result |
    | **Not verified** | Agreement for a body whose displacement field is known only as strain, not displacement — see [Strain is not displacement](#strain-is-not-displacement) |
    | **Known limits** | An open sub-volume embedded in surrounding material is not expected to satisfy the equivalence — this is documented, not a bug |
    | **Scope** | Research interface; input schema and return-shape changes remain possible |

Both functions evaluate only the first-order source-coordinate derivative for
discrete, non-negative point-mass elements. They accept real or complex
displacements and return read-only arrays. `backend="numpy"` is the reference;
`backend="numba"` is also available for these two derivatives, while C++ and
CuPy remain unsupported.

### Cell-local P1 density in a finite tetra body

`piecewise_tetra_p1_density_sources` accepts `(Nc, 4)` density values in each
tetra's local node order. Density traces may differ across a material face;
the builder uses their face-quadrature jump and keeps the exterior density at
zero. This is a finite, conforming P1-displacement model, not a general
higher-order or non-tetrahedral field importer.

The accepted P1-density representation checks use the NumPy reference path.
Its builder does not establish all-backend scientific equivalence; the Numba
addition concerns the underlying displacement-transfer potential and gradient.

### `displacement_transfer_potential` and `displacement_transfer_gravity_gradient` { #status-displacement-transfer-derivatives }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Experimental |
    | **Verified against** | closed-form analytic solution; independent numerical oracle; invariance property |
    | **Agreement** | direct identities use `rtol=1e-12`; finite-difference checks use `rtol=1e-5` after scale normalization |
    | **Test evidence** | `tests/test_kernel_displacement_contracts.py::test_displacement_kernel_axis_goldens`, `tests/test_kernel_displacement_contracts.py::test_displacement_contracts_match_dimensionless_source_shift_finite_difference`, `tests/test_displacement_observables.py::test_displacement_observable_axis_goldens_are_g_scaled_and_read_only` |
    | **Conditions** | discrete point masses and first-order displacement; NumPy is the reference and Numba is an optional CPU backend |
    | **Not verified** | equivalence with a general bulk-plus-surface field representation |
    | **Known limits** | Plummer softening makes the gradient trace generally nonzero |
    | **Scope** | potential `(M,)` and gravity gradient `(M, 3, 3)` only; C++ and CuPy remain unsupported |

### `displacement_field_acceleration` { #status-displacement-field-acceleration }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Experimental |
    | **Verified against** | closed-form analytic solution; invariance property (time-axis and chunk-size handling) |
    | **Agreement** | Matches independent linearized and central-difference formulas on synthetic GLL data; zero-displacement and rigid-translation cases are exact |
    | **Test evidence** | `tests/test_specfem_reader.py::test_displacement_acceleration_matches_independent_linearized_formula`, `tests/test_specfem_reader.py::test_displacement_acceleration_matches_independent_central_difference`, `tests/test_specfem_reader.py::test_displacement_acceleration_is_exact_zero_for_zero_displacement` |
    | **Conditions** | A `SpecfemGLLDataset` with the documented time-sampled schema and finite target points |
    | **Not verified** | General solver layouts, streaming large files, and targets inside the material without the local term |
    | **Known limits** | The reader loads snapshots into memory and rejects unsupported PML, mixed-material, and non-elastic inputs |
    | **Scope** | Research interface; broader solver layouts require separate implementation work |

## Visualizing fields with `plot_field` { #visualizing-fields }

> `gragra.plotting.plot_field` is present in the package but outside the
> documented public API scope for this release. New examples should visualize
> values from the documented calculation API with Matplotlib directly.

`gragra.plotting.plot_field` plots a `DensityPerturbationGrid` or
`DisplacementGrid` faithful to native grid coordinates without mutating the
field object.

### Key design contracts

- **Frame ownership**: `frame` is a required keyword argument specifying the
  physical coordinate frame in which the grid axes are defined. Field objects
  themselves do not own a `Frame`.
- **2D slices**: `plane` (`"xy"`, `"xz"`, or `"yz"`) and `index` (integer grid index
  along the orthogonal axis) must be specified together as a pair. This displays an
  exact slice through the native grid plane (not an arbitrary rotated projection).
- **3D representation**: Omitting `plane` produces a 3D scatter plot (for density
  or scalar displacement components) or a true-scale 3D quiver plot.
- **Complex decomposition**: For complex fields, `part` must be specified as
  `"real"`, `"imag"`, `"abs"`, or `"phase"`.
    - Phase $\phi = \operatorname{Arg}(z) \in (-\pi, \pi]$ is normalized such that exact
      $-\pi$ maps to $+\pi$, zero amplitude is masked, and the cyclic `twilight` colormap is applied.
- **True-scale vector quiver**: Arrows depict true physical displacement vectors
  ($\mathbf{x}_{\mathrm{end}} = \mathbf{x}_{\mathrm{start}} + \boldsymbol{\xi}$).
  For 2D slices, in-plane components are displayed with equal aspect ratio.
- **Microscopic displacement visibility**: When displacement amplitudes
  $\|\boldsymbol{\xi}\|$ are very small compared to the domain extent
  (e.g., $\sim 10^{-6}\,\mathrm{m}$ across $\sim 100\,\mathrm{m}$), true-scale
  arrows may appear as dots without visible extent. Rather than introducing
  unphysical auto-scaling, examine the spatial distribution using component
  scalar plots (`component="x"`, `part="abs"` or `component="norm"`).
- **Vector norm contract**: For complex displacements, `component="norm"` supports:
    - `part="real"`: $\|\operatorname{Re}(\boldsymbol{\xi})\|_2$
    - `part="imag"`: $\|\operatorname{Im}(\boldsymbol{\xi})\|_2$
    - `part="abs"`: $\sqrt{\sum_i |\xi_i|^2}$
    - `part="phase"` is explicitly disallowed.
- **Deterministic downsampling**: `max_points` limits displayed points or vectors
  without shrinking the spatial domain bounds or mutating the field.

### Usage examples

```python
import numpy as np
from astropy import units as u
from gragra.fields.regular_grid import DensityPerturbationGrid, DisplacementGrid
from gragra.geometry import Frame
from gragra.plotting import plot_field

root = Frame("root")
ground = Frame("ground", np.zeros(3) * u.m, np.eye(3), root)

# 1. Density perturbation 2D slice
x = np.linspace(-50.0, 50.0, 11)
y = np.linspace(-50.0, 50.0, 11)
z = np.linspace(-20.0, 0.0, 5)
density = np.ones((11, 11, 5)) * 100.0
density_grid = DensityPerturbationGrid(x, y, z, density)

fig_slice = plot_field(density_grid, frame=ground, plane="xy", index=2, unit="m")

# 2. Complex displacement field vector quiver & scalar component
disp_harmonic = (np.ones((11, 11, 5, 3)) + 1j * np.ones((11, 11, 5, 3))) * 1e-3
disp_grid = DisplacementGrid(x, y, z, disp_harmonic)

# True-scale real-part quiver
fig_quiver = plot_field(disp_grid, frame=ground, plane="xy", index=2, part="real")

# Displacement amplitude norm
fig_norm = plot_field(disp_grid, frame=ground, plane="xy", index=2, component="norm", part="abs")
```
