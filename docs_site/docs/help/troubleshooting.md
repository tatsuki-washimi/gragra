# Troubleshooting

## Installation problems

**The package is not importable.** Install it from a source checkout with
`pip install -e "."` (or `pip install -e ".[dev]"` for development), as
described on [Installation](../getting-started/installation.md).

**An optional extra's dependency import fails at runtime.** The base
installation includes NumPy, Astropy, pandas, and GWexpy. Mesh import,
HDF5 field I/O, and the `lebedev` directional grid have feature extras that
declare their dependencies, though a base dependency may bring some in
transitively; see
[Choosing extras and backends](../getting-started/choosing.md) for the full
list and which functions need which.

**The C++ backend does not build.** Building it is opt-in
(`--config-settings=cmake.define.GRAGRA_BUILD_CPP=ON`) and needs a C++17
compiler. Without that flag, a source install defaults to pure Python — the
`numpy` backend is the reference implementation, and `backend="cpp"`
(like every other non-`numpy` backend) raises `ImportError` at call time
rather than silently falling back to it — see
[Compute backends](../guide/backends.md) for the exact exception it raises.

## Import and backend errors

Backends never fall back to NumPy silently: requesting an unavailable
backend always raises rather than quietly computing on NumPy. The exact
exception type for each of `"numba"`, `"cpp"` and `"cupy"` (missing
package, missing compiled artifact, stale binary, no CUDA device) is
tabulated on [Compute backends](../guide/backends.md) — check that table
before assuming a traceback means broken code rather than an unmet
optional dependency. In particular, the exception a missing CuPy install
raises is both an `ImportError` and a `ValueError`, but the class itself is
defined locally inside gragra and is **not** importable as
`from gragra import CupyImportValueError`; catch it by the built-in base
class you actually need, not by name.

## Shape, dtype and validation errors

**`ValueError` on collision or negative softening.** A source and target at
the same point with `softening_m == 0` raises `ValueError` naming the
offending index, and `softening_m < 0` always raises — see
[Coordinates, signs and units](../concepts/conventions.md) and
[Chunking, softening and tolerances](../concepts/numerics.md).

**Shape mismatch on `positions_m` / `weights_kg`.** The source and target
dataclasses require strict `(N, 3)` / `(N,)` — a bare `(3,)` array is
rejected, not silently promoted. Only the public observable functions
accept an array-like `(3,)` single target and promote it to `(1, 3)`. See
[Point sources](../guide/point-sources.md).

**`chunk_size <= 0` raises `ValueError`.** `chunk_size` must be a positive
integer or `None` (meaning "no chunking"); see
[Chunking, softening and tolerances](../concepts/numerics.md).

## Results that look wrong

### An all-zero result

Two documented cases produce an exact, physically correct zero rather than
a bug:

- **A rigidly translated body's bulk density perturbation.** δρ = −∇·(ρ₀ξ)
  is identically zero for a constant displacement ξ; the entire signal is
  in the surface term. See
  [Displacement and density-perturbation fields](../guide/fields.md).
- **A pure S wave's bulk mass weights.** `plane_wave_bulk_mass_weights`
  is proportional to k̂·ê, which is zero by construction for a pure shear
  wave in an unbounded homogeneous medium. See
  [Waves](../guide/waves.md).

If your result is zero for a reason other than these two, check the six
equivalence conditions in
[kernel_contract.md §2.2](https://github.com/tatsuki-washimi/gragra/blob/main/docs/architecture/kernel_contract.md)
(closed manifold, consistent winding, verified outward normals, sign
convention, bulk/surface geometric agreement, background-density
truncation) before assuming a bug — a broken equivalence condition also
produces a suspicious near-zero or exactly-zero residual.

### A result that changes with discretization

The C5 sphere discretizer uses cell-centre inclusion on a midpoint Cartesian
grid, so the discrete total mass approaches but does not equal the analytic
mass. Refining the Quantity spacing passed to `gravity_sources.discretize`
should converge the result toward the closed form in
[Analytic benchmarks](../guide/analytic-benchmarks.md); a result that does
not converge as spacing shrinks is a real signal, not expected
discretization noise.

## Memory and performance

`chunk_size` bounds the size of the intermediate `(M, N, ...)` array
materialized during a direct sum, trading memory for more Python-level
iterations. On the `numpy`, `numba` and `cpp` backends the result is
bit-identical for any `chunk_size`; on `cupy` it becomes a VRAM tiling
parameter and the bit-identical guarantee relaxes to a tolerance — see
[Chunking, softening and tolerances](../concepts/numerics.md).
