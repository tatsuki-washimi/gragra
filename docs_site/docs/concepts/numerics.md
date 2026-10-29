
# Chunking, softening and tolerances

```python
--8<-- "snippets/quickstart_softening_and_chunking.py"
```

## Chunking

### What `chunk_size` does

`chunk_size` bounds the size of the intermediate `(M, N, ...)` array
materialized while summing over sources, trading memory for more
Python-level iterations. It is purely a memory/performance knob on the CPU
backends — `chunk_size=None` (no chunking) and any positive `chunk_size`
must produce the same result, and `chunk_size<=0` raises `ValueError`.

### Reproducibility across backends

On `"numpy"`, `"numba"` and `"cpp"`, the result is **bit-identical** for any
`chunk_size` — the compiled backends validate the argument but do not use it
for tiling at all. On `"cupy"` this guarantee **relaxes to a tolerance**:
`chunk_size` becomes a genuine VRAM-tiling parameter, the in-chunk reduction
is a non-associative parallel sum, and changing `chunk_size` changes the
last bits of the answer. Reproducibility there means "same device, same
CuPy version, same `chunk_size`" — not "same everywhere". See
[Compute backends](../guide/backends.md) for the full backend contract.

## Softening

### What softening changes

Plummer softening is a **smoothed-mass approximation, not a regularisation
of the exact field.** Setting `softening_m > 0` is exactly equivalent to
replacing the point mass with a Plummer sphere of density
`ρ(r) = 3Mε²/(4π r_eff⁵)`. The near field is a genuinely different mass
distribution, not the same field with a singularity removed.

### Why the gradient trace stops vanishing

Because softening changes the mass distribution, `tr(T) = 0` — the
vacuum Laplace-equation property of the exact point-mass gravity-gradient
tensor — holds **only** for `softening_m == 0`. With `softening_m > 0` the
observation point sits inside the (now-finite) mass distribution, so
Poisson's equation applies there instead, and the trace residual equals
`−4πGρ` evaluated at that point for the Plummer density above — not
numerical noise, a genuine physical effect of the approximation. A
collision (`r == 0`) with `softening_m == 0` raises `ValueError` naming the
offending source/target index pair; `softening_m < 0` always raises.

## Comparing results numerically

**NumPy's default `atol=1e-8` is meaningless at gragra's output scale.**
`G_SI ≈ 6.7 × 10⁻¹¹`, so `np.allclose` at default tolerances accepts a
result of exactly zero as "close to" a genuine, non-zero acceleration, and
accepts a sign-flipped result as "close to" the correct one. Every
comparison on this site states an explicit tolerance for exactly this
reason — never rely on `np.allclose`'s defaults when checking a gragra
result.

### Non-zero fixed points

Where the arithmetic is exact at a chosen configuration (unit mass, unit
distance, zero softening — see
[Coordinates, signs and units](conventions.md)), compare with an **exact
equality**, not a tolerance. Where the comparison must be a tolerance, use
an explicit `rtol` with `atol=0.0` so a structurally zero component cannot
silently pass. Prefer `abs(component)` over `np.linalg.norm` for a
scalar magnitude check: `norm` can dispatch to a BLAS routine whose
internal scaling makes bit-for-bit behaviour implementation-dependent
across OpenBLAS/MKL and Python versions, where a plain `abs` does not.

### Quantities whose target value is zero

A bare `rtol`/`atol` judgement against a target of exactly zero is
ill-posed — `rtol` scales with the target, so it scales with zero. Instead,
normalize the residual by the largest relevant component (as the chunking
snippet above does) or, where a closed-form magnitude is available, compare
against that magnitude directly.

### Cancellation-sensitive quantities

Some results (mixed-sign weighted sums, cross-statistics with cancelling
directional terms) are sensitive to the coefficients cancelling almost
exactly. For those, a residual tolerance is scaled by the sum of the
*absolute values* of the terms that were summed, not by the (possibly much
smaller) result — otherwise a genuinely large cancellation error can be
misread as a passing near-zero residual. See
[Coherent responses and cross statistics](../guide/coherent.md) for where
this applies.
