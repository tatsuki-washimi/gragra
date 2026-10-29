
# Coherent responses and cross statistics

## The coherent path

The `coherent_*` functions integrate a parametric source (typically a
plane-wave mass-weight field, see [Waves](waves.md)) the same way the
point-direct observables integrate a `WeightedPointSource`, but keep a
parametric axis (frequency, direction) alive through the calculation
instead of collapsing it. They share the point-direct kernels and the same
sign conventions (see [Coordinates, signs and units](../concepts/conventions.md));
what differs is the shape of the result and, for the second-order
quantities below, what the complex output means.

## First-order coherent observables

`coherent_point_acceleration` and `coherent_point_gravity_gradient` are the
coherent-path counterparts of `point_acceleration` and
`point_gravity_gradient` — first-order, linear complex amplitudes
(disclaimer A, see [Complex outputs](../concepts/complex-outputs.md)).
They are importable from `gragra` and from `gragra.observables`; their
interfaces remain research-oriented and may evolve with use.

## Second-order cross statistics

### What a cross average is

A cross average — `directional_cross_average`, `directional_outer_cross_average`,
`ensemble_outer_cross_average` — combines two first-order amplitudes
**element-wise**, conjugating the **second** argument: `⟨A·B*⟩`, not a dot
product or matrix product. `coherent_gravitoelastic_tensor` and
`coherent_acceleration_csd` build on this to produce second-order
cross-spectral tensors (disclaimer B, see
[Complex outputs](../concepts/complex-outputs.md)).

### Weights and normalization

Directional/ensemble weights are always real-valued and may be
**negative** — a Lebedev quadrature weight, for instance. Because of this,
`cross(A, A, w)` (the "auto"-cross of a quantity with itself) can be
legitimately **negative** under mixed-sign weights; this is not a sign
error. Weights are applied as a raw weighted sum — the caller normalizes,
gragra does not divide by `Σw` internally. The resulting cross tensor is
**not guaranteed positive semi-definite** under mixed-sign weights.

### Directional averages

A directional average `Σ_d` is the expectation value `⟨·⟩` under an
explicit **"directions are uncorrelated"** ensemble model. It does not
apply to a field with genuine directional correlation — using it there
silently discards the correlation rather than raising an error.

## Powers of the gravitational constant

Different coherent quantities carry different powers of `G_SI` even before
any observable multiplies by it explicitly: a witness/displacement
quantity is `G⁰`, `coherent_gravitoelastic_tensor` (`c_ij = ⟨δa_i ξ_j*⟩`) is
`G¹`, and an acceleration cross-spectral density (`coherent_acceleration_csd`)
is `G²`. A missing or doubled `G_SI` factor at the wrong order cannot be
caught by grepping for `G_SI` in the source — it is caught only by an
absolute-value golden test at the correct numerical scale; see
[How gragra is verified](../verification/index.md) for why this specific
error class needs a golden test rather than a structural one.

## Cross-spectral densities and phase

For `coherent_acceleration_csd`, the phase of `⟨a(r₁)·a(r₂)*⟩` is fixed:
`arg = k·(r₁ − r₂)`. A witness displacement used alongside a δa source must
be built with the **same** `(k̂, k, phase₀)` and a fixed `sign=+1` convention
— an inadvertently conjugated witness silently flips this phase without
raising any error, which is why the convention is stated explicitly here
rather than left to be inferred from the sign of an output.

## Status and verification

### `coherent_point_acceleration` { #status-coherent-point-acceleration }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Experimental |
    | **Verified against** | closed-form analytic solution; invariance property (backend equivalence, chunk-size invariance) |
    | **Agreement** | `rtol=1e-12` backend equivalence; exact at the documented fixed points |
    | **Test evidence** | `tests/test_coherent_observables.py::test_coherent_point_acceleration_basics`, `tests/test_coherent_observables.py::test_coherent_backend_matches_numpy` |
    | **Conditions** | Any `backend`; `chunk_size` any positive int or `None` |
    | **Not verified** | Agreement outside the tested fixed points and configurations |
    | **Known limits** | Linear complex amplitude only — see [Complex outputs](../concepts/complex-outputs.md) |
    | **Scope** | Research interface; argument and return-shape changes remain possible |

### `coherent_point_gravity_gradient` { #status-coherent-point-gravity-gradient }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Experimental |
    | **Verified against** | closed-form analytic solution; invariance property |
    | **Agreement** | Exact tensor-sign fixed point; `rtol=1e-15` chunk-size invariance |
    | **Test evidence** | `tests/test_coherent_observables.py::test_coherent_gradient_tensor_sign_fixed`, `tests/test_coherent_observables.py::test_coherent_gradient_chunk_invariance`, `tests/test_coherent_observables.py::test_coherent_gradient_complex_weights_dtype` |
    | **Conditions** | Complex-weighted sources produce complex128 output |
    | **Not verified** | Agreement outside the tested fixed points |
    | **Known limits** | Linear complex amplitude only |
    | **Scope** | Research interface; argument and return-shape changes remain possible |

### `coherent_gravitoelastic_tensor` { #status-coherent-gravitoelastic-tensor }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Experimental |
    | **Verified against** | invariance property (absolute-value golden, conjugate symmetry, S-wave structural zero) |
    | **Agreement** | Absolute-value golden at `G¹`; conjugate symmetry exact; zero exactly for a pure S wave (`k̂·ê = 0`) |
    | **Test evidence** | `tests/test_coherent_gravitoelastic.py::test_gc1_c_ij_absolute_golden`, `tests/test_coherent_gravitoelastic.py::test_s_wave_zero_fixed_point`, `tests/test_coherent_gravitoelastic.py::test_conjugate_symmetry`, `tests/test_coherent_gravitoelastic.py::test_chunk_invariance` |
    | **Conditions** | Second-order cross-spectral quantity — see [Complex outputs](../concepts/complex-outputs.md) disclaimer B. Weights real, may be negative |
    | **Not verified** | Positive semi-definiteness under mixed-sign weights — not guaranteed and not asserted |
    | **Known limits** | Directional-average interpretation assumes directions are uncorrelated — see [Directional averages](#directional-averages) |
    | **Scope** | Experimental API surface; schema not yet frozen |

### `coherent_acceleration_csd` { #status-coherent-acceleration-csd }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Experimental |
    | **Verified against** | invariance property (absolute-value golden at `G²`, phase fixed point) |
    | **Agreement** | Absolute-value golden at `G²`; phase `arg = k·(r₁ − r₂)` exact at the documented fixed point |
    | **Test evidence** | `tests/test_coherent_gravitoelastic.py::test_gc2_csd_absolute_golden`, `tests/test_coherent_gravitoelastic.py::test_shape_fixed_multi_point`, `tests/test_coherent_gravitoelastic.py::test_read_only_outputs` |
    | **Conditions** | Witness and δa source must share `(k̂, k, phase₀)` and `sign=+1` — see [Cross-spectral densities and phase](#cross-spectral-densities-and-phase) |
    | **Not verified** | Behaviour when witness and source disagree on convention — this is a silent phase error, not a raised exception |
    | **Known limits** | Second-order cross-spectral quantity, not a time-domain field |
    | **Scope** | Experimental API surface; schema not yet frozen |
