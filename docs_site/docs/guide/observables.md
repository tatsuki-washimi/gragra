
# Observables

## The observable layer

The three functions on this page are gragra's public entry points for
Newtonian gravity: potential, acceleration and gravity-gradient tensor. Each
dispatches to a G-free kernel contract and
multiplies the result by `G_SI` — that multiplication is the *only* thing
this layer does beyond argument normalization and dispatch (see
[What gragra computes](../concepts/model-and-scope.md)). All four accept a
`chunk_size` and (where relevant) a `softening_m` and a `backend` keyword;
see [Chunking, softening and tolerances](../concepts/numerics.md) and
[Compute backends](backends.md). The `"numpy"` backend is the reference
implementation and the default; the other backends are experimental
performance selectors checked for agreement against it.

## Potential

`point_potential(source, targets, *, softening_m=0.0, chunk_size=None)`
returns the scalar gravitational potential, shape `(M,)`, unit J/kg. Sign
convention: `K_Φ = −1/r_eff`, negative for a positive mass.

```python
--8<-- "snippets/quickstart_verify_install.py"
```

## Acceleration

`point_acceleration(source, targets, *, softening_m=0.0, chunk_size=None, backend="numpy")`
returns the vector gravitational acceleration, shape `(M, 3)`, unit m/s².
Sign convention: `K_a = Δ/r_eff³` with `Δ = r_source − r_target`, so the
result points from the observation point back towards the source.

```python
--8<-- "snippets/quickstart_point_mass_acceleration.py"
```

## Gravity gradient tensor

`point_gravity_gradient(source, targets, *, softening_m=0.0, chunk_size=None)`
returns the tensor `∂aᵢ/∂xⱼ`, shape `(M, 3, 3)`, unit s⁻². Sign convention:
`K_T = 3ΔΔ/r_eff⁵ − I/r_eff³`, differentiated with respect to the
observation coordinate. At the fixed point `softening_m=0`, source at the
origin, target at `(1, 0, 0)`: `T = G_SI · diag(2, −1, −1)`, and
`tr(T) = 0` — the vacuum Laplace-equation property, which holds **only**
for zero softening (see [Chunking, softening and tolerances](../concepts/numerics.md)).

```python
--8<-- "snippets/quickstart_sign_conventions.py"
```

## Status and verification

### `point_potential` { #status-point-potential }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Stable |
    | **Verified against** | closed-form analytic solution; independent numerical oracle (tests/oracles/) |
    | **Agreement** | Exact at the unit-mass, unit-distance fixed point |
    | **Test evidence** | `tests/test_observables.py::test_potential_analytic_value`, `tests/test_kernel_contract.py::test_kernel_vs_observable_real` |
    | **Conditions** | `softening_m >= 0`; `chunk_size` any positive int or `None` |
    | **Not verified** | Agreement outside the tested fixed points and configurations |
    | **Known limits** | none beyond [Chunking, softening and tolerances](../concepts/numerics.md) |
    | **Scope** | none |

### `point_acceleration` { #status-point-acceleration }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Stable |
    | **Verified against** | closed-form analytic solution; invariance property (chunk-size, backend equivalence) |
    | **Agreement** | Exact at the unit-mass, unit-distance fixed point; `rtol=1e-15` chunk-size invariance |
    | **Test evidence** | `tests/test_observables.py::test_acceleration_analytic_value`, `tests/test_observables.py::test_inverse_square_scaling`, `tests/test_observables.py::test_superposition_symmetric_sources`, `tests/test_kernel_contract.py::test_chunk_size_invariance` |
    | **Conditions** | `softening_m >= 0`; any `backend`; `chunk_size` any positive int or `None` |
    | **Not verified** | Error detection beyond the listed fixed points and configurations; the zero/sign-flip mutation controls cover only their named checks |
    | **Known limits** | none beyond [Chunking, softening and tolerances](../concepts/numerics.md) |
    | **Scope** | none |

### `point_gravity_gradient` { #status-point-gravity-gradient }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Stable |
    | **Verified against** | closed-form analytic solution; invariance property |
    | **Agreement** | Tensor-sign fixed point matches to `rtol=1e-14`; at zero softening the trace is bounded by `32 ε max(abs(T))` for the tested configuration |
    | **Test evidence** | `tests/test_observables.py::test_tensor_sign_convention`, `tests/test_observables.py::test_trace_free_gradient`, `tests/test_observables.py::test_numerical_derivatives` |
    | **Conditions** | `softening_m >= 0`; trace-free property holds only at `softening_m == 0` |
    | **Not verified** | Agreement outside the tested fixed points |
    | **Known limits** | With `softening_m > 0`, `tr(T)` is not zero — see [Chunking, softening and tolerances](../concepts/numerics.md) |
    | **Scope** | none |
