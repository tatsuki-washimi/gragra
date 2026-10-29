# Numerical reference API

The reference namespace contains an optional high precision scalar kernel for one centered spherical cavern volume calculation. It complements gragra's existing public box integration and directional averaging functions; it does not replace or generalize them.

## Centered cavern volume kernel

::: gragra.reference.centered_cavern_volume_kernel

The input `ka` is the dimensionless product of acoustic wavenumber `k` and cavern radius `a`. It accepts only a finite, nonnegative decimal string, exact Python `int`, or exact Python `float`. Decimal strings are preferred when the source value has more precision than a binary float can hold. Boolean, complex, array, and NumPy scalar inputs are invalid. The keyword `dps` accepts only the exact Python integer `60` or `80`.

For `x = ka`, the kernel evaluates

\[
V(x)=\int_0^1 (1-\cos(xu))\,du,
\]

with `V(0)=0`, closed form `1 - sin(x)/x` for `x > 0`, and small argument limit `V(x)/x² → 1/6`. The physical interpretation is limited to a centered half spherical cavern in an isotropic, incoherent acoustic field. It is not an arbitrary cavity boundary solution, a noncentral-cavern model, or a scattering calculation.

The optional `reference` extra installs `mpmath>=1.3,<2`. Core imports, `gragra.reference`, and its exception classes remain importable without mpmath; the numerical call raises an installation-guided `ImportError` if mpmath is unavailable at runtime. The extra's installation history does not control runtime use when mpmath is otherwise present.

Successful results are immutable `ReferenceScalar` records with nine fields:

- `value`: the selected scalar result.
- `estimated_abs_error`: the estimated absolute error.
- `requested_dps`: the requested working precision, 60 or 80.
- `actual_dps`: `requested_dps + 20`.
- `actual_resolution`: resolution of the selected result, 4 for nonzero input and 0 at zero.
- `actual_route`: route used for the selected result, `radial` for nonzero input and `exact_zero` at zero.
- `evaluations`: an immutable tuple of evaluation records, each with `route`, `dps`, `resolution`, and subdivision count `n`.
- `error_components`: `e_p`, `e_r`, and `e_a`, the precision, refinement, and route-difference estimates.
- `converged`: whether the fixed finite acceptance test passed.

The result value, estimated absolute error, and error components are mpmath scalars from one internal cloned context at `actual_dps`. For each of the angular and radial routes, nonzero input uses exactly `(dps, 1)`, `(dps + 20, 1)`, `(dps + 20, 2)`, and `(dps + 20, 4)`, for eight evaluations total. There are no automatic retries. The selected value is the high precision radial result at resolution 4. At zero, value and all error components are exact zero, `evaluations` is empty, `actual_resolution` is 0, and `actual_route` is `exact_zero`.

The estimated error is `max(e_p, e_r, e_a)`: `e_p` is the largest difference between requested and higher precision at resolution 1 across both routes, `e_r` is the largest higher precision resolution 2 versus 4 difference across both routes, and `e_a` is the route difference at higher precision and resolution 4. The scale is the larger absolute value from the two finest route results. Acceptance requires estimated error no greater than `1e-12 + 1e-10 * scale`, along with finite successful evaluations. This is a fixed provisional bounded acceptance test; it is not a rigorous enclosure, a universal accuracy guarantee, or a promise of `dps` correct digits.

The result and its diagnostic records use the following immutable public record types and errors:

::: gragra.reference.ReferenceScalar

`ReferenceScalar` is the immutable scalar result record returned by the kernel.

::: gragra.reference.ReferenceEvaluation

`ReferenceEvaluation` records one route, precision, resolution, and subdivision count.

::: gragra.reference.ReferenceErrorComponents

`ReferenceErrorComponents` stores the precision, refinement, and route-difference estimates.

::: gragra.reference.ReferenceConvergenceError

`ReferenceConvergenceError` reports a finite-evaluation or fixed acceptance failure.

::: gragra.reference.ReferenceResourceError

`ReferenceResourceError` reports a subdivision or evaluation resource-limit rejection.

Before subdivision, the kernel checks `r = x / pi <= 2500` in every evaluation context and the output context. It completes these checks before calculating `ceil(r)`, converting a subdivision count to an integer, generating points, or calling the kernel. It then checks every scheduled `n = max(1, int(ceil(r))) * resolution <= 10000` before beginning any evaluation. The quadrature `maxdegree` is 8. A resource-limit rejection raises `ReferenceResourceError`; invalid arguments raise `ValueError`; integration failures, nonfinite results, and failed acceptance raise `ReferenceConvergenceError`. These limits constrain subdivisions and evaluation count but do not impose a hard wall clock timeout.

Ordinary arithmetic on a returned mpmath value uses the context that created it. To choose a client precision, convert explicitly with `v = ctx.mpf(result.value)` and use `ctx.fmul`, `ctx.fadd`, and other functions from that same cloned context. Changing a context's `dps` alone does not move an existing mpmath value into it.

## Existing integration and directional average entry points

The following functions are already public in `gragra.observables`:

- `integrate_box_observables` and `integrate_displacement_box_observables` integrate Newtonian observables over rectangular boxes with tensor Gauss–Legendre quadrature.
- `adaptive_box_observables` and `adaptive_displacement_box_observables` compare quadrature orders and refine boxes when their error estimate requires it.

For directional response and polarization weighting, `gragra.coherent.directional_power_average` computes a weighted power sum and `directional_cross_average` computes a weighted element wise cross average. Neither normalizes weights internally. The optional `gragra.adapters.lebedev_grid` provides normalized direction weights and unit vectors through SciPy's Lebedev rule adapter; install it with `gragra[lebedev]`.

See the [numerical reference tutorial](../tutorials/numerical-reference.md) for an installed-use example and the limitations of the cavern model.
