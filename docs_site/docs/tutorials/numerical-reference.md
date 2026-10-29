# Numerical references and directional averages

This page introduces existing numerical integration and directional averaging APIs, then gives a high precision reference calculation for one centered cavern model.

## Existing integration and averaging APIs

For Newtonian observables over rectangular boxes, use `gragra.observables.integrate_box_observables` or `integrate_displacement_box_observables`. They perform tensor Gauss–Legendre quadrature with a caller selected order and return potential, acceleration, and gravity gradient observables. The adaptive counterparts, `adaptive_box_observables` and `adaptive_displacement_box_observables`, compare quadrature orders and bisect subboxes when needed. These functions integrate the documented box models. See the [observable guide](../guide/observables.md) and [analytic API reference](../api/physics.md).

For directional and polarization weighted response arrays, `gragra.coherent.directional_power_average` computes a weighted sum of squared magnitudes along a chosen direction axis. `directional_cross_average` computes the element wise weighted cross average, conjugating the second response. Weights are not normalized internally; normalize them before calling when an expectation rather than a raw weighted sum is intended. See [coherent responses](../guide/coherent.md) for interpretation and weight conventions.

The optional `gragra.adapters.lebedev_grid` helper returns unit directions and normalized Lebedev weights using SciPy's `scipy.integrate.lebedev_rule`. It is a grid adapter; integration or averaging remains the caller's task. Install it with `pip install gragra[lebedev]` (SciPy 1.15 or newer). Some supported Lebedev rules have negative weights, which matters for power and cross averages.

## Centered spherical cavern volume kernel

`gragra.reference.centered_cavern_volume_kernel(ka, *, dps=60)` evaluates a scalar finite volume kernel for dimensionless radius `ka`. Here `k` is the acoustic wavenumber in radians per metre and `a` is the cavern radius in metres:

\[
V(x)=\int_0^1 [1-\cos(xu)]\,du,\qquad x=ka.
\]

For positive `x`, the closed form is `1 - sin(x)/x`; at zero, `V(0)=0`, and `V(x)/x²` tends to `1/6` as `x` tends to zero. The implementation uses two numerical routes and a fixed refinement schedule to estimate an absolute error. That estimate is not a rigorous error enclosure, and `dps` does not guarantee that many correct digits.

This kernel applies to a centered half spherical cavern in an isotropic, incoherent acoustic field under the stated model assumptions. It does not describe an arbitrary cavity boundary, an off center cavern, or scattering. Underground atmospheric infrasound and finite cylindrical cavity modes are separate models and outside this reference example.

Install the optional dependency with `pip install gragra[reference]` (`mpmath>=1.3,<2`). Importing `gragra.reference` and its exception types does not require mpmath; calling the kernel without mpmath installed raises an `ImportError` with the installation command.

Inputs are a finite, nonnegative decimal string, Python `int`, or Python `float`; `bool`, complex values, arrays, and NumPy scalar values are rejected. `dps` must be the exact Python integer `60` or `80`. Decimal strings preserve supplied decimal digits; a float's earlier binary rounding cannot be recovered. The kernel rejects work beyond its fixed subdivision cap and raises an error if its finite calculation schedule fails or does not meet its acceptance criterion. The [API reference](../api/reference.md) gives the exact context preflight, subdivision limits, and quadrature degree.

The immutable `ReferenceScalar` result contains the value, estimated absolute error, requested and actual precision, route and resolution metadata, evaluation records, error components, and a convergence flag. For nonzero input it records eight evaluations across two routes and the fixed precision/resolution schedule. The reported error is the maximum of precision, refinement, and route difference estimates; acceptance uses `1e-12 + 1e-10 * scale`. These finite checks are provisional bounded acceptance criteria, not universal accuracy claims.

Returned values are `mpmath` scalars created in an internal cloned context at `actual_dps = dps + 20`. Their ordinary arithmetic uses that value's creation context. For calculations at a chosen client precision, explicitly convert the value to the client's own cloned context and perform arithmetic with that context's functions.

## Pressure PSD to acceleration CSD example

The following self contained example computes the diagonal acceleration cross spectral density for a one sided cavern pressure power spectral density (PSD). A cross spectral density (CSD) is the matrix valued counterpart of a PSD. Inputs are pressure PSD `S_p` in `Pa²/Hz`, frequency `f` in `Hz`, cavern radius `a` in `m`, sound speed `c_s` in `m/s`, and Newton's constant `G_SI` in `m³ kg⁻¹ s⁻²`.

The model assumes pressure to density conversion `p/c_s²`, isotropic uncorrelated propagation directions, and a centered half spherical cavern. The factor `1/3` comes from averaging the directional outer product over isotropic directions. The resulting G free CSD has units `kg² m⁻⁴ Hz⁻¹`; multiplying by `G_SI²` once gives acceleration CSD in `m² s⁻⁴ Hz⁻¹`. This model has no density, background pressure, or heat capacity ratio input.

<!-- reference-physical-example:start -->
```python
from decimal import Decimal

import mpmath as mp

from gragra.reference import centered_cavern_volume_kernel


def acceleration_csd(f_hz, radius_m, sound_speed_m_s, pressure_psd_pa2_hz, G_si):
    """Return G-free and SI acceleration CSD matrices for the cavern model."""
    ctx = mp.mp.clone()
    ctx.dps = 80
    f = ctx.mpf(str(f_hz))
    a = ctx.mpf(str(radius_m))
    c_s = ctx.mpf(str(sound_speed_m_s))
    S_p = ctx.mpf(str(pressure_psd_pa2_hz))
    G = ctx.mpf(str(G_si))
    if not (ctx.isfinite(f) and f > 0):
        raise ValueError("f_hz must be finite and positive")
    if not (ctx.isfinite(a) and a >= 0):
        raise ValueError("radius_m must be finite and nonnegative")
    if not (ctx.isfinite(c_s) and c_s > 0):
        raise ValueError("sound_speed_m_s must be finite and positive")
    if not (ctx.isfinite(S_p) and S_p >= 0):
        raise ValueError("pressure_psd_pa2_hz must be finite and nonnegative")
    if not (ctx.isfinite(G) and G > 0):
        raise ValueError("G_si must be finite and positive")

    two_pi = ctx.fmul(2, ctx.pi)
    k = ctx.fdiv(ctx.fmul(two_pi, f), c_s)
    x = ctx.fmul(k, a)
    result = centered_cavern_volume_kernel(ctx.nstr(x, ctx.dps))
    V = ctx.mpf(result.value)
    V2 = ctx.fmul(V, V)
    four_pi_over_ck = ctx.fdiv(ctx.fmul(4, ctx.pi), ctx.fmul(ctx.fmul(c_s, c_s), k))
    C = ctx.fdiv(ctx.fmul(S_p, ctx.fmul(four_pi_over_ck, four_pi_over_ck)), 3)
    q = ctx.fmul(C, V2)
    q_matrix = [[q if i == j else ctx.mpf(0) for j in range(3)] for i in range(3)]
    G2q = ctx.fmul(ctx.fmul(G, G), q)
    si_matrix = [[G2q if i == j else ctx.mpf(0) for j in range(3)] for i in range(3)]

    # Compare with the closed expression using a bound from the kernel tolerance.
    V_closed = ctx.fsub(1, ctx.fdiv(ctx.sin(x), x)) if x != 0 else ctx.mpf(0)
    tau = ctx.fadd(ctx.mpf("1e-12"), ctx.fmul(ctx.mpf("1e-10"), ctx.fabs(V_closed)))
    q_bound = ctx.fmul(C, ctx.fadd(ctx.fmul(2, ctx.fmul(ctx.fabs(V_closed), tau)), ctx.fmul(tau, tau)))
    q_truth = ctx.fmul(C, ctx.fmul(V_closed, V_closed))
    assert ctx.fabs(ctx.fsub(q, q_truth)) <= q_bound
    si_bound = ctx.fmul(ctx.fmul(G, G), q_bound)
    assert ctx.fabs(ctx.fsub(G2q, ctx.fmul(ctx.fmul(G, G), q_truth))) <= si_bound
    if a == 0 or S_p == 0:
        assert all(value == 0 for row in q_matrix for value in row)
        assert all(value == 0 for row in si_matrix for value in row)
    return q_matrix, si_matrix


G_SI = Decimal("6.67430e-11")
G_free_csd, acceleration_csd_si = acceleration_csd(
    Decimal("1"), Decimal("10"), Decimal("340"), Decimal("1"), G_SI
)
zero_radius_g_free, zero_radius_si = acceleration_csd(
    Decimal("1"), Decimal("0"), Decimal("340"), Decimal("1"), G_SI
)
zero_psd_g_free, zero_psd_si = acceleration_csd(
    Decimal("1"), Decimal("10"), Decimal("340"), Decimal("0"), G_SI
)
assert all(value == 0 for matrix in (zero_radius_g_free, zero_radius_si, zero_psd_g_free, zero_psd_si) for row in matrix for value in row)
assert all(G_free_csd[i][j] == G_free_csd[j][i] for i in range(3) for j in range(3))
assert all(acceleration_csd_si[i][j] == acceleration_csd_si[j][i] for i in range(3) for j in range(3))
try:
    acceleration_csd(Decimal("0"), Decimal("10"), Decimal("340"), Decimal("1"), G_SI)
except ValueError:
    pass
else:
    raise AssertionError("zero frequency is outside this mapping")
```
<!-- reference-physical-example:end -->

The example passes `ka` to the kernel as a decimal string generated with 80 client context digits, then converts `result.value` explicitly into that context before squaring it. Its assertions compare the computed CSD with the closed expression within the kernel's stated tolerance bound and check exact zero matrices at zero radius or zero pressure PSD. The mapping requires `f > 0`; it does not define a zero frequency limit.
