"""Centered spherical cavern volume reference kernel.

The integration bodies are a local extraction of the independently maintained
Gen 1 A3 ``volume_kernel`` (tests/oracles/gen1/stochastic_formal/a3/generator.py).
The source generator and its fixtures remain outside the installed package.
"""

from __future__ import annotations

from ._context import (
    convert_input,
    make_context,
    transfer_scalar,
    validate_dps,
    validate_input,
)
from ._types import (
    ReferenceConvergenceError,
    ReferenceErrorComponents,
    ReferenceEvaluation,
    ReferenceResourceError,
    ReferenceScalar,
)

_MAX_RATIO = "2500"
_MAX_SUBDIVISIONS = 10000
_SCHEDULE = ((0, 1), (20, 1), (20, 2), (20, 4))


def _volume_kernel(ctx, x, resolution: int, route: str, n: int):
    """Evaluate one fixed route and planned subdivision count."""
    if route == "angular":
        return ctx.quad(
            lambda u: 2 * ctx.sin(x * u / 2) ** 2,
            [ctx.fdiv(ctx.mpf(i), n) for i in range(n + 1)],
            maxdegree=8,
        )

    def radial(t):
        if t == 0:
            return ctx.mpf(0)
        if ctx.fabs(t) < ctx.mpf("0.5"):
            return ctx.nsum(
                lambda m: (-1) ** (m + 1) * 2 * m * t ** (2 * m - 1) / ctx.factorial(2 * m + 1),
                [1, ctx.inf],
            )
        return (ctx.sin(t) - t * ctx.cos(t)) / t**2

    return ctx.quad(
        radial,
        [ctx.fdiv(x * i, n) for i in range(n + 1)],
        maxdegree=8,
    )


def centered_cavern_volume_kernel(ka, *, dps=60) -> ReferenceScalar:
    """Compute the dimensionless centered spherical volume kernel ``V(ka)``.

    Uses two fixed quadrature routes at a bounded precision/refinement schedule.
    The reported error is an estimate from those evaluations, not a certified
    enclosure. ``ka`` accepts a decimal string, Python int, or finite float.
    """
    precision = validate_dps(dps)
    original = validate_input(ka)
    output = make_context(precision + 20)

    output_x = convert_input(output, original)
    if output_x == 0:
        value = output.mpf(0)
        components = ReferenceErrorComponents(value, value, value)
        return ReferenceScalar(value, value, precision, precision + 20, 0,
                               "exact_zero", (), components, True)

    # Build independent contexts for every evaluation and the output. Preflight
    # all ratios before any ceil/int conversion, points, or integration call.
    specs = []
    for extra, resolution in _SCHEDULE:
        for route in ("angular", "radial"):
            ctx = make_context(precision + extra)
            x = convert_input(ctx, original)
            ratio = ctx.fdiv(x, ctx.pi)
            specs.append((route, precision + extra, resolution, ctx, x, ratio))
    output_ratio = output.fdiv(output_x, output.pi)
    ratio_contexts = [(spec[5], spec[3]) for spec in specs]
    ratio_contexts.append((output_ratio, output))
    if any(ratio > ctx.mpf(_MAX_RATIO) for ratio, ctx in ratio_contexts):
        raise ReferenceResourceError("centered cavern kernel exceeds the ka/pi resource limit")

    # Now integerize all subdivision counts, then validate all before work.
    planned = []
    for route, used_dps, resolution, ctx, x, ratio in specs:
        n = max(1, int(ctx.ceil(ratio))) * resolution
        planned.append((route, used_dps, resolution, ctx, x, n))
    if any(n > _MAX_SUBDIVISIONS for *_, n in planned):
        raise ReferenceResourceError("centered cavern kernel exceeds the subdivision limit")

    raw = {}
    evaluations = []
    for route, used_dps, resolution, ctx, x, n in planned:
        try:
            value = _volume_kernel(ctx, x, resolution, route, n)
            if not ctx.isfinite(value):
                raise ArithmeticError("quadrature returned a non-finite value")
        except ReferenceResourceError:
            raise
        except Exception as exc:
            raise ReferenceConvergenceError("centered cavern quadrature failed") from exc
        raw[(route, used_dps, resolution)] = transfer_scalar(value, output)
        evaluations.append(ReferenceEvaluation(route, used_dps, resolution, n))

    def diff(left, right):
        return output.fabs(output.fsub(left, right))

    low, high = precision, precision + 20
    e_p = max(diff(raw[(route, low, 1)], raw[(route, high, 1)])
              for route in ("angular", "radial"))
    e_r = max(diff(raw[(route, high, 2)], raw[(route, high, 4)])
              for route in ("angular", "radial"))
    e_a = diff(raw[("angular", high, 4)], raw[("radial", high, 4)])
    error = max(e_p, e_r, e_a)
    angular_fine = raw[("angular", high, 4)]
    radial_fine = raw[("radial", high, 4)]
    scale = max(output.fabs(angular_fine), output.fabs(radial_fine))
    tolerance = output.fadd(output.mpf("1e-12"), output.fmul(output.mpf("1e-10"), scale))
    if not output.isfinite(error) or not output.isfinite(scale) or error > tolerance:
        raise ReferenceConvergenceError("centered cavern quadrature did not meet the fixed error criterion")

    components = ReferenceErrorComponents(e_p, e_r, e_a)
    return ReferenceScalar(radial_fine, error, precision, precision + 20, 4,
                           "radial", tuple(evaluations), components, True)
