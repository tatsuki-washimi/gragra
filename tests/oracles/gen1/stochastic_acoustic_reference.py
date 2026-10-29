"""Production-independent acoustic quadratures for review, not frozen evidence.

Sources (retained PDFs are outside Git in the acoustic lane's sources directory):
Harms, Living Reviews in Relativity 22, 6 (2019), journal PDF, Eqs. 131/132
pp. 69/70 and Eq. 160 p. 82, SHA256
150ec4b02097cd00a744cd67ed01dc3e9c3a3ebfd0a40c7ec68bce0467906376.
Amann et al., arXiv:2003.03434v2 (2020-06-14), p. 9, Eqs. 10--13, SHA256
cfaa10667acc51114f1e5eb48d3225084fe2f1339d0582d08096b4b5548f0069.

A1: density per total surface pressure is cos(b*z)/c_s**2. The horizontal
Fourier transform of the Newton Green function is 2*pi*exp(-q*|z-t|)/q.
At z>0 split the vertical integral at t=z; its value J and derivative are
(2*q*cos(b*z)-q*exp(-q*z))/(q*q+b*b) and
(-2*q*b*sin(b*z)+q*q*exp(-q*z))/(q*q+b*b). This nonsingular 1D Green route
justifies the z->0+ limit analytically for q>0. The measure-zero q=0 ray
has zero surface transfer. Dominated convergence on the compact hemisphere
holds for fixed k>0: the displayed J/q and J'/q are uniformly bounded.
No naive singular three-dimensional boundary quadrature is used.

A2: the defining angular quadrature and its derivatives use endpoint splits
on the physical scale 1/x, independently of production branch thresholds.
An alternative mu=cos(theta) quadrature checks the formulation.

A3: the centered spherical-volume Newton kernel pairs opposite mu directions
before integration: 2*pi*i*int_0^R dr int_-1^1 mu*sin(k*r*mu) dmu.
Its dimensionless kernel is int_0^1 (1-cos(x*mu)) dmu. An independent radial
integration uses int_0^x (sin(t)-t*cos(t))/t**2 dt. Neither calls sinc or
production special functions. Direction averaging is independently integrated.
This verifies only the spherical-volume kernel; Amann Eq.13's published
centered half-spherical-cavern interpretation is a separate model assumption.
No arbitrary boundary-acoustics theorem or detector response is claimed.

All uncertainties below are NUMERICAL ESTIMATES, never rigorous total bounds.
Input decimals are exact model inputs (not measured quantities). Finite domains
have zero omitted tails by construction. No roots occur. Quadrature/precision
refinements and alternate formulations estimate errors; operation-count
roundoff allowances are heuristic, and serialized rounding is explicitly added.
Unknown components must not be replaced by zero in any downstream admission.
Defaults are development proposals; formal policies and sealer are pending.
"""

import importlib.abc
import re
import sys
from contextlib import contextmanager
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext

import mpmath as mp

UNCERTAINTY_COMPONENTS = {
    "precision",
    "quadrature",
    "formulation",
    "input",
    "tail",
    "root",
    "roundoff",
    "serialization",
}
CALIBRATION = {
    "A1": {"q": "0.7", "b": "1.1", "z": "0.19"},
    "A2": ["0.0023", "0.41", "2.7", "47"],
    "A3": ["0.0041", "0.63", "3.7"],
}


def _ctx(dps):
    if not isinstance(dps, int) or isinstance(dps, bool) or dps < 25:
        raise ValueError("at least 25 decimal working digits required")
    ctx = mp.mp.clone()
    ctx.dps = dps
    return ctx


def _input(ctx, value, *, positive=False):
    if not isinstance(value, str):
        raise TypeError("scientific inputs must be decimal strings")
    try:
        dec = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("invalid decimal input") from exc
    if not dec.is_finite() or dec < 0 or (positive and dec == 0):
        raise ValueError("finite nonnegative input required")
    return ctx.mpf(value)


def decimal_string(value, digits=40):
    """Canonical decimal central value, rounded to nearest/ties-to-even.

    The caller must compute value with guard digits. Scientific JSON must keep
    this string intact; do not convert it to a JSON number or binary64.
    """
    if not isinstance(digits, int) or isinstance(digits, bool) or digits < 2:
        raise ValueError("at least two serialization digits required")
    if not mp.isfinite(value):
        raise ValueError("nonfinite scientific value")
    ctx = mp.mp.clone()
    ctx.dps = max(digits + 20, value._mpf_[3] * 30103 // 100000 + 10)
    raw = ctx.nstr(value, n=digits + 15, strip_zeros=False)
    with localcontext() as dc:
        dc.prec = digits
        dc.rounding = ROUND_HALF_EVEN
        dec = +Decimal(raw)
    if not dec:
        return "0e+0"
    sign, coeff, exp = dec.as_tuple()
    chars = "".join(map(str, coeff)).rstrip("0")
    power = len(coeff) + exp - 1
    mantissa = chars[0] + ("." + chars[1:] if len(chars) > 1 else "")
    return (
        ("-" if sign else "")
        + mantissa
        + "e"
        + ("+" if power >= 0 else "-")
        + str(abs(power))
    )


class _ProductionDenied(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, _path=None, _target=None):
        if fullname == "gragra" or fullname.startswith("gragra."):
            raise ImportError("production import denied in independent acoustic oracle")
        return


@contextmanager
def deny_production_imports():
    """Runtime independence gate; already-imported production fails closed."""
    if any(n == "gragra" or n.startswith("gragra.") for n in sys.modules):
        raise ImportError("production already imported before independent oracle")
    guard = _ProductionDenied()
    sys.meta_path.insert(0, guard)
    try:
        yield
    finally:
        sys.meta_path.remove(guard)


def green_closed(q, b, z, *, context=None):
    """Analytic split of the Fourier Green integral for q>0, b,z>=0.

    Inputs are arbitrary-precision numbers supplied by the caller; use the
    numerical green_integral for independently checking this identity.
    """
    if q <= 0 or b < 0 or z < 0:
        raise ValueError("q>0, b>=0, z>=0 required")
    ctx = context
    if ctx is None:
        ctx = mp.mp.clone()
        ctx.prec = max(mp.mp.prec, q._mpf_[3], b._mpf_[3], z._mpf_[3]) + 20
    den = q * q + b * b
    return (
        (2 * q * ctx.cos(b * z) - q * ctx.exp(-q * z)) / den,
        (-2 * q * b * ctx.sin(b * z) + q * q * ctx.exp(-q * z)) / den,
    )


def green_integral(q, b, z, *, dps=60):
    """Positive-height Green quadrature and explicit exponential tail bound.

    T=z+(dps+10)*ln(10)/q. Each interval spans <= pi/b to resolve oscillation.
    Tail(J) <= exp(-q*(T-z))/q; tail(J') <= q*tail(J). These tail bounds
    are analytic; quadrature/roundoff error remains an estimate elsewhere.
    """
    ctx = _ctx(dps)
    q, b, z = (
        _input(ctx, q, positive=True),
        _input(ctx, b),
        _input(ctx, z, positive=True),
    )
    end = z + (dps + 10) * ctx.log(10) / q
    count = max(1, int(ctx.ceil((end - z) * b / ctx.pi)))
    if count > 10000:
        raise ValueError("Green oscillation resource limit; unresolved")
    split = [z + (end - z) * i / count for i in range(count + 1)]
    lower = ctx.quad(lambda t: ctx.exp(-q * (z - t)) * ctx.cos(b * t), [0, z])
    upper = ctx.quad(lambda t: ctx.exp(-q * (t - z)) * ctx.cos(b * t), split)
    return lower + upper, q * (upper - lower), ctx.exp(-q * (end - z)) / q


def surface_pressure_psd(incident_psd, *, dps=60):
    """Coherent rigid reflection: p_surface=2*p_inc; PSD_surface=4*PSD_inc."""
    ctx = _ctx(dps)
    return 4 * _input(ctx, incident_psd)


def _direction_tensor(ctx, *, hemisphere):
    # Each entry integrates the actual conjugated directional outer product.
    if hemisphere:
        polar_end, norm = ctx.pi / 2, 2 * ctx.pi
    else:
        polar_end, norm = ctx.pi, 4 * ctx.pi
    # Separate polar and azimuth powers rather than costly nested adaptive sums.
    polar = (
        lambda t: ctx.sin(t),
        lambda t: ctx.sin(t),
        (lambda t: ctx.sin(t)) if hemisphere else (lambda t: ctx.cos(t)),
    )
    azi = (
        lambda p: ctx.j * ctx.cos(p) if hemisphere else ctx.cos(p),
        lambda p: ctx.j * ctx.sin(p) if hemisphere else ctx.sin(p),
        lambda _p: ctx.mpf(1),
    )
    return [
        [
            ctx.quad(
                lambda t, i=i, j=j: polar[i](t) * polar[j](t) * ctx.sin(t),
                [0, polar_end],
            )
            * ctx.quad(
                lambda p, i=i, j=j: azi[i](p) * ctx.conj(azi[j](p)),
                [0, ctx.pi, 2 * ctx.pi],
            )
            / norm
            for j in range(3)
        ]
        for i in range(3)
    ]


def hemisphere_tensor(*, dps=60):
    return _direction_tensor(_ctx(dps), hemisphere=True)


def isotropic_tensor(*, dps=60):
    return _direction_tensor(_ctx(dps), hemisphere=False)


def _angular_points(ctx, x, resolution):
    points = [ctx.mpf(0), ctx.pi / 2]
    if x > 1:
        points += [ctx.asin(j / x) for j in [1, 2, 4, 8, 16, 32, 64] if j < x]
    base = sorted(set(points))
    return sorted(
        set(
            base
            + [
                a + (b - a) * i / resolution
                for a, b in zip(base, base[1:], strict=False)
                for i in range(1, resolution)
            ]
        )
    )


def depth_integral(x, *, derivative=0, dps=60, resolution=1, route="angular"):
    """I(x), I'(x) or I''(x); theta and mu formulations, finite domains."""
    ctx = _ctx(dps)
    x = _input(ctx, x)
    if derivative not in (0, 1, 2) or resolution not in (1, 2, 4):
        raise ValueError("derivative=0/1/2 and resolution=1/2/4 required")
    theta = _angular_points(ctx, x, resolution)
    sign = (-1) ** derivative
    if route == "angular":
        return sign * ctx.quad(
            lambda t: ctx.sin(t) ** (3 + derivative) * ctx.exp(-x * ctx.sin(t)) / 2,
            theta,
            maxdegree=8,
        )
    if route == "mu":
        points = sorted({ctx.cos(t) for t in theta})
        return sign * ctx.quad(
            lambda u: (
                (1 - u * u) ** (ctx.mpf(1) + ctx.mpf(derivative) / 2)
                * ctx.exp(-x * ctx.sqrt(1 - u * u))
                / 2
            ),
            points,
            maxdegree=8,
        )
    raise ValueError("unknown depth formulation")


def volume_kernel(x, *, dps=60, resolution=1, route="angular"):
    """Spherical-volume dimensionless gravity amplitude, x=kR>=0.

    Radial route uses a cancellation-free small-t identity derived from the
    Newton angular integral; its series remainder is resolved by arbitrary-
    precision summation, not a fixed small-kR production branch.
    """
    ctx = _ctx(dps)
    x = _input(ctx, x)
    if resolution not in (1, 2, 4):
        raise ValueError("resolution=1/2/4 required")
    if route == "angular":
        n = max(resolution, int(ctx.ceil(x / ctx.pi)) * resolution)
        if n > 10000:
            raise ValueError("volume oscillation resource limit; unresolved")
        return ctx.quad(
            lambda u: 2 * ctx.sin(x * u / 2) ** 2,
            [ctx.mpf(i) / n for i in range(n + 1)],
            maxdegree=8,
        )
    if route == "radial":

        def integrand(t):
            if t == 0:
                return ctx.mpf(0)
            if abs(t) < ctx.mpf("0.5"):
                # sin(t)-t*cos(t) = sum (-1)^n*2n*t^(2n+1)/(2n+1)!
                return ctx.fsum(
                    [
                        (-1) ** (n + 1)
                        * 2
                        * n
                        * t ** (2 * n - 1)
                        / ctx.factorial(2 * n + 1)
                        for n in range(1, dps + 10)
                    ]
                )
            return (ctx.sin(t) - t * ctx.cos(t)) / (t * t)

        n = max(1, int(ctx.ceil(x / ctx.pi)) * resolution)
        return ctx.quad(integrand, [x * i / n for i in range(n + 1)], maxdegree=8)
    raise ValueError("unknown volume formulation")


def a2_required_grid():
    """Coordinate proposal ONLY; no formal fixtures or outputs generated."""
    ctx = _ctx(80)
    values = [ctx.mpf(x) for x in ["0", "1e-6", "0.75", "18", "30", "100", "1000"]]
    for boundary in ["0.75", "18"]:
        for eps in ["1e-8", "1e-6", "1e-4"]:
            for sign in [-1, 1]:
                values.append(ctx.mpf(boundary) * (1 + sign * ctx.mpf(eps)))
    return [decimal_string(x, 60) for x in sorted(set(values))]


def _estimate(function, x, *, dps, digits, alternate, derivative=0):
    if dps < digits + 15:
        raise ValueError("at least 15 working guard digits required")
    kwargs = {"derivative": derivative} if function is depth_integral else {}
    low = function(x, dps=dps, resolution=4, **kwargs)
    samples = [function(x, dps=dps + 20, resolution=n, **kwargs) for n in [1, 2, 4]]
    other = function(x, dps=dps + 20, resolution=4, route=alternate, **kwargs)
    ctx = _ctx(dps + 30)
    central = samples[-1]
    scale = max(abs(central), ctx.mpf("1e-" + str(dps)))
    rounding = ctx.mpf(10) ** (ctx.floor(ctx.log10(scale)) - digits + 1) / 2
    components = {
        "precision": 10 * abs(low - central),
        "quadrature": 10
        * max(abs(samples[2] - samples[1]), abs(samples[1] - samples[0])),
        "formulation": 10 * abs(other - central),
        "input": abs(ctx.mpf(x)) * ctx.mpf(10) ** (-dps + 1),
        "tail": (
            ctx.mpf(0)
            if function is depth_integral
            else min(ctx.mpf(x), ctx.mpf("0.5")) ** (2 * (dps + 30))
            / ctx.factorial(2 * (dps + 30) + 1)
        ),
        "root": ctx.mpf(0),
        # Heuristic 1e5 operation allowance at the LOW working precision.
        "roundoff": ctx.mpf("1e5") * ctx.mpf(10) ** (-dps) * scale,
        "serialization": rounding,
    }
    total = ctx.fsum(components.values())
    # Convergence threshold is a development method criterion, NOT acceptance.
    resolved = all(ctx.isfinite(v) and v >= 0 for v in components.values())
    resolved = resolved and total < scale * ctx.mpf(10) ** (-digits + 5)

    # Serialize component allowances upward with a decimal ulp; central values
    # use ties-to-even. This does not turn numerical estimates into bounds.
    def allowance(v):
        if not v:
            return "0e+0"
        ulp = ctx.mpf(10) ** (ctx.floor(ctx.log10(v)) - digits + 1)
        return decimal_string(v + ulp, digits)

    serial_components = {k: allowance(v) for k, v in components.items()}
    serial_total = ctx.fsum(ctx.mpf(v) for v in serial_components.values())
    return {
        "value": decimal_string(central, digits),
        "uncertainty_total": allowance(serial_total),
        "components": serial_components,
        "resolved": bool(resolved),
        "uncertainty_kind": "numerical_estimate_not_rigorous_bound",
        "precision_dps": [dps, dps + 20],
        "resolutions": [1, 2, 4],
        "trace": [decimal_string(v, dps) for v in [low, *samples, other]],
        "assumptions": {
            "input": "exact decimal input; conversion allowance via derivative <=1",
            "root": "no root solve in these quadratures",
            "tail": (
                "entire finite integration domain included"
                if function is depth_integral
                else "finite domain; alternate radial Taylor first-omitted-term bound"
            ),
            "roundoff": "heuristic 100000-operation relative allowance",
            "formulation": "paired coordinate formulation disagreement estimate",
        },
    }


def depth_estimate(x, *, derivative=0, dps=60, digits=40):
    return _estimate(
        depth_integral, x, dps=dps, digits=digits, alternate="mu", derivative=derivative
    )


def volume_estimate(x, *, dps=60, digits=40):
    return _estimate(volume_kernel, x, dps=dps, digits=digits, alternate="radial")


SOURCE_PINS = (
    {
        "version": "Harms LRR22,6 (2019) journal PDF",
        "equations_pages": "131:p69;132:p70;160:p82",
        "sha256": "150ec4b02097cd00a744cd67ed01dc3e9c3a3ebfd0a40c7ec68bce0467906376",
    },
    {
        "version": "Amann arXiv:2003.03434v2 (2020-06-14)",
        "equations_pages": "10-13:p9",
        "sha256": "cfaa10667acc51114f1e5eb48d3225084fe2f1339d0582d08096b4b5548f0069",
    },
)
CSD_COMPONENTS = {
    "kernel_linear",
    "kernel_quadratic",
    "direction",
    "kernel_direction_interaction",
    "source_input_roundoff",
    "g_conversion",
    "serialization",
}
CSD_CHECKS = {
    "kernel",
    "direction",
    "source",
    "g_conversion",
    "serialization_g_free",
    "serialization_si",
}
_CANONICAL = re.compile(r"(?:0e\+0|-?[1-9](?:\.\d*[1-9])?e(?:\+0|[+-][1-9]\d*))\Z")


def _newton_gradient(ctx, f, cs, theta, height, convention):
    """Radial-horizontal and vertical Newton gradients per pressure amplitude.

    Phi/G = -2*pi*J/(c_s^2*q), where J is the vertical Green integral.
    Thus H_horizontal/G = i*2*pi*J/c_s^2 and H_vertical/G =
    2*pi*J'/(c_s^2*q). The horizontal i/azimuth is applied by the caller.
    J,J' are the analytic split of the independently checked positive-height
    Green integral. z=0 is its analytically justified upper-side limit.
    For incident pressure, density=2*p_inc*cos(b*t)/c_s^2; for total surface
    pressure, density=p_surface*cos(b*t)/c_s^2. No stochastic prefactor is
    inserted independently of this gradient.
    """
    if convention not in ("surface_total", "incident"):
        raise ValueError("pressure convention must be surface_total or incident")
    k = 2 * ctx.pi * f / cs
    q, b = k * ctx.sin(theta), k * ctx.cos(theta)
    if q == 0:
        # q->0 at fixed k: the finite-height vertical limit is bounded.
        j, vertical = ctx.mpf(0), -2 * ctx.sin(k * height) / k
    else:
        j, dj = green_closed(q, b, height, context=ctx)
        vertical = dj / q
    density_per_pressure = (2 if convention == "incident" else 1) / (cs * cs)
    return (
        2 * ctx.pi * density_per_pressure * j,
        2 * ctx.pi * density_per_pressure * vertical,
    )


def newton_directional_transfer(
    frequency_hz,
    sound_speed_m_s,
    theta,
    azimuth,
    *,
    height_m="0",
    pressure_convention="surface_total",
    dps=60,
):
    """Independent G-free vector acceleration/pressure transfer, upper hemisphere."""
    ctx = _ctx(dps)
    f = _input(ctx, frequency_hz, positive=True)
    cs = _input(ctx, sound_speed_m_s, positive=True)
    theta, azimuth = _input(ctx, theta), _input(ctx, azimuth)
    if theta > ctx.pi / 2 or azimuth > 2 * ctx.pi:
        raise ValueError("upper-hemisphere direction required")
    horizontal, vertical = _newton_gradient(
        ctx, f, cs, theta, _input(ctx, height_m), pressure_convention
    )
    return [
        ctx.j * ctx.cos(azimuth) * horizontal,
        ctx.j * ctx.sin(azimuth) * horizontal,
        vertical,
    ]


def newton_covariance(
    frequency_hz,
    sound_speed_m_s,
    *,
    height_m="0",
    pressure_convention="surface_total",
    dps=60,
    resolution=1,
):
    """Integrate H H^dagger per integrated directional pressure PSD.

    This integrates the Newton-gradient amplitudes, including c_s and k,
    rather than supplying S_p/(c_s*f)^2 as a separate CSD prefactor.
    Polar/azimuth separation is exact because the Green function is azimuth
    independent. Every polar entry evaluates the actual Newton transfer.
    """
    ctx = _ctx(dps)
    f, cs = (
        _input(ctx, frequency_hz, positive=True),
        _input(ctx, sound_speed_m_s, positive=True),
    )
    height = _input(ctx, height_m)
    if resolution not in (1, 2, 4):
        raise ValueError("resolution=1/2/4 required")
    polar = [ctx.pi * i / (2 * resolution) for i in range(resolution + 1)]
    azimuth = [ctx.pi * i / resolution for i in range(2 * resolution + 1)]
    azi = (
        lambda p: ctx.j * ctx.cos(p),
        lambda p: ctx.j * ctx.sin(p),
        lambda _p: ctx.mpf(1),
    )

    def amplitude(t, i):
        horizontal, vertical = _newton_gradient(
            ctx, f, cs, t, height, pressure_convention
        )
        return horizontal if i < 2 else vertical

    return [
        [
            ctx.quad(
                lambda t, i=i, j=j: (
                    amplitude(t, i) * ctx.conj(amplitude(t, j)) * ctx.sin(t)
                ),
                polar,
                maxdegree=8,
            )
            * ctx.quad(
                lambda p, i=i, j=j: azi[i](p) * ctx.conj(azi[j](p)),
                azimuth,
                maxdegree=8,
            )
            / (2 * ctx.pi)
            for j in range(3)
        ]
        for i in range(3)
    ]


def _frobenius(ctx, values):
    return ctx.sqrt(ctx.fsum(abs(ctx.mpc(v)) ** 2 for v in values))


def _flat(matrix):
    return [v for row in matrix for v in row]


def _upper(ctx, value, digits):
    if not ctx.isfinite(value) or value < 0:
        raise ValueError("finite nonnegative uncertainty required")
    if value == 0:
        return "0e+0"
    return decimal_string(
        value + ctx.mpf(10) ** (ctx.floor(ctx.log10(value)) - digits + 1), digits
    )


def _public_components(ctx, components, digits):
    """Upward serialize each allowance, sum the PUBLIC strings, then upward total.

    Component sets are disjoint error accounting terms in the same CSD norm
    units. Higher-precision reparsing precedes the aggregate; no old unrounded
    aggregate is reused and dimensionless scalar U is never added to CSD U.
    """
    if set(components) != CSD_COMPONENTS:
        raise ValueError("missing or unknown CSD uncertainty component")
    public = {name: _upper(ctx, ctx.mpf(v), digits) for name, v in components.items()}
    with localcontext() as dc:
        dc.prec = digits + 80
        exact_sum = sum((Decimal(v) for v in public.values()), Decimal(0))
    return public, _upper(ctx, ctx.mpf(str(exact_sum)), digits)


def _serialization(ctx, values, digits):
    allowance = []
    for value in values:
        for part in (ctx.re(value), ctx.im(value)):
            allowance.append(
                ctx.mpf(0)
                if part == 0
                else ctx.mpf(10) ** (ctx.floor(ctx.log10(abs(part))) - digits + 1) / 2
            )
    return _frobenius(ctx, allowance)


def csd_record_resolved(record, *, digits=40):
    """Complete development-method check; NOT admission under frozen F.

    Missing/noncanonical/nonfinite scientific fields, invalid components,
    totals below PUBLIC component sums, and any failed method stage reject.
    Both G-free and SI value/uncertainty serialization must be valid. A method
    convergence criterion is checked separately for each observable norm.
    """
    try:
        ctx = _ctx(digits + 100)
        scientific_source = [
            record["source_scale_g_free"],
            *record["input_tuple"].values(),
        ]
        if any(
            not isinstance(s, str) or not _CANONICAL.fullmatch(s)
            for s in scientific_source
        ):
            return False
        if any(
            not ctx.isfinite(ctx.mpf(s)) or ctx.mpf(s) < 0 for s in scientific_source
        ):
            return False
        for name in ("frequency_hz", "sound_speed_m_s", "pressure_psd_pa2_hz", "g_si"):
            amount = ctx.mpf(record["input_tuple"][name])
            if amount < 0 or (name != "pressure_psd_pa2_hz" and amount == 0):
                return False
        expected_units = {"g_free": "(m/s^2)^2/Hz/G^2", "si": "m^2 s^-4 Hz^-1"}
        if (
            record["component_units"] != expected_units
            or record["units"] != expected_units
        ):
            return False
        if record["shape"] != [3, 3] or set(record["method_checks"]) != CSD_CHECKS:
            return False
        if not all(record["method_checks"][name] is True for name in CSD_CHECKS):
            return False
        for domain in ("g_free", "si"):
            values = record["values_" + domain]
            if len(values) != 9 or any(len(v) != 2 for v in values):
                return False
            scalar_strings = [s for v in values for s in v]
            components = record["uncertainty_components_" + domain]
            if set(components) != CSD_COMPONENTS:
                return False
            scalar_strings += list(components.values()) + [
                record["uncertainty_" + domain]
            ]
            if any(
                not isinstance(s, str) or not _CANONICAL.fullmatch(s)
                for s in scalar_strings
            ):
                return False
            amounts = [ctx.mpf(s) for s in components.values()]
            total = ctx.mpf(record["uncertainty_" + domain])
            if any(not ctx.isfinite(a) or a < 0 for a in [*amounts, total]):
                return False
            with localcontext() as dc:
                dc.prec = digits + 100
                if Decimal(record["uncertainty_" + domain]) < sum(
                    (Decimal(s) for s in components.values()), Decimal(0)
                ):
                    return False
            vector = [ctx.mpc(ctx.mpf(re), ctx.mpf(im)) for re, im in values]
            norm = _frobenius(ctx, vector)
            if norm == 0:
                if total != 0 or not record.get("zero_certificate"):
                    return False
            elif total <= 0 or total >= norm * ctx.mpf(10) ** (-digits + 5):
                return False
        return True
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return False


def _reference_metadata(fixture, convention):
    def exact_input(value):
        dec = Decimal(value)
        if not dec.is_finite():
            raise ValueError("nonfinite input tuple")
        if not dec:
            return "0e+0"
        sign, coeff, exp = dec.as_tuple()
        chars = "".join(map(str, coeff)).rstrip("0")
        power = len(coeff) + exp - 1
        mantissa = chars[0] + ("." + chars[1:] if len(chars) > 1 else "")
        return (
            ("-" if sign else "")
            + mantissa
            + "e"
            + ("+" if power >= 0 else "-")
            + str(abs(power))
        )

    inputs = {}
    for name in (
        "frequency_hz",
        "sound_speed_m_s",
        "pressure_psd_pa2_hz",
        "g_si",
        "depth_m",
        "radius_m",
    ):
        if name in fixture:
            # Retain the EXACT provided decimal tuple, without truncation to
            # output digits. Numeric field spelling is canonicalized only.
            inputs[name] = exact_input(fixture[name])
    return {
        "key": [
            fixture["family"],
            fixture.get("fixture_id", "development-unspecified"),
            inputs["frequency_hz"],
            fixture.get("target_id", "development-unspecified"),
            fixture.get("observable", "acceleration_csd"),
            fixture.get("representation", "one_sided_per_hz_csd"),
            fixture.get("set", "development"),
        ],
        "input_tuple": inputs,
        "source_pins": [dict(pin) for pin in SOURCE_PINS],
        "units": {"g_free": "(m/s^2)^2/Hz/G^2", "si": "m^2 s^-4 Hz^-1"},
        "pressure_convention": convention,
        "shape": [3, 3],
        "claim": "development_method_check_not_frozen_policy_admission",
    }


def acoustic_reference(fixture, *, dps=60, digits=40):
    """Review-only CSD; actual A1 Newton gradient and complete uncertainty gates.

    A1 directly integrates the surface-limit directional Newton gradient outer
    product. A2/A3 retain independent scalar and direction formulations. Public
    G-free/SI components are nonoverlapping Frobenius allowances. Scalar core
    estimates are dimensionless metadata and are propagated, not added twice.
    This provisional adapter NEVER establishes F/E/sealer formal admission.
    """
    if dps < digits + 15:
        raise ValueError("at least 15 working guard digits required")
    ctx = _ctx(dps + 100)
    family = fixture["family"]
    if family not in ("A1", "A2", "A3"):
        raise ValueError("unsupported acoustic family")
    f = _input(ctx, fixture["frequency_hz"], positive=True)
    cs = _input(ctx, fixture["sound_speed_m_s"], positive=True)
    pressure = _input(ctx, fixture["pressure_psd_pa2_hz"])
    grav = _input(ctx, fixture["g_si"], positive=True)
    convention = fixture.get(
        "pressure_convention", "cavern" if family == "A3" else "surface_total"
    )
    if (family == "A3" and convention != "cavern") or (
        family != "A3" and convention not in ("surface_total", "incident")
    ):
        raise ValueError("invalid family pressure convention")
    dim = ctx.mpf(0)
    if family == "A2":
        dim = 4 * ctx.pi * f * _input(ctx, fixture["depth_m"]) / cs
    elif family == "A3":
        dim = 2 * ctx.pi * f * _input(ctx, fixture["radius_m"]) / cs
    source_scale = (
        pressure
        / (cs * f) ** 2
        * (4 if family == "A3" or convention == "incident" else 1)
    )
    metadata = _reference_metadata(fixture, convention)
    checks = dict.fromkeys(CSD_CHECKS, False)
    try:
        zero = pressure == 0 or (family == "A3" and dim == 0)
        components = {name: ctx.mpf(0) for name in CSD_COMPONENTS}
        core = None
        if zero:
            values = [ctx.mpc(0) for _ in range(9)]
            metadata["zero_certificate"] = (
                "linear_transfer_times_exact_zero_input_psd"
                if pressure == 0
                else "zero_radius_empty_spherical_volume"
            )
            checks.update(dict.fromkeys(CSD_CHECKS, True))
        elif family == "A1":
            args = {"pressure_convention": convention}
            low = _flat(
                newton_covariance(
                    fixture["frequency_hz"],
                    fixture["sound_speed_m_s"],
                    dps=dps,
                    resolution=4,
                    **args,
                )
            )
            samples = [
                _flat(
                    newton_covariance(
                        fixture["frequency_hz"],
                        fixture["sound_speed_m_s"],
                        dps=dps + 20,
                        resolution=n,
                        **args,
                    )
                )
                for n in (1, 2, 4)
            ]
            high = samples[-1]

            def distance(a, b):
                return _frobenius(
                    ctx, [ctx.mpc(x) - ctx.mpc(y) for x, y in zip(a, b, strict=True)]
                )

            kernel_u = 10 * (
                distance(low, high)
                + max(
                    distance(samples[0], samples[1]), distance(samples[1], samples[2])
                )
            )
            kernel_u += _frobenius(ctx, high) * ctx.mpf("1e5") * ctx.mpf(10) ** (-dps)
            values = [pressure * ctx.mpc(v) for v in high]
            components["kernel_linear"] = pressure * kernel_u
            # Independent exact hemisphere moments are a direction/formulation
            # crosscheck of the Green-derived CSD, not its computational path.
            expected = [
                source_scale
                * (ctx.mpf(1) / 3 if i in (0, 4) else ctx.mpf(2) / 3 if i == 8 else 0)
                for i in range(9)
            ]
            components["direction"] = 10 * distance(values, expected)
            norm = _frobenius(ctx, values)
            checks["kernel"] = all(ctx.isfinite(v) for v in high) and components[
                "kernel_linear"
            ] < norm * ctx.mpf(10) ** (-digits + 5)
            checks["direction"] = components["direction"] < norm * ctx.mpf(10) ** (
                -digits + 5
            )
            core = {
                "method": "directional_newton_gradient_outer_product_surface_limit",
                "pressure_density_relation": (
                    "delta_rho=p_surface*cos(b*z)/c_s^2; p_surface=2*p_inc"
                ),
                "precision_dps": [dps, dps + 20],
                "resolutions": [1, 2, 4],
                "tail": "0e+0",
                "tail_reason": "analytic split to infinity; no truncation",
                "uncertainty_unit": "G-free acceleration CSD per pressure PSD",
            }
        else:
            x = decimal_string(dim, dps + 30)
            core = (
                volume_estimate(x, dps=dps, digits=digits)
                if family == "A3"
                else depth_estimate(x, dps=dps, digits=digits)
            )
            if set(core["components"]) != UNCERTAINTY_COMPONENTS:
                raise ValueError("missing scalar-kernel component")
            scalar_components = [ctx.mpf(v) for v in core["components"].values()]
            if not all(ctx.isfinite(v) and v >= 0 for v in scalar_components):
                raise ValueError("invalid scalar-kernel component")
            v, u = ctx.mpf(core["value"]), ctx.mpf(core["uncertainty_total"])
            if (
                not ctx.isfinite(v)
                or not ctx.isfinite(u)
                or u < ctx.fsum(scalar_components)
            ):
                raise ValueError("invalid scalar-kernel estimate or aggregate")
            core["uncertainty_unit"] = "dimensionless"
            angular = (
                isotropic_tensor(dps=dps + 20)
                if family == "A3"
                else hemisphere_tensor(dps=dps + 20)
            )
            flat = [ctx.mpc(a) for a in _flat(angular)]
            if len(flat) != 9 or not all(ctx.isfinite(a) for a in flat):
                raise ValueError("invalid direction tensor")
            tensor_norm = _frobenius(ctx, flat)
            ideal = [
                ctx.mpf(1) / 3,
                ctx.mpf(1) / 3,
                ctx.mpf(1) / 3 if family == "A3" else ctx.mpf(2) / 3,
            ]
            residual = _frobenius(
                ctx,
                [
                    flat[3 * i + j] - (ideal[i] if i == j else 0)
                    for i in range(3)
                    for j in range(3)
                ],
            )
            angular_u = (
                10 * residual + ctx.mpf("1e5") * ctx.mpf(10) ** (-dps) * tensor_norm
            )
            factor = v * v if family == "A3" else 3 * v
            linear = 2 * abs(v) * u if family == "A3" else 3 * u
            quadratic = u * u if family == "A3" else ctx.mpf(0)
            values = [source_scale * factor * a for a in flat]
            components["kernel_linear"] = source_scale * tensor_norm * linear
            components["kernel_quadratic"] = source_scale * tensor_norm * quadratic
            components["direction"] = source_scale * abs(factor) * angular_u
            components["kernel_direction_interaction"] = (
                source_scale * (linear + quadratic) * angular_u
            )
            checks["kernel"] = core["resolved"] is True
            checks["direction"] = angular_u < tensor_norm * ctx.mpf(10) ** (-digits + 5)
        norm = _frobenius(ctx, values)
        if not zero:
            # Exact decimal model inputs; numerical conversion/source arithmetic
            # estimate only. No empirical physical-parameter error is imputed.
            components["source_input_roundoff"] = (
                norm * ctx.mpf("1e5") * ctx.mpf(10) ** (-dps)
            )
            checks["source"] = ctx.isfinite(source_scale) and source_scale > 0
        si_values = [a * grav * grav for a in values]
        si_components = {
            name: amount * grav * grav for name, amount in components.items()
        }
        # Free central values and SI central values are serialized independently:
        # don't rescale free SERIALIZATION allowance into SI then add it again.
        components["serialization"] = _serialization(ctx, values, digits)
        si_components["serialization"] = _serialization(ctx, si_values, digits)
        si_components["g_conversion"] = (
            _frobenius(ctx, si_values) * ctx.mpf("1e5") * ctx.mpf(10) ** (-dps)
        )
        checks["g_conversion"] = (
            ctx.isfinite(grav) and grav > 0 and all(ctx.isfinite(a) for a in si_values)
        )
        free_public, free_total = _public_components(ctx, components, digits)
        si_public, si_total = _public_components(ctx, si_components, digits)

        def pairs(vector):
            return [
                [decimal_string(ctx.re(a), digits), decimal_string(ctx.im(a), digits)]
                for a in vector
            ]

        free_pairs, si_pairs = pairs(values), pairs(si_values)
        checks["serialization_g_free"] = all(
            _CANONICAL.fullmatch(s) for v in free_pairs for s in v
        )
        checks["serialization_si"] = all(
            _CANONICAL.fullmatch(s) for v in si_pairs for s in v
        )
        record = {
            **metadata,
            "values_g_free": free_pairs,
            "values_si": si_pairs,
            "uncertainty_components_g_free": free_public,
            "uncertainty_components_si": si_public,
            "uncertainty_g_free": free_total,
            "uncertainty_si": si_total,
            "source_scale_g_free": decimal_string(source_scale, digits),
            "method_checks": checks,
            "core_estimate": core,
            "uncertainty_kind": "exact_whole_observable_zero"
            if zero
            else "numerical_estimate_not_rigorous_bound",
            "component_units": metadata["units"],
            "component_definitions": {
                "kernel_linear": "kernel estimate once propagated into CSD units",
                "kernel_quadratic": "quadratic scalar-amplitude uncertainty; A3 only",
                "direction": "directional moment formulation/integration allowance",
                "kernel_direction_interaction": "kernel/direction allowance product",
                "source_input_roundoff": "pressure/scaling arithmetic after kernel",
                "g_conversion": "SI G parsing/squared scaling; G-free zero",
                "serialization": "final central CSD serialization in this domain only",
            },
            "scope": "centered_spherical_volume_kernel_and_published_cavern_model_only"
            if family == "A3"
            else "ideal_rigid_reflecting_atmospheric_halfspace",
        }
        record["resolved"] = csd_record_resolved(record, digits=digits)
        return record
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        # Missing components and numerical failures retain unknown uncertainty,
        # not zero. Invalid caller inputs are rejected above this block.
        return {
            **metadata,
            "resolved": False,
            "method_checks": checks,
            "values_g_free": None,
            "values_si": None,
            "uncertainty_g_free": None,
            "uncertainty_si": None,
            "failure": type(exc).__name__ + ": " + str(exc),
        }
