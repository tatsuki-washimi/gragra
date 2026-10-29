"""Independent positive parent-streamline integrals for T1 (development only).

Source: Creighton, arXiv:gr-qc/0007050v2, Eq.14/15 p6 and A1/A4/A7
pp13-14. PDF SHA256 a1e813967f1f822083b297c24008f7a843fa741e296082a0d4646e6426d80aff.

This module never imports gragra and never reads production output. mpmath's
arbitrary-precision unscaled K evaluates the streamline Fourier kernel. No
closed Bessel differences or production kve algorithm are used by the generator.
The independent Laplace representation is a kernel check, not a fitted formula.

Source discrepancies: printed Eq16 has negative differences. Detector-removed
Eq10/A7 coefficient ratio is pi/2. Printed A2 omits the one-sided factor 2;
A3 is half the spectrum derived from A1+A4. None anchors normalization.
Rederivation from A1/A4 retains A1's factor 2 and uses the transverse bundle
measure 2*pi*r0*dr0*v*dtau0. Formal admission still requires frozen family
policy and implementation review.

A1 defines S(f)=2 integral C(tau) exp(2*pi*i*f*tau) dtau, one-sided per Hz.
A4 assumes D_T(r)=cT^2*r^p, with covariance sigma^2-D_T/2. The parent
approximation replaces kernels on nearby streamlines by their central value;
this is a high-frequency narrow-bundle approximation, not exact homogeneous
field integration at all frequencies. Physical applicability requires
v/omega << rmin (x >> 1), smooth large-scale correlation cutoff and inertial
range statistics. Small/order-one x tests verify the mathematical integral only.

For uniform v, Gtilde=-2*omega/v^2*(i*cos(theta)*K0(omega*r/v)
 +sin(theta)*cos(psi)*K1(omega*r/v)), Ftilde=v^(p+3)*Gtilde.
The imaginary/real cross term vanishes. w=v and dA=r*dr*dpsi on the
above-ground half plane; integral dpsi=pi, integral cos(psi)^2 dpsi=pi/2.
With a(p)=2/sqrt(pi)*sin(pi*p/2)*Gamma((p+2)/2), duplication gives
Gamma((p+2)/2)*Gamma((p+3)/2)=2^(-p-1)*sqrt(pi)*Gamma(p+2).
A7's (pi*f)^(-p-3)*a(p)*(pi/2)*Gamma((p+3)/2) therefore equals
4*pi*Gamma(p+2)*sin(pi*p/2)*omega^(-p-3). No detector response is applied.
The resulting G-free conditional PSD is 16*pi^2*Gamma(p+2)*sin(pi*p/2)
*(rho/T)^2*cT^2*v^(p+2)*omega^(-p-3)*(cos^2(theta)*I0+sin^2(theta)*I1/2),
where I_n=integral_x^infinity t*K_n(t)^2 dt and x=omega*rmin/v.

All returned uncertainties are NUMERICAL ESTIMATES, not rigorous total-error
theorems. The omitted radial tail alone has an analytic bound. Model error,
measurement uncertainty, and physical approximation error are not assigned
zero or absorbed into numerical uncertainty. Formal acceptance is prohibited.
"""

import re
from decimal import ROUND_HALF_EVEN, Decimal, localcontext

import mpmath as mp

SOURCE = {
    "url": "https://arxiv.org/pdf/gr-qc/0007050v2",
    "version": "arXiv:gr-qc/0007050v2",
    "equations": ["14", "15", "A1", "A4", "A7"],
    "pages": [6, 13, 14],
    "sha256": "a1e813967f1f822083b297c24008f7a843fa741e296082a0d4646e6426d80aff",
}
NORMALIZATION_STATUS = "DEVELOPMENT_ONLY_PENDING_F_E_REVIEW"
DECIMAL_PATTERN = re.compile(
    r"(?:0e\+0|-?[1-9](?:\.[0-9]*[1-9])?e[+-](?:0|[1-9][0-9]*))\Z"
)


def decimal_string(value, digits=45):
    """Round nearest/ties-even to significant digits; never pass through float."""
    value = mp.mpf(value)
    if not mp.isfinite(value):
        raise ValueError("nonfinite scientific number")
    if value == 0:
        return "0e+0"
    if digits < 2:
        raise ValueError("at least two serialization digits required")
    # nstr includes extra digits so Decimal makes the final rounding decision.
    raw = mp.nstr(value, n=digits + 12, strip_zeros=False)
    with localcontext() as ctx:
        ctx.prec = digits
        ctx.rounding = ROUND_HALF_EVEN
        number = +Decimal(raw)
    sign, mantissa, exponent = number.as_tuple()
    significant = "".join(str(d) for d in mantissa)
    power = len(significant) + exponent - 1
    significant = significant.rstrip("0")
    fraction = "." + significant[1:] if len(significant) > 1 else ""
    return ("-" if sign else "") + significant[0] + fraction + "e" + f"{power:+d}"


def scientific_input(value):
    """Inputs must be exact canonical finite decimal strings, never JSON floats."""
    if not isinstance(value, str) or not DECIMAL_PATTERN.fullmatch(value):
        raise ValueError("noncanonical scientific input")
    return mp.mpf(value)


def parent_coefficient(p):
    """Literal A7 coefficient expressed in omega, before the geometry integral."""
    a = 2 / mp.sqrt(mp.pi) * mp.sin(mp.pi * p / 2) * mp.gamma((p + 2) / 2)
    return 2 ** (p + 3) * a * mp.pi / 2 * mp.gamma((p + 3) / 2)


def structure_function_coefficient(p):
    """Independent A1+A4 temporal Fourier and transverse-bundle normalization.

    At f>0 the constant covariance vanishes. One-sided 2 times covariance
    -cT²*r^p/2 gives -cT² times the Fourier transform of r^p. Analytic
    continuation of the Gaussian/Schwinger transform gives
    -2*sqrt(pi)*2^nu/Gamma(-p/2)*t^nu*Knu(t), nu=(p+1)/2,
    multiplying cT²*v^p*omega^(-p-1). Smooth large-scale cutoff is assumed.
    Bundle measure 2*pi*b*db*v*dtau0 supplies 2*pi*v³/omega²;
    integral t^(nu+1)*Knu(t) dt = 2^nu*Gamma(nu+1).
    Reflection/duplication reduces this coefficient to parent_coefficient(p).
    No printed A2/A3 normalization is used. Domain: 0<p<2, omega>0.
    """
    if not 0 < p < 2:
        raise ValueError("structure function requires 0<p<2")
    nu = (p + 1) / 2
    temporal = -2 * mp.sqrt(mp.pi) * 2**nu / mp.gamma(-p / 2)
    bundle = 2 * mp.pi * 2**nu * mp.gamma(nu + 1)
    return temporal * bundle


def refinement_error(values, floor):
    """Estimate from contracting differences or declared precision floor.

    This is a fail-closed development criterion, not a theorem. Final difference
    must contract by at least 2, or both differences must be below the explicit
    numerical floor. A freeze must review this criterion and its floor.
    """
    first, last = abs(values[1] - values[0]), abs(values[2] - values[1])
    if last > first / 2 and max(first, last) > floor:
        raise ValueError("nonconvergent whole-domain refinements")
    return max(10 * (first + last), floor)


def laplace_kernel_estimate(nu, t):
    """K_n(t)=integral exp(-t*cosh(u))*cosh(n*u) du, t>0.

    Finite U is fixed from working precision. Omitted tail is bounded by
    exp(-t*cosh(U)+U)/(t*sinh(U)-1), from cosh(n*u)<=exp(u) and convexity
    of t*cosh(u)-u. This formulation avoids any production special function.
    """
    if nu not in (0, 1) or t <= 0:
        raise ValueError("kernel requires order 0/1 and positive argument")

    cutoff = mp.acosh(1 + (mp.mp.dps + 20) * mp.log(10) / t)
    values = []
    for level in (0, 1, 2):
        count = 2 ** (level + 1)
        points = [cutoff * i / count for i in range(count + 1)]
        values.append(
            mp.quad(
                lambda u: mp.exp(-t * (mp.cosh(u) - 1)) * mp.cosh(nu * u),
                points,
                method="gauss-legendre",
            )
        )
    values = [v * mp.exp(-t) for v in values]
    floor = abs(values[-1]) * mp.power(10, -mp.mp.dps + 8)
    tail = mp.exp(-t * mp.cosh(cutoff) + cutoff) / (t * mp.sinh(cutoff) - 1)
    return {
        "value": values[-1],
        "quadrature": refinement_error(values, floor),
        "tail": tail,
        "cutoff": cutoff,
    }


def laplace_kernel(nu, t):
    """Central value of a kernel-only diagnostic; no full-PSD crosscheck claim."""
    return laplace_kernel_estimate(nu, t)["value"]


def radial_points(x, length, refinement):
    """Bisect EVERY base interval at each level, including the far endpoint."""
    if refinement not in (0, 1, 2):
        raise ValueError("resolution index must be 0, 1, or 2")
    points = [mp.mpf(0)]
    y = min(x, mp.mpf(1))
    while y < length:
        points.append(y)
        y *= 2
    points.append(length)
    for _ in range(refinement):
        refined = [points[0]]
        for left, right in zip(points[:-1], points[1:], strict=True):
            refined.extend([(left + right) / 2, right])
        points = refined
    return points


def radial_integrals(x, dps, refinement, length):
    """Positive parent radial integrals and analytic common absolute tail bound.

    t=x+y maps the finite interval y in [0,L]. At x>0 the endpoint is regular.
    Three subdivisions (refinement=0/1/2) provide resolution estimates; this
    is not a universal convergence guarantee. For R=x+L>1, K0(t) <=
    exp(-t)*sqrt(pi/(2t)); K1(t)<=exp(-t)*sqrt(pi/(2(t-1))) follows from
    cosh(u)>=1+u²/2 and cosh(u)<=exp(u²/2). Thus the omitted integrals are
    <=pi/4*exp(-2R) and <=pi/4*R/(R-1)*exp(-2R). Use the latter for both.
    """
    with mp.workdps(dps):
        x, length = mp.mpf(x), mp.mpf(length)
        if x <= 0 or length <= 0 or x + length <= 1:
            raise ValueError("positive x/L and tail cutoff greater than one required")
        points = radial_points(x, length, refinement)
        values = [
            mp.quad(
                lambda y, nu=nu: (x + y) * (mp.exp(x) * mp.besselk(nu, x + y)) ** 2,
                points,
                method="gauss-legendre",
            )
            for nu in (0, 1)
        ]
        cutoff = x + length
        tail = mp.pi / 4 * cutoff / (cutoff - 1) * mp.exp(-2 * cutoff)
        restore = mp.exp(-2 * x)
        return values[0] * restore, values[1] * restore, tail


def evaluate_calibration(inputs, *, dps=45, guard_digits=15, length="55", digits=40):
    """Conditional A7 PSD with component estimates; calibration/development only.

    Complete inputs: frequency_hz, rho_kg_m3, temperature_k, c_t_squared,
    wind_m_s, r_min_m, p, theta_rad. Exact decimal inputs define this
    mathematical problem; physical parameter uncertainty is excluded, not zero.
    No roots occur. No singular endpoint truncation occurs at x>0. The precision
    and roundoff floors below are heuristic numerical estimates.
    """
    if dps < 30 or guard_digits < 10 or digits > dps - 5:
        raise ValueError("insufficient precision/serialization guards")
    names = (
        "frequency_hz",
        "rho_kg_m3",
        "temperature_k",
        "c_t_squared",
        "wind_m_s",
        "r_min_m",
        "p",
        "theta_rad",
    )
    if set(inputs) != set(names):
        raise ValueError("incomplete or unknown physical inputs")
    with mp.workdps(dps + guard_digits):
        trials = []
        paths = []
        for precision in (dps, dps + guard_digits):
            # Reparse original strings and recompute ALL nonlinear factors at
            # this precision. Never recycle a high-precision geometry/prefactor.
            with mp.workdps(precision):
                f, rho, temp, ct2, wind, radius, p, theta = [
                    scientific_input(inputs[n]) for n in names
                ]
                if (
                    any(v <= 0 for v in (f, rho, temp, ct2, wind, radius))
                    or not 0 < p < 2
                ):
                    raise ValueError("positive physical inputs and 0<p<2 required")
                omega = 2 * mp.pi * f
                x = omega * radius / wind
                scale = (
                    structure_function_coefficient(p)
                    * 4
                    * mp.pi
                    * (rho / temp) ** 2
                    * ct2
                    * wind ** (p + 2)
                    * omega ** (-p - 3)
                )
                weights = (mp.cos(theta) ** 2, mp.sin(theta) ** 2 / 2)
                paths.append(
                    {
                        "dps": precision,
                        "x": decimal_string(x, precision),
                        "prefactor": decimal_string(scale, precision),
                        "weights": [decimal_string(w, precision) for w in weights],
                    }
                )
                for resolution in (0, 1, 2):
                    a, b, tail = radial_integrals(x, precision, resolution, length)
                    trials.append(
                        (
                            precision,
                            resolution,
                            scale * (weights[0] * a + weights[1] * b),
                            scale * (weights[0] + weights[1]) * tail,
                        )
                    )
        value = trials[-1][2]
        # Final two refinement differences at both precisions, conservative sum.
        floor = abs(value) * mp.power(10, -dps + 8)
        quad = sum(
            refinement_error([trials[i + j][2] for j in range(3)], floor)
            for i in (0, 3)
        )
        precision = 10 * max(abs(trials[i + 3][2] - trials[i][2]) for i in range(3))
        if precision > floor:
            raise ValueError("nonconvergent full-path precision agreement")
        kernel_checks = []
        for nu in (0, 1):
            diagnostic = laplace_kernel_estimate(nu, x)
            kernel_delta = abs(diagnostic["value"] - mp.besselk(nu, x))
            kernel_floor = abs(diagnostic["value"]) * mp.power(10, -dps + 8)
            if (
                kernel_delta
                > diagnostic["quadrature"] + diagnostic["tail"] + kernel_floor
            ):
                raise ValueError("nonconvergent Laplace kernel agreement")
            kernel_checks.append(
                {
                    "order": nu,
                    "scope": "STREAMLINE_KERNEL_ONLY_NOT_FULL_PSD",
                    **{
                        key: decimal_string(val, digits)
                        for key, val in diagnostic.items()
                    },
                    "difference": decimal_string(kernel_delta, digits),
                    "serialization": decimal_string(
                        abs(
                            diagnostic["value"]
                            - mp.mpf(decimal_string(diagnostic["value"], digits))
                        )
                        + abs(diagnostic["value"]) * mp.power(10, -digits),
                        digits,
                    ),
                }
            )
        central = decimal_string(value, digits)
        serialization = abs(value - mp.mpf(central)) + abs(value) * mp.power(
            10, -digits
        )
        components = {
            "quadrature": quad,
            "precision": precision,
            "tail": trials[-1][3],
            "input_arithmetic": floor,
            "roundoff": floor,
            "serialization": serialization,
            "endpoint": mp.mpf(0),
            "root": mp.mpf(0),
        }
        # Report a positive upward allowance for rounding each component.
        encoded = {
            k: decimal_string(v * (1 + mp.power(10, -digits + 2)), digits)
            for k, v in components.items()
        }
        total = sum(mp.mpf(v) for v in encoded.values())
        return {
            "status": NORMALIZATION_STATUS,
            "uncertainty_kind": "NUMERICAL_ESTIMATE_NOT_THEOREM",
            "source": SOURCE,
            "full_path_trials": paths,
            "kernel_checks": kernel_checks,
            "convergence_criterion": "differences contract by 2 or both below floor",
            "convergence_floor": decimal_string(floor, digits),
            "shape": [],
            "values": [[central, "0e+0"]],
            "uncertainty_components": encoded,
            "uncertainty": decimal_string(
                total * (1 + mp.power(10, -digits + 2)), digits
            ),
            "x": decimal_string(x, digits),
            "dps": dps,
            "guard_digits": guard_digits,
            "serialization_digits": digits,
            "tail_length": decimal_string(mp.mpf(length), digits),
            "component_assumptions": {
                "input_arithmetic": (
                    "exact decimal tuples; arithmetic estimate only; "
                    "measurement uncertainty excluded"
                ),
                "root": "not applicable: no roots",
                "endpoint": "regular x>0 endpoint; no endpoint truncation",
                "model": "A7 narrow-bundle approximation; requires x >> 1",
            },
            "refinements": [
                {"dps": prec, "resolution": res, "value": decimal_string(v, digits)}
                for prec, res, v, _ in trials
            ],
        }
