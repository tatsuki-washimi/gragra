"""Production-independent B1, R1 and B2 seismic references.

Harms (2019), LRR 22:6, publisher PDF SHA256
150ec4b02097cd00a744cd67ed01dc3e9c3a3ebfd0a40c7ec68bce0467906376:
Eq.55 p33 and Eq.60 p34. Spatial exp(+i k.x), a=-grad(Phi).
Exact mathematical decimal inputs, one-sided per-Hz displacement spectra.
No production modules, outputs, stopping helpers, or binary64 conversion.
Numerical uncertainty is an estimate, never a rigorous theorem.
"""

import re
from functools import lru_cache

import mpmath as mp

SOURCE = {
    "version": "Harms2019-LRR22-6-publisher-2019-10-14",
    "url": "https://link.springer.com/content/pdf/10.1007/s41114-019-0022-2.pdf",
    "sha256": "150ec4b02097cd00a744cd67ed01dc3e9c3a3ebfd0a40c7ec68bce0467906376",
    "equations": ["55", "60"],
    "pages": [33, 34],
}


def decimal_input(value):
    """Refuse silent binary64 promotion and unknown/nonfinite input."""
    if not isinstance(value, str):
        raise TypeError("scientific inputs must be decimal strings")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
        raise ValueError("invalid finite decimal input")
    number = mp.mpf(value)
    if not mp.isfinite(number):
        raise ValueError("nonfinite input")
    return number


def decimal_string(value, digits=40):
    """Canonical finite scientific string; central round-to-nearest."""
    value = mp.mpf(value)
    if not mp.isfinite(value):
        raise ValueError("nonfinite result")
    if value == 0:
        return "0e+0"
    spelling = mp.nstr(value, n=digits, min_fixed=0, max_fixed=0)
    mantissa, exponent = spelling.lower().split("e")
    mantissa = mantissa.rstrip("0").rstrip(".") if "." in mantissa else mantissa
    return f"{mantissa}e{int(exponent):+d}"


def upward_string(value, digits=40):
    """Round a nonnegative uncertainty upward with one decimal ulp padding."""
    if value < 0:
        raise ValueError("negative uncertainty")
    if value == 0:
        return "0e+0"
    exponent = int(mp.floor(mp.log10(value)))
    quantum = mp.power(10, exponent - digits + 1)
    return decimal_string((mp.ceil(value / quantum) + 1) * quantum, digits)


@lru_cache(maxsize=24)
def _gauss(order, dps):
    with mp.workdps(dps):
        nodes, weights = mp.gauss_quadrature(order, "legendre")
        return tuple(zip(nodes, weights, strict=True))


def integrate(function, left, right, order):
    midpoint, half = (left + right) / 2, (right - left) / 2
    return half * mp.fsum(
        weight * function(midpoint + half * node)
        for node, weight in _gauss(order, mp.mp.dps)
    )


def solid_angle_moment(order):
    """Integrate n_i n_j dOmega/(4pi) using smooth theta/phi coordinates.

    Factorization separates two compact integrals, without substituting the
    isotropic identity. No endpoint singularity or infinite-domain tail.
    """
    polar = (mp.sin, mp.sin, mp.cos)
    azimuth = (mp.cos, mp.sin, lambda _phi: mp.mpf(1))
    return [
        [
            integrate(
                lambda t, i=i, j=j: mp.sin(t) * polar[i](t) * polar[j](t),
                mp.mpf(0),
                mp.pi,
                order,
            )
            * integrate(
                lambda p, i=i, j=j: azimuth[i](p) * azimuth[j](p),
                mp.mpf(0),
                2 * mp.pi,
                order,
            )
            / (4 * mp.pi)
            for j in range(3)
        ]
        for i in range(3)
    ]


def _vector_number(value, *, real=False):
    if isinstance(value, (bool, float, complex)):
        raise TypeError("binary64/bool vector input prohibited")
    if isinstance(value, str):
        number = decimal_input(value)
    elif isinstance(value, (int, mp.mpf, mp.mpc)):
        number = value
    else:
        raise TypeError("exact integer/decimal or internal mpmath value required")
    if not mp.isfinite(number) or (real and mp.im(number) != 0):
        raise ValueError("finite real wavevector and finite displacement required")
    return mp.mpf(number) if real else mp.mpc(number)


def fourier_poisson_fields(density, wavevector, displacement):
    """Solve -k² Phi=4pi delta_rho with delta_rho=-i rho k.xi.

    This longitudinal projection also provides the independent transverse S
    null reference, mode by mode, without evaluating any stochastic tensor.
    """
    rho = decimal_input(density)
    k = [_vector_number(v, real=True) for v in wavevector]
    xi = [_vector_number(v) for v in displacement]
    if len(k) != 3 or len(xi) != 3:
        raise ValueError("three-dimensional vectors required")
    squared = mp.fsum(v * v for v in k)
    if squared <= 0:
        raise ValueError("nonzero wavevector required")
    contraction = mp.fsum(a * b for a, b in zip(k, xi, strict=True))
    delta_density = -mp.j * rho * contraction
    potential = -4 * mp.pi * delta_density / squared
    acceleration = [-mp.j * component * potential for component in k]
    gradient = [[mp.j * component * ai for component in k] for ai in acceleration]
    return {"potential": potential, "acceleration": acceleration, "gradient": gradient}


def fourier_poisson_acceleration(density, wavevector, displacement):
    """Acceleration view of the complete independent Fourier-Poisson solution."""
    return fourier_poisson_fields(density, wavevector, displacement)["acceleration"]


def _norm(values):
    return mp.sqrt(mp.fsum(abs(v) ** 2 for v in values))


def _difference(left, right):
    return _norm([a - b for a, b in zip(left, right, strict=True)])


def _b1_values(density, psd, convention, order):
    spectrum = decimal_input(psd)
    directional_power = spectrum * (3 if convention == "component" else 1)
    moment = solid_angle_moment(order)
    # Longitudinal Fourier-Poisson map, for a unit displacement along k.
    transfer = fourier_poisson_acceleration(density, [0, 0, 1], [0, 0, 1])[2]
    return [
        abs(transfer) ** 2 * directional_power * moment[i][j]
        for i in range(3)
        for j in range(3)
    ]


def b1_reference(
    density,
    psd,
    *,
    convention="component",
    dps=90,
    orders=(16, 32, 64),
    exact_inputs=True,
):
    """B1 numerical CSD with all components of the Frobenius uncertainty.

    10x the last two refinements, independent identity discrepancy, precision
    drift, an assumed arithmetic floor, and serialization padding are summed.
    The arithmetic floor is a declared estimate, not interval arithmetic.
    No finite physical-input measurement uncertainty is inferred from digits.
    """
    if convention not in ("component", "total"):
        raise ValueError("component/total convention must be explicit")
    if not exact_inputs:
        raise ValueError("unknown input uncertainty is unresolved")
    if dps < 50 or len(orders) != 3 or not (4 <= orders[0] < orders[1] < orders[2]):
        raise ValueError("three increasing resolutions and >=50 digits required")
    with mp.workdps(dps):
        rho, spectrum = decimal_input(density), decimal_input(psd)
        if rho < 0 or spectrum < 0:
            raise ValueError("nonnegative density and PSD required")
        scale = (4 * mp.pi * rho) ** 2 * spectrum / (3 if convention == "total" else 1)
        if rho == 0 or spectrum == 0:
            components = dict.fromkeys(
                (
                    "root",
                    "input",
                    "quadrature",
                    "tail",
                    "roundoff",
                    "serialization",
                    "precision",
                    "identity_crosscheck",
                ),
                "0e+0",
            )
            return {
                "status": "SELF_CHECK_ONLY_NOT_ADMITTED",
                "family": "B1",
                "convention": convention,
                "exact_inputs": True,
                "representation": "g_free",
                "units": "kg^2 m^-4 Hz^-1",
                "observable": "acceleration_csd",
                "shape": [3, 3],
                "values": [["0e+0", "0e+0"] for _ in range(9)],
                "source_scale": "0e+0",
                "zero_certificate": "exact rho=0"
                if rho == 0
                else "exact displacement PSD=0",
                "uncertainty": {
                    "kind": "exact_zero",
                    "total": "0e+0",
                    "components": components,
                },
                "source": SOURCE,
            }
        levels = [_b1_values(density, psd, convention, n) for n in orders]
        high = levels[-1]
        with mp.workdps(max(30, dps - 30)):
            low = _b1_values(density, psd, convention, orders[-1])
        identity = [scale if i == j else mp.mpf(0) for i in range(3) for j in range(3)]
        digits = dps - 10
        serialized = [
            [decimal_string(mp.re(v), digits), decimal_string(mp.im(v), digits)]
            for v in high
        ]
        restored = [mp.mpc(decimal_input(a), decimal_input(b)) for a, b in serialized]
        # Full scale is retained for symmetry zeros and cancellation residuals.
        components = {
            "root": mp.mpf(0),  # no root in B1
            "input": mp.mpf(0),  # exact mathematical fixture inputs declared
            "quadrature": 10
            * max(_difference(levels[2], levels[1]), _difference(levels[1], levels[0])),
            "tail": mp.mpf(0),  # compact angular domains; Fourier analytic solution
            "roundoff": abs(scale) * mp.power(10, -(dps - 8)),
            "serialization": _difference(high, restored)
            + 3 * abs(scale) * mp.power(10, -(digits - 1)),
            "precision": 10 * _difference(high, low),
            "identity_crosscheck": 10 * _difference(high, identity),
        }
        reported = {k: upward_string(v, digits) for k, v in components.items()}
        total = mp.fsum(decimal_input(v) for v in reported.values())
        return {
            "status": "SELF_CHECK_ONLY_NOT_ADMITTED",
            "family": "B1",
            "convention": convention,
            "exact_inputs": True,
            "representation": "g_free",
            "units": "kg^2 m^-4 Hz^-1",
            "observable": "acceleration_csd",
            "shape": [3, 3],
            "values": serialized,
            "source_scale": decimal_string(scale, digits),
            "zero_certificate": None,
            "uncertainty": {
                "kind": "numerical_estimate_not_theorem",
                "total": upward_string(total, digits),
                "components": reported,
            },
            "method": {
                "precision_digits": [max(30, dps - 30), dps],
                "orders": list(orders),
                "input_assumption": "exact mathematical decimal inputs",
                "root_assumption": "not applicable",
                "tail_assumption": "compact sphere; Fourier analytic map",
                "roundoff_assumption": "source_scale*10^(-dps+8), estimate",
                "serialization_digits": digits,
                "refinement_difference_norms": [
                    decimal_string(_difference(levels[1], levels[0]), digits),
                    decimal_string(_difference(levels[2], levels[1]), digits),
                ],
                "precision_difference_norm": decimal_string(
                    _difference(high, low), digits
                ),
                "literal_difference_norm": decimal_string(
                    _difference(high, identity), digits
                ),
            },
            "source": SOURCE,
        }


def rayleigh_root(vp, vs, dps=90):
    """Independent bisection of Harms Eq40 cubic, physical root y=(cR/vS)^2.

    Positive Poisson ratio domain beta=(vS/vP)^2<=1/2. On [0,1] the
    polynomial derivative is positive; f(0)<0 and f(1)=1. The numerical
    bracket is an estimate, not an interval-arithmetic enclosure.
    """
    with mp.workdps(dps + 20):
        cp, cs = decimal_input(vp), decimal_input(vs)
        if cp <= 0 or cs <= 0:
            raise ValueError("positive wave speeds required")
        beta = (cs / cp) ** 2
        if not (cp > 0 and cs > 0 and 0 < beta <= mp.mpf("0.5")):
            raise ValueError(
                "R1 reference domain: positive speeds and nonnegative Poisson ratio"
            )

        def polynomial(y):
            return y**3 - 8 * y**2 + (24 - 16 * beta) * y - 16 * (1 - beta)

        left, right = mp.mpf(0), mp.mpf(1)
        while right - left > mp.power(10, -(dps + 5)):
            middle = (left + right) / 2
            if polynomial(middle) < 0:
                left = middle
            else:
                right = middle
        y = (left + right) / 2
        return {
            "speed": cs * mp.sqrt(y),
            "left": left,
            "right": right,
            "residual": abs(polynomial(y)),
            "beta": beta,
        }


def _r1_transfer(density, vp, vs, frequency, height, speed):
    """Potential amplitude from literature Eq96+98, then target derivative.

    qzP and qzS from Eq38, zeta=sqrt(qzP/qzS) from Eq39. This does not
    construct the stochastic closed tensor. Vertical surface xi_z=1.
    """
    rho, cp, cs = map(decimal_input, (density, vp, vs))
    f, h = map(decimal_input, (frequency, height))
    horizontal_k = 2 * mp.pi * f / speed
    q_p = mp.sqrt(horizontal_k**2 - (2 * mp.pi * f / cp) ** 2)
    q_s = mp.sqrt(horizontal_k**2 - (2 * mp.pi * f / cs) ** 2)
    zeta = mp.sqrt(q_p / q_s)
    potential = (
        -2
        * mp.pi
        * rho
        * (1 - zeta)
        * mp.exp(-h * horizontal_k)
        / (q_p - horizontal_k * zeta)
    )
    # a=-grad Phi: x/y derivatives multiply -i*k; z derivative +k.
    return -horizontal_k * potential, horizontal_k, zeta, q_p


def _r1_values(density, vp, vs, frequency, height, psd, order, dps):
    root = rayleigh_root(vp, vs, dps)
    amplitude, *_ = _r1_transfer(density, vp, vs, frequency, height, root["speed"])
    spectrum = decimal_input(psd)

    def transfer(phi):
        return (
            mp.j * amplitude * mp.cos(phi),
            mp.j * amplitude * mp.sin(phi),
            -amplitude,
        )

    return [
        spectrum
        * integrate(
            lambda p, i=i, j=j: transfer(p)[i] * mp.conj(transfer(p)[j]),
            mp.mpf(0),
            2 * mp.pi,
            order,
        )
        / (2 * mp.pi)
        for i in range(3)
        for j in range(3)
    ]


def _estimated_record(
    family, levels, low, scale, literal, root_uncertainty, dps, orders, metadata, source
):
    """Complete estimated components, with no policy/admission decision."""
    high, digits = levels[-1], dps - 10
    serialized = [
        [decimal_string(mp.re(v), digits), decimal_string(mp.im(v), digits)]
        for v in high
    ]
    restored = [mp.mpc(decimal_input(a), decimal_input(b)) for a, b in serialized]
    components = {
        "root": root_uncertainty,
        "input": mp.mpf(0),  # exact mathematical fixtures, not measured inputs
        "quadrature": 10
        * max(_difference(levels[2], levels[1]), _difference(levels[1], levels[0])),
        "tail": mp.mpf(0),  # compact domains or analytically eliminated Abel tail
        "roundoff": abs(scale) * mp.power(10, -(dps - 8)),
        "serialization": _difference(high, restored)
        + 3 * abs(scale) * mp.power(10, -(digits - 1)),
        "precision": 10 * _difference(high, low),
        "identity_crosscheck": 10 * _difference(high, literal),
    }
    reported = {k: upward_string(v, digits) for k, v in components.items()}
    total = mp.fsum(decimal_input(v) for v in reported.values())
    return {
        "family": family,
        "exact_inputs": True,
        "representation": "g_free",
        "units": "kg^2 m^-4 Hz^-1",
        "observable": "acceleration_csd",
        "status": "SELF_CHECK_ONLY_NOT_ADMITTED",
        "shape": [3, 3],
        "values": serialized,
        "source_scale": decimal_string(scale, digits),
        "uncertainty": {
            "kind": "numerical_estimate_not_theorem",
            "total": upward_string(total, digits),
            "components": reported,
        },
        "method": {
            "precision_digits": [max(30, dps - 30), dps],
            "orders": list(orders),
            "serialization_digits": digits,
            "refinement_difference_norms": [
                decimal_string(_difference(levels[1], levels[0]), digits),
                decimal_string(_difference(levels[2], levels[1]), digits),
            ],
            "precision_difference_norm": decimal_string(_difference(high, low), digits),
            "literal_difference_norm": decimal_string(
                _difference(high, literal), digits
            ),
            "input_assumption": "exact mathematical decimal inputs",
            "roundoff_assumption": "source_scale*10^(-dps+8), estimate",
            **metadata,
        },
        "source": source,
    }


def r1_reference(
    density,
    vp,
    vs,
    frequency,
    height,
    psd,
    *,
    dps=90,
    orders=(16, 32, 64),
    exact_inputs=True,
):
    """Independent literature directional transfer and numerical azimuth CSD."""
    if not exact_inputs:
        raise ValueError("unknown input uncertainty is unresolved")
    if dps < 50 or len(orders) != 3 or not (4 <= orders[0] < orders[1] < orders[2]):
        raise ValueError("three increasing resolutions and >=50 digits required")
    with mp.workdps(dps):
        rho, f, h, spectrum = map(decimal_input, (density, frequency, height, psd))
        if rho < 0 or f <= 0 or h < 0 or spectrum < 0:
            raise ValueError(
                "nonnegative rho/height/PSD and positive frequency required"
            )
        root = rayleigh_root(vp, vs, dps)
        levels = [
            _r1_values(density, vp, vs, frequency, height, psd, n, dps) for n in orders
        ]
        with mp.workdps(max(30, dps - 30)):
            low = _r1_values(
                density, vp, vs, frequency, height, psd, orders[-1], max(30, dps - 30)
            )
        # Literal Eq112 evaluation, separate from the directional tensor integral.
        c = root["speed"]
        q = mp.sqrt(1 - (c / decimal_input(vp)) ** 2)
        s = mp.sqrt(1 - (c / decimal_input(vs)) ** 2)
        zeta = mp.sqrt(q / s)
        gamma = (1 - zeta) / (q - zeta)
        horizontal = (
            (2 * mp.pi * rho * gamma * mp.exp(-2 * mp.pi * f * h / c)) ** 2
            * spectrum
            / 2
        )
        literal = [
            horizontal * (2 if i == 2 else 1) if i == j else mp.mpf(0)
            for i in range(3)
            for j in range(3)
        ]
        scale = (2 * mp.pi * rho) ** 2 * spectrum
        endpoints = []
        for y in (root["left"], root["right"]):
            a, *_ = _r1_transfer(
                density, vp, vs, frequency, height, decimal_input(vs) * mp.sqrt(y)
            )
            endpoints.append(
                [
                    a * a * spectrum * (1 if i == 2 else mp.mpf("0.5"))
                    if i == j
                    else mp.mpf(0)
                    for i in range(3)
                    for j in range(3)
                ]
            )
        root_error = 10 * _difference(endpoints[0], endpoints[1]) + scale * mp.power(
            10, -(dps - 8)
        )
        source = {
            **SOURCE,
            "equations": ["38", "39", "40", "96", "98", "99", "112"],
            "pages": [29, 44, 45, 55],
        }
        with mp.workdps(dps + 20):
            bracket = [decimal_string(root[k], dps + 10) for k in ("left", "right")]
            residual = decimal_string(root["residual"], dps + 10)
        record = _estimated_record(
            "R1",
            levels,
            low,
            scale,
            literal,
            root_error,
            dps,
            orders,
            {
                "root_bracket": bracket,
                "root_residual": residual,
                "rayleigh_speed_m_s": decimal_string(c, dps - 10),
                "root_assumption": (
                    "bisection Eq40 y in (0,1); nonnegative Poisson ratio; estimate"
                ),
                "tail_assumption": "compact azimuth; Eq96 analytical halfspace field",
                "surface_limit": "height=0 is reference h->0+ only",
            },
            source,
        )
        record["literal_horizontal"] = decimal_string(horizontal, dps - 10)
        if rho == 0 or spectrum == 0:
            record["zero_certificate"] = (
                "exact rho=0" if rho == 0 else "exact vertical PSD=0"
            )
            record["uncertainty"]["kind"] = "exact_zero"
        return record


def eq65_factor(ka):
    """Literal spherical j1(x)/x via independent defining Bessel function."""
    x = decimal_input(ka)
    if x < 0:
        raise ValueError("nonnegative ka required")
    return (
        mp.mpf(1) / 3
        if x == 0
        else mp.sqrt(mp.pi / (2 * x)) * mp.besselj(mp.mpf("1.5"), x) / x
    )


def centered_newton_transfer(ka, order):
    """Centered Newton integral per rho, unit displacement, G=1, n=z.

    Cavity wall sigma=-rho xi.n gives -rho int n(xi.n)e^(i ka mu)dOmega.
    Longitudinal bulk delta_rho=-ik rho e^(ikr mu): angular integration and
    radial antiderivative give 4pi rho j0(ka). The omitted endpoint j0(kL)
    tends to zero in the Abel/infinite-radius limit. Thus no finite cutoff or
    numerical tail is asserted. S bulk is exactly zero by divergence freedom.
    Transverse wall uses phi integration int cos²phi dphi=pi. No scattering.
    """
    x = decimal_input(ka)
    if x < 0:
        raise ValueError("nonnegative ka required")
    j_zero = mp.mpf(1) if x == 0 else mp.sin(x) / x
    wall_p = (
        -2
        * mp.pi
        * integrate(lambda mu: mu**2 * mp.cos(x * mu), mp.mpf(-1), mp.mpf(1), order)
    )
    wall_s = -mp.pi * integrate(
        lambda mu: (1 - mu**2) * mp.cos(x * mu), mp.mpf(-1), mp.mpf(1), order
    )
    return 4 * mp.pi * j_zero + wall_p, wall_s


def transverse_polarization_moment(order):
    """Average e_theta e_theta^T + e_phi e_phi^T, both S polarizations.

    e_theta=(cosθ cosφ,cosθ sinφ,-sinθ), e_phi=(-sinφ,cosφ,0).
    Each polarization has directional weight 3/2*Scomponent, so their trace
    sums to 3*Scomponent. No coherent sum of the two polarization amplitudes.
    """
    polar = (mp.cos, mp.cos, lambda t: -mp.sin(t))
    azimuth = (mp.cos, mp.sin, lambda _p: mp.mpf(1))
    phi_vector = (lambda p: -mp.sin(p), mp.cos, lambda _p: mp.mpf(0))
    return [
        [
            (
                integrate(
                    lambda t, i=i, j=j: mp.sin(t) * polar[i](t) * polar[j](t),
                    mp.mpf(0),
                    mp.pi,
                    order,
                )
                * integrate(
                    lambda p, i=i, j=j: azimuth[i](p) * azimuth[j](p),
                    mp.mpf(0),
                    2 * mp.pi,
                    order,
                )
                + integrate(mp.sin, mp.mpf(0), mp.pi, order)
                * integrate(
                    lambda p, i=i, j=j: phi_vector[i](p) * phi_vector[j](p),
                    mp.mpf(0),
                    2 * mp.pi,
                    order,
                )
            )
            / (4 * mp.pi)
            for j in range(3)
        ]
        for i in range(3)
    ]


def _b2_values(density, vp, vs, frequency, radius, p_psd, s_psd, order):
    rho, cp, cs, f, a, p, s = map(
        decimal_input, (density, vp, vs, frequency, radius, p_psd, s_psd)
    )
    kp, ks = 2 * mp.pi * f * a / cp, 2 * mp.pi * f * a / cs
    # String conversion at full working precision preserves the numerical ka
    # value; its rounding is accounted by precision/roundoff crosschecks.
    p_transfer, _ = centered_newton_transfer(decimal_string(kp, mp.mp.dps), order)
    _, s_transfer = centered_newton_transfer(decimal_string(ks, mp.mp.dps), order)
    longitudinal = solid_angle_moment(order)
    transverse = transverse_polarization_moment(order)
    return [
        rho**2
        * (
            3 * p * p_transfer**2 * longitudinal[i][j]
            + mp.mpf("1.5") * s * s_transfer**2 * transverse[i][j]
        )
        for i in range(3)
        for j in range(3)
    ]


def b2_reference(
    density,
    vp,
    vs,
    frequency,
    radius,
    p_psd,
    s_psd,
    *,
    dps=90,
    orders=(32, 64, 128),
    exact_inputs=True,
    p_s_incoherent=True,
):
    """Newton bulk/wall plus independent directional/polarization average."""
    if not exact_inputs:
        raise ValueError("unknown input uncertainty is unresolved")
    if not p_s_incoherent:
        raise ValueError("B2 declares zero P-S cross-CSD; coherent input excluded")
    if dps < 50 or len(orders) != 3 or not (4 <= orders[0] < orders[1] < orders[2]):
        raise ValueError("three increasing resolutions and >=50 digits required")
    with mp.workdps(dps):
        rho, cp, cs, f, a, p, s = map(
            decimal_input, (density, vp, vs, frequency, radius, p_psd, s_psd)
        )
        if min(rho, a, p, s) < 0 or min(cp, cs, f) <= 0:
            raise ValueError(
                "nonnegative rho/radius/PSDs and positive speeds/frequency required"
            )
        args = (density, vp, vs, frequency, radius, p_psd, s_psd)
        levels = [_b2_values(*args, n) for n in orders]
        with mp.workdps(max(30, dps - 30)):
            low = _b2_values(*args, orders[-1])
        fp = eq65_factor(decimal_string(2 * mp.pi * f * a / cp, dps))
        fs = eq65_factor(decimal_string(2 * mp.pi * f * a / cs, dps))
        component = (4 * mp.pi * rho) ** 2 * (4 * fp**2 * p + fs**2 * s)
        literal = [
            component if i == j else mp.mpf(0) for i in range(3) for j in range(3)
        ]
        scale = (4 * mp.pi * rho) ** 2 * (4 * p + s)
        source = {
            **SOURCE,
            "equations": ["45", "60", "61", "64", "65"],
            "pages": [31, 34, 35, 36],
        }
        record = _estimated_record(
            "B2",
            levels,
            low,
            scale,
            literal,
            mp.mpf(0),
            dps,
            orders,
            {
                "root_assumption": "not applicable",
                "tail_assumption": "bulk radial antiderivative; j0(kL)->0 Abel limit",
                "p_s_cross_csd": "zero by declared incoherence",
                "s_polarizations": 2,
                "elastic_scattering": "excluded",
                "target_assumption": "cavity center only",
            },
            source,
        )
        record["literal_component"] = decimal_string(component, dps - 10)
        if rho == 0 or p + s == 0:
            record["zero_certificate"] = (
                "exact rho=0" if rho == 0 else "exact P and S PSD=0"
            )
            record["uncertainty"]["kind"] = "exact_zero"
        return record
