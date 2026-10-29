"""Independent first-order B1 references; integer/rational arithmetic only.

This module never imports gragra or other oracle computational helpers. All
accepted fixtures are origin plane waves; unknown geometry fails closed.
"""

from fractions import Fraction
from math import isqrt

COMPONENTS = (
    "input",
    "root",
    "endpoint",
    "quadrature",
    "tail",
    "formulation",
    "precision",
    "roundoff",
    "serialization",
)
SHAPES = {
    "p_acceleration_connection": [3],
    "s_potential_null": [],
    "s_acceleration_null": [3],
    "s_gradient_null": [3, 3],
}


def canonical(value, digits=80, upward=False):
    """Round rational at significant-digit quantum, with exact integer tests."""
    value = Fraction(value)
    if not value:
        return "0e+0"
    negative = value < 0
    if upward and negative:
        raise ValueError("upper uncertainty must be nonnegative")
    x = abs(value)
    exponent = len(str(x.numerator)) - len(str(x.denominator))
    while x < Fraction(10) ** exponent:
        exponent -= 1
    while x >= Fraction(10) ** (exponent + 1):
        exponent += 1
    quantum = Fraction(10) ** (exponent - digits + 1)
    ratio = x / quantum
    q, r = divmod(ratio.numerator, ratio.denominator)
    if upward:
        q += bool(r)
    elif 2 * r > ratio.denominator or (2 * r == ratio.denominator and q % 2):
        q += 1
    text = str(q)
    actual_exp = exponent - digits + len(text)
    text = text.rstrip("0")
    mantissa = text[0] + ("." + text[1:] if len(text) > 1 else "")
    return ("-" if negative else "") + mantissa + "e" + f"{actual_exp:+d}"


def sqrt_upper(value, digits=200):
    """Verified upper rational sqrt, relative grid; zero remains exact."""
    value = Fraction(value)
    if value < 0:
        raise ValueError("negative square")
    if not value:
        return Fraction(0)
    exponent = len(str(value.numerator)) - len(str(value.denominator))
    quantum = Fraction(10) ** (exponent // 2 - digits)
    scaled = value / (quantum * quantum)
    seed = isqrt(scaled.numerator // scaled.denominator)
    result = seed * quantum
    if result * result < value:
        result += quantum
    if result * result < value or (result - quantum) ** 2 >= value:
        raise ArithmeticError("square certificate failed")
    return result


def arctan_interval(inverse, n=128):
    x = Fraction(1, inverse)
    partial = sum(
        ((-1) ** j * x ** (2 * j + 1) / (2 * j + 1) for j in range(n)), Fraction(0)
    )
    remainder = x ** (2 * n + 1) / (2 * n + 1)
    return (
        (partial, partial + remainder) if n % 2 == 0 else (partial - remainder, partial)
    )


def pi_interval():
    a, b = arctan_interval(5), arctan_interval(239)
    lo, hi = 16 * a[0] - 4 * b[1], 16 * a[1] - 4 * b[0]
    if not 0 < lo < hi or hi - lo >= Fraction("1e-180"):
        raise ArithmeticError("pi enclosure invalid")
    return lo, hi


def norm_upper(values):
    return sqrt_upper(sum((v * v for v in values), Fraction(0)))


def checked_vectors(fixture):
    n = list(map(Fraction, fixture["direction_raw"]))
    p = list(map(Fraction, fixture["polarization_raw"]))
    if len(n) != 3 or len(p) != 3:
        raise ValueError("three coordinates required")
    squares = [sum(v * v for v in vector) for vector in (n, p)]
    if min(squares) <= 0:
        raise ValueError("zero direction or polarization")
    if fixture["kind"] == "P_transfer":
        roots = [isqrt(s.numerator) for s in squares]
        if any(r * r != s for r, s in zip(roots, squares, strict=False)) or n != p:
            raise ValueError("only exact rational longitudinal fixtures supported")
        return [v / roots[0] for v in n], p
    if fixture["kind"] != "S_null" or sum(x * y for x, y in zip(n, p, strict=False)):
        raise ValueError("transverse certificate invalid")
    return n, p


def make_record(key, fixture, source_pins, procedure_hash, source_equations):
    """Construct one first-order envelope, with full observable uncertainty."""
    observable = key["observable"]
    if observable not in SHAPES or observable not in fixture["observables"]:
        raise ValueError("unknown observable")
    if key["representation"] not in ("Gfree", "SI"):
        raise ValueError("unknown representation")
    if (
        key["frequency_hz"] not in fixture["frequency_hz"]
        or key["target_id"] != fixture["target_id"]
        or key["fixture_id"] != fixture["fixture_id"]
        or key["family"] != "B1"
        or key["set"] != "auxiliary"
    ):
        raise ValueError("key fixture mismatch")
    if any(Fraction(x) for x in fixture["target_m"]):
        raise ValueError("non-origin spatial phase not supported")
    if (
        fixture["spatial_phase"] != "exp(i*k dot target)"
        or fixture["frame"] != "Cartesian"
        or fixture["input_assumptions"] != "exact_input"
    ):
        raise ValueError("unknown phase/frame/input convention")
    if (
        Fraction(key["frequency_hz"]) <= 0
        or min(Fraction(fixture[name]) for name in ("p_speed_m_s", "s_speed_m_s")) <= 0
    ):
        raise ValueError("positive frequency and speeds required")
    direction, polarization = checked_vectors(fixture)
    shape = SHAPES[observable]
    count = 1 if not shape else (3 if len(shape) == 1 else 9)
    certificate = None
    zero_method = "Exact rational inputs and operations; no numerical truncation."
    parts = {name: {"value": "0e+0", "method": zero_method} for name in COMPONENTS}
    if fixture["kind"] == "S_null":
        if not observable.startswith("s_"):
            raise ValueError("S observable mismatch")
        real = imag = ["0e+0"] * count
        kind = "exact_zero"
        certificate = {
            "kind": "whole_field_transverse_null",
            "domain": "infinite homogeneous boundary-free medium",
            "raw_dot_product": "0e+0",
            "direction_raw": fixture["direction_raw"],
            "polarization_raw": fixture["polarization_raw"],
            "nonzero_direction_and_polarization": True,
            "normalization_preserves_transversality": True,
            "whole_field": True,
            "source_equations": source_equations,
            "reason": (
                "k dot xi = 0 implies delta_rho=0, Phi=0, acceleration=0, "
                "gradient=0 everywhere; no boundary source."
            ),
        }
        for part in parts.values():
            part["method"] = (
                "Exact whole-field transverse certificate; no evaluated cancellation."
            )
    else:
        if observable != "p_acceleration_connection":
            raise ValueError("P observable mismatch")
        lo, hi = pi_interval()
        middle, radius = (lo + hi) / 2, (hi - lo) / 2
        g = Fraction(fixture["g_si"]) if key["representation"] == "SI" else Fraction(1)
        amplitude = fixture["displacement_amplitude_m"]
        # Eq55/60: a = 4*pi*G*rho*xi at phase=1, G once for first order.
        factor = 4 * g * Fraction(fixture["density_kg_m3"])
        coefficients = [
            factor * Fraction(amplitude[part]) * axis
            for part in ("real", "imag")
            for axis in direction
        ]
        centers = [middle * c for c in coefficients]
        encoded = [canonical(v) for v in centers]
        real, imag = encoded[:3], encoded[3:]
        residuals = [Fraction(s) - v for s, v in zip(encoded, centers, strict=False)]
        parts["tail"] = {
            "value": canonical(radius * norm_upper(coefficients), upward=True),
            "method": (
                "Machin N=128 alternating-series rational pi enclosure; joint"
                " complex Euclidean norm, verified integer-square upward "
                "bound."
            ),
        }
        parts["serialization"] = {
            "value": canonical(norm_upper(residuals), upward=True),
            "method": (
                "Exact emitted 80-digit ties-even center residual in full "
                "complex Euclidean norm; verified integer-square upward "
                "bound."
            ),
        }
        kind = "rigorous_bound"
        if sum(Fraction(part["value"]) for part in parts.values()) <= 0:
            raise ValueError("nonzero P uncertainty required")
    si_units = {
        "s_potential_null": "m^2 s^-2",
        "s_gradient_null": "s^-2",
        "s_acceleration_null": "m s^-2",
        "p_acceleration_connection": "m s^-2",
    }
    gf_units = {
        "s_potential_null": "kg m^-1",
        "s_gradient_null": "kg m^-3",
        "s_acceleration_null": "kg m^-2",
        "p_acceleration_connection": "kg m^-2",
    }
    return {
        "schema_version": 1,
        "record_kind": "first_order_connection_reference",
        "key": key,
        "status": "resolved",
        "quantity": {"shape": shape, "real": real, "imag": imag},
        "uncertainty": {"kind": kind, "components": parts, "certificate": certificate},
        "input_assumptions": "exact_input",
        "source_pins": source_pins,
        "procedure_hash": procedure_hash,
        "inputs": fixture,
        "units": (si_units if key["representation"] == "SI" else gf_units)[observable],
        "input_convention": (
            "first_order_complex_displacement_amplitude; spatial exp(+ik.r)"
        ),
        "admission_status": "NOT_FORMALLY_ADMITTED",
    }
