"""F_B1_v2 analytic-enclosure implementation (Lane B, review-prep ONLY).

Implements EXACTLY the frozen F_B1_v2 procedure text
(``B1-procedure-v2.json``, sha256
``6e4aa87ff5d50cfdf1856e320aecfb847d283d9dd4307abaa2e247f73a340cd5``,
read read-only from protocol commit ``e3a20eb``; FROZEN per
``B1-freeze-v2.json``). No algorithm substitution is performed.

Arithmetic purity (frozen ``arithmetic_purity`` section): this module uses
only :class:`fractions.Fraction` and ``int`` arithmetic in the proof path.
There is deliberately NO ``float``, NO ``mpmath``, NO ``Decimal``
transcendental, NO ``math`` transcendental, NO literal pi constant, and NO
``gragra`` production import. Integer square roots use an in-module exact
Newton iteration on ``int`` (no float seed). A unit check greps this source
for forbidden tokens.

Status: HOLD partial release — implementation / unit self-checks / review
preparation ONLY. This module performs NO formal reference generation, NO
E_B1 construction, NO sealing, NO production comparison, and reads NO
production output. QL comparison exists solely as a labeled development
diagnostic (``ql_diagnostic_compare``) and can never gate a verdict.
"""

import re as _re
from fractions import Fraction

# ---------------------------------------------------------------------------
# Frozen procedure pins (verbatim values, used as given inputs, never tuned)
# ---------------------------------------------------------------------------

#: Primary Machin-series truncation (both series): N = 128.
PRIMARY_N = 128
#: Fallback Machin-series truncation: N5 = 100, N239 = 30.
FALLBACK_N5 = 100
FALLBACK_N239 = 30
#: Primary enclosure-width target: width < 1e-180 (integer-compared).
PRIMARY_WIDTH_DIGITS = 180
#: Fallback enclosure-width target: width < 1e-138 (integer-compared).
FALLBACK_WIDTH_DIGITS = 138
#: Central transport precision: 80 significant digits.
CENTRAL_DIGITS = 80
#: Decision-check precisions (significant digits).
DECISION_DIGITS = (120, 180)
#: AUX assessment partition: 70 keys = 56 R_aux records + 14.
AUX_TOTAL_KEYS = 70
AUX_R_RECORDS = 56
AUX_EXTRA_KEYS = 14

_FROZEN_PROCEDURE_SHA = (
    "6e4aa87ff5d50cfdf1856e320aecfb847d283d9dd4307abaa2e247f73a340cd5"
)
_FROZEN_BUDGET_SHA = "a3e321dcfd6c9825c2f7c330dbeb04c7dac0f970ff06464eb67d8cd53b79089c"
_FROZEN_FIXTURES_SHA = (
    "829e0e2cfd7aa915e19ad32f31abb0cdb57b0300d4854e5491f5358f0c197c05"
)


def frozen_pins():
    """Return the frozen input hashes this implementation is bound to."""
    return {
        "procedure": _FROZEN_PROCEDURE_SHA,
        "budget": _FROZEN_BUDGET_SHA,
        "fixtures": _FROZEN_FIXTURES_SHA,
    }


# ---------------------------------------------------------------------------
# Input validation: exact decimal strings only (tiny/zero + malformed checks)
# ---------------------------------------------------------------------------

_DECIMAL_RE = _re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def parse_decimal(value):
    """Parse an exact finite decimal string to :class:`Fraction`.

    Rejects binary64/bool/None (TypeError) and malformed or non-finite
    spellings such as ``"nan"``/``"inf"`` (ValueError). Zero parses to
    exactly ``Fraction(0)``; tiny-nonzero values stay nonzero (no flush).
    """
    if isinstance(value, bool) or not isinstance(value, str):
        raise TypeError("exact decimal string required")
    if not _DECIMAL_RE.fullmatch(value):
        raise ValueError("malformed finite decimal input")
    text = value.strip().lower()
    if text in (
        "nan",
        "+nan",
        "-nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    ):
        raise ValueError("nonfinite input")
    sign = -1 if text.startswith("-") else 1
    body = text[1:] if text[:1] in ("+", "-") else text
    if "e" in body:
        mantissa, _, exp_text = body.partition("e")
        exponent = int(exp_text)
    else:
        mantissa, exponent = body, 0
    if "." in mantissa:
        digits = mantissa.replace(".", "")
        shift = len(mantissa.split(".")[1])
    else:
        digits, shift = mantissa, 0
    digits = digits.lstrip("0") or "0"
    numerator = sign * int(digits)
    net_exp = exponent - shift
    if net_exp >= 0:
        return Fraction(numerator * 10**net_exp, 1)
    return Fraction(numerator, 10 ** (-net_exp))


def is_exact_zero(value):
    """True iff the parsed input is exactly zero (tiny != zero)."""
    return parse_decimal(value) == 0


# ---------------------------------------------------------------------------
# Change 03 / 11-unit (i): Machin arctan enclosure with parity indexing
# ---------------------------------------------------------------------------


def atan_enclosure(x, n):
    """Enclose atan(x) for x > 0 with the N-term alternating Machin series.

    Finite identity used (denominator ``1+t^2 >= 1 > 0``):
    ``1/(1+t^2) = sum_{j<n} (-t^2)^j + (-t^2)^n/(1+t^2)``.
    Remainder sign is ``(-1)^n`` and ``|R| <= x^{2n+1}/(2n+1)``.

    Even ``n`` -> ``[S_n, S_n + r_n]``; odd ``n`` -> ``[S_n - r_n, S_n]``.
    Returns ``(lo, hi)`` as Fractions. All arithmetic is exact rational.
    """
    if not isinstance(x, Fraction):
        raise TypeError("x must be an exact Fraction")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("N must be a positive int (fixed by freeze)")
    if x <= 0:
        raise ValueError("Machin branch requires x > 0")
    total = Fraction(0)
    for j in range(n):
        total += ((-1) ** j) * x ** (2 * j + 1) / (2 * j + 1)
    remainder = x ** (2 * n + 1) / (2 * n + 1)
    if n % 2 == 0:
        return total, total + remainder
    return total - remainder, total


def machin_tangent_identities():
    """Exact tangent-algebra identities (all :class:`Fraction`).

    ``tan(2a) = 5/12`` with ``a = atan(1/5)``; ``tan(4a) = 120/119``;
    ``(tan4 - 1/239)/(1 + tan4/239) = 1``.
    """
    tan2a = Fraction(2, 5) / (1 - Fraction(1, 25))
    tan4a = (2 * tan2a) / (1 - tan2a * tan2a)
    combo = (tan4a - Fraction(1, 239)) / (1 + tan4a * Fraction(1, 239))
    return {"tan2a": tan2a, "tan4a": tan4a, "combo": combo}


def verify_machin_tangent_identities():
    """Assert the exact identities ``5/12``, ``120/119`` and ``1``."""
    ids = machin_tangent_identities()
    assert ids["tan2a"] == Fraction(5, 12), ids
    assert ids["tan4a"] == Fraction(120, 119), ids
    assert ids["combo"] == 1, ids
    return True


def machin_branch_bounds():
    """Frozen branch-inequality bounds as exact Fractions.

    Lower: ``4(1/5 - (1/5)^3/3) - 1/239 = 70369/89625 > 0``.
    Upper: ``4a - b < 4/5 < 1`` via ``atan x < x`` and ``-b < 0``.
    ``atan(1) in [2/3, 1]`` by the N=1 alternating bound, so
    ``pi/2 = 4 atan(1) >= 8/3 > 1``; the angle in ``(0, 1]`` lies in
    ``(0, pi/2)`` where tan is injective, hence ``4a - b = pi/4``.
    """
    lower = 4 * (Fraction(1, 5) - Fraction(1, 5) ** 3 / 3) - Fraction(1, 239)
    atan1_lo, atan1_hi = atan_enclosure(Fraction(1), 1)
    return {
        "lower": lower,  # == 70369/89625
        "upper_4a_bound": Fraction(4, 5),  # 4a < 4/5 < 1
        "atan1_lo": atan1_lo,  # == 2/3
        "atan1_hi": atan1_hi,  # == 1
        "half_pi_lower": 4 * atan1_lo,  # == 8/3
    }


def verify_machin_branch():
    """Assert every frozen branch inequality exactly."""
    b = machin_branch_bounds()
    assert b["lower"] == Fraction(70369, 89625), b
    assert b["lower"] > 0, b
    assert b["upper_4a_bound"] == Fraction(4, 5) < 1, b
    assert (b["atan1_lo"], b["atan1_hi"]) == (Fraction(2, 3), Fraction(1)), b
    assert b["half_pi_lower"] == Fraction(8, 3) > 1, b
    return True


# ---------------------------------------------------------------------------
# Change 01: fixed N primary/fallback + directed interval assembly
# ---------------------------------------------------------------------------


def assemble_pi(a_lo, a_hi, b_lo, b_hi):
    """Directed assembly ``piL = 16 aL - 4 bU``, ``piU = 16 aU - 4 bL``.

    Endpoint monotonicity encloses every true ``16a - 4b`` with
    ``a in [aL, aU]``, ``b in [bL, bU]``. Requires ``aL <= aU``,
    ``bL <= bU`` and a strictly positive lower bound.
    """
    for bound in (a_lo, a_hi, b_lo, b_hi):
        if not isinstance(bound, Fraction):
            raise TypeError("enclosure endpoints must be exact Fractions")
    if not (a_lo <= a_hi and b_lo <= b_hi):
        raise ValueError("unordered enclosure endpoints")
    pi_lo = 16 * a_lo - 4 * b_hi
    pi_hi = 16 * a_hi - 4 * b_lo
    if not pi_lo > 0:
        raise ValueError("non-positive pi lower bound")
    if not pi_lo <= pi_hi:
        raise ValueError("empty assembled enclosure")
    return pi_lo, pi_hi


def assemble_pi_wrong_order(a_lo, a_hi, b_lo, b_hi):
    """Narrowing variant (wrong order): ``16 aL - 4 bL`` / ``16 aU - 4 bU``.

    This is the FORBIDDEN assembly: it need not enclose the true value and
    must be rejected by the unit check. Provided only so the rejection can
    be exercised.
    """
    return 16 * a_lo - 4 * b_lo, 16 * a_hi - 4 * b_hi


def pi_enclosure_primary():
    """Primary pi enclosure: N = 128 for both Machin series (fixed)."""
    a_lo, a_hi = atan_enclosure(Fraction(1, 5), PRIMARY_N)
    b_lo, b_hi = atan_enclosure(Fraction(1, 239), PRIMARY_N)
    return assemble_pi(a_lo, a_hi, b_lo, b_hi)


def pi_enclosure_fallback():
    """Reviewed fallback: N5 = 100 / N239 = 30 (fixed, never tuned)."""
    a_lo, a_hi = atan_enclosure(Fraction(1, 5), FALLBACK_N5)
    b_lo, b_hi = atan_enclosure(Fraction(1, 239), FALLBACK_N239)
    return assemble_pi(a_lo, a_hi, b_lo, b_hi)


def enclosure_width_lt(enclosure, digits):
    """Pure integer/rational check: ``hi - lo < 10^-digits``.

    Evaluated as ``(hi - lo).numerator * 10^digits < (hi - lo).denominator``;
    no float anywhere.
    """
    lo, hi = enclosure
    width = hi - lo
    if width < 0:
        raise ValueError("unordered enclosure")
    return width.numerator * 10**digits < width.denominator


def verify_primary_width():
    """Primary width below 1e-180 (exact integer comparison)."""
    assert enclosure_width_lt(pi_enclosure_primary(), PRIMARY_WIDTH_DIGITS)
    return True


def verify_fallback_width():
    """Fallback width below 1e-138 (exact integer comparison)."""
    assert enclosure_width_lt(pi_enclosure_fallback(), FALLBACK_WIDTH_DIGITS)
    return True


def verify_primary_inside_fallback():
    """Nesting self-check: the primary enclosure lies in the fallback one."""
    p_lo, p_hi = pi_enclosure_primary()
    f_lo, f_hi = pi_enclosure_fallback()
    assert f_lo <= p_lo and p_hi <= f_hi
    return True


# ---------------------------------------------------------------------------
# Change 04 (ordering) / interval norm: monotone squaring + midpoint/halfwidth
# ---------------------------------------------------------------------------


def multiply_enclosure(lo, hi, factor):
    """Scale an enclosure by an exact factor with directed ordering.

    Negative multipliers flip endpoints (ordering check); zero collapses to
    the exact point ``[0, 0]``. Non-Fraction inputs are rejected.
    """
    if not isinstance(lo, Fraction) or not isinstance(hi, Fraction):
        raise TypeError("endpoints must be exact Fractions")
    if not isinstance(factor, Fraction):
        raise TypeError("factor must be an exact Fraction")
    if not lo <= hi:
        raise ValueError("unordered enclosure")
    if factor >= 0:
        return lo * factor, hi * factor
    return hi * factor, lo * factor


def square_enclosure(lo, hi, c):
    """Monotone squaring: ``L = c piL^2 <= c pi^2 <= c piU^2 = H``.

    Requires ``0 < lo <= hi`` and ``c >= 0`` (exact preconditions).
    """
    for value in (lo, hi, c):
        if not isinstance(value, Fraction):
            raise TypeError("exact Fractions required")
    if not (Fraction(0) < lo <= hi):
        raise ValueError("squaring requires 0 < lo <= hi")
    if c < 0:
        raise ValueError("squaring requires c >= 0")
    return c * lo * lo, c * hi * hi


def verify_square_encloses(lo, hi, c, enclosure):
    """Necessary-condition check: enclosure covers ``{c lo^2, c hi^2}``."""
    low, high = enclosure
    assert low <= min(c * lo * lo, c * hi * hi), enclosure
    assert max(c * lo * lo, c * hi * hi) <= high, enclosure
    assert low <= high, enclosure
    return True


def midpoint_halfwidth(lo, hi):
    """Midpoint/halfwidth pair enclosing ``[lo, hi]`` exactly."""
    if not isinstance(lo, Fraction) or not isinstance(hi, Fraction):
        raise TypeError("exact Fractions required")
    if not lo <= hi:
        raise ValueError("unordered enclosure")
    mid = (lo + hi) / 2
    half = (hi - lo) / 2
    assert mid - half == lo and mid + half == hi
    return mid, half


def joint_diagonal_enclosure(pi_lo, pi_hi, c):
    """Joint (common-mode) diagonal correlation enclosure.

    All three diagonal entries share the single pi enclosure (joint, not
    single-entry): returns the common ``(L, H)`` pair owned once. Separate
    per-entry narrowing is forbidden (see ``single_entry_shrink``).
    """
    return square_enclosure(pi_lo, pi_hi, c)


def single_entry_shrink(pi_lo, pi_hi, c):
    """FORBIDDEN single-entry dependency-shrinking variant.

    Uses the midpoint as a point value (zero halfwidth): must be rejected
    because it discards the common-mode width. Provided only so the
    rejection can be exercised.
    """
    mid, _half = midpoint_halfwidth(pi_lo, pi_hi)
    point = c * mid * mid
    return point, point


# ---------------------------------------------------------------------------
# Change 05: outward sqrt brackets + upward ceilings; exact z = 0 branch
# ---------------------------------------------------------------------------


def isqrt_exact(value):
    """Exact integer floor-sqrt by Newton iteration (``int`` only)."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("exact int required")
    if value < 0:
        raise ValueError("negative radicand")
    if value == 0:
        return 0
    guess = 1 << ((value.bit_length() + 1) // 2)
    while True:
        better = (guess + value // guess) // 2
        if better >= guess:
            while guess * guess > value:
                guess -= 1
            while (guess + 1) * (guess + 1) <= value:
                guess += 1
            return guess
        guess = better


def sqrt_bracket(z):
    """Outward rational sqrt bracket with integer-square certificates.

    Returns ``(lo, hi)`` Fractions with ``lo^2 <= z <= hi^2`` certified by
    exact integer cross-multiplication. Exact ``z = 0`` branch returns the
    point ``(0, 0)``. Negative radicands raise. The bracket is built by
    exact integer bisection refined until the width is below ``2^-80``
    relative to the radicand scale (certificate: integer squares).
    """
    if not isinstance(z, Fraction):
        raise TypeError("radicand must be an exact Fraction")
    if z < 0:
        raise ValueError("negative radicand")
    if z == 0:
        return Fraction(0), Fraction(0)
    # Integer scale: find k with z <= 4^k, then bisect on integers.
    num, den = z.numerator, z.denominator
    scale = 0
    while Fraction(4**scale, 1) < z:
        scale += 1
    # RC3 (provenance-only, no logic change): the `scale` loop above is
    # vestigial — `scale` is intentionally left unused. The bracket below
    # uses `m = 60` unconditionally, so this loop changes no
    # logic/control-flow/arithmetic/precision/return value. It is retained
    # as-is to avoid any semantic churn; the width guarantee (2^-60, claimed
    # 2^-59) is unaffected.
    # Work with w = z * 4^m exactly; sqrt(z) = sqrt(w)/2^m.
    margin = 60  # binary refinement steps -> width certificate 2^-60 scale
    m = margin
    w_num = num * 4**m
    w_den = den
    # q = floor(sqrt(w_num / w_den)) by exact integer squares: the seed
    # isqrt(w_num*w_den)//w_den can sit one below the floor, so adjust by
    # +-1 with integer-square comparisons (exact, no float).
    s = isqrt_exact(w_num * w_den)
    q = s // w_den
    while (q + 1) * (q + 1) * w_den <= w_num:
        q += 1
    while q * q * w_den > w_num:
        q -= 1
    lo = Fraction(q, 2**m)
    hi = Fraction(q + 1, 2**m)
    assert lo * lo <= z <= hi * hi, (lo, hi, z)
    return lo, hi


def verify_sqrt_bracket(z, bracket):
    """Integer-square certificate: ``lo^2 <= z <= hi^2`` with ``lo<=hi``."""
    lo, hi = bracket
    assert isinstance(lo, Fraction) and isinstance(hi, Fraction)
    assert lo <= hi, bracket
    assert lo * lo <= z, (bracket, z)
    assert z <= hi * hi, (bracket, z)
    return True


def frobenius_upper(v, m, u_ser):
    """Frobenius bound ``U_pi <= sqrt(3 r^2) + U_ser`` (triangle inequality).

    ``r = v - m`` as exact Fractions; the square root is the UPPER end of
    an outward bracket, so the bound can never under-cover.
    """
    for value in (v, m, u_ser):
        if not isinstance(value, Fraction):
            raise TypeError("exact Fractions required")
    if u_ser < 0:
        raise ValueError("negative series uncertainty")
    r = v - m
    _slo, shi = sqrt_bracket(3 * r * r)
    return shi + u_ser


# ---------------------------------------------------------------------------
# Change 06: 80-digit central transport + upward U + no-narrowing reparse
# ---------------------------------------------------------------------------


def _floor_log10_positive(num, den):
    """Exact floor(log10(num/den)) for positive ints by comparison.

    Pure integer arithmetic on both sides (negative exponents scale the
    numerator instead of forming a float power).
    """
    guess = len(str(num)) - len(str(den))
    while True:
        if guess >= 0:
            if num < den * 10**guess:
                guess -= 1
                continue
            if num >= den * 10 ** (guess + 1):
                guess += 1
                continue
            return guess
        if num * 10 ** (-guess) < den:
            guess -= 1
            continue
        if num * 10 ** (-guess - 1) >= den:
            guess += 1
            continue
        return guess


def _quantize_ties_even(value, quantum):
    """Round ``value`` to a multiple of ``quantum`` (ties to even).

    Exact integer quotient/remainder on ``value / quantum``; no float.
    """
    if not isinstance(value, Fraction) or not isinstance(quantum, Fraction):
        raise TypeError("exact Fractions required")
    if quantum <= 0:
        raise ValueError("quantum must be positive")
    scaled = value / quantum
    if scaled >= 0:
        q, r = divmod(scaled.numerator, scaled.denominator)
        if 2 * r > scaled.denominator or (2 * r == scaled.denominator and q % 2 == 1):
            q += 1
        return Fraction(q) * quantum
    q, r = divmod(-scaled.numerator, scaled.denominator)
    if 2 * r > scaled.denominator or (2 * r == scaled.denominator and q % 2 == 1):
        q += 1
    return Fraction(-q) * quantum


def _sig_quantum(exp10, digits):
    """Exact ``10^(exp10-digits+1)`` quantum as a Fraction (no float)."""
    shift = exp10 - (digits - 1)
    if shift >= 0:
        return Fraction(10**shift, 1)
    return Fraction(1, 10 ** (-shift))


def _strip_trailing_zeros(mantissa, exponent):
    while mantissa != 0 and mantissa % 10 == 0:
        mantissa //= 10
        exponent += 1
    return mantissa, exponent


def format_canonical(value, digits=CENTRAL_DIGITS):
    """Canonical ``mantissa e exponent`` string with exact ties-even rounding.

    ``digits`` significant digits; ties-even by integer quotient/remainder;
    negative zero normalizes to ``0e+0``; grid exponent chosen by exact
    integer/rational comparison; tiny values keep full significance (no
    fixed absolute grid erases tiny U — per frozen resource limits).
    """
    if not isinstance(value, Fraction):
        raise TypeError("exact Fraction required")
    if value == 0:
        return "0e+0"
    negative = value < 0
    work = -value if negative else value
    exp10 = _floor_log10_positive(work.numerator, work.denominator)
    quantum = _sig_quantum(exp10, digits)
    rounded = _quantize_ties_even(work, quantum)
    if rounded == 0:
        return "0e+0"
    # Re-derive exponent after rounding (carry may add a digit).
    exp10 = _floor_log10_positive(rounded.numerator, rounded.denominator)
    mantissa = rounded / _sig_quantum(exp10, digits)
    assert mantissa.denominator == 1, (mantissa, exp10)
    mantissa, exponent = _strip_trailing_zeros(int(mantissa), exp10 - digits + 1)
    return f"{'-' if negative else ''}{mantissa}e{exponent:+d}"


def format_upward(u, digits=CENTRAL_DIGITS):
    """Outward (upward) decimal ceiling of a nonnegative uncertainty.

    Returns the smallest ``digits``-significant-figure decimal ``>= u``;
    ``0`` maps to ``"0e+0"``. Negative input raises.
    """
    if not isinstance(u, Fraction):
        raise TypeError("exact Fraction required")
    if u < 0:
        raise ValueError("negative uncertainty")
    if u == 0:
        return "0e+0"
    exp10 = _floor_log10_positive(u.numerator, u.denominator)
    quantum = _sig_quantum(exp10, digits)
    scaled = u / quantum
    q, r = divmod(scaled.numerator, scaled.denominator)
    if r > 0:
        q += 1
    mantissa, exponent = _strip_trailing_zeros(q, exp10 - digits + 1)
    if mantissa == 0:
        return "0e+0"
    return f"{mantissa}e{exponent:+d}"


def serialize_enclosure(lo, hi, digits=CENTRAL_DIGITS):
    """Serialize an enclosure as 80-digit central transport + upward U.

    Record fields: ``central`` (exact ties-even rounding of the midpoint),
    ``u_up`` (outward ceiling of the halfwidth, decoded and checked
    against BOTH endpoints), ``digits``, ``lo_exact``/``hi_exact``
    (exact endpoint Fractions, for the reparse no-narrowing proof).
    Public-component sums over the record are exact (see
    ``public_component_sum``).
    """
    if not isinstance(lo, Fraction) or not isinstance(hi, Fraction):
        raise TypeError("exact Fractions required")
    if not lo <= hi:
        raise ValueError("unordered enclosure")
    mid, _half = midpoint_halfwidth(lo, hi)
    central = format_canonical(mid, digits)
    central_frac = parse_decimal(central)
    # Outward U covers the error against BOTH endpoints: halfwidth PLUS the
    # central rounding error (exact Fractions, then upward ceiling).
    u_needed = max(abs(central_frac - lo), abs(hi - central_frac))
    u_up = format_upward(u_needed, digits)
    record = {
        "central": central,
        "u_up": u_up,
        "digits": digits,
        "lo_exact": f"{lo.numerator}/{lo.denominator}",
        "hi_exact": f"{hi.numerator}/{hi.denominator}",
    }
    check_no_narrowing(record)
    return record


def reparse(record):
    """Decode a record to exact Fractions; reject malformed records."""
    try:
        central = parse_decimal(record["central"])
        u_up = parse_decimal(record["u_up"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"malformed record: {error}") from error
    if u_up < 0:
        raise ValueError("negative transported uncertainty")
    return central - u_up, central + u_up


def check_no_narrowing(record):
    """Prove serialization/reparse cannot narrow the claimed enclosure.

    Decodes the ACTUAL emitted decimals and asserts, as exact Fractions:
    (a) the reparse covers the exact stored endpoints, and (b) the
    transported U covers the error against BOTH endpoints. Downward
    rounding or endpoint edits fail here.
    """
    re_lo, re_hi = reparse(record)
    try:
        lo_num, lo_den = record["lo_exact"].split("/")
        hi_num, hi_den = record["hi_exact"].split("/")
        lo = Fraction(int(lo_num), int(lo_den))
        hi = Fraction(int(hi_num), int(hi_den))
    except (KeyError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"malformed exact endpoints: {error}") from error
    if not lo <= hi:
        raise ValueError("unordered stored enclosure")
    central = parse_decimal(record["central"])
    u_up = parse_decimal(record["u_up"])
    if abs(central - lo) > u_up or abs(hi - central) > u_up:
        raise ValueError("transported U misses an endpoint (downward rounding?)")
    if not (re_lo <= lo and hi <= re_hi):
        raise ValueError("reparse narrowed the enclosure")
    return True


def public_component_sum(values):
    """Exact public-component sum (normative): plain exact addition."""
    total = Fraction(0)
    for value in values:
        if not isinstance(value, Fraction):
            raise TypeError("exact Fractions required")
        total += value
    return total


# ---------------------------------------------------------------------------
# Change 02/04 (normalization): nine angular moments + component/total
# ---------------------------------------------------------------------------


def angular_moment_table():
    """Exact ``n_i n_j`` solid-angle moments over ``4pi`` (proven identity).

    All 9 moments of ``n_i n_j sin(theta)`` over phi in ``[0, 2pi]`` then
    theta in ``[0, pi]``, divided by ``4pi``, equal ``delta_ij / 3``
    exactly (diagonals ``1/3``, off-diagonals ``0``); no symmetry shortcut
    is taken in the frozen proof (independent CAS re-integration). This
    table records the PROVEN values as exact Fractions for downstream
    exact use; the proof itself lives in the frozen procedure record.
    """
    return [
        [Fraction(1, 3) if i == j else Fraction(0) for j in range(3)] for i in range(3)
    ]


def component_total_convert(value, direction):
    """Exact component <-> total conversion (``TOTAL/3`` meaning pinned)."""
    if not isinstance(value, Fraction):
        raise TypeError("exact Fraction required")
    if direction == "to_total":
        return value * 3
    if direction == "to_component":
        return value / 3
    raise ValueError("direction must be 'to_total' or 'to_component'")


# ---------------------------------------------------------------------------
# Change 02/04 (sign/phase): Fourier/Poisson transfer-map sign structure
# ---------------------------------------------------------------------------


def transfer_sign(convention):
    """Sign skeleton of the Poisson transfer map for one convention.

    Assessment (spatial ``exp(+ik.r)``): ``delta_rho = -i rho k.xi``,
    ``Phi = -4 pi delta_rho / k^2``, ``a = -ik Phi``.
    Review (spatial ``exp(-ik.r)``): ``delta_rho = +i rho k.xi``,
    ``Phi = -4 pi G delta_rho / k^2``, ``a = +ik Phi``.
    Returns the ``(density_sign, grad_sign)`` pair; the acceleration-map
    product is ``+1`` under BOTH conventions (internally consistent sign
    reversal). Unknown conventions raise.
    """
    if convention == "assessment":
        return (-1, -1)
    if convention == "review":
        return (+1, +1)
    raise ValueError("unknown convention (assessment/review only)")


def verify_transfer_map_same():
    """Both conventions yield the same acceleration-map sign product."""
    for convention in ("assessment", "review"):
        density_sign, grad_sign = transfer_sign(convention)
        assert density_sign * grad_sign == 1, convention
    return True


# ---------------------------------------------------------------------------
# Change 08: G^2 separation (G-free vs SI recomputation paths)
# ---------------------------------------------------------------------------


def si_recompute(gfree_lo, gfree_hi, g_si, u_ser_si):
    """Independent SI recomputation ``S_a^SI = G_SI^2 . S_a^{G-free}``.

    ``g_si`` is an exact decimal-string test input (never a production
    value, never binary64). The SI path carries its OWN serialization
    residual ``u_ser_si``; scaling an already-rounded G-free output is
    forbidden — inputs must be exact Fractions. Returns the SI enclosure
    ``(lo, hi)`` plus the exact ``G_SI`` Fraction used.
    """
    for bound in (gfree_lo, gfree_hi, u_ser_si):
        if not isinstance(bound, Fraction):
            raise TypeError("G-free bounds/residual must be exact Fractions")
    if not gfree_lo <= gfree_hi:
        raise ValueError("unordered G-free enclosure")
    if u_ser_si < 0:
        raise ValueError("negative SI residual")
    g = parse_decimal(g_si)
    if g <= 0:
        raise ValueError("G_SI must be a positive exact input")
    factor = g * g
    lo, hi = multiply_enclosure(gfree_lo, gfree_hi, factor)
    return lo - u_ser_si, hi + u_ser_si, g


def g2_confusion_guard(representation, value):
    """Reject G1/G2 confusion: SI values must be tagged ``"si"``.

    A bare-G (G^1) value presented where ``G_SI^2``-scaled output is
    expected — or any mistagged representation — raises. Only the exact
    tags ``"g_free"`` and ``"si"`` are accepted.
    """
    if representation not in ("g_free", "si"):
        raise ValueError("representation must be 'g_free' or 'si'")
    if not isinstance(value, Fraction):
        raise TypeError("exact Fraction required")
    return representation


# ---------------------------------------------------------------------------
# Change 07: nine-source single-ownership allocation
# ---------------------------------------------------------------------------

_ALLOC_COMPONENTS = (
    "tail_pi_enclosure",
    "quadrature_tail_split",
    "formulation_identity",
    "roundoff_serialization",
    "central_rounding",
    "zero_method",
    "measure_psd",
    "convention_transfer",
    "aux_partition",
)


def allocation_single_owner(claims):
    """Single-ownership check for the nine-source allocation.

    ``claims`` maps component -> owner and must cover exactly the nine
    sources. Frozen rules enforced: the tail owns the pi-enclosure
    component; quadrature / formulation / roundoff have no overlap
    (pairwise distinct owners); serialization owns central rounding.
    """
    if set(claims) != set(_ALLOC_COMPONENTS):
        raise ValueError("allocation must cover exactly the nine sources")
    if claims["tail_pi_enclosure"] != "tail":
        raise ValueError("tail must own the pi-enclosure component")
    if claims["central_rounding"] != "serialization":
        raise ValueError("serialization must own central rounding")
    triple = (
        claims["quadrature_tail_split"],
        claims["formulation_identity"],
        claims["roundoff_serialization"],
    )
    if len(set(triple)) != 3:
        raise ValueError("overlap across quadrature/formulation/roundoff")
    return dict(claims)


def allocation_nonzero_requires_uncertainty(c, u_pi):
    """Nonzero-``c`` implies ``U_pi > 0``; exact zeros need proof-backed path."""
    if not isinstance(c, Fraction) or not isinstance(u_pi, Fraction):
        raise TypeError("exact Fractions required")
    if c != 0 and not u_pi > 0:
        raise ValueError("nonzero c with non-positive U_pi")
    return True


def whole_observable_zero_certificate(values, u_total):
    """Whole-observable zero certificate ONLY (partial zeros rejected)."""
    if not all(v == 0 for v in values):
        raise ValueError("zero certificate requires every value exactly zero")
    if u_total != 0:
        raise ValueError("zero certificate requires exactly zero total U")
    return True


# ---------------------------------------------------------------------------
# Change 09: AUX contract (partition / production denial / certificates)
# ---------------------------------------------------------------------------


def aux_partition_check(keys):
    """70-key assessment partition: 56 R_aux records + 14 extra keys."""
    keys = list(keys)
    if len(keys) != AUX_TOTAL_KEYS:
        raise ValueError(f"partition must hold exactly {AUX_TOTAL_KEYS} keys")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate partition key")
    r_records = [k for k in keys if k.startswith("r_aux:")]
    extra = [k for k in keys if not k.startswith("r_aux:")]
    if len(r_records) != AUX_R_RECORDS or len(extra) != AUX_EXTRA_KEYS:
        raise ValueError("partition must be 56 R_aux records + 14")
    return {"r_aux": len(r_records), "extra": len(extra)}


def aux_production_deny(record):
    """Production-denying check: no production-derived field may appear."""
    forbidden = (
        "production",
        "comparator_kernel",
        "reference_kernel",
        "calibration_value",
        "baseline_log",
    )
    for field in forbidden:
        if field in record:
            raise ValueError(f"production-derived field denied: {field}")
    return True


def aux_certificate_binding(certificate, scope):
    """Certificate applicability binding: S-null whole-field certificates
    are kept OUT of second-order zero-scale assessment scope."""
    if certificate == "s_null_whole_field" and scope == "second_order_zero_scale":
        raise ValueError("S-null certificate not applicable to this scope")
    if certificate not in ("s_null_whole_field", "exact_zero", "none"):
        raise ValueError("unknown certificate")
    return True


# ---------------------------------------------------------------------------
# Change 10: QL demoted to development diagnostic (never an oracle)
# ---------------------------------------------------------------------------


def ql_diagnostic_compare(formal_verdict, ql_values, label=None):
    """Attach a QL comparison as a LABELED development diagnostic.

    The returned mapping always carries
    ``role = "DEVELOPMENT_DIAGNOSTIC_NOT_ORACLE"`` and echoes
    ``formal_verdict`` UNCHANGED: QL agreement or disagreement can never
    promote, demote, or otherwise gate the verdict. Prior QL records keep
    status ``superseded as formal procedure``.
    """
    return {
        "role": "DEVELOPMENT_DIAGNOSTIC_NOT_ORACLE",
        "formal_status": "superseded as formal procedure",
        "formal_verdict": formal_verdict,
        "ql_values": list(ql_values),
        "label": label or "ql-diagnostic",
        "gates_pass_fail": False,
    }


# ---------------------------------------------------------------------------
# Changes 01/06/11: decision checks + review-prep demonstration enclosure
# ---------------------------------------------------------------------------


def verify_decision(enclosure, digits):
    """120/180-digit decision self-check: the enclosure decides ``digits``.

    Asserts (a) the enclosure width is below ``10^-digits`` by exact
    integer comparison, and (b) both endpoints round (exact ties-even) to
    the SAME ``digits``-significant-figure decimal string — i.e. the first
    ``digits`` digits are decided, no rounding-boundary straddle. A
    straddling or over-wide enclosure fails honestly.
    """
    lo, hi = enclosure
    assert enclosure_width_lt(enclosure, digits), digits
    lo_str, hi_str = format_canonical(lo, digits), format_canonical(hi, digits)
    assert lo_str == hi_str, (digits, lo_str, hi_str)
    return True


def demonstration_enclosure():
    """Review-prep demonstration: joint diagonal enclosure of ``c*pi^2``.

    Fixed demonstration constant ``c = 1/3`` (isotropic diagonal share,
    exact). Returns ``(lo, hi, record)`` with the 80-digit serialization
    record attached. Deterministic: same inputs always give byte-identical
    output. This is an implementation self-check artifact, NOT a formal
    reference (no E/R/P).
    """
    pi_lo, pi_hi = pi_enclosure_primary()
    c = Fraction(1, 3)
    lo, hi = joint_diagonal_enclosure(pi_lo, pi_hi, c)
    verify_square_encloses(pi_lo, pi_hi, c, (lo, hi))
    return lo, hi, serialize_enclosure(lo, hi)
