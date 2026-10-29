"""V4 contract controls; mock anchors exercise API binding, never real trust."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import itertools
import json
import random
import sys
import warnings
from fractions import Fraction
from pathlib import Path

import pytest


def _exact_mpf_decimal_for_test(value):
    """Return an exact decimal for an MPF tuple without using mpmath formatting."""
    sign, mantissa, exponent, _bitcount = value._mpf_
    if mantissa == 0:
        return "0"
    prefix = "-" if sign else ""
    if exponent >= 0:
        return prefix + _test_integer_decimal(mantissa << exponent)
    digits = _test_integer_decimal(mantissa * 5 ** (-exponent))
    decimal_exponent = len(digits) - 1 + exponent
    significand = digits[0]
    if len(digits) > 1:
        significand += "." + digits[1:]
    return f"{prefix}{significand}e{decimal_exponent}"


def _test_integer_decimal(value):
    if value == 0:
        return "0"
    chunks = []
    while value:
        value, chunk = divmod(value, 1_000_000_000)
        chunks.append(chunk)
    return str(chunks[-1]) + "".join(f"{chunk:09d}" for chunk in reversed(chunks[:-1]))


def _exact_binary_mpf(mp, sign, mantissa, exponent):
    if mantissa == 0:
        return mp.mpf(0)
    return mp.make_mpf((sign, mantissa, exponent, mantissa.bit_length()))


def _trend_for_pair(encoded):
    return {
        "classification": "unresolved_nondecreasing",
        "nondecreasing": True,
        "coarse_to_middle_si": encoded,
        "middle_to_fine_si": encoded,
        "independent_floor_si": None,
    }


def _fraction_from_mpf(value):
    """Recover an MPF's exact dyadic rational without decimal formatting."""
    sign, mantissa, exponent, _bitcount = value._mpf_
    numerator = -mantissa if sign else mantissa
    if exponent >= 0:
        return Fraction(numerator << exponent, 1)
    return Fraction(numerator, 1 << -exponent)


def _fraction_floor_log2(value):
    """Return floor(log2(value)) for a positive Fraction using integers."""
    numerator = value.numerator
    denominator = value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << -exponent < denominator:
        exponent -= 1
    return exponent


def _round_fraction_to_longdouble_oracle(value, finfo):
    """Independent exact-rational RNE oracle for the host longdouble format."""
    precision = int(finfo.nmant) + 1
    minimum_normal_exponent = int(finfo.minexp)
    maximum_exponent = int(finfo.maxexp)
    subnormal_quantum_exponent = minimum_normal_exponent - precision + 1
    negative = value < 0
    magnitude = abs(value)
    if magnitude == 0:
        return Fraction(0), negative

    leading_exponent = _fraction_floor_log2(magnitude)
    if leading_exponent >= maximum_exponent:
        raise OverflowError("oracle value overflows longdouble")
    quantum_exponent = (
        subnormal_quantum_exponent
        if leading_exponent < minimum_normal_exponent
        else leading_exponent - precision + 1
    )
    numerator = magnitude.numerator
    denominator = magnitude.denominator
    if quantum_exponent >= 0:
        denominator <<= quantum_exponent
    else:
        numerator <<= -quantum_exponent
    quotient, remainder = divmod(numerator, denominator)
    doubled_remainder = 2 * remainder
    if doubled_remainder > denominator or (
        doubled_remainder == denominator and quotient & 1
    ):
        quotient += 1
    if quotient == 0:
        return Fraction(0), negative

    rounded = (
        Fraction(quotient << quantum_exponent, 1)
        if quantum_exponent >= 0
        else Fraction(quotient, 1 << -quantum_exponent)
    )
    if _fraction_floor_log2(rounded) >= maximum_exponent:
        raise OverflowError("oracle rounded result overflows longdouble")
    return (-rounded if negative else rounded), negative


def _fraction_from_longdouble(value):
    numerator, denominator = value.as_integer_ratio()
    return Fraction(numerator, denominator)


def _exact_mpf_from_dyadic(mp, sign, mantissa, exponent):
    return mp.make_mpf((sign, mantissa, exponent, mantissa.bit_length()))


def _subnormal_rounding_probe(mp, sign, case, p, q_exponent):
    """Construct exact test MPFs around zero and subnormal rounding boundaries."""
    half_quantum_exponent = q_exponent - 1
    quarter_quantum_exponent = q_exponent - 2
    if case == "least_subnormal":
        mantissa, exponent = 1, q_exponent
    elif case == "zero_midpoint":
        mantissa, exponent = 1, half_quantum_exponent
    elif case == "zero_below":
        mantissa, exponent = 1, quarter_quantum_exponent
    elif case == "zero_above":
        mantissa, exponent = 3, quarter_quantum_exponent
    elif case == "normal_boundary_midpoint":
        mantissa, exponent = (1 << p) - 1, half_quantum_exponent
    elif case == "normal_boundary_midpoint_below":
        mantissa, exponent = (1 << (p + 1)) - 3, quarter_quantum_exponent
    elif case == "normal_boundary_midpoint_above":
        mantissa, exponent = (1 << (p + 1)) - 1, quarter_quantum_exponent
    elif case == "subnormal_midpoint":
        mantissa, exponent = (1 << (p - 1)) + 1, half_quantum_exponent
    elif case == "subnormal_midpoint_below":
        mantissa, exponent = (1 << p) + 1, quarter_quantum_exponent
    elif case == "subnormal_midpoint_above":
        mantissa, exponent = (1 << p) + 3, quarter_quantum_exponent
    else:
        raise AssertionError(f"unknown rounding probe {case!r}")
    return _exact_mpf_from_dyadic(mp, sign, mantissa, exponent)


def _longdouble_from_exact_fraction(value, *, negative_zero=False):
    """Construct an exact representable dyadic without decimal parsing."""
    import numpy as np

    if value == 0:
        return np.longdouble("-0" if negative_zero else "0")
    negative = value < 0
    magnitude = abs(value)
    denominator_exponent = magnitude.denominator.bit_length() - 1
    assert magnitude.denominator == 1 << denominator_exponent
    mantissa = magnitude.numerator
    scalar = np.longdouble(0)
    for bit in bin(mantissa)[2:]:
        scalar = np.ldexp(scalar, 1)
        if bit == "1":
            scalar += np.longdouble(1)
    result = np.ldexp(scalar, -denominator_exponent)
    return -result if negative else result


@pytest.fixture(scope="module")
def numeric_modules():
    modules = []
    for name, filename in (
        ("gen1_recorder_v4_numeric", "record_frozen_refinement_traces_v4.py"),
        ("gen1_auditor_v4_numeric", "audit_frozen_refinement_traces_v4.py"),
    ):
        path = Path(__file__).with_name(filename)
        assert path.exists(), "V4 numeric implementation is required"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


@pytest.mark.parametrize("digits", [15, 30, 50, 80, 100])
@pytest.mark.parametrize("ambient", [15, 30, 50, 80, 100])
def test_v4_mpf_precision_round_trip_context_independence(
    numeric_modules, digits, ambient
):
    import mpmath as mp

    recorder, auditor = numeric_modules
    with mp.workdps(digits):
        delta = mp.power(10, -(digits - 8))
        originals = [mp.mpf(1) + 2 * delta, mp.mpf(1) + delta, mp.mpf(1)]
        expected_tuple = [value._mpf_ for value in originals]
        expected_pair = (
            abs(originals[1] - originals[0]),
            abs(originals[2] - originals[1]),
        )
        encoded = recorder._capture_number(originals, digits)
    serialized = json.loads(json.dumps(encoded, allow_nan=False))
    assert serialized == encoded
    assert all(item["display_digits"] == max(50, digits + 4) for item in encoded)
    with mp.workdps(ambient):
        before = mp.mp.dps
        restored = recorder._decode_callback_value(serialized)
        audited = auditor.decode(serialized)
        assert mp.mp.dps == before
        assert [value._mpf_ for value in restored] == expected_tuple
        assert [value._mpf_ for value in audited] == expected_tuple
        assert recorder._capture_number(restored, digits) == encoded
    with mp.workdps(digits + 20):
        pair = recorder._callback_pair(
            [(v,) for v in restored], "potential", longdouble=False
        )
        audit_pair = auditor.observed_pairs([(v,) for v in audited])
        assert pair == audit_pair == expected_pair
        assert all(mp.isfinite(value) and value > 0 for value in pair)


@pytest.mark.parametrize("ambient", [15, 30, 50, 60, 80, 100])
def test_v4_longdouble_comparison_is_mpmath_context_independent(
    numeric_modules, ambient
):
    import mpmath as mp

    recorder, _auditor = numeric_modules
    saved = "4.14836323325840694038573959132848316478019426620422542606472146571e-28"
    with mp.workdps(100):
        expected = mp.mpf(saved)
    changed = "4.148363233258407389e-28"

    with mp.workdps(ambient):
        recorder._require_number_equal(
            saved,
            expected,
            longdouble=True,
            precision=100,
            message="longdouble comparison probe",
        )
        with pytest.raises(ValueError, match="longdouble comparison probe"):
            recorder._require_number_equal(
                changed,
                expected,
                longdouble=True,
                precision=100,
                message="longdouble comparison probe",
            )


@pytest.mark.parametrize("ambient", [15, 30, 50, 60, 80, 100])
@pytest.mark.parametrize("representation", ["decimal", "mpf-wrapper"])
@pytest.mark.parametrize("perturbed", [False, True])
def test_v4_longdouble_accepts_mpf_actual_only_when_exactly_equal(
    numeric_modules, ambient, representation, perturbed
):
    import mpmath as mp
    import numpy as np

    recorder, _auditor = numeric_modules
    source = "4.1483632332584069404e-28"
    with mp.workdps(100):
        expected = np.longdouble(source)
        changed = np.nextafter(expected, np.longdouble(np.inf), dtype=np.longdouble)
        changed_text = np.format_float_scientific(
            changed,
            unique=True,
            precision=int(np.finfo(np.longdouble).precision) + 4,
            trim="k",
        )
        assert np.longdouble(changed_text) == changed

        if representation == "mpf-wrapper":
            original_mpf = mp.mpf(changed_text if perturbed else source)
            encoded = recorder._capture_number(original_mpf, 100)
            serialized = json.loads(json.dumps(encoded, allow_nan=False))
            restored = recorder._decode_callback_value(serialized)
            assert restored._mpf_ == original_mpf._mpf_
            actual = serialized
        else:
            actual = changed_text if perturbed else source

    with mp.workdps(ambient):
        converted = recorder._callback_number(actual, longdouble=True, precision=100)
        assert converted == (changed if perturbed else expected)
        if perturbed:
            assert converted != expected
            with pytest.raises(ValueError, match="longdouble comparison probe"):
                recorder._require_number_equal(
                    actual,
                    expected,
                    longdouble=True,
                    precision=100,
                    message="longdouble comparison probe",
                )
        else:
            recorder._require_number_equal(
                actual,
                expected,
                longdouble=True,
                precision=100,
                message="longdouble comparison probe",
            )
        assert mp.mp.dps == ambient


@pytest.mark.parametrize("ambient", [15, 30, 50, 60, 80, 100])
@pytest.mark.parametrize("precision", [None, 60, 100])
@pytest.mark.parametrize("sign", [0, 1])
def test_v4_mpf_midpoint_ties_to_even_in_recorder_paths(
    numeric_modules, ambient, precision, sign
):
    import mpmath as mp
    import numpy as np

    recorder, _auditor = numeric_modules
    finfo = np.finfo(np.longdouble)
    p = int(finfo.nmant) + 1
    midpoint = _exact_binary_mpf(mp, sign, (1 << p) + 7, -p)
    below = _exact_binary_mpf(mp, sign, ((1 << p) + 7) * 1024 - 1, -p - 10)
    above = _exact_binary_mpf(mp, sign, ((1 << p) + 7) * 1024 + 1, -p - 10)
    lower_magnitude = np.longdouble(1) + np.longdouble(3) * np.spacing(np.longdouble(1))
    upper_magnitude = np.nextafter(
        lower_magnitude, np.longdouble(np.inf), dtype=np.longdouble
    )
    lower = -lower_magnitude if sign else lower_magnitude
    upper = -upper_magnitude if sign else upper_magnitude
    assert np.longdouble(_exact_mpf_decimal_for_test(midpoint)) == upper

    with mp.workdps(ambient):
        for value, expected in ((below, lower), (midpoint, upper), (above, upper)):
            encoded = recorder._capture_number(value, 100)
            restored = recorder._decode_callback_value(json.loads(json.dumps(encoded)))
            assert restored._mpf_ == value._mpf_
            actual = recorder._callback_number(
                encoded, longdouble=True, precision=precision
            )
            assert actual == expected
            recorder._require_number_equal(
                encoded,
                expected,
                longdouble=True,
                precision=precision,
                message="midpoint conversion mismatch",
            )
            wrong = lower if expected == upper else upper
            with pytest.raises(ValueError, match="midpoint conversion mismatch"):
                recorder._require_number_equal(
                    encoded,
                    wrong,
                    longdouble=True,
                    precision=precision,
                    message="midpoint conversion mismatch",
                )
            recorder._require_trend_pair(
                _trend_for_pair(encoded),
                (expected, expected),
                longdouble=True,
                index=0,
                observable="potential",
                key="probe",
                precision=precision,
            )
            with pytest.raises(ValueError, match="differs from captured callbacks"):
                recorder._require_trend_pair(
                    _trend_for_pair(encoded),
                    (wrong, wrong),
                    longdouble=True,
                    index=0,
                    observable="potential",
                    key="probe",
                    precision=precision,
                )
        assert mp.mp.dps == ambient


@pytest.mark.parametrize("ambient", [15, 30, 50, 60, 80, 100])
@pytest.mark.parametrize("sign", [0, 1])
def test_v4_auditor_mpf_wrapper_ties_and_one_ulp_mismatch(
    numeric_modules, ambient, sign
):
    import mpmath as mp
    import numpy as np

    recorder, auditor = numeric_modules
    p = int(np.finfo(np.longdouble).nmant) + 1
    midpoint = _exact_binary_mpf(mp, sign, (1 << p) + 7, -p)
    encoded = recorder._capture_number(midpoint, 100)
    lower_magnitude = np.longdouble(1) + np.longdouble(3) * np.spacing(np.longdouble(1))
    upper_magnitude = np.nextafter(
        lower_magnitude, np.longdouble(np.inf), dtype=np.longdouble
    )
    lower = -lower_magnitude if sign else lower_magnitude
    upper = -upper_magnitude if sign else upper_magnitude
    assert np.longdouble(_exact_mpf_decimal_for_test(midpoint)) == upper

    with mp.workdps(ambient):
        trend = _trend_for_pair(encoded)
        assert auditor.audit_trend(trend, (upper, upper), None, ld=True) == (
            "unresolved_nondecreasing"
        )
        with pytest.raises(ValueError, match="trend/named pair mismatch"):
            auditor.audit_trend(trend, (lower, lower), None, ld=True)
        assert auditor.number(encoded, ld=True) == upper
        assert mp.mp.dps == ambient


@pytest.mark.parametrize("ambient", [15, 30, 50, 60, 80, 100])
def test_v4_known_a3_mpf_wrapper_uses_both_trend_comparison_paths(
    numeric_modules, ambient
):
    import mpmath as mp
    import numpy as np

    recorder, auditor = numeric_modules
    source = "4.1483632332584069404e-28"
    with mp.workdps(100):
        exact = mp.mpf(source)
        encoded = recorder._capture_number(exact, 100)
        expected = np.longdouble(source)
        changed = np.nextafter(expected, np.longdouble(np.inf), dtype=np.longdouble)

    with mp.workdps(ambient):
        recorder._require_trend_pair(
            _trend_for_pair(encoded),
            (expected, expected),
            longdouble=True,
            index=0,
            observable="potential",
            key="a3-positive-control",
            precision=None,
        )
        with pytest.raises(ValueError, match="differs from captured callbacks"):
            recorder._require_trend_pair(
                _trend_for_pair(encoded),
                (changed, changed),
                longdouble=True,
                index=0,
                observable="potential",
                key="a3-negative-control",
                precision=None,
            )
        assert (
            auditor.audit_trend(
                _trend_for_pair(encoded), (expected, expected), None, ld=True
            )
            == "unresolved_nondecreasing"
        )
        with pytest.raises(ValueError, match="trend/named pair mismatch"):
            auditor.audit_trend(
                _trend_for_pair(encoded), (changed, changed), None, ld=True
            )


@pytest.mark.parametrize("ambient", [15, 60, 100])
@pytest.mark.parametrize("precision", [None, 60, 100])
@pytest.mark.parametrize("sign", [0, 1])
def test_v4_subnormal_mpf_wrapper_trend_rounding_is_exact_and_signed(
    numeric_modules, ambient, precision, sign
):
    """Exercise the saved MPF wrapper through recorder and auditor trend paths."""
    import mpmath as mp
    import numpy as np

    recorder, auditor = numeric_modules
    finfo = np.finfo(np.longdouble)
    p = int(finfo.nmant) + 1
    q_exponent = int(finfo.minexp) - p + 1
    # (2**(p-2) + 3/4) * q: the nearest value is one subnormal ULP
    # above the lower neighbor, with no midpoint ambiguity.
    exact = _exact_mpf_from_dyadic(mp, sign, (1 << p) + 3, q_exponent - 2)
    encoded = recorder._capture_number(exact, 100)
    exact_fraction = _fraction_from_mpf(exact)
    expected_fraction, negative_zero = _round_fraction_to_longdouble_oracle(
        exact_fraction, finfo
    )
    quantum = Fraction(1, 1 << -q_exponent)
    assert expected_fraction == ((1 << (p - 2)) + 1) * quantum * (-1 if sign else 1)
    expected = _longdouble_from_exact_fraction(
        expected_fraction, negative_zero=negative_zero
    )
    assert _fraction_from_longdouble(expected) == expected_fraction
    one_ulp_wrong = np.nextafter(expected, np.longdouble(0), dtype=np.longdouble)
    assert _fraction_from_longdouble(one_ulp_wrong) != expected_fraction

    with mp.workdps(ambient):
        restored = recorder._decode_callback_value(
            json.loads(json.dumps(encoded, allow_nan=False))
        )
        assert restored._mpf_ == exact._mpf_
        recorder._require_trend_pair(
            _trend_for_pair(encoded),
            (expected, expected),
            longdouble=True,
            index=0,
            observable="potential",
            key="subnormal-exact-rounding",
            precision=precision,
        )
        with pytest.raises(ValueError, match="differs from captured callbacks"):
            recorder._require_trend_pair(
                _trend_for_pair(encoded),
                (one_ulp_wrong, one_ulp_wrong),
                longdouble=True,
                index=0,
                observable="potential",
                key="subnormal-one-ulp-mismatch",
                precision=precision,
            )
        assert auditor.number(encoded, ld=True) == expected
        assert (
            auditor.audit_trend(
                _trend_for_pair(encoded), (expected, expected), None, ld=True
            )
            == "unresolved_nondecreasing"
        )
        with pytest.raises(ValueError, match="trend/named pair mismatch"):
            auditor.audit_trend(
                _trend_for_pair(encoded),
                (one_ulp_wrong, one_ulp_wrong),
                None,
                ld=True,
            )
        assert mp.mp.dps == ambient


@pytest.mark.parametrize("ambient", [15, 60, 100])
@pytest.mark.parametrize("sign", [0, 1])
@pytest.mark.parametrize(
    "case",
    [
        "least_subnormal",
        "zero_midpoint",
        "zero_below",
        "zero_above",
        "normal_boundary_midpoint",
        "normal_boundary_midpoint_below",
        "normal_boundary_midpoint_above",
        "subnormal_midpoint",
        "subnormal_midpoint_below",
        "subnormal_midpoint_above",
    ],
)
def test_v4_auditor_subnormal_rounding_boundaries_match_fraction_oracle(
    numeric_modules, ambient, sign, case
):
    """Check signed boundaries against exact Fraction RNE, not decimal parsing."""
    import mpmath as mp
    import numpy as np

    recorder, auditor = numeric_modules
    finfo = np.finfo(np.longdouble)
    p = int(finfo.nmant) + 1
    q_exponent = int(finfo.minexp) - p + 1
    value = _subnormal_rounding_probe(mp, sign, case, p, q_exponent)
    encoded = recorder._capture_number(value, 100)
    expected, negative_zero = _round_fraction_to_longdouble_oracle(
        _fraction_from_mpf(value), finfo
    )
    expected_longdouble = _longdouble_from_exact_fraction(
        expected, negative_zero=negative_zero
    )
    assert _fraction_from_longdouble(expected_longdouble) == expected

    with mp.workdps(ambient):
        actual = auditor.number(encoded, ld=True)
        assert _fraction_from_longdouble(actual) == expected
        if expected == 0:
            assert bool(np.signbit(actual)) is bool(negative_zero)
        else:
            assert actual == expected_longdouble
        recorder_actual = recorder._callback_number(
            encoded, longdouble=True, precision=ambient
        )
        assert _fraction_from_longdouble(recorder_actual) == expected
        assert mp.mp.dps == ambient


@pytest.mark.parametrize("sign", [0, 1])
def test_v4_mpf_binary_rounding_sign_zero_subnormal_and_overflow(numeric_modules, sign):
    import mpmath as mp
    import numpy as np

    recorder, auditor = numeric_modules
    finfo = np.finfo(np.longdouble)
    p = int(finfo.nmant) + 1
    subnormal_exponent = int(finfo.minexp) - p + 1
    least_subnormal = _exact_binary_mpf(mp, sign, 1, subnormal_exponent)
    half_subnormal = _exact_binary_mpf(mp, sign, 1, subnormal_exponent - 1)
    twice_half = _exact_binary_mpf(mp, sign, 3, subnormal_exponent - 1)
    least_positive_subnormal = np.nextafter(
        np.longdouble(0), np.longdouble(np.inf), dtype=np.longdouble
    )
    expected_subnormal = -least_positive_subnormal if sign else least_positive_subnormal

    assert recorder._callback_number(least_subnormal, longdouble=True) == (
        expected_subnormal
    )
    assert auditor.number(least_subnormal, ld=True) == (expected_subnormal)
    assert recorder._callback_number(half_subnormal, longdouble=True) == 0
    assert auditor.number(half_subnormal, ld=True) == 0
    assert recorder._callback_number(twice_half, longdouble=True) == (
        (2 * least_positive_subnormal) * (-1 if sign else 1)
    )

    largest_subnormal = _exact_binary_mpf(
        mp, sign, (1 << (p - 1)) - 1, subnormal_exponent
    )
    normal_boundary_midpoint = _exact_binary_mpf(
        mp, sign, (1 << p) - 1, subnormal_exponent - 1
    )
    signed_min_normal = -finfo.tiny if sign else finfo.tiny
    largest_positive_subnormal = np.nextafter(
        finfo.tiny, np.longdouble(0), dtype=np.longdouble
    )
    signed_largest_subnormal = (
        -largest_positive_subnormal if sign else largest_positive_subnormal
    )
    assert recorder._callback_number(largest_subnormal, longdouble=True) == (
        signed_largest_subnormal
    )
    assert recorder._callback_number(normal_boundary_midpoint, longdouble=True) == (
        signed_min_normal
    )
    assert auditor.number(normal_boundary_midpoint, ld=True) == signed_min_normal

    zero = mp.mpf(0)
    assert recorder._callback_number(zero, longdouble=True) == 0
    assert auditor.number(zero, ld=True) == 0

    maximum_finite = _exact_binary_mpf(mp, sign, (1 << p) - 1, int(finfo.maxexp) - p)
    signed_maximum = -finfo.max if sign else finfo.max
    assert recorder._callback_number(maximum_finite, longdouble=True) == signed_maximum
    assert auditor.number(maximum_finite, ld=True) == signed_maximum

    overflowing = _exact_binary_mpf(mp, sign, 1, int(finfo.maxexp))
    with pytest.raises(ValueError, match="captured callback difference is nonfinite"):
        recorder._callback_number(overflowing, longdouble=True)
    with pytest.raises(ValueError, match="nonfinite number"):
        auditor.number(overflowing, ld=True)


@pytest.mark.parametrize("ambient", [15, 60, 100])
def test_v4_binary_rounding_matches_exact_decimal_oracle_across_exponents(
    numeric_modules, ambient
):
    import mpmath as mp
    import numpy as np

    recorder, auditor = numeric_modules
    finfo = np.finfo(np.longdouble)
    rng = random.Random(1102026)
    leading_exponents = (
        int(finfo.minexp) - 2,
        int(finfo.minexp) - 1,
        int(finfo.minexp),
        -1000,
        -1,
        0,
        100,
        int(finfo.maxexp) - 2,
    )
    with mp.workdps(ambient):
        for leading_exponent in leading_exponents:
            for _ in range(4):
                bitcount = rng.randint(1, 130)
                mantissa = rng.getrandbits(bitcount) | (1 << (bitcount - 1)) | 1
                sign = rng.randrange(2)
                exponent = leading_exponent - mantissa.bit_length() + 1
                value = mp.make_mpf((sign, mantissa, exponent, mantissa.bit_length()))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    expected = np.longdouble(_exact_mpf_decimal_for_test(value))
                assert recorder._mpf_to_longdouble(value) == expected
                assert auditor.number(value, ld=True) == expected


@pytest.mark.parametrize("display_digits", [50, 100])
@pytest.mark.parametrize("ambient", [15, 30, 50, 80, 100])
def test_v4_mpf_low_precision_tuple_with_long_display(
    numeric_modules, display_digits, ambient
):
    import mpmath as mp

    recorder, auditor = numeric_modules
    with mp.workdps(15):
        original = mp.mpf(1) / 7
        encoded = recorder._capture_number(original, display_digits - 4)
    assert encoded["display_digits"] == display_digits
    assert original._mpf_[3] <= 53
    # Generate the presentation in a larger context independently of the
    # decoders' bit-count context; presentation precision is not arithmetic dps.
    with mp.workdps(200):
        assert encoded["decimal"] == mp.nstr(original, n=display_digits)
    with mp.workdps(ambient):
        for decode in (recorder._decode_callback_value, auditor.decode):
            restored = decode(json.loads(json.dumps(encoded)))
            assert restored._mpf_ == original._mpf_
            assert recorder._capture_number(restored, display_digits - 4) == encoded
        assert mp.mp.dps == ambient


@pytest.mark.parametrize("digits", [15, 30, 50, 80, 100])
@pytest.mark.parametrize("ambient", [15, 30, 50, 80, 100])
def test_v4_decimal_display_mutation_rejects_each_decoder(
    numeric_modules, digits, ambient
):
    import mpmath as mp

    recorder, auditor = numeric_modules
    with mp.workdps(digits + 20):
        encoded = recorder._capture_number(
            mp.mpf(1) + mp.power(10, -(digits - 8)), digits
        )
    tuple_before = list(encoded["mpf"])
    encoded["decimal"] = "2.0"
    with mp.workdps(ambient):
        with pytest.raises(
            ValueError, match="MPF decimal display does not match tuple"
        ):
            recorder._decode_callback_value(encoded)
        with pytest.raises(
            ValueError, match="MPF decimal display does not match tuple"
        ):
            auditor.decode(encoded)
    assert encoded["mpf"] == tuple_before


@pytest.mark.parametrize("bad_digits", [None, True, 0, -1, "50"])
def test_v4_mpf_display_precision_required(numeric_modules, bad_digits):
    import mpmath as mp

    recorder, auditor = numeric_modules
    encoded = recorder._capture_number(mp.mpf(1), 50)
    encoded["display_digits"] = bad_digits
    for decode in (recorder._decode_callback_value, auditor.decode):
        with pytest.raises(
            ValueError, match="MPF display precision must be a positive integer"
        ):
            decode(encoded)


@pytest.mark.parametrize(
    "parts", [[0, 2, -1, 2], [False, 1, 0, 1], [0, 1, 0], [0, -1, 0, 1]]
)
def test_v4_mpf_invalid_tuple_rejected_semantically(numeric_modules, parts):
    recorder, auditor = numeric_modules
    value = {"mpf": parts, "decimal": "1.0", "display_digits": 50}
    for decode in (recorder._decode_callback_value, auditor.decode):
        with pytest.raises(ValueError, match="invalid canonical MPF tuple"):
            decode(value)


def _v4_worker_evidence():
    import os

    return {
        "pid": os.getpid(),
        "modules_before": [],
        "modules_after": [],
        "production_module_names": [],
        "production_imported": False,
        "independence_check_passed": True,
    }


def _v4_plane_record(recorder):
    fixtures = recorder.oracle_module("noncavity_fixtures").noncavity_fixtures()
    fixture = next(item for item in fixtures if item["family"] == "temperature_plane")
    reference, metadata = recorder.oracle_module(
        "plane_static_reference"
    ).static_plane_reference(fixture)
    row = {
        "fixture": recorder._jsonable(fixture),
        "reference": recorder._jsonable(reference),
        "metadata": recorder._jsonable(metadata),
    }
    record = recorder.record_row("plane", row, source_row_index=0)
    record["process_independence"] = _v4_worker_evidence()
    return row, record


def test_v4_semantic_record_does_not_inherit_v3_trust(numeric_modules):
    recorder, auditor = numeric_modules
    row, record = _v4_plane_record(recorder)
    record["zero_certificate_supplement_sha256"] = auditor.ZERO_SHA256
    record.update(
        {"provenance_validation": "PROVENANCE_VERIFIED", "full_gate_complete": True}
    )

    class LegacyProvider:
        called = False

        def __call__(self, *_args, **_kwargs):
            self.called = True
            raise AssertionError("legacy provider must not supply V4 trust")

    provider = LegacyProvider()
    recorder._validate_trace_record(
        record, "plane", 0, row, zero_certificate_sha256=auditor.ZERO_SHA256
    )
    result = auditor.audit_record(
        "plane",
        0,
        row,
        record,
        provenance_provider=provider,
        provenance_binding={"run_id": "v3"},
    )
    assert result["semantic_validation"] == "SEMANTIC_VALID"
    assert result["provenance_validation"]["status"] == "PROVENANCE_UNVERIFIED"
    assert result["scientific_admission"] == "INCOMPLETE_EVIDENCE"
    assert not provider.called


def test_v4_decimal_display_mutation_rejects_record_validation(numeric_modules):
    recorder, auditor = numeric_modules
    row, record = _v4_plane_record(recorder)
    record["zero_certificate_supplement_sha256"] = auditor.ZERO_SHA256
    record["original_entrypoint"]["captured_calls"][0]["inputs"]["display_probe"] = {
        "mpf": [0, 1, 0, 1],
        "decimal": "2.0",
        "display_digits": 50,
    }
    with pytest.raises(ValueError, match="MPF decimal display does not match tuple"):
        recorder._validate_trace_record(
            record, "plane", 0, row, zero_certificate_sha256=auditor.ZERO_SHA256
        )
    with pytest.raises(ValueError, match="MPF decimal display does not match tuple"):
        auditor.audit_record("plane", 0, row, record)


def test_v4_numeric_scaffolds_do_not_run_legacy_orchestration(numeric_modules):
    recorder, auditor = numeric_modules
    for entrypoint in (
        recorder.main,
        recorder.run_dataset,
        auditor.main,
        auditor.audit_dataset,
    ):
        with pytest.raises(RuntimeError, match="V4.*not implemented"):
            entrypoint(None)


def test_v4_auditor_does_not_import_recorder_or_v3_provenance():
    source = (
        Path(__file__).with_name("audit_frozen_refinement_traces_v4.py").read_text()
    )
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            assert all("record_frozen_refinement_traces" not in name for name in names)
    assert "refinement_trace_provenance_v3.py" not in source


@pytest.fixture(scope="module")
def contract():
    path = Path(__file__).with_name("refinement_trace_provenance_v4.py")
    assert path.exists(), "V4 contract implementation is required"
    spec = importlib.util.spec_from_file_location("gen1_provenance_v4_contract", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def digest(value):
    """Independent test calculation, without the contract's serializer/helper."""
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def inputs(c):
    row = c.IndependentAuditInputs(
        dataset="sphere",
        dataset_sha256="a" * 64,
        source_row_index=7,
        fixture={"radius": "1.25", "point": ["2", "0", "0"]},
        reference={"potential": "-0.5", "gradient": ["0.25", "0", "0"]},
        required_observables=("potential", "gradient"),
        oracle_module_sha256="b" * 64,
        kernel_identities={"potential": "c" * 64, "gradient": "d" * 64},
        frozen_reference_sha256="e" * 64,
        certificate_sha256=("f" * 64,),
        root_supplement=None,
    )
    profile = c.TrustedRunProfile(
        run_id="new-v4-run",
        observer_sha256="1" * 64,
        auditor_sha256="2" * 64,
        provenance_sha256="3" * 64,
        runner_sha256="4" * 64,
        environment={"python": "3.12", "mpmath": "1.3.0", "platform": "test"},
        numeric_representation={"format": "mpf-tuple", "dps": 80},
    )
    callbacks = [
        {"kernel": "potential", "input": {"r": "2"}, "output": "-0.5"},
        {"kernel": "gradient", "input": {"r": "2"}, "output": ["0.25", "0", "0"]},
    ]
    journal = c.AttemptJournalInputs(
        events=[
            {
                "sequence": 0,
                "attempt_id": "failed-0",
                "status": "failed",
                "error": "worker died",
            },
            {
                "sequence": 1,
                "attempt_id": "retry-1",
                "retry_of": "failed-0",
                "status": "complete",
            },
        ],
        attempt_states={"failed-0": "failed", "retry-1": "complete"},
        retry_plan=[{"attempt_id": "retry-1", "retry_of": "failed-0"}],
        accepted_results={
            "sphere:7": {"attempt_id": "retry-1", "result_sha256": "3" * 64}
        },
    )
    return row, profile, callbacks, journal


def identity(c):
    row, profile, callbacks, journal = inputs(c)
    return c.derive_expected_row_identity(
        row,
        profile,
        callbacks=callbacks,
        attempt_journal=journal,
    )


def document(c):
    return c.commitment_document(identity(c))


class MockProvider:
    """Retrieval-only double. Its flags carry no authentication authority."""

    def __init__(self, value):
        self.value = value
        self.lookups = []

    def fetch_anchor(self, lookup_key):
        self.lookups.append(lookup_key)
        return copy.deepcopy(self.value)


class MockVerifier:
    """Test-only trust adapter. Matching tests demonstrate API matching only."""

    def __init__(self, c, *, authenticated=True, root_id="unit-test-root"):
        self.c = c
        self.authenticated = authenticated
        self.root_id = root_id

    def authenticate_anchor(self, raw_anchor, *, trust_root_id):
        assert trust_root_id == "unit-test-root"
        if not self.authenticated:
            return None
        return self.c.AuthenticatedAnchor(
            document=raw_anchor["document"],
            anchor_id="test-only-anchor",
            trust_root_id=self.root_id,
        )


def verify(c, doc, *, provider=None, verifier=None, profile=True):
    trust = (
        c.TrustedVerifierProfile(
            trust_root_id="unit-test-root",
            verifier=verifier or MockVerifier(c),
        )
        if profile
        else None
    )
    return c.verify_external_anchor(
        doc,
        provider=provider,
        verifier_profile=trust,
        lookup_key="sphere:7",
    )


def checks():
    return {
        "execution_state": "complete",
        "semantic_status": "SEMANTIC_VALID",
        "provenance_status": "PROVENANCE_VERIFIED",
        "required_evidence_complete": True,
        "independent_audit_valid": True,
        "certificate_validation_valid": True,
        "trend_validation_complete": True,
        "declared_scope_valid": True,
        "journal_binding_complete": True,
        "prior_failures": False,
        "accepted_retry_result": False,
        "accepted_result_anchored": True,
    }


def test_exact_v4_states_and_schema_ids(contract):
    c = contract
    assert {
        "not_started",
        "running",
        "complete",
        "failed",
        "interrupted",
    } == c.EXECUTION_STATES
    assert {
        "SEMANTIC_VALID",
        "SEMANTIC_INVALID",
        "SEMANTIC_NOT_RUN",
    } == c.SEMANTIC_STATES
    assert {
        "PROVENANCE_VERIFIED",
        "PROVENANCE_UNVERIFIED",
        "PROVENANCE_MISMATCH",
        "PROVENANCE_NOT_RUN",
    } == c.PROVENANCE_STATES
    assert {
        "ADMITTED_WITH_DECLARED_SCOPE",
        "NOT_ADMITTED",
        "INCOMPLETE_EVIDENCE",
    } == c.ADMISSION_STATES
    assert c.REPORT_SCHEMA == "gragra.gen1.refinement-trace-report.v4"
    assert c.IDENTITY_SCHEMA == "gragra.gen1.refinement-trace-row-identity.v4"
    assert (
        c.COMMITMENT_SCHEMA == "gragra.gen1.refinement-trace-provenance-commitment.v4"
    )
    assert all(value.endswith(".v4") for value in c.V4_SCHEMA_IDENTITIES.values())


def test_expected_identity_is_independently_computed(contract):
    c = contract
    row, profile, callbacks, journal = inputs(c)
    expected = {
        "schema": c.IDENTITY_SCHEMA,
        "dataset": "sphere",
        "dataset_sha256": "a" * 64,
        "source_row_index": 7,
        "fixture_sha256": digest({"radius": "1.25", "point": ["2", "0", "0"]}),
        "reference_values_sha256": digest(
            {"potential": "-0.5", "gradient": ["0.25", "0", "0"]}
        ),
        "required_observables": ["gradient", "potential"],
        "oracle_module_sha256": "b" * 64,
        "kernel_identities": {"potential": "c" * 64, "gradient": "d" * 64},
        "frozen_reference_sha256": "e" * 64,
        "certificate_sha256": ["f" * 64],
        "root_supplement": None,
        "run_id": "new-v4-run",
        "observer_sha256": "1" * 64,
        "auditor_sha256": "2" * 64,
        "provenance_sha256": "3" * 64,
        "runner_sha256": "4" * 64,
        "environment": {"python": "3.12", "mpmath": "1.3.0", "platform": "test"},
        "numeric_representation": {"format": "mpf-tuple", "dps": 80},
        "callback_count": 2,
        "callbacks_sha256": digest(callbacks),
        "attempt_journal_sha256": digest(
            {
                "events": journal.events,
                "attempt_states": journal.attempt_states,
                "retry_plan": journal.retry_plan,
                "accepted_results": journal.accepted_results,
            }
        ),
    }
    assert identity(c) == expected
    body = {"schema": c.COMMITMENT_SCHEMA, "payload": expected}
    assert document(c) == {**body, "commitment_sha256": digest(body)}


def test_v4_contract_and_runner_sources_are_bound_into_expected_identity(contract):
    from dataclasses import replace

    row, profile, callbacks, journal = inputs(contract)
    baseline = contract.derive_expected_row_identity(
        row, profile, callbacks=callbacks, attempt_journal=journal
    )
    for field in ("provenance_sha256", "runner_sha256"):
        changed = replace(profile, **{field: "9" * 64})
        derived = contract.derive_expected_row_identity(
            row, changed, callbacks=callbacks, attempt_journal=journal
        )
        assert derived != baseline
        assert derived[field] == "9" * 64


def test_expected_identity_has_no_provider_or_mutable_binding_argument(contract):
    import inspect

    names = set(inspect.signature(contract.derive_expected_row_identity).parameters)
    assert names == {"audit_inputs", "run_profile", "callbacks", "attempt_journal"}


TRUSTED_JSON_PATHS = [
    ("environment", ("packages", 0, "name")),
    ("environment", ("packages", 0)),
    ("numeric_representation", ("formats", 0, "dps")),
    ("numeric_representation", ("formats", 0)),
    ("fixture", ("point", 0)),
    ("reference", ("gradient", 0)),
    ("kernel_identities", ("potential",)),
    ("root_supplement", ("scope_sha256",)),
]


def trusted_json_inputs(c):
    from dataclasses import replace

    row, profile, callbacks, journal = inputs(c)
    sources = {
        "fixture": {"radius": "1.25", "point": ["2", "0", "0"]},
        "reference": {"potential": "-0.5", "gradient": ["0.25", "0", "0"]},
        "kernel_identities": {"potential": "c" * 64, "gradient": "d" * 64},
        "root_supplement": {
            "schema": c.ROOT_SUPPLEMENT_SCHEMA,
            "sha256": "4" * 64,
            "scope_sha256": "5" * 64,
        },
        "environment": {"packages": [{"name": "mpmath", "version": "1.3.0"}]},
        "numeric_representation": {"formats": [{"format": "mpf-tuple", "dps": 80}]},
    }
    row = replace(
        row,
        **{
            field: sources[field]
            for field in (
                "fixture",
                "reference",
                "kernel_identities",
                "root_supplement",
            )
        },
    )
    profile = replace(
        profile,
        **{
            field: sources[field] for field in ("environment", "numeric_representation")
        },
    )
    return row, profile, callbacks, journal, sources


@pytest.mark.parametrize("field,path", TRUSTED_JSON_PATHS)
def test_trusted_json_snapshots_are_detached_from_source_aliases(contract, field, path):
    c = contract
    row, profile, callbacks, journal, sources = trusted_json_inputs(c)
    baseline = c.derive_expected_row_identity(
        row, profile, callbacks=callbacks, attempt_journal=journal
    )
    target = sources[field]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = "changed"
    assert (
        c.derive_expected_row_identity(
            row, profile, callbacks=callbacks, attempt_journal=journal
        )
        == baseline
    )


@pytest.mark.parametrize("field,path", TRUSTED_JSON_PATHS)
def test_trusted_json_field_copies_isolate_nested_mutation(contract, field, path):
    c = contract
    row, profile, callbacks, journal, sources = trusted_json_inputs(c)
    baseline = c.derive_expected_row_identity(
        row, profile, callbacks=callbacks, attempt_journal=journal
    )
    owner = profile if field in {"environment", "numeric_representation"} else row
    target = getattr(owner, field)
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = "changed"
    assert getattr(owner, field) == sources[field]
    assert (
        c.derive_expected_row_identity(
            row, profile, callbacks=callbacks, attempt_journal=journal
        )
        == baseline
    )


class Capture:
    """Retain the equality operand, including a proxy's hidden backing dict."""

    def __eq__(self, other):
        self.value = other
        return True


def test_trusted_profile_hostile_equality_cannot_change_dps_identity(contract):
    c = contract
    row, profile, callbacks, journal = inputs(c)
    baseline = c.derive_expected_row_identity(
        row, profile, callbacks=callbacks, attempt_journal=journal
    )
    capture = Capture()
    assert profile.numeric_representation == capture
    capture.value["dps"] = 20
    assert profile.numeric_representation["dps"] == 80
    assert (
        c.derive_expected_row_identity(
            row, profile, callbacks=callbacks, attempt_journal=journal
        )
        == baseline
    )


@pytest.mark.parametrize("field,path", TRUSTED_JSON_PATHS)
def test_trusted_json_hostile_equality_retains_only_detached_copies(
    contract, field, path
):
    c = contract
    row, profile, callbacks, journal, sources = trusted_json_inputs(c)
    baseline = c.derive_expected_row_identity(
        row, profile, callbacks=callbacks, attempt_journal=journal
    )
    owner = profile if field in {"environment", "numeric_representation"} else row
    target = getattr(owner, field)
    for key in path[:-1]:
        target = target[key]
    capture = Capture()
    assert target == capture
    capture.value[path[-1]] = "changed"
    assert getattr(owner, field) == sources[field]
    assert (
        c.derive_expected_row_identity(
            row, profile, callbacks=callbacks, attempt_journal=journal
        )
        == baseline
    )


@pytest.mark.parametrize("with_root", [False, True])
def test_trusted_json_dataclass_asdict_and_replace_preserve_isolation(
    contract, with_root
):
    from dataclasses import asdict, replace

    c = contract
    row, profile, callbacks, journal, _ = trusted_json_inputs(c)
    if not with_root:
        row = replace(row, root_supplement=None)
    baseline = c.derive_expected_row_identity(
        row, profile, callbacks=callbacks, attempt_journal=journal
    )
    row_fields, profile_fields = asdict(row), asdict(profile)
    replaced_row = c.IndependentAuditInputs(**row_fields)
    replaced_profile = c.TrustedRunProfile(**profile_fields)
    assert (
        c.derive_expected_row_identity(
            replace(replaced_row),
            replace(replaced_profile),
            callbacks=callbacks,
            attempt_journal=journal,
        )
        == baseline
    )
    row_fields["fixture"]["point"][0] = "changed"
    row_fields["reference"]["gradient"][0] = "changed"
    row_fields["kernel_identities"]["potential"] = "0" * 64
    profile_fields["environment"]["packages"][0]["name"] = "changed"
    profile_fields["numeric_representation"]["formats"][0]["dps"] = 20
    if with_root:
        row_fields["root_supplement"]["scope_sha256"] = "0" * 64
    else:
        assert row_fields["root_supplement"] is None
    for current_row, current_profile in (
        (row, profile),
        (replaced_row, replaced_profile),
    ):
        assert (
            c.derive_expected_row_identity(
                current_row,
                current_profile,
                callbacks=callbacks,
                attempt_journal=journal,
            )
            == baseline
        )


def test_trusted_json_snapshots_support_replace_and_derive_plain_json(contract):
    from dataclasses import replace

    c = contract
    row, profile, callbacks, journal, sources = trusted_json_inputs(c)
    expected = c.derive_expected_row_identity(
        row, profile, callbacks=callbacks, attempt_journal=journal
    )
    changed = c.derive_expected_row_identity(
        replace(row), replace(profile), callbacks=callbacks, attempt_journal=journal
    )
    assert changed == expected
    assert changed["environment"] == sources["environment"]
    assert changed["numeric_representation"] == sources["numeric_representation"]
    assert changed["root_supplement"] == sources["root_supplement"]
    assert type(changed["environment"]) is dict
    assert type(changed["environment"]["packages"]) is list
    assert type(changed["environment"]["packages"][0]) is dict
    assert type(changed["numeric_representation"]["formats"]) is list
    c.canonical_json_bytes(changed)
    changed["environment"]["packages"][0]["name"] = "changed output"
    assert (
        c.derive_expected_row_identity(
            row, profile, callbacks=callbacks, attempt_journal=journal
        )
        == expected
    )


@pytest.mark.parametrize(
    "field",
    [
        "dataset",
        "dataset_sha256",
        "source_row_index",
        "fixture_sha256",
        "reference_values_sha256",
        "required_observables",
        "oracle_module_sha256",
        "kernel_identities",
        "run_id",
        "environment",
        "numeric_representation",
        "observer_sha256",
        "auditor_sha256",
        "provenance_sha256",
        "runner_sha256",
        "frozen_reference_sha256",
        "certificate_sha256",
        "root_supplement",
        "callback_count",
        "callbacks_sha256",
        "attempt_journal_sha256",
    ],
)
def test_provenance_scope_rejects_each_authenticated_a6_identity_mismatch(
    contract, field
):
    c = contract
    expected = document(c)
    altered = copy.deepcopy(expected)
    value = altered["payload"][field]
    if field == "root_supplement":
        altered["payload"][field] = {
            "schema": c.ROOT_SUPPLEMENT_SCHEMA,
            "sha256": "0" * 64,
            "scope_sha256": "0" * 64,
        }
    elif isinstance(value, str):
        altered["payload"][field] = "0" * 64 if field.endswith("sha256") else "other"
    elif isinstance(value, int):
        altered["payload"][field] += 1
    elif isinstance(value, list):
        altered["payload"][field] = (
            ["0" * 64] if field == "certificate_sha256" else ["potential"]
        )
        if field == "required_observables":
            altered["payload"]["kernel_identities"] = {
                "potential": altered["payload"]["kernel_identities"]["potential"]
            }
    elif isinstance(value, dict):
        altered["payload"][field] = {**value, next(iter(value)): "0" * 64}
    body = {"schema": altered["schema"], "payload": altered["payload"]}
    altered["commitment_sha256"] = digest(body)
    result = verify(c, expected, provider=MockProvider({"document": altered}))
    assert result.status == "PROVENANCE_MISMATCH"
    assert (
        result.reason == "authenticated identity differs from independent audit inputs"
    )
    assert (
        c.admission_decision(**{**checks(), "provenance_status": result.status})
        == "NOT_ADMITTED"
    )


@pytest.mark.parametrize(
    "part", ["events", "attempt_states", "retry_plan", "accepted_results"]
)
def test_expected_identity_binds_complete_journal_including_failures_and_retry(
    contract, part
):
    c = contract
    row, profile, callbacks, journal = inputs(c)
    baseline = identity(c)
    getattr(journal, part).clear()
    changed = c.derive_expected_row_identity(
        row, profile, callbacks=callbacks, attempt_journal=journal
    )
    assert changed["attempt_journal_sha256"] != baseline["attempt_journal_sha256"]


def test_expected_identity_binds_callback_order_input_and_output(contract):
    c = contract
    row, profile, callbacks, journal = inputs(c)
    baseline = identity(c)["callbacks_sha256"]
    for changed in (
        list(reversed(callbacks)),
        [{**callbacks[0], "input": {}}, callbacks[1]],
        [{**callbacks[0], "output": "wrong"}, callbacks[1]],
    ):
        assert (
            c.derive_expected_row_identity(
                row, profile, callbacks=changed, attempt_journal=journal
            )["callbacks_sha256"]
            != baseline
        )


@pytest.mark.parametrize(
    "case",
    [
        "no_provider",
        "no_profile",
        "no_run_profile",
        "provider_exception",
        "verifier_exception",
        "provider_verified_flag",
        "untrusted_verifier_flag",
        "wrong_root",
    ],
)
def test_provenance_never_trusts_missing_or_self_asserted_provider_material(
    contract, case
):
    c = contract
    doc = document(c)
    provider = MockProvider(
        {"document": doc, "status": "PROVENANCE_VERIFIED", "trust_root_verified": True}
    )
    verifier = MockVerifier(c)
    profile = True
    if case == "no_provider":
        provider = None
    elif case == "no_profile":
        profile = False
    elif case == "no_run_profile":
        doc = None
    elif case == "provider_exception":

        def fail(_):
            raise RuntimeError("fetch failed")

        provider.fetch_anchor = fail
    elif case == "verifier_exception":

        def fail(*_args, **_kwargs):
            raise RuntimeError("authentication failed")

        verifier.authenticate_anchor = fail
    elif case == "provider_verified_flag":
        verifier = MockVerifier(c, authenticated=False)
    elif case == "untrusted_verifier_flag":
        verifier.authenticate_anchor = lambda *_args, **_kwargs: {
            "document": doc,
            "trust_root_verified": True,
        }
    elif case == "wrong_root":
        verifier = MockVerifier(c, root_id="unconfigured-root")
    result = verify(c, doc, provider=provider, verifier=verifier, profile=profile)
    assert result.status == "PROVENANCE_UNVERIFIED"


def test_matching_mock_authenticated_anchor_demonstrates_api_matching_only(contract):
    c = contract
    provider = MockProvider({"document": document(c)})
    result = verify(c, document(c), provider=provider)
    assert result.status == "PROVENANCE_VERIFIED"
    assert result.anchor_id == "test-only-anchor"
    assert provider.lookups == ["sphere:7"]


@pytest.mark.parametrize("change", ["missing", "extra", "digest", "schema", "non_json"])
def test_authenticated_malformed_anchor_is_mismatch(contract, change):
    c = contract
    doc = document(c)
    bad = copy.deepcopy(doc)
    if change == "missing":
        del bad["payload"]["run_id"]
    elif change == "extra":
        bad["payload"]["trust_root_verified"] = True
    elif change == "digest":
        bad["commitment_sha256"] = "0" * 64
    elif change == "schema":
        bad["schema"] = "gragra.gen1.refinement-trace-provenance-commitment.v3"
    else:
        bad["payload"]["environment"]["nan"] = float("nan")
    assert (
        verify(c, doc, provider=MockProvider({"document": bad})).status
        == "PROVENANCE_MISMATCH"
    )


@pytest.mark.parametrize(
    "bad",
    [
        {1: "integer key"},
        (1, 2),
        {"x": float("nan")},
        {"x": float("inf")},
        {"x": object()},
        {"x": "\ud800"},
    ],
)
def test_strict_canonical_serialization_rejects_non_json_values(contract, bad):
    with pytest.raises((TypeError, ValueError)):
        contract.canonical_json_bytes(bad)


def test_canonical_json_is_stable_and_preserves_unicode(contract):
    assert (
        contract.canonical_json_bytes({"z": [1, True, None], "a": "日本"})
        == '{"a":"日本","z":[1,true,null]}'.encode()
    )


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', "[]"])
def test_strict_json_object_loader_rejects_duplicates_constants_and_nonobjects(
    contract, text
):
    with pytest.raises(ValueError):
        contract.strict_json_object(text)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_row_index", True),
        ("dataset_sha256", "short"),
        ("required_observables", ("potential", "potential")),
        ("kernel_identities", {}),
        ("certificate_sha256", ("not-a-digest",)),
        ("root_supplement", {"sha256": "0" * 64}),
    ],
)
def test_expected_identity_rejects_missing_or_malformed_independent_pins(
    contract, field, value
):
    from dataclasses import replace

    c = contract
    row, profile, callbacks, journal = inputs(c)
    with pytest.raises((TypeError, ValueError)):
        c.derive_expected_row_identity(
            replace(row, **{field: value}),
            profile,
            callbacks=callbacks,
            attempt_journal=journal,
        )


def test_absent_trusted_run_profile_cannot_derive_expected_identity(contract):
    row, _, callbacks, journal = inputs(contract)
    assert (
        contract.derive_expected_row_identity(
            row, None, callbacks=callbacks, attempt_journal=journal
        )
        is None
    )


@pytest.mark.parametrize(
    "execution,semantic,provenance",
    list(
        itertools.product(
            ("not_started", "running", "complete", "failed", "interrupted"),
            ("SEMANTIC_VALID", "SEMANTIC_INVALID", "SEMANTIC_NOT_RUN"),
            (
                "PROVENANCE_VERIFIED",
                "PROVENANCE_UNVERIFIED",
                "PROVENANCE_MISMATCH",
                "PROVENANCE_NOT_RUN",
            ),
        )
    ),
)
def test_admission_all_state_cases(contract, execution, semantic, provenance):
    if (
        semantic == "SEMANTIC_INVALID"
        or provenance == "PROVENANCE_MISMATCH"
        or execution == "failed"
    ):
        expected = "NOT_ADMITTED"
    elif (
        execution in {"not_started", "running", "interrupted"}
        or semantic == "SEMANTIC_NOT_RUN"
    ):
        expected = "INCOMPLETE_EVIDENCE"
    elif provenance != "PROVENANCE_VERIFIED":
        expected = "NOT_ADMITTED"
    else:
        expected = "ADMITTED_WITH_DECLARED_SCOPE"
    args = {
        **checks(),
        "execution_state": execution,
        "semantic_status": semantic,
        "provenance_status": provenance,
    }
    assert contract.admission_decision(**args) == expected


@pytest.mark.parametrize(
    "field",
    [
        "required_evidence_complete",
        "trend_validation_complete",
        "journal_binding_complete",
        "accepted_result_anchored",
    ],
)
def test_admission_missing_required_evidence_is_incomplete(contract, field):
    assert (
        contract.admission_decision(**{**checks(), field: False})
        == "INCOMPLETE_EVIDENCE"
    )


@pytest.mark.parametrize(
    "field",
    ["independent_audit_valid", "certificate_validation_valid", "declared_scope_valid"],
)
def test_admission_invalid_checks_are_denied_missing_checks_are_incomplete(
    contract, field
):
    assert contract.admission_decision(**{**checks(), field: False}) == "NOT_ADMITTED"
    assert (
        contract.admission_decision(**{**checks(), field: None})
        == "INCOMPLETE_EVIDENCE"
    )


@pytest.mark.parametrize(
    "field",
    ["independent_audit_valid", "certificate_validation_valid", "declared_scope_valid"],
)
@pytest.mark.parametrize(
    "patch",
    [
        {"execution_state": "running"},
        {"execution_state": "interrupted"},
        {"semantic_status": "SEMANTIC_NOT_RUN"},
        {"required_evidence_complete": False},
    ],
)
def test_admission_invalid_checks_precede_incomplete_states(contract, field, patch):
    args = {**checks(), **patch}
    assert contract.admission_decision(**{**args, field: False}) == "NOT_ADMITTED"
    assert contract.admission_decision(**{**args, field: None}) == "INCOMPLETE_EVIDENCE"


@pytest.mark.parametrize(
    "field,value",
    [
        ("execution_state", "done"),
        ("semantic_status", "VALID"),
        ("provenance_status", "VERIFIED"),
        ("required_evidence_complete", 1),
        ("prior_failures", "false"),
    ],
)
def test_admission_rejects_unknown_states_and_nonboolean_flags(contract, field, value):
    assert contract.admission_decision(**{**checks(), field: value}) == "NOT_ADMITTED"


def test_admission_explicit_retry_requires_result_and_history_anchor(
    contract,
):
    c = contract
    retry = {**checks(), "prior_failures": True, "accepted_retry_result": True}
    assert c.admission_decision(**retry) == "ADMITTED_WITH_DECLARED_SCOPE"
    assert (
        c.admission_decision(**{**retry, "accepted_retry_result": False})
        == "NOT_ADMITTED"
    )
    assert (
        c.admission_decision(**{**retry, "accepted_result_anchored": False})
        == "INCOMPLETE_EVIDENCE"
    )
    assert (
        c.admission_decision(**{**retry, "journal_binding_complete": False})
        == "INCOMPLETE_EVIDENCE"
    )
    assert (
        c.admission_decision(**{**retry, "execution_state": "failed"}) == "NOT_ADMITTED"
    )


def test_admission_known_invalidity_and_failure_precede_missing_evidence(contract):
    c = contract
    for patch in (
        {"semantic_status": "SEMANTIC_INVALID"},
        {"provenance_status": "PROVENANCE_MISMATCH"},
        {"execution_state": "failed"},
    ):
        assert (
            c.admission_decision(
                **{**checks(), "required_evidence_complete": False, **patch}
            )
            == "NOT_ADMITTED"
        )


def report(c):
    sources = {name: str(i) * 64 for i, name in enumerate(c.REPORT_SOURCE_FIELDS, 1)}
    args = checks()
    value = {
        "schema": c.REPORT_SCHEMA,
        "source_identity": copy.deepcopy(sources),
        "schema_identity": dict(c.V4_SCHEMA_IDENTITIES),
        "execution_state": args.pop("execution_state"),
        "semantic_status": args.pop("semantic_status"),
        "provenance_status": args.pop("provenance_status"),
        "checks": args,
        "admission": "ADMITTED_WITH_DECLARED_SCOPE",
    }
    return value, sources


@pytest.mark.parametrize("version", ["v1", "v3"])
def test_legacy_report_full_gate_complete_cannot_pass_v4_schema_gate(contract, version):
    value = {
        "schema": f"gragra.gen1.refinement-trace-report.{version}",
        "full_gate_complete": True,
    }
    with pytest.raises(ValueError, match="V4"):
        contract.validate_v4_report(value, expected_source_identity={})


def test_structurally_valid_v4_report_passes_schema_gate_only(contract):
    value, sources = report(contract)
    assert contract.validate_v4_report(value, expected_source_identity=sources) == value


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "source",
        "schema_identity",
        "admission",
        "bad_state",
        "bool_claim",
    ],
)
def test_report_schema_gate_rejects_inexact_sources_schemas_and_states(
    contract, change
):
    c = contract
    value, sources = report(c)
    if change == "missing":
        del value["checks"]["journal_binding_complete"]
    elif change == "extra":
        value["full_gate_complete"] = True
    elif change == "source":
        value["source_identity"]["auditor_sha256"] = "0" * 64
    elif change == "schema_identity":
        value["schema_identity"]["trace"] = "legacy.v3"
    elif change == "admission":
        value["provenance_status"] = "PROVENANCE_UNVERIFIED"
    elif change == "bad_state":
        value["execution_state"] = "done"
    else:
        value["checks"]["journal_binding_complete"] = "true"
    with pytest.raises(ValueError):
        c.validate_v4_report(value, expected_source_identity=sources)


@pytest.mark.parametrize("digits", [15, 30, 50, 80, 100])
def test_v4_precision_round_trip_and_semantic_pair_check(numeric_modules, digits):
    recorder, auditor = numeric_modules
    import mpmath as mp

    with mp.workdps(digits + 20):
        delta = mp.power(10, -(digits - 8))
        values = [mp.mpf(1) + 2 * delta, mp.mpf(1) + delta, mp.mpf(1)]
        encoded = [recorder._capture_number(value, digits) for value in values]
        expected_tuple = [tuple(item["mpf"]) for item in encoded]

    previous_dps = mp.mp.dps
    try:
        mp.mp.dps = 15
        recorder_values = [recorder._decode_callback_value(item) for item in encoded]
        auditor_values = [auditor.decode(item) for item in encoded]
    finally:
        mp.mp.dps = previous_dps

    assert [tuple(value._mpf_) for value in recorder_values] == expected_tuple
    assert [tuple(value._mpf_) for value in auditor_values] == expected_tuple
    with mp.workdps(digits + 20):
        recorder_pair = recorder._callback_pair(
            [(value,) for value in recorder_values], "potential", longdouble=False
        )
        auditor_pair = tuple(
            auditor.vector_difference((left,), (right,))
            for left, right in zip(
                recorder_values[:-1], recorder_values[1:], strict=True
            )
        )
        assert recorder_pair == auditor_pair
        trace = {
            "derivative_differences_si": {
                "4h_to_2h": recorder._exact_mp_string(recorder_pair[0]),
                "2h_to_h": recorder._exact_mp_string(recorder_pair[1]),
                "ten_max": recorder._exact_mp_string(10 * max(recorder_pair)),
            }
        }
        recorder._require_saved_pair(
            trace,
            "derivative_differences_si",
            recorder_pair,
            ("4h_to_2h", "2h_to_h"),
            longdouble=False,
            precision=digits + 20,
            index=0,
            observable="potential",
        )
        auditor_trend = {
            "classification": "unresolved_nondecreasing",
            "nondecreasing": True,
            "coarse_to_middle_si": recorder._exact_mp_string(recorder_pair[0]),
            "middle_to_fine_si": recorder._exact_mp_string(recorder_pair[1]),
            "independent_floor_si": None,
        }
        auditor.audit_trend(auditor_trend, auditor_pair, None)
        trace["derivative_differences_si"]["2h_to_h"] = mp.nstr(
            recorder_pair[1] * 2, digits + 30
        )
        with pytest.raises(ValueError, match="differs from captured callbacks"):
            recorder._require_saved_pair(
                trace,
                "derivative_differences_si",
                recorder_pair,
                ("4h_to_2h", "2h_to_h"),
                longdouble=False,
                precision=digits + 20,
                index=0,
                observable="potential",
            )
        auditor_trend["middle_to_fine_si"] = trace["derivative_differences_si"][
            "2h_to_h"
        ]
        with pytest.raises(ValueError, match="trend/named pair mismatch"):
            auditor.audit_trend(auditor_trend, auditor_pair, None)


def test_v4_a3_50_digit_fallback_positive_and_perturbed_negative_control(
    numeric_modules,
):
    recorder, auditor = numeric_modules
    import mpmath as mp

    digits = 50
    with mp.workdps(80):
        delta = mp.mpf("1e-42")
        vectors = [
            [mp.mpf(1) + 2 * delta] + [mp.mpf(0)] * 12,
            [mp.mpf(1) + delta] + [mp.mpf(0)] * 12,
            [mp.mpf(1)] + [mp.mpf(0)] * 12,
        ]
        calls = [
            {
                "kernel": "surface_high_precision_evaluate",
                "inputs": {"digits": digits, "order": order},
                "output": [
                    [recorder._capture_number(value, digits) for value in vector],
                    [recorder._capture_number(mp.mpf(1), digits) for _ in range(13)],
                    1,
                ],
            }
            for order, vector in zip((16, 24, 32), vectors, strict=True)
        ]
        expected_pair, _ = recorder._surface_callback_pair(
            calls, "cuboid", {}, {}, {"route": "mpmath_fallback"}, "potential"
        )
        auditor_pair, _auditor_vectors, _lookup = auditor.high_precision_callback_pair(
            calls, "potential"
        )
        assert expected_pair == auditor_pair
        saved = {
            "quadrature_differences_si": {
                "coarse_to_middle": recorder._exact_mp_string(expected_pair[0]),
                "middle_to_fine": recorder._exact_mp_string(expected_pair[1]),
                "ten_max": recorder._exact_mp_string(10 * max(expected_pair)),
            }
        }
        recorder._require_saved_pair(
            saved,
            "quadrature_differences_si",
            expected_pair,
            ("coarse_to_middle", "middle_to_fine"),
            longdouble=False,
            precision=80,
            index=0,
            observable="potential",
        )
        trend = {
            "classification": "unresolved_nondecreasing",
            "nondecreasing": True,
            "coarse_to_middle_si": recorder._exact_mp_string(expected_pair[0]),
            "middle_to_fine_si": recorder._exact_mp_string(expected_pair[1]),
            "independent_floor_si": None,
        }
        auditor.audit_trend(trend, expected_pair, None)

        calls[0]["output"][0][0] = recorder._capture_number(
            mp.mpf(1) + 3 * delta, digits
        )
        perturbed_pair, _ = recorder._surface_callback_pair(
            calls, "cuboid", {}, {}, {"route": "mpmath_fallback"}, "potential"
        )
        perturbed_auditor_pair, _values, _map = auditor.high_precision_callback_pair(
            calls, "potential"
        )
        assert perturbed_pair == perturbed_auditor_pair
        assert perturbed_pair != expected_pair
        with pytest.raises(ValueError, match="differs from captured callbacks"):
            recorder._require_saved_pair(
                saved,
                "quadrature_differences_si",
                perturbed_pair,
                ("coarse_to_middle", "middle_to_fine"),
                longdouble=False,
                precision=80,
                index=0,
                observable="potential",
            )
        with pytest.raises(ValueError, match="trend/named pair mismatch"):
            auditor.audit_trend(trend, perturbed_auditor_pair, None)


def test_v4_a3_fallback_positive_control_uses_pinned_high_precision_oracle_route(
    numeric_modules,
):
    recorder, auditor = numeric_modules
    fixture = next(
        item
        for item in recorder.oracle_module("cavity_fixtures").cuboid_fixtures()
        if item.target_kind == "exterior" and item.lengths_m == (1.0, 1.0, 1.0)
    )
    fixture_json = recorder._jsonable(fixture)
    _initial, metadata = recorder._invoke_original_entrypoint("cuboid", fixture_json)
    acceptance = recorder.oracle_module("acceptance")
    unresolved = {
        name: {
            "values_si": (),
            "uncertainty": acceptance.OracleUncertainty(
                None, None, None, None, "synthetic unresolved input", ()
            ),
        }
        for name in recorder.OBSERVABLE_SLICES
    }
    captures = []
    with recorder.capture_context("cuboid", captures):
        output, fallback_metadata = recorder.oracle_module(
            "surface_high_precision"
        ).high_precision_reference(
            recorder._surface_fixture("cuboid", fixture_json), 6.67430e-11, unresolved
        )
    metadata.update(fallback_metadata)
    row = {
        "fixture": fixture_json,
        "reference": recorder._jsonable(output),
        "metadata": recorder._jsonable(metadata),
    }
    record = recorder._case_trace(
        "cuboid",
        row,
        captures,
        output,
        metadata,
        source_row_index=0,
        zero_certificate_sha256=recorder.EXPECTED_ZERO_CERTIFICATES_SHA256,
        polynomial_certificate_sha256=None,
    )
    record["process_independence"] = _v4_worker_evidence()

    recorder._validate_trace_record(
        record,
        "cuboid",
        0,
        row,
        zero_certificate_sha256=recorder.EXPECTED_ZERO_CERTIFICATES_SHA256,
        polynomial_certificate_sha256=None,
    )
    audited = auditor.audit_record("cuboid", 0, row, record)

    assert {trace.get("route") for trace in record["observable_traces"].values()} == {
        "mpmath_fallback"
    }
    assert all(
        call["inputs"].get("digits") in {50, 80}
        for call in captures
        if call["kernel"] == "surface_high_precision_evaluate"
    )
    assert record["status"] == audited["status"]
    assert audited["provenance_validation"]["status"] == "PROVENANCE_UNVERIFIED"
    assert audited["scientific_admission"] == "INCOMPLETE_EVIDENCE"


# Task 3 controls exercise durable execution accounting, not provenance.
def _journal_inputs():
    rows = {
        index: {
            "fixture": {"id": index},
            "reference": {"potential": index},
            "metadata": {"precision": 50},
        }
        for index in range(3)
    }
    manifest = {
        "schema": "gragra.gen1.refinement-trace-dataset-manifest.v4",
        "run_id": "journal-run",
        "dataset": "plane",
        "selected_source_row_indices": list(rows),
        "observer_sha256": "observer-pin",
        "frozen_reference_sha256": "reference-pin",
    }
    return rows, manifest


def _open_journal(recorder, target, *, resume=False, rows=None, manifest=None):
    default_rows, default_manifest = _journal_inputs()
    return recorder.AttemptJournal(
        target,
        run_id="journal-run",
        dataset="plane",
        selected_rows=default_rows if rows is None else rows,
        manifest=default_manifest if manifest is None else manifest,
        resume=resume,
    )


def _journal_success(index):
    rows, _ = _journal_inputs()
    return (
        index,
        {
            "schema": "gragra.gen1.refinement-trace.v4",
            "dataset": "plane",
            "source_row_index": index,
            "fixture": rows[index]["fixture"],
            "frozen_reference": {
                "reference": rows[index]["reference"],
                "metadata": rows[index]["metadata"],
            },
            "status": "reproduced",
            "process_independence": _v4_worker_evidence(),
            "original_entrypoint": {"captured_calls": []},
        },
        None,
    )


def _journal_failure(index):
    return (
        index,
        {
            "status": "replay_error",
            "error": "ValueError: original worker error",
            "exception": {
                "type": "ValueError",
                "message": "original worker error",
                "traceback": "Traceback: original worker error",
            },
            "captured_calls_before_error": [],
            "process_independence": _v4_worker_evidence(),
        },
        "ValueError: original worker error",
    )


@pytest.mark.parametrize("partial", [False, True])
def test_v4_worker_failure_retains_original_traceback(
    numeric_modules, monkeypatch, partial
):
    import contextlib

    recorder, _ = numeric_modules

    @contextlib.contextmanager
    def capture(_dataset, captures):
        if partial:
            captures.append({"kernel": "partial", "inputs": {}, "output": 1})
        yield

    def oracle_failure(*_args):
        raise ValueError("original oracle failure")

    monkeypatch.setattr(recorder, "capture_context", capture)
    monkeypatch.setattr(recorder, "_invoke_original_entrypoint", oracle_failure)
    index, record, error = recorder._worker(("plane", 0, {"fixture": {}}))
    assert index == 0 and error == "ValueError: original oracle failure"
    assert record["exception"]["type"] == "ValueError"
    assert record["exception"]["message"] == "original oracle failure"
    assert "oracle_failure" in record["exception"]["traceback"]
    assert len(record["captured_calls_before_error"]) == int(partial)


def test_v4_failed_attempt_default_resume_and_explicit_retry(numeric_modules, tmp_path):
    recorder, _ = numeric_modules
    journal = _open_journal(recorder, tmp_path / "run")
    recorder.execute_attempts(
        journal, lambda i: (i,), worker=lambda p: _journal_failure(p[0])
    )
    assert journal.checkpoint["execution_state"] == "failed"
    assert journal.checkpoint["active_attempts"] == {}
    failure = journal.events[0]
    assert failure["schema"] == "gragra.gen1.refinement-trace-attempt-failure.v4"
    assert "success_commitment" not in failure and "record" not in failure
    assert failure["exception"]["message"] == "original worker error"
    resumed = _open_journal(recorder, journal.target, resume=True)
    called = []
    recorder.execute_attempts(
        resumed, lambda i: (i,), worker=lambda p: called.append(p)
    )
    assert called == [] and resumed.checkpoint["execution_state"] == "failed"
    recorder.execute_attempts(
        resumed,
        lambda i: (i,),
        retry_failed=True,
        worker=lambda p: _journal_success(p[0]),
    )
    assert resumed.checkpoint["execution_state"] == "complete"
    assert [e["source_row_index"] for e in resumed.events] == [0, 0, 1, 2]
    assert resumed.events[1]["retry_of"] == failure["attempt_id"]
    assert len({e["attempt_id"] for e in resumed.events}) == 4
    with pytest.raises(RuntimeError, match="accepted successful"):
        resumed.start(0)


@pytest.mark.parametrize("tail", [b'{"partial":', b'{"uncommitted":true}\n'])
def test_v4_journal_uncommitted_tail_isolated(numeric_modules, tmp_path, tail):
    recorder, _ = numeric_modules
    journal = _open_journal(recorder, tmp_path / "run")
    journal.finish(journal.start(0), _journal_success(0))
    prefix = journal.path.read_bytes()
    with journal.path.open("ab") as stream:
        stream.write(tail)
    resumed = _open_journal(recorder, journal.target, resume=True)
    assert resumed.path.read_bytes() == prefix
    entry = resumed.checkpoint["quarantined_uncommitted_tails"][-1]
    assert (resumed.target / entry["path"]).read_bytes() == tail
    assert len(resumed.events) == 1


@pytest.mark.parametrize(
    "corruption",
    [
        "changed",
        "deleted",
        "sequence",
        "attempt",
        "selection",
        "fixture",
        "manifest",
        "digest",
        "row_state",
    ],
)
def test_v4_journal_committed_corruption_rejects(numeric_modules, tmp_path, corruption):
    recorder, _ = numeric_modules
    journal = _open_journal(recorder, tmp_path / "run")
    journal.finish(journal.start(0), _journal_success(0))
    journal.finish(journal.start(1), _journal_success(1))
    rows, manifest = _journal_inputs()
    checkpoint = json.loads(journal.checkpoint_path.read_text())
    events = copy.deepcopy(journal.events)
    if corruption == "changed":
        events[0]["record"]["status"] = "forged"
    elif corruption == "deleted":
        events.pop()
    elif corruption == "sequence":
        events[1]["event_sequence"] = events[0]["event_sequence"]
    elif corruption == "attempt":
        events[1]["attempt_id"] = events[0]["attempt_id"]
    elif corruption == "selection":
        checkpoint["selected_source_row_indices"] = [0, 1]
    elif corruption == "fixture":
        rows[0]["fixture"]["id"] = "changed"
    elif corruption == "manifest":
        manifest["observer_sha256"] = "changed"
    elif corruption == "digest":
        checkpoint["event_digests"][0] = "changed"
    elif corruption == "row_state":
        checkpoint["accepted_source_row_indices"] = [0]
    journal.path.write_bytes(
        b"".join((recorder._canonical_json(e) + "\n").encode() for e in events)
    )
    journal.checkpoint_path.write_text(json.dumps(checkpoint))
    before = journal.path.read_bytes()
    with pytest.raises(RuntimeError):
        _open_journal(
            recorder, journal.target, resume=True, rows=rows, manifest=manifest
        )
    assert journal.path.read_bytes() == before


def test_v4_journal_mixed_checkpoint_unplanned_failure_rejects(
    numeric_modules, tmp_path
):
    recorder, _ = numeric_modules
    journal = _open_journal(recorder, tmp_path / "run")
    journal.finish(journal.start(0), _journal_failure(0))
    checkpoint = json.loads(journal.checkpoint_path.read_text())
    checkpoint["active_attempts"] = {"1": {"attempt_id": "unplanned", "retry_of": None}}
    checkpoint["execution_state"] = "running"
    journal.checkpoint_path.write_text(json.dumps(checkpoint))
    with pytest.raises(RuntimeError, match="unplanned.*failure|failure.*active"):
        _open_journal(recorder, journal.target, resume=True)


@pytest.mark.parametrize("boundary", ["append", "checkpoint"])
def test_v4_journal_crash_boundary_preserves_committed_prefix(
    numeric_modules, tmp_path, monkeypatch, boundary
):
    recorder, _ = numeric_modules
    journal = _open_journal(recorder, tmp_path / "run")
    attempt = journal.start(0)
    checkpoint_before = journal.checkpoint_path.read_bytes()
    original_append, original_atomic = recorder._append_jsonl, recorder._atomic_json

    def broken_append(path, _event):
        with path.open("ab") as stream:
            stream.write(b'{"broken":')
        raise OSError("append crash")

    def broken_checkpoint(path, value):
        if path == journal.checkpoint_path:
            raise OSError("checkpoint crash")
        original_atomic(path, value)

    monkeypatch.setattr(
        recorder,
        "_append_jsonl",
        broken_append if boundary == "append" else original_append,
    )
    monkeypatch.setattr(
        recorder,
        "_atomic_json",
        broken_checkpoint if boundary == "checkpoint" else original_atomic,
    )
    with pytest.raises(recorder.AttemptPersistenceError) as caught:
        journal.finish(attempt, _journal_failure(0))
    assert caught.value.event["exception"]["message"] == "original worker error"
    assert journal.checkpoint_path.read_bytes() == checkpoint_before
    monkeypatch.setattr(recorder, "_append_jsonl", original_append)
    monkeypatch.setattr(recorder, "_atomic_json", original_atomic)
    resumed = _open_journal(recorder, journal.target, resume=True)
    assert resumed.checkpoint["execution_state"] == "interrupted"
    assert resumed.events == []
    assert resumed.checkpoint["quarantined_uncommitted_tails"]


def test_v4_journal_fsync_precedes_checkpoint(numeric_modules, tmp_path, monkeypatch):
    recorder, _ = numeric_modules
    journal = _open_journal(recorder, tmp_path / "run")
    attempt = journal.start(0)
    order = []
    original_sync, original_atomic = recorder.os.fsync, recorder._atomic_json

    def sync(fd):
        order.append("fsync")
        return original_sync(fd)

    def atomic(path, value):
        if path == journal.checkpoint_path:
            order.append("checkpoint")
        return original_atomic(path, value)

    monkeypatch.setattr(recorder.os, "fsync", sync)
    monkeypatch.setattr(recorder, "_atomic_json", atomic)
    journal.finish(attempt, _journal_success(0))
    assert order.index("fsync") < order.index("checkpoint")


def test_v4_retry_plan_multiple_failures_is_sequential(numeric_modules, tmp_path):
    recorder, _ = numeric_modules
    journal = _open_journal(recorder, tmp_path / "run")
    # Already dispatched attempts may both fail; the retry plan must include both.
    first, second = journal.start(0), journal.start(1)
    journal.finish(first, _journal_failure(0))
    journal.finish(second, _journal_failure(1))
    journal.settle()
    seen = []

    def success(payload):
        seen.append((payload[0], copy.deepcopy(journal.checkpoint)))
        return _journal_success(payload[0])

    recorder.execute_attempts(
        journal, lambda i: (i,), workers=2, retry_failed=True, worker=success
    )
    assert [i for i, _ in seen] == [0, 1, 2]
    assert seen[0][1]["retry_plan"]["source_row_indices"] == [0, 1]
    assert len(seen[0][1]["active_attempts"]) == 1
    assert len(seen[1][1]["active_attempts"]) == 1
    assert journal.checkpoint["execution_state"] == "complete"


def test_v4_retry_interruption_requires_explicit_flag_again(numeric_modules, tmp_path):
    recorder, _ = numeric_modules
    journal = _open_journal(recorder, tmp_path / "run")
    journal.finish(journal.start(0), _journal_failure(0))
    journal.settle()

    def interrupt(_payload):
        raise KeyboardInterrupt("parent interrupted retry")

    with pytest.raises(KeyboardInterrupt):
        recorder.execute_attempts(
            journal, lambda i: (i,), retry_failed=True, worker=interrupt
        )
    assert journal.checkpoint["execution_state"] == "interrupted"
    assert journal.checkpoint["retry_plan"]["remaining_source_row_indices"] == [0]
    resumed = _open_journal(recorder, journal.target, resume=True)
    seen = []
    recorder.execute_attempts(resumed, lambda i: (i,), worker=lambda p: seen.append(p))
    assert seen == [] and resumed.checkpoint["execution_state"] == "interrupted"
    recorder.execute_attempts(
        resumed,
        lambda i: (i,),
        retry_failed=True,
        worker=lambda p: _journal_success(p[0]),
    )
    assert resumed.checkpoint["execution_state"] == "complete"


def test_v4_retry_terminal_failure_stops_phase(numeric_modules, tmp_path):
    recorder, _ = numeric_modules
    journal = _open_journal(recorder, tmp_path / "run")
    first, second = journal.start(0), journal.start(1)
    journal.finish(first, _journal_failure(0))
    journal.finish(second, _journal_failure(1))
    journal.settle()
    seen = []

    def fail(payload):
        seen.append(payload[0])
        return _journal_failure(payload[0])

    recorder.execute_attempts(journal, lambda i: (i,), retry_failed=True, worker=fail)
    assert seen == [0]
    assert journal.checkpoint["execution_state"] == "failed"
    assert not journal.checkpoint["active_attempts"]
    assert len(journal.events) == 3


def test_v4_journal_retry_final_event_crash_is_resumable(numeric_modules, tmp_path):
    recorder, _ = numeric_modules
    journal = _open_journal(recorder, tmp_path / "run")
    recorder.execute_attempts(
        journal, lambda i: (i,), worker=lambda p: _journal_failure(p[0])
    )
    journal.begin_retry()
    journal.finish(journal.start(0), _journal_success(0))
    resumed = _open_journal(recorder, journal.target, resume=True)
    assert resumed.checkpoint["retry_plan"]["remaining_source_row_indices"] == []
    seen = []
    recorder.execute_attempts(resumed, lambda i: (i,), worker=lambda p: seen.append(p))
    assert seen == []
    recorder.execute_attempts(
        resumed,
        lambda i: (i,),
        retry_failed=True,
        worker=lambda p: _journal_success(p[0]),
    )
    assert resumed.checkpoint["execution_state"] == "complete"


def test_v4_worker_failure_in_trace_builder_preserves_partial_calls(
    numeric_modules, monkeypatch
):
    import contextlib

    recorder, _ = numeric_modules

    @contextlib.contextmanager
    def capture(_dataset, captures):
        captures.append({"kernel": "partial", "inputs": {}, "output": 1})
        yield

    def builder_failure(*_args, **_kwargs):
        raise RuntimeError("trace builder failure")

    monkeypatch.setattr(recorder, "capture_context", capture)
    monkeypatch.setattr(
        recorder, "_invoke_original_entrypoint", lambda *_args: ({}, {})
    )
    monkeypatch.setattr(recorder, "_case_trace", builder_failure)
    _, record, _ = recorder._worker(("plane", 0, {"fixture": {}}))
    assert len(record["captured_calls_before_error"]) == 1
    assert "builder_failure" in record["exception"]["traceback"]


def test_v4_worker_failure_drains_inflight_before_stopping(
    numeric_modules, tmp_path, monkeypatch
):
    import concurrent.futures
    import threading

    recorder, _ = numeric_modules
    journal = _open_journal(recorder, tmp_path / "run")
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def worker(payload):
        index = payload[0]
        calls.append(index)
        if index == 0:
            assert started.wait(5)
            return _journal_failure(index)
        started.set()
        assert release.wait(5)
        finished.set()
        return _journal_success(index)

    original_finish = journal.finish

    def finish(attempt, result):
        event = original_finish(attempt, result)
        if result[0] == 0:
            assert len(journal.checkpoint["active_attempts"]) == 1
            assert journal.checkpoint["failure_drain_attempt_ids"]
            assert not finished.is_set()
            release.set()
        return event

    monkeypatch.setattr(journal, "finish", finish)
    monkeypatch.setattr(recorder, "_worker", worker)
    monkeypatch.setattr(
        recorder.concurrent.futures,
        "ProcessPoolExecutor",
        concurrent.futures.ThreadPoolExecutor,
    )
    recorder.execute_attempts(journal, lambda i: (i,), workers=2)
    assert finished.is_set()
    assert sorted(calls) == [0, 1]
    assert journal.checkpoint["execution_state"] == "failed"
    assert journal.checkpoint["active_attempts"] == {}
    assert journal.checkpoint["failure_drain_attempt_ids"] == []
    assert len(journal.events) == 2
    resumed = _open_journal(recorder, journal.target, resume=True)
    assert resumed.checkpoint["failed_source_row_indices"] == [0]


def test_v4_journal_duplicate_ids_reject_even_with_recomputed_local_hashes(
    numeric_modules, tmp_path
):
    recorder, _ = numeric_modules
    for field in ("attempt_id", "event_sequence"):
        journal = _open_journal(recorder, tmp_path / field)
        journal.finish(journal.start(0), _journal_success(0))
        journal.finish(journal.start(1), _journal_success(1))
        events = copy.deepcopy(journal.events)
        events[1][field] = events[0][field]
        events[1]["event_sha256"] = recorder._event_digest(events[1])
        data = b"".join((recorder._canonical_json(e) + "\n").encode() for e in events)
        journal.path.write_bytes(data)
        checkpoint = copy.deepcopy(journal.checkpoint)
        checkpoint["committed_journal_bytes"] = len(data)
        checkpoint["committed_journal_sha256"] = hashlib.sha256(data).hexdigest()
        checkpoint["event_digests"] = [e["event_sha256"] for e in events]
        journal.checkpoint_path.write_text(json.dumps(checkpoint))
        with pytest.raises(RuntimeError, match="duplicate or invalid journal"):
            _open_journal(recorder, journal.target, resume=True)


def test_v4_journal_final_retry_crash_with_all_rows_accepted(numeric_modules, tmp_path):
    recorder, _ = numeric_modules
    rows, manifest = _journal_inputs()
    rows = {0: rows[0]}
    manifest["selected_source_row_indices"] = [0]
    journal = _open_journal(recorder, tmp_path / "run", rows=rows, manifest=manifest)
    journal.finish(journal.start(0), _journal_failure(0))
    journal.begin_retry()
    journal.finish(journal.start(0), _journal_success(0))
    resumed = _open_journal(
        recorder, journal.target, resume=True, rows=rows, manifest=manifest
    )
    recorder.execute_attempts(
        resumed,
        lambda i: (i,),
        retry_failed=True,
        worker=lambda _p: pytest.fail("already accepted"),
    )
    assert resumed.checkpoint["execution_state"] == "complete"


def test_v4_journal_stale_writer_rejects(numeric_modules, tmp_path):
    recorder, _ = numeric_modules
    journal = _open_journal(recorder, tmp_path / "run")
    other = _open_journal(recorder, journal.target, resume=True)
    other.start(0)
    with pytest.raises(RuntimeError, match="another writer"):
        journal.start(1)


def _journal_dataset_files(recorder, tmp_path, monkeypatch):
    row, _ = _v4_plane_record(recorder)
    root = tmp_path / "frozen"
    directory = root / recorder.DATASETS["plane"]
    directory.mkdir(parents=True)
    reference = directory / "references.jsonl"
    reference.write_text(json.dumps(row) + "\n")
    (directory / "manifest.json").write_text('{"fixture":"local wiring test"}\n')
    monkeypatch.setitem(
        recorder.EXPECTED_REFERENCE_SHA256, "plane", recorder._sha256(reference)
    )
    monkeypatch.setattr(
        recorder,
        "_load_zero_certificates",
        lambda _root: {
            "path": "test-only",
            "sha256": None,
            "data": {"classification": "test-only", "certificates": []},
        },
    )
    monkeypatch.setattr(
        recorder, "_load_polynomial_quadrature_certificates", lambda *_args: None
    )
    return root


@pytest.mark.parametrize("workers", [1, 2])
def test_v4_journal_dataset_api_fixed_plane_wiring(
    numeric_modules, tmp_path, monkeypatch, workers
):
    recorder, _ = numeric_modules
    root = _journal_dataset_files(recorder, tmp_path, monkeypatch)
    result = recorder.run_dataset(
        "plane", root, tmp_path / "output", "fixed-run", workers=workers
    )
    assert result["execution_state"] == "complete"
    assert result["accepted_source_row_indices"] == [0]
    monkeypatch.setattr(
        recorder, "_worker", lambda _p: pytest.fail("completed row was rerun")
    )
    resumed = recorder.run_dataset(
        "plane", root, tmp_path / "output", "fixed-run", resume=True
    )
    assert resumed["execution_state"] == "complete"


def test_v4_retry_flag_is_explicit_in_recorder_parser(numeric_modules):
    recorder, _ = numeric_modules
    args = recorder._argument_parser().parse_args(
        [
            "--l6-root",
            "/frozen",
            "--output-root",
            "/out",
            "--run-id",
            "new",
            "--datasets",
            "plane",
            "--resume",
            "--retry-failed",
        ]
    )
    assert args.retry_failed is True and args.resume is True


def test_v4_worker_failure_escaping_adapter_is_committed(numeric_modules, tmp_path):
    recorder, _ = numeric_modules
    journal = _open_journal(recorder, tmp_path / "run")

    def worker(_payload):
        raise RuntimeError("escaping worker failure")

    recorder.execute_attempts(journal, lambda i: (i,), worker=worker)
    assert journal.checkpoint["execution_state"] == "failed"
    assert journal.checkpoint["active_attempts"] == {}
    assert journal.events[0]["exception"]["message"] == "escaping worker failure"
    assert "worker" in journal.events[0]["exception"]["traceback"]


def test_v4_journal_finish_refuses_to_commit_external_tail(numeric_modules, tmp_path):
    recorder, _ = numeric_modules
    journal = _open_journal(recorder, tmp_path / "run")
    attempt = journal.start(0)
    with journal.path.open("ab") as stream:
        stream.write(b'{"uncommitted":true}\n')
    with pytest.raises(RuntimeError, match="journal.*changed|uncommitted"):
        journal.finish(attempt, _journal_success(0))
    assert journal.checkpoint["event_sequence"] == 0


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("active_attempts", {"bad-row": {"attempt_id": "bad", "retry_of": None}}),
        ("active_attempts", {"0": "bad"}),
        ("retry_plan", []),
        ("interrupted_attempts", [None]),
        ("quarantined_uncommitted_tails", [None]),
        ("retry_plan_history", None),
    ],
)
def test_v4_journal_malformed_checkpoint_rejects_cleanly(
    numeric_modules, tmp_path, field, bad_value
):
    recorder, _ = numeric_modules
    journal = _open_journal(recorder, tmp_path / "run")
    checkpoint = copy.deepcopy(journal.checkpoint)
    checkpoint[field] = bad_value
    journal.checkpoint_path.write_text(json.dumps(checkpoint))
    with pytest.raises(RuntimeError):
        _open_journal(recorder, journal.target, resume=True)


def test_v4_journal_partial_retry_plan_rejected(numeric_modules, tmp_path):
    recorder, _ = numeric_modules
    journal = _open_journal(recorder, tmp_path / "run")
    first, second = journal.start(0), journal.start(1)
    journal.finish(first, _journal_failure(0))
    journal.finish(second, _journal_failure(1))
    journal.settle()
    journal.begin_retry()
    checkpoint = copy.deepcopy(journal.checkpoint)
    checkpoint["retry_plan"]["source_row_indices"] = [0]
    checkpoint["retry_plan"]["remaining_source_row_indices"] = [0]
    journal.checkpoint_path.write_text(json.dumps(checkpoint))
    with pytest.raises(RuntimeError, match="every unresolved failure"):
        _open_journal(recorder, journal.target, resume=True)


def test_v4_journal_resume_cannot_interrupt_live_parent(numeric_modules, tmp_path):
    import concurrent.futures
    import threading

    recorder, _ = numeric_modules
    journal = _open_journal(recorder, tmp_path / "run")
    started, release = threading.Event(), threading.Event()

    def worker(payload):
        started.set()
        assert release.wait(5)
        return _journal_failure(payload[0])

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            recorder.execute_attempts, journal, lambda i: (i,), worker=worker
        )
        assert started.wait(5)
        before = journal.checkpoint_path.read_bytes()
        try:
            with pytest.raises(RuntimeError, match="writer|worker phase"):
                _open_journal(recorder, journal.target, resume=True)
            assert journal.checkpoint_path.read_bytes() == before
        finally:
            release.set()
        future.result(timeout=5)
    assert journal.checkpoint["execution_state"] == "failed"


def test_v4_journal_attempt_id_collision_rejected_before_dispatch(
    numeric_modules, tmp_path, monkeypatch
):
    import types

    recorder, _ = numeric_modules
    journal = _open_journal(recorder, tmp_path / "run")
    monkeypatch.setattr(
        recorder.uuid, "uuid4", lambda: types.SimpleNamespace(hex="fixed-attempt")
    )
    journal.finish(journal.start(0), _journal_success(0))
    before = journal.checkpoint_path.read_bytes()
    with pytest.raises(RuntimeError, match="attempt ID"):
        journal.start(1)
    assert journal.checkpoint_path.read_bytes() == before


@pytest.mark.parametrize("field", ["fixture", "reference", "metadata"])
@pytest.mark.parametrize("boundary", ["finish", "resume"])
def test_v4_journal_success_binds_selected_frozen_payload(
    numeric_modules, tmp_path, field, boundary
):
    recorder, _ = numeric_modules
    rows, manifest = _journal_inputs()
    rows = {0: rows[0]}
    manifest["selected_source_row_indices"] = [0]
    journal = _open_journal(recorder, tmp_path / "run", rows=rows, manifest=manifest)
    result = _journal_success(0)
    if boundary == "resume":
        for index in journal.indices:
            journal.finish(journal.start(index), _journal_success(index))
        journal.settle()
        record = journal.events[0]["record"]
    else:
        attempt = journal.start(0)
        record = result[1]
    if field == "fixture":
        record[field] = {"id": "forged"}
    else:
        record["frozen_reference"][field] = {"forged": True}
    if boundary == "finish":
        before = journal.path.read_bytes(), journal.checkpoint_path.read_bytes()
        with pytest.raises(RuntimeError, match="fixture|frozen payload"):
            journal.finish(attempt, result)
        assert (
            journal.path.read_bytes(),
            journal.checkpoint_path.read_bytes(),
        ) == before
    else:
        journal.events[0]["event_sha256"] = recorder._event_digest(journal.events[0])
        data = b"".join(
            (recorder._canonical_json(event) + "\n").encode()
            for event in journal.events
        )
        journal.path.write_bytes(data)
        checkpoint = copy.deepcopy(journal.checkpoint)
        checkpoint["committed_journal_bytes"] = len(data)
        checkpoint["committed_journal_sha256"] = hashlib.sha256(data).hexdigest()
        checkpoint["event_digests"] = [e["event_sha256"] for e in journal.events]
        journal.checkpoint_path.write_text(json.dumps(checkpoint))
        with pytest.raises(RuntimeError, match="fixture|frozen payload"):
            _open_journal(
                recorder, journal.target, resume=True, rows=rows, manifest=manifest
            )


def _journal_hard_exit(_payload):
    import os

    os._exit(23)


def test_v4_process_pool_submission_failure_commits_and_drains(
    numeric_modules, tmp_path, monkeypatch
):
    import concurrent.futures
    import time

    recorder, _ = numeric_modules
    journal = _open_journal(recorder, tmp_path / "run")
    real_pool = concurrent.futures.ProcessPoolExecutor
    pools, dispatched = [], []

    def capture_pool(*args, **kwargs):
        pool = real_pool(*args, **kwargs)
        pools.append(pool)
        return pool

    def payload(index):
        dispatched.append(index)
        if index == 1:
            deadline = time.monotonic() + 5
            while not pools[0]._broken and time.monotonic() < deadline:
                time.sleep(0.01)
            assert pools[0]._broken, "hard exit must break the real pool before submit"
        return (index,)

    monkeypatch.setattr(recorder, "_worker", _journal_hard_exit)
    monkeypatch.setattr(
        recorder.concurrent.futures, "ProcessPoolExecutor", capture_pool
    )
    recorder.execute_attempts(journal, payload, workers=2)
    assert dispatched == [0, 1]
    assert journal.checkpoint["execution_state"] == "failed"
    assert journal.checkpoint["active_attempts"] == {}
    assert journal.checkpoint["interrupted_attempts"] == []
    assert journal.checkpoint["failure_drain_attempt_ids"] == []
    assert sorted(event["source_row_index"] for event in journal.events) == [0, 1]
    assert all(
        event["schema"] == recorder.ATTEMPT_FAILURE_SCHEMA for event in journal.events
    )
    assert all(
        event["exception"]["type"] == "BrokenProcessPool" for event in journal.events
    )
    assert all(event["exception"]["traceback"] for event in journal.events)
    resumed = _open_journal(recorder, journal.target, resume=True)
    called = []
    recorder.execute_attempts(
        resumed, lambda i: (i,), worker=lambda p: called.append(p)
    )
    assert called == []
    recorder.execute_attempts(
        resumed,
        lambda i: (i,),
        retry_failed=True,
        worker=lambda p: _journal_success(p[0]),
    )
    assert resumed.checkpoint["execution_state"] == "complete"
    assert [e["source_row_index"] for e in resumed.events[2:]] == [0, 1, 2]


@pytest.mark.parametrize("completed_retries", [0, 1])
def test_v4_retry_crash_without_active_attempt_remains_interrupted(
    numeric_modules, tmp_path, completed_retries
):
    recorder, _ = numeric_modules
    journal = _open_journal(recorder, tmp_path / "run")
    first, second = journal.start(0), journal.start(1)
    journal.finish(first, _journal_failure(0))
    journal.finish(second, _journal_failure(1))
    journal.settle()
    journal.begin_retry()
    if completed_retries:
        journal.finish(journal.start(0), _journal_success(0))
    plan = copy.deepcopy(journal.checkpoint["retry_plan"])
    resumed = _open_journal(recorder, journal.target, resume=True)
    assert resumed.checkpoint["execution_state"] == "interrupted"
    assert resumed.checkpoint["retry_plan"] == plan
    called = []
    recorder.execute_attempts(
        resumed, lambda i: (i,), worker=lambda p: called.append(p)
    )
    assert called == []
    assert resumed.checkpoint["execution_state"] == "interrupted"
    recorder.execute_attempts(
        resumed,
        lambda i: (i,),
        retry_failed=True,
        worker=lambda p: _journal_success(p[0]),
    )
    assert resumed.checkpoint["execution_state"] == "complete"


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("layer", ["entrypoint", "record_row"])
def test_v4_real_worker_interruption_preserves_retry_plan(
    numeric_modules, tmp_path, monkeypatch, interruption, layer
):
    import contextlib

    recorder, _ = numeric_modules
    journal = _open_journal(recorder, tmp_path / "run")
    journal.finish(journal.start(0), _journal_failure(0))
    journal.settle()

    def interrupt(*_args, **_kwargs):
        raise interruption("parent interrupted retry")

    monkeypatch.setattr(
        recorder, "capture_context", lambda *_args: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        recorder,
        "_invoke_original_entrypoint" if layer == "entrypoint" else "record_row",
        interrupt,
    )
    with pytest.raises(interruption, match="parent interrupted retry"):
        recorder.execute_attempts(
            journal, lambda i: ("plane", i, journal.rows[i]), retry_failed=True
        )
    assert len(journal.events) == 1
    assert journal.checkpoint["execution_state"] == "interrupted"
    assert journal.checkpoint["active_attempts"] == {}
    assert journal.checkpoint["retry_plan"]["remaining_source_row_indices"] == [0]
    assert len(journal.checkpoint["interrupted_attempts"]) == 1
    plan = copy.deepcopy(journal.checkpoint["retry_plan"])
    resumed = _open_journal(recorder, journal.target, resume=True)
    assert resumed.checkpoint["retry_plan"] == plan
    called = []
    recorder.execute_attempts(
        resumed, lambda i: (i,), worker=lambda p: called.append(p)
    )
    assert called == []
    recorder.execute_attempts(
        resumed,
        lambda i: (i,),
        retry_failed=True,
        worker=lambda p: _journal_success(p[0]),
    )
    assert resumed.checkpoint["execution_state"] == "complete"


@pytest.fixture(scope="module")
def audited_plane(numeric_modules):
    recorder, auditor = numeric_modules
    row, record = _v4_plane_record(recorder)
    record["zero_certificate_supplement_sha256"] = auditor.ZERO_SHA256
    return row, record


@pytest.mark.parametrize(
    ("mutation", "recorder_reason", "auditor_reason"),
    [
        ("missing_observable", "frozen observable set", "observable coverage"),
        ("renamed_observable", "frozen observable set", "observable coverage"),
        ("empty_observable", "trace status", "expected numerical trace"),
        (
            "missing_component_slot",
            "component-check slots",
            "MP component slots incomplete",
        ),
        ("unjustified_na", "not_applicable", "N/A trend mismatch"),
        ("stale_callback_trend", "trend", "trend/named pair mismatch"),
        ("status_mismatch", "status", "derived record status mismatch"),
        ("reproduction_flag", "reproduction result", "reproduction flags mismatch"),
        ("reference_drift", "value-match flag", "reproduction mismatch"),
        ("capture_count", "capture count", "capture count mismatch"),
        (
            "short_final_vector",
            "incomplete acceleration values",
            "captured final shape mismatch",
        ),
        (
            "captured_value_flag",
            "captured-value flag",
            "final frozen-value flag mismatch",
        ),
        (
            "component_match_flag",
            "component flag",
            "original component reconstruction failed",
        ),
    ],
)
def test_v4_task_a_semantic_negative_paths_rejected_by_recorder_and_auditor(
    numeric_modules,
    audited_plane,
    mutation,
    recorder_reason,
    auditor_reason,
):
    recorder, auditor = numeric_modules
    row, baseline = audited_plane
    recorder._validate_trace_record(
        baseline,
        "plane",
        0,
        row,
        zero_certificate_sha256=auditor.ZERO_SHA256,
    )
    assert (
        auditor.audit_record("plane", 0, row, baseline)["semantic_validation"]
        == "SEMANTIC_VALID"
    )

    record = copy.deepcopy(baseline)
    traces = record["observable_traces"]
    acceleration = traces["acceleration"]
    if mutation == "missing_observable":
        del traces["gradient"]
    elif mutation == "renamed_observable":
        traces["grad"] = traces.pop("gradient")
    elif mutation == "empty_observable":
        traces["gradient"] = {}
    elif mutation == "missing_component_slot":
        del acceleration["frozen_component_checks"]["derivative"]
    elif mutation == "unjustified_na":
        acceleration["derivative_trend"] = {
            "classification": "not_applicable",
            "reason": "caller supplied explanation",
        }
    elif mutation == "stale_callback_trend":
        acceleration["derivative_trend"]["coarse_to_middle_si"] = "0"
    elif mutation == "status_mismatch":
        record["status"] = "replay_mismatch"
    elif mutation == "reproduction_flag":
        record["reproduction"]["passed"] = False
    elif mutation == "reference_drift":
        record["original_entrypoint"]["actual_reference"]["potential"]["values_si"][
            0
        ] = "123.0"
    elif mutation == "capture_count":
        record["original_entrypoint"]["capture_call_count"] += 1
    elif mutation == "short_final_vector":
        acceleration["captured_final_values_si"] = acceleration[
            "captured_final_values_si"
        ][:2]
    elif mutation == "captured_value_flag":
        acceleration["captured_final_values_match_frozen"] = not acceleration[
            "captured_final_values_match_frozen"
        ]
    elif mutation == "component_match_flag":
        check = acceleration["frozen_component_checks"]["derivative"]
        check["matches_frozen_float"] = not check["matches_frozen_float"]

    with pytest.raises(ValueError, match=recorder_reason):
        recorder._validate_trace_record(
            record,
            "plane",
            0,
            row,
            zero_certificate_sha256=auditor.ZERO_SHA256,
        )
    with pytest.raises(ValueError, match=auditor_reason):
        auditor.audit_record("plane", 0, row, record)
    result = auditor.audit_saved_record("plane", 0, row, record)
    assert result["semantic_validation"] == "SEMANTIC_INVALID"
    assert result["scientific_admission"] == "NOT_ADMITTED"


def test_v4_duplicate_observable_json_key_rejected_by_both_decoders(numeric_modules):
    recorder, auditor = numeric_modules
    raw = b'{"observable_traces":{"gradient":{},"gradient":{}}}\n'

    with pytest.raises((ValueError, RuntimeError), match="duplicate"):
        recorder._decode_trace_bytes(raw, description="duplicate observable probe")
    with pytest.raises(ValueError, match="duplicate"):
        auditor.strict_json_loads(
            raw.decode("utf-8"), description="duplicate observable probe"
        )


def test_v4_digest_rehashed_semantic_invalid_row_is_not_admitted_after_resume(
    numeric_modules, audited_plane, tmp_path
):
    recorder, auditor = numeric_modules
    row, valid_record = audited_plane
    selected_rows = {0: row}
    manifest = {
        "schema": "gragra.gen1.refinement-trace-dataset-manifest.v4",
        "run_id": "semantic-resume-probe",
        "dataset": "plane",
        "selected_source_row_indices": [0],
        "observer_sha256": "observer-pin",
        "frozen_reference_sha256": "reference-pin",
    }
    journal = recorder.AttemptJournal(
        tmp_path / "semantic-resume-probe",
        run_id="semantic-resume-probe",
        dataset="plane",
        selected_rows=selected_rows,
        manifest=manifest,
    )
    journal.finish(journal.start(0), (0, copy.deepcopy(valid_record), None))

    event = copy.deepcopy(journal.events[0])
    del event["record"]["observable_traces"]["gradient"]
    event["event_sha256"] = recorder._event_digest(event)
    wire = (recorder._canonical_json(event) + "\n").encode("utf-8")
    journal.path.write_bytes(wire)
    checkpoint = copy.deepcopy(journal.checkpoint)
    checkpoint["committed_journal_bytes"] = len(wire)
    checkpoint["committed_journal_sha256"] = hashlib.sha256(wire).hexdigest()
    checkpoint["event_digests"] = [event["event_sha256"]]
    journal.checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    resumed = recorder.AttemptJournal(
        journal.target,
        run_id=manifest["run_id"],
        dataset="plane",
        selected_rows=selected_rows,
        manifest=manifest,
        resume=True,
    )
    assert resumed.checkpoint["execution_state"] == "complete"
    dispatched = []
    recorder.execute_attempts(
        resumed,
        lambda index: (index,),
        worker=lambda payload: dispatched.append(payload),
    )
    assert dispatched == []

    result = auditor.audit_saved_record(
        "plane",
        0,
        row,
        resumed.events[0]["record"],
        execution_state="complete",
        required_evidence_complete=True,
    )
    assert result["semantic_validation"] == "SEMANTIC_INVALID"
    assert result["scientific_admission"] == "NOT_ADMITTED"


def test_v4_coordinated_callback_mutation_resume_is_not_scientifically_admitted(
    numeric_modules, audited_plane, tmp_path
):
    recorder, auditor = numeric_modules
    row, valid_record = audited_plane
    selected_rows = {0: row}
    run_id = "test-api-run"
    manifest = {
        "schema": "gragra.gen1.refinement-trace-dataset-manifest.v4",
        "run_id": run_id,
        "dataset": "plane",
        "selected_source_row_indices": [0],
        "observer_sha256": "observer-pin",
        "frozen_reference_sha256": "reference-pin",
    }
    journal = recorder.AttemptJournal(
        tmp_path / run_id,
        run_id=run_id,
        dataset="plane",
        selected_rows=selected_rows,
        manifest=manifest,
    )
    journal.finish(journal.start(0), (0, copy.deepcopy(valid_record), None))

    original_event = copy.deepcopy(journal.events[0])
    event = copy.deepcopy(journal.events[0])
    _v4_coordinated_mutation(recorder, event["record"], row)
    event["event_sha256"] = recorder._event_digest(event)
    wire = (recorder._canonical_json(event) + "\n").encode("utf-8")
    journal.path.write_bytes(wire)
    checkpoint = copy.deepcopy(journal.checkpoint)
    checkpoint["committed_journal_bytes"] = len(wire)
    checkpoint["committed_journal_sha256"] = hashlib.sha256(wire).hexdigest()
    checkpoint["event_digests"] = [event["event_sha256"]]
    journal.checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    resumed = recorder.AttemptJournal(
        journal.target,
        run_id=run_id,
        dataset="plane",
        selected_rows=selected_rows,
        manifest=manifest,
        resume=True,
    )
    assert resumed.checkpoint["execution_state"] == "complete"
    dispatched = []
    recorder.execute_attempts(
        resumed,
        lambda index: (index,),
        worker=lambda payload: dispatched.append(payload),
    )
    assert dispatched == []

    record = resumed.events[0]["record"]

    def audited_journal(saved_event, saved_record):
        attempt_id = saved_event["attempt_id"]
        return auditor.PROVENANCE.AttemptJournalInputs(
            events=[saved_event],
            attempt_states={
                attempt_id: {
                    "state": "complete",
                    "source_row_index": 0,
                    "retry_of": None,
                }
            },
            retry_plan=[],
            accepted_results={
                "plane:0": {
                    "attempt_id": attempt_id,
                    "result_sha256": auditor.PROVENANCE.canonical_sha256(saved_record),
                }
            },
        )

    def audit_binding(saved_record, saved_event):
        binding = _audit_binding(auditor, row, saved_record)
        profile = binding["run_profile"]
        binding["run_profile"] = auditor.PROVENANCE.TrustedRunProfile(
            run_id=run_id,
            observer_sha256=profile.observer_sha256,
            auditor_sha256=profile.auditor_sha256,
            provenance_sha256=profile.provenance_sha256,
            runner_sha256=profile.runner_sha256,
            environment=profile.environment,
            numeric_representation=profile.numeric_representation,
        )
        binding["attempt_journal"] = audited_journal(saved_event, saved_record)
        return binding

    changed_binding = audit_binding(record, resumed.events[0])
    result = auditor.audit_record(
        "plane",
        0,
        row,
        record,
        **changed_binding,
        execution_state="complete",
        required_evidence_complete=True,
    )
    assert result["semantic_validation"] == "SEMANTIC_VALID"
    assert result["provenance_validation"]["status"] == "PROVENANCE_UNVERIFIED"
    assert result["scientific_admission"] == "NOT_ADMITTED"

    original_binding = audit_binding(valid_record, original_event)
    original_anchor = _audit_anchor(auditor, valid_record, original_binding)
    anchored_result = auditor.audit_record(
        "plane",
        0,
        row,
        record,
        **changed_binding,
        **_audit_mock_trust(auditor, original_anchor),
    )
    assert anchored_result["semantic_validation"] == "SEMANTIC_VALID"
    assert anchored_result["provenance_validation"]["status"] == "PROVENANCE_MISMATCH"
    assert anchored_result["scientific_admission"] == "NOT_ADMITTED"


def _audit_binding(auditor, row, record, *, failure=False):
    """Only test API inputs: these synthetic pins establish no real trust."""
    c = auditor.PROVENANCE
    dataset, index = record["dataset"], record["source_row_index"]
    selected = {index: row}
    independent = c.IndependentAuditInputs(
        dataset=dataset,
        dataset_sha256="a" * 64,
        source_row_index=index,
        fixture=row["fixture"],
        reference=row["reference"],
        required_observables=tuple(auditor.SECTIONS),
        oracle_module_sha256=digest(auditor.EXPECTED_ORACLE_SHA256),
        kernel_identities=dict.fromkeys(auditor.SECTIONS, "b" * 64),
        frozen_reference_sha256=auditor.EXPECTED_REFERENCE_SHA256[dataset],
        certificate_sha256=(auditor.ZERO_SHA256,),
        root_supplement=None,
    )
    profile = c.TrustedRunProfile(
        run_id="test-api-run",
        observer_sha256="1" * 64,
        auditor_sha256="2" * 64,
        provenance_sha256="3" * 64,
        runner_sha256="4" * 64,
        environment={"kind": "test-only"},
        numeric_representation={"format": "mpf-tuple", "digits": 80},
    )
    events, position = [], 0
    for failed in [True, False] if failure else [False]:
        event = {
            "schema": auditor.ATTEMPT_FAILURE_SCHEMA
            if failed
            else auditor.ATTEMPT_SUCCESS_SCHEMA,
            "run_id": profile.run_id,
            "dataset": dataset,
            "source_row_index": index,
            "attempt_id": "failed" if failed else "accepted",
            "retry_of": "failed" if failure and not failed else None,
            "event_sequence": len(events) + 1,
            "selected_row_sha256": digest(row),
            "last_durable_trace_offset": position,
            "process_independence": _v4_worker_evidence(),
        }
        if failed:
            event.update(
                exception={
                    "type": "RuntimeError",
                    "message": "test",
                    "traceback": "test original traceback",
                },
                captured_calls_before_error=[],
            )
        else:
            event["record"] = copy.deepcopy(record)
        event["event_sha256"] = hashlib.sha256(
            json.dumps(
                event, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        position += len(
            (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode()
        )
        events.append(event)
    journal = c.AttemptJournalInputs(
        events=events,
        attempt_states={
            e["attempt_id"]: {
                "state": "failed"
                if e["schema"] == auditor.ATTEMPT_FAILURE_SCHEMA
                else "complete",
                "source_row_index": index,
                "retry_of": e["retry_of"],
            }
            for e in events
        },
        retry_plan=[
            {
                "plan_id": "test-plan",
                "source_row_indices": [index],
                "remaining_source_row_indices": [],
                "execution_state": "complete",
            }
        ]
        if failure
        else [],
        accepted_results={
            f"{dataset}:{index}": {
                "attempt_id": "accepted",
                "result_sha256": digest(record),
            }
        },
    )
    return {
        "independent_inputs": independent,
        "run_profile": profile,
        "attempt_journal": journal,
        "selected_rows": selected,
    }


def _audit_anchor(auditor, record, binding):
    c = auditor.PROVENANCE
    expected = c.derive_expected_row_identity(
        binding["independent_inputs"],
        binding["run_profile"],
        callbacks=record["original_entrypoint"]["captured_calls"],
        attempt_journal=binding["attempt_journal"],
    )
    return c.commitment_document(expected)


def _audit_mock_trust(auditor, doc, *, authenticated=True, root_id="unit-test-root"):
    c = auditor.PROVENANCE
    return {
        "provenance_provider": MockProvider(
            {
                "document": doc,
                "status": "PROVENANCE_VERIFIED",
                "trust_root_verified": True,
            }
        ),
        "verifier_profile": c.TrustedVerifierProfile(
            "unit-test-root",
            MockVerifier(c, authenticated=authenticated, root_id=root_id),
        ),
    }


def test_v4_auditor_provenance_matching_mock_is_api_positive_only(
    numeric_modules, audited_plane
):
    _, auditor = numeric_modules
    row, record = audited_plane
    binding = _audit_binding(auditor, row, record, failure=True)
    trust = _audit_mock_trust(auditor, _audit_anchor(auditor, record, binding))
    result = auditor.audit_record("plane", 0, row, record, **binding, **trust)
    assert result["provenance_validation"]["status"] == "PROVENANCE_VERIFIED"
    assert result["journal_binding_complete"] is True
    assert result["scientific_admission"] == "ADMITTED_WITH_DECLARED_SCOPE"


@pytest.mark.parametrize(
    "field",
    [
        "dataset",
        "dataset_sha256",
        "source_row_index",
        "fixture_sha256",
        "reference_values_sha256",
        "required_observables",
        "oracle_module_sha256",
        "kernel_identities",
        "frozen_reference_sha256",
        "certificate_sha256",
        "root_supplement",
        "run_id",
        "observer_sha256",
        "auditor_sha256",
        "provenance_sha256",
        "runner_sha256",
        "environment",
        "numeric_representation",
        "callback_count",
        "callbacks_sha256",
        "attempt_journal_sha256",
    ],
)
def test_v4_auditor_provenance_every_authenticated_a6_field_mismatch(
    numeric_modules, audited_plane, field
):
    _, auditor = numeric_modules
    row, record = audited_plane
    binding = _audit_binding(auditor, row, record)
    doc = _audit_anchor(auditor, record, binding)
    payload = doc["payload"]
    if field == "required_observables":
        payload[field] = ["acceleration", "gradient"]
        payload["kernel_identities"].pop("potential")
    elif field == "kernel_identities":
        payload[field]["potential"] = "c" * 64
    elif field == "root_supplement":
        payload[field] = {
            "schema": auditor.PROVENANCE.ROOT_SUPPLEMENT_SCHEMA,
            "sha256": "c" * 64,
            "scope_sha256": "d" * 64,
        }
    elif isinstance(payload[field], dict):
        payload[field] = {"changed": True}
    elif isinstance(payload[field], list):
        payload[field] = ["c" * 64]
    elif isinstance(payload[field], int):
        payload[field] += 1
    else:
        payload[field] = "c" * 64 if field.endswith("sha256") else "changed"
    doc = auditor.PROVENANCE.commitment_document(payload)
    result = auditor.audit_record(
        "plane", 0, row, record, **binding, **_audit_mock_trust(auditor, doc)
    )
    assert result["provenance_validation"]["status"] == "PROVENANCE_MISMATCH"
    assert result["scientific_admission"] == "NOT_ADMITTED"


@pytest.mark.parametrize(
    "missing",
    [
        "provider",
        "verifier",
        "run_profile",
        "anchor",
        "wrong_root",
        "provider_assertion",
    ],
)
def test_v4_auditor_provenance_untrusted_material_never_authenticates(
    numeric_modules, audited_plane, missing
):
    _, auditor = numeric_modules
    row, record = audited_plane
    binding = _audit_binding(auditor, row, record)
    doc = _audit_anchor(auditor, record, binding)
    trust = _audit_mock_trust(auditor, doc)
    if missing == "provider":
        trust["provenance_provider"] = None
    elif missing == "verifier":
        trust["verifier_profile"] = None
    elif missing == "run_profile":
        binding["run_profile"] = None
    elif missing == "wrong_root":
        trust = _audit_mock_trust(auditor, doc, root_id="wrong-root")
    elif missing == "anchor":
        trust = _audit_mock_trust(auditor, doc, authenticated=False)
    else:
        trust["verifier_profile"] = None
    result = auditor.audit_record("plane", 0, row, record, **binding, **trust)
    assert result["provenance_validation"]["status"] == "PROVENANCE_UNVERIFIED"
    assert result["scientific_admission"] == (
        "INCOMPLETE_EVIDENCE" if missing == "run_profile" else "NOT_ADMITTED"
    )


@pytest.mark.parametrize("change", ["schema", "digest", "missing_payload"])
def test_v4_auditor_provenance_authenticated_malformed_anchor(
    numeric_modules, audited_plane, change
):
    _, auditor = numeric_modules
    row, record = audited_plane
    binding = _audit_binding(auditor, row, record)
    doc = _audit_anchor(auditor, record, binding)
    if change == "schema":
        doc["schema"] = "v3"
    elif change == "digest":
        doc["commitment_sha256"] = "0" * 64
    else:
        del doc["payload"]
    result = auditor.audit_record(
        "plane", 0, row, record, **binding, **_audit_mock_trust(auditor, doc)
    )
    assert result["provenance_validation"]["status"] == "PROVENANCE_MISMATCH"
    assert result["scientific_admission"] == "NOT_ADMITTED"


@pytest.mark.parametrize(
    "change",
    [
        "order",
        "duplicate_attempt",
        "row_pin",
        "retry",
        "failure_shape",
        "accepted_digest",
        "missing_state",
        "scope",
        "success_record",
        "missing_retry_plan",
        "event_digest",
        "durable_offset",
        "pending",
    ],
)
def test_v4_auditor_provenance_journal_requires_independent_reconciliation(
    numeric_modules, audited_plane, change
):
    _, auditor = numeric_modules
    row, record = audited_plane
    binding = _audit_binding(auditor, row, record, failure=True)
    journal = binding["attempt_journal"]
    if change == "order":
        journal.events.reverse()
    elif change == "duplicate_attempt":
        journal.events[1]["attempt_id"] = "failed"
    elif change == "row_pin":
        journal.events[0]["selected_row_sha256"] = "0" * 64
    elif change == "retry":
        journal.events[1]["retry_of"] = None
    elif change == "failure_shape":
        journal.events[0]["record"] = copy.deepcopy(record)
    elif change == "accepted_digest":
        journal.accepted_results["plane:0"]["result_sha256"] = "0" * 64
    elif change == "missing_state":
        del journal.attempt_states["failed"]
    elif change == "scope":
        binding["selected_rows"] = {0: row, 1: row}
    elif change == "success_record":
        journal.events[1]["record"]["fixture"] = {"changed": True}
    elif change == "missing_retry_plan":
        journal.retry_plan.clear()
    elif change == "event_digest":
        journal.events[1]["event_sha256"] = "0" * 64
    elif change == "durable_offset":
        journal.events[1]["last_durable_trace_offset"] += 1
    else:
        journal.attempt_states["pending"] = {
            "state": "pending",
            "source_row_index": 0,
            "retry_of": None,
        }
    if change not in {"event_digest", "durable_offset", "order"}:
        offset = 0
        for event in journal.events:
            event["last_durable_trace_offset"] = offset
            event["event_sha256"] = hashlib.sha256(
                json.dumps(
                    {k: v for k, v in event.items() if k != "event_sha256"},
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            offset += len(
                (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode()
            )
    # A matching test anchor cannot turn malformed/incomplete history into verification.
    doc = _audit_anchor(auditor, record, binding)
    result = auditor.audit_record(
        "plane", 0, row, record, **binding, **_audit_mock_trust(auditor, doc)
    )
    assert result["provenance_validation"]["status"] != "PROVENANCE_VERIFIED"
    assert result["journal_binding_complete"] is False
    assert result["scientific_admission"] != "ADMITTED_WITH_DECLARED_SCOPE"


def test_v4_auditor_provenance_known_invalidity_precedes_missing_evidence(
    numeric_modules, audited_plane
):
    _, auditor = numeric_modules
    row, original = audited_plane
    record = copy.deepcopy(original)
    record["fixture"] = {"changed": True}
    result = auditor.audit_saved_record(
        "plane", 0, row, record, required_evidence_complete=False
    )
    assert result["semantic_validation"] == "SEMANTIC_INVALID"
    assert result["scientific_admission"] == "NOT_ADMITTED"
    result = auditor.audit_saved_record(
        "plane", 0, row, original, required_evidence_complete=False
    )
    assert result["semantic_validation"] == "SEMANTIC_VALID"
    assert result["scientific_admission"] == "INCOMPLETE_EVIDENCE"


def _v4_coordinated_mutation(recorder, record, row):
    trace = record["observable_traces"]["acceleration"]
    calls = record["original_entrypoint"]["captured_calls"]
    events = [
        call
        for call in calls
        if call["kernel"] == "plane_observable_values"
        and (call["inputs"].get("digits") or call["inputs"].get("mp_dps_at_call")) == 80
    ][-3:]
    final = recorder._decode_callback_value(events[-1]["output"])["acceleration"]
    component_float = trace["frozen_component_checks"]["derivative"][
        "frozen_component_si"
    ]
    with recorder.mp.workdps(100):
        magnitude = recorder.mp.mpf(str(component_float)) / 10
        middle = (final[0] + magnitude / 2, final[1], final[2])
        coarse = (middle[0] + magnitude, middle[1], middle[2])
        for event, vector in zip(events[:2], (coarse, middle), strict=True):
            event["output"]["acceleration"] = [
                recorder._capture_number(value, 80) for value in vector
            ]
    with recorder.mp.workdps(80):
        pair = recorder._mp_callback_pairs(
            calls, "plane", row["fixture"], "acceleration"
        )[0]
        formula = 10 * max(pair)
        check = trace["frozen_component_checks"]["derivative"]
        check["formula_value_si"] = recorder._exact_mp_string(formula)
        check["formula_float_si"] = float(formula)
        check["matches_frozen_float"] = check["formula_float_si"] == component_float
        trace["derivative_differences_si"].update(
            {
                "4h_to_2h": recorder._exact_mp_string(pair[0]),
                "2h_to_h": recorder._exact_mp_string(pair[1]),
                "ten_max": recorder._exact_mp_string(formula),
            }
        )
        trace["derivative_trend"].update(
            recorder._change_trend(
                *pair,
                recorder.mp.mpf(trace["derivative_trend"]["independent_floor_si"]),
            )
        )
    raw, unresolved, status = recorder._derived_trace_status(
        record["observable_traces"], record["reproduction"]["failed_checks"]
    )
    record.update(
        raw_unresolved_trends=raw, unresolved_trends=unresolved, status=status
    )
    recorder._validate_trace_record(
        record,
        "plane",
        0,
        row,
        zero_certificate_sha256=record["zero_certificate_supplement_sha256"],
        polynomial_certificate_sha256=record[
            "polynomial_certificate_supplement_sha256"
        ],
    )


def test_v4_coordinated_callback_trend_component_summary_checkpoint_manifest_mutation(
    numeric_modules, audited_plane, tmp_path
):
    recorder, auditor = numeric_modules
    row, original = audited_plane
    record = copy.deepcopy(original)
    baseline = _audit_binding(auditor, row, record)
    old_document = _audit_anchor(auditor, record, baseline)
    _v4_coordinated_mutation(recorder, record, row)
    assert (
        record["original_entrypoint"]["captured_calls"]
        != original["original_entrypoint"]["captured_calls"]
    )
    binding = _audit_binding(auditor, row, record)
    new_document = _audit_anchor(auditor, record, binding)
    assert old_document != new_document
    # Recompute every mutable local claim, including checkpoint and manifest.
    for filename in ("checkpoint.json", "manifest.json"):
        path = tmp_path / filename
        path.write_text(
            json.dumps(
                {
                    "provenance_document": new_document,
                    "provenance_validation": "PROVENANCE_VERIFIED",
                    "full_gate_complete": True,
                }
            )
        )
    record["provenance_validation"] = "PROVENANCE_VERIFIED"
    record["full_gate_complete"] = True
    # These claims are also covered in the modified accepted-result journal.
    binding = _audit_binding(auditor, row, record)
    new_document = _audit_anchor(auditor, record, binding)
    result = auditor.audit_record(
        "plane",
        0,
        row,
        record,
        **binding,
        provenance_binding=json.loads((tmp_path / "manifest.json").read_text()),
    )
    assert result["semantic_validation"] == "SEMANTIC_VALID"
    assert result["provenance_validation"]["status"] == "PROVENANCE_UNVERIFIED"
    assert result["scientific_admission"] == "NOT_ADMITTED"
    anchored = auditor.audit_record(
        "plane", 0, row, record, **binding, **_audit_mock_trust(auditor, old_document)
    )
    assert anchored["provenance_validation"]["status"] == "PROVENANCE_MISMATCH"
    assert anchored["scientific_admission"] == "NOT_ADMITTED"
    # A matching mock demonstrates API matching only, never genuine execution.
    matching = auditor.audit_record(
        "plane", 0, row, record, **binding, **_audit_mock_trust(auditor, new_document)
    )
    assert matching["semantic_validation"] == "SEMANTIC_VALID"
    assert matching["provenance_validation"]["status"] == "PROVENANCE_VERIFIED"


@pytest.fixture(scope="module")
def certificate_cases(numeric_modules):
    recorder, auditor = numeric_modules
    cases = {}
    for kind in ("zero", "whole", "rigorous", "polynomial"):
        zero, polynomial = [], []
        if kind == "whole":
            fixture = next(
                item
                for item in recorder.oracle_module(
                    "sphere_high_degree_fixtures"
                ).sphere_high_degree_fixtures()
                if item.target_kind == "center"
            )
            fixture = recorder._jsonable(fixture)
            dataset, index = "high_sphere", 0
            reference, metadata = recorder._invoke_original_entrypoint(dataset, fixture)
        elif kind == "rigorous":
            fixture = next(
                item
                for item in recorder.oracle_module("source_fixtures").source_fixtures()
                if item["family"] == "finite_temperature"
                and item["shape"] == "cuboid"
                and item["target_kind"] == "center"
            )
            dataset, index = "source", 12
            reference, metadata = recorder.oracle_module(
                "source_reference"
            ).source_reference(fixture)
        else:
            fixtures = list(
                recorder.oracle_module("noncavity_fixtures").noncavity_fixtures()
            )
            dataset, index = "noncavity", 0 if kind == "zero" else 19
            fixture = (
                next(
                    item
                    for item in fixtures
                    if item["family"] == "gaussian" and item["target_kind"] == "center"
                )
                if kind == "zero"
                else fixtures[index]
            )
            reference, metadata = recorder.oracle_module(
                "noncavity_reference"
            ).noncavity_reference(fixture)
        row = {
            "fixture": recorder._jsonable(fixture),
            "reference": recorder._jsonable(reference),
            "metadata": recorder._jsonable(metadata),
        }
        if kind == "zero":
            zero = [
                {
                    "dataset": recorder.DATASETS[dataset],
                    "fixture": row["fixture"],
                    "observable": "acceleration",
                    "certificate": "even radial center gradient",
                    "original_uncertainty_kind": "estimate",
                    "original_used_uncertainty_si": 0.0,
                    "frozen_labels_and_values_unchanged": True,
                }
            ]
        if kind == "polynomial":
            polynomial = [
                {
                    "dataset": dataset,
                    "source_row_index": index,
                    "fixture": row["fixture"],
                    "certificate_kind": "retarded_cubic_pulse_polynomial_panels",
                    "covered_observables": list(auditor.SECTIONS),
                    "covered_original_stencils": (
                        recorder.POLYNOMIAL_CERTIFICATE_COVERED_STENCILS
                    ),
                    "certified_error": recorder.POLYNOMIAL_CERTIFICATE_CERTIFIED_ERROR,
                    "frozen_values_uncertainty_labels_unchanged": True,
                }
            ]
        record = recorder.record_row(
            dataset,
            row,
            source_row_index=index,
            zero_certificates=zero,
            zero_certificate_source="frozen-zero-estimate-certificates-integrated.json",
            zero_certificate_sha256=auditor.ZERO_SHA256,
            polynomial_certificates=polynomial,
            polynomial_certificate_source="polynomial-quadrature-certificates-integrated.json",
            polynomial_certificate_sha256=auditor.POLYNOMIAL_SHA256
            if dataset in {"source", "noncavity"}
            else None,
            polynomial_verifier_script_sha256=auditor.POLYNOMIAL_SCRIPT_SHA256,
        )
        record["process_independence"] = _v4_worker_evidence()
        cases[kind] = (dataset, index, row, record, zero, polynomial)
    return cases


@pytest.mark.parametrize("kind", ["zero", "whole", "rigorous", "polynomial"])
def test_v4_certificate_positive_scope_is_semantic_only(
    numeric_modules, certificate_cases, kind
):
    _, auditor = numeric_modules
    dataset, index, row, record, zero, polynomial = certificate_cases[kind]
    result = auditor.audit_record(
        dataset, index, row, record, zero=zero, polynomial=polynomial
    )
    assert result["semantic_validation"] == "SEMANTIC_VALID"
    assert result["certificate_validation"] == "CERTIFICATE_SCOPE_VALID"
    assert result["provenance_validation"]["status"] == "PROVENANCE_UNVERIFIED"
    assert result["scientific_admission"] != "ADMITTED_WITH_DECLARED_SCOPE"


@pytest.mark.parametrize(
    "kind,change",
    [
        ("zero", "fixture"),
        ("zero", "digest"),
        ("zero", "coherent_kind"),
        ("zero", "coherent_zero"),
        ("zero", "unchanged"),
        ("whole", "certificate"),
        ("whole", "reason"),
        ("rigorous", "certificate"),
        ("rigorous", "reason"),
        ("polynomial", "fixture"),
        ("polynomial", "digest"),
        ("polynomial", "coherent_kind"),
        ("polynomial", "coherent_error"),
        ("polynomial", "index"),
        ("polynomial", "unchanged"),
    ],
)
def test_v4_certificate_scope_rejects_independent_and_coordinated_changes(
    numeric_modules, certificate_cases, kind, change
):
    _, auditor = numeric_modules
    dataset, index, row, record, zero, polynomial = copy.deepcopy(
        certificate_cases[kind]
    )
    name = "acceleration" if kind in {"zero", "rigorous"} else "potential"
    trace = record["observable_traces"][name]
    if kind in {"whole", "rigorous"}:
        if change == "certificate":
            trace["certificate"] = "certificate for another fixture"
        else:
            trace["derivative_trend"]["reason"] = "certificate for another route"
    else:
        certificates = zero if kind == "zero" else polynomial
        annotation_key = (
            "zero_certificate_adjudication"
            if kind == "zero"
            else "polynomial_quadrature_certificate_adjudication"
        )
        if change == "fixture":
            certificates[0]["fixture"] = {"outside": True}
        elif change == "index":
            certificates[0]["source_row_index"] = 18
        elif change == "unchanged":
            certificates[0][
                "frozen_labels_and_values_unchanged"
                if kind == "zero"
                else "frozen_values_uncertainty_labels_unchanged"
            ] = False
        else:
            field, value = {
                "digest": ("supplement_sha256", "0" * 64),
                "coherent_kind": (
                    "original_uncertainty_kind"
                    if kind == "zero"
                    else "certificate_kind",
                    "invented",
                ),
                "coherent_zero": ("original_used_uncertainty_si", 1.0),
                "coherent_error": (
                    "certified_error",
                    "total numerical error fully certified",
                ),
            }[change]
            if change != "digest":
                certificates[0][field] = value
            for tr in record["observable_traces"].values():
                if annotation_key in tr:
                    tr[annotation_key][field] = value
    with pytest.raises(ValueError):
        auditor.audit_record(
            dataset, index, row, record, zero=zero, polynomial=polynomial
        )


@pytest.mark.parametrize("failed_retry", [False, True])
@pytest.mark.parametrize("audit_unrelated_row", [False, True])
def test_v4_provenance_real_recorder_journal_bytes_unicode_retry(
    numeric_modules, audited_plane, tmp_path, failed_retry, audit_unrelated_row
):
    from dataclasses import replace

    recorder, auditor = numeric_modules
    row, record = copy.deepcopy(audited_plane)
    selected = {0: row, 1: row} if audit_unrelated_row else {0: row}
    manifest = {
        "schema": "gragra.gen1.refinement-trace-dataset-manifest.v4",
        "run_id": "test-api-run",
        "dataset": "plane",
        "selected_source_row_indices": sorted(selected),
    }
    real = recorder.AttemptJournal(
        tmp_path / "real",
        run_id="test-api-run",
        dataset="plane",
        selected_rows=selected,
        manifest=manifest,
    )
    failure = _journal_failure(0)
    failure[1]["exception"]["message"] = "独立した失敗 Ω"
    failure[1]["exception"]["traceback"] = "RuntimeError: 独立した失敗 Ω\n"
    real.finish(real.start(0), failure)
    real.begin_retry()
    if failed_retry:
        real.finish(real.start(0), failure)
        real.settle()
        real.begin_retry()
    real.finish(real.start(0), (0, record, None))
    real.settle()
    index = 1 if audit_unrelated_row else 0
    if audit_unrelated_row:
        record["source_row_index"] = index
        real.finish(real.start(index), (index, copy.deepcopy(record), None))
        real.settle()
    binding = _audit_binding(auditor, row, record, failure=True)
    binding["selected_rows"] = selected
    binding["independent_inputs"] = replace(
        binding["independent_inputs"], source_row_index=index
    )
    c = auditor.PROVENANCE
    lines = real.path.read_bytes().splitlines(keepends=True)
    events = [json.loads(line) for line in lines]
    binding["attempt_journal"] = c.AttemptJournalInputs(
        events=events,
        attempt_states={
            e["attempt_id"]: {
                "state": "failed"
                if e["schema"] == auditor.ATTEMPT_FAILURE_SCHEMA
                else "complete",
                "source_row_index": e["source_row_index"],
                "retry_of": e["retry_of"],
            }
            for e in events
        },
        retry_plan=real.checkpoint["retry_plan_history"],
        accepted_results={
            f"plane:{e['source_row_index']}": {
                "attempt_id": e["attempt_id"],
                "result_sha256": digest(e["record"]),
            }
            for e in events
            if e["schema"] == auditor.ATTEMPT_SUCCESS_SCHEMA
        },
    )
    assert events[1]["last_durable_trace_offset"] == len(lines[0])
    result = auditor.audit_record(
        "plane",
        index,
        row,
        record,
        **binding,
        **_audit_mock_trust(auditor, _audit_anchor(auditor, record, binding)),
    )
    assert result["journal_binding_complete"] is True, result
    assert result["provenance_validation"]["status"] == "PROVENANCE_VERIFIED"
    assert result["scientific_admission"] == "ADMITTED_WITH_DECLARED_SCOPE"


def test_v4_provenance_interrupted_retry_history_positive_api_control(
    numeric_modules, audited_plane
):
    _, auditor = numeric_modules
    row, record = audited_plane
    binding = _audit_binding(auditor, row, record, failure=True)
    binding["attempt_journal"].attempt_states["interrupted-retry"] = {
        "state": "interrupted",
        "source_row_index": 0,
        "retry_of": "failed",
    }
    doc = _audit_anchor(auditor, record, binding)
    result = auditor.audit_record(
        "plane", 0, row, record, **binding, **_audit_mock_trust(auditor, doc)
    )
    assert result["journal_binding_complete"] is True
    assert result["provenance_validation"]["status"] == "PROVENANCE_VERIFIED"


@pytest.mark.parametrize(
    "schema",
    [
        "gragra.gen1.refinement-trace-success-event.v4",
        "gragra.gen1.refinement-trace-attempt-success.v3",
        "gragra.gen1.refinement-trace.v1",
    ],
)
def test_v4_provenance_journal_rejects_other_event_schema(
    numeric_modules, audited_plane, schema
):
    _, auditor = numeric_modules
    row, record = audited_plane
    binding = _audit_binding(auditor, row, record)
    event = binding["attempt_journal"].events[0]
    event["schema"] = schema
    event["event_sha256"] = auditor.journal_local_sha256(
        {k: v for k, v in event.items() if k != "event_sha256"}
    )
    result = auditor.audit_record(
        "plane",
        0,
        row,
        record,
        **binding,
        **_audit_mock_trust(auditor, _audit_anchor(auditor, record, binding)),
    )
    assert result["journal_binding_complete"] is False
    assert result["provenance_validation"]["status"] == "PROVENANCE_UNVERIFIED"
    assert result["scientific_admission"] == "NOT_ADMITTED"


def test_v4_provenance_mismatch_precedes_missing_evidence(
    numeric_modules, audited_plane
):
    _, auditor = numeric_modules
    row, record = audited_plane
    binding = _audit_binding(auditor, row, record)
    doc = _audit_anchor(auditor, record, binding)
    doc["payload"]["run_id"] = "another-run"
    doc = auditor.PROVENANCE.commitment_document(doc["payload"])
    result = auditor.audit_record(
        "plane",
        0,
        row,
        record,
        **binding,
        required_evidence_complete=False,
        **_audit_mock_trust(auditor, doc),
    )
    assert result["provenance_validation"]["status"] == "PROVENANCE_MISMATCH"
    assert result["scientific_admission"] == "NOT_ADMITTED"
    missing = auditor.audit_saved_record("plane", 0, row, None)
    assert missing["semantic_validation"] == "SEMANTIC_NOT_RUN"
    assert missing["scientific_admission"] == "INCOMPLETE_EVIDENCE"


def test_v4_provenance_missing_journal_is_incomplete_local_evidence(
    numeric_modules, audited_plane
):
    _, auditor = numeric_modules
    row, record = audited_plane
    result = auditor.audit_record("plane", 0, row, record)
    assert result["semantic_validation"] == "SEMANTIC_VALID"
    assert result["provenance_validation"]["status"] == "PROVENANCE_UNVERIFIED"
    assert result["journal_binding_complete"] is False
    assert result["scientific_admission"] == "INCOMPLETE_EVIDENCE"
