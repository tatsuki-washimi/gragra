"""Input validation and isolated mpmath context helpers.

This module deliberately does not import mpmath until a context is requested.
"""

import math
import re
import unicodedata
from typing import Any


_DECIMAL_RE = re.compile(
    r"^(?P<sign>[+-]?)(?P<mantissa>(?:\d+(?:\.\d*)?|\.\d+))"
    r"(?:[eE][+-]?\d+)?$"
)


def validate_dps(dps: Any) -> int:
    """Return supported precision, requiring an exact Python integer."""
    if type(dps) is not int or dps not in (60, 80):
        raise ValueError("dps must be the Python int 60 or 80")
    return dps


def validate_input(value: Any) -> str | int | float:
    """Validate a scalar input while retaining its original representation."""
    if type(value) is str:
        match = _DECIMAL_RE.fullmatch(value)
        if match is None:
            raise ValueError("input string must use finite decimal notation")
        mantissa_digits = match.group("mantissa").replace(".", "")
        is_zero = not any(unicodedata.decimal(digit) != 0 for digit in mantissa_digits)
        if match.group("sign") == "-" and not is_zero:
            raise ValueError("input must be finite and nonnegative")
        return value
    if type(value) is int:
        if value < 0:
            raise ValueError("input must be nonnegative")
        return value
    if type(value) is float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("input must be finite and nonnegative")
        return value
    raise ValueError("input must be a decimal string, Python int, or Python float")


def make_context(dps: int):
    """Create an independent mpmath clone configured to ``dps`` decimal places."""
    if type(dps) is not int or dps not in (60, 80, 100):
        raise ValueError("context dps must be the Python int 60, 80, or 100")
    try:
        import mpmath
    except ImportError as exc:
        raise ImportError(
            "Reference calculations require mpmath; install it with `pip install gragra[reference]`."
        ) from exc
    context = mpmath.mp.clone()
    context.dps = dps
    return context


def convert_input(context: Any, original: str | int | float):
    """Convert the retained original input directly in one cloned context."""
    return context.mpf(original)


def transfer_scalar(value: Any, target_context: Any):
    """Transfer an evaluated mpf directly into another cloned context."""
    return target_context.mpf(value)
