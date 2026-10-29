"""Pre-production source scales from frozen formulas, never production output."""

from __future__ import annotations

import hashlib
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache

from .transport import (
    GateError,
    artifact_pin,
    canonical_bytes,
    strict_json,
    verify_artifact,
)


@lru_cache(maxsize=1)
def pi_bounds() -> tuple[Fraction, Fraction]:
    """Machin identity with exact alternating-series remainder enclosures.

    tan(4*atan(1/5)-atan(1/239))=1 and the angle is in (0,pi/2).
    Each 256-term atan partial sum lies below its positive next remainder.
    """

    def atan_bounds(denominator):
        partial = sum(
            (
                Fraction((-1) ** n, (2 * n + 1) * denominator ** (2 * n + 1))
                for n in range(256)
            ),
            Fraction(),
        )
        return partial, partial + Fraction(1, 513 * denominator**513)

    a, b = atan_bounds(5), atan_bounds(239)
    return 16 * a[0] - 4 * b[1], 16 * a[1] - 4 * b[0]


def scale_bounds(family: str, key: dict, inputs: dict, kernel) -> tuple[Fraction, str]:
    """Return a conservative rational lower scale where the absolute term matters."""

    def q(name):
        return Fraction(Decimal(inputs[name]))

    si = q("g_si") ** 2 if key["representation"] == "SI" else Fraction(1)
    if family == "A2" and key["observable"] == "isotropic_depth_factor":
        return Fraction(1, 3), "exact rational scalar S_ref=1/3"
    if family in {"R1", "B2"}:
        pi = pi_bounds()[0]
        base = (
            (2 * pi * q("density_kg_m3")) ** 2
            * q("surface_vertical_displacement_psd_m2_hz")
            if family == "R1"
            else (4 * pi * q("density_kg_m3")) ** 2
            * (4 * q("p_displacement_psd_m2_hz") + q("s_displacement_psd_m2_hz"))
        )
        return (
            base * si,
            "exact rational Machin lower pi enclosure; "
            "frozen220/260-digit arithmetic and downward180-digit adoption",
        )
    if family in {"A1", "A3"}:
        if family == "A3" and q("radius_m") == 0:
            return Fraction(), "frozen whole-zero radius override; exact rational zero"
        return q("pressure_psd_pa2_hz") / (
            q("sound_speed_m_s") * q("frequency_hz")
        ) ** 2 * (
            4 if family == "A3" else 1
        ) * si, "exact rational frozen pressure/(c*f)^2" + (
            "*4" if family == "A3" else ""
        ) + ("*G^2" if key["representation"] == "SI" else "")
    if family == "A2":
        lower = min(kernel._sqrt_bounds(Fraction(6), d)[0] for d in (220, 260))
        return (
            lower
            * q("pressure_psd_pa2_hz")
            / (q("sound_speed_m_s") * q("frequency_hz")) ** 2
            * si,
            "verified exact rational sqrt6 lower brackets at220/260digits; "
            "downward180 serialization",
        )
    if family == "T1":
        import mpmath

        samples = []
        for precision in (220, 260):
            c = mpmath.mp.clone()
            c.dps = precision
            p = {n: c.mpf(v) for n, v in inputs.items()}
            exponent, wind = p["structure_exponent"], p["wind_speed_m_s"]
            omega = 2 * c.pi * p["frequency_hz"]
            x = omega * p["minimum_approach_m"] / wind
            coefficient = (
                16
                * c.pi
                * c.gamma(exponent + 2)
                * c.sin(c.pi * exponent / 2)
                * (p["density_kg_m3"] / p["temperature_k"]) ** 2
                * p["structure_coefficient_k_m_neg_p_over_2"] ** 2
                * wind ** (exponent + 2)
                * omega ** (-exponent - 3)
            )
            angular = c.pi * (
                c.cos(p["wind_axis_angle_rad"]) ** 2
                + c.sin(p["wind_axis_angle_rad"]) ** 2 / 2
            )
            factor = c.pi / 4 * x / (x - 1) * c.exp(-2 * x) if x > 1 else 1 + 1 / x**2
            value = coefficient * angular * factor
            if key["representation"] == "SI":
                value *= p["g_si"] ** 2
            samples.append(Fraction(Decimal(c.nstr(value, precision))))
        # T1 a=0: this tabulated positive scale has no role in B or acceptance.
        # No claim of a rigorous mpmath transcendental enclosure is made.
        return (
            min(samples),
            "frozen T1 source-scale formula220/260digits, downward decimal arithmetic; "
            "estimate of transcendental scale, no rigorous claim; "
            "a=0 makes S_ref numerically immaterial",
        )
    raise GateError("unknown frozen source-scale family")


def build_source_scales(
    *, fixture_pin: dict, budget_pin: dict, procedure_pin: dict, kernel
) -> dict:
    """Pure comparator arithmetic; pin complete F inputs and implementation bytes."""
    from .family_driver import frozen_expected_keys, frozen_input_rows

    fixtures = strict_json(verify_artifact(fixture_pin))
    budget = strict_json(verify_artifact(budget_pin))
    procedure = strict_json(verify_artifact(procedure_pin))
    family = fixtures["family"]
    if budget.get("family") != family or procedure.get("family") != family:
        raise GateError("source-scale frozen family mismatch")
    if family == "T1" and kernel.decimal_fraction(budget["absolute_rate"]) != 0:
        raise GateError(
            "T1 source-scale estimate cannot support a nonzero absolute acceptance term"
        )
    frozen = frozen_input_rows(fixtures)
    rows = []
    for key in frozen_expected_keys(fixtures):
        row = frozen[key["fixture_id"]]
        inputs = (
            dict(row["inputs"])
            if "inputs" in row
            else {
                **row,
                "frequency_hz": key["frequency_hz"],
                "g_si": fixtures.get("g_si", row.get("g_si")),
            }
        )
        if key["observable"] == "isotropic_depth_factor":
            inputs = {
                "nominal_exact_x": row["nominal_exact_x"],
                "frequency_hz": key["frequency_hz"],
            }
        else:
            inputs["frequency_hz"] = key["frequency_hz"]
        lower, method = scale_bounds(family, key, inputs, kernel)
        if lower < 0:
            raise GateError("negative source scale")
        values = [kernel.fraction_decimal(lower, d, downward=True) for d in (220, 260)]
        adopted = min(kernel.decimal_fraction(v) for v in values)
        value = kernel.fraction_decimal(adopted, 180, downward=True)
        if kernel.decimal_fraction(value) > lower:
            raise GateError("source-scale conservative serialization failed")
        rows.append(
            {
                "key": key,
                "value": value,
                "method": method,
                "input_sha256": hashlib.sha256(canonical_bytes(inputs)).hexdigest(),
            }
        )
    return {
        "schema": "gen1-stochastic-source-scales-v1",
        "family": family,
        "rows": rows,
        "pins": {
            "fixture": fixture_pin,
            "budget": budget_pin,
            "family_procedure": procedure_pin,
            "builder_source": artifact_pin(__file__, "source_scale_builder"),
        },
        "rounding": {
            "arithmetic_significant_digits": [220, 260],
            "serialization_significant_digits": 180,
            "direction": "downward",
            "reference_or_production_outputs_used": False,
        },
    }
