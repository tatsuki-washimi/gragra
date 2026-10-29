"""Independent scientific acceptance; production imports and helpers forbidden."""

import math
from dataclasses import dataclass

import numpy as np

RATES = {"potential": 1e-10, "acceleration": 1e-10, "gradient": 1e-9}
SHAPES = {"potential": (), "acceleration": (3,), "gradient": (3, 3)}


def norm_si(observable, values):
    array = np.asarray(values, dtype=complex)
    if observable not in SHAPES or array.shape != SHAPES[observable]:
        raise ValueError("observable shape mismatch")
    if not np.isfinite(array).all():
        raise ValueError("finite SI values required")
    norm = math.hypot(
        *(part for value in array.flat for part in (value.real, value.imag))
    )
    if not math.isfinite(norm):
        raise ValueError("finite derived SI norm required")
    return norm


@dataclass(frozen=True)
class OracleUncertainty:
    total_error_estimate_si: float | None
    total_error_bound_si: float | None
    used_uncertainty_si: float | None
    uncertainty_kind: str | None
    method: str
    components: tuple[tuple[str, float], ...]
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self):
        for value in (
            self.total_error_estimate_si,
            self.total_error_bound_si,
            self.used_uncertainty_si,
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("nonnegative finite uncertainty required")
        if self.uncertainty_kind not in {None, "estimate", "rigorous_bound"}:
            raise ValueError("unknown uncertainty kind")
        for name, value in self.components:
            if not name or not math.isfinite(value) or value < 0:
                raise ValueError("invalid uncertainty component")

    def resolved(self):
        selected = (
            self.total_error_estimate_si
            if self.uncertainty_kind == "estimate"
            else self.total_error_bound_si
            if self.uncertainty_kind == "rigorous_bound"
            else None
        )
        return (
            selected is not None
            and self.used_uncertainty_si == selected
            and bool(self.method)
            and selected >= math.fsum(value for _, value in self.components)
        )


def assess(observable, production_si, reference_si, scale_si, uncertainty):
    if observable not in RATES or not math.isfinite(scale_si) or scale_si < 0:
        raise ValueError("invalid observable or SI scale")
    reference_norm = norm_si(observable, reference_si)
    norm_si(observable, production_si)
    difference = np.asarray(production_si, dtype=complex) - np.asarray(
        reference_si, dtype=complex
    )
    difference_norm = norm_si(observable, difference)
    budget = 1e-12 * scale_si + RATES[observable] * reference_norm
    if not math.isfinite(budget):
        raise ValueError("finite derived acceptance budget required")
    resolved = uncertainty.resolved()
    used = uncertainty.used_uncertainty_si if resolved else None
    difference_condition = used is not None and difference_norm + used <= budget
    oracle_condition = used is not None and used <= 0.1 * budget
    accepted = difference_condition and oracle_condition
    return {
        "observable": observable,
        "reference_norm_si": reference_norm,
        "difference_norm_si": difference_norm,
        "budget_ref_si": budget,
        "used_uncertainty_si": used,
        "uncertainty_kind": uncertainty.uncertainty_kind,
        "difference_plus_uncertainty_condition": difference_condition,
        "oracle_fraction_condition": oracle_condition,
        "accepted": accepted,
        "status": "pass"
        if accepted
        else "fail"
        if resolved
        else "uncertainty_unresolved",
        "claim": (
            "based_on_recorded_oracle_uncertainty_estimate"
            if accepted and uncertainty.uncertainty_kind == "estimate"
            else "bounded_oracle_error"
            if accepted
            else None
        ),
    }
