"""Phase 16-b: full-space ANNA fixture-contract validation (A-1).

Design: docs/design/phase16b_anna_fullspace.md §1, §4, §5.

Compares gragra's closed-form analytic APIs (spherical_cavity_total_p,
spherical_cavity_bulk_p, spherical_cavity_wall) against published
closed-form expressions from the ANNA paper (arXiv:2603.15157v4 eqs.46-53,
eqs.63-65).

G-free outputs from gragra.analytic are multiplied by G_SI for physical
values, and then normalized by (G_SI * rho * xi) to match the dimensionless
golden values.
"""

import numpy as np
import pytest

from gragra.analytic.cavity_sphere import (
    spherical_cavity_bulk_p,
    spherical_cavity_total_p,
    spherical_cavity_wall,
)
from gragra.constants import G_SI
from gragra.waves import displacement_direction
from gragra.waves.directions import normalize_directions

R = 20.0
F = 5.0
VP = 5000.0
VS = 2500.0
RHO = 2800.0
XI = 1.0

# KP: 0.006283185307179587 rad/m for kp*r0 = 0.12566370614359174
KP = 2.0 * np.pi * F / VP
# KS: 0.012566370614359173 rad/m for ks*r0 = 0.25132741228718347
KS = 2.0 * np.pi * F / VS

REF_SRC = (
    "published closed-form (ANNA paper arXiv:2603.15157v4 eqs.46-53,63-65 = Harms 2019)"
)

# Default absolute tolerance used by test_fixture_cases_match_reference() below;
# cases may override it via an "atol" key (see s_wave_bulk_structural_zero).
_DEFAULT_ATOL = 1e-15

# A handful of generic (non-axis-aligned) propagation directions used to probe
# the structural transversality of the SH/SV polarization basis returned by
# gragra.waves.displacement_direction (see s_wave_bulk_structural_zero below).
_GENERIC_K_HATS = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, -2.0, 0.5],
        [0.3, 0.7, -1.0],
    ],
    dtype=np.float64,
)


def _max_s_wave_transversality_residual() -> float:
    """Max |k_hat . d| over SH/SV polarizations for several generic k_hat.

    There is no closed-form analytic expression for the S-wave bulk
    Newtonian-noise contribution in the ANNA paper: it is a *structural*
    zero, because S-waves are transverse (k_hat perpendicular to the
    displacement direction d), which makes div(u_S) = 0 identically and
    therefore the bulk (density-perturbation) term vanishes exactly. This
    helper validates that structural claim directly against the wave API's
    SH/SV output (gragra.waves.displacement_direction), instead of asserting
    an unverified 0.0 literal.

    k_hat . d is not exactly 0.0 in floating point because SH/SV are built
    via cross products (see waves/polarization.py
    _vertical_anchored_s_wave_basis), so a small (~1e-16) residual from
    floating-point rounding is expected; the comparison below therefore uses
    atol=1e-14 rather than exact equality.
    """
    k_norm = normalize_directions(_GENERIC_K_HATS)
    residuals = [
        np.max(
            np.abs(np.sum(k_norm * displacement_direction(wave_type, k_norm), axis=1))
        )
        for wave_type in ("SH", "SV")
    ]
    return float(max(residuals))


FIXTURE_CASES = [
    {
        "case_id": "p_wave_total_fullspace_eq48",
        "description": (
            "Full-space P-wave total Newtonian noise acceleration (ANNA eq.48)."
        ),
        "gragra_value": lambda: float(
            spherical_cavity_total_p(R, KP, RHO, XI) * G_SI / (G_SI * RHO * XI)
        ),
        "reference_value": 8.364358523725757,
        "reference_source": REF_SRC,
        "rtol": 1e-10,
    },
    {
        "case_id": "p_wave_bulk_fullspace_eq50",
        "description": (
            "Full-space P-wave bulk Newtonian noise acceleration (ANNA eq.50)."
        ),
        "gragra_value": lambda: float(
            spherical_cavity_bulk_p(R, KP, RHO, XI) * G_SI / (G_SI * RHO * XI)
        ),
        "reference_value": 12.533323356430422,
        "reference_source": REF_SRC,
        "rtol": 1e-10,
    },
    {
        "case_id": "p_wave_wall_fullspace_eq52",
        "description": (
            "Full-space P-wave cavity wall Newtonian noise acceleration (ANNA eq.52)."
        ),
        "gragra_value": lambda: float(
            spherical_cavity_wall(R, KP, RHO, XI, wave_type="P")
            * G_SI
            / (G_SI * RHO * XI)
        ),
        "reference_value": -4.168964832704666,
        "reference_source": REF_SRC,
        "rtol": 1e-10,
    },
    {
        "case_id": "p_wave_bulk_plus_wall_self_consistency",
        "description": (
            "Full-space P-wave bulk + wall sum self-consistency check (matches eq.48)."
        ),
        "gragra_value": lambda: float(
            (
                spherical_cavity_bulk_p(R, KP, RHO, XI)
                + spherical_cavity_wall(R, KP, RHO, XI, wave_type="P")
            )
            * G_SI
            / (G_SI * RHO * XI)
        ),
        "reference_value": 8.364358523725757,
        "reference_source": REF_SRC,
        "rtol": 1e-10,
    },
    {
        "case_id": "s_wave_total_fullspace_eq65",
        "description": (
            "Full-space S-wave total (=wall) Newtonian noise acceleration (ANNA eq.65)."
        ),
        "gragra_value": lambda: float(
            spherical_cavity_wall(R, KS, RHO, XI, wave_type="S")
            * G_SI
            / (G_SI * RHO * XI)
        ),
        "reference_value": -4.162391133994149,
        "reference_source": REF_SRC,
        "rtol": 1e-10,
    },
    {
        "case_id": "s_wave_bulk_structural_zero",
        "description": (
            "Full-space S-wave bulk noise structural zero (div u_S = 0). No "
            "closed-form analytic S-bulk expression exists (ANNA does not "
            "publish one), so this validates the structural identity that "
            "implies the zero: S-waves are transverse (k_hat . d == 0 for "
            "both SH and SV), which makes div(u_S) = 0 identically. See "
            "_max_s_wave_transversality_residual() docstring for details."
        ),
        "gragra_value": _max_s_wave_transversality_residual,
        "reference_value": 0.0,
        "reference_source": (
            "hand-computed structural identity (S-wave transversality via "
            "gragra.waves.displacement_direction, not an ANNA closed form)"
        ),
        "rtol": 0.0,
        "atol": 1e-14,
    },
]


def test_fixture_contract_shape():
    """Every fixture case exposes the Sec.4 contract fields."""
    required_keys = {
        "case_id",
        "description",
        "gragra_value",
        "reference_value",
        "reference_source",
        "rtol",
    }
    for case in FIXTURE_CASES:
        assert required_keys.issubset(case.keys())
        source = case["reference_source"]
        assert any(
            source.startswith(prefix)
            for prefix in ("hand-computed", "published closed-form")
        )


def test_fixture_cases_match_reference():
    """Cross-validation against published ANNA closed-form formulas."""
    for case in FIXTURE_CASES:
        gragra_val = case["gragra_value"]()
        np.testing.assert_allclose(
            gragra_val,
            case["reference_value"],
            rtol=case["rtol"],
            atol=case.get("atol", _DEFAULT_ATOL),
            err_msg=f"fixture case {case['case_id']} failed",
        )


def test_s_wave_bulk_zero_assertion():
    """Explicit structural-zero check: S-waves cause zero divergence.

    k_hat . d for SH/SV is not bitwise 0.0 (cross-product construction
    leaves floating-point rounding residue), so this uses atol=1e-14
    instead of exact equality; see _max_s_wave_transversality_residual().
    """
    val = FIXTURE_CASES[-1]["gragra_value"]()
    assert val == pytest.approx(0.0, abs=1e-14)
