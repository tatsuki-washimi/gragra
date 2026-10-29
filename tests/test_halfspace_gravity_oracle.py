"""Independent Newton-gravity-integral oracle vs. gragra's half-space
vertical-incidence P-wave transfer functions.

The oracle (``tests/oracles/generate_halfspace_oracle.py``) computes the
Newtonian gravity of the wave-induced bulk density perturbation and
free-surface mass sheet via a 3D integral over depth with the
horizontal/radial direction reduced analytically (not a literal triple
integral), sharing no code with ``gragra.analytic`` / ``gragra.waves`` (see
that module's docstring for the full derivation), then extrapolates the
disk radius R_cut -> infinity. This is compared against
``gragra.analytic.halfspace.halfspace_transfer_bulk`` /
``halfspace_transfer_full``, which use the closed-form
``T_bulk = 1 - 2 cos(kH)`` / ``T_full = -2 cos(kH)`` ported from nn-sus
(``docs/architecture/halfspace_conventions.md`` Sec.5/8).

This is an independent numerical re-derivation from the published formula,
not a comparison against ANNA (or any other external toolbox) execution
output.

Tolerance: the saved oracle's worst observed extrapolation error across the
13-point kH grid is ~1.4e-3 in T (see the JSON `points[*].bulk|full
.extrapolated_r_cut_inf`); ATOL/RTOL below carry a >2x margin over that.

In addition to the bulk/full/sheet value comparisons, two reproducibility
    checks are included here as part of the numerical comparison:

- vertical (Z_max) truncation error, via the ``z_max_sensitivity`` field
  (Z_MAX_FACTOR in {40,80,160} at the largest R_cut stage);
- cutoff-fit stability, via the ``fit_stability_max_diff`` field
  (extrapolation re-run on the last 7/6/5 R_cut stages).

Both are required to stay well inside the ATOL/RTOL used for the main
gragra comparison above -- otherwise the reported ``extrapolated_r_cut_inf``
could be under-verified. A separate fast, content-addressed, fail-closed
provenance check (``tests/test_validation_extra_available.py``'s
``test_halfspace_oracle_content_addressed_provenance_is_verified``, see
``tests/_provenance_contract.py``) verifies the saved oracle's
``provenance.source_content_sha256`` / ``derivation_input_sha256`` /
``golden_artifact_sha256`` against the current generator script and the
JSON's own content, to catch a generator edit or a hand-edited JSON that
wasn't followed by regeneration.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

# Skip this entire module if the `validation` extra isn't installed. Without
# this guard, the unconditional `scipy`/`mpmath` imports inside
# oracles.generate_halfspace_oracle raise ModuleNotFoundError at *collection*
# time -- which fails the whole pytest session even for a not-installed extra,
# since pytest.mark.slow deselection only applies after a module is
# successfully imported and collected.
pytest.importorskip("scipy")
pytest.importorskip("mpmath")

from oracles.generate_halfspace_oracle import (  # noqa: E402
    RHO_0,
    XI_INC,
    G,
    bulk_finite,
    extrapolate,
    sheet_finite,
)

from gragra.analytic.halfspace import (  # noqa: E402
    halfspace_transfer_bulk,
    halfspace_transfer_full,
)

ORACLE_PATH = Path(__file__).parent / "oracles" / "data" / "halfspace_v1.json"
DENOM = 4.0 * math.pi * G * RHO_0 * XI_INC

ATOL = 3e-3
RTOL = 1e-2


def _load_oracle() -> dict:
    with ORACLE_PATH.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def oracle_data() -> dict:
    return _load_oracle()


@pytest.mark.slow
def test_oracle_metadata_independent_of_gragra(oracle_data):
    assert oracle_data["schema_version"] == 1
    assert set(oracle_data["provenance"]["independent_of"]) >= {
        "gragra.analytic",
        "gragra.waves",
    }
    assert oracle_data["provenance"][
        "not_a_cross_validation_against_external_toolbox_execution"
    ]
    # Content-addressed provenance:
    # upstream_revision is declarative attribution only, not verified here
    # (see tests/_provenance_contract.py and
    # tests/test_validation_extra_available.py's
    # test_halfspace_oracle_content_addressed_provenance_is_verified for the
    # fail-closed hash checks that protect the saved scientific fixture).
    assert oracle_data["provenance"]["oracle_name"] == "halfspace_v1"
    assert oracle_data["provenance"]["upstream_revision"]
    assert oracle_data["provenance"]["source_content_sha256"]
    assert oracle_data["provenance"]["derivation_input_sha256"]
    assert oracle_data["provenance"]["golden_artifact_sha256"]


@pytest.mark.slow
def test_z_max_truncation_tail_error_is_small(oracle_data):
    # Vertical (Z_max) truncation error, re-verified per point via a
    # Z_MAX_FACTOR in {40,80,160} sensitivity sweep at the largest R_cut
    # stage (generator module docstring: z_max_sensitivity). Must stay well
    # inside the ATOL used for the main gragra comparison above, otherwise
    # the main sweep's Z_MAX_FACTOR=80 choice would be under-verified. The
    # saved oracle's worst observed value is ~1.1e-5 in T; this leaves a
    # >5x margin.
    for point in oracle_data["points"]:
        tail_t = point["z_max_sensitivity"]["tail_error_estimate"] / DENOM
        assert tail_t < 1e-4, f"kH={point['kH']}: tail_error(T)={tail_t}"


@pytest.mark.slow
def test_cutoff_fit_stability_is_small(oracle_data):
    # Cutoff-fit stability: extrapolation re-run on the last 7/6/5 R_cut
    # stages must agree within the ATOL used for the main gragra comparison
    # above -- otherwise the reported extrapolated_r_cut_inf could be an
    # artifact of which R_cut stages happened to be included, not a
    # converged R_cut -> infinity value.
    for point in oracle_data["points"]:
        for channel in ("bulk", "sheet", "full"):
            diff_t = point[channel]["fit_stability_max_diff"] / DENOM
            assert diff_t < ATOL, (
                f"kH={point['kH']} channel={channel}: fit_stability(T)={diff_t}"
            )


@pytest.mark.slow
def test_bulk_transfer_matches_saved_oracle(oracle_data):
    for point in oracle_data["points"]:
        k_h = point["kH"]
        t_bulk_oracle = -point["bulk"]["extrapolated_r_cut_inf"] / DENOM
        t_bulk_gragra = float(halfspace_transfer_bulk(k_h))
        assert t_bulk_oracle == pytest.approx(t_bulk_gragra, abs=ATOL, rel=RTOL), (
            f"kH={k_h}: oracle T_bulk={t_bulk_oracle}, gragra={t_bulk_gragra}"
        )


@pytest.mark.slow
def test_full_transfer_matches_saved_oracle(oracle_data):
    for point in oracle_data["points"]:
        k_h = point["kH"]
        t_full_oracle = -point["full"]["extrapolated_r_cut_inf"] / DENOM
        t_full_gragra = float(halfspace_transfer_full(k_h))
        assert t_full_oracle == pytest.approx(t_full_gragra, abs=ATOL, rel=RTOL), (
            f"kH={k_h}: oracle T_full={t_full_oracle}, gragra={t_full_gragra}"
        )


@pytest.mark.slow
def test_sheet_minus_one_identity_in_oracle(oracle_data):
    # Internal consistency of the oracle itself (not a gragra comparison):
    # the free-surface mass-sheet term is a fixed T_sheet = -1 contribution
    # (conventions Sec.5: T_full - T_bulk = -1 identically), independent of
    # kH. This exercises the "surface" channel that gragra does not expose
    # as a standalone function.
    for point in oracle_data["points"]:
        t_sheet_oracle = -point["sheet"]["extrapolated_r_cut_inf"] / DENOM
        assert t_sheet_oracle == pytest.approx(-1.0, abs=ATOL, rel=RTOL), (
            f"kH={point['kH']}: oracle T_sheet={t_sheet_oracle}"
        )


@pytest.mark.slow
def test_mpmath_cross_check_confirms_scipy_precision(oracle_data):
    # >=2 mpmath precision stages (dps=30, dps=50) must agree with each
    # other and with the scipy float64 value well within the ATOL used
    # above, confirming the scipy result is not floating-point-limited.
    for entry in oracle_data["mpmath_cross_check"]:
        scipy_val = entry["scipy_float64_value"]
        mp_vals = list(entry["mpmath_values_by_dps"].values())
        assert len(mp_vals) >= 2
        for mp_val in mp_vals:
            assert mp_val == pytest.approx(scipy_val, abs=1e-4, rel=1e-6), (
                f"kH={entry['kH']} r_cut={entry['r_cut']}: "
                f"scipy={scipy_val}, mpmath={mp_val}"
            )
        assert mp_vals[0] == pytest.approx(mp_vals[1], abs=1e-8)


@pytest.mark.slow
@pytest.mark.parametrize("k_h", [0.8, 2.0, 4.0])
def test_representative_points_reintegrated_in_ci(k_h):
    # Not just trusting the saved JSON: re-run the independent integral +
    # extrapolation live, at a subset of the grid, using a smaller R_cut
    # sweep for CI runtime (still >=3 stages for a meaningful extrapolation).
    r_cuts = np.array([20.0, 40.0, 80.0, 160.0], dtype=np.float64)
    h = k_h / 1.0

    bulk_vals = np.array([bulk_finite(h, r)[0] for r in r_cuts])
    sheet_vals = np.array([sheet_finite(h, r) for r in r_cuts])
    full_vals = bulk_vals + sheet_vals

    bulk_c0, _, _ = extrapolate(r_cuts, bulk_vals)
    full_c0, _, _ = extrapolate(r_cuts, full_vals)

    t_bulk_oracle = -bulk_c0 / DENOM
    t_full_oracle = -full_c0 / DENOM
    t_bulk_gragra = float(halfspace_transfer_bulk(k_h))
    t_full_gragra = float(halfspace_transfer_full(k_h))

    # Tighter than the saved-oracle ATOL/RTOL: the shorter 4-stage R_cut
    # sweep still lands within ~2e-4 of the closed form at these points
    # (verified empirically), so a tolerance an order of magnitude looser
    # than that residual would fail to catch a percent-level regression
    # (e.g. a sign or coefficient bug) while still tolerating the sweep's
    # own extrapolation noise.
    assert t_bulk_oracle == pytest.approx(t_bulk_gragra, abs=1e-3, rel=5e-3)
    assert t_full_oracle == pytest.approx(t_full_gragra, abs=1e-3, rel=5e-3)
