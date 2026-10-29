"""Independent Newton-gravity-integral oracle vs. the published
closed form for the Rayleigh-wave half-space Newtonian-noise acceleration
(ANNA eqs.75-76, arXiv:2603.15157v4 Sec.4.3 = Harms 2019).

The oracle (``tests/oracles/generate_rayleigh_oracle.py``) independently
re-derives the Rayleigh eigenfunctions and the bulk/free-surface mass
perturbations from elasticity, then computes their Newtonian gravity via a
y'-analytically-reduced ("line mass") 2D/1D integral truncated at
period-aligned x'_cut stages and extrapolated x'_cut -> infinity, sharing no
code with ``gragra.analytic`` / ``gragra.waves``. This is compared against
the closed-form relation, whose layout is re-coded in this test file (NOT
imported from ``tests.test_anna_halfspace_fixture_contract``). The saved
``gamma`` values do originate in the generator's independent secular solve,
so this closed-form assertion is not independent of that gamma calculation;
its independent evidence is the finite-integral/extrapolation route.

This is an independent numerical re-derivation from the published
formula, not a comparison against ANNA (or any other external toolbox)
execution output.

Tolerance: the saved oracle's worst observed extrapolation error across the
12-point (v_P/v_S, kH) grid is ~9.3e-5 absolute / ~0.72% relative (largest
kH, where the target itself is exponentially small); ATOL/RTOL below carry
a >2.7x margin over both.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

# Skip this entire module if the `validation` extra isn't installed (see
# tests/test_halfspace_gravity_oracle.py for why this must precede the
# scipy/mpmath-importing module import below).
pytest.importorskip("scipy")
pytest.importorskip("mpmath")

from oracles.generate_rayleigh_oracle import (  # noqa: E402
    RHO_0,
    G,
    bulk_finite,
    extrapolate,
    rayleigh_gamma,
    solve_rayleigh,
    surface_finite,
)

ORACLE_PATH = Path(__file__).parent / "oracles" / "data" / "rayleigh_v1.json"

ATOL = 5e-4
RTOL = 2e-2


def _closed_form(gamma: float, k_h: float) -> tuple[float, float]:
    """ANNA eq.75 relation, re-coded independently of gragra and evaluated
    with the generator-derived saved gamma: returns (a_z, a_x_imag_coeff)."""
    decay = math.exp(-k_h)
    a_z = -2.0 * math.pi * G * RHO_0 * gamma * decay
    a_x = 2.0 * math.pi * G * RHO_0 * gamma * decay
    return a_z, a_x


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
    # test_rayleigh_oracle_content_addressed_provenance_is_verified for the
    # fail-closed hash checks that protect the saved scientific fixture).
    assert oracle_data["provenance"]["oracle_name"] == "rayleigh_v1"
    assert oracle_data["provenance"]["upstream_revision"]
    assert oracle_data["provenance"]["source_content_sha256"]
    assert oracle_data["provenance"]["derivation_input_sha256"]
    assert oracle_data["provenance"]["golden_artifact_sha256"]


def test_generator_does_not_import_gragra():
    """Keep the numerical oracle independent of production implementation."""
    generator_path = Path(__file__).parent / "oracles" / "generate_rayleigh_oracle.py"
    tree = ast.parse(generator_path.read_text(), filename=str(generator_path))
    gragra_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            gragra_imports.extend(
                alias.name for alias in node.names if alias.name.startswith("gragra")
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (node.module == "gragra" or node.module.startswith("gragra."))
        ):
            gragra_imports.append(node.module)
    assert not gragra_imports, f"generator imports gragra: {gragra_imports}"


@pytest.mark.slow
def test_full_transfer_matches_closed_form(oracle_data):
    for point in oracle_data["points"]:
        target_az, target_ax = _closed_form(point["gamma"], point["kH"])
        az = point["full"]["az_extrapolated_x_cut_inf"]
        ax = point["full"]["ax_extrapolated_x_cut_inf"]
        assert az == pytest.approx(target_az, abs=ATOL, rel=RTOL), (
            f"vp_vs={point['vp_vs_ratio']} kH={point['kH']}: "
            f"oracle a_z={az}, closed-form={target_az}"
        )
        assert ax == pytest.approx(target_ax, abs=ATOL, rel=RTOL), (
            f"vp_vs={point['vp_vs_ratio']} kH={point['kH']}: "
            f"oracle a_x={ax}, closed-form={target_ax}"
        )


@pytest.mark.slow
def test_bulk_plus_sheet_equals_full(oracle_data):
    # Internal consistency of the oracle's own channel decomposition (not a
    # gragra comparison): bulk + sheet must equal full, to within the
    # per-channel extrapolation-fit precision (much tighter than ATOL/RTOL
    # above, since this is exact algebra on the saved per-channel values,
    # not an independent re-derivation).
    for point in oracle_data["points"]:
        for comp in ("az", "ax"):
            bulk = point["bulk"][f"{comp}_extrapolated_x_cut_inf"]
            sheet = point["sheet"][f"{comp}_extrapolated_x_cut_inf"]
            full = point["full"][f"{comp}_extrapolated_x_cut_inf"]
            assert (bulk + sheet) == pytest.approx(full, abs=1e-6, rel=1e-6), (
                f"vp_vs={point['vp_vs_ratio']} kH={point['kH']} comp={comp}"
            )


@pytest.mark.slow
def test_z_cut_truncation_tail_error_is_small(oracle_data):
    # Vertical (z_cut) truncation error, re-verified per point via a
    # Z_CUT_DECAY_LENGTHS in {30,60,120} sensitivity sweep (generator module
    # docstring: z_cut_sensitivity). Unlike the P-wave case, the
    # Rayleigh bulk density decays purely exponentially (no oscillation), so
    # this is expected to be at or near float64's own noise floor.
    for point in oracle_data["points"]:
        sensitivity = point["z_cut_sensitivity"]
        az_values = sensitivity["bulk_az_values"]
        ax_values = sensitivity["bulk_ax_values"]
        az_tail = abs(az_values[-1] - az_values[-2])
        ax_tail = abs(ax_values[-1] - ax_values[-2])

        assert sensitivity["az_tail_error_estimate"] == pytest.approx(az_tail)
        assert sensitivity["ax_tail_error_estimate"] == pytest.approx(ax_tail)
        assert (
            sensitivity["tail_error_estimate"] == sensitivity["az_tail_error_estimate"]
        )
        assert az_tail < 1e-6, f"kH={point['kH']}: az_tail_error={az_tail}"
        assert ax_tail < 1e-6, f"kH={point['kH']}: ax_tail_error={ax_tail}"


@pytest.mark.slow
def test_cutoff_fit_stability_is_small(oracle_data):
    # Cutoff-fit stability: extrapolation re-run on the last 7/6/5 x'_cut
    # stages must agree within ATOL -- otherwise the reported
    # extrapolated_x_cut_inf could be an artifact of which stages were
    # included, not a converged x'_cut -> infinity value.
    for point in oracle_data["points"]:
        for channel in ("bulk", "sheet", "full"):
            for comp in ("az", "ax"):
                diff = point[channel][f"{comp}_fit_stability"]["fit_stability_max_diff"]
                assert diff < ATOL, (
                    f"vp_vs={point['vp_vs_ratio']} kH={point['kH']} "
                    f"channel={channel} comp={comp}: fit_stability={diff}"
                )


@pytest.mark.slow
def test_mpmath_cross_check_confirms_scipy_precision(oracle_data):
    check = oracle_data["mpmath_cross_check"]
    scipy_vals = check["scipy_float64"]
    mp_by_dps = list(check["mpmath_values_by_dps"].values())
    assert len(mp_by_dps) >= 2
    for mp_vals in mp_by_dps:
        for key in ("bulk_az", "bulk_ax", "surf_az", "surf_ax"):
            assert mp_vals[key] == pytest.approx(scipy_vals[key], abs=1e-6, rel=1e-8), (
                f"{key}: scipy={scipy_vals[key]}, mpmath={mp_vals[key]}"
            )
    assert mp_by_dps[0]["bulk_az"] == pytest.approx(mp_by_dps[1]["bulk_az"], abs=1e-10)


@pytest.mark.slow
@pytest.mark.parametrize("vp_vs_ratio", [math.sqrt(3.0), 2.0])
@pytest.mark.parametrize("k_h", [0.5, 2.0])
def test_representative_points_reintegrated_in_ci(vp_vs_ratio, k_h):
    # Not just trusting the saved JSON: re-run the independent integral +
    # extrapolation live, at a subset of the (v_P/v_S, kH) grid, using a
    # shorter x'_cut sweep for CI runtime (still >=3 period-aligned stages
    # per the corresponding oracle contract).
    period = 2.0 * math.pi
    x_cuts = np.array([n * period for n in (4.0, 8.0, 16.0, 32.0)])

    _, q, s = solve_rayleigh(vp_vs_ratio)
    gamma = rayleigh_gamma(q, s)
    f_z0 = (s * s - 1.0) / (1.0 + s * s)
    a_amp = 1.0 / (q * f_z0)
    d_amp = RHO_0 * a_amp * (1.0 - q * q)

    z_cut = 60.0 / q
    az_vals = np.empty_like(x_cuts)
    ax_vals = np.empty_like(x_cuts)
    for i, x_cut in enumerate(x_cuts):
        bz, bx = bulk_finite(q, d_amp, 1.0, k_h, x_cut, z_cut)
        sz, sx = surface_finite(1.0, k_h, x_cut)
        az_vals[i] = bz + sz
        ax_vals[i] = bx + sx

    az_c0, _, _ = extrapolate(x_cuts, az_vals)
    ax_c0, _, _ = extrapolate(x_cuts, ax_vals)
    target_az, target_ax = _closed_form(gamma, k_h)

    # Tighter than the saved-oracle ATOL/RTOL: the shorter 4-stage sweep
    # still lands within ~1e-4 of the closed form at these points (verified
    # empirically), so a tolerance an order of magnitude looser than that
    # residual would fail to catch a percent-level regression while still
    # tolerating the shorter sweep's own extrapolation noise.
    assert az_c0 == pytest.approx(target_az, abs=2e-3, rel=5e-2)
    assert ax_c0 == pytest.approx(target_ax, abs=2e-3, rel=5e-2)
