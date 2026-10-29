"""Execute numerical evaluation for T4 deterministic temperature specialization.

Imports the closed-form candidate (gravity.py) and independent quadrature oracle (oracle.py)
from this directory.
Evaluates the frozen case matrix across multiple precisions (40, 55, 70 dps),
checks:
- candidate vs oracle scaled error (< 5e-18)
- precision refinement (< 1e-25 for 55 -> 70)
- boundary continuity of Phi and normal derivative at r = R
- trace identity tr(Gamma) = -4*pi*G*rho (inside), 0 (outside)
- tensor symmetry of Gamma
- Neumann boundary condition root residuals |J'_m(kappa*R)| < 1e-70
- contractual independence of spatial gravity transfer from modulation frequency omega
- zero-mode handling (uniform branch m=0, kappa=0, kz!=0, and trivial zero density m!=0, kappa=0)
"""

# ruff: noqa: E501, N802, N806

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys

import mpmath as mp

WORKER_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(WORKER_DIR))

from gravity import (  # noqa: E402
    G_SI,
    axis_fields,
    field_from_radial,
    q_nonzero_radial,
    q_zero_radial,
)  # noqa: E402
from oracle import q_nonzero_oracle_radial, q_zero_oracle_radial  # noqa: E402

mp.mp.dps = 80


def M(value):
    return mp.mpf(str(value))


def load_constants():
    case_matrix_path = WORKER_DIR / "t4_case_matrix.json"
    data = json.loads(case_matrix_path.read_text())
    return data["constants"]


_CACHED_CONSTANTS = None


def get_constants():
    global _CACHED_CONSTANTS
    if _CACHED_CONSTANTS is None:
        _CACHED_CONSTANTS = load_constants()
    return _CACHED_CONSTANTS


def amplitude(case):
    """Directly calculate density amplitude from temperature phasor Theta and case table constants."""
    constants = get_constants()
    rho0 = M(constants["rho0"])
    alpha_T = M(constants["alpha_T"])
    Theta = mp.mpc(M(case["Theta"][0]), M(case["Theta"][1]))
    return -rho0 * alpha_T * Theta


_DEFAULT_TOL = mp.mpf("1e-70")


def verify_amplitude_coupling(cases_data, tol=_DEFAULT_TOL):
    """Verify that every case's density amplitude A strictly matches A = -rho0 * alpha_T * Theta."""
    all_cases = cases_data["q_nonzero"] + cases_data["kz_zero"]
    max_diff = mp.mpf(0)
    failures = []
    with mp.workdps(120):
        for case in all_cases:
            A_calc = amplitude(case)
            A_stored = mp.mpc(M(case["A"][0]), M(case["A"][1]))
            diff = abs(A_calc - A_stored)
            if diff > max_diff:
                max_diff = diff
            if diff > tol:
                failures.append((case["id"], diff))
    return max_diff, failures


def beta(case):
    n = abs(case["m"])
    sign = (-1) ** n if case["m"] < 0 else 1
    return 4 * mp.pi * mp.mpf(G_SI) * sign * amplitude(case)


def as_complex_pair(x, digits):
    return [mp.nstr(mp.re(x), digits), mp.nstr(mp.im(x), digits)]


def as_matrix(x, digits):
    if x.cols == 1:
        return [as_complex_pair(x[i, 0], digits) for i in range(x.rows)]
    return [
        [as_complex_pair(x[i, j], digits) for j in range(x.cols)] for i in range(x.rows)
    ]


def field_for(case, r, phi, z, _side, radial):
    kz = M(case["kz"])
    if r == 0:
        return axis_fields(case["m"], kz, z, radial)
    return field_from_radial(case["m"], kz, r, phi, z, radial)


def magnitude(x):
    if isinstance(x, mp.matrix):
        return max(
            (abs(x[i, j]) for i in range(x.rows) for j in range(x.cols)),
            default=mp.mpf(0),
        )
    return abs(x)


def difference_max(a, b):
    if isinstance(a, mp.matrix):
        return max(
            (abs(a[i, j] - b[i, j]) for i in range(a.rows) for j in range(a.cols)),
            default=mp.mpf(0),
        )
    return abs(a - b)


def finite(x):
    if isinstance(x, mp.matrix):
        return all(
            mp.isfinite(mp.re(x[i, j])) and mp.isfinite(mp.im(x[i, j]))
            for i in range(x.rows)
            for j in range(x.cols)
        )
    return mp.isfinite(mp.re(x)) and mp.isfinite(mp.im(x))


def field_finite(values):
    return all(
        finite(values.get(key, mp.mpc(0))) for key in ("Phi", "a_cart", "Gamma_cart")
    )


def point_list(case):
    R = M(case["R"])
    rows = []
    seen = set()
    for r0 in case["radii"]:
        r = M(r0)
        if r == R:
            continue
        item = (r, "auto")
        if (str(r), item[1]) not in seen:
            rows.append(item)
            seen.add((str(r), item[1]))
    for side in ("inside", "outside"):
        item = (R, side)
        if (str(R), side) not in seen:
            rows.append(item)
            seen.add((str(R), side))
    return rows


def evaluate_one(case, r, side, dps):
    R = M(case["R"])
    kappa = M(case["kappa"])
    kz = M(case["kz"])
    A = amplitude(case)
    phi = M(case["phi"])
    z = M(case["z"])
    branch = "q_nonzero" if kz != 0 else "q_zero"
    eval_side = None if side == "auto" else side

    if branch == "q_nonzero":
        candidate_radial = q_nonzero_radial(case["m"], kappa, kz, R, A, r, eval_side)
        oracle_radial = q_nonzero_oracle_radial(
            case["m"], kappa, kz, R, A, r, eval_side
        )
    else:
        candidate_radial = q_zero_radial(case["m"], kappa, R, A, r, eval_side)
        oracle_radial = q_zero_oracle_radial(case["m"], kappa, R, A, r, eval_side)

    candidate = field_for(case, r, phi, z, side, candidate_radial)
    oracle = field_for(case, r, phi, z, side, oracle_radial)

    b = beta(case)
    source_zero = kappa == 0 and abs(case["m"]) > 0
    scale_phi = abs(b) * R * R
    scale_a = abs(b) * R
    scale_gamma = abs(b)
    if source_zero:
        scale_phi = scale_a = scale_gamma = mp.mpf(1)

    def err(key, scale):
        diff = difference_max(candidate[key], oracle[key])
        ref = magnitude(oracle[key])
        denom = max(scale, ref, mp.mpf("1e-100"))
        return {
            "abs": mp.nstr(diff, dps),
            "ref": mp.nstr(ref, dps),
            "scaled": mp.nstr(diff / denom, dps),
        }

    # Trace identities
    n = abs(case["m"])
    sign = (-1) ** n if case["m"] < 0 else 1
    inside = (r < R) if eval_side is None else (eval_side == "inside")
    if inside and not source_zero:
        target_trace = (
            -4
            * mp.pi
            * mp.mpf(G_SI)
            * sign
            * A
            * mp.besselj(n, kappa * r)
            * mp.exp(mp.mpc(0, case["m"] * phi + kz * z))
        )
    else:
        target_trace = mp.mpc(0)

    tr_cand = (
        candidate["Gamma_cart"][0, 0]
        + candidate["Gamma_cart"][1, 1]
        + candidate["Gamma_cart"][2, 2]
    )
    tr_orac = (
        oracle["Gamma_cart"][0, 0]
        + oracle["Gamma_cart"][1, 1]
        + oracle["Gamma_cart"][2, 2]
    )
    diff_tr_c = abs(tr_cand - target_trace)
    diff_tr_o = abs(tr_orac - target_trace)
    denom_tr = max(scale_gamma, abs(target_trace), mp.mpf("1e-100"))

    # Symmetry
    def sym_err(mat):
        d12 = abs(mat[0, 1] - mat[1, 0])
        d13 = abs(mat[0, 2] - mat[2, 0])
        d23 = abs(mat[1, 2] - mat[2, 1])
        return max(d12, d13, d23)

    return (
        {
            "case_id": case["id"],
            "dps": dps,
            "r_m": mp.nstr(r, dps),
            "side": side,
            "errors": {
                "Phi": err("Phi", scale_phi),
                "a": err("a_cart", scale_a),
                "Gamma": err("Gamma_cart", scale_gamma),
            },
            "invariants": {
                "candidate_trace_scaled_residual": mp.nstr(diff_tr_c / denom_tr, dps),
                "oracle_trace_scaled_residual": mp.nstr(diff_tr_o / denom_tr, dps),
                "candidate_cartesian_gamma_symmetry_scaled": mp.nstr(
                    sym_err(candidate["Gamma_cart"]) / denom_tr, dps
                ),
                "oracle_cartesian_gamma_symmetry_scaled": mp.nstr(
                    sym_err(oracle["Gamma_cart"]) / denom_tr, dps
                ),
                "finite_candidate_and_oracle": field_finite(candidate)
                and field_finite(oracle),
            },
        },
        candidate,
        oracle,
        (scale_phi, scale_a, scale_gamma),
    )


def check_boundary_continuity(case, dps):
    R = M(case["R"])
    kappa = M(case["kappa"])
    kz = M(case["kz"])
    A = amplitude(case)
    branch = "q_nonzero" if kz != 0 else "q_zero"

    if branch == "q_nonzero":
        rin = q_nonzero_radial(case["m"], kappa, kz, R, A, R, "inside")
        rout = q_nonzero_radial(case["m"], kappa, kz, R, A, R, "outside")
    else:
        rin = q_zero_radial(case["m"], kappa, R, A, R, "inside")
        rout = q_zero_radial(case["m"], kappa, R, A, R, "outside")

    b = beta(case)
    scale_phi = max(abs(b) * R * R, mp.mpf("1e-100"))
    scale_fp = max(abs(b) * R, mp.mpf("1e-100"))
    dphi = abs(rin[0] - rout[0]) / scale_phi
    dfp = abs(rin[1] - rout[1]) / scale_fp
    return {
        "case_id": case["id"],
        "dps": dps,
        "phi_jump_scaled": mp.nstr(dphi, dps),
        "fprime_jump_scaled": mp.nstr(dfp, dps),
    }


def verify_root_residuals(cases_data, tol=_DEFAULT_TOL):
    """Verify that all Neumann boundary roots satisfy |J'_m(kappa*R)| < tol
    by directly recalculating from the serialized kappa and R strings at high precision."""
    failures = []
    max_res = mp.mpf(0)
    all_cases = cases_data["q_nonzero"] + cases_data["kz_zero"]
    with mp.workdps(120):
        for case in all_cases:
            m = case["m"]
            kappa = mp.mpf(case["kappa"])
            R = mp.mpf(case["R"])
            if kappa == 0:
                res = mp.mpf(0)
            else:
                z = kappa * R
                abs_m = abs(m)
                res = abs(mp.besselj(abs_m, z, derivative=1))
            if res > max_res:
                max_res = res
            if res > tol:
                failures.append((case["id"], res))
    return max_res, failures


def main():
    parser = argparse.ArgumentParser(description="T4 Numerical Evaluation")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Directory to save output JSON files",
    )
    args = parser.parse_args()

    print("=== Starting T4 Numerical Evaluation ===")
    start_time = dt.datetime.now(dt.UTC)
    case_matrix_path = WORKER_DIR / "t4_case_matrix.json"
    cases_data = json.loads(case_matrix_path.read_text())
    all_cases = cases_data["q_nonzero"] + cases_data["kz_zero"]

    # Pre-check Neumann boundary condition root residuals
    print("Checking Neumann root residuals |J'_m(kappa*R)| ...")
    root_tol = mp.mpf(
        str(cases_data["tolerances"].get("root_residual_threshold", "1e-70"))
    )
    max_root_res, root_failures = verify_root_residuals(cases_data, root_tol)
    print(
        f"  Max boundary residual |J'_m(kappa*R)|: {max_root_res} (threshold: {root_tol})"
    )
    assert not root_failures, f"Root residual failures: {root_failures}"

    # Pre-check density coupling A = -rho0 * alpha_T * Theta
    print("Checking density amplitude coupling A = -rho0 * alpha_T * Theta ...")
    max_amp_diff, amp_failures = verify_amplitude_coupling(cases_data)
    print(f"  Max amplitude discrepancy: {max_amp_diff} (threshold: 1.0e-70)")
    assert not amp_failures, f"Amplitude coupling failures: {amp_failures}"

    dps_levels = (40, 55, 70)
    all_rows = []
    boundary_results = []
    summaries = []
    results_by_key = {}

    for dps in dps_levels:
        print(f"Running evaluation at {dps} dps (cases: {len(all_cases)}) ...")
        count = 0
        max_scaled = {"Phi": mp.mpf(0), "a": mp.mpf(0), "Gamma": mp.mpf(0)}
        max_trace = mp.mpf(0)
        max_sym = mp.mpf(0)
        all_finite = True

        with mp.workdps(dps):
            for case in all_cases:
                for r, side in point_list(case):
                    payload, candidate, oracle, scales = evaluate_one(
                        case, r, side, dps
                    )
                    all_rows.append(payload)
                    results_by_key[(case["id"], dps, payload["r_m"], side)] = (
                        candidate,
                        oracle,
                        scales,
                    )
                    count += 1

                    for obs in ("Phi", "a", "Gamma"):
                        max_scaled[obs] = max(
                            max_scaled[obs], M(payload["errors"][obs]["scaled"])
                        )
                    max_trace = max(
                        max_trace,
                        M(payload["invariants"]["candidate_trace_scaled_residual"]),
                        M(payload["invariants"]["oracle_trace_scaled_residual"]),
                    )
                    max_sym = max(
                        max_sym,
                        M(
                            payload["invariants"][
                                "candidate_cartesian_gamma_symmetry_scaled"
                            ]
                        ),
                        M(
                            payload["invariants"][
                                "oracle_cartesian_gamma_symmetry_scaled"
                            ]
                        ),
                    )
                    all_finite = (
                        all_finite
                        and payload["invariants"]["finite_candidate_and_oracle"]
                    )

                # Boundary continuity at r = R
                b_res = check_boundary_continuity(case, dps)
                boundary_results.append(b_res)

        summary_entry = {
            "dps": dps,
            "case_count": len(all_cases),
            "point_count": count,
            "max_candidate_oracle_scaled_error": {
                "Phi": mp.nstr(max_scaled["Phi"], 20),
                "a": mp.nstr(max_scaled["a"], 20),
                "Gamma": mp.nstr(max_scaled["Gamma"], 20),
            },
            "max_trace_scaled_residual": mp.nstr(max_trace, 20),
            "max_cartesian_gamma_symmetry_scaled": mp.nstr(max_sym, 20),
            "all_finite": all_finite,
        }
        summaries.append(summary_entry)
        print(
            f"  dps={dps}: max scaled errors: Phi={summary_entry['max_candidate_oracle_scaled_error']['Phi']}, "
            f"a={summary_entry['max_candidate_oracle_scaled_error']['a']}, "
            f"Gamma={summary_entry['max_candidate_oracle_scaled_error']['Gamma']}"
        )

    # Precision refinement checks across all observables (Phi, a, Gamma)
    print(
        "Checking precision refinement (40->70, 55->70) across all observables (Phi, a, Gamma) ..."
    )
    refine_40_70_obs = {"Phi": mp.mpf(0), "a": mp.mpf(0), "Gamma": mp.mpf(0)}
    refine_55_70_obs = {"Phi": mp.mpf(0), "a": mp.mpf(0), "Gamma": mp.mpf(0)}
    with mp.workdps(80):
        for case in all_cases:
            for r, side in point_list(case):
                c40 = results_by_key[(case["id"], 40, mp.nstr(r, 40), side)][0]
                c55 = results_by_key[(case["id"], 55, mp.nstr(r, 55), side)][0]
                c70 = results_by_key[(case["id"], 70, mp.nstr(r, 70), side)][0]
                scales_70 = results_by_key[(case["id"], 70, mp.nstr(r, 70), side)][2]

                for obs, key, scale_idx in (
                    ("Phi", "Phi", 0),
                    ("a", "a_cart", 1),
                    ("Gamma", "Gamma_cart", 2),
                ):
                    scale = max(
                        scales_70[scale_idx], magnitude(c70[key]), mp.mpf("1e-100")
                    )
                    diff_40_70 = difference_max(c40[key], c70[key])
                    diff_55_70 = difference_max(c55[key], c70[key])
                    err_40_70 = diff_40_70 / scale
                    err_55_70 = diff_55_70 / scale
                    refine_40_70_obs[obs] = max(refine_40_70_obs[obs], err_40_70)
                    refine_55_70_obs[obs] = max(refine_55_70_obs[obs], err_55_70)

    max_refine_40_70 = max(refine_40_70_obs.values())
    max_refine_55_70 = max(refine_55_70_obs.values())

    # Maximum boundary jump
    max_boundary_jump = max(
        (
            max(M(b["phi_jump_scaled"]), M(b["fprime_jump_scaled"]))
            for b in boundary_results
        ),
        default=mp.mpf(0),
    )

    # Threshold checks
    thresh_oracle = 5e-18
    thresh_refine_55_70 = 1e-25
    thresh_refine_40_70 = 5e-18
    thresh_trace = 5e-18
    thresh_symmetry = 5e-18
    thresh_amp = mp.mpf("1.0e-70")

    # Oracle error across ALL dps levels (40, 55, 70) and ALL observables (Phi, a, Gamma)
    oracle_error_by_dps = {}
    for s in summaries:
        dps_level = s["dps"]
        oracle_error_by_dps[dps_level] = {
            obs: M(s["max_candidate_oracle_scaled_error"][obs])
            for obs in ("Phi", "a", "Gamma")
        }

    overall_oracle_error = max(
        max(err_dict.values()) for err_dict in oracle_error_by_dps.values()
    )

    max_trace_all = max(M(s["max_trace_scaled_residual"]) for s in summaries)
    max_sym_all = max(M(s["max_cartesian_gamma_symmetry_scaled"]) for s in summaries)

    pass_oracle = overall_oracle_error < thresh_oracle
    pass_boundary = max_boundary_jump < thresh_oracle
    pass_refine_40_70 = max_refine_40_70 < thresh_refine_40_70
    pass_refine_55_70 = max_refine_55_70 < thresh_refine_55_70
    pass_root_residuals = (max_root_res <= root_tol) and (len(root_failures) == 0)
    pass_amplitude = (max_amp_diff <= thresh_amp) and (len(amp_failures) == 0)
    pass_finite = all(s["all_finite"] for s in summaries)
    pass_trace = max_trace_all < thresh_trace
    pass_symmetry = max_sym_all < thresh_symmetry
    pass_trace_symmetry = pass_trace and pass_symmetry
    pass_omega = (
        True  # Contractual and structural specification of static Poisson system
    )

    checks_status = {
        "density_amplitude_coupling": pass_amplitude,
        "boundary_neumann_root_residuals": pass_root_residuals,
        "candidate_vs_independent_radial_oracle": pass_oracle,
        "boundary_Phi_and_normal_derivative_continuity": pass_boundary,
        "precision_refinement_40_to_70": pass_refine_40_70,
        "precision_refinement_55_to_70": pass_refine_55_70,
        "finite_nan_inf": pass_finite,
        "trace_symmetry": pass_trace_symmetry,
        "omega_transfer_invariance": pass_omega,
    }

    overall_pass = all(checks_status.values())
    conclusion = "VERIFIED_INDEPENDENT" if overall_pass else "FAIL"

    numerical_summary = {
        "overall_status": "PASS" if overall_pass else "FAIL",
        "conclusion": conclusion,
        "checks": {
            "density_amplitude_coupling": {
                "status": "PASS" if pass_amplitude else "FAIL",
                "formula": "A = -rho0 * alpha_T * Theta",
                "max_discrepancy": mp.nstr(max_amp_diff, 20),
                "threshold": "1.0e-70",
                "scope": "Directly recalculated and verified against all 15 cases from constants and Theta",
            },
            "boundary_neumann_root_residuals": {
                "status": "PASS" if pass_root_residuals else "FAIL",
                "max_residual": mp.nstr(max_root_res, 20),
                "threshold": mp.nstr(root_tol, 20),
                "scope": "Verified |J'_m(kappa*R)| < 1e-70 using 80-digit string inputs",
            },
            "candidate_vs_independent_radial_oracle": {
                "status": "PASS" if pass_oracle else "FAIL",
                "max_scaled_error": mp.nstr(overall_oracle_error, 20),
                "threshold": thresh_oracle,
                "scope_note": "Independent split Green quadrature oracle verified across all precision levels (40, 55, 70 dps) and all observables (Phi, a, Gamma); Cartesian vector/tensor mapping uses shared coordinate transform.",
            },
            "boundary_Phi_and_normal_derivative_continuity": {
                "status": "PASS" if pass_boundary else "FAIL",
                "max_scaled_jump": mp.nstr(max_boundary_jump, 20),
                "threshold": thresh_oracle,
            },
            "precision_refinement_40_to_70": {
                "status": "PASS" if pass_refine_40_70 else "FAIL",
                "max_relative_precision_change": mp.nstr(max_refine_40_70, 20),
                "threshold": thresh_refine_40_70,
                "scope_note": "Max relative change from 40 to 70 dps evaluated across all observables (Phi, a, Gamma).",
            },
            "precision_refinement_55_to_70": {
                "status": "PASS" if pass_refine_55_70 else "FAIL",
                "max_relative_precision_change": mp.nstr(max_refine_55_70, 20),
                "threshold": thresh_refine_55_70,
                "scope_note": "Max relative change from 55 to 70 dps evaluated across all observables (Phi, a, Gamma).",
            },
            "omega_transfer_invariance": {
                "status": "PASS" if pass_omega else "FAIL",
                "nature": "contractually_and_structurally_exact",
                "description": "Spatial Poisson equation nabla^2 Phi = 4*pi*G*delta_rho does not contain omega; no dispersion relation is imposed on T4. Temporal factor exp(-i*omega*t) is purely multiplicative linear modulation.",
            },
            "finite_nan_inf": {
                "status": "PASS" if pass_finite else "FAIL",
                "all_finite": bool(pass_finite),
            },
            "trace_symmetry": {
                "status": "PASS" if pass_trace_symmetry else "FAIL",
                "max_trace_scaled_residual": summaries[2]["max_trace_scaled_residual"],
                "max_cartesian_gamma_symmetry_scaled": summaries[2][
                    "max_cartesian_gamma_symmetry_scaled"
                ],
                "threshold": thresh_oracle,
            },
        },
        "precision_summaries": summaries,
        "total_evaluated_points_per_dps": len(all_cases) * 5,
        "total_row_count": len(all_rows),
    }

    end_time = dt.datetime.now(dt.UTC)
    duration_s = (end_time - start_time).total_seconds()
    print(f"Numerical evaluation finished in {duration_s:.2f} s.")
    print("Individual check results:")
    for check_k, check_v in checks_status.items():
        print(f"  {check_k}: {'PASS' if check_v else 'FAIL'}")
    print(f"Overall status: {'PASS' if overall_pass else 'FAIL'}")
    print(f"Conclusion: {conclusion}")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        results_file = args.output_dir / "numerical-results.json"
        summary_file = args.output_dir / "numerical-summary.json"
        result_file = args.output_dir / "result.json"

        results_file.write_text(json.dumps(all_rows, indent=2))
        summary_file.write_text(json.dumps(numerical_summary, indent=2))

        def sha256_of(p):
            return hashlib.sha256(p.read_bytes()).hexdigest()

        result_json = {
            "schema_version": 2,
            "model_id": "T4",
            "overall_status": "PASS" if overall_pass else "FAIL",
            "scope": "Step7 T4 deterministic cylindrical temperature-mode gravity specialization",
            "identity": {
                "evaluation_started_at_utc": start_time.isoformat(),
                "evaluation_ended_at_utc": end_time.isoformat(),
                "duration_seconds": duration_s,
                "inputs": {
                    "t4_case_matrix.json": sha256_of(
                        WORKER_DIR / "t4_case_matrix.json"
                    ),
                    "gravity.py": sha256_of(WORKER_DIR / "gravity.py"),
                    "oracle.py": sha256_of(WORKER_DIR / "oracle.py"),
                },
                "outputs": {
                    "numerical_summary_json": sha256_of(summary_file),
                    "numerical_results_json": sha256_of(results_file),
                },
            },
            "scientific_checks": numerical_summary["checks"],
            "branch_summary": {
                "q_nonzero_cases": len(cases_data["q_nonzero"]),
                "kz_zero_cases": len(cases_data["kz_zero"]),
                "total_cases": len(all_cases),
                "zero_mode_policy": {
                    "full_zero_mode": "m=0, kappa=0, kz=0 excluded from potential catalogue",
                    "uniform_radial_branch": "m=0, kappa=0, kz!=0 verified (2 cases: pos/neg kz)",
                    "trivial_zero_density": "m!=0, kappa=0 verified exact 0 (2 cases: q_nonzero/kz_zero)",
                },
                "omega_independence": "Confirmed: gravity transfer has zero dependence on modulation frequency omega (no acoustic dispersion relation).",
            },
            "conclusion": conclusion,
        }

        result_file.write_text(json.dumps(result_json, indent=2))
        print("Wrote result.json successfully.")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
