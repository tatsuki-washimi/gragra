"""Execute candidate-vs-oracle numerical evaluation for AC3 Acoustic Specialization.

Loads gravity.py and oracle.py from this directory, evaluates all 13 acoustic cases
across r=0, interior, r=R (inside/outside limits), and exterior points at 40, 55, and 70 dps.
Outputs result.json and numerical-summary.json.
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

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gravity import (  # noqa: E402
    G_SI,
    axis_fields,
    field_from_radial,
    q_nonzero_radial,
    q_zero_radial,
)  # noqa: E402
from oracle import q_nonzero_oracle_radial, q_zero_oracle_radial  # noqa: E402

TOL_ORACLE_CANDIDATE = 5.0e-18
TOL_REFINEMENT_55_70 = 1.0e-25
TOL_REFINEMENT_40_70 = 5.0e-18


def M(val):
    return mp.mpf(str(val))


def amplitude(case):
    """Directly calculate density amplitude from serialized acoustic inputs P and cs."""
    cs = M(case["cs"])
    P = mp.mpc(M(case["P"][0]), M(case["P"][1]))
    return P / (cs**2)


_DEFAULT_TOL = mp.mpf("1e-70")


def verify_amplitude_coupling(cases, tol=_DEFAULT_TOL):
    """Verify that every case's density amplitude A strictly matches A = P / cs^2 directly from inputs."""
    max_diff = mp.mpf(0)
    with mp.workdps(120):
        for case in cases:
            cs = M(case["cs"])
            P = mp.mpc(M(case["P"][0]), M(case["P"][1]))
            A_calc = P / (cs**2)
            A_stored = mp.mpc(M(case["A"][0]), M(case["A"][1]))
            diff = abs(A_calc - A_stored)
            if diff > max_diff:
                max_diff = diff
            assert diff <= tol, (
                f"Amplitude coupling mismatch in {case['id']}: diff={diff} > {tol}"
            )
    return max_diff


def beta_eff(case):
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


def field_for(case, r, phi, z, radial):
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


def evaluate_point(case, r, side, dps):
    R = M(case["R"])
    kappa = M(case["kappa"])
    kz = M(case["kz"])
    A = amplitude(case)
    phi = M(case["phi"])
    z = M(case["z"])
    branch = case["branch"]
    eval_side = None if side == "auto" else side

    if branch == "q_nonzero":
        cand_radial = q_nonzero_radial(case["m"], kappa, kz, R, A, r, eval_side)
        ora_radial = q_nonzero_oracle_radial(case["m"], kappa, kz, R, A, r, eval_side)
    else:
        cand_radial = q_zero_radial(case["m"], kappa, R, A, r, eval_side)
        ora_radial = q_zero_oracle_radial(case["m"], kappa, R, A, r, eval_side)

    candidate = field_for(case, r, phi, z, cand_radial)
    oracle = field_for(case, r, phi, z, ora_radial)

    b = beta_eff(case)
    source_zero = kappa == 0 and abs(case["m"]) > 0
    scale_phi = abs(b) * R * R
    scale_a = abs(b) * R
    scale_gamma = abs(b)
    if source_zero:
        scale_phi = scale_a = scale_gamma = mp.mpf(1)

    def err(key, scale):
        diff = difference_max(candidate[key], oracle[key])
        ref = magnitude(oracle[key])
        scaled = diff / max(scale, ref, mp.mpf("1e-100"))
        relative = diff / max(ref, mp.mpf("1e-100"))
        return {
            "absolute": mp.nstr(diff, dps),
            "relative": mp.nstr(relative, dps),
            "scaled": mp.nstr(scaled, dps),
        }

    inside = (r < R) if side == "auto" else side == "inside"
    phase = mp.e ** (1j * case["m"] * phi + 1j * kz * z)
    expected_trace = (
        -b * mp.besselj(abs(case["m"]), kappa * r) * phase if inside else mp.mpc(0)
    )

    cand_trace = sum(candidate["Gamma_cart"][i, i] for i in range(3))
    ora_trace = sum(oracle["Gamma_cart"][i, i] for i in range(3))
    trace_cand_res = abs(cand_trace - expected_trace) / max(
        scale_gamma, mp.mpf("1e-100")
    )
    trace_ora_res = abs(ora_trace - expected_trace) / max(scale_gamma, mp.mpf("1e-100"))

    sym_cand = difference_max(candidate["Gamma_cart"], candidate["Gamma_cart"].T) / max(
        scale_gamma, mp.mpf("1e-100")
    )
    sym_ora = difference_max(oracle["Gamma_cart"], oracle["Gamma_cart"].T) / max(
        scale_gamma, mp.mpf("1e-100")
    )

    finite_all = field_finite(candidate) and field_finite(oracle)
    errors = {
        "Phi": err("Phi", scale_phi),
        "a": err("a_cart", scale_a),
        "Gamma": err("Gamma_cart", scale_gamma),
    }

    payload = {
        "case_id": case["id"],
        "dps": dps,
        "r_m": mp.nstr(r, dps),
        "side": side,
        "candidate": {
            "Phi": as_complex_pair(candidate["Phi"], dps),
            "a_cart": as_matrix(candidate["a_cart"], dps),
            "Gamma_cart": as_matrix(candidate["Gamma_cart"], dps),
        },
        "oracle": {
            "Phi": as_complex_pair(oracle["Phi"], dps),
            "a_cart": as_matrix(oracle["a_cart"], dps),
            "Gamma_cart": as_matrix(oracle["Gamma_cart"], dps),
        },
        "errors": errors,
        "invariants": {
            "candidate_trace_scaled_residual": mp.nstr(trace_cand_res, dps),
            "oracle_trace_scaled_residual": mp.nstr(trace_ora_res, dps),
            "candidate_gamma_symmetry_scaled": mp.nstr(sym_cand, dps),
            "oracle_gamma_symmetry_scaled": mp.nstr(sym_ora, dps),
            "finite_candidate_and_oracle": finite_all,
        },
        "acoustic_metadata": {
            "m": case["m"],
            "radial_n": case["radial_n"],
            "cs": case["cs"],
            "P": case["P"],
            "omega": case["omega"],
        },
    }
    return payload, candidate, oracle, (scale_phi, scale_a, scale_gamma)


def check_boundary_continuity(case, dps):
    R = M(case["R"])
    kappa = M(case["kappa"])
    kz = M(case["kz"])
    A = amplitude(case)
    branch = case["branch"]
    if branch == "q_nonzero":
        r_in = q_nonzero_radial(case["m"], kappa, kz, R, A, R, "inside")
        r_out = q_nonzero_radial(case["m"], kappa, kz, R, A, R, "outside")
    else:
        r_in = q_zero_radial(case["m"], kappa, R, A, R, "inside")
        r_out = q_zero_radial(case["m"], kappa, R, A, R, "outside")

    b = beta_eff(case)
    scale_phi = abs(b) * R * R
    scale_a = abs(b) * R
    if kappa == 0 and abs(case["m"]) > 0:
        scale_phi = scale_a = mp.mpf(1)

    diff_phi = abs(r_in[0] - r_out[0]) / max(scale_phi, mp.mpf("1e-100"))
    diff_fp = abs(r_in[1] - r_out[1]) / max(scale_a, mp.mpf("1e-100"))

    return {
        "case_id": case["id"],
        "dps": dps,
        "phi_jump_scaled": mp.nstr(diff_phi, dps),
        "fp_jump_scaled": mp.nstr(diff_fp, dps),
        "continuous": (
            diff_phi < TOL_ORACLE_CANDIDATE and diff_fp < TOL_ORACLE_CANDIDATE
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="AC3 Numerical Evaluation")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Directory to save output JSON files",
    )
    args = parser.parse_args()

    case_file = ROOT / "ac3_case_matrix.json"
    with open(case_file, encoding="utf-8") as f:
        matrix_data = json.load(f)
    cases = matrix_data["cases"]

    # Pre-check amplitude coupling A = P / cs^2 from serialized inputs
    max_amp_diff = verify_amplitude_coupling(cases)
    print(
        f"Verified amplitude coupling A = P / cs^2 (max diff: {max_amp_diff}) -> PASS"
    )

    precision_levels = [40, 55, 70]
    payloads = []
    boundary_results = []
    summaries = {}
    stored_candidates = {}  # (case_id, dps, r_str, side) -> candidate dict

    print(f"Starting evaluation of {len(cases)} cases at dps={precision_levels}...")

    for dps in precision_levels:
        print(f"\n--- Running dps = {dps} ---")
        max_scaled_error = {"Phi": mp.mpf(0), "a": mp.mpf(0), "Gamma": mp.mpf(0)}
        max_trace_res = mp.mpf(0)
        max_sym_res = mp.mpf(0)
        all_finite = True
        case_count = 0
        point_count = 0

        with mp.workdps(dps):
            for case in cases:
                case_count += 1
                pts = point_list(case)
                for r, side in pts:
                    point_count += 1
                    payload, cand, ora, scales = evaluate_point(case, r, side, dps)
                    payloads.append(payload)
                    stored_candidates[(case["id"], dps, str(r), side)] = (cand, scales)

                    for obs in ("Phi", "a", "Gamma"):
                        sc = M(payload["errors"][obs]["scaled"])
                        max_scaled_error[obs] = max(max_scaled_error[obs], sc)

                    tr_cand = M(
                        payload["invariants"]["candidate_trace_scaled_residual"]
                    )
                    tr_ora = M(payload["invariants"]["oracle_trace_scaled_residual"])
                    max_trace_res = max(max_trace_res, tr_cand, tr_ora)

                    sym_c = M(payload["invariants"]["candidate_gamma_symmetry_scaled"])
                    sym_o = M(payload["invariants"]["oracle_gamma_symmetry_scaled"])
                    max_sym_res = max(max_sym_res, sym_c, sym_o)

                    all_finite = (
                        all_finite
                        and payload["invariants"]["finite_candidate_and_oracle"]
                    )

                b_res = check_boundary_continuity(case, dps)
                boundary_results.append(b_res)

        overall_max_err = max(max_scaled_error.values())
        status = "PASS" if overall_max_err < TOL_ORACLE_CANDIDATE else "FAIL"
        print(
            f"  dps={dps}: points={point_count}, max_scaled_error={float(overall_max_err):12.4e}, trace_res={float(max_trace_res):12.4e}, status={status}"
        )

        summaries[f"dps_{dps}"] = {
            "dps": dps,
            "case_count": case_count,
            "point_count": point_count,
            "max_scaled_errors": {
                k: mp.nstr(v, 16) for k, v in max_scaled_error.items()
            },
            "overall_max_scaled_error": mp.nstr(overall_max_err, 16),
            "max_trace_residual": mp.nstr(max_trace_res, 16),
            "max_gamma_symmetry": mp.nstr(max_sym_res, 16),
            "all_finite": all_finite,
            "status": status,
        }

    # Precision refinement checks
    print("\n--- Running Precision Refinement Checks ---")
    # 55 vs 70 and 40 vs 70
    refinement_55_70 = {"Phi": mp.mpf(0), "a": mp.mpf(0), "Gamma": mp.mpf(0)}
    refinement_40_70 = {"Phi": mp.mpf(0), "a": mp.mpf(0), "Gamma": mp.mpf(0)}

    with mp.workdps(80):
        for case in cases:
            for r, side in point_list(case):
                cand_40, scales_40 = stored_candidates[(case["id"], 40, str(r), side)]
                cand_55, scales_55 = stored_candidates[(case["id"], 55, str(r), side)]
                cand_70, scales_70 = stored_candidates[(case["id"], 70, str(r), side)]

                for obs, key, scale_idx in (
                    ("Phi", "Phi", 0),
                    ("a", "a_cart", 1),
                    ("Gamma", "Gamma_cart", 2),
                ):
                    diff_55_70 = difference_max(cand_55[key], cand_70[key])
                    scale = max(
                        scales_70[scale_idx], magnitude(cand_70[key]), mp.mpf("1e-100")
                    )
                    err_55_70 = diff_55_70 / scale
                    refinement_55_70[obs] = max(refinement_55_70[obs], err_55_70)

                    diff_40_70 = difference_max(cand_40[key], cand_70[key])
                    err_40_70 = diff_40_70 / scale
                    refinement_40_70[obs] = max(refinement_40_70[obs], err_40_70)

    max_ref_55_70 = max(refinement_55_70.values())
    max_ref_40_70 = max(refinement_40_70.values())
    status_ref_55_70 = "PASS" if max_ref_55_70 < TOL_REFINEMENT_55_70 else "FAIL"
    status_ref_40_70 = "PASS" if max_ref_40_70 < TOL_REFINEMENT_40_70 else "FAIL"

    print(
        f"  Refinement 55 vs 70 max scaled error: {float(max_ref_55_70):12.4e} (threshold {TOL_REFINEMENT_55_70}) -> {status_ref_55_70}"
    )
    print(
        f"  Refinement 40 vs 70 max scaled error: {float(max_ref_40_70):12.4e} (threshold {TOL_REFINEMENT_40_70}) -> {status_ref_40_70}"
    )

    refinement_summary = {
        "refinement_55_vs_70": {
            "max_scaled_errors": {
                k: mp.nstr(v, 16) for k, v in refinement_55_70.items()
            },
            "overall_max": mp.nstr(max_ref_55_70, 16),
            "threshold": str(TOL_REFINEMENT_55_70),
            "status": status_ref_55_70,
        },
        "refinement_40_vs_70": {
            "max_scaled_errors": {
                k: mp.nstr(v, 16) for k, v in refinement_40_70.items()
            },
            "overall_max": mp.nstr(max_ref_40_70, 16),
            "threshold": str(TOL_REFINEMENT_40_70),
            "status": status_ref_40_70,
        },
    }

    # Boundary jumps summary
    max_phi_jump = max(M(b["phi_jump_scaled"]) for b in boundary_results)
    max_fp_jump = max(M(b["fp_jump_scaled"]) for b in boundary_results)
    boundary_status = (
        "PASS" if max(max_phi_jump, max_fp_jump) < TOL_ORACLE_CANDIDATE else "FAIL"
    )
    boundary_summary = {
        "max_phi_jump_scaled": mp.nstr(max_phi_jump, 16),
        "max_fp_jump_scaled": mp.nstr(max_fp_jump, 16),
        "status": boundary_status,
    }

    # Gate checks across all requirements
    pass_coupling = max_amp_diff <= mp.mpf("1e-70")
    pass_oracle_40 = summaries["dps_40"]["status"] == "PASS"
    pass_oracle_55 = summaries["dps_55"]["status"] == "PASS"
    pass_oracle_70 = summaries["dps_70"]["status"] == "PASS"
    pass_oracle = pass_oracle_40 and pass_oracle_55 and pass_oracle_70
    pass_boundary = boundary_status == "PASS"
    pass_refine_40_70 = status_ref_40_70 == "PASS"
    pass_refine_55_70 = status_ref_55_70 == "PASS"
    pass_finite = all(summaries[f"dps_{dps}"]["all_finite"] for dps in precision_levels)
    pass_trace = all(
        M(summaries[f"dps_{dps}"]["max_trace_residual"]) < TOL_ORACLE_CANDIDATE
        for dps in precision_levels
    )
    pass_symmetry = all(
        M(summaries[f"dps_{dps}"]["max_gamma_symmetry"]) < TOL_ORACLE_CANDIDATE
        for dps in precision_levels
    )

    checks_status = {
        "density_amplitude_coupling": pass_coupling,
        "candidate_vs_independent_radial_oracle": pass_oracle,
        "boundary_Phi_and_normal_derivative_continuity": pass_boundary,
        "precision_refinement_40_to_70": pass_refine_40_70,
        "precision_refinement_55_to_70": pass_refine_55_70,
        "finite_nan_inf": pass_finite,
        "trace_identity": pass_trace,
        "tensor_symmetry": pass_symmetry,
    }
    overall_pass = all(checks_status.values())
    overall_determination = "VERIFIED_INDEPENDENT" if overall_pass else "FAIL"

    summary_data = {
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "specialization": "AC3",
        "determination": overall_determination,
        "overall_status": "PASS" if overall_pass else "FAIL",
        "checks": checks_status,
        "amplitude_coupling_check": {
            "formula": "A = P / cs^2",
            "status": "PASS" if pass_coupling else "FAIL",
            "max_discrepancy": mp.nstr(max_amp_diff, 16),
            "threshold": "1.0e-70",
        },
        "dps_summaries": summaries,
        "refinement_summary": refinement_summary,
        "boundary_summary": boundary_summary,
    }

    result_data = {
        "schema_version": 2,
        "model_id": "AC3",
        "overall_status": "PASS" if overall_pass else "FAIL",
        "determination": overall_determination,
        "scope": "Step6b AC3 cylindrical acoustic-mode gravity specialization",
        "scientific_checks": checks_status,
        "summary": summary_data,
        "boundary_checks": boundary_results,
    }

    print("\nIndividual check results:")
    for check_k, check_v in checks_status.items():
        print(f"  {check_k}: {'PASS' if check_v else 'FAIL'}")
    print(f"Overall status: {'PASS' if overall_pass else 'FAIL'}")
    print(f"Conclusion: {overall_determination}")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        results_path = args.output_dir / "numerical-results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(payloads, f, indent=2)
        print(f"\nSaved numerical results to {results_path}")

        num_summary_path = args.output_dir / "numerical-summary.json"
        with open(num_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved numerical summary to {num_summary_path}")

        def sha256_of(p):
            return hashlib.sha256(p.read_bytes()).hexdigest()

        result_data["identity"] = {
            "inputs": {
                "ac3_case_matrix.json": sha256_of(ROOT / "ac3_case_matrix.json"),
                "gravity.py": sha256_of(ROOT / "gravity.py"),
                "oracle.py": sha256_of(ROOT / "oracle.py"),
            },
            "outputs": {
                "numerical_results_json": sha256_of(results_path),
                "numerical_summary_json": sha256_of(num_summary_path),
            },
        }

        result_path = args.output_dir / "result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2)
        print(f"Saved complete evaluation result to {result_path}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
